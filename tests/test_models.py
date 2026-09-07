"""Tests for the model factory and training pipeline.

The most important test here is
:func:`test_scaler_is_fit_on_training_data_only` — fitting the scaler on the
whole dataset is a quiet leak that inflates every downstream metric.
"""

import numpy as np
import pandas as pd
import pytest

from src.config import Config
from src.models import (
    DEFAULT_PARAMS,
    MODEL_LABELS,
    ModelPipeline,
    available_models,
    build_model,
    model_path,
    pipeline_from_config,
    xgboost_available,
)

MODELS = available_models()


@pytest.fixture
def xy():
    """A learnable signal: y depends linearly on the first two features."""
    rng = np.random.default_rng(3)
    n, k = 400, 6
    idx = pd.bdate_range("2020-01-01", periods=n, name="date")
    X = pd.DataFrame(
        rng.normal(0, 1, (n, k)),
        columns=[f"f{i}" for i in range(k)],
        index=idx,
    )
    y = pd.Series(
        0.5 * X["f0"] - 0.3 * X["f1"] + rng.normal(0, 0.1, n),
        index=idx, name="target",
    )
    return X, y


# ------------------------------------------------------------------ factory


@pytest.mark.parametrize("name", MODELS)
def test_build_model_for_each_name(name):
    assert build_model(name, seed=1) is not None


@pytest.mark.parametrize("name", MODELS)
def test_build_classification_variant(name):
    model = build_model(name, seed=1, task="classification")
    assert hasattr(model, "predict_proba")


def test_unknown_model_rejected():
    with pytest.raises(ValueError, match="Unknown model"):
        build_model("neural_net")


def test_unknown_task_rejected():
    with pytest.raises(ValueError, match="Unknown task"):
        build_model("linear", task="ranking")


def test_config_params_override_defaults():
    model = build_model("ridge", {"alpha": 99.0})
    assert model.alpha == 99.0


def test_defaults_are_conservative_for_trees():
    """Deep unconstrained trees memorise noise on daily financial data."""
    assert DEFAULT_PARAMS["random_forest"]["max_depth"] <= 8
    assert DEFAULT_PARAMS["random_forest"]["min_samples_leaf"] >= 20
    assert DEFAULT_PARAMS["xgboost"]["max_depth"] <= 6


def test_every_model_has_a_label():
    for name in MODELS:
        assert name in MODEL_LABELS


# ----------------------------------------------------------------- fitting


@pytest.mark.parametrize("name", MODELS)
def test_fit_predict_shape_and_index(name, xy):
    X, y = xy
    pipe = ModelPipeline(name, seed=0).fit(X, y)
    pred = pipe.predict(X)
    assert len(pred) == len(X)
    assert pred.index.equals(X.index)
    assert np.isfinite(pred).all()


@pytest.mark.parametrize("name", MODELS)
def test_model_learns_a_real_signal(name, xy):
    """Sanity floor: on data with a genuine linear signal, beat the mean."""
    X, y = xy
    train, test = slice(0, 300), slice(300, None)
    pipe = ModelPipeline(name, seed=0).fit(X.iloc[train], y.iloc[train])
    pred = pipe.predict(X.iloc[test])
    from sklearn.metrics import r2_score

    assert r2_score(y.iloc[test], pred) > 0.3, f"{name} failed to learn"


def test_predict_before_fit_raises(xy):
    X, _ = xy
    with pytest.raises(RuntimeError, match="fit"):
        ModelPipeline("linear").predict(X)


def test_fit_rejects_mismatched_lengths(xy):
    X, y = xy
    with pytest.raises(ValueError, match="rows"):
        ModelPipeline("linear").fit(X, y.iloc[:-5])


def test_fit_rejects_empty_data():
    with pytest.raises(ValueError, match="empty"):
        ModelPipeline("linear").fit(pd.DataFrame({"a": []}), pd.Series([], dtype=float))


def test_fit_rejects_missing_values(xy):
    X, y = xy
    X = X.copy()
    X.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="missing"):
        ModelPipeline("linear").fit(X, y)


def test_predict_rejects_missing_columns(xy):
    X, y = xy
    pipe = ModelPipeline("linear").fit(X, y)
    with pytest.raises(ValueError, match="Missing feature columns"):
        pipe.predict(X.drop(columns=["f0"]))


def test_predict_is_robust_to_column_reordering(xy):
    """A silent reorder would pair every feature with the wrong coefficient."""
    X, y = xy
    pipe = ModelPipeline("linear", seed=0).fit(X, y)
    shuffled = X[list(reversed(X.columns))]
    pd.testing.assert_series_equal(pipe.predict(X), pipe.predict(shuffled))


# ============================================================ SCALER LEAKAGE


def test_scaler_is_fit_on_training_data_only(xy):
    """The scaler must never see validation or test statistics."""
    X, y = xy
    train = slice(0, 300)
    pipe = ModelPipeline("linear", seed=0).fit(X.iloc[train], y.iloc[train])

    expected_mean = X.iloc[train].to_numpy().mean(axis=0)
    assert np.allclose(pipe.scaler.mean_, expected_mean), (
        "scaler statistics do not match the training rows — it saw more data"
    )
    full_mean = X.to_numpy().mean(axis=0)
    assert not np.allclose(pipe.scaler.mean_, full_mean, atol=1e-12), (
        "scaler was fit on the full dataset, leaking test statistics"
    )


def test_predictions_do_not_change_when_future_rows_are_added(xy):
    """Scoring extra future rows must not alter earlier predictions."""
    X, y = xy
    pipe = ModelPipeline("ridge", seed=0).fit(X.iloc[:300], y.iloc[:300])
    a = pipe.predict(X.iloc[300:350])
    b = pipe.predict(X.iloc[300:])
    pd.testing.assert_series_equal(a, b.iloc[:50])


def test_scaling_can_be_disabled(xy):
    X, y = xy
    pipe = ModelPipeline("linear", scale=False).fit(X, y)
    assert pipe.scaler is None
    assert np.isfinite(pipe.predict(X)).all()


# ------------------------------------------------------- reproducibility


@pytest.mark.parametrize("name", MODELS)
def test_same_seed_gives_identical_predictions(name, xy):
    X, y = xy
    a = ModelPipeline(name, seed=42).fit(X, y).predict(X)
    b = ModelPipeline(name, seed=42).fit(X, y).predict(X)
    pd.testing.assert_series_equal(a, b)


# --------------------------------------------------------- interpretation


@pytest.mark.parametrize("name", MODELS)
def test_feature_importance_is_normalised(name, xy):
    X, y = xy
    imp = ModelPipeline(name, seed=0).fit(X, y).feature_importance()
    assert set(imp.index) == set(X.columns)
    assert imp.sum() == pytest.approx(1.0)
    assert (imp >= 0).all()
    assert imp.is_monotonic_decreasing


@pytest.mark.parametrize("name", MODELS)
def test_important_features_are_the_informative_ones(name, xy):
    """y was built from f0 and f1, so they must dominate."""
    X, y = xy
    imp = ModelPipeline(name, seed=0).fit(X, y).feature_importance()
    assert set(imp.head(2).index) == {"f0", "f1"}


def test_feature_importance_before_fit_raises():
    with pytest.raises(RuntimeError):
        ModelPipeline("linear").feature_importance()


# --------------------------------------------------------- classification


@pytest.mark.parametrize("name", MODELS)
def test_classification_pipeline(name, xy):
    X, y = xy
    labels = (y > 0).astype(float)
    pipe = ModelPipeline(name, seed=0, task="classification").fit(X, labels)
    pred = pipe.predict(X)
    assert set(np.unique(pred)) <= {0.0, 1.0}
    proba = pipe.predict_proba(X)
    assert proba.between(0, 1).all()


def test_predict_proba_rejected_for_regression(xy):
    X, y = xy
    pipe = ModelPipeline("linear").fit(X, y)
    with pytest.raises(RuntimeError, match="classification"):
        pipe.predict_proba(X)


# ------------------------------------------------------------ persistence


@pytest.mark.parametrize("name", MODELS)
def test_save_and_load_roundtrip(name, xy, tmp_path):
    X, y = xy
    pipe = ModelPipeline(name, seed=0).fit(X, y)
    path = pipe.save(tmp_path / f"{name}.joblib")
    assert path.exists()

    reloaded = ModelPipeline.load(path)
    assert reloaded.is_fitted
    assert reloaded.feature_names_ == pipe.feature_names_
    pd.testing.assert_series_equal(pipe.predict(X), reloaded.predict(X))


# ------------------------------------------------------------ convenience


def test_pipeline_from_config_uses_config_values():
    cfg = Config(model="ridge", seed=7, model_params={"ridge": {"alpha": 5.0}})
    pipe = pipeline_from_config(cfg)
    assert pipe.name == "ridge"
    assert pipe.seed == 7
    assert pipe.model.alpha == 5.0


def test_model_path_is_config_specific():
    a = model_path(Config(ticker="AAPL"), "ridge")
    b = model_path(Config(ticker="MSFT"), "ridge")
    assert a != b
    assert a.suffix == ".joblib"


def test_xgboost_is_available_in_this_environment():
    assert xgboost_available(), "requirements.txt pins xgboost; it should import"
    assert "xgboost" in available_models()
