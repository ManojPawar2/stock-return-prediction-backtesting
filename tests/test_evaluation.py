"""Tests for evaluation metrics.

Metrics are checked against hand-computed values wherever a closed form
exists, so a formula change cannot pass silently.
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation import (
    classification_metrics,
    confusion,
    directional_accuracy,
    evaluate,
    format_metrics,
    hit_rate,
    information_coefficient,
    metrics_frame,
    regression_metrics,
    sanity_flags,
)


# ------------------------------------------------------------- regression


def test_perfect_prediction():
    y = pd.Series([0.01, -0.02, 0.03, -0.01])
    m = regression_metrics(y, y.copy())
    assert m["mae"] == pytest.approx(0.0)
    assert m["rmse"] == pytest.approx(0.0)
    assert m["r2"] == pytest.approx(1.0)
    assert m["directional_accuracy"] == pytest.approx(1.0)


def test_mae_and_rmse_are_hand_computable():
    y_true = pd.Series([1.0, 2.0, 3.0, 4.0])
    y_pred = pd.Series([1.0, 2.0, 3.0, 8.0])   # one error of 4
    m = regression_metrics(y_true, y_pred)
    assert m["mae"] == pytest.approx(4 / 4)
    assert m["rmse"] == pytest.approx(np.sqrt(16 / 4))
    assert m["rmse"] > m["mae"], "RMSE must penalise the large miss more"


def test_r2_of_the_mean_predictor_is_zero():
    y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    m = regression_metrics(y, pd.Series([y.mean()] * 5))
    assert m["r2"] == pytest.approx(0.0)


def test_r2_can_be_negative():
    """Worse than predicting the mean — the normal outcome on daily returns."""
    y = pd.Series([1.0, 2.0, 3.0, 4.0])
    m = regression_metrics(y, pd.Series([10.0, -10.0, 10.0, -10.0]))
    assert m["r2"] < 0


def test_directional_accuracy_counts_signs_only():
    y_true = pd.Series([0.01, -0.01, 0.02, -0.02])
    y_pred = pd.Series([0.99, -0.50, -0.10, -0.30])   # 3 of 4 signs right
    assert directional_accuracy(y_true, y_pred) == pytest.approx(0.75)


def test_directional_accuracy_ignores_flat_days():
    """A zero return has no direction to predict."""
    y_true = pd.Series([0.01, 0.0, 0.0, -0.01])
    y_pred = pd.Series([0.5, 0.5, -0.5, -0.5])
    assert directional_accuracy(y_true, y_pred) == pytest.approx(1.0)


def test_directional_accuracy_all_flat_is_nan():
    assert np.isnan(directional_accuracy(pd.Series([0.0, 0.0]), pd.Series([1.0, 1.0])))


def test_hit_rate_only_counts_traded_days():
    y_true = pd.Series([0.01, -0.01, 0.02, -0.02])
    y_pred = pd.Series([0.5, -0.5, 0.5, -0.5])   # would trade days 0 and 2
    assert hit_rate(y_true, y_pred, threshold=0.0) == pytest.approx(1.0)


def test_hit_rate_is_nan_when_nothing_would_trade():
    y_true = pd.Series([0.01, 0.02])
    y_pred = pd.Series([-0.5, -0.5])
    assert np.isnan(hit_rate(y_true, y_pred, threshold=0.0))


def test_information_coefficient_signs():
    y = pd.Series(np.arange(50, dtype=float))
    assert information_coefficient(y, y) == pytest.approx(1.0)
    assert information_coefficient(y, -y) == pytest.approx(-1.0)


def test_information_coefficient_of_a_constant_is_nan():
    y = pd.Series(np.arange(20, dtype=float))
    assert np.isnan(information_coefficient(y, pd.Series([1.0] * 20)))


def test_metrics_handle_nan_by_aligning():
    y_true = pd.Series([0.01, np.nan, 0.03, 0.04])
    y_pred = pd.Series([0.01, 0.02, np.nan, 0.04])
    m = regression_metrics(y_true, y_pred)
    assert m["n"] == 2
    assert m["mae"] == pytest.approx(0.0)


def test_empty_input_gives_nan_not_a_crash():
    m = regression_metrics(pd.Series([], dtype=float), pd.Series([], dtype=float))
    assert np.isnan(m["mae"])


def test_mismatched_arrays_rejected():
    with pytest.raises(ValueError, match="shape"):
        regression_metrics(np.array([1.0, 2.0]), np.array([1.0]))


def test_prediction_distribution_is_reported():
    m = regression_metrics(pd.Series([0.01, -0.01]), pd.Series([0.02, -0.02]))
    assert m["mean_prediction"] == pytest.approx(0.0)
    assert m["std_prediction"] == pytest.approx(0.02)
    assert m["pct_predicted_positive"] == pytest.approx(0.5)


# --------------------------------------------------------- classification


def test_classification_perfect_score():
    y = pd.Series([1.0, 0.0, 1.0, 0.0])
    m = classification_metrics(y, y.copy())
    assert m["accuracy"] == pytest.approx(1.0)
    assert m["precision"] == pytest.approx(1.0)
    assert m["recall"] == pytest.approx(1.0)
    assert m["f1"] == pytest.approx(1.0)


def test_classification_hand_computed():
    #             TP    FP    TN    FN
    y_true = pd.Series([1.0, 0.0, 0.0, 1.0])
    y_pred = pd.Series([1.0, 1.0, 0.0, 0.0])
    m = classification_metrics(y_true, y_pred)
    assert m["accuracy"] == pytest.approx(0.5)
    assert m["precision"] == pytest.approx(0.5)   # 1 TP / (1 TP + 1 FP)
    assert m["recall"] == pytest.approx(0.5)      # 1 TP / (1 TP + 1 FN)


def test_base_rate_is_the_score_to_beat():
    y_true = pd.Series([1.0] * 7 + [0.0] * 3)
    m = classification_metrics(y_true, pd.Series([1.0] * 10))
    assert m["base_rate"] == pytest.approx(0.7)
    assert m["accuracy"] == pytest.approx(0.7), "always-up must equal the base rate"


def test_roc_auc_with_probabilities():
    y_true = pd.Series([0.0, 0.0, 1.0, 1.0])
    y_prob = pd.Series([0.1, 0.2, 0.8, 0.9])
    m = classification_metrics(y_true, (y_prob > 0.5).astype(float), y_prob)
    assert m["roc_auc"] == pytest.approx(1.0)


def test_roc_auc_is_nan_for_a_single_class():
    y_true = pd.Series([1.0, 1.0, 1.0])
    m = classification_metrics(y_true, y_true, pd.Series([0.6, 0.7, 0.8]))
    assert np.isnan(m["roc_auc"])


def test_confusion_matrix_layout():
    y_true = pd.Series([1.0, 0.0, 0.0, 1.0])
    y_pred = pd.Series([1.0, 1.0, 0.0, 0.0])
    cm = confusion(y_true, y_pred)
    assert cm.shape == (2, 2)
    assert cm.loc["actual up", "predicted up"] == 1
    assert cm.loc["actual down", "predicted up"] == 1
    assert cm.to_numpy().sum() == 4


# -------------------------------------------------------------- dispatch


def test_evaluate_dispatches_on_task():
    y = pd.Series([1.0, 0.0, 1.0])
    assert "r2" in evaluate(y, y, task="regression")
    assert "accuracy" in evaluate(y, y, task="classification")


def test_metrics_frame_builds_a_comparison_table():
    frame = metrics_frame({
        "linear": {"rmse": 0.02, "r2": -0.01},
        "xgboost": {"rmse": 0.019, "r2": 0.001},
    })
    assert list(frame.index) == ["linear", "xgboost"]
    assert "rmse" in frame.columns


def test_format_metrics_handles_nan_and_ints():
    text = format_metrics({"n": 10, "mae": 0.0123, "r2": float("nan")})
    assert "n=10" in text and "mae=0.0123" in text and "r2=n/a" in text


# ---------------------------------------------------------- sanity flags


def test_no_flags_for_a_realistic_result():
    """The honest expectation from README section 25."""
    assert sanity_flags({
        "r2": -0.01, "directional_accuracy": 0.517,
        "information_coefficient": 0.04, "std_prediction": 0.004,
    }) == []


def test_flags_implausible_r2():
    flags = sanity_flags({"r2": 0.85})
    assert flags and "lookahead" in flags[0]


def test_flags_implausible_directional_accuracy():
    flags = sanity_flags({"directional_accuracy": 0.92})
    assert flags and "49-54" in flags[0]


def test_flags_implausible_information_coefficient():
    assert sanity_flags({"information_coefficient": 0.85})


def test_flags_a_constant_predictor():
    flags = sanity_flags({"std_prediction": 0.0})
    assert flags and "mean" in flags[0]
