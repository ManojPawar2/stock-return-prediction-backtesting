"""Model factory and training pipeline.

One interface (:class:`ModelPipeline`) wraps every estimator so that models are
swapped by name and compared on identical splits, identical features and an
identical seed.

The pipeline owns the scaler, and that placement is the point: ``StandardScaler``
is fit on the **training rows only** and merely applied to validation and test.
Fitting it on the full dataset would let the test period's mean and standard
deviation bleed into training — a quiet, easily-missed leak that inflates every
metric downstream.

Tree hyperparameters are deliberately conservative (shallow depth, large
minimum leaf size).  On daily financial data an unconstrained tree fits the
noise perfectly and then generalises at chance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from .config import MODEL_NAMES, MODELS as MODELS_DIR, Config

logger = logging.getLogger(__name__)


#: Human-readable labels for the dashboard.
MODEL_LABELS: dict[str, str] = {
    "linear": "Linear Regression",
    "ridge": "Ridge Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}

#: Sensible defaults when config.yaml supplies nothing.
DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    "linear": {},
    "ridge": {"alpha": 1.0},
    "random_forest": {
        "n_estimators": 300,
        "max_depth": 5,
        "min_samples_leaf": 50,
        "max_features": 0.5,
        "n_jobs": -1,
    },
    "xgboost": {
        "n_estimators": 300,
        "max_depth": 3,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "min_child_weight": 20,
        "n_jobs": -1,
    },
}


def xgboost_available() -> bool:
    try:
        import xgboost  # noqa: F401
        return True
    except ImportError:
        return False


def build_model(
    name: str,
    params: dict[str, Any] | None = None,
    seed: int = 42,
    task: str = "regression",
):
    """Construct a bare estimator by name.

    ``task`` selects the regression or classification variant of the same
    family, so the rest of the pipeline is identical either way.
    """
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown model {name!r}. Choose from {MODEL_NAMES}.")
    if task not in ("regression", "classification"):
        raise ValueError(f"Unknown task {task!r}")

    merged = {**DEFAULT_PARAMS.get(name, {}), **(params or {})}
    classify = task == "classification"

    if name == "linear":
        if classify:
            return LogisticRegression(max_iter=1000, random_state=seed)
        return LinearRegression(**merged)

    if name == "ridge":
        if classify:
            return LogisticRegression(
                penalty="l2",
                C=1.0 / max(merged.get("alpha", 1.0), 1e-9),
                max_iter=1000,
                random_state=seed,
            )
        return Ridge(random_state=seed, **merged)

    if name == "random_forest":
        cls = RandomForestClassifier if classify else RandomForestRegressor
        return cls(random_state=seed, **merged)

    # xgboost
    if not xgboost_available():
        raise ImportError(
            "xgboost is not installed. Run: pip install -r requirements.txt"
        )
    from xgboost import XGBClassifier, XGBRegressor

    cls = XGBClassifier if classify else XGBRegressor
    kwargs = dict(merged)
    if classify:
        kwargs.setdefault("eval_metric", "logloss")
    return cls(random_state=seed, **kwargs)


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------


@dataclass
class ModelPipeline:
    """Scaler + estimator, with the scaler fit on training data only."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)
    seed: int = 42
    task: str = "regression"
    scale: bool = True

    def __post_init__(self) -> None:
        self.model = build_model(self.name, self.params, self.seed, self.task)
        self.scaler: StandardScaler | None = None
        self.feature_names_: list[str] = []
        self.is_fitted: bool = False

    # ------------------------------------------------------------- fitting

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ModelPipeline":
        if len(X) != len(y):
            raise ValueError(f"X has {len(X)} rows but y has {len(y)}")
        if len(X) == 0:
            raise ValueError("cannot fit on an empty training set")
        if X.isna().any().any():
            raise ValueError("X contains missing values")

        self.feature_names_ = list(X.columns)

        values = X.to_numpy(dtype="float64")
        if self.scale:
            # Fit on train only. See the module docstring.
            self.scaler = StandardScaler()
            values = self.scaler.fit_transform(values)

        self.model.fit(values, y.to_numpy(dtype="float64"))
        self.is_fitted = True
        logger.info(
            "Fitted %s on %d rows x %d features", self.name, len(X), X.shape[1]
        )
        return self

    def _prepare(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict()")
        missing = [c for c in self.feature_names_ if c not in X.columns]
        if missing:
            raise ValueError(f"Missing feature columns at predict time: {missing}")
        # Reorder to the training column order — a silent reorder would
        # feed every feature into the wrong coefficient.
        values = X[self.feature_names_].to_numpy(dtype="float64")
        return self.scaler.transform(values) if self.scaler is not None else values

    # ---------------------------------------------------------- prediction

    def predict(self, X: pd.DataFrame) -> pd.Series:
        out = self.model.predict(self._prepare(X))
        return pd.Series(np.asarray(out, dtype="float64"), index=X.index,
                         name="prediction")

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        """Probability of the positive class (classification only)."""
        if self.task != "classification":
            raise RuntimeError("predict_proba is only defined for classification")
        proba = self.model.predict_proba(self._prepare(X))
        return pd.Series(proba[:, 1], index=X.index, name="probability")

    # ------------------------------------------------------ interpretation

    def feature_importance(self) -> pd.Series:
        """Importance per feature, normalised to sum to 1.

        Trees expose ``feature_importances_``; linear models expose
        coefficients, whose absolute value is comparable across features only
        because the inputs were standardised first.
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before feature_importance()")

        if hasattr(self.model, "feature_importances_"):
            raw = np.asarray(self.model.feature_importances_, dtype="float64")
        elif hasattr(self.model, "coef_"):
            raw = np.abs(np.asarray(self.model.coef_, dtype="float64")).ravel()
        else:
            raise AttributeError(f"{self.name} exposes no feature importance")

        total = raw.sum()
        if total > 0:
            raw = raw / total
        return (
            pd.Series(raw, index=self.feature_names_, name="importance")
            .sort_values(ascending=False)
        )

    # ------------------------------------------------------- serialisation

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("Saved model to %s", path)
        return path

    @staticmethod
    def load(path: str | Path) -> "ModelPipeline":
        return joblib.load(Path(path))

    @property
    def label(self) -> str:
        return MODEL_LABELS.get(self.name, self.name)

    def __repr__(self) -> str:  # pragma: no cover - display only
        state = "fitted" if self.is_fitted else "unfitted"
        return f"ModelPipeline({self.name!r}, {self.task}, {state})"


# --------------------------------------------------------------------------
# Convenience
# --------------------------------------------------------------------------


def pipeline_from_config(cfg: Config, model_name: str | None = None) -> ModelPipeline:
    name = model_name or cfg.model
    return ModelPipeline(
        name=name,
        params=cfg.params_for(name),
        seed=cfg.seed,
        task=cfg.target_kind,
    )


def model_path(cfg: Config, model_name: str | None = None) -> Path:
    name = model_name or cfg.model
    return MODELS_DIR / f"{cfg.ticker}_{name}_{cfg.fingerprint()}.joblib"


def available_models() -> list[str]:
    """Model names usable in this environment."""
    return [m for m in MODEL_NAMES if m != "xgboost" or xgboost_available()]
