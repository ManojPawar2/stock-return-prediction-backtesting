"""Model evaluation metrics.

Two families, kept apart on purpose.

**Statistical metrics** (MAE, RMSE, R²) measure how close the predicted number
is to the realised return.  **Directional metrics** measure whether the sign
was right.  They routinely disagree, because RMSE is dominated by a handful of
large moves that a threshold strategy may never trade.

On daily equity returns, expect R² near zero — often slightly negative — and
directional accuracy in the 49-54% band.  A high R² here is almost always a
leak rather than skill, so :func:`sanity_flags` says so explicitly.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def _align(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    """Align on the shared index, drop rows either side cannot supply."""
    if isinstance(y_true, pd.Series) and isinstance(y_pred, pd.Series):
        frame = pd.concat([y_true.rename("t"), y_pred.rename("p")], axis=1).dropna()
        return frame["t"].to_numpy("float64"), frame["p"].to_numpy("float64")
    t = np.asarray(y_true, dtype="float64")
    p = np.asarray(y_pred, dtype="float64")
    if t.shape != p.shape:
        raise ValueError(f"shape mismatch: {t.shape} vs {p.shape}")
    keep = ~(np.isnan(t) | np.isnan(p))
    return t[keep], p[keep]


# --------------------------------------------------------------------------
# Regression
# --------------------------------------------------------------------------


def directional_accuracy(y_true, y_pred) -> float:
    """Fraction of days where the predicted sign matched the realised sign.

    Days where the realised return is exactly zero are excluded: there is no
    direction to get right, and counting them would flatter the score.
    """
    t, p = _align(y_true, y_pred)
    mask = t != 0
    if not mask.any():
        return float("nan")
    return float((np.sign(t[mask]) == np.sign(p[mask])).mean())


def hit_rate(y_true, y_pred, threshold: float = 0.0) -> float:
    """Accuracy restricted to the days the strategy would actually trade.

    This is closer to what matters than overall directional accuracy: a model
    can be poor on average yet reliable on its high-conviction days.
    """
    t, p = _align(y_true, y_pred)
    traded = p > threshold
    if not traded.any():
        return float("nan")
    return float((t[traded] > 0).mean())


def information_coefficient(y_true, y_pred) -> float:
    """Spearman rank correlation between prediction and outcome.

    The standard quant measure of forecast quality.  Values around 0.03-0.05
    are considered good for daily equity signals.
    """
    t, p = _align(y_true, y_pred)
    if len(t) < 3 or np.std(p) == 0:
        return float("nan")
    from scipy.stats import spearmanr

    rho, _ = spearmanr(t, p)
    return float(rho)


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Full statistical + directional metric set for a return forecast."""
    t, p = _align(y_true, y_pred)
    if len(t) == 0:
        return {k: float("nan") for k in
                ("n", "mae", "rmse", "r2", "directional_accuracy", "hit_rate",
                 "information_coefficient", "mean_prediction", "std_prediction",
                 "pct_predicted_positive")}

    return {
        "n": int(len(t)),
        "mae": float(mean_absolute_error(t, p)),
        "rmse": float(np.sqrt(mean_squared_error(t, p))),
        "r2": float(r2_score(t, p)) if len(t) > 1 else float("nan"),
        "directional_accuracy": directional_accuracy(t, p),
        "hit_rate": hit_rate(t, p),
        "information_coefficient": information_coefficient(t, p),
        "mean_prediction": float(np.mean(p)),
        "std_prediction": float(np.std(p)),
        "pct_predicted_positive": float((p > 0).mean()),
    }


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------


def classification_metrics(y_true, y_pred, y_prob=None) -> dict[str, float]:
    """Accuracy, precision, recall, F1 and (when probabilities are given) AUC."""
    t, p = _align(y_true, y_pred)
    if len(t) == 0:
        return {k: float("nan") for k in
                ("n", "accuracy", "precision", "recall", "f1", "roc_auc",
                 "base_rate")}

    t_int, p_int = (t > 0.5).astype(int), (p > 0.5).astype(int)
    out = {
        "n": int(len(t)),
        "accuracy": float(accuracy_score(t_int, p_int)),
        "precision": float(precision_score(t_int, p_int, zero_division=0)),
        "recall": float(recall_score(t_int, p_int, zero_division=0)),
        "f1": float(f1_score(t_int, p_int, zero_division=0)),
        # The score to beat by always predicting the majority class.
        "base_rate": float(t_int.mean()),
        "roc_auc": float("nan"),
    }

    if y_prob is not None:
        tp, pp = _align(y_true, y_prob)
        if len(np.unique((tp > 0.5).astype(int))) > 1:
            out["roc_auc"] = float(roc_auc_score((tp > 0.5).astype(int), pp))
    return out


def confusion(y_true, y_pred) -> pd.DataFrame:
    """2x2 confusion matrix as a labelled frame."""
    from sklearn.metrics import confusion_matrix

    t, p = _align(y_true, y_pred)
    matrix = confusion_matrix((t > 0.5).astype(int), (p > 0.5).astype(int),
                              labels=[0, 1])
    return pd.DataFrame(
        matrix,
        index=pd.Index(["actual down", "actual up"], name=""),
        columns=["predicted down", "predicted up"],
    )


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def evaluate(y_true, y_pred, task: str = "regression", y_prob=None) -> dict[str, float]:
    if task == "classification":
        return classification_metrics(y_true, y_pred, y_prob)
    return regression_metrics(y_true, y_pred)


def metrics_frame(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Turn ``{split_or_model: metrics}`` into a comparison table."""
    return pd.DataFrame(results).T


def sanity_flags(metrics: dict[str, float]) -> list[str]:
    """Warn when a result is too good to be real.

    These thresholds encode the README's honest expectations (section 25).
    Every one of them firing means "look for a bug", not "celebrate".
    """
    flags: list[str] = []
    r2 = metrics.get("r2")
    if r2 is not None and not np.isnan(r2) and r2 > 0.10:
        flags.append(
            f"R2 of {r2:.3f} on daily returns is implausibly high — "
            "check for lookahead bias before trusting it."
        )
    acc = metrics.get("directional_accuracy")
    if acc is not None and not np.isnan(acc) and acc > 0.60:
        flags.append(
            f"Directional accuracy of {acc:.1%} far exceeds the 49-54% "
            "realistic band — treat as a bug until proven otherwise."
        )
    ic = metrics.get("information_coefficient")
    if ic is not None and not np.isnan(ic) and abs(ic) > 0.30:
        flags.append(
            f"Information coefficient of {ic:.3f} is far above the 0.03-0.05 "
            "range typical of real daily signals."
        )
    std = metrics.get("std_prediction")
    if std is not None and not np.isnan(std) and std < 1e-9:
        flags.append(
            "The model predicts a near-constant value — it has learned the "
            "mean and nothing else."
        )
    return flags


def format_metrics(metrics: dict[str, float]) -> str:
    """Compact one-line summary for logs and the CLI."""
    parts = []
    for key, value in metrics.items():
        if isinstance(value, (int, np.integer)):
            parts.append(f"{key}={value}")
        elif value is None or (isinstance(value, float) and np.isnan(value)):
            parts.append(f"{key}=n/a")
        else:
            parts.append(f"{key}={value:.4f}")
    return "  ".join(parts)
