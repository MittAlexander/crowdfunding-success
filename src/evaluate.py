"""Metrics, probability calibration and evaluation plots."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
    roc_curve,
)

# Fixed categorical order (reference data-viz palette), assigned per model so a
# model keeps its color across every figure. The dummy baseline is drawn in gray.
MODEL_COLORS = {
    "logistic_regression": "#2a78d6",
    "decision_tree": "#eb6834",
    "naive_bayes": "#1baf7a",
    "knn": "#eda100",
    "lightgbm": "#e87ba4",
    "catboost": "#4a3aa7",
    "dummy": "#8a8985",
}
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"


def compute_metrics(y_true, proba, threshold: float = 0.5) -> dict[str, float]:
    """Ranking, probability-quality and thresholded metrics for binary predictions."""
    y_true = np.asarray(y_true)
    proba = np.clip(np.asarray(proba, dtype=float), 1e-15, 1 - 1e-15)
    pred = (proba >= threshold).astype(int)
    single_class = len(np.unique(y_true)) < 2
    return {
        "roc_auc": np.nan if single_class else roc_auc_score(y_true, proba),
        "pr_auc": np.nan if single_class else average_precision_score(y_true, proba),
        "log_loss": log_loss(y_true, proba, labels=[0, 1]),
        "brier": brier_score_loss(y_true, proba),
        "accuracy": accuracy_score(y_true, pred),
        "f1": f1_score(y_true, pred, zero_division=0),
    }


def fast_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> dict:
    """NumPy re-implementation of ``compute_metrics`` for bootstrap loops.

    Gives the same values as the scikit-learn functions (checked in the tests)
    at a fraction of the cost, since it skips input validation.
    """
    y_true = np.asarray(y_true, dtype=int)
    proba = np.clip(np.asarray(proba, dtype=float), 1e-15, 1 - 1e-15)
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    pred = proba >= threshold
    tp = np.sum(pred & (y_true == 1))
    fp = np.sum(pred & (y_true == 0))

    if n_pos == 0 or n_neg == 0:
        roc_auc = pr_auc = np.nan
    else:
        # ROC-AUC via the Mann-Whitney U statistic with tie-averaged ranks.
        order = np.argsort(proba, kind="mergesort")
        sorted_p = proba[order]
        ranks = np.empty(len(proba))
        ranks[order] = np.arange(1, len(proba) + 1)
        _, first, counts = np.unique(sorted_p, return_index=True, return_counts=True)
        avg = first + (counts + 1) / 2  # average 1-based rank of each tie group
        ranks[order] = np.repeat(avg, counts)
        roc_auc = (ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

        # Average precision over distinct thresholds, as in scikit-learn.
        desc = order[::-1]
        y_desc = y_true[desc]
        p_desc = proba[desc]
        last_of_group = np.r_[np.nonzero(np.diff(p_desc))[0], len(p_desc) - 1]
        tps = np.cumsum(y_desc)[last_of_group]
        fps = (last_of_group + 1) - tps
        precision = tps / (tps + fps)
        recall = tps / n_pos
        pr_auc = np.sum(np.diff(np.r_[0, recall]) * precision)

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "log_loss": -np.mean(y_true * np.log(proba) + (1 - y_true) * np.log(1 - proba)),
        "brier": np.mean((proba - y_true) ** 2),
        "accuracy": np.mean(pred == y_true),
        "f1": 2 * tp / (2 * tp + fp + (n_pos - tp)) if (2 * tp + fp + n_pos - tp) else 0.0,
    }


def bootstrap_metrics(y_true, probas: dict[str, np.ndarray], n_resamples: int,
                      seed: int) -> pd.DataFrame:
    """Test metrics on bootstrap resamples of the test set, one row per (resample, model).

    Every model is scored on the same resamples, so differences between models
    can be compared pairwise (see ``compare_to_reference``).
    """
    y_true = np.asarray(y_true)
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_resamples):
        idx = rng.integers(0, len(y_true), len(y_true))
        for name, proba in probas.items():
            rows.append({"resample": i, "model": name,
                         **fast_metrics(y_true[idx], np.asarray(proba)[idx])})
    return pd.DataFrame(rows)


def confidence_intervals(samples: pd.DataFrame, confidence: float) -> pd.DataFrame:
    """Percentile intervals per model: columns ``<metric>_ci_low`` / ``<metric>_ci_high``."""
    alpha = (1 - confidence) / 2
    metrics = [c for c in samples.columns if c not in ("resample", "model")]
    grouped = samples.groupby("model")[metrics]
    low = grouped.quantile(alpha).add_suffix("_ci_low")
    high = grouped.quantile(1 - alpha).add_suffix("_ci_high")
    ordered = [f"{m}_ci_{side}" for m in metrics for side in ("low", "high")]
    return low.join(high)[ordered].reset_index()


# For these metrics lower is better.
LOWER_IS_BETTER = {"log_loss", "brier"}


def compare_to_reference(samples: pd.DataFrame, reference: str, metrics: list[str],
                         confidence: float) -> pd.DataFrame:
    """Paired bootstrap differences (model minus reference) with percentile intervals.

    ``p_reference_better`` is the share of resamples in which the reference model
    scores better; values near 0.5 mean the two cannot be told apart.
    """
    alpha = (1 - confidence) / 2
    wide = samples.pivot(index="resample", columns="model")
    rows = []
    for name in samples["model"].unique():
        if name == reference:
            continue
        for metric in metrics:
            diff = wide[(metric, name)] - wide[(metric, reference)]
            reference_better = diff > 0 if metric in LOWER_IS_BETTER else diff < 0
            rows.append({
                "model": name, "reference": reference, "metric": metric,
                "difference": diff.mean(),
                "ci_low": diff.quantile(alpha), "ci_high": diff.quantile(1 - alpha),
                "p_reference_better": reference_better.mean(),
            })
    return pd.DataFrame(rows)


def calibrate(fitted_model, X_cal, y_cal, method: str) -> CalibratedClassifierCV:
    """Fit a calibration map on a held-out set without refitting the model."""
    calibrated = CalibratedClassifierCV(FrozenEstimator(fitted_model), method=method)
    return calibrated.fit(X_cal, y_cal)


def _style_axes(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=12)
    ax.set_xlabel(xlabel, color=TEXT_SECONDARY)
    ax.set_ylabel(ylabel, color=TEXT_SECONDARY)
    ax.tick_params(colors=TEXT_SECONDARY, length=0)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")


def _line_style(name: str) -> dict:
    style = {"color": MODEL_COLORS.get(name, TEXT_SECONDARY), "linewidth": 2}
    if name == "dummy":
        style["linestyle"] = "--"
    return style


def plot_roc_curves(y_true, probas: dict[str, np.ndarray], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    for name, proba in probas.items():
        fpr, tpr, _ = roc_curve(y_true, proba)
        auc = roc_auc_score(y_true, proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC {auc:.3f})", **_line_style(name))
    _style_axes(ax, "ROC curves on the test set", "False positive rate", "True positive rate")
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_reliability(
    y_true,
    raw: dict[str, np.ndarray],
    calibrated: dict[str, np.ndarray],
    path: Path,
    n_bins: int = 10,
) -> None:
    """Reliability diagrams before and after calibration, one panel each."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.2))
    for ax, probas, title in (
        (axes[0], raw, "Before calibration"),
        (axes[1], calibrated, "After calibration"),
    ):
        ax.plot([0, 1], [0, 1], color=TEXT_SECONDARY, linewidth=1, linestyle=":",
                label="perfectly calibrated")
        for name, proba in probas.items():
            if name == "dummy":
                continue
            frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=n_bins,
                                                    strategy="quantile")
            ax.plot(mean_pred, frac_pos, marker="o", markersize=4, label=name,
                    **_line_style(name))
        _style_axes(ax, title, "Mean predicted probability of success",
                    "Observed success rate")
    axes[1].legend(frameon=False, loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def metrics_table(rows: list[dict]) -> pd.DataFrame:
    table = pd.DataFrame(rows)
    return table.sort_values(["split", "variant", "roc_auc"], ascending=[False, False, False]) \
        .reset_index(drop=True)
