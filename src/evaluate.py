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
