"""SHAP explanations for the tree-based models."""
from __future__ import annotations

import logging
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from catboost import CatBoostClassifier, Pool

from src.features import CATEGORICAL

logger = logging.getLogger(__name__)

SUPPORTED = ("catboost", "lightgbm", "decision_tree")


def shap_values(pipeline, X: pd.DataFrame) -> shap.Explanation:
    """SHAP values (log-odds of success) for a fitted pipeline ending in a tree model.

    Explanations refer to the uncalibrated model: calibration is a monotonic map
    on top of its output and does not change which features drive a prediction.
    """
    model = pipeline.named_steps["model"]
    if "preprocess" in pipeline.named_steps:
        X_model = pipeline.named_steps["preprocess"].transform(X)
    else:
        X_model = X

    if isinstance(model, CatBoostClassifier):
        # CatBoost computes exact SHAP values natively, including for categorical features.
        pool = Pool(X_model, cat_features=model.get_cat_feature_indices())
        raw = model.get_feature_importance(pool, type="ShapValues")
        return shap.Explanation(
            values=raw[:, :-1],
            base_values=raw[:, -1],
            data=_display_data(X),
            feature_names=list(X.columns),
        )

    if hasattr(X_model, "toarray"):  # one-hot output of the scikit-learn tree
        X_model = X_model.toarray()
    feature_names = (
        list(pipeline.named_steps["preprocess"].get_feature_names_out())
        if not isinstance(X_model, pd.DataFrame) else list(X_model.columns)
    )
    explainer = shap.TreeExplainer(model)
    explanation = explainer(X_model)
    if explanation.values.ndim == 3:  # per-class output: keep the "success" class
        explanation = explanation[:, :, 1]
    explanation.feature_names = feature_names
    if isinstance(X_model, pd.DataFrame):
        explanation.data = _display_data(X_model)
    return explanation


def _display_data(X: pd.DataFrame) -> np.ndarray:
    """Feature values for plot coloring: numeric columns as-is, categorical ones as NaN.

    SHAP colors points by feature value, which only makes sense for numbers;
    NaN makes categorical features render in neutral gray.
    """
    display = X.copy()
    for col in CATEGORICAL:
        if col in display:
            display[col] = np.nan
    return display.astype(float).to_numpy()


def explain_model(name: str, pipeline, X: pd.DataFrame, out_dir: Path, sample_size: int,
                  seed: int) -> pd.DataFrame:
    """Write a beeswarm and a bar plot, and return mean |SHAP| per feature."""
    if len(X) > sample_size:
        X = X.sample(sample_size, random_state=seed)
    explanation = shap_values(pipeline, X)

    importance = (
        pd.DataFrame({
            "feature": explanation.feature_names,
            "mean_abs_shap": np.abs(explanation.values).mean(axis=0),
        })
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    importance.insert(0, "model", name)

    for kind in ("beeswarm", "bar"):
        plt.figure()
        with warnings.catch_warnings():
            # Categorical features have NaN display values (drawn gray) by design.
            warnings.filterwarnings("ignore", message="All-NaN slice encountered")
            if kind == "beeswarm":
                shap.plots.beeswarm(explanation, max_display=15, show=False)
            else:
                shap.plots.bar(explanation, max_display=15, show=False)
        plt.title(f"{name}: SHAP {kind} (test sample, n={len(X)})", loc="left", fontsize=12)
        plt.tight_layout()
        plt.savefig(out_dir / f"shap_{kind}_{name}.png", dpi=150, bbox_inches="tight")
        plt.close("all")

    logger.info("SHAP for %s: top features %s", name, ", ".join(importance["feature"].head(5)))
    return importance
