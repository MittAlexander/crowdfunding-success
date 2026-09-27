"""Feature engineering. Only information available at launch time is used."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import TARGET

CATEGORICAL = ["main_category", "category", "country", "launch_month", "launch_weekday"]
NUMERIC = ["log_goal_usd", "duration_days", "launch_hour"]
FEATURES = CATEGORICAL + NUMERIC


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return the feature matrix ``X`` and target ``y`` for cleaned project rows.

    Categorical columns are returned as strings; each model pipeline encodes them
    in the way that suits it (one-hot, ordinal for Naive Bayes, or natively).
    """
    X = pd.DataFrame(index=df.index)
    X["main_category"] = df["main_category"].astype(str)
    X["category"] = df["category"].astype(str)
    X["country"] = df["country"].astype(str)
    # Month and weekday are cyclic with no meaningful order, so they are treated
    # as categories rather than numbers.
    X["launch_month"] = df["launched"].dt.month.astype(str)
    X["launch_weekday"] = df["launched"].dt.dayofweek.astype(str)

    # Goals span several orders of magnitude (0.01 to 1e8 USD).
    X["log_goal_usd"] = np.log1p(df["usd_goal_real"].astype(float))
    X["duration_days"] = (df["deadline"] - df["launched"]).dt.days.astype(float)
    X["launch_hour"] = df["launched"].dt.hour.astype(float)

    return X[FEATURES], df[TARGET].astype(int)
