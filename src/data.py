"""Loading, cleaning and splitting the Kickstarter projects dataset."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

TARGET = "success"

# Columns that are only known once a campaign has ended. Using them as features
# leaks the outcome (the legacy "all data" models reached ~99.9% accuracy this way).
LEAKAGE_COLUMNS = ["pledged", "usd pledged", "usd_pledged_real", "backers", "state"]


def load_raw(path: str | Path) -> pd.DataFrame:
    """Read the raw Kaggle CSV (``ks-projects-201801.csv``)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. See data/README.md for download instructions."
        )
    return pd.read_csv(path, encoding="utf-8", low_memory=False)


def clean(
    raw: pd.DataFrame,
    start_date: str | None = None,
    end_date: str | None = None,
    keep_states: tuple[str, ...] = ("successful", "failed"),
) -> pd.DataFrame:
    """Parse dates, drop placeholder and unfinished projects, and add the binary target.

    Rows are returned sorted by launch time, which the temporal split relies on.
    """
    df = raw.copy()
    df["launched"] = pd.to_datetime(df["launched"])
    df["deadline"] = pd.to_datetime(df["deadline"])

    # Placeholder projects carry a 1970 launch date.
    df = df[df["launched"].dt.year > 1970]
    if start_date:
        df = df[df["launched"] >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df["launched"] < pd.Timestamp(end_date)]

    # Live projects have no outcome yet; canceled/suspended ones are excluded by default.
    df = df[df["state"].isin(keep_states)]
    df[TARGET] = (df["state"] == "successful").astype(int)

    df = df.sort_values("launched", kind="stable").reset_index(drop=True)
    logger.info(
        "Cleaned data: %d projects from %s to %s, success rate %.3f",
        len(df), df["launched"].min(), df["launched"].max(), df[TARGET].mean(),
    )
    return df


@dataclass
class Split:
    train: pd.DataFrame
    calibration: pd.DataFrame
    test: pd.DataFrame


def temporal_split(df: pd.DataFrame, test_size: float, calibration_size: float) -> Split:
    """Split chronologically: oldest projects for training, then calibration, newest for test.

    ``df`` must already be sorted by launch time (``clean`` does this).
    """
    if not 0 < test_size < 1 or not 0 <= calibration_size < 1 or test_size + calibration_size >= 1:
        raise ValueError("test_size and calibration_size must be fractions summing to < 1")
    if not df["launched"].is_monotonic_increasing:
        raise ValueError("DataFrame must be sorted by 'launched' before a temporal split")

    n = len(df)
    test_start = int(round(n * (1 - test_size)))
    cal_start = int(round(n * (1 - test_size - calibration_size)))
    split = Split(
        train=df.iloc[:cal_start],
        calibration=df.iloc[cal_start:test_start],
        test=df.iloc[test_start:],
    )
    for name in ("train", "calibration", "test"):
        part = getattr(split, name)
        if len(part):
            logger.info(
                "%-11s %7d rows  %s -> %s  success rate %.3f",
                name, len(part), part["launched"].min().date(), part["launched"].max().date(),
                part[TARGET].mean(),
            )
    return split
