import pandas as pd
import pytest

from src.data import TARGET, clean, temporal_split
from tests.synthetic import make_raw


@pytest.fixture
def cleaned():
    return clean(make_raw(3000, seed=1))


def test_clean_keeps_only_finished_projects(cleaned):
    assert set(cleaned["state"]) == {"successful", "failed"}
    assert (cleaned[TARGET] == (cleaned["state"] == "successful")).all()


def test_clean_drops_placeholder_dates_and_sorts(cleaned):
    assert cleaned["launched"].dt.year.min() > 1970
    assert cleaned["launched"].is_monotonic_increasing


def test_clean_date_filter():
    df = clean(make_raw(3000, seed=1), start_date="2017-01-01", end_date="2017-07-01")
    assert df["launched"].min() >= pd.Timestamp("2017-01-01")
    assert df["launched"].max() < pd.Timestamp("2017-07-01")


def test_temporal_split_is_chronological_and_complete(cleaned):
    split = temporal_split(cleaned, test_size=0.2, calibration_size=0.1)
    assert len(split.train) + len(split.calibration) + len(split.test) == len(cleaned)
    assert split.train["launched"].max() <= split.calibration["launched"].min()
    assert split.calibration["launched"].max() <= split.test["launched"].min()
    assert len(split.test) == pytest.approx(0.2 * len(cleaned), abs=1)


def test_temporal_split_rejects_unsorted(cleaned):
    with pytest.raises(ValueError):
        temporal_split(cleaned.sample(frac=1, random_state=0), 0.2, 0.1)
