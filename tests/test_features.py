import numpy as np
import pandas as pd

from src.data import LEAKAGE_COLUMNS, clean
from src.features import CATEGORICAL, FEATURES, NUMERIC, build_features
from tests.synthetic import make_raw


def test_features_exclude_post_campaign_information():
    X, _ = build_features(clean(make_raw(500)))
    assert list(X.columns) == FEATURES
    assert not set(X.columns) & set(LEAKAGE_COLUMNS)


def test_feature_types_and_values():
    df = clean(make_raw(500))
    X, y = build_features(df)
    assert all(pd.api.types.is_string_dtype(X[c]) for c in CATEGORICAL)
    assert all(np.issubdtype(X[c].dtype, np.floating) for c in NUMERIC)
    assert not X.isna().any().any()
    np.testing.assert_allclose(X["log_goal_usd"], np.log1p(df["usd_goal_real"]))
    assert X["duration_days"].between(0, 61).all()
    assert set(y.unique()) <= {0, 1}
