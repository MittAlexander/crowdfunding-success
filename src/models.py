"""Model pipelines. Each model gets the categorical encoding that suits it:

* linear / distance / tree models from scikit-learn: one-hot encoding
  (the legacy LabelEncoder imposed an artificial order on nominal features),
* Categorical Naive Bayes: ordinal codes for categories plus quantile-binned numerics
  (the legacy version fed raw continuous values in as if they were categories),
* LightGBM and CatBoost: native categorical handling.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import CategoricalNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    KBinsDiscretizer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)
from sklearn.tree import DecisionTreeClassifier

from src.features import CATEGORICAL, NUMERIC


def _shift_unknown_to_zero(X: np.ndarray) -> np.ndarray:
    # OrdinalEncoder maps unseen categories to -1; CategoricalNB needs non-negative
    # codes, so shift everything by one. Code 0 is never seen in training and gets
    # the smoothed (alpha) probability for both classes.
    return X + 1


class CatBoostCategorical(CatBoostClassifier):
    """CatBoost that declares the categorical columns at fit time.

    Passing ``cat_features`` to the constructor breaks ``sklearn.base.clone``
    (CatBoost stores a modified copy), which hyperparameter search relies on.
    """

    def fit(self, X, y=None, **fit_params):
        fit_params.setdefault("cat_features", [c for c in CATEGORICAL if c in X.columns])
        return super().fit(X, y, **fit_params)


def _as_pandas_category(X):
    return X.astype({col: "category" for col in CATEGORICAL})


def _one_hot(scale_numeric: bool) -> ColumnTransformer:
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("num", StandardScaler() if scale_numeric else "passthrough", NUMERIC),
    ])


def _naive_bayes_encoding() -> ColumnTransformer:
    categorical = Pipeline([
        ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ("shift", FunctionTransformer(_shift_unknown_to_zero)),
    ])
    return ColumnTransformer([
        ("cat", categorical, CATEGORICAL),
        ("num", KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="quantile",
                                 quantile_method="averaged_inverted_cdf"), NUMERIC),
    ])


def _dummy(seed: int) -> Pipeline:
    return Pipeline([("model", DummyClassifier(strategy="prior"))])


def _logistic_regression(seed: int) -> Pipeline:
    return Pipeline([
        ("preprocess", _one_hot(scale_numeric=True)),
        ("model", LogisticRegression(max_iter=2000)),
    ])


def _decision_tree(seed: int) -> Pipeline:
    return Pipeline([
        ("preprocess", _one_hot(scale_numeric=False)),
        ("model", DecisionTreeClassifier(random_state=seed)),
    ])


def _naive_bayes(seed: int) -> Pipeline:
    return Pipeline([
        ("preprocess", _naive_bayes_encoding()),
        ("model", CategoricalNB()),
    ])


def _knn(seed: int) -> Pipeline:
    return Pipeline([
        ("preprocess", _one_hot(scale_numeric=True)),
        ("model", KNeighborsClassifier()),
    ])


def _lightgbm(seed: int) -> Pipeline:
    return Pipeline([
        ("preprocess", FunctionTransformer(_as_pandas_category)),
        ("model", LGBMClassifier(random_state=seed, verbose=-1)),
    ])


def _catboost(seed: int) -> Pipeline:
    return Pipeline([
        ("model", CatBoostCategorical(
            random_seed=seed,
            verbose=0,
            allow_writing_files=False,
        )),
    ])


# name -> (builder, whether the search itself should run folds in parallel).
# Boosting libraries already use all cores internally.
MODELS: dict[str, tuple[Callable[[int], Pipeline], bool]] = {
    "dummy": (_dummy, True),
    "logistic_regression": (_logistic_regression, True),
    "decision_tree": (_decision_tree, True),
    "naive_bayes": (_naive_bayes, True),
    "knn": (_knn, True),
    "lightgbm": (_lightgbm, False),
    "catboost": (_catboost, False),
}


def build_model(name: str, seed: int) -> Pipeline:
    if name not in MODELS:
        raise KeyError(f"Unknown model '{name}'. Available: {', '.join(MODELS)}")
    return MODELS[name][0](seed)


def param_grid(params: dict | None) -> dict[str, list]:
    """Translate config params to pipeline parameter names.

    Plain keys refer to the estimator (``C`` -> ``model__C``); keys that already
    contain ``__`` address another pipeline step, e.g.
    ``preprocess__num__n_bins``. Scalars become one-element lists.
    """
    grid = {}
    for key, values in (params or {}).items():
        full_key = key if "__" in key else f"model__{key}"
        grid[full_key] = values if isinstance(values, list) else [values]
    return grid
