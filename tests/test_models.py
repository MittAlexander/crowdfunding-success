import numpy as np
import pytest
from sklearn.base import clone

from src.data import clean
from src.evaluate import (
    bootstrap_metrics,
    calibrate,
    compare_to_reference,
    compute_metrics,
    confidence_intervals,
    fast_metrics,
)
from src.explain import shap_values
from src.features import build_features
from src.models import MODELS, build_model, param_grid
from tests.synthetic import make_raw


@pytest.fixture(scope="module")
def data():
    df = clean(make_raw(1500, seed=2))
    X, y = build_features(df)
    return X.iloc[:1000], y.iloc[:1000], X.iloc[1000:], y.iloc[1000:]


@pytest.mark.parametrize("name", list(MODELS))
def test_model_fits_and_predicts_probabilities(name, data):
    X_train, y_train, X_test, _ = data
    model = clone(build_model(name, seed=0))  # clone must work for hyperparameter search
    if name == "catboost":
        model.set_params(model__iterations=50)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    assert proba.shape == (len(X_test),)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_models_handle_unseen_categories(data):
    X_train, y_train, X_test, _ = data
    X_new = X_test.copy()
    X_new["category"] = "A category never seen in training"
    X_new["country"] = "ZZ"
    for name in ["logistic_regression", "naive_bayes", "lightgbm"]:
        model = build_model(name, seed=0).fit(X_train, y_train)
        assert np.isfinite(model.predict_proba(X_new)).all()


def test_param_grid_prefixes_estimator_params():
    grid = param_grid({"C": [0.1, 1], "preprocess__num__n_bins": 5})
    assert grid == {"model__C": [0.1, 1], "preprocess__num__n_bins": [5]}


def test_calibration_and_metrics(data):
    X_train, y_train, X_test, y_test = data
    model = build_model("naive_bayes", seed=0).fit(X_train, y_train)
    calibrated = calibrate(model, X_test.iloc[:250], y_test.iloc[:250], "isotonic")
    metrics = compute_metrics(y_test.iloc[250:], calibrated.predict_proba(X_test.iloc[250:])[:, 1])
    assert set(metrics) == {"roc_auc", "pr_auc", "log_loss", "brier", "accuracy", "f1"}
    assert 0.5 < metrics["roc_auc"] <= 1


@pytest.mark.parametrize("name", ["catboost", "lightgbm", "decision_tree"])
def test_shap_values_add_up_to_model_output(name, data):
    X_train, y_train, X_test, _ = data
    model = build_model(name, seed=0)
    if name == "catboost":
        model.set_params(model__iterations=50)
    if name == "decision_tree":
        model.set_params(model__max_depth=4)
    model.fit(X_train, y_train)
    explanation = shap_values(model, X_test.iloc[:50])
    total = explanation.values.sum(axis=1) + explanation.base_values
    proba = model.predict_proba(X_test.iloc[:50])[:, 1]
    # Tree SHAP explains log-odds for the boosters and probabilities for the sklearn tree.
    expected = proba if name == "decision_tree" else np.log(proba / (1 - proba))
    np.testing.assert_allclose(total, expected, atol=1e-4)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_fast_metrics_match_scikit_learn(seed):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, 500)
    # Rounded probabilities create ties, which both implementations must handle alike.
    proba = np.round(np.clip(0.3 * y + rng.random(500) * 0.7, 0, 1), 2)
    expected = compute_metrics(y, proba)
    actual = fast_metrics(y, proba)
    for metric, value in expected.items():
        assert actual[metric] == pytest.approx(value, abs=1e-10), metric


def test_bootstrap_intervals_and_paired_comparison():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    probas = {
        "good": np.clip(0.5 * y + rng.random(2000) * 0.5, 0, 1),
        "random": rng.random(2000),
    }
    samples = bootstrap_metrics(y, probas, n_resamples=200, seed=0)
    ci = confidence_intervals(samples, 0.95).set_index("model")
    assert ci.loc["random", "roc_auc_ci_low"] < 0.5 < ci.loc["random", "roc_auc_ci_high"]
    assert ci.loc["good", "roc_auc_ci_low"] > 0.8

    comparison = compare_to_reference(samples, "good", ["roc_auc", "log_loss"], 0.95)
    auc = comparison.set_index("metric").loc["roc_auc"]
    assert auc["ci_high"] < 0 and auc["p_reference_better"] == 1.0
    # For log loss lower is better, so "good" still wins although the difference is positive.
    loss = comparison.set_index("metric").loc["log_loss"]
    assert loss["ci_low"] > 0 and loss["p_reference_better"] == 1.0
