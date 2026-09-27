import pandas as pd

from src.pipeline import load_config, run
from tests.synthetic import make_raw


def test_pipeline_end_to_end(tmp_path):
    csv = tmp_path / "ks.csv"
    make_raw(3000, seed=3).to_csv(csv, index=False)

    config = load_config("configs/default.yaml")
    config["data"]["path"] = str(csv)
    config["data"]["start_date"] = None
    config["output"]["dir"] = str(tmp_path / "outputs")
    config["cv"]["n_splits"] = 2
    config["explain"]["sample_size"] = 200
    # Keep the search small so the test stays fast.
    config["models"]["decision_tree"]["params"] = {"max_depth": [3, 5]}
    config["models"]["lightgbm"] = {"n_iter": 1, "params": {"n_estimators": 50}}
    config["models"]["catboost"] = {"n_iter": 1, "params": {"iterations": 50}}

    run_dir = run(config)

    metrics = pd.read_csv(run_dir / "metrics.csv")
    expected_models = {"dummy", "logistic_regression", "decision_tree", "naive_bayes",
                       "lightgbm", "catboost"}
    assert set(metrics["model"]) == expected_models
    assert set(zip(metrics["split"], metrics["variant"])) == {
        ("train", "raw"), ("test", "raw"), ("test", "calibrated")}
    test_raw = metrics[(metrics["split"] == "test") & (metrics["variant"] == "raw")]
    assert test_raw.set_index("model").loc["catboost", "roc_auc"] > 0.6

    shap = pd.read_csv(run_dir / "shap_importance.csv")
    assert set(shap["model"]) == {"catboost", "lightgbm"}
    for name in ["roc_curves", "reliability", "shap_beeswarm_catboost", "shap_bar_lightgbm"]:
        assert (run_dir / "figures" / f"{name}.png").exists()
    assert (run_dir / "models" / "catboost_calibrated.joblib").exists()
