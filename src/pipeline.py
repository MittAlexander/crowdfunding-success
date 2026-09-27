"""End-to-end pipeline: data -> features -> tuned models -> calibration -> evaluation -> SHAP.

Usage:
    python -m src.pipeline --config configs/default.yaml [--models catboost lightgbm]
"""
from __future__ import annotations

import argparse
import json
import logging
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    TimeSeriesSplit,
)

from src import evaluate, explain
from src.data import clean, load_raw, temporal_split
from src.features import build_features
from src.models import MODELS, build_model, param_grid

logger = logging.getLogger("src")


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _cv_splitter(cfg: dict, seed: int):
    if cfg["strategy"] == "time":
        return TimeSeriesSplit(n_splits=cfg["n_splits"])
    if cfg["strategy"] == "stratified":
        return StratifiedKFold(n_splits=cfg["n_splits"], shuffle=True, random_state=seed)
    raise ValueError(f"Unknown cv.strategy '{cfg['strategy']}' (use 'time' or 'stratified')")


def _setup_output(config: dict, config_path: Path) -> Path:
    run_dir = Path(config["output"]["dir"]) / datetime.now().strftime("%Y%m%d-%H%M%S")
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    (run_dir / "models").mkdir(exist_ok=True)
    (run_dir / "cv_results").mkdir(exist_ok=True)
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    logger.info("Run directory: %s (config from %s)", run_dir, config_path)
    return run_dir


def _setup_logging(run_dir: Path | None = None) -> None:
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        console = logging.StreamHandler()
        console.setFormatter(fmt)
        logger.addHandler(console)
    if run_dir is not None:
        file_handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)


def tune(name: str, model_cfg: dict, X, y, cv_cfg: dict, seed: int, n_jobs: int):
    """Hyperparameter search on the training set; returns the fitted search object."""
    grid = param_grid(model_cfg.get("params"))
    parallel = MODELS[name][1]
    common = dict(
        scoring=cv_cfg["scoring"],
        cv=_cv_splitter(cv_cfg, seed),
        n_jobs=n_jobs if parallel else 1,
        refit=True,
        return_train_score=True,
        error_score="raise",
    )
    n_iter = model_cfg.get("n_iter")
    if n_iter:
        search = RandomizedSearchCV(build_model(name, seed), grid, n_iter=n_iter,
                                    random_state=seed, **common)
    else:
        search = GridSearchCV(build_model(name, seed), grid, **common)
    return search.fit(X, y)


def _add_confidence_intervals(metrics: pd.DataFrame, y_test, raw_probas: dict,
                              cal_probas: dict, boot_cfg: dict, seed: int,
                              run_dir: Path) -> pd.DataFrame:
    """Attach bootstrap intervals to the test rows and write pairwise model comparisons."""
    n_resamples, confidence = boot_cfg["n_resamples"], boot_cfg.get("confidence", 0.95)
    logger.info("Bootstrapping test metrics (%d resamples) ...", n_resamples)
    intervals, comparisons = [], []
    for variant, probas in (("raw", raw_probas), ("calibrated", cal_probas)):
        samples = evaluate.bootstrap_metrics(y_test, probas, n_resamples, seed)
        intervals.append(evaluate.confidence_intervals(samples, confidence)
                         .assign(split="test", variant=variant))
        # Compare every model with the best one of this variant by test ROC-AUC.
        observed = metrics[(metrics["split"] == "test") & (metrics["variant"] == variant)]
        reference = observed.loc[observed["roc_auc"].idxmax(), "model"]
        comparisons.append(evaluate.compare_to_reference(
            samples, reference, ["roc_auc", "log_loss", "brier"], confidence,
        ).assign(variant=variant))
    pd.concat(comparisons).to_csv(run_dir / "model_comparison.csv", index=False)
    return metrics.merge(pd.concat(intervals), on=["model", "split", "variant"], how="left")


def run(config: dict, config_path: Path = Path("<in-memory>"),
        only_models: list[str] | None = None) -> Path:
    # Expected: duration and launch hour have few distinct values, so some quantile
    # bins for Naive Bayes collapse and are merged.
    warnings.filterwarnings("ignore", message="Bins whose width are too small")
    _setup_logging()
    run_dir = _setup_output(config, config_path)
    _setup_logging(run_dir)
    seed = config["seed"]

    data_cfg = config["data"]
    df = clean(
        load_raw(data_cfg["path"]),
        start_date=data_cfg.get("start_date"),
        end_date=data_cfg.get("end_date"),
        keep_states=tuple(data_cfg["keep_states"]),
    )
    split = temporal_split(df, config["split"]["test_size"], config["split"]["calibration_size"])
    X_train, y_train = build_features(split.train)
    X_cal, y_cal = build_features(split.calibration)
    X_test, y_test = build_features(split.test)

    enabled = [
        name for name, cfg in config["models"].items()
        if cfg.get("enabled", True) and (only_models is None or name in only_models)
    ]
    rows, raw_probas, cal_probas, fitted, best_params = [], {}, {}, {}, {}

    for name in enabled:
        start = time.perf_counter()
        logger.info("Tuning %s ...", name)
        search = tune(name, config["models"][name], X_train, y_train, config["cv"], seed,
                      config.get("n_jobs", -1))
        model = search.best_estimator_
        pd.DataFrame(search.cv_results_).to_csv(run_dir / "cv_results" / f"{name}.csv",
                                                index=False)
        best_params[name] = {k.removeprefix("model__"): v for k, v in search.best_params_.items()}

        calibrated = evaluate.calibrate(model, X_cal, y_cal, config["calibration"]["method"])
        raw_probas[name] = model.predict_proba(X_test)[:, 1]
        cal_probas[name] = calibrated.predict_proba(X_test)[:, 1]
        fitted[name] = model

        elapsed = time.perf_counter() - start
        base = {"model": name, f"cv_{config['cv']['scoring']}": search.best_score_,
                "fit_seconds": round(elapsed, 1)}
        rows.append({**base, "split": "train", "variant": "raw",
                     **evaluate.compute_metrics(y_train, model.predict_proba(X_train)[:, 1])})
        rows.append({**base, "split": "test", "variant": "raw",
                     **evaluate.compute_metrics(y_test, raw_probas[name])})
        rows.append({**base, "split": "test", "variant": "calibrated",
                     **evaluate.compute_metrics(y_test, cal_probas[name])})
        joblib.dump(calibrated, run_dir / "models" / f"{name}_calibrated.joblib")
        logger.info("%s done in %.1fs: best params %s, test ROC-AUC %.4f",
                    name, elapsed, best_params[name], rows[-2]["roc_auc"])

    metrics = evaluate.metrics_table(rows)
    boot_cfg = config.get("bootstrap", {})
    if raw_probas and boot_cfg.get("n_resamples", 0) > 0:
        metrics = _add_confidence_intervals(metrics, y_test, raw_probas, cal_probas, boot_cfg,
                                            seed, run_dir)
    metrics.to_csv(run_dir / "metrics.csv", index=False)
    with open(run_dir / "best_params.json", "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2, default=str)

    if raw_probas:
        evaluate.plot_roc_curves(y_test, raw_probas, run_dir / "figures" / "roc_curves.png")
        evaluate.plot_reliability(y_test, raw_probas, cal_probas,
                                  run_dir / "figures" / "reliability.png")

    explain_cfg = config.get("explain", {})
    importances = []
    for name in explain_cfg.get("models", []):
        if name not in fitted:
            continue
        if name not in explain.SUPPORTED:
            logger.warning("SHAP is only set up for %s; skipping %s", explain.SUPPORTED, name)
            continue
        importances.append(explain.explain_model(
            name, fitted[name], X_test, run_dir / "figures",
            explain_cfg.get("sample_size", 2000), seed,
        ))
    if importances:
        pd.concat(importances).to_csv(run_dir / "shap_importance.csv", index=False)

    test_view = metrics.loc[metrics["split"] == "test",
                            [c for c in metrics.columns if "_ci_" not in c]]
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        logger.info("Test-set results:\n%s", test_view.round(4).to_string(index=False))
    logger.info("Finished. Outputs in %s", run_dir)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/default.yaml", type=Path)
    parser.add_argument("--models", nargs="+", help="Run only these models (must be enabled)")
    args = parser.parse_args()
    run(load_config(args.config), args.config, args.models)


if __name__ == "__main__":
    main()
