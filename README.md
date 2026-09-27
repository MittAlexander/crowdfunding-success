# crowdfunding-success

An analysis on the success of Kickstarter projects I first conducted during my Bachelor's degree. This is an extended and refactored version.

The question: **given only what is known when a project launches, how likely is it to reach its funding goal?**

## Quick start

```bash
pip install -r requirements.txt
# download the dataset first, see data/README.md
python -m src.pipeline --config configs/default.yaml
python -m src.pipeline --models catboost lightgbm   # run a subset of models
python -m pytest                                     # tests run on synthetic data
```

Every run writes to `outputs/<timestamp>/`:

| File | Content |
|---|---|
| `metrics.csv` | Train and test metrics for each model, before and after calibration, with 95% bootstrap intervals for the test metrics |
| `model_comparison.csv` | Paired bootstrap differences between each model and the best one (ROC-AUC, log loss, Brier) |
| `best_params.json`, `cv_results/` | Chosen hyperparameters and full search results |
| `figures/roc_curves.png` | ROC curves on the test set |
| `figures/reliability.png` | Reliability diagrams before and after calibration |
| `figures/shap_*.png`, `shap_importance.csv` | SHAP beeswarm and bar plots, mean \|SHAP\| per feature |
| `models/*_calibrated.joblib` | Fitted, calibrated pipelines |
| `config.yaml`, `run.log` | The exact configuration and log of the run |

## Method

1. **Data** (`src/data.py`): projects that ended as `successful` or `failed`, launched from
   2017 on by default (the legacy analysis used the same period). Placeholder 1970 dates are
   dropped, and live, canceled, suspended and undefined projects are excluded.
2. **Features** (`src/features.py`): only information available at launch. These are main
   category, category, country, launch month, launch weekday, log goal in USD, campaign
   duration and launch hour. `backers` and pledged amounts are excluded because they are
   only known after the campaign ends.
3. **Split**: chronological. The oldest 70% of projects are used for training, the next 10%
   for probability calibration, and the newest 20% for the test set.
4. **Models** (`src/models.py`): each model gets an encoding that suits it.

   | Model | Categorical encoding |
   |---|---|
   | Dummy baseline (predicts the base rate) | – |
   | Logistic regression | one-hot, numerics standardized |
   | Decision tree | one-hot |
   | Categorical Naive Bayes | ordinal codes, numerics quantile-binned |
   | KNN (off by default) | one-hot, numerics standardized |
   | LightGBM | native categorical support |
   | CatBoost | native categorical support (ordered target statistics) |

5. **Tuning**: grid or random search on the training set with time-series cross-validation,
   scored by ROC-AUC. The test set is not used for any decision.
6. **Calibration** (`src/evaluate.py`): isotonic (or sigmoid) regression fitted on the
   calibration set, on top of the frozen model.
7. **Evaluation**: ROC-AUC, PR-AUC, log loss, Brier score, accuracy and F1, on the test set.
   Uncertainty comes from 1000 bootstrap resamples of the test set, giving percentile
   intervals per model. All models are scored on the same resamples, so differences
   between two models are compared pairwise. This is far more precise than checking
   whether their individual intervals overlap.
8. **Explanation** (`src/explain.py`): SHAP values for CatBoost (native, exact) and LightGBM
   (TreeSHAP).

All settings live in `configs/default.yaml`.

## Changes from the original project

The original notebooks are in [`legacy/`](legacy/). The main changes are:

- **No leakage.** The legacy "all data" models reached ~99.9% accuracy because they used
  `backers` and `usd_pledged_real`. Those numbers do not reflect real predictive power; the
  comparable legacy results are the "reduced" models at 63–68% accuracy.
- **No model selection on the test set.** Hyperparameters are chosen by cross-validation
  on the training data only.
- **Proper categorical encoding.** The legacy code used `LabelEncoder`, which gives nominal
  features an artificial order. It also fed continuous values to `CategoricalNB` as
  categories.
- **Temporal instead of random split**, which matches how the model would be used.
- **Baseline and richer metrics** in addition to accuracy, plus calibrated probabilities.
- **The stability analysis bug is fixed by design.** The legacy notebook predicted
  random state 3 with the model from random state 1. The new pipeline has one code path
  per model instead of copied cells.

## Project layout

```
configs/default.yaml   experiment configuration
src/data.py            loading, cleaning, temporal split
src/features.py        launch-time features
src/models.py          model pipelines and encodings
src/evaluate.py        metrics, calibration, plots
src/explain.py         SHAP
src/pipeline.py        entry point
tests/                 unit and end-to-end tests on synthetic data
legacy/                original notebooks (WiSe 2022/23)
```
