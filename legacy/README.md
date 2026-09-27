# Legacy code

Original notebooks from the Data Mining project (WiSe 2022/23), kept unchanged
(including their outputs) as a reference and baseline for the refactored project.

| Notebook | Content |
|---|---|
| `Projektcode.ipynb` | Data preparation, exploratory analysis, association rules (apriori), and model comparison: Decision Tree, KNN, Naive Bayes, and a Keras neural network. Each is run on all features and on a reduced set without `backers` / `usd_pledged_real`, with 70/30, 80/20, and 90/10 splits and optional scaling. |
| `Projektcode_Stability.ipynb` | Stability check of the Categorical Naive Bayes model (reduced features, 90/10 split) across three random seeds. |

Environment at the time: Python 3.9.10, pandas 1.x, scikit-learn, mlxtend,
dtreeviz, TensorFlow/Keras. Parts of the code rely on APIs that have since
changed (e.g. positional `DataFrame.pivot` arguments), so the notebooks are not
expected to run as-is on current library versions.

Dataset: `ks-projects-201801.csv` (Kaggle "Kickstarter Projects", 378,661 rows),
not included in the repository.
