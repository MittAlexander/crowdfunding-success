# Data

The project uses the Kaggle dataset
[Kickstarter Projects](https://www.kaggle.com/datasets/kemical/kickstarter-projects)
by Mickaël Mouillé, file `ks-projects-201801.csv` (378,661 projects, 2009 to January 2018).

Download it and place it at:

```
data/raw/ks-projects-201801.csv
```

With the Kaggle CLI:

```bash
kaggle datasets download kemical/kickstarter-projects -f ks-projects-201801.csv -p data/raw --unzip
```

`data/raw/` is git-ignored; the dataset is not committed to the repository.
