# Data

Women's clothing e-commerce review dataset, provided as train/test splits for the
classification challenge. Columns include review title, review text, star rating,
and recommendation indicator.

The CSVs are **not** committed to this repo. To reproduce the analysis, place both
files in this folder with exactly these names:

```
data/
├── train_data.csv   # training reviews
└── test_data.csv    # held-out test reviews
```

The notebook expects them at `../data/train_data.csv` and `../data/test_data.csv`
relative to `notebooks/`.
