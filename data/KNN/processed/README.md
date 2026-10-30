# Prepared KNN data

Stratified 80/20 split (seed 42) of the generated datasets in `data/kNN/`.
Each folder has `train.csv`, `test.csv` and `schema.json` (feature order, ranges, classes).
Values are unscaled; the scaler is fitted on `train.csv` only during training.
Regenerate with `python ml/prepare_knn_data.py`.
