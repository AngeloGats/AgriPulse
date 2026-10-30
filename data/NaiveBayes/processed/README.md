# Prepared Naive Bayes datasets

Run `python ml/prepare_naive_bayes_data.py` from the project root to rebuild
these files. Raw source files in `data/NaiveBayes/` are never modified.

Each folder has `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`,
`feature_columns.txt` (feature order) and `schema.json`.

## `soybean_disease/` (for `CategoricalNB`)

- Source: UCI Large Soybean Database. The official train file (307 rows) and
  test file (376 rows) are kept as the train/test split.
- 35 categorical symptom and condition features, stored as integer codes. The
  code is the position of the value in `schema.json` (`features`).
- A blank cell means the symptom was not recorded. Do not treat it as a value.
- 15 diseases are kept. `diaporthe-pod-&-stem-blight`, `cyst-nematode`,
  `2-4-d-injury` and `herbicide-injury` are excluded because they have only
  1-6 training rows. Result: 290 train rows, 340 test rows.

## `crop_recommendation/` (for `GaussianNB`)

- Source: Crop Recommendation dataset (2,200 rows, 22 crops, 100 rows each).
- Features: N, P, K, temperature, humidity, ph, rainfall. Labels are lower-case
  crop names.
- Stratified 80/20 split with `random_state=42`: 1,760 train rows and 440 test
  rows. `schema.json` holds each feature's range for the demo form.

`preparation_summary.csv` lists the row counts and missing-value rates.
