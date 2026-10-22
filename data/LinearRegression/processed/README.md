# Prepared Linear Regression datasets

Run `python ml/prepare_linear_regression_data.py` from the project root to
rebuild these files from the original datasets. Raw source files are not
modified.

Each dataset folder contains `X_train.csv`, `y_train.csv`, and the applicable
validation/test equivalents. `X_*.csv` contains numeric, imputed model features;
categorical inputs are one-hot encoded. Imputation values and category columns
are derived from the training split only. The matching `y_*.csv` contains the
numeric `yield` target. `feature_columns.txt` records the feature order.

## Datasets and split choices

- `crop_yield_soil_weather/`: Joins yield rows to state soil and state/year
  weather with validated many-to-one joins. Uses years through 2017 for
  training, 2018 for validation, and 2019 for testing. The 37 rows from 2020
  are excluded because that year is incomplete. The `production` field is
  deliberately excluded from model inputs because it can reveal the yield
  target. Zero-valued targets are retained.
- `cycless/`: Keeps only field IDs shared by the Ratio, VH, and VV source tables
  and checks that yield, crop, and year agree before combining them. For each
  sensor, date measurements are reduced to mean, standard deviation, minimum,
  maximum, and observation count. A deterministic 80/20 split holds out entire
  `grid_ID` groups so a grid cannot appear in both train and test. Field IDs,
  grid IDs, and coordinates are not model inputs.
- `soybean_ndvi_argentina/`, `soybean_ndvi_brazil/`, and
  `soybean_ndvi_usa/`: Keep the source-provided train/dev/test splits and keep
  regions separate. The 32 NDVI positions become numeric features; missing
  feature values are imputed from training medians. SustainBench documents the
  target as metric tonnes per harvested hectare. The local archive does not
  include source metadata, and the original data license is not fully clear.

These files are prepared for experimentation, not evidence that the sources are
fully validated for real-world prediction. The Kaggle Indian yield data
describes yield in tonnes per hectare, but its values do not consistently match
production divided by area and include extreme values; the separately added
soil and weather files have no verified provenance. CYCleSS reports mean
tonnes per hectare per field. SustainBench documents soybean yield in metric
tonnes per harvested hectare, though the original data license remains
uncertain. Check feature availability at the intended prediction time before
using any model outside an experiment.

## Training the Linear Regression baseline

From the project root, run:

```powershell
.\.venv\Scripts\python.exe ml\train_linear_regression.py
```

This fits a `StandardScaler` plus ordinary `LinearRegression` pipeline for each
prepared dataset. For datasets with validation splits, validation metrics are
computed using the training split, then the final model is refit on train plus
validation before its single test evaluation. CYCleSS has no validation split,
so it is fit on train and evaluated on the grid-held-out test split. Metrics
include R-squared, MAE, and RMSE, with a training-mean baseline for comparison.

Models are saved under `trained_models/linear_regression/models/`, test-set
predictions under `trained_models/linear_regression/predictions/`, and
evaluation metrics in `trained_models/linear_regression/metrics.csv` and
`metrics.json`. These are baseline experiments, not deployment-ready models.
