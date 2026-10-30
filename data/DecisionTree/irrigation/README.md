# Synthetic irrigation dataset (Decision Tree)

This data is **simulated, not measured**, generated for the Decision Tree demo as
allowed by the course instructor. Do not present it as real field data.

Regenerate it (identical output, fixed seed 42) from the project root:

```powershell
.\.venv\Scripts\python.exe ml\generate_decision_tree_dataset.py
.\.venv\Scripts\python.exe ml\train_decision_tree.py
```

`irrigation.csv` has 1,600 rows: 698 `wait`, 362 `irrigate_now`, 344 `irrigate_soon`
and 196 `hold`. 3% of labels are deliberately wrong, and soil moisture readings carry
sensor noise. `processed/` holds a stratified 80/20 `train.csv` and `test.csv` plus
`schema.json`.

| Column | Meaning |
|---|---|
| `sample_id` | Row number |
| `soil_moisture_pct` | Soil moisture, % |
| `rain_last_3d_mm` | Rain in the last 3 days, mm |
| `forecast_rain_mm` | Forecast rain for the next 3 days, mm |
| `crop_stage` | 1 seedling, 2 vegetative, 3 flowering, 4 grain fill |
| `air_temp_c` | Air temperature, °C (deliberately irrelevant) |
| `label` | `irrigate_now`, `irrigate_soon`, `wait` or `hold` |

Rules used to label rows: hold if moisture is 36% or more. Wait if moisture is
comfortably above the crop stage's threshold (21% for seedling and vegetative, 28% for
flowering and grain fill), or if rain of 12 mm (past 3 days) or 15 mm (forecast) is
significant. Irrigate now if moisture is well below the threshold, otherwise irrigate
soon. Air temperature has no effect, so the tree should learn to ignore it.
