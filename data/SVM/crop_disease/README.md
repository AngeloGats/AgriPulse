# Synthetic crop-disease dataset (SVM)

This data is **simulated, not measured**, generated for the Support Vector Machine
demo as allowed by the course instructor. Do not present it as real field data.

Regenerate it (identical output, fixed seed 42) from the project root:

```powershell
.\.venv\Scripts\python.exe ml\generate_svm_dataset.py
.\.venv\Scripts\python.exe ml\train_svm.py
```

`crop_disease.csv` has 1,200 rows: 710 `healthy` and 490 `diseased`. 3% of labels are
deliberately wrong, like recording errors. `processed/` holds a stratified 80/20
`train.csv` and `test.csv` plus `schema.json`.

| Column | Meaning |
|---|---|
| `sample_id` | Row number |
| `air_temp_c` | Air temperature, °C |
| `humidity_pct` | Relative humidity, % |
| `leaf_wetness_hours` | Hours per day the leaves stay wet |
| `red_edge_index` | Leaf stress index from the red-edge band (lower = more stressed) |
| `label` | `healthy` or `diseased` |

Diseased samples sit in a middle band of temperature with high humidity. Healthy
samples surround that band (cool and humid, hot, or dry), so a straight line cannot
separate the classes but a curved (RBF) boundary can.
