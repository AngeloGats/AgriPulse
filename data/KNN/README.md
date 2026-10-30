# Synthetic KNN datasets

These datasets are **simulated, not measured**. They were generated for the
K-Nearest Neighbors demo, as allowed by the course instructor. Do not present
them as real field data.

Regenerate them (identical output, fixed seed 42) from the project root:

```powershell
.\.venv\Scripts\python.exe ml\generate_knn_datasets.py
```

## How they were built

- Each class has a profile of typical agronomic values (a mean and spread for
  every feature).
- Rows are drawn from those profiles, with a shared per-row factor so that
  related measurements move together, as they do in real samples.
- Values are clipped to physically sensible ranges and rounded to realistic
  precision.
- Classes **overlap on purpose**, and 2% of rows carry a wrong label (like a
  recording error). Without this, every neighbor would agree and changing K
  would never change a prediction.

## Datasets

| Folder | Rows | Features | Classes | Task |
|---|---|---|---|---|
| `soil_type/` | 1,200 | 7 | sandy, loamy, clay, silty (about 300 each) | Classify soil texture from lab and field measurements |
| `soil_fertility/` | 1,350 | 7 | low, medium, high (450 each) | Classify a soil sample's fertility from nutrient tests |
| `plant_condition/` | 1,400 | 5 | healthy, water_stressed, nutrient_deficient, pest_stressed (350 each) | Classify crop condition from canopy readings and soil moisture |

Each folder holds one CSV named after the folder, plus a `README.md` with the
column dictionary. Every CSV has a `sample_id` column, numeric feature columns
and a `label` column. There are no missing values and no duplicate rows.

## Notes for modeling

- Features are on very different scales (for example rainfall against pH), so
  scale them before running KNN.
- Quick check with scaled features and a stratified 80/20 split: accuracy is
  roughly 90-93% depending on K, against 25-33% for always guessing the biggest
  class. These are early numbers, not final results.
