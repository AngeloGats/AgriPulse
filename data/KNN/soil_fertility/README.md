# soil_fertility

Synthetic data (see `../README.md`). 1,350 rows, 3 balanced classes.

Task: classify a soil sample as low, medium or high fertility from nutrient tests.

| Column | Meaning | Unit | Typical range |
|---|---|---|---|
| `sample_id` | Row number | - | 1-1350 |
| `ph` | Soil acidity | pH | 4.5-8.8 |
| `organic_carbon_pct` | Organic carbon | % | 0.1-2.5 |
| `nitrogen_kg_ha` | Available nitrogen | kg/ha | 40-500 |
| `phosphorus_kg_ha` | Available phosphorus | kg/ha | 1-60 |
| `potassium_kg_ha` | Available potassium | kg/ha | 30-450 |
| `ec_ds_m` | Electrical conductivity (salts) | dS/m | 0.05-1.6 |
| `zinc_ppm` | Available zinc | ppm | 0.1-3.0 |
| `label` | Fertility class | low, medium, high | - |
