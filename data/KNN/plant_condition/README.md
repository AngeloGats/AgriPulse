# plant_condition

Synthetic data (see `../README.md`). 1,400 rows, 4 balanced classes.

Task: classify a crop's condition from canopy readings and soil moisture.

| Column | Meaning | Unit | Typical range |
|---|---|---|---|
| `sample_id` | Row number | - | 1-1400 |
| `ndvi` | Greenness index from imagery (higher is greener) | index | 0.15-0.95 |
| `chlorophyll_spad` | Leaf chlorophyll reading (SPAD meter) | SPAD | 10-60 |
| `canopy_temp_excess_c` | Canopy temperature above air temperature | degrees C | -2 to 10 |
| `soil_moisture_pct` | Soil water content | % | 3-45 |
| `leaf_damage_pct` | Leaf area with visible damage | % | 0-60 |
| `label` | Condition class | healthy, water_stressed, nutrient_deficient, pest_stressed | - |
