# soil_type

Synthetic data (see `../README.md`). 1,200 rows, 4 balanced classes.

Task: classify a soil sample's texture class.

| Column | Meaning | Unit | Typical range |
|---|---|---|---|
| `sample_id` | Row number | - | 1-1200 |
| `sand_pct` | Sand share of the mineral soil | % | 1-98 |
| `silt_pct` | Silt share | % | 1-89 |
| `clay_pct` | Clay share | % | 1-80 |
| `moisture_pct` | Soil water content | % | 3-50 |
| `ph` | Soil acidity | pH | 4.5-9.0 |
| `organic_matter_pct` | Organic matter | % | 0.2-8 |
| `bulk_density_g_cm3` | Dry soil weight per volume | g/cm3 | 1.0-1.8 |
| `label` | Texture class | sandy, loamy, clay, silty | - |

`sand_pct`, `silt_pct` and `clay_pct` add up to 100 in every row.
