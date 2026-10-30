"""Generate the synthetic KNN datasets in data/kNN.

The data is simulated, not measured. Class profiles are based on typical
agronomic ranges, and classes deliberately overlap (plus a small share of
mislabeled rows) so that neighbors can disagree and K changes some predictions.
Running the script again with the same seed reproduces the files exactly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "kNN"
SEED = 42
LABEL_NOISE = 0.02  # share of rows given a wrong label, like recording errors


def _sample(
    rng: np.random.Generator,
    profile: dict[str, dict[str, tuple[float, float]]],
    rows_per_class: int,
    shared_shift: float,
) -> pd.DataFrame:
    """Draw rows per class from normal profiles with a shared per-row factor.

    The shared factor makes features move together, as real measurements do.
    """
    frames = []
    for label, features in profile.items():
        shift = rng.normal(0, shared_shift, size=rows_per_class)
        data = {}
        for column, (mean, sd) in features.items():
            data[column] = rng.normal(mean, sd, size=rows_per_class) + shift * sd
        frame = pd.DataFrame(data)
        frame["label"] = label
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _add_label_noise(rng: np.random.Generator, frame: pd.DataFrame) -> pd.DataFrame:
    labels = sorted(frame["label"].unique())
    flip = rng.random(len(frame)) < LABEL_NOISE
    for index in frame.index[flip]:
        choices = [label for label in labels if label != frame.at[index, "label"]]
        frame.at[index, "label"] = rng.choice(choices)
    return frame


def _finish(rng: np.random.Generator, frame: pd.DataFrame, decimals: dict[str, int]) -> pd.DataFrame:
    frame = _add_label_noise(rng, frame)
    frame = frame.sample(frac=1, random_state=SEED).reset_index(drop=True)
    for column, places in decimals.items():
        frame[column] = frame[column].round(places)
    frame.insert(0, "sample_id", np.arange(1, len(frame) + 1))
    return frame


def soil_type(rng: np.random.Generator) -> pd.DataFrame:
    """Classify soil texture class from lab and field measurements."""
    profile = {
        "sandy": {
            "sand_pct": (70, 11), "silt_pct": (18, 8), "clay_pct": (12, 6),
            "moisture_pct": (12, 5), "ph": (6.0, 0.8), "organic_matter_pct": (1.2, 0.6),
            "bulk_density_g_cm3": (1.55, 0.10),
        },
        "loamy": {
            "sand_pct": (42, 11), "silt_pct": (38, 10), "clay_pct": (20, 8),
            "moisture_pct": (22, 6), "ph": (6.6, 0.6), "organic_matter_pct": (3.0, 1.0),
            "bulk_density_g_cm3": (1.35, 0.10),
        },
        "clay": {
            "sand_pct": (24, 11), "silt_pct": (28, 10), "clay_pct": (48, 11),
            "moisture_pct": (32, 7), "ph": (7.2, 0.7), "organic_matter_pct": (3.5, 1.2),
            "bulk_density_g_cm3": (1.22, 0.10),
        },
        "silty": {
            "sand_pct": (18, 9), "silt_pct": (62, 11), "clay_pct": (20, 8),
            "moisture_pct": (26, 6), "ph": (6.8, 0.6), "organic_matter_pct": (2.8, 1.0),
            "bulk_density_g_cm3": (1.30, 0.10),
        },
    }
    frame = _sample(rng, profile, rows_per_class=300, shared_shift=0.15)
    # Texture fractions must be positive and add up to 100%.
    texture = frame[["sand_pct", "silt_pct", "clay_pct"]].clip(lower=1)
    frame[["sand_pct", "silt_pct", "clay_pct"]] = texture.div(texture.sum(axis=1), axis=0) * 100
    frame["moisture_pct"] = frame["moisture_pct"].clip(lower=3, upper=50)
    frame["ph"] = frame["ph"].clip(4.5, 9.0)
    frame["organic_matter_pct"] = frame["organic_matter_pct"].clip(lower=0.2, upper=8)
    frame["bulk_density_g_cm3"] = frame["bulk_density_g_cm3"].clip(1.0, 1.8)
    return _finish(
        rng,
        frame,
        {
            "sand_pct": 1, "silt_pct": 1, "clay_pct": 1, "moisture_pct": 1, "ph": 2,
            "organic_matter_pct": 2, "bulk_density_g_cm3": 2,
        },
    )


def soil_fertility(rng: np.random.Generator) -> pd.DataFrame:
    """Classify a soil sample as low, medium or high fertility from nutrient tests."""
    profile = {
        "low": {
            "ph": (5.6, 0.9), "organic_carbon_pct": (0.45, 0.17),
            "nitrogen_kg_ha": (150, 38), "phosphorus_kg_ha": (8, 4),
            "potassium_kg_ha": (110, 38), "ec_ds_m": (0.25, 0.12), "zinc_ppm": (0.5, 0.22),
        },
        "medium": {
            "ph": (6.4, 0.65), "organic_carbon_pct": (0.75, 0.22),
            "nitrogen_kg_ha": (235, 42), "phosphorus_kg_ha": (16, 6),
            "potassium_kg_ha": (190, 48), "ec_ds_m": (0.45, 0.16), "zinc_ppm": (0.95, 0.3),
        },
        "high": {
            "ph": (6.8, 0.55), "organic_carbon_pct": (1.15, 0.3),
            "nitrogen_kg_ha": (320, 48), "phosphorus_kg_ha": (27, 8),
            "potassium_kg_ha": (280, 55), "ec_ds_m": (0.7, 0.2), "zinc_ppm": (1.4, 0.4),
        },
    }
    frame = _sample(rng, profile, rows_per_class=450, shared_shift=0.25)
    frame["ph"] = frame["ph"].clip(4.5, 8.8)
    frame["organic_carbon_pct"] = frame["organic_carbon_pct"].clip(lower=0.1, upper=2.5)
    frame["nitrogen_kg_ha"] = frame["nitrogen_kg_ha"].clip(lower=40, upper=500)
    frame["phosphorus_kg_ha"] = frame["phosphorus_kg_ha"].clip(lower=1, upper=60)
    frame["potassium_kg_ha"] = frame["potassium_kg_ha"].clip(lower=30, upper=450)
    frame["ec_ds_m"] = frame["ec_ds_m"].clip(lower=0.05, upper=1.6)
    frame["zinc_ppm"] = frame["zinc_ppm"].clip(lower=0.1, upper=3.0)
    return _finish(
        rng,
        frame,
        {
            "ph": 2, "organic_carbon_pct": 2, "nitrogen_kg_ha": 0, "phosphorus_kg_ha": 1,
            "potassium_kg_ha": 0, "ec_ds_m": 2, "zinc_ppm": 2,
        },
    )


def plant_condition(rng: np.random.Generator) -> pd.DataFrame:
    """Classify crop condition from canopy readings and soil moisture."""
    profile = {
        "healthy": {
            "ndvi": (0.80, 0.06), "chlorophyll_spad": (42, 4), "canopy_temp_excess_c": (1.0, 1.2),
            "soil_moisture_pct": (28, 5), "leaf_damage_pct": (3, 2.5),
        },
        "water_stressed": {
            "ndvi": (0.56, 0.09), "chlorophyll_spad": (36, 5), "canopy_temp_excess_c": (5.0, 1.5),
            "soil_moisture_pct": (12, 4), "leaf_damage_pct": (4, 3),
        },
        "nutrient_deficient": {
            "ndvi": (0.62, 0.08), "chlorophyll_spad": (26, 4), "canopy_temp_excess_c": (2.0, 1.3),
            "soil_moisture_pct": (26, 5), "leaf_damage_pct": (4, 3),
        },
        "pest_stressed": {
            "ndvi": (0.66, 0.09), "chlorophyll_spad": (35, 5), "canopy_temp_excess_c": (2.5, 1.4),
            "soil_moisture_pct": (25, 5), "leaf_damage_pct": (22, 8),
        },
    }
    frame = _sample(rng, profile, rows_per_class=350, shared_shift=0.2)
    frame["ndvi"] = frame["ndvi"].clip(0.15, 0.95)
    frame["chlorophyll_spad"] = frame["chlorophyll_spad"].clip(10, 60)
    frame["canopy_temp_excess_c"] = frame["canopy_temp_excess_c"].clip(-2, 10)
    frame["soil_moisture_pct"] = frame["soil_moisture_pct"].clip(3, 45)
    frame["leaf_damage_pct"] = frame["leaf_damage_pct"].clip(0, 60)
    return _finish(
        rng,
        frame,
        {
            "ndvi": 3, "chlorophyll_spad": 1, "canopy_temp_excess_c": 1,
            "soil_moisture_pct": 1, "leaf_damage_pct": 1,
        },
    )


GENERATORS = {
    "soil_type": soil_type,
    "soil_fertility": soil_fertility,
    "plant_condition": plant_condition,
}


def main() -> None:
    summary = []
    for index, (name, generator) in enumerate(GENERATORS.items()):
        rng = np.random.default_rng(SEED + index)
        frame = generator(rng)
        directory = OUTPUT / name
        directory.mkdir(parents=True, exist_ok=True)
        frame.to_csv(directory / f"{name}.csv", index=False)
        counts = frame["label"].value_counts()
        summary.append(
            {
                "dataset": name,
                "rows": len(frame),
                "features": frame.shape[1] - 2,
                "classes": len(counts),
                "smallest_class": int(counts.min()),
                "largest_class": int(counts.max()),
            }
        )
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
