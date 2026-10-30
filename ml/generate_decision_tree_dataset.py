"""Generate and prepare the synthetic irrigation dataset in data/DecisionTree/irrigation.

The data is simulated, not measured. Labels come from agronomy-style rules
(each crop stage has its own moisture threshold, coming
rain postpones irrigation, very wet soil means hold off). Soil moisture is
measured with noise and 3% of labels are flipped, so a tree finds real
structure but is not perfect.

Writes the full CSV, a stratified 80/20 train/test split and schema.json.
Running it again with the same seed reproduces every file.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "DecisionTree" / "irrigation"
SEED = 42
ROWS = 1600
LABEL_NOISE = 0.03
MEASUREMENT_NOISE = 1.0  # soil moisture sensor error, % points

STAGES = ["seedling", "vegetative", "flowering", "grain_fill"]
# Moisture (%) below which each stage needs water. Flowering and grain fill are
# the water-sensitive stages.
STAGE_THRESHOLD = {1: 21.0, 2: 21.0, 3: 28.0, 4: 28.0}
FEATURES = ["soil_moisture_pct", "rain_last_3d_mm", "forecast_rain_mm", "crop_stage", "air_temp_c"]


def decide(moisture: float, rain3d: float, forecast: float, stage: int, temp: float) -> str:
    threshold = STAGE_THRESHOLD[stage]
    if moisture >= 36:
        return "hold"
    if moisture >= threshold + 4:
        return "wait"
    if forecast >= 15 or rain3d >= 12:
        return "wait"
    if moisture < threshold - 3:
        return "irrigate_now"
    return "irrigate_soon"


def generate() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    stage = rng.integers(1, 5, size=ROWS)
    moisture = np.clip(rng.normal(25, 8, size=ROWS), 6, 48)
    rain3d = np.where(rng.random(ROWS) < 0.45, 0.0, rng.gamma(1.6, 7.0, size=ROWS))
    forecast = np.where(rng.random(ROWS) < 0.5, 0.0, rng.gamma(1.5, 8.0, size=ROWS))
    temp = np.clip(rng.normal(29, 5, size=ROWS), 14, 43)
    # Recent rain wets the soil a little, so the two readings are related.
    moisture = np.clip(moisture + 0.25 * rain3d, 6, 48)

    labels = [
        decide(m, r, f, int(s), t)
        for m, r, f, s, t in zip(moisture + rng.normal(0, MEASUREMENT_NOISE, ROWS), rain3d, forecast, stage, temp)
    ]
    labels = np.array(labels, dtype=object)
    classes = sorted(set(labels))
    flip = rng.random(ROWS) < LABEL_NOISE
    for index in np.flatnonzero(flip):
        labels[index] = rng.choice([c for c in classes if c != labels[index]])

    frame = pd.DataFrame(
        {
            "soil_moisture_pct": moisture.round(1),
            "rain_last_3d_mm": rain3d.round(1),
            "forecast_rain_mm": forecast.round(1),
            "crop_stage": stage,
            "air_temp_c": temp.round(1),
            "label": labels,
        }
    )
    frame.insert(0, "sample_id", np.arange(1, ROWS + 1))
    return frame


def main() -> None:
    frame = generate()
    processed = OUTPUT / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT / "irrigation.csv", index=False)

    train, test = train_test_split(frame, test_size=0.2, stratify=frame["label"], random_state=SEED)
    train.sort_values("sample_id").to_csv(processed / "train.csv", index=False)
    test.sort_values("sample_id").to_csv(processed / "test.csv", index=False)
    schema = {
        "dataset": "irrigation",
        "features": FEATURES,
        "classes": sorted(frame["label"].unique()),
        "crop_stages": STAGES,
        "ranges": {f: [float(frame[f].min()), float(frame[f].max())] for f in FEATURES},
        "train_rows": len(train),
        "test_rows": len(test),
    }
    (processed / "schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    print(frame["label"].value_counts().to_string())
    print(f"train={len(train)} test={len(test)}")


if __name__ == "__main__":
    main()
