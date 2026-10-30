"""Generate and prepare the synthetic SVM dataset in data/SVM/crop_disease.

The data is simulated, not measured. Diseased crops sit in a middle band of
temperature (a fungal disease favors moderate warmth) with high humidity, while
healthy crops surround that band: cool and humid, hot, or simply dry. A single
straight line cannot separate them, but a curved (RBF) boundary can.

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
OUTPUT = ROOT / "data" / "SVM" / "crop_disease"
SEED = 42
LABEL_NOISE = 0.03
DISEASED_ROWS = 480

# (share, {feature: (mean, sd)}) for each healthy sub-group.
HEALTHY_GROUPS = [
    (0.30, {"air_temp_c": (14, 3.0), "humidity_pct": (80, 8), "leaf_wetness_hours": (6.5, 3.0), "red_edge_index": (0.69, 0.08)}),
    (0.30, {"air_temp_c": (33, 2.5), "humidity_pct": (74, 9), "leaf_wetness_hours": (5.0, 3.0), "red_edge_index": (0.67, 0.08)}),
    (0.40, {"air_temp_c": (24, 5.5), "humidity_pct": (50, 10), "leaf_wetness_hours": (3.5, 2.5), "red_edge_index": (0.70, 0.08)}),
]
DISEASED = {"air_temp_c": (24, 3.2), "humidity_pct": (84, 7), "leaf_wetness_hours": (10.0, 3.0), "red_edge_index": (0.56, 0.08)}
HEALTHY_ROWS = 720

LIMITS = {
    "air_temp_c": (5, 45),
    "humidity_pct": (20, 100),
    "leaf_wetness_hours": (0, 24),
    "red_edge_index": (0.2, 0.95),
}
DECIMALS = {"air_temp_c": 1, "humidity_pct": 1, "leaf_wetness_hours": 1, "red_edge_index": 3}


def _draw(rng: np.random.Generator, profile: dict[str, tuple[float, float]], rows: int) -> pd.DataFrame:
    return pd.DataFrame({name: rng.normal(mean, sd, size=rows) for name, (mean, sd) in profile.items()})


def generate() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    frames = []
    diseased = _draw(rng, DISEASED, DISEASED_ROWS)
    diseased["label"] = "diseased"
    frames.append(diseased)
    for share, profile in HEALTHY_GROUPS:
        healthy = _draw(rng, profile, round(HEALTHY_ROWS * share))
        healthy["label"] = "healthy"
        frames.append(healthy)
    frame = pd.concat(frames, ignore_index=True)

    flip = rng.random(len(frame)) < LABEL_NOISE
    frame.loc[flip, "label"] = np.where(frame.loc[flip, "label"] == "healthy", "diseased", "healthy")
    for column, (low, high) in LIMITS.items():
        frame[column] = frame[column].clip(low, high).round(DECIMALS[column])
    frame = frame.sample(frac=1, random_state=SEED).reset_index(drop=True)
    frame.insert(0, "sample_id", np.arange(1, len(frame) + 1))
    return frame


def main() -> None:
    frame = generate()
    features = list(LIMITS)
    processed = OUTPUT / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT / "crop_disease.csv", index=False)

    train, test = train_test_split(frame, test_size=0.2, stratify=frame["label"], random_state=SEED)
    train.sort_values("sample_id").to_csv(processed / "train.csv", index=False)
    test.sort_values("sample_id").to_csv(processed / "test.csv", index=False)
    schema = {
        "dataset": "crop_disease",
        "features": features,
        "classes": ["healthy", "diseased"],
        "ranges": {f: [float(frame[f].min()), float(frame[f].max())] for f in features},
        "train_rows": len(train),
        "test_rows": len(test),
    }
    (processed / "schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    print(frame["label"].value_counts().to_string())
    print(f"train={len(train)} test={len(test)}")


if __name__ == "__main__":
    main()
