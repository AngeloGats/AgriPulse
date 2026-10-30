"""Prepare the generated KNN datasets: stratified 80/20 split and a schema file.

Reads data/kNN/<name>/<name>.csv and writes train.csv, test.csv and schema.json
to data/kNN/processed/<name>/. The scaler is fitted later, on training rows only.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "kNN"
OUTPUT = SOURCE / "processed"
DATASETS = ["soil_type", "soil_fertility", "plant_condition"]
RANDOM_STATE = 42
TEST_SIZE = 0.2


def prepare(name: str) -> dict:
    frame = pd.read_csv(SOURCE / name / f"{name}.csv")
    if frame.isna().any().any():
        raise ValueError(f"{name}: unexpected missing values")

    features = [c for c in frame.columns if c not in ("sample_id", "label")]
    train, test = train_test_split(
        frame, test_size=TEST_SIZE, stratify=frame["label"], random_state=RANDOM_STATE
    )
    train = train.sort_values("sample_id")
    test = test.sort_values("sample_id")

    out_dir = OUTPUT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(out_dir / "train.csv", index=False)
    test.to_csv(out_dir / "test.csv", index=False)

    schema = {
        "dataset": name,
        "features": features,
        "classes": sorted(frame["label"].unique()),
        "ranges": {f: [float(frame[f].min()), float(frame[f].max())] for f in features},
        "train_rows": len(train),
        "test_rows": len(test),
        "class_counts_train": train["label"].value_counts().sort_index().to_dict(),
    }
    (out_dir / "schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    return {
        "dataset": name,
        "train_rows": len(train),
        "test_rows": len(test),
        "features": len(features),
        "classes": len(schema["classes"]),
    }


def main() -> None:
    summary = pd.DataFrame([prepare(name) for name in DATASETS])
    summary.to_csv(OUTPUT / "preparation_summary.csv", index=False)
    (OUTPUT / "README.md").write_text(
        "# Prepared KNN data\n\n"
        "Stratified 80/20 split (seed 42) of the generated datasets in `data/kNN/`.\n"
        "Each folder has `train.csv`, `test.csv` and `schema.json` (feature order, ranges, classes).\n"
        "Values are unscaled; the scaler is fitted on `train.csv` only during training.\n"
        "Regenerate with `python ml/prepare_knn_data.py`.\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
