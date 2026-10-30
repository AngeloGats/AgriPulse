"""Prepare a small, reproducible leaf-image subset from the raw PlantVillage download.

Reads data/ANN/PlantVillage/<class>/*.JPG (the raw download is not committed),
picks PER_CLASS random images per class (seed 42), downsizes them to 64x64 JPEGs in
data/ANN/leaf_images/<class>/ and writes a stratified 80/20 train/test split plus
schema.json. The small subset IS committed, so groupmates need no raw download.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "ANN" / "PlantVillage"
OUTPUT = ROOT / "data" / "ANN" / "leaf_images"
SEED = 42
PER_CLASS = 900
SIZE = 64

# raw folder -> (key, display name)
CLASSES = {
    "Tomato_healthy": ("healthy", "Healthy"),
    "Tomato_Early_blight": ("early_blight", "Early blight"),
    "Tomato_Late_blight": ("late_blight", "Late blight"),
    "Tomato_Bacterial_spot": ("bacterial_spot", "Bacterial spot"),
    "Tomato__Tomato_YellowLeaf__Curl_Virus": ("yellow_leaf_curl", "Yellow leaf curl virus"),
}


def main() -> None:
    if OUTPUT.exists():
        for old in OUTPUT.iterdir():
            if old.name == "README.md":
                continue
            shutil.rmtree(old) if old.is_dir() else old.unlink()
    rows = []
    for folder, (key, _name) in CLASSES.items():
        files = sorted(p for p in (RAW / folder).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if len(files) < PER_CLASS:
            raise SystemExit(f"{folder} has only {len(files)} images, need {PER_CLASS}")
        chosen = pd.Series(files).sample(PER_CLASS, random_state=SEED).tolist()
        (OUTPUT / key).mkdir(parents=True)
        for index, source in enumerate(sorted(chosen), start=1):
            target = OUTPUT / key / f"{key}_{index:04d}.jpg"
            with Image.open(source) as image:
                image.convert("RGB").resize((SIZE, SIZE), Image.LANCZOS).save(target, quality=90)
            rows.append({"path": f"{key}/{target.name}", "label": key})

    frame = pd.DataFrame(rows)
    train, test = train_test_split(frame, test_size=0.2, stratify=frame["label"], random_state=SEED)
    train.sort_values("path").to_csv(OUTPUT / "train.csv", index=False)
    test.sort_values("path").to_csv(OUTPUT / "test.csv", index=False)
    schema = {
        "dataset": "plantvillage_tomato_subset",
        "image_size": SIZE,
        "classes": [key for key, _ in CLASSES.values()],
        "names": {key: name for key, name in CLASSES.values()},
        "per_class": PER_CLASS,
        "train_rows": len(train),
        "test_rows": len(test),
    }
    (OUTPUT / "schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    print(frame["label"].value_counts().to_string())
    print(f"train={len(train)} test={len(test)}")


if __name__ == "__main__":
    main()
