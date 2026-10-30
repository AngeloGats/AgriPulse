"""Prepare the Naive Bayes datasets (soybean disease and crop recommendation).

Raw files in data/NaiveBayes are never modified. Prepared files are written to
data/NaiveBayes/processed/ and are small enough to commit.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "NaiveBayes"
OUTPUT = SOURCE / "processed"
RANDOM_STATE = 42

SOYBEAN_DIR = SOURCE / "Soybeans"
CROP_FILE = SOURCE / "Crop Reccomendation" / "Crop_recommendation.csv"

# Attribute order and value labels come from soybean-large.names. A value's
# integer code in the raw files is its position in the list ("?" is missing).
SOYBEAN_SCHEMA: dict[str, list[str]] = {
    "date": ["april", "may", "june", "july", "august", "september", "october"],
    "plant-stand": ["normal", "lt-normal"],
    "precip": ["lt-norm", "norm", "gt-norm"],
    "temp": ["lt-norm", "norm", "gt-norm"],
    "hail": ["yes", "no"],
    "crop-hist": ["diff-lst-year", "same-lst-yr", "same-lst-two-yrs", "same-lst-sev-yrs"],
    "area-damaged": ["scattered", "low-areas", "upper-areas", "whole-field"],
    "severity": ["minor", "pot-severe", "severe"],
    "seed-tmt": ["none", "fungicide", "other"],
    "germination": ["90-100%", "80-89%", "lt-80%"],
    "plant-growth": ["norm", "abnorm"],
    "leaves": ["norm", "abnorm"],
    "leafspots-halo": ["absent", "yellow-halos", "no-yellow-halos"],
    "leafspots-marg": ["w-s-marg", "no-w-s-marg", "dna"],
    "leafspot-size": ["lt-1/8", "gt-1/8", "dna"],
    "leaf-shread": ["absent", "present"],
    "leaf-malf": ["absent", "present"],
    "leaf-mild": ["absent", "upper-surf", "lower-surf"],
    "stem": ["norm", "abnorm"],
    "lodging": ["yes", "no"],
    "stem-cankers": ["absent", "below-soil", "above-soil", "above-sec-nde"],
    "canker-lesion": ["dna", "brown", "dk-brown-blk", "tan"],
    "fruiting-bodies": ["absent", "present"],
    "external-decay": ["absent", "firm-and-dry", "watery"],
    "mycelium": ["absent", "present"],
    "int-discolor": ["none", "brown", "black"],
    "sclerotia": ["absent", "present"],
    "fruit-pods": ["norm", "diseased", "few-present", "dna"],
    "fruit-spots": ["absent", "colored", "brown-w/blk-specks", "distort", "dna"],
    "seed": ["norm", "abnorm"],
    "mold-growth": ["absent", "present"],
    "seed-discolor": ["absent", "present"],
    "seed-size": ["norm", "lt-norm"],
    "shriveling": ["absent", "present"],
    "roots": ["norm", "rotted", "galls-cysts"],
}

# The first 15 diseases are the ones used in the original study. The last four
# have 1-6 training rows, which is too few for reliable probabilities.
SOYBEAN_RARE_CLASSES = [
    "diaporthe-pod-&-stem-blight",
    "cyst-nematode",
    "2-4-d-injury",
    "herbicide-injury",
]

CROP_FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


def _read_soybean(path: Path) -> pd.DataFrame:
    columns = ["disease", *SOYBEAN_SCHEMA]
    frame = pd.read_csv(path, header=None, names=columns, na_values="?")
    # Row-level validation guards against a code that has no label.
    for name, values in SOYBEAN_SCHEMA.items():
        highest = frame[name].max()
        if pd.notna(highest) and highest >= len(values):
            raise ValueError(f"{name}: code {int(highest)} has no label in the schema.")
    return frame


def _write_split(directory: Path, name: str, features: pd.DataFrame, target: pd.Series) -> None:
    features.reset_index(drop=True).to_csv(directory / f"X_{name}.csv", index=False)
    target.reset_index(drop=True).to_frame("target").to_csv(directory / f"y_{name}.csv", index=False)


def prepare_soybean() -> dict[str, object]:
    directory = OUTPUT / "soybean_disease"
    directory.mkdir(parents=True, exist_ok=True)

    train = _read_soybean(SOYBEAN_DIR / "soybean-large.data")
    test = _read_soybean(SOYBEAN_DIR / "soybean-large.test")
    raw_rows = {"train": len(train), "test": len(test)}

    train = train[~train["disease"].isin(SOYBEAN_RARE_CLASSES)]
    test = test[~test["disease"].isin(SOYBEAN_RARE_CLASSES)]
    features = list(SOYBEAN_SCHEMA)

    unseen = set(test["disease"]) - set(train["disease"])
    if unseen:
        raise ValueError(f"Test classes missing from training data: {sorted(unseen)}")

    # Missing values stay blank (nullable integers); they mean "symptom not recorded".
    for name, split in (("train", train), ("test", test)):
        _write_split(directory, name, split[features].astype("Int64"), split["disease"])

    (directory / "feature_columns.txt").write_text("\n".join(features) + "\n", encoding="utf-8")
    classes = sorted(train["disease"].unique())
    schema = {
        "features": {name: values for name, values in SOYBEAN_SCHEMA.items()},
        "classes": classes,
        "excluded_classes": SOYBEAN_RARE_CLASSES,
        "missing_value": "blank cell = symptom not recorded",
    }
    (directory / "schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")

    return {
        "dataset": "soybean_disease",
        "model": "CategoricalNB",
        "raw_train_rows": raw_rows["train"],
        "raw_test_rows": raw_rows["test"],
        "train_rows": len(train),
        "test_rows": len(test),
        "features": len(features),
        "classes": len(classes),
        "missing_cells_train_pct": round(float(train[features].isna().mean().mean() * 100), 2),
        "missing_cells_test_pct": round(float(test[features].isna().mean().mean() * 100), 2),
        "smallest_train_class": int(train["disease"].value_counts().min()),
    }


def prepare_crop_recommendation() -> dict[str, object]:
    directory = OUTPUT / "crop_recommendation"
    directory.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(CROP_FILE)
    frame.columns = [column.strip().lower() if column != "label" else column for column in frame.columns]
    frame = frame.rename(columns={"n": "N", "p": "P", "k": "K"})
    missing = [column for column in [*CROP_FEATURES, "label"] if column not in frame.columns]
    if missing:
        raise ValueError(f"Crop data is missing columns: {missing}")
    if frame[CROP_FEATURES].isna().any().any() or frame.duplicated().any():
        raise ValueError("Crop data has unexpected missing values or duplicates.")

    frame["label"] = frame["label"].str.strip().str.lower()
    train_x, test_x, train_y, test_y = train_test_split(
        frame[CROP_FEATURES],
        frame["label"],
        test_size=0.2,
        stratify=frame["label"],
        random_state=RANDOM_STATE,
    )
    _write_split(directory, "train", train_x, train_y)
    _write_split(directory, "test", test_x, test_y)

    (directory / "feature_columns.txt").write_text("\n".join(CROP_FEATURES) + "\n", encoding="utf-8")
    ranges = {
        column: {
            "min": float(train_x[column].min()),
            "max": float(train_x[column].max()),
            "median": float(train_x[column].median()),
        }
        for column in CROP_FEATURES
    }
    schema = {"features": ranges, "classes": sorted(train_y.unique())}
    (directory / "schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")

    return {
        "dataset": "crop_recommendation",
        "model": "GaussianNB",
        "raw_train_rows": len(frame),
        "raw_test_rows": 0,
        "train_rows": len(train_x),
        "test_rows": len(test_x),
        "features": len(CROP_FEATURES),
        "classes": int(train_y.nunique()),
        "missing_cells_train_pct": 0.0,
        "missing_cells_test_pct": 0.0,
        "smallest_train_class": int(train_y.value_counts().min()),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame([prepare_soybean(), prepare_crop_recommendation()])
    summary.to_csv(OUTPUT / "preparation_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
