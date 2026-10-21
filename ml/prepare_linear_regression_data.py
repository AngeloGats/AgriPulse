"""Prepare the three yield datasets for Linear Regression experiments."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "LinearRegression"
OUTPUT = SOURCE / "processed"
RANDOM_STATE = 42


def _write_splits(
    output_dir: Path,
    feature_splits: dict[str, pd.DataFrame],
    target_splits: dict[str, pd.Series],
    categorical_columns: list[str] | None = None,
) -> dict[str, int]:
    """Impute and encode using training data only, then write numerical CSVs."""
    categorical_columns = categorical_columns or []
    train_features = feature_splits["train"].copy()
    numeric_columns = [
        column
        for column in train_features.columns
        if column not in categorical_columns
    ]

    medians = train_features[numeric_columns].median()
    medians = medians.fillna(0)
    encoded: dict[str, pd.DataFrame] = {}

    for split, features in feature_splits.items():
        numeric = features[numeric_columns].apply(pd.to_numeric, errors="coerce")
        numeric = numeric.fillna(medians).fillna(0)
        encoded[split] = numeric.reset_index(drop=True)

    if categorical_columns:
        train_categories = train_features[categorical_columns].fillna("__missing__").astype(str)
        train_dummies = pd.get_dummies(
            train_categories,
            columns=categorical_columns,
            dtype=float,
        )
        dummy_columns = train_dummies.columns
        encoded["train"] = pd.concat(
            [encoded["train"], train_dummies.reset_index(drop=True)],
            axis=1,
        )

        for split, features in feature_splits.items():
            if split == "train":
                continue
            categories = features[categorical_columns].fillna("__missing__").astype(str)
            dummies = pd.get_dummies(
                categories,
                columns=categorical_columns,
                dtype=float,
            ).reindex(columns=dummy_columns, fill_value=0)
            encoded[split] = pd.concat(
                [encoded[split], dummies.reset_index(drop=True)],
                axis=1,
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    row_counts: dict[str, int] = {}
    for split, features in encoded.items():
        target = target_splits[split].reset_index(drop=True)
        if len(features) != len(target):
            raise ValueError(f"{output_dir.name}: {split} feature/target length mismatch")
        if not np.isfinite(features.to_numpy(dtype=float)).all():
            raise ValueError(f"{output_dir.name}: non-finite feature values remain in {split}")
        if not np.isfinite(target.to_numpy(dtype=float)).all():
            raise ValueError(f"{output_dir.name}: non-finite targets remain in {split}")

        features.to_csv(output_dir / f"X_{split}.csv", index=False)
        target.rename("yield").to_csv(output_dir / f"y_{split}.csv", index=False)
        row_counts[split] = len(features)

    (output_dir / "feature_columns.txt").write_text(
        "\n".join(encoded["train"].columns) + "\n",
        encoding="utf-8",
    )
    return row_counts


def _prepare_soil_weather() -> dict[str, int]:
    source = SOURCE / "Crop yield Data with Soil And Weather Dataset"
    yield_data = pd.read_csv(source / "crop_yield.csv")
    soil = pd.read_csv(source / "state_soil_data.csv")
    weather = pd.read_csv(source / "state_weather_data_1997_2020.csv")

    joined = yield_data.merge(
        soil,
        on="state",
        how="left",
        validate="many_to_one",
    ).merge(
        weather,
        on=["state", "year"],
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        raise ValueError("Some yield records have no matching soil or weather data")

    # 2020 contains only 37 records, so it is treated as an incomplete year.
    joined = joined.loc[joined["year"] < 2020].copy()
    joined = joined.loc[joined["yield"].notna() & np.isfinite(joined["yield"])].copy()

    # Production is excluded because it can directly reveal the yield target.
    feature_columns = [
        "crop",
        "year",
        "season",
        "state",
        "area",
        "fertilizer",
        "pesticide",
        "N",
        "P",
        "K",
        "pH",
        "avg_temp_c",
        "total_rainfall_mm",
        "avg_humidity_percent",
    ]
    split_frames = {
        "train": joined.loc[joined["year"] <= 2017],
        "validation": joined.loc[joined["year"] == 2018],
        "test": joined.loc[joined["year"] == 2019],
    }
    if any(frame.empty for frame in split_frames.values()):
        raise ValueError("Expected non-empty train, validation, and test year splits")

    counts = _write_splits(
        OUTPUT / "crop_yield_soil_weather",
        {split: frame[feature_columns] for split, frame in split_frames.items()},
        {split: frame["yield"] for split, frame in split_frames.items()},
        categorical_columns=["crop", "season", "state"],
    )
    counts["excluded_partial_2020_rows"] = int((yield_data["year"] == 2020).sum())
    return counts


def _aggregate_satellite_series(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    date_columns = [column for column in frame.columns if str(column).startswith("X")]
    if not date_columns:
        raise ValueError(f"No satellite date columns found for {prefix}")

    readings = frame[date_columns].apply(pd.to_numeric, errors="coerce")
    result = pd.DataFrame({"ID": frame["ID"].astype(str)})
    result[f"{prefix}_mean"] = readings.mean(axis=1)
    result[f"{prefix}_std"] = readings.std(axis=1)
    result[f"{prefix}_min"] = readings.min(axis=1)
    result[f"{prefix}_max"] = readings.max(axis=1)
    result[f"{prefix}_observations"] = readings.count(axis=1)
    return result


def _prepare_cycless() -> dict[str, int]:
    source = SOURCE / "CYCleSS_dataset" / "data" / "crop_yield_type_and_satellite_data"
    yearly_frames: list[pd.DataFrame] = []

    for year in (2015, 2016, 2017):
        sensor_frames: dict[str, pd.DataFrame] = {}
        for sensor in ("Ratio", "VH", "VV"):
            frame = pd.read_csv(source / f"{sensor}_{year}_MeanYieldperField.csv")
            if frame["ID"].duplicated().any():
                raise ValueError(f"CYCleSS {sensor} {year} contains duplicate field IDs")
            sensor_frames[sensor] = frame

        common_ids = set.intersection(
            *(set(frame["ID"].astype(str)) for frame in sensor_frames.values())
        )
        base = sensor_frames["Ratio"].loc[
            sensor_frames["Ratio"]["ID"].astype(str).isin(common_ids),
            ["ID", "Year", "Crop", "Yield", "grid_ID"],
        ].copy()
        base["ID"] = base["ID"].astype(str)
        base["grid_ID"] = base["grid_ID"].astype(str)

        for sensor, frame in sensor_frames.items():
            matched = base[["ID", "Yield", "Crop", "Year"]].merge(
                frame.assign(ID=frame["ID"].astype(str))[
                    ["ID", "Yield", "Crop", "Year"]
                ],
                on="ID",
                how="left",
                suffixes=("_base", "_sensor"),
                validate="one_to_one",
            )
            if (
                not np.allclose(
                    matched["Yield_base"],
                    matched["Yield_sensor"],
                    rtol=1e-9,
                    atol=1e-10,
                )
                or not matched["Crop_base"].eq(matched["Crop_sensor"]).all()
                or not matched["Year_base"].eq(matched["Year_sensor"]).all()
            ):
                raise ValueError(f"CYCleSS target or metadata mismatch in {sensor} {year}")

            aggregate = _aggregate_satellite_series(frame, sensor.lower())
            base = base.merge(aggregate, on="ID", how="left", validate="one_to_one")

        yearly_frames.append(base)

    data = pd.concat(yearly_frames, ignore_index=True)
    groups = data["grid_ID"]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_indices, test_indices = next(splitter.split(data, groups=groups))
    splits = {
        "train": data.iloc[train_indices],
        "test": data.iloc[test_indices],
    }
    if set(splits["train"]["grid_ID"]) & set(splits["test"]["grid_ID"]):
        raise ValueError("CYCleSS grid IDs leaked across the train/test split")

    feature_columns = [
        column
        for column in data.columns
        if column not in {"ID", "grid_ID", "Yield"}
    ]
    counts = _write_splits(
        OUTPUT / "cycless",
        {split: frame[feature_columns] for split, frame in splits.items()},
        {split: frame["Yield"] for split, frame in splits.items()},
        categorical_columns=["Crop", "Year"],
    )
    counts["shared_field_seasons"] = len(data)
    counts["shared_grid_groups"] = int(data["grid_ID"].nunique())
    counts["held_out_grid_groups"] = int(splits["test"]["grid_ID"].nunique())
    return counts


def _load_npz_data(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        if "data" not in archive.files:
            raise ValueError(f"{path} does not contain a 'data' array")
        return np.asarray(archive["data"])


def _prepare_soybean_ndvi() -> dict[str, dict[str, int]]:
    source = SOURCE / "Crop Yield Prediction" / "soybeans"
    all_counts: dict[str, dict[str, int]] = {}

    for region in ("argentina", "brazil", "usa"):
        raw_splits: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for source_split, output_split in (
            ("train", "train"),
            ("dev", "validation"),
            ("test", "test"),
        ):
            folder = source / region
            features = _load_npz_data(folder / f"{source_split}_ndvi.npz")
            target = _load_npz_data(folder / f"{source_split}_yields.npz").reshape(-1)
            if features.ndim != 2 or len(features) != len(target):
                raise ValueError(f"Misaligned NDVI and yield arrays for {region}/{source_split}")
            if features.shape[1] != 32:
                raise ValueError(f"Expected 32 NDVI observations for {region}/{source_split}")
            if not np.isfinite(target).all():
                raise ValueError(f"Non-finite soybean yield target in {region}/{source_split}")
            raw_splits[output_split] = (features.astype(float), target.astype(float))

        train_features = raw_splits["train"][0]
        medians = np.nanmedian(
            np.where(np.isfinite(train_features), train_features, np.nan),
            axis=0,
        )
        medians = np.where(np.isfinite(medians), medians, 0)
        feature_splits: dict[str, pd.DataFrame] = {}
        target_splits: dict[str, pd.Series] = {}

        for split, (features, target) in raw_splits.items():
            features = np.where(np.isfinite(features), features, np.nan)
            features = np.where(np.isnan(features), medians, features)
            columns = [f"ndvi_{index:02d}" for index in range(1, 33)]
            feature_splits[split] = pd.DataFrame(features, columns=columns)
            target_splits[split] = pd.Series(target, name="yield")

        all_counts[region] = _write_splits(
            OUTPUT / f"soybean_ndvi_{region}",
            feature_splits,
            target_splits,
        )

    return all_counts


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results = {
        "crop_yield_soil_weather": _prepare_soil_weather(),
        "cycless": _prepare_cycless(),
        "soybean_ndvi_by_region": _prepare_soybean_ndvi(),
    }
    report = pd.json_normalize(results, sep=".")
    report.to_csv(OUTPUT / "preparation_summary.csv", index=False)
    print(report.to_string(index=False))
    print(f"\nPrepared files written to: {OUTPUT}")


if __name__ == "__main__":
    main()
