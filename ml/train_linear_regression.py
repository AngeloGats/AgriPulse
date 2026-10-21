"""Train and evaluate Linear Regression baselines on prepared yield datasets."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "LinearRegression" / "processed"
OUTPUT_ROOT = ROOT / "trained_models" / "linear_regression"

DATASETS = {
    "crop_yield_soil_weather": {
        "target_unit": "reported t/ha; values require further quality checks",
        "validation": True,
        "notes": "India state/crop/year records; incomplete 2020 excluded.",
    },
    "cycless": {
        "target_unit": "mean t/ha per field",
        "validation": False,
        "notes": "Grid-group holdout; full-season satellite summaries are exploratory.",
    },
    "soybean_ndvi_argentina": {
        "target_unit": "metric tonnes per harvested hectare",
        "validation": True,
        "notes": "Argentina soybean NDVI; source-provided splits preserved.",
    },
    "soybean_ndvi_brazil": {
        "target_unit": "metric tonnes per harvested hectare",
        "validation": True,
        "notes": "Brazil soybean NDVI; source-provided splits preserved.",
    },
    "soybean_ndvi_usa": {
        "target_unit": "metric tonnes per harvested hectare",
        "validation": True,
        "notes": "USA soybean NDVI; source-provided splits preserved.",
    },
}


def _load_split(dataset_dir: Path, split: str) -> tuple[pd.DataFrame, np.ndarray]:
    features_path = dataset_dir / f"X_{split}.csv"
    targets_path = dataset_dir / f"y_{split}.csv"
    if not features_path.is_file() or not targets_path.is_file():
        raise FileNotFoundError(f"Missing {split} features or target in {dataset_dir}")

    features = pd.read_csv(features_path)
    targets = pd.read_csv(targets_path)
    if len(features) != len(targets) or len(features) == 0:
        raise ValueError(f"{dataset_dir.name}/{split}: feature/target rows do not align")
    if not features.select_dtypes(exclude="number").empty:
        raise ValueError(f"{dataset_dir.name}/{split}: features must all be numeric")
    if targets.shape[1] != 1:
        raise ValueError(f"{dataset_dir.name}/{split}: expected one target column")

    x = features.to_numpy(dtype=float)
    y = targets.iloc[:, 0].to_numpy(dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError(f"{dataset_dir.name}/{split}: non-finite data found")
    return features, y


def _metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    baseline: np.ndarray,
) -> dict[str, float]:
    return {
        "r2": float(r2_score(actual, predicted)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "baseline_mean_mae": float(mean_absolute_error(actual, baseline)),
        "baseline_mean_rmse": float(np.sqrt(mean_squared_error(actual, baseline))),
    }


def _new_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("regressor", LinearRegression()),
        ]
    )


def train_dataset(dataset_name: str, config: dict[str, object]) -> dict[str, object]:
    dataset_dir = DATA_ROOT / dataset_name
    train_x, train_y = _load_split(dataset_dir, "train")
    test_x, test_y = _load_split(dataset_dir, "test")
    if list(train_x.columns) != list(test_x.columns):
        raise ValueError(f"{dataset_name}: train/test feature columns differ")

    initial_model = _new_model()
    initial_model.fit(train_x, train_y)
    fitting_x = train_x
    fitting_y = train_y
    result: dict[str, object] = {
        "dataset": dataset_name,
        "target_unit": config["target_unit"],
        "train_rows": len(train_y),
        "test_rows": len(test_y),
        "features": train_x.shape[1],
        "notes": config["notes"],
    }

    if config["validation"]:
        validation_x, validation_y = _load_split(dataset_dir, "validation")
        if list(train_x.columns) != list(validation_x.columns):
            raise ValueError(f"{dataset_name}: train/validation feature columns differ")

        validation_predictions = initial_model.predict(validation_x)
        train_mean = np.full(len(validation_y), train_y.mean())
        for metric, value in _metrics(
            validation_y,
            validation_predictions,
            train_mean,
        ).items():
            result[f"validation_{metric}"] = value
        result["validation_rows"] = len(validation_y)

        fitting_x = pd.concat([train_x, validation_x], ignore_index=True)
        fitting_y = np.concatenate([train_y, validation_y])
    else:
        result["validation_rows"] = 0

    if list(fitting_x.columns) != list(test_x.columns):
        raise ValueError(f"{dataset_name}: final/test feature columns differ")

    final_model = _new_model()
    final_model.fit(fitting_x, fitting_y)
    test_predictions = final_model.predict(test_x)
    test_baseline = np.full(len(test_y), fitting_y.mean())
    for metric, value in _metrics(
        test_y,
        test_predictions,
        test_baseline,
    ).items():
        result[f"test_{metric}"] = value
    result["final_fit_rows"] = len(fitting_y)

    model_dir = OUTPUT_ROOT / "models"
    predictions_dir = OUTPUT_ROOT / "predictions"
    model_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": final_model,
            "dataset": dataset_name,
            "target_unit": config["target_unit"],
            "feature_columns": list(fitting_x.columns),
            "notes": config["notes"],
        },
        model_dir / f"{dataset_name}.joblib",
    )
    pd.DataFrame(
        {
            "actual_yield": test_y,
            "predicted_yield": test_predictions,
            "training_mean_baseline": test_baseline,
        }
    ).to_csv(predictions_dir / f"{dataset_name}_test_predictions.csv", index=False)
    return result


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = [train_dataset(name, config) for name, config in DATASETS.items()]
    results_path = OUTPUT_ROOT / "metrics.json"
    results_path.write_text(
        json.dumps(results, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(results).to_csv(OUTPUT_ROOT / "metrics.csv", index=False)

    columns = [
        "dataset",
        "train_rows",
        "validation_rows",
        "test_rows",
        "features",
        "test_r2",
        "test_mae",
        "test_rmse",
        "test_baseline_mean_mae",
        "test_baseline_mean_rmse",
    ]
    print(pd.DataFrame(results)[columns].to_string(index=False))
    print(f"\nSaved models, test predictions, and metrics under: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
