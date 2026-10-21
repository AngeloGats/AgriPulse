"""Local Flask demo for the trained CYCleSS Linear Regression model."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from flask import Flask, abort, jsonify, render_template, request, send_from_directory
from werkzeug.exceptions import BadRequest


APP_DIR = Path(__file__).resolve().parent
SITE_DIR = APP_DIR.parent
PROJECT_ROOT = SITE_DIR.parent
TEMPLATE_DIR = APP_DIR / "templates"
SITE_PAGES_DIR = SITE_DIR / "pages"
SITE_ASSETS_DIR = SITE_DIR / "assets"
SITE_IMAGES_DIR = SITE_ASSETS_DIR / "images"
MODEL_DIR = PROJECT_ROOT / "trained_models" / "linear_regression"
MODEL_PATH = MODEL_DIR / "models" / "cycless.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
PREDICTIONS_PATH = MODEL_DIR / "predictions" / "cycless_test_predictions.csv"
EXAMPLE_PATH = (
    PROJECT_ROOT
    / "data"
    / "LinearRegression"
    / "processed"
    / "cycless"
    / "X_test.csv"
)
ALLOWED_SITE_PAGES = {"index", "about", "analytics", "demo"}
OBSERVATION_LIMITS = {2015: 61, 2016: 73, 2017: 177}
SENSORS = ("ratio", "vh", "vv")
SENSOR_FIELDS = ("mean", "std", "min", "max", "observations")


def _load_resources() -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    required = (MODEL_PATH, METRICS_PATH, EXAMPLE_PATH)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "The Linear Regression demo is missing required trained artifacts: "
            + ", ".join(missing)
            + ". Run ml/prepare_linear_regression_data.py and "
            "ml/train_linear_regression.py from the project root first."
        )

    bundle = joblib.load(MODEL_PATH)
    with METRICS_PATH.open(encoding="utf-8") as metrics_file:
        metrics = json.load(metrics_file)
    training_features = pd.read_csv(
        PROJECT_ROOT
        / "data"
        / "LinearRegression"
        / "processed"
        / "cycless"
        / "X_train.csv"
    )
    example_features = pd.read_csv(EXAMPLE_PATH, nrows=1)
    if not bundle.get("feature_columns") or not len(example_features):
        raise ValueError("The trained CYCleSS model or sample inputs are invalid")
    if list(training_features.columns) != bundle["feature_columns"]:
        raise ValueError("CYCleSS training feature schema does not match the model")
    if list(example_features.columns) != bundle["feature_columns"]:
        raise ValueError("CYCleSS sample feature schema does not match the model")

    return bundle, metrics, training_features


def _allowed_categories(feature_columns: list[str]) -> tuple[list[str], list[int]]:
    crops = sorted(
        column.removeprefix("Crop_")
        for column in feature_columns
        if column.startswith("Crop_")
    )
    years = sorted(
        int(column.removeprefix("Year_"))
        for column in feature_columns
        if column.startswith("Year_")
    )
    return crops, years


def _parse_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite")
    return number


def _build_input(
    payload: Any,
    feature_columns: list[str],
    training_features: pd.DataFrame,
) -> pd.DataFrame:
    if not isinstance(payload, dict):
        raise ValueError("The request body must be a JSON object")
    if set(payload) != {"crop", "year", *SENSORS}:
        raise ValueError("Provide crop, year, and summaries for Ratio, VH, and VV")

    crops, years = _allowed_categories(feature_columns)
    crop = payload["crop"]
    year_value = payload["year"]
    if not isinstance(crop, str) or crop not in crops:
        raise ValueError(f"crop must be one of: {', '.join(crops)}")
    if isinstance(year_value, bool) or not isinstance(year_value, int):
        raise ValueError(f"year must be one of: {', '.join(map(str, years))}")
    if year_value not in years:
        raise ValueError(f"year must be one of: {', '.join(map(str, years))}")

    feature_values = dict.fromkeys(feature_columns, 0.0)
    feature_values[f"Crop_{crop}"] = 1.0
    feature_values[f"Year_{year_value}"] = 1.0

    for sensor in SENSORS:
        summary = payload[sensor]
        if not isinstance(summary, dict) or set(summary) != set(SENSOR_FIELDS):
            raise ValueError(
                f"{sensor.upper()} must include mean, std, min, max, and observations"
            )

        count = _parse_number(
            summary["observations"],
            f"{sensor.upper()} observations",
        )
        if (
            count != int(count)
            or count < 0
            or count > OBSERVATION_LIMITS[year_value]
        ):
            raise ValueError(
                f"{sensor.upper()} observations must be a whole number from 0 to "
                f"{OBSERVATION_LIMITS[year_value]} for {year_value}"
            )

        summary_fields = ("mean", "std", "min", "max")
        if count == 0:
            if any(summary[field] is not None for field in summary_fields):
                raise ValueError(
                    f"{sensor.upper()} summary values must be empty when "
                    "observations is zero"
                )
            for field in summary_fields:
                feature_values[f"{sensor}_{field}"] = float(
                    training_features[f"{sensor}_{field}"].median()
                )
        else:
            values: dict[str, float] = {}
            for field in summary_fields:
                value = summary[field]
                if value is None and field == "std":
                    values[field] = float(
                        training_features[f"{sensor}_{field}"].median()
                    )
                elif value is None:
                    raise ValueError(
                        f"{sensor.upper()} {field} is required when observations "
                        "is greater than zero"
                    )
                else:
                    values[field] = _parse_number(
                        value,
                        f"{sensor.upper()} {field}",
                    )

            if values["std"] < 0:
                raise ValueError(f"{sensor.upper()} standard deviation cannot be negative")
            if values["min"] > values["mean"] or values["mean"] > values["max"]:
                raise ValueError(
                    f"{sensor.upper()} values must satisfy min <= mean <= max"
                )
            for field in summary_fields:
                feature_values[f"{sensor}_{field}"] = values[field]

        feature_values[f"{sensor}_observations"] = count

    return pd.DataFrame([feature_values], columns=feature_columns, dtype=float)


def _example_payload(features: pd.DataFrame, feature_columns: list[str]) -> dict[str, Any]:
    row = features.iloc[0]
    crop = next(
        (column.removeprefix("Crop_") for column in feature_columns
         if column.startswith("Crop_") and row[column] == 1),
        None,
    )
    year = next(
        (int(column.removeprefix("Year_")) for column in feature_columns
         if column.startswith("Year_") and row[column] == 1),
        None,
    )
    if crop is None or year is None:
        raise ValueError("The stored example is missing its crop or year category")

    return {
        "crop": crop,
        "year": year,
        **{
            sensor: {
                field: (
                    int(row[f"{sensor}_{field}"])
                    if field == "observations"
                    else float(row[f"{sensor}_{field}"])
                )
                for field in SENSOR_FIELDS
            }
            for sensor in SENSORS
        },
    }


def _load_test_predictions() -> list[dict[str, float]]:
    if not PREDICTIONS_PATH.is_file():
        raise FileNotFoundError(
            "The Linear Regression demo is missing held-out test predictions: "
            f"{PREDICTIONS_PATH}. Run ml/train_linear_regression.py from the "
            "project root first."
        )

    predictions = pd.read_csv(PREDICTIONS_PATH)
    required_columns = {"actual_yield", "predicted_yield"}
    if not required_columns.issubset(predictions.columns):
        raise ValueError(
            "CYCleSS test predictions must include actual_yield and predicted_yield"
        )

    values = predictions[["actual_yield", "predicted_yield"]].to_numpy(dtype=float)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("CYCleSS test predictions must contain finite field records")

    return [
        {"actual": float(actual), "predicted": float(predicted)}
        for actual, predicted in values
    ]


def create_app() -> Flask:
    app = Flask(__name__, template_folder=str(TEMPLATE_DIR), static_folder=None)
    app.config["MAX_CONTENT_LENGTH"] = 8 * 1024

    model_bundle, all_metrics, training_features = _load_resources()
    feature_columns = model_bundle["feature_columns"]
    model_metrics = next(
        (item for item in all_metrics if item.get("dataset") == "cycless"),
        None,
    )
    if model_metrics is None:
        raise ValueError("CYCleSS evaluation metrics are missing")

    test_predictions = _load_test_predictions()
    if len(test_predictions) != model_metrics["test_rows"]:
        raise ValueError("CYCleSS test prediction count does not match evaluation metrics")

    app.config["MODEL_BUNDLE"] = model_bundle
    app.config["TRAINING_FEATURES"] = training_features
    app.config["EXAMPLE_FEATURES"] = pd.read_csv(EXAMPLE_PATH, nrows=1)
    app.config["MODEL_METRICS"] = model_metrics
    @app.get("/")
    def home():
        return send_from_directory(SITE_PAGES_DIR, "index.html")

    @app.get("/linear-regression")
    def linear_regression_page():
        return render_template(
            "linear_regression.html",
            metrics=model_metrics,
            crops=_allowed_categories(feature_columns)[0],
            years=_allowed_categories(feature_columns)[1],
            diagnostic_points=test_predictions,
        )

    @app.get("/api/example")
    def example():
        return jsonify(
            _example_payload(app.config["EXAMPLE_FEATURES"], feature_columns)
        )

    @app.post("/api/predict")
    def predict():
        if not request.is_json:
            return jsonify(error="Send the inputs as a JSON request"), 415
        try:
            payload = request.get_json()
        except BadRequest:
            return jsonify(error="The request body is not valid JSON"), 400
        try:
            features = _build_input(
                payload,
                feature_columns,
                app.config["TRAINING_FEATURES"],
            )
        except (TypeError, ValueError) as error:
            return jsonify(error=str(error)), 400

        model = model_bundle["model"]
        prediction = float(model.predict(features)[0])
        training = app.config["TRAINING_FEATURES"]
        outside_training_range = [
            column
            for column in feature_columns
            if column.startswith(("ratio_", "vh_", "vv_"))
            and (
                features.iloc[0][column] < training[column].min()
                or features.iloc[0][column] > training[column].max()
            )
        ]

        return jsonify(
            prediction_t_per_ha=prediction,
            crop=payload["crop"],
            year=payload["year"],
            outside_training_range=outside_training_range,
            model_metrics={
                "test_r2": model_metrics["test_r2"],
                "test_mae_t_per_ha": model_metrics["test_mae"],
                "test_rows": model_metrics["test_rows"],
            },
        )

    @app.get("/ml-static/<path:filename>")
    def ml_static(filename: str):
        return send_from_directory(APP_DIR / "static", filename)

    @app.get("/assets/<path:filename>")
    def project_static(filename: str):
        return send_from_directory(SITE_ASSETS_DIR, filename)

    @app.get("/images/<path:filename>")
    def project_images(filename: str):
        return send_from_directory(SITE_IMAGES_DIR, filename)

    @app.get("/<page>.html")
    def project_page(page: str):
        if page not in ALLOWED_SITE_PAGES:
            abort(404)
        return send_from_directory(SITE_PAGES_DIR, f"{page}.html")

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5051, debug=False)
