"""Naive Bayes demo: model loading, prediction and evidence explanations."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import BadRequest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "trained_models" / "naive_bayes"
DATA_DIR = PROJECT_ROOT / "data" / "NaiveBayes" / "processed"
SOYBEAN_MODEL_PATH = MODEL_DIR / "models" / "soybean_disease.joblib"
CROP_MODEL_PATH = MODEL_DIR / "models" / "crop_recommendation.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
SOYBEAN_DIR = DATA_DIR / "soybean_disease"
CROP_DIR = DATA_DIR / "crop_recommendation"

# Hard physical limits for crop inputs; training ranges only produce a warning.
CROP_LIMITS = {
    "N": (0, 300),
    "P": (0, 300),
    "K": (0, 400),
    "temperature": (-10, 60),
    "humidity": (0, 100),
    "ph": (0, 14),
    "rainfall": (0, 1000),
}
CROP_LABELS = {
    "N": "Nitrogen (N)",
    "P": "Phosphorus (P)",
    "K": "Potassium (K)",
    "temperature": "Temperature",
    "humidity": "Humidity",
    "ph": "Soil pH",
    "rainfall": "Rainfall",
}
CROP_UNITS = {
    "N": "kg/ha",
    "P": "kg/ha",
    "K": "kg/ha",
    "temperature": "°C",
    "humidity": "%",
    "ph": "",
    "rainfall": "mm",
}

SOYBEAN_GROUPS: list[tuple[str, list[tuple[str, str]]]] = [
    (
        "Season and field conditions",
        [
            ("date", "Month observed"),
            ("plant-stand", "Plant stand"),
            ("precip", "Rainfall this season"),
            ("temp", "Temperature this season"),
            ("hail", "Hail damage"),
            ("crop-hist", "Crop history"),
            ("area-damaged", "Area damaged"),
            ("severity", "Disease severity"),
            ("seed-tmt", "Seed treatment"),
            ("germination", "Germination rate"),
            ("plant-growth", "Plant growth"),
        ],
    ),
    (
        "Leaves",
        [
            ("leaves", "Leaf condition"),
            ("leafspots-halo", "Leaf spot halos"),
            ("leafspots-marg", "Leaf spot margins"),
            ("leafspot-size", "Leaf spot size"),
            ("leaf-shread", "Shredded leaves"),
            ("leaf-malf", "Malformed leaves"),
            ("leaf-mild", "Leaf mildew"),
        ],
    ),
    (
        "Stem",
        [
            ("stem", "Stem condition"),
            ("lodging", "Lodging (plants fallen over)"),
            ("stem-cankers", "Stem cankers"),
            ("canker-lesion", "Canker lesion color"),
            ("fruiting-bodies", "Fungal fruiting bodies"),
            ("external-decay", "External stem decay"),
            ("mycelium", "Fungal threads (mycelium)"),
            ("int-discolor", "Inside-stem discoloration"),
            ("sclerotia", "Hard fungal lumps (sclerotia)"),
        ],
    ),
    (
        "Pods and seeds",
        [
            ("fruit-pods", "Pod condition"),
            ("fruit-spots", "Pod spots"),
            ("seed", "Seed condition"),
            ("mold-growth", "Mold on seed"),
            ("seed-discolor", "Seed discoloration"),
            ("seed-size", "Seed size"),
            ("shriveling", "Seed shriveling"),
        ],
    ),
    ("Roots", [("roots", "Root condition")]),
]
SOYBEAN_LABELS = {key: label for _, items in SOYBEAN_GROUPS for key, label in items}

VALUE_LABELS = {
    "lt-norm": "Below normal",
    "norm": "Normal",
    "gt-norm": "Above normal",
    "lt-normal": "Below normal",
    "normal": "Normal",
    "abnorm": "Abnormal",
    "dna": "Not applicable",
    "yes": "Yes",
    "no": "No",
    "diff-lst-year": "Different crop last year",
    "same-lst-yr": "Same crop last year",
    "same-lst-two-yrs": "Same crop last 2 years",
    "same-lst-sev-yrs": "Same crop last several years",
    "low-areas": "Low areas",
    "upper-areas": "Upper areas",
    "whole-field": "Whole field",
    "pot-severe": "Potentially severe",
    "lt-80%": "Under 80%",
    "w-s-marg": "With margin",
    "no-w-s-marg": "No margin",
    "lt-1/8": "Under 1/8 inch",
    "gt-1/8": "Over 1/8 inch",
    "upper-surf": "Upper surface",
    "lower-surf": "Lower surface",
    "above-sec-nde": "Above second node",
    "dk-brown-blk": "Dark brown / black",
    "firm-and-dry": "Firm and dry",
    "few-present": "Few present",
    "brown-w/blk-specks": "Brown with black specks",
    "galls-cysts": "Galls or cysts",
    "yellow-halos": "Yellow halos",
    "no-yellow-halos": "No yellow halos",
    "below-soil": "Below soil line",
    "above-soil": "Above soil line",
}
DISEASE_NAMES = {
    "alternarialeaf-spot": "Alternaria leaf spot",
    "diaporthe-stem-canker": "Diaporthe stem canker",
    "frog-eye-leaf-spot": "Frogeye leaf spot",
    "phyllosticta-leaf-spot": "Phyllosticta leaf spot",
}


def _pretty(value: str) -> str:
    return VALUE_LABELS.get(value) or value.replace("-", " ").capitalize()


def _disease_name(raw: str) -> str:
    return DISEASE_NAMES.get(raw) or raw.replace("-", " ").capitalize()


def _crop_name(raw: str) -> str:
    return raw.replace("_", " ").capitalize()


def _softmax(log_scores: np.ndarray) -> np.ndarray:
    shifted = np.exp(log_scores - log_scores.max())
    return shifted / shifted.sum()


class NaiveBayesDemo:
    """Loads both trained models and answers prediction requests."""

    def __init__(self) -> None:
        required = (
            SOYBEAN_MODEL_PATH,
            CROP_MODEL_PATH,
            METRICS_PATH,
            SOYBEAN_DIR / "schema.json",
            SOYBEAN_DIR / "X_test.csv",
            CROP_DIR / "schema.json",
            CROP_DIR / "X_test.csv",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "The Naive Bayes demo is missing required trained artifacts: "
                + ", ".join(missing)
                + ". Run ml/prepare_naive_bayes_data.py and ml/train_naive_bayes.py "
                "from the project root first."
            )

        self.soybean = joblib.load(SOYBEAN_MODEL_PATH)
        self.crop = joblib.load(CROP_MODEL_PATH)
        self.metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        self.soybean_schema = json.loads((SOYBEAN_DIR / "schema.json").read_text(encoding="utf-8"))
        self.crop_schema = json.loads((CROP_DIR / "schema.json").read_text(encoding="utf-8"))
        self.soybean_examples = pd.read_csv(SOYBEAN_DIR / "X_test.csv")
        self.soybean_example_labels = pd.read_csv(SOYBEAN_DIR / "y_test.csv")["target"]
        self.crop_examples = pd.read_csv(CROP_DIR / "X_test.csv")
        self.crop_example_labels = pd.read_csv(CROP_DIR / "y_test.csv")["target"]

        self.soybean_features = list(self.soybean_schema["features"])
        self.crop_features = list(self.crop_schema["features"])
        if list(self.soybean_examples.columns) != self.soybean_features:
            raise ValueError("Soybean feature order does not match schema.json")
        if list(self.crop_examples.columns) != self.crop_features:
            raise ValueError("Crop feature order does not match schema.json")
        if list(self.crop.feature_names_in_) != self.crop_features:
            raise ValueError("Crop model features do not match schema.json")
        if list(self.soybean["classes"]) != self.soybean_schema["classes"]:
            raise ValueError("Soybean model classes do not match schema.json")

    def soybean_form(self) -> list[dict[str, Any]]:
        schema = self.soybean_schema["features"]
        groups = []
        for title, items in SOYBEAN_GROUPS:
            groups.append(
                {
                    "title": title,
                    "features": [
                        {
                            "key": key,
                            "label": label,
                            "options": [
                                {"code": code, "label": _pretty(value)}
                                for code, value in enumerate(schema[key])
                            ],
                        }
                        for key, label in items
                    ],
                }
            )
        return groups

    def crop_form(self) -> list[dict[str, Any]]:
        return [
            {
                "key": key,
                "label": CROP_LABELS[key],
                "unit": CROP_UNITS[key],
                "min": CROP_LIMITS[key][0],
                "max": CROP_LIMITS[key][1],
                "median": self.crop_schema["features"][key]["median"],
                "train_min": self.crop_schema["features"][key]["min"],
                "train_max": self.crop_schema["features"][key]["max"],
            }
            for key in self.crop_features
        ]

    def soybean_example(self, index: int) -> dict[str, Any]:
        position = index % len(self.soybean_examples)
        row = self.soybean_examples.iloc[position]
        return {
            "symptoms": {key: int(value) for key, value in row.items() if pd.notna(value)},
            "actual": _disease_name(self.soybean_example_labels.iloc[position]),
        }

    def crop_example(self, index: int) -> dict[str, Any]:
        position = index % len(self.crop_examples)
        row = self.crop_examples.iloc[position]
        return {
            "inputs": {key: round(float(value), 1) for key, value in row.items()},
            "actual": _crop_name(self.crop_example_labels.iloc[position]),
        }

    def predict_soybean(self, payload: Any) -> dict[str, Any]:
        symptoms = self._parse_symptoms(payload)
        classes = self.soybean["classes"]
        tables = self.soybean["log_likelihoods"]
        scores = np.array(self.soybean["log_priors"], dtype=float)
        for column, code in symptoms.items():
            scores = scores + tables[column][:, code]
        probabilities = _softmax(scores)

        order = np.argsort(probabilities)[::-1]
        top, runner_up = int(order[0]), int(order[1])
        evidence = []
        for column, code in symptoms.items():
            delta = float(tables[column][top, code] - tables[column][runner_up, code])
            key = self.soybean_features[column]
            evidence.append(
                {
                    "feature": SOYBEAN_LABELS[key],
                    "value": _pretty(self.soybean_schema["features"][key][code]),
                    "log_ratio": delta,
                }
            )
        evidence.sort(key=lambda item: abs(item["log_ratio"]), reverse=True)
        prior_delta = float(self.soybean["log_priors"][top] - self.soybean["log_priors"][runner_up])

        return {
            "predictions": [
                {"name": _disease_name(classes[i]), "probability": float(probabilities[i])}
                for i in order
            ],
            "top": _disease_name(classes[top]),
            "runner_up": _disease_name(classes[runner_up]),
            "symptoms_used": len(symptoms),
            "symptoms_total": len(self.soybean_features),
            "prior_log_ratio": prior_delta,
            "evidence": evidence,
            "model_metrics": self._summary_metrics("soybean_disease"),
        }

    def predict_crop(self, payload: Any) -> dict[str, Any]:
        values = self._parse_crop(payload)
        frame = pd.DataFrame([values], columns=self.crop_features)
        probabilities = self.crop.predict_proba(frame)[0]
        classes = [str(name) for name in self.crop.classes_]
        order = np.argsort(probabilities)[::-1]
        top, runner_up = int(order[0]), int(order[1])

        x = np.array([values[key] for key in self.crop_features])
        variance = self.crop.var_
        log_density = -0.5 * (
            np.log(2 * math.pi * variance) + (x - self.crop.theta_) ** 2 / variance
        )
        evidence = []
        for column, key in enumerate(self.crop_features):
            evidence.append(
                {
                    "feature": CROP_LABELS[key],
                    "value": f"{round(values[key], 2):g} {CROP_UNITS[key]}".strip(),
                    "log_ratio": float(log_density[top, column] - log_density[runner_up, column]),
                }
            )
        evidence.sort(key=lambda item: abs(item["log_ratio"]), reverse=True)

        ranges = self.crop_schema["features"]
        outside = [
            CROP_LABELS[key]
            for key in self.crop_features
            if values[key] < ranges[key]["min"] or values[key] > ranges[key]["max"]
        ]
        return {
            "predictions": [
                {"name": _crop_name(classes[i]), "probability": float(probabilities[i])}
                for i in order
            ],
            "top": _crop_name(classes[top]),
            "runner_up": _crop_name(classes[runner_up]),
            "prior_log_ratio": float(
                np.log(self.crop.class_prior_[top]) - np.log(self.crop.class_prior_[runner_up])
            ),
            "evidence": evidence,
            "outside_training_range": outside,
            "model_metrics": self._summary_metrics("crop_recommendation"),
        }

    def _summary_metrics(self, name: str) -> dict[str, Any]:
        metrics = self.metrics[name]
        return {
            "accuracy": metrics["accuracy"],
            "top3_accuracy": metrics["top3_accuracy"],
            "test_rows": metrics["test_rows"],
        }

    def _parse_symptoms(self, payload: Any) -> dict[int, int]:
        if not isinstance(payload, dict) or not isinstance(payload.get("symptoms"), dict):
            raise ValueError("Send a JSON object with a 'symptoms' object")
        schema = self.soybean_schema["features"]
        parsed: dict[int, int] = {}
        for key, code in payload["symptoms"].items():
            if key not in schema:
                raise ValueError(f"Unknown symptom: {key}")
            if code is None or code == "":
                continue
            if isinstance(code, bool) or not isinstance(code, int):
                raise ValueError(f"{SOYBEAN_LABELS[key]} must be one of the listed options")
            if not 0 <= code < len(schema[key]):
                raise ValueError(f"{SOYBEAN_LABELS[key]} must be one of the listed options")
            parsed[self.soybean_features.index(key)] = code
        if not parsed:
            raise ValueError("Select at least one symptom or condition")
        return parsed

    def _parse_crop(self, payload: Any) -> dict[str, float]:
        if not isinstance(payload, dict) or not isinstance(payload.get("inputs"), dict):
            raise ValueError("Send a JSON object with an 'inputs' object")
        inputs = payload["inputs"]
        unknown = set(inputs) - set(self.crop_features)
        if unknown:
            raise ValueError(f"Unknown inputs: {', '.join(sorted(unknown))}")
        parsed: dict[str, float] = {}
        for key in self.crop_features:
            value = inputs.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{CROP_LABELS[key]} is required and must be a number")
            if not math.isfinite(value):
                raise ValueError(f"{CROP_LABELS[key]} must be a finite number")
            low, high = CROP_LIMITS[key]
            if not low <= value <= high:
                raise ValueError(f"{CROP_LABELS[key]} must be between {low} and {high}")
            parsed[key] = float(value)
        return parsed


def register_naive_bayes(app: Flask) -> None:
    """Add the Naive Bayes page and API. The rest of the site works without it."""
    try:
        demo = NaiveBayesDemo()
        load_error = None
    except (FileNotFoundError, ValueError) as error:
        demo, load_error = None, str(error)

    def unavailable():
        return jsonify(error="The Naive Bayes demo is not available. " + str(load_error)), 503

    def json_body():
        if not request.is_json:
            raise ValueError("Send the inputs as a JSON request")
        try:
            return request.get_json()
        except BadRequest as error:
            raise ValueError("The request body is not valid JSON") from error

    @app.get("/naive-bayes")
    def naive_bayes_page():
        if demo is None:
            return unavailable()
        return render_template(
            "naive_bayes.html",
            soybean_metrics=demo.metrics["soybean_disease"],
            crop_metrics=demo.metrics["crop_recommendation"],
            soybean_form=demo.soybean_form(),
            crop_form=demo.crop_form(),
            soybean_class_count=len(demo.soybean_schema["classes"]),
            crop_class_count=len(demo.crop_schema["classes"]),
            soybean_symptom_count=len(demo.soybean_features),
        )

    @app.get("/api/nb/soybean/example")
    def soybean_example():
        if demo is None:
            return unavailable()
        return jsonify(demo.soybean_example(request.args.get("n", 0, type=int)))

    @app.get("/api/nb/crop/example")
    def crop_example():
        if demo is None:
            return unavailable()
        return jsonify(demo.crop_example(request.args.get("n", 0, type=int)))

    @app.post("/api/nb/soybean/predict")
    def soybean_predict():
        if demo is None:
            return unavailable()
        try:
            return jsonify(demo.predict_soybean(json_body()))
        except ValueError as error:
            return jsonify(error=str(error)), 400

    @app.post("/api/nb/crop/predict")
    def crop_predict():
        if demo is None:
            return unavailable()
        try:
            return jsonify(demo.predict_crop(json_body()))
        except ValueError as error:
            return jsonify(error=str(error)), 400
