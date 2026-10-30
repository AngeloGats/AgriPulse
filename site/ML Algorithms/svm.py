"""SVM demo: live 2-D SVM fits (decision boundary, margin, support vectors)."""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from werkzeug.exceptions import BadRequest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "trained_models" / "svm"
DATA_DIR = PROJECT_ROOT / "data" / "SVM" / "crop_disease" / "processed"
METRICS_PATH = MODEL_DIR / "metrics.json"
GRID = 70
C_RANGE = (0.01, 100.0)
GAMMA_RANGE = (0.01, 20.0)
DEFAULT_PAIR = ("air_temp_c", "humidity_pct")

# (label, unit, hard min, hard max)
FEATURES: dict[str, tuple[str, str, float, float]] = {
    "air_temp_c": ("Air temperature", "°C", -10, 60),
    "humidity_pct": ("Humidity", "%", 0, 100),
    "leaf_wetness_hours": ("Leaf wetness", "hours/day", 0, 24),
    "red_edge_index": ("Leaf stress index (red edge)", "0–1", 0, 1),
}
CLASS_NAMES = {"healthy": "Healthy", "diseased": "Diseased"}


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)) or value == "":
        raise ValueError(f"{label} must be a number")
    try:
        number = float(value)
    except ValueError:
        raise ValueError(f"{label} must be a number") from None
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


class SvmDemo:
    def __init__(self) -> None:
        required = [METRICS_PATH, DATA_DIR / "schema.json", DATA_DIR / "train.csv", DATA_DIR / "test.csv"]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "The SVM demo is missing required artifacts: "
                + ", ".join(missing)
                + ". Run ml/generate_svm_dataset.py and ml/train_svm.py from the project root first."
            )
        self.metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        self.schema = json.loads((DATA_DIR / "schema.json").read_text(encoding="utf-8"))
        self.features = list(self.schema["features"])
        if self.features != list(FEATURES) or set(self.schema["classes"]) != set(CLASS_NAMES):
            raise ValueError("SVM schema.json does not match the demo configuration")
        self.train = pd.read_csv(DATA_DIR / "train.csv")
        self.test = pd.read_csv(DATA_DIR / "test.csv")
        self._fit = lru_cache(maxsize=128)(self._fit_uncached)

    def page_context(self) -> dict[str, Any]:
        medians = self.train[self.features].median()
        ranges = self.schema["ranges"]
        return {
            "features": [
                {
                    "key": key,
                    "label": FEATURES[key][0],
                    "unit": FEATURES[key][1],
                    "min": FEATURES[key][2],
                    "max": FEATURES[key][3],
                    "median": float(medians[key]),
                    "train_min": ranges[key][0],
                    "train_max": ranges[key][1],
                }
                for key in self.features
            ],
            "default_x": DEFAULT_PAIR[0],
            "default_y": DEFAULT_PAIR[1],
            "baseline": self.metrics["majority_baseline"],
            "full": self.metrics["views"]["+".join(self.features)],
            "train_rows": len(self.train),
            "test_rows": len(self.test),
            "c_range": C_RANGE,
            "gamma_range": GAMMA_RANGE,
        }

    def recommended(self) -> dict[str, Any]:
        return {
            key: {kernel: {"c": v["c"], "gamma": v["gamma"], "test_accuracy": v["test_accuracy"]} for kernel, v in kernels.items()}
            for key, kernels in self.metrics["views"].items()
            if key.count("+") == 1
        }

    def example(self, index: int) -> dict[str, Any]:
        row = self.test.iloc[index % len(self.test)]
        return {
            "inputs": {f: round(float(row[f]), 2) for f in self.features},
            "actual": CLASS_NAMES[row["label"]],
        }

    def _fit_uncached(self, x: str, y: str, kernel: str, c: float, gamma: float | None):
        pair = [x, y]
        options = {"kernel": kernel, "C": c}
        if kernel == "rbf":
            options["gamma"] = gamma
        model = make_pipeline(StandardScaler(), SVC(**options))
        model.fit(self.train[pair], self.train["label"])
        svc = model[-1]
        sign = 1.0 if list(svc.classes_)[1] == "diseased" else -1.0

        ranges = self.schema["ranges"]
        bounds = []
        for key in pair:
            low, high = ranges[key]
            pad = (high - low) * 0.05
            bounds.append((max(FEATURES[key][2], low - pad), min(FEATURES[key][3], high + pad)))
        gx = np.linspace(bounds[0][0], bounds[0][1], GRID)
        gy = np.linspace(bounds[1][0], bounds[1][1], GRID)
        mesh = pd.DataFrame(
            [(a, b) for b in gy for a in gx], columns=pair
        )
        grid = sign * model.decision_function(mesh)
        train_scores = sign * model.decision_function(self.train[pair])
        support = set(svc.support_.tolist())
        return {
            "model": model,
            "sign": sign,
            "bounds": bounds,
            "grid": grid,
            "train_scores": train_scores,
            "support": support,
            "train_accuracy": float(model.score(self.train[pair], self.train["label"])),
            "test_accuracy": float(model.score(self.test[pair], self.test["label"])),
        }

    def parse(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("Send the settings as a JSON object")
        x, y = payload.get("x"), payload.get("y")
        if x not in FEATURES or y not in FEATURES:
            raise ValueError("Choose two of the available measurements")
        if x == y:
            raise ValueError("Choose two different measurements")
        kernel = payload.get("kernel")
        if kernel not in ("linear", "rbf"):
            raise ValueError("Kernel must be 'linear' or 'rbf'")
        c = _number(payload.get("c"), "C")
        if not C_RANGE[0] <= c <= C_RANGE[1]:
            raise ValueError(f"C must be between {C_RANGE[0]:g} and {C_RANGE[1]:g}")
        gamma = None
        if kernel == "rbf":
            gamma = _number(payload.get("gamma"), "Curve tightness (gamma)")
            if not GAMMA_RANGE[0] <= gamma <= GAMMA_RANGE[1]:
                raise ValueError(f"Gamma must be between {GAMMA_RANGE[0]:g} and {GAMMA_RANGE[1]:g}")

        sample = None
        raw = payload.get("sample")
        if raw is not None:
            if not isinstance(raw, dict) or set(raw) != {x, y}:
                raise ValueError("The sample must contain exactly the two chosen measurements")
            sample = {}
            for key in (x, y):
                label, _unit, low, high = FEATURES[key]
                number = _number(raw[key], label)
                if not low <= number <= high:
                    raise ValueError(f"{label} must be between {low:g} and {high:g}")
                sample[key] = number
        return {"x": x, "y": y, "kernel": kernel, "c": round(c, 4), "gamma": None if gamma is None else round(gamma, 4), "sample": sample}

    def model(self, payload: Any) -> dict[str, Any]:
        p = self.parse(payload)
        fit = self._fit(p["x"], p["y"], p["kernel"], p["c"], p["gamma"])
        pair = [p["x"], p["y"]]
        labels = self.train["label"].tolist()
        points = [
            [round(float(a), 2), round(float(b), 2), 1 if label == "diseased" else 0, 1 if index in fit["support"] else 0]
            for index, (a, b, label) in enumerate(zip(self.train[p["x"]], self.train[p["y"]], labels))
        ]
        inside = int((np.abs(fit["train_scores"]) < 1).sum())
        result: dict[str, Any] = {
            "x": p["x"],
            "y": p["y"],
            "kernel": p["kernel"],
            "c": p["c"],
            "gamma": p["gamma"],
            "bounds": [[float(lo), float(hi)] for lo, hi in fit["bounds"]],
            "grid_size": GRID,
            "grid": [round(float(v), 2) for v in fit["grid"]],
            "points": points,
            "support_vectors": len(fit["support"]),
            "train_rows": len(self.train),
            "inside_margin": inside,
            "train_accuracy": fit["train_accuracy"],
            "test_accuracy": fit["test_accuracy"],
            "baseline": self.metrics["majority_baseline"],
            "sample": None,
        }
        if p["sample"] is not None:
            frame = pd.DataFrame([p["sample"]], columns=pair)
            score = float(fit["sign"] * fit["model"].decision_function(frame)[0])
            low, high = self.schema["ranges"][p["x"]], self.schema["ranges"][p["y"]]
            outside = any(
                not (r[0] - 0.1 * (r[1] - r[0]) <= p["sample"][k] <= r[1] + 0.1 * (r[1] - r[0]))
                for k, r in ((p["x"], low), (p["y"], high))
            )
            result["sample"] = {
                "score": round(score, 3),
                "prediction": CLASS_NAMES["diseased" if score > 0 else "healthy"],
                "in_margin": abs(score) < 1,
                "outside_training_range": outside,
            }
        return result


def register_svm(app: Flask) -> None:
    """Add the SVM page and API. The rest of the site works without it."""
    try:
        demo = SvmDemo()
        load_error = None
    except (FileNotFoundError, ValueError) as error:
        demo, load_error = None, str(error)

    def unavailable():
        return jsonify(error="The SVM demo is not available. " + str(load_error)), 503

    @app.get("/svm")
    def svm_page():
        if demo is None:
            return unavailable()
        return render_template("svm.html", ctx=demo.page_context(), recommended=demo.recommended())

    @app.get("/api/svm/example")
    def svm_example():
        if demo is None:
            return unavailable()
        return jsonify(demo.example(request.args.get("n", 0, type=int)))

    @app.post("/api/svm/model")
    def svm_model():
        if demo is None:
            return unavailable()
        if not request.is_json:
            return jsonify(error="Send the settings as a JSON request"), 400
        try:
            payload = request.get_json()
        except BadRequest:
            return jsonify(error="The request body is not valid JSON"), 400
        try:
            return jsonify(demo.model(payload))
        except ValueError as error:
            return jsonify(error=str(error)), 400
