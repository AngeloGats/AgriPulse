"""KNN demo: model loading, neighbor search, PCA chart projection and vote counts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.decomposition import PCA
from werkzeug.exceptions import BadRequest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "trained_models" / "knn"
DATA_DIR = PROJECT_ROOT / "data" / "kNN" / "processed"
METRICS_PATH = MODEL_DIR / "metrics.json"
MAX_K = 25

# Plain-language names; limits are hard physical bounds, not training ranges.
DATASETS: dict[str, dict[str, Any]] = {
    "soil_type": {
        "title": "Soil type",
        "tab_hint": "Lab measurements → soil texture",
        "question": "What kind of soil is this?",
        "result_title": "Predicted soil type",
        "classes": {"sandy": "Sandy", "loamy": "Loamy", "clay": "Clay", "silty": "Silty"},
        "features": {
            "sand_pct": ("Sand", "%", 0, 100),
            "silt_pct": ("Silt", "%", 0, 100),
            "clay_pct": ("Clay", "%", 0, 100),
            "moisture_pct": ("Soil moisture", "%", 0, 100),
            "ph": ("Soil pH", "", 0, 14),
            "organic_matter_pct": ("Organic matter", "%", 0, 100),
            "bulk_density_g_cm3": ("Soil compaction (bulk density)", "g/cm³", 0.3, 2.5),
        },
    },
    "soil_fertility": {
        "title": "Soil fertility",
        "tab_hint": "Nutrient tests → fertility level",
        "question": "How fertile is this soil?",
        "result_title": "Predicted fertility",
        "classes": {"low": "Low", "medium": "Medium", "high": "High"},
        "features": {
            "ph": ("Soil pH", "", 0, 14),
            "organic_carbon_pct": ("Organic carbon", "%", 0, 100),
            "nitrogen_kg_ha": ("Available nitrogen", "kg/ha", 0, 1000),
            "phosphorus_kg_ha": ("Available phosphorus", "kg/ha", 0, 500),
            "potassium_kg_ha": ("Available potassium", "kg/ha", 0, 1000),
            "ec_ds_m": ("Salinity (EC)", "dS/m", 0, 20),
            "zinc_ppm": ("Zinc", "ppm", 0, 50),
        },
    },
    "plant_condition": {
        "title": "Plant condition",
        "tab_hint": "Canopy readings → crop health",
        "question": "What condition is this crop in?",
        "result_title": "Predicted crop condition",
        "classes": {
            "healthy": "Healthy",
            "water_stressed": "Water stressed",
            "nutrient_deficient": "Nutrient deficient",
            "pest_stressed": "Pest stressed",
        },
        "features": {
            "ndvi": ("Greenness index (NDVI)", "0–1", 0, 1),
            "chlorophyll_spad": ("Leaf chlorophyll (SPAD)", "", 0, 100),
            "canopy_temp_excess_c": ("Canopy warmer than air", "°C", -15, 30),
            "soil_moisture_pct": ("Soil moisture", "%", 0, 100),
            "leaf_damage_pct": ("Leaf damage", "%", 0, 100),
        },
    },
}


class KnnDemo:
    """Loads every trained KNN model and answers neighbor-search requests."""

    def __init__(self) -> None:
        required = [METRICS_PATH]
        for name in DATASETS:
            required += [
                MODEL_DIR / f"{name}.joblib",
                DATA_DIR / name / "schema.json",
                DATA_DIR / name / "train.csv",
                DATA_DIR / name / "test.csv",
            ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "The KNN demo is missing required trained artifacts: "
                + ", ".join(missing)
                + ". Run ml/prepare_knn_data.py and ml/train_knn.py from the project root first."
            )

        self.metrics = {m["dataset"]: m for m in json.loads(METRICS_PATH.read_text(encoding="utf-8"))}
        self.sets: dict[str, dict[str, Any]] = {}
        for name, config in DATASETS.items():
            bundle = joblib.load(MODEL_DIR / f"{name}.joblib")
            schema = json.loads((DATA_DIR / name / "schema.json").read_text(encoding="utf-8"))
            features = list(schema["features"])
            if list(bundle["features"]) != features or list(config["features"]) != features:
                raise ValueError(f"{name}: feature order does not match schema.json")
            if list(bundle["classes"]) != schema["classes"] or set(config["classes"]) != set(schema["classes"]):
                raise ValueError(f"{name}: classes do not match schema.json")

            train = pd.read_csv(DATA_DIR / name / "train.csv")
            test = pd.read_csv(DATA_DIR / name / "test.csv")
            pipeline = bundle["pipeline"]
            scaler, knn = pipeline[0], pipeline[-1]
            scaled_train = scaler.transform(train[features])
            pca = PCA(n_components=2, svd_solver="full").fit(scaled_train)
            coords = pca.transform(scaled_train)

            self.sets[name] = {
                "bundle": bundle,
                "schema": schema,
                "features": features,
                "classes": list(bundle["classes"]),
                "scaler": scaler,
                "knn": knn,
                "pca": pca,
                "train": train,
                "train_coords": coords,
                "test": test,
                "k": int(bundle["k"]),
                "weights": bundle["weights"],
                "max_k": min(MAX_K, len(train)),
            }

    def dataset(self, name: str) -> dict[str, Any]:
        if name not in self.sets:
            raise ValueError(f"Unknown dataset '{name}'")
        return self.sets[name]

    def page_context(self) -> list[dict[str, Any]]:
        sets = []
        for name, config in DATASETS.items():
            state = self.sets[name]
            ranges = state["schema"]["ranges"]
            medians = state["train"][state["features"]].median()
            metrics = self.metrics[name]
            sets.append(
                {
                    "name": name,
                    "title": config["title"],
                    "tab_hint": config["tab_hint"],
                    "question": config["question"],
                    "result_title": config["result_title"],
                    "class_count": len(state["classes"]),
                    "feature_count": len(state["features"]),
                    "train_rows": len(state["train"]),
                    "test_rows": len(state["test"]),
                    "best_k": state["k"],
                    "weights": state["weights"],
                    "accuracy": metrics["test_accuracy"],
                    "baseline": metrics["majority_baseline"],
                    "features": [
                        {
                            "key": key,
                            "label": config["features"][key][0],
                            "unit": config["features"][key][1],
                            "min": config["features"][key][2],
                            "max": config["features"][key][3],
                            "median": float(medians[key]),
                            "train_min": ranges[key][0],
                            "train_max": ranges[key][1],
                        }
                        for key in state["features"]
                    ],
                }
            )
        return sets

    def chart_data(self, name: str) -> dict[str, Any]:
        state = self.dataset(name)
        config = DATASETS[name]
        coords = state["train_coords"]
        labels = state["train"]["label"].tolist()
        class_index = {c: i for i, c in enumerate(state["classes"])}
        return {
            "dataset": name,
            "classes": [{"key": c, "name": config["classes"][c]} for c in state["classes"]],
            "points": [
                [round(float(x), 3), round(float(y), 3), class_index[label]]
                for (x, y), label in zip(coords, labels)
            ],
            "explained_variance": [round(float(v), 4) for v in state["pca"].explained_variance_ratio_],
            "best_k": state["k"],
            "weights": state["weights"],
            "max_k": state["max_k"],
            "accuracy_by_k": [
                {"k": row["k"], "test_accuracy": row["test_accuracy"], "cv_accuracy": row["cv_accuracy"]}
                for row in self.metrics[name]["accuracy_by_k"]
            ],
        }

    def example(self, name: str, index: int) -> dict[str, Any]:
        state = self.dataset(name)
        config = DATASETS[name]
        position = index % len(state["test"])
        row = state["test"].iloc[position]
        return {
            "inputs": {f: round(float(row[f]), 3) for f in state["features"]},
            "actual": config["classes"][row["label"]],
        }

    def parse_inputs(self, name: str, payload: Any) -> tuple[dict[str, float], list[str]]:
        state = self.dataset(name)
        config = DATASETS[name]
        if not isinstance(payload, dict):
            raise ValueError("Send the inputs as a JSON object")
        values = payload.get("inputs")
        if not isinstance(values, dict):
            raise ValueError("Send the measurements as an 'inputs' object")
        unknown = sorted(set(values) - set(state["features"]))
        if unknown:
            raise ValueError("Unknown inputs: " + ", ".join(unknown))

        parsed: dict[str, float] = {}
        warnings: list[str] = []
        ranges = state["schema"]["ranges"]
        for key in state["features"]:
            label, _unit, low, high = config["features"][key]
            if key not in values or values[key] in (None, ""):
                raise ValueError(f"{label} is required")
            raw = values[key]
            if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
                raise ValueError(f"{label} must be a number")
            try:
                number = float(raw)
            except ValueError:
                raise ValueError(f"{label} must be a number") from None
            if not math.isfinite(number):
                raise ValueError(f"{label} must be a finite number")
            if not low <= number <= high:
                raise ValueError(f"{label} must be between {low:g} and {high:g}")
            parsed[key] = number
            lo, hi = ranges[key]
            span = hi - lo
            if number < lo - 0.1 * span or number > hi + 0.1 * span:
                warnings.append(
                    f"{label} is outside the range the model has seen ({lo:g} to {hi:g}), "
                    "so its neighbors are far away and the answer is less reliable."
                )
        if name == "soil_type":
            total = parsed["sand_pct"] + parsed["silt_pct"] + parsed["clay_pct"]
            if abs(total - 100) > 5:
                warnings.append(f"Sand, silt and clay add up to {total:.0f}%. They normally add up to 100%.")
        return parsed, warnings

    def predict(self, name: str, payload: Any) -> dict[str, Any]:
        state = self.dataset(name)
        config = DATASETS[name]
        inputs, warnings = self.parse_inputs(name, payload)
        k = payload.get("k", state["k"])
        if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= state["max_k"]:
            raise ValueError(f"K must be a whole number from 1 to {state['max_k']}")

        frame = pd.DataFrame([inputs], columns=state["features"])
        scaled = state["scaler"].transform(frame)
        distances, indices = state["knn"].kneighbors(scaled, n_neighbors=state["max_k"])
        distances, indices = distances[0], indices[0]
        neighbor_labels = state["train"]["label"].to_numpy()[indices]
        classes = state["classes"]

        def vote(count: int) -> dict[str, Any]:
            d = distances[:count]
            if state["weights"] == "distance":
                weights = (d == 0).astype(float) if (d == 0).any() else 1.0 / d
            else:
                weights = np.ones(count)
            totals = {c: float(weights[neighbor_labels[:count] == c].sum()) for c in classes}
            grand = sum(totals.values()) or 1.0
            share = {c: v / grand for c, v in totals.items()}
            counts = {c: int((neighbor_labels[:count] == c).sum()) for c in classes}
            winner = max(classes, key=lambda c: (share[c], -classes.index(c)))
            return {"winner": winner, "share": share, "counts": counts}

        by_k = []
        for count in range(1, state["max_k"] + 1):
            result = vote(count)
            by_k.append(
                {
                    "k": count,
                    "prediction": config["classes"][result["winner"]],
                    "prediction_key": result["winner"],
                    "votes": [
                        {
                            "key": c,
                            "name": config["classes"][c],
                            "count": result["counts"][c],
                            "share": result["share"][c],
                        }
                        for c in classes
                    ],
                }
            )

        point = state["pca"].transform(scaled)[0]
        class_index = {c: i for i, c in enumerate(classes)}
        train_rows = state["train"].iloc[indices]
        neighbors = [
            {
                "rank": rank + 1,
                "distance": round(float(distances[rank]), 3),
                "label": config["classes"][neighbor_labels[rank]],
                "class_index": class_index[neighbor_labels[rank]],
                "x": round(float(state["train_coords"][indices[rank]][0]), 3),
                "y": round(float(state["train_coords"][indices[rank]][1]), 3),
                "values": {f: round(float(train_rows.iloc[rank][f]), 3) for f in state["features"]},
            }
            for rank in range(state["max_k"])
        ]
        chosen = by_k[k - 1]
        previous = by_k[state["k"] - 1]
        return {
            "dataset": name,
            "k": k,
            "prediction": chosen["prediction"],
            "prediction_key": chosen["prediction_key"],
            "class_index": class_index[chosen["prediction_key"]],
            "votes": chosen["votes"],
            "by_k": by_k,
            "recommended_k": state["k"],
            "recommended_prediction": previous["prediction"],
            "neighbors": neighbors,
            "sample_point": [round(float(point[0]), 3), round(float(point[1]), 3)],
            "nearest_distance": round(float(distances[0]), 3),
            "warnings": warnings,
        }


def register_knn(app: Flask) -> None:
    """Add the KNN page and API. The rest of the site works without it."""
    try:
        demo = KnnDemo()
        load_error = None
    except (FileNotFoundError, ValueError) as error:
        demo, load_error = None, str(error)

    def unavailable():
        return jsonify(error="The KNN demo is not available. " + str(load_error)), 503

    @app.get("/knn")
    def knn_page():
        if demo is None:
            return unavailable()
        return render_template("knn.html", datasets=demo.page_context(), max_k=MAX_K)

    @app.get("/api/knn/<name>/chart")
    def knn_chart(name: str):
        if demo is None:
            return unavailable()
        try:
            return jsonify(demo.chart_data(name))
        except ValueError as error:
            return jsonify(error=str(error)), 404

    @app.get("/api/knn/<name>/example")
    def knn_example(name: str):
        if demo is None:
            return unavailable()
        try:
            return jsonify(demo.example(name, request.args.get("n", 0, type=int)))
        except ValueError as error:
            return jsonify(error=str(error)), 404

    @app.post("/api/knn/<name>/predict")
    def knn_predict(name: str):
        if demo is None:
            return unavailable()
        if name not in DATASETS:
            return jsonify(error=f"Unknown dataset '{name}'"), 404
        if not request.is_json:
            return jsonify(error="Send the inputs as a JSON request"), 400
        try:
            payload = request.get_json()
        except BadRequest:
            return jsonify(error="The request body is not valid JSON"), 400
        try:
            return jsonify(demo.predict(name, payload))
        except ValueError as error:
            return jsonify(error=str(error)), 400
