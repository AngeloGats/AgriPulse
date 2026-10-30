"""Decision Tree demo: tree structure, decision path and plain-language rules."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.tree import DecisionTreeClassifier
from werkzeug.exceptions import BadRequest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "trained_models" / "decision_tree"
DATA_DIR = PROJECT_ROOT / "data" / "DecisionTree" / "irrigation" / "processed"
MODEL_PATH = MODEL_DIR / "irrigation.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
MIN_DEPTH, MAX_DEPTH = 1, 5

# (label, unit, hard min, hard max)
FEATURES: dict[str, tuple[str, str, float, float]] = {
    "soil_moisture_pct": ("Soil moisture", "%", 0, 100),
    "rain_last_3d_mm": ("Rain in the last 3 days", "mm", 0, 500),
    "forecast_rain_mm": ("Forecast rain (next 3 days)", "mm", 0, 500),
    "crop_stage": ("Crop stage", "", 1, 4),
    "air_temp_c": ("Air temperature", "°C", -10, 60),
}
STAGE_NAMES = {1: "seedling", 2: "vegetative", 3: "flowering", 4: "grain fill"}
SHORT_LABELS = {
    "soil_moisture_pct": "Soil moisture",
    "rain_last_3d_mm": "Rain (3 days)",
    "forecast_rain_mm": "Forecast rain",
    "crop_stage": "Crop stage",
    "air_temp_c": "Air temp",
}
ACTIONS: dict[str, tuple[str, str]] = {
    "irrigate_now": ("Irrigate now", "The soil is drier than this crop stage can tolerate and no rain is coming. Water today."),
    "irrigate_soon": ("Irrigate soon", "Moisture is getting low for this crop stage. Plan to water in the next day or two."),
    "wait": ("Wait", "The soil has enough water for now, or rain is on the way. Check again in a day or two."),
    "hold": ("Hold off", "The soil is already wet. Watering now would waste water and risk waterlogging."),
}


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


def _g(value: float) -> str:
    return f"{value:.1f}".rstrip("0").rstrip(".")


class DecisionTreeDemo:
    def __init__(self) -> None:
        required = [MODEL_PATH, METRICS_PATH, DATA_DIR / "schema.json", DATA_DIR / "train.csv", DATA_DIR / "test.csv"]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "The Decision Tree demo is missing required artifacts: "
                + ", ".join(missing)
                + ". Run ml/generate_decision_tree_dataset.py and ml/train_decision_tree.py from the project root first."
            )
        self.metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        self.schema = json.loads((DATA_DIR / "schema.json").read_text(encoding="utf-8"))
        self.features = list(self.schema["features"])
        self.classes = list(self.schema["classes"])
        if self.features != list(FEATURES) or set(self.classes) != set(ACTIONS):
            raise ValueError("Decision Tree schema.json does not match the demo configuration")
        bundle = joblib.load(MODEL_PATH)
        if list(bundle["features"]) != self.features or list(bundle["classes"]) != self.classes:
            raise ValueError("The trained tree does not match schema.json")
        self.recommended = int(bundle["depth"])
        self.train = pd.read_csv(DATA_DIR / "train.csv")
        self.test = pd.read_csv(DATA_DIR / "test.csv")
        self.trees = {
            depth: DecisionTreeClassifier(max_depth=depth, min_samples_leaf=8, random_state=42).fit(
                self.train[self.features], self.train["label"]
            )
            for depth in range(MIN_DEPTH, MAX_DEPTH + 1)
        }
        self.by_depth = {row["depth"]: row for row in self.metrics["accuracy_by_depth"]}

    # ---- page and structure -------------------------------------------------

    def page_context(self) -> dict[str, Any]:
        medians = self.train[self.features].median()
        ranges = self.schema["ranges"]
        fields = []
        for key in self.features:
            label, unit, low, high = FEATURES[key]
            fields.append(
                {
                    "key": key,
                    "label": label,
                    "unit": unit,
                    "min": low,
                    "max": high,
                    "median": float(medians[key]),
                    "train_min": ranges[key][0],
                    "train_max": ranges[key][1],
                }
            )
        return {
            "fields": fields,
            "stages": [{"code": code, "name": name.capitalize()} for code, name in STAGE_NAMES.items()],
            "classes": [{"key": c, "name": ACTIONS[c][0]} for c in self.classes],
            "min_depth": MIN_DEPTH,
            "max_depth": MAX_DEPTH,
            "recommended": self.recommended,
            "accuracy": self.metrics["test_accuracy"],
            "baseline": self.metrics["majority_baseline"],
            "train_rows": len(self.train),
            "test_rows": len(self.test),
        }

    def _split_text(self, feature: str, threshold: float) -> tuple[str, str, str, str]:
        """Return (question, rule when yes, rule when no, short chart label) in plain words."""
        label, unit, _low, _high = FEATURES[feature]
        if feature == "crop_stage":
            low = [STAGE_NAMES[c] for c in STAGE_NAMES if c <= threshold]
            high = [STAGE_NAMES[c] for c in STAGE_NAMES if c > threshold]
            left, right = " or ".join(low), " or ".join(high)
            return (
                f"Crop stage: {left}?",
                f"Crop stage is {left}",
                f"Crop stage is {right}",
                f"Stage: {' / '.join(low)}",
            )
        value = f"{_g(threshold)}{unit if unit in ('%', '°C') else ' ' + unit}"
        return (
            f"{label} {value} or less?",
            f"{label} is {value} or less",
            f"{label} is above {value}",
            f"{SHORT_LABELS[feature]} ≤ {value}",
        )

    def tree_data(self, depth: int) -> dict[str, Any]:
        tree = self._tree(depth)
        t = tree.tree_
        nodes = []
        for node_id in range(t.node_count):
            fractions = t.value[node_id][0]
            fractions = fractions / fractions.sum()
            counts = np.rint(fractions * t.n_node_samples[node_id]).astype(int)
            top = int(np.argmax(counts))
            node: dict[str, Any] = {
                "id": node_id,
                "samples": int(t.n_node_samples[node_id]),
                "counts": {c: int(counts[i]) for i, c in enumerate(tree.classes_)},
                "prediction": str(tree.classes_[top]),
                "action": ACTIONS[str(tree.classes_[top])][0],
                "purity": float(counts[top] / max(counts.sum(), 1)),
                "left": int(t.children_left[node_id]) if t.children_left[node_id] != -1 else None,
                "right": int(t.children_right[node_id]) if t.children_right[node_id] != -1 else None,
            }
            if node["left"] is not None:
                feature = self.features[t.feature[node_id]]
                question, yes, no, short = self._split_text(feature, float(t.threshold[node_id]))
                node.update(feature=feature, question=question, yes=yes, no=no, short=short)
            nodes.append(node)
        metrics = self.by_depth[depth]
        return {
            "depth": depth,
            "nodes": nodes,
            "leaves": int(tree.get_n_leaves()),
            "classes": [{"key": c, "name": ACTIONS[c][0]} for c in self.classes],
            "train_accuracy": metrics["train_accuracy"],
            "test_accuracy": metrics["test_accuracy"],
            "cv_accuracy": metrics["cv_accuracy"],
            "recommended": self.recommended,
            "importance": sorted(
                (
                    {"key": f, "label": FEATURES[f][0], "value": float(v)}
                    for f, v in zip(self.features, tree.feature_importances_)
                ),
                key=lambda item: -item["value"],
            ),
            "accuracy_by_depth": [
                {"depth": d, "test_accuracy": r["test_accuracy"], "cv_accuracy": r["cv_accuracy"], "leaves": r["leaves"]}
                for d, r in sorted(self.by_depth.items())
            ],
        }

    def _tree(self, depth: Any) -> DecisionTreeClassifier:
        if isinstance(depth, bool) or not isinstance(depth, int) or depth not in self.trees:
            raise ValueError(f"Depth must be a whole number from {MIN_DEPTH} to {MAX_DEPTH}")
        return self.trees[depth]

    # ---- inputs and prediction ---------------------------------------------

    def example(self, index: int) -> dict[str, Any]:
        row = self.test.iloc[index % len(self.test)]
        return {
            "inputs": {f: (int(row[f]) if f == "crop_stage" else round(float(row[f]), 1)) for f in self.features},
            "actual": ACTIONS[row["label"]][0],
        }

    def parse_inputs(self, payload: Any) -> tuple[dict[str, float], list[str]]:
        if not isinstance(payload, dict) or not isinstance(payload.get("inputs"), dict):
            raise ValueError("Send the conditions as an 'inputs' object")
        values = payload["inputs"]
        unknown = sorted(set(values) - set(self.features))
        if unknown:
            raise ValueError("Unknown inputs: " + ", ".join(unknown))
        parsed: dict[str, float] = {}
        warnings: list[str] = []
        for key in self.features:
            label, _unit, low, high = FEATURES[key]
            if key not in values:
                raise ValueError(f"{label} is required")
            number = _number(values[key], label)
            if not low <= number <= high:
                raise ValueError(f"{label} must be between {low:g} and {high:g}")
            if key == "crop_stage" and number != int(number):
                raise ValueError("Crop stage must be one of the listed stages")
            parsed[key] = number
            lo, hi = self.schema["ranges"][key]
            if key != "crop_stage" and not lo - 0.1 * (hi - lo) <= number <= hi + 0.1 * (hi - lo):
                warnings.append(
                    f"{label} is outside the range the tree learned from ({_g(lo)} to {_g(hi)}), so this advice is less reliable."
                )
        return parsed, warnings

    def predict(self, payload: Any) -> dict[str, Any]:
        inputs, warnings = self.parse_inputs(payload)
        depth = payload.get("depth", self.recommended)
        tree = self._tree(depth)
        frame = pd.DataFrame([inputs], columns=self.features)
        t = tree.tree_
        path_ids = tree.decision_path(frame).indices.tolist()
        data = self.tree_data(depth)["nodes"]
        steps = []
        for node_id, next_id in zip(path_ids, path_ids[1:]):
            node = data[node_id]
            went_left = node["left"] == next_id
            steps.append(
                {
                    "node": node_id,
                    "question": node["question"],
                    "answer": "Yes" if went_left else "No",
                    "rule": node["yes"] if went_left else node["no"],
                    "value": inputs[node["feature"]],
                }
            )
        leaf = data[path_ids[-1]]
        by_depth = []
        for d, other in sorted(self.trees.items()):
            key = str(other.predict(frame)[0])
            by_depth.append({"depth": d, "prediction": key, "action": ACTIONS[key][0]})
        key = leaf["prediction"]
        return {
            "depth": depth,
            "recommended_depth": self.recommended,
            "path": path_ids,
            "steps": steps,
            "leaf": leaf["id"],
            "prediction": key,
            "action": ACTIONS[key][0],
            "advice": ACTIONS[key][1],
            "leaf_samples": leaf["samples"],
            "leaf_purity": leaf["purity"],
            "leaf_counts": [
                {"key": c, "name": ACTIONS[c][0], "count": leaf["counts"][c]} for c in self.classes
            ],
            "by_depth": by_depth,
            "warnings": warnings,
        }


def register_decision_tree(app: Flask) -> None:
    """Add the Decision Tree page and API. The rest of the site works without it."""
    try:
        demo = DecisionTreeDemo()
        load_error = None
    except (FileNotFoundError, ValueError) as error:
        demo, load_error = None, str(error)

    def unavailable():
        return jsonify(error="The Decision Tree demo is not available. " + str(load_error)), 503

    @app.get("/decision-tree")
    def decision_tree_page():
        if demo is None:
            return unavailable()
        return render_template("decision_tree.html", ctx=demo.page_context())

    @app.get("/api/dt/tree")
    def decision_tree_structure():
        if demo is None:
            return unavailable()
        raw = request.args.get("depth", str(demo.recommended))
        try:
            depth = int(raw)
            demo._tree(depth)
            return jsonify(demo.tree_data(depth))
        except ValueError as error:
            return jsonify(error=str(error)), 400

    @app.get("/api/dt/example")
    def decision_tree_example():
        if demo is None:
            return unavailable()
        return jsonify(demo.example(request.args.get("n", 0, type=int)))

    @app.post("/api/dt/predict")
    def decision_tree_predict():
        if demo is None:
            return unavailable()
        if not request.is_json:
            return jsonify(error="Send the conditions as a JSON request"), 400
        try:
            payload = request.get_json()
        except BadRequest:
            return jsonify(error="The request body is not valid JSON"), 400
        try:
            return jsonify(demo.predict(payload))
        except ValueError as error:
            return jsonify(error=str(error)), 400
