"""Artificial Neural Network demo: classify a leaf image (sample or uploaded) with class scores."""

from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import keras
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_from_directory
from PIL import Image, UnidentifiedImageError
from werkzeug.exceptions import BadRequest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "trained_models" / "ann"
DATA_DIR = PROJECT_ROOT / "data" / "ANN" / "leaf_images"
MODEL_PATH = MODEL_DIR / "leaf_cnn.keras"
METRICS_PATH = MODEL_DIR / "metrics.json"
HISTORY_PATH = MODEL_DIR / "training_history.csv"
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_PIXELS = 40_000_000
GRID = 8  # heat map is GRID x GRID patches
SAMPLES_PER_CLASS = 4

ADVICE = {
    "healthy": "No disease pattern found. Keep scouting regularly.",
    "early_blight": "Dark rings on older leaves suggest early blight, a fungal disease. Remove infected leaves and consider a fungicide.",
    "late_blight": "Dark, water-soaked patches suggest late blight, which spreads fast in cool, wet weather. Act quickly and isolate affected plants.",
    "bacterial_spot": "Small dark spots suggest bacterial spot. Avoid overhead watering and handling wet plants.",
    "yellow_leaf_curl": "Yellowing and curling suggest a virus spread by whiteflies. Remove infected plants and control whiteflies.",
}


class AnnDemo:
    def __init__(self) -> None:
        schema_path = DATA_DIR / "schema.json"
        required = [MODEL_PATH, METRICS_PATH, HISTORY_PATH, schema_path, DATA_DIR / "test.csv"]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "The Neural Network demo is missing required artifacts: "
                + ", ".join(missing)
                + ". Run ml/prepare_ann_images.py (needs the raw PlantVillage download) only if the"
                " committed leaf_images folder is missing, then ml/train_ann.py from the project root."
            )
        self.schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        bundle_classes = list(self.metrics["classes"])
        try:
            self.model = keras.saving.load_model(MODEL_PATH, compile=False)
        except (OSError, ValueError) as error:
            raise ValueError(f"The trained network could not be loaded: {error}") from None
        self.classes = bundle_classes
        self.size = int(self.metrics["image_size"])
        if self.model.input_shape[1:] != (self.size, self.size, 3) or self.model.output_shape[-1] != len(self.classes):
            raise ValueError("The trained network does not match the leaf_images schema")
        if sorted(self.schema["classes"]) != self.classes or set(self.classes) != set(ADVICE):
            raise ValueError("The trained network does not match the leaf_images schema")
        if self.size % GRID:
            raise ValueError("The heat map grid must divide the network input size")
        self.names = self.schema["names"]
        self.history = pd.read_csv(HISTORY_PATH).to_dict("records")
        self.test = pd.read_csv(DATA_DIR / "test.csv")
        self.test_paths = set(self.test["path"])
        self.samples = []
        for key in self.classes:
            for path in self.test[self.test["label"] == key]["path"].head(SAMPLES_PER_CLASS):
                self.samples.append({"path": path, "label": key, "name": self.names[key]})

    def page_context(self) -> dict[str, Any]:
        return {
            "classes": [{"key": c, "name": self.names[c]} for c in self.classes],
            "accuracy": self.metrics["test_accuracy"],
            "baseline": self.metrics["majority_baseline"],
            "parameters": self.metrics["parameters"],
            "filters": self.metrics["conv_filters"],
            "dense": self.metrics["dense"],
            "plain_accuracy": self.metrics["plain_network_accuracy"],
            "inputs": self.metrics["inputs"],
            "size": self.size,
            "epochs": self.metrics["epochs"],
            "train_rows": self.schema["train_rows"],
            "test_rows": self.schema["test_rows"],
            "per_class": self.schema["per_class"],
            "samples": self.samples,
        }

    def training_data(self) -> dict[str, Any]:
        return {
            "classes": self.classes,
            "names": self.names,
            "history": self.history,
            "confusion_matrix": self.metrics["confusion_matrix"],
            "test_accuracy": self.metrics["test_accuracy"],
        }

    # ---- images -------------------------------------------------------------

    def read_upload(self, data: bytes) -> Image.Image:
        try:
            with Image.open(io.BytesIO(data)) as probe:
                if probe.format not in ("JPEG", "PNG"):
                    raise ValueError("Upload a JPEG or PNG photo of a leaf")
                if probe.width * probe.height > MAX_PIXELS:
                    raise ValueError("That image is too large. Upload a photo under 40 megapixels")
            image = Image.open(io.BytesIO(data))
            image.load()
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
            raise ValueError("That file could not be read as an image. Upload a JPEG or PNG photo of a leaf") from None
        return image.convert("RGB")

    def to_input(self, image: Image.Image) -> np.ndarray:
        """Centre-crop to a square, then shrink to the size the network was trained on (0-255 values)."""
        side = min(image.size)
        left, top = (image.width - side) // 2, (image.height - side) // 2
        square = image.crop((left, top, left + side, top + side))
        small = square.resize((self.size, self.size), Image.LANCZOS)
        return np.asarray(small, dtype=np.float32)

    def probabilities(self, pixels: np.ndarray) -> np.ndarray:
        batch = pixels if pixels.ndim == 4 else pixels[None]
        return self.model.predict(batch, verbose=0)

    def classify(self, image: Image.Image) -> dict[str, Any]:
        pixels = self.to_input(image)
        probabilities = self.probabilities(pixels)[0]
        top = int(np.argmax(probabilities))
        key = self.classes[top]

        # Occlusion heat map: grey out one patch at a time and see how much the answer drops.
        patch = self.size // GRID
        fill = pixels.mean(axis=(0, 1))
        variants = []
        for row in range(GRID):
            for col in range(GRID):
                copy = pixels.copy()
                copy[row * patch:(row + 1) * patch, col * patch:(col + 1) * patch] = fill
                variants.append(copy)
        drops = probabilities[top] - self.probabilities(np.stack(variants))[:, top]
        drops = np.clip(drops, 0, None)
        heat = (drops / drops.max() if drops.max() > 0 else drops).reshape(GRID, GRID)

        buffer = io.BytesIO()
        Image.fromarray(pixels.round().astype(np.uint8)).resize((192, 192), Image.NEAREST).save(buffer, "PNG")
        ranked = sorted(zip(self.classes, probabilities), key=lambda item: -item[1])
        return {
            "prediction": key,
            "name": self.names[key],
            "advice": ADVICE[key],
            "confidence": float(probabilities[top]),
            "scores": [{"key": c, "name": self.names[c], "probability": float(p)} for c, p in ranked],
            "heatmap": heat.round(3).tolist(),
            "network_view": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"),
            "network_size": self.size,
            "uncertain": bool(probabilities[top] < 0.5),
        }

    def classify_sample(self, path: Any) -> dict[str, Any]:
        if not isinstance(path, str) or path not in self.test_paths:
            raise ValueError("Unknown sample leaf")
        with Image.open(DATA_DIR / path) as image:
            result = self.classify(image.convert("RGB"))
        actual = path.split("/")[0]
        result["actual"] = actual
        result["actual_name"] = self.names[actual]
        result["correct"] = actual == result["prediction"]
        return result


def register_ann(app: Flask) -> None:
    """Add the Neural Network page and API. The rest of the site works without it."""
    try:
        demo = AnnDemo()
        load_error = None
    except (FileNotFoundError, ValueError) as error:
        demo, load_error = None, str(error)

    def unavailable():
        return jsonify(error="The Neural Network demo is not available. " + str(load_error)), 503

    # Cap request bodies so a huge upload is refused before it is buffered.
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES + 1024 * 1024

    @app.errorhandler(413)
    def too_large(_error):
        return jsonify(error="That image is larger than 5 MB. Upload a smaller photo"), 413

    @app.get("/neural-network")
    def ann_page():
        if demo is None:
            return unavailable()
        return render_template("ann.html", ctx=demo.page_context())

    @app.get("/api/ann/training")
    def ann_training():
        if demo is None:
            return unavailable()
        return jsonify(demo.training_data())

    @app.get("/api/ann/sample-image/<path:path>")
    def ann_sample_image(path: str):
        if demo is None:
            return unavailable()
        if path not in demo.test_paths:
            return jsonify(error="Unknown sample leaf"), 404
        return send_from_directory(DATA_DIR, path, max_age=3600)

    @app.get("/api/ann/random-sample")
    def ann_random_sample():
        if demo is None:
            return unavailable()
        index = request.args.get("n", 0, type=int)
        return jsonify(path=demo.test.iloc[index % len(demo.test)]["path"])

    @app.post("/api/ann/predict")
    def ann_predict():
        if demo is None:
            return unavailable()
        try:
            if request.mimetype == "multipart/form-data":
                upload = request.files.get("image")
                if upload is None:
                    raise ValueError("Choose an image to upload")
                data = upload.stream.read(MAX_UPLOAD_BYTES + 1)
                if len(data) > MAX_UPLOAD_BYTES:
                    return jsonify(error="That image is larger than 5 MB. Upload a smaller photo"), 413
                return jsonify(demo.classify(demo.read_upload(data)))
            if not request.is_json:
                raise ValueError("Send a sample leaf or upload an image")
            try:
                payload = request.get_json()
            except BadRequest:
                raise ValueError("The request body is not valid JSON") from None
            if not isinstance(payload, dict) or set(payload) != {"sample"}:
                raise ValueError("Send the sample leaf as {\"sample\": path}")
            return jsonify(demo.classify_sample(payload["sample"]))
        except ValueError as error:
            return jsonify(error=str(error)), 400
