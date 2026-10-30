"""Train the leaf-image Artificial Neural Network (a small convolutional network).

A convolutional network (CNN) slides small filters over the picture, so it learns local
patterns such as spots, rings and yellowing instead of treating every pixel as unrelated.
Random flips, rotations, zooms and brightness changes are applied while training so the
network sees many versions of each leaf. 15% of the training split is held back for the
validation curve; the test split is used once for the final report.

For comparison, the plain pixel-by-pixel network used before (a scikit-learn MLP on 24x24
pixels) is trained on exactly the same split and its test score is saved as a baseline.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import keras
import numpy as np
import pandas as pd
from keras import layers
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "ANN" / "leaf_images"
OUTPUT = ROOT / "trained_models" / "ann"
MODEL_FILE = "leaf_cnn.keras"
EPOCHS = 60
BATCH = 32
RANDOM_STATE = 42
BASELINE_SIZE = 24


def load(paths: list[str], size: int) -> np.ndarray:
    images = []
    for path in paths:
        with Image.open(DATA / path) as image:
            image = image.convert("RGB")
            if image.size != (size, size):
                image = image.resize((size, size), Image.LANCZOS)
            images.append(np.asarray(image, dtype=np.uint8))
    return np.stack(images)


def build(size: int, classes: int) -> keras.Model:
    return keras.Sequential(
        [
            layers.Input((size, size, 3)),
            layers.Rescaling(1 / 255),
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.1),
            layers.RandomBrightness(0.15, value_range=(0, 1)),
            layers.Conv2D(32, 3, padding="same", use_bias=False, name="conv1"),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding="same", use_bias=False, name="conv2"),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(96, 3, padding="same", use_bias=False, name="conv3"),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(128, 3, padding="same", use_bias=False, name="conv4"),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),
            layers.Dense(64, activation="relu", name="dense"),
            layers.Dense(classes, activation="softmax", name="scores"),
        ]
    )


def baseline_mlp(fit: pd.DataFrame, test: pd.DataFrame, classes: list[str]) -> float:
    """The earlier plain network: 24x24 pixels, flattened, with flipped copies."""
    x_fit = load(fit["path"].tolist(), BASELINE_SIZE).astype(np.float32) / 255.0
    x_test = load(test["path"].tolist(), BASELINE_SIZE).astype(np.float32) / 255.0
    mean, std = x_fit.mean(axis=(0, 1, 2)), x_fit.std(axis=(0, 1, 2))

    def prep(images: np.ndarray) -> np.ndarray:
        return ((images - mean) / std).reshape(len(images), -1)

    flips = [x_fit, x_fit[:, :, ::-1], x_fit[:, ::-1], x_fit[:, ::-1, ::-1]]
    x_aug, y_aug = prep(np.concatenate(flips)), np.concatenate([fit["label"].to_numpy()] * 4)
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), alpha=0.5, batch_size=64, random_state=RANDOM_STATE)
    for _ in range(40):
        mlp.partial_fit(x_aug, y_aug, classes=np.array(classes))
    return float(accuracy_score(test["label"], mlp.predict(prep(x_test))))


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for old in ("leaf_images.joblib", "weights.json", "leaf_condition_test_predictions.csv"):
        (OUTPUT / old).unlink(missing_ok=True)
    keras.utils.set_random_seed(RANDOM_STATE)

    schema = json.loads((DATA / "schema.json").read_text(encoding="utf-8"))
    classes = sorted(schema["classes"])
    size = int(schema["image_size"])
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    fit, val = train_test_split(train, test_size=0.15, stratify=train["label"], random_state=RANDOM_STATE)

    index = {name: i for i, name in enumerate(classes)}
    x_fit, x_val, x_test = (load(d["path"].tolist(), size) for d in (fit, val, test))
    y_fit, y_val, y_test = (d["label"].map(index).to_numpy() for d in (fit, val, test))

    model = build(size, len(classes))
    model.compile(optimizer=keras.optimizers.Adam(0.001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    stop = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=12, restore_best_weights=True)
    reduce = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5)
    fitted = model.fit(
        x_fit, y_fit, validation_data=(x_val, y_val), epochs=EPOCHS, batch_size=BATCH,
        callbacks=[stop, reduce], verbose=2,
    )
    log = fitted.history
    history = [
        {
            "epoch": i + 1,
            "loss": float(log["loss"][i]),
            "train_accuracy": float(log["accuracy"][i]),
            "val_accuracy": float(log["val_accuracy"][i]),
        }
        for i in range(len(log["loss"]))
    ]

    probabilities = model.predict(x_test, verbose=0)
    predicted = probabilities.argmax(axis=1)
    actual_names = [classes[i] for i in y_test]
    predicted_names = [classes[i] for i in predicted]
    model.save(OUTPUT / MODEL_FILE)

    baseline = baseline_mlp(fit, test, classes)
    pd.DataFrame(history).to_csv(OUTPUT / "training_history.csv", index=False)
    pd.DataFrame({"path": test["path"], "actual": actual_names, "predicted": predicted_names}).to_csv(
        OUTPUT / "leaf_images_test_predictions.csv", index=False
    )
    metrics = {
        "dataset": schema["dataset"],
        "model_type": "cnn",
        "classes": classes,
        "image_size": size,
        "inputs": size * size * 3,
        "batch_norm": True,
        "conv_filters": [32, 64, 96, 128],
        "dense": 64,
        "epochs": len(history),
        "max_epochs": EPOCHS,
        "parameters": int(model.count_params()),
        "test_accuracy": float(accuracy_score(actual_names, predicted_names)),
        "test_macro_f1": float(f1_score(actual_names, predicted_names, average="macro")),
        "majority_baseline": float(1 / len(classes)),
        "plain_network_accuracy": baseline,
        "confusion_matrix": confusion_matrix(actual_names, predicted_names, labels=classes).tolist(),
        "final_loss": history[-1]["loss"],
        "final_val_accuracy": history[-1]["val_accuracy"],
    }
    (OUTPUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(
        f"CNN test={metrics['test_accuracy']:.3f} f1={metrics['test_macro_f1']:.3f} "
        f"params={metrics['parameters']} epochs={len(history)} | plain network test={baseline:.3f}"
    )
    print(np.array(metrics["confusion_matrix"]))


if __name__ == "__main__":
    main()
