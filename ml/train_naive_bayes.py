"""Train and evaluate the Naive Bayes models.

Soybean disease uses a categorical Naive Bayes that skips symptoms left blank,
so a visitor does not have to know every symptom. Crop recommendation uses
scikit-learn's GaussianNB.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.naive_bayes import GaussianNB


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "NaiveBayes" / "processed"
OUTPUT = ROOT / "trained_models" / "naive_bayes"
ALPHA = 1.0
RANDOM_STATE = 42


def fit_categorical(
    features: np.ndarray, labels: np.ndarray, classes: list[str], cardinalities: list[int], alpha: float
) -> dict[str, object]:
    """Fit class priors and per-feature likelihoods using non-missing cells only."""
    class_index = {name: i for i, name in enumerate(classes)}
    y = np.array([class_index[label] for label in labels])
    priors = np.bincount(y, minlength=len(classes)) / len(y)

    likelihoods = []
    for column, size in enumerate(cardinalities):
        counts = np.full((len(classes), size), alpha, dtype=float)
        for class_id in range(len(classes)):
            values = features[y == class_id, column]
            values = values[~np.isnan(values)].astype(int)
            counts[class_id] += np.bincount(values, minlength=size)
        likelihoods.append(counts / counts.sum(axis=1, keepdims=True))

    return {
        "classes": classes,
        "log_priors": np.log(priors),
        "log_likelihoods": [np.log(table) for table in likelihoods],
        "alpha": alpha,
    }


def predict_categorical(model: dict[str, object], features: np.ndarray) -> np.ndarray:
    """Return class probabilities. NaN cells contribute no evidence."""
    scores = np.tile(model["log_priors"], (len(features), 1))
    for column, table in enumerate(model["log_likelihoods"]):
        values = features[:, column]
        known = ~np.isnan(values)
        scores[known] += table[:, values[known].astype(int)].T
    scores -= scores.max(axis=1, keepdims=True)
    probabilities = np.exp(scores)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def _report(true: pd.Series, predicted: np.ndarray, classes: list[str]) -> dict[str, object]:
    return {
        "accuracy": float(accuracy_score(true, predicted)),
        "macro_f1": float(f1_score(true, predicted, average="macro", labels=classes, zero_division=0)),
        "per_class": classification_report(
            true, predicted, labels=classes, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(true, predicted, labels=classes).tolist(),
        "confusion_labels": classes,
    }


def _load(name: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    directory = PROCESSED / name
    return (
        pd.read_csv(directory / "X_train.csv"),
        pd.read_csv(directory / "X_test.csv"),
        pd.read_csv(directory / "y_train.csv")["target"],
        pd.read_csv(directory / "y_test.csv")["target"],
    )


def train_soybean() -> dict[str, object]:
    x_train, x_test, y_train, y_test = _load("soybean_disease")
    schema = json.loads((PROCESSED / "soybean_disease" / "schema.json").read_text(encoding="utf-8"))
    classes = schema["classes"]
    cardinalities = [len(values) for values in schema["features"].values()]
    if list(x_train.columns) != list(schema["features"]):
        raise ValueError("Soybean feature order does not match schema.json.")

    model = fit_categorical(
        x_train.to_numpy(dtype=float), y_train.to_numpy(), classes, cardinalities, ALPHA
    )
    probabilities = predict_categorical(model, x_test.to_numpy(dtype=float))
    predicted = np.array(classes)[probabilities.argmax(axis=1)]

    # Robustness: hide a share of the recorded symptoms, as a visitor might.
    rng = np.random.default_rng(RANDOM_STATE)
    hidden_accuracy = {}
    values = x_test.to_numpy(dtype=float)
    for share in (0.25, 0.5):
        masked = values.copy()
        masked[rng.random(masked.shape) < share] = np.nan
        masked_pred = np.array(classes)[predict_categorical(model, masked).argmax(axis=1)]
        hidden_accuracy[str(share)] = float(accuracy_score(y_test, masked_pred))

    majority = y_train.value_counts().idxmax()
    metrics = _report(y_test, predicted, classes)
    metrics.update(
        {
            "dataset": "soybean_disease",
            "model": "Categorical Naive Bayes (missing symptoms skipped)",
            "alpha": ALPHA,
            "train_rows": len(x_train),
            "test_rows": len(x_test),
            "majority_class_baseline_accuracy": float((y_test == majority).mean()),
            "accuracy_with_hidden_symptoms": hidden_accuracy,
            "top3_accuracy": float(
                np.mean(
                    [
                        truth in np.array(classes)[np.argsort(row)[-3:]]
                        for truth, row in zip(y_test, probabilities)
                    ]
                )
            ),
        }
    )

    (OUTPUT / "models").mkdir(parents=True, exist_ok=True)
    joblib.dump(model, OUTPUT / "models" / "soybean_disease.joblib")
    _save_predictions("soybean_disease", y_test, predicted, probabilities, classes)
    return metrics


def train_crop() -> dict[str, object]:
    x_train, x_test, y_train, y_test = _load("crop_recommendation")
    model = GaussianNB()
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)
    classes = [str(name) for name in model.classes_]
    predicted = model.predict(x_test)

    majority = y_train.value_counts().idxmax()
    metrics = _report(y_test, predicted, classes)
    metrics.update(
        {
            "dataset": "crop_recommendation",
            "model": "Gaussian Naive Bayes",
            "var_smoothing": model.var_smoothing,
            "train_rows": len(x_train),
            "test_rows": len(x_test),
            "majority_class_baseline_accuracy": float((y_test == majority).mean()),
            "top3_accuracy": float(
                np.mean(
                    [
                        truth in np.array(classes)[np.argsort(row)[-3:]]
                        for truth, row in zip(y_test, probabilities)
                    ]
                )
            ),
        }
    )

    joblib.dump(model, OUTPUT / "models" / "crop_recommendation.joblib")
    _save_predictions("crop_recommendation", y_test, predicted, probabilities, classes)
    return metrics


def _save_predictions(
    name: str, truth: pd.Series, predicted: np.ndarray, probabilities: np.ndarray, classes: list[str]
) -> None:
    (OUTPUT / "predictions").mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "actual": truth.to_numpy(),
            "predicted": predicted,
            "confidence": probabilities.max(axis=1),
        }
    )
    frame.to_csv(OUTPUT / "predictions" / f"{name}_test_predictions.csv", index=False)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    all_metrics = {"soybean_disease": train_soybean(), "crop_recommendation": train_crop()}
    (OUTPUT / "metrics.json").write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")

    summary = pd.DataFrame(
        [
            {
                "dataset": name,
                "model": m["model"],
                "train_rows": m["train_rows"],
                "test_rows": m["test_rows"],
                "accuracy": round(m["accuracy"], 4),
                "macro_f1": round(m["macro_f1"], 4),
                "top3_accuracy": round(m["top3_accuracy"], 4),
                "majority_baseline": round(m["majority_class_baseline_accuracy"], 4),
            }
            for name, m in all_metrics.items()
        ]
    )
    summary.to_csv(OUTPUT / "metrics.csv", index=False)
    print(summary.to_string(index=False))
    print("Soybean accuracy with hidden symptoms:", all_metrics["soybean_disease"]["accuracy_with_hidden_symptoms"])


if __name__ == "__main__":
    main()
