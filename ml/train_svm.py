"""Train and evaluate SVMs for the crop-disease dataset.

For every pair of features (the 2-D chart views) and for all features together,
this picks C (and gamma for RBF) by 5-fold cross-validation on the training
split, then reports held-out test accuracy. The website refits small 2-D SVMs
live when a visitor moves the sliders; these results supply the recommended
settings and the reference numbers.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "SVM" / "crop_disease" / "processed"
OUTPUT = ROOT / "trained_models" / "svm"
C_GRID = [0.1, 0.3, 1, 3, 10, 30, 100]
GAMMA_GRID = [0.1, 0.25, 0.5, 1, 2, 5]
RANDOM_STATE = 42


def search(x_train, y_train, x_test, y_test, kernel: str) -> tuple[dict, object]:
    grid = {"svc__C": C_GRID}
    if kernel == "rbf":
        grid["svc__gamma"] = GAMMA_GRID
    pipeline = make_pipeline(StandardScaler(), SVC(kernel=kernel))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    found = GridSearchCV(pipeline, grid, cv=cv, scoring="accuracy").fit(x_train, y_train)
    svc = found.best_estimator_[-1]
    result = {
        "kernel": kernel,
        "c": float(found.best_params_["svc__C"]),
        "gamma": float(found.best_params_["svc__gamma"]) if kernel == "rbf" else None,
        "cv_accuracy": float(found.best_score_),
        "test_accuracy": float(found.best_estimator_.score(x_test, y_test)),
        "support_vectors": int(svc.n_support_.sum()),
    }
    return result, found.best_estimator_


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    schema = json.loads((DATA / "schema.json").read_text(encoding="utf-8"))
    features = schema["features"]
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y_train, y_test = train["label"], test["label"]
    baseline = float(y_train.value_counts(normalize=True).max())

    views = {}
    for pair in [tuple(features)] + list(combinations(features, 2)):
        key = "+".join(pair)
        views[key] = {}
        for kernel in ("linear", "rbf"):
            result, model = search(train[list(pair)], y_train, test[list(pair)], y_test, kernel)
            views[key][kernel] = result
            if len(pair) == len(features):
                joblib.dump(
                    {"features": list(pair), "classes": schema["classes"], "kernel": kernel, "pipeline": model},
                    OUTPUT / f"crop_disease_all_features_{kernel}.joblib",
                )

    metrics = {"features": features, "majority_baseline": baseline, "views": views}
    (OUTPUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    for key, kernels in views.items():
        line = "  ".join(f"{k}: test={v['test_accuracy']:.3f} C={v['c']:g}" + (f" g={v['gamma']:g}" if v["gamma"] else "") for k, v in kernels.items())
        print(f"{key}\n   {line}")


if __name__ == "__main__":
    main()
