"""Train and evaluate KNN for each generated dataset.

The scaler is fitted on training rows only. K and the vote weighting are chosen
by 5-fold cross-validation on the training split; the test split is used once
for reporting, plus an accuracy-by-K curve for the website chart.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "kNN" / "processed"
OUTPUT = ROOT / "trained_models" / "knn"
DATASETS = ["soil_type", "soil_fertility", "plant_condition"]
K_VALUES = list(range(1, 26))
WEIGHTINGS = ["uniform", "distance"]
RANDOM_STATE = 42


def train(name: str) -> dict:
    schema = json.loads((DATA / name / "schema.json").read_text(encoding="utf-8"))
    features = schema["features"]
    train_df = pd.read_csv(DATA / name / "train.csv")
    test_df = pd.read_csv(DATA / name / "test.csv")
    x_train, y_train = train_df[features], train_df["label"]
    x_test, y_test = test_df[features], test_df["label"]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores: dict[str, dict[int, float]] = {w: {} for w in WEIGHTINGS}
    for weights in WEIGHTINGS:
        for k in K_VALUES:
            model = make_pipeline(StandardScaler(), KNeighborsClassifier(k, weights=weights))
            cv_scores[weights][k] = float(cross_val_score(model, x_train, y_train, cv=cv).mean())

    # Ties go to the larger K, which gives steadier votes.
    best_weights, best_k = max(
        ((w, k) for w in WEIGHTINGS for k in K_VALUES),
        key=lambda wk: (round(cv_scores[wk[0]][wk[1]], 6), wk[1]),
    )

    curve = []
    for k in K_VALUES:
        probe = make_pipeline(StandardScaler(), KNeighborsClassifier(k, weights=best_weights))
        probe.fit(x_train, y_train)
        curve.append({
            "k": k,
            "train_accuracy": float(probe.score(x_train, y_train)),
            "test_accuracy": float(probe.score(x_test, y_test)),
            "cv_accuracy": cv_scores[best_weights][k],
        })

    final = make_pipeline(StandardScaler(), KNeighborsClassifier(best_k, weights=best_weights))
    final.fit(x_train, y_train)
    predictions = final.predict(x_test)
    classes = list(final.classes_)

    joblib.dump({
        "dataset": name,
        "features": features,
        "classes": classes,
        "k": best_k,
        "weights": best_weights,
        "pipeline": final,
    }, OUTPUT / f"{name}.joblib")

    baseline = float(y_train.value_counts(normalize=True).max())
    pd.DataFrame({"sample_id": test_df["sample_id"], "actual": y_test, "predicted": predictions}) \
        .to_csv(OUTPUT / f"{name}_test_predictions.csv", index=False)

    return {
        "dataset": name,
        "features": features,
        "classes": classes,
        "best_k": best_k,
        "best_weights": best_weights,
        "cv_accuracy": cv_scores[best_weights][best_k],
        "test_accuracy": float(accuracy_score(y_test, predictions)),
        "test_macro_f1": float(f1_score(y_test, predictions, average="macro")),
        "majority_baseline": baseline,
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=classes).tolist(),
        "accuracy_by_k": curve,
        "weighting_comparison": {
            w: {"best_k": max(K_VALUES, key=lambda k: (round(cv_scores[w][k], 6), k)),
                "best_cv_accuracy": max(cv_scores[w].values())}
            for w in WEIGHTINGS
        },
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results = [train(name) for name in DATASETS]
    (OUTPUT / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    pd.DataFrame([
        {k: r[k] for k in ("dataset", "best_k", "best_weights", "cv_accuracy",
                           "test_accuracy", "test_macro_f1", "majority_baseline")}
        for r in results
    ]).to_csv(OUTPUT / "metrics.csv", index=False)
    for r in results:
        print(f"{r['dataset']}: K={r['best_k']} ({r['best_weights']}) cv={r['cv_accuracy']:.3f} "
              f"test={r['test_accuracy']:.3f} f1={r['test_macro_f1']:.3f} baseline={r['majority_baseline']:.3f}")


if __name__ == "__main__":
    main()
