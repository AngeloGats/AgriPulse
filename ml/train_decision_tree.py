"""Train and evaluate the irrigation Decision Tree.

Depth 1 to MAX_DEPTH is compared by 5-fold cross-validation on the training
split. The recommended depth is the shallowest one within one standard error of
the best score, which keeps the flowchart readable. The test split is used once
for reporting.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "DecisionTree" / "irrigation" / "processed"
OUTPUT = ROOT / "trained_models" / "decision_tree"
MAX_DEPTH = 5
RANDOM_STATE = 42


def new_tree(depth: int) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(max_depth=depth, min_samples_leaf=8, random_state=RANDOM_STATE)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    schema = json.loads((DATA / "schema.json").read_text(encoding="utf-8"))
    features = schema["features"]
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    x_train, y_train = train[features], train["label"]
    x_test, y_test = test[features], test["label"]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for depth in range(1, MAX_DEPTH + 1):
        scores = cross_val_score(new_tree(depth), x_train, y_train, cv=cv)
        model = new_tree(depth).fit(x_train, y_train)
        rows.append(
            {
                "depth": depth,
                "cv_accuracy": float(scores.mean()),
                "cv_std": float(scores.std()),
                "train_accuracy": float(model.score(x_train, y_train)),
                "test_accuracy": float(model.score(x_test, y_test)),
                "leaves": int(model.get_n_leaves()),
            }
        )
    best = max(rows, key=lambda r: r["cv_accuracy"])
    error = best["cv_std"] / np.sqrt(cv.get_n_splits())
    recommended = min((r for r in rows if r["cv_accuracy"] >= best["cv_accuracy"] - error), key=lambda r: r["depth"])

    final = new_tree(recommended["depth"]).fit(x_train, y_train)
    predictions = final.predict(x_test)
    classes = list(final.classes_)
    joblib.dump({"features": features, "classes": classes, "depth": recommended["depth"], "model": final}, OUTPUT / "irrigation.joblib")
    pd.DataFrame({"sample_id": test["sample_id"], "actual": y_test, "predicted": predictions}).to_csv(
        OUTPUT / "irrigation_test_predictions.csv", index=False
    )

    metrics = {
        "dataset": "irrigation",
        "features": features,
        "classes": classes,
        "recommended_depth": recommended["depth"],
        "test_accuracy": float(accuracy_score(y_test, predictions)),
        "test_macro_f1": float(f1_score(y_test, predictions, average="macro")),
        "majority_baseline": float(y_train.value_counts(normalize=True).max()),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=classes).tolist(),
        "feature_importance": {f: float(v) for f, v in zip(features, final.feature_importances_)},
        "accuracy_by_depth": rows,
    }
    (OUTPUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(OUTPUT / "accuracy_by_depth.csv", index=False)
    print(f"recommended depth {recommended['depth']}  test={metrics['test_accuracy']:.3f}  "
          f"f1={metrics['test_macro_f1']:.3f}  baseline={metrics['majority_baseline']:.3f}")
    for r in rows:
        print(f"  depth {r['depth']}: cv={r['cv_accuracy']:.3f} train={r['train_accuracy']:.3f} test={r['test_accuracy']:.3f} leaves={r['leaves']}")
    print({f: round(v, 3) for f, v in metrics["feature_importance"].items()})


if __name__ == "__main__":
    main()
