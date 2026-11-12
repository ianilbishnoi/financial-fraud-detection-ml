"""Credit card fraud detection on a highly imbalanced dataset.

Pipeline
  1. Load the Kaggle/ULB credit card dataset (284,807 transactions, 0.17% fraud).
  2. Stratified 60/20/20 train / validation / test split, so every split keeps the fraud rate.
  3. Train two models that account for the imbalance with class weights:
       - Logistic Regression (features scaled; the scaler is fit on training data only)
       - Random Forest
  4. Pick each model's decision threshold on the VALIDATION set (best F1), instead of using 0.5.
  5. Report precision, recall, F1, PR-AUC and ROC-AUC on the untouched TEST set,
     next to a "always predict legitimate" baseline that shows why accuracy is misleading.

Usage
  python fraud_detection.py --data creditcard.csv --out reports
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save plots to files; no window needed (works in CI)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    PrecisionRecallDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

TARGET = "Class"
SEED = 42


def load_data(path: str | Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    if TARGET not in data.columns:
        raise ValueError(f"expected a '{TARGET}' column (1 = fraud, 0 = legitimate)")
    return data


def split(data: pd.DataFrame, seed: int = SEED):
    """Stratified 60/20/20 split into train, validation and test sets."""
    X, y = data.drop(columns=TARGET), data[TARGET]
    X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size=0.4, stratify=y, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=seed)
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_models(seed: int = SEED) -> dict:
    """Both models weight the rare fraud class more heavily instead of resampling the data."""
    return {
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000)
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced_subsample", min_samples_leaf=2, n_jobs=-1, random_state=seed
        ),
    }


def best_threshold(y_true, scores) -> float:
    """Threshold that maximises F1 on the given (validation) data."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(thresholds[np.argmax(f1[:-1])])  # the last P/R point has no threshold


def evaluate(y_true, scores, threshold: float) -> dict:
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": round(threshold, 4),
        "precision": round(precision_score(y_true, pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, pred, zero_division=0), 4),
        "pr_auc": round(average_precision_score(y_true, scores), 4),
        "roc_auc": round(roc_auc_score(y_true, scores), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def run(data: pd.DataFrame, out_dir: str | Path, seed: int = SEED) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    X_train, X_val, X_test, y_train, y_val, y_test = split(data, seed)

    results = {
        "dataset": {
            "rows": int(len(data)),
            "fraud_rate_percent": round(100 * data[TARGET].mean(), 3),
            "train_rows": int(len(X_train)),
            "val_rows": int(len(X_val)),
            "test_rows": int(len(X_test)),
        },
        # Predicting "legitimate" for everything already gets ~99.8% accuracy and catches zero fraud.
        "baseline_always_legit": {
            "accuracy": round(float((y_test == 0).mean()), 4),
            "recall": 0.0,
        },
        "models": {},
    }

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, model in build_models(seed).items():
        model.fit(X_train, y_train)
        threshold = best_threshold(y_val, model.predict_proba(X_val)[:, 1])
        test_scores = model.predict_proba(X_test)[:, 1]
        results["models"][name] = evaluate(y_test, test_scores, threshold)
        PrecisionRecallDisplay.from_predictions(y_test, test_scores, name=name, ax=ax)

    ax.set_title("Precision-recall on the test set")
    fig.tight_layout()
    fig.savefig(out / "pr_curve.png", dpi=120)
    plt.close(fig)

    best = max(results["models"], key=lambda m: results["models"][m]["pr_auc"])
    results["best_model"] = best
    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="creditcard.csv", help="path to creditcard.csv")
    parser.add_argument("--out", default="reports", help="folder for metrics.json and pr_curve.png")
    args = parser.parse_args()

    results = run(load_data(args.data), args.out)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
