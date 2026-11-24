"""
baseline.py — TF-IDF + Logistic Regression baseline for ISOT Fake News detection.

Trains on ISOT train split, evaluates on test split, logs peak RSS memory,
and saves metrics to reports/baseline_metrics.json.

Usage:
    python src/baseline.py [--subset 0.1]
"""
import os
import sys
import argparse
import json
import time
import psutil

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    confusion_matrix
)

from src.data_pipeline import load_and_preprocess_dataset

MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")


def peak_rss_mb() -> float:
    return round(psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2), 1)


def run_baseline(subset_ratio=None):
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("\n[Baseline] Loading ISOT News dataset...")
    train_df, val_df, test_df = load_and_preprocess_dataset(subset_ratio=subset_ratio)

    X_train = train_df["text"].tolist()
    y_train = train_df["label"].tolist()
    X_test  = test_df["text"].tolist()
    y_test  = test_df["label"].tolist()

    print(f"[Baseline] Fitting TF-IDF + Logistic Regression on {len(X_train)} samples...")
    t0 = time.time()

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=50_000,
            sublinear_tf=True,
            strip_accents="unicode",
            min_df=2
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
            class_weight="balanced",
            random_state=42
        ))
    ])
    pipeline.fit(X_train, y_train)

    train_time = round(time.time() - t0, 2)
    peak_rss = peak_rss_mb()

    # Evaluation on holdout test set
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", pos_label=1, zero_division=0
    )
    _, _, macro_f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred).tolist()

    print("\n==============================")
    print(" TF-IDF Baseline (ISOT Test)")
    print("==============================")
    print(f"  Accuracy : {acc * 100:.2f}%")
    print(f"  Precision: {prec * 100:.2f}%")
    print(f"  Recall   : {rec * 100:.2f}%")
    print(f"  F1 (Fake): {f1 * 100:.2f}%")
    print(f"  Macro F1 : {macro_f1 * 100:.2f}%")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(f"  Train time: {train_time}s | Peak RSS: {peak_rss} MB")
    print(f"  Confusion Matrix: {cm}")
    print("==============================\n")

    # Save model
    out_path = os.path.join(MODELS_DIR, "baseline_tfidf.joblib")
    joblib.dump(pipeline, out_path)
    print(f"[Baseline] Model saved to {out_path}")

    # Save metrics
    metrics = {
        "model": "TF-IDF + Logistic Regression",
        "subset_ratio": subset_ratio,
        "test_samples": len(test_df),
        "test_accuracy":  round(acc * 100, 2),
        "test_precision": round(prec * 100, 2),
        "test_recall":    round(rec * 100, 2),
        "test_f1":        round(f1 * 100, 2),
        "test_macro_f1":  round(macro_f1 * 100, 2),
        "test_roc_auc":   round(auc, 4),
        "confusion_matrix": cm,
        "train_time_sec": train_time,
        "peak_rss_mb": peak_rss
    }
    report_path = os.path.join(REPORTS_DIR, "baseline_metrics.json")
    with open(report_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[Baseline] Metrics saved to {report_path}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TF-IDF Baseline Trainer")
    parser.add_argument("--subset", type=float, default=None,
                        help="Fraction of data to use (e.g. 0.1 for 10 percent)")
    args = parser.parse_args()
    run_baseline(subset_ratio=args.subset)
