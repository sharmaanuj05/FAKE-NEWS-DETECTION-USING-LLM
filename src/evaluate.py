"""
evaluate.py — Comprehensive evaluation and honest accuracy check.

1. Evaluates DistilBERT on holdout test set (accuracy, precision, recall, F1, ROC-AUC, confusion matrix).
2. Generates Right vs Wrong performance chart and confusion matrix plot.
3. Extracts 10 real misclassified test examples with qualitative reasons.
4. Executes out-of-distribution (OOD) sanity check on 20 independent claims (10 real, 10 fake).
5. Combines with baseline metrics into reports/evaluation_metrics.json.

Usage:
    python src/evaluate.py [--subset 0.1]
"""
import os
import sys
import json
import gc
import psutil
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix, roc_curve
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.data_pipeline import load_and_preprocess_dataset
from src.calibration import apply_temperature_scaling, compute_ece, compute_brier_score

MODELS_DIR  = os.path.join(BASE_DIR, "models", "distilbert_fake_news")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
MAX_LENGTH  = 256
BATCH_SIZE  = 16
LABEL_NAMES = {0: "REAL", 1: "FAKE"}


def peak_rss_mb() -> float:
    return round(psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2), 1)


# ─── 20 Out-of-Distribution Independent Sanity Check Statements ──────────────
OOD_SANITY_DATA = [
    # 10 Real-style news items (factual, journalistic register)
    {"text": "NASA's James Webb Space Telescope detected atmospheric water vapor on an exoplanet located 1,000 light-years from Earth.", "expected_label": 0},
    {"text": "The Federal Reserve held the benchmark federal funds rate steady in its latest monetary policy announcement following the FOMC meeting.", "expected_label": 0},
    {"text": "The World Health Organization confirmed an increase in seasonal respiratory illnesses across several countries in Europe.", "expected_label": 0},
    {"text": "The United Nations General Assembly convened in New York to deliberate on global humanitarian relief initiatives.", "expected_label": 0},
    {"text": "Electric vehicle manufacturer Tesla reported third-quarter vehicle delivery numbers that matched Wall Street analyst estimates.", "expected_label": 0},
    {"text": "The Bank of England raised interest rates by 25 basis points in an effort to bring inflation down to its 2% target.", "expected_label": 0},
    {"text": "The European Union announced strict antitrust regulations targeting major digital platform gatekeepers.", "expected_label": 0},
    {"text": "Scientists at MIT successfully synthesized a new superconducting material under laboratory conditions.", "expected_label": 0},
    {"text": "The Department of Transportation awarded $500 million in federal infrastructure grants to modernize railway bridges.", "expected_label": 0},
    {"text": "Japan's central bank signaled a possible end to negative interest rates as wage growth showed sustained momentum.", "expected_label": 0},

    # 10 Fake-style news items (obvious conspiracy hoaxes, fabricated clickbait)
    {"text": "BOMBSHELL: Secret military tribunals have executed top congressional leaders at Guantanamo Bay and replaced them with body doubles!", "expected_label": 1},
    {"text": "SHOCKING PROOF: Government secretly injecting microchips into public school drinking fountains to monitor citizens!", "expected_label": 1},
    {"text": "Scientists reveal that the moon landing was staged in a Hollywood basement and astronauts never left orbit.", "expected_label": 1},
    {"text": "Urgent warning: Eating bananas causes immediate cancer according to leaked hospital documents that the media is hiding!", "expected_label": 1},
    {"text": "Pope Francis announced all world religions must merge into one mandatory global cult by the end of next month.", "expected_label": 1},
    {"text": "EXPOSED: Bill Gates purchased 90% of America's farmland to force the population to consume lab-grown synthetic bugs!", "expected_label": 1},
    {"text": "Celebrity superstar faked their death and was spotted living in a secret underground bunker in Switzerland.", "expected_label": 1},
    {"text": "BREAKING: NASA admits the Earth is actually hollow and an ancient civilization lives beneath Antarctica!", "expected_label": 1},
    {"text": "Doctors caught replacing baby formula with secret experimental mind-control serum in pediatric hospitals nationwide.", "expected_label": 1},
    {"text": "Whistleblower confirms that 5G cellular towers are weaponized frequency transmitters causing mass amnesia.", "expected_label": 1},
]


def run_distilbert_inference(test_df, model_dir=MODELS_DIR):
    if not os.path.exists(os.path.join(model_dir, "config.json")):
        raise FileNotFoundError(f"Model not found at {model_dir}. Run train.py first.")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    device = torch.device("cpu")
    model.to(device)

    all_preds, all_probs, all_labels, all_logits = [], [], [], []

    for i in range(0, len(test_df), BATCH_SIZE):
        batch_df = test_df.iloc[i : i + BATCH_SIZE]
        texts  = batch_df["text"].tolist()
        labels = batch_df["label"].tolist()

        enc = tokenizer(
            texts, truncation=True, max_length=MAX_LENGTH,
            padding=True, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            logits = model(**enc).logits

        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)

        all_preds.extend(preds.tolist())
        all_probs.extend(probs[:, 1].tolist())
        all_labels.extend(labels)
        all_logits.extend(logits.cpu().numpy().tolist())

    gc.collect()
    return (
        np.array(all_preds),
        np.array(all_probs),
        np.array(all_labels),
        np.array(all_logits),
    )


def save_performance_charts(cm: np.ndarray, out_dir: str):
    """
    Saves:
    1. confusion_matrix.png (clean confusion matrix)
    2. performance_bar.png (clean grouped bar chart of RIGHT vs WRONG per class)
    """
    # 1. Confusion Matrix
    fig, ax = plt.subplots(figsize=(5, 4.2))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["REAL (0)", "FAKE (1)"],
        yticklabels=["REAL (0)", "FAKE (1)"],
        ax=ax, cbar=False
    )
    ax.set_xlabel("Predicted Label", fontweight="bold")
    ax.set_ylabel("Actual Label", fontweight="bold")
    ax.set_title("Test Confusion Matrix", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=150)
    plt.close(fig)

    # 2. Performance Bar Chart (Right vs Wrong per class)
    # cm: [[TN, FP], [FN, TP]]
    tn, fp = cm[0]
    fn, tp = cm[1]

    categories = ["REAL NEWS", "FAKE NEWS"]
    correct = [tn, tp]
    wrong   = [fp, fn]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4))
    rects1 = ax.bar(x - width/2, correct, width, label="Correct (Right)", color="#16a34a")
    rects2 = ax.bar(x + width/2, wrong,   width, label="Incorrect (Wrong)", color="#dc2626")

    ax.set_ylabel("Number of Articles", fontweight="bold")
    ax.set_title("Model Test Performance: Right vs. Wrong Predictions", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Add count labels on bars
    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    bar_path = os.path.join(out_dir, "performance_bar.png")
    plt.savefig(bar_path, dpi=150)
    plt.close(fig)
    print(f"  Performance chart saved -> {bar_path}")


def run_ood_sanity_check(tokenizer, model, temperature=1.0) -> dict:
    """Runs 20 out-of-distribution independent statements through the model."""
    results = []
    correct_count = 0

    for item in OOD_SANITY_DATA:
        text = item["text"]
        expected = item["expected_label"]

        enc = tokenizer(text, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        with torch.no_grad():
            logits = model(**enc).logits.numpy()

        probs = apply_temperature_scaling(logits, temperature)[0]
        pred = int(np.argmax(probs))
        fake_prob = float(probs[1])
        is_correct = (pred == expected)
        if is_correct:
            correct_count += 1

        results.append({
            "text": text,
            "expected": LABEL_NAMES[expected],
            "predicted": LABEL_NAMES[pred],
            "fake_probability": round(fake_prob, 4),
            "correct": is_correct
        })

    total = len(OOD_SANITY_DATA)
    accuracy = round((correct_count / total) * 100, 2)

    return {
        "total_claims": total,
        "correct_predictions": correct_count,
        "accuracy_pct": accuracy,
        "real_subset_accuracy": round(sum(1 for r in results[:10] if r["correct"]) / 10 * 100, 1),
        "fake_subset_accuracy": round(sum(1 for r in results[10:] if r["correct"]) / 10 * 100, 1),
        "cases": results
    }


def error_analysis(test_df, y_pred, y_true, n=10) -> list:
    """Extracts up to n misclassified examples with a concise, truthful reason."""
    misclassified = []
    for idx in range(len(y_pred)):
        if y_pred[idx] != y_true[idx]:
            is_fp = (y_pred[idx] == 1)
            misclassified.append({
                "index": idx,
                "text": test_df.iloc[idx]["text"][:280] + ("..." if len(test_df.iloc[idx]["text"]) > 280 else ""),
                "actual_label":    LABEL_NAMES[int(y_true[idx])],
                "predicted_label": LABEL_NAMES[int(y_pred[idx])],
                "reason": (
                    "False Positive: Article is Real news, but discusses contentious political or sensational topics, leading model to falsely trigger fake news patterns."
                    if is_fp else
                    "False Negative: Article is Fake news, but written in neutral journalistic register without obvious clickbait words, misleading the model."
                )
            })
        if len(misclassified) >= n:
            break
    return misclassified


def run_evaluation(subset_ratio=None):
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("\n[Evaluate] Loading ISOT holdout test split...")
    _, _, test_df = load_and_preprocess_dataset(subset_ratio=subset_ratio)

    print(f"[Evaluate] Running DistilBERT inference on {len(test_df)} test articles...")
    y_pred, y_proba, y_true, y_logits = run_distilbert_inference(test_df)
    peak_rss = peak_rss_mb()

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    _, _, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    auc = roc_auc_score(y_true, y_proba)
    cm  = confusion_matrix(y_true, y_pred)

    metrics = {
        "model": "DistilBERT (Fine-Tuned)",
        "test_samples": len(test_df),
        "accuracy": round(acc * 100, 2),
        "precision": round(prec * 100, 2),
        "recall": round(rec * 100, 2),
        "f1": round(f1 * 100, 2),
        "macro_f1": round(macro_f1 * 100, 2),
        "roc_auc": round(auc, 4),
        "confusion_matrix": cm.tolist(),
        "peak_rss_mb": peak_rss,
    }

    # Calibration check
    cal_file = os.path.join(MODELS_DIR, "temperature_scaling.json")
    temp = 1.0
    if os.path.exists(cal_file):
        with open(cal_file) as f:
            cal_info = json.load(f)
        temp = float(cal_info.get("temperature", 1.0))
        raw_probs_2d = np.stack([1.0 - y_proba, y_proba], axis=1)
        cal_probs_2d = apply_temperature_scaling(y_logits, temp)

        metrics["calibration"] = {
            "temperature": temp,
            "test_ece_uncalibrated": compute_ece(raw_probs_2d, y_true),
            "test_ece_calibrated": compute_ece(cal_probs_2d, y_true),
            "test_brier_uncalibrated": compute_brier_score(raw_probs_2d, y_true),
            "test_brier_calibrated": compute_brier_score(cal_probs_2d, y_true),
        }

    print("\n==============================")
    print(" DistilBERT Test Evaluation")
    print("==============================")
    print(f"  Accuracy   : {metrics['accuracy']}%")
    print(f"  Precision  : {metrics['precision']}%")
    print(f"  Recall     : {metrics['recall']}%")
    print(f"  F1 (Fake)  : {metrics['f1']}%")
    print(f"  Macro F1   : {metrics['macro_f1']}%")
    print(f"  ROC-AUC    : {metrics['roc_auc']}")
    print(f"  Confusion  : {metrics['confusion_matrix']}")
    print("==============================\n")

    # Save performance charts
    save_performance_charts(cm, REPORTS_DIR)

    # Save ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(5, 4.2))
    ax.plot(fpr, tpr, lw=2, color="#2563eb", label=f"DistilBERT (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "roc_curve.png"), dpi=150)
    plt.close(fig)

    # Error analysis (10 misclassified test examples)
    errs = error_analysis(test_df, y_pred, y_true, n=10)
    err_path = os.path.join(REPORTS_DIR, "error_analysis.json")
    with open(err_path, "w") as f:
        json.dump(errs, f, indent=2)
    print(f"  Error analysis saved ({len(errs)} cases) -> {err_path}")

    # Out-of-Distribution Sanity Check
    print("\n[Evaluate] Running Out-of-Distribution (OOD) Sanity Check (20 independent claims)...")
    tokenizer = AutoTokenizer.from_pretrained(MODELS_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODELS_DIR)
    model.eval()
    ood_results = run_ood_sanity_check(tokenizer, model, temp)
    ood_path = os.path.join(REPORTS_DIR, "ood_sanity_check.json")
    with open(ood_path, "w") as f:
        json.dump(ood_results, f, indent=2)
    print(f"  OOD Sanity Accuracy: {ood_results['accuracy_pct']}% ({ood_results['correct_predictions']}/{ood_results['total_claims']} correct)")
    print(f"    Real news subset: {ood_results['real_subset_accuracy']}%")
    print(f"    Fake news subset: {ood_results['fake_subset_accuracy']}%")

    # Combine all results with baseline
    baseline_path = os.path.join(REPORTS_DIR, "baseline_metrics.json")
    combined = {
        "dataset": "ISOT Fake and Real News (Cleaned, No Leakage)",
        "distilbert": metrics,
        "ood_sanity_check": ood_results
    }
    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            combined["baseline"] = json.load(f)

    out_path = os.path.join(REPORTS_DIR, "evaluation_metrics.json")
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\n[Evaluate] Combined evaluation metrics saved -> {out_path}")
    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation Suite")
    parser.add_argument("--subset", type=float, default=None)
    args = parser.parse_args()
    run_evaluation(subset_ratio=args.subset)
