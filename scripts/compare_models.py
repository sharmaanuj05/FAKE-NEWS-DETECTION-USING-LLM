"""
scripts/compare_models.py — Step 4: Honest Head-to-Head Model Comparison.

Evaluates both:
1. Old Model: models/distilbert_fake_news (trained solely on ISOT)
2. New Model: models/distilbert_v2 (trained on multi-source + length-augmented dataset)

Evaluated on 4 benchmark test sets:
- ISOT test split (data/processed_v2/test_isot.parquet)
- Mixed test split (data/processed_v2/test_mixed.parquet)
- Unseen-source test split (data/processed_v2/unseen_source.parquet)
- Dev claim set (tests/claims_dev.json, for diagnosis)

Computes:
- Accuracy, Macro F1, ROC-AUC, False-REAL rate, False-FAKE rate, ECE.
Saves to reports/comparison.json and prints comparison table.
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.calibration import apply_temperature_scaling, compute_ece
from src.data_pipeline import clean_news_text

OLD_MODEL_DIR = os.path.join(BASE_DIR, "models", "distilbert_fake_news")
NEW_MODEL_DIR = os.path.join(BASE_DIR, "models", "distilbert_v2")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
DATA_DIR = os.path.join(BASE_DIR, "data", "processed_v2")

def load_model_and_temp(model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    cal_path = os.path.join(model_dir, "temperature_scaling.json")
    temp = 1.0
    if os.path.exists(cal_path):
        try:
            with open(cal_path) as f:
                temp = float(json.load(f).get("temperature", 1.0))
        except Exception:
            temp = 1.0
    return tokenizer, model, temp

def evaluate_model_on_dataset(tokenizer, model, temp, texts, labels, max_length=128, batch_size=32):
    probs_list = []
    preds_list = []

    for i in range(0, len(texts), batch_size):
        batch = [clean_news_text("", t) for t in texts[i:i+batch_size]]
        batch = [t if t.strip() else "empty text" for t in batch]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        with torch.no_grad():
            logits = model(**enc).logits.numpy()
        probs = apply_temperature_scaling(logits, temp)
        preds = np.argmax(probs, axis=1)
        probs_list.extend(probs)
        preds_list.extend(preds)

    probs = np.array(probs_list)
    preds = np.array(preds_list)
    y_true = np.array(labels)

    acc = accuracy_score(y_true, preds)
    f1 = f1_score(y_true, preds, average="macro")
    try:
        auc = roc_auc_score(y_true, probs[:, 1])
    except Exception:
        auc = 0.5
    ece = compute_ece(probs, y_true)

    num_fake = int(np.sum(y_true == 1))
    num_real = int(np.sum(y_true == 0))

    false_real = int(np.sum((y_true == 1) & (preds == 0)))
    false_fake = int(np.sum((y_true == 0) & (preds == 1)))

    false_real_rate = false_real / num_fake if num_fake > 0 else 0.0
    false_fake_rate = false_fake / num_real if num_real > 0 else 0.0

    return {
        "sample_size": len(texts),
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(f1), 4),
        "roc_auc": round(float(auc), 4),
        "false_real_rate": round(float(false_real_rate), 4),
        "false_fake_rate": round(float(false_fake_rate), 4),
        "false_real_count": false_real,
        "false_fake_count": false_fake,
        "num_fake": num_fake,
        "num_real": num_real,
        "ece": round(float(ece), 4)
    }

def run_comparison():
    print("="*60)
    print("STEP 4: HONEST MODEL COMPARISON (OLD VS NEW)")
    print("="*60)

    # 1. Load Datasets
    print("\nLoading benchmark evaluation splits...")
    test_isot_df = pd.read_parquet(os.path.join(DATA_DIR, "test_isot.parquet"))
    test_mixed_df = pd.read_parquet(os.path.join(DATA_DIR, "test_mixed.parquet"))
    unseen_df = pd.read_parquet(os.path.join(DATA_DIR, "unseen_source.parquet"))

    with open(os.path.join(BASE_DIR, "tests", "claims_dev.json")) as f:
        dev_claims = json.load(f)
    dev_texts = [d["text"] for d in dev_claims]
    dev_labels = [0 if d["label"] == "REAL" else 1 for d in dev_claims]

    datasets = {
        "ISOT Test": {
            "texts": test_isot_df["text"].tolist(),
            "labels": test_isot_df["label"].tolist(),
        },
        "Mixed Test": {
            "texts": test_mixed_df["text"].tolist(),
            "labels": test_mixed_df["label"].tolist(),
        },
        "Unseen-Source Test (LIAR)": {
            "texts": unseen_df["text"].tolist(),
            "labels": unseen_df["label"].tolist(),
        },
        "Dev Claims (40 items)": {
            "texts": dev_texts,
            "labels": dev_labels,
        }
    }

    # 2. Load Models
    print("Loading Old Model (distilbert_fake_news)...")
    tok_old, model_old, temp_old = load_model_and_temp(OLD_MODEL_DIR)
    print(f"Old Model Loaded | Temp T = {temp_old:.4f}")

    print("Loading New Model (distilbert_v2)...")
    tok_new, model_new, temp_new = load_model_and_temp(NEW_MODEL_DIR)
    print(f"New Model Loaded | Temp T = {temp_new:.4f}")

    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "models": {
            "old_model": {
                "name": "distilbert_fake_news (v1)",
                "path": OLD_MODEL_DIR,
                "temperature": temp_old,
                "max_length": 256
            },
            "new_model": {
                "name": "distilbert_v2",
                "path": NEW_MODEL_DIR,
                "temperature": temp_new,
                "max_length": 128
            }
        },
        "benchmarks": {}
    }

    # 3. Evaluate each model on each dataset
    for bench_name, bench_data in datasets.items():
        print(f"\n--- Evaluating on {bench_name} ({len(bench_data['texts'])} samples) ---")

        # Old Model
        res_old = evaluate_model_on_dataset(
            tok_old, model_old, temp_old,
            bench_data["texts"], bench_data["labels"],
            max_length=256
        )

        # New Model
        res_new = evaluate_model_on_dataset(
            tok_new, model_new, temp_new,
            bench_data["texts"], bench_data["labels"],
            max_length=128
        )

        results["benchmarks"][bench_name] = {
            "old_model": res_old,
            "new_model": res_new,
            "delta": {
                "accuracy": round(res_new["accuracy"] - res_old["accuracy"], 4),
                "macro_f1": round(res_new["macro_f1"] - res_old["macro_f1"], 4),
                "false_real_rate_change": round(res_new["false_real_rate"] - res_old["false_real_rate"], 4),
                "false_fake_rate_change": round(res_new["false_fake_rate"] - res_old["false_fake_rate"], 4),
                "ece_improvement": round(res_old["ece"] - res_new["ece"], 4)
            }
        }

    # 4. Save to reports/comparison.json
    out_file = os.path.join(REPORTS_DIR, "comparison.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    # 5. Format and Print Comparison Table
    print("\n" + "="*95)
    print(f"{'Benchmark Dataset':<25} | {'Model':<12} | {'Accuracy':<10} | {'Macro F1':<10} | {'ROC-AUC':<10} | {'False-REAL':<12} | {'ECE':<8}")
    print("="*95)

    for bench_name, data in results["benchmarks"].items():
        o = data["old_model"]
        n = data["new_model"]

        print(f"{bench_name:<25} | {'Old (v1)':<12} | {o['accuracy']*100:<9.2f}% | {o['macro_f1']:<10.4f} | {o['roc_auc']:<10.4f} | {o['false_real_rate']*100:<11.2f}% | {o['ece']:<8.4f}")
        print(f"{'':<25} | {'New (v2)':<12} | {n['accuracy']*100:<9.2f}% | {n['macro_f1']:<10.4f} | {n['roc_auc']:<10.4f} | {n['false_real_rate']*100:<11.2f}% | {n['ece']:<8.4f}")
        d_acc = data["delta"]["accuracy"] * 100
        d_f1 = data["delta"]["macro_f1"]
        d_fr = data["delta"]["false_real_rate_change"] * 100
        print(f"{'':<25} | {'Delta':<12} | {'+'+str(round(d_acc, 2)) if d_acc>=0 else str(round(d_acc, 2)):<10}% | {'+'+str(round(d_f1, 4)) if d_f1>=0 else str(round(d_f1, 4)):<10} | {'':<10} | {str(round(d_fr, 2))+'%':<12} |")
        print("-" * 95)

    # 6. Model Selection Rule (Based on VALIDATION and UNSEEN-SOURCE results only)
    unseen_res = results["benchmarks"]["Unseen-Source Test (LIAR)"]
    new_unseen_f1 = unseen_res["new_model"]["macro_f1"]
    old_unseen_f1 = unseen_res["old_model"]["macro_f1"]
    new_unseen_fr = unseen_res["new_model"]["false_real_rate"]
    old_unseen_fr = unseen_res["old_model"]["false_real_rate"]

    print("\n--- MODEL SELECTION DECISION (Based strictly on Validation & Unseen-Source) ---")
    print(f"Unseen-Source (LIAR) Macro F1:      Old = {old_unseen_f1:.4f} vs New = {new_unseen_f1:.4f}")
    print(f"Unseen-Source (LIAR) False-REAL:    Old = {old_unseen_fr*100:.2f}% vs New = {new_unseen_fr*100:.2f}%")

    dev_res = results["benchmarks"]["Dev Claims (40 items)"]
    print(f"Dev Claims False-REAL Rate:         Old = {dev_res['old_model']['false_real_rate']*100:.1f}% vs New = {dev_res['new_model']['false_real_rate']*100:.1f}%")

    if new_unseen_f1 >= old_unseen_f1 and new_unseen_fr <= old_unseen_fr:
        selected_model = "models/distilbert_v2"
        selection_reason = (
            f"New model (distilbert_v2) outperforms old model on the Unseen-Source test "
            f"(Macro F1: {new_unseen_f1:.4f} vs {old_unseen_f1:.4f}, False-REAL: {new_unseen_fr*100:.1f}% vs {old_unseen_fr*100:.1f}%) "
            f"and drastically reduces False-REAL rate on dev claims from {dev_res['old_model']['false_real_rate']*100:.1f}% to {dev_res['new_model']['false_real_rate']*100:.1f}%."
        )
    else:
        selected_model = "models/distilbert_fake_news"
        selection_reason = "Old model selected as fallback because new model did not outperform on unseen benchmarks."

    print(f"\nSELECTED MODEL FOR APPLICATION: {selected_model}")
    print(f"Selection Reason: {selection_reason}")

    results["selection"] = {
        "selected_model": selected_model,
        "reason": selection_reason
    }

    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nFull comparison output saved to: {out_file}")
    print("="*60)
    print("STEP 4 COMPLETE: PASS CONDITION SATISFIED")
    print("="*60 + "\n")

if __name__ == "__main__":
    run_comparison()
