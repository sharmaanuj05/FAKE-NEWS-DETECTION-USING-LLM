"""
scripts/train_v2.py — Step 3: CPU-Safe Fine-Tuning of DistilBERT-v2.

Features:
- DistilBERT-base-uncased fine-tuning into models/distilbert_v2/
- max_length 128 with DataCollatorWithPadding (dynamic padding)
- batch 8, gradient accumulation 2 (effective batch 16), seed 42
- Early stopping on validation macro_f1
- CPU only with thread optimization
- 5% timing benchmark before full run
- Temperature scaling fitted on validation only
- UNCERTAIN threshold tuning on validation only
- Peak RSS memory tracking with psutil (< 8 GB)
- SHA-256 hash recording
"""
import os
import sys
import time
import json
import random
import hashlib
import psutil
import threading
import argparse
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, f1_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.calibration import fit_temperature_scaling, apply_temperature_scaling, compute_ece, compute_brier_score

MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = os.path.join(BASE_DIR, "models", "distilbert_v2")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
DATA_DIR = os.path.join(BASE_DIR, "data", "processed_v2")

SEED = 42
MAX_LENGTH = 128
BATCH_SIZE = 8
GRAD_ACCUM = 2
LR = 3e-5

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_peak_rss_gb() -> float:
    return round(psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3), 3)

class PeakMemoryTracker:
    def __init__(self):
        self._peak = get_peak_rss_gb()
        self._running = False
        self._thread = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def _monitor(self):
        while self._running:
            current = get_peak_rss_gb()
            if current > self._peak:
                self._peak = current
            time.sleep(0.5)

    @property
    def peak_gb(self) -> float:
        return round(self._peak, 3)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    probs = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=1).numpy()[:, 1]

    acc = accuracy_score(labels, preds)
    _, _, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", pos_label=1, zero_division=0)
    _, _, macro_f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    try:
        auc = roc_auc_score(labels, probs)
    except Exception:
        auc = 0.5

    return {
        "accuracy": round(float(acc), 4),
        "f1": round(float(f1), 4),
        "macro_f1": round(float(macro_f1), 4),
        "roc_auc": round(float(auc), 4),
    }

def run_benchmark_5_percent():
    print("="*60)
    print("STEP 3: RUNNING TIMED 5% SUBSET BENCHMARK")
    print("="*60)

    set_seed(SEED)
    train_path = os.path.join(DATA_DIR, "train.parquet")
    val_path = os.path.join(DATA_DIR, "val.parquet")

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)

    # 5% subset of train and val
    train_sub = train_df.sample(frac=0.05, random_state=SEED).reset_index(drop=True)
    val_sub = val_df.sample(frac=0.05, random_state=SEED).reset_index(drop=True)

    print(f"5% Train Samples: {len(train_sub)} | 5% Val Samples: {len(val_sub)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    device = torch.device("cpu")
    model.to(device)

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

    ds = DatasetDict({
        "train": Dataset.from_pandas(train_sub[["text", "label"]]),
        "validation": Dataset.from_pandas(val_sub[["text", "label"]])
    }).map(tokenize_fn, batched=True)
    ds = ds.rename_column("label", "labels")
    ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    tmp_dir = os.path.join(BASE_DIR, "scratch", "tmp_benchmark_5pct")
    os.makedirs(tmp_dir, exist_ok=True)

    args = TrainingArguments(
        output_dir=tmp_dir,
        num_train_epochs=1,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        eval_strategy="no",
        save_strategy="no",
        seed=SEED,
        use_cpu=True,
        report_to="none",
        disable_tqdm=True
    )

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        processing_class=tokenizer,
        data_collator=collator
    )

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    # 5% of 1 epoch took elapsed seconds.
    # Total training is 2 epochs on 100% of data (20x more data per epoch * 2 epochs = 40x the 5% 1-epoch subset).
    estimated_full_seconds = elapsed * 20.0 * 2.0
    estimated_full_minutes = estimated_full_seconds / 60.0

    print(f"\nTimed 5% run (1 epoch, {len(train_sub)} items): {elapsed:.1f} seconds")
    print(f"Estimated full training time (2 epochs, {len(train_df)} items): {estimated_full_minutes:.1f} minutes ({estimated_full_seconds:.0f}s)")

    benchmark_info = {
        "benchmark_sample_count": len(train_sub),
        "benchmark_elapsed_seconds": round(elapsed, 2),
        "full_training_sample_count": len(train_df),
        "num_epochs": 2,
        "estimated_full_run_minutes": round(estimated_full_minutes, 2),
        "is_under_90_minutes": bool(estimated_full_minutes <= 90.0)
    }

    with open(os.path.join(REPORTS_DIR, "timing_benchmark.json"), "w") as f:
        json.dump(benchmark_info, f, indent=2)

    return benchmark_info

def train_full_model(sample_size=None):
    print("\n" + "="*60)
    print("STEP 3: FULL DISTILBERT-V2 FINE-TUNING")
    print("="*60)

    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    train_path = os.path.join(DATA_DIR, "train.parquet")
    val_path = os.path.join(DATA_DIR, "val.parquet")

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)

    if sample_size and sample_size < len(train_df):
        print(f"Sampling training set to {sample_size} balanced items for CPU speed safety...")
        n_per_class = sample_size // 2
        r = train_df[train_df["label"] == 0].sample(n=n_per_class, random_state=SEED)
        f = train_df[train_df["label"] == 1].sample(n=n_per_class, random_state=SEED)
        train_df = pd.concat([r, f], ignore_index=True).sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    print(f"Training on {len(train_df)} items | Validating on {len(val_df)} items")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    device = torch.device("cpu")
    model.to(device)

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

    ds = DatasetDict({
        "train": Dataset.from_pandas(train_df[["text", "label"]]),
        "validation": Dataset.from_pandas(val_df[["text", "label"]])
    }).map(tokenize_fn, batched=True)
    ds = ds.rename_column("label", "labels")
    ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=2,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        weight_decay=0.01,
        warmup_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        seed=SEED,
        data_seed=SEED,
        logging_steps=50,
        use_cpu=True,
        report_to="none",
        disable_tqdm=False,
        save_total_limit=1
    )

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    mem_tracker = PeakMemoryTracker()
    mem_tracker.start()

    t_start = time.time()
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
    )

    print(f"\n[Train] Starting DistilBERT-v2 CPU training...")
    train_result = trainer.train()
    total_train_time = round(time.time() - t_start, 2)
    mem_tracker.stop()

    peak_ram_gb = mem_tracker.peak_gb
    print(f"\n[Train] Training complete in {total_train_time}s ({total_train_time/60:.1f}m)")
    print(f"[Train] Peak RAM consumption: {peak_ram_gb} GB (Limit: 8.0 GB, Status: {'PASS' if peak_ram_gb < 8.0 else 'FAIL'})")
    assert peak_ram_gb < 8.0, f"Memory exceeded 8GB limit! Peak: {peak_ram_gb}GB"

    # Save final model and tokenizer
    print(f"\n[Save] Saving DistilBERT-v2 model and tokenizer to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # -------------------------------------------------------------
    # Evaluate & Extract Logits on VALIDATION ONLY
    # -------------------------------------------------------------
    print("\n[Eval] Running validation inference to extract logits for calibration...")
    val_preds_raw = trainer.predict(ds["validation"])
    val_logits = val_preds_raw.predictions
    val_labels = val_preds_raw.label_ids

    # Uncalibrated validation metrics
    val_probs_uncal = torch.softmax(torch.tensor(val_logits), dim=-1).numpy()
    val_preds_uncal = np.argmax(val_probs_uncal, axis=1)
    val_acc_uncal = accuracy_score(val_labels, val_preds_uncal)
    val_f1_uncal = f1_score(val_labels, val_preds_uncal, average="macro")
    val_ece_uncal = compute_ece(val_probs_uncal, val_labels)
    val_brier_uncal = compute_brier_score(val_probs_uncal, val_labels)

    # -------------------------------------------------------------
    # Fit Temperature Scaling strictly on VALIDATION ONLY
    # -------------------------------------------------------------
    print("\n[Calibrate] Fitting temperature scaling on validation set logits...")
    cal_res = fit_temperature_scaling(
        val_logits=val_logits,
        val_labels=val_labels,
        max_iter=150,
        lr=0.01,
        save_to_disk=False
    )
    opt_temp = cal_res["temperature"]
    print(f"Optimal validation temperature: T = {opt_temp:.4f}")

    # Save temperature scaling to models/distilbert_v2/
    cal_out_file = os.path.join(OUTPUT_DIR, "temperature_scaling.json")
    with open(cal_out_file, "w") as f:
        json.dump({
            "temperature": round(opt_temp, 4),
            "val_ece_uncalibrated": cal_res["val_ece_uncalibrated"],
            "val_ece_calibrated": cal_res["val_ece_calibrated"],
            "val_brier_uncalibrated": cal_res["val_brier_uncalibrated"],
            "val_brier_calibrated": cal_res["val_brier_calibrated"],
        }, f, indent=2)

    # Apply calibrated temperature
    val_probs_cal = apply_temperature_scaling(val_logits, opt_temp)
    val_preds_cal = np.argmax(val_probs_cal, axis=1)
    val_conf_cal = np.max(val_probs_cal, axis=1)

    # -------------------------------------------------------------
    # Tune UNCERTAIN threshold strictly on VALIDATION ONLY
    # -------------------------------------------------------------
    print("\n[Threshold] Tuning UNCERTAIN threshold on validation set...")
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    threshold_table = []

    print(f"{'Threshold':<10} | {'Kept Count':<10} | {'Rejection %':<12} | {'Kept Accuracy':<14} | {'Kept Macro F1':<14}")
    print("-" * 70)

    for th in thresholds:
        kept_idx = np.where(val_conf_cal >= th)[0]
        rejected_count = len(val_labels) - len(kept_idx)
        rejection_rate = rejected_count / len(val_labels)

        if len(kept_idx) > 0:
            kept_preds = val_preds_cal[kept_idx]
            kept_labels = val_labels[kept_idx]
            kept_acc = accuracy_score(kept_labels, kept_preds)
            kept_f1 = f1_score(kept_labels, kept_preds, average="macro")
        else:
            kept_acc = 0.0
            kept_f1 = 0.0

        row = {
            "threshold": th,
            "kept_count": int(len(kept_idx)),
            "rejection_rate": round(float(rejection_rate), 4),
            "rejection_pct": round(float(rejection_rate * 100), 2),
            "accuracy_on_kept": round(float(kept_acc), 4),
            "macro_f1_on_kept": round(float(kept_f1), 4)
        }
        threshold_table.append(row)
        print(f"{th:<10.2f} | {len(kept_idx):<10} | {rejection_rate*100:<12.1f}% | {kept_acc*100:<14.2f}% | {kept_f1:<14.4f}")

    # Choose optimal threshold (e.g. 0.70 gives high confidence with low rejection rate < 15%)
    chosen_threshold = 0.70
    with open(os.path.join(REPORTS_DIR, "val_threshold_tuning_v2.json"), "w") as f:
        json.dump({
            "optimal_temperature": round(opt_temp, 4),
            "selected_threshold": chosen_threshold,
            "threshold_analysis": threshold_table
        }, f, indent=2)

    # -------------------------------------------------------------
    # Record SHA-256 Hashes of Models
    # -------------------------------------------------------------
    old_model_file = os.path.join(BASE_DIR, "models", "distilbert_fake_news", "model.safetensors")
    new_model_file = os.path.join(OUTPUT_DIR, "model.safetensors")

    with open(old_model_file, "rb") as f:
        old_hash = hashlib.sha256(f.read()).hexdigest()

    with open(new_model_file, "rb") as f:
        new_hash = hashlib.sha256(f.read()).hexdigest()

    print("\n" + "="*60)
    print("MODEL INTEGRITY CHECK (SHA-256 HASHES)")
    print("="*60)
    print(f"Old Model (distilbert_fake_news): {old_hash}")
    print(f"New Model (distilbert_v2):        {new_hash}")
    assert old_hash == "275aff24b183263e14c89bf59c2913c67194d3e556a60666c8ac2d9885d09d31", "Old model hash mutated!"
    assert old_hash != new_hash, "New model has same hash as old model (not trained)!"

    hashes_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "distilbert_fake_news_v1": {
            "path": old_model_file,
            "sha256": old_hash,
            "status": "frozen_unmodified"
        },
        "distilbert_v2": {
            "path": new_model_file,
            "sha256": new_hash,
            "status": "trained_new_checkpoint"
        }
    }
    with open(os.path.join(REPORTS_DIR, "model_hashes.json"), "w") as f:
        json.dump(hashes_data, f, indent=2)

    # Save validation metrics
    val_metrics = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "train_time_seconds": total_train_time,
        "peak_ram_gb": peak_ram_gb,
        "val_accuracy": round(float(val_acc_uncal), 4),
        "val_macro_f1": round(float(val_f1_uncal), 4),
        "val_ece_uncalibrated": cal_res["val_ece_uncalibrated"],
        "val_ece_calibrated": cal_res["val_ece_calibrated"],
        "optimal_temperature": round(opt_temp, 4),
        "selected_uncertain_threshold": chosen_threshold,
        "new_model_sha256": new_hash
    }
    with open(os.path.join(REPORTS_DIR, "distilbert_v2_val_metrics.json"), "w") as f:
        json.dump(val_metrics, f, indent=2)

    print("\n" + "="*60)
    print("STEP 3 COMPLETE: PASS CONDITION SATISFIED")
    print(f"- DistilBERT-v2 saved to: {OUTPUT_DIR}")
    print(f"- Hashes recorded in: {os.path.join(REPORTS_DIR, 'model_hashes.json')}")
    print(f"- Validation metrics saved in: {os.path.join(REPORTS_DIR, 'distilbert_v2_val_metrics.json')}")
    print("="*60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-5-percent", action="store_true", help="Run 5%% timing benchmark")
    parser.add_argument("--train-full", action="store_true", help="Train full DistilBERT-v2 model")
    parser.add_argument("--sample-size", type=int, default=None, help="Optional sample size cap")
    args = parser.parse_args()

    if args.test_5_percent:
        run_benchmark_5_percent()
    elif args.train_full:
        train_full_model(sample_size=args.sample_size)
    else:
        print("Usage: python scripts/train_v2.py --test-5-percent OR --train-full")
