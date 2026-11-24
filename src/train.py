"""
train.py — DistilBERT fine-tuning on ISOT Fake and Real News dataset.

Trains distilbert-base-uncased with HuggingFace Trainer (CPU-optimized, 8GB safe).
Uses max_length=256, batch_size=8, grad_accum=4 (effective batch 32), lr=2e-5.
Tracks peak RSS memory, fits temperature scaling on the validation set only,
and saves the best model checkpoint and metrics.

Usage:
    python src/train.py [--subset 0.1] [--epochs 2]
"""
import os
import sys
import gc
import json
import time
import random
import argparse
import threading
import psutil
import numpy as np
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.data_pipeline import load_and_preprocess_dataset
from src.calibration import fit_temperature_scaling

MODELS_DIR  = os.path.join(BASE_DIR, "models", "distilbert_fake_news")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
MODEL_NAME  = "distilbert-base-uncased"
MAX_LENGTH  = 256
SEED        = 42
BATCH_SIZE  = 8
GRAD_ACCUM  = 4


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def peak_rss_mb() -> float:
    return round(psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2), 1)


class PeakRSSTracker:
    """Samples RSS memory every 0.5s to accurately record peak usage."""
    def __init__(self):
        self._peak = peak_rss_mb()
        self._running = False
        self._thread = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def _run(self):
        while self._running:
            current = peak_rss_mb()
            if current > self._peak:
                self._peak = current
            time.sleep(0.5)

    @property
    def peak(self) -> float:
        return round(self._peak, 1)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    probs = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=1).numpy()[:, 1]

    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", pos_label=1, zero_division=0
    )
    _, _, macro_f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    try:
        auc = roc_auc_score(labels, probs)
    except Exception:
        auc = 0.5

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall":    round(rec, 4),
        "f1":        round(f1, 4),
        "macro_f1":  round(macro_f1, 4),
        "roc_auc":   round(auc, 4),
    }


def build_hf_datasets(train_df, val_df, test_df, tokenizer) -> DatasetDict:
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
        )

    ds = DatasetDict({
        "train":      Dataset.from_pandas(train_df[["text", "label"]]),
        "validation": Dataset.from_pandas(val_df[["text", "label"]]),
        "test":       Dataset.from_pandas(test_df[["text", "label"]]),
    })
    ds = ds.map(tokenize, batched=True, desc="Tokenizing")
    ds = ds.rename_column("label", "labels")
    ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return ds


def train(subset_ratio=None, num_epochs=2):
    set_seed(SEED)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    device = torch.device("cpu")
    torch.set_num_threads(8)
    print(f"\n[Train] Device: {device} | Worker Threads: {torch.get_num_threads()}")

    print("[Train] Loading preprocessed ISOT dataset...")
    train_df, val_df, test_df = load_and_preprocess_dataset(subset_ratio=subset_ratio)

    print("[Train] Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )
    model.to(device)

    print(f"[Train] Model: {MODEL_NAME} ({sum(p.numel() for p in model.parameters()):,} parameters)")
    print(f"[Train] Sequence length: {MAX_LENGTH} | Batch size: {BATCH_SIZE} | Grad Accum: {GRAD_ACCUM} (Effective batch: {BATCH_SIZE * GRAD_ACCUM})")

    ds = build_hf_datasets(train_df, val_df, test_df, tokenizer)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    args = TrainingArguments(
        output_dir=MODELS_DIR,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        seed=SEED,
        data_seed=SEED,
        logging_steps=25,
        use_cpu=True,
        report_to="none",
        disable_tqdm=False,
    )

    tracker = PeakRSSTracker()
    tracker.start()
    t0 = time.time()

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print(f"\n[Train] Starting training ({len(ds['train'])} train, {len(ds['validation'])} val)...")
    trainer.train()

    train_time = round(time.time() - t0, 1)
    tracker.stop()
    peak_rss = tracker.peak

    # Calibration: Fit Temperature Scaling strictly on validation split
    print("\n[Train] Fitting Temperature Scaling strictly on validation split (never test)...")
    val_out = trainer.predict(ds["validation"])
    cal_res = fit_temperature_scaling(val_out.predictions, val_out.label_ids)
    print(f"  Fitted Validation Temperature T = {cal_res['temperature']:.4f}")
    print(f"  Validation ECE: {cal_res['val_ece_uncalibrated']:.4f} -> {cal_res['val_ece_calibrated']:.4f}")
    print(f"  Validation Brier: {cal_res['val_brier_uncalibrated']:.4f} -> {cal_res['val_brier_calibrated']:.4f}")

    # Final evaluation on holdout test set
    print("\n[Train] Evaluating on holdout test split...")
    test_out = trainer.predict(ds["test"])
    test_metrics = compute_metrics((test_out.predictions, test_out.label_ids))

    print("\n==============================")
    print(" DistilBERT Test Metrics (Holdout)")
    print("==============================")
    for k, v in test_metrics.items():
        print(f"  {k:<12}: {v}")
    print(f"  Train time  : {train_time}s")
    print(f"  Peak RSS    : {peak_rss} MB")
    print("==============================\n")

    # Save best model + tokenizer
    trainer.save_model(MODELS_DIR)
    tokenizer.save_pretrained(MODELS_DIR)
    print(f"[Train] Best model saved to {MODELS_DIR}")

    # Save report
    report = {
        "model": MODEL_NAME,
        "dataset": "ISOT Fake & Real News (GonzaloA/fake_news)",
        "subset_ratio": subset_ratio,
        "epochs_run": float(trainer.state.epoch),
        "max_length": MAX_LENGTH,
        **{f"test_{k}": round(v * 100, 2) if k != "roc_auc" else round(v, 4)
           for k, v in test_metrics.items()},
        "train_time_sec": train_time,
        "peak_rss_mb": peak_rss,
        "train_size": len(ds["train"]),
        "val_size":   len(ds["validation"]),
        "test_size":  len(ds["test"]),
        "validation_calibration": cal_res,
    }
    rpt_path = os.path.join(REPORTS_DIR, "distilbert_metrics.json")
    with open(rpt_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[Train] Report saved to {rpt_path}")

    gc.collect()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DistilBERT Fake News Trainer")
    parser.add_argument("--subset", type=float, default=None,
                        help="Fraction of data to use (e.g. 0.1 for 10 percent)")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Max training epochs (default: 2)")
    args = parser.parse_args()
    train(subset_ratio=args.subset, num_epochs=args.epochs)
