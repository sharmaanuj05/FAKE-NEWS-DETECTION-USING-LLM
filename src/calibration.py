"""
calibration.py — Post-processing Temperature Scaling for probability calibration.

Fits a single temperature parameter T > 0 on validation set logits using NLL loss.
Never touches the test set during fitting.
Calculates Expected Calibration Error (ECE) and Brier Score before & after.
Generates a reliability diagram saved to reports/calibration_curve.png.
"""
import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

REPORTS_DIR = os.path.join(BASE_DIR, "reports")
MODELS_DIR = os.path.join(BASE_DIR, "models", "distilbert_fake_news")


class TemperatureScaler(nn.Module):
    """
    Learns a scalar temperature T to scale logits before softmax:
        p_calibrated = softmax(logits / T)
    Optimized strictly on the validation set.
    """
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        temp = self.temperature.clamp(min=0.01, max=10.0)
        return logits / temp


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """
    Calculates Expected Calibration Error (ECE).
    ECE = sum_m (|B_m| / N) * |acc(B_m) - conf(B_m)|
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total_samples = len(labels)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper) if i > 0 else (confidences >= bin_lower) & (confidences <= bin_upper)
        bin_size = np.sum(in_bin)

        if bin_size > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += (bin_size / total_samples) * np.abs(bin_acc - bin_conf)

    return float(round(ece, 4))


def compute_brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculates Brier score (mean squared error of probability for positive class):
    BS = (1/N) * sum_i (p_fake - y_fake)^2
    """
    fake_probs = probs[:, 1]
    return float(round(np.mean((fake_probs - labels) ** 2), 4))


def fit_temperature_scaling(
    val_logits: np.ndarray,
    val_labels: np.ndarray,
    max_iter: int = 100,
    lr: float = 0.01,
    save_to_disk: bool = True
) -> dict:
    """
    Fits temperature parameter T strictly on validation set logits.
    Returns:
        {
            "temperature": float,
            "val_ece_uncalibrated": float,
            "val_ece_calibrated": float,
            "val_brier_uncalibrated": float,
            "val_brier_calibrated": float,
        }
    """
    logits_tensor = torch.tensor(val_logits, dtype=torch.float32)
    labels_tensor = torch.tensor(val_labels, dtype=torch.long)

    # Initial metrics before calibration
    raw_probs = torch.softmax(logits_tensor, dim=1).numpy()
    ece_before = compute_ece(raw_probs, val_labels)
    brier_before = compute_brier_score(raw_probs, val_labels)

    scaler = TemperatureScaler()
    nll_criterion = nn.CrossEntropyLoss()
    optimizer = optim.LBFGS([scaler.temperature], lr=lr, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        scaled_logits = scaler(logits_tensor)
        loss = nll_criterion(scaled_logits, labels_tensor)
        loss.backward()
        return loss

    optimizer.step(closure)

    fitted_temp = float(round(scaler.temperature.item(), 4))
    # Calibrated metrics on validation set
    with torch.no_grad():
        cal_logits = scaler(logits_tensor)
        cal_probs = torch.softmax(cal_logits, dim=1).numpy()

    ece_after = compute_ece(cal_probs, val_labels)
    brier_after = compute_brier_score(cal_probs, val_labels)

    results = {
        "temperature": fitted_temp,
        "val_ece_uncalibrated": ece_before,
        "val_ece_calibrated": ece_after,
        "val_brier_uncalibrated": brier_before,
        "val_brier_calibrated": brier_after,
        "val_samples_used": len(val_labels)
    }

    # Save temperature file to models directory if requested
    if save_to_disk:
        os.makedirs(MODELS_DIR, exist_ok=True)
        temp_path = os.path.join(MODELS_DIR, "temperature_scaling.json")
        with open(temp_path, "w") as f:
            json.dump(results, f, indent=2)

    return results



def plot_reliability_diagram(
    uncal_probs: np.ndarray,
    cal_probs: np.ndarray,
    labels: np.ndarray,
    save_path: str,
    n_bins: int = 10
):
    """
    Generates a 2-panel reliability diagram (Calibration curve)
    comparing uncalibrated vs calibrated probabilities.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    for ax, probs, title, color in zip(
        axes,
        [uncal_probs, cal_probs],
        ["Before Calibration (Raw DistilBERT)", "After Temperature Scaling (Calibrated)"],
        ["#ef4444", "#3b82f6"]
    ):
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
        accuracies = (predictions == labels).astype(float)

        bin_confs = []
        bin_accs = []
        bin_counts = []

        for i in range(n_bins):
            b_low = bin_edges[i]
            b_high = bin_edges[i + 1]
            in_b = (confidences > b_low) & (confidences <= b_high) if i > 0 else (confidences >= b_low) & (confidences <= b_high)
            count = np.sum(in_b)
            bin_counts.append(count)
            if count > 0:
                bin_accs.append(np.mean(accuracies[in_b]))
                bin_confs.append(np.mean(confidences[in_b]))
            else:
                bin_accs.append(np.nan)
                bin_confs.append((b_low + b_high) / 2)

        # Plot perfect diagonal
        ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect Calibration")
        # Plot empirical accuracy vs confidence
        ax.plot(bin_confs, bin_accs, marker="o", lw=2, color=color, label="Observed Accuracy")
        ax.bar(bin_confs, bin_accs, width=0.08, alpha=0.2, color=color, edgecolor=color)

        ece = compute_ece(probs, labels, n_bins)
        brier = compute_brier_score(probs, labels)
        ax.set_title(f"{title}\nECE: {ece:.4f} | Brier: {brier:.4f}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Confidence", fontsize=10)
        ax.set_ylabel("Empirical Accuracy", fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[Calibration] Reliability diagram saved → {save_path}")


def apply_temperature_scaling(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Scales logits by fitted temperature and applies softmax."""
    t = max(0.01, float(temperature))
    scaled = logits / t
    exp_scaled = np.exp(scaled - np.max(scaled, axis=-1, keepdims=True))
    return exp_scaled / np.sum(exp_scaled, axis=-1, keepdims=True)
