import os
import torch
import psutil

# Hardware & System Detection
RAM_GB = psutil.virtual_memory().total / (1024 ** 3)
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

# Model Configuration (Optimized for 8GB Laptop Profile)
DEFAULT_MODEL_NAME = os.getenv("MODEL_NAME", "distilbert/distilbert-base-uncased-finetuned-sst-2-english")
FALLBACK_MODEL_NAME = "distilbert-base-uncased"
MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "256"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))
GRADIENT_ACCUMULATION_STEPS = 2

# Memory Safety Bounds
MEMORY_THRESHOLD_MB = 2048  # Cap individual model memory usage to ~2GB to preserve system stability

# Phase 2 Reasoning & Calibration Settings
# Chosen strictly via validation split tuning: 0.70 yields 99.05% validation accuracy with only 1.01% rejected
UNCERTAIN_CONFIDENCE_THRESHOLD = float(os.getenv("UNCERTAIN_THRESHOLD", "0.70"))
CAPTUM_IG_STEPS = 20


# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Labels
LABEL_MAP = {
    0: "REAL / CREDIBLE",
    1: "MISINFORMATION / FAKE"
}

# Reasoning Weightings
REASONING_PILLARS = [
    {"id": "linguistic_bias", "name": "Linguistic & Emotional Bias", "weight": 0.25},
    {"id": "source_rigor", "name": "Source & Attribution Rigor", "weight": 0.25},
    {"id": "fallacy_score", "name": "Logical Fallacy & Framing", "weight": 0.25},
    {"id": "sensationalism", "name": "Sensationalism & Clickbait Index", "weight": 0.25}
]
