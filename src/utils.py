import os
import re
import psutil
import torch
from typing import Dict, Any, List

def get_system_stats() -> Dict[str, Any]:
    """Retrieves current memory and CPU usage optimized for 8GB laptop monitoring."""
    mem = psutil.virtual_memory()
    device = "Apple Silicon MPS" if torch.backends.mps.is_available() else ("NVIDIA CUDA" if torch.cuda.is_available() else "CPU")
    return {
        "total_ram_gb": round(mem.total / (1024 ** 3), 2),
        "available_ram_gb": round(mem.available / (1024 ** 3), 2),
        "used_ram_gb": round(mem.used / (1024 ** 3), 2),
        "ram_percent": mem.percent,
        "cpu_percent": psutil.cpu_percent(interval=None),
        "compute_device": device,
        "is_8gb_profile": mem.total / (1024 ** 3) <= 9.0
    }

def clean_text(text: str) -> str:
    """Sanitizes raw text, strips excessive whitespace and unprintable characters."""
    if not text:
        return ""
    text = re.sub(r'http\S+|www\.\S+', '[URL]', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def calculate_readability_metrics(text: str) -> Dict[str, Any]:
    """Extracts surface-level linguistic features useful for forensic journalism."""
    words = re.findall(r'\b\w+\b', text)
    sentences = re.split(r'[.!?]+', text)
    sentences = [s for s in sentences if s.strip()]
    
    word_count = len(words)
    sentence_count = max(len(sentences), 1)
    avg_sentence_len = round(word_count / sentence_count, 1)
    
    # Capitalized / scream words
    caps_words = [w for w in words if w.isupper() and len(w) > 1 and w not in ["USA", "NASA", "WHO", "EU", "FBI", "CIA", "AI", "DNA"]]
    caps_ratio = round(len(caps_words) / max(word_count, 1) * 100, 2)
    
    # Exclamations
    exclamation_count = text.count('!')
    
    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_sentence_len": avg_sentence_len,
        "caps_ratio": caps_ratio,
        "caps_count": len(caps_words),
        "exclamation_count": exclamation_count
    }
