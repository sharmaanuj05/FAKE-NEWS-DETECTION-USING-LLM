import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import json
import pandas as pd
import torch
from typing import Tuple, List, Dict, Any, Optional

try:
    from torch.utils.data import Dataset
except ImportError:
    class Dataset:
        pass

from config import DATA_DIR, MAX_SEQ_LENGTH

class FakeNewsDataset(Dataset):
    """PyTorch Dataset for text classification fine-tuning."""
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = MAX_SEQ_LENGTH):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }

def load_curated_samples() -> List[Dict[str, Any]]:
    """Loads curated multi-domain real/fake news samples with reasoning annotations."""
    path = os.path.join(DATA_DIR, "sample_news.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def load_benchmark_dataframe(csv_path: Optional[str] = None) -> pd.DataFrame:
    """Loads the benchmark dataset CSV."""
    if not csv_path:
        csv_path = os.path.join(DATA_DIR, "test_benchmark.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return pd.DataFrame(columns=["id", "title", "text", "label"])
