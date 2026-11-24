"""
test_preprocessing.py — Unit tests for ISOT dataset leakage removal, deduplication, and splits.
"""
import os
import sys
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.data_pipeline import clean_news_text, load_and_preprocess_dataset


def test_leakage_removal_reuters_dateline():
    """Ensures publisher datelines like 'WASHINGTON (Reuters) -' are stripped."""
    raw_title = "Senate passes bipartisan budget bill"
    raw_text = "WASHINGTON (Reuters) - The U.S. Senate approved a compromise federal budget on Thursday."
    cleaned = clean_news_text(raw_title, raw_text)
    assert "(Reuters)" not in cleaned
    assert "WASHINGTON (Reuters) -" not in cleaned
    assert "The U.S. Senate approved a compromise" in cleaned


def test_leakage_removal_via_credits_and_urls():
    """Ensures 'via ...' credits, URLs, and Twitter handles are stripped."""
    raw_title = "Breaking: Secret meeting exposed [VIDEO]"
    raw_text = "Whistleblower contacted @journalist via Facebook. Details at https://example.com/news"
    cleaned = clean_news_text(raw_title, raw_text)
    assert "https://" not in cleaned
    assert "@journalist" not in cleaned
    assert "via Facebook" not in cleaned
    assert "[VIDEO]" not in cleaned


def test_dataset_splits_and_labels():
    """Verifies that splits contain only text and label, labels are binary {0, 1}."""
    train_df, val_df, test_df = load_and_preprocess_dataset(subset_ratio=0.01, seed=42)

    for df in [train_df, val_df, test_df]:
        assert list(df.columns) == ["text", "label"]
        assert set(df["label"].unique()).issubset({0, 1})
        assert df["text"].isna().sum() == 0
        assert len(df) > 0
