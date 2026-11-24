"""
tests/test_step7_data_integrity.py — Rigorous unit tests for Step 7:
1. Truncation augmentation & no cross-split leakage
2. Label mapping by name
3. Unseen-source holdout isolation
"""
import os
import sys
import pytest
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed_v2")

@pytest.fixture(scope="module")
def dataset_splits():
    train_df = pd.read_parquet(os.path.join(DATA_DIR, "train.parquet"))
    val_df = pd.read_parquet(os.path.join(DATA_DIR, "val.parquet"))
    test_mixed_df = pd.read_parquet(os.path.join(DATA_DIR, "test_mixed.parquet"))
    unseen_df = pd.read_parquet(os.path.join(DATA_DIR, "unseen_source.parquet"))
    return {
        "train": train_df,
        "val": val_df,
        "test_mixed": test_mixed_df,
        "unseen": unseen_df
    }

class TestDataIntegrityAndLeakage:
    def test_no_exact_text_leakage_across_splits(self, dataset_splits):
        """Verify zero text overlap between train and all evaluation splits."""
        train_texts = set(dataset_splits["train"]["text"].str.strip().str.lower())
        val_texts = set(dataset_splits["val"]["text"].str.strip().str.lower())
        test_mixed_texts = set(dataset_splits["test_mixed"]["text"].str.strip().str.lower())
        unseen_texts = set(dataset_splits["unseen"]["text"].str.strip().str.lower())

        val_overlap = train_texts.intersection(val_texts)
        assert len(val_overlap) == 0, f"Found {len(val_overlap)} leaking items between train and val: {list(val_overlap)[:3]}"

        test_mixed_overlap = train_texts.intersection(test_mixed_texts)
        assert len(test_mixed_overlap) == 0, f"Found {len(test_mixed_overlap)} leaking items between train and test_mixed: {list(test_mixed_overlap)[:3]}"

        unseen_overlap = train_texts.intersection(unseen_texts)
        assert len(unseen_overlap) == 0, f"Found {len(unseen_overlap)} leaking items between train and unseen: {list(unseen_overlap)[:3]}"

    def test_truncation_augmentation_isolated_to_train_only(self, dataset_splits):
        """
        Verify truncation augmentation is present only in training split
        and evaluation splits contain only full original texts.
        """
        train_df = dataset_splits["train"]
        val_df = dataset_splits["val"]
        test_mixed_df = dataset_splits["test_mixed"]
        unseen_df = dataset_splits["unseen"]

        # Train has augmentation types
        aug_count = (train_df["source"].str.endswith("_aug")).sum()
        assert aug_count > 0, f"Training set must contain length-augmented samples, found {aug_count}"

        # Holdout test splits must not contain synthetic/augmented clones
        assert (test_mixed_df["source"].str.endswith("_aug")).sum() == 0, "Test mixed set must not have augmented clones"
        assert (unseen_df["source"].str.endswith("_aug")).sum() == 0, "Unseen set must not have augmented clones"

    def test_label_mapping_by_name(self, dataset_splits):
        """
        Verify that all splits map labels strictly by name into binary integers (0=REAL, 1=FAKE).
        """
        for name, df in dataset_splits.items():
            assert "label" in df.columns, f"{name} split missing 'label' column"
            unique_labels = set(df["label"].unique())
            assert unique_labels.issubset({0, 1}), f"{name} contains unexpected labels: {unique_labels}"

            # Verify balance
            class_counts = df["label"].value_counts().to_dict()
            assert 0 in class_counts and 1 in class_counts, f"{name} missing one class entirely"

    def test_unseen_source_complete_isolation(self, dataset_splits):
        """Verify unseen source (LIAR test) was completely held out from train and val."""
        unseen_df = dataset_splits["unseen"]
        train_df = dataset_splits["train"]

        # Ensure unseen set has samples
        assert len(unseen_df) > 1000, f"Unseen split should have >1000 samples, got {len(unseen_df)}"

        # Ensure all unseen rows are from LIAR
        assert all(unseen_df["source"].str.contains("liar", case=False)), "All unseen samples must be from LIAR"

        # Ensure no sample text from unseen set appears in train
        train_set = set(train_df["text"])
        unseen_set = set(unseen_df["text"])
        assert len(train_set.intersection(unseen_set)) == 0
