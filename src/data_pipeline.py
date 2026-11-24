"""
data_pipeline.py — ISOT Fake and Real News dataset loader with strict leakage removal.

Loads GonzaloA/fake_news (ISOT dataset on HuggingFace Hub), strips publisher signatures,
datelines, 'via ...' credits, handles, URLs, and boilerplate. Deduplicates articles
and produces a stratified 80/10/10 train/val/test split.

Labels:
    0 = REAL NEWS (e.g. verified news reports)
    1 = FAKE NEWS (e.g. fabricated / satirical / misleading articles)
"""
import os
import sys
import re
import argparse
import pandas as pd
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split
from datasets import load_dataset

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
SEED = 42


def clean_news_text(title: str, text: str) -> str:
    """
    Combines title + text and rigorously strips source-identifying leakage:
    - Publisher names & datelines (e.g. 'WASHINGTON (Reuters) -', 'MOSCOW (Reuters) —')
    - Standalone publisher mentions like '(Reuters)', '(AP)', 'Reuters'
    - 'via ...' credits (e.g. 'Featured image via Facebook', 'via Getty Images')
    - Twitter handles (@username)
    - URLs (http://, https://, www.)
    - Boilerplate brackets like '[Video]', '[IMAGE]', '[Tweets]'
    """
    title_str = str(title).strip() if pd.notna(title) else ""
    text_str = str(text).strip() if pd.notna(text) else ""
    combined = (title_str + " " + text_str).strip()

    # 1. Strip URLs
    combined = re.sub(r"https?://\S+|www\.\S+", " ", combined)

    # 2. Strip Twitter handles
    combined = re.sub(r"@\w+", " ", combined)

    # 3. Strip datelines at start or within text: e.g. 'WASHINGTON (Reuters) -'
    combined = re.sub(
        r"^[A-Za-z\s,./-]+?\((?:Reuters|REUTERS|AP|AFP)\)\s*[-—:]\s*",
        "",
        combined,
        flags=re.MULTILINE
    )
    combined = re.sub(
        r"\b(?:WASHINGTON|NEW YORK|LONDON|MOSCOW|BEIJING|BERLIN|PARIS|TOKYO|GENEVA|BRUSSELS|JERUSALEM|BAGHDAD|KABUL|SEOUL|ANKARA|RIYADH|DUBAI|HONG KONG)\s*\((?:Reuters|REUTERS|AP|AFP)\)\s*[-—:]\s*",
        " ",
        combined,
        flags=re.IGNORECASE
    )
    combined = re.sub(r"\((?:Reuters|REUTERS|AP|AFP)\)", " ", combined)
    combined = re.sub(r"\bReuters\b", " ", combined, flags=re.IGNORECASE)

    # 4. Strip 'via ...' credits and 'Featured image via...'
    combined = re.sub(
        r"\b(?:Featured image|Image|Photo|Video)?\s*via\s+[A-Za-z0-9_.\s-]+(?:\.\w+)?",
        " ",
        combined,
        flags=re.IGNORECASE
    )

    # 5. Strip media boilerplate brackets
    combined = re.sub(r"\[(?:VIDEO|IMAGE|PICTURES?|TWEETS?|SCREENSHOTS?)\]", " ", combined, flags=re.IGNORECASE)

    # 6. Normalize whitespace
    combined = re.sub(r"\s+", " ", combined).strip()
    return combined


def build_and_cache_dataset() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Downloads GonzaloA/fake_news from Hub, cleans, deduplicates, and splits 80/10/10."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    train_cache = os.path.join(PROCESSED_DIR, "train.parquet")
    val_cache   = os.path.join(PROCESSED_DIR, "val.parquet")
    test_cache  = os.path.join(PROCESSED_DIR, "test.parquet")

    if os.path.exists(train_cache) and os.path.exists(val_cache) and os.path.exists(test_cache):
        print(f"[Data] Loading cached preprocessed splits from {PROCESSED_DIR}...")
        train_df = pd.read_parquet(train_cache)
        val_df   = pd.read_parquet(val_cache)
        test_df  = pd.read_parquet(test_cache)
        return train_df, val_df, test_df

    print("[Data] Downloading GonzaloA/fake_news from HuggingFace Hub...")
    ds = load_dataset("GonzaloA/fake_news")
    full_df = pd.concat(
        [pd.DataFrame(ds["train"]), pd.DataFrame(ds["validation"]), pd.DataFrame(ds["test"])],
        ignore_index=True
    )
    print(f"[Data] Loaded {len(full_df)} raw articles.")

    print("[Data] Removing label leakage (datelines, publisher credits, URLs, handles)...")
    full_df["text"] = [clean_news_text(t, x) for t, x in zip(full_df["title"], full_df["text"])]

    # Label harmonization: in GonzaloA/fake_news, 1=Real, 0=Fake
    # Harmonize to: 0=REAL, 1=FAKE
    full_df["label"] = full_df["label"].apply(lambda x: 0 if x == 1 else 1)
    full_df = full_df[["text", "label"]].copy()

    # Drop empty or trivial texts (<25 characters)
    full_df = full_df[full_df["text"].str.len() > 25].copy()

    # Deduplicate strictly on cleaned text
    before_dedup = len(full_df)
    full_df = full_df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    print(f"[Data] Deduplicated: {before_dedup} -> {len(full_df)} unique articles.")

    # Stratified 80 / 10 / 10 split with seed 42
    train_df, temp_df = train_test_split(
        full_df, test_size=0.20, random_state=SEED, stratify=full_df["label"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=SEED, stratify=temp_df["label"]
    )

    # Save to disk cache
    train_df.to_parquet(train_cache, index=False)
    val_df.to_parquet(val_cache, index=False)
    test_df.to_parquet(test_cache, index=False)
    print(f"[Data] Saved processed splits to {PROCESSED_DIR}")

    return train_df, val_df, test_df


def load_and_preprocess_dataset(
    subset_ratio: Optional[float] = None,
    seed: int = SEED
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns train, validation, and test splits with label counts printed.
    If subset_ratio is given (e.g. 0.1), takes a stratified subset of each split.
    """
    train_df, val_df, test_df = build_and_cache_dataset()

    if subset_ratio is not None and 0.0 < subset_ratio < 1.0:
        def sample_stratified(df: pd.DataFrame, ratio: float) -> pd.DataFrame:
            real_part = df[df["label"] == 0].sample(frac=ratio, random_state=seed)
            fake_part = df[df["label"] == 1].sample(frac=ratio, random_state=seed)
            return (
                pd.concat([real_part, fake_part], ignore_index=True)
                .sample(frac=1.0, random_state=seed)
                .reset_index(drop=True)
            )
        train_df = sample_stratified(train_df, subset_ratio)
        val_df   = sample_stratified(val_df, subset_ratio)
        test_df  = sample_stratified(test_df, subset_ratio)

    print("\n" + "=" * 60)
    print(f" ISOT News Dataset Class Counts {'(Subsample: ' + str(int(subset_ratio*100)) + '%)' if subset_ratio else '(Full)'}")
    print("=" * 60)
    for name, df in [("TRAIN", train_df), ("VALID", val_df), ("TEST", test_df)]:
        n_real = int((df["label"] == 0).sum())
        n_fake = int((df["label"] == 1).sum())
        print(f"{name:<10} Total: {len(df):<6} | REAL (0): {n_real:<6} | FAKE (1): {n_fake:<6}")
    print("=" * 60 + "\n")

    return train_df, val_df, test_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ISOT Fake News Data Pipeline")
    parser.add_argument("--subset", type=float, default=None, help="Fraction to subsample (e.g. 0.1)")
    args = parser.parse_args()
    load_and_preprocess_dataset(subset_ratio=args.subset)
