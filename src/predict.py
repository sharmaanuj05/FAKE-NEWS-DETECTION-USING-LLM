"""
predict.py — Minimal CLI predictor for Fake News detection.

Usage:
    python src/predict.py "text goes here"
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.data_pipeline import clean_news_text
from src.calibration import apply_temperature_scaling

MODELS_DIR = os.path.join(BASE_DIR, "models", "distilbert_fake_news")
MAX_LENGTH = 256

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves", "s", "t", "will", "just", "said"
}


def get_top_5_words(text: str, tokenizer, model, temp: float, target_class: int) -> list[str]:
    words = [w.strip(".,!?\"'()[]:;–—") for w in text.split()]
    words = [w for w in words if len(w) > 2 and w.lower() not in STOPWORDS]
    if not words:
        return []

    enc = tokenizer(text, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    with torch.no_grad():
        base_logits = model(**enc).logits.numpy()
    base_prob = apply_temperature_scaling(base_logits, temp)[0][target_class]

    unique_words = list(dict.fromkeys(words))[:30]
    impacts = []
    for w in unique_words:
        ablated = " ".join([x for x in words if x != w])
        if not ablated.strip():
            continue
        enc_ab = tokenizer(ablated, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        with torch.no_grad():
            ab_logits = model(**enc_ab).logits.numpy()
        ab_prob = apply_temperature_scaling(ab_logits, temp)[0][target_class]
        impacts.append((w, base_prob - ab_prob))

    impacts.sort(key=lambda x: x[1], reverse=True)
    return [w for w, _ in impacts[:5]]


def predict_text(text: str, model_dir: str = MODELS_DIR) -> dict:
    if not os.path.exists(os.path.join(model_dir, "config.json")):
        raise FileNotFoundError(f"Model not found at {model_dir}. Run train.py first.")

    cleaned = clean_news_text("", text)
    if not cleaned:
        raise ValueError("Input text is empty after cleaning.")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    temp = 1.0
    cal_file = os.path.join(model_dir, "temperature_scaling.json")
    if os.path.exists(cal_file):
        try:
            with open(cal_file) as f:
                temp = float(json.load(f).get("temperature", 1.0))
        except Exception:
            temp = 1.0

    enc = tokenizer(cleaned, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    with torch.no_grad():
        logits = model(**enc).logits.numpy()

    probs = apply_temperature_scaling(logits, temp)[0]
    fake_prob = float(probs[1])
    real_prob = float(probs[0])

    if 0.45 <= fake_prob <= 0.55:
        verdict = "UNCERTAIN"
        conf_pct = max(fake_prob, real_prob) * 100
        target_class = int(np.argmax(probs))
    elif fake_prob > 0.55:
        verdict = "FAKE"
        conf_pct = fake_prob * 100
        target_class = 1
    else:
        verdict = "REAL"
        conf_pct = real_prob * 100
        target_class = 0

    top_words = get_top_5_words(cleaned, tokenizer, model, temp, target_class)

    return {
        "verdict": verdict,
        "confidence_pct": round(conf_pct, 1),
        "fake_probability": round(fake_prob, 4),
        "real_probability": round(real_prob, 4),
        "top_5_words": top_words,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fake News CLI Predictor")
    parser.add_argument("text", type=str, help="News headline or article to check")
    args = parser.parse_args()

    res = predict_text(args.text)
    print("\n-----------------------------")
    print(f"Verdict    : {res['verdict']}")
    print(f"Confidence : {res['confidence_pct']}%")
    if res["top_5_words"]:
        print(f"Top Words  : {', '.join(res['top_5_words'])}")
    print("-----------------------------\n")
