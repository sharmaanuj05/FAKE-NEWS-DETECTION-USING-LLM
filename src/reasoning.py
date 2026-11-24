"""
src/reasoning.py — Word attribution layer using Captum Integrated Gradients.

Computes LayerIntegratedGradients on DistilBERT embedding layer,
merges subwords into whole words, filters special tokens, stopwords, and punctuation,
and returns the top 5 words pushing toward the predicted class with their scores.
Includes faithfulness validation via top-5 deletion vs random 5 deletion.
"""
import os
import sys
import string
import random
from typing import List, Tuple, Dict, Any
import numpy as np
import torch
from captum.attr import LayerIntegratedGradients
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

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
    "yourself", "yourselves", "s", "t", "will", "just", "said", "also", "would"
}


class WordAttributionExplainer:
    def __init__(self, model_dir: str = MODELS_DIR, model=None, tokenizer=None):
        self.model_dir = model_dir
        self.tokenizer = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(model_dir)
        self.model = model if model is not None else AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()

        def forward_func(input_ids, attention_mask):
            return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

        self.lig = LayerIntegratedGradients(forward_func, self.model.distilbert.embeddings)


    def attribute(self, text: str, target_class: int, n_steps: int = 20, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Computes Captum Integrated Gradients on embeddings, merges sub-words into whole words,
        filters stopwords and punctuation, and returns top_k words with scores.
        """
        if not text or not text.strip():
            return []

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=MAX_LENGTH
        )
        offset_mapping = inputs.pop("offset_mapping")[0].tolist()

        try:
            attrs = self.lig.attribute(
                inputs["input_ids"],
                additional_forward_args=(inputs["attention_mask"],),
                target=target_class,
                n_steps=n_steps
            )
        except Exception as e:
            # Fallback if Captum errors
            return []

        token_attrs = attrs.sum(dim=-1).squeeze(0).detach().cpu().numpy()
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        word_scores: Dict[str, float] = {}
        curr_word = ""
        curr_score = 0.0

        for tok, score, (start, end) in zip(tokens, token_attrs, offset_mapping):
            if start == end:  # [CLS], [SEP], [PAD]
                continue
            clean_tok = tok.strip()
            if clean_tok.startswith("##"):
                curr_word += clean_tok[2:]
                curr_score += float(score)
            else:
                if curr_word:
                    w_norm = curr_word.strip(string.punctuation).lower()
                    if len(w_norm) > 2 and w_norm not in STOPWORDS:
                        word_scores[w_norm] = word_scores.get(w_norm, 0.0) + curr_score
                curr_word = clean_tok
                curr_score = float(score)

        if curr_word:
            w_norm = curr_word.strip(string.punctuation).lower()
            if len(w_norm) > 2 and w_norm not in STOPWORDS:
                word_scores[w_norm] = word_scores.get(w_norm, 0.0) + curr_score

        # Filter out empty strings
        sorted_words = sorted(
            [{"word": w, "score": round(float(s), 4)} for w, s in word_scores.items() if w],
            key=lambda x: x["score"],
            reverse=True
        )

        return sorted_words[:top_k]


def run_faithfulness_check(
    explainer: WordAttributionExplainer,
    texts: List[str],
    labels: List[int],
    n_samples: int = 50,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Evaluates attribution faithfulness on n_samples:
    Compares probability drop of predicted class when deleting top-5 words
    vs deleting 5 random words.
    """
    random.seed(seed)
    np.random.seed(seed)

    indices = list(range(len(texts)))
    random.shuffle(indices)
    sample_indices = indices[:n_samples]

    top5_drops = []
    random5_drops = []

    for idx in sample_indices:
        text = texts[idx]
        words = text.split()
        if len(words) < 15:
            continue

        # Baseline prediction
        enc = explainer.tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
        with torch.no_grad():
            logits = explainer.model(**enc).logits.numpy()
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[0]
        pred_class = int(np.argmax(probs))
        base_prob = probs[pred_class]

        # Top 5 attribution
        top_attrs = explainer.attribute(text, target_class=pred_class, n_steps=20, top_k=5)
        top5_words = {a["word"].lower() for a in top_attrs}

        # 1. Delete top-5 words
        ablated_top = " ".join([w for w in words if w.strip(string.punctuation).lower() not in top5_words])
        if not ablated_top.strip():
            continue
        enc_top = explainer.tokenizer(ablated_top, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
        with torch.no_grad():
            logits_top = explainer.model(**enc_top).logits.numpy()
        prob_top = torch.softmax(torch.tensor(logits_top), dim=-1).numpy()[0][pred_class]
        top5_drops.append(float(base_prob - prob_top))

        # 2. Delete 5 random words
        eligible_random = [w for w in words if len(w) > 2 and w.lower() not in STOPWORDS]
        if len(eligible_random) >= 5:
            rand_chosen = set(random.sample(eligible_random, 5))
        else:
            rand_chosen = set(eligible_random)
        ablated_rand = " ".join([w for w in words if w not in rand_chosen])
        if not ablated_rand.strip():
            continue
        enc_rand = explainer.tokenizer(ablated_rand, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
        with torch.no_grad():
            logits_rand = explainer.model(**enc_rand).logits.numpy()
        prob_rand = torch.softmax(torch.tensor(logits_rand), dim=-1).numpy()[0][pred_class]
        random5_drops.append(float(base_prob - prob_rand))

    avg_top5_drop = float(np.mean(top5_drops)) if top5_drops else 0.0
    avg_rand5_drop = float(np.mean(random5_drops)) if random5_drops else 0.0
    is_reliable = avg_top5_drop > avg_rand5_drop

    return {
        "samples_evaluated": len(top5_drops),
        "avg_top5_prob_drop": round(avg_top5_drop, 4),
        "avg_random5_prob_drop": round(avg_rand5_drop, 4),
        "drop_ratio": round(avg_top5_drop / max(avg_rand5_drop, 1e-6), 2),
        "is_reliable": bool(is_reliable),
        "attribution_method": "Captum LayerIntegratedGradients (embedding layer, n_steps=20)",
    }
