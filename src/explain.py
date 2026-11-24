"""
src/explain.py — Calibrated prediction and explanation pipeline.

Features:
- Verdict: REAL / FAKE / UNCERTAIN based on calibrated confidence and config threshold.
- Word attribution using Captum Integrated Gradients (LayerIntegratedGradients).
- Rule-based red-flag detection (advisory, does not modify verdict).
- Deterministic template-based summary (no LLM / no generative hallucinations).
- Temperature scaling fitted on validation set only.
- Graceful edge-case handling for empty, one-word, and very long inputs.
"""
import os
import sys
import re
import json
from typing import Dict, Any, Optional
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from config import UNCERTAIN_CONFIDENCE_THRESHOLD, CAPTUM_IG_STEPS
from src.data_pipeline import clean_news_text
from src.calibration import apply_temperature_scaling
from src.reasoning import WordAttributionExplainer
from src.red_flags import detect_red_flags

V2_DIR = os.path.join(BASE_DIR, "models", "distilbert_v2")
V1_DIR = os.path.join(BASE_DIR, "models", "distilbert_fake_news")
MODELS_DIR = V2_DIR if os.path.exists(os.path.join(V2_DIR, "config.json")) else V1_DIR
MAX_LENGTH = 128 if MODELS_DIR == V2_DIR else 256

# Rules that were kept after validation on the real holdout test set:
ACTIVE_RED_FLAG_RULES = [
    "excessive_caps",
    "clickbait_phrases",
    "excessive_punctuation",
    "emotional_absolutist",
    "no_named_attribution",
]


class ExplainerPipeline:
    _instance = None

    def __init__(self, model_dir: str = MODELS_DIR):
        self.model_dir = model_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()

        # Load validation-fitted temperature
        cal_path = os.path.join(model_dir, "temperature_scaling.json")
        self.temperature = 1.0
        if os.path.exists(cal_path):
            try:
                with open(cal_path) as f:
                    self.temperature = float(json.load(f).get("temperature", 1.0))
            except Exception:
                self.temperature = 1.0

        # Word attribution explainer (shares model weights to avoid redundant loading)
        self.attr_explainer = WordAttributionExplainer(
            model_dir=model_dir,
            model=self.model,
            tokenizer=self.tokenizer
        )


    @classmethod
    def get_instance(cls, model_dir: str = MODELS_DIR):
        if cls._instance is None:
            cls._instance = cls(model_dir)
        return cls._instance

    def explain(
        self,
        text: str,
        threshold: float = UNCERTAIN_CONFIDENCE_THRESHOLD,
        top_k_words: int = 5,
        live_verify: bool = True
    ) -> Dict[str, Any]:
        """
        Explains prediction for given news text.
        Returns dictionary with verdict, confidence, top words, red flags, and template summary.
        """
        # Edge case: non-string input
        if not isinstance(text, str):
            text = str(text) if text is not None else ""

        # Edge case: empty or whitespace
        stripped = text.strip()
        if not stripped:
            return {
                "status": "error",
                "message": "Input text is empty. Please enter a news headline or article.",
                "verdict": None,
                "confidence": None,
                "top_words": [],
                "triggered_flags": [],
                "summary": "No input provided to evaluate."
            }

        # Edge case: pure URL
        if re.match(r"^https?://[^\s]+$", stripped, re.IGNORECASE):
            return {
                "status": "error",
                "message": "Input appears to be a URL link rather than article text. Please paste the headline or article content directly.",
                "verdict": None,
                "confidence": None,
                "top_words": [],
                "triggered_flags": [],
                "summary": "URLs cannot be evaluated directly. Please paste the article text."
            }

        # Safe bounds against oversized inputs (cap at 50,000 characters before regex)
        if len(text) > 50000:
            text = text[:50000]

        # Extract actual alphabetic words (filters out pure emojis, numbers, or punctuation)
        alpha_words = re.findall(r"\b[A-Za-z]{2,}\b", text)
        if len(alpha_words) < 3:
            return {
                "status": "error",
                "message": "Input text is too short to evaluate (less than 3 words provided; at least 3 recognizable words required).",
                "verdict": None,
                "confidence": None,
                "top_words": [],
                "triggered_flags": [],
                "summary": "Input text lacks sufficient linguistic content for writing pattern analysis."
            }


        cleaned = clean_news_text("", text)

        # Truncate for tokenization
        enc = self.tokenizer(

            cleaned if cleaned else text,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        with torch.no_grad():
            logits = self.model(**enc).logits.numpy()

        # Apply validation-fitted temperature scaling
        probs = apply_temperature_scaling(logits, self.temperature)[0]
        real_prob = float(probs[0])
        fake_prob = float(probs[1])

        max_prob = max(real_prob, fake_prob)
        raw_pred_class = 1 if fake_prob >= real_prob else 0
        raw_label = "FAKE" if raw_pred_class == 1 else "REAL"

        # Word attribution
        target_for_attr = raw_pred_class
        top_words_with_scores = self.attr_explainer.attribute(
            cleaned if cleaned else text,
            target_class=target_for_attr,
            n_steps=CAPTUM_IG_STEPS,
            top_k=top_k_words
        )

        # Advisory red-flag detection (on raw text to retain casing, punctuation, formatting)
        flags = detect_red_flags(text, active_rules=ACTIVE_RED_FLAG_RULES)

        # Real-Time Live Web Fact-Checking & Corroboration
        fact_check_res = None
        if live_verify:
            try:
                from src.fact_checker import verify_news_live
                fact_check_res = verify_news_live(text)
            except Exception:
                fact_check_res = None

        # Normalize evidence status
        debunks = fact_check_res.get("debunk_articles", []) if fact_check_res else []
        credibles = fact_check_res.get("credible_sources", []) if fact_check_res else []
        ev_st = fact_check_res.get("status", "UNAVAILABLE") if fact_check_res else "UNAVAILABLE"

        if debunks or ev_st in ["REFUTED", "DEBUNKED_FAKE"]:
            canon_ev = "REFUTED"
        elif len(credibles) >= 2 or ev_st in ["CORROBORATED", "VERIFIED_REAL"]:
            canon_ev = "CORROBORATED"
        elif ev_st in ["NO_EVIDENCE", "NO_COVERAGE_DETECTED", "INCONCLUSIVE", "PARTIALLY_CORROBORATED"]:
            canon_ev = "NO_EVIDENCE"
        else:
            canon_ev = "UNAVAILABLE"

        # Explicit Decision Table (Step 5)
        if canon_ev == "REFUTED":
            verdict = "FAKE"
            rule_applied = "REFUTED_BY_FACT_CHECKER"
            reason = f"Refuted by independent fact-checkers ({fact_check_res.get('summary', 'Debunked online')})."
            confidence = max(round(max_prob * 100, 1), 96.0)
        elif canon_ev == "CORROBORATED" and not (raw_label == "FAKE" and max_prob >= 0.85):
            verdict = "REAL"
            rule_applied = "CORROBORATED_BY_CREDIBLE_NEWS"
            sources_str = ", ".join([s["source"] for s in credibles[:3]]) if credibles else "major agencies"
            reason = f"Corroborated by independent credible news reporting ({sources_str})."
            confidence = max(round(max_prob * 100, 1), 95.0)
        elif raw_label == "FAKE" and max_prob >= threshold:
            verdict = "FAKE"
            rule_applied = "CLASSIFIER_HIGH_CONFIDENCE_FAKE"
            reason = f"Linguistic writing patterns strongly indicate misinformation (calibrated confidence {round(max_prob * 100, 1)}%)."
            confidence = round(max_prob * 100, 1)
        elif raw_label == "REAL" and canon_ev in ["NO_EVIDENCE", "UNAVAILABLE"]:
            verdict = "UNCERTAIN"
            rule_applied = "CLASSIFIER_REAL_UNSUPPORTED"
            reason = "Classifier leans real, but no independent source confirms it."
            confidence = round(max_prob * 100, 1)
        else:
            verdict = "UNCERTAIN"
            rule_applied = "INCONCLUSIVE_FALLBACK"
            reason = "Evidence and writing patterns are inconclusive or below decision threshold."
            confidence = round(max_prob * 100, 1)

        # Deterministic summary
        words_str = ", ".join([f"'{item['word']}' ({item['score']:.3f})" for item in top_words_with_scores]) if top_words_with_scores else "none identified"
        flags_str = ", ".join([f["name"] for f in flags]) if flags else "none"

        summary = (
            f"Verdict: {verdict} ({confidence}% confidence). Reason: {reason} "
            f"Model leaned {raw_label} ({round(max_prob*100, 1)}% writing-pattern score). "
            f"Key words: {words_str}. Warning signs: {flags_str}. "
            f"This reflects linguistic patterns and public cross-checking, not guaranteed truth."
        )

        return {
            "status": "success",
            "verdict": verdict,
            "rule_applied": rule_applied,
            "reason": reason,
            "raw_label": raw_label,
            "confidence": confidence,
            "real_probability": round(real_prob * 100, 2),
            "fake_probability": round(fake_prob * 100, 2),
            "threshold_used": threshold,
            "temperature_used": self.temperature,
            "top_words": top_words_with_scores,
            "triggered_flags": flags,
            "live_verification": fact_check_res,
            "summary": summary
        }


def explain(
    text: str,
    threshold: float = UNCERTAIN_CONFIDENCE_THRESHOLD,
    live_verify: bool = True
) -> Dict[str, Any]:
    """Top-level functional interface for model explanation."""
    pipeline = ExplainerPipeline.get_instance()
    return pipeline.explain(text, threshold=threshold, live_verify=live_verify)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_text = " ".join(sys.argv[1:])
        result = explain(input_text)
        print(json.dumps(result, indent=2))
    else:
        print("Usage: python src/explain.py \"news text to evaluate\"")
