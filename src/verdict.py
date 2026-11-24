"""
src/verdict.py — Step 5: Hybrid Decision Engine (Classifier Writing Patterns + Live Fact-Checking Evidence).

Combines probabilistic transformer predictions with live web journalistic corroboration
using an explicit, testable, deterministic Decision Table.

Explicit Decision Table:
1. REFUTED by a fact-checker
   -> FAKE (Fact-checking organizations Snopes/PolitiFact/Reuters debunked the claim)
2. CORROBORATED by >= 2 independent credible outlets and classifier NOT high-confidence FAKE
   -> REAL (Active reporting in mainstream journalistic agencies confirms the event)
3. Classifier FAKE with calibrated confidence >= threshold
   -> FAKE (Linguistic writing patterns strongly indicate deception/misinformation)
4. Classifier REAL but evidence NO_EVIDENCE or UNAVAILABLE for a checkable claim
   -> UNCERTAIN ("Classifier leans real, but no independent source confirms it")
5. Everything else (e.g. low-confidence, conflicting signals)
   -> UNCERTAIN

Goal: Prevent ever labeling an unsubstantiated statement REAL on model patterns alone.
"""
import os
import sys
import json
import re
from typing import Dict, Any, Optional
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.data_pipeline import clean_news_text
from src.calibration import apply_temperature_scaling
from src.fact_checker import LiveNewsFactChecker

# Default to newly trained distilbert_v2 if available, else fallback to distilbert_fake_news
V2_MODEL_DIR = os.path.join(BASE_DIR, "models", "distilbert_v2")
V1_MODEL_DIR = os.path.join(BASE_DIR, "models", "distilbert_fake_news")
DEFAULT_MODEL_DIR = V2_MODEL_DIR if os.path.exists(os.path.join(V2_MODEL_DIR, "config.json")) else V1_MODEL_DIR

DEFAULT_THRESHOLD = 0.70
HIGH_CONF_FAKE_OVERRIDE = 0.85
MAX_LENGTH = 128


class HybridDecisionEngine:
    _instance = None

    def __init__(self, model_dir: str = DEFAULT_MODEL_DIR):
        self.model_dir = model_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()

        cal_path = os.path.join(model_dir, "temperature_scaling.json")
        self.temperature = 1.0
        if os.path.exists(cal_path):
            try:
                with open(cal_path) as f:
                    self.temperature = float(json.load(f).get("temperature", 1.0))
            except Exception:
                self.temperature = 1.0

        self.fact_checker = LiveNewsFactChecker.get_instance(timeout=5)

    @classmethod
    def get_instance(cls, model_dir: str = DEFAULT_MODEL_DIR):
        if cls._instance is None or cls._instance.model_dir != model_dir:
            cls._instance = cls(model_dir=model_dir)
        return cls._instance

    def run_classifier(self, text: str) -> Dict[str, Any]:
        """Runs the calibrated neural classifier on the cleaned text."""
        cleaned = clean_news_text("", text)
        if not cleaned:
            cleaned = text.strip()

        enc = self.tokenizer(
            cleaned,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
        with torch.no_grad():
            logits = self.model(**enc).logits.numpy()

        probs = apply_temperature_scaling(logits, self.temperature)[0]
        real_prob = float(probs[0])
        fake_prob = float(probs[1])

        if fake_prob >= real_prob:
            verdict = "FAKE"
            conf = fake_prob
        else:
            verdict = "REAL"
            conf = real_prob

        return {
            "verdict": verdict,
            "confidence": round(conf, 4),
            "confidence_pct": round(conf * 100, 1),
            "real_probability": round(real_prob * 100, 2),
            "fake_probability": round(fake_prob * 100, 2),
            "temperature_used": self.temperature
        }

    def decide(
        self,
        text: str,
        threshold: float = DEFAULT_THRESHOLD,
        evidence_override: Optional[Dict[str, Any]] = None,
        live_verify: bool = True
    ) -> Dict[str, Any]:
        """
        Combines neural writing patterns with live fact-checking evidence
        under the explicit Decision Table.
        """
        # Edge cases
        if not isinstance(text, str) or not text.strip():
            return {
                "verdict": "UNCERTAIN",
                "confidence": 0.0,
                "rule_applied": "EMPTY_INPUT",
                "reason": "Input text is empty.",
                "classifier": None,
                "evidence": None
            }

        # 1. Run classifier
        clf_res = self.run_classifier(text)
        clf_verdict = clf_res["verdict"]
        clf_conf = clf_res["confidence"]

        # 2. Gather Evidence
        if evidence_override is not None:
            ev_res = evidence_override
        elif live_verify:
            try:
                ev_res = self.fact_checker.verify_claim(text)
            except Exception as e:
                ev_res = {
                    "status": "UNAVAILABLE",
                    "evidence": "UNAVAILABLE",
                    "credible_sources": [],
                    "debunk_articles": [],
                    "credible_sources_count": 0,
                    "debunk_articles_count": 0,
                    "summary": f"Live web checking unavailable: {str(e)}"
                }
        else:
            ev_res = {
                "status": "UNAVAILABLE",
                "evidence": "UNAVAILABLE",
                "credible_sources": [],
                "debunk_articles": [],
                "credible_sources_count": 0,
                "debunk_articles_count": 0,
                "summary": "Live web verification disabled."
            }

        ev_status = ev_res.get("evidence", ev_res.get("status", "UNAVAILABLE"))
        debunks = ev_res.get("debunk_articles", [])
        credibles = ev_res.get("credible_sources", [])

        # Normalize evidence status into one of 4 canonical states
        if debunks or ev_status in ["REFUTED", "DEBUNKED_FAKE"]:
            canon_ev = "REFUTED"
        elif len(credibles) >= 2 or ev_status in ["CORROBORATED", "VERIFIED_REAL"]:
            canon_ev = "CORROBORATED"
        elif ev_status in ["NO_EVIDENCE", "NO_COVERAGE_DETECTED", "INCONCLUSIVE", "PARTIALLY_CORROBORATED"]:
            canon_ev = "NO_EVIDENCE"
        else:
            canon_ev = "UNAVAILABLE"

        # 3. Explicit Decision Table
        # -------------------------------------------------------------
        # Rule 1: REFUTED by a fact-checker -> FAKE
        if canon_ev == "REFUTED":
            final_verdict = "FAKE"
            rule_applied = "REFUTED_BY_FACT_CHECKER"
            reason = f"Refuted by independent fact-checkers ({ev_res.get('summary', 'Debunked online')})."
            final_conf = max(clf_res["confidence_pct"], 96.0)

        # Rule 2: CORROBORATED and classifier not FAKE-with-high-confidence -> REAL
        elif canon_ev == "CORROBORATED" and not (clf_verdict == "FAKE" and clf_conf >= HIGH_CONF_FAKE_OVERRIDE):
            final_verdict = "REAL"
            rule_applied = "CORROBORATED_BY_CREDIBLE_NEWS"
            sources_str = ", ".join([s["source"] for s in credibles[:3]]) if credibles else "major agencies"
            reason = f"Corroborated by independent credible news reporting ({sources_str})."
            final_conf = max(clf_res["confidence_pct"], 95.0)

        # Rule 3: Classifier FAKE with calibrated confidence >= threshold -> FAKE
        elif clf_verdict == "FAKE" and clf_conf >= threshold:
            final_verdict = "FAKE"
            rule_applied = "CLASSIFIER_HIGH_CONFIDENCE_FAKE"
            reason = f"Linguistic writing patterns strongly indicate misinformation (calibrated confidence {clf_res['confidence_pct']}%)."
            final_conf = clf_res["confidence_pct"]

        # Rule 4: Classifier REAL but evidence NO_EVIDENCE or UNAVAILABLE for a checkable claim -> UNCERTAIN
        elif clf_verdict == "REAL" and canon_ev in ["NO_EVIDENCE", "UNAVAILABLE"]:
            final_verdict = "UNCERTAIN"
            rule_applied = "CLASSIFIER_REAL_UNSUPPORTED"
            reason = "Classifier leans real, but no independent source confirms it."
            final_conf = clf_res["confidence_pct"]

        # Rule 5: Everything else -> UNCERTAIN
        else:
            final_verdict = "UNCERTAIN"
            rule_applied = "INCONCLUSIVE_FALLBACK"
            reason = "Evidence and writing patterns are inconclusive or below decision threshold."
            final_conf = clf_res["confidence_pct"]

        return {
            "verdict": final_verdict,
            "confidence": round(final_conf, 1),
            "rule_applied": rule_applied,
            "reason": reason,
            "classifier": {
                "verdict": clf_verdict,
                "confidence": clf_res["confidence"],
                "confidence_pct": clf_res["confidence_pct"],
                "real_probability": clf_res["real_probability"],
                "fake_probability": clf_res["fake_probability"],
                "model_dir": self.model_dir
            },
            "evidence": {
                "status": canon_ev,
                "raw_status": ev_status,
                "credible_sources_count": len(credibles),
                "credible_sources": credibles[:4],
                "debunk_articles_count": len(debunks),
                "debunk_articles": debunks[:3],
                "summary": ev_res.get("summary", "")
            }
        }


def decide(
    text: str,
    model_dir: Optional[str] = None,
    threshold: float = DEFAULT_THRESHOLD,
    evidence_override: Optional[Dict[str, Any]] = None,
    live_verify: bool = True
) -> Dict[str, Any]:
    """Top-level functional interface for hybrid verdict engine."""
    engine = HybridDecisionEngine.get_instance(model_dir=model_dir or DEFAULT_MODEL_DIR)
    return engine.decide(
        text=text,
        threshold=threshold,
        evidence_override=evidence_override,
        live_verify=live_verify
    )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        text_arg = " ".join(sys.argv[1:])
        res = decide(text_arg)
        print("\n" + "="*50)
        print(f"VERDICT:      {res['verdict']} ({res['confidence']}%)")
        print(f"RULE:         {res['rule_applied']}")
        print(f"REASON:       {res['reason']}")
        print("="*50 + "\n")
    else:
        print("Usage: python src/verdict.py \"news headline or article text\"")
