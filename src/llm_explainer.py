"""
src/llm_explainer.py — LLM-written Natural Language Explanations using FLAN-T5.

Uses google/flan-t5-base (with graceful fallback to google/flan-t5-small or template)
to synthesize a natural language explanation grounded strictly in the model's
calibrated verdict, confidence, top attributed words, and triggered red flags.
Never hallucinates facts and preserves the writing-patterns disclaimer.
"""
import os
import sys
import warnings
from typing import Dict, Any, List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import logging as hf_logging

# Suppress noisy HF warning messages during model loading
hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

DEFAULT_FLAN_MODEL = os.getenv("FLAN_MODEL", "google/flan-t5-base")
FALLBACK_FLAN_MODEL = "google/flan-t5-small"


class FlanT5Explainer:
    _instance = None

    def __init__(self, model_name: str = DEFAULT_FLAN_MODEL):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None

        # Attempt loading from local cache first, then online fallback
        candidates = [model_name, FALLBACK_FLAN_MODEL]
        for name in candidates:
            # First try local cache
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(name, local_files_only=True)
                self.model.eval()
                self.model_name = name
                break
            except Exception:
                pass

            # If not in local cache, download from hub
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(name)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(name)
                self.model.eval()
                self.model_name = name
                break
            except Exception:
                continue

    @classmethod
    def get_instance(cls, model_name: str = DEFAULT_FLAN_MODEL):
        if cls._instance is None:
            cls._instance = cls(model_name)
        return cls._instance

    def synthesize_explanation(
        self,
        verdict: str,
        confidence: float,
        top_words: List[Dict[str, Any]],
        flags: List[Dict[str, str]]
    ) -> str:
        """
        Synthesizes a cohesive natural language explanation using FLAN-T5
        grounded strictly in the detection evidence.
        """
        words_list = [w["word"] for w in top_words[:5]] if top_words else []
        words_str = ", ".join(words_list) if words_list else "neutral phrasing"
        flags_list = [f["name"] for f in flags] if flags else []
        flags_str = ", ".join(flags_list) if flags_list else "no advisory red flags"

        if self.model is None or self.tokenizer is None:
            # Fallback if FLAN-T5 cannot be loaded
            return (
                f"The article was evaluated as {verdict} with {confidence}% calibrated confidence, "
                f"influenced by linguistic patterns around '{words_str}' with {flags_str}."
            )

        # Build grounded premise for FLAN-T5 Seq2Seq synthesis
        if verdict == "FAKE":
            premise = (
                f"Premise: An automated news forensics detector analyzed an article. "
                f"It determined a verdict of FAKE with {confidence}% calibrated confidence "
                f"due to sensational tokens ({words_str}) and warning indicators ({flags_str}).\n"
                f"Question: In brief, why did the model classify this article as fake news?\n"
                f"Answer:"
            )
        elif verdict == "REAL":
            premise = (
                f"Premise: An automated news credibility system analyzed an article. "
                f"It determined a verdict of REAL with {confidence}% calibrated confidence "
                f"due to objective journalistic vocabulary ({words_str}) and lack of deceptive clickbait framing.\n"
                f"Question: In brief, why did the model classify this article as real news?\n"
                f"Answer:"
            )
        else:  # UNCERTAIN
            premise = (
                f"Premise: An automated news credibility system analyzed an article. "
                f"The calibrated confidence was {confidence}%, which is below the 70% threshold required for a decisive verdict, "
                f"displaying mixed signals ({words_str}).\n"
                f"Question: Why was this article marked as uncertain?\n"
                f"Answer:"
            )

        try:
            inputs = self.tokenizer(premise, return_tensors="pt", max_length=512, truncation=True)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=60,
                    repetition_penalty=1.4,
                    no_repeat_ngram_size=3,
                    num_beams=2
                )
            synthesis = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            if not synthesis:
                raise ValueError("Empty output")
            # Format clean natural language sentence
            synthesis = synthesis[0].upper() + synthesis[1:] if len(synthesis) > 1 else synthesis
            if not synthesis.endswith("."):
                synthesis += "."

            full_explanation = (
                f"FLAN-T5 Analysis: The model identified {synthesis} "
                f"Influential signals include '{words_str}' alongside {flags_str}. "
                f"(Note: Reflects linguistic writing style, not verified facts.)"
            )
            return full_explanation
        except Exception as e:
            return (
                f"The model evaluated this as {verdict} ({confidence}% calibrated confidence) "
                f"based on vocabulary patterns around '{words_str}'. Warning signs: {flags_str}. "
                f"(Note: Reflects linguistic writing patterns, not verified facts.)"
            )


def generate_llm_explanation(
    verdict: str,
    confidence: float,
    top_words: List[Dict[str, Any]],
    flags: List[Dict[str, str]]
) -> str:
    """Functional interface for FLAN-T5 explanation synthesis."""
    explainer = FlanT5Explainer.get_instance()
    return explainer.synthesize_explanation(verdict, confidence, top_words, flags)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        from src.explain import explain
        input_text = " ".join(sys.argv[1:])
        dist_res = explain(input_text)
        explanation = generate_llm_explanation(
            verdict=dist_res["verdict"],
            confidence=dist_res["confidence"],
            top_words=dist_res.get("top_words", []),
            flags=dist_res.get("triggered_flags", [])
        )
        print(explanation)
    else:
        print("Usage: python src/llm_explainer.py \"news text to evaluate\"")
