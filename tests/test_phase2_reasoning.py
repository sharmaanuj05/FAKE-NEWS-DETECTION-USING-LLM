"""
tests/test_phase2_reasoning.py — Unit tests for Phase 2 Reasoning Layer.

Tests:
1. Red-flag rules positive & negative cases.
2. explain() output structure and keys.
3. UNCERTAIN threshold behavior (below threshold yields UNCERTAIN).
4. Token merging (subword tokens merged into complete words, special tokens removed).
"""
import pytest
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.red_flags import (
    check_excessive_caps,
    check_clickbait_phrases,
    check_excessive_punctuation,
    check_emotional_absolutist,
    check_no_named_attribution,
    detect_red_flags
)
from src.explain import explain, ExplainerPipeline


# ─── 1. Red-Flag Rules Positive & Negative Tests ─────────────────────────────
def test_excessive_caps_positive():
    text = "ALERT: MASSIVE CORRUPTION EXPOSED IN THE SENATE TODAY!"
    res = check_excessive_caps(text)
    assert res is not None
    assert res["name"] == "Excessive Capitalization"


def test_excessive_caps_negative():
    text = "The Federal Reserve and NASA announced regular budgetary projections today."
    res = check_excessive_caps(text)
    assert res is None  # Allowed acronyms do not trigger rule


def test_clickbait_phrases_positive():
    text = "You won't believe what they discovered inside the underground bunker."
    res = check_clickbait_phrases(text)
    assert res is not None
    assert res["name"] == "Clickbait Phrasing"


def test_clickbait_phrases_negative():
    text = "The committee held a hearing on Wednesday regarding environmental standards."
    res = check_clickbait_phrases(text)
    assert res is None


def test_excessive_punctuation_positive():
    text = "Could this be the end of civilization?!?! Check this out immediately!!!"
    res = check_excessive_punctuation(text)
    assert res is not None
    assert res["name"] == "Excessive Punctuation"


def test_excessive_punctuation_negative():
    text = "The Prime Minister met with European delegates in Paris on Monday."
    res = check_excessive_punctuation(text)
    assert res is None


def test_emotional_absolutist_positive():
    text = "This is 100% proven evidence of an evil cabal destroying the country."
    res = check_emotional_absolutist(text)
    assert res is not None
    assert res["name"] == "Absolutist / Emotional Framing"


def test_emotional_absolutist_negative():
    text = "Economic indicators show modest GDP growth in the second quarter."
    res = check_emotional_absolutist(text)
    assert res is None


def test_no_named_attribution_positive():
    # Long text (>=40 words) with no quotes or attribution verbs
    text = (
        "The mysterious energy field expanding across the northern mountains has baffled "
        "local observers for several weeks now as anomalous weather patterns persist without "
        "any clear explanation from meteorological stations stationed throughout the rural province. "
        "Residents continue reporting strange lights hovering near the tree lines at dusk."
    )
    res = check_no_named_attribution(text)
    assert res is not None
    assert res["name"] == "Missing Source Attribution"


def test_no_named_attribution_negative():
    text = (
        "The economic summit concluded today in Geneva. Treasury Secretary Miller said in an official "
        "press briefing that bilateral negotiations would continue next month according to department representatives."
    )
    res = check_no_named_attribution(text)
    assert res is None  # Contains 'said', 'according to'


# ─── 2. explain() Output Structure Tests ────────────────────────────────────
def test_explain_output_structure():
    text = "NASA's James Webb Space Telescope detected atmospheric water vapor on an exoplanet."
    res = explain(text)

    assert res["status"] == "success"
    assert "verdict" in res
    assert res["verdict"] in ["REAL", "FAKE", "UNCERTAIN"]
    assert "confidence" in res
    assert isinstance(res["confidence"], float)
    assert "top_words" in res
    assert isinstance(res["top_words"], list)
    assert "triggered_flags" in res
    assert isinstance(res["triggered_flags"], list)
    assert "summary" in res
    assert "reflects linguistic" in res["summary"].lower()


def test_explain_empty_input():
    res = explain("")
    assert res["status"] == "error"
    assert "Input text is empty" in res["message"]


def test_explain_short_input():
    res = explain("Hello world")
    assert res["status"] == "error"
    assert "less than 3 words" in res["message"]


# ─── 3. UNCERTAIN Threshold Behavior Test ────────────────────────────────────
def test_uncertain_threshold_behavior():
    pipeline = ExplainerPipeline.get_instance()
    # Use text that classifier leans FAKE on to test the confidence threshold cutoff
    text = "SHOCKING BOMBSHELL: Secret military tribunals have executed dozens of officials overnight!"

    # When threshold is set artificially high (0.999), confidence fails threshold yielding UNCERTAIN
    res_high = pipeline.explain(text, threshold=0.999, live_verify=False)
    assert res_high["verdict"] == "UNCERTAIN"

    # When threshold is set low (0.50), confidence meets threshold yielding FAKE
    res_low = pipeline.explain(text, threshold=0.50, live_verify=False)
    assert res_low["verdict"] == "FAKE"


# ─── 4. Token Merging Test ──────────────────────────────────────────────────
def test_token_merging_in_attribution():
    pipeline = ExplainerPipeline.get_instance()
    # "electrocardiogram" or "nanotechnology" split into multiple subwords in BERT vocabulary
    text = "Medical researchers utilized nanotechnology and advanced electrocardiogram monitoring in clinical trials."
    attrs = pipeline.attr_explainer.attribute(text, target_class=0, top_k=5)

    assert len(attrs) > 0
    for item in attrs:
        # Verify subword prefix '##' is never returned as a standalone word
        assert not item["word"].startswith("##")
        assert len(item["word"]) > 0
