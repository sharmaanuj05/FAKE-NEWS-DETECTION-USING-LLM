"""
tests/test_robustness.py — Comprehensive Robustness, Edge Case, and Fuzzing Tests.

Validates that the explain() pipeline and reasoning components never crash
when presented with non-standard, malformed, malicious, or edge-case inputs.
"""
import pytest
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.explain import explain, ExplainerPipeline


@pytest.fixture(scope="module")
def pipeline():
    return ExplainerPipeline.get_instance()


def test_none_input():
    res = explain(None)
    assert res["status"] == "error"
    assert "Input text is empty" in res["message"]


def test_empty_string():
    res = explain("")
    assert res["status"] == "error"
    assert "Input text is empty" in res["message"]


def test_whitespace_string():
    res = explain("   \n\t\r  ")
    assert res["status"] == "error"
    assert "Input text is empty" in res["message"]


def test_pure_url_input():
    url = "https://www.reuters.com/business/aerospace-defense/boeing-deliveries-rise-2026-09-30/"
    res = explain(url)
    assert res["status"] == "error"
    assert "URL link rather than article text" in res["message"]


def test_pure_emojis():
    res = explain("🚨📢🔥💣💥⚡️👀")
    assert res["status"] == "error"
    assert "at least 3 recognizable words" in res["message"]


def test_pure_numbers():
    res = explain("1984 2026 4058 32460 9828")
    assert res["status"] == "error"
    assert "at least 3 recognizable words" in res["message"]


def test_short_two_words():
    res = explain("Breaking News")
    assert res["status"] == "error"
    assert "at least 3 recognizable words" in res["message"]


def test_html_and_script_injection():
    text = "<b>BREAKING</b>: Scientists discovered water vapor on exoplanet. <script>alert('xss')</script>"
    res = explain(text)
    assert res["status"] == "success"
    assert res["verdict"] in ["REAL", "FAKE", "UNCERTAIN"]
    assert "summary" in res


def test_very_large_text():
    # 60,000 characters (exceeds typical article length)
    large_text = "The Federal Reserve released its quarterly economic outlook report today. " * 800
    assert len(large_text) > 50000
    res = explain(large_text)
    assert res["status"] == "success"
    assert res["verdict"] in ["REAL", "FAKE", "UNCERTAIN"]


def test_unicode_and_foreign_script():
    # French news excerpt
    french_text = "Le gouvernement français a approuvé de nouvelles réformes économiques aujourd'hui après d'intenses débats au parlement."
    res = explain(french_text)
    assert res["status"] == "success"
    assert res["verdict"] in ["REAL", "FAKE", "UNCERTAIN"]


def test_special_characters_and_symbols():
    text = "@@@ $$$ $$$ President announced new international trade agreement with European partners today ### ***"
    res = explain(text)
    assert res["status"] == "success"
    assert res["verdict"] in ["REAL", "FAKE", "UNCERTAIN"]


def test_deterministic_reentrancy():
    # Ensure repeated invocations with the same text produce identical outputs
    sample = "The Department of Transportation awarded infrastructure grants to modernize railway bridges."
    res1 = explain(sample)
    res2 = explain(sample)
    assert res1["verdict"] == res2["verdict"]
    assert res1["confidence"] == res2["confidence"]
    assert [w["word"] for w in res1["top_words"]] == [w["word"] for w in res2["top_words"]]
