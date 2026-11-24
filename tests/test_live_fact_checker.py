"""
tests/test_live_fact_checker.py — Unit tests for Live Web Fact-Checking & Verification Layer.
"""
import pytest
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.fact_checker import LiveNewsFactChecker, verify_news_live
from src.explain import explain


@pytest.fixture(scope="module")
def checker():
    return LiveNewsFactChecker.get_instance()


def test_extract_search_query(checker):
    text = "BREAKING: Scientists at NASA have observed deep subterranean water geysers on Europa!"
    query = checker.extract_search_query(text)
    assert len(query) > 5
    assert "Scientists" in query or "NASA" in query or "Europa" in query
    assert "BREAKING" not in query.split()


def test_empty_query_handling(checker):
    res = checker.verify_claim("")
    assert res["status"] == "UNAVAILABLE"
    assert res["credible_sources_count"] == 0


def test_live_fact_checker_debunk_query(checker):
    # Known medical myth query
    claim = "Doctors confirm drinking lemon water with salt permanently cures all forms of cancer without chemotherapy"
    res = checker.verify_claim(claim)
    assert res["status"] in ["REFUTED", "DEBUNKED_FAKE", "INCONCLUSIVE", "NO_COVERAGE_DETECTED"]
    assert "summary" in res


def test_explain_with_live_verification():
    claim = "Doctors confirm drinking lemon water with salt permanently cures all forms of cancer without chemotherapy"
    res = explain(claim, live_verify=True)
    assert res["status"] == "success"
    assert "live_verification" in res
    assert res["verdict"] in ["FAKE", "UNCERTAIN"]


def test_explain_without_live_verification():
    claim = "The Federal Reserve held interest rates steady during this week's monetary policy session."
    res = explain(claim, live_verify=False)
    assert res["status"] == "success"
    assert res["live_verification"] is None
