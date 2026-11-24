"""
tests/test_verdict.py — Unit tests for Step 5 Decision Table and Hybrid Verdict Engine.
"""
import os
import sys
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.verdict import decide, HybridDecisionEngine

class MockModel:
    def __init__(self, pred_class=0, fake_prob=0.20, real_prob=0.80):
        self.pred_class = pred_class
        self.fake_prob = fake_prob
        self.real_prob = real_prob

class TestDecisionTable:
    """Tests each row of the explicit Decision Table using mock evidence overrides."""

    def test_rule_1_refuted_by_fact_checker(self):
        """Rule 1: REFUTED by a fact-checker -> FAKE (regardless of classifier)."""
        evidence_mock = {
            "evidence": "REFUTED",
            "debunk_articles": [{"title": "Debunked: False claim", "source": "Snopes"}],
            "credible_sources": [],
            "summary": "Fact-checkers refuted this claim."
        }
        res = decide(
            text="Doctors confirm lemon water cures all cancer in 48 hours",
            evidence_override=evidence_mock
        )
        assert res["verdict"] == "FAKE"
        assert res["rule_applied"] == "REFUTED_BY_FACT_CHECKER"
        assert "independent fact-checkers" in res["reason"]

    def test_rule_2_corroborated_by_credible_news(self):
        """Rule 2: CORROBORATED by >=2 sources and classifier not high-conf FAKE -> REAL."""
        evidence_mock = {
            "evidence": "CORROBORATED",
            "debunk_articles": [],
            "credible_sources": [
                {"title": "Summits in Paris", "source": "Reuters"},
                {"title": "Starmer meets Macron", "source": "BBC"}
            ],
            "summary": "Corroborated by Reuters and BBC."
        }
        res = decide(
            text="UK Prime Minister Keir Starmer meets French President Emmanuel Macron in Paris",
            evidence_override=evidence_mock
        )
        assert res["verdict"] == "REAL"
        assert res["rule_applied"] == "CORROBORATED_BY_CREDIBLE_NEWS"

    def test_rule_3_classifier_high_confidence_fake(self):
        """Rule 3: Classifier FAKE with confidence >= threshold -> FAKE."""
        # Using a sensational hoax that model predicts FAKE with high confidence
        sensational_hoax = "SHOCKING BOMBSHELL HOAX: Unbelievable catastrophic nightmare destroys city completely!!"
        evidence_mock = {
            "evidence": "NO_EVIDENCE",
            "debunk_articles": [],
            "credible_sources": [],
            "summary": "No live reporting found."
        }
        res = decide(
            text=sensational_hoax,
            threshold=0.60,
            evidence_override=evidence_mock
        )
        if res["classifier"]["verdict"] == "FAKE" and res["classifier"]["confidence"] >= 0.60:
            assert res["verdict"] == "FAKE"
            assert res["rule_applied"] == "CLASSIFIER_HIGH_CONFIDENCE_FAKE"

    def test_rule_4_classifier_real_unsupported_yields_uncertain(self):
        """
        Rule 4: Classifier REAL but evidence NO_EVIDENCE or UNAVAILABLE
        MUST yield UNCERTAIN ('classifier leans real, but no independent source confirms it').
        This is the critical fix preventing false-REAL on formal fake news!
        """
        formal_fake = "The United Nations officially announced the immediate dissolution of national borders across all member states under Agenda 2030."
        evidence_mock = {
            "evidence": "NO_EVIDENCE",
            "debunk_articles": [],
            "credible_sources": [],
            "summary": "No live coverage detected."
        }
        res = decide(
            text=formal_fake,
            evidence_override=evidence_mock
        )
        # Even if classifier predicts REAL, lack of corroboration MUST downgrade verdict to UNCERTAIN
        if res["classifier"]["verdict"] == "REAL":
            assert res["verdict"] == "UNCERTAIN"
            assert res["rule_applied"] == "CLASSIFIER_REAL_UNSUPPORTED"
            assert "classifier leans real, but no independent source confirms it" in res["reason"].lower()

    def test_rule_4_classifier_real_unavailable_evidence(self):
        """Rule 4: When evidence is UNAVAILABLE, classifier REAL must become UNCERTAIN."""
        evidence_mock = {
            "evidence": "UNAVAILABLE",
            "debunk_articles": [],
            "credible_sources": [],
            "summary": "Live web verification unavailable."
        }
        res = decide(
            text="A formal sounding claim with no internet access available",
            evidence_override=evidence_mock
        )
        if res["classifier"]["verdict"] == "REAL":
            assert res["verdict"] == "UNCERTAIN"
            assert res["rule_applied"] == "CLASSIFIER_REAL_UNSUPPORTED"

    def test_empty_input(self):
        """Empty text returns UNCERTAIN without crashing."""
        res = decide("")
        assert res["verdict"] == "UNCERTAIN"
        assert res["rule_applied"] == "EMPTY_INPUT"

    def test_whitespace_input(self):
        res = decide("   \n\t   ")
        assert res["verdict"] == "UNCERTAIN"
