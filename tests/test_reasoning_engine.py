"""
test_reasoning_engine.py — Unit tests for linguistic reasoning engine and signal extraction.
"""
import os
import sys
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.reasoning_engine import (
    extract_linguistic_signals,
    build_reasoning,
    SENSATIONALIST_TERMS,
    HEDGE_TERMS
)


def test_sensationalist_signal_detection():
    text = "SHOCKING BOMBSHELL: Government exposed in massive fraud coverup!"
    sigs = extract_linguistic_signals(text)
    assert len(sigs["sensationalist_words"]) >= 2
    assert "shocking" in sigs["sensationalist_words"] or "bombshell" in sigs["sensationalist_words"]
    assert sigs["exclamation_marks"] == 1
    assert sigs["caps_ratio"] > 0.15


def test_hedging_signal_detection():
    text = "Anonymous insiders reportedly claim that officials allegedly held secret talks."
    sigs = extract_linguistic_signals(text)
    assert "reportedly" in sigs["hedge_words"] or "allegedly" in sigs["hedge_words"]


def test_credibility_signal_detection():
    text = "According to researchers at Johns Hopkins, studies show positive peer-reviewed clinical data."
    sigs = extract_linguistic_signals(text)
    assert len(sigs["credibility_signals"]) >= 1


def test_build_reasoning_output_structure():
    text = "Studies show verified official statistics from university of Chicago."
    reasoning = build_reasoning(
        text=text,
        verdict="REAL / CREDIBLE",
        fake_prob=0.15,
        real_prob=0.85,
        confidence=85.0
    )
    assert isinstance(reasoning, str)
    assert "CREDIBLE" in reasoning
    assert "Disclaimer" in reasoning
