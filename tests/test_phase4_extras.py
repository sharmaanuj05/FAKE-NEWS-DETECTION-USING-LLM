"""
tests/test_phase4_extras.py — Unit tests for Phase 4 Extras:
1. FLAN-T5 Natural Language Explanation Synthesis.
2. Model Comparison (DistilBERT vs. TF-IDF Baseline).
"""
import pytest
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.explain import explain
from src.llm_explainer import FlanT5Explainer, generate_llm_explanation
from src.model_comparison import ModelComparator, compare_models


@pytest.fixture(scope="module")
def flan_explainer():
    return FlanT5Explainer.get_instance()


@pytest.fixture(scope="module")
def comparator():
    return ModelComparator.get_instance()


def test_flan_t5_synthesis_fake(flan_explainer):
    words = [{"word": "bombshell", "score": 0.15}, {"word": "secret", "score": 0.11}]
    flags = [{"name": "Clickbait Phrasing", "reason": "Sensational terms"}]
    explanation = flan_explainer.synthesize_explanation("FAKE", 99.6, words, flags)

    assert isinstance(explanation, str)
    assert len(explanation) > 20
    assert "FLAN-T5" in explanation
    assert "Reflects linguistic writing style, not verified facts" in explanation


def test_flan_t5_synthesis_real(flan_explainer):
    words = [{"word": "telescope", "score": 0.86}, {"word": "detected", "score": 0.56}]
    flags = []
    explanation = flan_explainer.synthesize_explanation("REAL", 95.0, words, flags)

    assert isinstance(explanation, str)
    assert len(explanation) > 20
    assert "FLAN-T5" in explanation


def test_flan_t5_synthesis_uncertain(flan_explainer):
    words = [{"word": "hollywood", "score": 0.40}]
    flags = []
    explanation = flan_explainer.synthesize_explanation("UNCERTAIN", 57.1, words, flags)

    assert isinstance(explanation, str)
    assert len(explanation) > 20
    assert "FLAN-T5" in explanation


def test_model_comparison_execution(comparator):
    text = "NASA James Webb Space Telescope detected atmospheric water vapor on an exoplanet."
    distil_res = explain(text)
    comp = comparator.compare(text, distil_res)

    assert "baseline" in comp
    assert "distilbert" in comp
    assert "models_agree" in comp
    assert isinstance(comp["models_agree"], bool)
    assert "consensus_message" in comp
    assert "benchmark_comparison" in comp
    assert comp["baseline"]["verdict"] in ["REAL", "FAKE"]
    assert comp["distilbert"]["verdict"] in ["REAL", "FAKE", "UNCERTAIN"]


def test_model_comparison_benchmark_table(comparator):
    text = "BOMBSHELL: Secret tribunals have executed top leaders!"
    distil_res = explain(text)
    comp = comparator.compare(text, distil_res)
    benchmarks = comp["benchmark_comparison"]

    assert len(benchmarks) >= 5
    feature_names = [row["Feature / Metric"] for row in benchmarks]
    assert "Architecture" in feature_names
    assert "Holdout Test Accuracy" in feature_names
    assert "Holdout Test F1 Score" in feature_names


def test_model_comparison_empty_text(comparator):
    res = comparator.predict_baseline("")
    assert res["verdict"] == "UNAVAILABLE"


def test_streamlit_app_check_news_flow():
    """Verifies that clicking Check News produces the verdict card with NO white screen."""
    from streamlit.testing.v1 import AppTest
    app_path = os.path.join(BASE_DIR, "app.py")
    at = AppTest.from_file(app_path, default_timeout=30)
    at.run()
    assert len(at.text_area) == 1
    assert len(at.button) == 2

    # Simulate entering headline and clicking Check News
    at.text_area[0].input("BOMBSHELL: Secret military tribunals have executed dozens of officials overnight!").run()
    at.button[0].click().run()

    # Must render verdict card and expanded content without blanking
    assert any("FAKE" in m.value for m in at.markdown)
    assert len(at.expander) >= 1


def test_streamlit_app_clear_flow():
    """Verifies that clicking Clear properly resets the text area and removes the verdict card."""
    from streamlit.testing.v1 import AppTest
    app_path = os.path.join(BASE_DIR, "app.py")
    at = AppTest.from_file(app_path, default_timeout=30)
    at.run()
    at.text_area[0].input("Sample news text").run()
    at.button[0].click().run()

    # Click Clear button
    at.button[1].click().run()
    assert at.text_area[0].value == ""
    assert not any("FAKE" in m.value for m in at.markdown)
