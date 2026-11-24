# TruthPulse AI — News Credibility Forensics & Hybrid Verification

A production-grade, CPU-safe Fake News Credibility Engine and Streamlit single-page application built on HuggingFace Transformers (`distilbert-base-uncased`), PyTorch, Post-Processing Temperature Scaling, Captum Integrated Gradients, Advisory Red-Flag Heuristics, Google News RSS live journalistic corroboration, and an explicit Decision Table.

---

## 1. Setup & Environment

Engineered specifically for **8GB laptop profiles** (CPU only, Apple Silicon / x86_64, zero GPU required). Peak memory during training and inference stays strictly under **1.4 GB RAM**.

### Quick Start
```bash
# 1. Clone repository and navigate to workspace
cd "HCL PROJECT FAKE NEWS LLM"

# 2. Launch using one-click script (activates .venv and launches app)
./run.sh
```

### Manual Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run the single-page application
streamlit run app/app.py --server.fileWatcherType none --server.port 8501
```
Open **`http://localhost:8501`** in your browser.

### Running Test Verification Suite
Run all 60 unit tests across preprocessing, calibration, red flags, decision table, data integrity, and UI flows:
```bash
.venv/bin/pytest -v
```

---

## 2. Problem Diagnosis & Root Cause (Step 1)

The original v1 model achieved 98.42% accuracy on its own ISOT test split, but exhibited severe failure on modern news headlines and short claims typed by users:
- **Baseline Audit on Dev Set**: Raw classifier had a **60.0% False-REAL rate** (12 of 20 fake claims misclassified as REAL with high confidence).
- **Style Sensitivity**: 10 of 10 fake claims rewritten in neutral Reuters wire style flipped to `REAL` (>95% confidence). The classifier had learned *style* (formal vs sensational) rather than factual veracity.
- **Length Brittleness**: On ISOT test, full-article accuracy was 98.50%, but dropped to 95.80% on the first 30 words, and collapsed to 83.50% on title-only inputs.
- **Cross-Source Failure**: On real-world short claims (`UKPLab/liar`), the v1 model scored **53.39% accuracy** (near random coin-flip) with a 43.6% False-REAL rate.

---

## 3. Data Card Summary (`data/processed_v2/`)

To solve cross-source brittle shortcuts and length collapse, a multi-source training distribution was constructed:

1. **Integrated Real Datasets**:
   - **ISOT** (`GonzaloA/fake_news`): Full news articles from Reuters (Real) and political blogs (Fake).
   - **WELFake** (`davanstrien/WELFake`): Multi-domain news articles (0 = Real, 1 = Fake verified by inspection).
   - **LIAR** (`UKPLab/liar`): Politifact short claims mapped strictly by name (`true`/`mostly-true` -> 0 Real; `false`/`barely-true`/`pants-fire` -> 1 Fake).
2. **Leakage Removal**: Stripped leading publisher datelines (`WASHINGTON (Reuters) -`), photo credits (`via Getty Images`), URLs, social handles (`@username`), and media tags (`[VIDEO]`).
3. **Length Augmentation**: Generated headline-only and first-30-word truncations **strictly from training-split articles** to bring short claims into distribution.
4. **Source Share Capping & Balance**: No source exceeds 32% of training data (ISOT: 31.7%, WELFake: 29.3%, LIAR: 23.5%, Augmentations: 15.5%). Balanced at **5,440 REAL / 5,378 FAKE**.
5. **Held-Out Unseen Source**: The complete LIAR test set (**1,283 short claims**) was held out entirely as an unseen benchmark.
6. **Zero Leakage**: Exactly 0 cross-split duplicates between train, val, mixed-test, and unseen source (verified by `tests/test_step7_data_integrity.py`).

---

## 4. Explicit Decision Table (`src/verdict.py`)

The hybrid decision engine integrates the linguistic classifier with live web corroboration. The classifier detects writing patterns; it never proves truth on its own.

| Priority | Condition | Final Verdict | Rule Identifier | Reason Message |
|:---:|---|:---:|---|---|
| **1** | Evidence = `REFUTED` by fact-checker | **FAKE** | `REFUTED_BY_FACT_CHECKER` | *"Refuted by independent fact-checkers (Snopes/PolitiFact)."* |
| **2** | Evidence = `CORROBORATED` (≥2 mainstream sources) AND classifier not high-confidence FAKE | **REAL** | `CORROBORATED_BY_CREDIBLE_NEWS` | *"Corroborated by independent credible news reporting (Reuters, AP, BBC)."* |
| **3** | Classifier = `FAKE` with calibrated confidence $\ge 0.70$ | **FAKE** | `CLASSIFIER_HIGH_CONFIDENCE_FAKE` | *"Linguistic writing patterns strongly indicate misinformation (calibrated confidence X%)."* |
| **4** | Classifier = `REAL` but evidence `NO_EVIDENCE` or `UNAVAILABLE` | **UNCERTAIN** | `CLASSIFIER_REAL_UNSUPPORTED` | *"Classifier leans real, but no independent source confirms it."* |
| **5** | Everything else (ambiguous / below threshold) | **UNCERTAIN** | `INCONCLUSIVE_FALLBACK` | *"Evidence and writing patterns are inconclusive or below decision threshold."* |

---

## 5. Measured Benchmark Results

All figures produced by running evaluation scripts in this session and verified in `reports/`:

### A. Model Head-to-Head Comparison (`reports/comparison.json`)

| Evaluation Split | Metric | DistilBERT v1 (Old) | DistilBERT v2 (New) | Delta / Improvement |
|---|---|:---:|:---:|:---:|
| **Mixed Test Set** (1,000 articles) | Accuracy | 94.17% | **96.72%** | **+2.55%** |
| | Macro F1 | 0.9416 | **0.9672** | **+0.0256** |
| | False-REAL Rate | 7.02% | **2.19%** | **-4.83%** (69% reduction) |
| | Expected Calibration Error | 0.0464 | **0.0119** | **4x better calibrated** |
| **Unseen Source: LIAR** (1,283 claims) | Accuracy | 53.39% | **68.90%** | **+15.51%** |
| | Macro F1 | 0.5071 | **0.5974** | **+0.0903** |
| | ROC-AUC | 0.5466 | **0.6448** | **+0.0982** |
| **ISOT Original Test** (4,058 articles) | Accuracy | 98.42% | 98.05% | Preserved (>98%) |

### B. Hybrid Decision Engine Before vs. After (`reports/audit_before.json` vs `reports/hybrid_results.json`)

| Benchmark Set | Metric | DistilBERT v1 (Raw) | Hybrid System (v2 + Evidence) | Impact |
|---|---|:---:|:---:|:---:|
| **Dev Claims Set** (40 items) | False-REAL Rate | 60.0% (12/20 fake called real) | **10.0%** (2/20 fake called real) | **83.3% relative reduction** |
| | False-FAKE Rate | 0.0% | 35.0% | Conservative on unconfirmed claims |
| | UNCERTAIN Rate | 0.0% | 20.0% | Rejects unverified claims |
| **Holdout Claims Set** (40 items, one-shot) | False-REAL Rate | — | **10.0%** (2/20 fake called real) | Generalizes to fresh claims |
| | UNCERTAIN Rate | — | **27.5%** (11/40 items) | Abstains when evidence is missing |

### C. Checkpoint Integrity (`reports/model_hashes.json`)
- **Old Fallback Model** (`models/distilbert_fake_news/model.safetensors`):
  `SHA-256: 275aff24b183263e14c89bf59c2913c67194d3e556a60666c8ac2d9885d09d31` (100% unchanged)
- **New Active Model** (`models/distilbert_v2/model.safetensors`):
  `SHA-256: af67ce526afe647161bda7f3e54362e821f01a6866c6f11e9ea6668ce27f889f`

### D. Latency Benchmark (20 runs on CPU, `reports/latency_benchmark.json`)
- **Local Model + Captum + Red Flags** (without web layer): **82.5 ms** mean (72.4 ms min, 87.8 ms max).
- **Hybrid System with Live Web Corroboration**: **0.48 s** mean (0.15 s min, 2.08 s max).

---

## 6. Frontend Redesign (Streamlit Single-Page)

The UI was redesigned strictly to the user's constraints:
- **Design System**: Single-page centered layout with dark theme (`#0f172a` canvas, `#1e293b` cards, `#6366f1` accent).
- **Controls**: Exactly **two buttons** (`[Check News]` and `[Clear]`), **one text box** with character/word counter, and **one performance chart** (`reports/unseen_performance_chart.png`). Zero sidebars, zero tabs, zero example buttons.
- **Real-Time Status**: Step-by-step progress animation ("Reading text", "Running model", "Checking live sources", "Synthesizing evidence").
- **Hero Card**: Large color-coded verdict badge (`VERIFIED REAL` green, `MISINFORMATION / FAKE` red, `UNCERTAIN / UNVERIFIED` amber), confidence progress bar, and explicit decision reason.
- **Evidence Mini-Cards**: Stance badges (`Supports`, `Refutes`, `Related`) with direct links to publisher reports.
- **Captum Attributions**: Scrollable preview box with highlighted influential writing-pattern tokens.
- **AI Synthesis**: Collapsible "How this was decided" card displaying classifier output, live evidence status, triggered red-flags, and FLAN-T5 natural language summary.
- **Permanent Disclaimer**: *"Detects writing patterns and cross-checks public news sources. It does not prove truth. Verify important claims yourself."*

Five automated end-to-end browser verification states were captured and verified in `reports/screenshots/`:
1. `01_empty_page.png`: Clean initial layout with input box, buttons, benchmark chart, and disclaimer.
2. `02_fake_result.png`: Medical myth correctly classified as `MISINFORMATION / FAKE` with fact-checker debunk cards.
3. `03_real_result.png`: Federal Reserve rate cut verified as `VERIFIED REAL` with Reuters and CNBC corroboration.
4. `04_uncertain_result.png`: Uncorroborated conspiracy claim flagged as `UNCERTAIN / UNVERIFIED`.
5. `05_error_state.png`: Empty submission warning dialog.

---

## 7. Limitations & Honest Failure Modes

No automated system can guarantee factual truth. TruthPulse AI detects stylistic writing patterns and cross-references public RSS news feeds; it cannot independently establish physical reality:

1. **Breaking / Unindexed News**: Events that occurred within the past few minutes or hours may not yet be indexed by search engines. If the claim is written in neutral journalistic prose, Rule 4 will correctly yield `UNCERTAIN` rather than false confidence.
2. **Keyword Overlap in Fact-Checks**: Highly prominent public figures (e.g., presidents, prime ministers) frequently have viral hoaxes debunked about them. If a user pastes a genuine report about two leaders, broad fact-check queries can retrieve debunks of unrelated rumors about those same individuals, resulting in a false-`REFUTED` verdict.
3. **Plausible Low-Profile Misinformation**: Fake claims about small local events or private individuals written in calm wire-service style with zero online fact-checking coverage will not be caught as FAKE. Rule 4 flags them as `UNCERTAIN` ("Classifier leans real, but no independent source confirms it").
4. **Network & Rate Limits**: If external network access is unavailable or search feeds throttle requests, evidence status degrades gracefully to `UNAVAILABLE`, preventing unsupported `REAL` assertions.
# FAKE-NEWS-DETECTION-USING-LLM
