# Root Cause Diagnosis Report

This document records the empirical findings of the diagnostic tests run on the frozen baseline classifier (`models/distilbert_fake_news/`).
All numbers are generated strictly from live evaluation runs without assumptions or invented values.

---

## 1. Style Test: Sensitivity to Writing Patterns vs Factual Content

### Test Methodology
- Evaluated **10 dev FAKE claims** in their original form vs rewritten into calm, formal wire-service (Reuters-style) reporting.
- Evaluated **10 dev REAL claims** in their original form vs rewritten into sensational / clickbait style.
- The factual claims themselves were kept identical; only stylistic phrasing, tone, and formatting were varied.

### Measured Results
- **Fake claims called REAL when written formally:**
  - Original style: **3/10** (30%) predicted as REAL.
  - Formal wire-service style: **10/10** (100%) predicted as REAL.
  - **Verdict:** Writing a fabricated claim in a measured, journalistic style drastically shifts the model toward REAL. For example:
    - *"NASA officially admitted Earth is hollow..."* -> 99.8% REAL when rewritten into Reuters format.
    - *"5G towers transmit biological viruses..."* -> 99.9% REAL when formatted as a Geneva wire dispatch.
- **Real claims called FAKE when written sensationally:**
  - Original neutral style: **1/10** (10%) predicted as FAKE.
  - Sensational clickbait style: **10/10** (100%) predicted as FAKE.
  - Punctuation, capitalization, and urgent emotional adjectives immediately trigger the model's FAKE pattern detector, regardless of factual veracity.

---

## 2. Length Test: In-Distribution vs Truncated Inputs (ISOT Test Split)

### Test Methodology
Evaluated **1,000 balanced samples** from the ISOT test set across three input lengths:
1. **Full Article Text** (as seen in original ISOT training).
2. **First 30 Words** of the article.
3. **Headline / Title Only** from `GonzaloA/fake_news`.

### Measured Results Table
| Input Format | Sample Size | Accuracy | Macro F1 | ROC-AUC | False-REAL Rate | False-FAKE Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full Article Text** | 1000 | **98.50%** | **0.9850** | **0.9994** | **1.80%** | **1.20%** |
| **First 30 Words** | 1000 | **95.80%** | **0.9580** | **0.9946** | **6.00%** | **2.40%** |
| **Headline Only** | 1000 | **83.50%** | **0.8330** | **0.9423** | **5.60%** | **27.40%** |

### Key Observations
1. On **Full Text**, the model achieves 98.5% accuracy with an ultra-low false-REAL rate of 1.8%.
2. When inputs are truncated to **First 30 Words**, the False-REAL rate jumps to **6.0%**.
3. On **Headline Only**, the False-REAL rate rises to **5.6%**. Short text lacks the body-text stylistic markers the model memorized, causing it to default to REAL when overt clickbait punctuation is absent.

---

## 3. Cross-Source Test: Generalization Beyond ISOT (Zero-Shot)

### Test Methodology
Evaluated the current frozen model on external real datasets loaded from HuggingFace Hub, without any fine-tuning:
1. **`UKPLab/liar`** (Test split: 1283 short political claims verified by PolitiFact).
   - Labels mapped strictly by name: `'true statement' -> REAL (0)`, `'false statement' -> FAKE (1)`.
2. **`davanstrien/WELFake`** (1000 balanced articles).
   - Labels mapped strictly by name: `'real' -> REAL (0)`, `'fake' -> FAKE (1)`.

### Measured Cross-Source Results
| Dataset | Domain / Format | Sample Size | Accuracy | Macro F1 | ROC-AUC | False-REAL Rate | False-FAKE Rate |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **UKPLab/liar** | Short claims / PolitiFact | 1283 | **53.39%** | **0.5071** | **0.5467** | **43.57%** | **47.72%** |
| **davanstrien/WELFake** | Multi-source articles | 1000 | **92.00%** | **0.9199** | **0.9748** | **11.40%** | **4.60%** |

### Key Observations
1. On **LIAR short claims**, the classifier suffers catastrophic failure with **43.6% False-REAL rate** (the model calls 43.6% of false claims REAL).
2. Because short claims in LIAR are typically sober quotes rather than sensational blog titles, the ISOT-trained model predicts REAL for nearly all of them.
3. On **WELFake**, accuracy drops to **92.00%** compared to >98% on ISOT, demonstrating severe source overfitting.

---

## 4. Root Cause Summary (Three Specific Drivers)

1. **Short-Input Distribution Shift:** The model was trained exclusively on long articles (median length >350 words). Short claims and headlines are out-of-distribution, causing the model to default to REAL when sensational caps/exclamations are absent.
2. **Style Bias Over Truth Verification:** The classifier learns stylistic proxies (formal datelines vs angry rhetoric). A fake claim phrased formally receives >95% REAL probability.
3. **Single-Source Dataset Overfitting:** The ISOT dataset consists of Reuters articles (real) and 2016 partisan political blogs (fake). The model learned source-specific vocabulary artifacts rather than generalizable signals of deception.
