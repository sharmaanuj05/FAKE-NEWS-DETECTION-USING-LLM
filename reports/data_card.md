# Data Card: Multi-Source Robust Fake News Dataset (v2)

This dataset addresses generalization failure on short claims, headlines, and multi-domain news by combining three distinct benchmark sources with length augmentation and strict leakage removal.

---

## 1. Dataset Summary & Motivation
The original model trained on ISOT (`GonzaloA/fake_news`) suffered severe distribution shift on short claims and formal hoaxes because:
1. Training articles were long (median >350 words).
2. REAL news came from Reuters with formal datelines; FAKE news came from 2016 partisan blogs.
3. The model learned stylistic proxies (formal datelines = REAL) rather than generalizable signals.

### Improvements in Dataset v2:
- **Multi-Source Diversity:** Combines ISOT, WELFake, and LIAR.
- **Short Claims In-Distribution:** Integrates short political claims from PolitiFact (`UKPLab/liar`) and length-augmented headlines/30-word summaries.
- **Length Augmentation (Train Split Only):** Truncated versions (title-only, first 30 words) generated strictly from training articles.
- **Strict Leakage Removal:** Applied uniformly to every source.
- **Cross-Split Independence:** 0 duplicate texts across train, validation, mixed-test, and unseen-source holdout.
- **Source Share Cap:** No single source contributes more than 50% of the training split.

---

## 2. Included Sources & Explicit Label Mappings

Every source was inspected and mapped strictly by name to binary classification:
- **`0` = REAL** (verified factual reporting)
- **`1` = FAKE** (debunked claims, fabricated articles, misinformation)

| Source Name | HuggingFace Hub ID | Domain / Type | Raw Features / Labels | Mapped Semantics | Dropped Classes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ISOT** | `GonzaloA/fake_news` | Full articles (2016 geopolitical) | `label: int` (1=Real, 0=Fake) | 1 -> `0 (REAL)`, 0 -> `1 (FAKE)` | None |
| **WELFake** | `davanstrien/WELFake` | Multi-publisher full articles | `label: int` (0=Real, 1=Fake) | 0 -> `0 (REAL)`, 1 -> `1 (FAKE)` | Dropped null/empty texts |
| **LIAR** | `UKPLab/liar` | Short political claims / PolitiFact | `label_text: str` ('true statement', 'false statement') | 'true statement' -> `0 (REAL)`, 'false statement' -> `1 (FAKE)` | None |

---

## 3. Split Counts and Class Balance

| Split | File Path | Total Rows | REAL (0) | FAKE (1) | Balance Ratio |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Train** | `data/processed_v2/train.parquet` | **10818** | **5440** | **5378** | **50.3% / 49.7%** |
| **Validation** | `data/processed_v2/val.parquet` | **1910** | **960** | **950** | 50.0% / 50.0% |
| **Mixed Test** | `data/processed_v2/test_mixed.parquet` | **944** | **488** | **456** | 50.0% / 50.0% |
| **ISOT Test** | `data/processed_v2/test_isot.parquet` | **1000** | **500** | **500** | 50.0% / 50.0% |
| **Held-Out Unseen** | `data/processed_v2/unseen_source.parquet` | **1283** | **941** | **342** | Held out test split |

---

## 4. Source Contribution to Training Split (Cap <= 50%)

| Source Category | Training Samples | Percentage of Train Split |
| :--- | :---: | :---: |
| **ISOT** | 3431 | 31.7% |
| **WELFake** | 3170 | 29.3% |
| **LIAR** | 2541 | 23.5% |
| **ISOT_30w_aug** | 433 | 4.0% |
| **ISOT_title_aug** | 429 | 4.0% |
| **WELFake_30w_aug** | 420 | 3.9% |
| **WELFake_title_aug** | 394 | 3.6% |

---

## 5. Leakage Removal Pipeline
Applied uniformly to all texts before tokenization:
1. **Publisher Signatures & Datelines:** Stripped `WASHINGTON (Reuters) -`, `(Reuters)`, `(AP)`, `(AFP)` and leading metropolitan headers.
2. **URLs & Hyperlinks:** Removed all `http://`, `https://`, and `www.` URLs.
3. **Social Media Handles:** Stripped all `@username` handles.
4. **Media Boilerplate Tags:** Stripped `[VIDEO]`, `[IMAGE]`, `[TWEET]`, etc.
5. **Credit Attributions:** Stripped `via ...` and `Featured image via...` strings.
6. **Fact-Check Prefixes:** Stripped prefixes like `Fact Check:`, `Debunk:`, and fact-check outlet names.

---

## 6. Cross-Split Independence Check (0 Duplicates)
- **Train vs Validation Duplicates:** `0`
- **Train vs Mixed Test Duplicates:** `0`
- **Train vs Unseen Source Duplicates:** `0`
- **Validation vs Unseen Source Duplicates:** `0`
- **Cross-Split Leakage Check:** **PASSED (0 duplicates detected across splits)**
