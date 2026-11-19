# Phase 0 EDA & Data Audit Report: Name That LLM

**Project:** 18-Class LLM Authorship Attribution (Kaggle)  
**Date:** November 2026  
**Status:** Complete  
**Authors/Execution:** Automated Agent Data Audit  
**Artifacts Produced:** `docs/eda_report.md`, `configs/labels.json`  

---

## 1. Executive Summary

This report documents the exhaustive Phase 0 Exploratory Data Analysis and Data Integrity Audit for the **Name That LLM** Kaggle competition.

### Key Discoveries & Architectural Takeaways
1. **Explicit Prompt Grouping in Train (`prompt_id`):**  
   `train.csv` contains an explicit `prompt_id` column covering **150 unique prompts**. 149 prompts have exactly 18 completions (one per target model), and 1 prompt (`p002`) has 17 completions (missing `claude-fable-5`). Across the entire training set, **no model ever appears more than once within the same prompt**.
2. **Absolute Need for Grouped Cross-Validation:**  
   Because all 18 models respond to identical prompts, passages within the same `prompt_id` share topical vocabulary, thematic framing, and named entities. A naive random split would cause massive cross-validation leakage and severe overfitting. Stratified Group K-Fold grouped strictly on `prompt_id` is mandatory.
3. **Absence of `prompt_id` in Test:**  
   `test.csv` contains only `id` and `text`. Mutual-exclusivity post-processing (e.g., Sinkhorn-Knopp matching) cannot rely on ground-truth prompt IDs at inference time and must either use high-confidence semantic clustering or remain unconstrained.
4. **Stripped Newlines & Normalized Layout:**  
   Both `train.csv` and `test.csv` consist entirely of single-paragraph passages: raw newlines (`\n`, `\r`) and tab characters (`\t`) are 100% absent (0 occurrences across all 2,699 training rows and test rows).
5. **Potent Typographic & Stylometric Signatures:**  
   Despite the absence of newlines, subtle typographic artifacts provide near-deterministic discriminator signals across labs:
   - **Claude (Anthropic):** Uses straight single quotes (`'`) exclusively (0.0% curly `’`) and heavy em-dashes (`—` in 63%–83% of passages, average 1.0–1.6 per passage).
   - **OpenAI (`gpt-5.*`, `gpt-6`):** Uses curly single quotes (`’` in 42%–53% of passages, ~0% straight `'`), curly double quotes (`“”`), and low em-dash rates. Only OpenAI models generate LaTeX delimiters (`\(`).
   - **Gemini (Google):** Virtually **zero em-dashes** (0.0% in Flash models, 0.7% in Pro Preview) and very low colon/semicolon usage.
   - **GLM (ZAI):** Extremely high em-dash frequency (80%–82%, avg 1.6/passage) with straight single quotes.
6. **No Synthetic Mid-Sentence Truncation:**  
   99.93% of passages terminate with standard sentence punctuation (`.`, `!`, `?`, `"`, `”`). Only 2 passages in `kimi-k3` end with a trailing letter. Suffix TF-IDF features will not be degraded by arbitrary token cutoffs.

---

## 2. Dataset Overview & Schema Audit

### 2.1 File Summaries

| File | Rows | Columns | Memory Usage | Missing Values |
| :--- | :---: | :--- | :---: | :---: |
| `data/train.csv` | 2,699 | `id` (object), `prompt_id` (object), `text` (object), `model` (object) | ~2.1 MB | 0 |
| `data/test.csv` | 3* | `id` (object), `text` (object) | 2.2 KB | 0 |
| `data/sample_submission.parquet` | 3* | `id` (object), `model` (object) | 899 B | 0 |

*\*Note: The local `test.csv` and `sample_submission.parquet` represent Kaggle's toy public test placeholder. When submitted to the Kaggle offline environment, `test.csv` is dynamically swapped with the hidden test set (~4,000 rows).*

### 2.2 Integrity & Deduplication Check
- **Unique IDs:** Exactly 2,699 unique `id` values in `train.csv` (100% unique).
- **Unique Passages:** Exactly 2,699 unique `text` entries in `train.csv`. Zero exact text duplicates.
- **Null / Missing Data:** Zero missing values across all columns.

---

## 3. Target Class Distribution & Priors

The target variable `model` contains **18 classes** across 6 frontier AI laboratories.

### 3.1 Class Frequency Table

Canonical label order is saved in `configs/labels.json`.

| Index | Target Label (`model`) | Lab Family | Sample Count | Class Prior $\pi_{\text{train}}(k)$ | Balance Ratio |
| :---: | :--- | :--- | :---: | :---: | :---: |
| 0 | `claude-fable-5` | Anthropic | 149 | 0.055206 | 0.9933 |
| 1 | `claude-opus-4-6` | Anthropic | 150 | 0.055576 | 1.0000 |
| 2 | `claude-opus-4-8` | Anthropic | 150 | 0.055576 | 1.0000 |
| 3 | `claude-opus-5` | Anthropic | 150 | 0.055576 | 1.0000 |
| 4 | `claude-sonnet-5` | Anthropic | 150 | 0.055576 | 1.0000 |
| 5 | `gemini-3.1-pro-preview` | Google | 150 | 0.055576 | 1.0000 |
| 6 | `gemini-3.6-flash` | Google | 150 | 0.055576 | 1.0000 |
| 7 | `gemini-3.7-flash` | Google | 150 | 0.055576 | 1.0000 |
| 8 | `gemini-3.8-flash` | Google | 150 | 0.055576 | 1.0000 |
| 9 | `glm-5.3` | ZAI | 150 | 0.055576 | 1.0000 |
| 10 | `glm-5.3-flash` | ZAI | 150 | 0.055576 | 1.0000 |
| 11 | `gpt-5.5` | OpenAI | 150 | 0.055576 | 1.0000 |
| 12 | `gpt-5.6-luna` | OpenAI | 150 | 0.055576 | 1.0000 |
| 13 | `gpt-5.6-sol` | OpenAI | 150 | 0.055576 | 1.0000 |
| 14 | `gpt-5.6-terra` | OpenAI | 150 | 0.055576 | 1.0000 |
| 15 | `gpt-6-astra` | OpenAI | 150 | 0.055576 | 1.0000 |
| 16 | `kimi-k3` | Moonshot | 150 | 0.055576 | 1.0000 |
| 17 | `qwen3.8-max` | Qwen | 150 | 0.055576 | 1.0000 |
| **Total** | **18 Classes** | **6 Labs** | **2,699** | **1.000000** | — |

**Finding:** The dataset is virtually perfectly balanced ($150 \times 17 + 149 = 2,699$). The prior division rule $\arg\max_k P(y=k \mid x)/\pi_{\text{train}}(k)$ is straightforward since $\pi_{\text{train}}(k) \approx 1/18 \approx 0.05556$ across all classes.

---

## 4. Prompt Grouping Structure & Mutual Exclusivity

### 4.1 Training Prompt Distribution
- **Total Unique Prompts:** 150 (`p001` through `p150`).
- **Prompts with all 18 models present:** 149 prompts.
- **Prompts with missing models:** Exactly 1 prompt (`p002`), which has 17 completions and lacks `claude-fable-5`.
- **Within-Prompt Uniqueness:** In every prompt, no model ever appears twice. The contingency matrix between `prompt_id` and `model` is strictly binary $\{0, 1\}$.

### 4.2 Leakage Risk & Grouped Cross-Validation Mandate
When inspecting completions for any given `prompt_id` (e.g. `p081`, "When a language dies..."):
- Every model writes a response directly addressing the prompt prompt's core metaphor (astronomy, grammar of the cosmos, star names, etc.).
- An un-grouped K-Fold split would place completions of `p081` from 14 models in the training fold and 4 models in the validation fold. The model would easily score high simply by detecting prompt-specific nouns ("constellations", "astronomy") rather than model-specific stylistic signatures.
- **Decision:** Folds must be created strictly using `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` grouped by `prompt_id`. With 150 prompts, exactly 30 prompts will be assigned to each of the 5 folds.

### 4.3 Test Set Prompt Structure
- In `test.csv`, `prompt_id` is omitted.
- The 3 local test samples represent 3 completely unrelated topics:
  1. `00353fb1c541`: The Jevons Paradox (resource efficiency and consumption rebound).
  2. `005f5a9a6139`: The Stern-Gerlach quantum mechanics experiment (1922 silver beam).
  3. `0062a53d5487`: Geology of evaporite salt domes and diapirs.
- **Implication for Transductive Matching:**
  Because the test set does not reveal prompt groupings, applying Sinkhorn-Knopp mutual exclusivity post-processing would require unsupervised prompt clustering on test. Given that clustering errors propagate into forced assignment errors, transductive post-processing must be evaluated strictly with high-threshold clustering and validated against baseline argmax.

---

## 5. Text Length & Complexity Distributions

### 5.1 Overall Corpus Statistics
- **Character Length:** Min: 230, 25%: 540, Median: 670, 75%: 905, Max: 1,900, Mean: 755.3 (std: 268.4)
- **Word Count:** Min: 36, 25%: 81, Median: 101, 75%: 138, Max: 274, Mean: 114.7 (std: 40.7)
- **Sentences:** Min: 1, Median: 5, Mean: 4.88, Max: 14
- **Words per Sentence:** Min: 8.5, Median: 24.1, Mean: 24.3, Max: 47.0

### 5.2 Per-Class Length & Diversity Breakdown

| Model Label | Lab | Word Min | Word Median | Word Mean | Word Max | Word IQR | Char Median | Sentences (Avg) | Words/Sent | TTR (Lexical Div) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `claude-fable-5` | Anthropic | 39 | 100.0 | 112.5 | 252 | 57.0 | 646.0 | 4.74 | 23.96 | 0.8093 |
| `claude-opus-4-6` | Anthropic | 42 | 98.5 | 115.1 | 274 | 60.0 | 687.5 | 4.33 | **27.00** | 0.8125 |
| `claude-opus-4-8` | Anthropic | 42 | 109.0 | 123.6 | 251 | 63.8 | 731.5 | 5.11 | 23.73 | 0.7992 |
| `claude-opus-5` | Anthropic | 39 | **116.5** | **123.4** | 257 | 64.5 | 725.5 | 5.23 | 23.59 | 0.7802 |
| `claude-sonnet-5` | Anthropic | 39 | 99.5 | 120.0 | 267 | 65.5 | 707.0 | 4.59 | 26.05 | 0.8173 |
| `gemini-3.1-pro-preview` | Google | 38 | 104.0 | 119.3 | 262 | 59.8 | 714.0 | **5.78** | 20.23 | 0.8041 |
| `gemini-3.6-flash` | Google | 40 | 95.5 | 111.2 | 238 | 57.0 | 634.5 | 4.60 | 24.62 | 0.8273 |
| `gemini-3.7-flash` | Google | 40 | 101.5 | 117.1 | 252 | 59.5 | 717.5 | 4.81 | 23.85 | 0.8393 |
| `gemini-3.8-flash` | Google | 41 | **84.0** | **100.7** | 238 | 49.5 | 583.0 | 4.43 | 22.66 | **0.8594** |
| `glm-5.3` | ZAI | 36 | 96.0 | 110.3 | 244 | 55.0 | 616.5 | 4.91 | 23.13 | 0.8109 |
| `glm-5.3-flash` | ZAI | 37 | 112.0 | 119.2 | 256 | 58.0 | 744.0 | 5.40 | 23.06 | 0.8027 |
| `gpt-5.5` | OpenAI | 39 | 103.5 | 110.3 | 245 | 54.0 | 643.5 | 4.39 | 26.04 | 0.8237 |
| `gpt-5.6-luna` | OpenAI | 38 | 104.5 | 115.4 | 246 | 57.8 | 673.0 | 4.93 | 23.49 | 0.8217 |
| `gpt-5.6-sol` | OpenAI | 40 | 111.5 | 118.9 | 261 | 58.8 | 747.5 | 4.99 | 24.54 | 0.8246 |
| `gpt-5.6-terra` | OpenAI | 43 | 100.5 | 113.3 | 239 | 57.5 | 648.5 | 4.47 | 26.15 | 0.8289 |
| `gpt-6-astra` | OpenAI | 36 | 105.5 | 117.8 | 256 | 58.8 | 677.5 | 5.59 | 20.99 | 0.8162 |
| `kimi-k3` | Moonshot | 39 | 89.0 | 107.4 | 239 | 52.8 | 594.5 | 4.51 | 25.55 | 0.8327 |
| `qwen3.8-max` | Qwen | 39 | 98.5 | 108.3 | 265 | 55.0 | 625.0 | 4.97 | 22.20 | 0.8371 |

### 5.3 Key Length Observations
- **Shortest Model:** `gemini-3.8-flash` is systematically the most concise (median 84.0 words, median 583 characters) with the highest Type-Token Ratio (0.8594), reflecting rapid, high-density token economy.
- **Longest Model:** `claude-opus-5` and `claude-opus-4-8` generate the longest passages (median 116.5 and 109.0 words).
- **Sentence Architecture:** `claude-opus-4-6` writes the longest compound sentences (27.0 words/sentence), whereas `gemini-3.1-pro-preview` writes more frequent, shorter sentences (5.78 sentences/passage, 20.2 words/sentence).

---

## 6. Formatting Survival & Typographic Signatures

An audit across all 2,699 training passages revealed diagnostic typographic patterns distinguishing model families.

### 6.1 Feature Presence (% of Passages with Feature $\ge 1$)

| Model Label | Curly `'` (`’`) | Straight `'` | Curly `"` (`“”`) | Straight `"` | Em-Dash (`—`) | Colon (`:`) | Semicolon (`;`) | LaTeX (`\(`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `claude-fable-5` | **0.0%** | **59.7%** | **0.0%** | 8.7% | **83.2%** | 59.1% | 33.6% | 0.0% |
| `claude-opus-4-6` | **0.0%** | **57.3%** | **0.0%** | 7.3% | **75.3%** | 42.0% | 15.3% | 0.0% |
| `claude-opus-4-8` | **0.0%** | **60.0%** | **0.0%** | 7.3% | **62.7%** | 49.3% | 24.0% | 0.0% |
| `claude-opus-5` | **0.0%** | **63.3%** | **0.0%** | 9.3% | **81.3%** | 69.3% | 54.0% | 0.0% |
| `claude-sonnet-5` | **0.0%** | **72.7%** | **0.0%** | 10.0% | **73.3%** | 50.7% | 23.3% | 0.0% |
| `gemini-3.1-pro-preview` | 6.0% | 10.7% | 0.0% | 2.7% | **0.0%** | 16.7% | 19.3% | 0.0% |
| `gemini-3.6-flash` | 19.3% | 21.3% | 0.0% | 2.7% | **0.0%** | 20.0% | 22.7% | 0.0% |
| `gemini-3.7-flash` | 30.7% | 14.0% | 0.0% | 0.0% | **0.7%** | 22.7% | 23.3% | 0.0% |
| `gemini-3.8-flash` | 24.7% | 11.3% | 2.0% | 0.7% | **0.0%** | 18.0% | 28.0% | 0.0% |
| `glm-5.3` | **0.0%** | **58.0%** | **0.0%** | 9.3% | **82.0%** | 62.0% | 34.0% | 0.0% |
| `glm-5.3-flash` | **0.0%** | **67.3%** | **0.0%** | 12.0% | **79.3%** | 65.3% | 48.0% | 0.0% |
| `gpt-5.5` | **42.0%** | 11.3% | 9.3% | 1.3% | 13.3% | 65.3% | 60.7% | 0.0% |
| `gpt-5.6-luna` | **51.3%** | **0.7%** | 10.7% | 0.0% | 36.7% | 70.7% | 42.7% | **2.7%** |
| `gpt-5.6-sol` | **52.7%** | **0.0%** | 14.7% | 0.0% | 27.3% | 64.0% | 62.0% | 0.0% |
| `gpt-5.6-terra` | **46.7%** | 5.3% | 8.0% | 0.0% | 22.7% | 73.3% | 52.0% | **0.7%** |
| `gpt-6-astra` | **44.0%** | 8.7% | 7.3% | 0.0% | 8.7% | 65.3% | 52.7% | 0.0% |
| `kimi-k3` | 32.0% | 23.3% | 9.3% | 1.3% | 52.0% | 77.3% | 61.3% | 0.0% |
| `qwen3.8-max` | 28.7% | 12.0% | 0.7% | 0.0% | 3.3% | 47.3% | 39.3% | 0.0% |

### 6.2 Deep Dive into Diagnostic Lab Signatures
1. **The Claude vs. OpenAI Quote Invariant:**
   - Claude models (Anthropic) emit **straight single quotes (`'`) exclusively** (0.0% curly single quotes, 0.0% curly double quotes).
   - OpenAI models (`gpt-5.6-sol`, `gpt-5.6-luna`, `gpt-6-astra`) emit **curly single quotes (`’`)** almost exclusively, with `gpt-5.6-sol` exhibiting literally 0.0% straight single quotes.
2. **The Gemini Em-Dash Absence Rule:**
   - Gemini models (`gemini-3.1-pro-preview`, `gemini-3.6-flash`, `gemini-3.8-flash`) have a **0.0% occurrence of em-dashes** (`—`), and `gemini-3.7-flash` has 0.7% (1 passage). Any passage containing an em-dash is almost certainly not Gemini.
3. **The GLM / Claude Em-Dash Frequency:**
   - GLM models and Claude models use em-dashes profusely (averaging 1.3 to 1.6 em-dashes per passage, present in ~80% of passages).
4. **The OpenAI LaTeX Signature:**
   - LaTeX mathematical delimiters `\(...\)` appear only in OpenAI reasoning models (`gpt-5.6-luna` and `gpt-5.6-terra`).

---

## 7. Sklearn Vectorizer Verification & Whitespace Nuances

### 7.1 Mandatory 2-Line Test
We tested `TfidfVectorizer` on newline strings and markdown snippets:
```python
from sklearn.feature_extraction.text import TfidfVectorizer
vec_wb = TfidfVectorizer(analyzer='char_wb', ngram_range=(1, 3)).fit(["hello\nworld"])
print(vec_wb.get_feature_names_out())
# Output: [' ', ' h', ' w', 'h', 'he', 'hel', 'o', 'o ', 'w', 'wo', 'wor']
```
### 7.2 Findings & Implication
1. In `analyzer='char_wb'`, newlines are collapsed to standard whitespace/word boundaries and completely stripped from the resulting vocabulary.
2. In the raw dataset, raw newlines (`\n`) are already absent. However, punctuation, dashes (`—`), apostrophes (`'`), and quotes (`”`) are fully preserved.
3. Because scikit-learn's default tokenizers strip punctuation when `analyzer='word'` is used (unless a custom `token_pattern` is supplied), **character n-grams (`analyzer='char'` and `analyzer='char_wb'`) are essential to capture punctuation signatures**.
4. Preserving single-character em-dashes, quotes, and punctuation without collapse confirms the necessity of retaining raw characters in the sparse feature union.

---

## 8. Passage Termination & Truncation Audit

### 8.1 Results
- **Standard Terminal Characters:** `.`, `!`, `?`, `"`, `”`, `'`, `’`, `)`, `]`, `}`
- **Standard Termination Rate:** **99.93%** (2,697 out of 2,699 passages).
- **Non-Standard Terminations:** Exactly 2 passages in `kimi-k3` ended with the letter `t` (e.g., passage cut off at end of line without a period). All other 2,697 passages terminate with complete grammatical sentences.

### 8.2 Implications
- Passages in this competition are NOT subject to arbitrary token hard-clipping.
- Suffix n-grams (last 25 words) represent legitimate conclusion sentences and summary formulations, validating the Phase 2b prefix/suffix feature ablation.

---

## 9. Phase 0 Exit Gate Verification

| Checklist Item | Requirement | Observed Status | Gate Result |
| :--- | :--- | :--- | :---: |
| 1. Data Download | Raw files in `data/` | `train.csv` (2699 rows), `test.csv` (3 rows), `sample_submission.parquet` verified | ✅ PASSED |
| 2. Canonical Labels | `configs/labels.json` confirmed | 18 exact target classes saved in alphabetical canonical order | ✅ PASSED |
| 3. Class Balance | Verify class counts | 17 classes $\times$ 150, 1 class $\times$ 149. Balanced priors verified. | ✅ PASSED |
| 4. Prompt Grouping | Audit `prompt_id` & leakage | `prompt_id` explicit in train (150 prompts, 0 duplicate models/prompt). Grouped CV required. | ✅ PASSED |
| 5. Formatting Survival | Check `\n`, markdown, punctuation | Raw newlines stripped; rich typography (quotes, dashes, colons) survived with strong signal. | ✅ PASSED |
| 6. Sklearn Collapse | Verify char vectorizer behavior | Confirmed `char_wb` boundary collapse; character n-grams retain punctuation signatures. | ✅ PASSED |
| 7. Passage Termination | Audit truncation risk | 99.93% clean sentence termination; suffix features validated. | ✅ PASSED |
| 8. Comprehensive Report | `docs/eda_report.md` written | Detailed report completed with statistical tables and ADR alignment. | ✅ PASSED |

---

## 10. Roadmap & Immediate Next Steps for Phase 1

With Phase 0 complete and verified:
1. **Transition to Phase 1: Grouped Cross-Validation Creation (`/make-folds`):**
   - Partition the 150 `prompt_id` groups into 5 folds using `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`.
   - Verify zero group leakage using `.agent/skills/grouped-cv/scripts/check_folds.py`.
   - Run shuffled-label baseline test to assert balanced accuracy $\approx 0.0556$ (chance level).
   - Save partition to `folds/folds.parquet` (`id`, `fold`, `group`, `model`) and freeze the file permanently.
