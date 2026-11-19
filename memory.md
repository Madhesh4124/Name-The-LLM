# Agent Memory: Name That LLM

**Project:** 18-Class LLM Authorship Attribution (Kaggle)  
**Repository:** `https://github.com/Madhesh4124/Name-The-LLM.git`  
**Master Plan:** [docs/PLAN.md](file:///d:/CODE/Projects/Name%20the%20LLM/docs/PLAN.md)  
**Phases Roadmap:** [PHASES.md](file:///d:/CODE/Projects/Name%20the%20LLM/PHASES.md)  
**Last Updated:** Phase 0 Initialization  

---

## 1. Project Overview & Core Facts

- **Task:** Classify ~4,000 short unlabeled passages (40–250 words) into one of 18 frontier/lightweight LLM classes across 6 research labs:
  - **OpenAI (5):** `gpt-5.5`, `gpt-5.6-luna`, `gpt-5.6-sol`, `gpt-5.6-terra`, `gpt-6-astra`
  - **Google (4):** `gemini-3.1-pro-preview`, `gemini-3.6-flash`, `gemini-3.7-flash`, `gemini-3.8-flash`
  - **Anthropic (5):** `claude-fable-5`, `claude-opus-4-6`, `claude-opus-4-8`, `claude-opus-5`, `claude-sonnet-5`
  - **Qwen (1):** `qwen3.8-max`
  - **ZAI (2):** `glm-5.3`, `glm-5.3-flash`
  - **Moonshot (1):** `kimi-k3`
- **Evaluation Metric:** Balanced Accuracy (unweighted macro-average of recall across all 18 classes).
- **Core Decision Invariant:**
  
  $$\hat{y} = \arg\max_{k \in \{1,\dots,18\}} \frac{P_{\text{train}}(y=k \mid x)}{\pi_{\text{train}}(k)}$$
  
  *Never estimate or fit unregularized test class priors for the base decision rule.*

---

## 2. Absolute Non-Negotiables & Invariants

1. **Strictly In-Fold Fitting:** Vectorizers, scalers, calibrators, and stackers must be fit strictly on training-fold rows inside `sklearn.pipeline.Pipeline` or an in-fold cross-validation loop. Zero global fitting on full data.
2. **Frozen Folds:** Folds are created once using `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` and stored in `folds/folds.parquet` (`id`, `fold`, `group`, `model`). Never regenerate or alter folds once initialized.
3. **Layout Mapping (`src/text_utils.py`):** Before calling character TF-IDF, replace multi-character layout tokens with Unicode Private-Use Area (PUA) characters longest-pattern-first:
   - Line-anchored regex `(?m)^[ \t]*[-*•][ \t]+` $\to$ `\uE005` (never use `\s` which consumes newlines)
   - Line-anchored regex `(?m)^[ \t]*\d+[.)][ \t]+` $\to$ `\uE006`
   - Headers: `###` $\to$ `\uE002`, `##` $\to$ `\uE003`
   - Bolding: `**` $\to$ `\uE004`
   - Newlines (applied after lists): `\n\n` $\to$ `\uE000`, `\n` $\to$ `\uE001`
   - Single em-dash `—` is preserved natively.
4. **Standardized OOF Contract:** Every experiment `<exp>` writes to `artifacts/<exp>/`:
   - `oof.npy` (float32, $N_{\text{train}} \times 18$, rows sum to 1.0, row order matching `folds.parquet`, column order matching `configs/labels.json`).
   - `test.npy` (float32, $N_{\text{test}} \times 18$, fold-averaged probabilities).
   - `metrics.json` (`balanced_accuracy_argmax`, `balanced_accuracy_prior_divided`, `per_class_recall`).
   - `confusion_matrix.csv`, `config.yaml`, `env.txt`, `git_commit.txt`.
   - Validated via `.agent/skills/oof-contract/scripts/validate_oof.py`.
5. **Model Acceptance Threshold:**
   Let $\Delta_f = \text{score}_f(\text{new}) - \text{score}_f(\text{reference})$. Accept a branch or component if and only if:
   
   $$\text{mean}(\Delta) > \text{sd}(\Delta) \quad \text{AND} \quad \Delta_f > 0 \text{ on at least 4 of 5 folds}$$

6. **GCP Compute & Cost Safety:**
   - Explicit confirmation required before executing any `gcloud` create, resize, start, stop, or delete command, and any Cloud Run job creation/execution.
   - Mandatory 100-step throughput pilot before launching 5-fold training.
   - Spot VMs must use `scripts/run_job.sh` (checkpointing every $N$ steps to GCS, automated resume, automated `sudo shutdown -h now` on completion).
   - Teardown audit required at end of phase (`/gcp-teardown`).
7. **Offline Kaggle Invariance:**
   - Submission notebook operates with internet disabled, runtime $\le 9.0$ hours.
   - Target GPU is Kaggle T4 (no native `bf16`; all models must be verified in `fp16`).
   - Packaging: 1 base HF model + 5 lightweight LoRA adapters (not 5 full merged models).
   - Offline reproduction gate on Fold 0: $|\Delta \text{BA}| \le 0.002$, prediction agreement $\ge 99.5\%$.

---

## 3. Environment & Hardware Profiles

| Environment | Purpose | Hardware / Specs | Key Constraints |
| :--- | :--- | :--- | :--- |
| **Local Machine** | EDA, Folds, Baseline, Stacking, Packaging | Windows, Local CPU / Python 3.11+ | Memory-conscious TF-IDF parameter tuning (`min_df`, `max_features`). |
| **GCP Spot VM** | Primary Neural Training | `g2-standard-8` (1x NVIDIA L4 24GB, 8 vCPUs, 32 GB RAM, 200 GB disk) | Spot preemption risk; automated GCS checkpointing & resume mandatory. |
| **GCP Cloud Run** | GPU Contingency Fallback | Container on 1x L4, 8 vCPU, 32 GiB RAM (`--no-gpu-zonal-redundancy`) | Max 3 GPUs per region; no Spot discount; 24h max task timeout. |
| **Kaggle Notebook**| Offline Evaluation & Submission | 1x NVIDIA T4 (16 GB VRAM), 30 GB RAM, 2 vCPUs | Internet OFF, max 9h runtime; strictly `fp16` precision (no `bf16`). |

---

## 4. Current State & Working Memory

- **Active Phase:** `Phase 0: EDA & Data Audit`
- **Current Branch:** `master`
- **Git Remote:** `origin` $\to$ `https://github.com/Madhesh4124/Name-The-LLM.git`
- **Next Action Items:**
  1. Download competition data to `data/`: `kaggle competitions download -c name-that-llm -p data/`
  2. Extract data and audit files (`train.csv`, `test.csv`, `sample_submission.csv` or `parquet`).
  3. Extract canonical class labels to `configs/labels.json`.
  4. Run `/eda` workflow to produce `docs/eda_report.md`.

---

## 5. Architectural Decisions Log (ADRs)

- **ADR-001 (Prior Division Rule):** Due to balanced accuracy being unweighted macro-recall, test set predictions must be divided by training class frequencies $\pi_{\text{train}}(k)$ prior to $\text{argmax}$.
- **ADR-002 (Unicode PUA Mapping):** Sklearn's `TfidfVectorizer(analyzer='char')` collapses `\n` to spaces. To preserve structural line breaks and markdown syntax, we inject PUA code points (`\uE000`–`\uE006`) prior to tokenization.
- **ADR-003 (LoRA Adapter Packaging):** 5-fold neural checkpoints are packaged as 1 shared base model + 5 fold adapter weights rather than 5 full models, preventing Kaggle dataset storage exhaustion and enabling rapid offline weight switching.
- **ADR-004 (Dual Formulation for LinearSVC):** In sparse n-gram spaces where $N_{\text{features}} \gg N_{\text{samples}}$, default `dual` (`dual=True` / `'auto'`) must be used instead of `dual=False` (which solves the primal problem and scales poorly with feature dimensions).
- **ADR-005 (Automated Shutdown Pre-approval):** In-VM `sudo shutdown -h now` inside `scripts/run_job.sh` is pre-approved upon job launch confirmation to guarantee zero idle billing if a training run finishes overnight.

---

## 6. Known Pitfalls & Countermeasures

| Observed Pitfall | Root Cause | Countermeasure |
| :--- | :--- | :--- |
| **Sklearn newline erasure** | `analyzer='char'` treats `\n` as whitespace separator. | Map `\n\n` $\to$ `\uE000`, `\n` $\to$ `\uE001` in `src/text_utils.py` before vectorizer. |
| **Stylistic CV leakage** | Clustering raw text groups passages by writing style rather than prompt topic. | Strip formatting before embedding; verify high model entropy per cluster. |
| **Stacker distribution shift** | Fold-averaged test predictions are smoother than single OOF predictions. | Compare max-prob histograms; apply temperature scaling before training GBDT stacker. |
| **T4 GPU NaN / Slowdown** | Kaggle T4 lacks native `bfloat16` compute units. | Force `float16` on all neural inference pipelines and run stability checks. |
| **Public LB Overfitting** | Tuning hyperparameters on noisy public LB scores. | Trust only grouped-CV paired differences under Section 5.4 gate. |

---

## 7. Experiments Ledger

All experiments are permanently logged to `results.csv` and detailed under `artifacts/<exp>/`:

| Exp ID | Architecture / Model | Description | Grouped CV (Argmax) | Grouped CV (Prior-Div) | Status | Notes |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| *exp01* | *Baseline TF-IDF + LR* | *Char(2,6)+CharWB(3,5)+Word(1,3)* | *Pending* | *Pending* | *Queued* | *Reference model for Phase 2* |
