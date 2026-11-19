# Project Phases & Execution Roadmap: Name That LLM

**Master Plan Reference:** [docs/PLAN.md](file:///d:/CODE/Projects/Name%20the%20LLM/docs/PLAN.md)  
**Agent Memory Reference:** [memory.md](file:///d:/CODE/Projects/Name%20the%20LLM/memory.md)  
**Current Status:** `PHASE 0 - READY TO EXECUTE`  

---

## 1. Executive Status Dashboard

| Phase | Description | Workflow | Target Compute | Status | Exit Gate |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **Phase 0** | EDA & Data Audit | `/eda` | Local CPU | 🟡 `READY` | `docs/eda_report.md` written, `configs/labels.json` confirmed, formatting survival & newline collapse verified. |
| **Phase 1** | Grouped CV Split | `/make-folds` | Local CPU | ⚪ `PENDING` | `folds/folds.parquet` saved (never regenerated), 0 group leakage, cluster purity passed, shuffled labels $\approx 1/18$. |
| **Phase 2** | Stage 1 Sparse Baseline | `/baseline` | Local CPU | ⚪ `PENDING` | `artifacts/exp01_baseline/` verified via `validate_oof.py`, baseline balanced accuracy recorded in `results.csv`. |
| **Phase 2b** | Prefix/Suffix TF-IDF | `no workflow yet` | Local CPU | ⚪ `PENDING` | Ablation meets: $\text{mean}(\Delta) > \text{sd}(\Delta)$ and $\Delta_f > 0$ on $\ge 4/5$ folds. |
| **Phase 3** | Stylometrics + Stacker | `no workflow yet` | Local CPU | ⚪ `PENDING` | LightGBM stacker beats Stage 1 reference; OOF vs test probability distribution shift audited. |
| **Phase 4a** | GCP Preflight Audit | `/gcp-preflight` | Local Shell | ⚪ `PENDING` | Quotas, Spot L4 availability, GCS bucket access, and budget alerts verified. |
| **Phase 4b** | Neural Classifiers | `/gcp-train` | GCP L4 Spot VM | ⚪ `PENDING` | 100-step pilot cost gate cleared; 1-fold candidate selected; 5-fold neural run beats baseline on grouped CV. |
| **Phase 4c** | GCP Teardown | `/gcp-teardown` | Local Shell | ⚪ `PENDING` | All VMs and disks terminated; 0 lingering billable resources. |
| **Phase 5** | Perplexity Probes | `no workflow yet` | GCP GPU / Kaggle | ⚪ `OPTIONAL` | Isolated ablation improves CV without violating Kaggle 9h inference limit. |
| **Phase 6** | Sibling Specialists | `no workflow yet` | Local CPU / GPU | ⚪ `OPTIONAL` | Severe intradomain sibling confusion verified in confusion matrix; nested CV confirms gain. |
| **Phase 7** | Ensemble & Post-Process | `no workflow yet` | Local CPU | ⚪ `PENDING` | Nested CV calibrated blend outperforms single models; transductive Sinkhorn validated on simulated shift. |
| **Phase 8** | Kaggle Packaging | `/package-kaggle` | Local & Kaggle | ⚪ `PENDING` | Datasets uploaded; offline Fold 0 reproduction: $|\Delta\text{BA}| \le 0.002$, agreement $\ge 99.5\%$, 5-fold smoke test passes. |
| **Phase 9** | Submission Validation | `/validate-submission` | Kaggle Offline | ⚪ `PENDING` | `submission.parquet` valid schema, non-zero class coverage, full dry run $\le 4.5$ h. |

---

## 2. Phase Breakdown & Checklists

### Phase 0: EDA & Data Audit
- **Objective:** Characterize data distributions, identify prompt grouping structures, verify formatting preservation, and check text normalization nuances.
- **Workflow Command:** `/eda`
- **Checklist:**
  - [ ] Download raw competition data into `data/` via Kaggle CLI (`kaggle competitions download -c name-that-llm`).
  - [ ] Create `configs/labels.json` containing the exact 18 target class names in canonical order.
  - [ ] Run class frequency analysis (check for class imbalance in `train.csv`).
  - [ ] Measure word and character length distributions across each of the 18 classes.
  - [ ] Audit `train.csv` and `test.csv` for explicit `prompt_id` or grouping metadata.
  - [ ] Run formatting survival check: `\n`, `\n\n`, `**`, `###`, bullets (`-`, `*`, `•`), numbered lists, curly quotes, double spaces, trailing whitespace.
  - [ ] Run 2-line test confirming that sklearn's `char` analyzer collapses `\n` into spaces.
  - [ ] Audit passage ending patterns (detect truncation or signature closing tokens).
  - [ ] If reconstructing prompt groups: perform embedding clustering on sanitized text and execute the **cluster-purity check** (ensure high model diversity per cluster).
  - [ ] Generate comprehensive `docs/eda_report.md`.

### Phase 1: Grouped Cross-Validation Creation
- **Objective:** Establish the unalterable 5-fold grouped CV partition.
- **Workflow Command:** `/make-folds`
- **Checklist:**
  - [ ] Read findings from `docs/eda_report.md`.
  - [ ] Define groups: use explicit prompt IDs if present; otherwise, use conservative semantic clustering.
  - [ ] Split into 5 folds using `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`.
  - [ ] Save partition once to `folds/folds.parquet` (`id`, `fold`, `group`, `model`).
  - [ ] Run `.agent/skills/grouped-cv/scripts/check_folds.py folds/folds.parquet configs/labels.json`.
  - [ ] Run shuffled-label leakage test: confirm baseline accuracy is near chance ($\approx 0.0556$).
  - [ ] Freeze `folds/folds.parquet` (mark read-only / never overwrite).

### Phase 2: Stage 1 Sparse Linear Baseline
- **Objective:** Build a robust, reproducible TF-IDF + linear classifier baseline fitting strictly inside each fold.
- **Workflow Command:** `/baseline`
- **Checklist:**
  - [ ] Implement Unicode private-use token mapping in `src/text_utils.py` (longest-pattern-first, line-anchored regex `(?m)^[ \t]*[-*•][ \t]+` and `(?m)^[ \t]*\d+[.)][ \t]+`, post-list newline replacement).
  - [ ] Construct in-fold `Pipeline` with `FeatureUnion`:
    - `char` n-grams: (2, 6), `lowercase=False`, `sublinear_tf=True`, `dtype=float32`
    - `char_wb` n-grams: (3, 5), `lowercase=False`, `sublinear_tf=True`, `dtype=float32`
    - `word` n-grams: (1, 3), `lowercase=False`, `sublinear_tf=True`, `dtype=float32`
    - Layout scalar counts (double spaces, trailing whitespace) + `MaxAbsScaler`
  - [ ] Pilot solver on Fold 0: compare `lbfgs`, `saga`, and `SGDClassifier(loss='log_loss')`.
  - [ ] If evaluating `LinearSVC`, wrap inside in-fold `CalibratedClassifierCV` and use default `dual` (not `dual=False`).
  - [ ] Generate full OOF predictions (`oof.npy`) and fold-averaged test predictions (`test.npy`) under `artifacts/exp01_baseline/`.
  - [ ] Run `python .agent/skills/oof-contract/scripts/validate_oof.py artifacts/exp01_baseline/oof.npy folds/folds.parquet configs/labels.json --save-dir artifacts/exp01_baseline`.
  - [ ] Record baseline balanced accuracy in `results.csv` as the reference benchmark.

### Phase 2b: Stage 1b Prefix/Suffix TF-IDF Ablation
- **Objective:** Test whether distinct opening or closing n-grams provide orthogonal signal.
- **Workflow:** `no workflow yet`
- **Checklist:**
  - [ ] Extract TF-IDF on first 25 words and last 25 words inside each fold.
  - [ ] Measure impact on short (40-word) passages and evaluate truncation vulnerability.
  - [ ] Compare paired fold scores $\Delta_f = \text{score}_f(\text{exp01b}) - \text{score}_f(\text{exp01})$.
  - [ ] Accept only if $\text{mean}(\Delta) > \text{sd}(\Delta)$ and $\Delta_f > 0$ on $\ge 4/5$ folds.

### Phase 3: Stage 2 Stylometric Features + Tabular Stacker
- **Objective:** Extract rich stylometric features and train a gradient boosted tree (LightGBM) on combined features + Stage 1 OOF probabilities.
- **Workflow:** `no workflow yet`
- **Checklist:**
  - [ ] Extract punctuation frequencies, markdown structure ratios, lexical diversity (TTR), sentence/paragraph lengths, and whitespace signatures.
  - [ ] Train LightGBM stacker on Stage 1 OOF probabilities + stylometrics strictly inside CV.
  - [ ] Audit distribution shift between OOF probabilities and fold-averaged test probabilities ($\max_k P_k$ histograms).
  - [ ] Run `validate_oof.py` on `artifacts/exp02_stylometric_lgb/`.
  - [ ] Evaluate against Stage 1 reference via the improvement threshold.

### Phase 4: Stage 3 Neural Classifiers & GCP Lifecycle
- **Objective:** Train deep transformer models (RoBERTa / DeBERTa / small decoder LLMs with LoRA) on GCP Spot L4 instances.
- **Workflows:** `/gcp-preflight`, `/gcp-train`, `/gcp-teardown`
- **Checklist:**
  - [ ] Execute `/gcp-preflight`: audit auth, quotas, Spot L4 availability, and billing alerts.
  - [ ] Run 100-step pilot on Fold 0 to measure throughput ($\text{samples/sec}$) and compute GPU-hours formula.
  - [ ] Present exact cost estimate to user and request explicit confirmation.
  - [ ] Run 1-fold candidate selection pilot on Fold 0:
    - Compare small decoder LLM (0.5B–3B with LoRA, mean vs. last-token pooling) against `roberta-large` / `deberta-v3-large`.
  - [ ] Train winning architecture across all 5 folds using `scripts/run_job.sh` (tmux, GCS checkpoints every $N$ steps, automated resume, automated shutdown).
  - [ ] Mirror artifacts to `$BUCKET/oof/` and `$BUCKET/test/`.
  - [ ] Validate OOF predictions locally with `validate_oof.py` and log to `results.csv`.
  - [ ] Execute `/gcp-teardown`: audit and terminate all instances, disks, and Cloud Run jobs.

### Phase 5: Stage 4 Conditional Perplexity Probes
- **Objective:** Evaluate sequence log-probabilities and token rank distribution buckets from open-weight base models.
- **Workflow:** `no workflow yet` (Conditional)
- **Checklist:**
  - [ ] Benchmark feature extraction latency against Kaggle 9h inference budget.
  - [ ] Compute log-probs and token rank buckets (`top-1`, `top-10`, `top-100`, `>1000`) for representative open models.
  - [ ] Test isolated ablation on grouped CV.

### Phase 6: Stage 5 Conditional Sibling Specialists
- **Objective:** Resolve intradomain misclassifications among sibling model variants.
- **Workflow:** `no workflow yet` (Conditional)
- **Checklist:**
  - [ ] Inspect $18 \times 18$ confusion matrix from Stage 1–3 models.
  - [ ] Identify if errors are concentrated in specific sibling sub-clusters (`gpt-5.6-*`, `claude-opus-*`, `gemini-*-flash`).
  - [ ] If present, train dedicated sibling discriminators evaluated strictly via nested CV.

### Phase 7: Ensembling, Calibration & Post-Processing
- **Objective:** Blend diverse model outputs, optimize probability calibration, and evaluate transductive adjustments.
- **Workflow:** `no workflow yet`
- **Checklist:**
  - [ ] Perform temperature / vector scaling calibration per model family inside nested folds.
  - [ ] Optimize ensemble blend weights (weighted log-prob averaging or regularized multinomial LR).
  - [ ] Apply base decision rule: $\arg\max_k P_{\text{blend}}(y=k \mid x) / \pi_{\text{train}}(k)$.
  - [ ] Transductive Sinkhorn-Knopp testing (strictly conditional on test prompt grouping):
    - Implement unbalanced Sinkhorn / dummy slack column for within-prompt mutual exclusivity.
    - Validate global marginal estimation (EM/BBSE) under simulated prior shift before adopting.

### Phase 8: Offline Packaging & Kaggle Datasets
- **Objective:** Package all code, weights, adapters, tokenizers, and wheels for offline Kaggle execution.
- **Workflow Command:** `/package-kaggle`
- **Checklist:**
  - [ ] LoRA packaging: 1 base model + 5 fold adapters (safetensors).
  - [ ] Encoder packaging: 5 full fold checkpoints.
  - [ ] Package joblib pipelines, blend weights, prior vectors, `labels.json`, and any missing wheels (e.g. `peft`).
  - [ ] Create Kaggle Datasets via `kaggle datasets create -p <dir>`.
  - [ ] Build `kaggle_inference/` standalone inference notebook / script.
  - [ ] Execute offline verification gates on Kaggle T4:
    - Test `fp16` numerical stability (assert no NaNs).
    - Fold 0 offline reproduction test: assert $|\Delta \text{BA}| \le 0.002$ and prediction agreement $\ge 99.5\%$.
    - 5-fold load-and-smoke-test with synthetic dummy batch.

### Phase 9: Final Submission Validation & Timed Inference
- **Objective:** Validate submission file integrity, perform end-to-end timed run, and lock final submission selections.
- **Workflow Command:** `/validate-submission`
- **Checklist:**
  - [ ] Execute full end-to-end inference run on Kaggle offline notebook on `test.csv`.
  - [ ] Verify execution time is $\le 4.5$ hours (safe buffer under 9.0h limit).
  - [ ] Validate `submission.parquet` using `scripts/validate_submission.py`:
    - Columns exactly `['id', 'model']`.
    - Row count equals `test.csv`.
    - No NaN / missing values.
    - All predicted labels exist in `configs/labels.json`.
    - Check predicted class distribution; flag any zero-prediction classes.
  - [ ] Select 2 final submission slots on Kaggle UI:
    1. Highest grouped-CV ensemble blend.
    2. Simpler, highly-regularized robust alternative.

---

## 3. Phase Transition Protocol

1. **Gate Compliance:** An agent or developer must not transition to Phase $N+1$ until all checklist items and the formal exit gate for Phase $N$ are satisfied.
2. **Failure Recovery:** If an experiment fails its improvement threshold:
   - Do not discard the code silently.
   - Record the failure and exact metrics in `results.csv`.
   - Update `memory.md` with lessons learned.
   - Revert to the reference branch before proceeding to alternative experiments.
3. **Artifact Integrity:** All phase outputs must pass their respective validation scripts (`check_folds.py`, `validate_oof.py`, `validate_submission.py`) before being considered complete.
