---
trigger: always_on
---
# Project: Name That LLM (Kaggle, 18-class LLM authorship attribution)

- Metric: balanced accuracy. Final decision rule: argmax_k P_train(y=k|x) / pi_train(k). Never estimate or use test priors for the decision.
- Code competition: the final inference notebook is offline (no internet, <= 9 h). Training happens on GCP; only artifacts cross over.
- Source of truth: docs/PLAN.md. If a request conflicts with it, say so before acting.

## Non-negotiables
1. Never fit anything (TF-IDF, scalers, calibrators, stackers) on rows outside the training folds of the current split.
2. Folds are created once (folds/folds.parquet) and never regenerated.
3. Every model writes OOF probs (N x 18) and fold-averaged test probs (n_test x 18) following the oof-contract skill. No other format.
4. Report balanced accuracy, per-class recall and the 18x18 confusion matrix for every experiment. Never accuracy alone.
5. Keep a branch only if grouped-CV balanced accuracy improves beyond fold-to-fold noise as defined in docs/PLAN.md (paired per-fold Δ_f = score_f(new) - score_f(reference), accept only if mean(Δ) > sd(Δ) and Δ_f > 0 in at least 4 of 5 folds). Otherwise log it as cut.
6. Never tune on the public leaderboard.
7. State library versions used; pin to the Kaggle image versions.
8. Do not claim something works without running it. Mark unverified items [verify].

## Technical invariants
1. **Whitespace & markdown token mapping:** Before calling `TfidfVectorizer(analyzer='char', lowercase=False)`,
   map multi-char tokens to single private-use Unicode code points using longest-first matching (dropping
   single-char `—`) so sklearn's char analyzer does not collapse them (e.g., `\n\n` → `\uE000`, `\n` → `\uE001`,
   `###` → `\uE002`, `##` → `\uE003`, `**` → `\uE004`, and line-anchored regex for bullets `(?m)^[-*]\s+` → `\uE005`
   and numbered lists `(?m)^\d+\.\s+` → `\uE006`).
   Apply the same mapping at inference time. Keep the mapping in `src/text_utils.py`.
2. **Fold-strict preprocessing:** All vectorizers, scalers, calibrators, and stackers MUST be fit
   inside a `sklearn.pipeline.Pipeline` or strictly on training-fold rows only. No exceptions.
3. **Adapter packaging & T4 precision:** For LoRA/QLoRA decoder models, prefer shipping 1 base model
   + 5 lightweight LoRA adapters over 5 merged multi-GB checkpoints. Require an `fp16` inference
   stability test on a Kaggle T4 (which lacks native `bf16`) before packaging.
4. **Compute fallback (Cloud Run Jobs):** If Compute Engine L4 Spot quota is delayed, use
   Cloud Run Jobs with `--gpu=1 --no-gpu-zonal-redundancy` (auto-grants 3×L4 per region, no
   manual quota request). Still checkpoint to GCS and obey the same cost-safety rules.

## Conventions
- src/ layout, one configs/<exp>.yaml per experiment, seed 42.
- Label order = configs/labels.json. Never rely on implicit class order across files.
