---
name: grouped-cv
description: Use when creating or checking CV splits, reconstructing prompt groups, or when any step might leak information across folds.
---
# Grouped CV

## Rules
- Use StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42). Group = prompt. Stratify = model.
- If the data has prompt ids, use them and skip clustering.
- If not, reconstruct groups: strip markdown and formatting, embed passages with a sentence-embedding model, cluster with a conservative threshold. Err toward merging; under-merging leaks silently.
- Cluster purity check: each cluster must contain many different models. A cluster dominated by one model means you are grouping by style, not prompt. Report the distribution of distinct models per cluster.
- Folds are created once and saved to folds/folds.parquet (id, fold, group, model). Never regenerate.
- Everything fitted (vectorizers, scalers, calibrators, stackers) is fit on training folds only, inside the fold loop or a sklearn Pipeline. Stackers and calibrators use nested or OOF-only data.

## Checks to run
- No group appears in more than one fold.
- Per-fold class counts are close to balanced.
- A model trained on shuffled labels scores near chance (1/18 balanced accuracy).

Run: `python .agent/skills/grouped-cv/scripts/check_folds.py folds/folds.parquet configs/labels.json`
