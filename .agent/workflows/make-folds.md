---
description: Build fixed grouped CV folds (folds/folds.parquet) using the grouped-cv skill
---
1. Read the grouped-cv skill and docs/eda_report.md.
2. If prompt ids exist, group by them. Otherwise reconstruct groups as the skill describes, merging conservatively.
3. Run the cluster purity check and report it. Stop and ask if clusters are dominated by one model.
4. Create StratifiedGroupKFold(5, shuffle=True, random_state=42) folds.
5. Save folds/folds.parquet with columns: id, fold, group, model. Refuse to overwrite an existing file.
// turbo
6. Print per-fold class counts and group counts, and assert no group appears in two folds.
