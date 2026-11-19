---
description: Stage 1 sparse baseline (char + word TF-IDF + LR) with grouped CV
---
1. Read the oof-contract and grouped-cv skills.
2. Apply the PUA whitespace/markdown mapping from `src/text_utils.py` (rule §Technical-invariants-1)
   to train and test texts before any vectorizer sees them.
3. Build a sklearn Pipeline per fold: TfidfVectorizer (char, 2-6, lowercase=False) + word
   TfidfVectorizer (1-3, lowercase=False) → LogisticRegression(max_iter=1000, C=1.0).
   Fit only on training-fold rows.
4. Write OOF and fold-averaged test probs to artifacts/ using the contract.
// turbo
5. Run: python .agent/skills/oof-contract/scripts/validate_oof.py artifacts/<exp>/oof.npy folds/folds.parquet configs/labels.json --save-dir artifacts/<exp>
6. Report balanced accuracy (argmax and prior-divided), per-class recall and the confusion matrix. Record in results.csv. This is the reference every other branch must beat.
