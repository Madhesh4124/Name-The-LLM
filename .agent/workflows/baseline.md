---
description: Stage 1 sparse baseline (char + word TF-IDF + LR) with grouped CV
---
1. Read the oof-contract and grouped-cv skills.
2. Apply longest-first whitespace/markdown mapping from `src/text_utils.py` (rule §Technical-invariants-1, dropping single-char `—`) with line-anchored regex for bullets (`(?m)^[-*]\s+` → `\uE005`) and numbered lists (`(?m)^\d+\.\s+` → `\uE006`) to texts before vectorization.
3. Build a `sklearn.pipeline.Pipeline` per fold with `FeatureUnion`:
   - `TfidfVectorizer(analyzer='char', ngram_range=(2, 6), lowercase=False, sublinear_tf=True)`
   - `TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), lowercase=False, sublinear_tf=True)`
   - `TfidfVectorizer(analyzer='word', ngram_range=(1, 3), lowercase=False, sublinear_tf=True)`
   - Custom transformer inside the Pipeline extracting layout scalar counts (double spaces `  `, trailing whitespace `(?m)\s+$`, and indentation)
   - Classifier: `LogisticRegression(max_iter=1000, C=1.0)` or in-fold `CalibratedClassifierCV(LinearSVC(C=1.0, dual=False))`
   Fit strictly on training-fold rows only.
4. Write OOF and fold-averaged test probs to artifacts/<exp>/ using the contract.
// turbo
5. Run: python .agent/skills/oof-contract/scripts/validate_oof.py artifacts/<exp>/oof.npy folds/folds.parquet configs/labels.json --save-dir artifacts/<exp>
6. Report balanced accuracy (argmax and prior-divided), per-class recall and the confusion matrix. Record in results.csv. This is the reference every other branch must beat.
