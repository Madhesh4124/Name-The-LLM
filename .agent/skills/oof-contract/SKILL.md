---
name: oof-contract
description: Use whenever a model writes predictions or when blending, stacking or reporting results. Defines the artifact format and validation.
---
# OOF artifact contract

For experiment <exp>, write to artifacts/<exp>/ (and mirror to the bucket under oof/<exp>/ and test/<exp>/):
- oof.npy: float32, shape (N_train, 18), rows aligned to folds/folds.parquet order, columns in configs/labels.json order, rows sum to 1.
- test.npy: float32, shape (N_test, 18), mean of the 5 fold models' probabilities, rows aligned to test.csv order.
- config.yaml (copy), metrics.json (balanced accuracy for argmax and prior-divided, per-class recall), confusion_matrix.csv, env.txt (library versions), git_commit.txt (mirrored under oof/<exp>/).

## Validation
Run: python .agent/skills/oof-contract/scripts/validate_oof.py artifacts/<exp>/oof.npy folds/folds.parquet configs/labels.json --save-dir artifacts/<exp>
Reject any artifact that fails. Append one row per experiment to results.csv.

## Blending
Blend only OOF files that pass validation. Calibrate per model first. Keep the blender regularized and evaluate it with nested folds.
