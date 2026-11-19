---
name: kaggle-offline-packaging
description: Use when preparing artifacts, datasets or the notebook for offline inference on Kaggle, or validating a submission file.
---
# Kaggle offline packaging

## Artifacts to upload as Kaggle Datasets
- Model weights (safetensors):
  - **LoRA models:** 1 base model + 5 LoRA adapters (lightweight adapters instead of 5 multi-GB merged models).
  - **Non-LoRA models (e.g., RoBERTa, DeBERTa):** 5 full fold checkpoints.
  - Probe/base pretrained language models uploaded as separate Kaggle datasets as needed.
  - Require an `fp16` inference stability test on a Kaggle T4 (which lacks native `bf16`) before packaging.
- Tokenizers, fitted TF-IDF vectorizers and LR models (joblib), LightGBM models.
- Blend weights, calibration parameters, class prior vector, labels.json.
- Any wheels the Kaggle image lacks, built for the same Python version (e.g., peft pure-Python wheel).

## Upload (confirm with the user first)
Create dataset-metadata.json (title, id "$KAGGLE_USER/<slug>", licenses) inside the dataset folder, then:
`kaggle datasets create -p <folder>` (or `kaggle datasets version -p <folder> -m "msg"` for updates).

## Notebook checklist
- Reads test.csv, runs every model, fold-averages, blends, divides by class prior, argmax, writes submission.parquet (id, model).
- Ends by running the validation logic from scripts/validate_submission.py.
- Run once with internet off and record runtime (target: well under 9 h).
- **Reproduction test on held-out Fold 0:**
  - Fold 0 balanced accuracy delta ≤ 0.002 compared to CV recorded score.
  - Prediction agreement ≥ 99.5% (and mean absolute probability difference < 1e-2 for `fp16`).
  - **Smoke test:** Load all 5 fold models/adapters and run a dummy batch to ensure weights and keys resolve without error.
