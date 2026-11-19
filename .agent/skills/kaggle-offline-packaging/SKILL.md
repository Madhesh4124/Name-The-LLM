---
name: kaggle-offline-packaging
description: Use when preparing artifacts, datasets or the notebook for offline inference on Kaggle, or validating a submission file.
---
# Kaggle offline packaging

## Artifacts to upload as Kaggle Datasets
- Fold-model weights (safetensors; fp16 only if verified on a Kaggle T4, which lacks native bf16).
  For LoRA/QLoRA models, prefer shipping 1 base model + 5 lightweight LoRA adapters over
  5 merged multi-GB checkpoints. Run an fp16 inference stability test before packaging.
- Tokenizers, fitted TF-IDF vectorizers and LR models (joblib), LightGBM models.
- Blend weights, calibration parameters, class prior vector, labels.json.
- Any wheels the Kaggle image lacks, built for the same Python version (peft is a small pure-Python wheel; compare shipping adapters + one base model against merged models, which are much larger).

## Upload (confirm with the user first)
Create dataset-metadata.json (title, id "$KAGGLE_USER/<slug>", licenses) inside the dataset folder, then:
`kaggle datasets create -p <folder>` (or `kaggle datasets version -p <folder> -m "msg"` for updates).

## Notebook checklist
- Reads test.csv, runs every model, fold-averages, blends, divides by class prior, argmax, writes submission.parquet (id, model).
- Ends by running the validation logic from scripts/validate_submission.py.
- Run once with internet off and record runtime.
- Reproduction test: run one held-out fold through the same code; balanced accuracy must match the recorded CV value.
