---
description: Package trained artifacts into Kaggle datasets and prepare the offline notebook
---
1. Read the kaggle-offline-packaging skill.
2. Stage artifacts in kaggle_upload/<dataset>/:
   - LoRA models: **1 base model + 5 LoRA adapters** (safetensors); non-LoRA models (e.g. RoBERTa): 5 full fold checkpoints; probe/base LMs as separate datasets as needed.
   - Tokenizers, fitted vectorizers and classifier joblib models, blend weights, class prior vector, labels.json.
3. Write dataset-metadata.json and show the upload command. Ask for confirmation before uploading.
4. Update kaggle_inference/ notebook code to load only these artifacts.
5. Run the offline reproduction test on held-out Fold 0:
   - Assert Fold 0 balanced accuracy delta ≤ 0.002 compared to recorded CV value.
   - Assert prediction agreement ≥ 99.5% (and mean absolute probability difference < 1e-2 for fp16).
   - Perform a load-and-smoke-test for all 5 folds/adapters with a dummy batch.
6. Report mismatches or test failures. Do not proceed to submission until all checks pass.
