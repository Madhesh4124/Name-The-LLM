---
description: Package trained artifacts into Kaggle datasets and prepare the offline notebook
---
1. Read the kaggle-offline-packaging skill.
2. Stage artifacts in kaggle_upload/<dataset>/ (weights as safetensors, tokenizers, joblib models, blend weights, class prior, labels.json).
3. Write dataset-metadata.json and show the upload command. Ask for confirmation before uploading.
4. Update kaggle_inference/ notebook code to load only these artifacts.
5. Run the reproduction test: feed one held-out fold through the inference code and compare balanced accuracy to the recorded CV value.
6. Report mismatches. Do not proceed to submission until it matches.
