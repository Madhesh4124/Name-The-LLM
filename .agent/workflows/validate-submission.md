---
description: Validate submission.parquet against test.csv before submitting
---
// turbo
1. Run: python .agent/skills/kaggle-offline-packaging/scripts/validate_submission.py submission.parquet <path-to-test.csv> configs/labels.json
2. Report the result and the predicted class distribution. Flag any class with zero predictions.
