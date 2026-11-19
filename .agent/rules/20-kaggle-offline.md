---
trigger: glob
globs: kaggle_inference/**
---
# Rules for the Kaggle offline inference notebook

- No internet: no pip install from PyPI, no model downloads, no API calls. Read only from /kaggle/input/... and the competition data.
- Input: /kaggle/input/competitions/name-that-llm/test.csv. Output: submission.parquet with columns id, model.
- Load only artifacts produced by the oof-contract and kaggle-offline-packaging skills.
- Apply: fold-average -> blend -> divide by training class prior -> argmax.
- Run scripts/validate_submission.py logic at the end of the notebook and fail loudly on any problem.
- Total runtime target: well under 9 h; log time per stage.
