# Name That LLM

Kaggle 18-class LLM authorship-attribution competition.

## Setup
1. Export the project plan as `docs/PLAN.md` (source of truth).
2. Copy `configs/gcp.env.example` → `configs/gcp.env` and fill in your values.
3. Download competition data: `kaggle competitions download -c name-that-llm -p data/`.
4. Run the EDA workflow, then make-folds, then baseline.

See `.agent/workflows/` for the full pipeline and `.agent/rules/` for invariants.
