# AGENTS.md

## Project

PromoLift AI is a production-grade causal ML platform for retail promotion incrementality and campaign budget optimization. It uses the X5 RetailHero uplift dataset to estimate customer-level treatment effects and ultimately identify customers whose purchasing behavior is changed by promotions.

## Engineering Rules

Use Python 3.12 and `uv` for environments, dependencies, and reproducibility. Keep reusable production logic under `src/promolift/`; notebooks are for exploration and reporting only. Use type hints, concise docstrings, small testable functions, and `pathlib` for paths. Avoid premature abstractions and unnecessary dependencies.

Raw X5 files live under `data/raw/` and MUST NEVER be committed, modified, or deleted by agents. Never hardcode local absolute paths. `purchases.csv` is approximately 4.2 GB; never load it blindly into Pandas. Prefer Polars lazy operations (`scan_csv`) and memory-efficient processing.

## Quality

Use `pytest` for testing and Ruff for linting/formatting. Unit tests must be deterministic, fast, and independent of the real dataset. Use temporary synthetic fixtures. Real-data integration tests must be optional and excluded from CI.

Run before completion:

`uv sync`
`uv run ruff format --check .`
`uv run ruff check .`
`uv run pytest`

CI must not require raw data, AWS credentials, secrets, or external services.

## Scope

Done: ingestion, validation, data auditing, feature engineering, experiment tracking (MLflow: local SQLite or Databricks), and the canonical train/val/test split.

Current work: uplift evaluation (ranking metrics, uncertainty, noise floor), then baseline and causal uplift models.

The test split is locked: never use it for model selection or tuning -- only for the final evaluation.

Do not implement AWS deployment, budget optimization, or monitoring unless explicitly requested. Preserve existing configuration, inspect before modifying, and never commit or push automatically.
