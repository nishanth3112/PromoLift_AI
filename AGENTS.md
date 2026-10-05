# AGENTS.md

## Overview

PromoLift AI is a causal ML platform for retail promotion incrementality. It estimates customer-level uplift on the X5 RetailHero dataset to target customers whose purchases an SMS promotion changes.

## Tooling

- Python 3.12 and dependencies: `uv`
- Data processing: Polars, lazy by default
- Formatting/linting: Ruff
- Tests: pytest
- Tracking: MLflow (local SQLite; Databricks via `.env`)

## Verification

- `uv sync`
- `uv run ruff format --check .`
- `uv run ruff check .`
- `uv run pytest`: excludes `integration` tests, which need real data

Run all before completion; CI runs them on macOS, Linux, and Windows.

## Development Rules

Keep reusable logic in `src/promolift/`; notebooks and `scripts/` only orchestrate. Use type hints, `pathlib`, and small testable functions; avoid premature abstractions and new dependencies; never hardcode absolute paths. Never commit, modify, or delete raw files in `data/raw/`. `purchases.csv` is ~4.2 GB: use the lazy loaders in `promolift.data.loader`, never eager reads. Inspect before modifying, preserve existing configuration, and never commit or push automatically.

## Architecture

`data/` owns loading and the canonical split, `validation/` data and experiment checks, `features/` leakage-free client features, `evaluation/` uplift metrics, `optimization/` campaign economics and targeting depth, `serving/` the packaged model and its registry, and `tracking/` MLflow configuration and lineage. Start runs with `tracking.mlflow_tracking.start_run`, not bare `mlflow`. The test split is locked and its one final evaluation is done (`docs/final_results.md`): never load it again.

## Definition of Done

Add deterministic tests on synthetic temp fixtures. Keep CI free of raw data, credentials, and external services. Document new configuration and scripts in the README. Build only the current roadmap phase; don't implement deployment or monitoring unless asked. Never commit `.env`, credentials, data, or `mlruns/`.
