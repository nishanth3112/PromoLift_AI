# PromoLift AI

Production-grade causal machine learning platform for retail promotion incrementality, uplift modeling, and campaign budget optimization.

## Dataset

X5 RetailHero Uplift Modeling Dataset

Raw files are expected under `data/raw/` (`clients.csv`, `products.csv`,
`purchases.csv`, `uplift_train.csv`, `uplift_test.csv`,
`uplift_sample_submission.csv`).

**Raw X5 data is not version controlled.** `data/raw/`, `data/interim/`, and
`data/processed/` are gitignored — you must place the files there yourself.
`purchases.csv` is ~4.2 GB; never load it eagerly, use the Polars lazy
loaders in `src/promolift/data/loader.py`.

## Development

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Run tests:

```bash
uv run pytest
```

Run Ruff (formatting + linting):

```bash
uv run ruff format .
uv run ruff check .
```

Verify the causal-ML stack (imports, plus tiny mini-fits with `--fit`):

```bash
uv run python scripts/check_env.py --fit
```

## Experiment tracking

Runs are tracked with MLflow in a local SQLite store at `mlruns/mlflow.db`
(gitignored) — no MLflow server or account is needed. Use
`promolift.tracking.mlflow_tracking.start_run`, which pins artifacts to
`mlruns/artifacts/` regardless of working directory and tags every run with
its git commit, a `git_dirty` flag, the `uv.lock` hash, and a raw-data
fingerprint. Filter out `git_dirty=true` runs before choosing a final model.

Log the experiment-validity baseline (needs the raw data):

```bash
uv run python scripts/log_data_validation.py
```

Browse runs from the repository root:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

Open http://127.0.0.1:5000 — not `localhost:5000`, which on macOS can hit the
AirPlay Receiver on the same port and return 403. Add `--port 5001` if 5000 is
taken.

To log to a shared tracking server instead, set `MLFLOW_TRACKING_URI` (e.g. in
a gitignored `.env`); no code changes are needed.

## Status

Under active development.