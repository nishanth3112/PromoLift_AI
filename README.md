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

### Shared tracking on Databricks

One-time setup per machine (macOS; Windows: `winget install Databricks.DatabricksCLI`):

```bash
brew tap databricks/tap
brew trust --formula databricks/tap/databricks   # Homebrew 5 requires trusting third-party taps
brew install databricks
databricks auth login --host https://<workspace>.cloud.databricks.com --profile promolift
```

Use the workspace URL (`https://dbc-....cloud.databricks.com`), not the
`accounts.cloud.databricks.com` account console. Then create `.env` in the
repository root:

```bash
MLFLOW_TRACKING_URI=databricks://promolift
PROMOLIFT_EXPERIMENT_ROOT=/Shared/promolift
```

Databricks requires experiment names to be workspace paths, so
`PROMOLIFT_EXPERIMENT_ROOT` prefixes them (e.g.
`/Shared/promolift/promolift-data-validation`). `.env` is only loaded when
asked for, so plain `uv run` and CI keep using the local store:

```bash
uv run --env-file .env python scripts/log_data_validation.py
```

View runs under **Experiments** in the Databricks workspace.

## Train/validation/test split

Every model is trained and evaluated on one canonical client-level split of
`uplift_train`: stratified 60/20/20 on `treatment_flg × target`, seed 42,
stored at `data/processed/split_assignment.parquet` (gitignored). The shared
tracking server holds the canonical copy, and every MLflow run is tagged with
the `split_sha256` it used. **The test split is locked** — never used for
model selection or tuning, only for the final evaluation.

Get the split on a new machine (verifies the download against its recorded hash):

```bash
uv run --env-file .env python scripts/fetch_split.py
```

Creating it (done once; refuses to replace an existing split without `--force`,
which would invalidate every result evaluated on the old one):

```bash
uv run --env-file .env python scripts/make_split.py
```

## Evaluation

`promolift.evaluation.report.evaluate_ranking` is the one way to judge a model's
uplift scores on a split: normalized Qini AUC (primary model-selection metric),
uplift AUC, uplift at 10/20/30/50% targeted, stratified-bootstrap CIs, a
random-ranking noise floor, deciles, and the Qini curve. `log_evaluation` logs
it to the active MLflow run with split-prefixed names (`val_qini_auc`) and
Qini/decile plots. Compute the noise floor once per split with
`random_ranking_noise_floor` and pass it to every model's evaluation; compare two
models with `uncertainty.paired_comparison`. Evaluating on `test` raises unless
`final_evaluation=True`.

## Baseline models

Models implement `promolift.models.base.UpliftModel` and are built by name via
`promolift.models.registry.build_model`: `random` (sanity floor), `response`
(P(buy | contacted), trained on treated clients — the traditional approach),
and the scikit-uplift `s_learner`, `t_learner`, `class_transformation`, all on
one shared LightGBM configuration. Train all of them on the canonical split and
log a validation leaderboard (per-model runs plus a `leaderboard` run with
overlaid Qini curves and paired comparisons):

```bash
uv run --env-file .env python scripts/train_baselines.py            # ~8-9 min
uv run --env-file .env python scripts/train_baselines.py --n-bootstrap 200   # quick look
```

Models aren't stored: every run is reproducible from its commit, split hash,
and seed. The test split can't be loaded for training or evaluation outside
the final evaluation.

## Status

Under active development.