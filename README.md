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

## Features

Client features are built from `clients.csv`, `products.csv`, and the
purchase history up to its last transaction, in independently buildable
groups (`promolift.features.build.FeatureGroup`): the 19 Phase 10 features
(`demographics`, `purchase_behavior`, `product_mix`) plus
`promo_responsiveness`, `purchase_dynamics`, `basket_store`, and
`category_spend`. `promolift.features.store.load_feature_table(groups)` builds
a table once (~2 min for all groups) and caches it as parquet in
`data/interim/features/` (gitignored); delete that folder or pass
`rebuild=True` to force a rebuild. The cache key covers the feature code, raw
data, Polars version, and groups, so editing a builder or the data misses the
cache automatically.

Runs tag `feature_cache_key` (reproducible: same code + data + groups → same
key) and `feature_table_sha256` (the exact values used). Rebuilt tables differ
at ~1e-13 relative in float aggregates because Polars sums in parallel, so the
content hash is not stable across rebuilds — the key is the identity to
reproduce from.

### Feature ablation

Which groups to keep is decided by 3-fold CV **inside train** (val and test
untouched), with the Phase 10 tuned hyperparameters held fixed for
`class_transformation`, `s_learner`, and `dr_learner`. Configs: base, base +
each new group alone, and all groups. A group is kept if its paired
out-of-fold Qini gain over base (mean across the three models) has a 95% CI
above 0; all groups are chosen over the kept set only if significantly
better. Runs log to `promolift-feature-ablation`:

```bash
caffeinate -is uv run --env-file .env python scripts/run_ablation.py   # ~15 min
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

## Models

Models implement `promolift.models.base.UpliftModel` and are built by name via
`promolift.models.registry.build_model(name, overrides=...)`:

- `random` (sanity floor) and `response` — P(buy | contacted), trained on
  treated clients: the traditional approach this project argues against.
- scikit-uplift `s_learner`, `t_learner`, `class_transformation` on one shared
  LightGBM configuration.
- `x_learner`, `dr_learner`, `causal_forest` (econml) and `uplift_rf`
  (causalml), which receive a train-fitted NaN-free matrix
  (`models.preprocessing.NumericEncoder`: medians + missing flags + one-hot).

### Tuning

Hyperparameters are tuned by 3-fold cross-validation **inside train** (val and
test untouched): Optuna for the LightGBM-based models, whose first trial is
always the defaults, and small grids for the two forests. Uplift models are
tuned on Qini; the response model on purchase AUC among treated clients, its
own goal. Results are committed in `configs/tuned_params.yaml`:

```bash
caffeinate -is uv run --env-file .env python scripts/tune_models.py   # ~50 min, keep the Mac awake
```

### Training and the leaderboard

Train every model (and, with `--tuned`, a `<model>_tuned` variant from the
committed params) on train, evaluate on val, and log per-variant runs plus a
`leaderboard` run with Qini curves, paired comparisons, and
`tuning_effect.csv` (tuned vs default per model):

```bash
caffeinate -is uv run --env-file .env python scripts/train_models.py --tuned   # ~30 min
uv run --env-file .env python scripts/train_models.py --n-bootstrap 200          # quick look
```

Results:

- [docs/baseline_results.md](docs/baseline_results.md) — the response model
  is worse than random targeting; the uplift baselines beat it decisively.
- [docs/advanced_models_results.md](docs/advanced_models_results.md) —
  advanced models and tuning reach the same ceiling (Qini ≈ 0.015–0.016, ten
  variants statistically tied); tuning helped only the weakest defaults; the
  features are now the bottleneck.

Models aren't stored: every run is reproducible from its commit, split hash,
and seed. The test split can't be loaded for training or evaluation outside
the final evaluation.

## Status

Under active development.