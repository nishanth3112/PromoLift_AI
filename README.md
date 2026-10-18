# PromoLift AI

Production-grade causal machine learning platform for retail promotion incrementality, uplift modeling, and campaign budget optimization.

## Dataset

X5 RetailHero Uplift Modeling Dataset

Raw files are expected under `data/raw/` (`clients.csv`, `products.csv`,
`purchases.csv`, `uplift_train.csv`, `uplift_test.csv`,
`uplift_sample_submission.csv`).

**Raw X5 data is not version controlled.** `data/raw/`, `data/interim/`, and `data/processed/` are gitignored.
`purchases.csv` is ~4.2 GB; never load it eagerly, use the Polars lazy
loaders in `src/promolift/data/loader.py`.

## Development

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

`uv sync` installs the package's runtime dependencies plus the `research`
group (advanced models, tuning, plots) and `dev` group by default. The built
wheel declares only the runtime dependencies (loading, features, scoring),
which keeps the Databricks scoring job's install small.

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

Because the ablation holds base-tuned params fixed, a fairness check re-tunes
the three models on all groups (written to
`configs/tuned_params_all_features.yaml`, leaving the base params alone) and
compares tuned-all against tuned-base on the same folds:

```bash
caffeinate -is uv run --env-file .env python scripts/tune_models.py --feature-set all \
    --models class_transformation s_learner dr_learner                       # ~20 min
caffeinate -is uv run --env-file .env python scripts/compare_tuned_features.py   # ~10 min
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
- [docs/feature_engineering_results.md](docs/feature_engineering_results.md) —
  47 new features in four groups don't beat the 19 base features, even with
  hyperparameters re-tuned for them; the models keep the base features.
- [docs/final_results.md](docs/final_results.md) — the one-time test
  evaluation: the chosen model clearly beats likely-buyer targeting and
  modestly beats random targeting (test Qini +0.0078); these are the numbers
  to quote.

Models aren't stored: every run is reproducible from its commit, split hash,
and seed. The test split can't be loaded for training or evaluation outside
the final evaluation.

### Final evaluation

The test split is evaluated exactly once. `scripts/final_evaluation.py` chooses
the final model from the validation leaderboard by a rule fixed before test is
seen (`models/selection.py`: among uplift variants whose paired Qini CI with
the best includes zero, the cheapest to fit), refits it and the reference
models (`s_learner_tuned`, `causal_forest`, `response`, `random`) on
train + val with the leaderboard's parameters, and scores them on test in one
run in `promolift-final-evaluation`. The references are context only; the
choice is never changed after test is seen.

The official run refuses to start from a dirty or untracked working tree, a
non-shared tracking store, or a split or raw data that differ from the
leaderboard run's. It also refuses if a final evaluation already exists for
the split. There is no override; a failed run can only be redone by deleting
it in MLflow by hand. Models are fitted before the run opens, and test is
loaded only after, so any look at test is on record.

```bash
uv run --env-file .env python scripts/final_evaluation.py
```

`--dry-run` takes the same path on train -> val (test is never loaded, the
guard doesn't apply) and prints the leaderboard's val Qini beside each model,
which a correct pipeline reproduces. As a wiring check it defaults to 20
bootstrap resamples and 20 noise-floor rankings (the point estimates don't
depend on them). Log it to a scratch store and read the leaderboard from
Databricks:

```bash
MLFLOW_TRACKING_URI=sqlite:///<scratch>/mlflow.db uv run python scripts/final_evaluation.py \
  --dry-run --leaderboard-tracking-uri databricks://promolift
```

Options: `--leaderboard-run` (default: the Phase 10 leaderboard run),
`--references`, `--n-bootstrap`, `--n-rankings`, `--seed`, and `--workers`
(bootstrap evaluations run in parallel processes, default every core, with
results identical to a sequential run).

## Budget targeting

`scripts/targeting_analysis.py` turns the chosen model's ranking into a send
list: how deep to text down it for the most profit. Texting the top share `k`
of clients earns `k × (margin × uplift@k − SMS cost)` per client, so the best
depth depends only on the **break-even uplift** = SMS cost / margin per extra
purchase: keep texting while the next clients' uplift exceeds it.

- Every train + val client is scored out of fold (5-fold CV, ranks pooled
  across folds), alongside the `response` and `random` rankings; test is never
  loaded.
- Profit per 1,000 clients is computed at every depth from 0% to 100%, with
  bootstrap CIs (`optimization/targeting.py`).
- The decision is the depth with the **highest expected profit**; depth 0
  (text nobody) is always a candidate. The shallowest depth whose profit is
  statistically tied with it (paired bootstrap) is reported as a lower-spend
  option, not the decision: the tied range is wide, and its shallow end can
  earn less than texting everyone when SMS is cheap.
- A sensitivity table repeats the decision for each break-even uplift in the
  config, so it answers for any real cost and margin without a re-run.

The economics live in `configs/business.yaml`. X5 RetailHero is a public
dataset, so every value there is an assumption: the margin is X5's reported
2019 gross margin (24.6%), the SMS cost an assumed price. Replace them with
real figures before relying on the profits:

| Key | Meaning |
|---|---|
| `currency` | Label for money values |
| `sms_cost` | Cost per SMS |
| `gross_margin` | Fraction of basket value kept as profit |
| `margin_per_purchase` | Profit per extra purchase; `null` = average transaction in `purchases.csv` × `gross_margin` |
| `budget`, `campaign_clients` | Optional budget cap and the clients it covers (set both or neither) |
| `break_even_grid` | Break-even uplifts for the sensitivity table |

```bash
uv run --env-file .env python scripts/targeting_analysis.py   # a few minutes
```

Logs one run to `promolift-targeting` (decision, profit curves, sensitivity
table, plot). Results: [docs/targeting_results.md](docs/targeting_results.md)
— with the assumed economics, texting the top 38% earns ~2.6× texting
everyone, and the response model loses money at every depth up to 61%. Options: `--references`, `--n-folds`, `--n-bootstrap`, `--seed`,
`--leaderboard-run`, `--leaderboard-tracking-uri`.

## Model packaging and registry

`scripts/register_model.py` turns the final model into a versioned artifact
that batch scoring can load without retraining. It refits the final model
(the leaderboard rule's choice, with its tuned parameters) on train + val --
the same fit that was evaluated once on test -- and logs it in
`promolift-models` as an MLflow `pyfunc` model with:

- **a feature contract** (`serving/pyfunc.py`): the training columns in order
  and the allowed levels of each categorical. Scoring input with a missing or
  extra column, text in a numeric column, or an unseen category is refused
  instead of scored silently wrong;
- **a signature** and an input example. The canonical input is every numeric
  column as float64 and categoricals as strings (`FeatureContract.to_input`);
- **a drift reference**: the training clients' features binned, with the
  expected share of every bin (see [Monitoring](#monitoring)), carried by the
  model itself and logged as `contract/drift_reference.json`.

The model is then reloaded from the store and must score bit-identically
before it is registered as a new version of
`promolift_ai.models.uplift_targeting` in Unity Catalog. Each version carries
the run's lineage tags (commit, split, raw data, feature code, final
evaluation run). The `champion` alias -- what batch scoring loads, as
`models:/promolift_ai.models.uplift_targeting@champion` -- moves only with
`--set-champion`:

```bash
uv run --env-file .env python scripts/register_model.py --set-champion
```

The current champion and its limits: [docs/model_card.md](docs/model_card.md).

Against Databricks the working tree must be clean. Registering needs
`USE CATALOG` on `promolift_ai` and `USE SCHEMA` + `CREATE MODEL` on
`promolift_ai.models`. Options: `--model-name`, `--set-champion`,
`--leaderboard-run`, `--leaderboard-tracking-uri`, `--seed`.

## Batch scoring

`scripts/score_clients.py` turns the champion model into a send list for the
whole client base:

1. builds the base features for every client in `clients.csv` (~400k, not
   only the experiment's 200k);
2. resolves `promolift_ai.models.uplift_targeting@champion` to one version
   and loads that version, so a batch never mixes models;
3. converts the features to the model's canonical input
   (`serving/scoring.py: model_input`, booleans to 0/1 as in training);
4. checks the batch for drift against the champion's training reference
   ([Monitoring](#monitoring)) and stops before writing anything if a feature
   drifted severely; then scores every client;
5. ranks clients by score (ties broken by `client_id`), assigns deciles, and
   selects the top `send_share`;
6. holds back a random `holdout_share` of the selected clients as a control
   group -- ranked like the rest, but not texted -- so the campaign's real
   effect can be measured ([Campaign evaluation](#campaign-evaluation)). The
   rest are flagged `send`.

`send_share` lives in `configs/business.yaml` (0.38: the max-expected-profit
depth of the targeting analysis); update it when the economics change.
`holdout_share` (0.10) is the cost of knowing whether the model works: about
one in ten of the best prospects is not texted. The draw is seeded by the
batch's run id, so it is reproducible for a batch and new for every batch,
and independent of the scores. Each
batch is written to the gitignored `data/interim/scoring/` as
`send_list_<timestamp>_<run>.parquet` and never overwritten:

| Column | Meaning |
|---|---|
| `client_id` | Client |
| `uplift_score` | Predicted change in purchase probability from the SMS |
| `rank` | 1 = highest score |
| `decile` | 1 = top 10% |
| `send` | Text this client |
| `holdout` | Selected but held back as the control group (never `send`) |
| `model_name`, `model_version` | The model version that scored the batch |
| `scored_at`, `scoring_run_id` | When, and the MLflow run that records it |

The run in `promolift-scoring` records the model version, counts, a
per-decile score summary, and every feature's drift (`psi_<feature>`
metrics, a `drift_status` tag, `drift/drift_report.csv`), and the numbers
sent and held back, not the client list.

```bash
uv run --env-file .env python scripts/score_clients.py   # about a minute
```

Options: `--model-name`, `--output-dir`.

### Scheduled scoring on Databricks

`databricks.yml` (a Databricks Asset Bundle) runs the same scoring as a
serverless job, `promolift-score-clients`. Its wheel task
(`promolift.serving.job:main`) reads the raw CSVs from the Unity Catalog
volume `/Volumes/promolift_ai/scoring/raw_data`, the business config from the
bundle's synced files, and the champion from Unity Catalog. Each batch is
**appended** to the Delta table `promolift_ai.scoring.targets` (with a
`deployed_git_commit` column; the append adds the `holdout` column, which is
null for batches written before it existed), and the view
`promolift_ai.scoring.targets_latest` shows only the most recent batch. Each
batch's drift report is appended to `promolift_ai.scoring.drift` -- also when
severe drift stops the batch. The weekly schedule is defined but paused: the
X5 snapshot is static.

One-time upload of the raw files to the volume (byte-for-byte copies):

```powershell
foreach ($f in "clients","products","purchases","uplift_train","uplift_test","uplift_sample_submission") {
  databricks fs cp "data/raw/$f.csv" "dbfs:/Volumes/promolift_ai/scoring/raw_data/$f.csv" --profile promolift
}
```

Deploy from a clean commit (the commit is recorded with every batch) and run:

```bash
databricks bundle validate --profile promolift
databricks bundle deploy --profile promolift        # builds the wheel, creates the job
databricks bundle run score_clients --profile promolift
```

The `dev` target uses development mode: the job is created as
`[dev <user>] promolift-score-clients`, runs as the deploying user, and its
schedule stays paused. Needs `USE SCHEMA`, `CREATE TABLE`, `SELECT`, `MODIFY`
on `promolift_ai.scoring` and `READ VOLUME` on its `raw_data` volume.

## Monitoring

### Feature drift

Every scoring batch is compared, feature by feature, with the training
clients of the model that scores it (`monitoring/drift.py`). The registered
model carries a drift reference: each feature binned, with the expected share
of every bin --

- numeric features with at most 10 distinct values (flags, small counts):
  one bin per value, plus "other";
- other numeric features: decile bins;
- `gender`: one bin per level, plus "other";
- and a missing-value bin for every feature.

A batch is binned the same way and each feature scored with the population
stability index, PSI = Σ (actual − expected) · ln(actual / expected).
Thresholds live in `configs/monitoring.yaml`:

| Key | Default | Effect |
|---|---|---|
| `drift.warn_psi` | 0.1 | Any feature above it tags the run `drift_status=warn` |
| `drift.fail_psi` | 0.25 | Any feature above it fails the run before a send list is written |

Models registered before drift references carried bin shares (version 1)
can't be checked; scoring with them stops and asks to re-register.

### Campaign evaluation

Drift says the inputs still look like training; it can't say the model still
works. A campaign can: the batch texted its top clients and held a random
share of them back, so the difference in purchase rate between the two groups
is the campaign's real effect on the clients the model chose.
`scripts/evaluate_campaign.py` measures it and compares it with the
prediction (`monitoring/campaign.py`):

- **Overall:** realized uplift (texted minus held-back purchase rate) with a
  95% bootstrap interval; the mean predicted uplift of the same clients; the
  gap between them; and a flag when the prediction falls outside the realized
  interval. Also the purchases the texts caused.
- **By decile:** the same comparison in each predicted-uplift decile of the
  batch, to show where the model over- or under-promises.
- **Ranking:** Qini AUC and uplift in the top k% of the targeted clients,
  from the real outcomes, with bootstrap intervals, next to the predicted
  uplift at the same depths.

Give it a results file (CSV or parquet), one row per targeted client with a
result:

| Column | Meaning |
|---|---|
| `client_id` | A client the batch texted or held back |
| `treated` | 1 if texted, 0 if held back -- must match the batch's `send` / `holdout` |
| `purchased` | 1 if the client bought in the campaign window |
| `scoring_run_id` | The scoring run (batch) the campaign ran |

```bash
uv run --env-file .env python scripts/evaluate_campaign.py --results results.csv
```

The send list is found in `data/interim/scoring/` from the file's scoring
run; pass `--batch <send_list.parquet>` for a batch scored elsewhere (export
it from `promolift_ai.scoring.targets` for a job's batch). Options:
`--n-bootstrap`, `--ranking-bootstrap`, `--confidence`, `--seed`. One run is
logged to `promolift-campaigns`: metrics, a per-decile table, a summary, and
a predicted-vs-realized plot.

It refuses results that would make the comparison invalid: clients the batch
never targeted, a `treated` flag that disagrees with the batch, a batch with
no control group, or an arm with no results. Clients missing from the results
are counted per arm and flagged: unequal gaps between the arms bias the
comparison, so chase them before trusting the number.

No campaign has been run on these clients, and every client with a known
outcome is already used for training or evaluation, so the code is tested on
simulated campaigns with a known effect (`monitoring/campaign_simulation.py`).
`--demo` runs the evaluation on a simulated campaign over the latest local
send list (`--effect-scale 1` = the model's promise is real, `0` = the texts
do nothing) and writes nothing:

```bash
uv run python scripts/evaluate_campaign.py --demo --effect-scale 0.5
```

## Explainability

`scripts/explain_model.py` shows which features drive a client's predicted
uplift (`explainability/`). It loads the champion, takes a random sample of
train + val clients (20,000 by default; the test split is never loaded), and
splits each client's score among the features with exact TreeSHAP -- LightGBM's
own `pred_contrib`, so no extra dependency. It reports:

- **Global importance:** mean absolute contribution per feature and its share
  of the total, with a direction (higher value raises or lowers the score, or
  mixed) and the effect of a missing value.
- **Segments:** the same importance inside predicted-uplift tiers (top 10%,
  next 30%, rest), gender, and age bands.
- **Dependence:** how the score moves along the leading features, bin by bin.

The champion is a class-transformation model, `uplift = 2 * P(z) - 1` from one
LightGBM classifier, and SHAP splits that classifier's log-odds. Each client's
contributions are rescaled by the one factor that makes them add up exactly to
their score, so they read in percentage points of uplift and
`base + sum(contributions) = score` holds for every client (checked, and
logged as `additivity_error`). Other models (S-learner, forests) score uplift
as a difference of predictions and are refused.

```bash
uv run --env-file .env python scripts/explain_model.py
```

Options: `--model-name`, `--n-rows`, `--seed`, `--no-log` (print only). One run
is logged to `promolift-explainability`: the importance, segment, and
dependence tables, two figures, a summary, and each feature's importance as a
metric; no client-level contributions. Results: `docs/explainability_results.md`.

Read it for what it is. SHAP describes what the model does, not what the
features cause: correlated features share credit, and a feature the model
leans on may not move buying. It is also a view of a weak model (see the model
card), so small differences between features are noise.

### Age check

SHAP shows the champion leans on `age` more than on any other feature; it can't
say how much worse the model would be without it.
`scripts/age_ablation.py` answers that (`explainability/feature_removal.py`):
the chosen model, with its tuned parameters held fixed, is fitted twice on the
same train + val folds -- with every feature, and without `age` and
`age_is_valid` -- and every client is scored by a fold model that never saw
them. The two rankings are compared on the same bootstrap resamples:

- **Ranking quality:** Qini AUC and uplift at the deployed targeting depth
  (`send_share`), each with a paired 95% interval for the difference.
- **Who gets texted:** the age mix of the top `send_share` of clients under each
  ranking against the whole population, and how strongly each score still
  follows age (rank correlation). Removing `age` does not remove age
  information -- tenure, spend, and points correlate with it -- so the second
  number says how much is left.

```bash
uv run --env-file .env python scripts/age_ablation.py
```

Options: `--drop` (other features to remove), `--depth`, `--n-folds`,
`--n-bootstrap`, `--seed`, `--no-log` (print only). One run is logged to
`promolift-feature-ablation`. The test split is never loaded. It measures a
trade-off; whether to drop `age` is a decision for the campaign owner, and this
is not a fairness or legal assessment.

## CI/CD

Four GitHub Actions workflows in `.github/workflows/`:

| Workflow | When | What |
|---|---|---|
| `ci.yml` | every push to main and every pull request | format, lint, and the unit tests with coverage on Ubuntu, macOS, and Windows |
| `bundle-validate.yml` | pull requests | `databricks bundle validate --strict` for the `dev` and `prod` targets |
| `deploy.yml` | push to main | validates, then deploys the `prod` target; waits for approval |
| `score-prod.yml` | manual | runs the production scoring job and waits for it |

**Coverage.** CI runs `pytest --cov` (branch coverage of `promolift`) and fails
below the floor `fail_under` in `pyproject.toml` (97%; measured 97.78%, the
margin absorbs OS-specific branches). The Ubuntu job writes the table to the
run's summary page and uploads `coverage.xml`. Raise the floor as coverage
grows; locally, `uv run pytest --cov` checks it too.

**The prod target** (`databricks.yml`) runs in production mode: the job is
owned by, deployed by, and run as the service principal `promolift-cicd`
(OAuth M2M, never a person's token), deploys only from `main`, and keeps one
copy under the principal's workspace folder. Its tables are separate from dev:
`promolift_ai.scoring_prod.targets`, `.targets_latest`, and `.drift`. The
schedule stays paused, as in dev, since the X5 snapshot is static. Pull-request
CI validates but never deploys; a merge to `main` deploys once the
`production` environment's reviewer approves. Run a production batch from the
Actions tab (`Score clients (prod)`), or locally with
`databricks bundle run score_clients --target prod --profile promolift`.

**Supply chain.** Every action is pinned to a commit SHA (the tag is in the
comment beside it) and Dependabot proposes updates weekly
(`.github/dependabot.yml`). The Databricks credentials are passed only to the
single step that runs a `databricks bundle` command, never to the whole job.

Deploys come from CI, not a laptop. `databricks bundle deploy --target prod`
works only from a checkout of `main`, as the principal or an admin.

### One-time setup

1. **Service principal.** Workspace admin settings, Identity and access,
   Service principals: add `promolift-cicd` and generate an OAuth secret
   (shown once; it expires, so rotate it before then).
2. **GitHub.** Repository secrets `DATABRICKS_HOST` (the workspace URL, no
   trailing slash), `DATABRICKS_CLIENT_ID` (the principal's application ID),
   and `DATABRICKS_CLIENT_SECRET`; and an environment named `production` with
   a required reviewer and `main` as its only deployment branch.
3. **Unity Catalog**, as a metastore or catalog admin (replace `<client-id>`
   with the application ID):

   ```sql
   GRANT USE CATALOG ON CATALOG promolift_ai TO `<client-id>`;
   GRANT USE SCHEMA ON SCHEMA promolift_ai.models TO `<client-id>`;
   GRANT EXECUTE ON FUNCTION promolift_ai.models.uplift_targeting TO `<client-id>`;
   GRANT READ VOLUME ON VOLUME promolift_ai.scoring.raw_data TO `<client-id>`;
   CREATE SCHEMA IF NOT EXISTS promolift_ai.scoring_prod;
   GRANT USE SCHEMA, CREATE TABLE ON SCHEMA promolift_ai.scoring_prod TO `<client-id>`;
   ```

   The model grant is `ON FUNCTION`: this SQL warehouse rejects `ON MODEL`.
   The principal creates the prod tables and view, so it owns them; grant
   people `SELECT ON SCHEMA promolift_ai.scoring_prod` to read them.

The dev job (`databricks bundle deploy`, no target) is unchanged: it runs as
the deploying user against `promolift_ai.scoring.*`, and that user needs the
grants listed under [Scheduled scoring on Databricks](#scheduled-scoring-on-databricks).

## Status

Under active development.