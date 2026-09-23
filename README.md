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

## Status

Under active development.