# Model card: `promolift_ai.models.uplift_targeting`

| | |
|---|---|
| **Registered model** | `promolift_ai.models.uplift_targeting` (Unity Catalog) |
| **Champion** | version 1 — load with `models:/promolift_ai.models.uplift_targeting@champion` |
| **Model** | `class_transformation_tuned`: scikit-uplift class transformation on LightGBM |
| **Output** | One uplift score per client: higher = the SMS changes their buying more. A ranking score, not a calibrated probability. |
| **Trained on** | train + val, 160,031 clients of the X5 RetailHero randomized SMS experiment |
| **Registered from** | run `17c67f99322a468d90bbfefa27a4e49d` in `promolift-models`, commit `8b212de`, `git_dirty=false` |

## Intended use

Rank clients for an SMS promotion by how much the message *changes* their
probability of buying, then text the top share chosen by the economics
(`docs/targeting_results.md`: the top 38% with the placeholder costs). The
score ranks; the depth comes from `configs/business.yaml` at scoring time,
so new costs or margins never require a new model version.

**Not for:** predicting who will buy (that is the response model, which loses
money when used for targeting), other channels or offers than the one in the
experiment, or clients outside the X5 RetailHero population without
re-validation.

## How it was chosen and trained

- **Selection:** among 17 variants on the validation leaderboard, 11 were
  statistically tied with the best; the rule fixed before test was seen
  picked the cheapest to fit, which was also the best on validation
  (`docs/advanced_models_results.md`, `docs/final_results.md`).
- **Training:** class transformation turns the experiment into one
  classification target (treated buyers and control non-buyers vs the rest),
  valid because treatment was randomized 50/50. Hyperparameters were tuned by
  3-fold CV inside train (`configs/tuned_params.yaml`): 150 trees, 8 leaves,
  learning rate 0.0157, minimum 165 clients per leaf, L2 5.13, 69% row and
  64% column subsampling.
- **Fit:** refitted on train + val with those parameters, seed 42. This is the
  exact fit evaluated once on test; LightGBM is deterministic, so
  re-registering from the same commit reproduces it.

## Inputs

The 19 base client features, computed at the end of the purchase history
(`features/`): demographics (`age`, `age_is_valid`, `gender`, `tenure_days`,
`has_redeemed`, `days_to_redeem`), purchase behavior (`frequency`,
`recency_days`, `monetary_total`, `monetary_avg`, `monetary_std`,
`monetary_cv`, `points_received_total`, `points_spent_total`,
`recent_frequency_30d`, `recent_monetary_30d`), and product mix
(`n_distinct_categories`, `alcohol_purchase_rate`, `own_trademark_rate`).

The model's **feature contract** (`contract/feature_contract.json`) requires
exactly these columns. Numeric columns are passed as float64 and `gender` as
a string in {F, M, U}; missing values are allowed (clients with no purchases
have null purchase features). Build the input with
`FeatureContract.to_input`. A missing or extra column, text in a numeric
column, or an unseen `gender` value is refused rather than scored.

## Performance (test split, evaluated once)

From `docs/final_results.md` (40,008 clients never used before):

| Metric | Value [95% CI] |
|---|---|
| Qini AUC | +0.0078 [+0.0014, +0.0141] (random band ±0.007) |
| Uplift, top 10% | +8.97pp — 90 extra purchases per 1,000 texted vs 33 at random |
| Uplift, top 30% | +5.33pp [+3.57, +7.26] |
| vs response model (Qini) | +0.0124 [+0.0021, +0.0219] |

## Limitations

- **Weak signal.** The gain is concentrated at the top: the first decile's
  uplift is +9.0pp, the rest +0.9 to +4.6pp, and on test the Qini curve
  rejoins random targeting by ~80% of clients.
- **Validation and cross-validated numbers are optimistic.** The test Qini is
  about half the validation one; quote the test numbers.
- **Tied alternatives.** `s_learner_tuned` and `causal_forest` are
  statistically indistinguishable on test; this model was chosen for cost,
  not because it is better.
- **One experiment, one period.** Trained on a single randomized campaign;
  client behavior, offers, or SMS fatigue can shift the effect. The drift
  reference (`contract/drift_reference.json`: per-feature deciles, null
  shares, `gender` shares of the training clients) supports monitoring that.
- **Fairness.** `gender` and `age` are inputs. Their effect on who gets
  promotions has not been reviewed.

## Lineage (version tags)

`git_commit` `8b212de`, `split_sha256` `a22bf8eb…`, `raw_data_digest`
`f99dd016f1ed1ecc`, `feature_code_sha256` `6f77f597…`, `uv_lock_sha256`
`3fc56e93…`, `fit_split` train+val, `leaderboard_run_id` `6fc6717a…`,
`final_evaluation_run_id` `62d19750…`.

## Updating

```bash
uv run --env-file .env python scripts/register_model.py                  # new version, champion unchanged
uv run --env-file .env python scripts/register_model.py --set-champion   # new version becomes champion
```

Every version is reloaded and must score bit-identically before it is
registered. To roll back, point `champion` at an earlier version (Catalog
Explorer → the model → the version → aliases, or
`MlflowClient.set_registered_model_alias`).
