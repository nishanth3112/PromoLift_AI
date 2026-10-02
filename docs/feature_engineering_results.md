# Feature engineering: ablation results

**Result:** the 47 new features **did not improve targeting**. None of the four
new feature groups beats the 19 Phase 10 features, alone or together, and the
answer holds when the larger feature set gets hyperparameters tuned for it.
The model keeps the base features. Phase 10 concluded that the features were
the bottleneck; these features don't lift the Qini ≈ 0.015–0.016 ceiling, which
points at the heterogeneity signal itself being weak at this sample size rather
than at missing purchase-history aggregates.

All numbers are 3-fold cross-validation **inside train** (120,023 clients),
with paired bootstrap CIs over 1,000 stratified resamples. Validation and test
were not used in this phase.

## What was tested

| Group | Columns | What it measures |
|---|---|---|
| base (Phase 10) | 19 | demographics, RFM, points totals, 30-day activity, product mix |
| promo_responsiveness | 7 | express (promotional) points, redemption rate, discount depth, days since last redemption |
| purchase_dynamics | 14 | visit regularity, 7/14/60/90-day windows, spend and visit trend slopes |
| basket_store | 9 | basket size, product variety, store loyalty, weekend share, hour of day |
| category_spend | 17 | top-15 `level_2` spend shares, other share, spend entropy |

Models: class_transformation, s_learner, dr_learner — the cheap members of the
Phase 10 top cluster. Every client is scored out-of-fold, scores are ranked
within each fold before pooling, and each gain is the paired difference in
pooled out-of-fold Qini, averaged across the three models.

## Ablation: base-tuned params held fixed

| Config | class_transformation | s_learner | dr_learner | Gain vs base [95% CI] | P(better) |
|---|---|---|---|---|---|
| base | 0.0159 | 0.0175 | 0.0166 | — | — |
| + promo_responsiveness | 0.0145 | 0.0150 | 0.0165 | −0.0013 [−0.0027, +0.0000] | 0.03 |
| + purchase_dynamics | 0.0155 | 0.0144 | 0.0167 | −0.0011 [−0.0026, +0.0002] | 0.04 |
| + basket_store | 0.0141 | 0.0143 | 0.0148 | **−0.0023** [−0.0037, −0.0007] | 0.00 |
| + category_spend | 0.0137 | 0.0150 | 0.0165 | −0.0016 [−0.0034, +0.0001] | 0.03 |
| all groups | 0.0132 | 0.0148 | 0.0155 | **−0.0022** [−0.0044, −0.0001] | 0.02 |

Per-model columns are mean CV Qini across folds. The base row reproduces the
CV scores in `configs/tuned_params.yaml` exactly (same folds, params, seed).

- **No group helps.** Every point estimate is negative; `basket_store` and all
  groups together are significantly worse. The decision rule (keep a group only
  if its gain CI is above 0) kept nothing and chose base.
- dr_learner is the least affected (promo and dynamics within ±0.0001 of base);
  s_learner loses the most from every group.

## Fairness check: each feature set with its own tuned params

The ablation reused params tuned on 19 features, which could handicap a
66-column table. So the three models were re-tuned on all groups with the same
search (Optuna, 40 trials, defaults first, same folds) and compared, each arm
with its own params:

| Model | all, defaults | all, tuned | base, tuned | all tuned − base tuned |
|---|---|---|---|---|
| class_transformation | 0.0125 | 0.0151 | 0.0159 | −0.0008 |
| s_learner | 0.0134 | 0.0154 | 0.0175 | −0.0021 |
| dr_learner | 0.0117 | 0.0156 | 0.0166 | −0.0010 |
| **mean, paired [95% CI]** | | | | **−0.0013** [−0.0032, +0.0008], P(better) 0.10 |

- **Tuning closed about 40% of the gap, not all of it.** The pooled shortfall
  shrank from −0.0022 to −0.0013 — most for class_transformation (0.0132 →
  0.0151), little for s_learner and dr_learner — yet every model still scores
  below base and the paired difference stays negative. The negative result is
  not an artifact of the fixed params.
- Tuning on 66 features chose much heavier regularization for s_learner
  (`min_child_samples` 1,183 vs 209 on base): the extra columns mostly add
  variance the model has to be shielded from.
- Both arms are best-of-40 on the same folds, so the selection optimism is
  symmetric.

## What it means

- **Keep the 19 base features.** They are as good as or better than any larger
  set, and cheaper for the econml/causalml models.
- **The ceiling is not missing aggregates.** With a +3.3pp average effect and
  120k training clients, the base RFM and demographic features already capture
  the heterogeneity that can be resolved; finer behavioral detail adds noise
  faster than signal. Further gains would more likely need a different kind of
  information (e.g. client-level responses to past campaigns) than more
  aggregates of the same purchase history.
- **What this phase did not test:** individual features inside a group (a
  useful column could be diluted by noisy neighbours), and re-tuning for each
  single-group config. Both mean more selection on the same folds; given every
  estimate was negative, neither is likely to change the conclusion.

## Reproduce

Runs in `/Shared/promolift/promolift-feature-ablation` (ablation run
`83310e22d8e84766b8b7b261ac092f77`, commit `a2ea72e`; comparison run
`f7a6de1da7034545887fb82245c153e2`, commit `f04ebfe`) and
`/Shared/promolift/promolift-model-tuning` (all-feature tuning, commit
`4ee7eda`); feature cache key `40fab2bcd11c7731`, split `a22bf8eb…`, seed 42.

```bash
uv run --env-file .env python scripts/fetch_split.py
git checkout a2ea72e && caffeinate -is uv run --env-file .env python scripts/run_ablation.py
git checkout f04ebfe && caffeinate -is uv run --env-file .env python scripts/compare_tuned_features.py
```
