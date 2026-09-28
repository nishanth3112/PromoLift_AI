# Advanced models and tuning: validation results

**Result:** advanced causal models and hyperparameter tuning **did not break
the tie at the top**. Ten variants from very different model families sit
within Qini 0.0133–0.0160 and are statistically indistinguishable from the
best. Tuning clearly helped only the two models whose defaults were weakest.
The tuned response model — a better purchase predictor — is still worse than
random targeting. Since every method converges on the same ceiling, the
features, not the model choice, now limit performance.

All numbers are on the **validation** split (40,008 clients, 1,000 bootstrap
resamples). The test split is locked and has not been used.

## Leaderboard (17 variants)

| Model | Qini AUC [95% CI] | Uplift@30% | Fit time |
|---|---|---|---|
| class_transformation_tuned | **+0.0160** [+0.0103, +0.0224] | +6.88pp | 0.8s |
| s_learner_tuned | +0.0154 [+0.0091, +0.0215] | +6.32pp | 3.8s |
| causal_forest (= tuned: defaults won the grid) | +0.0151 [+0.0092, +0.0216] | +6.94pp | 28s |
| dr_learner_tuned | +0.0150 [+0.0089, +0.0215] | +6.51pp | 1.6s |
| s_learner | +0.0149 [+0.0087, +0.0213] | +6.58pp | 1.5s |
| uplift_rf | +0.0136 [+0.0076, +0.0199] | +6.44pp | 62s |
| t_learner_tuned | +0.0136 [+0.0074, +0.0198] | +7.08pp | 1.8s |
| uplift_rf_tuned | +0.0133 [+0.0073, +0.0196] | +6.63pp | 44s |
| x_learner_tuned | +0.0124 [+0.0063, +0.0188] | +6.81pp | 2.2s |
| t_learner | +0.0115 [+0.0052, +0.0180] | +6.07pp | 1.8s |
| x_learner | +0.0105 [+0.0040, +0.0170] | +6.61pp | 3.9s |
| class_transformation | +0.0104 [+0.0041, +0.0168] | +5.27pp | 1.2s |
| dr_learner | +0.0097 [+0.0030, +0.0162] | +5.98pp | 4.8s |
| random | −0.0041 [−0.0102, +0.0024] | +2.81pp | — |
| **response_tuned** | **−0.0098** [−0.0154, −0.0043] | +1.58pp | 1.3s |
| **response** | **−0.0098** [−0.0154, −0.0044] | +1.71pp | 1.6s |

Every uplift variant beats the random-ranking band (Qini > +0.0066); both
response variants fall below it (Qini < −0.0067).

## Did tuning help? (tuned − default, paired)

Tuning searched 3-fold CV inside train only; these differences are measured on
validation, which the search never saw.

| Model | Default → tuned Qini | Difference [95% CI] | P(tuned better) |
|---|---|---|---|
| class_transformation | 0.0104 → 0.0160 | **+0.0056** [+0.0005, +0.0111] | 0.984 |
| dr_learner | 0.0097 → 0.0150 | **+0.0052** [+0.0001, +0.0106] | 0.980 |
| t_learner | 0.0115 → 0.0136 | +0.0021 [−0.0021, +0.0068] | 0.827 |
| x_learner | 0.0105 → 0.0124 | +0.0019 [−0.0028, +0.0070] | 0.793 |
| s_learner | 0.0149 → 0.0154 | +0.0005 [−0.0031, +0.0039] | 0.600 |
| uplift_rf | 0.0136 → 0.0133 | −0.0003 [−0.0027, +0.0021] | 0.404 |
| causal_forest | unchanged (defaults best) | 0 | — |
| response (tuned on purchase AUC) | −0.0098 → −0.0098 | 0.0000 [−0.0006, +0.0006] | 0.534 |

- **Tuning mattered where the defaults were weak** (class transformation, DR-learner),
  and was noise elsewhere. CV had predicted exactly these two as the largest gains.
- **Treat the two significant gains as strong hints, not certainties.** Eight
  comparisons at 95% each; with a multiple-comparison correction both CIs would
  include zero.
- CV gains shrank on validation, as expected from choosing the best of 40 noisy
  trials (e.g. T-learner +0.0024 on CV, +0.0021 on val; S-learner +0.0017 → +0.0005).

## Is the best model actually better?

Against the top variant (class_transformation_tuned), paired on identical resamples:

| vs | Qini difference [95% CI] | P(best better) |
|---|---|---|
| s_learner_tuned | +0.0007 [−0.0039, +0.0056] | 0.625 |
| causal_forest | +0.0009 [−0.0030, +0.0045] | 0.685 |
| dr_learner_tuned | +0.0011 [−0.0014, +0.0035] | 0.787 |
| s_learner | +0.0011 [−0.0034, +0.0058] | 0.685 |
| t_learner_tuned | +0.0024 [−0.0024, +0.0079] | 0.849 |
| uplift_rf | +0.0024 [−0.0006, +0.0057] | 0.942 |
| x_learner_tuned | +0.0036 [−0.0006, +0.0077] | 0.959 |
| x_learner / dr_learner (defaults) | +0.0055 / +0.0063, CIs exclude 0 | 0.995 / 0.990 |
| random / response | +0.0201 / +0.0258, CIs exclude 0 | 1.000 |

**No variant is significantly better than the top cluster.** The ranking inside
it is noise; choosing among these models on validation Qini alone would be
chasing the winner's curse.

## What it means

- **The features are the bottleneck.** Meta-learners, doubly robust learners,
  and two kinds of causal forest all converge on Qini ≈ 0.015–0.016 with the
  current 19 features. More model capacity doesn't find more signal; richer
  features are the next lever.
- **A better purchase predictor is not a better targeter.** Tuning the
  response model raised its purchase AUC (0.7763 → 0.7775 on CV) and left its
  targeting worse than random.
- **Cheap models are enough.** The tuned class transformation fits in under a
  second and ties the 28–62s forests.
- Campaign view, targeting 30% of validation clients (12,002): the top variant
  causes ~825 extra purchases (uplift@30% × 12,002), versus ~399 for a random
  30% and ~205 with the response model.

## Reproduce

Runs: `/Shared/promolift/promolift-model-tuning` (tuning, commit `1cc6193`) and
`/Shared/promolift/promolift-uplift-models` (leaderboard run
`6fc6717a9b6a46d5b112c83120f2694d`, commit `ffcf83e`), split `a22bf8eb…`, seed 42.

```bash
git checkout ffcf83e
uv run --env-file .env python scripts/fetch_split.py
caffeinate -is uv run --env-file .env python scripts/train_models.py --tuned
```
