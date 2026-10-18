# Age check: what the model loses without age

**Result:** refitting the champion without `age` costs a **small** amount of
ranking quality and removes **most of the age skew** in who gets texted.
Qini AUC falls from +0.0143 to +0.0120, a difference of +0.0023 whose 95%
interval (+0.0003 to +0.0044) only just excludes zero. At the deployed
targeting depth (the top 38%) the difference in uplift is +0.28pp, and its
interval (-0.25pp to +0.88pp) includes zero. In exchange, the texted group
stops leaning old: 41.5% of texted clients are 60 or over with `age` (against
24.3% of all clients) and 27.5% without it.

This measures a trade-off; it does not decide it. Whether the campaign should
use `age` is for its owner, and this is not a fairness or legal assessment.

## How it was computed

| | |
|---|---|
| Model | `class_transformation_tuned` (the champion's configuration), parameters held fixed in both arms |
| Arms | **with** all 19 features; **without** `age` and `age_is_valid` (the flag is derived from age and was already unused) |
| Clients | the 160,031 train + val clients; the test split was not loaded |
| Method | 3-fold stratified cross-validation (the project's tuning default); every client is scored only by a fold model that never saw it; the two rankings are compared on the same 1,000 bootstrap resamples (`explainability/feature_removal.py`) |
| Depth | 38%, the deployed `send_share` |
| Run | `1dc06ca94af5406697d9e61838249e57` in `promolift-feature-ablation`, from clean commit `0cba273` (seed 42) |

## Ranking quality

| Metric | With age | Without age | With minus without (95% interval) | P(with > without) |
|---|---|---|---|---|
| Qini AUC | +0.0143 | +0.0120 | **+0.0023** (+0.0003 to +0.0044) | 98% |
| Uplift AUC | | | +0.0036 (+0.0005 to +0.0068) | 98% |
| Uplift in the top 38% | +5.65pp | +5.38pp | **+0.28pp** (-0.25 to +0.88) | 85% |

Read these together. Across the whole ranking, `age` helps a little, and the
interval's lower edge is +0.0003, so this is borderline: a preview of the same
comparison with 200 resamples had an interval of -0.0002 to +0.0041, and a
different seed could land either side of zero. At the depth the campaign
actually uses, the difference is too small to separate from noise. For scale,
the Qini difference is about a sixth of the model's own Qini (+0.0023 of
+0.0143), and the model's ranking quality is modest in both arms (see
`final_results.md`).

## Who gets texted

The top 38% of clients under each ranking, against everyone (ratio above 1 =
texted more often than the group's weight in the population):

| Age band | Population | Texted, with age | Texted, without age |
|---|---|---|---|
| Under 30 | 15.9% | 7.3% (x0.46) | 17.7% (x1.11) |
| 30-44 | 32.4% | 18.1% (x0.56) | 28.3% (x0.87) |
| 45-59 | 27.0% | 32.5% (x1.20) | 26.2% (x0.97) |
| 60 and over | 24.3% | 41.5% (x1.71) | 27.5% (x1.13) |
| Age not valid | 0.3% | 0.5% (x1.45) | 0.3% (x0.92) |

With `age`, the ranking follows age closely (rank correlation +0.48): clients
under 30 are texted at less than half their share, and those 60 and over at
1.7 times. Without it, the correlation is +0.05 and every band is within
roughly 13% of its population share.

## What this does and does not show

- **Age is not entirely gone.** Tenure, spend, and points spent correlate with
  age, so the model without `age` can still lean on them. The +0.05 score-age
  correlation says little is left, not that nothing is. Dropping the correlated
  features too is a different experiment.
- **The "with" arm has a small advantage.** The hyperparameters were tuned with
  `age` present and held fixed for the "without" arm. A refit with its own
  tuning might close part of the gap.
- **Borderline, not settled.** The Qini difference is small and its interval
  touches zero. Do not read it as "age adds nothing", and do not read it as
  "age is needed": the data are compatible with both a negligible and a
  modest gain.
- **Cross-validation on train + val.** Validation already informed model
  selection, so absolute levels are optimistic. The comparison is paired on the
  same folds and resamples, so the difference is less affected.
- **Association only.** This shows how a ranking changes, not whether older
  clients respond more to the SMS. A real campaign would show that:
  `scripts/evaluate_campaign.py` can compare realized and predicted uplift by
  age band once results exist.

## If you decide to drop age

The numbers above suggest the cost in ranking quality is small, but acting on
it is a change to the deployed model, which means: refit and register a new
version without `age` (`scripts/register_model.py` would need the feature
dropped from the contract), re-run the targeting analysis (the best depth and
profit may move slightly), re-run the drift reference, and score a batch. None
of that is done here.

## Reproduce

```bash
uv run --env-file .env python scripts/age_ablation.py
```

`--no-log` prints without recording a run; `--drop` removes other features;
`--n-folds`, `--n-bootstrap`, `--depth`, `--seed` change the setup.
