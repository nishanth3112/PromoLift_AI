# Baseline uplift models: validation results

**Result:** the traditional response model — ranking customers by how likely
they are to buy when sent the SMS — is **worse than random targeting**. All
three baseline uplift models beat both random targeting and the response model
decisively, and are statistically tied with each other.

All numbers are on the **validation** split (40,008 clients). The test split is
locked and has not been used.

## Leaderboard

| Model | Qini AUC [95% CI] | Uplift@10% | Uplift@30% [95% CI] | Verdict |
|---|---|---|---|---|
| S-learner | **+0.0149** [+0.0087, +0.0213] | +10.20pp | **+6.58pp** [+4.88, +8.45] | beats random |
| T-learner | +0.0115 [+0.0052, +0.0180] | +9.05pp | +6.07pp [+4.30, +7.78] | beats random |
| Class transformation | +0.0104 [+0.0041, +0.0168] | +9.47pp | +5.27pp [+3.47, +7.06] | beats random |
| Random ranking | −0.0041 [−0.0102, +0.0024] | +1.25pp | +2.81pp [+1.09, +4.52] | within noise |
| **Response model** | **−0.0098** [−0.0154, −0.0044] | **−0.34pp** | **+1.71pp** [+0.63, +2.96] | **worse than random** |

For reference, sending the SMS to everyone lifts conversion by the ATE,
**+3.32pp**. Random rankings on this split stay within Qini [−0.0067, +0.0066]
and uplift@30% [+1.89, +4.58]pp 95% of the time (200 random rankings); a model
"beats random" only if its estimate clears that band.

## What it means for a campaign

Targeting 30% of the validation clients (12,002 people):

| Targeting by | Extra purchases caused by the SMS |
|---|---|
| S-learner | ~790 |
| T-learner | ~729 |
| Class transformation | ~633 |
| Nobody in particular (ATE × 12,002) | ~399 |
| Response model | ~205 |

The best uplift model roughly **doubles** the purchases the same SMS budget
causes, compared with sending it to a random 30%, and gets **~3.9×** what the
response model gets. The response model spends the budget on customers who
would have bought anyway: its top 10% has *negative* uplift (−0.34pp).

## Is the difference real? (paired comparisons)

Each comparison scores both models on the same 1,000 bootstrap resamples, so
noise shared by both cancels out.

| Comparison | Qini difference [95% CI] | P(A better) |
|---|---|---|
| S-learner − response | +0.0247 [+0.0157, +0.0335] | 1.000 |
| T-learner − response | +0.0213 [+0.0129, +0.0295] | 1.000 |
| Class transformation − response | +0.0202 [+0.0118, +0.0282] | 1.000 |
| S-learner − random | +0.0190 [+0.0103, +0.0273] | 1.000 |
| S-learner − T-learner | +0.0034 [−0.0015, +0.0089] | 0.914 |
| S-learner − class transformation | +0.0045 [−0.0014, +0.0102] | 0.929 |

Every uplift model beats the response model with certainty. **The S-learner's
lead over the other two is not significant** — its CI includes zero — so the
top three are tied, and none should be called the winner yet.

## Caveats

- **One validation split, untuned models.** All models share one fixed
  LightGBM configuration; tuning could reorder the top three.
- **Uplift@10% is noisy.** The top 10% is only ~4,000 clients (random band
  +0.61 to +5.72pp), which is why Qini AUC is the selection metric.
- **19 features.** Richer features could raise every model's ceiling.
- Validation numbers are used for model selection, so they are slightly
  optimistic; the unbiased estimate comes from the one-time test evaluation.

## Reproduce

Runs are in `/Shared/promolift/promolift-uplift-models` on Databricks,
commit `280f5f5`, canonical split `a22bf8eb…`, seed 42, 1,000 bootstrap
resamples (leaderboard run `434d8e4b9f6c41e98694e66730878498`). Models aren't
stored; refitting from the commit reproduces every number exactly:

```bash
git checkout 280f5f5
uv run --env-file .env python scripts/fetch_split.py
uv run --env-file .env python scripts/train_baselines.py
```
