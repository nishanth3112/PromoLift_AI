# Explainability: what drives the predicted uplift

**Result:** the champion's predicted uplift depends mostly on **age** (22% of
total feature importance), then on **how much a client has spent recently and
in total** (14% and 11%), **tenure** (11%), and **points spent** (10%). Those
five features carry about two thirds of it. Older, newer, lower-spending
clients get the highest scores; heavy spenders and long-standing customers the
lowest. **Gender contributes about 2%.** Two features, `age_is_valid` and
`has_redeemed`, are never used.

This describes what the model does, not what causes clients to respond (see
[Caveats](#caveats)). **Age as the largest single driver has not been reviewed**
for whether targeting on it is acceptable; see [Age](#age).

## How it was computed

| | |
|---|---|
| Model | `promolift_ai.models.uplift_targeting` version 2 (`champion`), a class-transformation LightGBM model |
| Clients | 20,000 randomly sampled (seed 42) of the 160,031 train + val clients it was fitted on; the test split was not loaded |
| Method | exact TreeSHAP (LightGBM `pred_contrib`) of the classifier's log-odds, rescaled per client so contributions sum exactly to the predicted uplift (`explainability/shap_values.py`) |
| Check | for every client, base value + contributions = predicted uplift (largest gap 5.6e-17) |
| Base value | +3.33pp: the predicted uplift of an average client, in line with the experiment's measured effect of +3.32pp |
| Run | `cf8b1936c77c4a3f87c7677f39e364d0` in `promolift-explainability`, from clean commit `404f4f0` |

Contributions are in percentage points of uplift. "Importance" is the mean
absolute contribution; "share" is its fraction of the total over all features.

## Which features matter

| # | Feature | Importance | Share | Higher value means |
|---|---|---|---|---|
| 1 | `age` | 0.89pp | 22% | higher uplift |
| 2 | `recent_monetary_30d` | 0.55pp | 14% | lower uplift |
| 3 | `tenure_days` | 0.46pp | 11% | lower uplift |
| 4 | `monetary_total` | 0.43pp | 11% | lower uplift |
| 5 | `points_spent_total` | 0.41pp | 10% | higher uplift |
| 6 | `monetary_avg` | 0.18pp | 4% | lower uplift |
| 7 | `monetary_cv` | 0.17pp | 4% | lower uplift |
| 8 | `days_to_redeem` | 0.17pp | 4% | lower uplift |
| 9-17 | nine more | 0.04-0.14pp each | 1-3% each | |
| 18-19 | `age_is_valid`, `has_redeemed` | 0 | 0% | never used |

`has_redeemed` is probably redundant with `days_to_redeem`, which is missing
for the 20% of clients who never redeemed. In the logged tables these two
features read "constant"; that label really means "no contribution" and was
renamed afterwards, with no change in numbers.

## How the score moves along the leading features

Mean contribution by value (bins of about 2,000 clients; `points_spent_total` has one
large bin of clients with no points):

| Feature | Lowest bin | Middle | Highest bin |
|---|---|---|---|
| `age` | -1.5pp (age ~22) | +0.1pp (~48) | +1.4pp (~72) |
| `recent_monetary_30d` | +1.6pp (₽231) | -0.3pp (₽1,200-1,600) | -0.5pp (₽7,800) |
| `monetary_total` | +1.6pp (₽954) | -0.1pp (₽4,200-5,400) | -0.6pp (₽24,300) |
| `tenure_days` | +0.9pp (59 days) | -0.1pp (~460) | -0.7pp (~670) |
| `points_spent_total` | -0.4pp (none, 40% of clients) | +0.6pp (~100 points) | +0.4pp (~300) |

The shapes are smooth and mostly monotone, and they fit one story: clients who
already spend a lot or have been customers for long are likely to buy anyway,
so the SMS adds little; light, recent, lower-spending clients have room to be
persuaded. That is a reading of the model, not something these tables prove.

## Different clients, different reasons

By predicted-uplift tier (the campaign texts roughly the top 40%):

| Tier | Clients | Mean predicted uplift | Leading features (share) |
|---|---|---|---|
| Top 10% | 2,000 | +8.5pp | `age` 17%, `recent_monetary_30d` 17%, `monetary_total` 14% |
| Next 30% | 6,000 | +4.4pp | `age` 22%, `recent_monetary_30d` 14%, `tenure_days` 13% |
| Rest | 12,000 | +1.9pp | `age` 23%, `recent_monetary_30d` 12%, `tenure_days` 11% |

The clients with the very highest scores are scored for spending as much as for
age: low recent and total spend add +1.2pp and +1.0pp for them on average.
For the rest, the same features pull the score down.

Gender and age segments are in the run's `segments.csv`; the leading features
are the same across genders. By age band the leader changes: `age` itself
accounts for a third of the importance under 30 and a quarter above 60, while
for ages 45-59 recent spend and age tie at 15%.

## Age

`age` is the largest driver, and it separates clients clearly: mean predicted
uplift is **2.2pp under 30, 2.5pp for 30-44, 3.8pp for 45-59, and 4.7pp for
60 and over**. Gender matters little: **2.0% of importance**, with mean
predicted uplift of 3.4pp for women, 3.0pp for men, and 3.4pp where gender is
unknown.

A model that leans most on age will send more texts to older clients and fewer
to younger ones. Whether that is acceptable for this campaign has not been
reviewed, and this note is not a legal assessment. If it matters, a check worth
running is a refit without `age` to see how much of the ranking quality it
carries (not done here).

## Caveats

- **Association, not cause.** SHAP says which inputs the model uses and in
  which direction. It does not say that raising a client's tenure or spend would
  change how they respond to an SMS.
- **Correlated features share credit.** Recent spend, total spend, average
  basket, and frequency move together, so their individual shares depend on how
  the model happened to split the credit among them. Read the group (about 31%
  together for those four) more firmly than any one of them.
- **A weak model.** On the test split the model's ranking beat random only
  narrowly (Qini +0.0078, see `final_results.md`), so some of these patterns may
  be noise. The smooth, monotone shapes argue for signal, but treat them as
  hypotheses to check against a real campaign: `scripts/evaluate_campaign.py`
  reports realized against predicted uplift by decile, and the same comparison
  by age band or spend would test these directly.
- **In-sample.** The model was fitted on these clients, so contributions
  describe the fit, not how it behaves on new clients.
- **Sampled.** 20,000 of 160,031 clients, seed 42. A 5,000-client sample gave
  the same leading features in every tier, gender, and age band, with shares
  within a point or two.

## Reproduce

```bash
uv run --env-file .env python scripts/explain_model.py
```

Add `--no-log` to print without recording a run; `--n-rows` and `--seed` change
the sample.
