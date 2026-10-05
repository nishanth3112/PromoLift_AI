# Budget targeting: how deep to text

**Result:** with the placeholder economics (SMS ₽3, margin ₽102.75 per extra
purchase), the most profitable policy is to **text the top 38% of clients
ranked by the uplift model**: about **₽1,224 profit per 1,000 clients**
[₽925, ₽1,557]. That is **3.0× texting everyone** (₽414) and **2.8× the
response model at its best depth** (₽432), while sending 62% fewer SMS than
texting everyone. Ranking by the response model *loses money* at every depth
up to 50%.

These are cross-validated estimates on train + val, which also served model
selection, so they are optimistic. Adjusted to Phase 12's test uplift, the
targeted profit is roughly a third lower (see [Caveats](#caveats)) and still
about 1.8× texting everyone.

## How it was computed

- **Scores:** `class_transformation_tuned` (the Phase 12 model), `response`, and
  `random`, each scored on all 160,031 train + val clients by 5-fold
  cross-validation: every client was scored by a model that never saw it.
  Test was not loaded.
- **Profit:** texting the top share `k` earns
  `k × (margin × uplift@k − SMS cost)` per client, reported per 1,000 clients
  of the population, at every depth from 0% to 100%, with 1,000-resample
  bootstrap CIs (`optimization/targeting.py`).
- **Decision rule:** the depth with the highest expected profit. The shallowest
  depth whose profit is statistically tied with it is reported as a
  lower-spend option. (A first run chose that shallowest depth instead; it gave
  up expected profit and, when SMS is cheap, earned less than texting
  everyone, so the rule was changed before this run.)

## Economics (placeholders)

| Input | Value | Source |
|---|---|---|
| SMS cost | ₽3.00 | assumed bulk-SMS price |
| Average transaction | ₽428.13 | measured: mean `purchase_sum` per transaction, `purchases.csv` |
| Gross margin | 24% | assumed grocery margin |
| Margin per extra purchase | ₽102.75 | average transaction × gross margin |
| **Break-even uplift** | **2.92%** | SMS cost / margin per extra purchase |
| Average effect of the SMS (ATE) | 3.32% | train + val |

The break-even uplift sits just below the average effect, so texting everyone
barely pays, and targeting decides whether the campaign is worth running.

## The decision

| Policy | SMS per 1,000 | Extra purchases per 1,000 | Profit per 1,000 [95% CI] |
|---|---|---|---|
| **Uplift model, top 38%** (decision) | 380 | 23.0 | **₽1,224** [925, 1,557] |
| Uplift model, top 26% (lower-spend option) | 260 | 17.8 | ₽1,046 [784, 1,331] |
| Text everyone | 1,000 | 33.2 | ₽414 [−20, 922] |
| Response model, best depth (91%) | 910 | — | ₽432 |
| Text nobody | 0 | 0 | ₽0 |

- **Most of the effect, a third of the cost.** The top 38% hold 69% of all
  extra purchases the SMS can cause (23.0 of 33.2) for 38% of the SMS spend.
- **Targeting removes the risk.** Texting everyone has a CI that includes a
  loss; the targeted policy's CI is well above zero.
- **Returns fall with depth.** Each ₽1 spent on SMS returns ₽3.28 profit in
  the top 5%, ₽1.53 in the top 20%, and ₽1.07 at 38%. Beyond 38% the next
  clients' uplift is below break-even, and profit falls.
- **The lower-spend option** (top 26%) is statistically tied with the
  maximum and sends 32% fewer SMS, for ~15% less expected profit. A
  budget-constrained campaign should prefer it to an arbitrary cut.

## The response model loses money

| Top share texted | Uplift model | Response model | Random |
|---|---|---|---|
| 5% | ₽492 | −₽121 | −₽24 |
| 20% | ₽918 | −₽325 | ₽127 |
| 38% | ₽1,224 | −₽258 | ₽255 |
| 50% | ₽1,124 | −₽308 | ₽412 |
| 100% | ₽414 | ₽414 | ₽414 |

Profit per 1,000 clients. The response model's top clients are those who buy
anyway, so texting them costs SMS without causing purchases: its profit is
negative, with CIs below zero, at every depth up to 50%. Its best policy
(91%) is effectively texting everyone. Random rankings fluctuate around the
straight line from ₽0 to ₽414.

## Sensitivity: other costs and margins

The best depth depends only on the break-even uplift (SMS cost / margin per
extra purchase). Read off the row for the real values; no re-run needed.

| Break-even uplift | Text top | Profit per 1,000 [95% CI] | Lower-spend option | Text everyone |
|---|---|---|---|---|
| 0.5% | 100% | ₽2,901 [2,466, 3,408] | 73% (₽2,654) | ₽2,901 |
| 1.0% | 100% | ₽2,387 [1,953, 2,894] | 43% (₽2,048) | ₽2,387 |
| 2.0% | 51% | ₽1,630 [1,307, 2,015] | 30% (₽1,450) | ₽1,359 |
| **2.92% (configured)** | **38%** | **₽1,224 [925, 1,557]** | 26% (₽1,046) | ₽414 |
| 3.0% | 38% | ₽1,193 [893, 1,526] | 26% (₽1,025) | ₽332 |
| 4.0% | 31% | ₽839 [550, 1,134] | 16% (₽656) | ₽0 (don't send) |
| 5.0% | 30% | ₽525 [245, 808] | 3% (₽287) | ₽0 |
| 7.5% | 7% | ₽277 [139, 418] | 1% (₽165) | ₽0 |
| 10.0% | 2% | ₽151 [81, 220] | 1% (₽139) | ₽0 |

All rows use the measured margin (₽102.75) and set the SMS cost to
break-even × margin.

- **Cheap SMS (break-even ≤ 1%): text everyone.** Every client's uplift
  exceeds the cost; the model adds nothing over sending to all.
- **Break-even 2–5%: the model earns its keep.** Targeting the top 30–51%
  beats texting everyone, and above 3.3% (the ATE) texting everyone loses money
  while targeting still profits.
- **Expensive SMS (≥ 7.5%): text a small top slice.** Only the top few percent
  have uplift high enough; profit is small but positive.

## Caveats

- **Placeholder economics.** SMS cost and gross margin are assumptions;
  confirm them with the business and read the matching sensitivity row.
- **The estimates are optimistic.** Train and val were both used to choose
  the model (tuning CV in train, selection on val), and the cross-validated
  uplift here matches val's (uplift@30% +6.7pp) rather than Phase 12's
  untouched test (+5.33pp). With test's uplift, texting the top 30% earns
  about ₽743 per 1,000 instead of ₽1,167: roughly a third less, still about
  1.8× texting everyone (₽415 on test). The depth decision is less sensitive
  than the profit level, but a real campaign's profit should be expected
  nearer the test-adjusted figure.
- **One extra transaction per converted client.** Margin per extra purchase
  assumes the SMS causes one average-value transaction; larger or repeated
  purchases would raise it (lower the break-even), smaller ones lower it.
- **The ranking, not the score, is used.** The depth is a share of clients
  ("text the top 38%"), robust to the model's score scale changing between
  refits. Phase 15 applies it to the full client base.

## Reproduce

Run `f1f0b456896c492d926971c5c9024d8b` in
`/Shared/promolift/promolift-targeting` on Databricks: commit `3eaf684`,
`git_dirty=false`, split `a22bf8eb…`, raw data digest `f99dd016f1ed1ecc`,
5 folds, 1,000 bootstrap resamples, seed 42. Artifacts:
`targeting/decision.json`, `targeting/profit_curves.csv`,
`targeting/sensitivity.csv`, `targeting/profit_curves.png`. The run can be
repeated freely (it never loads test), e.g. after updating
`configs/business.yaml` with real values:

```bash
uv run --env-file .env python scripts/targeting_analysis.py
```
