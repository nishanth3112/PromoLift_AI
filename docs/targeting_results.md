# Budget targeting: how deep to text

**Result:** with the assumed economics (SMS ₽3, margin ₽105.32 per extra
purchase), the most profitable policy is to **text the top 38% of clients
ranked by the uplift model**: about **₽1,283 profit per 1,000 clients**
[₽976, ₽1,625]. That is **2.6× texting everyone** (₽500) and **2.5× the
response model at its best depth** (₽512), while sending 62% fewer SMS than
texting everyone. Ranking by the response model *loses money* at every depth
up to 61%.

X5 RetailHero is a public dataset, so the costs are assumptions, not a
business's figures (see [Economics](#economics-assumptions)). These are
cross-validated estimates on train + val, which also served model selection,
so they are optimistic. Adjusted to Phase 12's test uplift, the targeted profit
is roughly a third lower (see [Caveats](#caveats)) and still about 1.6×
texting everyone.

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
  everyone, so the rule was changed before the later runs.)

## Economics (assumptions)

| Input | Value | Source |
|---|---|---|
| SMS cost | ₽3.00 | **assumed** bulk-SMS price; no 2019 bulk price was found, and recent guides put domestic-operator SMS at roughly ₽1-2 |
| Average transaction | ₽428.13 | measured: mean `purchase_sum` per transaction, `purchases.csv` |
| Gross margin | 24.6% | X5 Retail Group's reported 2019 gross margin (IAS 17; 24.9% under IFRS 16), from its Q4 2019 financial results |
| Margin per extra purchase | ₽105.32 | average transaction × gross margin |
| **Break-even uplift** | **2.85%** | SMS cost / margin per extra purchase |
| Average effect of the SMS (ATE) | 3.32% | train + val |

The gross margin is a published figure, not profit after store costs, so it
flatters one extra purchase. The break-even uplift sits below the average
effect, so texting everyone pays a little, and targeting decides how much.
(An earlier run used an assumed 24% margin, ₽102.75 per purchase and a 2.92%
break-even; the depth did not change.)

## The decision

| Policy | SMS per 1,000 | Extra purchases per 1,000 | Profit per 1,000 [95% CI] |
|---|---|---|---|
| **Uplift model, top 38%** (decision) | 380 | 23.0 | **₽1,283** [976, 1,625] |
| Uplift model, top 27% (lower-spend option) | 270 | 18.4 | ₽1,130 [856, 1,412] |
| Text everyone | 1,000 | 33.2 | ₽500 [55, 1,020] |
| Response model, best depth (98%) | 980 | 32.8 | ₽512 |
| Text nobody | 0 | 0 | ₽0 |

- **Most of the effect, a third of the cost.** The top 38% hold 69% of all
  extra purchases the SMS can cause (23.0 of 33.2) for 38% of the SMS spend.
- **Targeting keeps the margin of safety.** Texting everyone now barely clears
  zero (its interval's lower edge is ₽55); the targeted policy's lower edge is
  ₽976.
- **Returns fall with depth.** Each ₽1 spent on SMS returns ₽3.38 profit in
  the top 5%, ₽1.59 in the top 20%, and ₽1.13 at 38%. Beyond 38% the next
  clients' uplift approaches break-even, and profit falls.
- **The lower-spend option** (top 27%) is statistically tied with the
  maximum and sends 29% fewer SMS, for ~12% less expected profit. A
  budget-constrained campaign should prefer it to an arbitrary cut.

## The response model loses money

| Top share texted | Uplift model | Response model | Random |
|---|---|---|---|
| 5% | ₽508 | −₽120 | −₽21 |
| 20% | ₽956 | −₽318 | ₽145 |
| 38% | ₽1,283 | −₽236 | ₽290 |
| 50% | ₽1,190 | −₽278 | ₽460 |
| 100% | ₽500 | ₽500 | ₽500 |

Profit per 1,000 clients. The response model's top clients are those who buy
anyway, so texting them costs SMS without causing purchases: its expected
profit is negative at every depth up to 61% (at 38% its interval is
[−₽472, −₽15]). Its best policy (98%) is effectively texting everyone. Random
rankings fluctuate around the straight line from ₽0 to ₽500.

## Sensitivity: other costs and margins

The best depth depends only on the break-even uplift (SMS cost / margin per
extra purchase). Read off the row for the real values; no re-run needed.

| Break-even uplift | Text top | Profit per 1,000 [95% CI] | Lower-spend option | Text everyone |
|---|---|---|---|---|
| 0.5% | 100% | ₽2,973 [2,528, 3,493] | 73% (₽2,720) | ₽2,973 |
| 1.0% | 100% | ₽2,446 [2,001, 2,967] | 43% (₽2,100) | ₽2,446 |
| 2.0% | 51% | ₽1,671 [1,340, 2,065] | 30% (₽1,486) | ₽1,393 |
| **2.85% (configured)** | **38%** | **₽1,283 [976, 1,625]** | 27% (₽1,130) | ₽500 |
| 3.0% | 38% | ₽1,223 [916, 1,564] | 26% (₽1,050) | ₽340 |
| 4.0% | 31% | ₽860 [564, 1,163] | 16% (₽673) | ₽0 (don't send) |
| 5.0% | 30% | ₽539 [251, 828] | 3% (₽294) | ₽0 |
| 7.5% | 7% | ₽284 [142, 429] | 1% (₽169) | ₽0 |
| 10.0% | 2% | ₽155 [83, 226] | 1% (₽142) | ₽0 |

All rows use the measured basket and the 24.6% margin (₽105.32) and set the
SMS cost to break-even × margin.

- **Cheap SMS (break-even ≤ 1%): text everyone.** Every client's uplift
  exceeds the cost; the model adds nothing over sending to all.
- **Break-even 2-5%: the model earns its keep.** Targeting the top 30-51%
  beats texting everyone, and above 3.3% (the ATE) texting everyone loses money
  while targeting still profits.
- **Expensive SMS (≥ 7.5%): text a small top slice.** Only the top few percent
  have uplift high enough; profit is small but positive.

## Caveats

- **Assumed economics.** The SMS cost is an assumption and the margin is a
  published gross margin; with a real campaign's figures, read the matching
  sensitivity row (and re-run the analysis to cite it).
- **The estimates are optimistic.** Train and val were both used to choose
  the model (tuning CV in train, selection on val), and the cross-validated
  uplift here matches val's (uplift@30% +6.7pp) rather than Phase 12's
  untouched test (+5.33pp). With test's uplift, texting the top 30% earns
  about ₽784 per 1,000 instead of ₽1,218: roughly a third less, still about
  1.6× texting everyone (₽500; the ATE on test is the same +3.32pp). The depth
  decision is less sensitive than the profit level, but a real campaign's
  profit should be expected nearer the test-adjusted figure.
- **One extra transaction per converted client.** Margin per extra purchase
  assumes the SMS causes one average-value transaction; larger or repeated
  purchases would raise it (lower the break-even), smaller ones lower it.
- **The ranking, not the score, is used.** The depth is a share of clients
  ("text the top 38%"), robust to the model's score scale changing between
  refits. Phase 15 applies it to the full client base.

## Reproduce

Run `7a29acf36bf9454fb767d09518601c79` in
`/Shared/promolift/promolift-targeting` on Databricks: commit `4d79df1`,
`git_dirty=false`, split `a22bf8eb…`, raw data digest `f99dd016f1ed1ecc`,
5 folds, 1,000 bootstrap resamples, seed 42. Artifacts:
`targeting/decision.json`, `targeting/profit_curves.csv`,
`targeting/sensitivity.csv`, `targeting/profit_curves.png`. The run can be
repeated freely (it never loads test), e.g. after updating
`configs/business.yaml` with real values:

```bash
uv run --env-file .env python scripts/targeting_analysis.py
```
