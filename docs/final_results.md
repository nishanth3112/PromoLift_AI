# Final evaluation: test results

**Result:** on the locked test split, the chosen model
(`class_transformation_tuned`, 19 base features) **beats random targeting and
clearly beats the traditional response model**. Its gain is concentrated in
the clients it ranks highest: targeting the top 10% causes **2.7×** the extra
purchases of a random 10%, and targeting 30% causes **3.6×** what the response
model achieves. Its Qini is about half its validation score, the expected
shrinkage after choosing among 17 variants on validation. **These are the
numbers to quote**; the validation numbers in the earlier docs are optimistic.

Test was evaluated exactly once: 40,008 clients, never used before, 1,000
bootstrap resamples. The ATE on test is +3.32pp, the same as on the full data.

## How the model was chosen

Everything below was fixed before test was loaded and is recorded in the run.

- **Selection rule** (`models/selection.py`): among the uplift variants on the
  validation leaderboard whose paired Qini CI with the best includes zero, take
  the cheapest to fit. Eleven variants were tied; `class_transformation_tuned`
  was both the best on validation and the cheapest (~1 s to fit).
- **Final fit:** refitted on train + val (160,031 clients) with the parameters
  tuned by CV inside train.
- **References** (context only; the choice can't change after test is seen):
  `s_learner_tuned` and `causal_forest` (two other tied variants), `response`,
  and `random`, all refitted the same way.

## Results

| Model | Qini AUC [95% CI] | Uplift@10% | Uplift@30% [95% CI] | Verdict |
|---|---|---|---|---|
| **class_transformation_tuned** (chosen) | **+0.0078** [+0.0014, +0.0141] | **+8.97pp** | **+5.33pp** [+3.57, +7.26] | beats random |
| s_learner_tuned | +0.0108 [+0.0042, +0.0171] | +8.09pp | +5.32pp [+3.44, +7.04] | beats random |
| causal_forest | +0.0100 [+0.0037, +0.0162] | +10.71pp | +5.29pp [+3.46, +7.12] | beats random |
| random ranking | +0.0047 [−0.0021, +0.0110] | +2.58pp | +4.20pp [+2.46, +5.85] | within noise |
| response model | −0.0046 [−0.0106, +0.0015] | +1.19pp | **+1.47pp** [+0.27, +2.48] | below random at 20–30% |

Random rankings on test stay within Qini [−0.0070, +0.0067] and uplift@30%
[+1.59, +4.78]pp 95% of the time (200 random rankings). The chosen model
clears the band on every metric: Qini, uplift AUC, and uplift at 10, 20, 30,
and 50%. The response model's Qini is inside the band, but its top 20% and 30%
convert *below* it (+1.35pp and +1.47pp): it targets clients who would buy
anyway.

## What it means for a campaign

Extra purchases caused by the SMS per 1,000 clients targeted:

| Targeting depth | Chosen model [95% CI] | Response model | Random clients (ATE) |
|---|---|---|---|
| top 10% | **90** [57, 120] | 12 | 33 |
| top 20% | 57 [34, 79] | 13 | 33 |
| top 30% | 53 [36, 73] | 15 | 33 |
| top 50% | 43 [29, 56] | 27 | 33 |

Targeting 30% of the test clients (12,002 people), the chosen model causes
~640 extra purchases, against ~399 for a random 30% and ~177 with the
response model. The advantage is largest at the top: the first decile's uplift
is +9.0pp, while deciles 2–10 range from +0.9 to +4.6pp. The uplift models'
gain is front-loaded, and their Qini curves meet the random line by ~80%
targeted. **A campaign should target narrowly** (Phase 13 sets the depth from
costs and margins).

## Is the difference real? (paired comparisons on test)

| Chosen model − | Qini difference [95% CI] | P(chosen better) | Uplift@30% difference [95% CI] |
|---|---|---|---|
| response | **+0.0124** [+0.0021, +0.0219] | 0.992 | **+3.86pp** [+1.93, +5.94] |
| random ranking | +0.0031 [−0.0062, +0.0125] | 0.744 | +1.14pp [−1.05, +3.36] |
| s_learner_tuned | −0.0030 [−0.0077, +0.0015] | 0.091 | +0.02pp [−1.24, +1.50] |
| causal_forest | −0.0022 [−0.0063, +0.0019] | 0.124 | +0.04pp [−1.04, +1.27] |

- **Uplift beats likely-buyer targeting, with certainty.** This is the
  project's central claim, and it holds on unseen data.
- **Against the one random ranking that was drawn, the Qini gain isn't
  significant.** That ranking happened to score +0.0047, high within its noise
  band, and its curve runs above the diagonal by chance. The fair test against
  random is the noise floor, built from 200 random rankings, and the chosen
  model clears it. Against that same random draw, the top-10% gain is
  significant (+6.38pp [+2.14, +10.33]).
- **The tie between the uplift models holds.** `s_learner_tuned` and
  `causal_forest` score higher Qini on test, but neither difference is
  significant, and all three give the same uplift@30% (~5.3pp). Choosing
  differently now would be selecting on test; the chosen model stands.

## Validation vs test

| Chosen model | Validation | Test |
|---|---|---|
| Qini AUC | +0.0160 [+0.0103, +0.0224] | +0.0078 [+0.0014, +0.0141] |
| Uplift@10% | +10.27pp | +8.97pp |
| Uplift@30% | +6.88pp | +5.33pp |

The drop is the expected price of selection: the validation winner of 17
variants is partly the luckiest, and test has no such selection. The two
Qini CIs overlap. The model was also refitted on 33% more data, so test
measures a slightly different model, but more training data would not be
expected to lower its score. Across Phases 9–12 the evidence is consistent: the
uplift signal in this data is real but weak, and most of it sits in the top
decile.

## Caveats

- **One test split, one look.** The CIs are the uncertainty; a second test set
  would land somewhere else inside them.
- **Weak signal.** A Qini of +0.008 is small. The business case rests on the
  top of the ranking (uplift@10–30%) and on the gap to the response model, not
  on the full-ranking Qini.
- **causal_forest is not bit-reproducible across machines.** A dry run
  (train → val, on Windows) reproduced the leaderboard's val Qini exactly for
  every LightGBM-based model and `random`, but `causal_forest` gave +0.0152 vs
  +0.0151 (leaderboard on macOS). It splits on exact feature values, and
  rebuilt features differ at ~1e-13 (see `features/store.py`). It's a
  reference model only.

## Reproduce

Run `62d1975014fc4f5d83d55cc80d21f3ac` in
`/Shared/promolift/promolift-final-evaluation` on Databricks: commit
`a08c45a`, `git_dirty=false`, split `a22bf8eb…`, raw data digest
`f99dd016f1ed1ecc` (both match the leaderboard run
`6fc6717a9b6a46d5b112c83120f2694d`), seed 42. Artifacts: `final/results.csv`,
`final/campaign.csv`, `final/paired_comparisons.json`, `final/qini_curves.png`,
`final/selection.json`, and the chosen model's report under `evaluation/test/`.

The script refuses to evaluate test a second time for this split
(`scripts/final_evaluation.py`; see the README). The dry run, which never
loads test, can be repeated:

```bash
MLFLOW_TRACKING_URI=sqlite:///<scratch>/mlflow.db uv run python scripts/final_evaluation.py \
  --dry-run --leaderboard-tracking-uri databricks://promolift
```
