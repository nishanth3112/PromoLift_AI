# PromoLift AI: Causal Uplift Modeling for Retail SMS Promotions

![Python](https://img.shields.io/badge/Python-3.12-3776AB)
![Polars](https://img.shields.io/badge/Data-Polars-CD792C)
![Models](https://img.shields.io/badge/Models-LightGBM%20%7C%20EconML%20%7C%20CausalML-2E7D32)
![Tracking](https://img.shields.io/badge/Tracking-MLflow-0194E2)
![Platform](https://img.shields.io/badge/Platform-Databricks-FF3621)
![CI](https://img.shields.io/badge/CI-GitHub%20Actions-2088FF)

Estimates each customer's **incremental** response to an SMS promotion (uplift), so the campaign texts only the customers whose purchase the message actually changes. Built on the public [X5 RetailHero](https://ods.ai/competitions/x5-retailhero-uplift-modeling) randomized experiment (200K labelled clients, 400K total).

## Why uplift, not response

A response model texts likely buyers, many of whom would buy anyway. Uplift modeling targets the *persuadable*. The test below shows how much that matters.

## Approach

- **Data:** randomized 50/50 treatment (balance and propensity checks pass); average effect **+3.32pp** purchase rate. Features are leakage-free (19 demographic, RFM and product-mix features).
- **Models:** 9 uplift learners (S/T/X/DR-learner, class transformation, causal forest, uplift RF) plus a response baseline and random. Tuned by cross-validation inside train.
- **Evaluation:** normalized Qini AUC and uplift@k with bootstrap CIs. One locked 60/20/20 split; **the test set was evaluated exactly once**, with the model chosen beforehand by a fixed rule.
- **Production:** champion model in Unity Catalog, batch scoring on Databricks, PSI drift gates, a 10% holdout to measure real lift, CI/CD with approval-gated deploys.

## Findings

| | Result | Basis |
|---|---|---|
| Model ceiling | ~10 variants statistically tied (Qini ≈ 0.015 to 0.016); features, not algorithms, are the bottleneck | validation |
| Response model | Worse than random (Qini −0.0098) | validation |
| Champion | Tuned class transformation, test Qini **+0.0078** [+0.0014, +0.0141] | test, once |
| Top 10% targeted | **90** extra purchases per 1,000 texted [57, 120] vs 33 random (**2.7×**) and 12 for response | test |
| Top 30% targeted | 53 per 1,000 [36, 73] vs 33 random (1.6×) and 15 for response | test |

Validation gains roughly halved on test, so quote the test numbers.

## Business impact

Profit per 1,000 clients, using **assumed** economics (SMS ₽3, margin ₽105 per extra purchase from X5's reported 24.6% gross margin; break-even uplift 2.85%):

| Strategy | Profit per 1,000 clients |
|---|---|
| **Text top 38% by uplift** | **₽1,283** [976, 1,625] |
| Text everyone | ₽500 [55, 1,020] |
| Response model (best depth) | ₽512, and negative up to 61% depth |

About 2.6× texting everyone (cross-validated); adjusting for the test-set shrinkage, closer to **1.6×**. The best depth depends only on the break-even uplift, so the sensitivity table in [targeting_results.md](docs/targeting_results.md) gives the answer for any real cost and margin.

**Age:** SHAP ranks age as the top driver (22%). A no-age refit costs a small, borderline amount of ranking quality (Qini +0.0143 → +0.0120) and cuts the share of texted clients aged 60+ from 41.5% to 27.5%. Whether to keep age is left to the campaign owner ([age_check_results.md](docs/age_check_results.md)).

## Limitations

- Economics are assumptions: replace `configs/business.yaml` with real SMS cost and margin.
- No real campaign has run. Campaign evaluation is built and tested on simulated results only.
- The signal is weak (test Qini under 0.01); see the [model card](docs/model_card.md).

## Quickstart

```bash
uv sync
uv run python -m pytest
```

Raw data goes in `data/raw/` (never committed; `purchases.csv` is ~4.2 GB, read lazily). Setup, every script and its options, configuration, Databricks deployment and CI/CD are in [docs/usage.md](docs/usage.md).

## Layout

`src/promolift/`: `data`, `validation`, `features`, `models`, `evaluation`, `optimization`, `serving`, `monitoring`, `explainability`, `tracking`. Scripts in `scripts/` only orchestrate.

## Results documents

[Baselines](docs/baseline_results.md) · [Advanced models](docs/advanced_models_results.md) · [Feature engineering](docs/feature_engineering_results.md) · [Final evaluation](docs/final_results.md) · [Targeting](docs/targeting_results.md) · [Explainability](docs/explainability_results.md) · [Age check](docs/age_check_results.md) · [Model card](docs/model_card.md)
