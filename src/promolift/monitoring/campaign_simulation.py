"""Simulated campaign results with a known effect, for tests and the evaluation demo.

No real campaign has been run on the X5 clients, and every client with a
known outcome is already used for training or evaluation, so the evaluation
code is checked on synthetic campaigns whose true effect is set here:
``effect_scale`` is how much of the predicted uplift is real. 1.0 is a
well-calibrated model, 0.5 one that promises twice the real effect, 0.0 one
whose texts do nothing.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from promolift.monitoring.campaign_data import RESULT_COLUMNS

DEFAULT_BASELINE_RATE = 0.6  # roughly the X5 control group's purchase rate


def simulate_campaign(
    batch: pl.DataFrame,
    *,
    effect_scale: float = 1.0,
    baseline_rate: float = DEFAULT_BASELINE_RATE,
    seed: int = 0,
) -> pl.DataFrame:
    """Results for the targeted clients of a send list, as if the campaign had run.

    Every targeted client buys with probability ``baseline_rate`` untreated;
    texting adds ``effect_scale * uplift_score`` to it. Treated clients are
    the batch's ``send`` clients, control its ``holdout``.

    Raises:
        ValueError: If the batch holds more than one scoring run or
            ``baseline_rate`` is outside [0, 1].
    """
    if not 0 <= baseline_rate <= 1:
        raise ValueError("baseline_rate must be in [0, 1]")
    if batch["scoring_run_id"].n_unique() != 1:
        raise ValueError("simulate one scoring run at a time")
    targeted = batch.filter(pl.col("send") | pl.col("holdout"))
    treated = targeted["send"].cast(pl.Int8).to_numpy()
    chance = np.clip(
        baseline_rate + effect_scale * treated * targeted["uplift_score"].to_numpy(), 0.0, 1.0
    )
    purchased = np.random.default_rng(seed).random(targeted.height) < chance
    return pl.DataFrame(
        {
            "client_id": targeted["client_id"],
            "treated": treated,
            "purchased": purchased.astype(np.int8),
            "scoring_run_id": targeted["scoring_run_id"],
        }
    ).select(RESULT_COLUMNS)
