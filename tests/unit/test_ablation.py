"""Unit tests for the cross-validated feature-group ablation."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest
from mlflow import MlflowClient

from promolift.features.build import FeatureGroup
from promolift.models.ablation import (
    AblationResult,
    FeatureConfig,
    FeatureGain,
    ablation_configs,
    bootstrap_qini,
    choose_feature_set,
    feature_gain,
    log_ablation,
    out_of_fold_scores,
    run_ablation,
)
from promolift.models.dataset import ModelFrame
from promolift.models.tuning import cross_validated_score, cv_folds
from promolift.tracking.mlflow_tracking import Experiment, TrackingConfig, local_tracking_config

DEMO, PROMO, DYN = (
    FeatureGroup.DEMOGRAPHICS,
    FeatureGroup.PROMO_RESPONSIVENESS,
    FeatureGroup.PURCHASE_DYNAMICS,
)
# Three columns per group, so no config is too small for the learners.
_GROUP_COLUMNS = {
    DEMO: ["base0", "base1", "base2"],
    PROMO: ["b", "promo0", "promo1"],
    DYN: ["dyn0", "dyn1", "dyn2"],
}
_MODELS = ("t_learner", "class_transformation")
_OVERRIDES = {name: {"n_estimators": 50} for name in _MODELS}
_FAST = {"n_folds": 3, "n_bootstrap": 60, "seed": 0}


def _synthetic(n: int = 4_000, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    # Only PROMO's column "b" carries uplift: sure things (b >= 0.5) buy 80%
    # regardless; persuadables 10% -> 40% if treated. The other groups are noise.
    rng = np.random.default_rng(seed)
    b = rng.random(n)
    treatment = rng.integers(0, 2, n)
    outcome = (rng.random(n) < np.where(b >= 0.5, 0.8, 0.1 + 0.3 * treatment)).astype(int)
    columns = pd.DataFrame(
        {name: rng.random(n) for names in _GROUP_COLUMNS.values() for name in names}
    )
    columns["b"] = b
    return columns, treatment, outcome


def _make_frame_factory(n: int = 4_000):
    columns, treatment, outcome = _synthetic(n)

    def make_frame(groups: Sequence[FeatureGroup]) -> ModelFrame:
        selected = [c for g in groups for c in _GROUP_COLUMNS[g]]
        return ModelFrame(
            split="train",
            client_ids=pl.Series([f"c{i}" for i in range(n)]),
            features=columns[selected].copy(),
            treatment=treatment,
            outcome=outcome,
        )

    return make_frame


def _gain(lower: float, upper: float, config: str = "x", baseline: str = "base") -> FeatureGain:
    return FeatureGain(
        config=config,
        baseline=baseline,
        difference=(lower + upper) / 2,
        lower=lower,
        upper=upper,
        win_rate=0.5,
        per_model={},
    )


def test_configs_are_base_each_candidate_alone_then_all() -> None:
    configs = ablation_configs(base=(DEMO,), candidates=(PROMO, DYN))

    assert [c.name for c in configs] == [
        "base",
        "+promo_responsiveness",
        "+purchase_dynamics",
        "all",
    ]
    assert configs[1].groups == (DEMO, PROMO)
    assert configs[-1].groups == (DEMO, PROMO, DYN)


def test_out_of_fold_scores_cover_every_client_once_as_within_fold_ranks() -> None:
    frame = _make_frame_factory()([DEMO, PROMO])
    folds = cv_folds(frame.treatment, frame.outcome, n_folds=3, seed=0)

    result = out_of_fold_scores(
        "t_learner", frame, folds, config="c", overrides=_OVERRIDES["t_learner"], seed=0
    )

    assert result.oof_scores.shape == (len(frame.outcome),)
    assert ((result.oof_scores > 0) & (result.oof_scores <= 1)).all()
    for _, score_idx in folds:
        # Each fold is ranked on its own: average ranks 1..n over n, even with ties.
        n = len(score_idx)
        assert result.oof_scores[score_idx].mean() == pytest.approx((n + 1) / (2 * n))


def test_fold_qini_matches_tuning_cross_validation() -> None:
    # Same folds, params, and seed give the same per-fold Qini as tuning, so the
    # real base config reproduces the CV scores in configs/tuned_params.yaml.
    frame = _make_frame_factory()([DEMO, PROMO])
    folds = cv_folds(frame.treatment, frame.outcome, n_folds=3, seed=0)

    result = out_of_fold_scores(
        "t_learner", frame, folds, config="c", overrides=_OVERRIDES["t_learner"], seed=0
    )
    tuning = cross_validated_score(
        "t_learner", frame, folds, overrides=_OVERRIDES["t_learner"], seed=0
    )

    assert result.fold_qini == pytest.approx(tuning.per_fold)


def test_identical_rankings_have_zero_paired_gain() -> None:
    _, treatment, outcome = _synthetic(2_000)
    scores = np.random.default_rng(1).random(2_000)
    boot = bootstrap_qini(
        {("a", "m"): scores, ("b", "m"): scores.copy()},
        treatment,
        outcome,
        n_bootstrap=50,
        seed=0,
    )

    gain = feature_gain(boot, "a", "b", models=["m"])

    assert gain.difference == 0.0
    assert gain.lower == gain.upper == 0.0


def test_bootstrap_draws_are_reproducible_for_a_later_config() -> None:
    # A config evaluated later must see the same resamples to stay paired.
    _, treatment, outcome = _synthetic(2_000)
    scores = np.random.default_rng(1).random(2_000)
    first = bootstrap_qini({("a", "m"): scores}, treatment, outcome, n_bootstrap=30, seed=7)
    later = bootstrap_qini({("a", "m"): scores}, treatment, outcome, n_bootstrap=30, seed=7)

    np.testing.assert_array_equal(first.draws[("a", "m")], later.draws[("a", "m")])


def test_rule_keeps_nothing_when_no_group_beats_base() -> None:
    chosen, kept, reason = choose_feature_set(
        {PROMO: _gain(-0.001, 0.002), DYN: _gain(-0.002, 0.001)},
        all_vs_kept=_gain(-0.001, 0.003),
        base=(DEMO,),
    )

    assert kept == ()
    assert chosen.name == "base"
    assert "tied" in reason


def test_rule_prefers_the_smaller_kept_set_when_all_is_tied() -> None:
    chosen, kept, _ = choose_feature_set(
        {PROMO: _gain(0.001, 0.004), DYN: _gain(-0.002, 0.001)},
        all_vs_kept=_gain(-0.001, 0.002),
        base=(DEMO,),
    )

    assert kept == (PROMO,)
    assert chosen.groups == (DEMO, PROMO)


def test_rule_takes_all_groups_when_they_beat_the_kept_set() -> None:
    chosen, kept, reason = choose_feature_set(
        {PROMO: _gain(0.001, 0.004), DYN: _gain(-0.002, 0.001)},
        all_vs_kept=_gain(0.0005, 0.003),
        base=(DEMO,),
    )

    assert kept == (PROMO,)
    assert chosen.name == "all"
    assert chosen.groups == (DEMO, PROMO, DYN)
    assert "better" in reason


def test_rule_needs_no_comparison_when_every_group_is_kept() -> None:
    chosen, kept, _ = choose_feature_set(
        {PROMO: _gain(0.001, 0.004), DYN: _gain(0.001, 0.003)}, all_vs_kept=None, base=(DEMO,)
    )

    assert kept == (PROMO, DYN)
    assert chosen.name == "all"


@pytest.fixture(scope="module")
def ablation() -> AblationResult:
    return run_ablation(
        _make_frame_factory(),
        models=_MODELS,
        overrides=_OVERRIDES,
        base=(DEMO,),
        candidates=(PROMO, DYN),
        **_FAST,
    )


def test_signal_group_is_kept_and_noise_group_is_not(ablation: AblationResult) -> None:
    assert ablation.gains["+promo_responsiveness"].lower > 0
    assert ablation.gains["+purchase_dynamics"].lower <= 0
    assert ablation.kept_groups == (PROMO,)


def test_kept_set_matching_a_config_is_not_refitted(ablation: AblationResult) -> None:
    # base + PROMO is already the "+promo_responsiveness" config.
    assert ablation.chosen.groups == (DEMO, PROMO)
    assert ablation.all_vs_kept is not None
    assert ablation.all_vs_kept.baseline == "+promo_responsiveness"
    configs = {r.config for r in ablation.cv_results}
    assert configs == {"base", "+promo_responsiveness", "+purchase_dynamics", "all"}


def test_every_config_and_model_has_cv_results(ablation: AblationResult) -> None:
    pairs = {(r.config, r.model_name) for r in ablation.cv_results}

    assert len(pairs) == 4 * len(_MODELS)
    assert all(len(r.fold_qini) == 3 for r in ablation.cv_results)


@pytest.fixture
def tracking(tmp_path: Path) -> TrackingConfig:
    return local_tracking_config(tmp_path / "mlruns")


def test_log_ablation_records_configs_and_decision(
    ablation: AblationResult, tracking: TrackingConfig, lineage_dirs: dict[str, Path]
) -> None:
    feature_tags = {"feature_cache_key": "k", "feature_table_sha256": "s"}

    summary_id = log_ablation(
        ablation, feature_tags=feature_tags, config=tracking, lineage=lineage_dirs
    )

    client = MlflowClient(tracking_uri=tracking.tracking_uri)
    experiment = client.get_experiment_by_name(
        tracking.experiment_name(Experiment.FEATURE_ABLATION)
    )
    runs = client.search_runs([experiment.experiment_id])
    configs = {r.data.tags["feature_config"]: r for r in runs if r.data.tags["run_type"] == "cv"}
    assert set(configs) == {"base", "+promo_responsiveness", "+purchase_dynamics", "all"}
    promo = configs["+promo_responsiveness"].data
    assert promo.tags["feature_groups"] == "demographics,promo_responsiveness"
    assert promo.tags["feature_cache_key"] == "k"
    assert "cv_qini_mean_t_learner" in promo.metrics

    summary = client.get_run(summary_id).data
    assert summary.tags["chosen_feature_set"] == ablation.chosen.name
    assert summary.tags["chosen_feature_groups"] == "demographics,promo_responsiveness"
    artifacts = {a.path for a in client.list_artifacts(summary_id, "ablation")}
    assert artifacts == {
        "ablation/cv_results.csv",
        "ablation/feature_gains.csv",
        "ablation/decision.json",
    }


def test_feature_config_name_lists_added_groups() -> None:
    config = FeatureConfig.added(base=(DEMO,), added=(PROMO, DYN))

    assert config.name == "+promo_responsiveness+purchase_dynamics"
    assert config.groups == (DEMO, PROMO, DYN)
