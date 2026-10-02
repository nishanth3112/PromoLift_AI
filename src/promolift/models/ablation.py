"""Feature-group ablation by cross-validation inside the training split.

Val has already been used to pick models in Phases 9-10, so feature groups
are judged on train alone: each configuration (base, base + one candidate
group, all groups) is fitted on the same stratified folds tuning used, with
the Phase 10 tuned hyperparameters held fixed. Val is touched once later, to
confirm the chosen feature set.

Every client gets an out-of-fold score from the fold model that held it out.
Scores are converted to within-fold percentile ranks before pooling, so fold
models whose raw score scales differ don't distort the pooled ranking (Qini
depends only on order within a fold, so per-fold Qini is unchanged). Gains
are paired-bootstrap differences in pooled out-of-fold Qini, averaged across
the models: every (config, model) ranking is scored on the same stratified
resamples, so shared client-level noise cancels.

The decision rule: keep a candidate group if its gain over base has a CI
above 0; then, if all groups together are not significantly better than the
kept set, prefer the kept set -- the smaller model.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import mlflow
import numpy as np
import polars as pl
from scipy.stats import rankdata

from promolift.evaluation.ranking_metrics import qini_auc
from promolift.evaluation.uncertainty import (
    DEFAULT_CONFIDENCE,
    DEFAULT_N_BOOTSTRAP,
    DEFAULT_SEED,
    stratified_resample_indices,
)
from promolift.features.build import ALL_GROUPS, BASE_GROUPS, FeatureGroup
from promolift.models.base import Params
from promolift.models.dataset import ModelFrame
from promolift.models.registry import build_model
from promolift.models.tuning import DEFAULT_N_FOLDS, cv_folds
from promolift.tracking.mlflow_tracking import Experiment, TrackingConfig, start_run

# The cheapest of the models statistically tied at the Phase 10 top.
ABLATION_MODELS = ("class_transformation", "s_learner", "dr_learner")
CANDIDATE_GROUPS = tuple(group for group in ALL_GROUPS if group not in BASE_GROUPS)
BASE_CONFIG = "base"
ALL_CONFIG = "all"

FrameFactory = Callable[[Sequence[FeatureGroup]], ModelFrame]
Key = tuple[str, str]  # (config name, model name)


@dataclass(frozen=True)
class FeatureConfig:
    """A named set of feature groups."""

    name: str
    groups: tuple[FeatureGroup, ...]

    @classmethod
    def added(cls, base: Sequence[FeatureGroup], added: Sequence[FeatureGroup]) -> FeatureConfig:
        """``base`` plus ``added``, named by what was added (``base`` if nothing)."""
        name = "".join(f"+{group.value}" for group in added) or BASE_CONFIG
        return cls(name, (*base, *added))


def ablation_configs(
    base: Sequence[FeatureGroup] = BASE_GROUPS,
    candidates: Sequence[FeatureGroup] = CANDIDATE_GROUPS,
) -> list[FeatureConfig]:
    """Base, base plus each candidate alone, then all groups."""
    return [
        FeatureConfig.added(base, ()),
        *(FeatureConfig.added(base, (group,)) for group in candidates),
        FeatureConfig(ALL_CONFIG, (*base, *candidates)),
    ]


@dataclass
class CVResult:
    """One model's out-of-fold scores and per-fold Qini on one feature config."""

    config: str
    model_name: str
    oof_scores: np.ndarray
    fold_qini: tuple[float, ...]
    fit_seconds: float

    @property
    def cv_mean(self) -> float:
        return float(np.mean(self.fold_qini))

    @property
    def cv_std(self) -> float:
        return float(np.std(self.fold_qini))


def out_of_fold_scores(
    model_name: str,
    frame: ModelFrame,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    config: str,
    overrides: Params | None,
    seed: int,
) -> CVResult:
    """Fit on each fold's training part; score and rank its held-out part.

    Per-fold Qini is computed on the raw scores exactly as tuning's
    ``cross_validated_score`` does; the returned out-of-fold scores are
    within-fold percentile ranks in (0, 1].
    """
    oof = np.empty(len(frame.outcome))
    fold_qini = []
    started = time.perf_counter()
    for fit_idx, score_idx in folds:
        model = build_model(model_name, seed=seed, overrides=overrides).fit(
            frame.features.iloc[fit_idx], frame.treatment[fit_idx], frame.outcome[fit_idx]
        )
        predictions = np.asarray(model.predict_uplift(frame.features.iloc[score_idx]))
        fold_qini.append(
            qini_auc(predictions, frame.treatment[score_idx], frame.outcome[score_idx])
        )
        oof[score_idx] = rankdata(predictions) / len(predictions)
    return CVResult(config, model_name, oof, tuple(fold_qini), time.perf_counter() - started)


@dataclass
class QiniBootstrap:
    """Pooled out-of-fold Qini per (config, model): the estimate and its bootstrap draws.

    Resample indices depend only on ``seed`` and the treatment vector, so a
    config bootstrapped later with the same seed stays paired with the rest.
    """

    estimates: dict[Key, float]
    draws: dict[Key, np.ndarray]
    n_bootstrap: int
    seed: int

    def merged(self, other: QiniBootstrap) -> QiniBootstrap:
        if (other.n_bootstrap, other.seed) != (self.n_bootstrap, self.seed):
            raise ValueError("can only merge bootstraps drawn with the same resamples")
        return QiniBootstrap(
            {**self.estimates, **other.estimates},
            {**self.draws, **other.draws},
            self.n_bootstrap,
            self.seed,
        )


def bootstrap_qini(
    scores: Mapping[Key, np.ndarray],
    treatment: np.ndarray,
    outcome: np.ndarray,
    *,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
) -> QiniBootstrap:
    """Qini of every ranking on the full sample and on shared stratified resamples."""
    t, y = np.asarray(treatment), np.asarray(outcome)
    estimates = {key: qini_auc(s, t, y) for key, s in scores.items()}
    draws = {key: np.empty(n_bootstrap) for key in scores}
    rng = np.random.default_rng(seed)
    for b in range(n_bootstrap):
        idx = stratified_resample_indices(t, rng)
        for key, s in scores.items():
            draws[key][b] = qini_auc(s[idx], t[idx], y[idx])
    return QiniBootstrap(estimates, draws, n_bootstrap, seed)


@dataclass(frozen=True)
class FeatureGain:
    """``config`` minus ``baseline`` pooled out-of-fold Qini, averaged across models."""

    config: str
    baseline: str
    difference: float
    lower: float
    upper: float
    win_rate: float
    per_model: dict[str, float] = field(default_factory=dict)


def feature_gain(
    boot: QiniBootstrap,
    config: str,
    baseline: str,
    *,
    models: Sequence[str],
    confidence: float = DEFAULT_CONFIDENCE,
) -> FeatureGain:
    """Paired gain of ``config`` over ``baseline`` with a percentile bootstrap CI."""
    per_model = {m: boot.estimates[(config, m)] - boot.estimates[(baseline, m)] for m in models}
    draws = np.mean([boot.draws[(config, m)] - boot.draws[(baseline, m)] for m in models], axis=0)
    alpha = (1 - confidence) / 2
    lower, upper = np.quantile(draws, [alpha, 1 - alpha])
    return FeatureGain(
        config=config,
        baseline=baseline,
        difference=float(np.mean(list(per_model.values()))),
        lower=float(lower),
        upper=float(upper),
        win_rate=float((draws > 0).mean()),
        per_model=per_model,
    )


def choose_feature_set(
    gains_vs_base: Mapping[FeatureGroup, FeatureGain],
    *,
    all_vs_kept: FeatureGain | None,
    base: Sequence[FeatureGroup] = BASE_GROUPS,
) -> tuple[FeatureConfig, tuple[FeatureGroup, ...], str]:
    """Apply the decision rule; return (chosen config, kept groups, reason).

    Args:
        gains_vs_base: Each candidate group's add-one gain over base, in order.
        all_vs_kept: All groups vs the kept set; unused (may be None) when
            every candidate is kept, since the two are then the same config.

    Raises:
        ValueError: If a comparison is needed but ``all_vs_kept`` is None.
    """
    candidates = tuple(gains_vs_base)
    kept = tuple(group for group, gain in gains_vs_base.items() if gain.lower > 0)
    everything = FeatureConfig(ALL_CONFIG, (*base, *candidates))
    if kept == candidates:
        return everything, kept, "every candidate group beats base"
    if all_vs_kept is None:
        raise ValueError("all groups must be compared with the kept set")
    kept_config = FeatureConfig.added(base, kept)
    if all_vs_kept.lower > 0:
        return everything, kept, f"all groups significantly better than {kept_config.name}"
    return kept_config, kept, f"all groups tied with {kept_config.name}; smaller set kept"


@dataclass
class AblationResult:
    """Every config's CV results, the paired gains, and the chosen feature set."""

    configs: list[FeatureConfig]
    models: tuple[str, ...]
    cv_results: list[CVResult]
    bootstrap: QiniBootstrap
    gains: dict[str, FeatureGain]
    all_vs_kept: FeatureGain | None
    kept_groups: tuple[FeatureGroup, ...]
    chosen: FeatureConfig
    reason: str
    n_folds: int
    confidence: float

    def reported_gains(self) -> list[FeatureGain]:
        """Every gain vs base, plus all vs kept unless that is the same comparison.

        With nothing kept, "all vs kept" *is* "all vs base".
        """
        gains = list(self.gains.values())
        seen = {(g.config, g.baseline) for g in gains}
        extra = self.all_vs_kept
        if extra is not None and (extra.config, extra.baseline) not in seen:
            gains.append(extra)
        return gains


def _run_config(
    config: FeatureConfig,
    make_frame: FrameFactory,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    reference: ModelFrame,
    *,
    models: Sequence[str],
    overrides: Mapping[str, Params],
    seed: int,
    progress: Callable[[str], None],
) -> list[CVResult]:
    frame = make_frame(config.groups)
    if not (
        frame.client_ids.equals(reference.client_ids)
        and np.array_equal(frame.treatment, reference.treatment)
        and np.array_equal(frame.outcome, reference.outcome)
    ):
        raise ValueError(f"{config.name} frame has different clients from the base frame")
    results = []
    for model_name in models:
        result = out_of_fold_scores(
            model_name,
            frame,
            folds,
            config=config.name,
            overrides=overrides.get(model_name),
            seed=seed,
        )
        progress(
            f"  {config.name:40s} {model_name:22s} CV Qini {result.cv_mean:+.4f} "
            f"(sd {result.cv_std:.4f})  {result.fit_seconds:.0f}s"
        )
        results.append(result)
    return results


def run_ablation(
    make_frame: FrameFactory,
    *,
    models: Sequence[str] = ABLATION_MODELS,
    overrides: Mapping[str, Params] | None = None,
    base: Sequence[FeatureGroup] = BASE_GROUPS,
    candidates: Sequence[FeatureGroup] = CANDIDATE_GROUPS,
    n_folds: int = DEFAULT_N_FOLDS,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_SEED,
    progress: Callable[[str], None] = lambda _message: None,
) -> AblationResult:
    """Cross-validate every config, compare gains, and choose the feature set.

    If the kept set is not already an evaluated config (two or more groups
    kept, but not all), it is cross-validated as one extra config so all
    groups can be compared with it.

    Args:
        make_frame: Builds the train ``ModelFrame`` for a list of groups; every
            frame must hold the same clients in the same order.
        overrides: Hyperparameters per model (e.g. the Phase 10 tuned params).
    """
    overrides = overrides or {}
    configs = ablation_configs(base, candidates)
    reference = make_frame(configs[0].groups)
    folds = cv_folds(reference.treatment, reference.outcome, n_folds=n_folds, seed=seed)
    run = {"models": models, "overrides": overrides, "seed": seed, "progress": progress}

    cv_results = [
        result
        for config in configs
        for result in _run_config(config, make_frame, folds, reference, **run)
    ]
    boot_args = {"n_bootstrap": n_bootstrap, "seed": seed}
    progress(f"Bootstrapping {len(cv_results)} rankings ({n_bootstrap} paired resamples)...")
    boot = bootstrap_qini(
        {(r.config, r.model_name): r.oof_scores for r in cv_results},
        reference.treatment,
        reference.outcome,
        **boot_args,
    )
    gains = {
        c.name: feature_gain(boot, c.name, BASE_CONFIG, models=models, confidence=confidence)
        for c in configs[1:]
    }
    add_one = {group: gains[FeatureConfig.added(base, (group,)).name] for group in candidates}

    kept = tuple(group for group, gain in add_one.items() if gain.lower > 0)
    all_vs_kept = None
    if kept != tuple(candidates):
        kept_config = FeatureConfig.added(base, kept)
        if kept_config.name not in {c.name for c in configs}:
            extra = _run_config(kept_config, make_frame, folds, reference, **run)
            cv_results += extra
            configs.append(kept_config)
            boot = boot.merged(
                bootstrap_qini(
                    {(r.config, r.model_name): r.oof_scores for r in extra},
                    reference.treatment,
                    reference.outcome,
                    **boot_args,
                )
            )
            gains[kept_config.name] = feature_gain(
                boot, kept_config.name, BASE_CONFIG, models=models, confidence=confidence
            )
        all_vs_kept = feature_gain(
            boot, ALL_CONFIG, kept_config.name, models=models, confidence=confidence
        )

    chosen, kept, reason = choose_feature_set(add_one, all_vs_kept=all_vs_kept, base=base)
    return AblationResult(
        configs=configs,
        models=tuple(models),
        cv_results=cv_results,
        bootstrap=boot,
        gains=gains,
        all_vs_kept=all_vs_kept,
        kept_groups=kept,
        chosen=chosen,
        reason=reason,
        n_folds=n_folds,
        confidence=confidence,
    )


def _groups_tag(groups: Sequence[FeatureGroup]) -> str:
    return ",".join(group.value for group in groups)


def _gain_row(gain: FeatureGain) -> dict:
    row = {k: v for k, v in asdict(gain).items() if k != "per_model"}
    return row | {f"diff_{model}": diff for model, diff in gain.per_model.items()}


def _metric_suffix(config: str) -> str:
    # MLflow metric names don't allow "+".
    return config.replace("+", "plus_")


def log_ablation(
    result: AblationResult,
    *,
    feature_tags: Mapping[str, str],
    config: TrackingConfig | None = None,
    lineage: dict[str, Path] | None = None,
) -> str:
    """Log one CV run per feature config and an ``ablation`` summary run; return its id.

    Args:
        feature_tags: ``StoredFeatureTable.lineage_tags(...)`` of the table the
            frames came from; each config run's ``feature_groups`` is its own.
    """
    common = {"n_folds": result.n_folds, "seed": result.bootstrap.seed}
    by_config: dict[str, list[CVResult]] = {}
    for cv in result.cv_results:
        by_config.setdefault(cv.config, []).append(cv)

    for feature_config in result.configs:
        with start_run(
            Experiment.FEATURE_ABLATION,
            run_name=f"cv-{feature_config.name}",
            tags={
                **feature_tags,
                "run_type": "cv",
                "feature_config": feature_config.name,
                "feature_groups": _groups_tag(feature_config.groups),
            },
            config=config,
            **(lineage or {}),
        ):
            mlflow.log_params({**common, "models": ",".join(result.models)})
            for cv in by_config[feature_config.name]:
                mlflow.log_metrics(
                    {
                        f"cv_qini_mean_{cv.model_name}": cv.cv_mean,
                        f"cv_qini_std_{cv.model_name}": cv.cv_std,
                        f"oof_qini_{cv.model_name}": result.bootstrap.estimates[
                            (cv.config, cv.model_name)
                        ],
                        f"fit_seconds_{cv.model_name}": cv.fit_seconds,
                    }
                )

    gains = result.reported_gains()
    cv_table = pl.DataFrame(
        [
            {
                "config": cv.config,
                "model": cv.model_name,
                "cv_qini_mean": cv.cv_mean,
                "cv_qini_std": cv.cv_std,
                "fold_qini": json.dumps(list(cv.fold_qini)),
                "oof_qini": result.bootstrap.estimates[(cv.config, cv.model_name)],
                "fit_seconds": cv.fit_seconds,
            }
            for cv in result.cv_results
        ]
    )
    decision = {
        "chosen": result.chosen.name,
        "chosen_groups": [group.value for group in result.chosen.groups],
        "kept_groups": [group.value for group in result.kept_groups],
        "reason": result.reason,
        "confidence": result.confidence,
        "n_bootstrap": result.bootstrap.n_bootstrap,
    }
    with start_run(
        Experiment.FEATURE_ABLATION,
        run_name="ablation",
        tags={
            **feature_tags,
            "run_type": "ablation",
            "chosen_feature_set": result.chosen.name,
            "chosen_feature_groups": _groups_tag(result.chosen.groups),
            "kept_groups": _groups_tag(result.kept_groups),
        },
        config=config,
        **(lineage or {}),
    ) as run:
        mlflow.log_params({**common, "n_bootstrap": result.bootstrap.n_bootstrap})
        mlflow.log_metrics(
            {
                f"qini_gain_{_metric_suffix(gain.config)}_vs_{_metric_suffix(gain.baseline)}": (
                    gain.difference
                )
                for gain in gains
            }
        )
        mlflow.log_text(cv_table.write_csv(), "ablation/cv_results.csv")
        mlflow.log_text(
            pl.DataFrame([_gain_row(g) for g in gains]).write_csv(), "ablation/feature_gains.csv"
        )
        mlflow.log_dict(decision, "ablation/decision.json")
        return run.info.run_id


@dataclass(frozen=True)
class Arm:
    """A feature set with its own per-model hyperparameters."""

    name: str
    groups: tuple[FeatureGroup, ...]
    overrides: Mapping[str, Params]


@dataclass
class ArmComparison:
    """Two arms' CV results and the challenger's paired gain over the baseline."""

    challenger: Arm
    baseline: Arm
    models: tuple[str, ...]
    cv_results: list[CVResult]
    bootstrap: QiniBootstrap
    gain: FeatureGain
    n_folds: int

    @property
    def verdict(self) -> str:
        if self.gain.lower > 0:
            return "challenger better"
        if self.gain.upper < 0:
            return "challenger worse"
        return "tied"


def compare_arms(
    make_frame: FrameFactory,
    challenger: Arm,
    baseline: Arm,
    *,
    models: Sequence[str] = ABLATION_MODELS,
    n_folds: int = DEFAULT_N_FOLDS,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_SEED,
    progress: Callable[[str], None] = lambda _message: None,
) -> ArmComparison:
    """Paired CV comparison of two feature sets, each with its own tuned params.

    The ablation holds one set of params fixed across feature sets; this asks
    whether a feature set wins once it gets params tuned for it. Both arms run
    on the same folds and bootstrap resamples as the ablation.
    """
    reference = make_frame(baseline.groups)
    folds = cv_folds(reference.treatment, reference.outcome, n_folds=n_folds, seed=seed)
    cv_results = []
    for arm in (baseline, challenger):
        cv_results += _run_config(
            FeatureConfig(arm.name, arm.groups),
            make_frame,
            folds,
            reference,
            models=models,
            overrides=arm.overrides,
            seed=seed,
            progress=progress,
        )
    progress(f"Bootstrapping {len(cv_results)} rankings ({n_bootstrap} paired resamples)...")
    boot = bootstrap_qini(
        {(r.config, r.model_name): r.oof_scores for r in cv_results},
        reference.treatment,
        reference.outcome,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    gain = feature_gain(boot, challenger.name, baseline.name, models=models, confidence=confidence)
    return ArmComparison(challenger, baseline, tuple(models), cv_results, boot, gain, n_folds)


def log_arm_comparison(
    result: ArmComparison,
    *,
    feature_tags: Mapping[str, str],
    config: TrackingConfig | None = None,
    lineage: dict[str, Path] | None = None,
) -> str:
    """Log the comparison (both arms' params, CV results, paired gain) as one run."""
    params: dict[str, object] = {"n_folds": result.n_folds, "seed": result.bootstrap.seed}
    for role, arm in (("challenger", result.challenger), ("baseline", result.baseline)):
        for model in result.models:
            for key, value in (arm.overrides.get(model) or {}).items():
                params[f"{role}_{model}_{key}"] = value
    with start_run(
        Experiment.FEATURE_ABLATION,
        run_name=f"compare-{result.challenger.name}-vs-{result.baseline.name}",
        tags={
            **feature_tags,
            "run_type": "tuned_comparison",
            "challenger": result.challenger.name,
            "baseline": result.baseline.name,
            "challenger_groups": _groups_tag(result.challenger.groups),
            "baseline_groups": _groups_tag(result.baseline.groups),
            "verdict": result.verdict,
        },
        config=config,
        **(lineage or {}),
    ) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(
            {
                "qini_gain": result.gain.difference,
                "qini_gain_ci_lower": result.gain.lower,
                "qini_gain_ci_upper": result.gain.upper,
                "qini_gain_win_rate": result.gain.win_rate,
                **{
                    f"{cv.config}_cv_qini_mean_{cv.model_name}": cv.cv_mean
                    for cv in result.cv_results
                },
            }
        )
        cv_table = pl.DataFrame(
            [
                {
                    "arm": cv.config,
                    "model": cv.model_name,
                    "cv_qini_mean": cv.cv_mean,
                    "cv_qini_std": cv.cv_std,
                    "fold_qini": json.dumps(list(cv.fold_qini)),
                    "oof_qini": result.bootstrap.estimates[(cv.config, cv.model_name)],
                }
                for cv in result.cv_results
            ]
        )
        mlflow.log_text(cv_table.write_csv(), "comparison/cv_results.csv")
        mlflow.log_dict(_gain_row(result.gain), "comparison/gain.json")
        return run.info.run_id
