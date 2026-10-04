"""Static plots logged as MLflow artifacts: Qini curves, decile uplift, and profit curves.

Built on ``matplotlib.figure.Figure`` directly rather than pyplot, so no GUI
backend or global figure state is involved (safe on headless CI). Colors,
ink, and mark weights follow the project's charting palette: categorical
hues in fixed order, ink-colored text, hairline recessive gridlines.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

if TYPE_CHECKING:
    from promolift.evaluation.report import EvaluationReport
    from promolift.optimization.targeting import ProfitCurve

# Categorical slots in fixed order (never cycled); validated colorblind-safe
# as a set. Light surface only: these are static images, not themed UI.
_SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
_SURFACE = "#fcfcfb"
_INK_PRIMARY = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_INK_MUTED = "#898781"
_GRID = "#e1e0d9"
_AXIS = "#c3c2b7"


def _axes(title: str) -> tuple[Figure, Axes]:
    figure = Figure(figsize=(7.0, 4.2), dpi=150, facecolor=_SURFACE, layout="constrained")
    axes = figure.add_subplot(facecolor=_SURFACE)
    axes.set_title(title, loc="left", color=_INK_PRIMARY, fontsize=11, fontweight="bold")
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(_AXIS)
        axes.spines[side].set_linewidth(0.8)
    axes.tick_params(colors=_INK_MUTED, labelsize=8, length=0)
    axes.grid(axis="y", color=_GRID, linewidth=0.5)
    axes.set_axisbelow(True)
    axes.xaxis.label.set_color(_INK_SECONDARY)
    axes.yaxis.label.set_color(_INK_SECONDARY)
    return figure, axes


def qini_curve_figure(reports: Sequence[EvaluationReport]) -> Figure:
    """Overlay the Qini curves of one or more models evaluated on the same split.

    The random-targeting reference is the straight line to the same end
    point: every ranking reaches the same total uplift at 100% targeted, so
    the area between a curve and this line is what the ranking adds.
    """
    if not reports:
        raise ValueError("need at least one report")
    if len({(r.split, r.n_clients) for r in reports}) > 1:
        raise ValueError("Qini curves can only be overlaid for reports on the same split")
    if len(reports) > len(_SERIES):
        msg = f"at most {len(_SERIES)} models per plot; facet or fold the rest"
        raise ValueError(msg)

    figure, axes = _axes(f"Qini curve, {reports[0].split} split")
    for report, color in zip(reports, _SERIES, strict=False):
        curve = report.qini_curve
        axes.plot(
            curve.fraction_targeted * 100,
            curve.incremental_conversions,
            color=color,
            linewidth=2,
            solid_capstyle="round",
            solid_joinstyle="round",
            label=report.model_name,
        )
    end = reports[0].qini_curve.incremental_conversions[-1]
    axes.plot([0, 100], [0, end], color=_INK_MUTED, linewidth=1.2, label="Random targeting")

    axes.set_xlim(0, 100)
    axes.xaxis.set_major_formatter(PercentFormatter())
    axes.set_xlabel("Clients targeted, ranked by score", fontsize=9)
    axes.set_ylabel("Incremental conversions", fontsize=9)
    axes.legend(frameon=False, loc="upper left", fontsize=8, labelcolor=_INK_PRIMARY)
    return figure


def decile_uplift_figure(report: EvaluationReport) -> Figure:
    """Observed uplift per score decile against the ATE of untargeted sending.

    A good ranking shows bars well above the ATE line on the left, falling
    below it on the right.
    """
    figure, axes = _axes(f"Uplift by score decile, {report.model_name}, {report.split} split")
    uplift_pp = report.deciles["uplift"].to_numpy() * 100
    positions = range(len(uplift_pp))
    axes.bar(positions, uplift_pp, width=0.45, color=_SERIES[0], linewidth=0)
    axes.axhline(0, color=_AXIS, linewidth=0.8)

    ate_pp = report.ate * 100
    axes.axhline(ate_pp, color=_INK_MUTED, linewidth=1.2)
    # Labelled just outside the plot's right edge, where no bar can collide with it.
    axes.annotate(
        f"ATE\n{ate_pp:+.1f}pp",
        xy=(1, ate_pp),
        xycoords=("axes fraction", "data"),
        xytext=(4, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=8,
        color=_INK_SECONDARY,
        annotation_clip=False,
    )

    axes.set_xticks(list(positions), report.deciles["bucket"].to_list())
    axes.set_xlabel("Score decile, highest predicted uplift first", fontsize=9)
    axes.set_ylabel("Uplift (pp)", fontsize=9)
    return figure


def profit_curve_figure(
    curves: Mapping[str, ProfitCurve], *, chosen: str, chosen_depth: float
) -> Figure:
    """Profit per 1,000 clients by targeting depth, with the chosen model's CI band.

    The chosen depth is marked on the chosen model's curve; profit 0 is
    texting nobody.
    """
    if chosen not in curves:
        raise ValueError(f"{chosen!r} has no curve")
    if len(curves) > len(_SERIES):
        raise ValueError(f"at most {len(_SERIES)} curves per plot")

    economics = curves[chosen].economics
    figure, axes = _axes(
        f"Profit by targeting depth (break-even uplift {economics.break_even_uplift:.1%})"
    )
    for (label, curve), color in zip(curves.items(), _SERIES, strict=False):
        depth_pct = curve.depths * 100
        if label == chosen:
            axes.fill_between(
                depth_pct, curve.profit_lower, curve.profit_upper, color=color, alpha=0.15, lw=0
            )
        axes.plot(depth_pct, curve.profit_per_1000, color=color, linewidth=2, label=label)
    axes.axhline(0, color=_AXIS, linewidth=0.8)

    chosen_curve = curves[chosen]
    index = int(abs(chosen_curve.depths - chosen_depth).argmin())
    axes.plot(
        chosen_depth * 100,
        chosen_curve.profit_per_1000[index],
        marker="o",
        markersize=7,
        color=_SERIES[list(curves).index(chosen)],
        markeredgecolor=_SURFACE,
        linestyle="none",
        label=f"chosen depth {chosen_depth:.0%}",
    )

    axes.set_xlim(0, 100)
    axes.xaxis.set_major_formatter(PercentFormatter())
    axes.set_xlabel("Clients texted, ranked by score", fontsize=9)
    axes.set_ylabel("Profit per 1,000 clients", fontsize=9)
    axes.legend(frameon=False, loc="best", fontsize=8, labelcolor=_INK_PRIMARY)
    return figure
