"""Static plots logged as MLflow artifacts: Qini, decile uplift, profit, campaign, and SHAP.

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
    import polars as pl

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
# The row label dependence tables give clients missing a feature
# (explainability.importance.MISSING_LABEL; a test keeps the two equal).
_MISSING_LABEL = "missing"


def _axes(title: str) -> tuple[Figure, Axes]:
    figure = Figure(figsize=(7.0, 4.2), dpi=150, facecolor=_SURFACE, layout="constrained")
    axes = figure.add_subplot(facecolor=_SURFACE)
    _style_axes(axes, title)
    return figure, axes


def _style_axes(axes: Axes, title: str, *, title_size: int = 11) -> None:
    axes.set_title(title, loc="left", color=_INK_PRIMARY, fontsize=title_size, fontweight="bold")
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


def decile_calibration_figure(deciles: pl.DataFrame) -> Figure:
    """Realized uplift (with its interval) against predicted uplift, per score decile of a campaign.

    ``deciles`` is ``monitoring.campaign.decile_comparison``'s table. A
    calibrated model has the realized points on the predicted line; deciles
    without both arms are left out.
    """
    shown = deciles.filter(deciles["realized_uplift"].is_not_null())
    figure, axes = _axes("Campaign uplift by predicted-uplift decile")
    deciles_axis = shown["decile"].to_list()
    realized = shown["realized_uplift"].to_numpy() * 100
    axes.errorbar(
        deciles_axis,
        realized,
        yerr=[
            realized - shown["realized_lower"].to_numpy() * 100,
            shown["realized_upper"].to_numpy() * 100 - realized,
        ],
        color=_SERIES[0],
        linestyle="none",
        marker="o",
        markersize=6,
        capsize=3,
        linewidth=1.4,
        label="realized (95% interval)",
    )
    axes.plot(
        deciles_axis,
        shown["predicted_uplift"].to_numpy() * 100,
        color=_SERIES[1],
        linewidth=2,
        marker="s",
        markersize=5,
        label="predicted",
    )
    axes.axhline(0, color=_AXIS, linewidth=0.8)
    axes.set_xticks(deciles_axis)
    axes.set_xlabel("Score decile, highest predicted uplift first", fontsize=9)
    axes.set_ylabel("Uplift (pp)", fontsize=9)
    axes.legend(frameon=False, loc="best", fontsize=8, labelcolor=_INK_PRIMARY)
    return figure


def importance_figure(importance: pl.DataFrame, *, top: int = 12) -> Figure:
    """Mean absolute SHAP contribution of the ``top`` features, in percentage points of uplift.

    ``importance`` is ``explainability.importance.global_importance``'s table
    (most important first); each bar is labelled with the feature's share of
    the total across all features.
    """
    shown = importance.head(top)
    figure, axes = _axes("What drives a client's predicted uplift")
    labels = shown["feature"].to_list()
    values = shown["mean_abs_contribution"].to_numpy() * 100
    axes.barh(range(len(labels)), values, color=_SERIES[0], height=0.65)
    axes.set_yticks(range(len(labels)), labels)
    axes.invert_yaxis()
    axes.grid(axis="y", visible=False)
    axes.grid(axis="x", color=_GRID, linewidth=0.5)
    for position, (value, share) in enumerate(zip(values, shown["share"].to_list(), strict=True)):
        axes.annotate(
            f"{share:.0%}",
            xy=(value, position),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=_INK_SECONDARY,
        )
    axes.set_xlabel("Mean absolute contribution to uplift (pp); labels: share of total", fontsize=9)
    return figure


def dependence_figure(tables: Mapping[str, pl.DataFrame], *, columns: int = 3) -> Figure:
    """Mean SHAP contribution along each feature, one small panel per feature.

    ``tables`` maps a feature name to ``explainability.importance.dependence_table``'s
    table. Numeric features are drawn as a line through their bins, categorical
    ones as bars; clients missing the feature get a hollow marker at the end.
    """
    if not tables:
        raise ValueError("need at least one feature to plot")
    rows = -(-len(tables) // columns)
    figure = Figure(
        figsize=(3.6 * min(columns, len(tables)), 2.9 * rows),
        dpi=150,
        facecolor=_SURFACE,
        layout="constrained",
    )
    for axes, (feature, table) in zip(
        figure.subplots(rows, min(columns, len(tables)), squeeze=False).ravel(),
        tables.items(),
        strict=False,
    ):
        axes.set_facecolor(_SURFACE)
        _style_axes(axes, feature, title_size=9)
        positions = list(range(table.height))
        effect = table["mean_contribution"].to_numpy() * 100
        missing = table["label"].to_numpy() == _MISSING_LABEL
        ordered = ~missing
        numeric = table["mean_value"].drop_nulls().len() > 0
        if numeric:
            axes.plot(
                [p for p, keep in zip(positions, ordered, strict=True) if keep],
                effect[ordered],
                color=_SERIES[0],
                linewidth=1.8,
                marker="o",
                markersize=4,
            )
        else:
            axes.bar(
                [p for p, keep in zip(positions, ordered, strict=True) if keep],
                effect[ordered],
                color=_SERIES[0],
                width=0.6,
            )
        if missing.any():
            axes.plot(
                [p for p, miss in zip(positions, missing, strict=True) if miss],
                effect[missing],
                linestyle="none",
                marker="o",
                markersize=5,
                markerfacecolor=_SURFACE,
                markeredgecolor=_SERIES[1],
            )
        axes.axhline(0, color=_AXIS, linewidth=0.8)
        axes.set_xticks(positions, table["label"].to_list(), rotation=45, ha="right")
        axes.tick_params(axis="x", labelsize=7)
        axes.set_ylabel("pp", fontsize=8)
    for axes in figure.axes[len(tables) :]:
        axes.set_visible(False)
    return figure
