"""Unit tests for the SHAP importance and dependence figures."""

import numpy as np
import polars as pl
import pytest
from matplotlib.figure import Figure

from promolift.evaluation import plots
from promolift.explainability.importance import (
    MISSING_LABEL,
    dependence_table,
    global_importance,
)
from promolift.explainability.shap_values import explain


@pytest.fixture(scope="module")
def explanation(explained_model):
    return explain(explained_model.model, explained_model.features)


def _tables(explanation, features=("b", "noise", "segment")) -> dict[str, pl.DataFrame]:
    return {feature: dependence_table(explanation, feature) for feature in features}


# --- importance ---------------------------------------------------------------------------


def test_importance_bars_follow_the_ranking_with_share_labels(explanation) -> None:
    table = global_importance(explanation)

    figure = plots.importance_figure(table)

    assert isinstance(figure, Figure)
    (axes,) = figure.axes
    assert [t.get_text() for t in axes.get_yticklabels()] == table["feature"].to_list()
    assert len(axes.patches) == table.height
    heights = [patch.get_width() for patch in axes.patches]
    assert heights == pytest.approx(list(table["mean_abs_contribution"] * 100))
    labels = [t.get_text() for t in axes.texts]
    assert labels == [f"{share:.0%}" for share in table["share"]]
    assert "pp" in axes.get_xlabel()


def test_importance_shows_only_the_top_features(explanation) -> None:
    figure = plots.importance_figure(global_importance(explanation), top=2)

    (axes,) = figure.axes
    assert len(axes.patches) == 2


# --- dependence ---------------------------------------------------------------------------


def test_one_panel_per_feature_titled_by_name(explanation) -> None:
    figure = plots.dependence_figure(_tables(explanation))

    assert [axes.get_title(loc="left") for axes in figure.axes] == ["b", "noise", "segment"]


def test_numeric_features_are_lines_and_categorical_ones_bars(explanation) -> None:
    figure = plots.dependence_figure(_tables(explanation))

    numeric, _, categorical = figure.axes
    # The axis-zero reference line is a line too, so look for the data line by point count.
    assert any(len(line.get_xdata()) > 2 for line in numeric.lines)
    assert len(numeric.patches) == 0
    assert len(categorical.patches) == 3  # F, M, U


def test_x_tick_labels_are_the_bin_labels(explanation) -> None:
    tables = _tables(explanation, features=("b",))

    (axes,) = plots.dependence_figure(tables).axes

    assert [t.get_text() for t in axes.get_xticklabels()] == tables["b"]["label"].to_list()


def test_clients_missing_a_feature_get_a_hollow_marker() -> None:
    table = pl.DataFrame(
        {
            "label": ["1", "2", MISSING_LABEL],
            "n": [5, 5, 2],
            "mean_contribution": [0.02, 0.04, -0.01],
            "mean_value": [1.0, 2.0, None],
        }
    )

    (axes,) = plots.dependence_figure({"x": table}).axes

    markers = [line for line in axes.lines if len(line.get_xdata()) == 1]
    assert len(markers) == 1
    assert list(markers[0].get_xdata()) == [2]
    assert markers[0].get_ydata()[0] == pytest.approx(-1.0)  # percentage points


def test_unused_panels_are_hidden() -> None:
    table = pl.DataFrame(
        {
            "label": ["a", "b"],
            "n": [1, 1],
            "mean_contribution": [0.1, -0.1],
            "mean_value": [None, None],
        }
    )

    figure = plots.dependence_figure({"only": table, "two": table}, columns=3)

    visible = [axes.get_visible() for axes in figure.axes]
    assert visible.count(True) == 2


def test_a_plot_needs_at_least_one_feature() -> None:
    with pytest.raises(ValueError, match="at least one feature"):
        plots.dependence_figure({})


def test_the_missing_label_matches_the_one_the_tables_use() -> None:
    assert plots._MISSING_LABEL == MISSING_LABEL


def test_values_are_in_percentage_points(explanation) -> None:
    table = dependence_table(explanation, "b", bins=4)

    (axes,) = plots.dependence_figure({"b": table}).axes

    line = max(axes.lines, key=lambda line: len(line.get_xdata()))
    assert np.asarray(line.get_ydata()) == pytest.approx(
        table["mean_contribution"].to_numpy() * 100
    )
