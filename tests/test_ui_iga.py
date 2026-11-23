"""Headless GUI tests for knot display, knot insertion and the IGA Analysis tab,
including full load -> refine -> tag -> export -> reopen workflows through the main window."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PyQt5.QtWidgets import QFileDialog

from core import io_json
from core.bspline import bspline_curve, element_boundaries
from core.nurbs import nurbs_curve


@pytest.fixture
def window(qapp):
    from app import MainWindow
    return MainWindow()


def _labels(fig):
    return [line.get_label() for line in fig.axes[0].get_lines()]


def _knot_marker_count(fig):
    lines = [l for l in fig.axes[0].get_lines() if l.get_label() == "Knots (element ends)"]
    return len(lines[0].get_xdata()) if lines else 0


def _open(monkeypatch, path):
    monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(lambda *a: (str(path), "")))


def _save(monkeypatch, path):
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a: (str(path), "")))


# ---------------------------------------------------------------- B-spline tab

def test_bspline_knots_toggle_marks_every_knot(window, dialogs):
    tab = window._bspline_tab
    tab._draw_curve()
    assert _knot_marker_count(tab._figure) == 0
    tab._show_knots_chk.setChecked(True)                      # redraws by itself
    assert _knot_marker_count(tab._figure) == 6               # knots 0..5
    assert "5 elements (knot spans)" in window._status_bar.currentMessage()
    line = [l for l in tab._figure.axes[0].get_lines() if l.get_label() == "Knots (element ends)"][0]
    knots = tab.parse_knots()
    expected = bspline_curve(knots, 2, tab.get_control_points(), u_values=element_boundaries(knots, 2))
    np.testing.assert_allclose(np.column_stack(line.get_data()), expected, atol=1e-12)
    texts = [t.get_text() for t in tab._figure.axes[0].texts]
    assert "u=2.5" not in texts and "u=3" in texts
    assert dialogs == []


def test_bspline_insert_knot_and_split_keep_shape(window, dialogs):
    tab = window._bspline_tab
    knots0, cp0 = tab.parse_knots(), tab.get_control_points()
    u = np.linspace(0, 5, 301)
    before = bspline_curve(knots0, 2, cp0, u_values=u)

    tab._refine_box.u_edit.setText("2.5")
    tab._insert_knot()
    assert dialogs == []
    assert tab._table.rowCount() == 9
    assert 2.5 in tab.parse_knots()
    np.testing.assert_allclose(bspline_curve(tab.parse_knots(), 2, tab.get_control_points(), u_values=u),
                               before, atol=1e-12)
    msg = window._status_bar.currentMessage()
    assert msg.startswith("Inserted knot u=2.5 (1x): 8 -> 9 control points, max shape change")

    tab._split_elements()
    assert len(element_boundaries(tab.parse_knots(), 2)) - 1 == 12
    np.testing.assert_allclose(bspline_curve(tab.parse_knots(), 2, tab.get_control_points(), u_values=u),
                               before, atol=1e-11)


@pytest.mark.parametrize("text, times, msg", [
    ("", 1, "Enter the knot value"), ("abc", 1, "is not a number"),
    ("7", 1, "strictly inside"), ("4", 1, "already appears 2 time"),
])
def test_bspline_insert_knot_errors(window, dialogs, text, times, msg):
    tab = window._bspline_tab
    tab._refine_box.u_edit.setText(text)
    tab._refine_box.times_spin.setValue(times)
    tab._insert_knot()
    assert dialogs[-1][1] == "Insert Knot" and msg in dialogs[-1][2]
    assert tab._table.rowCount() == 8                          # nothing changed


# ---------------------------------------------------------------- NURBS curve tab

def test_nurbs_circle_insert_keeps_exact_circle(window, dialogs):
    tab = window._nurbs_curve_tab
    tab._load_circle()
    tab._show_knots_chk.setChecked(True)                      # nothing drawn yet -> no redraw
    assert tab._figure.axes == []
    tab._draw()
    assert _knot_marker_count(tab._figure) == 5               # 0, .25, .5, .75, 1 (1 == 0 on the circle)
    tab._refine_box.u_edit.setText("0.125")
    tab._insert_knot()
    assert dialogs == []
    assert tab._table.rowCount() == 10
    msg = window._status_bar.currentMessage()
    assert "9 -> 10 control points" in msg and "Circle: centre" in msg and "5 elements" in msg
    err = float(msg.rsplit("max radius error = ", 1)[1])
    assert err < 1e-12
    assert _knot_marker_count(tab._figure) == 6


def test_nurbs_knots_toggle_does_not_replace_basis_plot(window, dialogs):
    tab = window._nurbs_curve_tab
    tab._load_t_profile()
    tab._plot_rational_basis()
    tab._show_knots_chk.setChecked(True)
    assert tab._figure.axes[0].get_title().startswith("NURBS Rational Basis")
    tab._split_elements()                                     # 10 -> 20 elements
    assert "20 elements" in window._status_bar.currentMessage()
    data = tab.get_data()
    u = np.linspace(0, 1, 500)
    from core.nurbs import t_profile
    cp, w, k, d = t_profile()
    np.testing.assert_allclose(nurbs_curve(data["knots"], 2, data["control_points"], data["weights"], u_values=u),
                               nurbs_curve(k, d, cp, w, u_values=u), atol=1e-12)


# ---------------------------------------------------------------- NURBS surface tab

def test_surface_knots_toggle_draws_element_lines(window, dialogs):
    tab = window._nurbs_surface_tab
    tab._load_t_shape()
    tab._draw()
    assert "Element boundaries" not in _labels(tab._figure)
    assert "10 element(s) in u x 1 in v = 10" in window._status_bar.currentMessage()
    tab._show_knots_chk.setChecked(True)
    assert "Element boundaries" in _labels(tab._figure)
    assert dialogs == []


# ---------------------------------------------------------------- IGA Analysis tab

@pytest.fixture
def analysis(window):
    window._tabs.setCurrentWidget(window._analysis_tab)
    return window._analysis_tab


def _load_example(tab, name):
    tab._example_combo.setCurrentText(name)
    tab._load_example()


def test_analysis_empty_state_guides_the_user(analysis, dialogs):
    assert "No geometry loaded yet" in analysis._summary.toPlainText()
    assert "Step 1" in analysis._summary.toPlainText()
    analysis._split("both")
    analysis._insert()
    analysis.export_json("never.json")
    assert [d[2] for d in dialogs] == ["Load a geometry first (Step 1).",
                                       "Load a geometry first (Step 1).",
                                       "Nothing to export yet: load a geometry in Step 1."]
    with pytest.raises(ValueError, match="No analysis model yet"):
        analysis.get_data()


def test_analysis_plate_workflow(analysis, window, dialogs, tmp_path, monkeypatch):
    _load_example(analysis, "Plate with a hole (2D)")
    assert analysis._elem_table.rowCount() == 2
    assert "Elements:                  2 (u) x 1 (v) = 2" in analysis.summary_text()

    analysis._split("both")
    analysis._split("u")
    assert analysis._elem_table.rowCount() == 16
    assert "Split every element in U: now 8 x 2 = 16 elements" in window._status_bar.currentMessage()

    analysis._dir_combo.setCurrentText("V")
    analysis._knot_edit.setText("0.25")
    analysis._insert()
    assert analysis._model.n_elements_v == 3 and dialogs == []

    analysis._bc_combos["u = 1"].setCurrentText("Fixed (Dirichlet)")
    analysis._bc_combos["v = 1"].setCurrentText("Load (Neumann)")
    assert "u = 1:  Fixed (Dirichlet)" in analysis.summary_text()
    labels = _labels(analysis._figure)
    assert "u = 1: Fixed (Dirichlet)" in labels and "v = 1: Load (Neumann)" in labels

    # All displays on, then select an edge and an element
    for chk in (analysis._chk_net, analysis._chk_gauss):
        chk.setChecked(True)
    assert "Gauss points" in _labels(analysis._figure) and "Control net" in _labels(analysis._figure)
    analysis._bc_table.selectRow(1)
    assert "Selected edge u = 1" in _labels(analysis._figure)
    analysis._results.setCurrentIndex(1)
    analysis._elem_table.selectRow(5)
    assert "Selected element 5" in _labels(analysis._figure)
    statuses = {analysis._elem_table.item(r, 7).text() for r in range(analysis._elem_table.rowCount())}
    assert statuses == {"OK"}

    # Export both formats; the JSON reopens through File -> Open with conditions intact
    js, mf = tmp_path / "plate.json", tmp_path / "plate.m"
    assert analysis.export_json(str(js)) == str(js)
    assert analysis.export_matlab(str(mf)) == str(mf)
    assert "geometry.form = 'B-NURBS';" in mf.read_text()
    expected = analysis.get_data()

    window._file_new()
    assert analysis._model is None and "No geometry loaded yet" in analysis._summary.toPlainText()
    window._tabs.setCurrentWidget(window._bezier_tab)
    _open(monkeypatch, js)
    window._file_open()
    assert window._tabs.currentWidget() is analysis
    reopened = analysis.get_data()
    assert reopened["geometry"] == expected["geometry"]
    assert reopened["boundaries"] == expected["boundaries"]
    assert reopened["elements"] == expected["elements"]


def test_analysis_reset_refinement_and_insert_errors(analysis, dialogs):
    _load_example(analysis, "Quarter annulus (2D)")
    analysis._split("both")
    assert analysis._model.n_control_points == 12
    analysis._reset_refinement()
    assert analysis._model.n_control_points == 6
    for text, msg in (("", "Enter the knot value"), ("x", "is not a number"),
                      ("1", "strictly inside")):
        analysis._knot_edit.setText(text)
        analysis._insert()
        assert msg in dialogs[-1][2]


def test_analysis_uses_nurbs_surface_tab(window, analysis, dialogs):
    surf = window._nurbs_surface_tab
    surf._load_cylinder()
    analysis._load_from_surface_tab()
    assert dialogs == []
    assert "NURBS Surface tab" in analysis._source_label.text()
    assert not analysis._model.planar
    assert "3D surface" in analysis.summary_text()
    assert analysis._figure.axes[0].name == "3d"
    surf._knots_u_edit.setText("garbage")
    analysis._load_from_surface_tab()
    assert dialogs[-1][1] == "Use NURBS Surface Tab"
    assert "Knots U: all knot values must be numbers" in dialogs[-1][2]
    assert analysis._model.n_control_points == 18                # previous model kept


def test_analysis_flags_inverted_elements(analysis, tmp_path, dialogs):
    from core.nurbs import quarter_annulus
    net, w, ku, kv, du, dv = quarter_annulus()
    analysis.set_geometry(net[:, ::-1], w[:, ::-1], ku, kv, du, dv, "flipped annulus")
    assert analysis._elem_table.item(0, 7).text() == "Inverted"
    assert analysis._elem_table.item(0, 7).background().color().red() == 255
    assert "inverted or degenerate" in analysis.summary_text()
    assert any(line.get_color() == "red" for line in analysis._figure.axes[0].get_lines())


def test_analysis_open_file_rejects_other_types(analysis, dialogs):
    analysis._open_file(str(Path("examples") / "circle.json"))
    assert dialogs[-1][0] == "critical" and "needs a NURBS surface" in dialogs[-1][2]
    analysis._open_file(str(Path("examples") / "plate_with_hole.json"))
    assert analysis._model is not None and "plate_with_hole.json" in analysis._source_label.text()


def test_analysis_element_limit(analysis, dialogs, monkeypatch):
    import ui.analysis_tab as at
    monkeypatch.setattr(at, "MAX_ELEMENTS", 10)
    _load_example(analysis, "Plate with a hole (2D)")
    analysis._split("both")                                     # 8 elements: fine
    analysis._split("both")                                     # 32: refused
    assert "limit is 10" in dialogs[-1][2]
    assert len(analysis._model.elements) == 8


def test_save_as_from_analysis_tab(window, analysis, monkeypatch, tmp_path, dialogs):
    _load_example(analysis, "T-shape (3D surface)")
    analysis._bc_combos["v = 0"].setCurrentText("Fixed (Dirichlet)")
    path = tmp_path / "t_model.json"
    _save(monkeypatch, path)
    window._file_save_as()
    data = io_json.load(path)
    assert data["type"] == "iga_analysis_model"
    assert data["conditions"]["v = 0"] == "Fixed (Dirichlet)"
    assert json.loads(path.read_text())["mesh"]["elements_u"] == 10


def test_every_example_loads_and_draws(analysis, dialogs):
    for i in range(analysis._example_combo.count()):
        analysis._example_combo.setCurrentIndex(i)
        analysis._load_example()
        analysis._split("u")
        assert analysis._figure.axes
        assert all(e.status == "OK" for e in analysis._model.elements)
    assert dialogs == []
