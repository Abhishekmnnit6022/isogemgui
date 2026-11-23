"""Headless GUI tests for the NURBS Curve and NURBS Surface tabs and the main window."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
from PyQt5.QtWidgets import QTableWidgetItem

from core import io_json
from core.nurbs import nurbs_cylinder, nurbs_t_shape, t_profile
from ui.common import format_knots, format_number, parse_knots_text


# ---------------------------------------------------------------- helpers

class StatusLog:
    """Collects status-bar messages sent by a tab."""

    def __init__(self):
        self.messages = []

    def __call__(self, msg):
        self.messages.append(msg)

    @property
    def last(self):
        return self.messages[-1] if self.messages else ""


def _labels(figure):
    ax = figure.axes[0]
    return [line.get_label() for line in ax.get_lines()]


@pytest.fixture
def curve_tab(qapp):
    from ui.nurbs_curve_tab import NurbsCurveTab
    log = StatusLog()
    tab = NurbsCurveTab(status_callback=log)
    tab.log = log
    return tab


@pytest.fixture
def surface_tab(qapp):
    from ui.nurbs_surface_tab import NurbsSurfaceTab
    log = StatusLog()
    tab = NurbsSurfaceTab(status_callback=log)
    tab.log = log
    return tab


# ---------------------------------------------------------------- knot text helpers

def test_format_knots_is_lossless():
    knots = sorted([0, 0, 1 / 3, 0.1, 2 / 7, 1e-17, 12345.678901234567, 1])
    assert parse_knots_text(format_knots(knots)) == [float(k) for k in knots]
    assert format_knots([0.0, 0.25, 1.0, 2.0]) == "0, 0.25, 1, 2"
    assert format_number(-0.0) == "0"


def test_parse_knots_text_errors():
    with pytest.raises(ValueError, match="Knots U is empty"):
        parse_knots_text("  ", "Knots U")
    with pytest.raises(ValueError, match="must be numbers"):
        parse_knots_text("0, a, 1")
    with pytest.raises(ValueError, match="non-decreasing"):
        parse_knots_text("0 1 0.5")


# ---------------------------------------------------------------- NURBS curve tab

def test_circle_draw_reports_radius_error(curve_tab, dialogs):
    curve_tab._load_circle()
    curve_tab._draw()
    assert dialogs == []
    m = re.search(r"Circle: centre \((.+?), (.+?)\), radius (\S+), max radius error = (\S+)",
                  curve_tab.log.last)
    assert m, curve_tab.log.last
    assert abs(float(m.group(1))) < 1e-9 and abs(float(m.group(2))) < 1e-9
    assert float(m.group(3)) == pytest.approx(1.0)
    assert float(m.group(4)) < 1e-12


def test_non_circle_has_no_circle_message(curve_tab, dialogs):
    curve_tab._load_t_profile()
    curve_tab._draw()
    assert dialogs == []
    assert "Circle" not in curve_tab.log.last
    assert "degree 2, 21 points" in curve_tab.log.last


def test_bspline_overlay_in_2d_and_3d(curve_tab, dialogs):
    curve_tab._load_circle()
    curve_tab._show_bspline_chk.setChecked(True)
    curve_tab._draw()
    assert "B-spline (w=1)" in _labels(curve_tab._figure)

    # Give every point a Z value -> 3D plot must still show the overlay
    for r in range(curve_tab._table.rowCount()):
        curve_tab._table.setItem(r, 2, QTableWidgetItem(str(0.1 * r)))
    curve_tab._draw()
    ax = curve_tab._figure.axes[0]
    assert ax.name == "3d"
    assert "B-spline (w=1)" in _labels(curve_tab._figure)
    assert dialogs == []

    curve_tab._show_bspline_chk.setChecked(False)
    curve_tab._draw()
    assert "B-spline (w=1)" not in _labels(curve_tab._figure)


def test_rational_basis_plot(curve_tab, dialogs):
    curve_tab._load_circle()
    curve_tab._plot_rational_basis()
    assert dialogs == []
    lines = curve_tab._figure.axes[0].get_lines()
    assert len(lines) == 9
    total = sum(line.get_ydata() for line in lines)
    np.testing.assert_allclose(total, 1.0, atol=1e-12)
    assert "Rational basis plotted: 9 functions" in curve_tab.log.last

    curve_tab._show_bspline_chk.setChecked(True)
    curve_tab._plot_rational_basis()
    assert len(curve_tab._figure.axes[0].get_lines()) == 18


def test_rational_basis_excel_export(curve_tab, dialogs, tmp_path):
    pd = pytest.importorskip("pandas")
    curve_tab._load_circle()
    out = tmp_path / "rational.xlsx"
    curve_tab.export_rational_basis_excel(str(out))
    assert dialogs and dialogs[-1][0] == "information"
    df = pd.read_excel(out)
    assert list(df.columns) == ["u"] + [f"R_{i}_2" for i in range(9)]
    np.testing.assert_allclose(df.iloc[:, 1:].sum(axis=1), 1.0, atol=1e-12)


def test_curve_get_set_data_keeps_full_precision(curve_tab, dialogs):
    data = {"type": "nurbs_curve", "degree": 2,
            "knots": [0, 0, 0, 1 / 3, 2 / 3, 1, 1, 1],
            "control_points": [[0, 0], [1 / 3, 2], [2, 3], [3, 1 / 7], [4, 0]],
            "weights": [1, 0.7071067811865476, 2 / 3, 1, 1]}
    curve_tab.set_data(data)
    out = curve_tab.get_data()
    assert out["knots"] == [float(k) for k in data["knots"]]
    np.testing.assert_array_equal(out["control_points"], data["control_points"])
    np.testing.assert_array_equal(out["weights"], data["weights"])


def test_curve_invalid_inputs_show_clear_messages(curve_tab, dialogs):
    curve_tab._load_circle()
    curve_tab._table.setItem(3, 3, QTableWidgetItem("-2"))
    curve_tab._draw()
    assert "Weight at row 4 must be strictly positive" in dialogs[-1][2]

    curve_tab._load_circle()
    curve_tab._knots_edit.setText("0, 0, 1, 1")
    curve_tab._plot_rational_basis()
    assert "B-spline constraint violated" in dialogs[-1][2]


def test_curve_reset(curve_tab):
    curve_tab._load_t_profile()
    curve_tab._draw()
    curve_tab.reset()
    assert curve_tab._table.rowCount() == 0
    assert curve_tab._knots_edit.text() == ""
    assert curve_tab._figure.axes == []


# ---------------------------------------------------------------- NURBS surface tab

def test_surface_reset_gives_drawable_default(surface_tab, dialogs):
    surface_tab._load_t_shape()
    surface_tab.reset()
    assert (surface_tab._rows_spin.value(), surface_tab._cols_spin.value()) == (3, 2)
    assert surface_tab._coord_table.rowCount() == 6
    assert surface_tab._wt_table.rowCount() == 6
    weights = [surface_tab._wt_table.item(r, 2).text() for r in range(6)]
    assert weights == ["1.0"] * 6
    assert surface_tab._figure.axes == []
    surface_tab._draw()
    assert dialogs == []
    assert "3x2 control net" in surface_tab.log.last


def test_surface_tables_follow_spin_boxes(surface_tab):
    surface_tab._coord_table.setItem(0, 4, QTableWidgetItem("7.5"))
    surface_tab._rows_spin.setValue(5)
    assert surface_tab._coord_table.rowCount() == 10
    assert surface_tab._wt_table.rowCount() == 10
    assert surface_tab._coord_table.item(0, 4).text() == "7.5"  # existing value kept
    surface_tab._cols_spin.setValue(4)
    assert surface_tab._coord_table.rowCount() == 20
    surface_tab._generate_open_knots()
    surface_tab._draw()
    assert surface_tab._figure.axes


def test_cylinder_draws_exact_round_surface(surface_tab, dialogs):
    surface_tab._load_cylinder()
    assert (surface_tab._rows_spin.value(), surface_tab._cols_spin.value()) == (9, 2)
    surface_tab._draw()
    assert dialogs == []
    ax = surface_tab._figure.axes[0]
    assert any(c.__class__.__name__ == "Poly3DCollection" for c in ax.collections)
    # The surface built from the tab's own tables is an exact unit cylinder of height 1
    from core.nurbs import nurbs_surface, sample_parameters
    net, w, ku, kv, du, dv = surface_tab._read_inputs()
    X, Y, Z = nurbs_surface(ku, kv, du, dv, net, w,
                            u_values=sample_parameters(ku, 200), v_values=[0.0, 0.5, 1.0])
    assert np.max(np.abs(np.hypot(X, Y) - 1.0)) < 1e-12
    assert Z.min() == pytest.approx(0.0) and Z.max() == pytest.approx(1.0)
    # Equal axes: x and y extents are equal, so the cylinder looks round
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    assert (x1 - x0) == pytest.approx(y1 - y0)
    assert "degree 2x1, 9x2" in surface_tab.log.last


def test_cylinder_save_open_roundtrip_is_exact(surface_tab, tmp_path):
    surface_tab._load_cylinder()
    path = tmp_path / "cyl.json"
    io_json.save(path, surface_tab.get_data())
    surface_tab.reset()
    surface_tab.set_data(io_json.load(path))
    out = surface_tab.get_data()
    net, w, ku, kv, du, dv = nurbs_cylinder()
    np.testing.assert_array_equal(out["control_points"], net)
    np.testing.assert_array_equal(out["weights"], w)
    assert out["knots_u"] == ku and out["knots_v"] == kv


def test_t_shape_loads_draws_and_roundtrips(surface_tab, dialogs, tmp_path):
    surface_tab._load_t_shape()
    assert (surface_tab._rows_spin.value(), surface_tab._cols_spin.value()) == (21, 2)
    surface_tab._draw()
    assert dialogs == []
    assert "degree 2x1, 21x2" in surface_tab.log.last
    data = surface_tab.get_data()
    net, w, ku, kv, du, dv = nurbs_t_shape()
    np.testing.assert_array_equal(data["control_points"], net)
    np.testing.assert_array_equal(data["weights"], w)
    assert data["knots_u"] == ku

    path = tmp_path / "t.json"
    io_json.save(path, data)
    surface_tab.reset()
    surface_tab.set_data(io_json.load(path))
    assert surface_tab.get_data() == data


def test_t_shape_example_file_opens_in_tab(surface_tab, dialogs):
    surface_tab.set_data(io_json.load(Path("examples") / "t_shape.json"))
    surface_tab._draw()
    assert dialogs == []
    assert surface_tab._rows_spin.value() == 21


def test_display_toggles_redraw_silently(surface_tab, dialogs):
    surface_tab._load_cylinder()
    surface_tab._draw()
    surface_tab._show_surface_chk.setChecked(False)
    assert surface_tab._figure.axes[0].collections == [] or all(
        c.__class__.__name__ != "Poly3DCollection" for c in surface_tab._figure.axes[0].collections)
    surface_tab._show_lines_chk.setChecked(False)
    assert "Control net lines" not in _labels(surface_tab._figure)
    surface_tab._show_points_chk.setChecked(False)
    assert "Control points" not in _labels(surface_tab._figure)
    # Broken input + toggle must not pop up a dialog
    surface_tab._knots_u_edit.setText("garbage")
    surface_tab._show_points_chk.setChecked(True)
    surface_tab._line_color_combo.setCurrentIndex(2)
    assert dialogs == []
    # But an explicit Draw does report the problem
    surface_tab._draw()
    assert "Knots U: all knot values must be numbers" in dialogs[-1][2]


def test_surface_invalid_weight_and_direction_messages(surface_tab, dialogs):
    surface_tab._load_cylinder()
    surface_tab._wt_table.setItem(4, 2, QTableWidgetItem("0"))
    surface_tab._draw()
    assert "must be strictly positive" in dialogs[-1][2]
    surface_tab._reset_weights()
    surface_tab._knots_v_edit.setText("0, 0, 0, 1, 1, 1")
    surface_tab._draw()
    assert dialogs[-1][2].startswith("V direction (cols)")


# ---------------------------------------------------------------- main window

@pytest.fixture
def window(qapp):
    from app import MainWindow
    return MainWindow()


def test_file_new_resets_nurbs_tabs(window):
    window._tabs.setCurrentWidget(window._nurbs_surface_tab)
    window._nurbs_surface_tab._load_t_shape()
    window._file_new()
    assert window._nurbs_surface_tab._rows_spin.value() == 3
    assert window._nurbs_surface_tab._coord_table.rowCount() == 6

    window._tabs.setCurrentWidget(window._nurbs_curve_tab)
    window._nurbs_curve_tab._load_circle()
    window._file_new()
    assert window._nurbs_curve_tab._table.rowCount() == 0


def test_excel_menu_is_context_aware(window, monkeypatch):
    calls = []
    monkeypatch.setattr(window._nurbs_curve_tab, "export_rational_basis_excel",
                        lambda *a: calls.append("nurbs"))
    monkeypatch.setattr(window._bspline_tab, "export_basis_excel",
                        lambda *a: calls.append("bspline"))
    window._tabs.setCurrentWidget(window._nurbs_curve_tab)
    window._export_excel()
    window._tabs.setCurrentWidget(window._bezier_tab)
    window._export_excel()
    assert calls == ["nurbs", "bspline"]
    assert window._tabs.currentWidget() is window._bspline_tab


def test_every_example_opens_in_its_tab(window, dialogs, monkeypatch):
    from PyQt5.QtWidgets import QFileDialog
    tab_for = {
        "bezier_curve": window._bezier_tab, "bezier_surface": window._surface_tab,
        "bspline_curve": window._bspline_tab, "nurbs_curve": window._nurbs_curve_tab,
        "nurbs_surface": window._nurbs_surface_tab,
    }
    for path in sorted(Path("examples").glob("*.json")):
        monkeypatch.setattr(QFileDialog, "getOpenFileName",
                            staticmethod(lambda *a, p=str(path): (p, "")))
        window._file_open()
        geom = io_json.load(path)["type"]
        assert window._tabs.currentWidget() is tab_for[geom], path.name
        draw = getattr(window._tabs.currentWidget(), "_draw", None) or window._bspline_tab._draw_curve
        draw()
    assert [d for d in dialogs if d[0] != "information"] == []
