"""Regression tests for bugs found in the project review.

Covers: unclamped (uniform) knot vectors, the vectorised basis evaluation, silent
display-option redraws, clear table-cell errors, and loading files whose values
do not fit the UI spin boxes.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pytest
from PyQt5.QtWidgets import QTableWidgetItem

from core.bspline import _basis_scalar, basis_functions, bspline_curve, valid_domain
from core.knots import open_uniform, uniform
from core.nurbs import nurbs_curve, nurbs_surface, rational_basis_matrix, sample_parameters

KNOT_CASES = [
    ([0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 5], 2),       # thesis example, double interior knot
    ([0, 1, 2, 3, 4, 5, 6, 7], 2),                # uniform, unclamped
    ([0, 0, 0, 0, 0.3, 0.3, 0.7, 1, 1, 1, 1], 3),  # clamped cubic with repeated knot
    ([0, 0.5, 1, 2, 2, 2, 3, 3.5], 1),            # irregular, unclamped
    ([0, 0, 1, 1], 0),                            # degree 0
    (open_uniform(9, 5), 5),
]


# ---------------------------------------------------------------- vectorised basis

@pytest.mark.parametrize("knots, p", KNOT_CASES)
def test_vectorised_basis_equals_recursive_reference(knots, p):
    u = np.unique(np.concatenate([np.linspace(knots[0], knots[-1], 157), knots]))
    fast = basis_functions(knots, p, u)
    ref = np.array([[_basis_scalar(i, p, x, [float(k) for k in knots]) for x in u]
                    for i in range(len(knots) - p - 1)])
    np.testing.assert_allclose(fast, ref, atol=1e-14)


def test_basis_evaluation_is_fast():
    knots = open_uniform(20, 10)
    cp = np.random.default_rng(0).random((20, 2))
    t = time.perf_counter()
    bspline_curve(knots, 10, cp, n_eval=300)
    assert time.perf_counter() - t < 0.5  # was about 3-4 s with the recursive version


# ---------------------------------------------------------------- unclamped knots

def test_valid_domain():
    assert valid_domain([0, 0, 0, 1, 2, 2, 2], 2) == (0.0, 2.0)
    assert valid_domain(uniform(5, 2), 2) == (2.0, 5.0)
    with pytest.raises(ValueError, match="too short"):
        valid_domain([0, 1, 2], 2)
    with pytest.raises(ValueError, match="empty valid range"):
        valid_domain([0, 0, 0, 0, 0, 1, 1, 1], 3)  # knots[3] == knots[4] == 0


def test_uniform_knot_curve_no_longer_drops_to_origin():
    P = np.array([[0, 0], [1, 2], [3, 3], [4, 0], [6, 1]], float)
    c = bspline_curve(uniform(5, 2), 2, P, n_eval=101)
    # A uniform quadratic B-spline starts/ends at the midpoints of the first/last legs
    np.testing.assert_allclose(c[0], (P[0] + P[1]) / 2, atol=1e-14)
    np.testing.assert_allclose(c[-1], (P[3] + P[4]) / 2, atol=1e-14)
    assert np.min(np.linalg.norm(c, axis=1)) > 0.5  # never collapses to (0, 0)


def test_uniform_knot_nurbs_matches_bspline_and_basis_sums_to_one():
    P = np.array([[0, 0], [1, 2], [3, 3], [4, 0], [6, 1]], float)
    knots = uniform(5, 2)
    np.testing.assert_allclose(nurbs_curve(knots, 2, P, np.ones(5), n_eval=60),
                               bspline_curve(knots, 2, P, n_eval=60), atol=1e-14)
    u, R = rational_basis_matrix(knots, 2, [1, 2, 0.5, 1, 3], n_eval=80)
    assert u[0] == 2.0 and u[-1] == 5.0
    np.testing.assert_allclose(R.sum(axis=0), 1.0, atol=1e-14)


def test_uniform_knot_surface_corners_are_not_zero():
    rng = np.random.default_rng(3)
    net = rng.random((5, 4, 3)) + 1.0
    X, Y, Z = nurbs_surface(uniform(5, 2), uniform(4, 1), 2, 1, net, np.ones((5, 4)), 6, 6)
    corner = np.array([X[0, 0], Y[0, 0], Z[0, 0]])
    # u = 2 on a uniform quadratic -> mean of rows 0 and 1; v = 1 on uniform linear -> column 0
    np.testing.assert_allclose(corner, (net[0, 0] + net[1, 0]) / 2, atol=1e-14)


def test_sample_parameters_respects_valid_domain():
    u = sample_parameters(uniform(6, 3), 50, degree=3)
    assert u[0] == 3.0 and u[-1] == 6.0
    assert {3.0, 4.0, 5.0, 6.0} <= set(u.tolist())


# ---------------------------------------------------------------- GUI fixtures

@pytest.fixture
def window(qapp):
    from app import MainWindow
    return MainWindow()


def _title(tab):
    return tab._figure.axes[0].get_title() if tab._figure.axes else ""


# ---------------------------------------------------------------- display toggles

def test_bezier_toggles_are_silent_and_need_a_plot(window, dialogs):
    tab = window._bezier_tab
    tab.set_control_points(np.array([[0, 0], [1, 2], [2, 0]]))
    tab._show_poly_chk.setChecked(False)            # nothing drawn yet -> no redraw
    assert tab._figure.axes == []
    tab._draw()
    tab._table.setItem(1, 0, QTableWidgetItem("abc"))
    tab._show_points_chk.setChecked(False)          # bad input -> keep old plot, no pop-up
    assert dialogs == []
    tab._draw()                                     # explicit Draw still reports it
    assert dialogs[-1][2] == "Row 2: X value 'abc' is not a number."


def test_bspline_toggles_keep_basis_plot(window, dialogs):
    tab = window._bspline_tab
    tab._plot_basis()
    tab._show_poly_chk.setChecked(False)
    tab._curve_color_combo.setCurrentIndex(2)
    assert _title(tab).startswith("B-Spline Basis Functions")
    tab._draw_curve()
    tab._show_points_chk.setChecked(False)
    labels = [l.get_label() for l in tab._figure.axes[0].get_lines()]
    assert "Control points" not in labels and "B-spline curve" in labels
    tab._knots_edit.setText("0 0 1")
    tab._show_points_chk.setChecked(True)
    assert dialogs == []


def test_bspline_cell_errors_are_clear(window, dialogs):
    tab = window._bspline_tab
    tab._table.setItem(2, 1, QTableWidgetItem(""))
    tab._draw_curve()
    assert dialogs[-1][2] == "Row 3: Y is empty."
    tab._load_matlab_preset()
    tab._table.setItem(0, 2, QTableWidgetItem("z?"))
    tab._draw_curve()
    assert dialogs[-1][2] == "Row 1: Z value 'z?' is not a number."


def test_bspline_uniform_knots_in_gui(window, dialogs):
    tab = window._bspline_tab
    tab._generate_uniform_knots()
    tab._draw_curve()
    assert dialogs == []
    curve = [l for l in tab._figure.axes[0].get_lines() if l.get_label() == "B-spline curve"][0]
    xy = np.column_stack(curve.get_data())
    assert np.min(np.linalg.norm(xy, axis=1)) > 1.0  # Matlab preset points are far from (0, 0)
    tab._plot_basis()
    assert any(p.get_label() == "Valid curve range" for p in tab._figure.axes[0].patches)


def test_bezier_surface_toggles_are_silent(window, dialogs):
    tab = window._surface_tab
    tab._draw()
    tab._table.setItem(0, 2, QTableWidgetItem("x"))
    tab._show_points_chk.setChecked(False)
    tab._show_lines_chk.setChecked(False)
    assert dialogs == []


# ---------------------------------------------------------------- loading out-of-range files

def _open(window, tmp_path, monkeypatch, payload):
    from PyQt5.QtWidgets import QFileDialog
    path = tmp_path / "geom.json"
    path.write_text(json.dumps(payload))
    monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(lambda *a: (str(path), "")))
    window._file_open()


def test_bezier_surface_up_to_50x50_loads(window, tmp_path, monkeypatch, dialogs):
    net = np.random.default_rng(1).random((25, 3, 3))
    _open(window, tmp_path, monkeypatch, {"type": "bezier_surface", "control_points": net.tolist()})
    assert dialogs == []
    np.testing.assert_array_equal(window._surface_tab.get_control_net(), net)


@pytest.mark.parametrize("payload, message", [
    ({"type": "bspline_curve", "degree": 16, "knots": open_uniform(20, 16),
      "control_points": np.ones((20, 2)).tolist()},
     "Degree 16 is outside the supported range 1 to 15."),
    ({"type": "bspline_curve", "degree": 0, "knots": [0, 1, 2, 3],
      "control_points": [[0, 0], [1, 1], [2, 0]]},
     "Degree 0 is outside the supported range 1 to 15."),
    ({"type": "nurbs_curve", "degree": 0, "knots": [0, 1, 2, 3],
      "control_points": [[0, 0], [1, 1], [2, 0]], "weights": [1, 1, 1]},
     "Degree 0 is outside the supported range 1 to 15."),
    ({"type": "nurbs_surface", "degree_u": 1, "degree_v": 0, "knots_u": [0, 0, 1, 1],
      "knots_v": [0, 1, 2], "control_points": np.zeros((2, 2, 3)).tolist(),
      "weights": [[1, 1], [1, 1]]},
     "Degree V 0 is outside the supported range 1 to 15."),
    ({"type": "bezier_surface", "control_points": np.zeros((60, 2, 3)).tolist()},
     "Number of control net rows 60 is outside the supported range 2 to 50."),
])
def test_out_of_range_files_are_rejected_clearly(window, tmp_path, monkeypatch, dialogs,
                                                 payload, message):
    before = {name: getattr(window, name).get_data() for name in
              ("_bspline_tab", "_surface_tab")}
    _open(window, tmp_path, monkeypatch, payload)
    assert dialogs[-1][0] == "critical"
    assert message in dialogs[-1][2]
    # The tabs that could be affected keep their previous content
    for name, data in before.items():
        assert getattr(window, name).get_data() == data
