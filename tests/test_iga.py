"""IGA tests: knot insertion, the analysis model and the IGA Analysis tab."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pytest
from PyQt5.QtWidgets import QFileDialog

from core import io_json
from core.analysis import (
    CONDITIONS,
    EDGE_NAMES,
    STATUS_DEGENERATE,
    STATUS_DISTORTED,
    STATUS_INVERTED,
    STATUS_OK,
    analyze_surface,
    element_mesh_lines,
    export_json,
    export_matlab,
    is_clamped,
    is_planar,
    jacobian_field,
    surface_derivatives,
)
from core.bspline import (
    basis_function_derivatives,
    basis_functions,
    bspline_curve,
    element_boundaries,
)
from core.knots import open_uniform
from core.nurbs import (
    nurbs_circle,
    nurbs_curve,
    nurbs_cylinder,
    nurbs_surface,
    nurbs_t_shape,
    plate_with_hole,
    quarter_annulus,
    t_profile,
)
from core.refinement import insert_knot, insert_knot_surface, refine_surface_uniform, refine_uniform


# ---------------- Knot insertion and the analysis model

P2 = np.array([[5, 20], [5, 70], [50, 2], [50, 40], [70, 40], [80, 2], [90, 30], [100, 30]], float)


K2 = [0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 5]


def _surface_points(g, n=41):
    net, w, ku, kv, du, dv = g
    return np.stack(nurbs_surface(ku, kv, du, dv, net, w, n, n), axis=-1)


@pytest.mark.parametrize("knots, p", [(K2, 2), (open_uniform(7, 3), 3),
                                      ([0, 1, 2, 3, 4, 5, 6], 2), ([0, 0, 1, 1], 1)])
def test_basis_derivatives_match_finite_differences(knots, p):
    lo, hi = knots[p], knots[-p - 1]
    u = np.linspace(lo, hi, 97)[1:-1]
    u = u[np.min(np.abs(u[:, None] - np.array(knots)[None, :]), axis=1) > 1e-3]  # away from knots
    h = 1e-6
    _, dN = basis_function_derivatives(knots, p, u)
    from core.bspline import basis_functions
    fd = (basis_functions(knots, p, u + h) - basis_functions(knots, p, u - h)) / (2 * h)
    np.testing.assert_allclose(dN, fd, atol=1e-6)
    np.testing.assert_allclose(dN.sum(axis=0), 0.0, atol=1e-12)  # partition of unity -> sum dN = 0


def test_element_boundaries():
    np.testing.assert_array_equal(element_boundaries(K2, 2), [0, 1, 2, 3, 4, 5])
    np.testing.assert_array_equal(element_boundaries([0, 1, 2, 3, 4, 5, 6], 2), [2, 3, 4])


def test_insert_knot_keeps_bspline_shape():
    u = np.linspace(0, 5, 501)
    before = bspline_curve(K2, 2, P2, u_values=u)
    k, cp, w = insert_knot(K2, 2, P2, 2.5)
    assert w is None and cp.shape == (9, 2) and k == [0, 0, 0, 1, 2, 2.5, 3, 4, 4, 5, 5, 5]
    np.testing.assert_allclose(bspline_curve(k, 2, cp, u_values=u), before, atol=1e-12)


def test_insert_knot_keeps_nurbs_circle_exact():
    cp, w, knots, d = nurbs_circle()
    u = np.linspace(0, 1, 801)
    before = nurbs_curve(knots, d, cp, w, u_values=u)
    k, cp2, w2 = insert_knot(knots, d, cp, 0.125, weights=w)
    assert cp2.shape == (10, 2) and len(w2) == 10 and np.all(w2 > 0)
    after = nurbs_curve(k, d, cp2, w2, u_values=u)
    np.testing.assert_allclose(after, before, atol=1e-14)
    assert np.max(np.abs(np.hypot(after[:, 0], after[:, 1]) - 1)) < 1e-14


def test_insert_knot_3d_and_multiple_times():
    rng = np.random.default_rng(5)
    cp = rng.random((7, 3))
    knots = open_uniform(7, 3)
    u = np.linspace(0, 1, 301)
    before = bspline_curve(knots, 3, cp, u_values=u)
    k, cp2, _ = insert_knot(knots, 3, cp, 0.3, times=3)
    assert cp2.shape == (10, 3) and k.count(0.3) == 3
    np.testing.assert_allclose(bspline_curve(k, 3, cp2, u_values=u), before, atol=1e-12)
    # With multiplicity 3 = degree, the curve passes through a control point at u = 0.3
    point = bspline_curve(k, 3, cp2, u_values=[0.3])[0]
    assert np.min(np.linalg.norm(cp2 - point, axis=1)) < 1e-12


@pytest.mark.parametrize("u, times, msg", [
    (0.0, 1, "strictly inside"), (5.0, 1, "strictly inside"), (-1, 1, "strictly inside"),
    (4.0, 1, "already appears 2 time"), (2.0, 2, "allows at most 2"),
    ("abc", 1, "must be a number"), (1.5, 0, "whole number >= 1"),
])
def test_insert_knot_errors(u, times, msg):
    with pytest.raises(ValueError, match=msg):
        insert_knot(K2, 2, P2, u, times)


def test_refine_uniform_splits_every_element():
    cp, w, knots, d = nurbs_circle()
    k, cp2, w2 = refine_uniform(knots, d, cp, w)
    assert len(element_boundaries(k, d)) - 1 == 8
    u = np.linspace(0, 1, 401)
    np.testing.assert_allclose(nurbs_curve(k, d, cp2, w2, u_values=u),
                               nurbs_curve(knots, d, cp, w, u_values=u), atol=1e-14)
    k3, _, _ = refine_uniform(K2, 2, P2)
    assert len(element_boundaries(k3, 2)) - 1 == 10


@pytest.mark.parametrize("make", [nurbs_t_shape, plate_with_hole, quarter_annulus, nurbs_cylinder])
def test_surface_refinement_keeps_shape(make):
    g = make()
    net, w, ku, kv, du, dv = g
    before = _surface_points(g)
    a = insert_knot_surface(ku, kv, du, dv, net, w, "u", 0.37)
    assert a[2].shape[0] == net.shape[0] + 1
    b = insert_knot_surface(ku, kv, du, dv, net, w, "v", 0.61)
    assert b[2].shape[1] == net.shape[1] + 1
    c = refine_surface_uniform(ku, kv, du, dv, net, w, "both")
    for ku2, kv2, net2, w2 in (a, b, c):
        np.testing.assert_allclose(_surface_points((net2, w2, ku2, kv2, du, dv)), before, atol=1e-12)


def test_surface_refinement_errors():
    net, w, ku, kv, du, dv = plate_with_hole()
    with pytest.raises(ValueError, match="'u' or 'v'"):
        insert_knot_surface(ku, kv, du, dv, net, w, "x", 0.3)
    with pytest.raises(ValueError, match="strictly inside"):
        insert_knot_surface(ku, kv, du, dv, net, w, "v", 1.0)
    with pytest.raises(ValueError, match="Direction must be one of"):
        refine_surface_uniform(ku, kv, du, dv, net, w, "diagonal")


def test_quarter_annulus_model():
    m = analyze_surface(*_args(quarter_annulus()))
    assert m.planar and m.clamped_u and m.clamped_v
    assert (m.n_elements_u, m.n_elements_v, len(m.elements)) == (1, 1, 1)
    e = m.elements[0]
    assert e.control_points == [0, 1, 2, 3, 4, 5] and e.status == STATUS_OK and e.jacobian_min > 0
    assert m.dofs() == {"control_points": 6, "scalar_field": 6, "elasticity_2d": 12, "elasticity_3d": 18}
    assert m.boundaries == {"u = 0": [0, 1, 2], "u = 1": [3, 4, 5], "v = 0": [0, 3], "v = 1": [2, 5]}
    assert (m.gauss_u, m.gauss_v) == (2, 3) and m.gauss_points.shape == (6, 3)
    exact = analyze_surface(*_args(quarter_annulus()), n_gauss_u=10, n_gauss_v=10)
    assert exact.total_area == pytest.approx(3 * math.pi / 4, abs=1e-12)


def _args(g):
    net, w, ku, kv, du, dv = g
    return ku, kv, du, dv, net, w


def test_plate_with_hole_model_and_refinement():
    g = plate_with_hole()
    m = analyze_surface(*_args(g))
    assert (m.n_elements_u, m.n_elements_v) == (2, 1)
    assert [e.control_points for e in m.elements] == [[0, 1, 2, 3, 4, 5], [4, 5, 6, 7, 8, 9]]
    assert all(e.status == STATUS_OK for e in m.elements)
    area = 16 - math.pi / 4
    assert analyze_surface(*_args(g), n_gauss_u=12, n_gauss_v=4).total_area == pytest.approx(area, abs=1e-12)
    net, w, ku, kv, du, dv = g
    r = refine_surface_uniform(ku, kv, du, dv, net, w, "both")
    r = refine_surface_uniform(*r[:2], du, dv, *r[2:], "both")
    m2 = analyze_surface(r[0], r[1], du, dv, r[2], r[3])
    assert (m2.n_elements_u, m2.n_elements_v) == (8, 4) and m2.n_control_points == 11 * 5  # double knot at 0.5
    assert sum(e.area for e in m2.elements) == pytest.approx(m2.total_area)
    assert m2.total_area == pytest.approx(area, abs=1e-7)
    # Every control point is used by some element; every element has (p+1)(q+1) of them
    used = set().union(*(e.control_points for e in m2.elements))
    assert used == set(range(m2.n_control_points))
    assert all(len(e.control_points) == 6 for e in m2.elements)
    # Geometry checks: inner edge lies on the hole, outer edges on the plate
    pts = m2.net.reshape(-1, 3)
    assert np.allclose(np.hypot(*pts[m2.boundaries["v = 0"]][[0, -1], :2].T), 1.0)


def test_plate_boundary_edges_trace_the_geometry():
    g = plate_with_hole()
    lines = element_mesh_lines(*_args(g))
    _, _, ku, kv, du, dv = g
    assert len(lines) == len(element_boundaries(ku, du)) + len(element_boundaries(kv, dv))
    hole = lines[len(element_boundaries(ku, du))]          # v = 0 line
    assert np.max(np.abs(np.hypot(hole[:, 0], hole[:, 1]) - 1.0)) < 1e-12


def test_cylinder_is_3d_with_exact_area():
    m = analyze_surface(*_args(nurbs_cylinder(1.0, 2.0)), n_gauss_u=10, n_gauss_v=3)
    assert not m.planar
    assert m.total_area == pytest.approx(4 * math.pi, abs=1e-10)
    assert all(e.status == STATUS_OK for e in m.elements)


def test_quality_flags_inverted_degenerate_distorted():
    net, w, ku, kv, du, dv = quarter_annulus()
    flipped = analyze_surface(ku, kv, du, dv, net[:, ::-1], w[:, ::-1])
    assert flipped.elements[0].status == STATUS_INVERTED and flipped.elements[0].jacobian_max < 0

    collapsed = net.copy()
    collapsed[0, :, :] = 0.0                                  # u = 0 edge shrunk to a point
    assert analyze_surface(ku, kv, du, dv, collapsed, w).elements[0].status == STATUS_DEGENERATE

    quad = np.array([[[0, 0, 0], [0, 1, 0]], [[1, 0, 0], [0.53, 0.53, 0]]], float)
    m = analyze_surface([0, 0, 1, 1], [0, 0, 1, 1], 1, 1, quad, np.ones((2, 2)))
    assert m.elements[0].status == STATUS_DISTORTED


def test_surface_derivatives_match_finite_differences():
    net, w, ku, kv, du, dv = plate_with_hole()
    u, v, h = np.array([0.2, 0.7]), np.array([0.3, 0.8]), 1e-6
    _, Su, Sv = surface_derivatives(ku, kv, du, dv, net, w, u, v)
    fu = (np.stack(nurbs_surface(ku, kv, du, dv, net, w, u_values=u + h, v_values=v), -1)
          - np.stack(nurbs_surface(ku, kv, du, dv, net, w, u_values=u - h, v_values=v), -1)) / (2 * h)
    fv = (np.stack(nurbs_surface(ku, kv, du, dv, net, w, u_values=u, v_values=v + h), -1)
          - np.stack(nurbs_surface(ku, kv, du, dv, net, w, u_values=u, v_values=v - h), -1)) / (2 * h)
    np.testing.assert_allclose(Su, fu, atol=1e-6)
    np.testing.assert_allclose(Sv, fv, atol=1e-6)


def test_jacobian_field_and_helpers():
    m = analyze_surface(*_args(plate_with_hole()))
    X, _, _, J = jacobian_field(m, per_element=5)
    assert X.shape == J.shape == (10, 5) and np.all(J > 0)
    assert is_planar(m.net) and not is_planar(nurbs_cylinder()[0])
    assert is_clamped([0, 0, 0, 1, 1, 1], 2) and not is_clamped([0, 1, 2, 3, 4, 5], 2)


def test_conditions_are_validated():
    with pytest.raises(ValueError, match="Unknown boundary condition"):
        analyze_surface(*_args(quarter_annulus()), conditions={"u = 0": "Glued"})
    with pytest.raises(ValueError, match="Unknown boundary edge"):
        analyze_surface(*_args(quarter_annulus()), conditions={"top": "Free"})


def test_export_json_reloads_with_conditions(tmp_path: Path):
    m = analyze_surface(*_args(plate_with_hole()),
                        conditions={"u = 1": "Fixed (Dirichlet)", "v = 1": "Load (Neumann)"})
    path = tmp_path / "model.json"
    export_json(m, path)
    data = io_json.load(path)
    assert data["type"] == "iga_analysis_model"
    assert data["conditions"] == {"u = 1": "Fixed (Dirichlet)", "v = 1": "Load (Neumann)",
                                  "u = 0": "Free", "v = 0": "Free"}
    np.testing.assert_array_equal(data["geometry"]["control_points"], m.net)
    assert len(data["elements"]) == 2 and data["dofs"]["elasticity_2d"] == 20
    assert data["elements"][1]["control_points"] == m.elements[1].control_points


@pytest.mark.parametrize("change, msg", [
    (lambda d: d.pop("geometry"), "missing required field"),
    (lambda d: d["geometry"].pop("knots_u"), "geometry is missing required field"),
    (lambda d: d["boundaries"]["u = 0"].update(condition="Glued"), "unknown boundary condition"),
    (lambda d: d["boundaries"].update({"top": {"condition": "Free"}}), "unknown boundary edge"),
])
def test_broken_model_files_are_rejected(tmp_path: Path, change, msg):
    import json
    m = analyze_surface(*_args(quarter_annulus()))
    d = m.to_dict()
    change(d)
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(d))
    with pytest.raises(ValueError, match=msg):
        io_json.load(path)


def _matlab_numbers(text: str, name: str):
    m = re.search(rf"^{re.escape(name)} = \[(.*?)\];$", text, re.M)
    return m.group(1)


def test_export_matlab_layout(tmp_path: Path):
    m = analyze_surface(*_args(plate_with_hole()), conditions={"u = 1": "Fixed (Dirichlet)"})
    path = tmp_path / "model.m"
    export_matlab(m, path)
    text = path.read_text()
    # coefs: 4 x nu x nv, column-major, weighted coordinates
    mm = re.search(r"geometry\.coefs = reshape\(\[(.*?)\], \[4 (\d+) (\d+)\]\);", text)
    flat = np.array([float(x) for x in mm.group(1).split()])
    coefs = flat.reshape((4, int(mm.group(2)), int(mm.group(3))), order="F")
    np.testing.assert_allclose(coefs[3], m.weights)
    np.testing.assert_allclose(coefs[:3] / coefs[3], np.moveaxis(m.net, 2, 0))
    assert "geometry.knots = {[0 0 0 0.5 0.5 1 1 1], [0 0 1 1]};" in text
    assert "geometry.order = [3 2];" in text
    conn = [list(map(int, row.split())) for row in _matlab_numbers(text, "elements.connectivity").split(";")]
    assert conn == [[c + 1 for c in e.control_points] for e in m.elements]
    assert "boundaries(2).name = 'u = 1';" in text
    assert "boundaries(2).condition = 'Fixed (Dirichlet)';" in text
    assert f"boundaries(1).control_points = [{' '.join(str(c + 1) for c in m.boundaries['u = 0'])}];" in text
    assert "elements.status = {'OK'; 'OK'};" in text
    assert "dofs.elasticity_2d = 20;" in text
    assert text.count("(") == text.count(")") and text.count("[") == text.count("]")
    assert text.count("{") == text.count("}")


def test_conditions_list_is_stable():
    assert EDGE_NAMES == ("u = 0", "u = 1", "v = 0", "v = 1")
    assert CONDITIONS[0] == "Free"


def test_fold_between_gauss_points_is_inverted():
    """Element folded near the middle of one edge must be 'Inverted' (bug found by MATLAB)."""
    xs = np.linspace(0.0, 1.0, 4)
    net = np.array([[[x, y, 0.0] for y in xs] for x in xs])
    net[1, 3, 1] = net[2, 3, 1] = 0.5
    k3 = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    m = analyze_surface(k3, k3, 3, 3, net, np.ones((4, 4)))
    e = m.elements[0]
    assert e.status == STATUS_INVERTED
    assert e.jacobian_min == pytest.approx(-0.125, abs=1e-8)   # minimum at (0.5, 1), sampled 1e-9 inside


# ---------------- Knot tools and the IGA Analysis tab

def _labels(fig):
    return [line.get_label() for line in fig.axes[0].get_lines()]


def _knot_marker_count(fig):
    lines = [l for l in fig.axes[0].get_lines() if l.get_label() == "Knots (element ends)"]
    return len(lines[0].get_xdata()) if lines else 0


def _open(monkeypatch, path):
    monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(lambda *a: (str(path), "")))


def _save(monkeypatch, path):
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a: (str(path), "")))


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


def test_surface_knots_toggle_draws_element_lines(window, dialogs):
    tab = window._nurbs_surface_tab
    tab._load_t_shape()
    tab._draw()
    assert "Element boundaries" not in _labels(tab._figure)
    assert "10 element(s) in u x 1 in v = 10" in window._status_bar.currentMessage()
    tab._show_knots_chk.setChecked(True)
    assert "Element boundaries" in _labels(tab._figure)
    assert dialogs == []


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
