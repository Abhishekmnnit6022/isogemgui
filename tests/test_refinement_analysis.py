"""Core tests for knot insertion, element/Jacobian analysis and model export."""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pytest

from core import io_json
from core.analysis import (
    CONDITIONS, EDGE_NAMES, STATUS_DEGENERATE, STATUS_DISTORTED, STATUS_INVERTED, STATUS_OK,
    analyze_surface, element_mesh_lines, export_json, export_matlab, is_clamped, is_planar,
    jacobian_field, surface_derivatives,
)
from core.bspline import basis_function_derivatives, bspline_curve, element_boundaries
from core.knots import open_uniform
from core.nurbs import (
    nurbs_circle, nurbs_curve, nurbs_cylinder, nurbs_surface, nurbs_t_shape, plate_with_hole,
    quarter_annulus,
)
from core.refinement import (
    insert_knot, insert_knot_surface, refine_surface_uniform, refine_uniform,
)

P2 = np.array([[5, 20], [5, 70], [50, 2], [50, 40], [70, 40], [80, 2], [90, 30], [100, 30]], float)
K2 = [0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 5]


def _surface_points(g, n=41):
    net, w, ku, kv, du, dv = g
    return np.stack(nurbs_surface(ku, kv, du, dv, net, w, n, n), axis=-1)


# ---------------------------------------------------------------- basis derivatives

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


# ---------------------------------------------------------------- curve knot insertion

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


# ---------------------------------------------------------------- surface knot insertion

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


# ---------------------------------------------------------------- analysis

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


# ---------------------------------------------------------------- export

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
