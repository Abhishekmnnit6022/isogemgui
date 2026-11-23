"""Tests for shape generators (circle, cylinder, T-shape), extrusion, sampling and circle fitting."""

from __future__ import annotations

import math

import numpy as np
import pytest

from core.bspline import _basis_scalar
from core.nurbs import (
    extrude_curve, fit_circle, nurbs_circle, nurbs_curve, nurbs_cylinder,
    nurbs_surface, nurbs_t_shape, sample_parameters, t_profile,
)

# Default T dimensions used by t_profile()
W, TF, WS, HS, RF = 4.0, 1.0, 1.0, 3.0, 0.4


def _distance_to_t_outline(p, W=W, tf=TF, ws=WS, hs=HS, rf=RF):
    """Distance from a 2D point to the exact T outline (straight edges + two fillet arcs)."""
    a, b, top = ws / 2, W / 2, hs + tf

    def seg(p, s, e):
        s, e = np.asarray(s, float), np.asarray(e, float)
        d = e - s
        t = np.clip(np.dot(p - s, d) / np.dot(d, d), 0.0, 1.0)
        return np.linalg.norm(p - (s + t * d))

    edges = [((-a, 0), (a, 0)), ((a, 0), (a, hs - rf)), ((a + rf, hs), (b, hs)),
             ((b, hs), (b, top)), ((b, top), (-b, top)), ((-b, top), (-b, hs)),
             ((-b, hs), (-a - rf, hs)), ((-a, hs - rf), (-a, 0))]
    best = min(seg(p, s, e) for s, e in edges)
    # Fillet arcs: quarter circles centred at (+-(a + rf), hs - rf), on the side facing the corner
    for sx in (1.0, -1.0):
        c = np.array([sx * (a + rf), hs - rf])
        v = p - c
        if sx * v[0] <= 1e-12 and v[1] >= -1e-12:
            best = min(best, abs(np.linalg.norm(v) - rf))
    return best


def test_t_profile_structure():
    cp, w, knots, degree = t_profile()
    assert degree == 2
    assert cp.shape == (21, 2)
    assert len(knots) == cp.shape[0] + degree + 1
    np.testing.assert_allclose(cp[0], cp[-1])  # closed outline
    # Exactly two fillet arcs carry the sqrt(2)/2 weight
    assert np.sum(np.isclose(w, math.sqrt(2) / 2)) == 2
    assert np.all(w > 0)


def test_t_profile_points_lie_exactly_on_outline():
    cp, w, knots, degree = t_profile()
    curve = nurbs_curve(knots, degree, cp, w, u_values=sample_parameters(knots, 2000))
    errors = [_distance_to_t_outline(p) for p in curve]
    assert max(errors) < 1e-12


def test_t_profile_hits_every_corner_at_its_knot():
    cp, w, knots, degree = t_profile()
    breaks = sorted(set(knots))
    pts = nurbs_curve(knots, degree, cp, w, u_values=breaks)
    # Segment end points are the even-indexed control points
    np.testing.assert_allclose(pts, cp[0::2], atol=1e-14)


def test_t_profile_area_matches_formula():
    cp, w, knots, degree = t_profile()
    c = nurbs_curve(knots, degree, cp, w, u_values=sample_parameters(knots, 20000))
    x, y = c[:, 0], c[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    # Flange + stem + the two concave fillet fillings (square minus quarter disc each)
    expected = W * TF + WS * HS + 2 * (RF ** 2 - math.pi * RF ** 2 / 4)
    assert area == pytest.approx(expected, abs=1e-6)


def test_t_profile_is_symmetric_about_y_axis():
    cp, w, knots, degree = t_profile()
    c = nurbs_curve(knots, degree, cp, w, n_eval=801)
    mirrored = c * np.array([-1.0, 1.0])
    # Every mirrored point must also be on the outline
    assert max(_distance_to_t_outline(p) for p in mirrored) < 1e-12


def test_t_profile_without_fillets_has_sharp_corners():
    cp, w, knots, degree = t_profile(fillet_radius=0.0)
    assert cp.shape == (17, 2)
    assert np.allclose(w, 1.0)
    pts = nurbs_curve(knots, degree, cp, w, u_values=sorted(set(knots)))
    corners = {(round(x, 12), round(y, 12)) for x, y in pts}
    assert (0.5, 3.0) in corners and (-0.5, 3.0) in corners


@pytest.mark.parametrize("kwargs, msg", [
    ({"flange_width": 0.0}, "flange_width must be positive"),
    ({"stem_height": -1.0}, "stem_height must be positive"),
    ({"stem_width": 5.0}, "must be smaller than flange_width"),
    ({"fillet_radius": -0.1}, "fillet_radius must be >= 0"),
    ({"fillet_radius": 3.0}, "must be smaller than stem_height"),
    ({"fillet_radius": 1.5}, "flange overhang"),
])
def test_t_profile_invalid_parameters(kwargs, msg):
    with pytest.raises(ValueError, match=msg):
        t_profile(**kwargs)


def test_t_shape_surface_geometry():
    net, w, ku, kv, du, dv = nurbs_t_shape(depth=2.5)
    assert net.shape == (21, 2, 3) and w.shape == (21, 2)
    assert (du, dv) == (2, 1) and kv == [0.0, 0.0, 1.0, 1.0]
    u = sample_parameters(ku, 300)
    v = np.linspace(0.0, 1.0, 7)
    X, Y, Z = nurbs_surface(ku, kv, du, dv, net, w, u_values=u, v_values=v)
    # Every surface point sits on the T outline in (x, y), and z grows linearly with v
    for i in range(0, X.shape[0], 7):
        for j in range(X.shape[1]):
            assert _distance_to_t_outline(np.array([X[i, j], Y[i, j]])) < 1e-12
    np.testing.assert_allclose(Z, np.outer(np.ones(len(u)), 2.5 * v), atol=1e-13)


def test_t_shape_cross_sections_equal_profile():
    cp, wp, kp, dp = t_profile()
    net, w, ku, kv, du, dv = nurbs_t_shape(depth=1.0)
    u = sample_parameters(ku, 200)
    X, Y, Z = nurbs_surface(ku, kv, du, dv, net, w, u_values=u, v_values=[0.0, 0.5, 1.0])
    prof = nurbs_curve(kp, dp, cp, wp, u_values=u)
    for j in range(3):
        np.testing.assert_allclose(X[:, j], prof[:, 0], atol=1e-13)
        np.testing.assert_allclose(Y[:, j], prof[:, 1], atol=1e-13)


def test_t_shape_invalid_depth():
    with pytest.raises(ValueError, match="height must be positive"):
        nurbs_t_shape(depth=0.0)


def test_cylinder_still_exact_with_explicit_parameters():
    net, w, ku, kv, du, dv = nurbs_cylinder(radius=2.0, height=3.0)
    X, Y, Z = nurbs_surface(ku, kv, du, dv, net, w,
                            u_values=sample_parameters(ku, 500), v_values=np.linspace(0, 1, 9))
    assert np.max(np.abs(np.hypot(X, Y) - 2.0)) < 1e-12
    assert Z.min() == pytest.approx(0.0) and Z.max() == pytest.approx(3.0)


def test_cylinder_invalid_radius():
    with pytest.raises(ValueError, match="radius must be positive"):
        nurbs_cylinder(radius=0.0)


def test_extrude_rejects_3d_curve():
    with pytest.raises(ValueError, match="Only 2D curves"):
        extrude_curve(np.zeros((3, 3)) + np.arange(3)[:, None], [1, 1, 1], [0, 0, 0, 1, 1, 1], 2)


def test_vectorised_surface_matches_naive_double_sum():
    rng = np.random.default_rng(7)
    net = rng.normal(size=(5, 4, 3))
    w = rng.uniform(0.3, 3.0, size=(5, 4))
    ku = [0, 0, 0, 0.4, 0.7, 1, 1, 1]
    kv = [0, 0, 0.3, 0.6, 1, 1]
    du, dv = 2, 1
    X, Y, Z = nurbs_surface(ku, kv, du, dv, net, w, n_eval_u=9, n_eval_v=8)
    for a, u in enumerate(np.linspace(0, 1, 9)):
        for b, v in enumerate(np.linspace(0, 1, 8)):
            num, den = np.zeros(3), 0.0
            for i in range(5):
                for j in range(4):
                    nw = _basis_scalar(i, du, u, ku) * _basis_scalar(j, dv, v, kv) * w[i, j]
                    num += nw * net[i, j]
                    den += nw
            np.testing.assert_allclose([X[a, b], Y[a, b], Z[a, b]], num / den, atol=1e-13)


def test_sample_parameters_contains_all_knots():
    knots = [0, 0, 0, 0.1, 0.1, 0.37, 1, 1, 1]
    u = sample_parameters(knots, 10)
    for k in set(knots):
        assert np.any(u == k)
    assert np.all(np.diff(u) > 0)
    assert u[0] == 0 and u[-1] == 1


def test_fit_circle_on_exact_circle_and_arc():
    cp, w, k, d = nurbs_circle(radius=2.5, center=(3.0, -4.0))
    c = nurbs_curve(k, d, cp, w, n_eval=400)
    cx, cy, r, err = fit_circle(c)
    assert (cx, cy, r) == pytest.approx((3.0, -4.0, 2.5), abs=1e-12)
    assert err < 1e-12
    # A quarter arc is detected too
    q = nurbs_curve([0, 0, 0, 1, 1, 1], 2, [[1, 0], [1, 1], [0, 1]], [1, math.sqrt(2) / 2, 1], 50)
    assert fit_circle(q)[3] < 1e-12


def test_fit_circle_rejects_line_and_flags_non_circles():
    with pytest.raises(ValueError, match="collinear"):
        fit_circle(np.column_stack([np.linspace(0, 1, 20), 2 * np.linspace(0, 1, 20)]))
    cp, w, k, d = t_profile()
    assert fit_circle(nurbs_curve(k, d, cp, w, n_eval=200))[3] > 0.1
