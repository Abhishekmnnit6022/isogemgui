"""Tests for NURBS curves and surfaces."""

from __future__ import annotations

import math
import numpy as np
import pytest

from core.nurbs import (
    rational_basis_matrix,
    nurbs_curve,
    nurbs_surface,
    nurbs_circle,
    nurbs_cylinder,
)
from core.bspline import bspline_curve
from core.validation import validate_weights, validate_surface_weights


# Weights = 1 equivalence

def test_nurbs_curve_unit_weights_equals_bspline():
    """NURBS curve with all weights = 1 must equal the B-spline curve."""
    knots = [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0]
    degree = 2
    cp = np.array([
        [0.0, 0.0],
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 2.0],
        [6.0, 0.0],
    ])
    w = [1.0, 1.0, 1.0, 1.0, 1.0]
    n_eval = 200

    nurbs = nurbs_curve(knots, degree, cp, w, n_eval=n_eval)
    bsp = bspline_curve(knots, degree, cp, n_eval=n_eval)
    assert np.allclose(nurbs, bsp, atol=1e-12)


def test_nurbs_surface_unit_weights_equals_bspline_tensor():
    """NURBS surface with all weights = 1 matches the un-weighted tensor product."""
    knots_u = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    knots_v = [0.0, 0.0, 1.0, 1.0]
    du, dv = 2, 1
    net = np.array([
        [[0.0, 0.0, 0.0], [0.0, 1.0, 1.0]],
        [[1.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
        [[2.0, 0.0, 0.0], [2.0, 1.0, 1.0]],
    ])
    w_ones = np.ones((3, 2))
    n_u, n_v = 15, 10

    X, Y, Z = nurbs_surface(knots_u, knots_v, du, dv, net, w_ones,
                             n_eval_u=n_u, n_eval_v=n_v)

    # Manual un-weighted tensor product using bspline basis
    from core.bspline import _basis_scalar
    u_arr = np.linspace(knots_u[0], knots_u[-1], n_u)
    v_arr = np.linspace(knots_v[0], knots_v[-1], n_v)
    Xe = np.zeros((n_u, n_v))
    Ye = np.zeros((n_u, n_v))
    Ze = np.zeros((n_u, n_v))
    for ui, uv in enumerate(u_arr):
        for vi, vv in enumerate(v_arr):
            for i in range(3):
                for j in range(2):
                    b = _basis_scalar(i, du, uv, knots_u) * _basis_scalar(j, dv, vv, knots_v)
                    Xe[ui, vi] += b * net[i, j, 0]
                    Ye[ui, vi] += b * net[i, j, 1]
                    Ze[ui, vi] += b * net[i, j, 2]

    assert np.allclose(X, Xe, atol=1e-12)
    assert np.allclose(Y, Ye, atol=1e-12)
    assert np.allclose(Z, Ze, atol=1e-12)


# Rational basis: partition of unity

def test_rational_basis_partition_of_unity():
    """Rational basis functions must sum to 1 everywhere, including at the last knot."""
    knots = [0.0, 0.0, 0.0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0, 1.0]
    degree = 2
    w1 = math.sqrt(2.0) / 2.0
    weights = [1.0, w1, 1.0, w1, 1.0, w1, 1.0, w1, 1.0]

    u, R = rational_basis_matrix(knots, degree, weights, n_eval=400)
    sums = R.sum(axis=0)
    assert np.allclose(sums, 1.0, atol=1e-12)
    # Specifically check the last knot
    assert np.isclose(sums[-1], 1.0, atol=1e-14)


# Endpoint interpolation

def test_nurbs_curve_endpoint_interpolation():
    """Clamped NURBS curve must interpolate first and last control points."""
    knots = [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0]
    degree = 2
    cp = np.array([
        [0.0, 0.0],
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 2.0],
        [6.0, 0.0],
    ])
    w = [1.0, 2.0, 0.5, 3.0, 1.0]
    curve = nurbs_curve(knots, degree, cp, w, n_eval=201)
    assert np.allclose(curve[0], cp[0], atol=1e-12)
    assert np.allclose(curve[-1], cp[-1], atol=1e-12)


def test_nurbs_surface_corner_interpolation():
    """NURBS surface with clamped knots must interpolate the four corner points."""
    # Build a simple NURBS surface with clamped knots in both directions
    knots_u = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    knots_v = [0.0, 0.0, 1.0, 1.0]
    du, dv = 2, 1
    net = np.array([
        [[0.0, 0.0, 0.0], [0.0, 1.0, 1.0]],
        [[1.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
        [[2.0, 0.0, 0.0], [2.0, 1.0, 1.0]],
    ])
    w_surf = np.array([[1.0, 1.0], [2.0, 2.0], [1.0, 1.0]])

    X, Y, Z = nurbs_surface(knots_u, knots_v, du, dv, net, w_surf,
                             n_eval_u=20, n_eval_v=20)
    # Four corners of the surface
    assert np.allclose([X[0, 0], Y[0, 0], Z[0, 0]], net[0, 0], atol=1e-12)
    assert np.allclose([X[0, -1], Y[0, -1], Z[0, -1]], net[0, -1], atol=1e-12)
    assert np.allclose([X[-1, 0], Y[-1, 0], Z[-1, 0]], net[-1, 0], atol=1e-12)
    assert np.allclose([X[-1, -1], Y[-1, -1], Z[-1, -1]], net[-1, -1], atol=1e-12)


# Weight invariance: scaling all weights by a constant

def test_weight_scaling_invariance():
    """Multiplying all weights by a constant must leave the curve unchanged."""
    knots = [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0]
    degree = 2
    cp = np.array([
        [0.0, 0.0],
        [1.0, 3.0],
        [3.0, 5.0],
        [5.0, 2.0],
        [7.0, 0.0],
    ])
    w = [1.0, 2.0, 0.5, 3.0, 1.5]
    n_eval = 300

    curve1 = nurbs_curve(knots, degree, cp, w, n_eval=n_eval)
    curve2 = nurbs_curve(knots, degree, cp, [7.3 * wi for wi in w], n_eval=n_eval)
    assert np.allclose(curve1, curve2, atol=1e-12)


# Raising one weight pulls curve toward that control point

def test_raising_weight_pulls_curve():
    """Increasing one weight must pull the curve closer to that control point."""
    knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    degree = 2
    cp = np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 0.0]])
    n_eval = 101

    # Midpoint index
    mid = n_eval // 2

    curve_w1 = nurbs_curve(knots, degree, cp, [1.0, 1.0, 1.0], n_eval=n_eval)
    curve_w10 = nurbs_curve(knots, degree, cp, [1.0, 10.0, 1.0], n_eval=n_eval)

    # Distance from P1 at midpoint
    d1 = np.linalg.norm(curve_w1[mid] - cp[1])
    d10 = np.linalg.norm(curve_w10[mid] - cp[1])
    assert d10 < d1


# Quarter circle exact value

def test_quarter_circle_midpoint():
    """Quarter circle at u = 0.5 must be (sqrt(2)/2, sqrt(2)/2) exactly."""
    knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    degree = 2
    cp = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    w1 = math.sqrt(2.0) / 2.0
    weights = [1.0, w1, 1.0]

    # Evaluate at exactly 3 points: u = 0, 0.5, 1
    curve = nurbs_curve(knots, degree, cp, weights, n_eval=3)
    expected_mid = np.array([math.sqrt(2.0) / 2.0, math.sqrt(2.0) / 2.0])
    assert np.allclose(curve[1], expected_mid, atol=1e-14)


# Full circle accuracy

def test_full_circle_unit_radius():
    """All 400 samples of the unit circle must satisfy |r - 1| < 1e-12."""
    cp, w, knots, degree = nurbs_circle(radius=1.0, center=(0.0, 0.0))
    curve = nurbs_curve(knots, degree, cp, w, n_eval=400)
    radii = np.sqrt(curve[:, 0] ** 2 + curve[:, 1] ** 2)
    max_err = np.max(np.abs(radii - 1.0))
    assert max_err < 1e-12, f"Max radius error {max_err}"


def test_full_circle_shifted_and_scaled():
    """Circle with radius 2.5 centered at (3, -1): same accuracy."""
    r = 2.5
    cx, cy = 3.0, -1.0
    cp, w, knots, degree = nurbs_circle(radius=r, center=(cx, cy))
    curve = nurbs_curve(knots, degree, cp, w, n_eval=400)
    radii = np.sqrt((curve[:, 0] - cx) ** 2 + (curve[:, 1] - cy) ** 2)
    max_err = np.max(np.abs(radii - r))
    assert max_err < 1e-12, f"Max radius error {max_err}"


# Cylinder

def test_cylinder_geometry():
    """All surface points of the cylinder must satisfy x^2+y^2 = r^2 and 0 <= z <= h."""
    r = 1.5
    h = 3.0
    net, w, ku, kv, du, dv = nurbs_cylinder(radius=r, height=h)
    X, Y, Z = nurbs_surface(ku, kv, du, dv, net, w, n_eval_u=40, n_eval_v=10)

    radii = np.sqrt(X ** 2 + Y ** 2)
    max_r_err = np.max(np.abs(radii - r))
    assert max_r_err < 1e-12, f"Max radial error {max_r_err}"

    assert np.all(Z >= -1e-14)
    assert np.all(Z <= h + 1e-14)


# Validation errors

def test_invalid_weights_zero():
    """Zero weight raises ValueError."""
    with pytest.raises(ValueError, match="strictly positive"):
        validate_weights([1.0, 0.0, 1.0], 3)


def test_invalid_weights_negative():
    """Negative weight raises ValueError."""
    with pytest.raises(ValueError, match="strictly positive"):
        validate_weights([1.0, -0.5, 1.0], 3)


def test_invalid_weights_wrong_length():
    """Wrong number of weights raises ValueError."""
    with pytest.raises(ValueError, match="Expected 3 weights"):
        validate_weights([1.0, 1.0], 3)


def test_invalid_surface_weights():
    """Surface weights with wrong shape raise ValueError."""
    with pytest.raises(ValueError, match="shape"):
        validate_surface_weights([[1.0, 1.0]], 2, 2)


# 3D NURBS curve

def test_nurbs_curve_3d():
    """NURBS curve works with 3D control points."""
    knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    degree = 2
    cp = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 0.0, 0.0]])
    w = [1.0, 1.0, 1.0]
    curve = nurbs_curve(knots, degree, cp, w, n_eval=50)
    assert curve.shape == (50, 3)
    assert np.allclose(curve[0], cp[0], atol=1e-12)
    assert np.allclose(curve[-1], cp[-1], atol=1e-12)
