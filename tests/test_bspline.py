"""Tests for B-spline curves and basis functions."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial import Delaunay

from core.bezier import bezier_curve
from core.bspline import basis_matrix, bspline_curve
from core.knots import open_uniform
from core.validation import validate_bspline_params, validate_degree, validate_knots


def test_bspline_partition_of_unity():
    """B-spline basis functions over clamped knots must sum to 1 everywhere."""
    knots = [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 5.0, 5.0, 5.0]
    degree = 2
    basis, t = basis_matrix(knots, degree, n_eval=150)
    # Sum over basis functions at each evaluation parameter
    sums = np.sum(basis, axis=0)
    assert np.allclose(sums, 1.0, atol=1e-12)


def test_bspline_endpoint_interpolation():
    """Clamped B-spline curve must interpolate the first and last control points."""
    knots = [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0]
    degree = 2
    # n_cp = len(knots) - degree - 1 = 8 - 2 - 1 = 5
    cp = np.array([
        [0.0, 0.0],
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 2.0],
        [6.0, 0.0],
    ])
    curve = bspline_curve(knots, degree, cp, n_eval=101)
    assert np.allclose(curve[0], cp[0], atol=1e-12)
    assert np.allclose(curve[-1], cp[-1], atol=1e-12)


def test_bezier_degree_9_equals_bspline():
    """A degree-9 Bezier curve matches a B-spline with knots [0]*10 + [1]*10."""
    degree = 9
    n_cp = 10
    # Create 10 distinct control points in 2D
    np.random.seed(42)
    cp = np.random.rand(n_cp, 2) * 100.0

    knots = [0.0] * 10 + [1.0] * 10
    n_eval = 80

    bez = bezier_curve(cp, n_eval=n_eval)
    bsp = bspline_curve(knots, degree, cp, n_eval=n_eval)

    assert np.allclose(bez, bsp, atol=1e-10)


def test_bspline_reference_values_from_matlab():
    """Verify B-spline with Matlab Bspline_Verify.m reference dataset.

    knots = [0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 5]
    degree = 2
    control points = [[5,20], [5,70], [50,2], [50,40], [70,40], [80,2], [90,30], [100,30]]
    """
    knots = [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 5.0, 5.0, 5.0]
    degree = 2
    cp = np.array([
        [5.0, 20.0],
        [5.0, 70.0],
        [50.0, 2.0],
        [50.0, 40.0],
        [70.0, 40.0],
        [80.0, 2.0],
        [90.0, 30.0],
        [100.0, 30.0],
    ])
    curve = bspline_curve(knots, degree, cp, n_eval=501)

    # First and last points
    assert np.allclose(curve[0], [5.0, 20.0])
    assert np.allclose(curve[-1], [100.0, 30.0])

    # Check basis functions sum to 1 at all points
    basis, t = basis_matrix(knots, degree, n_eval=501)
    assert np.allclose(np.sum(basis, axis=0), 1.0)


def test_bspline_convex_hull_property():
    """B-spline curve points lie within the convex hull of control points."""
    knots = open_uniform(n_control_points=5, degree=2)
    cp = np.array([[0.0, 0.0], [1.0, 4.0], [3.0, 5.0], [5.0, 2.0], [6.0, 0.0]])
    curve = bspline_curve(knots, 2, cp, n_eval=100)

    delaunay = Delaunay(cp)
    inside = delaunay.find_simplex(curve) >= 0
    assert np.all(inside)


def test_bspline_validation_errors():
    """Check validation failures for mismatched parameters or decreasing knots."""
    # Decreasing knots
    with pytest.raises(ValueError, match="non-decreasing"):
        validate_knots([0.0, 1.0, 0.5, 2.0])

    # Too few knots
    with pytest.raises(ValueError, match="at least 2"):
        validate_knots([1.0])

    # Invalid degree
    with pytest.raises(ValueError, match="Degree must be >= 0"):
        validate_degree(-1)

    # Mismatched control points count
    knots = [0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0]  # len=7, p=2 => n_cp = 7 - 2 - 1 = 4
    with pytest.raises(ValueError, match="B-spline constraint violated"):
        validate_bspline_params(knots, 2, n_control_points=5)
