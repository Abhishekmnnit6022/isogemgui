"""Tests for Bezier curves and surfaces."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial import Delaunay

from core.bezier import bernstein, bezier_curve, bezier_surface
from core.validation import validate_control_points, validate_surface_net


def test_bernstein_partition_of_unity():
    """Bernstein polynomials must sum to 1.0 across [0, 1] for any degree."""
    u = np.linspace(0.0, 1.0, 100)
    for n in [1, 2, 3, 5, 9]:
        total = np.zeros_like(u)
        for i in range(n + 1):
            total += bernstein(i, n, u)
        assert np.allclose(total, 1.0, atol=1e-12)


def test_bezier_endpoint_interpolation():
    """Bezier curve must start at P_0 and end at P_n."""
    cp_2d = np.array([[1.0, 1.0], [2.0, 3.0], [4.0, 3.0], [3.0, 1.0]])
    curve_2d = bezier_curve(cp_2d, n_eval=101)
    assert np.allclose(curve_2d[0], cp_2d[0])
    assert np.allclose(curve_2d[-1], cp_2d[-1])

    cp_3d = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [4.0, 4.0, 4.0]])
    curve_3d = bezier_curve(cp_3d, n_eval=101)
    assert np.allclose(curve_3d[0], cp_3d[0])
    assert np.allclose(curve_3d[-1], cp_3d[-1])


def test_bezier_closed_form_matches():
    """Degrees 1, 2, and 3 match exact closed-form algebraic formulas."""
    u = np.linspace(0.0, 1.0, 50)

    # Degree 1
    p0, p1 = np.array([1.0, 2.0]), np.array([4.0, 6.0])
    cp1 = np.array([p0, p1])
    expected1 = np.outer(1.0 - u, p0) + np.outer(u, p1)
    actual1 = bezier_curve(cp1, n_eval=50)
    assert np.allclose(actual1, expected1)

    # Degree 2
    p2 = np.array([3.0, 8.0])
    cp2 = np.array([p0, p1, p2])
    expected2 = (
        np.outer((1.0 - u) ** 2, p0)
        + np.outer(2.0 * u * (1.0 - u), p1)
        + np.outer(u ** 2, p2)
    )
    actual2 = bezier_curve(cp2, n_eval=50)
    assert np.allclose(actual2, expected2)

    # Degree 3
    p3 = np.array([7.0, 1.0])
    cp3 = np.array([p0, p1, p2, p3])
    expected3 = (
        np.outer((1.0 - u) ** 3, p0)
        + np.outer(3.0 * u * ((1.0 - u) ** 2), p1)
        + np.outer(3.0 * (u ** 2) * (1.0 - u), p2)
        + np.outer(u ** 3, p3)
    )
    actual3 = bezier_curve(cp3, n_eval=50)
    assert np.allclose(actual3, expected3)


def test_bezier_reference_values_from_matlab():
    """Verify exact reference values from Matlab Bezier_3rd_Degree.m.

    Control points: (1, 1), (2, 3), (4, 3), (3, 1)
    At u = 0.5:
      x = (1/8)*1 + (3/8)*2 + (3/8)*4 + (1/8)*3 = 2.75
      y = (1/8)*1 + (3/8)*3 + (3/8)*3 + (1/8)*1 = 2.50
    """
    cp = np.array([[1.0, 1.0], [2.0, 3.0], [4.0, 3.0], [3.0, 1.0]])
    curve = bezier_curve(cp, n_eval=3)  # u = [0.0, 0.5, 1.0]
    # Midpoint u=0.5
    assert np.isclose(curve[1, 0], 2.75, atol=1e-12)
    assert np.isclose(curve[1, 1], 2.50, atol=1e-12)


def test_bezier_convex_hull_property():
    """Curve points must lie within the convex hull of control points."""
    cp = np.array([[0.0, 0.0], [2.0, 5.0], [4.0, 6.0], [6.0, 1.0]])
    curve = bezier_curve(cp, n_eval=100)

    # Delaunay triangulation can check if points are inside convex hull
    delaunay = Delaunay(cp)
    # find_simplex returns -1 for points strictly outside
    inside = delaunay.find_simplex(curve) >= 0
    assert np.all(inside)


def test_bezier_surface_corners_and_shape():
    """Tensor-product Bezier surface interpolates the 4 corner net points."""
    net = np.array([
        [[-0.5, 2.0, 1.0], [-2.0, 1.0, -1.0], [0.0, 0.0, 2.0]],
        [[1.0, 2.0, 0.0], [1.0, 8.0, -8.5], [1.0, -1.0, 2.0]],
        [[2.0, 2.0, 0.5], [2.0, 1.0, 1.0], [2.0, 1.0, 2.0]],
    ])
    X, Y, Z = bezier_surface(net, u_eval=20, v_eval=20)
    assert X.shape == (20, 20)
    assert Y.shape == (20, 20)
    assert Z.shape == (20, 20)

    # Corners
    assert np.allclose([X[0, 0], Y[0, 0], Z[0, 0]], net[0, 0])
    assert np.allclose([X[0, -1], Y[0, -1], Z[0, -1]], net[0, -1])
    assert np.allclose([X[-1, 0], Y[-1, 0], Z[-1, 0]], net[-1, 0])
    assert np.allclose([X[-1, -1], Y[-1, -1], Z[-1, -1]], net[-1, -1])


def test_bezier_validation_errors():
    """Test ValueError is raised for invalid control point inputs."""
    with pytest.raises(ValueError, match="at least 2 control points"):
        validate_control_points(np.empty((0, 2)))

    with pytest.raises(ValueError, match="2-D or 3-D"):
        validate_control_points(np.zeros((4, 4)))

    with pytest.raises(ValueError, match="at least 2x2"):
        validate_surface_net(np.zeros((1, 3, 3)))

    with pytest.raises(ValueError, match="at least 2 control points"):
        bezier_curve(np.array([[1.0, 1.0]]))
