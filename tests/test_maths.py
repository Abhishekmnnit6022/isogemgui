"""Maths tests: Bezier, knots, B-spline, NURBS and the built-in shapes."""

from __future__ import annotations

import math
import time

import numpy as np
import pytest
from scipy.spatial import Delaunay

from core.bezier import bernstein, bezier_curve, bezier_surface
from core.bspline import _basis_scalar, basis_functions, basis_matrix, bspline_curve, valid_domain
from core.knots import open_uniform, uniform
from core.nurbs import (
    extrude_curve,
    fit_circle,
    nurbs_circle,
    nurbs_curve,
    nurbs_cylinder,
    nurbs_surface,
    nurbs_t_shape,
    rational_basis_matrix,
    sample_parameters,
    t_profile,
)
from core.validation import (
    validate_bspline_params,
    validate_control_points,
    validate_degree,
    validate_knots,
    validate_surface_net,
    validate_surface_weights,
    validate_weights,
)


# ---------------- Bezier curves and surfaces

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
    """Same points as Bezier_3rd_Degree.m: at u = 0.5 the curve is at (2.75, 2.5)."""
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


# ---------------- Knot vectors

def test_open_uniform_clamped():
    """Verify open clamped knot structure."""
    n_cp = 6
    p = 3
    knots = open_uniform(n_cp, p)
    # Total knots = n_cp + p + 1 = 6 + 3 + 1 = 10
    assert len(knots) == 10
    assert knots[:p + 1] == [0.0] * 4
    assert knots[-(p + 1):] == [1.0] * 4
    # Internal knots are non-decreasing
    for i in range(len(knots) - 1):
        assert knots[i] <= knots[i + 1]


def test_open_uniform_insufficient_points():
    """ValueError raised if n_control_points < degree + 1."""
    with pytest.raises(ValueError, match="Need at least"):
        open_uniform(3, 3)


def test_uniform_knots():
    """Uniform knots are sequential integers."""
    n_cp = 4
    p = 2
    knots = uniform(n_cp, p)
    assert len(knots) == 7
    assert knots == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


# ---------------- B-spline curves

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
    """Same data as Bspline_Verify.m (degree 2, knots 0,0,0,1,2,3,4,4,5,5,5)."""
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


# ---------------- NURBS curves and surfaces

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


# ---------------- T-shape, cylinder, sampling and circle fit

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


# ---------------- Fixed bugs: uniform knots and fast basis

KNOT_CASES = [
    ([0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 5], 2),       # thesis example, double interior knot
    ([0, 1, 2, 3, 4, 5, 6, 7], 2),                # uniform, unclamped
    ([0, 0, 0, 0, 0.3, 0.3, 0.7, 1, 1, 1, 1], 3),  # clamped cubic with repeated knot
    ([0, 0.5, 1, 2, 2, 2, 3, 3.5], 1),            # irregular, unclamped
    ([0, 0, 1, 1], 0),                            # degree 0
    (open_uniform(9, 5), 5),
]


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
