"""NURBS (Non-Uniform Rational B-Spline) curve and surface evaluation.

Reuses the Cox-de Boor basis from core.bspline and adds rational
weighting on top.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np

from .bspline import basis_functions, valid_domain
from .validation import (
    validate_bspline_params,
    validate_control_points,
    validate_degree,
    validate_knots,
    validate_surface_net,
    validate_weights,
    validate_surface_weights,
)


# Rational basis functions

def rational_basis_matrix(
    knots: Sequence[float],
    degree: int,
    weights: Sequence[float],
    n_eval: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the NURBS rational basis functions R_i(u).

    R_i(u) = N_i(u) * w_i / sum_j N_j(u) * w_j

    Args:
        knots: Knot vector (non-decreasing).
        degree: Polynomial degree.
        weights: One positive weight per basis function.
        n_eval: Number of evenly spaced evaluation points.

    Returns:
        (u, R) where u has shape (n_eval,) and R has shape (n_basis, n_eval).
    """
    knots_list = validate_knots(knots)
    degree = validate_degree(degree)
    n_basis = len(knots_list) - degree - 1
    w = validate_weights(weights, n_basis)

    # Sample the valid range only: outside it the basis does not sum to 1
    start, end = valid_domain(knots_list, degree)
    u = np.linspace(start, end, n_eval)
    return u, _rational(basis_functions(knots_list, degree, u), w)


# R = N * w / sum(N * w), column by column; columns with zero denominator stay zero
def _rational(N: np.ndarray, w: np.ndarray) -> np.ndarray:
    W = N * w[:, np.newaxis]
    denom = W.sum(axis=0)
    denom[denom == 0.0] = 1.0
    return W / denom[np.newaxis, :]


# NURBS curve

def nurbs_curve(
    knots: Sequence[float],
    degree: int,
    control_points: np.ndarray,
    weights: Sequence[float],
    n_eval: int = 200,
    u_values: Sequence[float] | None = None,
) -> np.ndarray:
    """Evaluate a NURBS curve.

    C(u) = sum_i R_i(u) * P_i

    Args:
        knots: Knot vector.
        degree: Polynomial degree.
        control_points: Shape (n, 2) or (n, 3).
        weights: One positive weight per control point.
        n_eval: Number of evaluation points.
        u_values: Optional explicit parameter values (overrides n_eval),
            e.g. from sample_parameters() so sharp corners at knots are hit exactly.

    Returns:
        Curve points, shape (len(u), dim).
    """
    knots_list = validate_knots(knots)
    degree = validate_degree(degree)
    cp = validate_control_points(control_points)
    n_cp = cp.shape[0]
    w = validate_weights(weights, n_cp)
    validate_bspline_params(knots_list, degree, n_cp)

    if u_values is None:
        start, end = valid_domain(knots_list, degree)
        u = np.linspace(start, end, n_eval)
    else:
        u = np.asarray(u_values, dtype=float)
    R = _rational(basis_functions(knots_list, degree, u), w)  # (n_cp, len(u))
    return R.T @ cp


# NURBS surface

def nurbs_surface(
    knots_u: Sequence[float],
    knots_v: Sequence[float],
    degree_u: int,
    degree_v: int,
    net: np.ndarray,
    weights: np.ndarray,
    n_eval_u: int = 30,
    n_eval_v: int = 30,
    u_values: Sequence[float] | None = None,
    v_values: Sequence[float] | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate a NURBS surface.

    S(u,v) = sum_ij R_ij(u,v) * P_ij
    R_ij   = N_i(u) * N_j(v) * w_ij / sum_kl N_k(u) * N_l(v) * w_kl

    Args:
        knots_u: Knot vector in the u-direction.
        knots_v: Knot vector in the v-direction.
        degree_u: Degree in the u-direction.
        degree_v: Degree in the v-direction.
        net: Control point net, shape (rows, cols, 3).
        weights: Weight net, shape (rows, cols).
        n_eval_u: Evaluation points along u.
        n_eval_v: Evaluation points along v.
        u_values, v_values: Optional explicit parameter values (override n_eval_u / n_eval_v).

    Returns:
        (X, Y, Z) grids, each shape (len(u), len(v)).
    """
    ku = validate_knots(knots_u)
    kv = validate_knots(knots_v)
    du = validate_degree(degree_u)
    dv = validate_degree(degree_v)
    net = validate_surface_net(net)
    rows, cols, _ = net.shape
    w = validate_surface_weights(weights, rows, cols)
    validate_bspline_params(ku, du, rows)
    validate_bspline_params(kv, dv, cols)

    u_arr = (np.linspace(*valid_domain(ku, du), n_eval_u) if u_values is None
             else np.asarray(u_values, dtype=float))
    v_arr = (np.linspace(*valid_domain(kv, dv), n_eval_v) if v_values is None
             else np.asarray(v_values, dtype=float))

    # Basis values are computed once per direction: Nu (len(u), rows), Nv (len(v), cols)
    Nu = basis_functions(ku, du, u_arr).T
    Nv = basis_functions(kv, dv, v_arr).T

    # Weighted tensor-product sums: denom = sum_ij Nu_i Nv_j w_ij, numer = same times P_ij
    denom = np.einsum("ai,bj,ij->ab", Nu, Nv, w)
    numer = np.einsum("ai,bj,ij,ijk->abk", Nu, Nv, w, net)
    safe = np.where(denom == 0.0, 1.0, denom)
    pts = numer / safe[:, :, np.newaxis]
    pts[denom == 0.0] = 0.0

    X = pts[:, :, 0]
    Y = pts[:, :, 1]
    Z = pts[:, :, 2]
    return X, Y, Z


# Standard exact circle (9-point quadratic NURBS)

def nurbs_circle(
    radius: float = 1.0,
    center: Tuple[float, float] = (0.0, 0.0),
) -> Tuple[np.ndarray, np.ndarray, List[float], int]:
    """Generate the standard 9-point exact NURBS circle (degree 2).

    Returns:
        (control_points, weights, knots, degree)
        control_points: shape (9, 2)
        weights: shape (9,)
        knots: list of 12 floats
        degree: 2
    """
    w1 = math.sqrt(2.0) / 2.0
    cx, cy = center
    r = radius

    # 9 control points at 0, 45, 90, 135, 180, 225, 270, 315, 360 degrees
    control_points = np.array([
        [r,        0.0],       # 0 deg
        [r,        r],         # 45 deg  (off-curve)
        [0.0,      r],         # 90 deg
        [-r,       r],         # 135 deg (off-curve)
        [-r,       0.0],       # 180 deg
        [-r,      -r],         # 225 deg (off-curve)
        [0.0,     -r],         # 270 deg
        [r,       -r],         # 315 deg (off-curve)
        [r,        0.0],       # 360 deg = 0 deg
    ], dtype=float)

    # Translate to center
    control_points[:, 0] += cx
    control_points[:, 1] += cy

    weights = np.array([1.0, w1, 1.0, w1, 1.0, w1, 1.0, w1, 1.0])

    knots = [0.0, 0.0, 0.0,
             0.25, 0.25,
             0.5, 0.5,
             0.75, 0.75,
             1.0, 1.0, 1.0]

    degree = 2

    return control_points, weights, knots, degree


# Cylinder: circle extruded along z

def nurbs_cylinder(
    radius: float = 1.0,
    height: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, List[float], List[float], int, int]:
    """Generate a NURBS cylinder by extruding a circle along z.

    Degree 2 in the u-direction (around), degree 1 in the v-direction (height).

    Returns:
        (net, weights, knots_u, knots_v, degree_u, degree_v)
        net: shape (9, 2, 3)
        weights: shape (9, 2)
        knots_u: circle knot vector (12 knots)
        knots_v: [0, 0, 1, 1]
        degree_u: 2
        degree_v: 1
    """
    if not float(radius) > 0.0:
        raise ValueError(f"Cylinder radius must be positive, got {radius}.")
    cp2d, w1d, knots_u, degree_u = nurbs_circle(radius=radius)
    return extrude_curve(cp2d, w1d, knots_u, degree_u, height)


# Extrude a planar NURBS curve along z into a NURBS surface

def extrude_curve(
    control_points: np.ndarray,
    weights: Sequence[float],
    knots: Sequence[float],
    degree: int,
    height: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, List[float], List[float], int, int]:
    """Sweep a 2D NURBS curve straight up along z from z = 0 to z = height.

    The curve becomes the u-direction; the v-direction is linear (degree 1,
    knots [0, 0, 1, 1]) with one row of control points at the bottom and one
    at the top, both carrying the curve weights.

    Returns:
        (net, weights, knots_u, knots_v, degree_u, degree_v)
        net: shape (n, 2, 3); weights: shape (n, 2).
    """
    cp = validate_control_points(control_points)
    if cp.shape[1] != 2:
        raise ValueError("Only 2D curves (x, y) can be extruded along z.")
    knots_list = validate_knots(knots)
    degree = validate_degree(degree)
    n_cp = cp.shape[0]
    w = validate_weights(weights, n_cp)
    validate_bspline_params(knots_list, degree, n_cp)
    height = float(height)
    if not height > 0.0:
        raise ValueError(f"Extrusion height must be positive, got {height}.")

    net = np.zeros((n_cp, 2, 3))
    net[:, :, 0] = cp[:, 0:1]
    net[:, :, 1] = cp[:, 1:2]
    net[:, 1, 2] = height
    surf_weights = np.column_stack([w, w])

    return net, surf_weights, knots_list, [0.0, 0.0, 1.0, 1.0], degree, 1


# T-shape: T cross-section with exact rounded inner corners

def t_profile(
    flange_width: float = 4.0,
    flange_thickness: float = 1.0,
    stem_width: float = 1.0,
    stem_height: float = 3.0,
    fillet_radius: float = 0.4,
) -> Tuple[np.ndarray, np.ndarray, List[float], int]:
    """Closed 2D outline of a T section as a degree-2 NURBS curve.

    The stem runs from y = 0 to y = stem_height and the flange (the top bar)
    sits on it from y = stem_height to y = stem_height + flange_thickness. The
    shape is centred on x = 0. The two inner corners where the stem meets the
    flange are rounded with exact quarter-circle arcs (middle weight sqrt(2)/2);
    every other corner is sharp. Each straight edge or arc is one rational
    quadratic Bezier piece, joined by double knots, so corners are exact.
    Set fillet_radius = 0 for a T with all-sharp corners.

    Returns:
        (control_points, weights, knots, degree) with control_points shape (n, 2).
    """
    W, tf = float(flange_width), float(flange_thickness)
    ws, hs, rf = float(stem_width), float(stem_height), float(fillet_radius)
    for name, val in (("flange_width", W), ("flange_thickness", tf),
                      ("stem_width", ws), ("stem_height", hs)):
        if not val > 0.0:
            raise ValueError(f"T-shape {name} must be positive, got {val}.")
    if not ws < W:
        raise ValueError(
            f"T-shape stem_width ({ws}) must be smaller than flange_width ({W})."
        )
    if rf < 0.0:
        raise ValueError(f"T-shape fillet_radius must be >= 0, got {rf}.")
    if rf >= hs or rf >= (W - ws) / 2.0:
        raise ValueError(
            f"T-shape fillet_radius ({rf}) must be smaller than stem_height ({hs}) "
            f"and the flange overhang ({(W - ws) / 2.0})."
        )

    a, b, h = ws / 2.0, W / 2.0, hs
    top = hs + tf
    w_arc = math.sqrt(2.0) / 2.0

    # Corner points in counter-clockwise order, starting at the bottom-left of the stem.
    # arcs maps a segment index to the sharp corner that its quarter-circle rounds off.
    if rf > 0.0:
        corners = [(-a, 0.0), (a, 0.0), (a, h - rf), (a + rf, h), (b, h), (b, top),
                   (-b, top), (-b, h), (-a - rf, h), (-a, h - rf)]
        arcs = {2: (a, h), 8: (-a, h)}
    else:
        corners = [(-a, 0.0), (a, 0.0), (a, h), (b, h), (b, top),
                   (-b, top), (-b, h), (-a, h)]
        arcs = {}

    # Each segment contributes its start point and a middle control point:
    # the midpoint (weight 1) for a straight edge, the corner (weight sqrt(2)/2) for an arc.
    n_seg = len(corners)
    cps: List[Tuple[float, float]] = []
    wts: List[float] = []
    for k in range(n_seg):
        p, q = corners[k], corners[(k + 1) % n_seg]
        if k in arcs:
            mid, wm = arcs[k], w_arc
        else:
            mid, wm = ((p[0] + q[0]) / 2.0, (p[1] + q[1]) / 2.0), 1.0
        cps.extend([p, mid])
        wts.extend([1.0, wm])
    cps.append(corners[0])  # close the loop
    wts.append(1.0)

    knots: List[float] = [0.0, 0.0, 0.0]
    for k in range(1, n_seg):
        knots.extend([k / n_seg, k / n_seg])
    knots.extend([1.0, 1.0, 1.0])

    return np.array(cps, dtype=float), np.array(wts), knots, 2


def nurbs_t_shape(
    flange_width: float = 4.0,
    flange_thickness: float = 1.0,
    stem_width: float = 1.0,
    stem_height: float = 3.0,
    fillet_radius: float = 0.4,
    depth: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, List[float], List[float], int, int]:
    """3D T-shaped NURBS surface: the T outline from t_profile() extruded along z.

    Returns:
        (net, weights, knots_u, knots_v, degree_u, degree_v)
        net: shape (n, 2, 3) with n = 21 control points (17 without fillets).
    """
    cp, w, knots, degree = t_profile(flange_width, flange_thickness, stem_width,
                                     stem_height, fillet_radius)
    return extrude_curve(cp, w, knots, degree, depth)


# Parameter sampling that always includes the knots

def sample_parameters(knots: Sequence[float], n_eval: int = 200,
                      degree: int | None = None) -> np.ndarray:
    """Evenly spaced parameters merged with every distinct knot inside the range.

    Sharp corners of a curve (repeated knots) sit exactly at knot values, so
    including the knots makes plots show corners exactly instead of cutting them.
    With a degree, only the valid range [knots[p], knots[-p-1]] is sampled
    (needed for unclamped knots); without one, the whole knot range is used.
    """
    k = validate_knots(knots)
    start, end = (k[0], k[-1]) if degree is None else valid_domain(k, degree)
    base = np.linspace(start, end, max(int(n_eval), 2))
    kk = np.asarray(k, dtype=float)
    inside = kk[(kk >= start) & (kk <= end)]
    return np.unique(np.concatenate([base, inside]))


# Exact circle check for evaluated 2D points

def fit_circle(points: np.ndarray) -> Tuple[float, float, float, float]:
    """Least-squares circle through 2D points.

    Returns (cx, cy, radius, max_radius_error), where max_radius_error is
    max | |p - c| - radius | over all points. Raises ValueError if the points
    are (nearly) collinear, so no finite circle fits them.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 2 or pts.shape[0] < 3:
        raise ValueError("Need at least 3 two-dimensional points to fit a circle.")
    x, y = pts[:, 0], pts[:, 1]
    # Solve x^2 + y^2 + a x + b y + c = 0 in the least-squares sense
    A = np.column_stack([x, y, np.ones_like(x)])
    rhs = -(x ** 2 + y ** 2)
    sol, _, rank, _ = np.linalg.lstsq(A, rhs, rcond=None)
    if rank < 3:
        raise ValueError("Points are collinear; no circle fits them.")
    a, b, c = sol
    cx, cy = -a / 2.0, -b / 2.0
    r_sq = cx ** 2 + cy ** 2 - c
    if r_sq <= 0.0:
        raise ValueError("Points do not lie on a circle.")
    radius = float(np.sqrt(r_sq))
    extent = float(np.ptp(pts[:, :2], axis=0).max())
    if extent == 0.0 or radius > 1e6 * extent:
        raise ValueError("Points are collinear; no circle fits them.")
    dist = np.hypot(x - cx, y - cy)
    return float(cx), float(cy), radius, float(np.max(np.abs(dist - radius)))
