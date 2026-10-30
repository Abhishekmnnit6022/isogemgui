"""NURBS (Non-Uniform Rational B-Spline) curve and surface evaluation.

Reuses the Cox-de Boor recursion from core.bspline and adds rational
weighting on top.  Pure math: no GUI, no plotting, no I/O.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np

from .bspline import _basis_scalar, basis_matrix
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

    N, u = basis_matrix(knots_list, degree, n_eval)  # (n_basis, n_eval)

    # Weighted basis: W_i = N_i * w_i
    W = N * w[:, np.newaxis]
    denom = W.sum(axis=0)  # (n_eval,)

    # Avoid division by zero at degenerate points
    denom[denom == 0.0] = 1.0

    R = W / denom[np.newaxis, :]
    return u, R


# NURBS curve

def nurbs_curve(
    knots: Sequence[float],
    degree: int,
    control_points: np.ndarray,
    weights: Sequence[float],
    n_eval: int = 200,
) -> np.ndarray:
    """Evaluate a NURBS curve.

    C(u) = sum_i R_i(u) * P_i

    Args:
        knots: Knot vector.
        degree: Polynomial degree.
        control_points: Shape (n, 2) or (n, 3).
        weights: One positive weight per control point.
        n_eval: Number of evaluation points.

    Returns:
        Curve points, shape (n_eval, dim).
    """
    knots_list = validate_knots(knots)
    degree = validate_degree(degree)
    cp = validate_control_points(control_points)
    n_cp = cp.shape[0]
    w = validate_weights(weights, n_cp)
    validate_bspline_params(knots_list, degree, n_cp)

    dim = cp.shape[1]
    u = np.linspace(knots_list[0], knots_list[-1], n_eval)
    curve = np.zeros((n_eval, dim))

    for j, u_val in enumerate(u):
        numer = np.zeros(dim)
        denom = 0.0
        for i in range(n_cp):
            Ni = _basis_scalar(i, degree, u_val, knots_list)
            nw = Ni * w[i]
            numer += nw * cp[i]
            denom += nw
        if denom != 0.0:
            curve[j] = numer / denom

    return curve


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

    Returns:
        (X, Y, Z) grids, each shape (n_eval_u, n_eval_v).
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

    u_arr = np.linspace(ku[0], ku[-1], n_eval_u)
    v_arr = np.linspace(kv[0], kv[-1], n_eval_v)

    X = np.zeros((n_eval_u, n_eval_v))
    Y = np.zeros((n_eval_u, n_eval_v))
    Z = np.zeros((n_eval_u, n_eval_v))

    for ui, u_val in enumerate(u_arr):
        # Pre-compute N_i(u) for all rows
        Nu = np.array([_basis_scalar(i, du, u_val, ku) for i in range(rows)])
        for vi, v_val in enumerate(v_arr):
            # Pre-compute N_j(v) for all cols
            Nv = np.array([_basis_scalar(j, dv, v_val, kv) for j in range(cols)])
            # Weighted sum
            denom = 0.0
            sx, sy, sz = 0.0, 0.0, 0.0
            for i in range(rows):
                for j in range(cols):
                    nw = Nu[i] * Nv[j] * w[i, j]
                    sx += nw * net[i, j, 0]
                    sy += nw * net[i, j, 1]
                    sz += nw * net[i, j, 2]
                    denom += nw
            if denom != 0.0:
                X[ui, vi] = sx / denom
                Y[ui, vi] = sy / denom
                Z[ui, vi] = sz / denom

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
    cp2d, w1d, knots_u, degree_u = nurbs_circle(radius=radius)
    n_cp = cp2d.shape[0]  # 9

    # Two rows along v (bottom z=0, top z=height)
    net = np.zeros((n_cp, 2, 3))
    weights = np.zeros((n_cp, 2))

    for i in range(n_cp):
        # Bottom
        net[i, 0, 0] = cp2d[i, 0]
        net[i, 0, 1] = cp2d[i, 1]
        net[i, 0, 2] = 0.0
        weights[i, 0] = w1d[i]

        # Top
        net[i, 1, 0] = cp2d[i, 0]
        net[i, 1, 1] = cp2d[i, 1]
        net[i, 1, 2] = height
        weights[i, 1] = w1d[i]

    knots_v = [0.0, 0.0, 1.0, 1.0]
    degree_v = 1

    return net, weights, knots_u, knots_v, degree_u, degree_v
