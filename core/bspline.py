"""B-spline basis function and curve evaluation using Cox-de Boor algorithm."""

from __future__ import annotations

from typing import List, Optional, Sequence
import numpy as np

from .validation import (
    validate_bspline_params,
    validate_control_points,
    validate_degree,
    validate_knots,
)


# Recursive Cox-de Boor function for a single parameter value u
def _basis_scalar(i: int, p: int, u: float, knots: List[float]) -> float:
    """Computes the basis function N_{i,p}(u) at a single number u.
    Zero division is treated as zero.
    The last knot span includes the right endpoint so clamped curves reach the final control point.
    """
    if p == 0:
        if knots[i] == knots[i + 1]:
            return 0.0
        if knots[i] <= u < knots[i + 1]:
            return 1.0
        # Include right endpoint when u reaches the last knot
        if u == knots[i + 1] and knots[i + 1] == knots[-1] and knots[i] < knots[i + 1]:
            return 1.0
        return 0.0

    d1 = knots[i + p] - knots[i]
    d2 = knots[i + p + 1] - knots[i + 1]

    c1 = 0.0 if d1 == 0.0 else (u - knots[i]) / d1
    c2 = 0.0 if d2 == 0.0 else (knots[i + p + 1] - u) / d2

    return c1 * _basis_scalar(i, p - 1, u, knots) + c2 * _basis_scalar(i + 1, p - 1, u, knots)


# Generate the full basis function matrix across parameter array t
def basis_matrix(
    knots: Sequence[float],
    degree: int,
    n_eval: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculates all B-spline basis functions over an evenly spaced range of u values.
    Returns (basis_matrix, parameter_array).
    """
    knots_list = validate_knots(knots)
    degree = validate_degree(degree)
    n_basis = len(knots_list) - degree - 1
    if n_basis < 1:
        raise ValueError(
            f"len(knots) - degree - 1 must be >= 1 (got {n_basis}). "
            f"Add more knots or reduce the degree."
        )

    t = np.linspace(knots_list[0], knots_list[-1], n_eval)
    basis = np.zeros((n_basis, n_eval))
    for i in range(n_basis):
        for j, u_val in enumerate(t):
            basis[i, j] = _basis_scalar(i, degree, u_val, knots_list)
    return basis, t


# Calculate 2D or 3D B-spline curve points from knots and control points
def bspline_curve(
    knots: Sequence[float],
    degree: int,
    control_points: np.ndarray,
    n_eval: int = 200,
) -> np.ndarray:
    """Calculates points along the B-spline curve by multiplying basis functions with control points."""
    knots_list = validate_knots(knots)
    degree = validate_degree(degree)
    cp = validate_control_points(control_points)
    validate_bspline_params(knots_list, degree, cp.shape[0])

    n_basis = cp.shape[0]
    dim = cp.shape[1]

    t = np.linspace(knots_list[0], knots_list[-1], n_eval)
    curve = np.zeros((n_eval, dim))
    for j, u_val in enumerate(t):
        for i in range(n_basis):
            curve[j] += _basis_scalar(i, degree, u_val, knots_list) * cp[i]
    return curve
