"""B-spline basis function and curve evaluation using Cox-de Boor algorithm."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple
import numpy as np

from .validation import (
    validate_bspline_params,
    validate_control_points,
    validate_degree,
    validate_knots,
)


# Recursive Cox-de Boor function for a single parameter value u (kept as the reference)
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


# All basis functions N_{i,p} at many parameter values at once (same recurrence, vectorised)
def basis_functions(knots: Sequence[float], degree: int, u: Sequence[float]) -> np.ndarray:
    """Evaluates every B-spline basis function N_{i,p}(u) for an array of parameters.

    Uses the Cox-de Boor recurrence bottom-up (degree 0, 1, ..., p) on whole arrays,
    with exactly the same rules as _basis_scalar (0/0 = 0, right end point included).

    Returns:
        Array of shape (len(knots) - degree - 1, len(u)).
    """
    k = np.asarray(validate_knots(knots), dtype=float)
    p = validate_degree(degree)
    u = np.atleast_1d(np.asarray(u, dtype=float))
    n_spans = len(k) - 1
    if n_spans - p < 1:
        raise ValueError(
            f"len(knots) - degree - 1 must be >= 1 (got {n_spans - p}). "
            f"Add more knots or reduce the degree."
        )

    # Degree 0: indicator of each half-open span [k_i, k_{i+1})
    N = np.zeros((n_spans, u.size))
    for i in range(n_spans):
        if k[i] < k[i + 1]:
            N[i] = (k[i] <= u) & (u < k[i + 1])
    # The last non-empty span also contains the final knot value
    nonempty = [i for i in range(n_spans) if k[i] < k[i + 1]]
    if nonempty:
        last = nonempty[-1]
        if k[last + 1] == k[-1]:
            N[last, u == k[-1]] = 1.0

    # Raise the degree one step at a time
    for q in range(1, p + 1):
        nxt = np.zeros((n_spans - q, u.size))
        for i in range(n_spans - q):
            d1 = k[i + q] - k[i]
            d2 = k[i + q + 1] - k[i + 1]
            if d1 != 0.0:
                nxt[i] += (u - k[i]) / d1 * N[i]
            if d2 != 0.0:
                nxt[i] += (k[i + q + 1] - u) / d2 * N[i + 1]
        N = nxt
    return N


# Parameter range on which the basis functions add up to 1
def valid_domain(knots: Sequence[float], degree: int) -> Tuple[float, float]:
    """Returns (u_start, u_end) = (knots[p], knots[-p-1]), where the curve is defined.

    For clamped (open) knot vectors this is the whole knot range. For unclamped
    knots (for example uniform 0, 1, 2, ...) the first and last p spans are not
    covered by a full set of basis functions, so a curve must not be drawn there.
    """
    k = validate_knots(knots)
    p = validate_degree(degree)
    if len(k) < 2 * p + 2:
        raise ValueError(
            f"Knot vector is too short for degree {p}: need at least {2 * p + 2} knots, got {len(k)}."
        )
    start, end = k[p], k[-p - 1]
    if not start < end:
        raise ValueError(
            f"Knot vector has an empty valid range: knots[{p}] = {start} must be "
            f"smaller than knots[{len(k) - p - 1}] = {end}."
        )
    return start, end


# Generate the full basis function matrix across parameter array t
def basis_matrix(
    knots: Sequence[float],
    degree: int,
    n_eval: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculates all B-spline basis functions over an evenly spaced range of u values
    covering the whole knot vector (so the plot shows every function completely).
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
    return basis_functions(knots_list, degree, t), t


# Calculate 2D or 3D B-spline curve points from knots and control points
def bspline_curve(
    knots: Sequence[float],
    degree: int,
    control_points: np.ndarray,
    n_eval: int = 200,
    u_values: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Calculates points along the B-spline curve by multiplying basis functions with control points.

    The curve is sampled on its valid range [knots[p], knots[-p-1]] (the whole range for
    clamped knots). Pass u_values to evaluate at chosen parameters instead.
    """
    knots_list = validate_knots(knots)
    degree = validate_degree(degree)
    cp = validate_control_points(control_points)
    validate_bspline_params(knots_list, degree, cp.shape[0])

    if u_values is None:
        start, end = valid_domain(knots_list, degree)
        u = np.linspace(start, end, n_eval)
    else:
        u = np.asarray(u_values, dtype=float)
    N = basis_functions(knots_list, degree, u)  # (n_cp, len(u))
    return N.T @ cp


# Basis functions and their first derivatives dN_{i,p}/du at many parameters
def basis_function_derivatives(
    knots: Sequence[float], degree: int, u: Sequence[float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (N, dN), both of shape (len(knots) - degree - 1, len(u)).

    Uses the standard formula
        dN_{i,p}/du = p * ( N_{i,p-1} / (u_{i+p} - u_i) - N_{i+1,p-1} / (u_{i+p+1} - u_{i+1}) )
    with 0/0 treated as 0. At a knot the right-hand span is used, like basis_functions().
    """
    k = np.asarray(validate_knots(knots), dtype=float)
    p = validate_degree(degree)
    u = np.atleast_1d(np.asarray(u, dtype=float))
    N = basis_functions(k, p, u)
    dN = np.zeros_like(N)
    if p == 0:
        return N, dN
    lower = basis_functions(k, p - 1, u)  # one more function than N
    for i in range(N.shape[0]):
        d1 = k[i + p] - k[i]
        d2 = k[i + p + 1] - k[i + 1]
        if d1 != 0.0:
            dN[i] += p / d1 * lower[i]
        if d2 != 0.0:
            dN[i] -= p / d2 * lower[i + 1]
    return N, dN


# Distinct knot values inside the valid range: the element (knot span) boundaries
def element_boundaries(knots: Sequence[float], degree: int) -> np.ndarray:
    """Sorted distinct knots in [knots[p], knots[-p-1]].

    Consecutive values bound one element (a non-empty knot span), so the number
    of elements is len(result) - 1.
    """
    start, end = valid_domain(knots, degree)
    k = np.asarray(validate_knots(knots), dtype=float)
    return np.unique(k[(k >= start) & (k <= end)])
