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


def _basis_scalar(i: int, p: int, u: float, knots: List[float]) -> float:
    """N_{i,p}(u) for one u by plain recursion. Kept as a reference for the tests."""
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


def basis_functions(knots: Sequence[float], degree: int, u: Sequence[float]) -> np.ndarray:
    """All basis functions N_{i,p} at many u values at once.

    Same Cox-de Boor rule as _basis_scalar, built up degree by degree.
    Returns an array of shape (number of functions, len(u))."""
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


def valid_domain(knots: Sequence[float], degree: int) -> Tuple[float, float]:
    """(knots[p], knots[-p-1]): the range where the curve is defined."""
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


def basis_matrix(
    knots: Sequence[float],
    degree: int,
    n_eval: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Basis functions over the whole knot range, for plots and Excel."""
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


def bspline_curve(
    knots: Sequence[float],
    degree: int,
    control_points: np.ndarray,
    n_eval: int = 200,
    u_values: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Points on a B-spline curve (2D or 3D)."""
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


def basis_function_derivatives(
    knots: Sequence[float], degree: int, u: Sequence[float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Basis functions N and their first derivatives dN/du."""
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


def element_boundaries(knots: Sequence[float], degree: int) -> np.ndarray:
    """Distinct knots inside the valid range (the ends of each element)."""
    start, end = valid_domain(knots, degree)
    k = np.asarray(validate_knots(knots), dtype=float)
    return np.unique(k[(k >= start) & (k <= end)])
