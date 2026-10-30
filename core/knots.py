"""Knot vector generators for B-spline curves.

Provides functions for open/clamped, uniform, and periodic knot vectors.
"""

from __future__ import annotations

from typing import List


def open_uniform(n_control_points: int, degree: int) -> List[float]:
    """Generate an open (clamped) uniform knot vector.

    The first and last (degree+1) knots are clamped to 0 and 1 respectively.
    Internal knots are uniformly spaced.

    Args:
        n_control_points: Number of control points (n+1).
        degree: Polynomial degree p.

    Returns:
        Knot vector of length n_control_points + degree + 1.

    Raises:
        ValueError: if n_control_points < degree + 1.
    """
    n = n_control_points
    p = degree
    if n < p + 1:
        raise ValueError(
            f"Need at least degree+1 = {p + 1} control points, got {n}."
        )
    m = n + p  # last index, total knots = m + 1
    n_internal = m + 1 - 2 * (p + 1)

    knots: List[float] = [0.0] * (p + 1)
    for i in range(1, n_internal + 1):
        knots.append(i / (n_internal + 1))
    knots.extend([1.0] * (p + 1))
    return knots


def uniform(n_control_points: int, degree: int) -> List[float]:
    """Generate a uniform (periodic) knot vector.

    Knots are integers 0, 1, 2, ... , n_control_points + degree.

    Args:
        n_control_points: Number of control points.
        degree: Polynomial degree.

    Returns:
        Knot vector of length n_control_points + degree + 1.
    """
    total = n_control_points + degree + 1
    return [float(i) for i in range(total)]
