"""Knot vector generators."""

from __future__ import annotations

from typing import List


def open_uniform(n_control_points: int, degree: int) -> List[float]:
    """Clamped knots: p+1 zeros, evenly spaced inner knots, p+1 ones."""
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
    """Unclamped knots 0, 1, 2, ... (curve only valid on knots[p]..knots[-p-1])."""
    total = n_control_points + degree + 1
    return [float(i) for i in range(total)]
