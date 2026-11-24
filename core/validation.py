"""Input checks that raise ValueError with a clear message."""

from __future__ import annotations

from typing import List, Sequence

import numpy as np


def validate_knots(knots: Sequence[float]) -> List[float]:
    """At least 2 knots, never decreasing. Returns a list of floats."""
    knots = [float(k) for k in knots]
    if len(knots) < 2:
        raise ValueError("Knot vector must contain at least 2 values.")
    for i in range(len(knots) - 1):
        if knots[i + 1] < knots[i]:
            raise ValueError(
                f"Knot vector must be non-decreasing, but knots[{i}]={knots[i]} "
                f"> knots[{i + 1}]={knots[i + 1]}."
            )
    return knots


def validate_degree(degree: int) -> int:
    """Degree must be a non-negative integer (>= 0 for basis, >= 1 for curves)."""
    degree = int(degree)
    if degree < 0:
        raise ValueError(f"Degree must be >= 0, got {degree}.")
    return degree


def validate_control_points(points: np.ndarray) -> np.ndarray:
    """Points as an (n, 2) or (n, 3) array with at least 2 points."""
    pts = np.asarray(points, dtype=float)
    if pts.ndim == 1:
        # Single point -- reshape to (1, dim)
        pts = pts.reshape(1, -1)
    if pts.ndim != 2:
        raise ValueError(
            f"Control points must be a 2-D array (n_points x dimension), "
            f"got shape {pts.shape}."
        )
    if pts.shape[0] < 2:
        raise ValueError("Need at least 2 control points.")
    if pts.shape[1] not in (2, 3):
        raise ValueError(
            f"Control points must be 2-D or 3-D (columns=2 or 3), "
            f"got {pts.shape[1]} columns."
        )
    return pts


def validate_bspline_params(
    knots: Sequence[float], degree: int, n_control_points: int
) -> None:
    """Check len(knots) == number of control points + degree + 1."""
    n = n_control_points - 1
    p = degree
    expected_knots = n + p + 1 + 1  # m+1
    if len(knots) != expected_knots:
        raise ValueError(
            f"B-spline constraint violated: len(knots) must equal "
            f"n_control_points + degree + 1 = {n_control_points} + {degree} + 1 "
            f"= {expected_knots}, but got {len(knots)}."
        )


def validate_surface_net(net: np.ndarray) -> np.ndarray:
    """Net as a (rows, cols, 3) array, at least 2 x 2."""
    net = np.asarray(net, dtype=float)
    if net.ndim != 3:
        raise ValueError(
            f"Surface control net must be 3-D (rows x cols x dim), "
            f"got {net.ndim}-D array with shape {net.shape}."
        )
    if net.shape[2] != 3:
        raise ValueError(
            f"Surface control net must have 3 coordinates per point (x,y,z), "
            f"got {net.shape[2]}."
        )
    if net.shape[0] < 2 or net.shape[1] < 2:
        raise ValueError(
            f"Surface control net must be at least 2x2, got {net.shape[0]}x{net.shape[1]}."
        )
    return net


def validate_weights(weights: Sequence[float], n_expected: int) -> np.ndarray:
    """One positive weight per control point. Returns an array."""
    w = np.asarray(weights, dtype=float).ravel()
    if w.shape[0] != n_expected:
        raise ValueError(
            f"Expected {n_expected} weights (one per control point), "
            f"got {w.shape[0]}."
        )
    if np.any(w <= 0.0):
        raise ValueError("All weights must be strictly positive.")
    return w


def validate_surface_weights(
    weights: Sequence[Sequence[float]], rows: int, cols: int
) -> np.ndarray:
    """Positive weights with the same shape as the net."""
    w = np.asarray(weights, dtype=float)
    if w.ndim != 2 or w.shape[0] != rows or w.shape[1] != cols:
        raise ValueError(
            f"Weights must have shape ({rows}, {cols}) matching the control net, "
            f"got shape {w.shape}."
        )
    if np.any(w <= 0.0):
        raise ValueError("All weights must be strictly positive.")
    return w
