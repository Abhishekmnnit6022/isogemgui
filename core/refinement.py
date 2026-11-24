"""Knot insertion for curves and surfaces. Adds control points, shape stays the same."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from .bspline import element_boundaries, valid_domain
from .validation import (
    validate_bspline_params,
    validate_control_points,
    validate_degree,
    validate_knots,
    validate_surface_net,
    validate_surface_weights,
    validate_weights,
)

DIRECTIONS = ("u", "v", "both")


# Insert u once into knots, updating homogeneous control points Pw (n, d)
def _insert_once(knots: List[float], p: int, Pw: np.ndarray, u: float):
    k = int(np.searchsorted(knots, u, side="right")) - 1  # knots[k] <= u < knots[k+1]
    n = Pw.shape[0]
    Q = np.empty((n + 1, Pw.shape[1]))
    Q[:k - p + 1] = Pw[:k - p + 1]
    for i in range(k - p + 1, k + 1):
        a = (u - knots[i]) / (knots[i + p] - knots[i])
        Q[i] = a * Pw[i] + (1.0 - a) * Pw[i - 1]
    Q[k + 1:] = Pw[k:]
    return knots[:k + 1] + [u] + knots[k + 1:], Q


# Check u and times, snap u onto an existing knot if it is (numerically) equal
def _prepare_insert(knots: List[float], p: int, u: float, times: int) -> float:
    try:
        u = float(u)
    except (TypeError, ValueError):
        raise ValueError(f"Knot value must be a number, got {u!r}.") from None
    if int(times) != times or times < 1:
        raise ValueError(f"Number of insertions must be a whole number >= 1, got {times}.")
    if p < 1:
        raise ValueError("Knot insertion needs degree >= 1.")
    start, end = valid_domain(knots, p)
    tol = 1e-12 * (end - start)
    if not (start + tol < u < end - tol):
        raise ValueError(
            f"Knot value u = {u:g} must lie strictly inside the parameter range "
            f"({start:g}, {end:g})."
        )
    for k in knots:
        if abs(k - u) <= tol:
            u = k
            break
    s = sum(1 for k in knots if k == u)
    if s + times > p:
        raise ValueError(
            f"Cannot insert u = {u:g} {int(times)} more time(s): it already appears {s} time(s) "
            f"and degree {p} allows at most {p} (the curve would break apart)."
        )
    return u


# Insert into homogeneous points: the shared engine for curves and surfaces
def _insert_homogeneous(knots: List[float], p: int, Pw: np.ndarray, u: float, times: int):
    u = _prepare_insert(knots, p, u, times)
    for _ in range(int(times)):
        knots, Pw = _insert_once(knots, p, Pw, u)
    return knots, Pw


# (x, y, z), w -> (w*x, w*y, w*z, w): knot insertion is linear in these
def _to_homogeneous(cp: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.concatenate([cp * w[..., np.newaxis], w[..., np.newaxis]], axis=-1)


# Back from homogeneous coordinates to points and weights
def _from_homogeneous(Pw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    w = Pw[..., -1]
    return Pw[..., :-1] / w[..., np.newaxis], w


# Curves

def insert_knot(
    knots: Sequence[float],
    degree: int,
    control_points: np.ndarray,
    u: float,
    times: int = 1,
    weights: Optional[Sequence[float]] = None,
) -> Tuple[List[float], np.ndarray, Optional[np.ndarray]]:
    """Insert knot u (times times) into a B-spline or NURBS curve.

    Returns (knots, control_points, weights); weights is None for a B-spline."""
    k = validate_knots(knots)
    p = validate_degree(degree)
    cp = validate_control_points(control_points)
    validate_bspline_params(k, p, cp.shape[0])
    w = np.ones(cp.shape[0]) if weights is None else validate_weights(weights, cp.shape[0])
    new_k, Pw = _insert_homogeneous(k, p, _to_homogeneous(cp, w), u, times)
    new_cp, new_w = _from_homogeneous(Pw)
    return new_k, new_cp, (None if weights is None else new_w)


def refine_uniform(
    knots: Sequence[float],
    degree: int,
    control_points: np.ndarray,
    weights: Optional[Sequence[float]] = None,
) -> Tuple[List[float], np.ndarray, Optional[np.ndarray]]:
    """Split every element in two by inserting the midpoint of each knot span once."""
    k = validate_knots(knots)
    p = validate_degree(degree)
    bounds = element_boundaries(k, p)
    mids = (bounds[:-1] + bounds[1:]) / 2.0
    cp, w = control_points, weights
    for m in mids:
        k, cp, w = insert_knot(k, p, cp, float(m), 1, w)
    return k, np.asarray(cp, dtype=float), w


# Surfaces

def insert_knot_surface(
    knots_u: Sequence[float],
    knots_v: Sequence[float],
    degree_u: int,
    degree_v: int,
    net: np.ndarray,
    weights: np.ndarray,
    direction: str,
    value: float,
    times: int = 1,
) -> Tuple[List[float], List[float], np.ndarray, np.ndarray]:
    """Insert a knot in u (adds a row) or v (adds a column) of a NURBS surface."""
    ku, kv = validate_knots(knots_u), validate_knots(knots_v)
    du, dv = validate_degree(degree_u), validate_degree(degree_v)
    net = validate_surface_net(net)
    rows, cols, _ = net.shape
    w = validate_surface_weights(weights, rows, cols)
    validate_bspline_params(ku, du, rows)
    validate_bspline_params(kv, dv, cols)
    if direction not in ("u", "v"):
        raise ValueError(f"Direction must be 'u' or 'v', got {direction!r}.")

    Pw = _to_homogeneous(net, w)  # (rows, cols, 4)
    if direction == "u":
        new_cols = [_insert_homogeneous(ku, du, Pw[:, j, :], value, times) for j in range(cols)]
        ku = new_cols[0][0]
        Pw = np.stack([c[1] for c in new_cols], axis=1)
    else:
        new_rows = [_insert_homogeneous(kv, dv, Pw[i, :, :], value, times) for i in range(rows)]
        kv = new_rows[0][0]
        Pw = np.stack([r[1] for r in new_rows], axis=0)
    new_net, new_w = _from_homogeneous(Pw)
    return ku, kv, new_net, new_w


def refine_surface_uniform(
    knots_u: Sequence[float],
    knots_v: Sequence[float],
    degree_u: int,
    degree_v: int,
    net: np.ndarray,
    weights: np.ndarray,
    direction: str = "both",
) -> Tuple[List[float], List[float], np.ndarray, np.ndarray]:
    """Split every element in two in u, in v, or in both directions."""
    if direction not in DIRECTIONS:
        raise ValueError(f"Direction must be one of {DIRECTIONS}, got {direction!r}.")
    ku, kv = validate_knots(knots_u), validate_knots(knots_v)
    net, w = np.asarray(net, dtype=float), np.asarray(weights, dtype=float)
    for d in (("u", "v") if direction == "both" else (direction,)):
        knots, deg = (ku, degree_u) if d == "u" else (kv, degree_v)
        b = element_boundaries(knots, deg)
        for m in (b[:-1] + b[1:]) / 2.0:
            ku, kv, net, w = insert_knot_surface(ku, kv, degree_u, degree_v, net, w, d, float(m))
    return ku, kv, net, w
