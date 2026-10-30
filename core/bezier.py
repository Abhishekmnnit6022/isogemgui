"""Bezier curve and surface calculations using Bernstein polynomials."""

from __future__ import annotations

from math import comb
from typing import Optional
import numpy as np

from .validation import validate_control_points, validate_surface_net


# Calculate single Bernstein basis value B_{i,n}(u)
def bernstein(i: int, n: int, u: np.ndarray) -> np.ndarray:
    """Calculates the i-th Bernstein polynomial of degree n for parameter array u."""
    return comb(n, i) * (u ** i) * ((1.0 - u) ** (n - i))


# Calculate 2D or 3D Bezier curve points from control points
def bezier_curve(
    control_points: np.ndarray,
    n_eval: int = 200,
) -> np.ndarray:
    """Calculates points on a Bezier curve of any degree given an array of control points."""
    cp = validate_control_points(control_points)
    n = cp.shape[0] - 1
    dim = cp.shape[1]

    if n < 1:
        raise ValueError("A Bezier curve requires at least 2 control points.")

    u = np.linspace(0.0, 1.0, n_eval)
    curve = np.zeros((n_eval, dim))
    for i in range(n + 1):
        b = bernstein(i, n, u)
        curve += np.outer(b, cp[i])
    return curve


# Calculate 3D Bezier surface grid from a control net
def bezier_surface(
    net: np.ndarray,
    u_eval: int = 30,
    v_eval: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculates X, Y, Z coordinate matrices for a 3D Bezier surface from a rectangular control net."""
    net = validate_surface_net(net)
    n = net.shape[0] - 1
    m = net.shape[1] - 1

    u = np.linspace(0.0, 1.0, u_eval)
    v = np.linspace(0.0, 1.0, v_eval)

    X = np.zeros((u_eval, v_eval))
    Y = np.zeros((u_eval, v_eval))
    Z = np.zeros((u_eval, v_eval))

    for i in range(n + 1):
        bi = bernstein(i, n, u)
        for j in range(m + 1):
            bj = bernstein(j, m, v)
            w = np.outer(bi, bj)
            X += w * net[i, j, 0]
            Y += w * net[i, j, 1]
            Z += w * net[i, j, 2]

    return X, Y, Z
