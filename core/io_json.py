"""Save and load geometry objects to and from JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .validation import (
    validate_bspline_params,
    validate_control_points,
    validate_degree,
    validate_knots,
    validate_surface_net,
    validate_surface_weights,
    validate_weights,
)

VALID_TYPES = {"bezier_curve", "bezier_surface", "bspline_curve",
               "nurbs_curve", "nurbs_surface"}


# Creates a list of default 1.0 weights
def _make_weights(n: int) -> list[float]:
    """Returns a list of 1.0 weights, one for each control point."""
    return [1.0] * n


# Save geometry dictionary to JSON file
def save(path: str | Path, data: Dict[str, Any]) -> None:
    """Writes a geometry dictionary to a JSON file on disk. Converts numpy arrays to lists."""
    data = dict(data)
    geom_type = data.get("type")
    if geom_type not in VALID_TYPES:
        raise ValueError(f"Unknown geometry type '{geom_type}'. Must be one of {VALID_TYPES}.")

    # Convert numpy arrays to plain lists for JSON serialization
    for key in ("control_points", "knots", "weights",
                "knots_u", "knots_v"):
        if key in data and isinstance(data[key], np.ndarray):
            data[key] = data[key].tolist()

    if "weights" not in data or data["weights"] is None:
        if geom_type == "bezier_surface":
            net = np.asarray(data["control_points"])
            data["weights"] = _make_weights(int(net.size // 3))
        elif geom_type == "nurbs_surface":
            pass  # weights are mandatory for NURBS surface
        else:
            data["weights"] = _make_weights(len(data["control_points"]))

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# Load geometry from JSON file and validate
def load(path: str | Path) -> Dict[str, Any]:
    """Reads a JSON file, verifies its geometry properties, and returns verified numpy arrays."""
    with open(path) as f:
        data = json.load(f)

    geom_type = data.get("type")
    if geom_type not in VALID_TYPES:
        raise ValueError(f"Unknown geometry type '{geom_type}' in file.")

    if geom_type == "bezier_curve":
        cp = validate_control_points(data["control_points"])
        degree = cp.shape[0] - 1
        data["control_points"] = cp
        data["degree"] = degree

    elif geom_type == "bezier_surface":
        net = validate_surface_net(data["control_points"])
        data["control_points"] = net
        data["degree"] = [int(net.shape[0] - 1), int(net.shape[1] - 1)]

    elif geom_type == "bspline_curve":
        cp = validate_control_points(data["control_points"])
        knots = validate_knots(data["knots"])
        degree = validate_degree(data["degree"])
        validate_bspline_params(knots, degree, cp.shape[0])
        data["control_points"] = cp
        data["knots"] = knots
        data["degree"] = degree

    elif geom_type == "nurbs_curve":
        cp = validate_control_points(data["control_points"])
        knots = validate_knots(data["knots"])
        degree = validate_degree(data["degree"])
        validate_bspline_params(knots, degree, cp.shape[0])
        w = validate_weights(data["weights"], cp.shape[0])
        data["control_points"] = cp
        data["knots"] = knots
        data["degree"] = degree
        data["weights"] = w.tolist()

    elif geom_type == "nurbs_surface":
        net = validate_surface_net(data["control_points"])
        ku = validate_knots(data["knots_u"])
        kv = validate_knots(data["knots_v"])
        du = validate_degree(data["degree_u"])
        dv = validate_degree(data["degree_v"])
        rows, cols, _ = net.shape
        validate_bspline_params(ku, du, rows)
        validate_bspline_params(kv, dv, cols)
        w = validate_surface_weights(data["weights"], rows, cols)
        data["control_points"] = net
        data["knots_u"] = ku
        data["knots_v"] = kv
        data["degree_u"] = du
        data["degree_v"] = dv
        data["weights"] = w.tolist()

    return data
