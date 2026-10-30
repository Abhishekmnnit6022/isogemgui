"""Tests for JSON save/load serialization."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from core import io_json


def test_bezier_curve_roundtrip(tmp_path: Path):
    """Save and load a Bezier curve; verify identical data."""
    file_path = tmp_path / "curve.json"
    orig_data = {
        "type": "bezier_curve",
        "degree": 3,
        "control_points": [[1.0, 1.0], [2.0, 3.0], [4.0, 3.0], [3.0, 1.0]],
    }
    io_json.save(file_path, orig_data)
    loaded = io_json.load(file_path)

    assert loaded["type"] == "bezier_curve"
    assert loaded["degree"] == 3
    assert np.allclose(loaded["control_points"], orig_data["control_points"])
    assert loaded["weights"] == [1.0, 1.0, 1.0, 1.0]


def test_bezier_surface_roundtrip(tmp_path: Path):
    """Save and load a Bezier surface; verify identical data."""
    file_path = tmp_path / "surface.json"
    orig_net = [
        [[-0.5, 2.0, 1.0], [-2.0, 1.0, -1.0], [0.0, 0.0, 2.0]],
        [[1.0, 2.0, 0.0], [1.0, 8.0, -8.5], [1.0, -1.0, 2.0]],
        [[2.0, 2.0, 0.5], [2.0, 1.0, 1.0], [2.0, 1.0, 2.0]],
    ]
    orig_data = {
        "type": "bezier_surface",
        "control_points": orig_net,
    }
    io_json.save(file_path, orig_data)
    loaded = io_json.load(file_path)

    assert loaded["type"] == "bezier_surface"
    assert loaded["degree"] == [2, 2]
    assert np.allclose(loaded["control_points"], orig_net)
    assert len(loaded["weights"]) == 9


def test_bspline_roundtrip(tmp_path: Path):
    """Save and load a B-spline curve; verify identical data."""
    file_path = tmp_path / "bspline.json"
    knots = [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0]
    cp = [[0.0, 0.0], [1.0, 2.0], [3.0, 4.0], [5.0, 2.0], [6.0, 0.0]]
    orig_data = {
        "type": "bspline_curve",
        "degree": 2,
        "knots": knots,
        "control_points": cp,
    }
    io_json.save(file_path, orig_data)
    loaded = io_json.load(file_path)

    assert loaded["type"] == "bspline_curve"
    assert loaded["degree"] == 2
    assert np.allclose(loaded["knots"], knots)
    assert np.allclose(loaded["control_points"], cp)
    assert loaded["weights"] == [1.0] * 5


def test_invalid_type_raises(tmp_path: Path):
    """Unrecognized geometry types must raise ValueError."""
    file_path = tmp_path / "invalid.json"
    with pytest.raises(ValueError, match="Unknown geometry type"):
        io_json.save(file_path, {"type": "unknown_shape"})


def test_nurbs_curve_roundtrip(tmp_path: Path):
    """Save and load a NURBS curve; verify identical data and weights."""
    file_path = tmp_path / "nurbs_curve.json"
    knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    cp = [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    weights = [1.0, 0.7071067811865476, 1.0]
    orig_data = {
        "type": "nurbs_curve",
        "degree": 2,
        "knots": knots,
        "control_points": cp,
        "weights": weights,
    }
    io_json.save(file_path, orig_data)
    loaded = io_json.load(file_path)

    assert loaded["type"] == "nurbs_curve"
    assert loaded["degree"] == 2
    assert np.allclose(loaded["knots"], knots)
    assert np.allclose(loaded["control_points"], cp)
    assert np.allclose(loaded["weights"], weights)


def test_nurbs_surface_roundtrip(tmp_path: Path):
    """Save and load a NURBS surface; verify identical data and 2D weights."""
    file_path = tmp_path / "nurbs_surface.json"
    knots_u = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    knots_v = [0.0, 0.0, 1.0, 1.0]
    net = [
        [[0.0, 0.0, 0.0], [0.0, 1.0, 1.0]],
        [[1.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
        [[2.0, 0.0, 0.0], [2.0, 1.0, 1.0]],
    ]
    weights = [
        [1.0, 1.0],
        [2.0, 2.0],
        [1.0, 1.0],
    ]
    orig_data = {
        "type": "nurbs_surface",
        "degree_u": 2,
        "degree_v": 1,
        "knots_u": knots_u,
        "knots_v": knots_v,
        "control_points": net,
        "weights": weights,
    }
    io_json.save(file_path, orig_data)
    loaded = io_json.load(file_path)

    assert loaded["type"] == "nurbs_surface"
    assert loaded["degree_u"] == 2
    assert loaded["degree_v"] == 1
    assert np.allclose(loaded["knots_u"], knots_u)
    assert np.allclose(loaded["knots_v"], knots_v)
    assert np.allclose(loaded["control_points"], net)
    assert np.allclose(loaded["weights"], weights)


def test_nurbs_examples_load():
    """Verify that all example JSON files in examples/ load cleanly."""
    for example_name in ["circle.json", "quarter_circle.json", "cylinder.json"]:
        path = Path("examples") / example_name
        data = io_json.load(path)
        assert data["type"] in ("nurbs_curve", "nurbs_surface")
        assert "control_points" in data
        assert "weights" in data

