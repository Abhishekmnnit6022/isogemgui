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
