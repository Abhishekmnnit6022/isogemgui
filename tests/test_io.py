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
    for example_name in ["circle.json", "quarter_circle.json", "cylinder.json",
                         "t_profile.json", "t_shape.json"]:
        path = Path("examples") / example_name
        data = io_json.load(path)
        assert data["type"] in ("nurbs_curve", "nurbs_surface")
        assert "control_points" in data
        assert "weights" in data



def test_every_example_file_loads():
    """Every JSON file shipped in examples/ must load and validate."""
    files = sorted(Path("examples").glob("*.json"))
    assert len(files) >= 10
    for path in files:
        data = io_json.load(path)
        assert data["type"] in io_json.VALID_TYPES


def test_t_shape_example_matches_generator():
    from core.nurbs import nurbs_t_shape
    data = io_json.load(Path("examples") / "t_shape.json")
    net, w, ku, kv, du, dv = nurbs_t_shape()
    np.testing.assert_array_equal(data["control_points"], net)
    np.testing.assert_array_equal(data["weights"], w)
    assert data["knots_u"] == ku and data["knots_v"] == kv
    assert (data["degree_u"], data["degree_v"]) == (du, dv)


def test_knots_roundtrip_with_full_precision(tmp_path: Path):
    knots = [0, 0, 0, 1 / 3, 2 / 3, 1, 1, 1]
    data = {"type": "bspline_curve", "degree": 2, "knots": knots,
            "control_points": [[0, 0], [1, 2], [2, 3], [3, 1], [4, 0]]}
    p = tmp_path / "k.json"
    io_json.save(p, data)
    assert io_json.load(p)["knots"] == [float(k) for k in knots]


@pytest.mark.parametrize("payload, msg", [
    ({"type": "nurbs_curve", "control_points": [[0, 0], [1, 1]], "degree": 1},
     r"missing required field\(s\): knots, weights"),
    ({"type": "nurbs_surface", "control_points": [[[0, 0, 0], [1, 0, 0]], [[0, 1, 0], [1, 1, 0]]]},
     r"missing required field\(s\): knots_u, knots_v, degree_u, degree_v, weights"),
    ({"type": "bspline_curve", "control_points": [[0, 0], [1, 1]], "degree": 1,
      "knots": [0, 0, 1]}, "Invalid 'bspline_curve' data: B-spline constraint violated"),
    ({"type": "nurbs_curve", "control_points": [[0, 0], [1, 1]], "degree": 1,
      "knots": [0, 0, 1, 1], "weights": [1, -1]}, "strictly positive"),
    ({"type": "nurbs_curve", "control_points": [[0, 0], [1, "a"]], "degree": 1,
      "knots": [0, 0, 1, 1], "weights": [1, 1]}, "Invalid 'nurbs_curve' data"),
])
def test_load_reports_clear_errors(tmp_path: Path, payload, msg):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=msg):
        io_json.load(p)


def test_load_rejects_broken_json(tmp_path: Path):
    p = tmp_path / "broken.json"
    p.write_text('{"type": "bezier_curve", "control_points": [[0, 0], [1, 1]')
    with pytest.raises(ValueError, match=r"not valid JSON \(line 1"):
        io_json.load(p)


def test_load_rejects_non_object(tmp_path: Path):
    p = tmp_path / "list.json"
    p.write_text("[1, 2, 3]")
    with pytest.raises(ValueError, match="JSON object at the top level"):
        io_json.load(p)
