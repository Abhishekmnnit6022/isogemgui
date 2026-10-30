"""Tests for knot vector generation."""

from __future__ import annotations

import pytest
from core.knots import open_uniform, uniform


def test_open_uniform_clamped():
    """Verify open clamped knot structure."""
    n_cp = 6
    p = 3
    knots = open_uniform(n_cp, p)
    # Total knots = n_cp + p + 1 = 6 + 3 + 1 = 10
    assert len(knots) == 10
    assert knots[:p + 1] == [0.0] * 4
    assert knots[-(p + 1):] == [1.0] * 4
    # Internal knots are non-decreasing
    for i in range(len(knots) - 1):
        assert knots[i] <= knots[i + 1]


def test_open_uniform_insufficient_points():
    """ValueError raised if n_control_points < degree + 1."""
    with pytest.raises(ValueError, match="Need at least"):
        open_uniform(3, 3)


def test_uniform_knots():
    """Uniform knots are sequential integers."""
    n_cp = 4
    p = 2
    knots = uniform(n_cp, p)
    assert len(knots) == 7
    assert knots == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
