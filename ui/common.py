"""Small helpers shared by several UI tabs (knot text formatting and parsing)."""

from __future__ import annotations

from typing import Iterable, List

from core.validation import validate_knots


def format_number(value: float) -> str:
    """Shortest text that converts back to exactly the same float.

    Uses repr() so no precision is lost (1/3 -> '0.3333333333333333'),
    and drops a trailing '.0' so whole numbers stay short ('2' not '2.0').
    """
    text = repr(float(value))
    if text.endswith(".0"):
        text = text[:-2]
    if text == "-0":
        text = "0"
    return text


def format_knots(knots: Iterable[float]) -> str:
    """Join knot values into the comma-separated text shown in the knot boxes."""
    return ", ".join(format_number(k) for k in knots)


def parse_knots_text(text: str, label: str = "Knot vector") -> List[float]:
    """Parse comma- or space-separated knot text into a validated list of floats."""
    text = text.strip().replace(",", " ")
    if not text:
        raise ValueError(f"{label} is empty. Please enter knots or generate them.")
    tokens = text.split()
    try:
        raw = [float(t) for t in tokens]
    except ValueError:
        raise ValueError(f"{label}: all knot values must be numbers.")
    return validate_knots(raw)
