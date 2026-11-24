"""Small helpers shared by several UI tabs (knot text, table cells, spin-box ranges)."""

from __future__ import annotations

from typing import Iterable, List, Optional

from core.validation import validate_knots


def format_number(value: float) -> str:
    """Shortest text that gives back exactly the same float ('2' not '2.0')."""
    text = repr(float(value))
    if text.endswith(".0"):
        text = text[:-2]
    if text == "-0":
        text = "0"
    return text


def format_knots(knots: Iterable[float]) -> str:
    """Join knot values into the comma-separated text shown in the knot boxes."""
    return ", ".join(format_number(k) for k in knots)


def read_cell_float(table, row: int, col: int, name: str,
                    optional: bool = False) -> Optional[float]:
    """Number from a table cell, or a clear error naming the row. Empty -> None if optional."""
    item = table.item(row, col)
    text = item.text().strip() if item is not None else ""
    if not text:
        if optional:
            return None
        raise ValueError(f"Row {row + 1}: {name} is empty.")
    try:
        return float(text)
    except ValueError:
        raise ValueError(f"Row {row + 1}: {name} value '{text}' is not a number.") from None


def check_spin_value(spin, value, label: str) -> int:
    """Error if a loaded value is outside the spin box range (instead of clipping it)."""
    try:
        as_int = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be a whole number, got {value!r}.") from None
    if as_int != value:
        raise ValueError(f"{label} must be a whole number, got {value!r}.")
    lo, hi = spin.minimum(), spin.maximum()
    if not lo <= as_int <= hi:
        raise ValueError(f"{label} {as_int} is outside the supported range {lo} to {hi}.")
    return as_int


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
