"""Reusable widgets and plot helpers for knot display and knot insertion."""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
from PyQt5.QtWidgets import (
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPushButton, QSpinBox, QVBoxLayout,
)

MAX_KNOT_LABELS = 30  # above this many knots, markers are drawn without text labels


class CurveRefinementBox(QGroupBox):
    """'Refinement (Knot Insertion)' group: insert one knot value or split every element.

    on_insert() and on_split() are called when the buttons are pressed; the
    owning tab reads the value with read_insert().
    """

    def __init__(self, on_insert: Callable[[], None], on_split: Callable[[], None]):
        super().__init__("Refinement (Knot Insertion)")
        layout = QVBoxLayout(self)

        row = QHBoxLayout()
        row.addWidget(QLabel("Knot u:"))
        self.u_edit = QLineEdit()
        self.u_edit.setPlaceholderText("e.g. 0.125")
        self.u_edit.setToolTip("Parameter value strictly inside the curve's range")
        row.addWidget(self.u_edit)
        row.addWidget(QLabel("Times:"))
        self.times_spin = QSpinBox()
        self.times_spin.setRange(1, 15)
        self.times_spin.setToolTip("How many copies of the knot to insert (at most the degree in total)")
        row.addWidget(self.times_spin)
        self.insert_btn = QPushButton("Insert Knot")
        self.insert_btn.clicked.connect(lambda: on_insert())
        row.addWidget(self.insert_btn)
        layout.addLayout(row)

        self.split_btn = QPushButton("Split All Elements (insert every span midpoint)")
        self.split_btn.clicked.connect(lambda: on_split())
        layout.addWidget(self.split_btn)

        hint = QLabel("Adds control points; the curve keeps exactly the same shape.")
        hint.setWordWrap(True)
        layout.addWidget(hint)

    def read_insert(self) -> Tuple[float, int]:
        """The typed knot value and number of insertions; ValueError if the value is not a number."""
        text = self.u_edit.text().strip()
        if not text:
            raise ValueError("Enter the knot value u to insert.")
        try:
            u = float(text)
        except ValueError:
            raise ValueError(f"Knot value '{text}' is not a number.") from None
        return u, self.times_spin.value()


def draw_knot_markers(ax, points: np.ndarray, values: np.ndarray, three_d: bool) -> None:
    """Black square markers at the curve points that belong to each distinct knot.

    Each marker is labelled with its u value (unless there are very many knots).
    """
    pts = np.asarray(points, dtype=float)
    if three_d:
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], linestyle="None", marker="s",
                color="black", markersize=6, label="Knots (element ends)")
    else:
        ax.plot(pts[:, 0], pts[:, 1], linestyle="None", marker="s",
                color="black", markersize=6, label="Knots (element ends)")
    if len(values) > MAX_KNOT_LABELS:
        return
    seen = set()
    for p, u in zip(pts, values):
        key = tuple(np.round(p, 9))
        text = f"u={u:.4g}"
        if key in seen:      # closed curve: first and last knot meet; label once
            continue
        seen.add(key)
        if three_d:
            ax.text(p[0], p[1], p[2], "  " + text, fontsize=7, color="black")
        else:
            ax.annotate(text, (p[0], p[1]), textcoords="offset points", xytext=(5, 5),
                        fontsize=7, color="black")


# '1 element' / 'N elements' for status messages
def element_count_text(n_elements: int) -> str:
    return f"{n_elements} element{'s' if n_elements != 1 else ''} (knot spans)"


def draw_polylines(ax, lines, three_d: bool, **style) -> None:
    """Draw several polylines as one plot artist (NaN-separated), so they share one legend entry."""
    if not lines:
        return
    gap = np.full((1, 3), np.nan)
    pts = np.concatenate([np.vstack([ln, gap]) for ln in lines])
    if three_d:
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], **style)
    else:
        ax.plot(pts[:, 0], pts[:, 1], **style)


def scrollable(panel):
    """Wrap a left-hand control panel in a vertical scroll area so its tables never get squashed."""
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QScrollArea
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    area.setWidget(panel)
    area.setMinimumWidth(panel.sizeHint().width() + 20)
    return area
