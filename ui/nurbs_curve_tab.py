"""NURBS Curve tab for editing control points with weights and displaying 2D/3D NURBS curves."""

from __future__ import annotations

import math
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QLineEdit,
    QPushButton, QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox,
    QSplitter, QCheckBox, QSpinBox,
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

from core.nurbs import nurbs_curve, nurbs_circle
from core.bspline import bspline_curve
from core.validation import validate_knots, validate_bspline_params


class NurbsCurveTab(QWidget):
    """Tab widget for NURBS curves with rational weights."""

    def __init__(self, status_callback=None):
        super().__init__()
        self._status = status_callback or (lambda msg: None)
        self._init_ui()

    # Build the user interface
    def _init_ui(self):
        main = QHBoxLayout(self)

        left = QVBoxLayout()

        # B-Spline parameters group
        param_group = QGroupBox("NURBS Parameters")
        param_layout = QVBoxLayout(param_group)

        deg_row = QHBoxLayout()
        deg_row.addWidget(QLabel("&Degree:"))
        self._deg_spin = QSpinBox()
        self._deg_spin.setRange(1, 15)
        self._deg_spin.setValue(2)
        deg_row.addWidget(self._deg_spin)
        param_layout.addLayout(deg_row)

        param_layout.addWidget(QLabel("Knot Vector (comma or space separated):"))
        self._knots_edit = QLineEdit()
        self._knots_edit.setPlaceholderText("e.g. 0, 0, 0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1, 1, 1")
        param_layout.addWidget(self._knots_edit)
        left.addWidget(param_group)

        # Control points table with weights
        cp_group = QGroupBox("Control Points")
        cp_layout = QVBoxLayout(cp_group)

        btn_row = QHBoxLayout()
        add_btn = QPushButton("&Add Point")
        add_btn.clicked.connect(self._add_point)
        btn_row.addWidget(add_btn)
        rm_btn = QPushButton("&Remove Selected")
        rm_btn.clicked.connect(self._remove_selected)
        btn_row.addWidget(rm_btn)
        clr_btn = QPushButton("&Clear")
        clr_btn.clicked.connect(self._clear_points)
        btn_row.addWidget(clr_btn)
        cp_layout.addLayout(btn_row)

        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["X", "Y", "Z", "W"])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        cp_layout.addWidget(self._table)

        example_row = QHBoxLayout()
        circle_btn = QPushButton("&Load Circle Example")
        circle_btn.clicked.connect(self._load_circle)
        example_row.addWidget(circle_btn)
        cp_layout.addLayout(example_row)

        left.addWidget(cp_group)

        # Options
        opt_group = QGroupBox("Display Options")
        opt_layout = QVBoxLayout(opt_group)

        self._show_bspline_chk = QCheckBox("Show B-spline (all weights = 1)")
        opt_layout.addWidget(self._show_bspline_chk)
        left.addWidget(opt_group)

        draw_btn = QPushButton("&Draw Curve")
        draw_btn.clicked.connect(self._draw)
        left.addWidget(draw_btn)
        left.addStretch()

        left_widget = QWidget()
        left_widget.setLayout(left)

        # Plot view on the right
        right = QVBoxLayout()
        self._figure = Figure()
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._toolbar = NavigationToolbar2QT(self._canvas, self)
        right.addWidget(self._toolbar)
        right.addWidget(self._canvas)

        right_widget = QWidget()
        right_widget.setLayout(right)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        main.addWidget(splitter)

    # Parse knots from text
    def parse_knots(self):
        text = self._knots_edit.text().strip().replace(",", " ")
        if not text:
            raise ValueError("Knot vector is empty.")
        tokens = text.split()
        try:
            raw = [float(t) for t in tokens]
        except ValueError:
            raise ValueError("All knot values must be numbers.")
        return validate_knots(raw)

    # Add an empty row to the table
    def _add_point(self):
        row = self._table.rowCount()
        self._table.insertRow(row)
        self._table.setItem(row, 0, QTableWidgetItem("0.0"))
        self._table.setItem(row, 1, QTableWidgetItem("0.0"))
        self._table.setItem(row, 2, QTableWidgetItem(""))
        self._table.setItem(row, 3, QTableWidgetItem("1.0"))

    # Remove selected rows
    def _remove_selected(self):
        rows = sorted(set(idx.row() for idx in self._table.selectedIndexes()), reverse=True)
        for r in rows:
            self._table.removeRow(r)

    # Clear all rows
    def _clear_points(self):
        self._table.setRowCount(0)

    # Load the 9-point circle preset
    def _load_circle(self):
        cp, w, knots, degree = nurbs_circle()
        self._deg_spin.setValue(degree)
        self._knots_edit.setText(", ".join(f"{k:g}" for k in knots))
        self._table.setRowCount(0)
        for i in range(cp.shape[0]):
            row = self._table.rowCount()
            self._table.insertRow(row)
            self._table.setItem(row, 0, QTableWidgetItem(str(cp[i, 0])))
            self._table.setItem(row, 1, QTableWidgetItem(str(cp[i, 1])))
            self._table.setItem(row, 2, QTableWidgetItem(""))
            self._table.setItem(row, 3, QTableWidgetItem(str(w[i])))
        self._status("Loaded 9-point NURBS circle example")

    # Read control points and weights from the table
    def _get_table_data(self):
        rows = self._table.rowCount()
        if rows < 2:
            raise ValueError("Need at least 2 control points.")
        pts = []
        weights = []
        has_z = False
        for r in range(rows):
            try:
                x = float(self._table.item(r, 0).text())
                y = float(self._table.item(r, 1).text())
            except (ValueError, AttributeError):
                raise ValueError(f"Invalid X or Y at row {r + 1}.")
            z_item = self._table.item(r, 2)
            z_text = z_item.text().strip() if z_item else ""
            if z_text:
                try:
                    z = float(z_text)
                except ValueError:
                    raise ValueError(f"Invalid Z at row {r + 1}.")
                has_z = True
                pts.append([x, y, z])
            else:
                pts.append([x, y, 0.0])

            w_item = self._table.item(r, 3)
            w_text = w_item.text().strip() if w_item else "1.0"
            if not w_text:
                w_text = "1.0"
            try:
                wv = float(w_text)
            except ValueError:
                raise ValueError(f"Invalid weight at row {r + 1}.")
            weights.append(wv)

        arr = np.array(pts)
        if not has_z:
            arr = arr[:, :2]
        return arr, np.array(weights)

    # get_data for JSON save
    def get_data(self) -> dict:
        knots = self.parse_knots()
        degree = self._deg_spin.value()
        cp, w = self._get_table_data()
        validate_bspline_params(knots, degree, cp.shape[0])
        return {
            "type": "nurbs_curve",
            "degree": degree,
            "knots": knots,
            "control_points": cp.tolist(),
            "weights": w.tolist(),
        }

    # set_data for JSON load
    def set_data(self, data: dict):
        if "degree" in data:
            self._deg_spin.setValue(int(data["degree"]))
        if "knots" in data:
            self._knots_edit.setText(", ".join(f"{k:g}" for k in data["knots"]))
        if "control_points" in data:
            cp = np.atleast_2d(np.array(data["control_points"], dtype=float))
            w = data.get("weights")
            if w is None:
                w = [1.0] * cp.shape[0]
            w = list(w)
            is_3d = cp.shape[1] == 3
            self._table.setRowCount(0)
            for i in range(cp.shape[0]):
                row = self._table.rowCount()
                self._table.insertRow(row)
                self._table.setItem(row, 0, QTableWidgetItem(str(cp[i, 0])))
                self._table.setItem(row, 1, QTableWidgetItem(str(cp[i, 1])))
                if is_3d:
                    self._table.setItem(row, 2, QTableWidgetItem(str(cp[i, 2])))
                else:
                    self._table.setItem(row, 2, QTableWidgetItem(""))
                self._table.setItem(row, 3, QTableWidgetItem(str(w[i])))

    # Draw the NURBS curve
    def _draw(self):
        try:
            knots = self.parse_knots()
            degree = self._deg_spin.value()
            cp, w = self._get_table_data()
            validate_bspline_params(knots, degree, cp.shape[0])
            curve = nurbs_curve(knots, degree, cp, w, n_eval=400)
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))
            return

        self._figure.clear()
        dim = cp.shape[1]

        if dim == 3:
            ax = self._figure.add_subplot(111, projection="3d")
            ax.plot(cp[:, 0], cp[:, 1], cp[:, 2],
                    color="blue", linestyle="--", linewidth=1.0,
                    label="Control polygon")
            ax.plot(cp[:, 0], cp[:, 1], cp[:, 2],
                    linestyle="None", marker="o", color="blue", markersize=5,
                    label="Control points")
            ax.plot(curve[:, 0], curve[:, 1], curve[:, 2],
                    color="red", linestyle="-", linewidth=1.5,
                    label="NURBS curve")
            ax.set_zlabel("Z")
        else:
            ax = self._figure.add_subplot(111)
            ax.plot(cp[:, 0], cp[:, 1],
                    color="blue", linestyle="--", linewidth=1.0,
                    label="Control polygon")
            ax.plot(cp[:, 0], cp[:, 1],
                    linestyle="None", marker="o", color="blue", markersize=5,
                    label="Control points")
            ax.plot(curve[:, 0], curve[:, 1],
                    color="red", linestyle="-", linewidth=1.5,
                    label="NURBS curve")

            # Overlay unweighted B-spline if checkbox is checked
            if self._show_bspline_chk.isChecked():
                try:
                    bsp = bspline_curve(knots, degree, cp, n_eval=400)
                    ax.plot(bsp[:, 0], bsp[:, 1],
                            color="green", linestyle=":", linewidth=1.5,
                            label="B-spline (w=1)")
                except Exception:
                    pass

            ax.set_aspect("equal", adjustable="box")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f"NURBS Curve (degree {degree}, {cp.shape[0]} pts)")
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend()
        ax.grid(True)
        self._canvas.draw()

        # Compute circle error if it looks like a circle
        status_msg = f"NURBS curve drawn: degree {degree}, {cp.shape[0]} points"
        try:
            # Check if all points roughly equidistant from center (circle test)
            cx = np.mean(curve[:, 0])
            cy = np.mean(curve[:, 1])
            radii = np.sqrt((curve[:, 0] - cx) ** 2 + (curve[:, 1] - cy) ** 2)
            mean_r = np.mean(radii)
            max_err = np.max(np.abs(radii - mean_r))
            if max_err < 1e-6:
                status_msg += f"  |  Circle: max radius error = {max_err:.2e}"
        except Exception:
            pass
        self._status(status_msg)
