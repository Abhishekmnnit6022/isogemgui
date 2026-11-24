"""NURBS Curve tab for editing control points with weights and displaying 2D/3D NURBS curves."""

from __future__ import annotations

from typing import Optional

import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QLineEdit,
    QPushButton, QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox,
    QSplitter, QCheckBox, QSpinBox, QFileDialog,
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

from core.nurbs import (
    nurbs_curve, nurbs_circle, t_profile, rational_basis_matrix,
    sample_parameters, fit_circle,
)
from core.bspline import bspline_curve, basis_matrix, element_boundaries, valid_domain
from core.refinement import insert_knot, refine_uniform
from core.validation import validate_bspline_params
from ui.common import check_spin_value, format_knots, format_number, parse_knots_text
from ui.widgets import CurveRefinementBox, draw_knot_markers, element_count_text, scrollable


class NurbsCurveTab(QWidget):
    """Tab widget for NURBS curves with rational weights."""

    def __init__(self, status_callback=None):
        super().__init__()
        self._status = status_callback or (lambda msg: None)
        self._has_curve = False   # True while the plot shows the curve (not the basis)
        self._last_status = ""
        self._init_ui()

    # Build the user interface
    def _init_ui(self):
        main = QHBoxLayout(self)

        left = QVBoxLayout()

        # NURBS parameters group
        param_group = QGroupBox("NURBS Parameters")
        param_layout = QVBoxLayout(param_group)

        deg_row = QHBoxLayout()
        deg_row.addWidget(QLabel("Degree:"))
        self._deg_spin = QSpinBox()
        self._deg_spin.setRange(1, 15)
        self._deg_spin.setValue(2)
        deg_row.addWidget(self._deg_spin)
        param_layout.addLayout(deg_row)

        param_layout.addWidget(QLabel("Knot Vector (comma or space separated):"))
        self._knots_edit = QLineEdit()
        self._knots_edit.setPlaceholderText("e.g. 0, 0, 0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1, 1, 1")
        param_layout.addWidget(self._knots_edit)

        basis_row = QHBoxLayout()
        basis_btn = QPushButton("Plot Rational &Basis")
        basis_btn.setToolTip("Plot the rational basis functions R_i(u) = N_i(u) w_i / sum_j N_j(u) w_j")
        basis_btn.clicked.connect(self._plot_rational_basis)
        basis_row.addWidget(basis_btn)
        export_btn = QPushButton("Export Basis to Excel")
        export_btn.setToolTip("Save the rational basis matrix R_i(u) to an .xlsx file")
        export_btn.clicked.connect(lambda: self.export_rational_basis_excel())
        basis_row.addWidget(export_btn)
        param_layout.addLayout(basis_row)
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
        self._table.setMinimumHeight(150)
        self._table.setHorizontalHeaderLabels(["X", "Y", "Z", "W"])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        cp_layout.addWidget(self._table)

        example_row = QHBoxLayout()
        circle_btn = QPushButton("&Load Circle Example")
        circle_btn.clicked.connect(self._load_circle)
        example_row.addWidget(circle_btn)
        t_btn = QPushButton("Load &T-Profile Example")
        t_btn.setToolTip("Closed T cross-section with exact rounded inner corners")
        t_btn.clicked.connect(self._load_t_profile)
        example_row.addWidget(t_btn)
        cp_layout.addLayout(example_row)

        left.addWidget(cp_group)

        self._refine_box = CurveRefinementBox(self._insert_knot, self._split_elements)
        left.addWidget(self._refine_box)

        # Options
        opt_group = QGroupBox("Display Options")
        opt_layout = QVBoxLayout(opt_group)

        self._show_bspline_chk = QCheckBox("Show B-spline (all weights = 1)")
        self._show_bspline_chk.setToolTip(
            "Overlay the non-rational B-spline on the same control points (curve plot), "
            "or the non-rational basis N_i (basis plot)"
        )
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
        self._toolbar.addSeparator()
        self._show_knots_chk = QCheckBox("Knots")
        self._show_knots_chk.setToolTip("Mark the knots (element boundaries) on the curve")
        self._show_knots_chk.stateChanged.connect(self._on_display_option_changed)
        self._toolbar.addWidget(self._show_knots_chk)
        right.addWidget(self._toolbar)
        right.addWidget(self._canvas)

        right_widget = QWidget()
        right_widget.setLayout(right)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(scrollable(left_widget))
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        main.addWidget(splitter)

    # Parse knots from text
    def parse_knots(self):
        return parse_knots_text(self._knots_edit.text())

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

    # Reset the whole tab (used by File -> New)
    def reset(self):
        self._table.setRowCount(0)
        self._knots_edit.clear()
        self._deg_spin.setValue(2)
        self._show_bspline_chk.setChecked(False)
        self._has_curve = False
        self._figure.clear()
        self._canvas.draw()

    # Fill the table, degree and knots in one go
    def _fill(self, cp: np.ndarray, weights, knots, degree: int):
        cp = np.atleast_2d(np.asarray(cp, dtype=float))
        weights = list(weights)
        is_3d = cp.shape[1] == 3
        degree = check_spin_value(self._deg_spin, degree, "Degree")
        self._deg_spin.setValue(degree)
        self._knots_edit.setText(format_knots(knots))
        self._table.setRowCount(0)
        for i in range(cp.shape[0]):
            row = self._table.rowCount()
            self._table.insertRow(row)
            self._table.setItem(row, 0, QTableWidgetItem(format_number(cp[i, 0])))
            self._table.setItem(row, 1, QTableWidgetItem(format_number(cp[i, 1])))
            z_text = format_number(cp[i, 2]) if is_3d else ""
            self._table.setItem(row, 2, QTableWidgetItem(z_text))
            self._table.setItem(row, 3, QTableWidgetItem(format_number(weights[i])))

    # Load the 9-point circle preset
    def _load_circle(self):
        cp, w, knots, degree = nurbs_circle()
        self._fill(cp, w, knots, degree)
        self._status("Loaded 9-point NURBS circle example")

    # Load the closed T cross-section preset
    def _load_t_profile(self):
        cp, w, knots, degree = t_profile()
        self._fill(cp, w, knots, degree)
        self._status(f"Loaded T-profile example: {cp.shape[0]} control points, degree {degree}")

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
            if wv <= 0.0:
                raise ValueError(f"Weight at row {r + 1} must be strictly positive (got {wv}).")
            weights.append(wv)

        arr = np.array(pts)
        if not has_z:
            arr = arr[:, :2]
        return arr, np.array(weights)

    # Read and check everything needed to evaluate the curve
    def _read_inputs(self):
        knots = self.parse_knots()
        degree = self._deg_spin.value()
        cp, w = self._get_table_data()
        validate_bspline_params(knots, degree, cp.shape[0])
        return knots, degree, cp, w

    # get_data for JSON save
    def get_data(self) -> dict:
        knots, degree, cp, w = self._read_inputs()
        return {
            "type": "nurbs_curve",
            "degree": degree,
            "knots": knots,
            "control_points": cp.tolist(),
            "weights": w.tolist(),
        }

    # set_data for JSON load
    def set_data(self, data: dict):
        cp = np.atleast_2d(np.array(data.get("control_points", []), dtype=float))
        if cp.size == 0:
            return
        w = data.get("weights")
        if w is None:
            w = [1.0] * cp.shape[0]
        self._fill(cp, w, data.get("knots", []), data.get("degree", self._deg_spin.value()))

    # Plot the rational basis functions R_i(u)
    def _plot_rational_basis(self):
        try:
            knots, degree, _, w = self._read_inputs()
            u, R = rational_basis_matrix(knots, degree, w, n_eval=400)
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))
            return

        self._figure.clear()
        ax = self._figure.add_subplot(111)
        for i in range(R.shape[0]):
            ax.plot(u, R[i], linewidth=1.5, label=f"R_{i},{degree}  (w={w[i]:.4g})")

        if self._show_bspline_chk.isChecked():
            N, t = basis_matrix(knots, degree, n_eval=400)
            for i in range(N.shape[0]):
                ax.plot(t, N[i], color="gray", linestyle=":", linewidth=1.0,
                        label="N_i (w = 1)" if i == 0 else None)

        ax.set_xlabel("Parameter u")
        ax.set_ylabel("Basis value")
        self._has_curve = False
        ax.set_title(f"NURBS Rational Basis Functions (degree {degree}, {R.shape[0]} functions)")
        ax.grid(True)
        if R.shape[0] <= 12:
            ax.legend(fontsize="small")
        self._canvas.draw()
        self._status(f"Rational basis plotted: {R.shape[0]} functions, degree {degree}  |  "
                     f"max |sum R_i - 1| = {np.max(np.abs(R.sum(axis=0) - 1.0)):.2e}")

    def export_rational_basis_excel(self, filepath: Optional[str] = None):
        """Export R_i(u) over the knot range to an Excel workbook (one column per function)."""
        try:
            knots, degree, _, w = self._read_inputs()
            u, R = rational_basis_matrix(knots, degree, w, n_eval=200)
        except Exception as e:
            QMessageBox.warning(self, "Export Error", f"Cannot compute rational basis matrix:\n{e}")
            return

        if not filepath:
            filepath, _ = QFileDialog.getSaveFileName(
                self, "Export Rational Basis to Excel", "nurbs_rational_basis.xlsx",
                "Excel Files (*.xlsx)"
            )
            if not filepath:
                return

        try:
            import pandas as pd
            col_names = ["u"] + [f"R_{i}_{degree}" for i in range(R.shape[0])]
            df = pd.DataFrame(np.column_stack([u, R.T]), columns=col_names)
            df.to_excel(filepath, index=False)
            self._status(f"Rational basis matrix exported to {filepath}")
            QMessageBox.information(self, "Export Successful",
                                    f"Rational basis matrix saved to:\n{filepath}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to save Excel file:\n{e}")

    # Redraw quietly when a display option changes, but only while the curve is shown
    def _on_display_option_changed(self, *_):
        if self._has_curve:
            self._draw(silent=True)

    # Draw the NURBS curve; silent=True skips the error dialog
    def _draw(self, silent: bool = False):
        try:
            knots, degree, cp, w = self._read_inputs()
            u_vals = sample_parameters(knots, 400, degree)
            curve = nurbs_curve(knots, degree, cp, w, u_values=u_vals)
        except Exception as e:
            if not silent:
                QMessageBox.warning(self, "Error", str(e))
            return

        overlay = None
        if self._show_bspline_chk.isChecked():
            try:
                overlay = bspline_curve(knots, degree, cp, n_eval=400)
            except Exception:
                overlay = None

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
            if overlay is not None:
                ax.plot(overlay[:, 0], overlay[:, 1], overlay[:, 2],
                        color="green", linestyle=":", linewidth=1.5,
                        label="B-spline (w=1)")
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
            if overlay is not None:
                ax.plot(overlay[:, 0], overlay[:, 1],
                        color="green", linestyle=":", linewidth=1.5,
                        label="B-spline (w=1)")
            ax.set_aspect("equal", adjustable="datalim")

        bounds = element_boundaries(knots, degree)
        if self._show_knots_chk.isChecked():
            knot_pts = nurbs_curve(knots, degree, cp, w, u_values=bounds)
            draw_knot_markers(ax, knot_pts, bounds, dim == 3)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f"NURBS Curve (degree {degree}, {cp.shape[0]} pts)")
        if ax.get_legend_handles_labels()[1]:  # only when something is labelled
            ax.legend()
        ax.grid(True)
        self._canvas.draw()
        self._has_curve = True

        status = self._curve_status(curve, degree, cp.shape[0])
        parts = status.split("  |  ", 1)
        parts[0] += f", {element_count_text(len(bounds) - 1)}"
        self._last_status = "  |  ".join(parts)
        self._status(self._last_status)

    # Swap in the refined curve and report the shape change
    def _apply_refinement(self, old, new_knots, new_cp, new_w, action: str):
        knots, degree, cp, w = old
        u = np.linspace(*valid_domain(knots, degree), 400)
        change = float(np.max(np.abs(nurbs_curve(knots, degree, cp, w, u_values=u)
                                     - nurbs_curve(new_knots, degree, new_cp, new_w, u_values=u))))
        self._fill(new_cp, new_w, new_knots, degree)
        self._draw()
        self._status(f"{action}: {cp.shape[0]} -> {new_cp.shape[0]} control points, "
                     f"max shape change {change:.1e}  |  {self._last_status}")

    # Insert the knot typed in the Refinement box
    def _insert_knot(self):
        try:
            old = self._read_inputs()
            u, times = self._refine_box.read_insert()
            new_knots, new_cp, new_w = insert_knot(old[0], old[1], old[2], u, times, weights=old[3])
        except Exception as e:
            QMessageBox.warning(self, "Insert Knot", str(e))
            return
        self._apply_refinement(old, new_knots, new_cp, new_w, f"Inserted knot u={u:g} ({times}x)")

    # Split every element in two
    def _split_elements(self):
        try:
            old = self._read_inputs()
            new_knots, new_cp, new_w = refine_uniform(old[0], old[1], old[2], old[3])
        except Exception as e:
            QMessageBox.warning(self, "Split All Elements", str(e))
            return
        self._apply_refinement(old, new_knots, new_cp, new_w, "Split all elements")

    # Status text, including the radius error when the curve is a circle or circular arc
    @staticmethod
    def _curve_status(curve: np.ndarray, degree: int, n_points: int) -> str:
        msg = f"NURBS curve drawn: degree {degree}, {n_points} points"
        planar = curve.shape[1] == 2 or np.ptp(curve[:, 2]) < 1e-12
        if not planar:
            return msg
        try:
            cx, cy, r, max_err = fit_circle(curve[:, :2])
        except ValueError:
            return msg
        if max_err <= 1e-9 * max(r, 1.0):
            msg += (f"  |  Circle: centre ({cx:.6g}, {cy:.6g}), radius {r:.6g}, "
                    f"max radius error = {max_err:.2e}")
        return msg
