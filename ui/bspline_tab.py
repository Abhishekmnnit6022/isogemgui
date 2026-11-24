"""B-Spline tab for knot vector generation, basis function plotting, curve drawing, and Excel export."""

from __future__ import annotations

from typing import List, Optional
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QLineEdit,
    QSpinBox, QPushButton, QTableWidget, QTableWidgetItem, QHeaderView,
    QMessageBox, QSplitter, QFileDialog, QCheckBox, QComboBox, QDoubleSpinBox,
    QDialog, QGridLayout, QToolButton
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

from core.bspline import basis_matrix, bspline_curve, element_boundaries, valid_domain
from core.refinement import insert_knot, refine_uniform
from core.knots import open_uniform, uniform
from core.validation import validate_bspline_params
from ui.common import check_spin_value, format_knots, parse_knots_text, read_cell_float
from ui.widgets import CurveRefinementBox, draw_knot_markers, element_count_text, scrollable


class BsplineTab(QWidget):
    """Tab widget for B-spline curves, basis functions, and Excel export."""

    def __init__(self, status_callback=None):
        super().__init__()
        self._status = status_callback or (lambda msg: None)
        self._last_plot: Optional[str] = None  # "basis" or "curve", whatever is shown now
        self._last_status = ""
        self._init_ui()
        self._load_matlab_preset()

    # Construct the user interface layout
    def _init_ui(self):
        main = QHBoxLayout(self)

        left = QVBoxLayout()

        # Group box for degree and knots
        param_group = QGroupBox("B-Spline Parameters")
        param_layout = QVBoxLayout(param_group)

        deg_row = QHBoxLayout()
        deg_row.addWidget(QLabel("Degree (p):"))
        self._deg_spin = QSpinBox()
        self._deg_spin.setRange(1, 15)
        self._deg_spin.setValue(2)
        deg_row.addWidget(self._deg_spin)
        param_layout.addLayout(deg_row)

        param_layout.addWidget(QLabel("Knot Vector (comma or space separated):"))
        self._knots_edit = QLineEdit()
        self._knots_edit.setPlaceholderText("e.g. 0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 5")
        param_layout.addWidget(self._knots_edit)

        knot_btn_row = QHBoxLayout()
        gen_open_btn = QPushButton("Gen Open Knots")
        gen_open_btn.setToolTip("Generate clamped open knot vector matching degree and control points")
        gen_open_btn.clicked.connect(self._generate_open_knots)
        knot_btn_row.addWidget(gen_open_btn)

        gen_unif_btn = QPushButton("Gen Uniform Knots")
        gen_unif_btn.clicked.connect(self._generate_uniform_knots)
        knot_btn_row.addWidget(gen_unif_btn)
        param_layout.addLayout(knot_btn_row)

        plot_basis_btn = QPushButton("Plot Basis Functions")
        plot_basis_btn.clicked.connect(self._plot_basis)
        param_layout.addWidget(plot_basis_btn)

        left.addWidget(param_group)

        # Group box for control point coordinates
        cp_group = QGroupBox("Control Points (Required for Curve)")
        cp_layout = QVBoxLayout(cp_group)

        pt_row = QHBoxLayout()
        pt_row.addWidget(QLabel("X:"))
        self._x_edit = QLineEdit()
        pt_row.addWidget(self._x_edit)
        pt_row.addWidget(QLabel("Y:"))
        self._y_edit = QLineEdit()
        pt_row.addWidget(self._y_edit)
        pt_row.addWidget(QLabel("Z:"))
        self._z_edit = QLineEdit()
        self._z_edit.setPlaceholderText("(opt)")
        pt_row.addWidget(self._z_edit)
        cp_layout.addLayout(pt_row)

        cp_btn_row = QHBoxLayout()
        add_pt_btn = QPushButton("Add Point")
        add_pt_btn.clicked.connect(self._add_point)
        cp_btn_row.addWidget(add_pt_btn)

        rm_pt_btn = QPushButton("Remove")
        rm_pt_btn.clicked.connect(self._remove_point)
        cp_btn_row.addWidget(rm_pt_btn)

        clr_pt_btn = QPushButton("Clear")
        clr_pt_btn.clicked.connect(self._clear_points)
        cp_btn_row.addWidget(clr_pt_btn)
        cp_layout.addLayout(cp_btn_row)

        self._table = QTableWidget(0, 3)
        self._table.setMinimumHeight(150)
        self._table.setHorizontalHeaderLabels(["X", "Y", "Z"])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        cp_layout.addWidget(self._table)

        preset_row = QHBoxLayout()
        preset_btn = QPushButton("Load Matlab Preset")
        preset_btn.clicked.connect(self._load_matlab_preset)
        preset_row.addWidget(preset_btn)
        cp_layout.addLayout(preset_row)

        left.addWidget(cp_group)

        self._refine_box = CurveRefinementBox(self._insert_knot, self._split_elements)
        left.addWidget(self._refine_box)

        draw_btn = QPushButton("Draw Curve")
        draw_btn.clicked.connect(self._draw_curve)
        left.addWidget(draw_btn)

        excel_btn = QPushButton("Export Basis Matrix to Excel")
        excel_btn.clicked.connect(self.export_basis_excel)
        left.addWidget(excel_btn)

        left.addStretch()

        left_widget = QWidget()
        left_widget.setLayout(left)

        # Style Options Dialog
        self._style_dialog = QDialog(self)
        self._style_dialog.setWindowTitle("Display & Style Options")
        dlg_layout = QVBoxLayout(self._style_dialog)
        dlg_layout.setSpacing(10)

        grid = QGridLayout()
        grid.addWidget(QLabel("Poly Color:"), 0, 0)
        self._poly_color_combo = QComboBox()
        self._poly_color_combo.addItems(["blue", "red", "green", "black", "orange", "magenta", "gray"])
        self._poly_color_combo.currentIndexChanged.connect(self._on_display_option_changed)
        grid.addWidget(self._poly_color_combo, 0, 1)

        grid.addWidget(QLabel("Point Color:"), 0, 2)
        self._point_color_combo = QComboBox()
        self._point_color_combo.addItems(["blue", "red", "green", "black", "orange", "magenta", "cyan", "yellow"])
        self._point_color_combo.currentIndexChanged.connect(self._on_display_option_changed)
        grid.addWidget(self._point_color_combo, 0, 3)

        grid.addWidget(QLabel("Curve Color:"), 1, 0)
        self._curve_color_combo = QComboBox()
        self._curve_color_combo.addItems(["red", "blue", "green", "black", "orange", "purple"])
        self._curve_color_combo.currentIndexChanged.connect(self._on_display_option_changed)
        grid.addWidget(self._curve_color_combo, 1, 1)

        grid.addWidget(QLabel("Poly Style:"), 1, 2)
        self._poly_style_combo = QComboBox()
        self._poly_style_combo.addItem("dashed", "--")
        self._poly_style_combo.addItem("solid", "-")
        self._poly_style_combo.addItem("dotted", ":")
        self._poly_style_combo.currentIndexChanged.connect(self._on_display_option_changed)
        grid.addWidget(self._poly_style_combo, 1, 3)

        grid.addWidget(QLabel("Line Width:"), 2, 0)
        self._line_width_spin = QDoubleSpinBox()
        self._line_width_spin.setRange(0.5, 8.0)
        self._line_width_spin.setSingleStep(0.5)
        self._line_width_spin.setValue(1.5)
        self._line_width_spin.valueChanged.connect(self._on_display_option_changed)
        grid.addWidget(self._line_width_spin, 2, 1)

        grid.addWidget(QLabel("Point Size:"), 2, 2)
        self._point_size_spin = QDoubleSpinBox()
        self._point_size_spin.setRange(2.0, 20.0)
        self._point_size_spin.setSingleStep(1.0)
        self._point_size_spin.setValue(7.0)
        self._point_size_spin.valueChanged.connect(self._on_display_option_changed)
        grid.addWidget(self._point_size_spin, 2, 3)
        dlg_layout.addLayout(grid)

        dlg_btn_row = QHBoxLayout()
        dlg_btn_row.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self._style_dialog.accept)
        dlg_btn_row.addWidget(close_btn)
        dlg_layout.addLayout(dlg_btn_row)

        # Plot area on the right
        right = QVBoxLayout()
        self._figure = Figure()
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._toolbar = NavigationToolbar2QT(self._canvas, self)

        # Add display options to toolbar ribbon
        self._toolbar.addSeparator()

        self._show_poly_chk = QCheckBox("Polygon")
        self._show_poly_chk.setChecked(True)
        self._show_poly_chk.setToolTip("Show/hide control polygon")
        self._show_poly_chk.stateChanged.connect(self._on_display_option_changed)
        self._toolbar.addWidget(self._show_poly_chk)

        self._show_points_chk = QCheckBox("Points")
        self._show_points_chk.setChecked(True)
        self._show_points_chk.setToolTip("Show/hide control points")
        self._show_points_chk.stateChanged.connect(self._on_display_option_changed)
        self._toolbar.addWidget(self._show_points_chk)

        self._show_curve_chk = QCheckBox("Curve")
        self._show_curve_chk.setChecked(True)
        self._show_curve_chk.setToolTip("Show/hide B-spline curve")
        self._show_curve_chk.stateChanged.connect(self._on_display_option_changed)
        self._toolbar.addWidget(self._show_curve_chk)

        self._show_knots_chk = QCheckBox("Knots")
        self._show_knots_chk.setChecked(False)
        self._show_knots_chk.setToolTip("Mark the knots (element boundaries) on the curve")
        self._show_knots_chk.stateChanged.connect(self._on_display_option_changed)
        self._toolbar.addWidget(self._show_knots_chk)

        self._toolbar.addSeparator()
        self._style_btn = QToolButton(self._toolbar)
        self._style_btn.setText("Style Options...")
        self._style_btn.setToolTip("Configure line colors, styles, widths, and point sizes")
        self._style_btn.clicked.connect(self._open_style_dialog)
        self._toolbar.addWidget(self._style_btn)

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

    # Show the style dialog; redraw with the chosen style afterwards
    def _open_style_dialog(self):
        self._style_dialog.exec_()
        self._on_display_option_changed()

    # Display options only redraw the curve, never a basis plot
    def _on_display_option_changed(self, *_):
        if self._last_plot == "curve":
            self._draw_curve(silent=True)

    # Parse and validate the knot values entered in the text box
    def parse_knots(self) -> List[float]:
        return parse_knots_text(self._knots_edit.text())

    # Automatically generate clamped open knot vector matching degree and points
    def _generate_open_knots(self):
        n_cp = self._table.rowCount()
        deg = self._deg_spin.value()
        if n_cp < deg + 1:
            QMessageBox.warning(
                self, "Insufficient Points",
                f"Need at least degree + 1 = {deg + 1} control points to generate open knots. Currently have {n_cp}."
            )
            return
        knots = open_uniform(n_cp, deg)
        self._knots_edit.setText(format_knots(knots))
        self._status(f"Generated open uniform knots for {n_cp} points, degree {deg}")

    # Generate uniform integer knots
    def _generate_uniform_knots(self):
        n_cp = self._table.rowCount()
        deg = self._deg_spin.value()
        if n_cp < deg + 1:
            n_cp = deg + 1
        knots = uniform(n_cp, deg)
        self._knots_edit.setText(format_knots(knots))
        self._status(f"Generated uniform knots for {n_cp} points, degree {deg}")

    # Add a control point from the text boxes to the table
    def _add_point(self):
        try:
            x = float(self._x_edit.text())
            y = float(self._y_edit.text())
        except ValueError:
            QMessageBox.warning(self, "Input Error", "X and Y must be numbers.")
            return

        z_text = self._z_edit.text().strip()
        z = None
        if z_text:
            try:
                z = float(z_text)
            except ValueError:
                QMessageBox.warning(self, "Input Error", "Z must be a number.")
                return

        row = self._table.rowCount()
        self._table.insertRow(row)
        self._table.setItem(row, 0, QTableWidgetItem(str(x)))
        self._table.setItem(row, 1, QTableWidgetItem(str(y)))
        self._table.setItem(row, 2, QTableWidgetItem(str(z) if z is not None else ""))
        self._x_edit.clear()
        self._y_edit.clear()
        self._z_edit.clear()

    # Remove selected control points
    def _remove_point(self):
        rows = sorted(set(idx.row() for idx in self._table.selectedIndexes()), reverse=True)
        for r in rows:
            self._table.removeRow(r)

    # Clear all control points from table
    def _clear_points(self):
        self._table.setRowCount(0)

    # Read control points from table as numpy array
    def get_control_points(self) -> np.ndarray:
        rows = self._table.rowCount()
        if rows == 0:
            raise ValueError("No control points provided.")
        pts = []
        has_z = False
        for r in range(rows):
            x = read_cell_float(self._table, r, 0, "X")
            y = read_cell_float(self._table, r, 1, "Y")
            z = read_cell_float(self._table, r, 2, "Z", optional=True)
            if z is not None:
                has_z = True
                pts.append([x, y, z])
            else:
                pts.append([x, y, 0.0])
        arr = np.array(pts)
        if not has_z:
            arr = arr[:, :2]
        return arr

    # Populate table from array of control points
    def set_control_points(self, cp: np.ndarray):
        self._table.setRowCount(0)
        cp = np.atleast_2d(cp)
        is_3d = cp.shape[1] == 3
        for row in cp:
            r = self._table.rowCount()
            self._table.insertRow(r)
            self._table.setItem(r, 0, QTableWidgetItem(str(row[0])))
            self._table.setItem(r, 1, QTableWidgetItem(str(row[1])))
            if is_3d:
                self._table.setItem(r, 2, QTableWidgetItem(str(row[2])))
            else:
                self._table.setItem(r, 2, QTableWidgetItem(""))

    # Load predefined test dataset matching Matlab Bspline_Verify.m
    def _load_matlab_preset(self):
        self._deg_spin.setValue(2)
        self._knots_edit.setText("0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 5")
        cp = np.array([
            [5.0, 20.0],
            [5.0, 70.0],
            [50.0, 2.0],
            [50.0, 40.0],
            [70.0, 40.0],
            [80.0, 2.0],
            [90.0, 30.0],
            [100.0, 30.0],
        ])
        self.set_control_points(cp)

    # Return B-spline data dictionary for JSON save
    def get_data(self) -> dict:
        knots = self.parse_knots()
        degree = self._deg_spin.value()
        cp = self.get_control_points()
        validate_bspline_params(knots, degree, cp.shape[0])
        return {
            "type": "bspline_curve",
            "degree": degree,
            "knots": knots,
            "control_points": cp.tolist(),
        }

    # Populate tab fields from loaded JSON data
    def set_data(self, data: dict):
        if "degree" in data:
            degree = check_spin_value(self._deg_spin, data["degree"], "Degree")
            self._deg_spin.setValue(degree)
        if "knots" in data:
            self._knots_edit.setText(format_knots(data["knots"]))
        if "control_points" in data:
            self.set_control_points(np.array(data["control_points"]))

    # Calculate and plot the basis functions N_{i,p}(u)
    def _plot_basis(self):
        try:
            knots = self.parse_knots()
            deg = self._deg_spin.value()
            basis, t = basis_matrix(knots, deg, n_eval=300)
        except Exception as e:
            QMessageBox.warning(self, "Validation Error", str(e))
            return

        self._last_plot = "basis"

        self._figure.clear()
        ax = self._figure.add_subplot(111)
        n_basis = basis.shape[0]
        for i in range(n_basis):
            ax.plot(t, basis[i], label=f"N_{i},{deg}")

        # Unclamped knots: shade the range where the curve exists
        try:
            start, end = valid_domain(knots, deg)
            if (start, end) != (knots[0], knots[-1]):
                ax.axvspan(start, end, color="gray", alpha=0.12, label="Valid curve range")
        except ValueError:
            pass

        ax.set_xlabel("Parameter u")
        ax.set_ylabel("Basis Value")
        ax.set_title(f"B-Spline Basis Functions (degree {deg}, {n_basis} functions)")
        ax.legend()
        ax.grid(True)
        self._canvas.draw()
        self._status(f"Basis functions plotted: {n_basis} basis functions, degree {deg}")

    # Calculate and draw the B-spline curve and control polygon
    # silent=True skips the error dialog (used for redraws after display-option changes)
    def _draw_curve(self, silent: bool = False):
        try:
            knots = self.parse_knots()
            deg = self._deg_spin.value()
            cp = self.get_control_points()
            validate_bspline_params(knots, deg, cp.shape[0])
            curve = bspline_curve(knots, deg, cp, n_eval=300)
        except Exception as e:
            if not silent:
                QMessageBox.warning(self, "Validation Error", str(e))
            return
        self._last_plot = "curve"

        poly_color = self._poly_color_combo.currentText()
        point_color = self._point_color_combo.currentText()
        poly_style = self._poly_style_combo.currentData() or "--"
        curve_color = self._curve_color_combo.currentText()
        line_w = self._line_width_spin.value()
        pt_size = self._point_size_spin.value()

        self._figure.clear()
        dim = cp.shape[1]
        if dim == 3:
            ax = self._figure.add_subplot(111, projection="3d")
            if self._show_poly_chk.isChecked():
                ax.plot(
                    cp[:, 0], cp[:, 1], cp[:, 2],
                    color=poly_color, linestyle=poly_style, linewidth=line_w,
                    label="Control polygon"
                )
            if self._show_points_chk.isChecked():
                ax.plot(
                    cp[:, 0], cp[:, 1], cp[:, 2],
                    linestyle="None", marker="o", color=point_color, markersize=pt_size,
                    label="Control points"
                )
            if self._show_curve_chk.isChecked():
                ax.plot(
                    curve[:, 0], curve[:, 1], curve[:, 2],
                    color=curve_color, linestyle="-", linewidth=line_w + 0.5,
                    label="B-spline curve"
                )
            ax.set_zlabel("Z")
        else:
            ax = self._figure.add_subplot(111)
            if self._show_poly_chk.isChecked():
                ax.plot(
                    cp[:, 0], cp[:, 1],
                    color=poly_color, linestyle=poly_style, linewidth=line_w,
                    label="Control polygon"
                )
            if self._show_points_chk.isChecked():
                ax.plot(
                    cp[:, 0], cp[:, 1],
                    linestyle="None", marker="o", color=point_color, markersize=pt_size,
                    label="Control points"
                )
            if self._show_curve_chk.isChecked():
                ax.plot(
                    curve[:, 0], curve[:, 1],
                    color=curve_color, linestyle="-", linewidth=line_w + 0.5,
                    label="B-spline curve"
                )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        bounds = element_boundaries(knots, deg)
        if self._show_knots_chk.isChecked():
            knot_pts = bspline_curve(knots, deg, cp, u_values=bounds)
            draw_knot_markers(ax, knot_pts, bounds, dim == 3)

        ax.set_title(f"B-Spline Curve (degree {deg}, {cp.shape[0]} control points)")
        if ax.get_legend_handles_labels()[1]:  # only when something is labelled
            ax.legend()
        ax.grid(True)
        self._canvas.draw()
        self._last_status = (f"Curve drawn: degree {deg}, {cp.shape[0]} control points, "
                             f"{element_count_text(len(bounds) - 1)}")
        self._status(self._last_status)

    # Read the curve currently described by the knot box and the table
    def _read_curve(self):
        knots = self.parse_knots()
        deg = self._deg_spin.value()
        cp = self.get_control_points()
        validate_bspline_params(knots, deg, cp.shape[0])
        return knots, deg, cp

    # Swap in the refined curve and report the shape change
    def _apply_refinement(self, old, new_knots, new_cp, action: str):
        knots, deg, cp = old
        u = np.linspace(*valid_domain(knots, deg), 400)
        change = float(np.max(np.abs(bspline_curve(knots, deg, cp, u_values=u)
                                     - bspline_curve(new_knots, deg, new_cp, u_values=u))))
        self._knots_edit.setText(format_knots(new_knots))
        self.set_control_points(new_cp)
        self._draw_curve()
        self._status(f"{action}: {cp.shape[0]} -> {new_cp.shape[0]} control points, "
                     f"max shape change {change:.1e}  |  {self._last_status}")

    # Insert the knot typed in the Refinement box
    def _insert_knot(self):
        try:
            old = self._read_curve()
            u, times = self._refine_box.read_insert()
            new_knots, new_cp, _ = insert_knot(old[0], old[1], old[2], u, times)
        except Exception as e:
            QMessageBox.warning(self, "Insert Knot", str(e))
            return
        self._apply_refinement(old, new_knots, new_cp, f"Inserted knot u={u:g} ({times}x)")

    # Split every element in two
    def _split_elements(self):
        try:
            old = self._read_curve()
            new_knots, new_cp, _ = refine_uniform(old[0], old[1], old[2])
        except Exception as e:
            QMessageBox.warning(self, "Split All Elements", str(e))
            return
        self._apply_refinement(old, new_knots, new_cp, "Split all elements")

    def export_basis_excel(self, filepath: Optional[str] = None):
        """Export basis matrix over parameter t to an Excel workbook."""
        try:
            knots = self.parse_knots()
            deg = self._deg_spin.value()
            basis, t = basis_matrix(knots, deg, n_eval=200)
        except Exception as e:
            QMessageBox.warning(self, "Export Error", f"Cannot compute basis matrix:\n{e}")
            return

        if not filepath:
            filepath, _ = QFileDialog.getSaveFileName(
                self, "Export Basis Matrix to Excel", "bspline_basis.xlsx", "Excel Files (*.xlsx)"
            )
            if not filepath:
                return

        try:
            n_basis = basis.shape[0]
            col_names = ["u"] + [f"N_{i}_{deg}" for i in range(n_basis)]
            data_matrix = np.column_stack([t, basis.T])
            import pandas as pd
            df = pd.DataFrame(data_matrix, columns=col_names)
            df.to_excel(filepath, index=False)
            self._status(f"Basis matrix exported to {filepath}")
            QMessageBox.information(self, "Export Successful", f"Basis matrix saved to:\n{filepath}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to save Excel file:\n{e}")
