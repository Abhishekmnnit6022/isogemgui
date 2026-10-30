"""B-Spline tab for knot vector generation, basis function plotting, curve drawing, and Excel export."""

from __future__ import annotations

from typing import List, Optional
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QLineEdit,
    QSpinBox, QPushButton, QTableWidget, QTableWidgetItem, QHeaderView,
    QMessageBox, QSplitter, QFileDialog, QCheckBox, QComboBox, QDoubleSpinBox,
    QDialog, QGridLayout, QToolButton
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

from core.bspline import basis_matrix, bspline_curve
from core.knots import open_uniform, uniform
from core.validation import validate_bspline_params, validate_knots


class BsplineTab(QWidget):
    """Tab widget for B-spline curves, basis functions, and Excel export."""

    def __init__(self, status_callback=None):
        super().__init__()
        self._status = status_callback or (lambda msg: None)
        self._last_basis: Optional[np.ndarray] = None
        self._last_t: Optional[np.ndarray] = None
        self._last_degree: int = 2
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
        self._table.setHorizontalHeaderLabels(["X", "Y", "Z"])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        cp_layout.addWidget(self._table)

        preset_row = QHBoxLayout()
        preset_btn = QPushButton("Load Matlab Preset")
        preset_btn.clicked.connect(self._load_matlab_preset)
        preset_row.addWidget(preset_btn)
        cp_layout.addLayout(preset_row)

        left.addWidget(cp_group)

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
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        main.addWidget(splitter)

    def _open_style_dialog(self):
        self._style_dialog.exec_()
        self._on_display_option_changed()

    def _on_display_option_changed(self):
        if self._table.rowCount() >= 2:
            try:
                self._draw_curve()
            except Exception:
                pass

    # Parse and validate the knot values entered in the text box
    def parse_knots(self) -> List[float]:
        text = self._knots_edit.text().strip().replace(",", " ")
        if not text:
            raise ValueError("Knot vector is empty. Please enter knots or generate them.")
        tokens = text.split()
        try:
            raw_knots = [float(t) for t in tokens]
        except ValueError:
            raise ValueError("All knot values must be numbers.")
        return validate_knots(raw_knots)

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
        self._knots_edit.setText(", ".join(f"{k:g}" for k in knots))
        self._status(f"Generated open uniform knots for {n_cp} points, degree {deg}")

    # Generate uniform integer knots
    def _generate_uniform_knots(self):
        n_cp = self._table.rowCount()
        deg = self._deg_spin.value()
        if n_cp < deg + 1:
            n_cp = deg + 1
        knots = uniform(n_cp, deg)
        self._knots_edit.setText(", ".join(f"{k:g}" for k in knots))
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
            x = float(self._table.item(r, 0).text())
            y = float(self._table.item(r, 1).text())
            z_item = self._table.item(r, 2)
            z_text = z_item.text().strip() if z_item else ""
            if z_text:
                has_z = True
                pts.append([x, y, float(z_text)])
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
            self._deg_spin.setValue(int(data["degree"]))
        if "knots" in data:
            self._knots_edit.setText(", ".join(f"{k:g}" for k in data["knots"]))
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

        self._last_basis = basis
        self._last_t = t
        self._last_degree = deg

        self._figure.clear()
        ax = self._figure.add_subplot(111)
        n_basis = basis.shape[0]
        for i in range(n_basis):
            ax.plot(t, basis[i], label=f"N_{i},{deg}")

        ax.set_xlabel("Parameter u")
        ax.set_ylabel("Basis Value")
        ax.set_title(f"B-Spline Basis Functions (degree {deg}, {n_basis} functions)")
        ax.legend()
        ax.grid(True)
        self._canvas.draw()
        self._status(f"Basis functions plotted: {n_basis} basis functions, degree {deg}")

    # Calculate and draw the B-spline curve and control polygon
    def _draw_curve(self):
        try:
            knots = self.parse_knots()
            deg = self._deg_spin.value()
            cp = self.get_control_points()
            validate_bspline_params(knots, deg, cp.shape[0])
            curve = bspline_curve(knots, deg, cp, n_eval=300)
        except Exception as e:
            QMessageBox.warning(self, "Validation Error", str(e))
            return

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
        ax.set_title(f"B-Spline Curve (degree {deg}, {cp.shape[0]} control points)")
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend()
        ax.grid(True)
        self._canvas.draw()
        self._status(f"Curve drawn: degree {deg}, {cp.shape[0]} control points")

    # Export calculated basis function matrix to Excel spreadsheet
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
            df = pd.DataFrame(data_matrix, columns=col_names)
            df.to_excel(filepath, index=False)
            self._status(f"Basis matrix exported to {filepath}")
            QMessageBox.information(self, "Export Successful", f"Basis matrix saved to:\n{filepath}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to save Excel file:\n{e}")
