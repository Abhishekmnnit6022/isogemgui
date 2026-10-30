"""Bezier Curve tab for entering control points and displaying 2D or 3D curves."""

from __future__ import annotations

import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QLineEdit,
    QPushButton, QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox,
    QSplitter, QCheckBox, QComboBox, QDoubleSpinBox
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

from core.bezier import bezier_curve


class BezierTab(QWidget):
    """Tab widget for drawing arbitrary degree Bezier curves in 2D and 3D."""

    def __init__(self, status_callback=None):
        super().__init__()
        self._status = status_callback or (lambda msg: None)
        self._is_3d = False
        self._init_ui()

    # Build the user interface with left control panel and right plot view
    def _init_ui(self):
        main = QHBoxLayout(self)

        left = QVBoxLayout()

        # Input group for entering point coordinates
        inp_group = QGroupBox("Add Control Point")
        inp_layout = QVBoxLayout(inp_group)

        row_xy = QHBoxLayout()
        row_xy.addWidget(QLabel("X:"))
        self._x_edit = QLineEdit()
        row_xy.addWidget(self._x_edit)
        row_xy.addWidget(QLabel("Y:"))
        self._y_edit = QLineEdit()
        row_xy.addWidget(self._y_edit)
        row_xy.addWidget(QLabel("Z:"))
        self._z_edit = QLineEdit()
        self._z_edit.setPlaceholderText("(optional)")
        row_xy.addWidget(self._z_edit)
        inp_layout.addLayout(row_xy)

        btn_row = QHBoxLayout()
        add_btn = QPushButton("Add Point")
        add_btn.clicked.connect(self._add_point)
        btn_row.addWidget(add_btn)
        rm_btn = QPushButton("Remove Selected")
        rm_btn.clicked.connect(self._remove_selected)
        btn_row.addWidget(rm_btn)
        clr_btn = QPushButton("Clear")
        clr_btn.clicked.connect(self._clear_points)
        btn_row.addWidget(clr_btn)
        inp_layout.addLayout(btn_row)

        left.addWidget(inp_group)

        # Table showing current list of control points
        tbl_group = QGroupBox("Control Points")
        tbl_layout = QVBoxLayout(tbl_group)
        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(["X", "Y", "Z"])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        tbl_layout.addWidget(self._table)
        self._degree_label = QLabel("Degree: 0")
        tbl_layout.addWidget(self._degree_label)
        left.addWidget(tbl_group)

        # Display and Line Options
        opt_group = QGroupBox("Line & Point Display Options")
        opt_layout = QVBoxLayout(opt_group)

        chk_row = QHBoxLayout()
        self._show_poly_chk = QCheckBox("Show Polygon")
        self._show_poly_chk.setChecked(True)
        chk_row.addWidget(self._show_poly_chk)

        self._show_points_chk = QCheckBox("Show Points")
        self._show_points_chk.setChecked(True)
        chk_row.addWidget(self._show_points_chk)

        self._show_curve_chk = QCheckBox("Show Curve")
        self._show_curve_chk.setChecked(True)
        chk_row.addWidget(self._show_curve_chk)
        opt_layout.addLayout(chk_row)

        color_row1 = QHBoxLayout()
        color_row1.addWidget(QLabel("Poly Color:"))
        self._poly_color_combo = QComboBox()
        self._poly_color_combo.addItems(["blue", "red", "green", "black", "orange", "magenta", "gray"])
        color_row1.addWidget(self._poly_color_combo)

        color_row1.addWidget(QLabel("Point Color:"))
        self._point_color_combo = QComboBox()
        self._point_color_combo.addItems(["blue", "red", "green", "black", "orange", "magenta", "cyan", "yellow"])
        color_row1.addWidget(self._point_color_combo)
        opt_layout.addLayout(color_row1)

        color_row2 = QHBoxLayout()
        color_row2.addWidget(QLabel("Curve Color:"))
        self._curve_color_combo = QComboBox()
        self._curve_color_combo.addItems(["red", "blue", "green", "black", "orange", "purple"])
        color_row2.addWidget(self._curve_color_combo)

        color_row2.addWidget(QLabel("Poly Style:"))
        self._poly_style_combo = QComboBox()
        self._poly_style_combo.addItem("dashed", "--")
        self._poly_style_combo.addItem("solid", "-")
        self._poly_style_combo.addItem("dotted", ":")
        color_row2.addWidget(self._poly_style_combo)
        opt_layout.addLayout(color_row2)

        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("Line Width:"))
        self._line_width_spin = QDoubleSpinBox()
        self._line_width_spin.setRange(0.5, 8.0)
        self._line_width_spin.setSingleStep(0.5)
        self._line_width_spin.setValue(1.5)
        size_row.addWidget(self._line_width_spin)

        size_row.addWidget(QLabel("Point Size:"))
        self._point_size_spin = QDoubleSpinBox()
        self._point_size_spin.setRange(2.0, 20.0)
        self._point_size_spin.setSingleStep(1.0)
        self._point_size_spin.setValue(7.0)
        size_row.addWidget(self._point_size_spin)
        opt_layout.addLayout(size_row)

        left.addWidget(opt_group)

        draw_btn = QPushButton("Draw Curve")
        draw_btn.clicked.connect(self._draw)
        left.addWidget(draw_btn)
        left.addStretch()

        left_widget = QWidget()
        left_widget.setLayout(left)

        # Plot view on the right using Matplotlib canvas
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

    # Add a new point from the text boxes into the table
    def _add_point(self):
        try:
            x = float(self._x_edit.text())
            y = float(self._y_edit.text())
        except ValueError:
            QMessageBox.warning(self, "Input Error", "X and Y must be numbers.")
            return
        z_text = self._z_edit.text().strip()
        if z_text:
            try:
                z = float(z_text)
            except ValueError:
                QMessageBox.warning(self, "Input Error", "Z must be a number.")
                return
            self._is_3d = True
        else:
            z = None

        row = self._table.rowCount()
        self._table.insertRow(row)
        self._table.setItem(row, 0, QTableWidgetItem(str(x)))
        self._table.setItem(row, 1, QTableWidgetItem(str(y)))
        self._table.setItem(row, 2, QTableWidgetItem(str(z) if z is not None else ""))
        self._degree_label.setText(f"Degree: {row}")
        self._x_edit.clear()
        self._y_edit.clear()
        self._z_edit.clear()

    # Remove the selected rows from the table
    def _remove_selected(self):
        rows = sorted(set(idx.row() for idx in self._table.selectedIndexes()), reverse=True)
        for r in rows:
            self._table.removeRow(r)
        n = self._table.rowCount()
        self._degree_label.setText(f"Degree: {max(0, n - 1)}")

    # Clear all control points from the table
    def _clear_points(self):
        self._table.setRowCount(0)
        self._degree_label.setText("Degree: 0")
        self._is_3d = False

    # Extract all control points from the table as a numpy array
    def get_control_points(self) -> np.ndarray:
        rows = self._table.rowCount()
        if rows < 2:
            raise ValueError("Need at least 2 control points.")
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

    # Populate the table from a numpy array of points
    def set_control_points(self, cp: np.ndarray):
        self._table.setRowCount(0)
        cp = np.atleast_2d(cp)
        is_3d = cp.shape[1] == 3
        self._is_3d = is_3d
        for row in cp:
            r = self._table.rowCount()
            self._table.insertRow(r)
            self._table.setItem(r, 0, QTableWidgetItem(str(row[0])))
            self._table.setItem(r, 1, QTableWidgetItem(str(row[1])))
            if is_3d:
                self._table.setItem(r, 2, QTableWidgetItem(str(row[2])))
            else:
                self._table.setItem(r, 2, QTableWidgetItem(""))
        self._degree_label.setText(f"Degree: {max(0, cp.shape[0] - 1)}")

    # Return curve dictionary for saving to JSON
    def get_data(self) -> dict:
        cp = self.get_control_points()
        return {
            "type": "bezier_curve",
            "degree": cp.shape[0] - 1,
            "control_points": cp.tolist(),
        }

    # Load curve dictionary from JSON
    def set_data(self, data: dict):
        if "control_points" in data:
            self.set_control_points(np.array(data["control_points"]))

    # Calculate and draw the Bezier curve and control polygon
    def _draw(self):
        try:
            cp = self.get_control_points()
            curve = bezier_curve(cp, n_eval=500)
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))
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
                    label="Bezier curve"
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
                    label="Bezier curve"
                )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend()
        ax.grid(True)
        ax.set_title(f"Bezier Curve (degree {cp.shape[0] - 1})")
        self._canvas.draw()
        self._status(f"Curve drawn: degree {cp.shape[0]-1}, {cp.shape[0]} control points")
