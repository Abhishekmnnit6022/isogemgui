"""Bezier Surface tab for setting up a control net and displaying an interactive 3D surface."""

from __future__ import annotations

import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QSpinBox,
    QPushButton, QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox,
    QSplitter, QCheckBox, QComboBox, QDoubleSpinBox
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from core.bezier import bezier_surface
from core.validation import validate_surface_net


class SurfaceTab(QWidget):
    """Tab widget for editing and rendering 3D tensor-product Bezier surfaces."""

    def __init__(self, status_callback=None):
        super().__init__()
        self._status = status_callback or (lambda msg: None)
        self._init_ui()
        self._load_default_preset()

    # Construct the tab layout with grid options, line settings, and 3D plot
    def _init_ui(self):
        main = QHBoxLayout(self)

        left = QVBoxLayout()

        # Dimension group to specify row and column counts
        dim_group = QGroupBox("Control Net Grid")
        dim_layout = QVBoxLayout(dim_group)

        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("Rows (U):"))
        self._rows_spin = QSpinBox()
        self._rows_spin.setRange(2, 20)
        self._rows_spin.setValue(3)
        size_row.addWidget(self._rows_spin)

        size_row.addWidget(QLabel("Cols (V):"))
        self._cols_spin = QSpinBox()
        self._cols_spin.setRange(2, 20)
        self._cols_spin.setValue(3)
        size_row.addWidget(self._cols_spin)

        apply_dim_btn = QPushButton("Resize Table")
        apply_dim_btn.clicked.connect(self._resize_table)
        size_row.addWidget(apply_dim_btn)
        dim_layout.addLayout(size_row)

        self._dim_label = QLabel("Degree: U=2, V=2")
        dim_layout.addWidget(self._dim_label)
        left.addWidget(dim_group)

        # Table holding coordinates for each net point
        tbl_group = QGroupBox("Control Net Coordinates")
        tbl_layout = QVBoxLayout(tbl_group)
        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(["Row (i)", "Col (j)", "X", "Y", "Z"])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        tbl_layout.addWidget(self._table)

        btn_row = QHBoxLayout()
        preset_btn = QPushButton("Load Preset (3x3)")
        preset_btn.clicked.connect(self._load_default_preset)
        btn_row.addWidget(preset_btn)

        clear_btn = QPushButton("Clear Coordinates")
        clear_btn.clicked.connect(self._clear_coords)
        btn_row.addWidget(clear_btn)
        tbl_layout.addLayout(btn_row)
        left.addWidget(tbl_group)

        # Line and Display Options
        opt_group = QGroupBox("Line & Point Display Options")
        opt_layout = QVBoxLayout(opt_group)

        chk_row = QHBoxLayout()
        self._show_lines_chk = QCheckBox("Show Net Lines")
        self._show_lines_chk.setChecked(True)
        chk_row.addWidget(self._show_lines_chk)

        self._show_points_chk = QCheckBox("Show Points")
        self._show_points_chk.setChecked(True)
        chk_row.addWidget(self._show_points_chk)

        self._show_surface_chk = QCheckBox("Show Surface")
        self._show_surface_chk.setChecked(True)
        chk_row.addWidget(self._show_surface_chk)
        opt_layout.addLayout(chk_row)

        line_prop_row = QHBoxLayout()
        line_prop_row.addWidget(QLabel("Line Color:"))
        self._line_color_combo = QComboBox()
        self._line_color_combo.addItems(["blue", "red", "green", "black", "orange", "magenta", "gray"])
        line_prop_row.addWidget(self._line_color_combo)

        line_prop_row.addWidget(QLabel("Style:"))
        self._line_style_combo = QComboBox()
        self._line_style_combo.addItem("dashed", "--")
        self._line_style_combo.addItem("solid", "-")
        self._line_style_combo.addItem("dotted", ":")
        self._line_style_combo.addItem("dash-dot", "-.")
        line_prop_row.addWidget(self._line_style_combo)

        line_prop_row.addWidget(QLabel("Width:"))
        self._line_width_spin = QDoubleSpinBox()
        self._line_width_spin.setRange(0.5, 8.0)
        self._line_width_spin.setSingleStep(0.5)
        self._line_width_spin.setValue(1.5)
        line_prop_row.addWidget(self._line_width_spin)
        opt_layout.addLayout(line_prop_row)

        pt_prop_row = QHBoxLayout()
        pt_prop_row.addWidget(QLabel("Point Color:"))
        self._point_color_combo = QComboBox()
        self._point_color_combo.addItems(["blue", "red", "green", "black", "orange", "magenta", "cyan", "yellow"])
        pt_prop_row.addWidget(self._point_color_combo)

        pt_prop_row.addWidget(QLabel("Marker:"))
        self._point_marker_combo = QComboBox()
        self._point_marker_combo.addItem("circle", "o")
        self._point_marker_combo.addItem("square", "s")
        self._point_marker_combo.addItem("triangle", "^")
        self._point_marker_combo.addItem("diamond", "D")
        self._point_marker_combo.addItem("cross", "x")
        pt_prop_row.addWidget(self._point_marker_combo)

        pt_prop_row.addWidget(QLabel("Size:"))
        self._point_size_spin = QDoubleSpinBox()
        self._point_size_spin.setRange(2.0, 20.0)
        self._point_size_spin.setSingleStep(1.0)
        self._point_size_spin.setValue(7.0)
        pt_prop_row.addWidget(self._point_size_spin)
        opt_layout.addLayout(pt_prop_row)

        left.addWidget(opt_group)

        draw_btn = QPushButton("Plot Surface")
        draw_btn.clicked.connect(self._draw)
        left.addWidget(draw_btn)
        left.addStretch()

        left_widget = QWidget()
        left_widget.setLayout(left)

        # 3D interactive plot canvas on the right
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

    # Adjust table rows according to rows and cols spinbox values
    def _resize_table(self):
        nu = self._rows_spin.value()
        nv = self._cols_spin.value()
        self._dim_label.setText(f"Degree: U={nu - 1}, V={nv - 1}")

        existing = {}
        for r in range(self._table.rowCount()):
            try:
                i = int(self._table.item(r, 0).text())
                j = int(self._table.item(r, 1).text())
                x = self._table.item(r, 2).text()
                y = self._table.item(r, 3).text()
                z = self._table.item(r, 4).text()
                existing[(i, j)] = (x, y, z)
            except Exception:
                pass

        self._table.setRowCount(nu * nv)
        idx = 0
        for i in range(nu):
            for j in range(nv):
                item_i = QTableWidgetItem(str(i))
                item_i.setFlags(item_i.flags() & ~Qt.ItemIsEditable)
                item_j = QTableWidgetItem(str(j))
                item_j.setFlags(item_j.flags() & ~Qt.ItemIsEditable)
                self._table.setItem(idx, 0, item_i)
                self._table.setItem(idx, 1, item_j)

                if (i, j) in existing:
                    x, y, z = existing[(i, j)]
                else:
                    x, y, z = str(float(i)), str(float(j)), "0.0"

                self._table.setItem(idx, 2, QTableWidgetItem(str(x)))
                self._table.setItem(idx, 3, QTableWidgetItem(str(y)))
                self._table.setItem(idx, 4, QTableWidgetItem(str(z)))
                idx += 1

    # Reset all coordinate values in the table to 0
    def _clear_coords(self):
        for r in range(self._table.rowCount()):
            self._table.setItem(r, 2, QTableWidgetItem("0.0"))
            self._table.setItem(r, 3, QTableWidgetItem("0.0"))
            self._table.setItem(r, 4, QTableWidgetItem("0.0"))

    # Load 3x3 preset control net values
    def _load_default_preset(self):
        net = np.array([
            [[-0.5, 2.0, 1.0], [-2.0, 1.0, -1.0], [0.0, 0.0, 2.0]],
            [[1.0, 2.0, 0.0], [1.0, 8.0, -8.5], [1.0, -1.0, 2.0]],
            [[2.0, 2.0, 0.5], [2.0, 1.0, 1.0], [2.0, 1.0, 2.0]],
        ])
        self.set_control_net(net)

    # Read all control points from the table as a 3D numpy array
    def get_control_net(self) -> np.ndarray:
        nu = self._rows_spin.value()
        nv = self._cols_spin.value()
        expected_rows = nu * nv
        if self._table.rowCount() != expected_rows:
            raise ValueError(f"Table row count ({self._table.rowCount()}) does not match grid ({nu}x{nv}).")

        net = np.zeros((nu, nv, 3), dtype=float)
        idx = 0
        for i in range(nu):
            for j in range(nv):
                try:
                    x = float(self._table.item(idx, 2).text())
                    y = float(self._table.item(idx, 3).text())
                    z = float(self._table.item(idx, 4).text())
                except (ValueError, AttributeError):
                    raise ValueError(f"Invalid coordinate at Row {i}, Col {j}.")
                net[i, j] = [x, y, z]
                idx += 1
        return net

    # Populate table and spinboxes from a 3D control net array
    def set_control_net(self, net: np.ndarray):
        net = validate_surface_net(net)
        nu, nv, _ = net.shape
        self._rows_spin.setValue(nu)
        self._cols_spin.setValue(nv)
        self._dim_label.setText(f"Degree: U={nu - 1}, V={nv - 1}")

        self._table.setRowCount(nu * nv)
        idx = 0
        for i in range(nu):
            for j in range(nv):
                item_i = QTableWidgetItem(str(i))
                item_i.setFlags(item_i.flags() & ~Qt.ItemIsEditable)
                item_j = QTableWidgetItem(str(j))
                item_j.setFlags(item_j.flags() & ~Qt.ItemIsEditable)
                self._table.setItem(idx, 0, item_i)
                self._table.setItem(idx, 1, item_j)
                self._table.setItem(idx, 2, QTableWidgetItem(str(net[i, j, 0])))
                self._table.setItem(idx, 3, QTableWidgetItem(str(net[i, j, 1])))
                self._table.setItem(idx, 4, QTableWidgetItem(str(net[i, j, 2])))
                idx += 1

    # Extract surface dictionary for JSON export
    def get_data(self) -> dict:
        net = self.get_control_net()
        nu, nv, _ = net.shape
        return {
            "type": "bezier_surface",
            "degree": [nu - 1, nv - 1],
            "control_points": net.tolist(),
        }

    # Populate surface tab from loaded JSON data
    def set_data(self, data: dict):
        if "control_points" in data:
            self.set_control_net(np.array(data["control_points"]))

    # Evaluate the surface and plot both the 3D surface and control net
    def _draw(self):
        try:
            net = self.get_control_net()
            X, Y, Z = bezier_surface(net, u_eval=30, v_eval=30)
        except Exception as e:
            QMessageBox.warning(self, "Validation Error", str(e))
            return

        nu, nv, _ = net.shape
        self._figure.clear()
        ax = self._figure.add_subplot(111, projection="3d")

        # Plot 3D surface mesh if enabled
        if self._show_surface_chk.isChecked():
            ax.plot_surface(X, Y, Z, color="cyan", alpha=0.7, edgecolor="gray", linewidth=0.5)

        # Plot control net vertices as a Line3D artist so it appears in Figure Options -> Curves
        # and responds directly to Point Color and Marker selection
        if self._show_points_chk.isChecked():
            p_color = self._point_color_combo.currentText()
            p_marker = self._point_marker_combo.currentData() or "o"
            p_size = self._point_size_spin.value()
            ax.plot(
                net[:, :, 0].flatten(),
                net[:, :, 1].flatten(),
                net[:, :, 2].flatten(),
                linestyle="None",
                marker=p_marker,
                color=p_color,
                markersize=p_size,
                label="Control points",
            )

        # Plot all control net lines in a single call using NaN separators
        # This groups all grid lines under a single "Control net lines" label
        # so Figure Options edits all lines together instead of multiple _child entries.
        if self._show_lines_chk.isChecked():
            lines_x: list[float] = []
            lines_y: list[float] = []
            lines_z: list[float] = []

            for i in range(nu):
                lines_x.extend(net[i, :, 0].tolist() + [np.nan])
                lines_y.extend(net[i, :, 1].tolist() + [np.nan])
                lines_z.extend(net[i, :, 2].tolist() + [np.nan])

            for j in range(nv):
                lines_x.extend(net[:, j, 0].tolist() + [np.nan])
                lines_y.extend(net[:, j, 1].tolist() + [np.nan])
                lines_z.extend(net[:, j, 2].tolist() + [np.nan])

            l_color = self._line_color_combo.currentText()
            l_style = self._line_style_combo.currentData() or "--"
            l_width = self._line_width_spin.value()

            ax.plot(
                lines_x, lines_y, lines_z,
                color=l_color,
                linestyle=l_style,
                linewidth=l_width,
                alpha=0.8,
                label="Control net lines",
            )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"Bezier Surface (degree {nu - 1} x {nv - 1})")
        ax.legend()
        self._canvas.draw()
        self._status(f"Surface drawn: degree {nu-1}x{nv-1}, {nu}x{nv} control net")
