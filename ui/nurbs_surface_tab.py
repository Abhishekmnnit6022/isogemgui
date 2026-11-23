"""NURBS Surface tab for editing a control net with weights and displaying a rotatable 3D surface."""

from __future__ import annotations

import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QLineEdit,
    QSpinBox, QPushButton, QTableWidget, QTableWidgetItem, QHeaderView,
    QMessageBox, QSplitter, QCheckBox, QComboBox, QDoubleSpinBox, QDialog,
    QGridLayout, QToolButton,
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

from core.analysis import element_mesh_lines
from core.bspline import element_boundaries
from core.knots import open_uniform
from core.nurbs import nurbs_surface, nurbs_cylinder, nurbs_t_shape, sample_parameters
from core.validation import validate_bspline_params
from ui.common import check_spin_value, format_knots, format_number, parse_knots_text
from ui.widgets import draw_polylines, scrollable

DEFAULT_ROWS = 3
DEFAULT_COLS = 2
DEFAULT_DEG_U = 2
DEFAULT_DEG_V = 1
MAX_NET_SIZE = 50


class NurbsSurfaceTab(QWidget):
    """Tab widget for NURBS surfaces with rational weights and 3D view."""

    def __init__(self, status_callback=None):
        super().__init__()
        self._status = status_callback or (lambda msg: None)
        self._has_plot = False
        self._init_ui()
        self.reset(quiet=True)

    # Build the layout
    def _init_ui(self):
        main = QHBoxLayout(self)

        left = QVBoxLayout()

        # Grid size (tables follow the spin boxes automatically)
        dim_group = QGroupBox("Control Net Grid")
        dim_layout = QVBoxLayout(dim_group)

        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("Rows (U):"))
        self._rows_spin = QSpinBox()
        self._rows_spin.setRange(2, MAX_NET_SIZE)
        self._rows_spin.setValue(DEFAULT_ROWS)
        self._rows_spin.valueChanged.connect(self._resize_table)
        size_row.addWidget(self._rows_spin)

        size_row.addWidget(QLabel("Cols (V):"))
        self._cols_spin = QSpinBox()
        self._cols_spin.setRange(2, MAX_NET_SIZE)
        self._cols_spin.setValue(DEFAULT_COLS)
        self._cols_spin.valueChanged.connect(self._resize_table)
        size_row.addWidget(self._cols_spin)
        dim_layout.addLayout(size_row)

        left.addWidget(dim_group)

        # Parameters: degrees and knots
        param_group = QGroupBox("NURBS Parameters")
        param_layout = QVBoxLayout(param_group)

        deg_row = QHBoxLayout()
        deg_row.addWidget(QLabel("Degree U:"))
        self._deg_u_spin = QSpinBox()
        self._deg_u_spin.setRange(1, 15)
        self._deg_u_spin.setValue(DEFAULT_DEG_U)
        deg_row.addWidget(self._deg_u_spin)

        deg_row.addWidget(QLabel("Degree V:"))
        self._deg_v_spin = QSpinBox()
        self._deg_v_spin.setRange(1, 15)
        self._deg_v_spin.setValue(DEFAULT_DEG_V)
        deg_row.addWidget(self._deg_v_spin)
        param_layout.addLayout(deg_row)

        param_layout.addWidget(QLabel("Knots U:"))
        self._knots_u_edit = QLineEdit()
        self._knots_u_edit.setPlaceholderText("e.g. 0, 0, 0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1, 1, 1")
        param_layout.addWidget(self._knots_u_edit)

        param_layout.addWidget(QLabel("Knots V:"))
        self._knots_v_edit = QLineEdit()
        self._knots_v_edit.setPlaceholderText("e.g. 0, 0, 1, 1")
        param_layout.addWidget(self._knots_v_edit)

        gen_btn = QPushButton("Gen Open Knots (U and V)")
        gen_btn.setToolTip("Generate clamped open uniform knots matching the grid size and degrees")
        gen_btn.clicked.connect(self._generate_open_knots)
        param_layout.addWidget(gen_btn)

        left.addWidget(param_group)

        # Coordinates table
        coord_group = QGroupBox("Control Net Coordinates")
        coord_layout = QVBoxLayout(coord_group)
        self._coord_table = QTableWidget(0, 5)
        self._coord_table.setMinimumHeight(130)
        self._coord_table.setHorizontalHeaderLabels(["Row (i)", "Col (j)", "X", "Y", "Z"])
        self._coord_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        coord_layout.addWidget(self._coord_table)
        left.addWidget(coord_group)

        # Weights table
        wt_group = QGroupBox("Weights")
        wt_layout = QVBoxLayout(wt_group)
        self._wt_table = QTableWidget(0, 3)
        self._wt_table.setMinimumHeight(110)
        self._wt_table.setHorizontalHeaderLabels(["Row (i)", "Col (j)", "Weight"])
        self._wt_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        wt_layout.addWidget(self._wt_table)
        reset_w_btn = QPushButton("Reset Weights to 1")
        reset_w_btn.clicked.connect(self._reset_weights)
        wt_layout.addWidget(reset_w_btn)
        left.addWidget(wt_group)

        # Example buttons
        btn_row = QHBoxLayout()
        cyl_btn = QPushButton("Load C&ylinder Example")
        cyl_btn.clicked.connect(self._load_cylinder)
        btn_row.addWidget(cyl_btn)
        t_btn = QPushButton("Load &T-Shape Example")
        t_btn.setToolTip("3D T-section (rounded inner corners) extruded along Z")
        t_btn.clicked.connect(self._load_t_shape)
        btn_row.addWidget(t_btn)
        left.addLayout(btn_row)

        draw_btn = QPushButton("&Draw Surface")
        draw_btn.clicked.connect(self._draw)
        left.addWidget(draw_btn)
        left.addStretch()

        left_widget = QWidget()
        left_widget.setLayout(left)

        self._build_style_dialog()

        # 3D plot with display toggles in the toolbar ribbon
        right = QVBoxLayout()
        self._figure = Figure()
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._toolbar = NavigationToolbar2QT(self._canvas, self)
        self._toolbar.addSeparator()

        self._show_surface_chk = QCheckBox("Surface")
        self._show_surface_chk.setChecked(True)
        self._show_surface_chk.setToolTip("Show/hide the NURBS surface")
        self._show_surface_chk.stateChanged.connect(self._on_display_option_changed)
        self._toolbar.addWidget(self._show_surface_chk)

        self._show_lines_chk = QCheckBox("Net Lines")
        self._show_lines_chk.setChecked(True)
        self._show_lines_chk.setToolTip("Show/hide control net wireframe lines")
        self._show_lines_chk.stateChanged.connect(self._on_display_option_changed)
        self._toolbar.addWidget(self._show_lines_chk)

        self._show_points_chk = QCheckBox("Points")
        self._show_points_chk.setChecked(True)
        self._show_points_chk.setToolTip("Show/hide control net points")
        self._show_points_chk.stateChanged.connect(self._on_display_option_changed)
        self._toolbar.addWidget(self._show_points_chk)

        self._equal_axes_chk = QCheckBox("Equal Axes")
        self._equal_axes_chk.setChecked(True)
        self._equal_axes_chk.setToolTip("Keep true proportions (a cylinder looks round)")
        self._equal_axes_chk.stateChanged.connect(self._on_display_option_changed)
        self._toolbar.addWidget(self._equal_axes_chk)

        self._show_knots_chk = QCheckBox("Knots")
        self._show_knots_chk.setToolTip("Draw the element boundaries (lines at every distinct knot)")
        self._show_knots_chk.stateChanged.connect(self._on_display_option_changed)
        self._toolbar.addWidget(self._show_knots_chk)

        self._toolbar.addSeparator()
        self._style_btn = QToolButton(self._toolbar)
        self._style_btn.setText("Style Options...")
        self._style_btn.setToolTip("Configure surface colour, net lines and point markers")
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

    # Style dialog for surface, net line and point appearance
    def _build_style_dialog(self):
        self._style_dialog = QDialog(self)
        self._style_dialog.setWindowTitle("Display & Style Options")
        dlg_layout = QVBoxLayout(self._style_dialog)
        dlg_layout.setSpacing(10)
        grid = QGridLayout()

        grid.addWidget(QLabel("Surface Color:"), 0, 0)
        self._surface_color_combo = QComboBox()
        self._surface_color_combo.addItems(["cyan", "lightgray", "orange", "lightgreen", "gold", "pink"])
        grid.addWidget(self._surface_color_combo, 0, 1)

        grid.addWidget(QLabel("Opacity:"), 0, 2)
        self._surface_alpha_spin = QDoubleSpinBox()
        self._surface_alpha_spin.setRange(0.1, 1.0)
        self._surface_alpha_spin.setSingleStep(0.1)
        self._surface_alpha_spin.setValue(0.7)
        grid.addWidget(self._surface_alpha_spin, 0, 3)

        grid.addWidget(QLabel("Line Color:"), 1, 0)
        self._line_color_combo = QComboBox()
        self._line_color_combo.addItems(["blue", "red", "green", "black", "orange", "magenta", "gray"])
        grid.addWidget(self._line_color_combo, 1, 1)

        grid.addWidget(QLabel("Line Style:"), 1, 2)
        self._line_style_combo = QComboBox()
        self._line_style_combo.addItem("dashed", "--")
        self._line_style_combo.addItem("solid", "-")
        self._line_style_combo.addItem("dotted", ":")
        self._line_style_combo.addItem("dash-dot", "-.")
        grid.addWidget(self._line_style_combo, 1, 3)

        grid.addWidget(QLabel("Line Width:"), 2, 0)
        self._line_width_spin = QDoubleSpinBox()
        self._line_width_spin.setRange(0.5, 8.0)
        self._line_width_spin.setSingleStep(0.5)
        self._line_width_spin.setValue(1.0)
        grid.addWidget(self._line_width_spin, 2, 1)

        grid.addWidget(QLabel("Point Color:"), 2, 2)
        self._point_color_combo = QComboBox()
        self._point_color_combo.addItems(["blue", "red", "green", "black", "orange", "magenta"])
        grid.addWidget(self._point_color_combo, 2, 3)

        grid.addWidget(QLabel("Point Size:"), 3, 0)
        self._point_size_spin = QDoubleSpinBox()
        self._point_size_spin.setRange(2.0, 20.0)
        self._point_size_spin.setSingleStep(1.0)
        self._point_size_spin.setValue(5.0)
        grid.addWidget(self._point_size_spin, 3, 1)
        dlg_layout.addLayout(grid)

        for combo in (self._surface_color_combo, self._line_color_combo,
                      self._line_style_combo, self._point_color_combo):
            combo.currentIndexChanged.connect(self._on_display_option_changed)
        for spin in (self._surface_alpha_spin, self._line_width_spin, self._point_size_spin):
            spin.valueChanged.connect(self._on_display_option_changed)

        dlg_btn_row = QHBoxLayout()
        dlg_btn_row.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self._style_dialog.accept)
        dlg_btn_row.addWidget(close_btn)
        dlg_layout.addLayout(dlg_btn_row)

    # Show the style dialog (changes redraw immediately)
    def _open_style_dialog(self):
        self._style_dialog.exec_()

    # Redraw quietly after a display option changes (only if something is plotted)
    def _on_display_option_changed(self, *_):
        if self._has_plot:
            self._draw(silent=True)

    # Parse knot text field
    def _parse_knots(self, edit: QLineEdit, label: str = "Knot vector") -> list:
        return parse_knots_text(edit.text(), label)

    # Read the (i, j) -> text values currently in a table
    @staticmethod
    def _table_values(table: QTableWidget, n_values: int) -> dict:
        values = {}
        for r in range(table.rowCount()):
            try:
                i = int(table.item(r, 0).text())
                j = int(table.item(r, 1).text())
                values[(i, j)] = tuple(table.item(r, 2 + c).text() for c in range(n_values))
            except (AttributeError, ValueError):
                pass
        return values

    # Write one grid of rows into a table; values(i, j) returns the cell texts
    @staticmethod
    def _write_table(table: QTableWidget, nu: int, nv: int, values):
        table.setRowCount(nu * nv)
        idx = 0
        for i in range(nu):
            for j in range(nv):
                for col, text in enumerate((str(i), str(j))):
                    item = QTableWidgetItem(text)
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    table.setItem(idx, col, item)
                for c, text in enumerate(values(i, j)):
                    table.setItem(idx, 2 + c, QTableWidgetItem(text))
                idx += 1

    # Rebuild coordinate and weight tables to match the grid size, keeping existing values
    def _resize_table(self, *_):
        nu = self._rows_spin.value()
        nv = self._cols_spin.value()
        coords = self._table_values(self._coord_table, 3)
        wts = self._table_values(self._wt_table, 1)
        self._write_table(self._coord_table, nu, nv,
                          lambda i, j: coords.get((i, j), (str(float(i)), str(float(j)), "0.0")))
        self._write_table(self._wt_table, nu, nv,
                          lambda i, j: wts.get((i, j), ("1.0",)))

    # Set every weight back to 1 (turns the NURBS into a plain B-spline surface)
    def _reset_weights(self):
        nu, nv = self._rows_spin.value(), self._cols_spin.value()
        self._write_table(self._wt_table, nu, nv, lambda i, j: ("1.0",))
        self._status("All weights reset to 1.0")

    # Generate clamped open uniform knots for both directions
    def _generate_open_knots(self):
        nu, nv = self._rows_spin.value(), self._cols_spin.value()
        du, dv = self._deg_u_spin.value(), self._deg_v_spin.value()
        try:
            ku = open_uniform(nu, du)
            kv = open_uniform(nv, dv)
        except ValueError as e:
            QMessageBox.warning(self, "Knot Generation",
                                f"{e}\nIncrease the grid size or lower the degree.")
            return
        self._knots_u_edit.setText(format_knots(ku))
        self._knots_v_edit.setText(format_knots(kv))
        self._status(f"Generated open knots: U for {nu} rows (degree {du}), V for {nv} cols (degree {dv})")

    # Reset the tab to a fresh, drawable default net (used by File -> New)
    def reset(self, quiet: bool = False):
        for spin, value in ((self._rows_spin, DEFAULT_ROWS), (self._cols_spin, DEFAULT_COLS),
                            (self._deg_u_spin, DEFAULT_DEG_U), (self._deg_v_spin, DEFAULT_DEG_V)):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)
        self._coord_table.setRowCount(0)
        self._wt_table.setRowCount(0)
        self._resize_table()
        self._knots_u_edit.setText(format_knots(open_uniform(DEFAULT_ROWS, DEFAULT_DEG_U)))
        self._knots_v_edit.setText(format_knots(open_uniform(DEFAULT_COLS, DEFAULT_DEG_V)))
        self._figure.clear()
        self._canvas.draw()
        self._has_plot = False
        if not quiet:
            self._status("NURBS surface reset to a new 3x2 control net")

    # Read control net from coordinate table
    def _get_net(self):
        nu = self._rows_spin.value()
        nv = self._cols_spin.value()
        expected = nu * nv
        if self._coord_table.rowCount() != expected:
            raise ValueError(f"Table rows ({self._coord_table.rowCount()}) don't match grid ({nu}x{nv}).")
        net = np.zeros((nu, nv, 3))
        idx = 0
        for i in range(nu):
            for j in range(nv):
                try:
                    x = float(self._coord_table.item(idx, 2).text())
                    y = float(self._coord_table.item(idx, 3).text())
                    z = float(self._coord_table.item(idx, 4).text())
                except (ValueError, AttributeError):
                    raise ValueError(f"Invalid coordinate at Row {i}, Col {j}.")
                net[i, j] = [x, y, z]
                idx += 1
        return net

    # Read weights from weight table
    def _get_weights(self):
        nu = self._rows_spin.value()
        nv = self._cols_spin.value()
        expected = nu * nv
        if self._wt_table.rowCount() != expected:
            raise ValueError(f"Weight table rows ({self._wt_table.rowCount()}) don't match grid ({nu}x{nv}).")
        w = np.zeros((nu, nv))
        idx = 0
        for i in range(nu):
            for j in range(nv):
                try:
                    w[i, j] = float(self._wt_table.item(idx, 2).text())
                except (ValueError, AttributeError):
                    raise ValueError(f"Invalid weight at Row {i}, Col {j}.")
                if w[i, j] <= 0.0:
                    raise ValueError(f"Weight at Row {i}, Col {j} must be strictly positive (got {w[i, j]}).")
                idx += 1
        return w

    # Read and check everything needed to evaluate the surface
    def _read_inputs(self):
        net = self._get_net()
        w = self._get_weights()
        ku = self._parse_knots(self._knots_u_edit, "Knots U")
        kv = self._parse_knots(self._knots_v_edit, "Knots V")
        du = self._deg_u_spin.value()
        dv = self._deg_v_spin.value()
        try:
            validate_bspline_params(ku, du, net.shape[0])
        except ValueError as e:
            raise ValueError(f"U direction (rows): {e}")
        try:
            validate_bspline_params(kv, dv, net.shape[1])
        except ValueError as e:
            raise ValueError(f"V direction (cols): {e}")
        return net, w, ku, kv, du, dv

    # Populate tables from net and weights arrays
    def _set_net_and_weights(self, net, weights, knots_u, knots_v, degree_u, degree_v):
        net = np.asarray(net, dtype=float)
        weights = np.asarray(weights, dtype=float)
        nu, nv = net.shape[0], net.shape[1]
        if nu > MAX_NET_SIZE or nv > MAX_NET_SIZE:
            raise ValueError(f"Control net {nu}x{nv} is larger than the supported {MAX_NET_SIZE}x{MAX_NET_SIZE}.")
        degree_u = check_spin_value(self._deg_u_spin, degree_u, "Degree U")
        degree_v = check_spin_value(self._deg_v_spin, degree_v, "Degree V")

        for spin, value in ((self._rows_spin, nu), (self._cols_spin, nv),
                            (self._deg_u_spin, int(degree_u)), (self._deg_v_spin, int(degree_v))):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)
        self._knots_u_edit.setText(format_knots(knots_u))
        self._knots_v_edit.setText(format_knots(knots_v))

        self._write_table(self._coord_table, nu, nv,
                          lambda i, j: tuple(format_number(net[i, j, c]) for c in range(3)))
        self._write_table(self._wt_table, nu, nv,
                          lambda i, j: (format_number(weights[i, j]),))

    # Load cylinder preset
    def _load_cylinder(self):
        net, w, ku, kv, du, dv = nurbs_cylinder()
        self._set_net_and_weights(net, w, ku, kv, du, dv)
        self._status("Loaded NURBS cylinder example (9x2 net, degree 2x1)")

    # Load 3D T-shape preset
    def _load_t_shape(self):
        net, w, ku, kv, du, dv = nurbs_t_shape()
        self._set_net_and_weights(net, w, ku, kv, du, dv)
        self._status(f"Loaded 3D T-shape example ({net.shape[0]}x{net.shape[1]} net, degree {du}x{dv})")

    # get_data for JSON save
    def get_data(self) -> dict:
        net, w, ku, kv, du, dv = self._read_inputs()
        return {
            "type": "nurbs_surface",
            "degree_u": du,
            "degree_v": dv,
            "knots_u": ku,
            "knots_v": kv,
            "control_points": net.tolist(),
            "weights": w.tolist(),
        }

    # set_data for JSON load
    def set_data(self, data: dict):
        net = np.asarray(data["control_points"], dtype=float)
        w = np.asarray(data["weights"], dtype=float)
        ku = data.get("knots_u", [])
        kv = data.get("knots_v", [])
        du = data.get("degree_u", 1)
        dv = data.get("degree_v", 1)
        self._set_net_and_weights(net, w, ku, kv, du, dv)

    # Draw the NURBS surface; silent=True skips error dialogs (used for display-option redraws)
    def _draw(self, silent: bool = False):
        try:
            net, w, ku, kv, du, dv = self._read_inputs()
            u_vals = sample_parameters(ku, 40, du)
            v_vals = sample_parameters(kv, 15, dv)
            X, Y, Z = nurbs_surface(ku, kv, du, dv, net, w, u_values=u_vals, v_values=v_vals)
        except Exception as e:
            if not silent:
                QMessageBox.warning(self, "Error", str(e))
            return

        nu, nv, _ = net.shape
        self._figure.clear()
        ax = self._figure.add_subplot(111, projection="3d")
        ax.computed_zorder = False  # draw in call order so knot lines stay visible on the surface

        if self._show_surface_chk.isChecked():
            ax.plot_surface(X, Y, Z,
                            color=self._surface_color_combo.currentText(),
                            alpha=self._surface_alpha_spin.value(),
                            edgecolor="gray", linewidth=0.3)

        if self._show_points_chk.isChecked():
            ax.plot(
                net[:, :, 0].flatten(),
                net[:, :, 1].flatten(),
                net[:, :, 2].flatten(),
                linestyle="None", marker="o",
                color=self._point_color_combo.currentText(),
                markersize=self._point_size_spin.value(),
                label="Control points",
            )

        if self._show_lines_chk.isChecked():
            # All net lines in one artist, separated by NaN
            lx, ly, lz = [], [], []
            for i in range(nu):
                lx.extend(net[i, :, 0].tolist() + [np.nan])
                ly.extend(net[i, :, 1].tolist() + [np.nan])
                lz.extend(net[i, :, 2].tolist() + [np.nan])
            for j in range(nv):
                lx.extend(net[:, j, 0].tolist() + [np.nan])
                ly.extend(net[:, j, 1].tolist() + [np.nan])
                lz.extend(net[:, j, 2].tolist() + [np.nan])
            ax.plot(lx, ly, lz,
                    color=self._line_color_combo.currentText(),
                    linestyle=self._line_style_combo.currentData() or "--",
                    linewidth=self._line_width_spin.value(), alpha=0.8,
                    label="Control net lines")

        if self._show_knots_chk.isChecked():
            draw_polylines(ax, element_mesh_lines(ku, kv, du, dv, net, w), True,
                           color="black", linewidth=1.2, label="Element boundaries")

        if self._equal_axes_chk.isChecked():
            pts = np.concatenate([net.reshape(-1, 3),
                                  np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])])
            lo, hi = pts.min(axis=0), pts.max(axis=0)
            span = hi - lo
            biggest = max(float(span.max()), 1e-12)
            span = np.maximum(span, 0.05 * biggest)
            centre = (lo + hi) / 2.0
            ax.set_xlim(centre[0] - span[0] / 2, centre[0] + span[0] / 2)
            ax.set_ylim(centre[1] - span[1] / 2, centre[1] + span[1] / 2)
            ax.set_zlim(centre[2] - span[2] / 2, centre[2] + span[2] / 2)
            ax.set_box_aspect(tuple(span))

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"NURBS Surface (deg {du}x{dv}, {nu}x{nv} net)")
        if ax.get_legend_handles_labels()[1]:  # only when something is labelled
            ax.legend()
        self._canvas.draw()
        self._has_plot = True
        neu = len(element_boundaries(ku, du)) - 1
        nev = len(element_boundaries(kv, dv)) - 1
        self._status(f"NURBS surface drawn: degree {du}x{dv}, {nu}x{nv} control net, "
                     f"{neu} element(s) in u x {nev} in v = {neu * nev}")
