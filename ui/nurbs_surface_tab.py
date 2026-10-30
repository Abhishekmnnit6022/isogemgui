"""NURBS Surface tab for editing a control net with weights and displaying a rotatable 3D surface."""

from __future__ import annotations

import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QLineEdit,
    QSpinBox, QPushButton, QTableWidget, QTableWidgetItem, QHeaderView,
    QMessageBox, QSplitter, QCheckBox,
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from core.nurbs import nurbs_surface, nurbs_cylinder
from core.validation import (
    validate_knots, validate_bspline_params, validate_surface_net,
)


class NurbsSurfaceTab(QWidget):
    """Tab widget for NURBS surfaces with rational weights and 3D view."""

    def __init__(self, status_callback=None):
        super().__init__()
        self._status = status_callback or (lambda msg: None)
        self._init_ui()

    # Build the layout
    def _init_ui(self):
        main = QHBoxLayout(self)

        left = QVBoxLayout()

        # Grid size
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
        self._cols_spin.setValue(2)
        size_row.addWidget(self._cols_spin)

        resize_btn = QPushButton("Resize Table")
        resize_btn.clicked.connect(self._resize_table)
        size_row.addWidget(resize_btn)
        dim_layout.addLayout(size_row)

        left.addWidget(dim_group)

        # Parameters: degrees and knots
        param_group = QGroupBox("NURBS Parameters")
        param_layout = QVBoxLayout(param_group)

        deg_row = QHBoxLayout()
        deg_row.addWidget(QLabel("Degree &U:"))
        self._deg_u_spin = QSpinBox()
        self._deg_u_spin.setRange(1, 15)
        self._deg_u_spin.setValue(2)
        deg_row.addWidget(self._deg_u_spin)

        deg_row.addWidget(QLabel("Degree &V:"))
        self._deg_v_spin = QSpinBox()
        self._deg_v_spin.setRange(1, 15)
        self._deg_v_spin.setValue(1)
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

        left.addWidget(param_group)

        # Coordinates table
        coord_group = QGroupBox("Control Net Coordinates")
        coord_layout = QVBoxLayout(coord_group)
        self._coord_table = QTableWidget(0, 5)
        self._coord_table.setHorizontalHeaderLabels(["Row (i)", "Col (j)", "X", "Y", "Z"])
        self._coord_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        coord_layout.addWidget(self._coord_table)
        left.addWidget(coord_group)

        # Weights table
        wt_group = QGroupBox("Weights")
        wt_layout = QVBoxLayout(wt_group)
        self._wt_table = QTableWidget(0, 3)
        self._wt_table.setHorizontalHeaderLabels(["Row (i)", "Col (j)", "Weight"])
        self._wt_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        wt_layout.addWidget(self._wt_table)
        left.addWidget(wt_group)

        # Buttons row
        btn_row = QHBoxLayout()
        cyl_btn = QPushButton("Load C&ylinder Example")
        cyl_btn.clicked.connect(self._load_cylinder)
        btn_row.addWidget(cyl_btn)
        left.addLayout(btn_row)

        # Options
        opt_group = QGroupBox("Display Options")
        opt_layout = QVBoxLayout(opt_group)
        self._show_net_chk = QCheckBox("Show control net")
        self._show_net_chk.setChecked(True)
        opt_layout.addWidget(self._show_net_chk)
        left.addWidget(opt_group)

        draw_btn = QPushButton("&Draw Surface")
        draw_btn.clicked.connect(self._draw)
        left.addWidget(draw_btn)
        left.addStretch()

        left_widget = QWidget()
        left_widget.setLayout(left)

        # 3D plot
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

    # Parse knot text field
    def _parse_knots(self, edit: QLineEdit) -> list:
        text = edit.text().strip().replace(",", " ")
        if not text:
            raise ValueError("Knot vector is empty.")
        tokens = text.split()
        try:
            raw = [float(t) for t in tokens]
        except ValueError:
            raise ValueError("All knot values must be numbers.")
        return validate_knots(raw)

    # Rebuild coordinate and weight tables
    def _resize_table(self):
        nu = self._rows_spin.value()
        nv = self._cols_spin.value()

        # Save existing coordinate data
        existing_coords = {}
        for r in range(self._coord_table.rowCount()):
            try:
                i = int(self._coord_table.item(r, 0).text())
                j = int(self._coord_table.item(r, 1).text())
                x = self._coord_table.item(r, 2).text()
                y = self._coord_table.item(r, 3).text()
                z = self._coord_table.item(r, 4).text()
                existing_coords[(i, j)] = (x, y, z)
            except Exception:
                pass

        existing_wts = {}
        for r in range(self._wt_table.rowCount()):
            try:
                i = int(self._wt_table.item(r, 0).text())
                j = int(self._wt_table.item(r, 1).text())
                w = self._wt_table.item(r, 2).text()
                existing_wts[(i, j)] = w
            except Exception:
                pass

        total = nu * nv
        self._coord_table.setRowCount(total)
        self._wt_table.setRowCount(total)
        idx = 0
        for i in range(nu):
            for j in range(nv):
                # Coordinate table
                item_i = QTableWidgetItem(str(i))
                item_i.setFlags(item_i.flags() & ~Qt.ItemIsEditable)
                item_j = QTableWidgetItem(str(j))
                item_j.setFlags(item_j.flags() & ~Qt.ItemIsEditable)
                self._coord_table.setItem(idx, 0, item_i)
                self._coord_table.setItem(idx, 1, item_j)

                if (i, j) in existing_coords:
                    x, y, z = existing_coords[(i, j)]
                else:
                    x, y, z = str(float(i)), str(float(j)), "0.0"
                self._coord_table.setItem(idx, 2, QTableWidgetItem(x))
                self._coord_table.setItem(idx, 3, QTableWidgetItem(y))
                self._coord_table.setItem(idx, 4, QTableWidgetItem(z))

                # Weight table
                wi = QTableWidgetItem(str(i))
                wi.setFlags(wi.flags() & ~Qt.ItemIsEditable)
                wj = QTableWidgetItem(str(j))
                wj.setFlags(wj.flags() & ~Qt.ItemIsEditable)
                self._wt_table.setItem(idx, 0, wi)
                self._wt_table.setItem(idx, 1, wj)

                wval = existing_wts.get((i, j), "1.0")
                self._wt_table.setItem(idx, 2, QTableWidgetItem(wval))
                idx += 1

    # Read control net from coordinate table
    def _get_net(self):
        nu = self._rows_spin.value()
        nv = self._cols_spin.value()
        expected = nu * nv
        if self._coord_table.rowCount() != expected:
            raise ValueError(f"Table rows ({self._coord_table.rowCount()}) don't match grid ({nu}x{nv}). Click Resize Table.")
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
                idx += 1
        return w

    # Populate tables from net and weights arrays
    def _set_net_and_weights(self, net, weights, knots_u, knots_v, degree_u, degree_v):
        net = np.asarray(net)
        weights = np.asarray(weights)
        nu, nv = net.shape[0], net.shape[1]

        self._rows_spin.setValue(nu)
        self._cols_spin.setValue(nv)
        self._deg_u_spin.setValue(degree_u)
        self._deg_v_spin.setValue(degree_v)
        self._knots_u_edit.setText(", ".join(f"{k:g}" for k in knots_u))
        self._knots_v_edit.setText(", ".join(f"{k:g}" for k in knots_v))

        total = nu * nv
        self._coord_table.setRowCount(total)
        self._wt_table.setRowCount(total)
        idx = 0
        for i in range(nu):
            for j in range(nv):
                # Coord table
                item_i = QTableWidgetItem(str(i))
                item_i.setFlags(item_i.flags() & ~Qt.ItemIsEditable)
                item_j = QTableWidgetItem(str(j))
                item_j.setFlags(item_j.flags() & ~Qt.ItemIsEditable)
                self._coord_table.setItem(idx, 0, item_i)
                self._coord_table.setItem(idx, 1, item_j)
                self._coord_table.setItem(idx, 2, QTableWidgetItem(str(net[i, j, 0])))
                self._coord_table.setItem(idx, 3, QTableWidgetItem(str(net[i, j, 1])))
                self._coord_table.setItem(idx, 4, QTableWidgetItem(str(net[i, j, 2])))

                # Weight table
                wi = QTableWidgetItem(str(i))
                wi.setFlags(wi.flags() & ~Qt.ItemIsEditable)
                wj = QTableWidgetItem(str(j))
                wj.setFlags(wj.flags() & ~Qt.ItemIsEditable)
                self._wt_table.setItem(idx, 0, wi)
                self._wt_table.setItem(idx, 1, wj)
                self._wt_table.setItem(idx, 2, QTableWidgetItem(str(weights[i, j])))
                idx += 1

    # Load cylinder preset
    def _load_cylinder(self):
        net, w, ku, kv, du, dv = nurbs_cylinder()
        self._set_net_and_weights(net, w, ku, kv, du, dv)
        self._status("Loaded NURBS cylinder example")

    # get_data for JSON save
    def get_data(self) -> dict:
        net = self._get_net()
        w = self._get_weights()
        ku = self._parse_knots(self._knots_u_edit)
        kv = self._parse_knots(self._knots_v_edit)
        du = self._deg_u_spin.value()
        dv = self._deg_v_spin.value()
        validate_bspline_params(ku, du, net.shape[0])
        validate_bspline_params(kv, dv, net.shape[1])
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

    # Draw the NURBS surface
    def _draw(self):
        try:
            net = self._get_net()
            w = self._get_weights()
            ku = self._parse_knots(self._knots_u_edit)
            kv = self._parse_knots(self._knots_v_edit)
            du = self._deg_u_spin.value()
            dv = self._deg_v_spin.value()
            validate_bspline_params(ku, du, net.shape[0])
            validate_bspline_params(kv, dv, net.shape[1])
            X, Y, Z = nurbs_surface(ku, kv, du, dv, net, w, n_eval_u=30, n_eval_v=30)
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))
            return

        nu, nv, _ = net.shape
        self._figure.clear()
        ax = self._figure.add_subplot(111, projection="3d")

        # Surface
        ax.plot_surface(X, Y, Z, color="cyan", alpha=0.7, edgecolor="gray", linewidth=0.5)

        # Control net
        if self._show_net_chk.isChecked():
            # Points
            ax.plot(
                net[:, :, 0].flatten(),
                net[:, :, 1].flatten(),
                net[:, :, 2].flatten(),
                linestyle="None", marker="o", color="blue", markersize=5,
                label="Control points",
            )
            # Lines via NaN separators
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
                    color="blue", linestyle="--", linewidth=1.0, alpha=0.8,
                    label="Control net lines")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"NURBS Surface (deg {du}x{dv}, {nu}x{nv} net)")
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend()
        self._canvas.draw()
        self._status(f"NURBS surface drawn: degree {du}x{dv}, {nu}x{nv} control net")
