"""IGA Analysis tab: load a NURBS surface, refine it, set edge conditions and export.

Five steps on the left, plot and results on the right. No solver."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
from matplotlib import colormaps
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QFontDatabase
from PyQt5.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QFileDialog, QGroupBox, QHBoxLayout,
    QHeaderView, QLabel, QLineEdit, QMessageBox, QPlainTextEdit, QPushButton,
    QScrollArea, QSpinBox, QSplitter, QTableWidget, QTableWidgetItem, QTabWidget,
    QVBoxLayout, QWidget,
)

from core import io_json
from core.analysis import (
    CONDITIONS, EDGE_NAMES, STATUS_DEGENERATE, STATUS_DISTORTED, STATUS_INVERTED, STATUS_OK,
    AnalysisModel, analyze_surface, element_mesh_lines, export_json, export_matlab,
    jacobian_field,
)
from core.nurbs import (
    nurbs_cylinder, nurbs_surface, nurbs_t_shape, plate_with_hole, quarter_annulus,
    sample_parameters,
)
from core.refinement import insert_knot_surface, refine_surface_uniform


def _short_knots(knots) -> str:
    """Knot list for reading (6 significant digits); the exported files keep full precision."""
    return ", ".join(f"{k:.6g}" for k in knots)
from ui.widgets import draw_polylines

EXAMPLES: Dict[str, Callable] = {
    "Plate with a hole (2D)": plate_with_hole,
    "Quarter annulus (2D)": quarter_annulus,
    "Cylinder (3D surface)": nurbs_cylinder,
    "T-shape (3D surface)": nurbs_t_shape,
}
MAX_ELEMENTS = 10000         # refinement beyond this is refused to keep the app responsive
MAX_NUMBERED_ELEMENTS = 150  # element numbers are only written on smaller meshes
CONDITION_COLORS = {"Fixed (Dirichlet)": "red", "Load (Neumann)": "green", "Symmetry": "purple"}
STATUS_BACKGROUND = {
    STATUS_OK: QColor(215, 245, 215),
    STATUS_DISTORTED: QColor(255, 228, 170),
    STATUS_INVERTED: QColor(255, 190, 190),
    STATUS_DEGENERATE: QColor(255, 190, 190),
}


class AnalysisTab(QWidget):
    """Tab that turns a NURBS surface into an analysis-ready IGA model."""

    def __init__(self, status_callback=None, geometry_provider: Optional[Callable[[], dict]] = None):
        super().__init__()
        self._status = status_callback or (lambda msg: None)
        self._geometry_provider = geometry_provider
        self._geom: Optional[dict] = None       # current (possibly refined) geometry
        self._original: Optional[dict] = None   # geometry as loaded in step 1
        self._source = ""
        self._model: Optional[AnalysisModel] = None
        self._conditions = {e: "Free" for e in EDGE_NAMES}
        self._highlight = None                   # ("edge", name) or ("element", index)
        self._init_ui()
        self._show_empty()

    # ------------------------------------------------------------ layout

    # Left: scrollable column of the five step boxes. Right: plot above the results tabs
    def _init_ui(self):
        main = QHBoxLayout(self)

        steps = QVBoxLayout()
        steps.addWidget(self._build_step1())
        steps.addWidget(self._build_step2())
        steps.addWidget(self._build_step3())
        steps.addWidget(self._build_step4())
        steps.addWidget(self._build_step5())
        steps.addStretch()
        steps_widget = QWidget()
        steps_widget.setLayout(steps)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(steps_widget)

        # Right: plot on top, results below
        plot_box = QWidget()
        plot_layout = QVBoxLayout(plot_box)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        self._figure = Figure()
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._toolbar = NavigationToolbar2QT(self._canvas, self)
        plot_layout.addWidget(self._toolbar)
        plot_layout.addWidget(self._canvas)

        self._results = QTabWidget()
        self._summary = QPlainTextEdit()
        self._summary.setReadOnly(True)
        self._summary.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
        self._summary.setLineWrapMode(QPlainTextEdit.NoWrap)
        self._results.addTab(self._summary, "Summary")
        self._elem_table = QTableWidget(0, 8)
        self._elem_table.setHorizontalHeaderLabels(
            ["Element", "u range", "v range", "Area", "Control points", "min J", "max J", "Status"])
        self._elem_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self._elem_table.horizontalHeader().setStretchLastSection(True)
        self._elem_table.verticalHeader().setVisible(False)
        self._elem_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._elem_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._elem_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._elem_table.itemSelectionChanged.connect(self._on_element_selected)
        self._results.addTab(self._elem_table, "Elements")

        right = QSplitter(Qt.Vertical)
        right.addWidget(plot_box)
        right.addWidget(self._results)
        right.setStretchFactor(0, 3)
        right.setStretchFactor(1, 2)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(scroll)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        splitter.setSizes([440, 660])
        main.addWidget(splitter)

    # Step 1 box: where the geometry comes from
    def _build_step1(self) -> QGroupBox:
        box = QGroupBox("Step 1 - Geometry")
        lay = QVBoxLayout(box)
        self._source_label = QLabel()
        self._source_label.setWordWrap(True)
        lay.addWidget(self._source_label)

        row = QHBoxLayout()
        self._from_tab_btn = QPushButton("Use NURBS Surface Tab")
        self._from_tab_btn.setToolTip("Analyse the surface currently entered in the NURBS Surface tab")
        self._from_tab_btn.clicked.connect(self._load_from_surface_tab)
        self._from_tab_btn.setEnabled(self._geometry_provider is not None)
        row.addWidget(self._from_tab_btn)
        open_btn = QPushButton("Open File...")
        open_btn.setToolTip("Open a nurbs_surface or iga_analysis_model JSON file")
        open_btn.clicked.connect(lambda: self._open_file())
        row.addWidget(open_btn)
        lay.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Example:"))
        self._example_combo = QComboBox()
        self._example_combo.addItems(list(EXAMPLES))
        row.addWidget(self._example_combo, 1)
        ex_btn = QPushButton("Load Example")
        ex_btn.clicked.connect(self._load_example)
        row.addWidget(ex_btn)
        lay.addLayout(row)
        return box

    # Step 2 box: split elements / insert a knot / undo
    def _build_step2(self) -> QGroupBox:
        box = QGroupBox("Step 2 - Refinement (shape is kept)")
        lay = QVBoxLayout(box)

        row = QHBoxLayout()
        row.addWidget(QLabel("Split every element:"))
        for text, direction in (("In U", "u"), ("In V", "v"), ("Both", "both")):
            btn = QPushButton(text)
            btn.clicked.connect(lambda _=False, d=direction: self._split(d))
            row.addWidget(btn)
        lay.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Insert knot in"))
        self._dir_combo = QComboBox()
        self._dir_combo.addItems(["U", "V"])
        row.addWidget(self._dir_combo)
        row.addWidget(QLabel("at"))
        self._knot_edit = QLineEdit()
        self._knot_edit.setPlaceholderText("value, e.g. 0.25")
        row.addWidget(self._knot_edit, 1)
        lay.addLayout(row)
        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(QLabel("Times:"))
        self._times_spin = QSpinBox()
        self._times_spin.setRange(1, 15)
        row.addWidget(self._times_spin)
        ins_btn = QPushButton("Insert")
        ins_btn.clicked.connect(self._insert)
        row.addWidget(ins_btn)
        lay.addLayout(row)

        reset_btn = QPushButton("Reset to Original Geometry")
        reset_btn.setToolTip("Undo all refinement done in this step")
        reset_btn.clicked.connect(self._reset_refinement)
        lay.addWidget(reset_btn)
        self._mesh_label = QLabel()
        self._mesh_label.setWordWrap(True)
        lay.addWidget(self._mesh_label)
        return box

    # Step 3 box: one table row per edge with a condition drop-down
    def _build_step3(self) -> QGroupBox:
        box = QGroupBox("Step 3 - Boundary Conditions")
        lay = QVBoxLayout(box)
        self._bc_table = QTableWidget(len(EDGE_NAMES), 3)
        self._bc_table.setHorizontalHeaderLabels(["Edge", "Control points", "Condition"])
        self._bc_table.verticalHeader().setVisible(False)
        self._bc_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self._bc_table.horizontalHeader().setStretchLastSection(True)
        self._bc_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._bc_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._bc_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._bc_table.itemSelectionChanged.connect(self._on_edge_selected)
        self._bc_combos: Dict[str, QComboBox] = {}
        for r, name in enumerate(EDGE_NAMES):
            self._bc_table.setItem(r, 0, QTableWidgetItem(name))
            self._bc_table.setItem(r, 1, QTableWidgetItem("-"))
            combo = QComboBox()
            combo.addItems(CONDITIONS)
            combo.currentTextChanged.connect(lambda text, n=name: self._on_condition_changed(n, text))
            self._bc_table.setCellWidget(r, 2, combo)
            self._bc_combos[name] = combo
        self._bc_table.setMinimumHeight(140)
        lay.addWidget(self._bc_table)
        hint = QLabel("Select a row to highlight that edge and its control points on the plot.")
        hint.setWordWrap(True)
        lay.addWidget(hint)
        return box

    # Step 4 box: what the plot shows (each checkbox redraws)
    def _build_step4(self) -> QGroupBox:
        box = QGroupBox("Step 4 - Display")
        lay = QVBoxLayout(box)
        self._chk_mesh = QCheckBox("Element mesh (knot lines)")
        self._chk_jac = QCheckBox("Jacobian colours (mesh quality)")
        self._chk_bc = QCheckBox("Boundary conditions")
        self._chk_numbers = QCheckBox("Element numbers")
        self._chk_net = QCheckBox("Control net")
        self._chk_gauss = QCheckBox("Gauss points")
        for chk, on in ((self._chk_mesh, True), (self._chk_jac, True), (self._chk_bc, True),
                        (self._chk_numbers, True), (self._chk_net, False), (self._chk_gauss, False)):
            chk.setChecked(on)
            chk.stateChanged.connect(lambda _=0: self._draw())
            lay.addWidget(chk)
        return box

    # Step 5 box: export buttons
    def _build_step5(self) -> QGroupBox:
        box = QGroupBox("Step 5 - Export")
        lay = QVBoxLayout(box)
        row = QHBoxLayout()
        json_btn = QPushButton("Export JSON...")
        json_btn.clicked.connect(lambda: self.export_json())
        row.addWidget(json_btn)
        m_btn = QPushButton("Export MATLAB (.m)...")
        m_btn.clicked.connect(lambda: self.export_matlab())
        row.addWidget(m_btn)
        lay.addLayout(row)
        hint = QLabel("JSON can be opened again here (File -> Open). The .m script uses the "
                      "NURBS toolbox (GeoPDEs) geometry layout plus elements and boundaries.")
        hint.setWordWrap(True)
        lay.addWidget(hint)
        return box

    # ------------------------------------------------------------ step 1: geometry

    def set_geometry(self, net, weights, knots_u, knots_v, degree_u, degree_v,
                     source: str, conditions: Optional[Dict[str, str]] = None):
        """Load a NURBS surface as the new starting geometry (raises ValueError if invalid)."""
        geom = {"ku": list(map(float, knots_u)), "kv": list(map(float, knots_v)),
                "du": int(degree_u), "dv": int(degree_v),
                "net": np.asarray(net, dtype=float), "w": np.asarray(weights, dtype=float)}
        conds = {e: "Free" for e in EDGE_NAMES}
        conds.update(conditions or {})
        analyze_surface(geom["ku"], geom["kv"], geom["du"], geom["dv"], geom["net"], geom["w"],
                        conditions=conds)  # validates before anything changes
        self._original = dict(geom)
        self._geom = geom
        self._source = source
        self._conditions = conds
        self._highlight = None
        self._rebuild()

    # Analyse whatever is currently entered in the NURBS Surface tab
    def _load_from_surface_tab(self):
        try:
            data = self._geometry_provider()
            self.set_geometry(data["control_points"], data["weights"], data["knots_u"],
                              data["knots_v"], data["degree_u"], data["degree_v"],
                              "NURBS Surface tab")
        except Exception as e:
            QMessageBox.warning(self, "Use NURBS Surface Tab",
                                f"The NURBS Surface tab does not hold a valid surface:\n{e}")

    # Open a nurbs_surface or iga_analysis_model JSON file (path given in tests)
    def _open_file(self, path: Optional[str] = None):
        if not path:
            path, _ = QFileDialog.getOpenFileName(
                self, "Open Surface for Analysis", "", "JSON Files (*.json);;All Files (*)")
            if not path:
                return
        try:
            data = io_json.load(path)
            if data["type"] not in ("nurbs_surface", "iga_analysis_model"):
                raise ValueError(f"This file holds a '{data['type']}'. The IGA Analysis tab needs a "
                                 f"NURBS surface (nurbs_surface or iga_analysis_model).")
            self.set_data(data, source=Path(path).name)
        except Exception as e:
            QMessageBox.critical(self, "Open File", f"Failed to open file:\n{e}")

    # Load the example chosen in the drop-down
    def _load_example(self):
        name = self._example_combo.currentText()
        net, w, ku, kv, du, dv = EXAMPLES[name]()
        self.set_geometry(net, w, ku, kv, du, dv, f"Example: {name}")

    # ------------------------------------------------------------ step 2: refinement

    # Refinement and export need a geometry from Step 1
    def _require_model(self):
        if self._geom is None:
            raise ValueError("Load a geometry first (Step 1).")

    # Make a refined geometry the current one and refresh everything
    def _apply(self, ku, kv, net, w, action: str):
        g = self._geom
        n_el = (len(set(ku)) - 1) * (len(set(kv)) - 1)
        if n_el > MAX_ELEMENTS:
            raise ValueError(f"The mesh would have about {n_el} elements; the limit is {MAX_ELEMENTS}.")
        self._geom = {"ku": list(ku), "kv": list(kv), "du": g["du"], "dv": g["dv"],
                      "net": np.asarray(net), "w": np.asarray(w)}
        self._highlight = None
        self._rebuild()
        m = self._model
        self._status(f"{action}: now {m.n_elements_u} x {m.n_elements_v} = {len(m.elements)} elements, "
                     f"{m.n_control_points} control points (shape unchanged)")

    # Split every element in two along u, v or both
    def _split(self, direction: str):
        try:
            self._require_model()
            g = self._geom
            bu = len(set(g["ku"])) - 1
            bv = len(set(g["kv"])) - 1
            factor = {"u": (2, 1), "v": (1, 2), "both": (2, 2)}[direction]
            if bu * factor[0] * bv * factor[1] > MAX_ELEMENTS:
                raise ValueError(f"Splitting would give about {bu * factor[0] * bv * factor[1]} "
                                 f"elements; the limit is {MAX_ELEMENTS}.")
            ku, kv, net, w = refine_surface_uniform(g["ku"], g["kv"], g["du"], g["dv"],
                                                    g["net"], g["w"], direction)
            self._apply(ku, kv, net, w, f"Split every element in {direction.upper()}"
                        if direction != "both" else "Split every element in U and V")
        except Exception as e:
            QMessageBox.warning(self, "Split Elements", str(e))

    # Insert the typed knot value in the chosen direction
    def _insert(self):
        try:
            self._require_model()
            text = self._knot_edit.text().strip()
            if not text:
                raise ValueError("Enter the knot value to insert.")
            try:
                value = float(text)
            except ValueError:
                raise ValueError(f"Knot value '{text}' is not a number.") from None
            direction = self._dir_combo.currentText().lower()
            times = self._times_spin.value()
            g = self._geom
            ku, kv, net, w = insert_knot_surface(g["ku"], g["kv"], g["du"], g["dv"], g["net"], g["w"],
                                                 direction, value, times)
            self._apply(ku, kv, net, w, f"Inserted knot {direction} = {value:g} ({times}x)")
        except Exception as e:
            QMessageBox.warning(self, "Insert Knot", str(e))

    # Go back to the geometry as loaded in Step 1
    def _reset_refinement(self):
        if self._original is None:
            return
        self._geom = dict(self._original)
        self._highlight = None
        self._rebuild()
        self._status("Refinement undone: back to the original geometry")

    # ------------------------------------------------------------ step 3: boundaries

    # A condition drop-down changed: store it and refresh summary and plot
    def _on_condition_changed(self, name: str, text: str):
        self._conditions[name] = text
        if self._model is not None:
            self._model.conditions[name] = text
            self._update_summary()
            self._draw()

    # Edge row selected: highlight that edge and its control points
    def _on_edge_selected(self):
        rows = {i.row() for i in self._bc_table.selectedIndexes()}
        if rows and self._model is not None:
            self._elem_table.clearSelection()
            self._highlight = ("edge", EDGE_NAMES[min(rows)])
            self._draw()

    # Element row selected: outline that element and its control points
    def _on_element_selected(self):
        rows = {i.row() for i in self._elem_table.selectedIndexes()}
        if rows and self._model is not None:
            self._bc_table.clearSelection()
            self._highlight = ("element", min(rows))
            self._draw()

    # ------------------------------------------------------------ model and results

    # Recompute the analysis model and refresh labels, tables, summary and plot
    def _rebuild(self):
        g = self._geom
        self._model = analyze_surface(g["ku"], g["kv"], g["du"], g["dv"], g["net"], g["w"],
                                      conditions=self._conditions)
        m = self._model
        self._source_label.setText(
            f"<b>{self._source}</b><br>{'Planar (2D)' if m.planar else '3D'} surface, "
            f"degree {m.degree_u} x {m.degree_v}, {m.n_rows} x {m.n_cols} control points")
        self._mesh_label.setText(
            f"Mesh: {m.n_elements_u} x {m.n_elements_v} = {len(m.elements)} elements, "
            f"{m.n_control_points} control points")
        for r, name in enumerate(EDGE_NAMES):
            self._bc_table.item(r, 1).setText(str(len(m.boundaries[name])))
            combo = self._bc_combos[name]
            combo.blockSignals(True)
            combo.setCurrentText(m.conditions[name])
            combo.blockSignals(False)
        self._fill_element_table()
        self._update_summary()
        self._draw()
        q = m.quality_counts()
        self._status(f"IGA model: {len(m.elements)} elements, {m.n_control_points} control points, "
                     f"quality {q[STATUS_OK]} OK / {q[STATUS_DISTORTED]} distorted / "
                     f"{q[STATUS_INVERTED] + q[STATUS_DEGENERATE]} inverted or degenerate")

    # One row per element; the Status cell is coloured
    def _fill_element_table(self):
        m = self._model
        t = self._elem_table
        t.blockSignals(True)
        t.clearSelection()
        t.setRowCount(len(m.elements))
        for r, e in enumerate(m.elements):
            cells = [
                str(e.index),
                f"{e.u_range[0]:.4g} - {e.u_range[1]:.4g}",
                f"{e.v_range[0]:.4g} - {e.v_range[1]:.4g}",
                f"{e.area:.6g}",
                " ".join(str(c) for c in e.control_points),
                f"{e.jacobian_min:.4g}",
                f"{e.jacobian_max:.4g}",
                e.status,
            ]
            for c, text in enumerate(cells):
                t.setItem(r, c, QTableWidgetItem(text))
            t.item(r, 7).setBackground(STATUS_BACKGROUND[e.status])
        t.blockSignals(False)

    # Plain-text report shown in the Summary tab
    def summary_text(self) -> str:
        m = self._model
        if m is None:
            return "No geometry loaded yet."
        q = m.quality_counts()
        d = m.dofs()
        jmins = [e.jacobian_min for e in m.elements]
        jmaxs = [e.jacobian_max for e in m.elements]
        jname = "det J" if m.planar else "area stretch |S_u x S_v|"
        lines = [
            f"Geometry:   {self._source}",
            f"Surface:    {'planar (2D domain, z constant)' if m.planar else '3D surface (shell-like)'}",
            f"Degrees:    p = {m.degree_u} (u), q = {m.degree_v} (v)",
            f"Knots U:    {_short_knots(m.knots_u)}",
            f"Knots V:    {_short_knots(m.knots_v)}",
            "",
            f"Control points:            {m.n_rows} x {m.n_cols} = {m.n_control_points}",
            f"Elements:                  {m.n_elements_u} (u) x {m.n_elements_v} (v) = {len(m.elements)}",
            f"Control points / element:  {m.degree_u + 1} x {m.degree_v + 1} = "
            f"{(m.degree_u + 1) * (m.degree_v + 1)}",
            f"Gauss points / element:    {m.gauss_u} x {m.gauss_v} = {m.gauss_u * m.gauss_v} "
            f"(total {len(m.gauss_points)})",
            "",
            "Degrees of freedom (unknowns a solver would have):",
            f"  scalar field (heat, potential):  {d['scalar_field']}",
            f"  2D elasticity (u_x, u_y):        {d['elasticity_2d']}",
            f"  3D / shell (u_x, u_y, u_z):      {d['elasticity_3d']}",
            "",
            f"Total area (Gauss quadrature):   {m.total_area:.10g}",
            f"Mesh quality: {q[STATUS_OK]} OK, {q[STATUS_DISTORTED]} distorted, "
            f"{q[STATUS_INVERTED]} inverted, {q[STATUS_DEGENERATE]} degenerate",
            f"  {jname}: min {min(jmins):.4g}, max {max(jmaxs):.4g}",
            "",
            "Boundary conditions:",
        ]
        for name in EDGE_NAMES:
            lines.append(f"  {name}:  {m.conditions[name]:<18}  "
                         f"control points {' '.join(map(str, m.boundaries[name]))}")
        notes = []
        if not m.clamped_u or not m.clamped_v:
            notes.append("Knots are not clamped in " + " and ".join(
                d for d, c in (("u", m.clamped_u), ("v", m.clamped_v)) if not c)
                + ": edge control-point sets only describe the edge exactly for clamped knots.")
        if q[STATUS_INVERTED] or q[STATUS_DEGENERATE]:
            notes.append("Some elements are inverted or degenerate (red in the Elements table): "
                         "a solver would fail or give wrong results. Fix the control net first.")
        if q[STATUS_DISTORTED]:
            notes.append(f"Some elements are strongly distorted (|J| varies more than "
                         f"10x inside the element); refining there usually helps.")
        if notes:
            lines += ["", "Notes:"] + [f"  - {n}" for n in notes]
        return "\n".join(lines)

    # Refresh the Summary tab
    def _update_summary(self):
        self._summary.setPlainText(self.summary_text())

    # ------------------------------------------------------------ plot

    # State before any geometry is loaded: short instructions instead of results
    def _show_empty(self):
        self._source_label.setText("No geometry loaded yet. Use one of the buttons below.")
        self._mesh_label.setText("Mesh: -")
        for r in range(len(EDGE_NAMES)):
            self._bc_table.item(r, 1).setText("-")
        self._elem_table.setRowCount(0)
        self._summary.setPlainText(
            "No geometry loaded yet.\n\n"
            "Step 1: pick a surface (NURBS Surface tab, a JSON file, or an example).\n"
            "Step 2: refine the mesh if needed - the shape stays exactly the same.\n"
            "Step 3: tag the edges with boundary conditions.\n"
            "Step 4: choose what the plot shows.\n"
            "Step 5: export the analysis model for a solver.")
        self._figure.clear()
        self._canvas.draw()

    # Points along one boundary edge (for highlighting and condition colours)
    def _edge_line(self, name: str) -> np.ndarray:
        m = self._model
        su = sample_parameters(m.knots_u, 80, m.degree_u)
        sv = sample_parameters(m.knots_v, 80, m.degree_v)
        u0, u1 = m.bounds_u[0], m.bounds_u[-1]
        v0, v1 = m.bounds_v[0], m.bounds_v[-1]
        uu, vv = {"u = 0": ([u0], sv), "u = 1": ([u1], sv),
                  "v = 0": (su, [v0]), "v = 1": (su, [v1])}[name]
        X, Y, Z = nurbs_surface(m.knots_u, m.knots_v, m.degree_u, m.degree_v, m.net, m.weights,
                                u_values=uu, v_values=vv)
        return np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    # Closed outline of one element in physical space
    def _element_outline(self, e) -> np.ndarray:
        m = self._model
        a, b = e.u_range
        c, d = e.v_range
        t = np.linspace(0.0, 1.0, 25)
        # Four sides walked in order: v = c (u: a->b), u = b (v: c->d), v = d (u: b->a), u = a (v: d->c)
        sides = [(a + (b - a) * t, [c]), ([b], c + (d - c) * t),
                 (b - (b - a) * t, [d]), ([a], d - (d - c) * t)]
        pts = []
        for uu, vv in sides:
            X, Y, Z = nurbs_surface(m.knots_u, m.knots_v, m.degree_u, m.degree_v, m.net,
                                    m.weights, u_values=uu, v_values=vv)
            pts.append(np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]))
        return np.concatenate(pts)

    # ax.plot for 2D or 3D axes from an (n, 3) point array
    def _plot(self, ax, pts, three_d, **style):
        if three_d:
            return ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], **style)
        return ax.plot(pts[:, 0], pts[:, 1], **style)

    # Redraw the plot from the current model and the Step 4 checkboxes
    def _draw(self):
        m = self._model
        if m is None:
            return
        three_d = not m.planar
        self._figure.clear()
        ax = self._figure.add_subplot(111, projection="3d") if three_d else self._figure.add_subplot(111)
        if three_d:
            ax.computed_zorder = False  # draw in call order so mesh lines stay on top of the surface

        # Jacobian colours (mesh quality)
        n_dir = max(m.n_elements_u, m.n_elements_v)
        X, Y, Z, J = jacobian_field(m, per_element=max(3, min(10, 240 // max(n_dir, 1))))
        if self._chk_jac.isChecked():
            jmin, jmax = float(J.min()), float(J.max())
            if jmax <= 0.0:
                norm, cmap = Normalize(jmin, min(jmax, -1e-300)), colormaps["Reds_r"]
            elif jmin < 0.0:
                norm, cmap = TwoSlopeNorm(vmin=jmin, vcenter=0.0, vmax=jmax), colormaps["RdYlGn"]
            else:
                norm, cmap = Normalize(jmin, jmax if jmax > jmin else jmin + 1e-12), colormaps["viridis"]
            if three_d:
                ax.plot_surface(X, Y, Z, facecolors=cmap(norm(J)), shade=False, linewidth=0,
                                antialiased=False, alpha=0.9)
            else:
                ax.pcolormesh(X, Y, J, shading="gouraud", cmap=cmap, norm=norm)
            label = "det J (negative = folded)" if m.planar else "area stretch |S_u x S_v|"
            self._figure.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, label=label,
                                  shrink=0.8)
        elif three_d:
            ax.plot_surface(X, Y, Z, color="lightgray", alpha=0.5, linewidth=0)

        if self._chk_mesh.isChecked():
            draw_polylines(ax, element_mesh_lines(m.knots_u, m.knots_v, m.degree_u, m.degree_v,
                                                  m.net, m.weights),
                           three_d, color="black", linewidth=0.9, label="Element boundaries")

        # Flag bad elements with a coloured outline
        for e in m.elements:
            if e.status in (STATUS_INVERTED, STATUS_DEGENERATE):
                self._plot(ax, self._element_outline(e), three_d, color="red", linewidth=2.5)
            elif e.status == STATUS_DISTORTED:
                self._plot(ax, self._element_outline(e), three_d, color="orange", linewidth=2.0)

        if self._chk_net.isChecked():
            lines = [m.net[i] for i in range(m.n_rows)] + [m.net[:, j] for j in range(m.n_cols)]
            draw_polylines(ax, lines, three_d, color="blue", linestyle="--", linewidth=0.8,
                           alpha=0.8, label="Control net")
            self._plot(ax, m.net.reshape(-1, 3), three_d, linestyle="None", marker="o",
                       color="blue", markersize=4)

        if self._chk_gauss.isChecked():
            self._plot(ax, m.gauss_points, three_d, linestyle="None", marker="x", color="red",
                       markersize=3, label="Gauss points")

        if self._chk_bc.isChecked():
            for name in EDGE_NAMES:
                cond = m.conditions[name]
                if cond in CONDITION_COLORS:
                    self._plot(ax, self._edge_line(name), three_d, color=CONDITION_COLORS[cond],
                               linewidth=3.5, label=f"{name}: {cond}")

        if self._chk_numbers.isChecked() and len(m.elements) <= MAX_NUMBERED_ELEMENTS:
            mu = (m.bounds_u[:-1] + m.bounds_u[1:]) / 2.0
            mv = (m.bounds_v[:-1] + m.bounds_v[1:]) / 2.0
            CX, CY, CZ = nurbs_surface(m.knots_u, m.knots_v, m.degree_u, m.degree_v, m.net,
                                       m.weights, u_values=mu, v_values=mv)
            for e in m.elements:
                x, y, z = CX[e.eu, e.ev], CY[e.eu, e.ev], CZ[e.eu, e.ev]
                if three_d:
                    ax.text(x, y, z, str(e.index), fontsize=8, ha="center")
                else:
                    ax.text(x, y, str(e.index), fontsize=8, ha="center", va="center")

        # Highlight selected edge or element
        if self._highlight:
            kind, key = self._highlight
            if kind == "edge":
                self._plot(ax, self._edge_line(key), three_d, color="orange", linewidth=5,
                           alpha=0.8, label=f"Selected edge {key}")
                pts = m.net.reshape(-1, 3)[m.boundaries[key]]
                self._plot(ax, pts, three_d, linestyle="None", marker="o", color="orange",
                           markeredgecolor="black", markersize=8)
            elif kind == "element" and key < len(m.elements):
                e = m.elements[key]
                self._plot(ax, self._element_outline(e), three_d, color="magenta", linewidth=3,
                           label=f"Selected element {e.index}")
                pts = m.net.reshape(-1, 3)[e.control_points]
                self._plot(ax, pts, three_d, linestyle="None", marker="o", color="magenta",
                           markeredgecolor="black", markersize=7)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        if three_d:
            ax.set_zlabel("Z")
            pts = m.net.reshape(-1, 3)
            span = np.maximum(np.ptp(pts, axis=0), 0.05 * max(float(np.ptp(pts, axis=0).max()), 1e-12))
            centre = (pts.min(axis=0) + pts.max(axis=0)) / 2.0
            ax.set_xlim(centre[0] - span[0] / 2, centre[0] + span[0] / 2)
            ax.set_ylim(centre[1] - span[1] / 2, centre[1] + span[1] / 2)
            ax.set_zlim(centre[2] - span[2] / 2, centre[2] + span[2] / 2)
            ax.set_box_aspect(tuple(span))
        else:
            ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(f"IGA mesh: {m.n_elements_u} x {m.n_elements_v} elements, "
                     f"degree {m.degree_u} x {m.degree_v}")
        if ax.get_legend_handles_labels()[1]:  # only when something is labelled
            # Below the plot, so it never hides part of the mesh
            ax.legend(fontsize="small", loc="upper center", bbox_to_anchor=(0.5, -0.12),
                      ncol=3, frameon=False)
            self._figure.subplots_adjust(bottom=0.25)
        self._canvas.draw()

    # ------------------------------------------------------------ step 5: export

    # Shared export code; asks for a file name when no path is given
    def _export(self, kind: str, path: Optional[str]):
        if self._model is None:
            QMessageBox.warning(self, "Export", "Nothing to export yet: load a geometry in Step 1.")
            return None
        if not path:
            filt = "JSON Files (*.json)" if kind == "json" else "MATLAB Scripts (*.m)"
            default = "iga_model.json" if kind == "json" else "iga_model.m"
            path, _ = QFileDialog.getSaveFileName(self, "Export Analysis Model", default, filt)
            if not path:
                return None
        try:
            (export_json if kind == "json" else export_matlab)(self._model, path)
        except Exception as e:
            QMessageBox.critical(self, "Export", f"Failed to export:\n{e}")
            return None
        self._status(f"Analysis model exported to {path}")
        QMessageBox.information(self, "Export Successful", f"Analysis model saved to:\n{path}")
        return path

    # Step 5: JSON export (returns the path, or None if cancelled/failed)
    def export_json(self, path: Optional[str] = None):
        return self._export("json", path)

    # Step 5: MATLAB .m export (returns the path, or None if cancelled/failed)
    def export_matlab(self, path: Optional[str] = None):
        return self._export("m", path)

    # ------------------------------------------------------------ File menu integration

    # File -> Save As: the full analysis model
    def get_data(self) -> dict:
        if self._model is None:
            raise ValueError("No analysis model yet: load a geometry in Step 1.")
        return self._model.to_dict()

    def set_data(self, data: dict, source: str = "Opened file"):
        """Load a validated nurbs_surface or iga_analysis_model dictionary (from io_json.load)."""
        if data.get("type") == "iga_analysis_model":
            g, conds = data["geometry"], data.get("conditions", {})
        else:
            g, conds = data, {}
        self.set_geometry(g["control_points"], g["weights"], g["knots_u"], g["knots_v"],
                          g["degree_u"], g["degree_v"], source, conditions=conds)

    # File -> New: forget the model and show the instructions again
    def reset(self):
        self._geom = self._original = self._model = None
        self._source = ""
        self._conditions = {e: "Free" for e in EDGE_NAMES}
        self._highlight = None
        for combo in self._bc_combos.values():
            combo.blockSignals(True)
            combo.setCurrentIndex(0)
            combo.blockSignals(False)
        self._show_empty()
