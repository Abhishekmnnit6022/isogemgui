# CAGD Suite - Computer-Aided Geometric Design

A unified desktop application for CAGD curves and surfaces: Bézier curves, Bézier surfaces, B-spline curves, and rational B-splines (NURBS curves & NURBS surfaces). Built with a pure-math `core/` package and a native PyQt5 / Matplotlib desktop interface.

---

## Folder Layout

```text
gui2/
├── main.py                  # Application launcher entry point
├── app.py                   # Main window with tabs, menu bar, and status bar
├── requirements.txt         # Project dependencies
├── README.md                # Project documentation and guide
├── documentation.txt        # Comprehensive technical documentation
├── core/                    # Pure-math geometry engine (no GUI dependencies)
│   ├── __init__.py
│   ├── bezier.py            # Bernstein basis, arbitrary-degree curves & surfaces
│   ├── bspline.py           # Cox-de Boor recursion, basis matrices, curve evaluation
│   ├── nurbs.py             # Rational basis, NURBS curves/surfaces, exact circle & cylinder
│   ├── knots.py             # Knot vector generators (open/clamped, uniform)
│   ├── validation.py        # Strict input validation (knots, degree, points, weights)
│   └── io_json.py           # Geometry JSON serialization (Bézier, B-spline, NURBS)
├── ui/                      # PyQt5 user interface components
│   ├── __init__.py
│   ├── bezier_tab.py        # 2D/3D Bézier curve tab
│   ├── surface_tab.py       # 3D Bézier surface tab
│   ├── bspline_tab.py       # B-spline tab with basis plot, curve draw & Excel export
│   ├── nurbs_curve_tab.py   # NURBS curve tab: weights, B-spline overlay, rational basis, circle & T-profile demos
│   ├── nurbs_surface_tab.py # NURBS surface tab: weight grid, cylinder & 3D T-shape demos, rotatable 3D
│   └── common.py            # Shared knot text formatting/parsing (lossless)
├── tests/                   # Pytest test suite (92 tests: math, JSON and headless GUI)
│   ├── test_bezier.py       # Partition of unity, endpoints, closed-forms, convex hull
│   ├── test_bspline.py      # B-spline partition of unity, endpoints, Matlab reference
│   ├── test_knots.py        # Knot generator validation
│   ├── test_io.py           # JSON round-trip serialization tests (including NURBS)
│   ├── test_nurbs.py        # Weights equivalence, circle/cylinder accuracy, partition of unity
│   ├── test_shapes.py       # T-shape/cylinder exactness, extrusion, sampling, circle fitting
│   ├── test_ui_nurbs.py     # Headless GUI tests for the NURBS tabs and main window
│   └── conftest.py          # Offscreen Qt application and message-box capture
├── examples/                # Sample geometry JSON files
│   ├── cubic_bezier_2d.json
│   ├── cubic_bezier_3d.json
│   ├── bicubic_surface.json
│   ├── bspline_deg2.json
│   ├── bspline_clamped_deg3.json
│   ├── circle.json          # 9-point exact quadratic NURBS circle
│   ├── quarter_circle.json  # 3-point quarter circle
│   ├── cylinder.json        # Degree 2x1 extruded NURBS cylinder
│   ├── t_profile.json       # Closed T cross-section (degree 2, exact rounded inner corners)
│   └── t_shape.json         # 3D T-shape: the T profile extruded along Z (21x2 net)
├── Matlab/                  # Validation scripts and reference outputs (untouched)
└── oldcode/archive/         # Legacy individual scripts preserved unmodified
```

---

## Installation

Install the required Python dependencies:

```bash
pip install -r requirements.txt
```

Dependencies:
* `numpy`
* `matplotlib`
* `PyQt5`
* `scipy`
* `pytest`
* `pandas`
* `openpyxl`

---

## Running the Application

To launch the desktop suite:

```bash
python main.py
```

Or directly via `app.py`:

```bash
python app.py
```

---

## Features

1. **Bézier Curves**:
   - Add/remove arbitrary 2D and 3D control points.
   - Real-time degree calculation ($n = \text{points} - 1$).
   - Interactive Matplotlib plot showing both control polygon and evaluated curve.

2. **Bézier Surfaces**:
   - Customizable rectangular control nets ($M \times N$).
   - Interactive, rotatable 3D surface rendering with wireframe control net overlay.
   - Built-in 3x3 preset and custom net coordinates.

3. **B-Spline Curves**:
   - Arbitrary polynomial degree and knot vector configuration.
   - Automatic generation of open clamped and uniform knot vectors.
   - Plot basis functions independently or draw curves.
   - Export calculated basis matrices directly to Microsoft Excel (`.xlsx`).

4. **NURBS Curves (Rational B-Splines)**:
   - Positive rational weighting $w_i$ for each control point.
   - 2D and 3D curve support.
   - Comparison overlay: "Show B-spline (all weights = 1)" checkbox to compare rational curve with standard unweighted B-spline on the same polygon (2D and 3D, green dotted line).
   - **Plot Rational Basis** button showing $R_i(u)$ (optionally with the plain $N_i(u)$ dotted), and **Export Basis to Excel** for the rational basis matrix.
   - Built-in exact 9-point circle and T-profile examples. Whenever the drawn curve is a circle or circular arc, the status bar shows its centre, radius and maximum radius error.

5. **NURBS Surfaces**:
   - Bidirectional tensor-product rational surfaces with $(M \times N)$ coordinate and weight nets.
   - Interactive rotatable 3D surface visualization.
   - Toolbar toggles for Surface, Net Lines, Points and Equal Axes (true proportions), plus a Style Options dialog.
   - Tables follow the Rows/Cols spin boxes automatically; **Gen Open Knots (U and V)** and **Reset Weights to 1** buttons.
   - Built-in extruded cylinder and 3D T-shape examples. `File -> New` resets to a fresh, drawable 3x2 net.

6. **File I/O**:
   - Save and open curves and surfaces using unified JSON formats (`File -> Open...` / `File -> Save As...`).
   - Supports `bezier_curve`, `bezier_surface`, `bspline_curve`, `nurbs_curve`, and `nurbs_surface`.
   - Knot values keep full precision through Open -> Save, and broken files give clear messages (missing fields, invalid JSON with line/column, bad values).

---

## Running the Demos

### Exact Circle Demo
1. Launch `python main.py`.
2. Click on the **NURBS Curve** tab.
3. Click the **&Load Circle Example** button. (Alternatively, use `File -> Open...` and select `examples/circle.json`).
4. Click **&Draw Curve**.
5. The exact circle is plotted with its control polygon. The status bar shows, for example, `Circle: centre (0, 0), radius 1, max radius error = 4.44e-16`.
6. Check **Show B-spline (all weights = 1)** and click **&Draw Curve** to see the unweighted B-spline overlay (green dotted) versus the exact rational circle (red).
7. Click **Plot Rational &Basis** to see the rational basis functions $R_i(u)$; with the checkbox on, the plain $N_i(u)$ are drawn dotted for comparison.

### NURBS Cylinder Demo
1. Launch `python main.py`.
2. Click on the **NURBS Surface** tab.
3. Click the **&Load Cylinder Example** button. (Alternatively, use `File -> Open...` and select `examples/cylinder.json`).
4. Click **&Draw Surface**.
5. The 3D cylinder surface is rendered with its wireframe net.
6. Click and drag with the left mouse button directly on the plot to interactively rotate the 3D cylinder.

### 3D T-Shape Demo
1. In the **NURBS Surface** tab click **Load T-Shape Example** (or `File -> Open...` and select `examples/t_shape.json`).
2. Click **&Draw Surface**. A T-section (flange 4 x 1, stem 1 x 3, rounded inner corners of radius 0.4) extruded to depth 1 is drawn with its 21x2 control net.
3. The profile alone is available in the **NURBS Curve** tab via **Load T-Profile Example** (`examples/t_profile.json`).
4. Each straight edge and each fillet is one rational quadratic piece joined by double knots, so the corners are sharp and the fillets are exact circular arcs. In code: `core.nurbs.t_profile(...)` and `core.nurbs.nurbs_t_shape(..., depth=...)`.

---

## What Changed

* **Pure-Math Core (`core/nurbs.py`)**: Implemented `rational_basis_matrix`, `nurbs_curve`, `nurbs_surface`, `nurbs_circle`, and `nurbs_cylinder`. Reuses the existing Cox-de Boor recursion without duplication.
* **Validation (`core/validation.py`)**: Added `validate_weights` and `validate_surface_weights` enforcing strictly positive weights and dimension consistency.
* **JSON Serialization (`core/io_json.py`)**: Added `nurbs_curve` and `nurbs_surface` to `VALID_TYPES`, handling `knots_u`, `knots_v`, `degree_u`, `degree_v`, and 2D `weights`.
* **Example Geometries (`examples/`)**: Added `circle.json`, `quarter_circle.json`, and `cylinder.json`.
* **UI Components (`ui/nurbs_curve_tab.py`, `ui/nurbs_surface_tab.py`)**: Added two new tabs matching classic desktop ergonomics, error dialogs via `QMessageBox`, and `get_data()`/`set_data()` serialization.
* **Main Application (`app.py`)**: Registered the new tabs, wired menu actions (`New`, `Open`, `Save As`), and updated About information.
* **Toolbar Ribbon Display Options**: Moved the bulky "Line & Point Display Options" group boxes from the left control panels into the plot toolbar ribbon across Bézier, B-spline, and Surface tabs, freeing up left panel vertical space for comfortable editing.
* **Automated Tests (`tests/test_nurbs.py`, `tests/test_io.py`)**: Added 19 new tests bringing the test suite to 39 passing unit tests.

### NURBS fixes and T-shape
* **NURBS Curve tab**: B-spline overlay now also works for 3D curves; circle detection uses a least-squares circle fit (the old check never triggered); new rational basis plot and Excel export (`File -> Export basis matrix to Excel...` exports the rational basis when the NURBS Curve tab is active).
* **NURBS Surface tab**: `File -> New` now really resets the tab; display toggles and style options in the toolbar; tables resize automatically; equal-axis 3D view; net size up to 50x50.
* **Core (`core/nurbs.py`)**: `extrude_curve`, `t_profile`, `nurbs_t_shape`, `sample_parameters` (sampling that always includes the knots so sharp corners are drawn exactly) and `fit_circle`. `nurbs_curve`/`nurbs_surface` accept explicit parameter values; `nurbs_surface` computes each basis once and sums with NumPy (much faster); `nurbs_cylinder` is now built with `extrude_curve`.
* **JSON (`core/io_json.py`)**: clear errors for missing fields, invalid JSON and bad values; UTF-8 file encoding. The UI no longer rounds knots to 6 digits (`ui/common.py`).
* **Tests**: 53 new tests (92 total), including headless GUI tests that drive the NURBS tabs and main window.

---

## Running Tests

Run the complete test suite with `pytest`:

```bash
pytest -v
```

The GUI tests run headless (`tests/conftest.py` sets `QT_QPA_PLATFORM=offscreen`), so no window opens.

