# CAGD Suite - Computer-Aided Geometric Design

A unified desktop application for CAGD curves and surfaces: Bézier curves, Bézier surfaces, B-spline curves, and rational B-splines (NURBS curves & NURBS surfaces), plus knot insertion (h-refinement) and an **IGA Analysis** tab that prepares a NURBS surface for isogeometric analysis (elements, connectivity, degrees of freedom, boundary edges, Jacobian mesh-quality check, JSON/MATLAB export - no solver). Built with a pure-math `core/` package and a native PyQt5 / Matplotlib desktop interface.

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
│   ├── refinement.py        # Knot insertion / element splitting for curves and surfaces (shape kept)
│   ├── analysis.py          # IGA analysis prep: elements, connectivity, Jacobian, Gauss points, export
│   └── io_json.py           # Geometry JSON serialization (Bézier, B-spline, NURBS, IGA model)
├── ui/                      # PyQt5 user interface components
│   ├── __init__.py
│   ├── bezier_tab.py        # 2D/3D Bézier curve tab
│   ├── surface_tab.py       # 3D Bézier surface tab
│   ├── bspline_tab.py       # B-spline tab with basis plot, curve draw & Excel export
│   ├── nurbs_curve_tab.py   # NURBS curve tab: weights, B-spline overlay, rational basis, circle & T-profile demos
│   ├── nurbs_surface_tab.py # NURBS surface tab: weight grid, cylinder & 3D T-shape demos, rotatable 3D
│   ├── analysis_tab.py      # IGA Analysis tab (Step 1 Geometry ... Step 5 Export)
│   ├── widgets.py           # Refinement box, knot markers, scrollable panels
│   └── common.py            # Shared knot text formatting/parsing (lossless)
├── tests/                   # Pytest test suite (169 tests: math, JSON and headless GUI)
│   ├── test_bezier.py       # Partition of unity, endpoints, closed-forms, convex hull
│   ├── test_bspline.py      # B-spline partition of unity, endpoints, Matlab reference
│   ├── test_knots.py        # Knot generator validation
│   ├── test_io.py           # JSON round-trip serialization tests (including NURBS)
│   ├── test_nurbs.py        # Weights equivalence, circle/cylinder accuracy, partition of unity
│   ├── test_shapes.py       # T-shape/cylinder exactness, extrusion, sampling, circle fitting
│   ├── test_ui_nurbs.py     # Headless GUI tests for the NURBS tabs and main window
│   ├── test_regressions.py  # Uniform knots, fast basis, silent toggles, clear load/cell errors
│   ├── test_refinement_analysis.py # Knot insertion keeps shape; areas, Jacobian, quality flags; export
│   ├── test_ui_iga.py       # GUI workflows: knots, insert/split, IGA tab load -> refine -> export -> reopen
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
│   ├── t_shape.json         # 3D T-shape: the T profile extruded along Z (21x2 net)
│   ├── quarter_annulus.json # Planar quarter ring, radii 1 and 2 (IGA benchmark)
│   ├── plate_with_hole.json # Planar quarter plate 4 x 4 with a hole of radius 1 (IGA benchmark)
│   ├── nurbscurve2.json, nurbscurve3.json, airfoil_nurbs_curve.json  # Extra NURBS curve samples
│   └── nurbs_saddle_surface.json                                     # Extra NURBS surface sample
├── Matlab/                  # Validation scripts and reference outputs (untouched)
└── verify_project.py        # Independent audit script (structure, maths, GUI smoke test)
```

**Where to look first:** the maths lives in `core/` (start with `bspline.py` and `nurbs.py`); each tab of the window is one file in `ui/`; `app.py` wires the tabs to the File menu. Every module starts with a short docstring explaining its job. The legacy scripts that used to sit in `oldcode/` were removed once everything they did was in `core/` and `ui/`; they remain in the git history.

---

## Installation

Install the required Python dependencies:

```bash
pip install -r requirements.txt
```

Dependencies:
* `numpy`, `matplotlib`, `PyQt5` - needed to run the app
* `pandas`, `openpyxl` - Excel export of basis matrices (loaded only when exporting)
* `pytest`, `scipy` - needed only for the test suite

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
   - Automatic generation of open clamped and uniform knot vectors. Curves are drawn on their valid range [u_p, u_(m-p)], so uniform (unclamped) knots give correct curves; the basis plot shades that range.
   - Plot basis functions independently or draw curves.
   - Export calculated basis matrices directly to Microsoft Excel (`.xlsx`).
   - **Knots** toolbar checkbox: black square markers where each knot sits on the curve (the element ends), labelled `u=...`; the status bar shows the number of elements.
   - **Refinement (Knot Insertion)** box: insert a knot value (optionally several times) or **Split All Elements**. Control points are added but the curve keeps exactly the same shape; the status bar reports the measured shape change (about 1e-16).

4. **NURBS Curves (Rational B-Splines)**:
   - Positive rational weighting $w_i$ for each control point.
   - 2D and 3D curve support.
   - Comparison overlay: "Show B-spline (all weights = 1)" checkbox to compare rational curve with standard unweighted B-spline on the same polygon (2D and 3D, green dotted line).
   - **Plot Rational Basis** button showing $R_i(u)$ (optionally with the plain $N_i(u)$ dotted), and **Export Basis to Excel** for the rational basis matrix.
   - Built-in exact 9-point circle and T-profile examples. Whenever the drawn curve is a circle or circular arc, the status bar shows its centre, radius and maximum radius error.
   - The same **Knots** checkbox and **Refinement (Knot Insertion)** box as the B-spline tab; weights are refined too, so a circle stays an exact circle.

5. **NURBS Surfaces**:
   - Bidirectional tensor-product rational surfaces with $(M \times N)$ coordinate and weight nets.
   - Interactive rotatable 3D surface visualization.
   - Toolbar toggles for Surface, Net Lines, Points and Equal Axes (true proportions), plus a Style Options dialog.
   - Tables follow the Rows/Cols spin boxes automatically; **Gen Open Knots (U and V)** and **Reset Weights to 1** buttons.
   - Built-in extruded cylinder and 3D T-shape examples. `File -> New` resets to a fresh, drawable 3x2 net.
   - **Knots** toolbar checkbox draws the element boundaries (lines at every distinct knot) on the surface.

6. **IGA Analysis** (prepares a NURBS surface for isogeometric analysis; it does not solve anything). The left panel is split into five numbered steps:
   - **Step 1 - Geometry**: *Use NURBS Surface Tab*, *Open File...* (a `nurbs_surface` or saved `iga_analysis_model`), or an example: plate with a hole, quarter annulus, cylinder, T-shape.
   - **Step 2 - Refinement**: split every element in U, V or both, or insert a knot at a chosen value; *Reset to Original Geometry* undoes it. The shape never changes.
   - **Step 3 - Boundary Conditions**: tag each edge (`u = 0`, `u = 1`, `v = 0`, `v = 1`) as Free, Fixed (Dirichlet), Load (Neumann) or Symmetry. Selecting a row highlights that edge and its control points.
   - **Step 4 - Display**: element mesh, Jacobian colours (mesh quality), boundary conditions, element numbers, control net, Gauss points.
   - **Step 5 - Export**: JSON (reopens in the app) or a MATLAB/Octave `.m` script whose `geometry` uses the NURBS toolbox layout (`nrbmak`, as used by GeoPDEs) plus elements, 1-based connectivity, DOF counts and boundary sets.
   - Results below the plot: **Summary** (control points, elements, DOFs for scalar / 2D / 3D problems, total area, quality counts, boundary sets) and **Elements** (parameter ranges, area, control points, min/max Jacobian, status). Status is *OK*, *Distorted* (|J| varies more than 10x inside the element), *Inverted* (det J <= 0 on a planar surface) or *Degenerate* (e.g. an edge collapsed to a point); bad elements are outlined on the plot and coloured in the table.
   - Planar surfaces (constant z) use the signed det J and are drawn in 2D; other surfaces use the area stretch |S_u x S_v| in 3D.

7. **File I/O**:
   - Save and open curves and surfaces using unified JSON formats (`File -> Open...` / `File -> Save As...`).
   - Supports `bezier_curve`, `bezier_surface`, `bspline_curve`, `nurbs_curve`, `nurbs_surface`, and `iga_analysis_model` (saved from the IGA Analysis tab, opened back into it).
   - Knot values keep full precision through Open -> Save, and broken files give clear messages (missing fields, invalid JSON with line/column, bad values).

---

## Running the Demos

### Exact Circle Demo
1. Launch `python main.py`.
2. Click on the **NURBS Curve** tab.
3. Click the **Load Circle Example** button. (Alternatively, use `File -> Open...` and select `examples/circle.json`).
4. Click **Draw Curve**.
5. The exact circle is plotted with its control polygon. The status bar shows, for example, `Circle: centre (0, 0), radius 1, max radius error = 4.44e-16`.
6. Check **Show B-spline (all weights = 1)** and click **Draw Curve** to see the unweighted B-spline overlay (green dotted) versus the exact rational circle (red).
7. Click **Plot Rational Basis** to see the rational basis functions $R_i(u)$; with the checkbox on, the plain $N_i(u)$ are drawn dotted for comparison.

### NURBS Cylinder Demo
1. Launch `python main.py`.
2. Click on the **NURBS Surface** tab.
3. Click the **Load Cylinder Example** button. (Alternatively, use `File -> Open...` and select `examples/cylinder.json`).
4. Click **Draw Surface**.
5. The 3D cylinder surface is rendered with its wireframe net.
6. Click and drag with the left mouse button directly on the plot to interactively rotate the 3D cylinder.

### 3D T-Shape Demo
1. In the **NURBS Surface** tab click **Load T-Shape Example** (or `File -> Open...` and select `examples/t_shape.json`).
2. Click **Draw Surface**. A T-section (flange 4 x 1, stem 1 x 3, rounded inner corners of radius 0.4) extruded to depth 1 is drawn with its 21x2 control net.
3. The profile alone is available in the **NURBS Curve** tab via **Load T-Profile Example** (`examples/t_profile.json`).
4. Each straight edge and each fillet is one rational quadratic piece joined by double knots, so the corners are sharp and the fillets are exact circular arcs. In code: `core.nurbs.t_profile(...)` and `core.nurbs.nurbs_t_shape(..., depth=...)`.
5. Tick **Knots** in the toolbar to see its 10 elements.

### Knot Insertion Demo
1. In the **NURBS Curve** tab click **Load Circle Example**, tick **Knots** in the toolbar and click **Draw Curve**: knot markers at u = 0, 0.25, 0.5 and 0.75 (u = 1 is the same point as u = 0), so 4 elements.
2. In **Refinement (Knot Insertion)** type `0.125` and click **Insert Knot**. The table grows from 9 to 10 points and the control polygon gets an extra corner, but the red circle is unchanged; the status bar shows `max shape change` of about 1e-16 and the radius error is still about 1e-16.
3. Click **Split All Elements** to halve every element at once.

### IGA Analysis Demo (plate with a hole)
1. Open the **IGA Analysis** tab. In **Step 1** choose *Plate with a hole (2D)* and click **Load Example**: 2 elements, coloured by det J.
2. In **Step 2** click **Both** twice: 8 x 4 = 32 elements, 55 control points; the shape is unchanged.
3. In **Step 3** set `u = 1` to *Fixed (Dirichlet)* and `v = 1` to *Load (Neumann)*; the edges are drawn thick in red and green. Click a row to highlight an edge and its control points.
4. In **Step 4** tick *Gauss points* to see the 3 x 2 integration points of each element.
5. Read the **Summary** tab (110 unknowns for 2D elasticity, total area close to 16 - pi/4 = 15.2146) and the **Elements** tab (click a row to outline that element).
6. In **Step 5** export JSON or MATLAB. `File -> Open...` on the JSON brings the model back, including the boundary conditions.

To analyse your own surface, build it in the **NURBS Surface** tab and click **Use NURBS Surface Tab** in Step 1. Planar surfaces (all z equal) give 2D results; the sign of det J reveals folded (inverted) elements.

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

### Review fixes
* **Uniform knots**: B-spline/NURBS curves, surfaces and the rational basis are evaluated on the valid range `[knots[p], knots[-p-1]]` (`core.bspline.valid_domain`); before, uniform knots made curves jump to (0, 0) at both ends.
* **Speed**: `core.bspline.basis_functions` evaluates the Cox-de Boor recurrence on whole arrays (a degree-10 curve went from about 3 s to a few milliseconds); the recursive `_basis_scalar` stays as the test reference.
* **Display toggles** in the Bezier curve, Bezier surface and B-spline tabs redraw silently and only when something is plotted; on the B-spline tab they no longer replace the basis plot.
* **Clear errors** for bad table cells (e.g. `Row 2: X value 'abc' is not a number.`) and for files whose degree or net size does not fit the UI (e.g. `Degree 16 is outside the supported range 1 to 15.`) instead of silently changing the loaded values. The Bezier surface tab now accepts nets up to 50x50.
* **Tests**: 23 new regression tests (115 total).

### Knot display, knot insertion and IGA Analysis
* **Knots** checkbox in the B-spline, NURBS Curve and NURBS Surface toolbars shows the element boundaries.
* **Knot insertion** (`core/refinement.py`): `insert_knot`, `refine_uniform`, `insert_knot_surface`, `refine_surface_uniform` (Boehm's algorithm in homogeneous coordinates, so NURBS weights are handled). Available in the B-spline and NURBS Curve tabs and in the IGA Analysis tab.
* **IGA Analysis tab** (`ui/analysis_tab.py`, `core/analysis.py`): elements, connectivity, DOFs, boundary edges with conditions, Jacobian quality check, Gauss points, JSON and MATLAB export, File menu integration.
* **Core**: basis-function derivatives (`basis_function_derivatives`), `element_boundaries`, and the planar benchmarks `quarter_annulus` and `plate_with_hole`.
* **Layout**: the left panels of the B-spline, NURBS Curve and NURBS Surface tabs scroll instead of squashing their tables; stray `&` removed from labels.
* **Tests**: 54 new tests (169 total).

### Code clean-up
* Deleted the `oldcode/` folder of legacy scripts (all of their functionality is in `core/` and `ui/`; they remain in the git history).
* Moved the loose sample JSON files from the project root into `examples/`.
* Removed unused imports, an unused alias, an unused attribute and the duplicated start-up code in `main.py` (it now calls `app.main()`).
* Added package overviews in `core/__init__.py` and `ui/__init__.py`, and short comments on every non-obvious function.

---

## Running Tests

Run the complete test suite with `pytest`:

```bash
pytest -v
```

The GUI tests run headless (`tests/conftest.py` sets `QT_QPA_PLATFORM=offscreen`), so no window opens.

