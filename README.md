# CAGD Suite - Computer-Aided Geometric Design

A unified desktop application for CAGD curves and surfaces: Bézier curves, Bézier surfaces, and B-spline curves. Built with a pure-math `core/` package and a native PyQt5 / Matplotlib desktop interface.

---

## Folder Layout

```text
gui/
├── main.py                  # Application launcher entry point
├── app.py                   # Main window with tabs, menu bar, and status bar
├── requirements.txt         # Project dependencies
├── README.md                # Project documentation and guide
├── core/                    # Pure-math geometry engine (no GUI dependencies)
│   ├── __init__.py
│   ├── bezier.py            # Bernstein basis, arbitrary-degree curves & surfaces
│   ├── bspline.py           # Cox-de Boor recursion, basis matrices, curve evaluation
│   ├── knots.py             # Knot vector generators (open/clamped, uniform)
│   ├── validation.py        # Strict input validation and clear error messaging
│   └── io_json.py           # Geometry JSON serialization and deserialization
├── ui/                      # PyQt5 user interface components
│   ├── __init__.py
│   ├── bezier_tab.py        # 2D/3D Bézier curve tab
│   ├── surface_tab.py       # 3D Bézier surface tab
│   └── bspline_tab.py       # B-spline tab with basis plot, curve draw & Excel export
├── tests/                   # Pytest test suite
│   ├── test_bezier.py       # Partition of unity, endpoints, closed-forms, convex hull
│   ├── test_bspline.py      # B-spline partition of unity, endpoints, Matlab reference
│   ├── test_knots.py        # Knot generator validation
│   └── test_io.py           # JSON round-trip serialization tests
├── examples/                # Sample geometry JSON files
│   ├── cubic_bezier_2d.json
│   ├── cubic_bezier_3d.json
│   ├── bicubic_surface.json
│   ├── bspline_deg2.json
│   └── bspline_clamped_deg3.json
├── Matlab/                  # Validation scripts and reference outputs (untouched)
└── archive/                 # Legacy individual scripts preserved unmodified
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

4. **File I/O**:
   - Save and open curves and surfaces using the unified JSON format (`File -> Open...` / `File -> Save As...`).

---

## Running Tests

Run the complete test suite with `pytest`:

```bash
pytest -v
```
