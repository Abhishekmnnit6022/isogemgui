CAGD SUITE - COMPUTER-AIDED GEOMETRIC DESIGN
============================================

A desktop application for CAGD curves and surfaces:

  - Bezier curves and Bezier surfaces
  - B-spline curves
  - NURBS curves and NURBS surfaces (rational B-splines)
  - Knot insertion (h-refinement) that keeps the shape exactly
  - An IGA Analysis tab that prepares a NURBS surface for isogeometric
    analysis: elements, connectivity, degrees of freedom, boundary edges,
    Jacobian mesh-quality check, JSON and MATLAB export. It does not solve
    anything (no solver is included).

The maths is in a pure-Python core/ package; the window is built with
PyQt5 and Matplotlib.


1. FOLDER LAYOUT
----------------

gui2/
  main.py                   Starts the application
  app.py                    Main window: tabs, menu bar, status bar
  requirements.txt          Python dependencies
  README.txt                This guide
  documentation.txt         Full technical documentation

  core/                     Geometry engine (no GUI code)
    __init__.py             Package overview
    bezier.py               Bernstein basis, Bezier curves and surfaces
    bspline.py              Cox-de Boor basis, derivatives, curve evaluation
    nurbs.py                Rational basis, NURBS curves/surfaces, circle,
                            cylinder, T-shape, IGA benchmark shapes
    knots.py                Knot vector generators (open/clamped, uniform)
    validation.py           Input checks (knots, degree, points, weights)
    refinement.py           Knot insertion / element splitting (shape kept)
    analysis.py             IGA prep: elements, connectivity, Jacobian,
                            Gauss points, JSON and MATLAB export
    io_json.py              Reading and writing geometry JSON files

  ui/                       PyQt5 interface, one file per tab
    __init__.py             Package overview
    bezier_tab.py           2D/3D Bezier curve tab
    surface_tab.py          3D Bezier surface tab
    bspline_tab.py          B-spline tab: basis plot, curve, Excel export
    nurbs_curve_tab.py      NURBS curve tab: weights, B-spline overlay,
                            rational basis, circle and T-profile examples
    nurbs_surface_tab.py    NURBS surface tab: weight grid, cylinder and
                            3D T-shape examples, rotatable 3D view
    analysis_tab.py         IGA Analysis tab (Step 1 Geometry ... Step 5 Export)
    widgets.py              Refinement box, knot markers, scrollable panels
    common.py               Shared helpers: lossless knot text, cell checks

  tests/                    Pytest suite (173 tests: maths, JSON, headless GUI)
    conftest.py             Shared setup: headless Qt, message-box recorder, main window
    test_maths.py           66 tests: Bezier, knots, B-spline, NURBS, exact shapes
    test_files.py           23 tests: JSON save/open, examples, clear file errors
    test_iga.py             55 tests: knot insertion, analysis model, IGA Analysis tab
    test_gui.py             29 tests: curve/surface tabs, display options, project rules

  examples/                 Sample geometry files (JSON)
    cubic_bezier_2d.json, cubic_bezier_3d.json, bicubic_surface.json
    bspline_deg2.json, bspline_clamped_deg3.json
    circle.json             Exact 9-point quadratic NURBS circle
    quarter_circle.json     3-point quarter circle
    cylinder.json           Degree 2x1 extruded NURBS cylinder
    t_profile.json          Closed T cross-section (rounded inner corners)
    t_shape.json            3D T-shape: the T profile extruded along Z
    quarter_annulus.json    Planar quarter ring, radii 1 and 2 (IGA benchmark)
    plate_with_hole.json    Quarter plate 4 x 4 with a hole of radius 1
    nurbscurve2.json, nurbscurve3.json, airfoil_nurbs_curve.json
    nurbs_saddle_surface.json

  Matlab/                   Validation scripts and reference outputs (untouched)
  MATLAB_Verification/      Independent MATLAB check of all IGA results
                            (start with COMPARISON_SUMMARY.txt)

Where to look first: the maths is in core/ (start with bspline.py and
nurbs.py). Each tab of the window is one file in ui/. app.py connects the
tabs to the File menu. Every module starts with a short docstring that says
what it does. The old scripts from oldcode/ were removed once everything
they did was in core/ and ui/; they are still in the git history.


2. INSTALLATION
---------------

    pip install -r requirements.txt

Dependencies:
    numpy, matplotlib, PyQt5   needed to run the app
    pandas, openpyxl           Excel export of basis matrices (only loaded
                               when you export)
    pytest, scipy              needed only for the tests


3. RUNNING THE APPLICATION
--------------------------

    python main.py

or

    python app.py


4. FEATURES
-----------

4.1 Bezier Curves
    - Add and remove 2D or 3D control points.
    - The degree updates automatically (degree = number of points - 1).
    - The plot shows the control polygon and the curve.

4.2 Bezier Surfaces
    - Rectangular control nets of any size M x N (up to 50 x 50).
    - Rotatable 3D surface with the control net drawn on top.
    - Built-in 3x3 preset.

4.3 B-Spline Curves
    - Any degree and any knot vector.
    - Buttons to generate open (clamped) and uniform knot vectors. Curves are
      drawn on their valid range [u_p, u_(m-p)], so uniform (unclamped)
      knots give correct curves; the basis plot shades that range.
    - Plot the basis functions or draw the curve.
    - Export the basis matrix to Excel (.xlsx).
    - Knots checkbox (toolbar): square markers where each knot sits on the
      curve, labelled u=...; the status bar shows the number of elements.
    - Refinement (Knot Insertion) box: insert a knot value (one or more
      times) or Split All Elements. Control points are added but the curve
      keeps exactly the same shape; the status bar shows the measured shape
      change (about 1e-16).

4.4 NURBS Curves
    - A positive weight w_i for each control point; 2D and 3D curves.
    - "Show B-spline (all weights = 1)" overlays the plain B-spline on the
      same polygon (green dotted line) for comparison.
    - Plot Rational Basis shows R_i(u) (and, with the checkbox on, the plain
      N_i(u) dotted). Export Basis to Excel saves the rational basis matrix.
    - Built-in exact circle and T-profile examples. When the curve is a
      circle or circular arc, the status bar shows its centre, radius and
      largest radius error.
    - The same Knots checkbox and Refinement box as the B-spline tab. Weights
      are refined too, so a circle stays an exact circle.

4.5 NURBS Surfaces
    - Tensor-product rational surfaces with M x N point and weight tables.
    - Rotatable 3D view.
    - Toolbar checkboxes: Surface, Net Lines, Points, Equal Axes, Knots,
      plus a Style Options dialog.
    - Tables follow the Rows/Cols boxes automatically. Buttons: Gen Open
      Knots (U and V) and Reset Weights to 1.
    - Built-in cylinder and 3D T-shape examples. File -> New resets the tab
      to a fresh, drawable 3x2 net.
    - Knots draws the element boundaries (lines at every distinct knot).

4.6 IGA Analysis
    Prepares a NURBS surface for isogeometric analysis. It does not solve
    anything. The left panel has five numbered steps:

    Step 1 - Geometry
        Use NURBS Surface Tab, Open File... (a nurbs_surface or a saved
        iga_analysis_model), or load an example: plate with a hole,
        quarter annulus, cylinder, T-shape.

    Step 2 - Refinement
        Split every element in U, V or both, or insert a knot at a chosen
        value. Reset to Original Geometry undoes it. The shape never changes.

    Step 3 - Boundary Conditions
        Tag each edge (u = 0, u = 1, v = 0, v = 1) as Free, Fixed
        (Dirichlet), Load (Neumann) or Symmetry. Selecting a row highlights
        that edge and its control points.

    Step 4 - Display
        Element mesh, Jacobian colours (mesh quality), boundary conditions,
        element numbers, control net, Gauss points.

    Step 5 - Export
        JSON (opens again in the app) or a MATLAB/Octave .m script. Its
        "geometry" uses the NURBS toolbox layout (nrbmak, as used by
        GeoPDEs) and it also contains the elements, 1-based connectivity,
        DOF counts and boundary sets.

    Below the plot:
        Summary   control points, elements, DOFs for scalar / 2D / 3D
                  problems, total area, quality counts, boundary sets.
        Elements  parameter ranges, area, control points, min/max Jacobian
                  and status for each element.

    Element status:
        OK          nothing wrong
        Distorted   |J| changes more than 10 times inside the element
        Inverted    det J <= 0 on a planar surface (folded element)
        Degenerate  e.g. an edge collapsed to a point
    Bad elements are outlined on the plot and coloured in the table.

    Planar surfaces (all z equal) use the signed det J and are drawn in 2D.
    Other surfaces use the area stretch |S_u x S_v| and are drawn in 3D.

4.7 File Input/Output
    - File -> Open... and File -> Save As... use one JSON format for all
      types: bezier_curve, bezier_surface, bspline_curve, nurbs_curve,
      nurbs_surface and iga_analysis_model (saved from the IGA Analysis tab
      and opened back into it).
    - Knot values keep full precision through Open -> Save.
    - Broken files give clear messages: missing fields, invalid JSON (with
      line and column), bad values.


5. DEMOS
--------

5.1 Exact circle
    1. Run python main.py and open the NURBS Curve tab.
    2. Click Load Circle Example (or File -> Open... examples/circle.json).
    3. Click Draw Curve. The status bar shows, for example:
           Circle: centre (0, 0), radius 1, max radius error = 4.44e-16
    4. Tick "Show B-spline (all weights = 1)" and click Draw Curve to compare
       the plain B-spline (green dotted) with the exact circle (red).
    5. Click Plot Rational Basis to see R_i(u).

5.2 NURBS cylinder
    1. Open the NURBS Surface tab.
    2. Click Load Cylinder Example (or open examples/cylinder.json).
    3. Click Draw Surface.
    4. Drag on the plot with the left mouse button to rotate it.

5.3 3D T-shape
    1. In the NURBS Surface tab click Load T-Shape Example (or open
       examples/t_shape.json).
    2. Click Draw Surface. A T-section (flange 4 x 1, stem 1 x 3, rounded
       inner corners of radius 0.4) extruded to depth 1 is drawn with its
       21x2 control net.
    3. The profile alone is in the NURBS Curve tab: Load T-Profile Example
       (examples/t_profile.json).
    4. Each straight edge and each fillet is one rational quadratic piece
       joined by double knots, so the corners are sharp and the fillets are
       exact circular arcs. In code: core.nurbs.t_profile(...) and
       core.nurbs.nurbs_t_shape(..., depth=...).
    5. Tick Knots in the toolbar to see its 10 elements.

5.4 Knot insertion
    1. In the NURBS Curve tab click Load Circle Example, tick Knots and click
       Draw Curve: markers at u = 0, 0.25, 0.5 and 0.75 (u = 1 is the same
       point as u = 0), so 4 elements.
    2. In Refinement (Knot Insertion) type 0.125 and click Insert Knot. The
       table grows from 9 to 10 points and the control polygon gets an extra
       corner, but the circle is unchanged: the status bar shows a max shape
       change of about 1e-16.
    3. Click Split All Elements to halve every element at once.

5.5 IGA Analysis (plate with a hole)
    1. Open the IGA Analysis tab. In Step 1 choose "Plate with a hole (2D)"
       and click Load Example: 2 elements, coloured by det J.
    2. In Step 2 click Both twice: 8 x 4 = 32 elements, 55 control points.
       The shape does not change.
    3. In Step 3 set u = 1 to Fixed (Dirichlet) and v = 1 to Load (Neumann).
       The edges are drawn thick in red and green. Click a row to highlight
       an edge and its control points.
    4. In Step 4 tick Gauss points to see the 3 x 2 points of each element.
    5. Read the Summary tab (110 unknowns for 2D elasticity, total area close
       to 16 - pi/4 = 15.2146) and the Elements tab (click a row to outline
       that element).
    6. In Step 5 export JSON or MATLAB. File -> Open... on the JSON file
       brings the model back, boundary conditions included.

    To analyse your own surface, build it in the NURBS Surface tab and click
    Use NURBS Surface Tab in Step 1.


6. RUNNING THE TESTS
--------------------

    pytest -v

The GUI tests run headless (tests/conftest.py sets
QT_QPA_PLATFORM=offscreen), so no window opens.



    - Moved loose sample JSON files into examples/.
    - Removed unused imports and duplicated start-up code (main.py now calls
      app.main()).
    - Added package overviews and short comments.
