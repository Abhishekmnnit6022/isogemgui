"""Pure-math geometry engine (no GUI code, no printing).

Modules:
    bezier      Bernstein polynomials, Bezier curves and surfaces
    bspline     Cox-de Boor basis (vectorised), derivatives, valid range, B-spline curves
    nurbs       Rational curves/surfaces, circle, cylinder, T-shape, IGA benchmark shapes
    knots       Open (clamped) and uniform knot vector generators
    refinement  Knot insertion / element splitting (shape never changes)
    analysis    IGA analysis prep: elements, connectivity, Jacobian, Gauss points, export
    validation  Input checks that raise clear ValueError messages
    io_json     Save / load geometry and analysis-model JSON files
"""
