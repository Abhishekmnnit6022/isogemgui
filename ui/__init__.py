"""PyQt5 tabs of the main window, plus shared UI helpers.

One module per tab (in the order they appear):
    bezier_tab, surface_tab, bspline_tab, nurbs_curve_tab, nurbs_surface_tab, analysis_tab
Shared helpers:
    common   knot text formatting/parsing, table-cell and spin-box checks
    widgets  refinement box, knot markers, polylines, scrollable side panels
"""

from .bezier_tab import BezierTab
from .surface_tab import SurfaceTab
from .bspline_tab import BsplineTab
from .nurbs_curve_tab import NurbsCurveTab
from .nurbs_surface_tab import NurbsSurfaceTab
from .analysis_tab import AnalysisTab

__all__ = ["BezierTab", "SurfaceTab", "BsplineTab", "NurbsCurveTab", "NurbsSurfaceTab",
           "AnalysisTab"]
