"""The six tabs of the main window and two helper modules (common, widgets)."""

from .bezier_tab import BezierTab
from .surface_tab import SurfaceTab
from .bspline_tab import BsplineTab
from .nurbs_curve_tab import NurbsCurveTab
from .nurbs_surface_tab import NurbsSurfaceTab
from .analysis_tab import AnalysisTab

__all__ = ["BezierTab", "SurfaceTab", "BsplineTab", "NurbsCurveTab", "NurbsSurfaceTab",
           "AnalysisTab"]
