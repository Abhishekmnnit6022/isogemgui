"""IGA analysis prep for a NURBS surface: elements, Jacobian, Gauss points, export."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .bspline import basis_function_derivatives, element_boundaries
from .validation import (
    validate_bspline_params,
    validate_degree,
    validate_knots,
    validate_surface_net,
    validate_surface_weights,
)

EDGE_NAMES = ("u = 0", "u = 1", "v = 0", "v = 1")
CONDITIONS = ("Free", "Fixed (Dirichlet)", "Load (Neumann)", "Symmetry")
DISTORTION_RATIO = 10.0   # max|J| / min|J| inside one element above this -> "Distorted"
STATUS_OK, STATUS_DISTORTED, STATUS_INVERTED, STATUS_DEGENERATE = (
    "OK", "Distorted", "Inverted", "Degenerate")
MODEL_TYPE = "iga_analysis_model"


# ---------------------------------------------------------------- geometry evaluation

def surface_derivatives(ku, kv, du, dv, net, weights, u, v):
    """Surface points S and tangents S_u, S_v on a (u, v) grid."""
    net = np.asarray(net, dtype=float)
    w = np.asarray(weights, dtype=float)
    Nu, dNu = basis_function_derivatives(ku, du, u)
    Nv, dNv = basis_function_derivatives(kv, dv, v)
    Pw = net * w[:, :, np.newaxis]

    # Tensor-product sums done as two matrix products (u first, then v): much faster
    # than one 4-index einsum on large meshes, same result up to rounding.
    def combine(Bu, Bv):
        A = np.tensordot(np.tensordot(Bu, Pw, axes=(0, 0)), Bv, axes=(1, 0)).transpose(0, 2, 1)
        W = Bu.T @ w @ Bv
        return A, W

    A, W = combine(Nu, Nv)
    Au, Wu = combine(dNu, Nv)
    Av, Wv = combine(Nu, dNv)
    W = W[:, :, np.newaxis]
    S = A / W
    Su = (Au - Wu[:, :, np.newaxis] * S) / W
    Sv = (Av - Wv[:, :, np.newaxis] * S) / W
    return S, Su, Sv


def is_planar(net: np.ndarray) -> bool:
    """True when every control point has the same z (a flat 2D domain in the x-y plane)."""
    net = np.asarray(net, dtype=float)
    extent = max(float(np.ptp(net.reshape(-1, 3), axis=0).max()), 1.0)
    return float(np.ptp(net[:, :, 2])) <= 1e-12 * extent


def jacobian(Su: np.ndarray, Sv: np.ndarray, planar: bool) -> np.ndarray:
    """det J = x_u y_v - x_v y_u for planar surfaces (signed: < 0 means folded),
    otherwise the area stretch |S_u x S_v| (always >= 0)."""
    if planar:
        return Su[..., 0] * Sv[..., 1] - Su[..., 1] * Sv[..., 0]
    return np.linalg.norm(np.cross(Su, Sv), axis=-1)


def gauss_legendre(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """n-point Gauss-Legendre nodes and weights on [-1, 1]."""
    if n < 1:
        raise ValueError(f"Number of Gauss points must be >= 1, got {n}.")
    return np.polynomial.legendre.leggauss(int(n))


def is_clamped(knots: Sequence[float], degree: int) -> bool:
    """True when the first and last degree + 1 knots are equal (open knot vector)."""
    k = list(knots)
    p = int(degree)
    return len(set(k[:p + 1])) == 1 and len(set(k[-p - 1:])) == 1


# ---------------------------------------------------------------- the model

@dataclass
class Element:
    """One element (non-empty knot span in u times one in v)."""
    index: int
    eu: int                       # element number along u (0-based)
    ev: int                       # element number along v (0-based)
    u_range: Tuple[float, float]
    v_range: Tuple[float, float]
    control_points: List[int]     # global control point indices acting on this element
    area: float
    jacobian_min: float
    jacobian_max: float
    status: str


@dataclass
class AnalysisModel:
    """Everything a solver would need, computed from one NURBS surface."""
    knots_u: List[float]
    knots_v: List[float]
    degree_u: int
    degree_v: int
    net: np.ndarray
    weights: np.ndarray
    planar: bool
    bounds_u: np.ndarray
    bounds_v: np.ndarray
    elements: List[Element]
    boundaries: Dict[str, List[int]]
    gauss_u: int
    gauss_v: int
    gauss_points: np.ndarray            # (n_points, 3) physical positions
    total_area: float
    clamped_u: bool
    clamped_v: bool
    conditions: Dict[str, str] = field(default_factory=lambda: {e: "Free" for e in EDGE_NAMES})

    @property
    def n_rows(self) -> int:
        return self.net.shape[0]

    @property
    def n_cols(self) -> int:
        return self.net.shape[1]

    @property
    def n_control_points(self) -> int:
        return self.n_rows * self.n_cols

    @property
    def n_elements_u(self) -> int:
        return len(self.bounds_u) - 1

    @property
    def n_elements_v(self) -> int:
        return len(self.bounds_v) - 1

    # Unknowns a solver would have for each kind of problem
    def dofs(self) -> Dict[str, int]:
        n = self.n_control_points
        return {"control_points": n, "scalar_field": n,
                "elasticity_2d": 2 * n, "elasticity_3d": 3 * n}

    # How many elements have each status
    def quality_counts(self) -> Dict[str, int]:
        counts = {s: 0 for s in (STATUS_OK, STATUS_DISTORTED, STATUS_INVERTED, STATUS_DEGENERATE)}
        for e in self.elements:
            counts[e.status] += 1
        return counts

    def geometry_dict(self) -> dict:
        """The surface in the app's nurbs_surface JSON format."""
        return {
            "type": "nurbs_surface",
            "degree_u": self.degree_u,
            "degree_v": self.degree_v,
            "knots_u": list(self.knots_u),
            "knots_v": list(self.knots_v),
            "control_points": self.net.tolist(),
            "weights": self.weights.tolist(),
        }

    def to_dict(self) -> dict:
        """Complete analysis model as a JSON-ready dictionary."""
        return {
            "type": MODEL_TYPE,
            "version": 1,
            "geometry": self.geometry_dict(),
            "numbering": "control point index = i * n_cols + j (i along u, j along v), 0-based",
            "planar": self.planar,
            "dofs": self.dofs(),
            "mesh": {"elements_u": self.n_elements_u, "elements_v": self.n_elements_v,
                     "element_boundaries_u": self.bounds_u.tolist(),
                     "element_boundaries_v": self.bounds_v.tolist()},
            "gauss": {"points_u": self.gauss_u, "points_v": self.gauss_v},
            "total_area": self.total_area,
            "quality": self.quality_counts(),
            "elements": [
                {"id": e.index, "u_range": list(e.u_range), "v_range": list(e.v_range),
                 "control_points": e.control_points, "area": e.area,
                 "jacobian_min": e.jacobian_min, "jacobian_max": e.jacobian_max,
                 "status": e.status}
                for e in self.elements
            ],
            "boundaries": {name: {"control_points": idx, "condition": self.conditions[name]}
                           for name, idx in self.boundaries.items()},
        }


# Even quality samples per element side (odd, so the midpoint is included).
# Corners + Gauss points alone missed a fold between them.
def quality_samples(degree: int) -> int:
    return max(2 * int(degree) + 3, 7)


# Samples in one element: Gauss nodes first, then n_uniform even points
# (ends moved in a tiny bit so we stay inside this element).
def _element_samples(a: float, b: float, nodes: np.ndarray, n_uniform: int = 0) -> np.ndarray:
    mid, half = (a + b) / 2.0, (b - a) / 2.0
    pts = mid + half * nodes
    if n_uniform:
        eps = 1e-9 * (b - a)
        pts = np.concatenate([pts, np.linspace(a + eps, b - eps, int(n_uniform))])
    return pts


def analyze_surface(
    knots_u: Sequence[float],
    knots_v: Sequence[float],
    degree_u: int,
    degree_v: int,
    net: np.ndarray,
    weights: np.ndarray,
    n_gauss_u: Optional[int] = None,
    n_gauss_v: Optional[int] = None,
    conditions: Optional[Dict[str, str]] = None,
) -> AnalysisModel:
    """Build the analysis model of a NURBS surface.

    Uses p+1 Gauss points per direction for the areas. Quality is checked on the
    Gauss points plus an even grid that includes the element corners and edges."""
    ku, kv = validate_knots(knots_u), validate_knots(knots_v)
    du, dv = validate_degree(degree_u), validate_degree(degree_v)
    net = validate_surface_net(net)
    rows, cols, _ = net.shape
    w = validate_surface_weights(weights, rows, cols)
    validate_bspline_params(ku, du, rows)
    validate_bspline_params(kv, dv, cols)
    if du < 1 or dv < 1:
        raise ValueError("Analysis needs degree >= 1 in both directions.")

    gu = int(n_gauss_u) if n_gauss_u else du + 1
    gv = int(n_gauss_v) if n_gauss_v else dv + 1
    xu, wu = gauss_legendre(gu)
    xv, wv = gauss_legendre(gv)
    bu, bv = element_boundaries(ku, du), element_boundaries(kv, dv)
    neu, nev = len(bu) - 1, len(bv) - 1
    planar = is_planar(net)

    # Evaluate all samples of all elements in one go
    qu, qv = quality_samples(du), quality_samples(dv)
    mu, mv = gu + qu, gv + qv
    U = np.concatenate([_element_samples(bu[a], bu[a + 1], xu, qu) for a in range(neu)])
    V = np.concatenate([_element_samples(bv[b], bv[b + 1], xv, qv) for b in range(nev)])
    S, Su, Sv = surface_derivatives(ku, kv, du, dv, net, w, U, V)
    J = jacobian(Su, Sv, planar)
    j_scale = float(np.max(np.abs(J))) or 1.0

    span_u = [int(np.searchsorted(ku, bu[a], side="right")) - 1 for a in range(neu)]
    span_v = [int(np.searchsorted(kv, bv[b], side="right")) - 1 for b in range(nev)]

    elements: List[Element] = []
    gauss_pts = []
    total = 0.0
    for a in range(neu):
        for b in range(nev):
            block = J[a * mu:(a + 1) * mu, b * mv:(b + 1) * mv]
            jg = block[:gu, :gv]                          # Gauss points only
            hu = (bu[a + 1] - bu[a]) / 2.0
            hv = (bv[b + 1] - bv[b]) / 2.0
            area = float(np.sum(np.outer(wu, wv) * np.abs(jg)) * hu * hv)
            total += area
            gauss_pts.append(S[a * mu:a * mu + gu, b * mv:b * mv + gv].reshape(-1, 3))

            jmin, jmax = float(block.min()), float(block.max())
            amin = float(np.min(np.abs(block)))
            if planar and jmin <= 0.0:
                status = STATUS_INVERTED
            elif amin <= 1e-6 * j_scale:   # e.g. an edge collapsed to a point
                status = STATUS_DEGENERATE
            elif float(np.max(np.abs(block))) / amin > DISTORTION_RATIO:
                status = STATUS_DISTORTED
            else:
                status = STATUS_OK

            cps = [i * cols + j
                   for i in range(span_u[a] - du, span_u[a] + 1)
                   for j in range(span_v[b] - dv, span_v[b] + 1)]
            elements.append(Element(
                index=len(elements), eu=a, ev=b,
                u_range=(float(bu[a]), float(bu[a + 1])), v_range=(float(bv[b]), float(bv[b + 1])),
                control_points=cps, area=area, jacobian_min=jmin, jacobian_max=jmax, status=status,
            ))

    boundaries = {
        "u = 0": [j for j in range(cols)],
        "u = 1": [(rows - 1) * cols + j for j in range(cols)],
        "v = 0": [i * cols for i in range(rows)],
        "v = 1": [i * cols + cols - 1 for i in range(rows)],
    }
    conds = {e: "Free" for e in EDGE_NAMES}
    if conditions:
        for name, cond in conditions.items():
            if name not in conds:
                raise ValueError(f"Unknown boundary edge '{name}'. Expected one of {EDGE_NAMES}.")
            if cond not in CONDITIONS:
                raise ValueError(f"Unknown boundary condition '{cond}'. Expected one of {CONDITIONS}.")
            conds[name] = cond

    return AnalysisModel(
        knots_u=ku, knots_v=kv, degree_u=du, degree_v=dv, net=net, weights=w,
        planar=planar, bounds_u=bu, bounds_v=bv, elements=elements, boundaries=boundaries,
        gauss_u=gu, gauss_v=gv, gauss_points=np.concatenate(gauss_pts), total_area=total,
        clamped_u=is_clamped(ku, du), clamped_v=is_clamped(kv, dv), conditions=conds,
    )


def jacobian_field(model: AnalysisModel, per_element: int = 8):
    """Jacobian on a fine grid inside each element, for colouring the plot."""
    nodes = np.linspace(-1.0, 1.0, max(int(per_element), 2))
    U = np.concatenate([_element_samples(model.bounds_u[a], model.bounds_u[a + 1], nodes * (1 - 1e-9))
                        for a in range(model.n_elements_u)])
    V = np.concatenate([_element_samples(model.bounds_v[b], model.bounds_v[b + 1], nodes * (1 - 1e-9))
                        for b in range(model.n_elements_v)])
    S, Su, Sv = surface_derivatives(model.knots_u, model.knots_v, model.degree_u, model.degree_v,
                                    model.net, model.weights, U, V)
    J = jacobian(Su, Sv, model.planar)
    return S[:, :, 0], S[:, :, 1], S[:, :, 2], J


# ---------------------------------------------------------------- export

def export_json(model: AnalysisModel, path) -> None:
    """Write the analysis model as JSON (can be opened again by the app)."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(model.to_dict(), f, indent=2)


# Numbers as a MATLAB row: integers without '.0', others with full precision
def _m_row(values) -> str:
    return " ".join(repr(float(x)) if not float(x).is_integer() else str(int(x)) for x in values)


# Quote text for MATLAB (single quotes doubled)
def _m_str(text: str) -> str:
    return "'" + str(text).replace("'", "''") + "'"


def export_matlab(model: AnalysisModel, path) -> None:
    """Write the model as a MATLAB/Octave script (NURBS toolbox layout, 1-based)."""
    rows, cols = model.n_rows, model.n_cols
    coefs = np.empty((4, rows, cols))
    coefs[:3] = np.moveaxis(model.net * model.weights[:, :, np.newaxis], 2, 0)
    coefs[3] = model.weights
    flat = coefs.ravel(order="F")  # MATLAB stores column-major

    lines = [
        "% IGA analysis model exported from IGA Designer (no solver included).",
        "% Control point k (1-based) is row i, column j of the net: k = (i-1)*n_cols + j,",
        "% with i along u and j along v.",
        "",
        "% Geometry in NURBS toolbox format (nrbmak / GeoPDEs)",
        "geometry.form = 'B-NURBS';",
        "geometry.dim = 4;",
        f"geometry.number = [{rows} {cols}];",
        f"geometry.coefs = reshape([{_m_row(flat)}], [4 {rows} {cols}]);",
        f"geometry.knots = {{[{_m_row(model.knots_u)}], [{_m_row(model.knots_v)}]}};",
        f"geometry.order = [{model.degree_u + 1} {model.degree_v + 1}];",
        "",
        "% Mesh",
        f"mesh.planar = {str(model.planar).lower()};",
        f"mesh.elements_u = {model.n_elements_u};",
        f"mesh.elements_v = {model.n_elements_v};",
        f"mesh.gauss_points = [{model.gauss_u} {model.gauss_v}];",
        f"mesh.total_area = {model.total_area!r};",
        "",
        "% Elements: one row each",
        "elements.u_range = [" + "; ".join(_m_row(e.u_range) for e in model.elements) + "];",
        "elements.v_range = [" + "; ".join(_m_row(e.v_range) for e in model.elements) + "];",
        "elements.connectivity = [" + "; ".join(
            _m_row([c + 1 for c in e.control_points]) for e in model.elements) + "];",
        "elements.area = [" + "; ".join(repr(e.area) for e in model.elements) + "];",
        "elements.jacobian_min = [" + "; ".join(repr(e.jacobian_min) for e in model.elements) + "];",
        "elements.jacobian_max = [" + "; ".join(repr(e.jacobian_max) for e in model.elements) + "];",
        "elements.status = {" + "; ".join(_m_str(e.status) for e in model.elements) + "};",
        "",
        "% Degrees of freedom",
    ]
    for key, val in model.dofs().items():
        lines.append(f"dofs.{key} = {val};")
    lines += ["", "% Boundary edges (control points on each edge) and their conditions"]
    for k, name in enumerate(EDGE_NAMES, start=1):
        idx = [c + 1 for c in model.boundaries[name]]
        lines.append(f"boundaries({k}).name = {_m_str(name)};")
        lines.append(f"boundaries({k}).control_points = [{_m_row(idx)}];")
        lines.append(f"boundaries({k}).condition = {_m_str(model.conditions[name])};")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- plotting data

def element_mesh_lines(ku, kv, du, dv, net, weights, samples: int = 60) -> List[np.ndarray]:
    """Curves along every element boundary, for drawing the mesh."""
    from .nurbs import nurbs_surface, sample_parameters
    bu, bv = element_boundaries(ku, du), element_boundaries(kv, dv)
    su = sample_parameters(ku, samples, du)
    sv = sample_parameters(kv, samples, dv)
    lines = []
    X, Y, Z = nurbs_surface(ku, kv, du, dv, net, weights, u_values=bu, v_values=sv)
    lines += [np.column_stack([X[a], Y[a], Z[a]]) for a in range(len(bu))]
    X, Y, Z = nurbs_surface(ku, kv, du, dv, net, weights, u_values=su, v_values=bv)
    lines += [np.column_stack([X[:, b], Y[:, b], Z[:, b]]) for b in range(len(bv))]
    return lines
