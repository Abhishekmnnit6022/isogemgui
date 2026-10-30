#!/usr/bin/env python3
"""
verify_project.py  -  independent audit of the restructured CAGD project.

Put this file in the project root (next to main.py / app.py / core/) and run:

    python verify_project.py            # full audit
    python verify_project.py --no-gui   # skip the PyQt5 checks
    python verify_project.py --no-pytest

What it does
  1. STRUCTURE : required files/folders, forbidden imports, style rules
  2. MATH      : compares core/ against its OWN reference implementations
                 (math.comb Bernstein, Cox-de Boor, scipy BSpline, convex-hull LP)
  3. VALIDATION: bad input must raise a clear error
  4. JSON      : save/load round trip, default weights, bundled examples
  5. PYTEST    : runs the project's own test-suite
  6. GUI       : headless start-up, tabs, menu, palette, screenshots, button smoke test

Core functions are located automatically by name and parameter names, so it works
with whatever names the builder chose. If auto-discovery fails for something, set
the OVERRIDES dict below, e.g.  "bezier_curve": "core.bezier:bezier_curve".

Output: console + verify_report.txt + verify_output/*.png (tab screenshots)
Exit code: 0 if no FAIL, 1 otherwise.
"""
import argparse
import importlib
import inspect
import json
import math
import os
import pkgutil
import re
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Manual overrides (only needed if auto-discovery picks the wrong function)
# keys: bezier_curve, bezier_surface, bspline_basis, bspline_curve, knot_vector,
#       json_save, json_load
# --------------------------------------------------------------------------- #
OVERRIDES = {
    # "bezier_curve": "core.bezier:bezier_curve",
}

ROOT = Path(".").resolve()
RESULTS = []


def rec(status, name, detail=""):
    RESULTS.append((status, name, detail))
    tag = {"PASS": "[ PASS ]", "FAIL": "[ FAIL ]", "WARN": "[ WARN ]", "SKIP": "[ SKIP ]"}[status]
    line = f"{tag} {name}"
    if detail:
        line += f"\n         {detail}"
    print(line)


class AdapterError(Exception):
    """Could not find / call a function in the project."""


def check(name, fn):
    try:
        out = fn()
        if out is None:
            rec("PASS", name)
        elif isinstance(out, tuple):
            rec(out[0], name, out[1])
        else:
            rec("PASS", name, str(out))
    except AdapterError as e:
        rec("SKIP", name, f"could not locate/call: {e}")
    except AssertionError as e:
        rec("FAIL", name, str(e))
    except Exception as e:  # noqa
        tb = traceback.format_exc().strip().splitlines()[-1]
        rec("FAIL", name, f"{type(e).__name__}: {e}  ({tb})")


# --------------------------------------------------------------------------- #
# Independent reference mathematics
# --------------------------------------------------------------------------- #
def ref_bernstein(n, i, u):
    return math.comb(n, i) * u ** i * (1 - u) ** (n - i)


def ref_bezier_curve(P, us):
    P = np.asarray(P, float)
    n = len(P) - 1
    B = np.array([[ref_bernstein(n, i, u) for i in range(n + 1)] for u in us])
    return B @ P


def ref_decasteljau(P, u):
    pts = np.asarray(P, float)
    while len(pts) > 1:
        pts = (1 - u) * pts[:-1] + u * pts[1:]
    return pts[0]


def ref_bezier_surface(net, us, vs):
    net = np.asarray(net, float)
    m, n = net.shape[0] - 1, net.shape[1] - 1
    Bu = np.array([[ref_bernstein(m, i, u) for i in range(m + 1)] for u in us])
    Bv = np.array([[ref_bernstein(n, j, v) for j in range(n + 1)] for v in vs])
    return np.einsum("ai,bj,ijk->abk", Bu, Bv, net)


def ref_N(i, p, u, U):
    """Cox-de Boor; right end point u == U[-1] handled (last non-empty span closed)."""
    if p == 0:
        if U[i] <= u < U[i + 1]:
            return 1.0
        if u == U[-1] and U[i] < U[i + 1] == U[-1]:
            return 1.0
        return 0.0
    a = b = 0.0
    d1 = U[i + p] - U[i]
    d2 = U[i + p + 1] - U[i + 1]
    if d1 > 0:
        a = (u - U[i]) / d1 * ref_N(i, p - 1, u, U)
    if d2 > 0:
        b = (U[i + p + 1] - u) / d2 * ref_N(i + 1, p - 1, u, U)
    return a + b


def ref_basis_matrix(U, p, us):
    n1 = len(U) - p - 1
    return np.array([[ref_N(i, p, u, U) for i in range(n1)] for u in us])


def ref_bspline_curve(P, U, p, us):
    return ref_basis_matrix(U, p, us) @ np.asarray(P, float)


def in_hull(points, P):
    """LP test: every row of `points` is a convex combination of rows of P."""
    from scipy.optimize import linprog
    P = np.asarray(P, float)
    bad = 0
    for q in points:
        A = np.vstack([P.T, np.ones(len(P))])
        b = np.append(q, 1.0)
        r = linprog(np.zeros(len(P)), A_eq=A, b_eq=b, bounds=(0, None), method="highs")
        if r.status != 0:
            bad += 1
    return bad


# --------------------------------------------------------------------------- #
# Discovery + flexible calling
# --------------------------------------------------------------------------- #
def import_project_modules():
    sys.path.insert(0, str(ROOT))
    mods = {}
    try:
        core = importlib.import_module("core")
    except Exception as e:
        raise AdapterError(f"cannot import package 'core' ({e})")
    for mi in pkgutil.iter_modules(core.__path__):
        try:
            mods[f"core.{mi.name}"] = importlib.import_module(f"core.{mi.name}")
        except Exception as e:
            rec("FAIL", f"import core.{mi.name}", f"{type(e).__name__}: {e}")
    return mods


def all_funcs(mods):
    out = []
    for mn, m in mods.items():
        for n, f in inspect.getmembers(m, inspect.isfunction):
            if not n.startswith("_") and f.__module__.startswith("core"):
                out.append((f"{mn}.{n}".lower(), f))
    return out


def find_funcs(mods, include, exclude="", prefer=""):
    inc, exc = re.compile(include), (re.compile(exclude) if exclude else None)
    pref = re.compile(prefer) if prefer else None
    c = [(q, f) for q, f in all_funcs(mods) if inc.search(q) and not (exc and exc.search(q))]
    c.sort(key=lambda t: (0 if (pref and pref.search(t[0])) else 1, len(t[0])))
    return [f for _, f in c]


def override(key):
    if key in OVERRIDES:
        mod, fn = OVERRIDES[key].split(":")
        return [getattr(importlib.import_module(mod), fn)]
    return None


RULES = [
    ("knots", r"knot"),
    ("net", r"net|grid"),
    ("x", r"^x(_?(coords?|points?|cp|ctrl|vals?))?$"),
    ("y", r"^y(_?(coords?|points?|cp|ctrl|vals?))?$"),
    ("z", r"^z(_?(coords?|points?|cp|ctrl|vals?))?$"),
    ("weights", r"weight"),
    ("path", r"path|file|fname|^fp$"),
    ("type", r"^(type|kind|geom_?type|geometry_?type)$"),
    ("ncp", r"n_?cps?$|num_?cps?|n_?control|num_?control|n_?ctrl|num_?ctrl"),
    ("num", r"num|cells|n_?pts|n_?points|samples|resolution|steps|^count$|n_?eval|_eval$|^eval"),
    ("cp", r"control|ctrl|^cps?$|^pts$|^points?$"),
    ("idx", r"^(i|idx|index)$"),
    ("degree", r"^(p|n|deg|degree)$|degree"),
    ("order", r"^order$"),
    ("u", r"^(u|v|us|vs|ts?|params?|u_?vals?|t_?vals?|u_?values|t_?values|u_?array|parameter.*)$"),
    ("data", r"data|geom|geometry|obj|spec|model|doc|content|curve|surface"),
]


def classify(name, ctx=None):
    if name == "U":
        return "knots"
    if name == "P":
        return "cp"
    if ctx == "knots" and name == "n":
        return "ncp"
    low = name.lower()
    for key, rx in RULES:
        if re.search(rx, low):
            return key
    return None


def bind(fn, sem, ctx=None):
    sem = dict(sem)
    if "degree" in sem:
        sem.setdefault("order", sem["degree"] + 1)
    sig = inspect.signature(fn)
    args, info = [], {"used_u": False, "used_num": False}
    kwargs = {}
    for pname, par in sig.parameters.items():
        if par.kind in (par.VAR_POSITIONAL, par.VAR_KEYWORD):
            continue
        cls = classify(pname, ctx)
        if cls in sem:
            val = sem[cls]
            info["used_u"] |= cls == "u"
            info["used_num"] |= cls == "num"
            if par.kind == par.POSITIONAL_ONLY:
                args.append(val)
            else:
                kwargs[pname] = val
        elif par.default is not inspect.Parameter.empty:
            continue
        else:
            raise AdapterError(f"{fn.__module__}.{fn.__name__}: unknown required parameter '{pname}'")
    return args, kwargs, info


def call(fn, sem, ctx=None):
    a, k, info = bind(fn, sem, ctx)
    return fn(*a, **k), info


def first_working(cands, sem, accept, ctx=None):
    """Return (fn, result, info) of first candidate that runs and passes `accept`."""
    errs = []
    for fn in cands:
        try:
            res, info = call(fn, sem, ctx)
            if accept(res):
                return fn, res, info
            errs.append(f"{fn.__name__}: unexpected return shape")
        except AdapterError as e:
            errs.append(str(e))
        except Exception as e:  # noqa
            errs.append(f"{fn.__name__}: {type(e).__name__}: {e}")
    raise AdapterError("; ".join(errs) if errs else "no candidate functions found")


def as_points(res, dim=None):
    if isinstance(res, (tuple, list)) and len(res) > 0 and not np.isscalar(res[0]):
        arrs = [np.asarray(r, float) for r in res]
        if all(a.ndim == 1 and a.shape == arrs[0].shape for a in arrs) and len(arrs) in (2, 3):
            return np.stack(arrs, axis=1)
        res = arrs[0]
    a = np.asarray(res, float)
    if a.ndim == 2 and dim and a.shape[0] == dim and a.shape[1] != dim:
        a = a.T
    return a


def as_grid3(res):
    if isinstance(res, (tuple, list)) and len(res) == 3 and np.asarray(res[0]).ndim == 2:
        return np.stack([np.asarray(r, float) for r in res], axis=-1)
    a = np.asarray(res, float)
    if a.ndim == 3 and a.shape[0] == 3 and a.shape[-1] != 3:
        a = np.moveaxis(a, 0, -1)
    return a


# --------------------------------------------------------------------------- #
# Adapters
# --------------------------------------------------------------------------- #
class Adapters:
    def __init__(self, mods):
        self.mods = mods
        self.cache = {}

    def _run(self, key, make_cands, sem):
        """First call discovers the function (swallowing errors); later calls use the
        cached function directly so genuine exceptions (validation tests) propagate."""
        if key in self.cache:
            fn = self.cache[key]
            res, info = call(fn, sem)
            return fn, res, info
        fn, res, info = first_working(make_cands(), sem, lambda r: r is not None)
        self.cache[key] = fn
        return fn, res, info

    # ---- Bezier curve -----------------------------------------------------
    def bezier_curve(self, P, us):
        P = np.asarray(P, float)
        mk = lambda: override("bezier_curve") or find_funcs(
            self.mods, r"bezier.*curve|curve.*bezier|bezier\.(bezier|evaluate|curve|general)",
            r"surface|basis|bernstein|plot|draw|deriv|valid|load|save|json")
        sem = {"cp": P, "u": us, "num": len(us), "degree": len(P) - 1}
        fn, res, info = self._run("bezier_curve", mk, sem)
        pts = as_points(res, P.shape[1])
        return pts, info, fn

    # ---- B-spline basis (matrix) -----------------------------------------
    def bspline_basis(self, U, p, us):
        cands = override("bspline_basis") or find_funcs(
            self.mods, r"basis", r"plot|draw|bernstein|bezier|valid|excel|store")
        n1 = len(U) - p - 1
        sem = {"knots": list(U), "degree": p, "u": us, "num": len(us)}
        errs = []
        for fn in cands:
            try:
                # matrix-style
                res, info = call(fn, sem)
                if isinstance(res, (tuple, list)):  # e.g. (u, N) -> keep the 2-D matrix part
                    mats = [np.asarray(r, float) for r in res if np.asarray(r).ndim == 2]
                    if not mats:
                        raise AdapterError(f"{fn.__name__}: returned tuple without a 2-D matrix")
                    res = mats[0]
                a = np.asarray(res, float)
                if a.ndim == 2 and n1 in a.shape:
                    return (a if a.shape[0] == len(us) else a.T), fn
                if a.ndim == 2 and a.shape == (len(us), n1):
                    return a, fn
            except AdapterError as e:
                errs.append(str(e))
            except Exception as e:  # noqa
                errs.append(f"{fn.__name__}: {type(e).__name__}: {e}")
            try:
                # scalar-style  N(i, p, u, U)
                M = np.zeros((len(us), n1))
                for k, u in enumerate(us):
                    for i in range(n1):
                        r, _ = call(fn, {"idx": i, "degree": p, "u": float(u), "knots": list(U), "num": 1})
                        M[k, i] = float(r)
                return M, fn
            except Exception:  # noqa
                pass
        raise AdapterError("; ".join(errs) if errs else "no basis function found")

    # ---- B-spline curve ---------------------------------------------------
    def bspline_curve(self, P, U, p, us):
        P = np.asarray(P, float)
        mk = lambda: override("bspline_curve") or find_funcs(
            self.mods, r"(spline|bspline).*curve|curve.*spline|bspline\.(curve|evaluate)",
            r"basis|plot|draw|valid|load|save|json|bezier|knot_?vec")
        sem = {"cp": P, "knots": list(U), "degree": p, "u": us, "num": len(us)}
        fn, res, info = self._run("bspline_curve", mk, sem)
        return as_points(res, P.shape[1]), info, fn

    # ---- Bezier surface ---------------------------------------------------
    def bezier_surface(self, net, us, vs):
        if "bezier_surface" in self.cache:  # validation path: pass raw input, never pre-convert
            fn = self.cache["bezier_surface"]
            sem = {"net": net, "cp": net, "u": us, "num": len(us)}
            try:
                a = np.asarray(net, float)
                sem.update(x=a[..., 0], y=a[..., 1], z=a[..., 2])
            except Exception:  # noqa
                sem.update(x=[r[0] for r in net], y=[r[1] for r in net], z=[r[2] for r in net])
            res, info = call(fn, sem)
            return as_grid3(res), info, fn
        net = np.asarray(net, float)
        mk = lambda: override("bezier_surface") or find_funcs(
            self.mods, r"surface", r"plot|draw|valid|load|save|json|basis|hull")
        try:
            sem = {"net": net, "cp": net, "x": net[..., 0], "y": net[..., 1], "z": net[..., 2],
                   "u": us, "num": len(us)}
        except IndexError:
            sem = {"net": net, "cp": net, "u": us, "num": len(us)}
        fn, res, info = self._run("bezier_surface", mk, sem)
        return as_grid3(res), info, fn

    # ---- knots ------------------------------------------------------------
    def knot_vector(self, ncp, p):
        cands = override("knot_vector") or find_funcs(self.mods, r"knot", r"valid|plot|draw|json", prefer=r"open|clamp")
        sem = {"ncp": ncp, "degree": p, "num": ncp, "cp": [[0, 0]] * ncp}
        fn, res, _ = first_working(cands, sem, lambda r: r is not None, ctx="knots")
        return np.asarray(res, float).ravel(), fn


# --------------------------------------------------------------------------- #
# 1. STRUCTURE
# --------------------------------------------------------------------------- #
def py_files(*subs):
    out = []
    for s in subs:
        p = ROOT / s
        if p.is_file():
            out.append(p)
        elif p.is_dir():
            out += [f for f in p.rglob("*.py") if "__pycache__" not in f.parts]
    return out


def grep(files, rx, flags=re.I):
    hits = []
    r = re.compile(rx, flags)
    for f in files:
        for ln, line in enumerate(f.read_text(errors="ignore").splitlines(), 1):
            if line.strip().startswith("#"):
                continue
            if r.search(line):
                hits.append(f"{f.relative_to(ROOT)}:{ln}: {line.strip()[:80]}")
    return hits


def run_structure():
    print("\n=== 1. STRUCTURE ===")
    for rel in ["main.py", "app.py", "core/__init__.py", "core/bezier.py", "core/bspline.py",
                "core/knots.py", "core/io_json.py", "core/validation.py",
                "ui/bezier_tab.py", "ui/surface_tab.py", "ui/bspline_tab.py",
                "requirements.txt", "README.md"]:
        check(f"exists: {rel}", lambda rel=rel: None if (ROOT / rel).exists() else
              (_ for _ in ()).throw(AssertionError("missing")))

    def tests_present():
        n = len(list((ROOT / "tests").glob("test_*.py"))) if (ROOT / "tests").is_dir() else 0
        assert n >= 3, f"found {n} test files, expected >= 3"
    check("tests/ has >= 3 test files", tests_present)

    def examples():
        n = len(list((ROOT / "examples").glob("*.json"))) if (ROOT / "examples").is_dir() else 0
        assert n >= 3, f"found {n} example JSON files, expected >= 3"
    check("examples/ has >= 3 JSON files", examples)

    def archive():
        n = len(list((ROOT / "archive").rglob("*.py"))) if (ROOT / "archive").is_dir() else 0
        assert n >= 10, f"archive/ has {n} .py files, old scripts should be kept (>= 10)"
    check("archive/ keeps old scripts", archive)

    def matlab():
        d = next((ROOT / n for n in ("Matlab", "matlab") if (ROOT / n).is_dir()), None)
        assert d, "Matlab folder missing"
        n = len(list(d.rglob("*.m")))
        assert n >= 5, f"only {n} .m files found"
    check("Matlab folder kept", matlab)

    core = py_files("core")
    ui = py_files("ui", "app.py", "main.py")

    def core_pure():
        h = grep(core, r"^\s*(import|from)\s+(matplotlib|PyQt5|tkinter|PySide)")
        h += grep(core, r"\binput\(")
        assert not h, "GUI/plot/input code in core/:\n         " + "\n         ".join(h[:6])
    check("core/ is pure (no matplotlib/PyQt/tkinter/input)", core_pure)

    def no_print():
        h = grep(core, r"\bprint\(")
        return ("WARN", f"{len(h)} print() calls in core/, e.g. {h[0]}") if h else None
    check("core/ has no print() calls", no_print)

    def no_tk():
        h = grep(py_files("core", "ui", "app.py", "main.py", "tests"), r"^\s*(import|from)\s+tkinter")
        assert not h, "Tkinter still used:\n         " + "\n         ".join(h[:6])
    check("no Tkinter outside archive/", no_tk)

    def no_sub():
        h = grep(ui, r"subprocess|os\.system|Popen")
        assert not h, "tabs/launcher still spawn subprocesses:\n         " + "\n         ".join(h[:6])
    check("no subprocess launching in app/ui/main", no_sub)

    def no_npmath():
        h = grep(py_files("core", "ui", "app.py", "main.py"), r"np\.math\.|numpy\.math\.")
        assert not h, "np.math is removed in new NumPy:\n         " + "\n         ".join(h[:4])
    check("no np.math usage", no_npmath)

    def style_rules():
        h = grep(ui, r"setStyleSheet|gradient|border-radius|box-shadow|QPalette\(|setPalette|Fusion")
        if h:
            return ("WARN", "modern/custom styling found (you asked for plain old-school):\n         "
                    + "\n         ".join(h[:8]))
    check("no custom stylesheet / palette / Fusion", style_rules)

    def colors():
        h = grep(ui, r"dark|navy|#[0-3][0-9a-f]{5}\b|rgb\(\s*[0-5]?\d\s*,\s*[0-5]?\d\s*,\s*[0-9]{2,3}\s*\)")
        if h:
            return ("WARN", "possible dark / navy colors:\n         " + "\n         ".join(h[:8]))
    check("no dark or navy colour values", colors)

    def reqs():
        t = (ROOT / "requirements.txt").read_text().lower()
        miss = [p for p in ("numpy", "matplotlib", "pyqt5", "pytest") if p not in t]
        assert not miss, f"requirements.txt missing {miss}"
    check("requirements.txt lists key packages", reqs)


# --------------------------------------------------------------------------- #
# 2. MATH
# --------------------------------------------------------------------------- #
THESIS_U = [0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 5]
THESIS_P = np.array([[5, 20], [5, 70], [50, 2], [50, 40], [70, 2], [80, 40], [90, 30], [100, 30]], float)


def sample_us(U, p, n=101):
    return np.linspace(U[p], U[-p - 1], n)


def curve_vs_ref(A, P, U, p, label):
    """Compare adapter bspline_curve with reference."""
    us = sample_us(U, p)
    pts, info, fn = A.bspline_curve(P, U, p, us)
    ref = ref_bspline_curve(P, U, p, us)
    if pts.shape != ref.shape:
        # function sampled itself (num) -> compare as point cloud against dense reference
        if pts.ndim == 2 and pts.shape[1] == ref.shape[1]:
            dense = ref_bspline_curve(P, U, p, np.linspace(U[p], U[-p - 1], 4000))
            d = max(np.min(np.linalg.norm(dense - q, axis=1)) for q in pts)
            assert d < 0.01 * np.ptp(P), f"{label}: points not on reference curve (max dist {d:.3g})"
            return
        raise AssertionError(f"{label}: unexpected output shape {pts.shape}, expected {ref.shape}")
    err = np.abs(pts - ref).max()
    assert err < 1e-8, f"{label}: max error vs reference {err:.3g}"


def run_math(mods):
    print("\n=== 2. MATH (independent references) ===")
    A = Adapters(mods)
    rng = np.random.default_rng(7)

    # --- sanity of my own references ---------------------------------------
    def selfcheck():
        P = rng.random((5, 2))
        us = np.linspace(0, 1, 9)
        r1 = ref_bezier_curve(P, us)
        r2 = np.array([ref_decasteljau(P, u) for u in us])
        assert np.allclose(r1, r2), "reference Bernstein != de Casteljau"
        assert abs(ref_N(0, 2, 0.5, THESIS_U) - 0.25) < 1e-12, "reference basis hand value wrong"
    check("(self-test) reference implementations agree", selfcheck)

    # --- Bezier curve ------------------------------------------------------
    def bez_random(dim, degs):
        for n in degs:
            P = rng.random((n + 1, dim)) * 10
            us = np.linspace(0, 1, 101)
            pts, info, fn = A.bezier_curve(P, us)
            ref = ref_bezier_curve(P, us)
            if pts.shape != ref.shape:
                dense = ref_bezier_curve(P, np.linspace(0, 1, 4000))
                d = max(np.min(np.linalg.norm(dense - q, axis=1)) for q in pts)
                assert d < 0.01 * np.ptp(P), f"degree {n}: points not on reference curve (dist {d:.3g})"
            else:
                e = np.abs(pts - ref).max()
                assert e < 1e-8, f"degree {n} {dim}D: max error {e:.3g}"
    check("Bezier curve 2D, degrees 1..9 match Bernstein reference", lambda: bez_random(2, range(1, 10)))
    check("Bezier curve 3D, degrees 1..5 match Bernstein reference", lambda: bez_random(3, range(1, 6)))

    def bez_closed_form():
        P = np.array([[1, 1], [2, 3], [4, 3], [3, 1]], float)
        pts, _, _ = A.bezier_curve(P, np.array([0.0, 0.5, 1.0]))
        if pts.shape[0] != 3:
            raise AdapterError("function resamples internally; cannot probe u=0.5")
        assert np.allclose(pts[1], [2.75, 2.5]), f"cubic at u=.5 should be (2.75, 2.5), got {pts[1]}"
    check("cubic Bezier (1,1),(2,3),(4,3),(3,1) at u=0.5 = (2.75, 2.5)", bez_closed_form)

    def bez_endpoints():
        for n in (1, 2, 3, 6):
            P = rng.random((n + 1, 2)) * 10
            pts, _, _ = A.bezier_curve(P, np.linspace(0, 1, 51))
            assert np.allclose(pts[0], P[0]) and np.allclose(pts[-1], P[-1]), f"degree {n}: endpoints not interpolated"
    check("Bezier curve interpolates first and last control point", bez_endpoints)

    def bez_hull():
        P = rng.random((6, 2)) * 10
        pts, _, _ = A.bezier_curve(P, np.linspace(0, 1, 60))
        bad = in_hull(pts[::3], P)
        assert bad == 0, f"{bad} sampled points outside convex hull"
    check("Bezier curve lies in convex hull of control points", bez_hull)

    # --- B-spline basis ----------------------------------------------------
    cases = [
        ("thesis open non-uniform, p=2", THESIS_U, 2),
        ("clamped uniform, p=2", [0, 0, 0, 1, 2, 3, 3, 3], 2),
        ("clamped, p=3 with double knot", [0, 0, 0, 0, .25, .5, .75, .75, 1, 1, 1, 1], 3),
        ("Bezier-like p=4", [0] * 5 + [1] * 5, 4),
    ]
    for label, U, p in cases:
        def basis_ref(U=U, p=p, label=label):
            us = sample_us(U, p)
            M, fn = A.bspline_basis(U, p, us)
            ref = ref_basis_matrix(U, p, us)
            assert M.shape == ref.shape, f"shape {M.shape}, expected {ref.shape}"
            e = np.abs(M - ref).max()
            assert e < 1e-9, f"max error vs reference {e:.3g}"
        check(f"basis matrix matches Cox-de Boor reference ({label})", basis_ref)

        def pou(U=U, p=p):
            us = sample_us(U, p)
            M, fn = A.bspline_basis(U, p, us)
            s = M.sum(axis=1)
            bad = np.where(np.abs(s - 1) > 1e-9)[0]
            assert len(bad) == 0, f"sum != 1 at u={us[bad][:3]} (sums {s[bad][:3]}), last-knot handling?"
            assert M.min() > -1e-12, "negative basis value"
        check(f"partition of unity incl. right end point ({label})", pou)

    def basis_hand():
        us = np.linspace(0, 5, 11)  # contains u = 0.5 at index 1 (works for self-sampling functions too)
        M, _ = A.bspline_basis(THESIS_U, 2, us)
        assert abs(M[1, 0] - 0.25) < 1e-12, f"N0,2(0.5) should be 0.25, got {M[1, 0]}"
    check("hand value N0,2(0.5) = 0.25 for thesis knots", basis_hand)

    # --- B-spline curve ----------------------------------------------------
    check("B-spline curve vs reference (thesis example)",
          lambda: curve_vs_ref(A, THESIS_P, THESIS_U, 2, "thesis"))

    def bs_scipy():
        from scipy.interpolate import BSpline
        U = np.array([0, 0, 0, 0, .25, .5, .75, .75, 1, 1, 1, 1], float)
        P = rng.random((8, 2)) * 10
        us = np.linspace(0, 1, 101)
        pts, _, _ = A.bspline_curve(P, U, 3, us)
        ref = BSpline(U, P, 3)(us)
        if pts.shape != ref.shape:
            raise AdapterError("function resamples internally")
        e = np.abs(pts - ref).max()
        assert e < 1e-8, f"max error vs scipy BSpline {e:.3g}"
    check("B-spline curve matches scipy.interpolate.BSpline (p=3)", bs_scipy)

    def bs_endpoints():
        for U, p, n in [(THESIS_U, 2, 8), ([0, 0, 0, 1, 2, 3, 3, 3], 2, 5)]:
            P = rng.random((n, 2)) * 10
            pts, _, _ = A.bspline_curve(P, U, p, sample_us(U, p))
            assert np.allclose(pts[0], P[0]), "does not start at first control point"
            assert np.allclose(pts[-1], P[-1]), "does not end at last control point (u = last knot bug?)"
    check("clamped B-spline interpolates end control points", bs_endpoints)

    def bs_hull():
        P = THESIS_P
        pts, _, _ = A.bspline_curve(P, THESIS_U, 2, sample_us(THESIS_U, 2, 60))
        bad = in_hull(pts[::3], P)
        assert bad == 0, f"{bad} points outside convex hull"
    check("clamped B-spline lies in convex hull", bs_hull)

    def bs_equals_bezier():
        P = rng.random((10, 2)) * 10
        U = [0] * 10 + [1] * 10
        us = np.linspace(0, 1, 101)
        pts, _, _ = A.bspline_curve(P, U, 9, us)
        ref = ref_bezier_curve(P, us)
        if pts.shape != ref.shape:
            raise AdapterError("function resamples internally")
        e = np.abs(pts - ref).max()
        assert e < 1e-8, f"degree-9 B-spline != Bezier, error {e:.3g}"
    check("B-spline with knots [0]*10+[1]*10 equals degree-9 Bezier", bs_equals_bezier)

    # --- surface -----------------------------------------------------------
    thesis_x = np.array([[-.5, -2, 0], [1, 1, 1], [2, 2, 2]])
    thesis_y = np.array([[2, 1, 0], [2, 8, -1], [2, 1, 1]])
    thesis_z = np.array([[1, -1, 2], [0, -8.5, 2], [.5, 1, 2]])
    net33 = np.stack([thesis_x, thesis_y, thesis_z], axis=-1).astype(float)
    net43 = rng.random((4, 3, 3)) * 5
    us = np.linspace(0, 1, 21)

    def surf(net, hard=True):
        S, info, fn = A.bezier_surface(net, us, us)
        ref = ref_bezier_surface(net, us, us)
        if S.shape != ref.shape:
            raise AssertionError(f"output grid shape {S.shape}, expected {ref.shape}")
        e1 = np.abs(S - ref).max()
        e2 = np.abs(S - np.transpose(ref, (1, 0, 2))).max()
        if e1 < 1e-8:
            return None
        if e2 < 1e-8:
            return ("WARN", "surface matches only after transposing u/v (row/column convention differs)")
        raise AssertionError(f"max error vs tensor-product reference {min(e1, e2):.3g}")

    check("Bezier surface 3x3 (thesis general net) matches reference", lambda: surf(net33))
    check("Bezier surface non-square 4x3 net matches reference",
          lambda: ("WARN", "non-square net failed") if _try(lambda: surf(net43)) is False else surf(net43))

    def surf_corners():
        S, _, _ = A.bezier_surface(net33, us, us)
        corners = {S[0, 0], S[0, -1], S[-1, 0], S[-1, -1]} if False else None
        got = np.array([S[0, 0], S[0, -1], S[-1, 0], S[-1, -1]])
        want = np.array([net33[0, 0], net33[0, -1], net33[-1, 0], net33[-1, -1]])
        for q in want:
            assert np.min(np.linalg.norm(got - q, axis=1)) < 1e-8, f"corner {q} not on surface"
    check("Bezier surface passes through the 4 corner control points", surf_corners)

    def surf_hull():
        S, _, _ = A.bezier_surface(net43, us, us)
        pts = S.reshape(-1, 3)[::37]
        bad = in_hull(pts, net43.reshape(-1, 3))
        assert bad == 0, f"{bad} surface points outside convex hull"
    check("Bezier surface lies in convex hull of control net", surf_hull)

    # --- knots -------------------------------------------------------------
    def knots_ok():
        for ncp, p in [(6, 2), (8, 3), (4, 3)]:
            U, fn = A.knot_vector(ncp, p)
            assert np.all(np.diff(U) >= 0), f"knots not non-decreasing for ncp={ncp}, p={p}: {U}"
            assert np.all(U[:p + 1] == U[0]) and np.all(U[-p - 1:] == U[-1]), f"not clamped: {U}"
            assert len(U) in (ncp + p + 1, ncp + p + 2), f"length {len(U)} for ncp={ncp}, p={p}"
    check("knot generator returns clamped, non-decreasing vector", knots_ok)


def _try(fn):
    try:
        fn()
        return True
    except Exception:  # noqa
        return False


# --------------------------------------------------------------------------- #
# 3. VALIDATION
# --------------------------------------------------------------------------- #
def run_validation(mods):
    print("\n=== 3. VALIDATION ===")
    A = Adapters(mods)
    U = THESIS_U
    us = sample_us(U, 2)
    # warm-up with VALID input so function discovery succeeds; the bad-input calls below
    # then hit the discovered function directly and its exceptions are judged.
    for _w in (lambda: A.bspline_curve(THESIS_P, U, 2, us),
               lambda: A.bezier_curve(np.array([[0., 0.], [1, 2], [3, 1]]), np.linspace(0, 1, 5)),
               lambda: A.bezier_surface(np.zeros((3, 3, 3)), np.linspace(0, 1, 5), np.linspace(0, 1, 5))):
        try:
            _w()
        except Exception:  # noqa
            pass

    def must_raise(fn, what):
        try:
            fn()
        except AdapterError:
            raise
        except Exception as e:  # noqa
            msg = str(e).strip()
            assert msg, f"{what}: exception {type(e).__name__} has an empty message"
            assert isinstance(e, (ValueError, TypeError)), (
                f"{what}: raised {type(e).__name__}; ValueError expected (message: {msg[:60]})")
            return None
        raise AssertionError(f"{what}: no error raised")

    check("too few control points -> clear ValueError",
          lambda: must_raise(lambda: A.bspline_curve(THESIS_P[:5], U, 2, us), "5 CPs for 11 knots, p=2"))
    check("too many control points -> clear ValueError",
          lambda: must_raise(lambda: A.bspline_curve(np.vstack([THESIS_P, THESIS_P[:2]]), U, 2, us), "10 CPs"))
    check("decreasing knot vector -> clear ValueError",
          lambda: must_raise(lambda: A.bspline_curve(THESIS_P, [0, 0, 0, 2, 1, 3, 4, 4, 5, 5, 5], 2, us), "decreasing"))
    check("degree 0 / negative degree -> clear ValueError",
          lambda: must_raise(lambda: A.bspline_curve(THESIS_P, U, -1, us), "degree -1"))
    check("Bezier with a single control point -> clear ValueError",
          lambda: must_raise(lambda: A.bezier_curve(np.array([[1.0, 2.0]]), np.linspace(0, 1, 5)), "1 CP"))

    def ragged_surface():
        bad = [[[0, 0, 0], [1, 0, 0], [2, 0, 0]], [[0, 1, 0], [1, 1, 0]]]
        try:
            A.bezier_surface(bad, us, us)
        except AdapterError:
            raise
        except Exception:  # noqa
            return None
        raise AssertionError("ragged control net accepted silently")
    check("non-rectangular surface net rejected", ragged_surface)

    def val_module():
        fs = find_funcs(mods, r"valid")
        assert fs, "core/validation.py exposes no functions"
    check("validation module exposes functions", val_module)


# --------------------------------------------------------------------------- #
# 4. JSON
# --------------------------------------------------------------------------- #
def run_json(mods):
    print("\n=== 4. JSON save / load ===")
    saves = override("json_save") or find_funcs(mods, r"io_json.*(save|write|dump|export)")
    loads = override("json_load") or find_funcs(mods, r"io_json.*(load|read|import)")

    geom = {"type": "bspline_curve", "degree": 2, "knots": THESIS_U,
            "control_points": THESIS_P.tolist(), "weights": [1] * 8}
    geom_noweights = {k: v for k, v in geom.items() if k != "weights"}
    tmp = Path(tempfile.mkdtemp()) / "t.json"

    def do_save(data):
        if tmp.exists():
            tmp.unlink()
        sem = {"path": str(tmp), "data": data, "type": data["type"], "degree": data["degree"],
               "knots": data["knots"], "cp": data["control_points"], "weights": data.get("weights")}
        errs = []
        for fn in saves:
            for p in (str(tmp), tmp):
                try:
                    a, k, _ = bind(fn, {**sem, "path": p})
                    fn(*a, **k)
                    if tmp.exists():
                        return fn
                except Exception as e:  # noqa
                    errs.append(f"{fn.__name__}: {e}")
            try:  # positional fallbacks
                fn(data, str(tmp))
                if tmp.exists():
                    return fn
            except Exception as e:  # noqa
                errs.append(f"{fn.__name__}: {e}")
        raise AdapterError("; ".join(errs[:3]) or "no save function")

    def do_load(path):
        errs = []
        for fn in loads:
            for p in (str(path), Path(path)):
                try:
                    a, k, _ = bind(fn, {"path": p})
                    return fn(*a, **k)
                except Exception as e:  # noqa
                    errs.append(f"{fn.__name__}: {e}")
        raise AdapterError("; ".join(errs[:3]) or "no load function")

    def get(obj, key):
        if isinstance(obj, dict):
            return obj.get(key)
        if isinstance(obj, (tuple, list)) and len(obj) == 1:
            return get(obj[0], key)
        return getattr(obj, key, None)

    def schema():
        do_save(geom)
        raw = json.loads(tmp.read_text())
        miss = [k for k in ("type", "degree", "knots", "control_points", "weights") if k not in raw]
        assert not miss, f"saved JSON lacks keys {miss}; has {list(raw)}"
    check("saved file has keys type/degree/knots/control_points/weights", schema)

    def roundtrip():
        do_save(geom)
        back = do_load(tmp)
        for k in ("degree", "knots", "control_points"):
            a, b = get(back, k), geom[k]
            assert a is not None, f"loaded object lacks '{k}'"
            assert np.allclose(np.asarray(a, float), np.asarray(b, float)), f"'{k}' changed in round trip"
        assert str(get(back, "type")) == geom["type"], "type changed in round trip"
    check("save -> load round trip is lossless", roundtrip)

    def default_weights():
        do_save(geom_noweights)
        raw = json.loads(tmp.read_text())
        w = raw.get("weights")
        if w is None:
            w = get(do_load(tmp), "weights")
        assert w is not None, "weights missing when not supplied"
        assert np.allclose(np.asarray(w, float), 1.0) and len(w) == 8, f"default weights should be eight 1s, got {w}"
    check("weights default to 1 for every control point", default_weights)

    def bad_load():
        bad = dict(geom)
        bad["control_points"] = THESIS_P[:5].tolist()
        bad["weights"] = [1] * 5
        tmp.write_text(json.dumps(bad))
        try:
            do_load(tmp)
        except AdapterError:
            return None  # loader raised -> adapter wraps; accept
        except Exception:  # noqa
            return None
        raise AssertionError("loader accepted 5 control points with 11 knots / degree 2")
    check("loading inconsistent geometry is rejected", bad_load)

    def examples():
        files = sorted((ROOT / "examples").glob("*.json"))
        assert files, "no examples/*.json"
        types = set()
        for f in files:
            d = json.loads(f.read_text())
            types.add(d.get("type"))
            do_load(f)
        miss = {"bezier_curve", "bezier_surface", "bspline_curve"} - types
        assert not miss, f"no example for {sorted(miss)}"
    check("examples/*.json all load; one per geometry type", examples)


# --------------------------------------------------------------------------- #
# 5. PYTEST
# --------------------------------------------------------------------------- #
def run_pytest():
    print("\n=== 5. PYTEST ===")
    try:
        p = subprocess.run([sys.executable, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider"],
                           cwd=ROOT, capture_output=True, text=True, timeout=600)
    except Exception as e:  # noqa
        rec("FAIL", "pytest run", str(e))
        return
    tail = "\n         ".join(p.stdout.strip().splitlines()[-6:])
    if p.returncode == 0:
        rec("PASS", "project's own pytest suite", tail.splitlines()[-1].strip())
    elif p.returncode == 5:
        rec("FAIL", "project's own pytest suite", "no tests collected")
    else:
        rec("FAIL", "project's own pytest suite", tail)

    def quality():
        txt = "\n".join(f.read_text(errors="ignore") for f in (ROOT / "tests").glob("test_*.py"))
        n_assert = len(re.findall(r"\bassert\b|allclose|pytest\.raises", txt))
        has_hard = bool(re.search(r"2\.75|0\.25|\[0,\s*0,\s*0,\s*1,\s*2,\s*3,\s*4,\s*4,\s*5,\s*5,\s*5\]", txt))
        need = [k for k, v in {"raises": "raises" in txt, "hull": "hull" in txt.lower(),
                               "partition": "partition" in txt.lower() or "sum" in txt.lower()}.items() if not v]
        msgs = []
        if n_assert < 15:
            msgs.append(f"only {n_assert} assertions")
        if not has_hard:
            msgs.append("no hard-coded reference numbers (MATLAB/closed-form) found")
        if need:
            msgs.append(f"no tests mentioning {need}")
        if msgs:
            return ("WARN", "; ".join(msgs))
    check("project tests look substantive", quality)


# --------------------------------------------------------------------------- #
# 6. GUI (headless)
# --------------------------------------------------------------------------- #
def run_gui():
    print("\n=== 6. GUI (headless smoke test) ===")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        from PyQt5 import QtWidgets, QtGui
    except Exception as e:  # noqa
        rec("SKIP", "GUI checks", f"PyQt5 not importable ({e})")
        return

    errors, dialogs = [], []
    sys.excepthook = lambda t, v, tb: errors.append("".join(traceback.format_exception(t, v, tb))[-400:])
    for nm in ("warning", "critical", "information", "question"):
        setattr(QtWidgets.QMessageBox, nm,
                staticmethod(lambda *a, _n=nm, **k: (dialogs.append((_n, [str(x) for x in a[1:3]])),
                                                     QtWidgets.QMessageBox.Ok)[1]))
    QtWidgets.QMessageBox.exec_ = lambda self: dialogs.append(("exec", [self.text()])) or 0
    QtWidgets.QMessageBox.exec = QtWidgets.QMessageBox.exec_

    qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    sys.path.insert(0, str(ROOT))
    holder = {}

    def make_window():
        mod = importlib.import_module("app")
        cls = [c for _, c in inspect.getmembers(mod, inspect.isclass)
               if issubclass(c, QtWidgets.QMainWindow) and c.__module__ == "app"]
        if not cls:
            raise AssertionError("app.py defines no QMainWindow subclass")
        w = cls[0]()
        w.show()
        qapp.processEvents()
        holder["w"] = w
    check("app.py: main window starts without errors", make_window)
    w = holder.get("w")
    if w is None:
        return
    out = ROOT / "verify_output"
    out.mkdir(exist_ok=True)

    def tabs():
        tw = [t for t in w.findChildren(QtWidgets.QTabWidget) if t.count() >= 3]
        assert tw, "no QTabWidget with >= 3 tabs"
        holder["tabs"] = tw[0]
        names = [tw[0].tabText(i).lower() for i in range(tw[0].count())]
        assert any("bezier" in n or "bézier" in n for n in names), f"no Bezier curve tab in {names}"
        assert any("surface" in n for n in names), f"no surface tab in {names}"
        assert any("spline" in n for n in names), f"no B-spline tab in {names}"
        return f"tabs: {names}"
    check("QTabWidget with Bezier / Surface / B-Spline tabs", tabs)

    def menu():
        mb = w.menuBar()
        top = {a.text().replace("&", "").lower(): a.menu() for a in mb.actions() if a.menu()}
        assert "file" in top, f"no File menu; menus: {list(top)}"
        items = " | ".join(a.text().replace("&", "").lower() for a in top["file"].actions())
        miss = [k for k in ("open", "save", "exit") if k not in items]
        assert not miss, f"File menu lacks {miss}; has: {items}"
    check("File menu has Open / Save / Exit", menu)

    def palette():
        lum = qapp.palette().color(QtGui.QPalette.Window).lightness()
        assert lum >= 180, f"window colour lightness {lum}: looks like a dark theme"
        ss = [x for x in [qapp.styleSheet(), w.styleSheet()] if x.strip()]
        sw = [x.objectName() or x.__class__.__name__ for x in w.findChildren(QtWidgets.QWidget) if x.styleSheet().strip()]
        if ss or sw:
            return ("WARN", f"stylesheets in use ({len(ss)} global, {len(sw)} widgets) - you asked for plain native look")
    check("light native palette, no stylesheets", palette)

    def canvases():
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
        tw = holder["tabs"]
        for i in range(tw.count()):
            tw.setCurrentIndex(i)
            qapp.processEvents()
            page = tw.widget(i)
            cs = page.findChildren(FigureCanvasQTAgg)
            assert cs, f"tab '{tw.tabText(i)}' has no embedded matplotlib canvas"
            if "surface" in tw.tabText(i).lower():
                has3d = any(hasattr(a, "zaxis") for c in cs for a in c.figure.axes) or True
        return None
    check("each tab embeds a FigureCanvasQTAgg", canvases)

    def shots():
        tw = holder["tabs"]
        paths = []
        for i in range(tw.count()):
            tw.setCurrentIndex(i)
            qapp.processEvents()
            p = out / f"tab{i}_{re.sub(r'[^a-z0-9]+', '_', tw.tabText(i).lower())}.png"
            w.grab().save(str(p))
            paths.append(p.name)
        return f"saved {paths} in verify_output/ - LOOK AT THEM (old-school, no dark blue?)"
    check("screenshots of every tab", shots)

    def smoke():
        tw = holder["tabs"]
        pressed = []
        for i in range(tw.count()):
            tw.setCurrentIndex(i)
            qapp.processEvents()
            for b in tw.widget(i).findChildren(QtWidgets.QPushButton):
                if re.search(r"draw|plot|generate|curve|surface", b.text(), re.I) and b.isEnabled():
                    errors_before = len(errors)
                    b.click()
                    qapp.processEvents()
                    pressed.append(b.text())
                    assert len(errors) == errors_before, f"button '{b.text()}' raised: {errors[-1]}"
        assert pressed, "found no Draw/Plot buttons"
        return f"clicked {pressed} on empty input: no crash, {len(dialogs)} message box(es) shown"
    check("Draw/Plot buttons survive empty input (no uncaught exception)", smoke)


# --------------------------------------------------------------------------- #
def manual_checklist():
    return """
MANUAL CHECKLIST (5 minutes, things a script cannot judge)
  [ ] python main.py opens ONE window; no extra console prompts / input() calls
  [ ] B-Spline tab: knots 0,0,0,1,2,3,4,4,5,5,5 / degree 2 / points from thesis Fig 4.19
      -> basis plot has 8 curves and the curve matches thesis Fig 4.20
  [ ] Bezier tab: add (1,1) (2,3) (4,3) (3,1) -> shape matches thesis Fig 4.8
  [ ] Surface tab: thesis 3x3 net -> surface matches Fig 4.14 / 4.16; mouse rotates it
  [ ] Type garbage / leave fields empty -> message box, not a crash
  [ ] File > Save then File > Open restores the same plot
  [ ] Window looks plain and old-school: grey system colours, no cards, no dark blue
  [ ] Window resizes sensibly; plots are not clipped
"""


def main():
    global ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default=".")
    ap.add_argument("--no-gui", action="store_true")
    ap.add_argument("--no-pytest", action="store_true")
    args = ap.parse_args()
    ROOT = Path(args.root).resolve()
    os.chdir(ROOT)
    print(f"Auditing project at: {ROOT}\n")

    run_structure()
    try:
        mods = import_project_modules()
        run_math(mods)
        run_validation(mods)
        run_json(mods)
    except AdapterError as e:
        rec("FAIL", "import core package", str(e))
    if not args.no_pytest:
        run_pytest()
    if not args.no_gui:
        run_gui()

    n = {k: sum(1 for r in RESULTS if r[0] == k) for k in ("PASS", "FAIL", "WARN", "SKIP")}
    print("\n" + "=" * 60)
    print(f"SUMMARY: {n['PASS']} passed, {n['FAIL']} failed, {n['WARN']} warnings, {n['SKIP']} skipped")
    if n["FAIL"]:
        print("\nFAILED CHECKS (paste these back to the AI to fix):")
        for s, nm, d in RESULTS:
            if s == "FAIL":
                print(f" - {nm}\n     {d}")
    if n["SKIP"]:
        print("\nSKIPPED = the script could not find/call that function. Either the function is missing,")
        print("or set OVERRIDES at the top of this file.")
    print(manual_checklist())
    (ROOT / "verify_report.txt").write_text(
        "\n".join(f"{s}\t{nm}\t{d}" for s, nm, d in RESULTS) + "\n" + manual_checklist())
    print("Report written to verify_report.txt")
    sys.exit(1 if n["FAIL"] else 0)


if __name__ == "__main__":
    main()