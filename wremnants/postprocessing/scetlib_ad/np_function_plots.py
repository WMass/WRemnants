"""Two-panel NP-function plot for scetlib_ad fits: CS γ_ν^NP(b_T) | TMD F_eff(b_T, y).

    python -m wremnants.postprocessing.scetlib_ad.np_function_plots \\
        --num-fitresult <A>/fitresults.hdf5 [--den-fitresult <B>/fitresults.hdf5] \\
        [--num-label ...] [--den-label ...] [--y 0 2.5] -o <dir>/

Each fit is drawn as its prefit (λ_central, dashed) and postfit (solid) with a 68 %
band from Gaussian toys of its stored covariance; the numerator takes the viridis
ramp, the denominator plasma (per-y shades on the TMD panel), external references
are grey: the lattice-fit CS kernel of arXiv:2506.13874 Eq. (3.34) (provisional
covariance, a visual guide only) and the MAP22 N3LL intrinsic F_eff analogue
(evolution factor stripped, two-beam product at x = (Q/√s)e^{±Y}; different b*
scheme). At most two fits (num / den).

Ported 2026-10-15 from ``scetlib_np/np_function_plots.py`` (branch
``scetlib-np-param-model`` @ 332b7028): same layout, colour contract, legends, λ
tables and references; the btgrid machinery is replaced by the closed-form tanh_2 /
tanh_6 NP functions SCETlib evaluates (the forms ``np_damping_wall`` supports) and
a reader for scetlib_ad fitresults.

READER. A scetlib_ad fit stores θ, not λ: λ = anchor + width·θ, the anchor from the
card's theory correction. The map is rebuilt offline from the fit's own card with
:func:`np_damping_wall.resolve_wall_inputs` (the wall / NPDomainConstraints use the
same), so walled, trust-constr and unconstrained fits read identically. Toys use
``multivariate_normal(method="eigh")`` on the floating λ only, so a FACE-HELD
(trust-constr) or otherwise singular covariance is fine; on an active face the band
is pinched along the face normal (that width is the face's, not a measurement), and
the active faces are printed. alphaS is never read: only the NP rows of parms/cov.
"""

import argparse
import os
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from wremnants.postprocessing.scetlib_ad import np_damping_wall as ndw

# λ print order of the parameter tables
GNU_ORDER = ("lambda2_nu", "lambda4_nu", "lambda6_nu", "lambda_inf_nu")
EFF_ORDER = ("lambda2", "lambda4", "lambda6", "delta_lambda2", "lambda_inf")
FACE_TOL = 1e-5  # value - bound below this = on an active damping face


# ---------------------------------------------------------------------------
# NP functions (SCETlib np_effective / gamma_nu_np_model, raw b_T)
# ---------------------------------------------------------------------------


@dataclass
class Tune:
    """Both NP forms and the λ they use."""

    np_model: str
    np_model_nu: str
    values: dict = field(default_factory=dict)

    @property
    def eff(self):
        d = {k: self.values[k] for k in ndw._TMD_LAMBDAS[self.np_model]}
        return dict(d, np_model=self.np_model)

    @property
    def gnu(self):
        d = {k: self.values[k] for k in ndw._CS_LAMBDAS[self.np_model_nu]}
        return dict(d, np_model_nu=self.np_model_nu)


def gamma_nu_curve(bT, gnu):
    """γ_ν^NP = -λ∞ν tanh[(λ2ν b² + λ4ν b⁴ [+ λ6ν b⁶]) / λ∞ν]."""
    model, linf = gnu["np_model_nu"], gnu["lambda_inf_nu"]
    if model not in ndw._CS_LAMBDAS:
        raise ValueError(f"unsupported np_model_nu {model!r}")
    bT = np.asarray(bT, dtype=float)
    den = linf if linf != 0 else 1.0
    bT2 = bT * bT
    a = (gnu["lambda2_nu"] + gnu["lambda4_nu"] * bT2) * bT2 / den
    if model == "tanh_6":
        a = a + gnu["lambda6_nu"] * bT2**3 / den
    return np.where(linf == 0, 0.0, -(linf * np.tanh(a)))


def f_eff_curve(bT, y, eff):
    """F_eff = exp[-2 λ∞ b tanh(a)], a = (L2 b + λ4 b³ [+ λ6 b⁵])/λ∞ + (L2 b/λ∞)³/3,
    L2 = λ2 + δλ2 y² (the λ∞-exact arctanh correction of np_damping_wall)."""
    model, linf = eff["np_model"], eff["lambda_inf"]
    if model not in ndw._TMD_LAMBDAS:
        raise ValueError(f"unsupported np_model {model!r}")
    bT = np.asarray(bT, dtype=float)
    den = linf if linf != 0 else 1.0
    l2y = eff["lambda2"] + eff["delta_lambda2"] * y * y
    a = (l2y + eff["lambda4"] * bT**2) * bT / den
    if model == "tanh_6":
        a = a + eff["lambda6"] * bT**5 / den
    a = a + (1.0 / 3.0) * (l2y * bT / den) ** 3
    return np.where(linf == 0, 1.0, np.exp(-(2.0 * linf * bT * np.tanh(a))))


# ---------------------------------------------------------------------------
# scetlib_ad fitresults reader
# ---------------------------------------------------------------------------
_CARDS = {}


def _flat(entry):
    return " ".join(
        " ".join(map(str, e)) if isinstance(e, (list, tuple)) else str(e)
        for e in (entry or [])
    )


def read_ad_fitresult(path, card=None, result=None, n_toys=500, seed=0):
    """``(prefit Tune, postfit Tune, toys [Tune] | None, active faces)``."""
    from rabbit import inputdata, io_tools

    if os.path.isdir(path):
        path = os.path.join(path, "fitresults.hdf5")
    fr, meta = io_tools.get_fitresult(path, result, meta=True)
    args = (meta.get("meta_info") or {}).get("args") or {}
    pm = _flat(args.get("paramModel"))
    if pm and "SCETlibADParamModel" not in pm:
        raise ValueError(f"{path} is not a scetlib_ad fit")
    for tok in ("anchor_override", "anchor_source=cache"):
        if tok in pm:
            print(f"[np_function_plots] WARNING {path}: '{tok}' in the paramModel")
    card = card or args.get("filename")
    if card not in _CARDS:
        _CARDS[card] = ndw.resolve_wall_inputs(inputdata.FitInputData(card))
    inp = _CARDS[card]
    lin = {}
    for n in inp["names"]:
        kind, c = inp["specs"][n]
        if kind is not None and (kind != "quad" or c[2] != 0.0):
            raise ValueError(f"theta -> {n} is not linear: {kind} {c}")
        lin[n] = (0.0, 1.0) if kind is None else (float(c[0]), float(c[1]))

    def physical(hname, floating):
        h = fr[hname].get()
        nm = [str(n) for n in h.axes[0]]
        x = np.asarray(h.values(), float)[[nm.index(n) for n in floating]]
        return {n: lin[n][0] + lin[n][1] * v for n, v in zip(floating, x)}

    nm = [str(n) for n in fr["parms"].get().axes[0]]
    floating = [n for n in inp["names"] if n in nm]
    held = {n: float(inp["anchors"][n]) for n in inp["names"] if n not in floating}
    post = dict(held, **physical("parms", floating))
    pre = dict(held, **physical("parms_prefit", floating))
    forms = (inp["np_model"], inp["np_model_nu"])
    faces = {}
    for c in ndw.damping_conditions(*forms, inp["ymax"]):
        if set(c.names) & set(floating):
            v = float(c.value(post, lambda z: np.maximum(0.0, z) ** 2) - c.bound)
            if v < FACE_TOL:
                faces[c.label] = v
    print(
        f"[np_function_plots] {path}: "
        + ", ".join(f"{n}={post[n]:+.5g}" for n in floating)
        + f"; held {held}"
        + (f"; ON ACTIVE FACE(S): {faces} (band pinched there)" if faces else "")
    )
    toys = None
    if "cov" in fr and n_toys > 0:
        idx = [nm.index(n) for n in floating]
        scale = np.array([lin[n][1] for n in floating])
        cov = np.asarray(fr["cov"].get().values(), float)[np.ix_(idx, idx)]
        cov = cov * np.outer(scale, scale)
        # eigh, never Cholesky: a face-held covariance is singular along the face
        draws = np.random.default_rng(seed).multivariate_normal(
            [post[n] for n in floating], cov, size=n_toys, method="eigh"
        )
        toys = [Tune(*forms, dict(post, **dict(zip(floating, d)))) for d in draws]
    return Tune(*forms, pre), Tune(*forms, post), toys, faces


# ---------------------------------------------------------------------------
# Plotting (from scetlib_np/np_function_plots.py)
# ---------------------------------------------------------------------------


@dataclass
class Series:
    """A curve to draw: a λ set, styling, and optional toys for an error band."""

    label: str
    lam: Tune
    color: Optional[str] = None
    linestyle: str = "-"
    lw: float = 2.0
    toys: Optional[List[Tune]] = None
    band_pct: Tuple[float, float] = (16.0, 84.0)
    cmap: Optional[str] = None


LATTICE_CS_MU = np.array([1.6853, 0.087, 0.0074])


LATTICE_CS_SIGMA = np.array([0.5069, 0.0332, 0.0066])


LATTICE_CS_CORR = np.array(
    [[1.0, 0.5212, -0.7249], [0.5212, 1.0, -0.9135], [-0.7249, -0.9135, 1.0]]
)


LATTICE_CS_COV = np.outer(LATTICE_CS_SIGMA, LATTICE_CS_SIGMA) * LATTICE_CS_CORR


def lattice_gamma_nu(bT, lam_inf, lambda2_nu, lambda4_nu):
    """γ̃_ν^NP(b_T) in the paper's own tanh_2 form, Eq. (3.32)."""
    bT = np.asarray(bT, dtype=float)
    return -lam_inf * np.tanh((lambda2_nu * bT**2 + lambda4_nu * bT**4) / lam_inf)


def lattice_cs_band(bT, nsig=1.0, n_samples=20000, seed=0):
    """Envelope of Eq. (3.32) over the ``nsig`` covariance ellipsoid of Eq. (3.34)."""
    chol = np.linalg.cholesky(LATTICE_CS_COV)
    rng = np.random.default_rng(seed)
    u = rng.normal(size=(3, n_samples))
    u /= np.linalg.norm(u, axis=0)
    pars = LATTICE_CS_MU[:, None] + nsig * (chol @ u)
    pars = pars[:, pars[0] > 0]
    curves = lattice_gamma_nu(
        np.asarray(bT)[None, :], pars[0][:, None], pars[1][:, None], pars[2][:, None]
    )
    return (curves.min(axis=0), curves.max(axis=0))


def draw_lattice_cs_reference(ax, bT, nsig=1.0, band=True):
    """Overlay the lattice-fit CS kernel on a γ̃_ν panel (drawn behind the series)."""
    if band:
        lo, hi = lattice_cs_band(bT, nsig=nsig)
        ax.fill_between(bT, lo, hi, color="0.55", alpha=0.28, lw=0, zorder=0)
    ax.plot(
        bT,
        lattice_gamma_nu(bT, *LATTICE_CS_MU),
        color="0.15",
        lw=1.8,
        ls=(0, (5, 2)),
        zorder=1,
        label="lattice fit (2506.13874)",
    )


MAP22_N3LL_PARS = (
    0.2482420850967863,
    0.3156048435703042,
    1.1349706417780698,
    0.5154187912648195,
    1.8893385924128125,
    0.004392854889525612,
    10.843398922979528,
    0.007035551324624491,
    1.5397221152343268,
    0.061287263131035774,
    0.21739975134899683,
    0.12887993135476603,
    0.016302544233918744,
    0.01960274771272947,
    4.26533498117754,
    4.323926466248531,
    0.417958063109992,
    12.673323632665454,
)


MAP22_XHAT = 0.1


MAP22_Q_DEFAULT = 91.1876


MAP22_SQRTS_DEFAULT = 13000.0


def map22_f_int(x, bT, pars=MAP22_N3LL_PARS):
    """MAP22 intrinsic (evolution-stripped) NP function for ONE hadron."""
    x = np.asarray(x, dtype=float)
    bT = np.asarray(bT, dtype=float)
    N1, al1, sg1, lam = (pars[1], pars[2], pars[3], pars[4])
    N1B, N1C, lam2 = (pars[11], pars[12], pars[13])
    al2, al3, sg2, sg3 = (pars[14], pars[15], pars[16], pars[17])

    def g(N, sg, al):
        return N * (x / MAP22_XHAT) ** sg * ((1 - x) / (1 - MAP22_XHAT)) ** al**2

    g1, g1B, g1C = (g(N1, sg1, al1), g(N1B, sg2, al2), g(N1C, sg3, al3))
    q = (bT / 2.0) ** 2
    num = (
        g1 * np.exp(-g1 * q)
        + lam**2 * g1B**2 * (1 - g1B * q) * np.exp(-g1B * q)
        + lam2**2 * g1C * np.exp(-g1C * q)
    )
    return num / (g1 + lam**2 * g1B**2 + lam2**2 * g1C)


def map22_F_eff(
    bT, Y, Q=MAP22_Q_DEFAULT, sqrt_s=MAP22_SQRTS_DEFAULT, pars=MAP22_N3LL_PARS
):
    """MAP22 analogue of F_eff(b_T, Y): the two-beam product at x = (Q/√s)e^{±Y}."""
    x1 = Q / sqrt_s * np.exp(+abs(Y))
    x2 = Q / sqrt_s * np.exp(-abs(Y))
    return map22_f_int(x1, bT, pars) * map22_f_int(x2, bT, pars)


def draw_map22_tmd_reference(ax, bT, y_values, colors, Q=None, sqrt_s=None):
    """Overlay the MAP22 F_eff analogue on a TMD panel, one curve per y."""
    Q = MAP22_Q_DEFAULT if Q is None else Q
    sqrt_s = MAP22_SQRTS_DEFAULT if sqrt_s is None else sqrt_s
    for yi, y in enumerate(y_values):
        ax.plot(
            bT,
            map22_F_eff(bT, y, Q, sqrt_s),
            color=colors[yi],
            lw=1.8,
            ls=(0, (1, 1.3)),
            zorder=1,
            label=f"MAP22, y={y:g}",
        )


def _band(curves, pct):
    """(lo, hi) percentile envelope across a stack of curves, or None if empty."""
    if not len(curves):
        return None
    stack = np.asarray(curves)
    return (np.percentile(stack, pct[0], axis=0), np.percentile(stack, pct[1], axis=0))


_CORNERS = {
    "upper right": (0.97, 0.97, "top", "right"),
    "upper left": (0.03, 0.97, "top", "left"),
    "lower left": (0.03, 0.03, "bottom", "left"),
    "lower right": (0.97, 0.03, "bottom", "right"),
}


_EFF_LABELS = {
    "lambda2": "\\lambda_2",
    "lambda4": "\\lambda_4",
    "delta_lambda2": "\\delta\\lambda_2",
    "lambda6": "\\lambda_6",
    "lambda_inf": "\\lambda_\\infty",
}


_GNU_LABELS = {
    "lambda2_nu": "\\lambda_2^\\nu",
    "lambda4_nu": "\\lambda_4^\\nu",
    "lambda6_nu": "\\lambda_6^\\nu",
    "lambda_inf_nu": "\\lambda_\\infty^\\nu",
}


def _sector_view(lam, sector):
    """``(model, active λ names, λ dict, print order)`` for one panel's sector."""
    if sector == "gnu":
        return (
            lam.np_model_nu,
            set(ndw._CS_LAMBDAS[lam.np_model_nu]),
            lam.gnu,
            GNU_ORDER,
        )
    return lam.np_model, set(ndw._TMD_LAMBDAS[lam.np_model]), lam.eff, EFF_ORDER


def _param_inset(ax, lam, sector, corner="upper right"):
    """Small text box listing only the λ that drive the panel for its model."""
    model, active, src, order = _sector_view(lam, sector)
    labels = _GNU_LABELS if sector == "gnu" else _EFF_LABELS
    lines = [f"${labels[k]} = {src.get(k, 0):+.4f}$" for k in order if k in active]
    lines.append(f"model: {model}")
    box = dict(boxstyle="round,pad=0.35", fc="white", ec="0.6", alpha=0.85)
    x, y, va, ha = _CORNERS[corner]
    ax.text(
        x,
        y,
        "\n".join(lines),
        transform=ax.transAxes,
        fontsize=8,
        va=va,
        ha=ha,
        bbox=box,
    )


def _inset_table_text(lams, titles, sector, title_width=26):
    """One λ per row, one λ SET per column: the two-tune version of the inset."""
    views = [_sector_view(lam, sector) for lam in lams]
    rows = [p for p in views[0][3] if any((p in v[1] for v in views))]
    heads = [
        t if len(t) <= title_width else "…" + t[-(title_width - 1) :]  # noqa: E203
        for t in titles
    ]
    w = max([len(p) for p in rows] + [len("model")])
    cw = max([len(h) for h in heads] + [8])
    out = [" " * w + "  " + "  ".join((f"{h:>{cw}}" for h in heads))]
    for p in rows:
        cells = [f"{v[2][p]:+.4f}" if p in v[1] else "--" for v in views]
        out.append(f"{p:<{w}}  " + "  ".join((f"{c:>{cw}}" for c in cells)))
    out.append(f"{'model':<{w}}  " + "  ".join((f"{v[0]:>{cw}}" for v in views)))
    return "\n".join(out)


def _param_inset_compare(ax, lams, titles, sector, corner="upper right"):
    """Draw the multi-column λ table (:func:`_inset_table_text`) in a corner."""
    x, y, va, ha = _CORNERS[corner]
    ax.text(
        x,
        y,
        _inset_table_text(lams, titles, sector),
        transform=ax.transAxes,
        fontsize=6.5,
        family="monospace",
        va=va,
        ha=ha,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.6", alpha=0.88),
    )


def _fixed_color(c):
    """Resolve a ``"CN"`` property-cycle reference to a concrete colour NOW."""
    import matplotlib.colors as mcolors

    return mcolors.to_hex(mcolors.to_rgba(c)) if isinstance(c, str) else c


NUM_CMAP = "viridis"


DEN_CMAP = "plasma"


CMAP_SHADE_RANGE = (0.25, 0.85)


REFERENCE_GREY = "0.35"


def _cmap_shades(name, n):
    """``n`` colours across one colormap, over :data:`CMAP_SHADE_RANGE`."""
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap(name)
    lo, hi = CMAP_SHADE_RANGE
    if n <= 1:
        return [_to_hex(cmap(lo)[:3])]
    return [_to_hex(cmap(t)[:3]) for t in np.linspace(lo, hi, n)]


def _y_shades(base, n):
    """``n`` shades of ONE hue, dark (first y) -> light (last y)."""
    rgb = np.asarray(_to_rgb(base), dtype=float)
    if n <= 1:
        return [_to_hex(rgb)]
    dark = rgb * 0.55
    light = 1.0 - (1.0 - rgb) * 0.55
    return [_to_hex(dark + (light - dark) * t) for t in np.linspace(0.0, 1.0, n)]


def _series_colors(s, n_y, fallback):
    """``(CS-panel colour, [per-y TMD colours])`` for one series."""
    if s.cmap:
        shades = _cmap_shades(s.cmap, n_y)
        return (_fixed_color(s.color) if s.color else shades[0], shades)
    base = _fixed_color(s.color or fallback)
    return (base, _y_shades(base, n_y))


def _to_rgb(c):
    import matplotlib.colors as mcolors

    return mcolors.to_rgb(c)


def _to_hex(rgb):
    import matplotlib.colors as mcolors

    return mcolors.to_hex(np.clip(rgb, 0.0, 1.0))


def _legend(ax, loc):
    """Panel legend, wrapped to two columns once a two-tune overlay fills it."""
    n = len(ax.get_legend_handles_labels()[1])
    ax.legend(
        loc=loc,
        fontsize=8 if n <= 8 else 6.5,
        ncol=1 if n <= 8 else 2,
        handlelength=3.6,
    )


def plot_np_functions(
    series: Sequence[Series],
    *,
    y_values: Sequence[float] = (0.0, 2.5, 5.0),
    bT_max: float = 4.0,
    n_points: int = 401,
    outpath: str,
    inset_from: Optional[Series] = None,
    insets: Optional[Sequence[Tuple[str, Series]]] = None,
    f_ymax: Optional[float] = None,
    lattice_reference: bool = True,
    lattice_nsig: float = 1.0,
    map22_reference: bool = True,
    map22_Q: Optional[float] = None,
    map22_sqrts: Optional[float] = None,
    args=None,
):
    """Draw the two NP form factors for one or more λ sets."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bT = np.linspace(0.0, bT_max, n_points)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 4.5))
    auto_colors = [c for c in plt.rcParams["axes.prop_cycle"].by_key()["color"]]
    line_fmax = 0.0
    if lattice_reference:
        draw_lattice_cs_reference(axL, bT, nsig=lattice_nsig)
    n_y = len(y_values)
    for si, s in enumerate(series):
        color, shades = _series_colors(s, n_y, auto_colors[si % len(auto_colors)])
        axL.plot(
            bT,
            gamma_nu_curve(bT, s.lam.gnu),
            label=s.label,
            color=color,
            lw=s.lw,
            ls=s.linestyle,
        )
        if s.toys:
            band = _band([gamma_nu_curve(bT, t.gnu) for t in s.toys], s.band_pct)
            if band is not None:
                axL.fill_between(
                    bT,
                    band[0],
                    band[1],
                    color=color,
                    alpha=0.22,
                    label=f"{s.label} {int(s.band_pct[1] - s.band_pct[0])}% band",
                )
        for yi, y in enumerate(y_values):
            yc = shades[yi]
            line = f_eff_curve(bT, y, s.lam.eff)
            line_fmax = max(line_fmax, float(np.nanmax(line)))
            axR.plot(
                bT, line, color=yc, lw=s.lw, ls=s.linestyle, label=f"{s.label}, y={y:g}"
            )
            if s.toys:
                band = _band([f_eff_curve(bT, y, t.eff) for t in s.toys], s.band_pct)
                if band is not None:
                    axR.fill_between(bT, band[0], band[1], color=yc, alpha=0.18)
    if map22_reference:
        y_colors = _y_shades(REFERENCE_GREY, len(y_values))
        draw_map22_tmd_reference(axR, bT, y_values, y_colors, map22_Q, map22_sqrts)
    axL.axhline(0, color="k", lw=0.5)
    axL.set_xlabel("$b_T$ [GeV$^{-1}$]")
    axL.set_ylabel("$\\tilde\\gamma_\\nu^{\\rm NP}(b_T)$")
    axL.set_title(
        "CS rapidity anomalous dimension $\\tilde\\gamma_\\nu^{\\rm NP}(b_T)$",
        fontsize=11,
    )
    _legend(axL, loc="lower left")
    axL.grid(alpha=0.3)
    if args.ylim_cs is not None:
        axL.set_ylim(args.ylim_cs[0], args.ylim_cs[1])
    axR.set_xlabel("$b_T$ [GeV$^{-1}$]")
    axR.set_ylabel("$F_{\\rm eff}(b_T, y)$")
    axR.set_title("TMD-effective NP factor $F_{\\rm eff}(b_T, y)$", fontsize=11)
    _legend(axR, loc="upper right")
    axR.grid(alpha=0.3)
    if args.ylim_tmd is not None:
        axR.set_ylim(args.ylim_tmd[0], args.ylim_tmd[1])
    else:
        top = f_ymax if f_ymax is not None else max(1.1, 1.2 * line_fmax)
        axR.set_ylim(0.0, top)
    if insets:
        titles = [t for (t, _) in insets]
        lams = [s.lam for (_, s) in insets]
        _param_inset_compare(axL, lams, titles, "gnu", corner="upper right")
        _param_inset_compare(axR, lams, titles, "eff", corner="lower left")
    else:
        inset = inset_from if inset_from is not None else series[-1] if series else None
        if inset is not None:
            _param_inset(axL, inset.lam, "gnu", corner="upper right")
            _param_inset(axR, inset.lam, "eff", corner="lower left")
    fig.tight_layout()
    outdir, basename = _split_outpath(outpath, default_name="np_functions.png")
    _save_plot(outdir, basename, fig=fig, args=args, dpi=140)
    plt.close(fig)
    print(f"Wrote {outdir}/{basename}.png(.pdf) + {basename}.log")


def _save_plot(outdir, basename, fig=None, args=None, meta_info=None, dpi=None):
    """Write ``fig`` to ``{outdir}/{basename}.{pdf,png}`` + a provenance log/index."""
    from wums import output_tools, plot_tools

    outdir = outdir or "."
    os.makedirs(outdir, exist_ok=True)
    if dpi is not None and fig is not None:
        fig.set_dpi(dpi)
    plot_tools.save_pdf_and_png(outdir, basename, fig=fig)
    output_tools.write_index_and_log(
        outdir, basename, analysis_meta_info=meta_info or {}, args=args
    )


def _split_outpath(out_path, default_name=None):
    """Split a user ``--out``-style path into ``(outdir, basename)``."""
    if default_name is not None and (
        out_path.endswith(("/", os.sep))
        or os.path.isdir(out_path)
        or (not os.path.splitext(out_path)[1])
    ):
        out_path = os.path.join(out_path, default_name)
    outdir = os.path.dirname(out_path) or "."
    basename = os.path.splitext(os.path.basename(out_path))[0]
    return (outdir, basename)


def _run_label(fitresult_path):
    """A fit's run name: the directory the fitresults sits in."""
    return os.path.basename(os.path.dirname(os.path.abspath(fitresult_path)))


def _inset_title(side, label, width=18):
    """``"num: <label>"`` for the λ-table header, keeping the side marker."""
    short = label if len(label) <= width else "…" + label[-(width - 1) :]  # noqa
    return f"{side}: " + short


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def make_parser():
    p = argparse.ArgumentParser(
        description="Plot the NP form factors γ_ν^NP(b_T) and F_eff(b_T, y) of one "
        "or two scetlib_ad fits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    for side, word in (("num", "numerator"), ("den", "denominator")):
        p.add_argument(
            f"--{side}-fitresult",
            required=side == "num",
            help=f"scetlib_ad fitresults (or its run directory) of the {word} tune",
        )
        p.add_argument(f"--{side}-label", help=f"legend label of the {word}")
        p.add_argument(f"--{side}-result", help="result group suffix")
    p.add_argument("--card", help="card for the θ -> λ map (default: the fit's)")
    p.add_argument("--n-toys", type=int, default=500, help="band toys per fit")
    p.add_argument("--seed", type=int, default=0, help="band RNG seed")
    p.add_argument("--y", type=float, nargs="+", default=[0.0, 2.5, 5.0])
    p.add_argument("--bT-max", type=float, default=4.0)
    p.add_argument("--f-ymax", type=float, default=None)
    p.add_argument("--ylim-cs", type=float, nargs=2, default=None)
    p.add_argument("--ylim-tmd", type=float, nargs=2, default=None)
    p.add_argument("--no-map22-reference", dest="map22_reference", action="store_false")
    p.add_argument("--map22-Q", type=float, default=None)
    p.add_argument("--map22-sqrts", type=float, default=None)
    p.add_argument(
        "--no-lattice-reference", dest="lattice_reference", action="store_false"
    )
    p.add_argument("--lattice-nsig", type=float, default=1.0)
    p.add_argument("--no-prefit", dest="prefit", action="store_false")
    p.add_argument("--outpath", "-o", required=True)
    return p


def _side(args, side):
    pre, post, toys, _ = read_ad_fitresult(
        getattr(args, f"{side}_fitresult"),
        card=args.card,
        result=getattr(args, f"{side}_result"),
        n_toys=args.n_toys,
        seed=args.seed,
    )
    cmap = NUM_CMAP if side == "num" else DEN_CMAP
    print(f"[np_function_plots] {side}: {len(toys or [])}-toy 68% band")
    return [
        Series("prefit (λ_central)", pre, None, "--", 1.8, cmap=cmap),
        Series("postfit", post, None, "-", 2.0, toys=toys, cmap=cmap),
    ]


def main(argv=None):
    args = make_parser().parse_args(argv)
    num = _side(args, "num")
    insets = None
    if args.num_label:
        num[-1].label = args.num_label
    series = num if args.prefit else num[-1:]
    if args.den_fitresult:
        den = _side(args, "den")
        num[-1].label = args.num_label or _run_label(args.num_fitresult)
        den[-1].label = args.den_label or _run_label(args.den_fitresult)
        pre_den = den[:-1]
        if num[0].lam == den[0].lam:  # one prefit curve covers both sides
            pre_den = []
            num[0].label = "prefit (λ_central, both sides)"
        else:
            num[0].label = f"prefit ({num[-1].label})"
            den[0].label = f"prefit ({den[-1].label})"
        series = [*num, *pre_den, den[-1]]
        if not args.prefit:
            series = [num[-1], den[-1]]
        insets = [
            (_inset_title("num", num[-1].label), num[-1]),
            (_inset_title("den", den[-1].label), den[-1]),
        ]
    plot_np_functions(
        series,
        lattice_reference=args.lattice_reference,
        lattice_nsig=args.lattice_nsig,
        map22_reference=args.map22_reference,
        map22_Q=args.map22_Q,
        map22_sqrts=args.map22_sqrts,
        y_values=args.y,
        bT_max=args.bT_max,
        outpath=args.outpath,
        insets=insets,
        f_ymax=args.f_ymax,
        args=args,
    )


if __name__ == "__main__":
    main()
