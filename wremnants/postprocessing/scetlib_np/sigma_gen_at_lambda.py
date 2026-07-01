"""Evaluate the SCETlib NP matched σ_gen at a given λ tune, via the core.

Builds the datacard-free core ``SigmaGenModel`` and runs its matched gen-level
prediction σ_gen(λ; g) = σ_resum(λ; g) + σ_ns on the (ptVGen, absYVGen) gen grid
— Steps 1–2 of ``param_model.py`` (the object that folds through R into the fit,
BEFORE the gen→reco fold and ratio). Prints σ_gen on the gen grid and can plot a
1-D projection (e.g. the ptZ = ptVGen distribution).

The λ to evaluate at = a physical BASE tune + overrides:
  * base: ``--meta-from HDF5`` / the ``--theory-corr`` file's Nonperturbative
    runcard / else the canonical FranksVals tanh_2 default. The model is BUILT at
    this base (positive σ_gen, which the constructor requires); λ are evaluated on
    top, so params not set stay at the BASE value, not 0;
  * ``--fitresult HDF5`` postfit λ (optional);
  * ``--lambdas name=val,...`` explicit values (optional; win over the rest).
Common use is ``--lambdas lambda2=0.5``, the rest staying at FranksVals; no
λ_central source needed. Evaluating, unlike constructing, has no positivity
guard, so a weak-NP tune can be inspected; non-positive bins are warned, not
rejected.

Can ALSO overlay the gen distribution in an official ``TheoryCorrection`` hist
(``--theory-corr``). Those ``.pkl.lz4`` files carry the ``{generator}_hist``
object — the official SCETlib+DYTurbo prediction on a (Q, absY, qT, charge, vars)
grid — so its central (``pdf0``) entry is the same physical object the param-model
σ_gen reconstructs. Overlaying is a direct end-to-end check that the bt-grid +
on-the-fly reconstruction reproduces the official run; pick any ``vars`` label,
e.g. ``lambda21.0``, to overlay a λ-shifted official run against the model at the
matching λ.

The bT-grid is required (``--btgrid``). The gen-bin edges (ptVGen, absYVGen) are
chosen per axis, in order: explicit ``--ptv-edges`` / ``--absy-edges``, then a
``--gen-edges-from`` / ``--datacard`` hdf5, then a built-in default (1-GeV ptVGen
bins over [0, 40]; a single rapidity-inclusive absYVGen bin [0, 5]) — so the
script runs with no gen-edge input. ``--datacard`` feeds ONLY the gen edges; λ
are not sourced from it (use ``--meta-from``).

Run inside a container that binds the inputs (same as the validation scripts):

    export APPTAINER_BIND="/scratch,/cvmfs,/work,/ceph,/home"
    singularity run --cleanenv <wmassdevrolling> bash -c \\
      "source main/WRemnants/setup.sh; \\
       python3 -m wremnants.postprocessing.scetlib_np.sigma_gen_at_lambda \\
         --lambdas lambda2=0.4,lambda4=0.1,lambda2_nu=0.15 \\
         --theory-corr <wremnants-data>/data/TheoryCorrections/scetlib_dyturbo_..._CorrZ.pkl.lz4 \\
         --plot ~/public_html/alphaS/YYMMDD_sigmagen/ptZ.png"
"""

import argparse
import os
import sys
import time

import numpy as np

from wremnants.postprocessing.scetlib_np.params import (
    EFF_PARAMS,
    GNU_PARAMS,
    parse_lambda_overrides,
)

# btgrid default mirrors the validation scripts; the datacard is intentionally
# NOT defaulted — pass --datacard (or explicit --*-edges) for the gen edges.
BTGRID_DIR = "/scratch/submit/cms/wmass/scetlib_np/Z_COM13_CT18Z_N3p0LL_btgrid_fineall/"
Q_LO, Q_HI = 60.0, 120.0
# Canonical FranksVals (CT18Z N3+0LL lattice λ4-bugfix) tanh_2 runcard — the
# production λ_central. Construction BASE when no base λ is sourced: the model
# must be built at a PHYSICAL tune (positive σ_gen, so the constructor's response
# guard passes), and the requested λ evaluated on top. Source of truth: a
# correction file's Nonperturbative section (file_meta_data → config →
# Nonperturbative); the LatticeNPLambda4Bugfix_FranksVals_CT18Z values.
CANONICAL_BASE = {
    "eff_params": {
        "np_model": "tanh_2", "lambda2": 0.4, "lambda4": 0.4,
        "lambda6": 0.0, "delta_lambda2": 0.0, "lambda_inf": 1.0,
    },
    "gnu_params": {
        "np_model_nu": "tanh_2", "lambda2_nu": 0.15,
        "lambda4_nu": 0.0, "lambda6_nu": 0.0, "lambda_inf_nu": 2.0,
    },
}

# Built-in gen grid used when neither explicit edges nor a datacard/hdf5 source is
# given: 1-GeV ptVGen bins over [0, 40], rapidity-inclusive in a single absYVGen
# bin [0, 5] (5.0 is a TheoryCorrection absY edge, so the overlay still aligns).
DEFAULT_PTV_EDGES = np.arange(0.0, 41.0, 1.0)
DEFAULT_ABSY_EDGES = np.array([0.0, 5.0])

# corr-hist axis ↔ model gen axis (the TheoryCorrection _hist uses SCETlib names).
_CORR_AXIS = {"ptVGen": "qT", "absYVGen": "absY"}


def _parse_edges(s):
    """``a,b,c,...`` -> float ndarray of bin edges."""
    return np.array([float(x) for x in s.split(",") if x.strip()], dtype=np.float64)


def _merge_matrix(fine_edges, coarse_edges, name="axis", tol=1e-6):
    """(N_coarse, N_fine) 0/1 matrix summing fine bins into coarse bins.

    Requires every coarse edge to coincide with a fine edge (coarse is a
    sub-binning of fine): exact merge, no interpolation. Fine bins whose centre
    lies outside every coarse bin (e.g. qT beyond the model's ptVGen overflow
    edge) get weight 0 and are dropped, matching the model.
    """
    fine_edges = np.asarray(fine_edges, dtype=np.float64)
    coarse_edges = np.asarray(coarse_edges, dtype=np.float64)
    for e in coarse_edges:
        if not np.any(np.isclose(fine_edges, e, atol=tol)):
            raise SystemExit(
                f"_merge_matrix[{name}]: model edge {e} is not a TheoryCorrection "
                f"bin edge (its binning is not a refinement of the model grid on "
                f"this axis). corr edges: {fine_edges}"
            )
    centers = 0.5 * (fine_edges[:-1] + fine_edges[1:])
    W = np.zeros((coarse_edges.size - 1, fine_edges.size - 1), dtype=np.float64)
    for i in range(coarse_edges.size - 1):
        m = (centers >= coarse_edges[i]) & (centers <= coarse_edges[i + 1])
        W[i, m] = 1.0
    return W


def resolve_base_lambda(args):
    """Physical BASE λ tune (eff_params/gnu_params) the model is CONSTRUCTED at.

    Priority: ``--meta-from HDF5`` > the ``--theory-corr`` file's embedded
    Nonperturbative runcard > the canonical FranksVals tanh_2 default. Always a
    complete physical tune (never None), so construction lands on a positive-σ_gen
    point (the constructor's response guard); the requested λ are evaluated on top.
    """
    from wremnants.postprocessing.scetlib_np import lambda_central as lc

    if args.meta_from:
        print(f"[λ base] from hdf5 metadata {args.meta_from}")
        return lc.read_lambda_central(args.meta_from)
    if args.theory_corr:
        import pickle

        import lz4.frame

        with lz4.frame.open(args.theory_corr) as fh:
            corr = pickle.load(fh)
        base = lc.extract_lambda_central(
            corr, tag=os.path.basename(args.theory_corr),
            proc=args.theory_corr_proc or "Z",
        )
        print(f"[λ base] from the --theory-corr Nonperturbative runcard "
              f"({base.get('basename')})")
        return {"eff_params": base["eff_params"], "gnu_params": base["gnu_params"]}
    print("[λ base] none given -> canonical FranksVals tanh_2 default")
    return CANONICAL_BASE


def assemble_tune(base, overrides):
    """Full (eff_params, gnu_params) for the EVAL point = base tune + overrides,
    plus the explicitly-set names.

    ``base`` is a physical lambda_central dict (with the np_model form strings);
    params not in ``overrides`` stay at the base value (NOT 0). Each override is
    routed to eff or gnu by membership."""
    eff = dict(base["eff_params"])
    gnu = dict(base["gnu_params"])
    explicit = {}
    for name, val in overrides.items():
        if name in EFF_PARAMS:
            eff[name] = val
        elif name in GNU_PARAMS:
            gnu[name] = val
        else:
            raise SystemExit(
                f"unknown λ {name!r}; valid: {list(GNU_PARAMS) + list(EFF_PARAMS)}"
            )
        explicit[name] = val
    return eff, gnu, explicit


def resolve_gen_axes(args):
    """gen_axes = [(ptVGen, edges), (absYVGen, edges)], chosen per axis in order:
    explicit --ptv-edges/--absy-edges, then a --gen-edges-from/--datacard hdf5,
    then the built-in defaults (1-GeV ptVGen [0,40]; single absYVGen [0,5])."""
    ptv = _parse_edges(args.ptv_edges) if args.ptv_edges else None
    absy = _parse_edges(args.absy_edges) if args.absy_edges else None
    src = args.gen_edges_from or args.datacard

    src_axes = None
    if (ptv is None or absy is None) and src:
        print(f"[gen-axes] reading the scetlib_np auxiliary of {src}")
        from rabbit.inputdata import FitInputData

        from wremnants.postprocessing.scetlib_np.param_model import (
            _R_info_from_auxiliary,
        )

        indata = FitInputData(src)
        src_axes = {
            n: np.asarray(e, dtype=np.float64)
            for n, e in _R_info_from_auxiliary(indata)["gen_axes"]
        }

    def pick(name, explicit, default):
        if explicit is not None:
            print(f"[gen-axes] {name}: explicit ({explicit.size - 1} bins)")
            return explicit
        if src_axes is not None and name in src_axes:
            print(f"[gen-axes] {name}: from {src} ({src_axes[name].size - 1} bins)")
            return src_axes[name]
        print(f"[gen-axes] {name}: built-in default ({default.size - 1} bins, "
              f"[{default[0]:g}, {default[-1]:g}])")
        return default

    return [
        ("ptVGen", pick("ptVGen", ptv, DEFAULT_PTV_EDGES)),
        ("absYVGen", pick("absYVGen", absy, DEFAULT_ABSY_EDGES)),
    ]


def load_theory_corr_hist(path, proc=None):
    """Load the ``{generator}_hist`` (SCETlib+DYTurbo) Hist from a TheoryCorrection
    ``.pkl.lz4``.

    The file maps ``corr[proc][histname]``. ``proc`` defaults to the single
    physics key (``meta_data`` / ``file_meta_data`` excluded); the hist is the
    lone ``*_hist`` that is not ``minnlo_ref_hist`` (the prediction, not the
    MiNNLO reference or the ratio).
    """
    import pickle

    import lz4.frame

    with lz4.frame.open(path) as fh:
        corr = pickle.load(fh)

    meta_keys = {"meta_data", "file_meta_data"}
    procs = [k for k in corr.keys() if k not in meta_keys]
    if proc is None:
        if len(procs) != 1:
            raise SystemExit(
                f"--theory-corr-proc needed: {os.path.basename(path)} has procs {procs}"
            )
        proc = procs[0]
    elif proc not in corr:
        raise SystemExit(f"proc {proc!r} not in {list(corr.keys())}")

    entry = corr[proc]
    cands = [k for k in entry if k.endswith("_hist") and k != "minnlo_ref_hist"]
    if len(cands) != 1:
        raise SystemExit(
            f"expected one {{generator}}_hist in {proc}; found {list(entry.keys())}"
        )
    histname = cands[0]

    print(f"[theory-corr] {os.path.basename(path)} :: {proc} / {histname}")
    return entry[histname]


def theory_corr_projection(h, gen_axes, plot_axis, var="pdf0", q_window=(Q_LO, Q_HI),
                           tol=1e-6):
    """Project a TheoryCorrection ``_hist`` onto the model's ``plot_axis`` gen bins.

    Reduces the (Q, absY, qT, charge, vars) Hist to a 1-D bin-integrated σ on the
    model's ``plot_axis`` edges, restricted to the model's gen-grid extent on the
    OTHER axis so it covers the same phase space the model σ_gen projection does:

      1. select the ``vars`` entry (default ``pdf0`` = central tune);
      2. sum the Q bins whose centre falls in ``q_window`` (in-range only);
      3. sum the charge axis (in-range), if present;
      4. sum the OTHER gen axis over the model's extent [0, other_max];
      5. rebin the projection axis onto the model's ``plot_axis`` edges (model
         edges must be a sub-binning of the corr hist's: qT is fine enough that
         ptVGen always aligns; absY uses SCETlib's binning so absYVGen may not).

    Returns an ndarray of length ``len(plot_axis edges) - 1`` (bin-integrated σ).
    """
    names = [n for n, _ in gen_axes]
    if plot_axis not in names or plot_axis not in _CORR_AXIS:
        raise SystemExit(f"--plot-axis {plot_axis!r} not a model gen axis {names}")
    edges_by_name = {n: np.asarray(e, dtype=np.float64) for n, e in gen_axes}
    other_model = names[1] if plot_axis == names[0] else names[0]
    proj_corr = _CORR_AXIS[plot_axis]
    other_corr = _CORR_AXIS[other_model]

    have = [a.name for a in h.axes]
    for need in ("Q", proj_corr, other_corr, "vars"):
        if need not in have:
            raise SystemExit(
                f"theory-corr hist missing {need!r} axis; has {have}"
            )

    # 1. vars selection.
    vlist = list(h.axes["vars"])
    if var not in vlist:
        raise SystemExit(
            f"--theory-corr-var {var!r} not in corr hist vars; have {vlist}"
        )
    h = h[{"vars": vlist.index(var)}]

    # 2. Q window (sum in-range bins whose centre is inside the window).
    qe = np.asarray(h.axes["Q"].edges, dtype=np.float64)
    qc = 0.5 * (qe[:-1] + qe[1:])
    qsel = np.where((qc >= q_window[0] - tol) & (qc <= q_window[1] + tol))[0]
    if not qsel.size:
        raise SystemExit(f"no Q bins in window {q_window}; corr Q edges {qe}")
    h = h[{"Q": slice(int(qsel[0]), int(qsel[-1]) + 1, sum)}]

    # 3. charge sum (in-range), if a charge axis is present.
    if "charge" in [a.name for a in h.axes]:
        h = h[{"charge": slice(0, h.axes["charge"].size, sum)}]

    # 4. sum the OTHER axis over the model's extent [0, other_max]. Non-coinciding
    #    upper edge (SCETlib's absY binning): cut at the nearest corr edge, warn.
    other_max = edges_by_name[other_model][-1]
    oe = np.asarray(h.axes[other_corr].edges, dtype=np.float64)
    oc = 0.5 * (oe[:-1] + oe[1:])
    osel = np.where(oc <= other_max + tol)[0]
    if not osel.size:
        raise SystemExit(
            f"theory-corr {other_corr} has no bins below the model {other_model} "
            f"max {other_max}; corr edges {oe}"
        )
    cut_idx = int(osel[-1]) + 1
    actual_edge = oe[cut_idx]
    if abs(actual_edge - other_max) > tol:
        print(
            f"[theory-corr] WARNING: model {other_model} max {other_max} does not "
            f"coincide with a corr {other_corr} edge; summing corr up to "
            f"{actual_edge} ({abs(actual_edge - other_max):.3g} off)."
        )
    h = h[{other_corr: slice(0, cut_idx, sum)}]

    # 5. rebin the projection axis onto the model's plot_axis edges.
    W = _merge_matrix(
        np.asarray(h.axes[proj_corr].edges, dtype=np.float64),
        edges_by_name[plot_axis],
        name=plot_axis,
        tol=tol,
    )
    return W @ np.asarray(h.values(flow=False), dtype=np.float64)


def _lambda_box_text(eff, gnu):
    """Compact multi-line λ-tune annotation: np_model form(s) + non-zero params."""
    lines = [f"np_model = {eff.get('np_model')}"]
    if gnu.get("np_model_nu") != eff.get("np_model"):
        lines.append(f"np_model_nu = {gnu.get('np_model_nu')}")
    for p in GNU_PARAMS:
        if abs(gnu.get(p, 0.0)) > 0:
            lines.append(f"{p} = {gnu[p]:.4g}")
    for p in EFF_PARAMS:
        if abs(eff.get(p, 0.0)) > 0:
            lines.append(f"{p} = {eff[p]:.4g}")
    return "λ tune:\n" + "\n".join(lines)


def make_projection_plot(sigma_gen, gen_axes, axis, out_path, eff, gnu,
                         s_corr=None, corr_label=None):
    """Step histogram of the matched σ_gen(λ) projection onto one gen axis
    (default ptVGen = ptZ), summing over the other.

    With a TheoryCorrection projection ``s_corr`` (bin-integrated σ on the SAME
    ``axis`` edges), it is overlaid and TWO residual panels added: a ratio (param
    model ÷ SCETlib+DYTurbo) and a DIFFERENTIAL difference Δ(dσ/dx) =
    (model − corr)/width. The diff is in the top panel's units, so an additive
    pedestal in the density reads as a horizontal line while a multiplicative bias
    slopes with the spectrum — discriminating a constant offset from a fractional
    one. Without ``s_corr`` the figure is a single panel. Values are plotted as
    DIFFERENTIAL dσ/dx (bin-integrated σ ÷ bin width) so the variable binning,
    notably the wide ptVGen overflow bin, reads correctly; the ratio is
    width-independent.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [n for n, _ in gen_axes]
    if axis not in names:
        raise SystemExit(f"--plot-axis {axis!r} not in gen axes {names}")
    ai = names.index(axis)
    other = 1 - ai  # exactly 2 gen axes (ptVGen, absYVGen)
    edges = np.asarray(gen_axes[ai][1], dtype=np.float64)
    widths = np.diff(edges)
    s = sigma_gen.sum(axis=other)
    ds = s / widths
    corr_label = corr_label or "SCETlib+DYTurbo"
    show_ratio = s_corr is not None
    if show_ratio:
        ratio = np.divide(s, s_corr, out=np.ones_like(s), where=s_corr != 0)
        diff = (s - s_corr) / widths  # differential difference Δ(dσ/dx)

    if show_ratio:
        fig, (ax, axr, axd) = plt.subplots(
            3, 1, sharex=True, figsize=(7, 7.2),
            gridspec_kw={"height_ratios": [3, 1, 1], "hspace": 0.06},
        )
    else:
        fig, ax = plt.subplots(figsize=(7, 5))
        axr = axd = None

    lab = "p$_T^Z$ (ptVGen) [GeV]" if axis == "ptVGen" else axis
    ax.stairs(ds, edges, color="C3", lw=1.6, label="σ_gen(λ) (param model)")
    if s_corr is not None:
        ax.stairs(s_corr / widths, edges, color="C0", lw=1.6, ls=(0, (4, 2)),
                  label=corr_label)
    ax.set_ylabel(r"d$\sigma_{\mathrm{gen}}$/d(" + axis + ")  [a.u.]")
    ax.margins(x=0)
    ax.legend(loc="upper right", fontsize=9)
    ax.text(
        0.975, 0.60, _lambda_box_text(eff, gnu), transform=ax.transAxes,
        ha="right", va="top", fontsize=7.5, family="monospace",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.7", alpha=0.9),
    )

    if show_ratio:
        axr.stairs(ratio, edges, color="k", lw=1.4)
        axr.axhline(1.0, color="0.5", lw=0.8, ls="--")
        axr.set_ylabel("param model /\nSCETlib+DYTurbo")
        axr.margins(x=0)
        # Zoom around 1 to show the (often sub-%) residual, keeping 1.0 in frame;
        # small window if the ratio is flat.
        rlo, rhi = float(np.min(ratio)), float(np.max(ratio))
        pad = max((rhi - rlo) * 0.25, 0.003)
        axr.set_ylim(min(rlo, 1.0) - pad, max(rhi, 1.0) + pad)
        # Differential difference: horizontal ⇒ additive pedestal in density;
        # tracks the spectrum shape ⇒ multiplicative (fractional) offset.
        axd.stairs(diff, edges, color="C2", lw=1.4)
        axd.axhline(0.0, color="0.5", lw=0.8, ls="--")
        axd.set_xlabel(lab)
        axd.set_ylabel("(model − corr)\n/ d(" + axis + ")")
        axd.margins(x=0)
        rng = (f"; model/corr [{rlo:.4f}, {rhi:.4f}]"
               f"; Δ(dσ/dx) [{float(np.min(diff)):.3g}, {float(np.max(diff)):.3g}]")
    else:
        ax.set_xlabel(lab)
        rng = ""

    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_path}  (axis={axis}, summed over {names[other]}{rng})")


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--btgrid", default=BTGRID_DIR, help="SCETlib bT-grid directory")
    p.add_argument("--datacard", default=None,
                   help="hdf5 fallback for the GEN EDGES (no default). λ are NOT "
                        "sourced from it — use --meta-from for that")
    # λ tune: base source (optional) + functional form
    p.add_argument("--meta-from", default=None,
                   help="hdf5 to read the base λ tune from (datacard/fitresults metadata); optional")
    p.add_argument("--np-model", default=None,
                   help="F_eff functional-form override (default: the base tune's form — "
                        "tanh_2 for the canonical / --theory-corr base)")
    p.add_argument("--np-model-nu", default=None,
                   help="γ_ν^NP functional-form override (default: the base tune's form)")
    # λ values to evaluate at (applied on top of the base, in this order)
    p.add_argument("--fitresult", default=None,
                   help="fitresults hdf5 to read the POSTFIT λ from (optional)")
    p.add_argument("--result", default=None, help="fitresult group suffix (e.g. 'nominal')")
    p.add_argument("--lambdas", default=None,
                   help="λ values 'name=val,...' evaluated on top of the base tune "
                        "(e.g. lambda2=0.5); unset params stay at the base (FranksVals "
                        "by default), NOT 0")
    # gen-edge source
    p.add_argument("--ptv-edges", default=None,
                   help="ptVGen edges 'a,b,c,...' (default: 1-GeV bins over [0,40])")
    p.add_argument("--absy-edges", default=None,
                   help="absYVGen edges 'a,b,c,...' (default: single bin [0,5])")
    p.add_argument("--gen-edges-from", default=None,
                   help="hdf5 whose scetlib_np auxiliary gives the gen edges "
                        "(default: --datacard, else the built-in defaults)")
    # TheoryCorrection overlay
    p.add_argument("--theory-corr", default=None,
                   help="TheoryCorrection .pkl.lz4 to overlay the official "
                        "SCETlib+DYTurbo gen distribution (its {generator}_hist)")
    p.add_argument("--theory-corr-proc", default=None,
                   help="proc key in the corr file (default: the single physics key)")
    p.add_argument("--theory-corr-var", default="pdf0",
                   help="vars label to read from the corr hist (default pdf0 = central)")
    p.add_argument("--theory-corr-normalize", action="store_true",
                   help="rescale the corr curve to the model σ_gen integral "
                        "(shape-only comparison; default off = absolute overlay)")
    # model / output
    p.add_argument("--q-lo", type=float, default=Q_LO)
    p.add_argument("--q-hi", type=float, default=Q_HI)
    p.add_argument("--no-nonsingular", action="store_true",
                   help="resum-only σ_gen (σ_ns = 0; skips the FO inputs)")
    p.add_argument("--plot", default=None,
                   help="optional path (e.g. .png/.pdf) to write a 1-D projection plot "
                        "of σ_gen(λ) [+ theory-corr overlay/ratio]; see --plot-axis")
    p.add_argument("--plot-axis", default="ptVGen", choices=["ptVGen", "absYVGen"],
                   help="gen axis to project onto for --plot (default ptVGen = ptZ)")
    args = p.parse_args(argv)

    # ---- λ: build a PHYSICAL base tune (model CONSTRUCTED there so the
    # positive-σ_gen guard passes), then EVALUATE at base + the requested
    # overrides (--fitresult postfit, then --lambdas). Params not set stay at the
    # base, NOT at 0.
    import copy

    overrides = {}
    if args.fitresult:
        from wremnants.postprocessing.scetlib_np.fitresult_lambdas import _flat_values

        pf = _flat_values(args.fitresult, which="postfit", result=args.result)
        overrides.update(pf)
        print(f"[λ] postfit from {args.fitresult}: {pf}")
    try:
        overrides.update(parse_lambda_overrides(args.lambdas))
    except ValueError as e:
        p.error(str(e))

    base = copy.deepcopy(resolve_base_lambda(args))
    if args.np_model:
        base["eff_params"]["np_model"] = args.np_model
    if args.np_model_nu:
        base["gnu_params"]["np_model_nu"] = args.np_model_nu
    eff, gnu, explicit = assemble_tune(base, overrides)

    gen_axes = resolve_gen_axes(args)

    from wremnants.postprocessing.scetlib_np.sigma_gen import SigmaGenModel

    print("\n[core] constructing SigmaGenModel at the base tune (bt-grid integral) …",
          flush=True)
    t0 = time.time()
    core = SigmaGenModel(
        btgrid_dir=args.btgrid,
        lambda_central=base,
        gen_axes=gen_axes,
        Q_lo=args.q_lo,
        Q_hi=args.q_hi,
        include_nonsingular=not args.no_nonsingular,
    )
    print(f"  constructed in {time.time()-t0:.1f}s; gen grid {core.gen_shape} "
          f"({[n for n, _ in core.gen_axes]})")

    print(f"\n[λ] evaluating matched σ_gen at:")
    print(f"  F_eff  : {eff}")
    print(f"  γ_ν^NP : {gnu}")
    if explicit:
        print(f"  (set via --lambdas/--fitresult: {explicit}; the rest stay at the base)")
    else:
        print("  (no overrides — evaluating at the base tune itself)")

    # ---- σ_gen on the (ptVGen, absYVGen) gen grid. Reuse the construction central
    # when no overrides; else evaluate the tune (no positivity guard, so a weak-NP
    # tune can be inspected — it can dip negative at the lowest qT).
    t0 = time.time()
    if explicit:
        sigma_gen = np.asarray(core.sigma_gen(eff, gnu).numpy(), dtype=np.float64)
    else:
        sigma_gen = np.asarray(core.sigma_gen_central.numpy(), dtype=np.float64)
    print(f"  σ_gen computed in {time.time()-t0:.1f}s; shape {sigma_gen.shape}")

    n_bad = int(np.sum(sigma_gen <= 0))
    if n_bad:
        print(f"  [warning] {n_bad}/{sigma_gen.size} σ_gen bins are non-positive at "
              f"this tune (expected where the NP damping is weak, esp. low qT).")

    print(f"\n  Σ σ_gen        = {sigma_gen.sum():.6g}")
    print(f"  per-bin σ_gen  : min {sigma_gen.min():.4g}  max {sigma_gen.max():.4g}")
    print("\n  σ_gen(λ) per (ptVGen × absY) bin:")
    with np.printoptions(precision=4, suppress=True, linewidth=140):
        print(sigma_gen)

    # ---- NP physical-validity detectors at THIS λ. A wrong-sign tune's pathology
    # (anti-damping NP → oscillating, negative native σ(qT)) is AVERAGED AWAY in
    # the binned σ_gen above, so check the native spectrum and form factors
    # directly. Detectors only — change nothing; fit-time enforcement is the
    # np_damping_wall.NPDampingWall regularizer.
    from wremnants.postprocessing.scetlib_np import param_model_diagnostics as ppd

    rep = ppd.np_physical_report(core, eff, gnu)
    damp, neg = rep["damp"], rep["neg"]
    print("\n  NP physical-validity:")
    print(f"    γ_ν^NP damping : {'OK' if not damp['gamma_nu_wrong_sign'] else 'WRONG SIGN'}"
          f"  (max γ_ν over probe bT = {damp['gamma_nu_max']:+.3g}; must be ≤ 0)")
    print(f"    F_eff decays   : {'OK' if not damp['F_eff_growing'] else 'GROWING (bT-integral divergence sign)'}")
    print(f"    native σ(qT)≥0 : neg_area_frac={neg['neg_area_frac']:.3g} "
          f"(λ_central {rep['central_neg_area']:.3g}), min/peak={neg['min_over_peak']:+.3g}, "
          f"n_neg_bins={neg['n_neg_bins']}")

    # WHERE the negativity sits (native Y × qT). The σ(qT)<0 dip is laundered by
    # the gen-binning, so locating it is the discriminator: negative cells inside
    # the |Y|≤2.5 acceptance at accessible qT matter for interpretation; cells only
    # at |Y|>2.5 or beyond the fit's qT reach are doubly invisible. ACCEPT_ABSY is
    # the Z dilepton |yll| acceptance edge (boson-Y proxy).
    ACCEPT_ABSY = 2.5
    if neg.get("n_neg_bins") and neg.get("worst") is not None:
        w = neg["worst"]
        print(f"    worst neg cell : σ={w['value']:.4g} ({w['frac_of_peak']*100:+.1f}% of peak) "
              f"at Y={w['Y']:+.3g}, qT={w['qT']:.3g} GeV")
        cells = neg.get("neg_bins") or []
        in_cells = [c for c in cells if abs(c["Y"]) <= ACCEPT_ABSY]
        n_in, n_out = len(in_cells), len(cells) - len(in_cells)
        qr = neg.get("neg_qT_range")
        print(f"    neg-cell split : {n_in} inside |Y|≤{ACCEPT_ABSY}, {n_out} outside "
              f"(|Y|≤{neg.get('neg_absY_max', float('nan')):.2g} reached); "
              f"qT∈[{qr[0]:.3g}, {qr[1]:.3g}] GeV" if qr else "")
        if in_cells:
            wi = min(in_cells, key=lambda c: c["value"])
            print(f"    worst in-acc   : σ={wi['value']:.4g} ({wi['frac_of_peak']*100:+.1f}% of peak) "
                  f"at Y={wi['Y']:+.3g}, qT={wi['qT']:.3g} GeV   "
                  f"[the |Y|≤{ACCEPT_ABSY} pathology the fit region could see]")
        n_show = min(10, len(cells))
        if n_show:
            print(f"    most-negative {n_show} cells (Y, qT[GeV], σ, %peak):")
            for c in cells[:n_show]:
                inacc = "in " if abs(c["Y"]) <= ACCEPT_ABSY else "out"
                print(f"        [{inacc}] Y={c['Y']:+.3g}  qT={c['qT']:7.3g}  "
                      f"σ={c['value']:+.4g}  ({c['frac_of_peak']*100:+.1f}%)")

    if not rep["ok"]:
        print("    ⚠  UNPHYSICAL NP TUNE — the differential σ(qT) is negative / the NP is "
              "anti-damping.\n       This is hidden in the binned σ_gen above; do not treat "
              "this point as a physical prediction.")

    # ---- optional: project an official TheoryCorrection run onto the same axis.
    s_corr = None
    corr_label = None
    if args.theory_corr:
        h_corr = load_theory_corr_hist(args.theory_corr, args.theory_corr_proc)
        s_corr = theory_corr_projection(
            h_corr, core.gen_axes, args.plot_axis, var=args.theory_corr_var,
            q_window=(args.q_lo, args.q_hi),
        )
        other = 1 - [n for n, _ in core.gen_axes].index(args.plot_axis)
        s_model = sigma_gen.sum(axis=other)
        sum_corr, sum_model = float(s_corr.sum()), float(s_model.sum())
        norm_label = ""
        if args.theory_corr_normalize and sum_corr != 0:
            scale = sum_model / sum_corr
            s_corr = s_corr * scale
            norm_label = f", ×{scale:.4f} (shape-normalized)"
            print(f"\n[theory-corr] shape-normalized to the model integral (×{scale:.5f})")
        corr_label = f"SCETlib+DYTurbo ({args.theory_corr_var}){norm_label}"
        rcorr = np.divide(s_model, s_corr, out=np.ones_like(s_model), where=s_corr != 0)
        print(f"\n[theory-corr] projection onto {args.plot_axis} (var={args.theory_corr_var}):")
        print(f"  Σ corr           = {sum_corr:.6g}")
        print(f"  Σ model / Σ corr = {sum_model/sum_corr:.5f}"
              + ("  (== 1 after --theory-corr-normalize)" if args.theory_corr_normalize else ""))
        print(f"  model / corr per bin : min {rcorr.min():.4f}  max {rcorr.max():.4f}")
        with np.printoptions(precision=4, suppress=True, linewidth=140):
            print(f"  model/corr: {rcorr}")
        if not args.plot:
            print("[theory-corr] (pass --plot to also write the overlay figure)")

    if args.plot:
        make_projection_plot(
            sigma_gen, core.gen_axes, args.plot_axis, args.plot, eff, gnu,
            s_corr=s_corr, corr_label=corr_label,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
