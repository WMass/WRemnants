"""Agreement diagnostics for :class:`SCETlibNPParamModel`: does the model
reproduce, at λ_central, the SHAPE present in the datacard?

The model builds its λ_central prediction two ways:

    σ_gen(λ_c)   the bt-grid Hankel + Q integral, matched (Steps 1-2)
    σ_reco(λ_c)  = R · σ_gen(λ_c)               folded through the response (Step 3)

and the fit applies ``rnorm = σ_reco(λ)/σ_reco(λ_c)`` on the datacard's signal
nominal template. For that transport to be valid the model's λ_central shape must
match the shape baked into the card (see :mod:`param_model`). This module checks
against the references that live IN THE CARD, so it needs no external histmaker:

    reco :  σ_reco(λ_c)        vs  indata.norm[:, signal]   (the template the
                                                             ratio multiplies)
    gen  :  σ_gen(λ_c)         vs  N_gen   (the gen-total in the scetlib_np
                                            auxiliary, == the histmaker's
                                            NP-corrected gen σ at λ_central)

Both are SHAPE comparisons: σ_reco/σ_gen are folded theory cross sections,
norm/N_gen are weighted yields, so they differ by an overall normalization (it
cancels in the fit's ratio). Density plots unit-normalize both curves, so the
comparison carries no ad-hoc scale.

Entry points:
  * :func:`run_reco_guard` — PURE-NUMPY per-bin reco check for the in-fit
    auto-guard (warns, or raises with ``strict``; never imports plotting).
  * :func:`run_card_diagnostics` — full reco + gen comparison (+ optional plots)
    for interactive use; returns the per-bin residuals.
  * ``python -m wremnants.postprocessing.scetlib_np.param_model_diagnostics
    --datacard <card> [--outdir <dir>]`` — construct from a card, write the
    reco + gen agreement plots.

Heavy deps (``hist``, ``wums.plot_tools``, and the plotting/projection helpers
from ``scetlib_np.validation_plots``) are imported LAZILY in the print/plot paths
only, so this module — and the in-fit guard — stays numpy-only.
"""

import os

import numpy as np

# Per-reco/gen axis projection order for the standalone plots (shared names).
from wremnants.postprocessing.scetlib_np.params import (
    GEN_AXES as GEN_PROJ_AXES,
    RECO_AXES as RECO_PROJ_AXES,
)


# =============================================================================
# Postfit NP physical-validity detectors (standalone — NOT part of the fit).
# =============================================================================
# A wrong-sign NP point anti-damps the form factors and makes the *differential*
# σ(qT) oscillate negative; the qT→ptVGen rebin AVERAGES that away, so it is
# invisible in the binned σ_gen / σ_reco / NLL the fit sees. FIT-TIME enforcement
# is ``np_damping_wall.NPDampingWall`` (the exact tanh_2/tanh_6 damping walls); these are
# cheap POSTFIT cross-checks on a constructed :class:`SigmaGenModel` ``core`` at a
# λ point (eff/gnu dicts, same shape as ``core.eff_central`` / ``core.gnu_central``).
# ``np_damping_ok`` probes the CAUSE (the forms must damp); ``spectrum_negativity``
# measures the EFFECT (σ(qT) ≥ 0).
NP_PROBE_BT = (0.3, 1.0, 2.0, 5.0, 10.0, 20.0)


def np_damping_ok(core, eff_params, gnu_params, b_probe=NP_PROBE_BT, gamma_tol=1e-3):
    """Probe the NP form factors (no bT integral) for the physical DAMPING sign.

    Evaluates the actual ``btgrid_tf`` forms the fit integrates at a few bT:
      γ_ν^NP(b) ≤ 0   — CS Sudakov damping; γ_ν^NP > 0 is the anti-damping wrong
                        sign (λ2_ν < 0 / λ4_ν < 0) that makes σ(qT) oscillate neg.
      F_eff(b) decays — TMD damping / bT-integral convergence; F_eff growing with
                        bT is the λ2_eff < 0 divergence trap.
    Empirical proxy for the exact ``np_damping_wall.NPDampingWall`` walls: the γ_ν
    probes test the SAME damping condition the CS walls enforce (form-agnostically
    — the b⁶ term is in the evaluated form, so this also covers tanh_6); the F_eff
    endpoint test at Y=0 is CRUDER than the wall's exact a≥0 ∀b at Y=0 and Y_max.
    NOTE this probes the CARD form (``core.np_model``/``np_model_nu``), so it does
    not see a numerator-form override (``np_model_(nu_)fit``). Cheap (1-D evals)."""
    from wremnants.postprocessing.scetlib_np import btgrid_tf as fz_tf

    b = np.asarray(b_probe, dtype=np.float64)
    eff = {k: v for k, v in eff_params.items() if k != "np_model"}
    gnu = {k: v for k, v in gnu_params.items() if k != "np_model_nu"}
    g = fz_tf.gamma_nu_NP_tf(b, gnu, np_model_nu=core.np_model_nu).numpy()
    F = fz_tf.F_eff_tf(0.0, b, eff, np_model=core.np_model).numpy()
    gamma_max = float(np.max(g))
    feff_growing = bool(F[-1] > F[0])
    return {
        "probe_b": b,
        "gamma_nu": g,
        "F_eff": F,
        "gamma_nu_max": gamma_max,
        "gamma_nu_wrong_sign": bool(gamma_max > gamma_tol),
        "F_eff_growing": feff_growing,
        "ok": (gamma_max <= gamma_tol) and (not feff_growing),
    }


def spectrum_negativity(core, eff_params, gnu_params, sigma_YqT=None, locate=True):
    """Negativity of the native (Y, qT) resummed spectrum at a λ point — the
    σ(qT) < 0 pathology the gen-binning averages away. Scale-free metrics:
      neg_area_frac = Σ|min(σ,0)| / Σ|σ|   (0 physical; → O(1) pathological)
      min_over_peak = min(σ) / max(σ)       (≈0 physical; ≤ −O(1) pathological)
    Pass ``sigma_YqT`` (e.g. ``core.sigma_YqT_central``) to skip recomputation,
    else reconstructed via ``core.sigma_YqT_native``. Judge relative to the
    λ_central baseline (``np_physical_report`` does this): the singular-only
    spectrum carries a tiny benign qT→0 dip.

    With ``locate`` (default True) and a 2-D spectrum on the core's native grids
    (``core.Y_unique`` × ``core.qT_unique``), also returns WHERE the negativity
    sits — the key discriminator for whether it touches the fit region or is
    laundered/out-of-acceptance:
      ``worst``    {iY, iqT, Y, qT, value, frac_of_peak} of the most-negative cell
      ``neg_bins`` per-cell {Y, qT, value, frac_of_peak}, most-negative first
      ``neg_qT_range`` / ``neg_absY_max`` extent of the negative region
    (omitted if the grids are absent or the shape doesn't match)."""
    s = (
        core.sigma_YqT_native(eff_params, gnu_params)
        if sigma_YqT is None
        else sigma_YqT
    )
    s = np.asarray(s)
    peak = float(np.max(s))
    neg = float(np.sum(np.abs(np.minimum(s, 0.0))))
    tot = float(np.sum(np.abs(s)))
    out = {
        "neg_area_frac": neg / tot,
        "min_over_peak": float(np.min(s)) / peak,
        "n_neg_bins": int(np.sum(s < 0)),
    }
    Yg = np.asarray(getattr(core, "Y_unique", None)) if locate else None
    qTg = np.asarray(getattr(core, "qT_unique", None)) if locate else None
    if (
        locate
        and s.ndim == 2
        and Yg is not None and qTg is not None
        and Yg.ndim == 1 and qTg.ndim == 1
        and s.shape == (Yg.size, qTg.size)
    ):
        jmin = np.unravel_index(int(np.argmin(s)), s.shape)
        out["worst"] = dict(
            iY=int(jmin[0]), iqT=int(jmin[1]),
            Y=float(Yg[jmin[0]]), qT=float(qTg[jmin[1]]),
            value=float(s[jmin]), frac_of_peak=float(s[jmin] / peak),
        )
        negidx = np.argwhere(s < 0)
        if negidx.size:
            order = np.argsort(s[negidx[:, 0], negidx[:, 1]])  # most negative first
            out["neg_bins"] = [
                dict(
                    Y=float(Yg[negidx[k, 0]]), qT=float(qTg[negidx[k, 1]]),
                    value=float(s[negidx[k, 0], negidx[k, 1]]),
                    frac_of_peak=float(s[negidx[k, 0], negidx[k, 1]] / peak),
                )
                for k in order
            ]
            out["neg_qT_range"] = (
                float(qTg[negidx[:, 1]].min()), float(qTg[negidx[:, 1]].max())
            )
            out["neg_absY_max"] = float(np.abs(Yg[negidx[:, 0]]).max())
        else:
            out["neg_bins"] = []
            out["neg_qT_range"] = None
            out["neg_absY_max"] = None
    return out


def np_physical_report(
    core, eff_params, gnu_params, sigma_YqT=None, central_neg_area=None
):
    """Combine both detectors into a postfit verdict (no printing, no raising).

    Returns ``{ok, issues, damp, neg, central_neg_area}``: ``ok`` the overall
    verdict, ``issues`` human-readable problems, ``damp``/``neg`` the raw
    sub-results. ``central_neg_area`` anchors the relative negativity threshold
    (default: from ``core`` at λ_central). Callers format/act."""
    damp = np_damping_ok(core, eff_params, gnu_params)
    neg = spectrum_negativity(core, eff_params, gnu_params, sigma_YqT=sigma_YqT)
    if central_neg_area is None:
        central_neg_area = spectrum_negativity(
            core,
            core.eff_central,
            core.gnu_central,
            sigma_YqT=getattr(core, "sigma_YqT_central", None),
            locate=False,
        )["neg_area_frac"]
    neg_bad = neg["neg_area_frac"] > max(0.01, 5.0 * central_neg_area)
    issues = []
    if damp["gamma_nu_wrong_sign"]:
        issues.append(
            f"γ_ν^NP > 0 (anti-damping, wrong CS sign; max={damp['gamma_nu_max']:+.3g}) "
            f"on probe bT={list(damp['probe_b'])}"
        )
    if damp["F_eff_growing"]:
        issues.append(
            "F_eff grows with bT (TMD divergence sign — λ2_eff likely < 0; the bT "
            "integral is then finite only by grid truncation)"
        )
    if neg_bad:
        issues.append(
            f"native σ(qT) significantly negative: neg_area_frac="
            f"{neg['neg_area_frac']:.3g} (λ_central {central_neg_area:.3g}), "
            f"min/peak={neg['min_over_peak']:+.3g}, n_neg_bins={neg['n_neg_bins']}"
        )
    return {
        "ok": damp["ok"] and not neg_bad,
        "issues": issues,
        "damp": damp,
        "neg": neg,
        "central_neg_area": central_neg_area,
    }


# =============================================================================
# Pure-numpy core (no heavy imports) — shared by the guard and the full report.
# =============================================================================
def _shape_residual(model_vals, ref_vals):
    """Per-bin shape residual after matching the overall integral.

    Returns ``(resid, stats)``: ``resid`` has the input shape and is
    ``scale·model/ref - 1`` (NaN where ref<=0), ``scale = Σref/Σmodel``. ``stats``
    carries what a threshold guard keys on: yield-weighted mean |residual| and the
    worst bin (value + flat index)."""
    m = np.asarray(model_vals, dtype=np.float64)
    n = np.asarray(ref_vals, dtype=np.float64)
    if m.shape != n.shape:
        raise ValueError(f"shape mismatch: model {m.shape} vs ref {n.shape}")
    msum, nsum = m.sum(), n.sum()
    scale = nsum / msum if msum != 0 else np.nan
    good = n > 0
    resid = np.full(m.shape, np.nan)
    resid[good] = scale * m[good] / n[good] - 1.0
    rg = np.abs(resid[good])
    wmad = float(np.average(rg, weights=n[good])) if good.any() else np.nan
    imax = int(np.nanargmax(np.abs(resid))) if good.any() else -1
    stats = dict(
        scale=float(scale),
        n_bins=int(good.sum()),
        max_abs=float(rg.max()) if rg.size else np.nan,
        yield_weighted_mean_abs=wmad,
        worst_flat_idx=imax,
        worst_value=float(resid.flat[imax]) if imax >= 0 else np.nan,
    )
    return resid, stats


def card_reco_reference(model, indata):
    """Signal nominal reco template from the card, on the model's reco binning.

    ``indata.norm`` is ``(nbins, nproc)``; the signal column is the histmaker
    nominal yield ``rnorm`` multiplies. Reshaped to ``model.reco_shape`` (same
    axis order as ``sigma_reco_central``)."""
    norm = np.asarray(indata.norm, dtype=np.float64)
    n_reco = int(np.prod(model.reco_shape))
    if norm.shape[0] < n_reco:
        raise ValueError(
            f"indata.norm has {norm.shape[0]} bins < reco bins {n_reco}"
        )
    # v1 single non-masked channel: the reco bins are the leading rows of norm.
    col = norm[:n_reco, model.signal_proc_idx]
    return col.reshape(model.reco_shape)


def card_gen_reference(model):
    """Gen-total ``N_gen`` from the scetlib_np auxiliary, on the model gen grid.

    ``N_gen`` is the prefsr xnorm UL gen-total in the card — the histmaker's
    NP-corrected gen σ at λ_central. It normalizes R, and ``P = R_raw/N_gen`` is
    built theory-independent, so N_gen carries the same nominal correction as
    R_raw. Shape == ``model.gen_shape``."""
    return np.asarray(model._N_gen_flat, dtype=np.float64).reshape(model.gen_shape)


def reco_offending_bins(model, indata, threshold, min_yield_frac=0.0):
    """Reco bins whose |shape residual| exceeds ``threshold`` (fractional).

    Pure numpy (no plotting/hist imports), so the in-fit guard can call it.
    ``min_yield_frac`` (of the max reference bin) optionally drops near-empty bins,
    where a shape residual is meaningless. Returns ``(offenders, stats)``,
    offenders sorted by |residual| descending."""
    ref = card_reco_reference(model, indata)
    mdl = np.asarray(model.sigma_reco_central, np.float64).reshape(model.reco_shape)
    resid, stats = _shape_residual(mdl, ref)
    absr = np.abs(resid)
    if min_yield_frac > 0:
        floor = float(min_yield_frac) * float(np.nanmax(ref))
        absr = np.where(ref >= floor, absr, np.nan)
    names = [n for n, _ in model._reco_axes_meta]
    offenders = []
    for idx in np.argwhere(absr > threshold):
        idx = tuple(int(i) for i in idx)
        offenders.append(
            dict(
                coord=dict(zip(names, idx)),
                residual=float(resid[idx]),
                ref_yield=float(ref[idx]),
            )
        )
    offenders.sort(key=lambda d: -abs(d["residual"]))
    return offenders, stats


def run_reco_guard(
    model, indata, threshold=0.005, strict=False, min_yield_frac=0.0, max_list=12
):
    """In-fit reco agreement guard (pure numpy). Trips on ANY bin exceeding
    ``threshold`` (|σ_reco(λ_c)/card_nominal − 1|).

    Warns and lists the worst offenders by default; raises iff ``strict``. A
    diagnostic failure never takes down the fit — any error is caught and logged.
    """
    tag = "[SCETlibNPParamModel]"
    try:
        offenders, stats = reco_offending_bins(model, indata, threshold, min_yield_frac)
    except Exception as e:  # a diagnostic must never crash the minimizer
        print(f"{tag} reco agreement check SKIPPED (error: {e})", flush=True)
        return None
    wmean = stats["yield_weighted_mean_abs"] * 100
    if not offenders:
        print(
            f"{tag} reco agreement OK: all {stats['n_bins']} bins within "
            f"{threshold*100:.2f}% (yield-weighted mean |shape−1| = {wmean:.3f}%).",
            flush=True,
        )
        return offenders
    head = "\n".join(
        f"    {d['residual']*100:+.2f}%  ("
        + ", ".join(f"{k}={v}" for k, v in d["coord"].items())
        + f")  ref_yield={d['ref_yield']:.3g}"
        for d in offenders[:max_list]
    )
    more = "" if len(offenders) <= max_list else f"\n    … +{len(offenders)-max_list} more"
    msg = (
        f"{tag} reco agreement: {len(offenders)} bin(s) exceed {threshold*100:.2f}% "
        f"|σ_reco(λ_c)/card_nominal − 1| (yield-weighted mean = {wmean:.3f}%). "
        f"The model's λ_central shape does not match the card's signal template "
        f"in these bins — the rnorm it applies there is transported off the wrong "
        f"baseline. (Trips concentrated in the top ptll bin [37,44] are the KNOWN "
        f"qT>100 grid truncation — see the param_model module docstring 'Known "
        f"limitation' — not misuse; they do not bias the fit.) Worst:\n{head}{more}"
    )
    if strict:
        raise ValueError(
            msg
            + "\n(check_agreement_strict=1 → raising. Pass check_agreement=0 to "
            "disable, raise check_agreement_threshold, or set check_agreement_min_yield "
            "to ignore sparse bins.)"
        )
    print("WARNING: " + msg, flush=True)
    return offenders


# =============================================================================
# Full report (+ optional plots) — heavy deps imported lazily here.
# =============================================================================
def compare_level(model_vals, ref_vals, axes_meta, label):
    """Print detailed per-bin diagnostics + return ``(resid, stats)``.

    ``model_vals`` / ``ref_vals`` are ndarrays on the same binning; ``axes_meta``
    the ``(name, edges)`` list for that level (reco or gen)."""
    from wremnants.postprocessing.scetlib_np.validation_plots import summarize

    print(f"\n{'='*70}\n{label}\n{'='*70}")
    summarize(np.asarray(model_vals, np.float64), np.asarray(ref_vals, np.float64), axes_meta)
    resid, stats = _shape_residual(model_vals, ref_vals)
    names = [n for n, _ in axes_meta]
    shape = tuple(len(e) - 1 for _, e in axes_meta)
    coord = (
        np.unravel_index(stats["worst_flat_idx"], shape)
        if stats["worst_flat_idx"] >= 0
        else None
    )
    coord_str = (
        ", ".join(f"{nm}={c}" for nm, c in zip(names, coord)) if coord is not None else "n/a"
    )
    print(
        f"\n  >> {label}: yield-weighted mean|shape−1| = "
        f"{stats['yield_weighted_mean_abs']*100:.3f}%  |  "
        f"worst bin = {stats['worst_value']*100:+.2f}% at ({coord_str})"
    )
    return resid, stats


def run_card_diagnostics(
    model, indata, outdir=None, do_plots=True, ref_label_reco=None,
    gen_exclude_overflow=True,
):
    """Full reco + gen comparison of the model's λ_central shape to the card's
    references, with optional per-axis shape plots.

    The gen `ptVGen` axis has a known-TRUNCATED overflow bin: the model integrates
    qT only to the bt-grid ceiling (PTVGEN_OVERFLOW_EDGE), while N_gen's overflow
    holds all qT beyond the last edge (unbounded). A global Σ/Σ shape
    normalization forces the totals to match, smearing that one large-yield
    deficit into a flat pedestal across every other bin — an artifact masquerading
    as a constant offset (verified 260625). With ``gen_exclude_overflow`` (default
    True) the gen shape is normalized on the RESOLVED bins only, so the bulk
    closure is faithful and the truncation shows as its own step in the `ptVGen`
    ratio; the `absYVGen` plot then projects resolved-qT only (overflow zeroed)
    for a clean rapidity-shape comparison.

    Returns ``{'reco': (resid, stats), 'gen': (resid, stats)}`` (stats are the
    global-norm ones from compare_level; the resolved-norm bulk figure is
    printed). Reco is unaffected: its overflow ptll bin was cropped to the fit
    binning, so there is no truncated bin to smear."""
    out = {}
    reco_ref = card_reco_reference(model, indata)
    reco_model = np.asarray(model.sigma_reco_central, np.float64).reshape(model.reco_shape)
    out["reco"] = compare_level(
        reco_model, reco_ref, model._reco_axes_meta, "RECO  σ_reco(λ_c) vs card norm[signal]"
    )

    gen_ref = card_gen_reference(model)
    gen_model = np.asarray(model.sigma_gen_central, np.float64).reshape(model.gen_shape)
    out["gen"] = compare_level(
        gen_model, gen_ref, model._gen_axes_meta, "GEN   σ_gen(λ_c) vs card N_gen"
    )

    # Resolved-qT (overflow-excluded) gen normalization — the faithful bulk view.
    gen_names = [n for n, _ in model._gen_axes_meta]
    ptv_ax = gen_names.index("ptVGen") if "ptVGen" in gen_names else 0
    drop = 1 if gen_exclude_overflow else 0
    resolved = [slice(None)] * gen_model.ndim
    resolved[ptv_ax] = slice(0, gen_model.shape[ptv_ax] - drop)
    resolved = tuple(resolved)
    gscale = gen_ref[resolved].sum() / gen_model[resolved].sum()
    if gen_exclude_overflow:
        rres, wres = gscale * gen_model[resolved] / gen_ref[resolved], gen_ref[resolved]
        print(
            f"\n  >> GEN resolved-qT norm (overflow bin excluded): bulk "
            f"yield-weighted mean|shape−1| = "
            f"{np.average(np.abs(rres - 1.0), weights=wres) * 100:.3f}%  "
            f"(the global-norm {out['gen'][1]['yield_weighted_mean_abs']*100:.2f}% is that "
            f"truncation smeared into a pedestal — see docstring)"
        )

    if do_plots and outdir:
        from wremnants.postprocessing.scetlib_np.validation_plots import (
            plot_ptll_ratio,
            tf_to_hist,
        )

        os.makedirs(outdir, exist_ok=True)
        rlabel_reco = ref_label_reco or "card nominal (signal)"
        h_reco_m = tf_to_hist(reco_model, model._reco_axes_meta)
        h_reco_n = tf_to_hist(reco_ref, model._reco_axes_meta)
        for ax in RECO_PROJ_AXES:
            plot_ptll_ratio(
                h_reco_m, h_reco_n, axis=ax,
                out_path=os.path.join(outdir, f"reco_{ax}.png"),
                ref_label=rlabel_reco, model_label=r"ParamModel $\sigma_{reco}(\lambda_c)$",
                rlabel="model / card", density=True,
                title=f"σ_reco(λ_c) vs card nominal — {ax}",
            )
        # gen ptVGen: normalize on resolved bins but keep all bins, so the
        # truncated overflow shows as a step (not a pedestal on the bulk).
        # Pre-scale the model and pass scale=1.0 so the legend has no "(×scale)";
        # the normalization choice lives in the title.
        norm_tag = "resolved-qT norm" if gen_exclude_overflow else "global norm"
        plot_ptll_ratio(
            tf_to_hist(gen_model * gscale, model._gen_axes_meta),
            tf_to_hist(gen_ref, model._gen_axes_meta),
            axis="ptVGen", density=False,
            out_path=os.path.join(outdir, "gen_ptVGen.png"),
            ref_label=r"card $N_{gen}$",
            model_label=r"ParamModel $\sigma_{gen}(\lambda_c)$",
            rlabel="model / $N_{gen}$", rrange=(0.78, 1.05),
            title=f"σ_gen(λ_c) vs card N_gen — ptVGen ({norm_tag})",
        )
        # gen absYVGen: project resolved-qT only (overflow zeroed in both) so the
        # rapidity shape isn't contaminated by the truncated bin.
        gm_res, gn_res = gen_model.copy(), gen_ref.copy()
        if gen_exclude_overflow:
            ov = [slice(None)] * gen_model.ndim
            ov[ptv_ax] = gen_model.shape[ptv_ax] - 1
            gm_res[tuple(ov)] = 0.0
            gn_res[tuple(ov)] = 0.0
        plot_ptll_ratio(
            tf_to_hist(gm_res, model._gen_axes_meta),
            tf_to_hist(gn_res, model._gen_axes_meta),
            axis="absYVGen", density=True,
            out_path=os.path.join(outdir, "gen_absYVGen.png"),
            ref_label=r"card $N_{gen}$",
            model_label=r"ParamModel $\sigma_{gen}(\lambda_c)$",
            rlabel="model / $N_{gen}$", rrange=(0.95, 1.05),
            title="σ_gen(λ_c) vs card N_gen — |y|" + (", resolved qT" if gen_exclude_overflow else ""),
        )
        print(f"\n  plots written under: {outdir}")
    return out


def main(argv=None):
    import argparse
    import time

    from rabbit.inputdata import FitInputData

    from wremnants.postprocessing.scetlib_np.param_model import SCETlibNPParamModel

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datacard", required=True, help="fit-input hdf5 (must carry the scetlib_np auxiliary)")
    p.add_argument("--btgrid", default=None, help="SCETlib bt-grid dir (default: model's data-area copy)")
    p.add_argument("--signal-proc", default="Zmumu")
    p.add_argument("--outdir", default=None, help="plot output dir ('' / unset to skip plotting)")
    args = p.parse_args(argv)

    print("Loading FitInputData …", flush=True)
    t0 = time.time()
    indata = FitInputData(args.datacard)
    print(f"  loaded in {time.time()-t0:.1f}s; nproc={indata.nproc}", flush=True)

    print("Constructing SCETlibNPParamModel (runs the bt integral at λ_central) …", flush=True)
    t0 = time.time()
    kw = dict(signal_proc=args.signal_proc, check_agreement=False)  # report below, not the guard
    if args.btgrid:
        kw["btgrid_dir"] = args.btgrid
    model = SCETlibNPParamModel(indata, **kw)
    print(f"  constructed in {time.time()-t0:.1f}s", flush=True)

    run_card_diagnostics(
        model, indata, outdir=(args.outdir or None), do_plots=bool(args.outdir)
    )


if __name__ == "__main__":
    main()
