"""Moved from wremnants.postprocessing.scetlib_np on 2026-08-26: the response
loader is used by the autodiff param model's card path, and scetlib_np is
superseded. The scetlib_np copy is left in place for that package's own
workflow.

Load the (reco × gen) response matrix R for the ParamModel.

R lives in the unfolding histmaker output (a separate hdf5 from the fit tensor)
as ``nominal_prefsr_yieldsUnfolding`` under the Z sample group. SUM over
``acceptance`` (True + False), project to reco × (ptVGen, absYVGen): this
SUMS the helicitySig axis. R is filled with the weight PARTITION
``nominal_weight_helicity`` whose 8 pieces add back up, so the physical yield is
the helicitySig SUM (NOT UL). The gen-total normalizer N_gen is filled with
``csAngularMoments`` and takes the UL component (``helicitySig=-1``); see
``_select_ul_helicity`` and the inline comment in ``load_R``. Full response-fold
formula and this SUM-vs-UL subtlety: :mod:`param_model` module docstring.

Single entry point:

    load_R(unfolding_hdf5_path,
           sample_key="Zmumu_2016PostVFP",
           hist_name="nominal_prefsr_yieldsUnfolding") -> dict
"""

import h5py
import numpy as np

from wums import ioutils as wums_io

# Axes kept, in canonical order: reco first, then gen.
# Carried here rather than imported from scetlib_np: this module is now part
# of the autodiff line, and a cross-package import would leave scetlib_ad
# depending on a package that is superseded.
RECO_AXES = ("ptll", "yll", "cosThetaStarll_quantile", "phiStarll_quantile")
GEN_AXES = ("ptVGen", "absYVGen")

# Pre-FSR: the btgrid σ_gen is resummed *boson* qT/Y (QCD, pre-QED-FSR), so R
# and N_gen must also be pre-FSR for σ_gen, R, N_gen to share a gen level.
# (postfsr variants — nominal_postfsr_yieldsUnfolding / "postfsr" — also exist.)
DEFAULT_HIST = "nominal_prefsr_yieldsUnfolding"
DEFAULT_GENTOTAL = "prefsr"  # xnorm gen-total denominator (pre-reco-selection)
# mz_dilepton --responseGenBinning writes a SECOND reco x gen pair on a finer gen
# grid (the theory correction's own cells), in parallel to the unfolding one. Same
# structure, no helicity axis: filled with `nominal_weight`, which is what
# summing the helicity partition of the unfolding hist gives.
RESPONSE_HIST = "nominal_prefsr_yieldsResponse"
RESPONSE_GENTOTAL = "prefsr_response"
DEFAULT_SAMPLE = "Zmumu_2016PostVFP"
# helicitySig: angular-moment axis; take UL (value -1), the angular-integrated
# total (see _select_ul_helicity). acceptance is SUMMED (True + False) at use.
HELICITY_AXIS = "helicitySig"

# Gen ptVGen overflow. The ptVGen axis ends at 44 (last reco-ptll edge), but
# ~3.6% of the Z yield has true gen qT > 44 and resolution-migrates into the
# high-ptll reco bins (6.2% of the last reco bin); dropping that column makes
# σ_reco low there. Instead fold the overflow into an extra gen bin so the model
# can supply a σ_gen for it (btgrid integral over qT ∈ (44, PTVGEN_OVERFLOW_EDGE]).
# The edge must be ≤ the btgrid qT max (fineall runs to 100) and should coincide
# with a gen-histmaker ptVgen edge so the cross-check's _merge_matrix is exact;
# 100 satisfies both. (absYVGen has zero overflow: |Y| ≤ 2.5 is fully contained.)
PTVGEN_OVERFLOW_EDGE = 100.0


def _select_ul_helicity(h):
    """Select the UL angular component (helicitySig = -1), if that axis exists.

    For the gen-total denominator N_gen ONLY, NOT R. N_gen is filled with
    ``csAngularMoments``, a moment expansion whose A_i bins (0..7) are signed and
    do NOT sum to σ; only UL (value -1) is the angular-integrated total, so N_gen
    takes UL. R is filled with the weight PARTITION ``nominal_weight_helicity``
    and is recovered by SUMMING helicitySig (``project``), not by UL (see the
    inline comment in ``load_R``). Same axis, opposite reduction; taking UL of R
    would inflate the closure ~15×. Returning h unchanged when the axis is absent
    keeps non-helicity inputs working.
    """
    if HELICITY_AXIS in [a.name for a in h.axes]:
        ul_idx = h.axes[HELICITY_AXIS].index(-1)
        h = h[{HELICITY_AXIS: ul_idx}]
    return h


def _append_axis_overflow(h, axis_name):
    """Values array for ``h`` with ``axis_name``'s OVERFLOW bin appended as one
    extra in-range bin along that axis; every other axis in-range (no flow).
    I.e. ``flow=False`` everywhere except ``axis_name``'s overflow kept as a
    trailing bin."""
    full = h.values(flow=True)
    inr = h.values(flow=False).astype(np.float64)
    idx, pos = [], None
    for p, ax in enumerate(h.axes):
        uf = 1 if ax.traits.underflow else 0
        if ax.name == axis_name:
            pos = p
            idx.append(slice(uf + ax.size, uf + ax.size + 1))  # the overflow bin
        else:
            idx.append(slice(uf, uf + ax.size))  # in-range only
    if pos is None:
        raise ValueError(
            f"_append_axis_overflow: no {axis_name!r} axis in {[a.name for a in h.axes]}"
        )
    over = full[tuple(idx)].astype(np.float64)
    return np.concatenate([inr, over], axis=pos)


def corr_generator_of(unfolding_hdf5_path):
    """Name of the theory correction the response grid was taken from, or None.

    The histmaker records its own ``args`` in ``meta_info``; when it ran with
    ``--responseGenBinning theoryCorr`` the response gen axes ARE the first
    ``--theoryCorr`` generator's grid, so that name is what a downstream consumer
    needs to check its binning against (see
    ``theory_corrections.check_gen_grid_vs_correction``). Returns None when the
    file does not say, so the check degrades to a no-op rather than an error.
    """
    try:
        with h5py.File(unfolding_hdf5_path, "r") as f:
            if "meta_info" not in f:
                return None
            meta = wums_io.pickle_load_h5py(f["meta_info"])
        args = (meta.get("meta_info", meta) or {}).get("args", {}) or {}
        if args.get("responseGenBinning") != "theoryCorr":
            return None
        corrs = args.get("theoryCorr") or []
        return str(corrs[0]) if corrs else None
    except (OSError, KeyError, TypeError, IndexError, AttributeError):
        return None


def has_response(
    unfolding_hdf5_path,
    sample_key=DEFAULT_SAMPLE,
    hist_name=DEFAULT_HIST,
    gen_total_name=DEFAULT_GENTOTAL,
):
    """True iff this histmaker output carries BOTH the reco x gen response hist
    and the gen-total xnorm hist (needed for R *and* N_gen).

    setupRabbit uses this to decide whether to embed the SCETlib-NP response in
    the datacard. Requiring both makes a generic unfolding run (response hist but
    no gen-total) a silent no-op, not an error. Never raises (any structural
    problem -> False); materializes no histogram.
    """
    try:
        with h5py.File(unfolding_hdf5_path, "r") as f:
            if sample_key not in f:
                return False
            sample = wums_io.pickle_load_h5py(f[sample_key])
            output = sample["output"]
            return hist_name in output and gen_total_name in output
    except (OSError, KeyError, TypeError):
        return False


def load_R(
    unfolding_hdf5_path,
    sample_key=DEFAULT_SAMPLE,
    hist_name=DEFAULT_HIST,
    reco_axes=RECO_AXES,
    gen_axes=GEN_AXES,
    gen_total_name=DEFAULT_GENTOTAL,
    ptVGen_overflow=True,
):
    """Load R from the unfolding histmaker output.

    Returns a dict with:
        R           : ndarray shape (*reco_sizes, *gen_sizes), float64
        reco_axes   : list of (name, edges) tuples in the canonical order
        gen_axes    : list of (name, edges) tuples
        reco_shape  : tuple of axis sizes (reco)
        gen_shape   : tuple of axis sizes (gen)
        source      : (path, sample_key, hist_name) for traceability

    ``ptVGen_overflow`` (default True): append the gen ptVGen overflow (true
    qT > last gen edge) as a trailing gen bin in R and N_gen, edge
    ``PTVGEN_OVERFLOW_EDGE``, so the model can fold σ_gen(qT>44) through the
    migration into the high-ptll reco bins (see the PTVGEN_OVERFLOW_EDGE note).
    False = legacy in-range-only response.
    """
    with h5py.File(unfolding_hdf5_path, "r") as f:
        if sample_key not in f:
            raise KeyError(
                f"{unfolding_hdf5_path}: no '{sample_key}' group. "
                f"Available top-level: {list(f.keys())[:10]}"
            )
        sample = wums_io.pickle_load_h5py(f[sample_key])
        try:
            output = sample["output"]
        except (KeyError, TypeError) as exc:
            raise KeyError(
                f"{sample_key}: no 'output' dict — schema mismatch?"
            ) from exc
        if hist_name not in output:
            joint_candidates = [
                k for k in output.keys() if "yieldsUnfolding" in k or "Unfolding" in k
            ]
            raise KeyError(
                f"{sample_key}: '{hist_name}' missing. "
                f"Joint-hist candidates: {joint_candidates[:5]}"
            )
        proxy = output[hist_name]
        # Force materialization while the file is open.
        h = proxy.get() if hasattr(proxy, "get") else proxy

        # Sanity-check the axes.
        ax_names = [a.name for a in h.axes]
        # helicitySig is OPTIONAL: the unfolding hist carries the weight
        # partition and is recovered by summing it, while the parallel response
        # hist is filled with `nominal_weight` and has no such axis. Both give
        # the same R.
        required = set(reco_axes) | set(gen_axes) | {"acceptance"}
        missing = required - set(ax_names)
        if missing:
            raise ValueError(
                f"{hist_name}: missing expected axes {missing}. " f"Got: {ax_names}"
            )

        # SUM over `acceptance` (True + False); do NOT slice True. A reco-selected
        # event belongs in the response regardless of its GEN acceptance flag: it
        # is in the data, and the card's own signal template carries its yield. The
        # response the model hands rabbit is a per-reco-bin AVERAGE over the gen
        # cells feeding that bin, and the correct weights are the template's own
        # per-gen-cell yields, N(acceptance=True) + N(acceptance=False). Slicing
        # True dropped the acceptance=False gen cells out of that mixture while the
        # template yield kept the events, so those events were assigned the
        # in-acceptance mixture's response instead of their own.
        #
        # MEASURED, not argued (studies/scetlib-ad-param-model/260908-absolute-B):
        # on the 770-bin theory-correction-grid card the ABSOLUTE reco closure
        # `total model/nominal` goes 0.999026 -> 0.999775, i.e. |1 - ratio| improves
        # 4.3x, and the raw numerator change `sum(R_summed)/sum(R_True) - 1 =
        # +7.596e-04` accounts for the whole shift. The effect on a fit-consumed
        # response is much smaller, <= 2.4e-05 (260908-acceptance-response); the
        # absolute number is a VALIDATION-metric gain, not a fit gain, because the
        # model enters as a ratio normalised to 1 at the anchor.
        #
        # THE ASSUMPTION, stated plainly: the numerator now contains events whose
        # gen Q lies OUTSIDE the theory correction's Q window, for which sigma_gen
        # has NO prediction. Their reco yield is carried proportionally to the
        # in-window gen yield of the same (qT, |Y|) cell -- an MC extrapolation, of
        # exactly the same status as the reco<-gen migration already encoded in R.
        # It shrinks automatically as the correction's Q coverage grows, so it is a
        # property of today's correction files, not a permanent limitation.
        #
        # `slice(None, None, sum)` sums the IN-RANGE bins only. `acceptance` is a
        # hist.axis.Boolean and has no flow, so this is exactly True + False; being
        # explicit keeps it flow-safe if that ever changes.
        #
        # project() SUMS helicitySig — correct *for R*: the joint yield is filled with
        # `nominal_weight_helicity` (= nominal_weight × helWeight_tensor, see
        # helicity_utils), a PARTITION of the event weight into the 8 helicity
        # pieces g_i(cosθ,φ) that ADD BACK UP to the full angular weight. So
        # Σ_helicitySig R is the physical angular-resolved reco×gen yield (angular
        # dependence lives in the cosThetaStar*/phiStar* reco bins). OPPOSITE
        # reduction from N_gen below: N_gen is filled with `csAngularMoments` (a
        # moment expansion whose 0..7 bins do NOT sum to σ), so it takes UL (-1).
        # Same axis, different fill tensor → different recovery. Taking UL of R
        # would discard the angular partition and inflate the closure (~15×).
        h_sel = h[{"acceptance": slice(None, None, sum)}]
        h_proj = h_sel.project(*reco_axes, *gen_axes)

        # The ptVGen overflow is appended as a trailing gen bin [last edge,
        # PTVGEN_OVERFLOW_EDGE]. On a gen axis that already reaches (or passes)
        # that edge -- the correction's own grid runs to 100 -- that bin would be
        # empty-width and meaningless, so drop the overflow instead. What it holds
        # is then gen qT beyond the correction's range, where the correction file's
        # flow bins are exactly 1, i.e. events the templates leave uncorrected.
        if ptVGen_overflow:
            last_edge = float(h_proj.axes["ptVGen"].edges[-1])
            if last_edge >= PTVGEN_OVERFLOW_EDGE - 1e-9:
                ptVGen_overflow = False

        # Gen-total denominator N_gen(g): the xnorm histogram (e.g. "postfsr"),
        # filled on fiducial gen events BEFORE reco selection. Its gen marginal
        # is the generated total per gen bin (no efficiency yet), so
        # N_reco(b,g)/N_gen(g) = efficiency × migration, the theory-independent
        # gen→reco response. (R's own gen marginal is reco-passing, i.e. already
        # × efficiency — the wrong normalizer.)
        N_gen_hist = None
        if gen_total_name is not None and gen_total_name in output:
            gp = output[gen_total_name]
            hg = gp.get() if hasattr(gp, "get") else gp
            # postfsr/prefsr axes: (count, ptVGen, absYVGen, helicitySig). Filled
            # with `csAngularMoments`: a moment expansion whose UL bin (-1) is the
            # angular-integrated total σ, while bins 0..7 are the A_i moments
            # (orthogonal, signed) that do NOT sum to σ (summing overcounts ~19%
            # here). So take UL. (R above is the opposite: a weight partition,
            # recovered by SUMMING helicitySig.) Then project to the gen axes
            # (sums the trivial 'count' axis) to match R's binning.
            hg = _select_ul_helicity(hg)
            hg_gen = hg.project(*gen_axes)
            # Fold the gen ptVGen overflow into a trailing bin (or drop it).
            N_gen_hist = (
                _append_axis_overflow(hg_gen, "ptVGen")
                if ptVGen_overflow
                else hg_gen.values(flow=False).astype(np.float64)
            )

    # Out of the with-block: hist materialized. Keep the gen ptVGen overflow as a
    # trailing gen bin so the model can supply σ_gen(qT>44); else the high-ptll
    # reco bins (fed by true qT>44 migrating down) come out low.
    R = (
        _append_axis_overflow(h_proj, "ptVGen")
        if ptVGen_overflow
        else h_proj.values(flow=False).astype(np.float64)
    )

    def _gen_edges(name):
        e = np.asarray(h_proj.axes[name].edges, dtype=np.float64)
        if ptVGen_overflow and name == "ptVGen":
            e = np.concatenate([e, [PTVGEN_OVERFLOW_EDGE]])  # (44, 100] overflow bin
        return e

    reco_meta = [(name, h_proj.axes[name].edges) for name in reco_axes]
    gen_meta = [(name, _gen_edges(name)) for name in gen_axes]
    # Derive shapes from the (possibly overflow-extended) arrays, not the hist.
    reco_shape = R.shape[: len(reco_axes)]
    gen_shape = R.shape[len(reco_axes) :]

    if N_gen_hist is not None and N_gen_hist.shape != gen_shape:
        raise ValueError(
            f"gen-total {gen_total_name!r} shape {N_gen_hist.shape} != R gen "
            f"shape {gen_shape}; gen binning mismatch."
        )

    return dict(
        R=R,
        N_gen=N_gen_hist,
        reco_axes=reco_meta,
        gen_axes=gen_meta,
        reco_shape=reco_shape,
        gen_shape=gen_shape,
        source=(unfolding_hdf5_path, sample_key, hist_name),
        corr_generator=corr_generator_of(unfolding_hdf5_path),
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(
            f"usage: python -m {__name__.replace('.', '/')} <unfolding_hdf5> [sample_key]",
            file=sys.stderr,
        )
        sys.exit(2)
    sample = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_SAMPLE
    info = load_R(sys.argv[1], sample_key=sample)
    print(f"R shape         : {info['R'].shape}")
    print(f"R sum           : {info['R'].sum():.6g}")
    print(f"reco_shape      : {info['reco_shape']}")
    print(f"gen_shape       : {info['gen_shape']}")
    for name, edges in info["reco_axes"]:
        print(
            f"  reco {name}: size={len(edges)-1} edges=[{edges[0]:.3g}, {edges[-1]:.3g}]"
        )
    for name, edges in info["gen_axes"]:
        print(
            f"  gen  {name}: size={len(edges)-1} edges=[{edges[0]:.3g}, {edges[-1]:.3g}]"
        )
