"""Load the (reco × gen) response matrix R for the ParamModel.

R lives in the unfolding histmaker output (a separate hdf5 from the fit tensor)
as ``nominal_prefsr_yieldsUnfolding`` under the Z sample group. We slice
``acceptance=True`` (gen-fiducial) and project to the reco × (ptVGen, absYVGen)
axes — which SUMS the helicitySig axis: R is filled with the weight PARTITION
``nominal_weight_helicity`` whose 8 pieces add back up, so the physical yield
is the helicitySig SUM (NOT UL). The gen-total normalizer N_gen, by contrast,
is filled with ``csAngularMoments`` and takes the UL component
(``helicitySig=-1``); see ``_select_ul_helicity`` and the inline comment in
``load_R``. The full response-fold formula and this SUM-vs-UL subtlety are
documented once in the :mod:`param_model` module docstring (single source of
truth) — consult it rather than re-deriving here.

Single entry point:

    load_R(unfolding_hdf5_path,
           sample_key="Zmumu_2016PostVFP",
           hist_name="nominal_prefsr_yieldsUnfolding") -> dict
"""

import h5py
import numpy as np

from wums import ioutils as wums_io

# Pre-FSR gen level: the SCETlib btgrid σ_gen is the resummed *boson* qT/Y
# (QCD, before QED FSR), so the response and gen-total must also be pre-FSR for
# σ_gen, R, and N_gen to live at the same gen level. (postfsr variants exist in
# the same file — nominal_postfsr_yieldsUnfolding / "postfsr" — for comparison.)
DEFAULT_HIST = "nominal_prefsr_yieldsUnfolding"
DEFAULT_GENTOTAL = "prefsr"  # xnorm gen-total denominator (pre-reco-selection)
DEFAULT_SAMPLE = "Zmumu_2016PostVFP"

# Axes we keep, in canonical order: reco first, then gen.
RECO_AXES = ("ptll", "yll", "cosThetaStarll_quantile", "phiStarll_quantile")
GEN_AXES = ("ptVGen", "absYVGen")
# helicitySig is an angular-moment axis; we take the UL component (value -1),
# the angular-integrated total — see _select_ul_helicity. acceptance is sliced
# to True (gen-fiducial) at use.
HELICITY_AXIS = "helicitySig"

# Gen ptVGen overflow handling. The unfolding hist's ptVGen axis ends at 44 (the
# last reco-ptll edge), but ~3.6% of the Z yield has true gen qT > 44 and
# resolution-migrates into the high-ptll reco bins (6.2% of the last reco bin).
# Dropping that gen-overflow column makes σ_reco low there. We instead fold the
# ptVGen overflow into an extra gen bin so the model can supply a σ_gen for it
# (the btgrid integral over qT ∈ (44, PTVGEN_OVERFLOW_EDGE]). The edge must be
# ≤ the btgrid qT max (the fineall grid runs to 100) and should coincide with a
# gen-histmaker ptVgen edge so the gen-level cross-check's _merge_matrix is exact
# — 100 satisfies both. (absYVGen has zero overflow: |Y| ≤ 2.5 is fully contained,
# so only ptVGen needs this.)
PTVGEN_OVERFLOW_EDGE = 100.0


def _select_ul_helicity(h):
    """Select the UL angular component (helicitySig = -1), if that axis exists.

    Applied to the gen-total denominator N_gen ONLY — NOT to R. N_gen is filled
    with ``csAngularMoments``, a moment expansion whose A_i bins (0..7) are
    signed and do NOT sum to σ; only UL (value -1) is the angular-integrated
    total, so N_gen takes UL. R is the opposite: filled with the weight
    PARTITION ``nominal_weight_helicity``, it is recovered by SUMMING
    helicitySig (``project``), not by taking UL — see the inline comment in
    ``load_R``. Same axis, opposite reduction; taking UL of R would inflate the
    closure ~15×. Returning h unchanged when the axis is absent keeps
    non-helicity inputs working.
    """
    if HELICITY_AXIS in [a.name for a in h.axes]:
        ul_idx = h.axes[HELICITY_AXIS].index(-1)
        h = h[{HELICITY_AXIS: ul_idx}]
    return h


def _append_axis_overflow(h, axis_name):
    """Values array for ``h`` with ``axis_name``'s OVERFLOW bin appended as one
    extra in-range bin along that axis; every other axis is taken in-range (no
    flow). I.e. ``flow=False`` everywhere except that we keep ``axis_name``'s
    overflow as a genuine trailing bin."""
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
    qT > last gen edge) as a trailing gen bin in R and N_gen, with edge
    ``PTVGEN_OVERFLOW_EDGE``. This lets the model fold a σ_gen(qT>44) through the
    migration into the high-ptll reco bins (see the PTVGEN_OVERFLOW_EDGE note);
    set False for the legacy in-range-only response.
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
        required = set(reco_axes) | set(gen_axes) | {"acceptance", HELICITY_AXIS}
        missing = required - set(ax_names)
        if missing:
            raise ValueError(
                f"{hist_name}: missing expected axes {missing}. " f"Got: {ax_names}"
            )

        # Select acceptance=True (gen events in fiducial), then keep the reco +
        # gen axes. project() SUMS helicitySig out — which is correct *for R*:
        # the joint yield is filled with `nominal_weight_helicity`
        # (= nominal_weight × helWeight_tensor, see helicity_utils), a PARTITION
        # of the event weight into the 8 helicity pieces g_i(cosθ,φ) that ADD
        # BACK UP to the full angular weight. So Σ_helicitySig R is the physical
        # angular-resolved reco×gen yield (the angular dependence lives in the
        # cosThetaStar*/phiStar* reco bins). NB this is the OPPOSITE reduction
        # from N_gen below: N_gen is filled with `csAngularMoments` (a moment
        # expansion whose 0..7 bins do NOT sum to σ), so it takes UL (-1). Same
        # axis, different fill tensor → different recovery. Taking UL of R would
        # discard the angular partition and inflate the closure (~15×).
        h_sel = h[{"acceptance": True}]
        h_proj = h_sel.project(*reco_axes, *gen_axes)

        # Gen-total denominator N_gen(g): the xnorm histogram (e.g. "postfsr"),
        # filled on fiducial gen events BEFORE reco selection. Its gen marginal
        # is the generated total per gen bin (efficiency NOT yet applied), so
        # N_reco(b,g)/N_gen(g) = efficiency × migration — the theory-independent
        # gen→reco response. (The gen marginal of R itself is reco-passing, i.e.
        # already × efficiency, which is the wrong normalizer.)
        N_gen_hist = None
        if gen_total_name is not None and gen_total_name in output:
            gp = output[gen_total_name]
            hg = gp.get() if hasattr(gp, "get") else gp
            # postfsr/prefsr axes: (count, ptVGen, absYVGen, helicitySig). This
            # one is filled with `csAngularMoments`: a moment expansion whose UL
            # bin (-1) is the angular-integrated total σ, while bins 0..7 are the
            # A_i moments (orthogonal, signed) that do NOT sum to σ — summing them
            # overcounts by ~19% here. So take UL. (R above is the opposite: a
            # weight partition, recovered by SUMMING helicitySig.) Then project
            # to the gen axes (sums the trivial 'count' axis) to match R's binning.
            hg = _select_ul_helicity(hg)
            hg_gen = hg.project(*gen_axes)
            # Fold the gen ptVGen overflow into a trailing bin (or drop it).
            N_gen_hist = (
                _append_axis_overflow(hg_gen, "ptVGen")
                if ptVGen_overflow
                else hg_gen.values(flow=False).astype(np.float64)
            )

    # Out from the with-block: hist is materialized. Keep the gen ptVGen overflow
    # as a trailing gen bin so the model can supply σ_gen(qT>44) — else the high-
    # ptll reco bins (fed by true qT>44 migrating down) come out low.
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
