import pickle

import hist
import lz4.frame
import numpy as np
import ROOT

import narf
from wremnants.production.muon_efficiencies_smooth import cloneAxis
from wremnants.utilities import common
from wums import logging

logger = logging.child_logger(__name__)

narf.clingutils.Declare('#include "muon_efficiencies_insitu.hpp"')

data_dir = common.data_dir

# In-situ efficiency steps floated as Chebyshev polynomials.
# IDIP   : 1D in pt,           charge-dependent  (NCoeffPt per (eta, q))
# Trigger: 2D in (pt, ut),     charge-dependent  (NCoeffPt*NCoeffUt per (eta,q))
# Iso    : 2D in (pt, ut),     charge-inclusive  (NCoeffPt*NCoeffUt per eta)
# matching the SMP-23-002 measurement variables (pt^3 x ut^2 polynomials).
insitu_eff_steps = ["idip", "trigger", "iso"]

# 48-bin probe-eta decorrelation.
insitu_n_eta = 48
insitu_n_coeff_pt = 4  # 3rd-order Chebyshev in pt
insitu_n_coeff_ut = 3  # 2nd-order Chebyshev in ut (HLT, Iso only)
# Chebyshev x̃ windows shared by ALL analyses (Z dilepton and W single-muon).
# In a combined fit the same-named coefficient nuisances must mean the same
# polynomial, so the windows are fixed here and must NOT follow the per-analysis
# --pt range; the C++ helper clamps pt/ut to the window before the transform
#   x̃ = 2*(clamp(v) - lo)/(hi - lo) - 1.
# pt window = the effMC probe-pt range (the fixed insitu_effMC_pt_edges grid),
# so the effMC and the SF parameterisation share one basis over the full
# measured range; the fit probes populate x̃_pt only up to their analysis pt
# cut, beyond which the polynomial is an extrapolation evaluated with the
# clamp. NB: changing this window redefines the meaning of the coefficient
# nuisances -- accumulated theta_central (insituSFFile) pkls and fit results
# from a previous window are NOT compatible.
# ut window [-30, +100] matches the published smoothSF3D_uTm30to100 file.
insitu_pt_range = (26.0, 65.0)
insitu_ut_range = (-30.0, 100.0)
# Coarse uT bin edges shared by the genUT axis of the effMC probe spectra (both
# histmakers) and the dilepton failHLT/failIso recoUT fit axis. Coarsened in the
# tails because the Z-pT distribution makes |uT| > 20 GeV rare -> avoid empty
# bins in the high-|uT| corners.
insitu_ut_edges = [-30, -10, -5, 0, 5, 10, 20, 40, 100]
# variation step folded into the stored response (units of the unconstrained
# nuisance: theta_c = delta * n_c). Under the logit parameterisation the
# per-leg derivatives are bounded in [-1,1], so delta only sets the nuisance
# units (the physics is delta-independent for unconstrained coefficients).
insitu_delta = 0.01
# effMC is clamped to (0, effMC_max] so that 1-e > 0 (the fail denominator
# D = e*expS + (1-e) stays positive) and to absorb e >= 1 low-stats MC bins.
# 0.9999 sits above the largest genuine effMC (idip peaks at ~0.99976), so no
# real efficiency is clamped while e >= 1 fluctuations still are.
insitu_effMC_max = 0.9999
# The coefficients are unconstrained, so the fit can leave the physical region
# where little data constrains it (the corners of the pt/ut window), giving an
# implied data efficiency eMC*(1+P) >= 1 and hence a negative fail probability.
# Feeding such a point back as the iterative linearisation point used to make
# the helper throw mid-event-loop; 1+P is now capped so eMC*(1+P) <=
# insitu_effData_max. This guards the linearisation point only -- it is not a
# constraint on the fit, which still needs a bounded parameterisation.
insitu_effData_max = 0.9999
# name of the auxiliary bundle carrying the efficiency grid for rabbit's
# InSituEfficiencyBound penalty; must match rabbit.regularization.insitu.AUX_NAME
INSITU_BOUND_AUX_NAME = "insitu_efficiency_bound"

# group-name prefix per step (used for nuisance grouping in setupRabbit)
insitu_step_group = {
    "idip": "effInsituID",
    "trigger": "effInsituHLT",
    "iso": "effInsituIso",
}


def insitu_pt_edges(pt_max):
    """Probe-pt bin edges of the dilepton fail-category fit axes. failID has
    falling probe-pt stats above ~45 GeV (few/empty bins on a 1 GeV grid at
    high pt x forward eta): 1 GeV bins up to 38 GeV, then coarser toward the
    analysis upper pt cut ``pt_max`` (the last edge follows the cut so the
    axis covers all probes).
    """
    edges = [26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 40, 42, 45, 50]
    last = int(round(pt_max))
    assert last > edges[-1], f"upper pt cut {pt_max} must be above {edges[-1]}"
    return [*edges, last]


def insitu_effMC_pt_edges():
    """Probe-pt bin edges of the effMC probe spectra (both histmakers): the
    same grid as the dilepton fit axes, spanning the shared Chebyshev pt
    window (insitu_pt_range)."""
    return insitu_pt_edges(insitu_pt_range[1])


def insitu_ut_clamp_expr(var):
    """C++ expression clamping a uT column into the (flow-less) effMC genUT
    axis window. Without it, out-of-window probes (~1%, fail-enriched 2-3x in
    trigger/iso) silently drop from the effMC spectra while the C++ helper
    clamps uT at evaluation time and applies the EDGE-bin effMC to them;
    clamping at fill time measures the edge bins on exactly the population
    they are applied to (a <=1e-3 effect on the integrated trigger/iso effMC,
    none on idip)."""
    lo = float(insitu_ut_edges[0])
    hi = float(insitu_ut_edges[-1]) - 1e-3
    return f"std::min(std::max({var}, {lo}f), {hi}f)"


def make_insitu_effMC_axes(n_eta, eta_low, eta_high):
    """(axes, cols) of the effMCprobe_* probe spectra: (eta, pt, charge, genUT).
    Shared by the dilepton and single-muon histmakers so the effMC files from
    make_insitu_effMC.py follow one convention regardless of which histmaker
    produced them (the helper looks up effMC via boost-hist axis.index(), so the
    variable pt axis is safe). The pt axis is the fixed insitu_effMC_pt_edges
    grid (..., 45, 50, 65), independent of the per-analysis pt cut.
    """
    axes = [
        hist.axis.Regular(
            int(n_eta), eta_low, eta_high, name="eta", underflow=False, overflow=False
        ),
        hist.axis.Variable(
            insitu_effMC_pt_edges(), name="pt", underflow=False, overflow=False
        ),
        hist.axis.Regular(2, -2.0, 2.0, name="charge", underflow=False, overflow=False),
        hist.axis.Variable(
            insitu_ut_edges, name="genUT", underflow=False, overflow=False
        ),
    ]
    cols = [
        "probeMuons_eta0",
        "probeMuons_pt0",
        "probeMuons_charge0",
        "probeMuons_tnpUT0",
    ]
    return axes, cols


def _build_effMC_hist(h_in, axes_spec, name):
    """Clone an input boost hist into a Weight-storage hist with flow filled
    from the adjacent in-range bin. ``axes_spec`` is a list of
    (axis_name, in_axis, with_flow) tuples giving the desired output axes
    (with flow as requested). The hist is filled positionally from h_in.
    """
    out_axes = []
    flow_dims = []
    for ax_name, ax_in, with_flow in axes_spec:
        if isinstance(ax_in, hist.axis.StrCategory):
            out_axes.append(hist.axis.StrCategory(list(ax_in), name=ax_name))
            flow_dims.append(False)
        else:
            out_axes.append(
                cloneAxis(
                    ax_in,
                    overflow=with_flow,
                    underflow=with_flow,
                    newName=ax_name,
                )
            )
            flow_dims.append(with_flow)
    h = hist.Hist(*out_axes, name=name, storage=hist.storage.Weight())
    h.view(flow=False)["value"] = h_in.values(flow=False)
    # Set under/overflow per axis to the adjacent in-range bin
    view = h.view(flow=True)
    for i_axis, has_flow in enumerate(flow_dims):
        if not has_flow:
            continue
        size_with_flow = out_axes[i_axis].extent
        # underflow -> first in-range
        idx_under = [slice(None)] * view.ndim
        idx_under[i_axis] = 0
        idx_first = [slice(None)] * view.ndim
        idx_first[i_axis] = 1
        view[tuple(idx_under)] = view[tuple(idx_first)]
        # overflow -> last in-range
        idx_over = [slice(None)] * view.ndim
        idx_over[i_axis] = size_with_flow - 1
        idx_last = [slice(None)] * view.ndim
        idx_last[i_axis] = size_with_flow - 2
        view[tuple(idx_over)] = view[tuple(idx_last)]
    return h


def make_muon_insitu_effMC_helper(effMCFile):
    """Load the per-step MC-efficiency hists for the in-situ method.

    ``effMCFile`` is the .pkl.lz4 produced by scripts/corrections/
    make_insitu_effMC.py via wums write_lz4_pkl_output. It holds a dict
        { <process>: { "process": str, "effMC": { "idip": Hist, "trigger":
          Hist, "iso": Hist }, "steps": [...] }, "meta_data": {...},
          "file_meta_data": {...} }
    i.e. the effMC content is nested under a top-level process key alongside
    the provenance metadata. Here:
      - idip   : axes (eta(48), pt(34), charge(2)),                 3D
      - trigger: axes (eta(48), pt(34), charge(2), ut(N_ut)),       4D
      - iso    : axes (eta(48), pt(34), ut(N_ut)),                  3D (charge-incl)

    Returns a dict {step: hist.Hist with flow filled from adjacent bins} so
    the C++ helper does direct per-bin lookups without ever reading an empty
    flow cell.
    """
    with lz4.frame.open(effMCFile, "rb") as fin:
        payload = pickle.load(fin)
    # write_lz4_pkl_output nests the content under a top-level process key,
    # alongside the meta_data / file_meta_data provenance entries.
    content = next(
        v
        for k, v in payload.items()
        if k not in ("meta_data", "file_meta_data", "meta_info")
    )
    effMC_in = content["effMC"]
    assert isinstance(effMC_in, dict) and set(effMC_in.keys()) >= set(
        insitu_eff_steps
    ), f"effMC pkl must be a dict with keys {insitu_eff_steps}, got {list(effMC_in)}"
    logger.info(
        f"Loaded in-situ effMC from {effMCFile} "
        f"(process={content.get('process')}, shapes="
        f"{ {k: effMC_in[k].shape for k in insitu_eff_steps} })"
    )

    h_idip_in = effMC_in["idip"]
    h_trig_in = effMC_in["trigger"]
    h_iso_in = effMC_in["iso"]

    eta_in_idip, pt_in_idip, q_in_idip = h_idip_in.axes
    eta_in_trig, pt_in_trig, q_in_trig, ut_in_trig = h_trig_in.axes
    eta_in_iso, pt_in_iso, ut_in_iso = h_iso_in.axes

    h_idip = _build_effMC_hist(
        h_idip_in,
        [
            ("SF eta", eta_in_idip, True),
            ("SF pt", pt_in_idip, True),
            ("SF charge", q_in_idip, False),
        ],
        name="effMC_insitu_idip",
    )
    h_trig = _build_effMC_hist(
        h_trig_in,
        [
            ("SF eta", eta_in_trig, True),
            ("SF pt", pt_in_trig, True),
            ("SF charge", q_in_trig, False),
            ("SF ut", ut_in_trig, True),
        ],
        name="effMC_insitu_trigger",
    )
    h_iso = _build_effMC_hist(
        h_iso_in,
        [
            ("SF eta", eta_in_iso, True),
            ("SF pt", pt_in_iso, True),
            ("SF ut", ut_in_iso, True),
        ],
        name="effMC_insitu_iso",
    )
    logger.info(
        f"Built in-situ effMC hists: idip={h_idip.shape}, trigger={h_trig.shape}, "
        f"iso={h_iso.shape}"
    )
    return {"idip": h_idip, "trigger": h_trig, "iso": h_iso}


def insitu_parameter_labels(
    n_eta=insitu_n_eta,
    n_coeff_pt=insitu_n_coeff_pt,
    n_coeff_ut=insitu_n_coeff_ut,
):
    """Ordered list of nuisance labels, one per flat tensor index.

    Layout must match muon_efficiencies_insitu.hpp:
      [ idip (q x eta x kPt)
        trigger (q x eta x kPt x mUt)
        iso     (eta x kPt x mUt) ]
    """
    labels = []
    # idip: charge x eta x pt
    for qbit in range(2):
        qtag = "plus" if qbit else "minus"
        for b in range(n_eta):
            for k in range(n_coeff_pt):
                labels.append(f"{insitu_step_group['idip']}_eta{b}_q{qtag}_cPt{k}")
    # trigger: charge x eta x pt x ut
    for qbit in range(2):
        qtag = "plus" if qbit else "minus"
        for b in range(n_eta):
            for k in range(n_coeff_pt):
                for m in range(n_coeff_ut):
                    labels.append(
                        f"{insitu_step_group['trigger']}_eta{b}_q{qtag}"
                        f"_cPt{k}_cUt{m}"
                    )
    # iso: eta x pt x ut (charge-inclusive)
    for b in range(n_eta):
        for k in range(n_coeff_pt):
            for m in range(n_coeff_ut):
                labels.append(f"{insitu_step_group['iso']}_eta{b}_cPt{k}_cUt{m}")
    return labels


def load_insitu_central(
    sf_file,
    n_eta=insitu_n_eta,
    n_coeff_pt=insitu_n_coeff_pt,
    n_coeff_ut=insitu_n_coeff_ut,
):
    """Flat ``theta_central`` array (NSF doubles, in insitu_parameter_labels
    order) for the iterative fit. ``sf_file`` is the accumulated-coefficient
    pkl written by scripts/corrections/make_insitu_effSF.py; ``None`` -> all
    zeros (iteration 0, SF = 1).

    The pkl stores theta as a *label-keyed dict* (not a bare array) so the
    label<->flat-index mapping round-trips through insitu_parameter_labels() on
    both write and read, robust against eta-binning / ordering changes. Both
    the histmaker and the producer import this loader (single source of truth).
    """
    labels = insitu_parameter_labels(n_eta, n_coeff_pt, n_coeff_ut)
    if sf_file is None:
        return [0.0] * len(labels)
    with lz4.frame.open(sf_file, "rb") as fin:
        payload = pickle.load(fin)
    content = next(
        v
        for k, v in payload.items()
        if k not in ("meta_data", "file_meta_data", "meta_info")
    )
    theta = content["theta_central"]
    assert isinstance(
        theta, dict
    ), f"theta_central must be a label-keyed dict, got {type(theta)}"
    missing = [lbl for lbl in labels if lbl not in theta]
    if missing:
        raise KeyError(
            f"theta_central pkl {sf_file} missing {len(missing)} labels, "
            f"e.g. {missing[:3]}"
        )
    arr = [float(theta[lbl]) for lbl in labels]
    logger.info(
        f"Loaded in-situ central theta from {sf_file} ({len(labels)} coeffs, "
        f"max|theta|={max(abs(x) for x in arr):.4g})"
    )
    return arr


def _cheb(x, n):
    """First ``n`` Chebyshev polynomials of the first kind, as the C++ helper
    evaluates them."""
    return [np.ones_like(x), x, 2.0 * x * x - 1.0, 4.0 * x**3 - 3.0 * x][:n]


def _xtil(v, lo, hi):
    return 2.0 * (np.clip(v, lo, hi) - lo) / (hi - lo) - 1.0


def _centres(axis):
    edges = np.asarray(axis.edges, dtype=float)
    return 0.5 * (edges[:-1] + edges[1:])


def _sample_points(centres, lo, hi):
    """Bin centres plus the two window edges, and the bin each sample reads.

    eval_leg clamps pt and ut to the Chebyshev window before the transform, so
    the polynomial's value at an edge is its value *everywhere* beyond it.
    Sampling only bin centres therefore leaves the extrapolation unconstrained:
    the effMC centres stop at pt = 57.5 while the window runs to 65, so 7.5 GeV
    at the top -- and the ut tails beyond +70 -- were never seen by the bound.
    That is where the fitted scale factor curves diverge. The edge samples reuse
    the nearest bin's efficiency, which is what the helper would read there.
    """
    pts = np.concatenate([[lo], centres, [hi]])
    idx = np.concatenate([[0], np.arange(centres.size), [centres.size - 1]])
    return pts, idx


def _insitu_cell_blocks(effMC, n_eta, n_coeff_pt, n_coeff_ut, include_edges=True):
    """Yield one (step, eta, charge) block of the effMC grid at a time as
    ``(step, offset, basis, eff)``.

    ``basis`` has shape (n_pt, n_ut, n_coeff) with the product basis flattened
    as the C++ helper flattens it (j = k*n_coeff_ut + m), ``eff`` the matching
    MC efficiencies, and ``offset`` the position of the block's first
    coefficient in the flat parameter layout.

    This is the single source of the flat layout: both the physical-region
    report and the fit penalty bundle are built from it, so they cannot drift
    apart from each other or from muon_efficiencies_insitu.hpp.
    """
    n_c2d = n_coeff_pt * n_coeff_ut
    n_id = n_eta * 2 * n_coeff_pt
    n_hlt = n_eta * 2 * n_c2d
    for step in insitu_eff_steps:
        axes = list(effMC[step].axes)
        values = effMC[step].values()
        pt_c = _centres(axes[1])
        pt_s, pt_i = (
            _sample_points(pt_c, *insitu_pt_range)
            if include_edges
            else (pt_c, np.arange(pt_c.size))
        )
        b_pt = np.stack(_cheb(_xtil(pt_s, *insitu_pt_range), n_coeff_pt), -1)
        if step == "idip":
            charges = (-1, 1)
            basis_full = b_pt[:, None, :]
            ut_i = np.array([0])
        else:
            ut_c = _centres(axes[3] if step == "trigger" else axes[2])
            ut_s, ut_i = (
                _sample_points(ut_c, *insitu_ut_range)
                if include_edges
                else (ut_c, np.arange(ut_c.size))
            )
            b_ut = np.stack(_cheb(_xtil(ut_s, *insitu_ut_range), n_coeff_ut), -1)
            charges = (-1, 1) if step == "trigger" else (0,)
            basis_full = (b_pt[:, None, :, None] * b_ut[None, :, None, :]).reshape(
                b_pt.shape[0], b_ut.shape[0], n_c2d
            )
        for i_eta in range(n_eta):
            for charge in charges:
                qbit = 1 if charge > 0 else 0
                if step == "idip":
                    offset = (qbit * n_eta + i_eta) * n_coeff_pt
                    eff = np.asarray(values[i_eta, :, qbit], dtype=float)[pt_i]
                    eff = eff[:, None]
                else:
                    offset = (
                        n_id + (qbit * n_eta + i_eta) * n_c2d
                        if step == "trigger"
                        else n_id + n_hlt + i_eta * n_c2d
                    )
                    eff = np.asarray(
                        (
                            values[i_eta, :, qbit, :]
                            if step == "trigger"
                            else values[i_eta, :, :]
                        ),
                        dtype=float,
                    )[np.ix_(pt_i, ut_i)]
                yield step, offset, basis_full, eff


def report_capped_cells(
    effMC,
    theta_central,
    n_eta,
    n_coeff_pt=insitu_n_coeff_pt,
    n_coeff_ut=insitu_n_coeff_ut,
    effMC_max=insitu_effMC_max,
    effData_max=insitu_effData_max,
):
    """Log the effMC cells where the linearisation point leaves the physical
    region, i.e. where the fitted eMC*(1+P) exceeds ``effData_max`` and the
    helper caps 1+P back onto the boundary.

    Capping silently would hide a fit that has walked outside the physical
    region -- which is exactly what the unconstrained coefficients do in the
    corners of the pt/ut window -- so the count and the smallest resulting fail
    factor are reported per step. A large count, or a fail factor near zero,
    means the SFs should not be trusted there.

    Mirrors the C++ ``fpass_capped`` on the effMC bin centres; returns the total
    number of capped cells.
    """
    theta = np.asarray(theta_central, dtype=float)
    per_step = {}
    for step, offset, basis, eff in _insitu_cell_blocks(
        effMC, n_eta, n_coeff_pt, n_coeff_ut
    ):
        pol = basis @ theta[offset : offset + basis.shape[-1]]
        live = eff > 0.0
        eff = np.where(eff > effMC_max, effMC_max, eff)
        capped = live & (eff * (1.0 + pol) > effData_max)
        n_capped, worst = per_step.get(step, (0, 1.0))
        if capped.any():
            fail = (1.0 - effData_max) / (1.0 - eff[capped])
            worst = min(worst, float(fail.min()))
        per_step[step] = (n_capped + int(capped.sum()), worst)

    total = 0
    for step in insitu_eff_steps:
        n_capped, worst = per_step.get(step, (0, 1.0))
        total += n_capped
        if n_capped:
            logger.warning(
                f"In-situ {step}: capped {n_capped} effMC cells where the "
                f"linearisation point implies a data efficiency above "
                f"{effData_max} (smallest resulting fail factor {worst:.3g})"
            )
    if total == 0:
        logger.info("In-situ linearisation point is physical in every effMC cell")
    return total


def build_insitu_bound_aux(
    effMC,
    n_eta=None,
    n_coeff_pt=insitu_n_coeff_pt,
    n_coeff_ut=insitu_n_coeff_ut,
    effMC_max=insitu_effMC_max,
    delta=insitu_delta,
    include_edges=True,
    theta_central=None,
):
    """Arrays for rabbit's ``InSituEfficiencyBound`` penalty, one row per live
    effMC cell.

    Returns ``{"effmc", "basis", "coeff_index", "coeff_scale", "labels"}``: the
    MC efficiency of each cell, the Chebyshev basis evaluated there, the
    position within ``labels`` of each coefficient it multiplies, and the
    conversion from fit parameter to polynomial coefficient. The penalty forms
    ``u = effmc * (1 + coeff_scale * sum(basis * n[coeff_index]))`` and pushes
    back where ``u`` approaches 1, so the fit is discouraged from leaving the
    region where the fail probability is positive.

    ``coeff_scale`` is ``insitu_delta``: the helper folds the variation step
    into the stored response (``exp(delta * dlnW/dtheta_c)``), so the fitted
    nuisance ``n_c`` and the polynomial coefficient differ by that factor. Using
    the nuisance directly evaluates the polynomial 100x too large.

    Empty effMC cells are dropped (they carry no constraint) and blocks
    narrower than the widest one -- idip, which has no ut dependence -- are zero
    padded, so the penalty is a single vectorised expression.
    """
    if n_eta is None:
        n_eta = effMC["idip"].axes[0].size
    labels = insitu_parameter_labels(n_eta, n_coeff_pt, n_coeff_ut)
    width = n_coeff_pt * n_coeff_ut
    effmc, basis_rows, index_rows, offsets = [], [], [], []
    for step, offset, basis, eff in _insitu_cell_blocks(
        effMC, n_eta, n_coeff_pt, n_coeff_ut, include_edges=include_edges
    ):
        live = eff > 0.0
        if not live.any():
            continue
        n_coeff = basis.shape[-1]
        rows = np.broadcast_to(basis, eff.shape + (n_coeff,))[live]
        idx = np.broadcast_to(offset + np.arange(n_coeff), (rows.shape[0], n_coeff))
        if n_coeff < width:  # idip: pad with zero-weighted entries
            pad = width - n_coeff
            rows = np.pad(rows, ((0, 0), (0, pad)))
            idx = np.pad(idx, ((0, 0), (0, pad)))
        effmc.append(np.minimum(eff[live], effMC_max))
        basis_rows.append(rows)
        index_rows.append(idx)
        # P(theta_central) at this cell. Zero on iteration 0; afterwards the
        # histmaker has already reweighted the MC by 1+P(theta_central), so
        # the total scale factor the fit realises is
        #     1 + P(theta_central) + delta*P(n),
        # not 1 + delta*P(n). Without this the bound is imposed at the wrong
        # point -- by up to 15% where the accumulated correction is largest.
        if theta_central is None:
            offsets.append(np.zeros(rows.shape[0]))
        else:
            th = np.asarray(theta_central, dtype=np.float64)
            pol = (rows * th[idx]).sum(-1)
            # Cap exactly as fpass_capped does in the helper: the
            # histmaker applied min(1+P, effData_max/eMC), so where
            # theta_central is unphysical the MC was reweighted by the
            # CAPPED factor. Storing the uncapped P makes the bound
            # disagree with the histograms about where theta=0 is, and the
            # fit then starts in a region the model does not represent.
            # With the external T&P scale factors that left 130 cells at
            # u>=1 and the fit stalled with a negative fail probability.
            e = np.minimum(eff[live], effMC_max)
            offsets.append(np.minimum(1.0 + pol, effMC_max / e) - 1.0)

    out = {
        "effmc": np.concatenate(effmc).astype(np.float64),
        "basis": np.concatenate(basis_rows).astype(np.float64),
        "coeff_index": np.concatenate(index_rows).astype(np.int64),
        "offset": np.concatenate(offsets).astype(np.float64),
        "coeff_scale": np.array([delta], dtype=np.float64),
        "labels": labels,
    }
    logger.info(
        f"In-situ efficiency bound: {out['effmc'].size} live cells over "
        f"{len(labels)} coefficients"
    )
    return out


def merge_insitu_bound_aux(bundles):
    """Stack per-analysis bound bundles into one.

    A combined fit shares the coefficients between analyses but each carries its
    own MC efficiency -- the W single-muon effMC and the Z tag-and-probe effMC --
    and the scale factors have to stay physical against both, since both
    evaluate fail factors (the W through its non-isolated sideband). Every cell
    is kept, so the binding constraint per coefficient is whichever analysis has
    the largest eMC there.

    The coefficient layout must be identical across bundles; that is what makes
    the same nuisance mean the same polynomial in a combined fit.
    """
    bundles = list(bundles)
    if len(bundles) == 1:
        return bundles[0]
    first = bundles[0]
    for other in bundles[1:]:
        if list(other["labels"]) != list(first["labels"]):
            raise ValueError(
                "in-situ bound bundles have different coefficient layouts; "
                "they cannot share nuisances in a combined fit"
            )
        if other["coeff_scale"][0] != first["coeff_scale"][0]:
            raise ValueError("in-situ bound bundles disagree on coeff_scale")
    merged = {
        "effmc": np.concatenate([b["effmc"] for b in bundles]),
        "basis": np.concatenate([b["basis"] for b in bundles]),
        "coeff_index": np.concatenate([b["coeff_index"] for b in bundles]),
        "offset": np.concatenate([b["offset"] for b in bundles]),
        "coeff_scale": first["coeff_scale"],
        "labels": first["labels"],
    }
    logger.info(
        f"In-situ efficiency bound: merged {len(bundles)} efficiency grids into "
        f"{merged['effmc'].size} cells"
    )
    return merged


def make_muon_insitu_efficiency_helper(
    effMCFile,
    n_coeff_pt=insitu_n_coeff_pt,
    n_coeff_ut=insitu_n_coeff_ut,
    delta=insitu_delta,
    effMC_max=insitu_effMC_max,
    effData_max=insitu_effData_max,
    single_leg=False,
    central_sf_file=None,
):
    """Build the RDF helpers for the in-situ efficiency method.

    The Chebyshev x̃ windows are deliberately not parameters: they come from the
    module-level insitu_pt_range / insitu_ut_range so every analysis shares the
    same transform (required to correlate the coefficient nuisances in a
    combined fit).

    The number of eta bins is inferred from the effMC histograms so it follows
    the histmaker --eta binning automatically. Returns
    ``(tensor_helper, central_helper, labels)``:
      - ``tensor_helper`` emits the per-event 2112-bin variation tensor,
      - ``central_helper`` returns the scalar central reweight W(theta_central)
        to fold into nominal_weight before the templates are filled.

    ``single_leg=True`` selects the W single-muon helpers (no tag leg); the
    default two-leg helpers are used for the Z dilepton tag-and-probe topology.
    A single effMC triplet is used (the passing SF is 1+P and is MC-free, so the
    always-passing dilepton tag leg needs no effMC; effMC enters only the fail
    factor). Each analysis passes its own effMC: W for the single-muon
    histmaker, Z tag-and-probe for the dilepton.
    ``central_sf_file`` is the accumulated theta_central pkl (None -> zeros, i.e.
    iteration 0, SF = 1, == previous behaviour with W(theta)=1 and the gradient
    evaluated at SF=1).
    """
    effMC = make_muon_insitu_effMC_helper(effMCFile)
    h_idip = effMC["idip"]
    h_trig = effMC["trigger"]
    h_iso = effMC["iso"]
    n_eta = h_idip.axes[0].size
    assert h_trig.axes[0].size == n_eta and h_iso.axes[0].size == n_eta, (
        n_eta,
        h_trig.axes[0].size,
        h_iso.axes[0].size,
    )

    theta_central = load_insitu_central(central_sf_file, n_eta, n_coeff_pt, n_coeff_ut)
    if central_sf_file is not None:
        report_capped_cells(
            effMC,
            theta_central,
            n_eta,
            n_coeff_pt,
            n_coeff_ut,
            effMC_max,
            effData_max,
        )
    theta_vec = ROOT.std.vector("double")(theta_central)

    def _instantiate(helper_class):
        # fresh pyroot copies per helper (the boost hists are std::move'd in)
        ip = narf.hist_to_pyroot_boost(h_idip)
        tp = narf.hist_to_pyroot_boost(h_trig)
        sp = narf.hist_to_pyroot_boost(h_iso)
        return helper_class[
            n_eta, n_coeff_pt, n_coeff_ut, type(ip), type(tp), type(sp)
        ](
            ROOT.std.move(ip),
            ROOT.std.move(tp),
            ROOT.std.move(sp),
            insitu_pt_range[0],
            insitu_pt_range[1],
            insitu_ut_range[0],
            insitu_ut_range[1],
            delta,
            effMC_max,
            effData_max,
            theta_vec,
        )

    tensor_class = (
        ROOT.wrem.muon_insitu_efficiency_helper_singleleg
        if single_leg
        else ROOT.wrem.muon_insitu_efficiency_helper
    )
    central_class = (
        ROOT.wrem.muon_insitu_central_weight_helper_singleleg
        if single_leg
        else ROOT.wrem.muon_insitu_central_weight_helper
    )
    helper = _instantiate(tensor_class)
    central_helper = _instantiate(central_class)

    n_id = n_eta * 2 * n_coeff_pt
    n_hlt = n_eta * 2 * n_coeff_pt * n_coeff_ut
    n_iso_p = n_eta * n_coeff_pt * n_coeff_ut
    n_sf = n_id + n_hlt + n_iso_p
    axis_insitu = hist.axis.Integer(
        0, n_sf, underflow=False, overflow=False, name="insituEffParm"
    )
    helper.tensor_axes = [axis_insitu]

    labels = insitu_parameter_labels(n_eta, n_coeff_pt, n_coeff_ut)
    assert len(labels) == n_sf, (len(labels), n_sf)
    logger.info(
        f"Built in-situ efficiency helper with {n_sf} unconstrained "
        f"Chebyshev coefficients "
        f"(eta={n_eta}, order_pt={n_coeff_pt - 1}, order_ut={n_coeff_ut - 1}; "
        f"idip={n_id}, trigger={n_hlt}, iso={n_iso_p}); "
        f"x̃ windows pt={insitu_pt_range}, ut={insitu_ut_range}; "
        f"central theta {'from ' + central_sf_file if central_sf_file else '= 0'}"
    )
    return helper, central_helper, labels


def setup_muon_insitu_helpers(
    effMC_file, central_sf_file, make_effMC, single_leg=False
):
    """One-stop in-situ setup shared by the histmakers: validate the CLI
    combination and build ``(tensor_helper, central_helper, labels)``, or
    ``(None, None, None)`` when the method is not enabled (no effMC file —
    e.g. the --makeInsituEffMC pre-step that produces it).

    The in-situ central reweight folds W(theta_central) into nominal_weight,
    which also feeds the effMCprobe_* spectra. effMC must stay
    iteration-independent, so the effMC pre-step and an iteration reweight
    cannot run together.
    """
    if make_effMC and central_sf_file is not None:
        raise ValueError(
            "--makeInsituEffMC and --insituSFFile are mutually exclusive: the "
            "central reweight would distort the (iteration-independent) effMC. "
            "Compute effMC at iteration 0 (no --insituSFFile), then reuse it "
            "for all iterations."
        )
    if effMC_file is None:
        return None, None, None
    helper, central_helper, labels = make_muon_insitu_efficiency_helper(
        effMC_file,
        single_leg=single_leg,
        central_sf_file=central_sf_file,
    )
    logger.info(f"In-situ effMC file: {effMC_file}")
    if central_sf_file is not None:
        logger.info(f"In-situ central SF file (iteration): {central_sf_file}")
    return helper, central_helper, labels
