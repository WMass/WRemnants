import argparse
import os

import hist
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from numpy.polynomial import chebyshev as npcheb
from scipy.optimize import least_squares
from scipy.special import expit
from scipy.special import logit as logit_fn

from wremnants.postprocessing.datagroups.datagroups import Datagroups
from wremnants.production.muon_efficiencies_insitu import (
    insitu_effMC_max,
    insitu_n_coeff_pt,
    insitu_n_coeff_ut,
    insitu_pt_range,
    insitu_ut_range,
)
from wremnants.utilities import common
from wremnants.utilities.io_tools import base_io
from wums import logging, output_tools, plot_tools

hep.style.use(hep.style.ROOT)

# Produce the MC efficiency (effMC) input for the in-situ muon efficiency
# method from the 4-category probe spectra emitted by mz_dilepton.py
# (effMCprobe_{nominal,failIso,failHLT,failID}, MC only).
#
#   nominal : 2HLT, probe passes ID & HLT & Iso
#   failIso : 2HLT, probe passes ID & HLT, fails Iso
#   failHLT : 1HLT, probe passes ID, fails HLT
#   failID  : 1HLT, probe fails ID
#
# Per-step MC efficiency (raw per-bin), computed independently for each listed
# process from its own probe spectra:
#   eff_iso     =  nominal                       / (nominal + failIso)
#   eff_trigger = (nominal + failIso)            / (nominal + failIso + failHLT)
#   eff_idip    = (nominal + failIso + failHLT)  / (nominal + failIso + failHLT
#                                                              + failID)
#
# Differential structure (matches the SMP-23-002 parameterisation variables):
#   idip   : (eta, pt, charge)        -- charge-dependent, no uT
#   trigger: (eta, pt, charge, uT)    -- charge-dependent, 2D (pt, uT)
#   iso    : (eta, pt, uT)            -- charge-inclusive, 2D (pt, uT)
#
# One output pkl is written per process: insitu_effMC_<process>[_<postfix>].pkl.lz4.
# Diagnostic plots overlay all listed processes for a side-by-side comparison.
# Multiple input files may be given (e.g. the dilepton output for Zmumu and the
# single-muon output for Wmunu); each process is taken from the first input
# that provides its probe spectra. The pT binning may differ between inputs
# (axes are kept per process), but eta/charge/genUT must match.


# Color and label palette for the diagnostic-plot overlay across processes.
PROCESS_COLORS = {
    "Zmumu": "tab:blue",
    "Ztautau": "tab:red",
    "Wmunu": "tab:green",
    "Wtaunu": "tab:purple",
    "Other": "tab:gray",
    "PhotonInduced": "tab:orange",
    "Diboson": "tab:cyan",
    "Top": "tab:brown",
}
PROCESS_LABELS = {
    "Zmumu": r"Z $\to \mu\mu$",
    "Ztautau": r"Z $\to \tau\tau$",
    "Wmunu": r"W $\to \mu\nu$",
    "Wtaunu": r"W $\to \tau\nu$",
}

parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--inputFile",
    type=str,
    nargs="+",
    required=True,
    help="Histmaker output file(s) with the effMCprobe spectra. Each requested "
    "process is read from the first file that provides it.",
)
parser.add_argument("--outpath", type=str, default="./")
parser.add_argument(
    "-p", "--postfix", type=str, help="Postfix for output file names", default=None
)
parser.add_argument("--plotdir", type=str, help="Output directory for plots")
parser.add_argument(
    "--eoscp",
    action="store_true",
    help="Copy folder to eos with xrdcp rather than using the mount",
)
parser.add_argument(
    "--process",
    type=str,
    nargs="+",
    default=["Zmumu"],
    help="Process group(s) to measure effMC for. One output pkl per entry; "
    "the diagnostic plots overlay all listed entries. Each must be a "
    "datagroup present in --inputFile (e.g. Zmumu, Ztautau). An entry "
    "'<group>:unbiased' makes an additional curve/pkl from the tag-unbiased "
    "probe spectra of the same group, so tag-based and tag-unbiased can be "
    "overlaid in one go (e.g. --process Zmumu Zmumu:unbiased Wmunu).",
)
parser.add_argument(
    "--tagAndProbeDir",
    type=str,
    default=None,
    help="Directory with the external tag-and-probe efficiency files "
    "(allEfficiencies_2D_<step>_<charge>.root, e.g. "
    "wremnants-data/data/muonSF/tagAndProbe/2016). When set, the per-cell "
    "eff-vs-pT plots overlay the T&P EffMC2D and EffData2D as data points "
    "with stat error bars.",
)
parser.add_argument(
    "--unbiased",
    action="store_true",
    help="Use the tag-unbiased probe spectra "
    "(effMCprobeUnbiased_{cat}_{minus,plus}, summed over the two charge legs) "
    "instead of the tag-based effMCprobe_*. Processes whose input lacks them "
    "fall back to effMCprobe_* with an info message (the single-muon histmaker "
    "output is tag-unbiased by construction).",
)
parser.add_argument(
    "--smooth",
    action="store_true",
    help="Smooth the per-step effMC with the same logit-Chebyshev form as the "
    "in-situ SF parameterisation: logit(e) = sum_k c_k T_k(x_pt) [x T_m(x_ut)], "
    "with the shared clamped x-tilde windows, fitted per (eta[, charge]) cell to "
    "the binned efficiencies (variance-weighted least squares). The pkl then "
    "carries the smoothed effMC (what the in-situ helper consumes) plus the raw "
    "one as effMC_raw; the per-cell plots overlay the smooth curves.",
)
parser.add_argument(
    "--smoothCoeffPt",
    type=int,
    default=insitu_n_coeff_pt,
    help="Number of Chebyshev coefficients in pt for --smooth (default: same "
    "order as the in-situ SF parameterisation).",
)
parser.add_argument(
    "--smoothCoeffUt",
    type=int,
    default=insitu_n_coeff_ut,
    help="Number of Chebyshev coefficients in uT for --smooth (trigger/iso; "
    "default: same order as the in-situ SF parameterisation).",
)
parser.add_argument(
    "--plotPtRange",
    type=float,
    nargs=2,
    default=[26.0, 65.0],
    help="Displayed pt-axis range of every plot with a pt axis, so productions "
    "with different upper pt cuts (e.g. 26-70 vs 26-90) are directly "
    "comparable. The underlying histograms, fits and pkl outputs keep the "
    "full effMC range.",
)
parser.add_argument("--debug", action="store_true", help="Print debug output")
args = parser.parse_args()

logger = logging.setup_logger("make_insitu_effMC", 4 if args.debug else 3)

categories = ["nominal", "failIso", "failHLT", "failID"]
steps = ["idip", "trigger", "iso"]


def _parse_process_entry(entry):
    """--process entries are '<group>' or '<group>:unbiased'. The latter makes
    a separate curve/pkl from the tag-unbiased probe spectra of the same
    datagroup, so biased and unbiased can be overlaid in one invocation
    (e.g. --process Zmumu Zmumu:unbiased Wmunu)."""
    group, _, variant = entry.partition(":")
    if variant not in ("", "unbiased"):
        raise ValueError(
            f"Unknown variant '{variant}' in process entry '{entry}' "
            "(supported: '<group>' or '<group>:unbiased')"
        )
    return group, variant == "unbiased"


# Both the dilepton (z_dilepton) and single-muon (w_mass) histmakers emit the
# same effMCprobe_{nominal,failIso,failHLT,failID} probe spectra (eta, pt,
# charge, genUT), so the per-step recombination below is identical for both.
supported_modes = ("z_dilepton", "w_mass")

# Map each requested process to the first input file that provides its probe
# spectra (group present with members). Each input gets its own Datagroups, so
# processes from different histmakers (e.g. Zmumu from the dilepton output,
# Wmunu from the single-muon output) can be overlaid in one go.
process_source = {}  # process -> (Datagroups, input file path)
probe_names = {}  # process -> {category: [hist names whose sum is the spectrum]}


def _probe_hist_names(datagroups, process, cat):
    """Histogram name(s) whose sum gives the probe spectrum of `cat`."""
    group, unbiased = _parse_process_entry(process)
    if args.unbiased or unbiased:
        names = [
            f"effMCprobeUnbiased_{cat}_minus",
            f"effMCprobeUnbiased_{cat}_plus",
        ]
        member = datagroups.groups[group].members[0].name
        if all(n in datagroups.results[member]["output"] for n in names):
            return names
        if cat == categories[0]:
            logger.info(
                f"{process}: no tag-unbiased probe spectra in input, falling "
                "back to effMCprobe_* (already tag-unbiased for the "
                "single-muon histmaker output)"
            )
    return [f"effMCprobe_{cat}"]


for input_file in args.inputFile:
    datagroups = Datagroups(input_file)
    if datagroups.mode not in supported_modes:
        raise ValueError(
            f"Input mode '{datagroups.mode}' of {input_file} not supported; "
            f"expected the output of the dilepton or single-muon histmaker "
            f"(one of {supported_modes})"
        )
    wanted = [
        p
        for p in args.process
        if p not in process_source
        and _parse_process_entry(p)[0] in datagroups.groups
        and datagroups.groups[_parse_process_entry(p)[0]].members
    ]
    if not wanted:
        logger.warning(f"No requested process provided by {input_file}; skipping")
        continue
    for p in wanted:
        group = _parse_process_entry(p)[0]
        probe_names[p] = {
            cat: _probe_hist_names(datagroups, p, cat) for cat in categories
        }
        for names in probe_names[p].values():
            for name in names:
                datagroups.loadHistsForDatagroups(name, syst="", procsToRead=[group])
        process_source[p] = (datagroups, input_file)
        logger.info(f"Reading process '{p}' from {input_file}")


def _ratio(num, den):
    out = np.zeros_like(num, dtype=float)
    np.divide(num, den, out=out, where=den > 0)
    return out


def _safe_ratio_1d(num_1d, den_1d):
    out = np.zeros_like(num_1d, dtype=float)
    np.divide(num_1d, den_1d, out=out, where=den_1d > 0)
    return out


def _eff_and_var(num, den, var_num, var_den):
    """Efficiency = num/den with the binomial-like variance for the case where
    the numerator events are a subset of the denominator (so
    Cov(num, den) = Var(num)). Propagation:

        Var(eff) = (Var(num)*(1-eff)^2 + eff^2 * Var(fail)) / den^2

    with Var(fail) = Var(den) - Var(num) >= 0. For unweighted Poisson counts
    this reduces to the familiar eff*(1-eff)/den. Inputs are arbitrary-shape
    numpy arrays; outputs match. Bins with den<=0 get eff=var=0.
    """
    eff = _ratio(num, den)
    var_fail = np.maximum(var_den - var_num, 0.0)
    var_eff = np.zeros_like(eff, dtype=float)
    safe = den > 0
    var_eff[safe] = (
        var_num[safe] * (1.0 - eff[safe]) ** 2 + eff[safe] ** 2 * var_fail[safe]
    ) / den[safe] ** 2
    return eff, var_eff


def _ylim_from_arr(arr, var=None, margin_up=0.03, margin_down=0.01):
    """Adaptive y-axis range for an efficiency array.

    ymin = max(0, min(eff - err) - margin) over positive cells   (lower bound
                                                                  clipped at 0)
    ymax = 1.0 + margin                                          (hard cap, the
                                                                  physical
                                                                  efficiency
                                                                  ceiling)

    Bins above 1 from negative MC weights or large stat error bars are clipped
    by matplotlib at ymax; the dashed unity line at y=1 is always inside the
    visible range. Fallback (no positive entries): (0, 1+margin).
    """
    ymax_cap = 1.0 + margin_up
    positive = arr > 0
    if not positive.any():
        return 0.0, ymax_cap
    if var is None:
        lo = arr[positive].min()
    else:
        err = np.sqrt(np.maximum(var, 0.0))
        lo = float((arr - err)[positive].min())
    return max(0.0, float(lo) - margin_down), ymax_cap


def _draw_unity_line(ax):
    """Dashed grey horizontal line at y=1 to mark the physical efficiency
    ceiling. No legend entry."""
    ax.axhline(1.0, linestyle="--", color="gray", linewidth=0.8, zorder=0)


def compute_effMC(process):
    """Build the per-step effMC histograms and the intermediate count arrays
    needed by the diagnostic projections, for one process group.

    Returns a dict containing:
        axes      : (eta, pt, charge, genUT) input boost-hist axes
        n         : raw counts per category (eta, pt, charge, uT)
        pass2HLT, passID, allProbes : combined sums used by the ratios
        effMC     : {step: hist.Hist}  -- the per-step boost hists for the pkl
        eff_etapt : {step: ndarray}    -- uT-integrated eff for per-cell plots
                    shapes: (eta, pt, charge) for idip/trigger; (eta, pt) for iso
    """
    src_dg, _ = process_source[process]
    group = _parse_process_entry(process)[0]

    def _probe_hist(cat):
        # sum of the configured spectra (one tag-based hist, or the two
        # charge legs of the tag-unbiased spectra)
        hists = [src_dg.groups[group].hists[n] for n in probe_names[process][cat]]
        out = hists[0]
        for x in hists[1:]:
            out = out + x
        return out

    h = {cat: _probe_hist(cat) for cat in categories}

    ref_axes = list(h["nominal"].axes)
    axis_names = tuple(a.name for a in ref_axes)
    expected = ("eta", "pt", "charge", "genUT")
    if axis_names != expected:
        raise ValueError(
            f"effMCprobe[{process}] axes {axis_names} != expected {expected}; "
            "histmaker has the wrong cols/axes ordering"
        )
    for cat in categories:
        cat_names = tuple(a.name for a in h[cat].axes)
        if cat_names != axis_names:
            raise ValueError(
                f"effMCprobe_{cat} axes for {process} {cat_names} != reference {axis_names}"
            )

    eta_ax, pt_ax, charge_ax, ut_ax = ref_axes

    # raw counts (sumw) and sumw2 per category, used both for the central
    # ratios and for the binomial-like variance propagation that drives the
    # plot error bars.
    n = {cat: h[cat].values(flow=False) for cat in categories}
    var_n = {cat: h[cat].variances(flow=False) for cat in categories}
    pass2HLT = n["nominal"] + n["failIso"]
    var_pass2HLT = var_n["nominal"] + var_n["failIso"]

    # Dilepton tag-and-probe counting: a 2HLT event provides TWO valid
    # (tag, probe) pairs (either leg can be the probe; both pass ID & trigger),
    # while a 1HLT event provides a single valid probe. The histmaker already
    # fills BOTH legs of each 2HLT event into the nominal/failIso histograms
    # (each at its own kinematics), so the category sums here are the exact
    # probe counts and need no double-counting factor:
    #   - trigger: num = pass2HLT, den = pass2HLT + failHLT
    #   - idip:    num = pass2HLT + failHLT (probes passing ID),
    #              den = num + failID
    #   - iso:     num = nominal, den = nominal + failIso (both 2HLT)
    # The two 2HLT legs always have opposite charge, so they never share a bin
    # -> each bin sees at most one leg per event -> clean Poisson sumw2.

    # Per-step (num, den) sumw and sumw2 in their natural shapes:
    #   idip/trigger -> (eta, pt, charge, uT);  iso -> (eta, pt, uT)
    step_num, step_den, step_var_num, step_var_den = {}, {}, {}, {}

    step_num["trigger"] = pass2HLT
    step_den["trigger"] = pass2HLT + n["failHLT"]
    step_var_num["trigger"] = var_pass2HLT
    step_var_den["trigger"] = var_pass2HLT + var_n["failHLT"]

    passID = pass2HLT + n["failHLT"]
    var_passID = var_pass2HLT + var_n["failHLT"]
    step_num["idip"] = passID
    step_den["idip"] = passID + n["failID"]
    step_var_num["idip"] = var_passID
    step_var_den["idip"] = var_passID + var_n["failID"]

    step_num["iso"] = n["nominal"].sum(axis=2)  # charge-inclusive
    step_den["iso"] = pass2HLT.sum(axis=2)
    step_var_num["iso"] = var_n["nominal"].sum(axis=2)
    step_var_den["iso"] = var_pass2HLT.sum(axis=2)

    # idip: uT and charge dependence collapsed -> (eta, pt, charge)
    eff_idip = _ratio(step_num["idip"].sum(axis=-1), step_den["idip"].sum(axis=-1))
    h_idip = hist.Hist(
        eta_ax, pt_ax, charge_ax, name="effMC_idip", storage=hist.storage.Double()
    )
    h_idip.view(flow=False)[...] = eff_idip

    # trigger: full 4D (eta, pt, charge, uT)
    eff_trig = _ratio(step_num["trigger"], step_den["trigger"])
    h_trig = hist.Hist(
        eta_ax,
        pt_ax,
        charge_ax,
        ut_ax,
        name="effMC_trigger",
        storage=hist.storage.Double(),
    )
    h_trig.view(flow=False)[...] = eff_trig

    # iso: charge-inclusive (eta, pt, uT)
    eff_iso = _ratio(step_num["iso"], step_den["iso"])
    h_iso = hist.Hist(
        eta_ax, pt_ax, ut_ax, name="effMC_iso", storage=hist.storage.Double()
    )
    h_iso.view(flow=False)[...] = eff_iso

    # uT-integrated eff per cell with its propagated variance, used by the
    # per-cell pt curves with error bars.
    eff_etapt, var_etapt = {}, {}
    for step in steps:
        eff_etapt[step], var_etapt[step] = _eff_and_var(
            step_num[step].sum(axis=-1),
            step_den[step].sum(axis=-1),
            step_var_num[step].sum(axis=-1),
            step_var_den[step].sum(axis=-1),
        )

    return {
        "axes": ref_axes,
        "n": n,
        "var_n": var_n,
        "step_num": step_num,
        "step_den": step_den,
        "step_var_num": step_var_num,
        "step_var_den": step_var_den,
        "effMC": {"idip": h_idip, "trigger": h_trig, "iso": h_iso},
        "eff_etapt": eff_etapt,
        "var_etapt": var_etapt,
    }


def _cheb_basis(x, n):
    """Chebyshev basis [T_0(x), ..., T_{n-1}(x)] at array x (the convention of
    the in-situ SF parameterisation, cf. plot_insitu_effSF.py)."""
    return np.array([npcheb.chebval(x, [0] * i + [1]) for i in range(n)])


def _cheb_x(v, lo, hi):
    """Clamped x-tilde transform of the in-situ Chebyshev windows."""
    return 2.0 * (np.clip(v, lo, hi) - lo) / (hi - lo) - 1.0


def _fit_logit_cheb(y, sig, n_eff, valid, B):
    """Variance-weighted least-squares fit of expit(c @ B) to the per-bin
    efficiencies y. ``valid`` masks fitted bins (den > 0). The per-bin sigma is
    floored at the boundary-aware binomial scale 1/(n_eff + 2), with n_eff =
    den^2/var_den effective counts, so eff >= 1 bins -- whose naive binomial
    variance is zero -- enter with a finite, statistically sensible weight
    instead of dominating the fit. Returns (coeffs, chi2, ndf), or (None, 0, 0)
    when the cell has fewer valid bins than coefficients or the fit fails."""
    n_coeff = B.shape[0]
    if int(valid.sum()) < n_coeff:
        return None, 0.0, 0
    sig_floor = 1.0 / (n_eff[valid] + 2.0)
    w = 1.0 / np.maximum(np.maximum(sig[valid], sig_floor), 1e-6)
    Bm, ym = B[:, valid], y[valid]
    e0 = np.clip(np.average(ym, weights=w**2), 1e-3, 1.0 - 1e-3)
    c0 = np.zeros(n_coeff)
    c0[0] = logit_fn(e0)
    res = least_squares(lambda c: (expit(c @ Bm) - ym) * w, c0)
    if not res.success:
        return None, 0.0, 0
    return res.x, float(2.0 * res.cost), max(int(valid.sum()) - n_coeff, 1)


def _smooth_effMC(process):
    """Replace the per-step effMC of one process by logit-Chebyshev fits: the
    same functional form, clamped x-tilde windows and Chebyshev convention as
    the in-situ SF parameterisation (orders from --smoothCoeffPt/Ut, coefficient
    layout pt-major like the SF nuisances). Fitted per (eta, charge) cell for
    idip/trigger and per eta for iso, at the bin centers (matching the per-event
    bin lookup of the helper). Cells with too few valid bins or a failed fit
    keep the raw values (logged). Bins that are empty in the raw effMC get the
    model value (the fit covers the full grid). Modifies process_data[process]:
        effMC          smoothed hists (drop-in for the pkl / in-situ helper)
        effMC_raw      the raw hists
        smooth_coeffs  {step: {(i_eta, j_q): coeffs}}  (j_q None for iso)
    """
    pd = process_data[process]
    eta_ax, pt_ax, charge_ax, ut_ax = pd["axes"]
    nk, nm = args.smoothCoeffPt, args.smoothCoeffUt
    npt, nut = pt_ax.size, ut_ax.size
    Tp = _cheb_basis(_cheb_x(pt_ax.centers, *insitu_pt_range), nk)  # (nk, npt)
    Tu = _cheb_basis(_cheb_x(ut_ax.centers, *insitu_ut_range), nm)  # (nm, nut)
    # flattened (pt, ut) tensor-product basis, C-order matching .reshape(-1)
    B2 = (Tp[:, None, :, None] * Tu[None, :, None, :]).reshape(nk * nm, npt * nut)

    pd["effMC_raw"] = pd["effMC"]
    pd["smooth_coeffs"] = {}
    smoothed = {}
    for step in steps:
        num, den = pd["step_num"][step], pd["step_den"][step]
        vn, vd = pd["step_var_num"][step], pd["step_var_den"][step]
        if step == "idip":
            # the idip effMC is uT-collapsed -> (eta, pt, charge)
            num, den, vn, vd = (a.sum(axis=-1) for a in (num, den, vn, vd))
        eff, var = _eff_and_var(num, den, vn, vd)
        sig = np.sqrt(np.maximum(var, 0.0))
        # effective (unweighted-equivalent) counts per bin, for the boundary-
        # aware sigma floor in the fit
        neff = np.where(vd > 0, den**2 / np.maximum(vd, 1e-12), 0.0)

        h_new = pd["effMC_raw"][step].copy()
        view = h_new.view(flow=False)
        coeffs = {}
        chi2_ndf = []
        n_fallback = 0
        charge_iter = range(charge_ax.size) if step != "iso" else [None]
        for i_eta in range(eta_ax.size):
            for j_q in charge_iter:
                if step == "idip":
                    y, s = eff[i_eta, :, j_q], sig[i_eta, :, j_q]
                    d, ne = den[i_eta, :, j_q], neff[i_eta, :, j_q]
                    B = Tp
                elif step == "trigger":
                    y = eff[i_eta, :, j_q, :].reshape(-1)
                    s = sig[i_eta, :, j_q, :].reshape(-1)
                    d = den[i_eta, :, j_q, :].reshape(-1)
                    ne = neff[i_eta, :, j_q, :].reshape(-1)
                    B = B2
                else:
                    y = eff[i_eta].reshape(-1)
                    s = sig[i_eta].reshape(-1)
                    d = den[i_eta].reshape(-1)
                    ne = neff[i_eta].reshape(-1)
                    B = B2
                c, chi2, ndf = _fit_logit_cheb(y, s, ne, d > 0, B)
                if c is None:
                    n_fallback += 1
                    continue  # this cell keeps the raw values
                sm = np.clip(expit(c @ B), 1e-6, insitu_effMC_max)
                if step == "idip":
                    view[i_eta, :, j_q] = sm
                elif step == "trigger":
                    view[i_eta, :, j_q, :] = sm.reshape(npt, nut)
                else:
                    view[i_eta] = sm.reshape(npt, nut)
                coeffs[(i_eta, j_q)] = c
                chi2_ndf.append(chi2 / ndf)
        smoothed[step] = h_new
        pd["smooth_coeffs"][step] = coeffs
        msg = f"smooth effMC[{process}][{step}]: {len(coeffs)} cells fitted"
        if chi2_ndf:
            arr = np.array(chi2_ndf)
            msg += f" (chi2/ndf mean {arr.mean():.2f}, max {arr.max():.2f})"
        if n_fallback:
            msg += f"; {n_fallback} cells kept raw (too few bins or failed fit)"
        logger.info(msg)
    pd["effMC"] = smoothed


def _smooth_eval_cell(process, step, i_eta, j_q, pt_vals, j_ut=None):
    """Smoothed effMC curve of one fitted cell evaluated at pt_vals.
    idip: e(pt). trigger/iso with j_ut: e(pt) at that uT bin center. trigger/iso
    without j_ut: the den-weighted uT average at the pt BIN of each pt value
    (matching how the raw uT-integrated points are built). None if the cell has
    no fit."""
    pd = process_data[process]
    c = pd.get("smooth_coeffs", {}).get(step, {}).get((i_eta, j_q))
    if c is None:
        return None
    nk, nm = args.smoothCoeffPt, args.smoothCoeffUt
    pt_vals = np.asarray(pt_vals, dtype=float)
    Tp = _cheb_basis(_cheb_x(pt_vals, *insitu_pt_range), nk)  # (nk, npts)
    if step == "idip":
        return expit(c @ Tp)
    _, pt_ax_p, _, ut_ax_p = pd["axes"]
    if j_ut is not None:
        Tu1 = _cheb_basis(
            _cheb_x(np.array([ut_ax_p.centers[j_ut]]), *insitu_ut_range), nm
        )[:, 0]
        B = (Tp[:, None, :] * Tu1[None, :, None]).reshape(nk * nm, pt_vals.size)
        return expit(c @ B)
    Tu = _cheb_basis(_cheb_x(ut_ax_p.centers, *insitu_ut_range), nm)
    B = (Tp[:, None, :, None] * Tu[None, :, None, :]).reshape(
        nk * nm, pt_vals.size * ut_ax_p.size
    )
    e2 = expit(c @ B).reshape(pt_vals.size, ut_ax_p.size)
    den = pd["step_den"][step]
    w_bin = den[i_eta, :, j_q, :] if j_q is not None else den[i_eta]
    ipt = np.clip(
        np.searchsorted(pt_ax_p.edges, pt_vals, side="right") - 1, 0, pt_ax_p.size - 1
    )
    w = w_bin[ipt]  # (npts, nut)
    wsum = w.sum(axis=1)
    return np.where(wsum > 0, (e2 * w).sum(axis=1) / np.maximum(wsum, 1e-9), np.nan)


def _smooth_grid_cell(process, step, i_eta, j_q, pt_vals=None, ut_vals=None):
    """Smoothed effMC of one fitted cell on a (pt_vals x ut_vals) grid (the bin
    centers by default), i.e. expit(c @ B(pt, uT)) without the uT averaging of
    _smooth_eval_cell. Returns ndarray (len(pt_vals), len(ut_vals)) or None if
    the cell has no fit. uT-dependent steps only (idip has no uT)."""
    pd = process_data[process]
    c = pd.get("smooth_coeffs", {}).get(step, {}).get((i_eta, j_q))
    if c is None or step == "idip":
        return None
    nk, nm = args.smoothCoeffPt, args.smoothCoeffUt
    _, pt_ax_p, _, ut_ax_p = pd["axes"]
    pt_vals = pt_ax_p.centers if pt_vals is None else np.asarray(pt_vals, dtype=float)
    ut_vals = ut_ax_p.centers if ut_vals is None else np.asarray(ut_vals, dtype=float)
    Tp = _cheb_basis(_cheb_x(pt_vals, *insitu_pt_range), nk)  # (nk, npt)
    Tu = _cheb_basis(_cheb_x(ut_vals, *insitu_ut_range), nm)  # (nm, nut)
    B = (Tp[:, None, :, None] * Tu[None, :, None, :]).reshape(
        nk * nm, pt_vals.size * ut_vals.size
    )
    return expit(c @ B).reshape(pt_vals.size, ut_vals.size)


# Compute effMC for every requested process
process_data = {}
for process in args.process:
    if process not in process_source:
        logger.warning(
            f"Process group '{process}' not provided by any input file; skipping"
        )
        continue
    process_data[process] = compute_effMC(process)
    if args.smooth:
        _smooth_effMC(process)
    for step, h_out in process_data[process]["effMC"].items():
        e = h_out.view(flow=False)
        inrange = (e > 0) & (e < 1)
        nfilled = int(inrange.sum())
        if nfilled:
            sel = e[inrange]
            logger.info(
                f"effMC[{process}][{step}] shape={e.shape}: {nfilled} bins in (0,1), "
                f"mean={sel.mean():.4f} min={sel.min():.4f} max={sel.max():.4f}; "
                f"{int((e >= 1).sum())} bins >=1 (low stats), "
                f"{int((e <= 0).sum())} empty"
            )
        else:
            logger.warning(
                f"effMC[{process}][{step}] shape={e.shape}: no bins in (0,1) "
                f"({int((e >= 1).sum())} bins >=1, {int((e <= 0).sum())} empty) "
                f"- expected only with very low statistics"
            )

if not process_data:
    raise ValueError(
        f"None of the requested processes {args.process} were found in input"
    )

# Reference axes from the first process. Different inputs may use different pT
# binnings (e.g. single-muon vs dilepton acceptance), so pT is handled per
# process in the plots; eta/charge/genUT must be identical across processes for
# the overlay cells to line up.
ref_axes = process_data[next(iter(process_data))]["axes"]
eta_ax, pt_ax, charge_ax, ut_ax = ref_axes
for process, pd in process_data.items():
    p_eta, p_pt, p_charge, p_ut = pd["axes"]
    for ref_a, p_a in ((eta_ax, p_eta), (charge_ax, p_charge), (ut_ax, p_ut)):
        if not np.array_equal(ref_a.edges, p_a.edges):
            raise ValueError(
                f"Axis '{ref_a.name}' of process '{process}' does not match the "
                f"reference binning; only the pT axis may differ across inputs"
            )


def _proc_axis(process, name):
    """The named input axis (eta/pt/charge/genUT) of one process. Only pT may
    differ across processes; the others are validated identical above."""
    for a in process_data[process]["axes"]:
        if a.name == name:
            return a
    raise KeyError(name)


def _step_num_den(process, step):
    """Numerator/denominator (sumw and sumw2) arrays for one (process, step),
    with the 2HLT double counting already applied (see compute_effMC). Shapes
    (eta, pt, charge, uT) for idip/trigger, (eta, pt, uT) for iso.
    Returns (num, den, var_num, var_den, axis_names, charge_dep).
    """
    pd = process_data[process]
    if step in ("idip", "trigger"):
        axis_names, charge_dep = ("eta", "pt", "charge", "genUT"), True
    elif step == "iso":
        axis_names, charge_dep = ("eta", "pt", "genUT"), False
    else:
        raise KeyError(step)
    return (
        pd["step_num"][step],
        pd["step_den"][step],
        pd["step_var_num"][step],
        pd["step_var_den"][step],
        axis_names,
        charge_dep,
    )


def _step_proj_axes(step):
    """Projection axes to draw per step (IDIP is uT-independent by design)."""
    if step == "idip":
        return ("eta", "pt")
    return ("eta", "pt", "genUT")


def _project_eff(num, den, var_num, var_den, ax_idx, q_idx=None):
    """Project num/den (sumw) and var_num/var_den (sumw2) onto the requested
    axis, optionally keeping charge separately. Returns (eff, var_eff) on the
    projected shape.

    Sums of (sumw, sumw2) are additive across summed bins, so the variance of
    the projected num/den is just the sum of the input variances. The variance
    of the resulting efficiency is then computed by _eff_and_var.
    """
    if q_idx is not None:
        sum_axes = tuple(i for i in range(num.ndim) if i not in (ax_idx, q_idx))
        num_p = num.sum(axis=sum_axes)
        den_p = den.sum(axis=sum_axes)
        var_num_p = var_num.sum(axis=sum_axes)
        var_den_p = var_den.sum(axis=sum_axes)
        if ax_idx > q_idx:
            num_p, den_p = num_p.T, den_p.T
            var_num_p, var_den_p = var_num_p.T, var_den_p.T
    else:
        sum_axes = tuple(i for i in range(num.ndim) if i != ax_idx)
        num_p = num.sum(axis=sum_axes)
        den_p = den.sum(axis=sum_axes)
        var_num_p = var_num.sum(axis=sum_axes)
        var_den_p = var_den.sum(axis=sum_axes)
    return _eff_and_var(num_p, den_p, var_num_p, var_den_p)


def _proc_color(process):
    if process in PROCESS_COLORS:
        return PROCESS_COLORS[process]
    return PROCESS_COLORS.get(_parse_process_entry(process)[0])


def _is_wo_tag(process):
    """Whether this entry's probe spectra are tag-unbiased: an explicit
    ':unbiased' variant, an entry resolved to the unbiased spectra (global
    --unbiased), or an entry read from the single-muon histmaker, which is
    tag-unbiased by construction (gen-based probes, no tag leg)."""
    if _parse_process_entry(process)[1]:
        return True
    if probe_names[process][categories[0]][0].startswith("effMCprobeUnbiased"):
        return True
    src_dg, _ = process_source[process]
    return src_dg.mode == "w_mass"


def _proc_label(process):
    group, _ = _parse_process_entry(process)
    base = PROCESS_LABELS.get(group, group)
    return f"{base} (w/o tag)" if _is_wo_tag(process) else base


def _proc_linestyle(process):
    # dashed = w/o tag (incl. the single-muon histmaker entries), so the
    # linestyle uniformly encodes the probe definition
    return "--" if _is_wo_tag(process) else "-"


# step -> (filename charge token); idip/trigger are charge-split, iso is "both".
_TNP_FILE_CHARGE = {
    "idip": {"minus": "minus", "plus": "plus"},
    "trigger": {"minus": "minus", "plus": "plus"},
    "iso": {"both": "both"},
}
_tnp_cache = {}


def load_tnp_eff(tnp_dir, step, charge_tag):
    """Read the external tag-and-probe EffMC2D and EffData2D for one
    (step, charge) from allEfficiencies_2D_<step>_<charge>.root.

    Returns a dict {"mc": (pt_centers, pt_halfwidths, eff[48,npt], err[48,npt]),
                    "data": (pt_centers, pt_halfwidths, eff[48,npt], err[48,npt])}
    with eta on the first axis (matching our 48-bin grid) and pT on the
    second. ``pt_halfwidths`` is half the (variable) pT bin width, used as the
    horizontal error bar so each point shows its pT bin extent. Results are
    cached per (step, charge). Returns None if the file is missing.
    """
    import ROOT

    key = (tnp_dir, step, charge_tag)
    if key in _tnp_cache:
        return _tnp_cache[key]

    file_charge = _TNP_FILE_CHARGE[step][charge_tag]
    path = os.path.join(tnp_dir, f"allEfficiencies_2D_{step}_{file_charge}.root")
    if not os.path.isfile(path):
        logger.warning(f"T&P file not found, skipping overlay: {path}")
        _tnp_cache[key] = None
        return None

    f = ROOT.TFile.Open(path)
    out = {}
    for which, hname in (("mc", "EffMC2D"), ("data", "EffData2D")):
        h2 = f.Get(hname)
        if not h2:
            logger.warning(f"{hname} not found in {path}")
            continue
        n_eta = h2.GetNbinsX()
        n_pt = h2.GetNbinsY()
        pt_centers = np.array([h2.GetYaxis().GetBinCenter(j + 1) for j in range(n_pt)])
        pt_halfwidths = np.array(
            [0.5 * h2.GetYaxis().GetBinWidth(j + 1) for j in range(n_pt)]
        )
        eff = np.empty((n_eta, n_pt))
        err = np.empty((n_eta, n_pt))
        for i in range(n_eta):
            for j in range(n_pt):
                eff[i, j] = h2.GetBinContent(i + 1, j + 1)
                err[i, j] = h2.GetBinError(i + 1, j + 1)
        out[which] = (pt_centers, pt_halfwidths, eff, err)
    f.Close()
    _tnp_cache[key] = out
    return out


# Marker styling for the T&P overlay points (distinct from the process lines).
_TNP_STYLE = {
    "mc": dict(fmt="o", color="black", label="T&P MC", markersize=4, capsize=2),
    "data": dict(fmt="s", color="dimgray", label="T&P data", markersize=4, capsize=2),
}


def make_effMC_plots(plotdir):
    """Layout:

        <plotdir>/<step>/effMC_<step>_proj_<axis>[_q<tag>].{pdf,png}
            1D projection: one figure per axis × charge (charge-dep steps) or
            per axis (iso). One curve per process, coloured by process.

        <plotdir>/<step>/per_cell/effMC_<step>_pt_eta<i>[_q<tag>].{pdf,png}
            eff(pT) at a fixed eta (and charge), uT integrated, one curve per
            process.

        <plotdir>/<step>/per_cell_ut/eta<i>/effMC_<step>_pt_eta<i>_ut<j>[_q<tag>].{pdf,png}
            eff(pT) at a fixed (eta, uT) cell (and charge), one curve per
            process. Only for the uT-dependent steps (trigger, iso); these are
            the cells entering the (pT, uT) parameterisation. No T&P overlay
            (the external T&P efficiencies are uT-integrated).

        <plotdir>/<step>/per_cell_pt/eta<i>/effMC_<step>_ut_eta<i>_pt<j>[_q<tag>].{pdf,png}
            Transpose of per_cell_ut: eff(uT) at a fixed (eta, pT) cell (and
            charge), one curve per process, smooth overlay at the pT bin center.
            uT-dependent steps only.

        <plotdir>/<step>/per_cell_2d/effMC_<step>_2d_eta<i>[_q<tag>].{pdf,png}
            2D eff(pT, uT) map per (eta, charge) cell for the reference process.
            With --smooth: raw | smoothed | pull = (smooth-raw)/sigma side by
            side, to localise mis-description. uT-dependent steps only.

    Per-step shared y-axis range across (cells × processes × charges) so the
    plots are directly comparable. Dashed unity line on every figure.
    """
    outdir = output_tools.make_plot_dir(*plotdir.rsplit("/", 1), eoscp=args.eoscp)
    # Nest each input file's metadata under its basename so write_logfile's
    # `{**info, **meta_info}` merge does *not* clobber this script's own
    # command/time/args/git_hash entries -- same pattern as make_theory_corr.py.
    input_meta_local = {}
    for input_file in args.inputFile:
        meta = base_io.get_metadata(input_file)
        if meta:
            input_meta_local[os.path.basename(input_file)] = meta

    axis_xlabels = {
        "eta": r"$\eta_\mu$",
        "pt": r"$p_T^\mu$ [GeV]",
        "genUT": r"$u_T$ [GeV]",
    }
    charge_tags = {-1: ("minus", r"$\mu^{-}$"), +1: ("plus", r"$\mu^{+}$")}

    # w/o-tag entries first: with the two-column legend (column-major fill)
    # they then group in one column, and the dashed w/o-tag curves are drawn
    # below the solid tag-based ones.
    procs_present = [p for p in process_data if _is_wo_tag(p)] + [
        p for p in process_data if not _is_wo_tag(p)
    ]
    # Reference process whose efficiency drives the shared y-axis range. Other
    # processes (e.g. Ztautau) have noisy low-stat per-bin efficiencies that
    # would otherwise blow up the range, so they are excluded from the ylim
    # computation but still drawn.
    ref_proc = "Zmumu" if "Zmumu" in procs_present else procs_present[0]

    for step in steps:
        step_dir = os.path.join(outdir, step)
        per_cell_dir = os.path.join(step_dir, "per_cell")
        os.makedirs(step_dir, exist_ok=True)
        os.makedirs(per_cell_dir, exist_ok=True)

        # Determine step layout from the first process (consistent across procs).
        _, _, _, _, axis_names, charge_dep = _step_num_den(procs_present[0], step)

        # ---- 1D projections (per axis × per charge for charge-dep steps) ----
        for axis_name in _step_proj_axes(step):
            ax_idx = axis_names.index(axis_name)
            q_idx = axis_names.index("charge") if charge_dep else None

            charge_iter = list(range(charge_ax.size)) if charge_dep else [None]
            for j_q in charge_iter:
                # Compute the projection (eff + stat variance) per process.
                per_proc_eff = {}
                per_proc_var = {}
                for process in procs_present:
                    num, den, var_num, var_den, _, _ = _step_num_den(process, step)
                    eff_p, var_p = _project_eff(
                        num, den, var_num, var_den, ax_idx, q_idx
                    )
                    if j_q is not None:
                        per_proc_eff[process] = eff_p[:, j_q]
                        per_proc_var[process] = var_p[:, j_q]
                    else:
                        per_proc_eff[process] = eff_p
                        per_proc_var[process] = var_p

                # Shared y-axis range from the reference process only (incl.
                # its error bars); noisy processes are excluded from the range.
                ymin, ymax = _ylim_from_arr(
                    per_proc_eff[ref_proc], per_proc_var[ref_proc]
                )

                fig, ax = plt.subplots(figsize=(8, 6))
                for process in procs_present:
                    x_ax = _proc_axis(process, axis_name)
                    y = per_proc_eff[process]
                    err = np.sqrt(np.maximum(per_proc_var[process], 0.0))
                    color = _proc_color(process)
                    ax.step(
                        x_ax.edges,
                        np.concatenate([y[:1], y]),
                        where="pre",
                        color=color,
                        linestyle=_proc_linestyle(process),
                        label=_proc_label(process),
                    )
                    ax.errorbar(
                        x_ax.centers,
                        y,
                        yerr=err,
                        fmt="none",
                        ecolor=color,
                        elinewidth=1.0,
                        capsize=0,
                    )
                ax.set_xlabel(axis_xlabels[axis_name])
                if axis_name == "pt":
                    ax.set_xlim(*args.plotPtRange)
                ax.set_ylabel(f"MC efficiency ({step})")
                ax.set_ylim(ymin, ymax)
                _draw_unity_line(ax)
                ax.legend(loc="upper left", ncol=2, frameon=False)

                if j_q is not None:
                    qkey = int(round(charge_ax.centers[j_q]))
                    tag, qlabel = charge_tags.get(qkey, (str(qkey), f"q={qkey}"))
                    ax.text(
                        0.04,
                        0.06,
                        qlabel,
                        transform=ax.transAxes,
                        ha="left",
                        va="bottom",
                        fontsize=18,
                    )
                    suffix = f"_q{tag}"
                else:
                    suffix = ""

                plot_tools.add_decor(ax, "CMS", None, data=False, lumi=None, loc=0)
                plot_name = f"effMC_{step}_proj_{axis_name}{suffix}"
                plot_tools.save_pdf_and_png(step_dir, plot_name)
                plt.close(fig)
                output_tools.write_index_and_log(
                    step_dir,
                    plot_name,
                    args=args,
                    analysis_meta_info=input_meta_local,
                )

        # ---- per-(eta, [q]) eff(pT) cells, uT integrated, overlaid procs ----
        # IDIP/trigger have a charge axis; iso does not.
        eff_is_charge_dep = process_data[procs_present[0]]["eff_etapt"][step].ndim == 3

        # T&P overlay (optional): preload EffMC2D/EffData2D per charge tag.
        # charge tag for the file lookup: "minus"/"plus" per cell for the
        # charge-dep steps, "both" for iso.
        tnp_by_tag = {}
        if args.tagAndProbeDir:
            if eff_is_charge_dep:
                tags_needed = [
                    charge_tags[int(round(charge_ax.centers[jq]))][0]
                    for jq in range(charge_ax.size)
                ]
            else:
                tags_needed = ["both"]
            for tag in tags_needed:
                tnp_by_tag[tag] = load_tnp_eff(args.tagAndProbeDir, step, tag)

        # Shared y-axis range from the reference process only (across cells ×
        # charges), plus the T&P points (eff ± err); noisy processes excluded.
        ylim_vals = process_data[ref_proc]["eff_etapt"][step].ravel()
        ylim_vars = process_data[ref_proc]["var_etapt"][step].ravel()
        for tnp in tnp_by_tag.values():
            if not tnp:
                continue
            for which in ("mc", "data"):
                if which in tnp:
                    _, _, e_tnp, err_tnp = tnp[which]
                    ylim_vals = np.concatenate([ylim_vals, e_tnp.ravel()])
                    ylim_vars = np.concatenate([ylim_vars, (err_tnp**2).ravel()])
        ymin, ymax = _ylim_from_arr(ylim_vals, ylim_vars)

        for i_eta in range(eta_ax.size):
            eta_lo, eta_hi = eta_ax.edges[i_eta], eta_ax.edges[i_eta + 1]
            eta_text = f"${eta_lo:+.2f}\\,\\leq\\,\\eta\\,<\\,{eta_hi:+.2f}$"
            cell_iter = range(charge_ax.size) if eff_is_charge_dep else [None]
            for j_q in cell_iter:
                fig, ax = plt.subplots(figsize=(8, 6))
                for process in procs_present:
                    pt_ax_p = _proc_axis(process, "pt")
                    eff_arr = process_data[process]["eff_etapt"][step]
                    var_arr = process_data[process]["var_etapt"][step]
                    if j_q is not None:
                        y = eff_arr[i_eta, :, j_q]
                        v = var_arr[i_eta, :, j_q]
                    else:
                        y = eff_arr[i_eta, :]
                        v = var_arr[i_eta, :]
                    err = np.sqrt(np.maximum(v, 0.0))
                    color = _proc_color(process)
                    ax.step(
                        pt_ax_p.edges,
                        np.concatenate([y[:1], y]),
                        where="pre",
                        color=color,
                        linestyle=_proc_linestyle(process),
                        label=_proc_label(process),
                    )
                    ax.errorbar(
                        pt_ax_p.centers,
                        y,
                        yerr=err,
                        fmt="none",
                        ecolor=color,
                        elinewidth=1.0,
                        capsize=0,
                    )
                    if args.smooth:
                        # smooth logit-Chebyshev fit (uT-dependent steps:
                        # den-weighted uT average, matching the raw points)
                        pt_fine = np.linspace(pt_ax_p.edges[0], pt_ax_p.edges[-1], 200)
                        y_sm = _smooth_eval_cell(process, step, i_eta, j_q, pt_fine)
                        if y_sm is not None:
                            ax.plot(
                                pt_fine, y_sm, color=color, linewidth=1.2, alpha=0.85
                            )
                if j_q is not None:
                    qkey = int(round(charge_ax.centers[j_q]))
                    tag, qlabel = charge_tags.get(qkey, (str(qkey), f"q={qkey}"))
                    annot = f"{qlabel}, {eta_text}"
                    suffix = f"_q{tag}"
                else:
                    tag = "both"
                    annot = eta_text
                    suffix = ""

                # Overlay the external T&P efficiencies for this (step, eta, q)
                tnp = tnp_by_tag.get(tag)
                if tnp:
                    for which in ("mc", "data"):
                        if which not in tnp:
                            continue
                        pt_tnp, pt_hw_tnp, e_tnp, err_tnp = tnp[which]
                        ax.errorbar(
                            pt_tnp,
                            e_tnp[i_eta],
                            yerr=err_tnp[i_eta],
                            xerr=pt_hw_tnp,
                            **_TNP_STYLE[which],
                        )
                ax.text(
                    0.04,
                    0.03,
                    annot,
                    transform=ax.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=18,
                )
                ax.set_xlabel(r"$p_T^\mu$ [GeV]")
                ax.set_xlim(*args.plotPtRange)
                ax.set_ylabel(f"MC efficiency ({step})")
                ax.set_ylim(ymin, ymax)
                _draw_unity_line(ax)
                ax.legend(loc="upper left", ncol=2, frameon=False)
                plot_tools.add_decor(ax, "CMS", None, data=False, lumi=None, loc=0)
                plot_name = f"effMC_{step}_pt_eta{i_eta:02d}{suffix}"
                plot_tools.save_pdf_and_png(per_cell_dir, plot_name)
                plt.close(fig)
                output_tools.write_index_and_log(
                    per_cell_dir,
                    plot_name,
                    args=args,
                    analysis_meta_info=input_meta_local,
                )

        # ---- per-(eta, uT, [q]) eff(pT) cells (uT-dependent steps only) ----
        # These are the cells that enter the (pT, uT) parameterisation. One
        # subfolder per eta bin, one figure per uT slice (× charge for the
        # charge-dependent steps). No T&P overlay: the external T&P
        # efficiencies are uT-integrated and would not be comparable.
        if step == "idip":
            continue

        eff_ut, var_ut = {}, {}
        for process in procs_present:
            num, den, var_num, var_den, _, _ = _step_num_den(process, step)
            eff_ut[process], var_ut[process] = _eff_and_var(num, den, var_num, var_den)

        ymin, ymax = _ylim_from_arr(eff_ut[ref_proc], var_ut[ref_proc])

        for i_eta in range(eta_ax.size):
            eta_dir = os.path.join(step_dir, "per_cell_ut", f"eta{i_eta:02d}")
            os.makedirs(eta_dir, exist_ok=True)
            eta_lo, eta_hi = eta_ax.edges[i_eta], eta_ax.edges[i_eta + 1]
            eta_text = f"${eta_lo:+.2f}\\,\\leq\\,\\eta\\,<\\,{eta_hi:+.2f}$"
            for j_ut in range(ut_ax.size):
                ut_lo, ut_hi = ut_ax.edges[j_ut], ut_ax.edges[j_ut + 1]
                ut_text = f"${ut_lo:.0f}\\,\\leq\\,u_T\\,<\\,{ut_hi:.0f}$ GeV"
                cell_iter = range(charge_ax.size) if charge_dep else [None]
                for j_q in cell_iter:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    for process in procs_present:
                        pt_ax_p = _proc_axis(process, "pt")
                        if j_q is not None:
                            y = eff_ut[process][i_eta, :, j_q, j_ut]
                            v = var_ut[process][i_eta, :, j_q, j_ut]
                        else:
                            y = eff_ut[process][i_eta, :, j_ut]
                            v = var_ut[process][i_eta, :, j_ut]
                        err = np.sqrt(np.maximum(v, 0.0))
                        color = _proc_color(process)
                        ax.step(
                            pt_ax_p.edges,
                            np.concatenate([y[:1], y]),
                            where="pre",
                            color=color,
                            linestyle=_proc_linestyle(process),
                            label=_proc_label(process),
                        )
                        ax.errorbar(
                            pt_ax_p.centers,
                            y,
                            yerr=err,
                            fmt="none",
                            ecolor=color,
                            elinewidth=1.0,
                            capsize=0,
                        )
                        if args.smooth:
                            # smooth logit-Chebyshev fit at this uT bin center
                            pt_fine = np.linspace(
                                pt_ax_p.edges[0], pt_ax_p.edges[-1], 200
                            )
                            y_sm = _smooth_eval_cell(
                                process, step, i_eta, j_q, pt_fine, j_ut=j_ut
                            )
                            if y_sm is not None:
                                ax.plot(
                                    pt_fine,
                                    y_sm,
                                    color=color,
                                    linewidth=1.2,
                                    alpha=0.85,
                                )
                    if j_q is not None:
                        qkey = int(round(charge_ax.centers[j_q]))
                        tag, qlabel = charge_tags.get(qkey, (str(qkey), f"q={qkey}"))
                        annot = f"{qlabel}, {ut_text}, {eta_text}"
                        suffix = f"_q{tag}"
                    else:
                        annot = f"{ut_text}, {eta_text}"
                        suffix = ""
                    ax.text(
                        0.04,
                        0.03,
                        annot,
                        transform=ax.transAxes,
                        ha="left",
                        va="bottom",
                        fontsize=18,
                    )
                    ax.set_xlabel(r"$p_T^\mu$ [GeV]")
                    ax.set_xlim(*args.plotPtRange)
                    ax.set_ylabel(f"MC efficiency ({step})")
                    ax.set_ylim(ymin, ymax)
                    _draw_unity_line(ax)
                    ax.legend(loc="upper left", ncol=2, frameon=False)
                    plot_tools.add_decor(ax, "CMS", None, data=False, lumi=None, loc=0)
                    plot_name = f"effMC_{step}_pt_eta{i_eta:02d}_ut{j_ut:02d}{suffix}"
                    plot_tools.save_pdf_and_png(eta_dir, plot_name)
                    plt.close(fig)
                    output_tools.write_index_and_log(
                        eta_dir,
                        plot_name,
                        args=args,
                        analysis_meta_info=input_meta_local,
                    )

        # ---- per-(eta, pt, [q]) eff(uT) cells (uT-dependent steps only) ----
        # Transpose of per_cell_ut: fix a pT slice and show the uT dependence,
        # one subfolder per eta bin, one figure per pT bin (x charge for the
        # charge-dependent steps). No T&P overlay (external T&P is uT-integrated).
        pt_ax_ref = _proc_axis(ref_proc, "pt")
        for i_eta in range(eta_ax.size):
            eta_dir = os.path.join(step_dir, "per_cell_pt", f"eta{i_eta:02d}")
            os.makedirs(eta_dir, exist_ok=True)
            eta_lo, eta_hi = eta_ax.edges[i_eta], eta_ax.edges[i_eta + 1]
            eta_text = f"${eta_lo:+.2f}\\,\\leq\\,\\eta\\,<\\,{eta_hi:+.2f}$"
            for j_pt in range(pt_ax_ref.size):
                pt_lo, pt_hi = pt_ax_ref.edges[j_pt], pt_ax_ref.edges[j_pt + 1]
                if pt_lo >= args.plotPtRange[1] or pt_hi <= args.plotPtRange[0]:
                    continue  # pT bin outside the displayed range
                pt_text = f"${pt_lo:.0f}\\,\\leq\\,p_T\\,<\\,{pt_hi:.0f}$ GeV"
                cell_iter = range(charge_ax.size) if charge_dep else [None]
                for j_q in cell_iter:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    for process in procs_present:
                        ut_ax_p = _proc_axis(process, "genUT")
                        pt_ax_p = _proc_axis(process, "pt")
                        # nearest pT bin of this process to the reference center
                        jp = int(
                            np.argmin(np.abs(pt_ax_p.centers - pt_ax_ref.centers[j_pt]))
                        )
                        if j_q is not None:
                            y = eff_ut[process][i_eta, jp, j_q, :]
                            v = var_ut[process][i_eta, jp, j_q, :]
                        else:
                            y = eff_ut[process][i_eta, jp, :]
                            v = var_ut[process][i_eta, jp, :]
                        err = np.sqrt(np.maximum(v, 0.0))
                        color = _proc_color(process)
                        ax.step(
                            ut_ax_p.edges,
                            np.concatenate([y[:1], y]),
                            where="pre",
                            color=color,
                            linestyle=_proc_linestyle(process),
                            label=_proc_label(process),
                        )
                        ax.errorbar(
                            ut_ax_p.centers,
                            y,
                            yerr=err,
                            fmt="none",
                            ecolor=color,
                            elinewidth=1.0,
                            capsize=0,
                        )
                        if args.smooth:
                            ut_fine = np.linspace(
                                ut_ax_p.edges[0], ut_ax_p.edges[-1], 200
                            )
                            grid = _smooth_grid_cell(
                                process,
                                step,
                                i_eta,
                                j_q,
                                pt_vals=[pt_ax_p.centers[jp]],
                                ut_vals=ut_fine,
                            )
                            if grid is not None:
                                ax.plot(
                                    ut_fine,
                                    grid[0],
                                    color=color,
                                    linewidth=1.2,
                                    alpha=0.85,
                                )
                    if j_q is not None:
                        qkey = int(round(charge_ax.centers[j_q]))
                        tag, qlabel = charge_tags.get(qkey, (str(qkey), f"q={qkey}"))
                        annot = f"{qlabel}, {pt_text}, {eta_text}"
                        suffix = f"_q{tag}"
                    else:
                        annot = f"{pt_text}, {eta_text}"
                        suffix = ""
                    ax.text(
                        0.04,
                        0.03,
                        annot,
                        transform=ax.transAxes,
                        ha="left",
                        va="bottom",
                        fontsize=18,
                    )
                    ax.set_xlabel(axis_xlabels["genUT"])
                    ax.set_ylabel(f"MC efficiency ({step})")
                    ax.set_ylim(ymin, ymax)
                    _draw_unity_line(ax)
                    ax.legend(loc="upper left", ncol=2, frameon=False)
                    plot_tools.add_decor(ax, "CMS", None, data=False, lumi=None, loc=0)
                    plot_name = f"effMC_{step}_ut_eta{i_eta:02d}_pt{j_pt:02d}{suffix}"
                    plot_tools.save_pdf_and_png(eta_dir, plot_name)
                    plt.close(fig)
                    output_tools.write_index_and_log(
                        eta_dir,
                        plot_name,
                        args=args,
                        analysis_meta_info=input_meta_local,
                    )

        # ---- 2D eff(pT, uT) maps per (eta, [q]) cell, reference process ----
        # raw measured efficiency; with --smooth also the smoothed surface and
        # the pull (smooth-raw)/sigma, side by side, to localise mis-description.
        twod_dir = os.path.join(step_dir, "per_cell_2d")
        os.makedirs(twod_dir, exist_ok=True)
        pt_ax_r = _proc_axis(ref_proc, "pt")
        ut_ax_r = _proc_axis(ref_proc, "genUT")
        _e = eff_ut[ref_proc]
        _m = (_e > 0) & (_e < 1)
        evmin = float(_e[_m].min()) if _m.any() else 0.0
        evmax = float(_e[_m].max()) if _m.any() else 1.0
        for i_eta in range(eta_ax.size):
            eta_lo, eta_hi = eta_ax.edges[i_eta], eta_ax.edges[i_eta + 1]
            eta_text = f"${eta_lo:+.2f}\\,\\leq\\,\\eta\\,<\\,{eta_hi:+.2f}$"
            cell_iter = range(charge_ax.size) if charge_dep else [None]
            for j_q in cell_iter:
                if j_q is not None:
                    raw = eff_ut[ref_proc][i_eta, :, j_q, :]
                    vraw = var_ut[ref_proc][i_eta, :, j_q, :]
                    qkey = int(round(charge_ax.centers[j_q]))
                    tag, qlabel = charge_tags.get(qkey, (str(qkey), f"q={qkey}"))
                    annot = f"{qlabel}, {eta_text}"
                    suffix = f"_q{tag}"
                else:
                    raw = eff_ut[ref_proc][i_eta, :, :]
                    vraw = var_ut[ref_proc][i_eta, :, :]
                    annot = eta_text
                    suffix = ""
                panels = [
                    ("raw eff", raw, "viridis", (evmin, evmax), f"MC eff ({step})")
                ]
                if args.smooth:
                    grid = _smooth_grid_cell(ref_proc, step, i_eta, j_q)
                    if grid is not None:
                        sig = np.sqrt(np.maximum(vraw, 0.0))
                        with np.errstate(divide="ignore", invalid="ignore"):
                            pull = np.where(sig > 0, (grid - raw) / sig, np.nan)
                        panels.append(
                            (
                                "smoothed",
                                grid,
                                "viridis",
                                (evmin, evmax),
                                f"smoothed eff ({step})",
                            )
                        )
                        panels.append(
                            (
                                "pull",
                                pull,
                                "RdBu_r",
                                (-5.0, 5.0),
                                r"(smooth$-$raw)/$\sigma$",
                            )
                        )
                fig, axes = plt.subplots(
                    1, len(panels), figsize=(6.0 * len(panels), 5.0), squeeze=False
                )
                for ax, (title, arr, cmap, (vmn, vmx), clab) in zip(axes[0], panels):
                    if title == "pull":
                        masked = np.ma.masked_invalid(arr)
                    else:
                        masked = np.ma.masked_where(~((arr > 0) & (arr < 1)), arr)
                    mesh = ax.pcolormesh(
                        pt_ax_r.edges,
                        ut_ax_r.edges,
                        masked.T,
                        cmap=cmap,
                        vmin=vmn,
                        vmax=vmx,
                    )
                    fig.colorbar(mesh, ax=ax, label=clab)
                    ax.set_xlabel(axis_xlabels["pt"])
                    ax.set_ylabel(axis_xlabels["genUT"])
                    ax.set_xlim(*args.plotPtRange)
                    ax.set_title(title)
                axes[0][0].text(
                    0.04,
                    0.94,
                    annot,
                    transform=axes[0][0].transAxes,
                    ha="left",
                    va="top",
                    fontsize=14,
                )
                fig.tight_layout()
                plot_name = f"effMC_{step}_2d_eta{i_eta:02d}{suffix}"
                plot_tools.save_pdf_and_png(twod_dir, plot_name)
                plt.close(fig)
                output_tools.write_index_and_log(
                    twod_dir,
                    plot_name,
                    args=args,
                    analysis_meta_info=input_meta_local,
                )

    logger.info(f"Wrote effMC plots to {outdir}")
    if output_tools.is_eosuser_path(plotdir) and args.eoscp:
        output_tools.copy_to_eos(outdir, plotdir)


def make_corrC_plots(plotdir):
    """Tag-conditioning correlation factor C = eff(T&P-like) / eff(tag-unbiased),
    per step, for every group with both a tag-based and a ':unbiased' entry in
    --process (e.g. --process Zmumu Zmumu:unbiased).

    C is the per-bin ratio of the conditional-on-tag to the inclusive probe
    efficiency. In the in-situ weight model it multiplies the probe-leg
    efficiency of the (tag-conditioned) Z fit categories: it cancels exactly in
    the pass weight Ce'/(Ce) but survives in the fail weight (1-Ce')/(1-Ce),
    where the fail-category leverage on the SF scales as Ce/(1-Ce) -- so a
    per-mille C-1 matters wherever the fail fraction is small (idip at high
    |eta|). Using the tag-based effMC in the Z helper implements C from MC to
    first order; these plots quantify it.

    Layout:
        <plotdir>/corrC/corrC_<step>_proj_<axis>[_q<tag>]      1D projections
        <plotdir>/corrC/corrC_<step>_{etapt,ptut}[_q<tag>]     2D maps
        <plotdir>/corrC/ptut/...                               2D per-eta maps
        <plotdir>/corrC/per_cell/corrC_idip_pt_eta<i>_q<tag>   C(pT) per cell
        <plotdir>/corrC/per_cell_ut/eta<i>/corrC_<step>_pt_eta<i>_ut<j>[_q<tag>]
                                       C(pT) per native (eta[, q], uT) cell

    With --smooth, C is additionally derived from the smoothed efficiencies:
    the 1D projections gain a smooth-C line (den-weighted projection of the
    smoothed effMC per sample, same composition convention as the raw points),
    and every 2D map gets a parallel '_smooth' file (direct ratio at the
    native bins; den-weighted averages where an axis is integrated).

    NB: the error bars combine the two efficiencies' stat variances in
    quadrature; the probe samples overlap strongly (tag-based probes are
    mostly a subset of the unbiased ones), so this overestimates the
    uncertainty on C.
    """
    pairs = []
    for entry in process_data:
        group, unbiased = _parse_process_entry(entry)
        if unbiased and group in process_data:
            pairs.append((group, entry))
    if not pairs:
        return

    outdir = output_tools.make_plot_dir(*plotdir.rsplit("/", 1), eoscp=args.eoscp)
    corr_dir = os.path.join(outdir, "corrC")
    os.makedirs(corr_dir, exist_ok=True)
    input_meta_local = {}
    for input_file in args.inputFile:
        meta = base_io.get_metadata(input_file)
        if meta:
            input_meta_local[os.path.basename(input_file)] = meta

    axis_xlabels = {
        "eta": r"$\eta_\mu$",
        "pt": r"$p_T^\mu$ [GeV]",
        "genUT": r"$u_T$ [GeV]",
    }
    charge_tags = {-1: ("minus", r"$\mu^{-}$"), +1: ("plus", r"$\mu^{+}$")}

    def _C_of(num_t, den_t, num_u, den_u):
        """Masked C array from tag/unbiased counts (masked where either
        efficiency is empty)."""
        eff_t, eff_u = _ratio(num_t, den_t), _ratio(num_u, den_u)
        return np.ma.masked_where((eff_t <= 0) | (eff_u <= 0), _ratio(eff_t, eff_u))

    def _C_ratio(eff_t, eff_u):
        """Masked C array from two efficiency arrays."""
        return np.ma.masked_where((eff_t <= 0) | (eff_u <= 0), _ratio(eff_t, eff_u))

    def _smooth_eff_and_w(entry, step):
        """Smoothed per-bin efficiencies and their den weights on the native
        smoothed grid: idip (eta, pt, charge) [uT-collapsed den], trigger
        (eta, pt, charge, uT), iso (eta, pt, uT). (None, None) without
        --smooth fits."""
        pdd = process_data[entry]
        if "smooth_coeffs" not in pdd:
            return None, None
        e = pdd["effMC"][step].values()
        w = pdd["step_den"][step]
        if step == "idip":
            w = w.sum(axis=-1)
        return e, w

    def _proj_weighted(vals, w, ax_idx, q_idx=None):
        """Weighted average of per-bin values onto one axis (optionally keeping
        the charge axis), mirroring _project_eff's axis handling."""
        keep = (ax_idx,) if q_idx is None else (ax_idx, q_idx)
        sum_axes = tuple(i for i in range(vals.ndim) if i not in keep)
        num_p = (vals * w).sum(axis=sum_axes)
        den_p = w.sum(axis=sum_axes)
        if q_idx is not None and ax_idx > q_idx:
            num_p, den_p = num_p.T, den_p.T
        return _ratio(num_p, den_p)

    def _wavg(e, w, axis):
        """Den-weighted average of smoothed efficiencies over one axis; 0
        (-> masked downstream) where the weight sum vanishes."""
        ws = w.sum(axis=axis)
        return np.where(ws > 0, (e * w).sum(axis=axis) / np.maximum(ws, 1e-9), 0.0)

    def _halfrange(masked):
        """Robust symmetric color half-range around 1 (noisy bins excluded)."""
        dev = np.abs(np.ma.compressed(masked) - 1.0)
        return max(0.002, float(np.percentile(dev, 98)) if dev.size else 0.005)

    def _plot_C_map(
        masked, x_edges, y_edges, x_name, y_name, halfrange, step, label, out_dir, name
    ):
        fig, ax = plt.subplots(figsize=(9, 7))
        mesh = ax.pcolormesh(
            x_edges,
            y_edges,
            masked.T,
            cmap="RdBu_r",
            vmin=1.0 - halfrange,
            vmax=1.0 + halfrange,
        )
        fig.colorbar(mesh, ax=ax, label=rf"$C$ = eff(T&P) / eff(unbiased) ({step})")
        ax.set_xlabel(axis_xlabels[x_name])
        ax.set_ylabel(axis_xlabels[y_name])
        if x_name == "pt":
            ax.set_xlim(*args.plotPtRange)
        if y_name == "pt":
            ax.set_ylim(*args.plotPtRange)
        ax.text(
            0.04, 0.94, label, transform=ax.transAxes, ha="left", va="top", fontsize=16
        )
        plot_tools.add_decor(ax, "CMS", None, data=False, lumi=None, loc=0)
        plot_tools.save_pdf_and_png(out_dir, name)
        plt.close(fig)
        output_tools.write_index_and_log(
            out_dir, name, args=args, analysis_meta_info=input_meta_local
        )

    for step in steps:
        _, _, _, _, axis_names, charge_dep = _step_num_den(pairs[0][0], step)
        for axis_name in _step_proj_axes(step):
            ax_idx = axis_names.index(axis_name)
            q_idx = axis_names.index("charge") if charge_dep else None

            charge_iter = list(range(charge_ax.size)) if charge_dep else [None]
            for j_q in charge_iter:
                per_pair_C = {}
                per_pair_var = {}
                for group, unb_entry in pairs:
                    eff = {}
                    var = {}
                    for key, entry in (("tag", group), ("unb", unb_entry)):
                        num, den, var_num, var_den, _, _ = _step_num_den(entry, step)
                        eff_p, var_p = _project_eff(
                            num, den, var_num, var_den, ax_idx, q_idx
                        )
                        if j_q is not None:
                            eff_p, var_p = eff_p[:, j_q], var_p[:, j_q]
                        eff[key], var[key] = eff_p, var_p
                    C = _safe_ratio_1d(eff["tag"], eff["unb"])
                    # quadrature of the two relative variances (overestimate,
                    # see docstring)
                    rel2 = _safe_ratio_1d(var["tag"], eff["tag"] ** 2) + _safe_ratio_1d(
                        var["unb"], eff["unb"] ** 2
                    )
                    per_pair_C[group] = C
                    per_pair_var[group] = C**2 * rel2

                fig, ax = plt.subplots(figsize=(8, 6))
                lo, hi = 1.0, 1.0
                for group, unb_entry in pairs:
                    x_ax = _proc_axis(group, axis_name)
                    y = per_pair_C[group]
                    err = np.sqrt(np.maximum(per_pair_var[group], 0.0))
                    color = _proc_color(group)
                    ax.step(
                        x_ax.edges,
                        np.concatenate([y[:1], y]),
                        where="pre",
                        color=color,
                        label=_proc_label(group),
                    )
                    ax.errorbar(
                        x_ax.centers,
                        y,
                        yerr=err,
                        fmt="none",
                        ecolor=color,
                        elinewidth=1.0,
                        capsize=0,
                    )
                    if args.smooth:
                        # C from the smoothed efficiencies (den-weighted
                        # projection per sample, same convention as the raw)
                        e_t, w_t = _smooth_eff_and_w(group, step)
                        e_u, w_u = _smooth_eff_and_w(unb_entry, step)
                        if e_t is not None and e_u is not None:
                            p_t = _proj_weighted(e_t, w_t, ax_idx, q_idx)
                            p_u = _proj_weighted(e_u, w_u, ax_idx, q_idx)
                            if j_q is not None:
                                p_t, p_u = p_t[:, j_q], p_u[:, j_q]
                            C_sm = _safe_ratio_1d(p_t, p_u)
                            ax.plot(
                                x_ax.centers,
                                np.where(C_sm > 0, C_sm, np.nan),
                                color=color,
                                linewidth=1.2,
                                alpha=0.85,
                            )
                    good = y > 0
                    if axis_name == "pt":
                        # y-range from the displayed pt window only
                        good &= (x_ax.centers >= args.plotPtRange[0]) & (
                            x_ax.centers <= args.plotPtRange[1]
                        )
                    if good.any():
                        lo = min(lo, float((y - err)[good].min()))
                        hi = max(hi, float((y + err)[good].max()))
                margin = max(0.002, 0.2 * (hi - lo))
                ax.set_xlabel(axis_xlabels[axis_name])
                if axis_name == "pt":
                    ax.set_xlim(*args.plotPtRange)
                ax.set_ylabel(rf"$C$ = eff(T&P) / eff(unbiased) ({step})")
                ax.set_ylim(lo - margin, hi + margin)
                _draw_unity_line(ax)
                ax.legend(loc="upper left", ncol=2, frameon=False)

                if j_q is not None:
                    qkey = int(round(charge_ax.centers[j_q]))
                    tag, qlabel = charge_tags.get(qkey, (str(qkey), f"q={qkey}"))
                    ax.text(
                        0.04,
                        0.06,
                        qlabel,
                        transform=ax.transAxes,
                        ha="left",
                        va="bottom",
                        fontsize=18,
                    )
                    suffix = f"_q{tag}"
                else:
                    suffix = ""

                plot_tools.add_decor(ax, "CMS", None, data=False, lumi=None, loc=0)
                plot_name = f"corrC_{step}_proj_{axis_name}{suffix}"
                plot_tools.save_pdf_and_png(corr_dir, plot_name)
                plt.close(fig)
                output_tools.write_index_and_log(
                    corr_dir,
                    plot_name,
                    args=args,
                    analysis_meta_info=input_meta_local,
                )

        # ---- 2D maps of C ----
        # (eta, pt) uT-integrated, one per charge for the charge-dep steps;
        # for the uT-dependent steps (trigger, iso) also (pt, uT) maps:
        # eta-integrated in the main folder, plus one per eta slice in ptut/
        # (the cells entering the (pT, uT) parameterisation), the latter with
        # a color range shared across the eta slices of one (step, charge) so
        # the maps are directly comparable.
        charge_iter = list(range(charge_ax.size)) if charge_dep else [None]
        if step in ("trigger", "iso"):
            ptut_dir = os.path.join(corr_dir, "ptut")
            os.makedirs(ptut_dir, exist_ok=True)
        for group, unb_entry in pairs:
            num_t, den_t, *_ = _step_num_den(group, step)
            num_u, den_u, *_ = _step_num_den(unb_entry, step)
            eta_edges = _proc_axis(group, "eta").edges
            pt_edges = _proc_axis(group, "pt").edges
            ut_edges = _proc_axis(group, "genUT").edges
            group_tag = f"_{group}" if len(pairs) > 1 else ""
            # smoothed efficiencies (with their den weights) when available
            es_t, ws_t = _smooth_eff_and_w(group, step) if args.smooth else (None, None)
            es_u, ws_u = (
                _smooth_eff_and_w(unb_entry, step) if args.smooth else (None, None)
            )
            have_smooth = es_t is not None and es_u is not None

            for j_q in charge_iter:
                if j_q is not None:
                    nt, dt = num_t[:, :, j_q, :], den_t[:, :, j_q, :]
                    nu, du = num_u[:, :, j_q, :], den_u[:, :, j_q, :]
                    if have_smooth:
                        # idip smoothed grid is uT-collapsed (eta, pt, charge)
                        if step == "idip":
                            et, wt = es_t[:, :, j_q], ws_t[:, :, j_q]
                            eu, wu = es_u[:, :, j_q], ws_u[:, :, j_q]
                        else:
                            et, wt = es_t[:, :, j_q, :], ws_t[:, :, j_q, :]
                            eu, wu = es_u[:, :, j_q, :], ws_u[:, :, j_q, :]
                    qkey = int(round(charge_ax.centers[j_q]))
                    tag, qlabel = charge_tags.get(qkey, (str(qkey), f"q={qkey}"))
                    base_label = f"{_proc_label(group)}, {qlabel}"
                    suffix = f"_q{tag}"
                else:
                    nt, dt, nu, du = num_t, den_t, num_u, den_u
                    if have_smooth:
                        et, wt, eu, wu = es_t, ws_t, es_u, ws_u
                    base_label = _proc_label(group)
                    suffix = ""

                # (eta, pt), uT integrated
                C = _C_of(nt.sum(-1), dt.sum(-1), nu.sum(-1), du.sum(-1))
                _plot_C_map(
                    C,
                    eta_edges,
                    pt_edges,
                    "eta",
                    "pt",
                    _halfrange(C),
                    step,
                    base_label,
                    corr_dir,
                    f"corrC_{step}_etapt{group_tag}{suffix}",
                )
                if have_smooth:
                    # idip is already (eta, pt); uT-dependent steps use the
                    # den-weighted uT average per sample
                    if step == "idip":
                        C_sm = _C_ratio(et, eu)
                    else:
                        C_sm = _C_ratio(_wavg(et, wt, -1), _wavg(eu, wu, -1))
                    _plot_C_map(
                        C_sm,
                        eta_edges,
                        pt_edges,
                        "eta",
                        "pt",
                        _halfrange(C_sm),
                        step,
                        f"{base_label} (smooth)",
                        corr_dir,
                        f"corrC_{step}_etapt{group_tag}{suffix}_smooth",
                    )

                if step not in ("trigger", "iso"):
                    continue

                # (pt, uT), eta integrated
                C = _C_of(nt.sum(0), dt.sum(0), nu.sum(0), du.sum(0))
                _plot_C_map(
                    C,
                    pt_edges,
                    ut_edges,
                    "pt",
                    "genUT",
                    _halfrange(C),
                    step,
                    base_label,
                    corr_dir,
                    f"corrC_{step}_ptut{group_tag}{suffix}",
                )
                if have_smooth:
                    C_sm = _C_ratio(_wavg(et, wt, 0), _wavg(eu, wu, 0))
                    _plot_C_map(
                        C_sm,
                        pt_edges,
                        ut_edges,
                        "pt",
                        "genUT",
                        _halfrange(C_sm),
                        step,
                        f"{base_label} (smooth)",
                        corr_dir,
                        f"corrC_{step}_ptut{group_tag}{suffix}_smooth",
                    )

                # (pt, uT) per eta slice, shared color range
                C_all = _C_of(nt, dt, nu, du)
                halfrange = _halfrange(C_all)
                # fully smooth per-eta maps: direct ratio at the native bins
                C_sm_all = _C_ratio(et, eu) if have_smooth else None
                halfrange_sm = _halfrange(C_sm_all) if have_smooth else None
                for i_eta in range(len(eta_edges) - 1):
                    eta_label = (
                        rf"${eta_edges[i_eta]:.2f} \leq \eta < "
                        rf"{eta_edges[i_eta + 1]:.2f}$"
                    )
                    _plot_C_map(
                        C_all[i_eta],
                        pt_edges,
                        ut_edges,
                        "pt",
                        "genUT",
                        halfrange,
                        step,
                        f"{base_label}, {eta_label}",
                        ptut_dir,
                        f"corrC_{step}_ptut_eta{i_eta:02d}{group_tag}{suffix}",
                    )
                    if C_sm_all is not None:
                        _plot_C_map(
                            C_sm_all[i_eta],
                            pt_edges,
                            ut_edges,
                            "pt",
                            "genUT",
                            halfrange_sm,
                            step,
                            f"{base_label} (smooth), {eta_label}",
                            ptut_dir,
                            f"corrC_{step}_ptut_eta{i_eta:02d}{group_tag}{suffix}_smooth",
                        )

        # ---- 1D C(pT) per-cell plots ----
        # idip: per (eta, charge) [native, uT-collapsed]; trigger: per
        # (eta, charge, uT); iso: per (eta, uT) -- the native cells of the
        # parameterisation. Raw C as step + error bars per pair; with --smooth
        # also the fully smooth C curve (ratio of the fitted cell models).
        if step == "idip":
            cell_dir = os.path.join(corr_dir, "per_cell")
            os.makedirs(cell_dir, exist_ok=True)
            ut_iter = [None]
        else:
            ut_iter = list(range(_proc_axis(pairs[0][0], "genUT").size))
        charge_iter = list(range(charge_ax.size)) if charge_dep else [None]
        # per-pair raw eff/var on the native grid
        cell_eff = {}
        for group, unb_entry in pairs:
            for key, entry in (("tag", group), ("unb", unb_entry)):
                num, den, vn, vd, _, _ = _step_num_den(entry, step)
                if step == "idip":
                    num, den, vn, vd = (a.sum(axis=-1) for a in (num, den, vn, vd))
                cell_eff[(group, key)] = _eff_and_var(num, den, vn, vd)
        eta_edges_c = _proc_axis(pairs[0][0], "eta").edges
        ut_edges_c = _proc_axis(pairs[0][0], "genUT").edges

        for i_eta in range(len(eta_edges_c) - 1):
            if step != "idip":
                cell_dir = os.path.join(corr_dir, "per_cell_ut", f"eta{i_eta:02d}")
                os.makedirs(cell_dir, exist_ok=True)
            eta_lo, eta_hi = eta_edges_c[i_eta], eta_edges_c[i_eta + 1]
            eta_text = f"${eta_lo:+.2f}\\,\\leq\\,\\eta\\,<\\,{eta_hi:+.2f}$"
            for j_ut in ut_iter:
                for j_q in charge_iter:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    lo, hi = 1.0, 1.0
                    for group, unb_entry in pairs:
                        pt_ax_p = _proc_axis(group, "pt")
                        ys, vs = {}, {}
                        for key in ("tag", "unb"):
                            e_, v_ = cell_eff[(group, key)]
                            if step == "idip":
                                ys[key], vs[key] = e_[i_eta, :, j_q], v_[i_eta, :, j_q]
                            elif step == "trigger":
                                ys[key] = e_[i_eta, :, j_q, j_ut]
                                vs[key] = v_[i_eta, :, j_q, j_ut]
                            else:
                                ys[key], vs[key] = (
                                    e_[i_eta, :, j_ut],
                                    v_[i_eta, :, j_ut],
                                )
                        C = _safe_ratio_1d(ys["tag"], ys["unb"])
                        rel2 = _safe_ratio_1d(
                            vs["tag"], ys["tag"] ** 2
                        ) + _safe_ratio_1d(vs["unb"], ys["unb"] ** 2)
                        err = np.sqrt(np.maximum(C**2 * rel2, 0.0))
                        color = _proc_color(group)
                        y_draw = np.where(C > 0, C, np.nan)
                        ax.step(
                            pt_ax_p.edges,
                            np.concatenate([y_draw[:1], y_draw]),
                            where="pre",
                            color=color,
                            label=_proc_label(group),
                        )
                        ax.errorbar(
                            pt_ax_p.centers,
                            y_draw,
                            yerr=err,
                            fmt="none",
                            ecolor=color,
                            elinewidth=1.0,
                            capsize=0,
                        )
                        if args.smooth:
                            pt_fine = np.linspace(
                                pt_ax_p.edges[0], pt_ax_p.edges[-1], 200
                            )
                            c_t = _smooth_eval_cell(
                                group, step, i_eta, j_q, pt_fine, j_ut=j_ut
                            )
                            c_u = _smooth_eval_cell(
                                unb_entry, step, i_eta, j_q, pt_fine, j_ut=j_ut
                            )
                            if c_t is not None and c_u is not None:
                                ax.plot(
                                    pt_fine,
                                    c_t / np.maximum(c_u, 1e-9),
                                    color=color,
                                    linewidth=1.2,
                                    alpha=0.85,
                                )
                        good = (
                            (C > 0)
                            # y-range from the displayed pt window only
                            & (pt_ax_p.centers >= args.plotPtRange[0])
                            & (pt_ax_p.centers <= args.plotPtRange[1])
                        )
                        if good.any():
                            lo = min(lo, float((C - err)[good].min()))
                            hi = max(hi, float((C + err)[good].max()))
                    margin = max(0.002, 0.2 * (hi - lo))
                    ax.set_xlabel(r"$p_T^\mu$ [GeV]")
                    ax.set_xlim(*args.plotPtRange)
                    ax.set_ylabel(rf"$C$ = eff(T&P) / eff(unbiased) ({step})")
                    ax.set_ylim(lo - margin, hi + margin)
                    _draw_unity_line(ax)
                    ax.legend(loc="upper left", ncol=2, frameon=False)
                    annot = eta_text
                    if j_ut is not None:
                        ut_lo, ut_hi = ut_edges_c[j_ut], ut_edges_c[j_ut + 1]
                        annot = (
                            f"${ut_lo:.0f}\\,\\leq\\,u_T\\,<\\,{ut_hi:.0f}$ GeV, "
                            + annot
                        )
                    if j_q is not None:
                        qkey = int(round(charge_ax.centers[j_q]))
                        tag, qlabel = charge_tags.get(qkey, (str(qkey), f"q={qkey}"))
                        annot = f"{qlabel}, {annot}"
                        suffix = f"_q{tag}"
                    else:
                        suffix = ""
                    ax.text(
                        0.04,
                        0.03,
                        annot,
                        transform=ax.transAxes,
                        ha="left",
                        va="bottom",
                        fontsize=16,
                    )
                    plot_tools.add_decor(ax, "CMS", None, data=False, lumi=None, loc=0)
                    ut_tag = f"_ut{j_ut:02d}" if j_ut is not None else ""
                    plot_name = f"corrC_{step}_pt_eta{i_eta:02d}{ut_tag}{suffix}"
                    plot_tools.save_pdf_and_png(cell_dir, plot_name)
                    plt.close(fig)
                    output_tools.write_index_and_log(
                        cell_dir,
                        plot_name,
                        args=args,
                        analysis_meta_info=input_meta_local,
                    )

    logger.info(f"Wrote correlation-factor C plots to {corr_dir}")


if args.plotdir:
    make_corrC_plots(args.plotdir)
    make_effMC_plots(args.plotdir)


# Write one pkl per process, carrying the provenance of its own source file.
filename_base = "insitu_effMC"
if args.unbiased:
    filename_base += "_unbiased"
if args.postfix:
    filename_base += f"_{args.postfix}"
for process, pd in process_data.items():
    _, source_file = process_source[process]
    input_meta = base_io.get_metadata(source_file)
    # 'Zmumu:unbiased' -> 'Zmumu_unbiased' (filesystem/key friendly)
    out_name = process.replace(":", "_")
    outfile = f"{args.outpath}/{filename_base}_{out_name}.pkl.lz4"
    output_dict = {"effMC": pd["effMC"], "process": process, "steps": steps}
    if args.smooth:
        # effMC above is the smoothed one (what the in-situ helper consumes);
        # keep the raw one and the fit configuration/coefficients alongside
        output_dict["effMC_raw"] = pd["effMC_raw"]
        output_dict["smooth"] = {
            "n_coeff_pt": args.smoothCoeffPt,
            "n_coeff_ut": args.smoothCoeffUt,
            "pt_range": insitu_pt_range,
            "ut_range": insitu_ut_range,
            "coeffs": pd["smooth_coeffs"],
        }
    # Carry over the input histmaker's provenance as file_meta_data, and let
    # write_lz4_pkl_output attach this script's own meta_data (command, git
    # hash, timestamp) so the output is inspectable with
    # scripts/inspect/print_command.py.
    output_tools.write_lz4_pkl_output(
        outfile,
        out_name,
        output_dict,
        common.base_dir,
        args,
        input_meta,
    )
    logger.info(f"Wrote in-situ effMC[{process}] to {outfile}")
