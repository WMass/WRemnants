import argparse
import os
import pickle

import lz4.frame
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import chebyshev as C

from rabbit.io_tools import get_fitresult
from wremnants.production.muon_efficiencies_insitu import (
    insitu_delta,
    insitu_n_coeff_pt,
    insitu_n_coeff_ut,
    insitu_parameter_labels,
    insitu_pt_range,
    insitu_ut_range,
    load_insitu_central,
)
from wums import logging, output_tools, plot_tools

# Postfit diagnostic for the in-situ tag-and-probe muon efficiency fit.
#
# The fit floats the ID/HLT/Iso data/MC scale factors as Chebyshev polynomials,
# with the passing-leg SF the polynomial directly (linear parameterisation):
#   SF_X(pt[,uT]) = 1 + P_X,  P_X = sum_{k,m} theta_{X,eta,k,m} T_k(x_pt) T_m(x_uT),
# with theta = theta_central + delta * n_hat (n_hat = fitted nuisance). The SF is
# MC-independent (effMC enters only the failing factor, not plotted here). The
# *marginal* per-coefficient uncertainties are large and strongly correlated
# within each (eta, charge) block, so they are not the meaningful quantity --
# the SF (a fixed linear combination of the coefficients) is what the data
# constrains. Since SF is linear in theta, the band is sigma_SF = sqrt(B^T Cov B).
# This script propagates the FULL parameter covariance into sigma_SF(pt[,uT]) and
# overlays the external tag-and-probe SF (= EffData2D / EffMC2D), in the same
# per-(eta, charge) pt-cell layout as the effMC diagnostic plots
# (make_insitu_effMC.py).

logger = logging.child_logger(__name__)

# Chebyshev coefficient block per step: prefix + whether it is charge-dependent
# and has a uT dependence. Mirrors insitu_parameter_labels().
STEP_INFO = {
    "idip": dict(prefix="effInsituID", charge=True, has_ut=False, ylabel="ID"),
    "trigger": dict(prefix="effInsituHLT", charge=True, has_ut=True, ylabel="trigger"),
    "iso": dict(prefix="effInsituIso", charge=False, has_ut=True, ylabel="iso"),
}
# step -> external T&P file token; idip/trigger charge-split, iso "both".
TNP_FILE = {"idip": "idip", "trigger": "trigger", "iso": "iso"}
TNP_STYLE = dict(fmt="s", color="black", markersize=4, capsize=2, zorder=5)
CHARGE_TAGS = {-1: ("minus", r"$\mu^{-}$"), +1: ("plus", r"$\mu^{+}$")}


def cheb(x, n):
    """Chebyshev basis [T_0(x), ..., T_{n-1}(x)] at scalar/array x."""
    return np.array([C.chebval(x, [0] * i + [1]) for i in range(n)])


def norm(v, lo, hi):
    return 2.0 * (np.clip(v, lo, hi) - lo) / (hi - lo) - 1.0


_tnp_cache = {}


_sf3d_cache = {}

# Our step names -> the keys in smoothSF3D_*.pkl.lz4. idip is absent there (the
# smoothing is 3D only for the steps that actually depend on uT), so it keeps
# the 2D T&P points.
SF3D_KEY = {
    ("trigger", "plus"): "smoothSF3D_triggerplus",
    ("trigger", "minus"): "smoothSF3D_triggerminus",
    ("iso", None): "smoothSF3D_iso",
}


def load_sf3d(path):
    """The default analysis' smoothed 3D (eta, pt, uT) scale factors.

    These are what muon_efficiencies_smooth actually applies, unlike the 2D
    tag-and-probe efficiencies, which are uT-integrated. Their uT axis spans
    exactly insitu_ut_range and their 48 eta bins are ours, so a slice can be
    drawn directly against our bands.
    """
    if path in _sf3d_cache:
        return _sf3d_cache[path]
    with lz4.frame.open(path, "rb") as f:
        _sf3d_cache[path] = pickle.load(f)
    return _sf3d_cache[path]


def eval_sf3d(d, step, charge_tag, i_eta, pt_grid, ut, iso_key="smoothSF3D_iso"):
    """External 3D SF for one (step, charge, eta) at fixed uT, on pt_grid."""
    key = SF3D_KEY.get((step, charge_tag if step == "trigger" else None))
    if step == "iso":
        key = iso_key
    if key is None or key not in d:
        return None
    h = d[key]
    pt_c = h.axes["pt"].centers
    ut_c = h.axes["ut"].centers
    j = int(np.argmin(np.abs(ut_c - ut)))
    vals = h.values()[i_eta, :, j, 0]  # nominal eigenvariation
    return np.interp(pt_grid, pt_c, vals)


def load_tnp_sf(tnp_dir, step, charge_tag):
    """External T&P scale factor SF = EffData2D / EffMC2D for one (step, charge).

    Returns (pt_centers[npt], pt_halfwidths[npt], sf[n_eta, npt], err[n_eta, npt])
    with eta on the first axis (the 48-bin grid, identical to ours) and pt on
    the second, or None if the file is missing. Cached per (step, charge).
    """
    import ROOT

    key = (tnp_dir, step, charge_tag)
    if key in _tnp_cache:
        return _tnp_cache[key]
    path = os.path.join(
        tnp_dir, f"allEfficiencies_2D_{TNP_FILE[step]}_{charge_tag}.root"
    )
    if not os.path.isfile(path):
        logger.warning(f"T&P file not found, skipping overlay: {path}")
        _tnp_cache[key] = None
        return None
    f = ROOT.TFile.Open(path)
    eff = {}
    for which, hname in (("mc", "EffMC2D"), ("data", "EffData2D")):
        h2 = f.Get(hname)
        n_eta, n_pt = h2.GetNbinsX(), h2.GetNbinsY()
        pt_c = np.array([h2.GetYaxis().GetBinCenter(j + 1) for j in range(n_pt)])
        pt_hw = np.array([0.5 * h2.GetYaxis().GetBinWidth(j + 1) for j in range(n_pt)])
        e = np.array(
            [
                [h2.GetBinContent(i + 1, j + 1) for j in range(n_pt)]
                for i in range(n_eta)
            ]
        )
        u = np.array(
            [[h2.GetBinError(i + 1, j + 1) for j in range(n_pt)] for i in range(n_eta)]
        )
        eff[which] = (pt_c, pt_hw, e, u)
    f.Close()
    pt_c, pt_hw, ed, errd = eff["data"]
    _, _, em, errm = eff["mc"]
    with np.errstate(divide="ignore", invalid="ignore"):
        sf = ed / em
        rel = np.sqrt((errd / ed) ** 2 + (errm / em) ** 2)
        err = np.abs(sf) * rel
    sf = np.where(np.isfinite(sf), sf, np.nan)
    err = np.where(np.isfinite(err), err, 0.0)
    out = (pt_c, pt_hw, sf, err)
    _tnp_cache[key] = out
    return out


def ylim_for(step, args):
    """Per-step y range, falling back to --ylim."""
    for entry in args.ylimStep:
        name, _, rng = entry.partition("=")
        if name == step:
            lo, _, hi = rng.partition(",")
            return float(lo), float(hi)
    return tuple(args.ylim)


def build_blocks(fitresult, prev_file, pt_range):
    """Per (step, eta, charge) Chebyshev coefficients theta and covariance.

    Returns (n_eta, blocks) where blocks[step] is a dict keyed by (eta, charge
    or None) -> (theta[ncoeff], cov[ncoeff, ncoeff]). theta already includes the
    previous central value; cov = delta^2 * cov(n_hat) (the fixed central offset
    does not contribute to the uncertainty).
    """
    parms = fitresult["parms"].get()
    labels = np.array(parms.axes["parms"]).astype(str)
    nhat = parms.values()
    # A fit run with --noHessian has no covariance. That is fine for the
    # comparison overlay, which is drawn as a dashed line with no band; only
    # the band of the primary input needs one, and main() checks for it there.
    if "cov" in fitresult:
        cov = fitresult["cov"].get().values()
    else:
        cov = np.zeros((len(labels), len(labels)))
        logger.info("No covariance in this fitresult; curves drawn without a band")
    idx = {l: i for i, l in enumerate(labels)}

    n_eta = 1 + max(
        int(l.split("_eta")[1].split("_")[0])
        for l in labels
        if l.startswith("effInsituIso_eta")
    )
    prev = dict(
        zip(
            insitu_parameter_labels(n_eta, insitu_n_coeff_pt, insitu_n_coeff_ut),
            load_insitu_central(prev_file, n_eta, insitu_n_coeff_pt, insitu_n_coeff_ut),
        )
    )

    blocks = {}
    for step, info in STEP_INFO.items():
        charges = ("minus", "plus") if info["charge"] else (None,)
        blocks[step] = {}
        for b in range(n_eta):
            for q in charges:
                names = block_names(step, b, q, info["has_ut"])
                ids = np.array([idx[n] for n in names])
                theta = np.array([prev[n] for n in names]) + insitu_delta * nhat[ids]
                cblock = insitu_delta**2 * cov[np.ix_(ids, ids)]
                blocks[step][(b, q)] = (theta, cblock)
    return n_eta, blocks


def block_names(step, b, q, has_ut):
    """Coefficient nuisance names of one (step, eta, charge) cell, in the
    insitu_parameter_labels order."""
    pref = STEP_INFO[step]["prefix"]
    nk, nm = insitu_n_coeff_pt, (insitu_n_coeff_ut if has_ut else 1)
    names = []
    for k in range(nk):
        for m in range(nm):
            n = f"{pref}_eta{b}"
            if q is not None:
                n += f"_q{q}"
            n += f"_cPt{k}"
            if has_ut:
                n += f"_cUt{m}"
            names.append(n)
    return names


def build_blocks_from_sf(sf_file):
    """Blocks straight from a theta_central pkl (the accumulated SF), with zero
    covariance -> SF curves without an uncertainty band. Useful to plot a
    produced SF without rerunning/locating the fit."""
    with lz4.frame.open(sf_file, "rb") as fin:
        payload = pickle.load(fin)
    content = next(
        v
        for k, v in payload.items()
        if k not in ("meta_data", "file_meta_data", "meta_info")
    )
    theta_d = content["theta_central"]
    n_eta = 1 + max(
        int(l.split("_eta")[1].split("_")[0])
        for l in theta_d
        if l.startswith("effInsituIso_eta")
    )
    blocks = {}
    for step, info in STEP_INFO.items():
        charges = ("minus", "plus") if info["charge"] else (None,)
        blocks[step] = {}
        for b in range(n_eta):
            for q in charges:
                names = block_names(step, b, q, info["has_ut"])
                theta = np.array([theta_d[n] for n in names])
                blocks[step][(b, q)] = (theta, np.zeros((len(names), len(names))))
    return n_eta, blocks


def sf_band(theta, cblock, pt_grid, ut, pt_range, ut_range, has_ut):
    """data/MC SF(pt) = 1 + P and sigma_SF(pt) at fixed uT (linear param, full cov).

    The passing-leg SF is the polynomial directly, SF = 1 + P with
    P = sum_c theta_c B_c(pt[,uT]); it is MC-independent. SF is linear in theta,
    so d SF/d theta_c = B_c and sigma_SF = sqrt(B^T Cov B).
    """
    xpt = norm(pt_grid, *pt_range)
    Tp = cheb(xpt, insitu_n_coeff_pt)  # (nk, npt)
    if has_ut:
        Tu = cheb(norm(ut, *ut_range), insitu_n_coeff_ut)  # (nm,)
        B = (Tp[:, None, :] * Tu[None, :, None]).reshape(-1, pt_grid.size)  # (nc, npt)
    else:
        B = Tp
    sf = 1.0 + theta @ B
    var = np.einsum("ip,ij,jp->p", B, cblock, B)
    return sf, np.sqrt(np.maximum(var, 0.0))


def report_sf_difference(n_eta, blocks, blocks_cmp, args):
    """Largest SF difference between the two fits, in absolute terms and in
    units of the first fit's uncertainty.

    Two coefficient sets can differ a lot while describing the same scale
    factor, so the difference that matters is the one in SF space, measured
    against the band.
    """
    plot_range = args.ptPlotRange if args.ptPlotRange is not None else args.ptRange
    pt_grid = np.linspace(plot_range[0], plot_range[1], 120)
    for step, info in STEP_INFO.items():
        worst_abs, worst_pull, where = 0.0, 0.0, ""
        charges = (None,) if not info["charge"] else ("minus", "plus")
        for b in range(n_eta):
            for q in charges:
                theta, cblock = blocks[step][(b, q)]
                theta2, _ = blocks_cmp[step][(b, q)]
                for ut in (args.utSlices if info["has_ut"] else [0.0]):
                    sf1, sig = sf_band(
                        theta,
                        cblock,
                        pt_grid,
                        ut,
                        args.ptRange,
                        insitu_ut_range,
                        info["has_ut"],
                    )
                    sf2, _ = sf_band(
                        theta2,
                        cblock,
                        pt_grid,
                        ut,
                        args.ptRange,
                        insitu_ut_range,
                        info["has_ut"],
                    )
                    d = np.abs(sf1 - sf2)
                    pull = d / np.where(sig > 0, sig, np.inf)
                    if d.max() > worst_abs:
                        worst_abs = d.max()
                        where = f"eta bin {b}" + (f" q{q}" if q else "")
                    worst_pull = max(worst_pull, pull.max())
        logger.info(
            f"{step}: max |SF1-SF2| = {worst_abs:.5f} ({where}), "
            f"max |SF1-SF2|/sigma = {worst_pull:.3f}"
        )


def make_plots(n_eta, blocks, args, blocks_cmp=None):
    outdir = output_tools.make_plot_dir(*args.plotdir.rsplit("/", 1), eoscp=args.eoscp)
    eta_edges = np.linspace(-2.4, 2.4, n_eta + 1)
    plot_range = args.ptPlotRange if args.ptPlotRange is not None else args.ptRange
    pt_grid = np.linspace(plot_range[0], plot_range[1], 120)
    ut_slices = args.utSlices
    ut_cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(ut_slices)))
    sf3d = load_sf3d(args.smoothSF3DFile) if args.smoothSF3DFile else None

    for step, info in STEP_INFO.items():
        has_ut = info["has_ut"]
        step_dir = os.path.join(outdir, step, "per_cell")
        os.makedirs(step_dir, exist_ok=True)
        charges = (-1, +1) if info["charge"] else (None,)
        for i_eta in range(n_eta):
            eta_lo, eta_hi = eta_edges[i_eta], eta_edges[i_eta + 1]
            eta_text = f"${eta_lo:+.2f}\\,\\leq\\,\\eta\\,<\\,{eta_hi:+.2f}$"
            for qkey in charges:
                q = None if qkey is None else CHARGE_TAGS[qkey][0]
                theta, cblock = blocks[step][(i_eta, q)]
                theta_cmp = (
                    None if blocks_cmp is None else blocks_cmp[step][(i_eta, q)][0]
                )
                fig, ax = plt.subplots(figsize=(8, 6))

                if has_ut:
                    for ut, col in zip(ut_slices, ut_cmap):
                        sf, sig = sf_band(
                            theta,
                            cblock,
                            pt_grid,
                            ut,
                            args.ptRange,
                            insitu_ut_range,
                            True,
                        )
                        ax.plot(pt_grid, sf, color=col, label=f"$u_T={ut:g}$ GeV")
                        ax.fill_between(
                            pt_grid, sf - sig, sf + sig, color=col, alpha=0.25, lw=0
                        )
                        # the default analysis' smoothed 3D SF at the SAME uT, dashed in the
                        # same colour: like-for-like, unlike the uT-integrated T&P points
                        if sf3d is not None:
                            ext = eval_sf3d(
                                sf3d, step, q, i_eta, pt_grid, ut, args.sf3dIsoKey
                            )
                            if ext is not None:
                                ax.plot(
                                    pt_grid,
                                    ext,
                                    color=col,
                                    ls="--",
                                    lw=1.4,
                                    label=(
                                        "smoothed 3D SF" if ut == ut_slices[0] else None
                                    ),
                                )
                        if theta_cmp is not None:
                            sf2, _ = sf_band(
                                theta_cmp,
                                cblock,
                                pt_grid,
                                ut,
                                args.ptRange,
                                insitu_ut_range,
                                True,
                            )
                            ax.plot(pt_grid, sf2, color=col, ls="--", lw=1.4)
                else:
                    sf, sig = sf_band(
                        theta,
                        cblock,
                        pt_grid,
                        0.0,
                        args.ptRange,
                        insitu_ut_range,
                        False,
                    )
                    ax.plot(pt_grid, sf, color="#5790FC", label="in-situ SF")
                    if theta_cmp is not None:
                        sf2, _ = sf_band(
                            theta_cmp,
                            cblock,
                            pt_grid,
                            0.0,
                            args.ptRange,
                            insitu_ut_range,
                            False,
                        )
                        ax.plot(
                            pt_grid,
                            sf2,
                            color="#E42536",
                            ls="--",
                            lw=1.4,
                            label=args.compareLabel,
                        )
                    if cblock.any():  # no band when plotting from --sfFile (no cov)
                        ax.fill_between(
                            pt_grid,
                            sf - sig,
                            sf + sig,
                            color="#5790FC",
                            alpha=0.3,
                            lw=0,
                            label=r"$\pm\,\sigma_{SF}$ (full cov)",
                        )

                # external T&P SF overlay (uT-integrated 2D), reference points
                if args.tagAndProbeDir:
                    tag = "both" if q is None else q
                    tnp = load_tnp_sf(args.tagAndProbeDir, step, tag)
                    if tnp:
                        pt_c, pt_hw, sf_t, err_t = tnp
                        lbl = "T&P SF" + (" ($u_T$ incl.)" if has_ut else "")
                        ax.errorbar(
                            pt_c,
                            sf_t[i_eta],
                            yerr=err_t[i_eta],
                            xerr=pt_hw,
                            label=lbl,
                            **TNP_STYLE,
                        )

                annot = eta_text if q is None else f"{eta_text}\n{CHARGE_TAGS[qkey][1]}"
                suffix = "" if q is None else f"_q{q}"
                ax.text(
                    0.04,
                    0.06,
                    annot,
                    transform=ax.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=18,
                )
                ax.axhline(1.0, ls="--", color="gray", lw=0.8, zorder=0)
                ax.set_xlim(*plot_range)
                ax.set_ylim(*ylim_for(step, args))
                ax.set_xlabel(r"$p_T^\mu$ [GeV]")
                ax.set_ylabel(f"data/MC SF ({info['ylabel']})")
                ax.legend(loc="upper right", frameon=False, fontsize=12, ncol=2)
                plot_tools.add_decor(
                    ax, "CMS", "Preliminary", data=True, lumi=16.8, loc=0
                )
                name = f"effSF_{step}_pt_eta{i_eta:02d}{suffix}"
                plot_tools.save_pdf_and_png(step_dir, name)
                plt.close(fig)
                output_tools.write_index_and_log(step_dir, name, args=args)
        logger.info(f"Wrote {step} SF band plots to {step_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "-i",
        "--inputFile",
        default=None,
        help="rabbit fitresults_shapes.hdf5 (must contain the parameter "
        "covariance, i.e. fit run with the Hessian). Draws the SF with its "
        "full-covariance band. Mutually exclusive with --sfFile.",
    )
    p.add_argument(
        "--sfFile",
        default=None,
        help="accumulated theta_central pkl (insitu_effSF*.pkl.lz4) to plot the "
        "SF curves directly, without a fit result (no covariance band). Use to "
        "inspect a produced SF, e.g. for high-uT singularities.",
    )
    p.add_argument(
        "--prevSFFile",
        default=None,
        help="theta_central pkl of the iteration that produced this fit "
        "(None -> zeros, iteration 0)",
    )
    p.add_argument(
        "--smoothSF3DFile",
        default=None,
        help="smoothSF3D_*.pkl.lz4 holding the default analysis' 3D "
        "(eta, pt, uT) scale factors -- what muon_efficiencies_smooth actually "
        "applies. Drawn dashed at the same uT slices as our bands, which is the "
        "like-for-like comparison; --tagAndProbeDir gives uT-integrated points.",
    )
    p.add_argument(
        "--sf3dIsoKey",
        default="smoothSF3D_iso",
        help="iso key to compare against: smoothSF3D_iso (triggering muon) or "
        "smoothSF3D_isonotrig (no trigger requirement)",
    )
    p.add_argument(
        "--tagAndProbeDir",
        default=None,
        help="dir with allEfficiencies_2D_<step>_<charge>.root for the "
        "external T&P SF overlay",
    )
    p.add_argument(
        "--compareFile",
        default=None,
        help="second rabbit fitresults to overlay as dashed curves, to compare "
        "two fits of the same model (e.g. two minima). The band shown is always "
        "the one from -i/--inputFile.",
    )
    p.add_argument(
        "--compareLabel", default="comparison", help="legend label for --compareFile"
    )
    p.add_argument("--plotdir", required=True, help="output plot directory")
    p.add_argument(
        "--ptRange",
        type=float,
        nargs=2,
        default=list(insitu_pt_range),
        help="Chebyshev pt WINDOW used in production, i.e. the xtilde transform "
        "(default: the central insitu_pt_range). Must match production; do NOT "
        "set this to the fit range or the SF is mis-normalised.",
    )
    p.add_argument(
        "--ptPlotRange",
        type=float,
        nargs=2,
        default=None,
        help="Displayed pt x-axis range (default: --ptRange). Use to zoom (e.g. "
        "to the fit range 26 65) without changing the Chebyshev window.",
    )
    p.add_argument(
        "--utSlices",
        type=float,
        nargs="+",
        default=[-10.0, 0.0, 20.0],
        help="uT values (GeV) at which to draw the HLT/Iso SF bands",
    )
    p.add_argument("--ylim", type=float, nargs=2, default=[0.7, 1.3])
    p.add_argument(
        "--ylimStep",
        nargs="+",
        default=[],
        metavar="STEP=LO,HI",
        help="per-step y range overriding --ylim, e.g. 'iso=0.9,1.05 idip=0.9,1.05'. "
        "The steps differ by an order of magnitude in how far the SF moves, so one "
        "range for all of them either clips the trigger or leaves iso and idip in a "
        "flat band.",
    )
    p.add_argument("--result", default=None, help="named fit result to read")
    p.add_argument(
        "--eoscp",
        action="store_true",
        help="copy to eos with xrdcp instead of the mount",
    )
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()
    logging.setup_logger("plot_insitu_effSF", 4 if args.debug else 3)
    if (args.inputFile is None) == (args.sfFile is None):
        raise ValueError("provide exactly one of -i/--inputFile or --sfFile")
    if args.sfFile is not None:
        n_eta, blocks = build_blocks_from_sf(args.sfFile)
        logger.info(f"Plotting SF from {args.sfFile} (no covariance band)")
    else:
        fitresult = get_fitresult(args.inputFile, result=args.result)
        if "cov" not in fitresult:
            raise ValueError(
                f"{args.inputFile} has no parameter covariance; rerun the fit with "
                "the Hessian (use the *_shapes.hdf5 output)."
            )
        n_eta, blocks = build_blocks(fitresult, args.prevSFFile, args.ptRange)

    blocks_cmp = None
    if args.compareFile is not None:
        cmp_result = get_fitresult(args.compareFile, result=args.result)
        _, blocks_cmp = build_blocks(cmp_result, args.prevSFFile, args.ptRange)
        logger.info(f"Overlaying {args.compareFile} as {args.compareLabel}")
        report_sf_difference(n_eta, blocks, blocks_cmp, args)

    make_plots(n_eta, blocks, args, blocks_cmp=blocks_cmp)


if __name__ == "__main__":
    main()
