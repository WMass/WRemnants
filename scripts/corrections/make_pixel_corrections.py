"""Pixel multiplicity correction from the output of the dilepton histmaker.

Stores, for wremnants.production.muon_calibration.make_pixel_multiplicity_helpers:
  - the valid pixel hit multiplicity of the triggering and the non-triggering muon in
    data and MC (hNValidPixelHits{Trig,NonTrig}_{data,mc}), from which the fractions
    of muons without a valid pixel hit and the weights w0, w1 follow;
  - the probability of no valid pixel hit against the generated pt and the reco/gen
    curvature ratio of a muon (hZeroPixelProb), from the response of the MC muons
    with and without a valid pixel hit (hPixelResponse{Trig,NonTrig}), for the
    response-based weights; see wremnants/postprocessing/pixel_multiplicity_utils.py.
"""

import argparse
import pickle

import hist
import lz4.frame
import numpy as np

from wremnants.postprocessing import pixel_multiplicity_utils
from wremnants.postprocessing.datagroups.datagroups import Datagroups
from wums import logging

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--inputFile", type=str, required=True)
parser.add_argument(
    "-o",
    "--outfile",
    type=str,
    default="pixelcorr.pkl.lz4",
    help="Output file",
)
parser.add_argument(
    "--inputFilePixelHits",
    type=str,
    default=None,
    help="""Dilepton histmaker output to take the pixel hit multiplicities of data and MC
    from, if not the input file (which then needs the Z MC only)""",
)
parser.add_argument(
    "--noResponse",
    action="store_true",
    help="Store the pixel hit multiplicities only, without the probability map of the response-based weights",
)
parser.add_argument(
    "--genPtFitRange",
    type=float,
    nargs=2,
    default=[34.0, 50.0],
    help="Generated-pt range in which the response of the two classes of muons is fitted",
)
parser.add_argument("--debug", action="store_true", help="Print debug output")
args = parser.parse_args()

logger = logging.setup_logger("make_pixel_corrections", 4 if args.debug else 3)


def load_hists(filename, histnames, procs):
    """Histograms of the dilepton histmaker output per group, {group: {name: hist}}"""
    datagroups = Datagroups(filename, splitGroups=True)
    if datagroups.mode != "z_dilepton":
        raise ValueError("Expected input is the output from the dilepton histmaker")
    for histname in histnames:
        datagroups.loadHistsForDatagroups(histname, syst="", procsToRead=procs)
    return {p: datagroups.groups[p].hists for p in procs}


mc_procs = ["Zmumu", "Ztautau"]
histnames_hits = ["hNValidPixelHitsTrig", "hNValidPixelHitsNonTrig"]
# the response is that of the generated muons: MC only
histnames_response = (
    [] if args.noResponse else ["hPixelResponseTrig", "hPixelResponseNonTrig"]
)

if args.inputFilePixelHits is None:
    hists = load_hists(args.inputFile, histnames_hits, ["Data", *mc_procs])
    hists_response = load_hists(args.inputFile, histnames_response, mc_procs)
else:
    hists = load_hists(args.inputFilePixelHits, histnames_hits, ["Data", *mc_procs])
    hists_response = load_hists(args.inputFile, histnames_response, mc_procs)


def mc_hist(source, name):
    return source["Zmumu"][name] + source["Ztautau"][name]


res = {
    "hNValidPixelHitsTrig_data": hists["Data"]["hNValidPixelHitsTrig"],
    "hNValidPixelHitsNonTrig_data": hists["Data"]["hNValidPixelHitsNonTrig"],
    "hNValidPixelHitsTrig_mc": mc_hist(hists, "hNValidPixelHitsTrig"),
    "hNValidPixelHitsNonTrig_mc": mc_hist(hists, "hNValidPixelHitsNonTrig"),
}

for name, h in res.items():
    v = h[{"pt": hist.sum, "eta": hist.sum, "charge": hist.sum}].values()
    logger.info(
        f"{name}: {v.sum():.4g} muons, {100 * v[0] / v.sum():.2f}% without a valid pixel hit"
    )

if not args.noResponse:
    hresp = [
        mc_hist(hists_response, "hPixelResponseNonTrig"),
        mc_hist(hists_response, "hPixelResponseTrig"),
    ]
    # leading axis in the order of wrem::TriggerCat, as in make_pixel_multiplicity_helpers
    axis_nontrig_trig = hist.axis.Integer(
        0, 2, name="nontrig_trig", underflow=False, overflow=False
    )
    hresponse = hist.Hist(
        axis_nontrig_trig, *hresp[0].axes, storage=hresp[0].storage_type()
    )
    for i, h in enumerate(hresp):
        hresponse.view(flow=True)[i] = h.view(flow=True)

    pt_edges = res["hNValidPixelHitsTrig_mc"].axes["pt"].edges
    hprob, info = pixel_multiplicity_utils.make_zero_pixel_probability(
        hresponse,
        pt_range=(pt_edges[0], pt_edges[-1]),
        genpt_fit_range=args.genPtFitRange,
    )
    res["hZeroPixelProb"] = hprob
    res["zeroPixelProbModel"] = info
    # the response of both trigger categories and charges, for checks of the model
    res["hPixelResponse_mc"] = hresponse[{"nontrig_trig": hist.sum, "charge": hist.sum}]

    eta = 0.5 * (info["eta_edges"][1:] + info["eta_edges"][:-1])
    pt = 40.0
    sigma1 = np.sqrt(info["sigma1_a2"] + info["sigma1_b2"] * pt**2)
    delta = np.sqrt(info["delta_a2"] + info["delta_b2"] * pt**2)

    def ratio(chi2, ndf):
        return chi2.sum() / ndf.sum() if ndf.sum() > 0 else float("nan")

    for ie in range(len(eta)):
        logger.debug(
            f"eta {eta[ie]:+.2f}: sigma1 {100 * sigma1[ie]:.2f}%, delta {100 * delta[ie]:.2f}%, shift {100 * info['mu'][ie]:+.3f}%, "
            f"chi2/ndf of the fits {ratio(info['chi2_fit'][ie : ie + 1], info['ndf_fit'][ie : ie + 1]):.2f}, "
            f"against all bins {ratio(info['chi2_all'][ie : ie + 1], info['ndf_all'][ie : ie + 1]):.2f}"
        )
    logger.info(
        f"Response model at generated pt {pt:g} GeV, per |eta| range: width with a valid pixel hit, extra width "
        "without, shift, chi2/ndf of the fits and of the probability against all generated-pt and qopr bins"
    )
    for lo, hi in [
        (0.0, 0.8),
        (0.8, 1.6),
        (1.6, 2.0),
        (2.0, 2.1),
        (2.1, 2.2),
        (2.2, 2.3),
        (2.3, 2.4),
    ]:
        m = (np.abs(eta) >= lo) & (np.abs(eta) < hi)
        logger.info(
            f"  {lo:.1f} <= |eta| < {hi:.1f}: sigma1 {100 * sigma1[m].mean():.2f}%, delta {100 * delta[m].mean():.2f}%, "
            f"shift {100 * info['mu'][m].mean():+.3f}%, chi2/ndf {ratio(info['chi2_fit'][m], info['ndf_fit'][m]):.2f}, "
            f"{ratio(info['chi2_all'][m], info['ndf_all'][m]):.2f}; "
            f"muons without a valid pixel hit {100 * info['frac0_observed'][:, m].mean():.2f}%"
        )
    logger.info(
        f"All eta bins: chi2/ndf of the fits {ratio(info['chi2_fit'], info['ndf_fit']):.2f}, "
        f"of the probability against all bins {ratio(info['chi2_all'], info['ndf_all']):.2f}"
    )

res["meta"] = {
    "inputFile": args.inputFile,
    "inputFilePixelHits": args.inputFilePixelHits,
    "genPtFitRange": args.genPtFitRange,
}

with lz4.frame.open(args.outfile, "wb") as fout:
    pickle.dump(res, fout, protocol=pickle.HIGHEST_PROTOCOL)
logger.info(f"Written {args.outfile}")
