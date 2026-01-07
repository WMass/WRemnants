#!/usr/bin/env python3
import argparse
import os
import pickle
import sys
from array import array

import hist
import lz4.frame
import ROOT
import tensorflow as tf

# from narf import histutils
import narf
import wums.output_tools
from utilities import common
from wremnants.datasets.datagroups import Datagroups
from wums import boostHistHelpers as hh
from wums import logging

args = sys.argv[:]
sys.argv = ["-b"]
sys.argv = args

ROOT.gROOT.SetBatch(True)
ROOT.PyConfig.IgnoreCommandLineOptions = True

from scripts.analysisTools.plotUtils.utility import (
    safeOpenFile,
)


def pol4_root(xvals, parms, xLowVal=0.0, xFitRange=1.0):
    xscaled = (xvals[0] - xLowVal) / xFitRange
    return (
        parms[0]
        + parms[1] * xscaled
        + parms[2] * xscaled**2
        + parms[3] * xscaled**3
        + parms[4] * xscaled**4
    )


def crystal_ball_right_tf(xvals, parms):
    # parms: [A, MPV, sigma, alpha, n]
    x = tf.convert_to_tensor(xvals[0])
    A = tf.cast(parms[0], x.dtype)
    MPV = tf.cast(parms[1], x.dtype)
    sigma = tf.maximum(tf.cast(parms[2], x.dtype), 1e-8)
    alpha = tf.cast(parms[3], x.dtype)
    n = tf.cast(parms[4], x.dtype)

    t = (x - MPV) / sigma

    # constants for continuity
    # for right-tail CrystalBall: tail when t > alpha
    # tail shape: A * ( (n/alpha)^n * exp(-alpha^2/2) ) / ( (n/alpha) + t )^n
    # gaussian part: A * exp(-t^2/2)
    prefactor = tf.pow(n / alpha, n) * tf.exp(-0.5 * alpha * alpha)

    gauss = A * tf.exp(-0.5 * t * t)
    tail = A * prefactor / tf.pow((n / alpha) + t, n)

    return tf.where(t > alpha, tail, gauss)


def convert_binEdges_idx(ed_list, binning):
    low, high = 0, 0
    for binEdge in binning:
        if binEdge + 0.01 < ed_list[0]:
            low += 1
        if binEdge + 0.01 < ed_list[1]:
            high += 1
    return (low, high)


parser = argparse.ArgumentParser()

parser.add_argument(
    "-i",
    "--infile",
    type=str,
    default="/scratch/rforti/wremnants_hists/mw_with_mu_eta_pt_scetlib_dyturboCorr_maxFiles_m1_x0p50_y3p00_THAGNV0_fromSV.hdf5",
    help="Input file with histograms",
)
parser.add_argument("-o", "--outdir", type=str, default="./", help="Output directory")
parser.add_argument(
    "-p",
    "--postfix",
    type=str,
    default=None,
    help="Postfix appended to output file name",
)
parser.add_argument(
    "--nEtaBins",
    type=int,
    default=1,
    choices=[1, 2, 3, 4, 6, 8],
    help="Number of eta bins where to evaluate the templates",
)
parser.add_argument(
    "--nChargeBins",
    type=int,
    default=1,
    choices=[1, 2],
    help="Number of charge bins where to evaluate the templates",
)
parser.add_argument(
    "--doQCD",
    action="store_true",
    help="Make templates with QCD instead of nonprompt contribution",
)

args = parser.parse_args()

logger = logging.setup_logger(os.path.basename(__file__), 4)
ROOT.TH1.SetDefaultSumw2()

groupsToConsider = (
    [
        "Data",
        "Wmunu",
        "Wtaunu",
        "Diboson",
        "Zmumu",
        "DYlowMass",
        "PhotonInduced",
        "Ztautau",
        "Top",
    ]
    if not args.doQCD
    else ["QCD"]
)

eta_genBinning = array("d", [round(-2.4 + 0.1 * i, 1) for i in range(49)])
charge_genBinning = array("d", [-2, 0, 2])

delta_eta = (eta_genBinning[-1] - eta_genBinning[0]) / args.nEtaBins
delta_ch = (charge_genBinning[-1] - charge_genBinning[0]) / args.nChargeBins

decorrBins_eta = [
    (
        round((eta_genBinning[0] + i * delta_eta), 1),
        round((eta_genBinning[0] + (i + 1) * delta_eta), 1),
    )
    for i in range(args.nEtaBins)
]
decorrBins_ch = [
    (
        round((charge_genBinning[0] + i * delta_ch), 1),
        round((charge_genBinning[0] + (i + 1) * delta_ch), 1),
    )
    for i in range(args.nChargeBins)
]

logger.info(f"Decorrelating in the eta bins: {decorrBins_eta}")
logger.info(f"Decorrelating in the charge bins: {decorrBins_ch}")

groups = Datagroups(
    args.infile,
    filterGroups=groupsToConsider,
    excludeGroups=None,
)

# There is probably a better way to do this but I don't want to deal with it
datasets = groups.getNames()
logger.info(f"Will work on datasets {datasets}")

exclude = ["Data"] if not args.doQCD else []
prednames = list(
    groups.getNames(
        [d for d in datasets if d not in exclude], exclude=False, match_exact=True
    )
)

logger.info(f"Unstacked processes are {exclude}")
logger.info(f"Stacked processes are {prednames}")

histInfo = groups.groups

select_utMinus = {"utAngleSign": hist.tag.Slicer()[0 : 1 : hist.sum]}
select_utPlus = {"utAngleSign": hist.tag.Slicer()[1 : 2 : hist.sum]}

groups.set_histselectors(
    datasets,
    "nominal",
    smoothing_mode="full",
    smoothingOrderSpectrum=3,
    smoothingPolynomialSpectrum="chebyshev",
    integrate_x=all("mt" not in x.split("-") for x in ["pt"]),
    mode="extended1D",
    forceGlobalScaleFakes=None,
    mcCorr=[None],
)

groups.setNominalName("nominal")
groups.loadHistsForDatagroups(
    "nominal", syst="", procsToRead=datasets, applySelection=True
)

out_hist = ROOT.TH3D(
    f"fakeRatio_utAngleSign_{'Data' if not args.doQCD else 'QCD'}",
    "",
    len(eta_genBinning) - 1,
    eta_genBinning,
    30,
    array("d", [round(26.0 + 1.0 * i, 1) for i in range(31)]),
    len(charge_genBinning) - 1,
    charge_genBinning,
)

for ch_edges in decorrBins_ch:
    for eta_edges in decorrBins_eta:

        ch_low_idx, ch_high_idx = convert_binEdges_idx(ch_edges, charge_genBinning)
        eta_low_idx, eta_high_idx = convert_binEdges_idx(eta_edges, eta_genBinning)

        logger.info(f"{ch_low_idx}, {ch_high_idx}")
        logger.info(f"{eta_low_idx}, {eta_high_idx}")

        select_utMinus["charge"] = hist.tag.Slicer()[
            ch_low_idx : ch_high_idx : hist.sum
        ]
        select_utMinus["eta"] = hist.tag.Slicer()[eta_low_idx : eta_high_idx : hist.sum]

        select_utPlus["charge"] = hist.tag.Slicer()[ch_low_idx : ch_high_idx : hist.sum]
        select_utPlus["eta"] = hist.tag.Slicer()[eta_low_idx : eta_high_idx : hist.sum]

        logger.info(f"Processing charge bin [{ch_edges}] and eta bin [{eta_edges}]")

        boost_h_utMinus = histInfo["Data"].copy("Data_utMinus").hists["nominal"]
        boost_h_utMinus = boost_h_utMinus[select_utMinus]
        boost_h_utMinus = hh.projectNoFlow(boost_h_utMinus, ["pt"], ["relIso", "mt"])
        root_h_utMinus = narf.hist_to_root(boost_h_utMinus)

        boost_h_utPlus = histInfo["Data"].copy("Data_utPlus").hists["nominal"]
        boost_h_utPlus = boost_h_utPlus[select_utPlus]
        boost_h_utPlus = hh.projectNoFlow(boost_h_utPlus, ["pt"], ["relIso", "mt"])
        root_h_utPlus = narf.hist_to_root(boost_h_utPlus)

        logger.info(f"Integrals BEFORE prompt subraction (uT < 0, uT > 0)")
        logger.info(f"{root_h_utMinus.Integral()}, {root_h_utPlus.Integral()}")

        for mcName in prednames:
            if args.doQCD:
                continue
            logger.info(f"Subtracting {mcName} from data")
            boost_h_mc_utMinus = (
                histInfo[mcName].copy(f"{mcName}_utMinus").hists["nominal"]
            )
            boost_h_mc_utMinus = boost_h_mc_utMinus[select_utMinus]
            boost_h_mc_utMinus = hh.projectNoFlow(
                boost_h_mc_utMinus, ["pt"], ["relIso", "mt"]
            )
            root_h_mc_utMinus = narf.hist_to_root(boost_h_mc_utMinus)
            root_h_utMinus.Add(root_h_mc_utMinus, -1)

            boost_h_mc_utPlus = (
                histInfo[mcName].copy(f"{mcName}_utPlus").hists["nominal"]
            )
            boost_h_mc_utPlus = boost_h_mc_utPlus[select_utPlus]
            boost_h_mc_utPlus = hh.projectNoFlow(
                boost_h_mc_utPlus, ["pt"], ["relIso", "mt"]
            )
            root_h_mc_utPlus = narf.hist_to_root(boost_h_mc_utPlus)
            root_h_utPlus.Add(root_h_mc_utPlus, -1)

        logger.info(f"Integrals AFTER prompt subraction (uT < 0, uT > 0)")
        logger.info(f"{root_h_utMinus.Integral()}, {root_h_utPlus.Integral()}")

        ratio_h = root_h_utMinus.Clone(f"fakeRatio_utAngleSign")
        ratio_h.Sumw2()
        ratio_h.Divide(root_h_utPlus)

        for idx_ch in range(ch_low_idx + 1, ch_high_idx + 1):
            for idx_eta in range(eta_low_idx + 1, eta_high_idx + 1):
                # logger.debug(f"Setting weights for chBin={idx_ch}, etaBin={idx_eta}")
                for idx_pt in range(1, 31):
                    out_hist.SetBinContent(
                        idx_eta, idx_pt, idx_ch, ratio_h.GetBinContent(idx_pt)
                    )
                    out_hist.SetBinError(
                        idx_eta, idx_pt, idx_ch, ratio_h.GetBinError(idx_pt)
                    )
                    if ratio_h.GetBinContent(idx_pt) <= 0.0:
                        logger.info(
                            "WARNING - found negative value in bin: ({idx_eta}, {idx_pt}, {idx_ch})"
                        )


boost_out_hist = narf.root_to_hist(out_hist)
resultDict = {"fakeCorr": boost_out_hist}
base_dir = common.base_dir
resultDict.update(
    {"meta_info": wums.output_tools.make_meta_info_dict(args=args, wd=base_dir)}
)

outfileName = "fakeTransferTemplates"
if args.postfix:
    outfileName += f"_{args.postfix}"
if args.doQCD:
    outfileName += "_QCD"

pklfileName = f"{args.outdir}/{outfileName}.pkl.lz4"
with lz4.frame.open(pklfileName, "wb") as fout:
    pickle.dump(resultDict, fout, protocol=pickle.HIGHEST_PROTOCOL)
logger.warning(f"Created file {pklfileName}")

rootfileName = f"{args.outdir}/{outfileName}.root"
fout = safeOpenFile(rootfileName, mode="RECREATE")
out_hist.Write()
fout.Close()
logger.warning(f"Created file {rootfileName}")


"""
x_axis = hist.axis.Regular(30, 26, 56, name="pt", flow=False)

tr_hist = hist.Hist(x_axis, storage=hist.storage.Weight())

for i in range(30):
    tr_hist[i] = (arr_val[i], arr_var[i])


params = np.array([1.0, 0.0, 0.0, 0.0, 0.0])  # Initial parameters for pol4_root

res = fit_hist(
    tr_hist,
    partial(pol4_root, xLowVal=26.0, xFitRange=30.0),
    params,
)

tr_func = []
for i in range(len(bincenters)):
    tr_func.append(
        float(
            pol4_root(
                [bincenters[i]],
                res["x"],
                xLowVal=26.0,
                xFitRange=30.0,
    )))
logger.info(tr_func)
logger.info("Params:", res["x"])

chi2 = res["loss_val"]
ndof = len(bincenters) - len(res["x"])
chi2Prob = ROOT.TMath.Prob(chi2, ndof)

logger.info(chi2, ndof, chi2Prob)

"""


fout.Close()
