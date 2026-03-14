#!/usr/bin/env python3
import math
import os
import pickle
import sys
from array import array
from functools import partial

import hist
import lz4.frame
import numpy as np
import ROOT
import tensorflow as tf

# from narf import histutils
import narf
import wums.fitutils
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
    adjustSettings_CMS_lumi,
    common_plot_parser,
    copyOutputToEos,
    createPlotDirAndCopyPhp,
    drawNTH1,
    getMinMaxMultiHisto,
    safeOpenFile,
)

INFLATION_ERRORS = 9.78**0.5
INFLATION_FIT_SIGMA = 0

inflation = np.sqrt(INFLATION_ERRORS**2 + INFLATION_FIT_SIGMA**2)


def pol4_withCut_root(xvals, parms, xLowVal=0.0, xFitRange=1.0, xCut=47.84):
    xScaled = (xvals[0] - xLowVal) / xFitRange
    xCutScaled = (xCut - xLowVal) / xFitRange

    polN = tf.exp(
        parms[0]
        + parms[1] * xScaled
        + parms[2] * xScaled**2
        + parms[3] * xScaled**3
        + parms[4] * xScaled**4
    )

    polN_cut = tf.exp(
        parms[0]
        + parms[1] * xCutScaled
        + parms[2] * xCutScaled**2
        + parms[3] * xCutScaled**3
        + parms[4] * xCutScaled**4
    )

    der_polN_cut = (
        parms[1]
        + 2 * parms[2] * xCutScaled
        + 3 * parms[3] * xCutScaled**2
        + 4 * parms[4] * xCutScaled**3
    ) * polN_cut

    return tf.where(
        xScaled < xCutScaled, polN, polN_cut + der_polN_cut * (xScaled - xCutScaled)
    )


def pol4_root(xvals, parms, xLowVal=0.0, xFitRange=1.0):
    xscaled = (xvals[0] - xLowVal) / xFitRange
    return tf.exp(
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


def smoothTF(ratio_boost, x_bins, xLowVal=26.0, xFitRange=30.0):

    pars = np.array([1.0, 0.0, 0.0, 0.0, 0.0])  # Initial parameters for polN_root
    fitFunc = partial(pol4_root, xLowVal=xLowVal, xFitRange=xFitRange)
    fitRes = wums.fitutils.fit_hist(
        ratio_boost,
        fitFunc,
        pars,
    )
    tf1_fit = ROOT.TF1("tf1_fit", fitFunc, ptEdges[0], ptEdges[-1], len(pars))
    tf1_fit.SetParameters(np.array(fitRes["x"], dtype=np.float64))

    npar = tf1_fit.GetNpar()
    altPars = np.array(
        [np.zeros(npar, dtype=np.float64)] * (npar * 2), dtype=np.float64
    )

    e, v = np.linalg.eigh(fitRes["cov"])
    for ivar in range(npar):
        shift = np.sqrt(e[ivar]) * v[:, ivar] * inflation
        altPars[ivar] = fitRes["x"] + shift
        altPars[ivar + npar] = fitRes["x"] - shift

    nomiVals = np.zeros(len(x_bins))
    altVals_sep = [np.zeros(len(x_bins))] * npar
    altVals_all = np.zeros(len(x_bins))

    for iBin in range(len(x_bins)):
        pt = ptBinCenters[iBin]
        fitVal = max(0.00001, tf1_fit.Eval(pt))
        nomiVals[iBin] = fitVal

        err = 0
        for ivar in range(npar):
            tf1_alt = ROOT.TF1()
            tf1_alt.SetName(f"tf1_alt_{ivar}")
            tf1_fit.Copy(tf1_alt)

            # set parameters for a given hessian
            tf1_alt.SetParameters(altPars[ivar])
            altFuncVal = max(0.001, tf1_alt.Eval(pt))
            altVals_sep[ivar][iBin] = altFuncVal
            diff = altFuncVal - fitVal
            err += diff * diff

        altVals_all[iBin] = fitVal + math.sqrt(err)
        print(f"{iBin}: {altVals_all[iBin] / fitVal}")

    return fitRes, nomiVals, altVals_all, altVals_sep


parser = common_plot_parser()
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
parser.add_argument(
    "--addClosure",
    action="store_true",
    help="Save variations related to closure with QCD MC and with signal region",
)
parser.add_argument(
    "--noSmoothing",
    action="store_true",
    default=False,
    help="Save binned TF instead of smoothed one",
)
parser.add_argument(
    "--plotdir", type=str, default=None, help="Output directory for plots"
)

args = parser.parse_args()

logger = logging.setup_logger(os.path.basename(__file__), 4)
ROOT.TH1.SetDefaultSumw2()
proc = "Data" if not args.doQCD else "QCD"

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

s = hist.tag.Slicer()

histInfo = groups.groups

select_utMinus = {"utAngleSign": s[0 : 1 : hist.sum]}
select_utPlus = {"utAngleSign": s[1 : 2 : hist.sum]}

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

hnomi = histInfo[datasets[0]].hists["nominal"]
nPtBins = hnomi.axes["pt"].size
ptEdges = hnomi.axes["pt"].edges
nEtaBins = hnomi.axes["eta"].size
etaEdges = hnomi.axes["eta"].edges
nChargeBins = hnomi.axes["charge"].size
chargeEdges = hnomi.axes["charge"].edges
ptBinCenters = [round((ptEdges[i + 1] + ptEdges[i]) / 2, 1) for i in range(nPtBins)]

eta_genBinning = array("d", [round(x, 1) for x in etaEdges])
pt_genBinning = array("d", [round(x, 0) for x in ptEdges])
charge_genBinning = array("d", chargeEdges)

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

out_hist_nomi = hist.Hist(
    hist.axis.Regular(
        len(etaEdges) - 1, etaEdges[0], etaEdges[-1], name="eta", flow=False
    ),
    hist.axis.Regular(len(ptEdges) - 1, ptEdges[0], ptEdges[-1], name="pt", flow=False),
    hist.axis.Regular(
        len(chargeEdges) - 1, chargeEdges[0], chargeEdges[-1], name="charge", flow=False
    ),
    # hist.axis.Regular(2, -2.0, 2.0, name="utAngleSign")
    storage=hist.storage.Weight(),
)
outNomi = out_hist_nomi.view()

out_hist_altStat = hist.Hist(
    *out_hist_nomi.axes,
    # FIXME: why not starting from -0.5 such that the first bin is centered at 0?
    hist.axis.Regular(
        6, -1.5, 4.5, name="varTF"
    ),  # 1 inclusive variation (first bin), plus the other 5
    storage=hist.storage.Weight(),
)
outAltStat = out_hist_altStat.view()

if args.addClosure:
    out_hist_closQCDsv = hist.Hist(*out_hist_nomi.axes, storage=hist.storage.Weight())
    out_hist_closQCDsignal = hist.Hist(
        *out_hist_nomi.axes, storage=hist.storage.Weight()
    )
else:
    out_hist_closQCDsv, out_hist_closQCDsignal = None, None

out_info = {}

for ch_edges in decorrBins_ch:
    for eta_edges in decorrBins_eta:

        ch_low_idx, ch_high_idx = convert_binEdges_idx(ch_edges, charge_genBinning)
        eta_low_idx, eta_high_idx = convert_binEdges_idx(eta_edges, eta_genBinning)

        logger.info(f"{ch_low_idx}, {ch_high_idx}")
        logger.info(f"{eta_low_idx}, {eta_high_idx}")

        select_utMinus["charge"] = s[ch_low_idx : ch_high_idx : hist.sum]
        select_utMinus["eta"] = s[eta_low_idx : eta_high_idx : hist.sum]

        select_utPlus["charge"] = s[ch_low_idx : ch_high_idx : hist.sum]
        select_utPlus["eta"] = s[eta_low_idx : eta_high_idx : hist.sum]

        logger.info(f"Processing charge bin [{ch_edges}] and eta bin [{eta_edges}]")

        boost_h_utMinus = histInfo[proc].copy(f"{proc}_utMinus").hists["nominal"]
        boost_h_utMinus = boost_h_utMinus[select_utMinus]
        boost_h_utMinus = hh.projectNoFlow(boost_h_utMinus, ["pt"], ["relIso", "mt"])
        root_h_utMinus = narf.hist_to_root(boost_h_utMinus)

        boost_h_utPlus = histInfo[proc].copy(f"{proc}_utMinus").hists["nominal"]
        boost_h_utPlus = boost_h_utPlus[select_utPlus]
        boost_h_utPlus = hh.projectNoFlow(boost_h_utPlus, ["pt"], ["relIso", "mt"])
        root_h_utPlus = narf.hist_to_root(boost_h_utPlus)

        logger.debug(f"Integrals BEFORE prompt subraction (uT < 0, uT > 0)")
        logger.debug(f"{root_h_utMinus.Integral()}, {root_h_utPlus.Integral()}")

        for mcName in prednames:
            if args.doQCD:
                continue
            logger.debug(f"Subtracting {mcName} from data")
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

        logger.debug(f"Integrals AFTER prompt subraction (uT < 0, uT > 0)")
        logger.debug(f"{root_h_utMinus.Integral()}, {root_h_utPlus.Integral()}")

        ratio_h = root_h_utMinus.Clone(f"fakeRatio_utAngleSign_TH1")
        ratio_h.Sumw2()
        ratio_h.Divide(root_h_utPlus)

        ratio_h_boost = narf.root_to_hist(ratio_h)

        sel = (
            slice(eta_low_idx, eta_high_idx),
            slice(None),
            slice(ch_low_idx, ch_high_idx),
        )

        if args.noSmoothing:
            outNomi.value[sel] = np.broadcast_to(
                ratio_h_boost.values()[None, :, None],
                (eta_high_idx - eta_low_idx, nPtBins, ch_high_idx - ch_low_idx),
            )
            outNomi.variance[sel] = np.broadcast_to(
                ratio_h_boost.variances()[None, :, None],
                (eta_high_idx - eta_low_idx, nPtBins, ch_high_idx - ch_low_idx),
            )

        else:
            fitRes, nomiVals, altVals_all, altVals_sep = smoothTF(
                ratio_h_boost,
                ptBinCenters,
                xLowVal=ptEdges[0],
                xFitRange=(ptEdges[-1] - ptEdges[0]),
            )

            outNomi.value[sel] = np.broadcast_to(
                nomiVals[None, :, None],
                (eta_high_idx - eta_low_idx, nPtBins, ch_high_idx - ch_low_idx),
            )

            outAltStat.value[sel + (0,)] = (
                np.broadcast_to(
                    altVals_all[None, :, None],
                    (eta_high_idx - eta_low_idx, nPtBins, ch_high_idx - ch_low_idx),
                )
                / outNomi.value[sel]
            )

            # TODO: now have 5 parameters for pol4, can generalize to polN
            for iv in range(1, 6):
                outAltStat.value[sel + (iv,)] = (
                    np.broadcast_to(
                        altVals_sep[iv - 1][None, :, None],
                        (eta_high_idx - eta_low_idx, nPtBins, ch_high_idx - ch_low_idx),
                    )
                    / outNomi.value[sel]
                )

            if args.addClosure:

                logger.debug("Elaborating corrections evaluated on QCD... ")
                logger.debug(
                    "... be sure that the files are present and the TF has been already smoothed!"
                )

                path_corr_QCD_sv = f"{common.data_dir}/fakesWmass/test/fakeTransferTemplates_QCD.pkl.lz4"
                path_corr_QCD_signal = f"{common.data_dir}/fakesWmass/test/fakeTransferTemplates_signalRegion_QCD.pkl.lz4"

                if os.path.exists(path_corr_QCD_sv):
                    with lz4.frame.open(path_corr_QCD_sv) as fTens:
                        hist_corr_QCDsv = pickle.load(fTens)["fakeCorr"]
                    out_hist_closQCDsv.values()[sel] = (
                        hist_corr_QCDsv[sel].values() / outNomi.value[sel]
                    )
                else:
                    logger.warning(
                        "File with TF correction on QCD (SV region) not found! Skipping..."
                    )
                    out_hist_closQCDsv = None

                if os.path.exists(path_corr_QCD_signal) and out_hist_closQCDsv:
                    with lz4.frame.open(path_corr_QCD_signal) as fTens:
                        hist_corr_QCDsignal = pickle.load(fTens)["fakeCorr"]
                    out_hist_closQCDsignal.values()[sel] = (
                        hist_corr_QCDsignal[sel].values()
                        / hist_corr_QCDsv[sel].values()
                    )
                else:
                    logger.warning(
                        "File with TF correction on QCD (signal region) not found! Skipping..."
                    )
                    out_hist_closQCDsignal = None

resultDict = {
    "fakeCorr": out_hist_nomi,
}

if args.noSmoothing is False:
    resultDict.update({"fakeCorr_altStat": out_hist_altStat})
if out_hist_closQCDsv:
    resultDict.update({"fakeCorr_closQCDsv": out_hist_closQCDsv})
if out_hist_closQCDsignal:
    resultDict.update({"fakeCorr_closQCDsignal": out_hist_closQCDsignal})

resultDict.update(
    {"meta_info": wums.output_tools.make_meta_info_dict(args=args, wd=common.base_dir)}
)


if args.plotdir is not None:

    outRootNomi = narf.hist_to_root(out_hist_nomi.copy())

    plotdir_original = args.plotdir
    plotdir = createPlotDirAndCopyPhp(plotdir_original, eoscp=args.eoscp)
    hists_corr = []
    legEntries = []
    etaID = 0
    # for 1D plots
    canvas1D = ROOT.TCanvas("canvas1D", "", 800, 900)
    adjustSettings_CMS_lumi()
    idxs_ch = [0] if args.nChargeBins == 1 else [1, 2]
    idxs_eta = (
        [0]
        if args.nEtaBins == 1
        else [(1 + int(48 * i / args.nEtaBins)) for i in range(args.nEtaBins)]
    )
    for ieta in idxs_eta:
        etamu = "#eta^{#mu}"
        etaleg = (
            f"{decorrBins_eta[etaID][0]} < {etamu} < {decorrBins_eta[etaID][1]}"
            if len(idxs_eta) != 1
            else ""
        )
        etaID += 1 if len(idxs_eta) != 1 else 0
        for icharge in idxs_ch:
            chargeleg = "#it{q}^{#mu} = " + (
                ("-1" if icharge == 1 else "+1") if len(idxs_ch) != 1 else "#pm 1"
            )
            hists_corr.append(
                outRootNomi.ProjectionY(
                    f"projPt_{ieta}_{icharge}", ieta, ieta, icharge, icharge
                )
            )
            legInfo = f"{etaleg}, {chargeleg}".lstrip()
            legEntries.append(
                legInfo if legInfo != "" else "TF integrated on #eta^{#mu}-#it{q}^{#mu}"
            )

    miny, maxy = getMinMaxMultiHisto(hists_corr)
    if miny < 0:
        miny = 0
    maxy = 1.4 * (maxy - miny)
    plotfilename = "correction_uT_vs_pT_eta_charge"
    if args.postfix:
        plotfilename += f"_{args.postfix}"
    drawNTH1(
        hists_corr,
        legEntries,
        "#it{p}_{T}^{#mu}",
        "Correction: #it{u}_{T}^{#mu} > 0 #rightarrow #it{u}_{T}^{#mu} < 0"
        + f"::{miny},{maxy}",
        plotfilename,
        plotdir,
        lowerPanelHeight=0.4,
        legendCoords=(
            "0.4,0.98,0.74,0.92" if len(idxs_ch) == 1 else "0.16,0.98,0.74,0.92;2"
        ),
        labelRatioTmp="Ratio to first::0.2,1.8",
        topMargin=0.06,
        rightMargin=0.02,
        drawLumiLatex=True,
        onlyLineColor=True,
        useLineFirstHistogram=True,
        drawErrorAll=True,
        yAxisExtendConstant=1.0,
    )

    copyOutputToEos(plotdir, plotdir_original, eoscp=args.eoscp)


outfileName = "fakeTransferTemplates"
if args.postfix:
    outfileName += f"_{args.postfix}"
if args.doQCD:
    outfileName += "_QCD"
# if args.noSmoothing:
#    outfileName += "_noSmoothing"

pklfileName = f"{args.outdir}/{outfileName}.pkl.lz4"
with lz4.frame.open(pklfileName, "wb") as fout:
    pickle.dump(resultDict, fout, protocol=pickle.HIGHEST_PROTOCOL)
logger.info(f"Created file {pklfileName}")

rootfileName = f"{args.outdir}/{outfileName}.root"
fout = safeOpenFile(rootfileName, mode="RECREATE")
ratio_h.Write()

outRootNomi = narf.hist_to_root(out_hist_nomi.copy()[(1, slice(None), 1)])
outRootNomi.SetName("fakeCorr_Nominal")
outRootNomi.Write()
outRootAltStat_all = narf.hist_to_root(out_hist_altStat.copy()[(0, slice(None), 0, 0)])
outRootAltStat_all.SetName("fakeCorr_altStat_all")
outRootAltStat_all.Write()
if args.addClosure:
    outClosSV = narf.hist_to_root(out_hist_closQCDsv.copy()[(0, slice(None), 0)])
    outClosSV.SetName("fakeCorr_closure_QCDsv")
    outClosSV.Write()
    outClosSignal = narf.hist_to_root(
        out_hist_closQCDsignal.copy()[(0, slice(None), 0)]
    )
    outClosSignal.SetName("fakeCorr_closure_QCDsignal")
    outClosSignal.Write()

fout.Close()
logger.info(f"Created file {rootfileName}")
