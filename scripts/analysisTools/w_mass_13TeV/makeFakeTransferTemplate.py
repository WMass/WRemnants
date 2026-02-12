#!/usr/bin/env python3
import os
import pickle
import sys
import numpy as np
from array import array

import hist
import lz4.frame
import ROOT
import tensorflow as tf

from functools import partial

# from narf import histutils
import narf
import wums.output_tools
import wums.fitutils
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


def polN_root(xvals, parms, xLowVal=0.0, xFitRange=1.0, xCut=47.84):
    xScaled = (xvals[0] - xLowVal) / xFitRange
    xCutScaled = (xCut - xLowVal) / xFitRange

    polN = tf.exp(
            parms[0]
            + parms[1] * xScaled
            + parms[2] * xScaled**2
            + parms[3] * xScaled**3
            + parms[4] * xScaled**4)
    
    polN_cut = tf.exp(
            parms[0]
            + parms[1] * xCutScaled
            + parms[2] * xCutScaled**2
            + parms[3] * xCutScaled**3
            + parms[4] * xCutScaled**4)
    
    der_polN_cut = (
        parms[1] 
        + 2*parms[2]*xCutScaled
        + 3*parms[3]*xCutScaled**2
        + 4*parms[4]*xCutScaled**3
        ) * polN_cut

    return tf.where(
        xScaled < xCutScaled,
        polN,
        polN_cut + der_polN_cut * (xScaled - xCutScaled)
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
    "--noSmoothing",
    action="store_true",
    default=False,
    help="Save binned TF instead of smoothed one"
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

hnomi = histInfo[datasets[0]].hists["nominal"]
nPtBins = hnomi.axes["pt"].size
ptEdges = hnomi.axes["pt"].edges
nEtaBins = hnomi.axes["eta"].size
etaEdges = hnomi.axes["eta"].edges
nChargeBins = hnomi.axes["charge"].size
chargeEdges = hnomi.axes["charge"].edges
ptBinCenters = [round((ptEdges[i+1]+ptEdges[i])/2, 1) for i in range(nPtBins)]

eta_genBinning = array("d", [round(x, 1) for x in etaEdges])
pt_genBinning  = array("d", [round(x, 0) for x in ptEdges])
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

out_hist = ROOT.TH3D(
    f"fakeRatio_utAngleSign_{proc}",
    "",
    nEtaBins, eta_genBinning,
    nPtBins, pt_genBinning,
    nChargeBins, charge_genBinning,
)

out_hist_varSeparate = ROOT.THnD(
    f"fakeRatio_utAngleSign_{proc}_variations",
    "",
    4,
    array("i", [nEtaBins,      nPtBins,    nChargeBins,      5]),
    array("d", [etaEdges[0],  ptEdges[0],  chargeEdges[0],  -0.5]),
    array("d", [etaEdges[-1], ptEdges[-1], chargeEdges[-1],  5.5])
)
out_hist_varAll = out_hist.Clone(f"fakeRatio_utAngleSign_{proc}_variations_all")

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

        ratio_h = root_h_utMinus.Clone(f"fakeRatio_utAngleSign")
        ratio_h.Sumw2()
        ratio_h.Divide(root_h_utPlus)

        if args.noSmoothing is False:

            for iBin in range(1, nPtBins):
                ratio_h.SetBinError(iBin, ratio_h.GetBinError(iBin) * (9.78**0.5))

            ratio_h_boost = narf.root_to_hist(ratio_h)
            pars = np.array([1.0, 0.0, 0.0, 0.0, 0.0]) # Initial parameters for polN_root
            fitFunc = partial(polN_root, xLowVal=26.0, xFitRange=30.0)
            fitRes = wums.fitutils.fit_hist(
                ratio_h_boost,
                fitFunc,
                pars,
                )
            tf1_fit = ROOT.TF1("tf1_fit", fitFunc, ptEdges[0], ptEdges[-1], len(pars))
            tf1_fit.SetParameters(np.array(fitRes["x"], dtype=np.float64))

            npar = tf1_fit.GetNpar()
            altPars = np.array([np.zeros(npar, dtype=np.float64)] * (npar*2), dtype=np.float64)

            e, v = np.linalg.eigh(fitRes["cov"])
            for ivar in range(npar):
                shift = np.sqrt(e[ivar]) * v[:, ivar] * 4.0
                altPars[ivar] = fitRes["x"] + shift
                altPars[ivar + npar] = fitRes["x"] - shift

            altValPoints = [np.zeros(nPtBins)]*npar

            ratio_alt = ROOT.TH2D("fakeRatio_utAngleSign_variations_separate", "",
                                  nPtBins, ptEdges[0], ptEdges[-1],
                                  npar, -0.5, -0.5+npar)
            ratio_alt.Sumw2()
            ratio_alt_all = ROOT.TH1D("fakeRatio_utAngleSign_variations_all", "",
                                      nPtBins, ptEdges[0], ptEdges[-1])
            ratio_alt_all.Sumw2()

            for iBin in range(1, nPtBins):
                pt = ptBinCenters[iBin-1]
                ratio_h.SetBinContent(iBin, max(0.001, tf1_fit.Eval(pt)))

                for ivar in range(npar):
                    tf1_alt = ROOT.TF1()
                    tf1_alt.SetName(f"tf1_alt_{ivar}")
                    tf1_fit.Copy(tf1_alt)

                    # set parameters for a given hessian
                    tf1_alt.SetParameters(altPars[ivar])
                    altValPoints[ivar][iBin-1] = max(0.001, tf1_alt.Eval(pt))

        for idx_ch in range(ch_low_idx + 1, ch_high_idx + 1):
            for idx_eta in range(eta_low_idx + 1, eta_high_idx + 1):
                # logger.debug(f"Setting weights for chBin={idx_ch}, etaBin={idx_eta}")
                for idx_pt in range(1, nPtBins+1):
                    out_hist.SetBinContent(
                        idx_eta, idx_pt, idx_ch, ratio_h.GetBinContent(idx_pt)
                    )
                    out_hist.SetBinError(
                        idx_eta, idx_pt, idx_ch, ratio_h.GetBinError(idx_pt)
                    )
                    if ratio_h.GetBinContent(idx_pt) <= 0.0:
                        logger.warning(f"Found negative value in bin: ({idx_eta}, {idx_pt}, {idx_ch})")

                    if args.noSmoothing is False:
                        totVar = 0.0
                        for ivar in range(npar):
                            altVal = altValPoints[ivar][idx_pt-1]
                            glBin = out_hist_varSeparate.GetBin(array("i", [idx_eta, idx_pt, idx_ch, ivar+1]))
                            out_hist_varSeparate.SetBinContent(glBin, altVal)
                            diff = altVal - ratio_h.GetBinContent(idx_pt)
                            totVar += diff * diff

                        out_hist_varAll.SetBinContent(
                            idx_eta, idx_pt, idx_ch, ratio_h.GetBinContent(idx_pt) + np.sqrt(totVar)
                        )


boost_out_hist = narf.root_to_hist(out_hist)
resultDict = {"fakeCorr": boost_out_hist}
if args.noSmoothing is False:
    resultDict.update({
        "fakeCorr_variations" : narf.root_to_hist(out_hist_varSeparate),
        "fakeCorr_varAll" : narf.root_to_hist(out_hist_varAll)
    })


base_dir = common.base_dir
resultDict.update(
    {"meta_info": wums.output_tools.make_meta_info_dict(args=args, wd=base_dir)}
)

if args.plotdir is not None:

    plotdir_original = args.plotdir
    plotdir = createPlotDirAndCopyPhp(plotdir_original, eoscp=args.eoscp)
    hists_corr = []
    legEntries = []
    etaID = 0
    # for 1D plots
    canvas1D = ROOT.TCanvas("canvas1D", "", 800, 900)
    adjustSettings_CMS_lumi()
    idxs_ch  = [0] if args.nChargeBins==1 else [1, 2]
    idxs_eta = [0] if args.nEtaBins==1    else [int(48*i/args.nEtaBins) for i in range(args.nEtaBins+1)]
    for ieta in idxs_eta:
        etamu = "#eta^{#mu}"
        etaleg = f"{decorrBins_eta[etaID][0]} < {etamu} < {decorrBins_eta[etaID][1]}," if len(idxs_eta)!=1 else ""
        etaID += (1 if len(idxs_eta)!=1 else 0)
        for icharge in idxs_ch:
            chargeleg = "#it{q}^{#mu} = " + ("-1" if icharge == 1 else "+1") if len(idxs_ch)!=1 else ""
            hists_corr.append(
                out_hist.ProjectionY(
                    f"projPt_{ieta}_{icharge}", ieta, ieta, icharge, icharge
                )
            )
            legInfo = f"{etaleg} {chargeleg}".lstrip()
            legEntries.append(legInfo if legInfo!="" else "TF integrated on #eta and #it{q}")

    miny, maxy = getMinMaxMultiHisto(hists_corr)
    if miny < 0:
        miny = 0
    maxy = 1.4 * (maxy - miny)
    drawNTH1(
        hists_corr,
        legEntries,
        "#it{p}_{T}^{#mu}",
        "Correction: #it{u}_{T}^{#mu} > 0 #rightarrow #it{u}_{T}^{#mu} < 0"
        + f"::{miny},{maxy}",
        "correction_uT_vs_pT_eta_charge",
        plotdir,
        lowerPanelHeight=0.4,
        legendCoords=(
            "0.4,0.98,0.74,0.92" if len(idxs_ch)==1 else "0.16,0.98,0.74,0.92;2"
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

pklfileName = f"{args.outdir}/{outfileName}.pkl.lz4"
with lz4.frame.open(pklfileName, "wb") as fout:
    pickle.dump(resultDict, fout, protocol=pickle.HIGHEST_PROTOCOL)
logger.info(f"Created file {pklfileName}")

rootfileName = f"{args.outdir}/{outfileName}.root"
fout = safeOpenFile(rootfileName, mode="RECREATE")
out_hist.Write()
if args.noSmoothing is False:
    out_hist_varSeparate.Write()
    out_hist_varAll.Write()
fout.Close()
logger.info(f"Created file {rootfileName}")



fout.Close()
