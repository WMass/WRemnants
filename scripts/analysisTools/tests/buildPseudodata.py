#!/usr/bin/env python3

# quick script to build a pseudodata histogram summing whatever you need for tests
import os
## safe batch mode
import sys

from utilities import common

args = sys.argv[:]
sys.argv = ["-b"]
import ROOT

sys.argv = args
ROOT.gROOT.SetBatch(True)
ROOT.PyConfig.IgnoreCommandLineOptions = True

from copy import *

from scripts.analysisTools.plotUtils.utility import *

## Dictionary with info to build pseudodata
# file is the file to read from
# procs is the list of process names to form the histogram name to get (make name as x_{proc}_{charge})
# hists is a list of actual histogram names (optional)
## This assumes we read TH2, will see if the logic needs to be extended
pseudodataMaker = {
    1: {
        "file": "/scratch/mciprian/CombineStudies/TRASHTEST/testMyPRafterMerge/withQCD_fixJetSyst/WMass_eta_pt/WMassCombineInput.root",
        "procs": [
            "Wmunu",
            "Wtaunu",
            "Zmumu",
            "Ztautau",
            "Top",
            "Diboson",
            "Fake",
        ],  # list of names or processes
        "hists": [],
    },
    # 2: {"file": "something",
    #     "procs": [], # list of names or processes
    #     "hists": []
    #     },
    # 3: {"file": "something",
    #     "procs": [], # list of names or processes
    #     "hists": []
    #     },
}

# choose if applying a correction to fakes, use {} if None

# fakesCorr = {"file": "/eos/user/m/mciprian/www/WMassAnalysis/fromMyWremnants/fitResults/TRASHTEST/testFakes_PR131/WMass/testFakesVsMt_testNoJetCutInWremnants//sidebandIntegral_fit0to40_pol1_rebinEta4_rebinPt2_jetInclusiveFRF_maxPt50//fakerateFactorMtBasedCorrection_vsEtaPt.root",
#             "hist": "etaPtCharge_mtCorrection",
#             "offsetCorr": 1.0}

# fakesCorr = {"file": "/eos/user/m/mciprian/www/WMassAnalysis/fromMyWremnants/fitResults/TRASHTEST/testFakes_PR131/WMass/testFakesVsMt_testNoJetCutInWremnants//sidebandIntegral_fit0to40_pol1_rebinEta4_rebinPt2_jetInclusiveFRF_maxPt50//fakerateFactorMtBasedCorrection_vsEtaPt.root",
#              "hist": "fakerateFactorCorrection_CHARGE",
#              "offsetCorr": 1.0}

fakesCorr = {}

# apply a correction to the final histogram directly
finalhistCorr = {
    "file": "/eos/user/m/mciprian/www/WMassAnalysis/TnP/egm_tnp_analysis/results_globalMuons_ntuplesXYZ_1orMoreNvalidHitsStandalone/smoothLeptonScaleFactors/GtoH/mu_iso_both/smoothedSFandEffi_iso_GtoH_both.root",
    "hist": "ratioSF_smoothEffiOverSmoothDirectly_GtoH_iso_both",
    "ops": "divide",  # can be multiply or add or subtract, multiply/divide also works with histograms with different binning
    "zeroUnc": True,
}  # reset uncertainty in correction histogram to 0 if the target must not have modified uncertainty


if __name__ == "__main__":

    parser = common_plot_parser()
    parser.add_argument("outfolder", type=str, nargs=1)
    parser.add_argument("outfilename", type=str, nargs=1)
    parser.add_argument(
        "outhistname", type=str, nargs=1, help="Charge added to name automatically"
    )
    parser.add_argument(
        "-p", "--plot", dest="plot", action="store_true", help="Plot pseudodata"
    )
    args = parser.parse_args()

    logger = common.setup_color_logger(os.path.basename(__file__), args.verbose)

    charges = ["plus", "minus"]
    fakeCorrHist = {x: None for x in charges}
    # this part is quite hardcoded but fine
    if fakesCorr:
        f = safeOpenFile(fakesCorr["file"], mode="READ")
        if "CHARGE" in fakesCorr["hist"]:
            fakeCorrHist["plus"] = safeGetObject(
                f, fakesCorr["hist"].replace("CHARGE", "plus"), detach=True
            )
            fakeCorrHist["minus"] = safeGetObject(
                f, fakesCorr["hist"].replace("CHARGE", "minus"), detach=True
            )
        else:
            corr = safeGetObject(f, fakesCorr["hist"], detach=True)
            if fakesCorr["offsetCorr"]:
                offsetHist = corr.Clone("offsetHist")
                ROOT.wrem.initializeRootHistogram(offsetHist, fakesCorr["offsetCorr"])
                corr.Add(offsetHist)
            fakeCorrHist["minus"] = ROOT.wrem.projectTH2FromTH3(
                corr, "fakeCorrHist_minus", 1, 1
            )
            fakeCorrHist["plus"] = ROOT.wrem.projectTH2FromTH3(
                corr, "fakeCorrHist_plus", 2, 2
            )
        f.Close()

    charges = ["plus", "minus"]
    pseudodataHist = {x: None for x in charges}
    for k in pseudodataMaker.keys():
        fin = safeOpenFile(pseudodataMaker[k]["file"], mode="READ")
        for c in charges:
            # loop on processes to add up, this is the typical procedure
            for p in pseudodataMaker[k]["procs"]:
                htmp = safeGetObject(fin, f"x_{p}_{c}")
                if p == "Fake" and fakeCorrHist[c] is not None:
                    print(f"Multiplying histogram for {p} by fakeCorrHist[c]")
                    print(f"Integral before correction: {htmp.Integral()}")
                    scaleTH2byOtherTH2(htmp, fakeCorrHist[c], scaleUncertainty=False)
                    print(f"Integral after correction: {htmp.Integral()}")
                if pseudodataHist[c] is None:
                    pseudodataHist[c] = copy.deepcopy(
                        htmp.Clone(f"{args.outhistname[0]}_{c}")
                    )
                else:
                    pseudodataHist[c].Add(htmp)
            # repeat summing histograms given full names, if any
            for h in pseudodataMaker[k]["hists"]:
                htmp = safeGetObject(fin, h)
                if pseudodataHist[c] is None:
                    pseudodataHist[c] = copy.deepcopy(
                        htmp.Clone(f"{args.outhistname[0]}_{c}")
                    )
                else:
                    pseudodataHist[c].Add(htmp)
        fin.Close()

    # correct final histogram
    if finalhistCorr:
        totCorrHist = {c: None for c in charges}
        print(
            f"Going to {finalhistCorr['ops']} total histogram by {finalhistCorr['hist']}"
        )
        f = safeOpenFile(finalhistCorr["file"], mode="READ")
        for c in charges:
            finalhistCorr[c] = safeGetObject(
                f, finalhistCorr["hist"].replace("CHARGE", c), detach=True
            )
            if finalhistCorr["zeroUnc"]:
                for ib in range(finalhistCorr[c].GetNcells()):
                    finalhistCorr[c].SetBinError(ib, 0)
            if finalhistCorr["ops"] == "divide":
                scaleTH2byOtherTH2(pseudodataHist[c], finalhistCorr[c], divide=True)
            elif finalhistCorr["ops"] == "multiply":
                scaleTH2byOtherTH2(pseudodataHist[c], finalhistCorr[c])
            elif finalhistCorr["ops"] == "add":
                pseudodataHist[c].Add(finalhistCorr[c])
            elif finalhistCorr["ops"] == "subtract":
                pseudodataHist[c].Add(finalhistCorr[c], -1.0)
            else:
                print(
                    "Error: operation {finalhistCorr['ops']} not implemented. Please check"
                )
                quit()

    for c in charges:
        pseudodataHist[c].SetTitle(f"Pseudodata for charge {c}")

    outdir_original = args.outfolder[0]
    outdir = createPlotDirAndCopyPhp(outdir_original, eoscp=args.eoscp)
    outname = f"{outdir}/{args.outfilename[0]}"
    if not outname.endswith(".root"):
        outname += ".root"
    fout = safeOpenFile(outname, mode="RECREATE")
    fout.cd()

    # do some plots of the final histograms (should be better to plot ratio with uncorrected histogram, but this can be done with
    # scripts/analysisTools/w_mass_13TeV/plotPrefitTemplatesWRemnants.py selecting pseudodata as data histogram
    if args.plot:
        canvas = ROOT.TCanvas("canvas", "", 800, 800)
        for c in charges:
            drawCorrelationPlot(
                pseudodataHist[c],
                pseudodataHist[c].GetXaxis().GetTitle(),
                pseudodataHist[c].GetYaxis().GetTitle(),
                "Events",
                pseudodataHist[c].GetName(),
                plotLabel="ForceTitle",
                outdir=outdir,
                draw_both0_noLog1_onlyLog2=1,
                nContours=args.nContours,
                palette=args.palette,
                invertPalette=args.invertPalette,
                passCanvas=canvas,
                skipLumi=True,
            )

    for c in charges:
        pseudodataHist[c].Write()
        logger.info(
            f"Writing histogram {pseudodataHist[c].GetName()}: integral = {pseudodataHist[c].Integral()}"
        )
    fout.Close()
    logger.info(f"Output saved in file {outname}")

    copyOutputToEos(outdir, outdir_original, eoscp=args.eoscp)
