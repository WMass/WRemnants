#!/usr/bin/env python3

# to check if SF Up/Down variations are effectively the mirrored image of each other wrt to the nominal

import argparse
## safe batch mode
import sys

args = sys.argv[:]
sys.argv = ["-b"]
import ROOT

sys.argv = args
ROOT.gROOT.SetBatch(True)
ROOT.PyConfig.IgnoreCommandLineOptions = True

from copy import *

from scripts.analysisTools.plotUtils.utility import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("rootfile", type=str, nargs=1, help="Input root file")
    parser.add_argument("outdir", type=str, nargs=1, help="Folder for plots")
    args = parser.parse_args()

    fname = args.rootfile[0]
    outdir = args.outdir[0] + "/"
    createPlotDirAndCopyPhp(outdir)
    adjustSettings_CMS_lumi()

    steps_byCharge = ["reco", "tracking", "idip", "trigger"]
    steps_inclusive = ["iso"]
    allSteps = steps_byCharge[:] + steps_inclusive[:]

    # SF_nomiAndAlt_GtoH_tracking_plus
    # SF_nomiAndAlt_onlyDataVar_GtoH_iso_both
    # SF_nomiAndAlt_onlyMCVar_GtoH_iso_both

    canvas = ROOT.TCanvas("canvas", "", 800, 700)
    canvasWide = ROOT.TCanvas("canvasWide", "", 2400, 600)
    tfile = safeOpenFile(fname)

    for step in allSteps:
        if step in steps_byCharge:
            charges = ["plus", "minus"]
        else:
            charges = ["both"]
        for charge in charges:
            sfhistname = f"SF_nomiAndAlt_GtoH_{step}_{charge}"
            hsf = safeGetObject(tfile, sfhistname)
            hnomi = ROOT.wrem.projectTH2FromTH3(hsf, f"hnomi_{step}_{charge}", 1)
            nVars = int((hsf.GetNbinsZ() - 2) / 2)  # number of Up or Down variations
            for nv in range(nVars):
                # variations organized as nomi, Nup,Ndown, syst
                hup = ROOT.wrem.projectTH2FromTH3(
                    hsf, f"hup{nv}_{step}_{charge}", 2 + nv
                )
                hdown = ROOT.wrem.projectTH2FromTH3(
                    hsf, f"hdown{nv}_{step}_{charge}", 2 + nv + nVars
                )
                hdownMirror = copy.deepcopy(
                    hnomi.Clone(f"hdownMirror{nv}_{step}_{charge}")
                )
                hdownMirror.Multiply(hnomi)
                hdownMirror.Divide(hup)
                ROOT.wrem.setRootHistogramError(hdownMirror, 0.0)
                hratio = copy.deepcopy(
                    hdown.Clone(f"hratioDownEigenOverMirror{nv}_{step}_{charge}")
                )
                # hratio.SetTitle("Eigen / mirror (down=nomi^2/up)")
                hratio.SetTitle(f"{step} {charge}: eigen {nv}")
                hratio.Divide(hdownMirror)
                rmin, rmax = getMinMaxHisto(hratio, sumError=False)
                maxdiff = max(abs(rmax - 1.0), abs(rmin - 1.0))
                rmax = 1.0 + maxdiff
                rmin = 1.0 - maxdiff
                ztitle = f"SF_{{down}} ratio: eigen / mirror::{rmin},{rmax}"
                drawCorrelationPlot(
                    hratio,
                    "Muon #eta",
                    "Muon p_{T} (GeV)",
                    ztitle,
                    hratio.GetName(),
                    "ForceTitle",
                    outdir,
                    palette=87,
                    passCanvas=canvas,
                    skipLumi=True,
                )
                # Better not to plot the unrolled, the histogram has the fine pt bins from the smoothing
                # so the unrolled plot will be huge and impossible to read
                #
                # ratio_unrolled = unroll2Dto1D(hratio, newname=f"{hratio.GetName()}_unrolled")
                # XlabelUnroll = "unrolled template along #eta:  #eta #in [%.1f, %.1f]" % (hratio.GetXaxis().GetBinLowEdge(1),
                #                                                                          hratio.GetXaxis().GetBinLowEdge(1+hratio.GetNbinsX()))
                # ptBinRanges = []
                # ptBins = [round(hratio.GetYaxis().GetBinLowEdge(i),1) for i in range(1,hratio.GetNbinsY()+2)]
                # for ipt in range(0,hratio.GetNbinsY()):
                #     ptBinRanges.append("#splitline{{[{ptmin},{ptmax}]}}{{GeV}}".format(ptmin=int(ptBins[ipt]), ptmax=int(ptBins[ipt+1])))

                # drawSingleTH1(ratio_unrolled, XlabelUnroll, ztitle, ratio_unrolled.GetName(),
                #               outdir, drawLineLowerPanel="", lowerPanelHeight=0.0, labelRatioTmp="",
                #               passCanvas=canvasWide,
                #               legendCoords="0.15,0.85,0.82,0.9;2",
                #               leftMargin=0.05,rightMargin=0.01,lumi=None,
                #               drawVertLines="{a},{b}".format(a=hratio.GetNbinsY(),b=hratio.GetNbinsX()),
                #               textForLines=ptBinRanges, ytextOffsetFromTop=0.3, textSize=0.04, textAngle=30, drawLineTopPanel=1.0)
