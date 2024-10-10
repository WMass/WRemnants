#!/usr/bin/env python3

## safe batch mode
import sys
from copy import *

args = sys.argv[:]
sys.argv = ["-b"]
import ROOT

sys.argv = args
ROOT.gROOT.SetBatch(True)
ROOT.PyConfig.IgnoreCommandLineOptions = True

from scripts.analysisTools.plotUtils.utility import *

step = "tracking"  # tracking
outdir_original = f"scripts/analysisTools/plots/TESTPLOTS/{step}_altBkg/"
fname = "wremnants-data/data/muonSF/allSmooth_GtoHout_vtxAgnIso_altBkg.root"
charge = "plus"
hname = f"SF_nomiAndAlt_GtoH_{step}_{charge}_altBkg"
rf = safeOpenFile(fname)
h = safeGetObject(rf, hname)

hnomi = getTH2fromTH3(h, f"hnomi_{charge}", 1)
hsyst = getTH2fromTH3(h, f"hsyst_{charge}", 10 if step == "reco" else 8)

canvas = ROOT.TCanvas("canvas", "", 900, 800)
setTDRStyle()  # this one removes the stat box

outdir = createPlotDirAndCopyPhp(outdir_original, eoscp=True)
drawCorrelationPlot(
    hnomi,
    "Muon #eta",
    "Muon p_{T} (GeV)",
    f"Scale factors (nomi)",
    hnomi.GetName(),
    plotLabel="ForceTitle",
    outdir=outdir,
    smoothPlot=False,
    drawProfileX=False,
    scaleToUnitArea=False,
    draw_both0_noLog1_onlyLog2=1,
    palette=112,
    nContours=51,
    invertPalette=False,
    passCanvas=canvas,
    drawOption="COLZ0",
)
drawCorrelationPlot(
    hsyst,
    "Muon #eta",
    "Muon p_{T} (GeV)",
    f"Scale factors (syst)",
    hsyst.GetName(),
    plotLabel="ForceTitle",
    outdir=outdir,
    smoothPlot=False,
    drawProfileX=False,
    scaleToUnitArea=False,
    draw_both0_noLog1_onlyLog2=1,
    palette=112,
    nContours=51,
    invertPalette=False,
    passCanvas=canvas,
    drawOption="COLZ0",
)

hratio = hsyst.Clone(f"hratio_{charge}")
hratio.Divide(hnomi)
drawCorrelationPlot(
    hratio,
    "Muon #eta",
    "Muon p_{T} (GeV)",
    f"SF ratio (syst/nomi)",
    hratio.GetName(),
    plotLabel="ForceTitle",
    outdir=outdir,
    smoothPlot=False,
    drawProfileX=False,
    scaleToUnitArea=False,
    draw_both0_noLog1_onlyLog2=1,
    palette=112,
    nContours=51,
    invertPalette=False,
    passCanvas=canvas,
    drawOption="COLZ0",
)

copyOutputToEos(outdir, outdir_original, eoscp=True)
print()
