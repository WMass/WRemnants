#!/usr/bin/env python3
from wremnants import CardTool,theory_tools,syst_tools
from wremnants import histselections as sel
from wremnants.datasets.datagroups import datagroups2016
import argparse
import os
import pathlib

scriptdir = f"{pathlib.Path(__file__).parent}"

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--outfolder", type=str, default="/scratch/kelong/CombineStudies")
parser.add_argument("-i", "--inputFile", type=str, required=True)
parser.add_argument("--qcdScale", choices=["byHelicityPt", "byPt", "integrated"], default="byHelicityPt", 
        help="Decorrelation for QCDscale (additionally always by charge)")
parser.add_argument("--wlike", action='store_true', help="Run W-like analysis of mZ")
parser.add_argument("--pdf", type=str, default="nnpdf31", choices=theory_tools.pdfMap.keys(), help="PDF to use")
args = parser.parse_args()

if not os.path.isdir(args.outfolder):
    os.mkdir(args.outfolder)

datagroups = datagroups2016(args.inputFile, wlike=args.wlike)
templateDir = f"{scriptdir}/Templates/WMass"
name = "WMass" if not args.wlike else "ZMassWLike"
cardTool = CardTool.CardTool(f"{args.outfolder}/{name}_{{chan}}.txt")
cardTool.setNominalTemplate(f"{templateDir}/main.txt")
cardTool.setOutfile(os.path.abspath(f"{args.outfolder}/{name}CombineInput.root"))
cardTool.setDatagroups(datagroups)
cardTool.setSpacing(36)

print("All processes", cardTool.allMCProcesses())
single_v_samples = cardTool.filteredProcesses(lambda x: x[0] in ["W", "Z"])
single_v_and_fake_samples = cardTool.filteredProcesses(lambda x: x[0] in ["W", "Z"] or x == "Fake")
single_vmu_samples = list(filter(lambda x: "mu" in x, single_v_samples))
signal_samples = list(filter(lambda x: x[0] == ("Z" if args.wlike else "W"), single_vmu_samples))
signal_samples_inctau = list(filter(lambda x: x[0] == ("Z" if args.wlike else "W"), single_v_samples))
print("Single V samples", single_v_samples)
print("Single Vmu samples", single_vmu_samples)
print("signal samples", signal_samples)
print("single_c_fake_samples", single_v_and_fake_samples)

pdfName = theory_tools.pdfMap[args.pdf]["name"]
cardTool.addSystematic(pdfName, 
    processes=single_v_and_fake_samples,
    mirror=True,
    group=pdfName,
    systAxes=["tensor_axis_0"],
    labelsByAxis=[pdfName.replace("pdf", "pdf{i}")],
    # Needs to be a tuple, since for multiple axis it would be (ax1, ax2, ax3)...
    # -1 means all possible values of the mirror axis
    skipEntries=[(0, -1)],
)

cardTool.addSystematic(f"alphaS002{pdfName}", 
    processes=single_v_and_fake_samples,
    mirror=False,
    group=pdfName,
    systAxes=["tensor_axis_0"],
    outNames=[pdfName+"AlphaSUp", pdfName+"AlphaSDown"],
    scale=0.75,
)
for name,num in zip(["effSystIsoTnP", "effStatTnP",], [2, 624*4]):
    axes = ["idiptrig-iso"] if num == 2 else ["SF eta", "SF pt", "SF charge", "idiptrig-iso"]
    axlabels = ["IDIPTrig"] if num == 2 else ["eta", "pt", "q", "Trig"]
    cardTool.addSystematic(name, 
        mirror=True,
        group="muon_eff",
        systAxes=axes,
        labelsByAxis=axlabels,
        baseName=name+"_",
        processes=cardTool.filteredProcesses(lambda x: x != "Data"),
    )

scaleSystAxes = ["chargeVgen", "muRfact", "muFfact"] 
scaleLabelsByAxis = ["q", "muR", "muF"]
scaleGroupName = "QCDscale"
scaleActionArgs = {"sum_ptV" : True, "sum_helicity" : True}
scaleSkipEntries = [(-1, 1, 1), (-1, 0, 2), (-1, 2, 0)]
# TODO: reuse some code here
if "Helicity" in args.qcdScale:
    scaleActionArgs.pop("sum_helicity")
    scaleGroupName += "ByHelicity"
    scaleSystAxes.insert(0, "helicity")
    scaleLabelsByAxis.insert(0, "Coeff")
    scaleSkipEntries = [(-1, *x) for x in scaleSkipEntries]
if "Pt" in args.qcdScale:
    scaleActionArgs.pop("sum_ptV")
    scaleGroupName += "ByPtV"
    scaleSystAxes.insert(0, "ptVgen")
    scaleLabelsByAxis.insert(0, "genPtV")
    scaleSkipEntries = [(-1, *x) for x in scaleSkipEntries]

cardTool.addSystematic("qcdScaleByHelicity", 
    action=syst_tools.scale_helicity_hist_to_variations,
    actionArgs=scaleActionArgs,
    processes=signal_samples,
    group=scaleGroupName,
    systAxes=scaleSystAxes,
    labelsByAxis=scaleLabelsByAxis,
    # Exclude all combinations where muR = muF = 1 (nominal) or where
    # they are extreme values (ratio = 4 or 1/4)
    skipEntries=scaleSkipEntries,
    # This is hacky but it's the best idea I have for now...
    systNameReplace=[("muR2muF2", "muRmuFUp"), ("muR0muF0", "muRmuFDown"), ("muR2muF1", "muRUp"), 
        ("muR0muF1", "muRDown"), ("muR1muF0", "muFDown"), ("muR1muF2", "muFUp")],
    baseName="QCDscale_",
    )

cardTool.addSystematic("muonScaleSyst", 
    processes=single_vmu_samples,
    group="muonScale",
    baseName="CMS_scale_m_",
    systAxes=["downUpVar", "scaleEtaSlice"],
    labelsByAxis=["downUpVar", "ieta"],
)
cardTool.addSystematic("muonL1PrefireSyst", 
    processes=cardTool.allMCProcesses(),
    group="muonPrefire",
    baseName="CMS_prefire_syst_m",
    systAxes=["downUpVar"],
    labelsByAxis=["downUpVar"],
)
# TODO: Allow to be appended to previous group
cardTool.addSystematic("muonL1PrefireStat", 
    processes=cardTool.allMCProcesses(),
    group="muonPrefire",
    baseName="CMS_prefire_stat_m_",
    systAxes=["downUpVar", "etaPhiRegion"],
    labelsByAxis=["downUpVar", "etaPhiReg"],
)
cardTool.addSystematic("massWeight", 
    # TODO: Add the mass weights to the tau samples
    processes=signal_samples_inctau,
    outNames=theory_tools.massWeightNames(["massShift100MeV"], wlike=args.wlike),
    group="massShift",
    groupFilter=lambda x: x == "massShift100MeV",
    mirror=False,
    #TODO: Name this
    noConstraint=True,
    systAxes=["tensor_axis_0"],
)
# TODO: This needs to be handled by shifting the norm before subtracting from the fakes
# cardTool.addSystematic("lumi", outNames=["", "lumiDown", "lumiUp"], group="luminosity")
if not args.wlike:
    cardTool.addLnNSystematic("CMS_Fakes", processes=["Fake"], size=1.05)
    cardTool.addLnNSystematic("CMS_Top", processes=["Top"], size=1.06)
    cardTool.addLnNSystematic("CMS_VV", processes=["Diboson"], size=1.16)
else:
    cardTool.addLnNSystematic("CMS_background", processes=["Other"], size=1.15)
cardTool.writeOutput()

