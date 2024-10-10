#!/usr/bin/env python3

import os
## safe batch mode
import sys

args = sys.argv[:]
sys.argv = ["-b"]
import ROOT

sys.argv = args
ROOT.gROOT.SetBatch(True)
ROOT.PyConfig.IgnoreCommandLineOptions = True

# sys.path.append(os.getcwd() + "/plotUtils/")
# from utility import *

# mainPath = "/eos/user/m/mciprian/www/WMassAnalysis/TnP/egm_tnp_analysis/results_Sept2022_binnedInPtEta_mass60to120"
# workingPoints = ["reco", "trigger", "iso"]
mainPath = "/eos/user/m/mciprian/www/WMassAnalysis/TnP/egm_tnp_analysis/results_test_globalMuons_testByParity"
inputhPath = (
    "/home/m/mciprian/tnp/egm_tnp_analysis/localplots/results_globalMuons_testByParity"
)
workingPoints = ["tracking"]

tag = "oddOverEven"  # just something to name the ratio
elements = ["odd", "even"]
ratioTitle = f"{elements[0]}/{elements[1]}"
asymTitle = f"({elements[0]}-{elements[1]})/({elements[0]}+{elements[1]})"
hToPlot = ["SF2D_nominal", "EffData2D", "EffMC2D"]
eras = ["GtoH"]  # ["BtoF", "GtoH"]:

skipAsym = True

scriptDir = os.path.dirname(sys.argv[0])
if len(scriptDir):
    scriptDir += "/"

for wp in workingPoints:
    for era in eras:
        outdir = f"{mainPath}/compareEffAndSFbyEventParity/{era}/{wp}/"
        file1 = f"{inputhPath}/efficiencies_{era}/mu_{wp}_{elements[0]}/allEfficiencies_2D.root"
        file2 = f"{inputhPath}/efficiencies_{era}/mu_{wp}_{elements[1]}/allEfficiencies_2D.root"

        for n in hToPlot:
            h1 = f"{n}_{era}"

            basecmd = f"python {scriptDir}w_mass_13TeV/makeRatioTH2.py {file1} {n} {file2} {n} -o {outdir} -p -n ratio_{tag}_{n}_{era} -x 'Muon #eta' -y 'Muon p_{{T}} (GeV)' -z '{era} {n} ratio::0.97,1.03' --skip1DPlot -t '{ratioTitle}' --palette -1 "
            print()
            print(basecmd)

            basecmd_asym = f"python {scriptDir}w_mass_13TeV/makeRatioTH2.py {file1} {n} {file2} {n} -o {outdir} -a -n asymmetry_{tag}_{n}_{era} -x 'Muon #eta' -y 'Muon p_{{T}} (GeV)' -z '{era} {n} asymmetry::-0.05,0.05' --skip1DPlot -t '{asymTitle}' --palette -1 "
            if not skipAsym:
                print()
                print(basecmd_asym)

    print()
    print()
