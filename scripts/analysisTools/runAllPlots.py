#!/usr/bin/env python3

### Wrapper to run all plotting scripts in one go out of the fit results
### This runs the minimal command, which is usually sufficient
### Each script has additional options for more customization, please check them

import os
import sys

sys.path.append(os.getcwd() + "/plotUtils/")
from plotUtils.utility import safeSystem

# from runAllPlots_customSetting import *

# general settings
# could make these command-line options, but it would hide other necessary changes inside this script)
dryRun = 1  # run or just print commands
isWlike = 0  # Wmass or Wlike
skipData = 1  # or set fits = ["Asimov"]
onlyData = 0  # or set fits = ["Data"]
fits = ["Asimov", "Data"]

# select what to plot
skipHistograms = 1  # prefit histograms, can't be made also before running the fit
skipImpacts = 1
skipNuisances = 0
skipSystRatios = 1
skipPostfitHistograms = 1  # prefit and postfit histograms, from fitresults.root

# SPECIFIC PATH CUSTOMIZED BY EACH USER (PATH INSIDE $COMBINE_STUDIES)
customPath = "smoothSF/muonCorr_none/scetlibCorr_nnpdf31/byHelicityPtCharge"  # contains the root file with TH2
subFolder = (
    "nominal"  # folder inside customPath: it contains the final cards and fit results
)

##################################
## These should not be touched
##################################
combine_studies_path = os.environ["COMBINE_STUDIES"]
plots_path = os.environ["PLOTS"]
what = "ZMassWLike" if isWlike else "WMass"
inputPath = f"{combine_studies_path}/{what}/{customPath}/"  # contains the TH2 histograms for combine
outputPath = (
    f"{plots_path}/fromMyWremnants/{what}_fit/{customPath}/"  # where to store plots
)
combineInputFile = f"{inputPath}/{what}CombineInput.root"
useSmoothSF = (
    False if "binnedSF" in inputPath else True
)  # FIXME: a bit hardcoded for now
##################################

#################################################
# dictionaries to customize some specific plots #
#################################################
#
# postfix for plot name and regular expression to pick histogram variations to plot
# there are infinite combinations, so this is mainly an example for Up variations
systRatiosDict = {
    "effStat_trigger_eta20_plus_Up": ".*effStat.*_trigger_eta20.*q1.*Up",
    "pdfsAndAlphaS": ".*pdf(12|25|50|.*AlphaS).*Up",
}
#
# unique string for output plot name and regular expression for nuisances used to plot pulls and constraints
diffNuisanceDict = {
    "effStat_triggerPlus": ".*effStat.*_trigger.*q1",
    "effStat_triggerMinus": ".*effStat.*_trigger.*q0",
    "effStat_idipPlus": ".*effStat.*_idip.*q1",
    "effStat_idipMinus": ".*effStat.*_idip.*q0",
    "effStat_recoPlus": ".*effStat.*_reco.*q1",
    "effStat_recoMinus": ".*effStat.*_reco.*q0",
    "effStat_trackingPlus": ".*effStat.*_tracking.*q1",
    "effStat_trackingMinus": ".*effStat.*_tracking.*q0",
    "effSyst": ".*effSyst.*",
    "prefireL1": ".*refire.*",
    "pdfAndAlphaS": ".*pdf(\d+|.*AlphaS)",
}
if useSmoothSF:
    diffNuisanceDict["effStat_isoEffData"] = ".*effStat.*_iso_effData"
    diffNuisanceDict["effStat_isoEffMC"] = ".*effStat.*_iso_effMC"
else:
    diffNuisanceDict["effStat_isoSF"] = ".*effStat.*_iso"
#################################################

print()


##############################
# utility functions used here
def printText(text):
    print("")
    print("=" * 30)
    print(text)
    print("")


#################################################

scriptDir = os.path.dirname(sys.argv[0])
if len(scriptDir):
    scriptDir += "/"

printText("PREFIT HISTOGRAMS")
command = f"python {scriptDir}w_mass_13TeV/plotPrefitTemplatesWRemnants.py {combineInputFile} {outputPath}/plotPrefitTemplatesWRemnants/"
if isWlike:
    command += " --wlike"
if not skipHistograms:
    print()
    safeSystem(command, dryRun=dryRun)


printText("HISTOGRAM SYST RATIOS")
# the command below does charge plus, but it is just for example
processes = "Zmumu" if isWlike else "Wmunu,Fake"
command = f"python {scriptDir}w_mass_13TeV/makeSystRatios.py {combineInputFile} {outputPath}/makeSystRatios/"
command += f" -p {processes} -c plus --plotStat"
if not skipSystRatios:
    for key, value in systRatiosDict.items():
        trueCommand = command + f" --systPostfix {key} -s '{value}'"
        print()
        safeSystem(trueCommand, dryRun=dryRun)


## Now produce plots from fit results for data or Asimov (toys to be implemented)
for fit in fits:

    if skipData and fit == "Data":
        continue
    if onlyData and fit != "Data":
        continue

    typedir = "hessian" if fit == "Asimov" else "data"

    printText(f"Running plots for {fit} fit")
    fitresultsFile = f"{inputPath}/{subFolder}/fit/{typedir}/fitresults_123456789_{fit}_bbb1_cxs0.root"

    ########################################
    printText("IMPACTS")
    statImpact = "0.0710" if isWlike else "0.0230"
    # TODO: add cases for single charge (change input to --set-stat and use "--postfix charge" not to overwrite plots)
    command = f"python {scriptDir}w_mass_13TeV/makeImpactsOnMW.py {fitresultsFile} -o {outputPath}/makeImpactsOnMW/"
    command += f" --set-stat {statImpact} --showTotal --scaleToMeV"
    if not skipImpacts:
        print()
        safeSystem(command, dryRun=dryRun)

    ########################################
    printText("NUISANCES AND COSTRAINTS")
    yAxisSetting = (
        " --y-setting -1.0 -0.5 0 0.5 1.0"
        if fit == "Asimov"
        else " --y-setting -5.0 -3.0 0 3.0 5.0"
    )
    command = f"python {scriptDir}w_mass_13TeV/diffNuisances.py {fitresultsFile} -o {outputPath}/diffNuisances/ "
    command += f" -a --format html --type hessian  --suffix {fit} {yAxisSetting}"
    if not skipNuisances:
        for key, value in diffNuisanceDict.items():
            trueCommand = command + f" --uniqueString {key} -p '{value}'"
            print()
            safeSystem(trueCommand, dryRun=dryRun)

    ########################################
    printText("PREFIT AND POSTFIT HISTOGRAMS (YIELDS AND UNCERTAINTIES)")
    command = f"python {scriptDir}w_mass_13TeV/postFitHistograms.py {fitresultsFile} -o {outputPath}/postFitHistograms/ "
    command += f" --suffix postVFP -l 16.8 --no2Dplot"  # remove --no2Dplot to add all 2D histograms
    if not skipPostfitHistograms:
        print()
        safeSystem(command, dryRun=dryRun)

print()
print("THE END!")
print()
