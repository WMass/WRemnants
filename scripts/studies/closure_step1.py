import os
from utilities import logging
import argparse 
import itertools
import glob

logger = logging.setup_logger(__file__, 3, True)

# Perform bias tests on different (pseudo) data sets. Use one dataset + uncertainties as nominal and fit the other set as pseudodata.
#   The shift in the mass parameter is the bias
# This script is step 1 and done within the WRemnants singularity and produces the histograms and combine inputs
#   A second script, `biastests_step2.py` , has to be run in the cmsssw-7 environment and the fits are performed and the information is read out

parser = argparse.ArgumentParser()
parser.add_argument("--histmaker", type=str, help="histmaker to perform bias tests with", 
    default="mw_with_mu_eta_pt", choices=["mw_with_mu_eta_pt","mz_wlike_with_mu_eta_pt", "mz_dilepton"])
parser.add_argument("-o", "--outfolder", type=str, default="./hdf5Files/nonClosureCorl", help="Output folder")
parser.add_argument("--combineOutFolder", type=str, default="./CombineStudies/nonClosureCorlSigOnly/", help='output folder for combine files')
parser.add_argument("--defaultPostfix", type=str, default="scetlib_dyturboCorr", help='the postfix string added to the hdf5 file by default by the histmaker')
parser.add_argument("-j", "--nThreads", type=int, default=192, help="number of threads to run the histmaker")
parser.add_argument("--infoOnly", action="store_true", help="only print the commands without actually running them") 

args = parser.parse_args()

datasets = [
    # ("default", "--smearing"),
    # ("default", "--smearing --bias-calibration parameterized"),
    # ("default", "--smearing --bias-calibration A"),
    # ("default", "--smearing --bias-calibration M"),
    ("--smearing --nonClosureScheme A-M-separated  --correlatedNonClosureNP", "--smearing --biasCalibration parameterized"),
    ("--smearing --nonClosureScheme binned --correlatedNonClosureNP", "--smearing --biasCalibration binned"),
    #("--smearing --nonClosureScheme binned-plus-M", "--smearing --biasCalibration parameterized"),
    #("--smearing --nonClosureScheme binned-plus-M", "--smearing --biasCalibration binned")
    #("--smearing", "--smearing --biasCalibration binned"),
    #("--smearing", "--smearing --biasCalibration A"),
    #("--smearing", "--smearing --biasCalibration M"),
]

freeze_uncertainties = [
    ("reducedUncertainties", "-x '.*' -k '(mass.*|muonScaleSyst.*|CMS_scale_m.*|Z_non_closure.*|Z_nonClosure.*)'"), # freeze all nuisances except momentum scale #-k '(CMS_scale_m.*|mass.*|Z_nonClosure.*)'
    ("fullUncertainties", "")
]

hists_to_plot = {
    "mw_with_mu_eta_pt": "pt eta pt-eta",
    "mz_wlike_with_mu_eta_pt": "pt eta pt-eta",
    "mz_dilepton": "mll",
}
channels_to_plot = {
    "mw_with_mu_eta_pt": ["plus","minus","all"],
    "mz_wlike_with_mu_eta_pt": ["plus","minus","all"],
    "mz_dilepton": ["all",],
}

histmaker = args.histmaker

hists_to_plot = hists_to_plot[histmaker]
channels_to_plot = channels_to_plot[histmaker]

options = [
    # "--theory_corr scetlib",
    # "--no-recoil"
]
options = " ".join(options)

if not os.path.isdir(args.outfolder):
    os.mkdir(args.outfolder)

def EXE(command):
    logger.info(command) 
    if not args.infoOnly:
        os.system(command)  # for testing comment out this line

def make_appendix(name):
        
    parts = []
    for p in [pa.replace("-"," ").replace("_"," ") for pa in name.split("--")]:
        if p=="":
            continue
        ps = p.split(" ")
        ps = "".join([ps[0],]+[p.capitalize() for p in ps[1:]])
        parts.append(ps)
    
    return "_".join(parts)

    file_sep_impact = ""
for nominal, pseudodata in datasets:
    
    if nominal == pseudodata:
        continue

    logger.info(f"Now at {nominal} vs {pseudodata}")
    
    arg_nominal = nominal if nominal != "default" else ""
    str_nominal = make_appendix(nominal)

    arg_pseudodata = pseudodata if pseudodata != "default" else ""
    str_pseudodata = make_appendix(pseudodata)

    # we never use biased datasets as nominal, thus we can skip expensive uncertainty calculation
    if "bias" in arg_pseudodata:
        arg_pseudodata += " --onlyMainHistograms"

    file_nominal = f"{args.outfolder}/{histmaker}_{args.defaultPostfix}_{str_nominal}.hdf5"
    file_sep_impact = file_nominal if not "BinnedPlus" in file_nominal else file_sep_impact
    file_pseudodata = f"{args.outfolder}/{histmaker}_{args.defaultPostfix}_{str_pseudodata}.hdf5"

    if not os.path.isfile(file_nominal):
        # run histmaker for nominal
        #EXE(f"python3 scripts/histmakers/{histmaker}.py -j {args.nThreads} -o {args.outfolder} {options} {arg_nominal} -p {str_nominal}")
        print('.')
    else:
        logger.info(f"Found file for nominal {nominal}")

    if not os.path.isfile(file_pseudodata):
        # run histmaker for pseudodata
        #EXE(f"python3 scripts/histmakers/{histmaker}.py -j {args.nThreads} -o {args.outfolder} {options} {arg_pseudodata} -p {str_pseudodata}")
        print('.')
    else:
        logger.info(f"Found file for pseudodata {pseudodata}")
        
    # make control plots
    # webDir = args.outfolder.split("/")[-1] +f"/{str_nominal}_vs_{str_pseudodata}"
    
    # if args.mode=="pdf":
    #     variation = f"variation --varName pdf{str_nominal.upper()}Up pdf{str_nominal.upper()}Down pdf{str_pseudodata.upper()} --transform --varLabel {str_nominal.upper()} $\pm 1 \sigma$  {str_pseudodata.upper()} --color grey grey --fill-between"
    # else:
    #     variation = ""
    
    # for channel in channels_to_plot:
    #     if len(glob.glob(f"/home/d/dwalter/www/WMassAnalysis/{webDir}/*_{channel}_*.pdf")) == 0:
    #         EXE(f"python3 scripts/plotting/makeDataMCStackPlot.py --hists {hists_to_plot} --yscale 1.6 -r 0.95 1.05 --channel {channel} -f {webDir} -a {str_nominal} {file_nominal} {variation}")
    #         EXE(f"python3 scripts/plotting/makeDataMCStackPlot.py --hists {hists_to_plot} --yscale 1.6 -r 0.95 1.05 --channel {channel} -f {webDir} -a {str_pseudodata} {file_pseudodata} {variation}")

    # make combine input        
    for freeze_name, freeze_command in freeze_uncertainties:

        dir_combine = f"{args.combineOutFolder}/{str_nominal}_vs_{str_pseudodata}_{freeze_name}"
        
        #if os.path.isdir(dir_combine):
        #    logger.warning(f"The combine file for {dir_combine} already exists, continue with the next one!")
        #    continue
        #if "reduced" in dir_combine: continue
        file_nominal_comb = file_nominal.replace("correlatedNonClosureNP", "correlatedNonClosureNP_NonClosureCorl")
        EXE(f"python3 scripts/combine/setupCombineWMass.py -o {dir_combine} -i {file_nominal_comb} --pseudoDataFile {file_pseudodata} --pseudoData nominal --muonScaleVariation smearingWeights {freeze_command} --correlatedNonClosureNuisances --filterProcGroups Wmunu --excludeProcGroups Fake")
#dir_combine = f"{args.combineOutFolder}/sepImpact"
#if not os.path.isfile(f"{dir_combine}/WMassCombineInput.root"):
#    EXE(f"python3 scripts/combine/setupCombineWMass.py -o {dir_combine} -i {file_sep_impact} --muonScaleVariation smearingWeights --sepImpactForNC")
