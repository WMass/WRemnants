#!/usr/bin/env python3

# this script takes an existing root file with the SF in 2D, and updates some steps
# this can be useful when changes are made or a particular step is modified or being studies with some variations
# a copy of the original file is created, with the new histograms

# example
# python scripts/analysisTools/w_mass_13TeV/mergeFilesSF2D.py wremnants-data/data/muonSF/allSmooth_GtoHout.root allSmooth_GtoHout_vtxAgnIso.root /eos/user/m/mciprian/www/WMassAnalysis/TnP/egm_tnp_analysis/filesFromDavide/SF_vtxAgnostic_10oct2023/GtoH/mu_isonotrig_both/smoothedSFandEffi_isonotrig_GtoH_both.root /eos/user/m/mciprian/www/WMassAnalysis/TnP/egm_tnp_analysis/filesFromDavide/SF_vtxAgnostic_10oct2023/GtoH/mu_iso_both/smoothedSFandEffi_iso_GtoH_both.root /eos/user/m/mciprian/www/WMassAnalysis/TnP/egm_tnp_analysis/filesFromDavide/SF_vtxAgnostic_10oct2023/GtoH/mu_isoantitrig_both/smoothedSFandEffi_isoantitrig_GtoH_both.root /eos/user/m/mciprian/www/WMassAnalysis/TnP/egm_tnp_analysis/filesFromDavide/SF_vtxAgnostic_10oct2023/GtoH/mu_trigger_plus/smoothedSFandEffi_trigger_GtoH_plus.root /eos/user/m/mciprian/www/WMassAnalysis/TnP/egm_tnp_analysis/filesFromDavide/SF_vtxAgnostic_10oct2023/GtoH/mu_trigger_minus/smoothedSFandEffi_trigger_GtoH_minus.root

import os
from copy import *

import utilitiesCMG

utilities = utilitiesCMG.util()

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
from scripts.analysisTools.plotUtils.utility import *

## TODO: move this script to scripts/analysisTools/w_mass_13TeV/

if __name__ == "__main__":

    parser = common_plot_parser()
    parser.add_argument("inputfile", type=str, nargs=1, help="Input root file with TH2")
    parser.add_argument(
        "outputfile", type=str, nargs=1, help="Output file absolute path"
    )
    parser.add_argument(
        "mergefiles",
        type=str,
        nargs="+",
        help="List of files to merge to inputfile into outputfile, keys already in inputfile are overridden, unless --noOverwriteDuplicate is specified",
    )
    parser.add_argument(
        "--noOverwriteDuplicate",
        dest="noOverwriteDuplicate",
        action="store_true",
        help="If a histogram from any of the mergefiles is already present in inputfile, keep the version in inputfile",
    )

    args = parser.parse_args()

    logger = logging.setup_logger(os.path.basename(__file__), 3, True)

    ROOT.TH1.SetDefaultSumw2()

    # protection against deleting original file
    if os.path.abspath(args.outputfile[0]) == os.path.abspath(args.inputfile[0]):
        raise ValueError(
            f"Invalid outputfile name {args.outputfile[0]}, it would overwrite the input file {args.inputfile[0]}"
        )

    # allSmooth_GtoHout.root
    outdir_original = os.path.dirname(os.path.abspath(args.outputfile[0])) + "/"
    outdir = createPlotDirAndCopyPhp(outdir_original)

    outfilenameLocal = outdir + "/" + os.path.basename(args.outputfile[0])
    outfile = safeOpenFile(outfilenameLocal, mode="RECREATE")

    infile = safeOpenFile(
        args.inputfile[0]
    )  # might not work if the input is on eos and one uses the mount
    inputHistnames = []
    for k in infile.GetListOfKeys():
        name = k.GetName()
        inputHistnames.append(name)
        h = safeGetObject(infile, name)
        outfile.cd()
        h.Write(name)
        logger.info(f"Copying {h.ClassName()} {name}")
    infile.Close()

    for mf in args.mergefiles:
        logger.info(f"Opening file {mf}")
        infile = safeOpenFile(mf)
        for k in infile.GetListOfKeys():
            name = k.GetName()
            if name in inputHistnames and args.noOverwriteDuplicate:
                continue
            else:
                h = safeGetObject(infile, name)
                outfile.cd()
                if name in inputHistnames:
                    logger.warning(f"Overwriting {h.ClassName()} {name}")
                    h.Write(name, ROOT.TObject.kOverwrite)
                else:
                    logger.info(f"Copying {h.ClassName()} {name}")
                    h.Write(name)
        infile.Close()

    logger.info(f"Done, closing file {outfile.GetName()}")
    outfile.Close()
    copyOutputToEos(outdir, outdir_original, eoscp=args.eoscp)
