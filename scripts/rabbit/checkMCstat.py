#!/usr/bin/env python3

# create dummy histogram with 1 bin changing number of events and event weight, and run expected fit with data or data+MC stat only
# currently works only after creating singularity image with cmssw-cc7, to use cmsenv from the combine fit

import argparse
import math
import os
import string

## safe batch mode
import sys

args = sys.argv[:]
sys.argv = ["-b"]
import ROOT

from wums import logging

logger = logging.child_logger(__name__)

sys.argv = args
ROOT.gROOT.SetBatch(True)
ROOT.PyConfig.IgnoreCommandLineOptions = True


def safeOpenFile(fileName, quitOnFail=True, silent=False, mode="READ"):
    fileObject = ROOT.TFile.Open(fileName, mode)
    if not fileObject or fileObject.IsZombie():
        if not silent:
            logger.error("Could not open file {f}".format(f=fileName))
        if quitOnFail:
            quit()
        else:
            return None
    elif not fileObject.IsOpen():
        if not silent:
            logger.error("File {f} was not opened".format(f=fileName))
        if quitOnFail:
            quit()
        else:
            return None
    else:
        return fileObject


def safeSystem(cmd, dryRun=False, quitOnFail=True):
    print(cmd)
    if not dryRun:
        res = os.system(cmd)
        if res:
            print("-" * 30)
            print(
                "safeSystem(): error occurred when executing the following command. Aborting"
            )
            print(cmd)
            print("-" * 30)
            if quitOnFail:
                quit()
        return res
    else:
        return 0


def createFolder(checkdir, dryRun=False):
    if not os.path.exists(checkdir):
        print("Creating folder", checkdir)
        safeSystem("mkdir -p " + checkdir, dryRun=dryRun)


def readTemplate(templateFile, templateDict, filt=None):
    if not os.path.isfile(templateFile):
        raise ValueError("Template file %s is not a valid file!" % templateFile)
    with open(templateFile, "r") as tf:
        lines = filter(filt, tf.readlines()) if filt else tf.readlines()
        source = string.Template("".join(lines))
    filled = source.substitute(templateDict)
    return filled


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "outdir", type=str, nargs=1, help="output directory to save things"
    )
    parser.add_argument(
        "-w",
        "--weight",
        default="1.0",
        type=float,
        help="Weight to fill dummy histogram",
    )
    parser.add_argument(
        "-d",
        "--dryRun",
        action="store_true",
        help="Do not execute commands, just print them",
    )

    args = parser.parse_args()

    if not args.dryRun:
        try:
            cmssw = os.environ["CMSSW_BASE"]
        except:
            cmssw = ""
        if cmssw == "":
            print("\n")
            print("Error: to use combinetf you need to activate cmsenv from a release.")
            print(
                "You should work from a cmssw-cc7 singularity environment to get the release."
            )
            print("Aborting ...")
            print("\n")
            quit()

    outdir = args.outdir[0]
    ROOT.TH1.SetDefaultSumw2()
    createFolder(outdir, False)

    nominalTemplate = "scripts/combine/Templates/WMass/main.txt"

    histName = "x"
    signalName = "sig"
    procs = [signalName]
    processLabels = [0]
    nprocs = len(procs)
    dataName = "Data"
    chan = "chan"

    cardArgs = {
        "channel": chan,
        "channelPerProc": chan.ljust(10) * nprocs,
        "processes": "".join([x.ljust(10) for x in procs]),
        "labels": "".join([str(x).ljust(10) for x in processLabels]),
        "rates": "-1".ljust(10) * nprocs,
        "inputfile": None,
        "dataName": dataName,
        "histName": histName,
        "pseudodataHist": "{h}_{d}".format(h=histName, d=dataName),
    }

    eventsForTest = [math.pow(100, i) for i in range(1, 6)]
    for i, nEvts in enumerate(eventsForTest):
        hdata = ROOT.TH1D(
            "{h}_{d}_{i}".format(h=histName, d=dataName, i=i),
            "Nevts = {n}".format(n=nEvts),
            1,
            0,
            1,
        )
        hdata.SetBinContent(1, nEvts * args.weight)
        hdata.SetBinError(1, math.sqrt(nEvts * args.weight))
        hmc = ROOT.TH1D(
            "{h}_{s}_{i}".format(h=histName, s=signalName, i=i),
            "Nevts = {n}, weight = {w}".format(n=nEvts, w=args.weight),
            1,
            0,
            1,
        )
        hmc.SetBinContent(1, nEvts * args.weight)
        hmc.SetBinError(1, math.sqrt(nEvts) * args.weight)

        fdir = outdir + "/" + str(int(nEvts)) + "/"
        createFolder(fdir, False)
        fname = fdir + "checkMCstat.root"
        rf = safeOpenFile(fname, mode="RECREATE")
        hdata.Write(histName + "_" + dataName + "_" + chan)
        hmc.Write(histName + "_" + signalName + "_" + chan)
        rf.Close()

        cardArgs["inputfile"] = fname
        cardName = fdir + "/card.txt"
        cardContent = readTemplate(nominalTemplate, cardArgs)
        include = ["1.000001".ljust(10) for x in procs]
        cardContent += "syst     lnN      {}\n".format("".join(include))
        cardContent += "\n"
        cardContent += "group allSysts = syst\n"

        with open(cardName, "w") as card:
            card.write(cardContent)
            card.write("\n")

        txt2hdf5Cmd = (
            "text2hdf5.py {i} --dataset data_obs --X-allow-no-background".format(
                i=cardName
            )
        )
        safeSystem(txt2hdf5Cmd, dryRun=args.dryRun)

        hdf5 = cardName.replace(".txt", ".hdf5")
        combineCmd = "combinetf.py -t -1 --binByBinStat {hdf5} --doImpacts --saveHists --computeHistErrors --doh5Output --postfix bbb1 --outputDir {fd}".format(
            hdf5=hdf5, fd=fdir
        )
        print("\n### Fit with BBB")
        safeSystem(combineCmd, dryRun=args.dryRun)
        combineCmd = "combinetf.py -t -1 {hdf5} --doImpacts --saveHists --computeHistErrors --doh5Output --postfix bbb0 --outputDir {fd}".format(
            hdf5=hdf5, fd=fdir
        )
        print("\n### Fit without BBB")
        safeSystem(combineCmd, dryRun=args.dryRun)
        print("\n\n")
