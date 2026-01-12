#!/usr/bin/env python3

# example
# python scripts/utilities/check_root_files.py /eos/cms/store/cmst3/group/wmass/w-mass-13TeV/NanoAOD/ -m ".*ST_"

import argparse
import os
import re

## safe batch mode
import sys

args = sys.argv[:]
sys.argv = ["-b"]
import ROOT

sys.argv = args
ROOT.gROOT.SetBatch(True)
ROOT.PyConfig.IgnoreCommandLineOptions = True

from wums import logging


def isBadRootFile(fname, checkTree=True, treeName="Events"):
    try:
        tf = ROOT.TFile.Open(fname)
        if not tf or tf.IsZombie():
            return True
        if tf.TestBit(ROOT.TFile.kRecovered):
            return True
        if checkTree:
            tree = tf.Get(treeName)
            if not tree:
                return True
    except IOError as e:
        logger.error(e)
        return True
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("inputpath", type=str, help="Input path where files are stored")
    parser.add_argument(
        "-m",
        "--match",
        type=str,
        default=None,
        help="Regular expression to select only specific subpaths",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        type=int,
        default=3,
        choices=[0, 1, 2, 3, 4],
        help="Set verbosity level with logging, the larger the more verbose",
    )
    parser.add_argument(
        "-s",
        "--save",
        type=str,
        default=None,
        help="Save list of bad files in a file, specifying its name",
    )
    parser.add_argument(
        "-a",
        "--append",
        action="store_true",
        help="Whenb using -s, append list to existing file",
    )
    args = parser.parse_args()

    logger = logging.setup_logger(os.path.basename(__file__), args.verbose)

    regexp = None
    if args.match:
        regexp = re.compile(args.match)

    badFiles = []

    for dirpath, dirnames, filenames in os.walk(args.inputpath):
        if regexp is None or regexp.match(dirpath):
            for f in filenames:
                fullName = os.path.join(dirpath, f)
                if not (f.endswith(".root") and os.path.isfile(fullName)):
                    continue
                if isBadRootFile(fullName):
                    badFiles.append(fullName)

    if len(badFiles):
        logger.warning("List of bad files")
        logger.warning("-" * 30)
        for i, f in enumerate(badFiles):
            logger.warning(f"{str(i).rjust(4)}: {f}")
        logger.warning("-" * 30)
        logger.warning(f"Found {len(badFiles)} bad files")
        if args.save is not None:
            fname = args.save
            outfileDir = os.path.abspath(os.path.dirname(fname)) + "/"
            if not os.path.exists(outfileDir):
                os.makedirs(outfileDir)
            with open(fname, "a" if args.append else "w") as f:
                for l in badFiles:
                    f.write(f"{l}\n")
            logger.info(f"List of bad files saved in {fname}")
    else:
        logger.info("No bad files found")
        if args.save is not None:
            logger.warning("Skipping creation of output file to store the list")
