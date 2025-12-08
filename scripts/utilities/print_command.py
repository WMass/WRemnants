import argparse
import os
import pathlib

from utilities.io_tools import input_tools
from wremnants.datasets.datagroups import Datagroups
from wums import logging

parser = argparse.ArgumentParser()
parser.add_argument("infile", type=str, help=".pkl.lz4 from with meta_info")
parser.add_argument("-a", "--all", action="store_true", help="Print all meta info")
parser.add_argument("--timestamp", action="store_true", help="Print timestamp")
parser.add_argument("--hash", action="store_true", help="Print git hash")
parser.add_argument("--diff", action="store_true", help="Print git diff")

args = parser.parse_args()

logger = logging.setup_logger(__file__)

if not os.path.isfile(args.infile):
    raise ValueError(f"{args.infile} is not a valid file!")

exts = pathlib.Path(args.infile).suffixes


def print_command_from_root(rtfile_name):
    import ROOT

    rtfile = ROOT.TFile.Open(rtfile_name)
    command = rtfile.Get("meta_info/command")
    logger.info(command.GetTitle())


def print_command_from_dict(infile):
    meta_data = input_tools.get_metadata(infile)
    if meta_data is not None:

        def get(arg):
            return meta_data.get(
                arg,
                meta_data["meta_info"][arg] if "meta_info" in meta_data else None,
            )

        if not args.all:
            logger.info(get("command"))
            if args.timestamp:
                logger.info("Timestamp: " + get("time"))
            if args.hash:
                logger.info("Git hash: " + get("git_hash"))
            if args.diff:
                logger.info("Git diff: " + get("git_diff"))
        else:
            logger.info("Full metadata:")
            for k in meta_data.keys():
                v = get(k)
                if type(v) is dict:
                    logger.info(
                        f"{k}: "
                        + "\n"
                        + "\n\t".join([f"{kk}: {vv}" for kk, vv in v.items()])
                        + "\n"
                    )
                else:
                    logger.info(f"{k}: {v}")
    else:
        dg = Datagroups(args.infile)
        logger.info(dg.getScriptCommand())


if args.infile.endswith(".root"):
    print_command_from_root(args.infile)
else:
    print_command_from_dict(args.infile)
