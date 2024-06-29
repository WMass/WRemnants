import lz4.frame
import pickle
import argparse
import os
import pathlib
from utilities import logging
from utilities.io_tools import input_tools
from wremnants.datasets.datagroups import Datagroups

parser = argparse.ArgumentParser()
parser.add_argument("infile", type=str, help=".pkl.lz4 from with meta_info")

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
        logger.info(meta_data.get("command", meta_data["meta_info"]["command"] if "meta_info" in meta_data else None))
    else:
        dg = Datagroups(args.infile)
        logger.info(dg.getScriptCommand())

if args.infile.endswith(".root"):
    print_command_from_root(args.infile)
else:
    print_command_from_dict(args.infile)
