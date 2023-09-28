#!/usr/bin/env python3

import argparse
import os
import sys

import lz4.frame
import pickle
import hist
from utilities import boostHistHelpers as hh, logging
#import wremnants
import hdf5plugin
import h5py
import narf
from narf import ioutils
import ROOT

logger = logging.child_logger(__name__)

if __name__ == "__main__":    

     parser = argparse.ArgumentParser()
     parser.add_argument("inputfile", type=str, nargs=1, help="Input file")
     parsers = parser.add_subparsers(dest='printMode')
     printAll = parsers.add_parser("all", help="Print everything in the input file")
     printProcs = parsers.add_parser("proc", help="Print only names of processes")
     printHist = parsers.add_parser("hist", help="Print more information about histograms")
     # printHist.add_argument("-w", "--what", type=str, default="all",
     #                       choices=["all", "procs", "hists", "axes"],
     #                       help="What to print")
     printHist.add_argument("-p", "--process", type=str, default=None,
                           help="Select this process to print")
     printHist.add_argument("-n", "--histo", type=str, default=None,
                           help="Select single histogram to print with this name, and printout will have more information")
     printHist.add_argument("-a", "--axis", type=str, default=None,
                           help="Select single axis to print with this name")
     printHist.add_argument("--noAxes", "--noAxes", action='store_true',
                           help="Don't print axes")
     args = parser.parse_args()

     h5file = h5py.File(args.inputfile[0], "r")
     results = narf.ioutils.pickle_load_h5py(h5file["results"])

     if args.printMode == "all":
          print(results)
     elif args.printMode == "proc":
          procs = list(filter(lambda x: x != "meta_info", results.keys() ))
          print("\nList of processes:\n")
          for ip,p in enumerate(procs):
               print(f"{ip+1}: {p}")
          print()
     elif args.printMode == "hist":
           print("="*30)
           print("GOING TO PRINT HISTOGRAMS")
           print("="*30)
           #print(results.keys())
           for p in results.keys():
                if p == "meta_info":
                     continue
                if args.process and p != args.process:
                     continue
                print("-"*30)
                print(f"Process {p}")
                space = " "*5
                for k in results[p]["output"].keys():
                     if args.histo and k != args.histo:
                          continue
                     print(f"{space}{k}")                         
                     if not args.noAxes:
                          histObj = results[p]['output'][k]
                          if isinstance(histObj, ioutils.H5PickleProxy):
                               histObj = histObj.get()
                          print(f"{space}  Axes = {histObj.axes.name}")
                          for n in histObj.axes:
                               if args.axis and n.name != args.axis:
                                    continue
                               print(f"{space}{space} {n}")
