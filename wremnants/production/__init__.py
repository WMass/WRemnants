import pathlib

import ROOT

import narf

ROOT.gInterpreter.AddIncludePath(f"{pathlib.Path(__file__).parent}/include/")

narf.clingutils.Load("libHist")
