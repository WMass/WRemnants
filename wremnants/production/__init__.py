import pathlib

import ROOT

import narf  # noqa: F401 (imported for its side effects)

ROOT.gInterpreter.AddIncludePath(f"{pathlib.Path(__file__).parent}/include/")
