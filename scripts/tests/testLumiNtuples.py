import argparse
from utilities import output_tools,common

parser,initargs = common.common_parser()

import narf
import wremnants
from wremnants import theory_tools,syst_tools,theory_corrections
import hist
import lz4.frame
import logging
import math
import time
import pathlib

data_dir = f"{pathlib.Path(__file__).parent}/../../wremnants/data/"

logging.basicConfig(level=logging.INFO)

parser.add_argument("-e", "--era", type=str, choices=["2016PreVFP","2016PostVFP"], help="Data set to process", default="2016PostVFP")
parser.add_argument("--noScaleFactors", action="store_true", help="Don't use scale factors for efficiency")
parser.add_argument("--muonCorrMag", default=1.e-4, type=float, help="Magnitude of dummy muon momentum calibration uncertainty")
parser.add_argument("--muonCorrEtaBins", default=1, type=int, help="Number of eta bins for dummy muon momentum calibration uncertainty")
parser.add_argument("--lumiUncertainty", type=float, help="Uncertainty for luminosity in excess to 1 (e.g. 1.012 means 1.2\%)", default=1.012)
parser.add_argument("--nano", type=str, help="NanoAOD version to run on", default="tnp")
args = parser.parse_args()

filt = lambda x,filts=args.filterProcs: any([f in x.name for f in filts]) 
datasets = wremnants.datasets2016.getDatasets(maxFiles=args.maxFiles, filt=filt if args.filterProcs else None, 
    nanoVersion=args.nano)

print("Setting option --debug to True")
args.debug = True

era = args.era

# custom template binning
template_neta = int(args.eta[0])
template_mineta = args.eta[1]
template_maxeta = args.eta[2]
print(f"Eta binning: {template_neta} bins from {template_mineta} to {template_maxeta}")
template_npt = int(args.pt[0])
template_minpt = args.pt[1]
template_maxpt = args.pt[2]
print(f"Pt binning: {template_npt} bins from {template_minpt} to {template_maxpt}")

# standard regular axes
axis_eta = hist.axis.Regular(template_neta, template_mineta, template_maxeta, name = "eta")
axis_pt = hist.axis.Regular(template_npt, template_minpt, template_maxpt, name = "pt")

# categorical axes in python bindings always have an overflow bin, so use a regular
# axis for the charge
axis_charge = hist.axis.Regular(2, -2., 2., underflow=False, overflow=False, name = "charge")

# TODO: get from common
axis_passIso = hist.axis.Boolean(name = "passIso")
axis_passMT = hist.axis.Boolean(name = "passMT")
nominal_axes = [axis_eta, axis_pt, axis_charge, axis_passIso, axis_passMT]

axis_ptVgen = hist.axis.Variable(
    common.ptV_10quantiles_binning, 
    name = "ptVgen", underflow=False,
)

pileup_helper = wremnants.make_pileup_helper(era = era)
vertex_helper = wremnants.make_vertex_helper(era = era)


def build_graph(df, dataset):
    print("build graph", dataset.name)
    results = []

    if dataset.is_data:
        df = df.DefinePerSample("weight", "1.0")
    else:
        df = df.Define("weight", "std::copysign(1.0, genWeight)")

    weightsum = df.SumAndCount("weight")

    df = df.Filter("HLT_IsoTkMu24 || HLT_IsoMu24")

    isW = dataset.name in common.wprocs
    isZ = dataset.name in common.zprocs
    isTop = dataset.group == "Top"

    df = df.Define("passMT", "1")
    df = df.Define("passIso", "1")

    df = df.Filter("Sum(Muon_eta) > 0")
    nominal_cols = ["Muon_eta", "Muon_pt", "Muon_charge", "passIso", "passMT"]

    if dataset.is_data:
        df = df.Define("nominal_weight", "1.0")
        nominal = df.HistoBoost("nominal", nominal_axes, nominal_cols)
        results.append(nominal)
                
    else:
        df = df.Define("weight_pu", pileup_helper, ["Pileup_nTrueInt"])
        df = df.Define("weight_vtx", vertex_helper, ["GenVtx_z", "Pileup_nTrueInt"])

        weight_expr = "weight*weight_pu"
        if not args.noVertexWeight:
            weight_expr += "*weight_vtx"
            
        nominal = df.HistoBoost("nominal", nominal_axes, [*nominal_cols, "nominal_weight"])
        results.append(nominal)

    return results, weightsum

resultdict = narf.build_and_run(datasets, build_graph)
output_tools.write_analysis_output(resultdict, "mw_with_mu_eta_pt.pkl.lz4", args)
