import argparse
import narf
import wremnants
import pickle
import gzip
from wremnants import theory_tools
import hist
import lz4.frame
import logging
import math

parser = argparse.ArgumentParser()
parser.add_argument("--nThreads", type=int, help="number of threads", default=None)
parser.add_argument("--maxFiles", type=int, help="Max number of files (per dataset)", default=-1)
parser.add_argument("--filterProcs", type=str, nargs="*", help="Only run over processes matched by (subset) of name", default=["Wplus", "Wminus", "Zmumu", "Ztautau"])
parser.add_argument("--pdfs", type=str, nargs="*", default=["nnpdf31"], choices=theory_tools.pdfMap.keys(), help="PDF sets to produce error hists for")
args = parser.parse_args()

import ROOT
ROOT.gInterpreter.ProcessLine(".O3")
if not args.nThreads:
    ROOT.ROOT.EnableImplicitMT()
elif args.nThreads != 1:
    ROOT.ROOT.EnableImplicitMT(args.nThreads)

filt = lambda x,filts=args.filterProcs: any([f in x.name for f in filts])
datasets = wremnants.datasets2016.getDatasets(maxFiles=args.maxFiles, filt=filt if args.filterProcs else None)

axis_massWgen = hist.axis.Variable([0., math.inf], name="massVgen")

axis_massZgen = hist.axis.Regular(12, 60., 120., name="massVgen")

axis_absYVgen = hist.axis.Variable(
    [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 10], name = "absYVgen"
)
axis_ptVgen = hist.axis.Variable(
    list(range(41))+[45, 50, 55, 60, 75, 100], name = "ptVgen"
)
axis_chargeWgen = hist.axis.Regular(
    2, -2, 2, name="chargeVgen", underflow=False, overflow=False
)

axis_chargeZgen = hist.axis.Integer(
    0, 1, name="chargeVgen", underflow=False, overflow=False
)

wprocs = ["WplusmunuPostVFP", "WminusmunuPostVFP", "WminustaunuPostVFP", "WplustaunuPostVFP"]
zprocs = ["ZmumuPostVFP", "ZtautauPostVFP"]


def build_graph(df, dataset):
    print("build graph")
    print(dataset.name)
    results = []

    df = df.Define("nominal_weight", "std::copysign(1.0, genWeight)")

    weightsum = df.SumAndCount("nominal_weight")

    df = theory_tools.define_prefsr_vars(df)
    df = theory_tools.define_scale_tensor(df)

    if dataset.name in zprocs:
        nominal_axes = [axis_massZgen, axis_absYVgen, axis_ptVgen, axis_chargeZgen]
    else:
        nominal_axes = [axis_massWgen, axis_absYVgen, axis_ptVgen, axis_chargeWgen]

    nominal_cols = ["massVgen", "absYVgen", "ptVgen", "chargeVgen"]
    results.append(theory_tools.make_scale_hist(df, nominal_axes, nominal_cols))
    for pdf in args.pdfs:
        results.extend(theory_tools.define_and_make_pdf_hists(df, nominal_axes, nominal_cols, pdfset=pdf))

    nominal_gen = df.HistoBoost("nominal_gen", nominal_axes, [*nominal_cols, "nominal_weight"])
    results.append(nominal_gen)

    df = df.Define("helicity_moments_scale_tensor", "wrem::makeHelicityMomentScaleTensor(csSineCosThetaPhi, scaleWeights_tensor, nominal_weight)")
    helicity_moments_scale = df.HistoBoost("helicity_moments_scale", nominal_axes, [*nominal_cols, "helicity_moments_scale_tensor"], tensor_axes = [wremnants.axis_helicity, *wremnants.scale_tensor_axes])
    results.append(helicity_moments_scale)

    return results, weightsum

resultdict = narf.build_and_run(datasets, build_graph)

fname = "w_z_gen_dists.pkl.lz4"

print("writing output")
with lz4.frame.open(fname, "wb") as f:
    pickle.dump(resultdict, f, protocol = pickle.HIGHEST_PROTOCOL)

print("computing angular coefficients")

z_moments = None
w_moments = None

for key, val in resultdict.items():
    moments = val["output"]["helicity_moments_scale"]
    if key in zprocs:
        if z_moments is None:
            z_moments = moments
        else:
            z_moments += moments
    elif key in wprocs:
        if w_moments is None:
            w_moments = moments
        else:
            w_moments += moments

z_coeffs = wremnants.moments_to_angular_coeffs(z_moments)
w_coeffs = wremnants.moments_to_angular_coeffs(w_moments)

with lz4.frame.open("z_coeffs.pkl.lz4", "wb") as f:
    pickle.dump(z_coeffs, f, protocol = pickle.HIGHEST_PROTOCOL)

with lz4.frame.open("w_coeffs.pkl.lz4", "wb") as f:
    pickle.dump(w_coeffs, f, protocol = pickle.HIGHEST_PROTOCOL)
