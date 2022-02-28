import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--nThreads", type=int, help="number of threads", default=None)
parser.add_argument("--maxFiles", type=int, help="Max number of files (per dataset)", default=-1)
parser.add_argument("--filterProcs", type=str, nargs="*", help="Only run over processes matched by (subset) of name", default=["Wplus", "Wminus", "Zmumu", "Ztautau"])
args = parser.parse_args()

import ROOT
ROOT.gInterpreter.ProcessLine(".O3")
if not args.nThreads:
    ROOT.ROOT.EnableImplicitMT()
elif args.nThreads != 1:
    ROOT.ROOT.EnableImplicitMT(args.nThreads)

import pickle
import gzip

import narf
import wremnants
import hist
import lz4.frame
import logging
import math

filt = lambda x,filts=args.filterProcs: any([f in x.name for f in filts])
datasets = wremnants.datasets2016.getDatasets(maxFiles=args.maxFiles, filt=filt if args.filterProcs else None)

axis_massWgen = hist.axis.Variable([0., math.inf], name="massVgen")

axis_massZgen = hist.axis.Regular(12, 60., 120., name="massVgen")

axis_absYVgen = hist.axis.Variable(
    [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 10], name = "absYVgen"
)
axis_ptVgen = hist.axis.Variable(
    [0, 2, 3, 4, 4.75, 5.5, 6.5, 8, 9, 10, 12, 14, 16, 18, 20, 23, 27, 32, 40, 55, 100], name = "ptVgen"
)
axis_chargeWgen = hist.axis.Regular(
    2, -2, 2, name="chargeVgen", underflow=False, overflow=False
)

axis_chargeZgen = hist.axis.Integer(
    0, 1, name="chargeVgen", underflow=False, overflow=False
)

axis_l_eta_gen = hist.axis.Regular(48, -2.4, 2.4, name = "prefsr_lepton_eta_gen")
axis_l_pt_gen = hist.axis.Regular(29, 26., 55., name = "prefsr_lepton_pt_gen")
axes_l_gen = [axis_l_eta_gen, axis_l_pt_gen]
wprocs = ["WplusmunuPostVFP", "WminusmunuPostVFP", "WminustaunuPostVFP", "WplustaunuPostVFP"]
zprocs = ["ZmumuPostVFP", "ZtautauPostVFP"]


def build_graph(df, dataset):
    print("build graph")
    print(dataset.name)
    results = []

    df = df.Define("weight", "std::copysign(1.0, genWeight)")

    weightsum = df.SumAndCount("weight")

    df = wremnants.define_prefsr_vars(df)

    if dataset.name in zprocs:
        nominal_axes = [axis_massZgen, axis_absYVgen, axis_ptVgen, axis_chargeZgen]
    else:
        nominal_axes = [axis_massWgen, axis_absYVgen, axis_ptVgen, axis_chargeWgen]

    nominal_cols = ["massVgen", "absYVgen", "ptVgen", "chargeVgen"]

    df = df.Define("helicity_moments_scale_tensor", "wrem::makeHelicityMomentScaleTensor(csSineCosThetaPhi, scaleWeights_tensor, weight)")
    helicity_moments_scale = df.HistoBoost("helicity_moments_scale", nominal_axes, [*nominal_cols, "helicity_moments_scale_tensor"], tensor_axes = [wremnants.axis_helicity, *wremnants.scale_tensor_axes])
    results.append(helicity_moments_scale)

    if dataset.name == 'WplusmunuPostVFP':
        df = df.Define('ptPrefsrMuon', 'genlanti.pt()')
        df = df.Define('etaPrefsrMuon', 'genlanti.eta()')
        print("gen info created")
    elif dataset.name == 'WminusmunuPostVFP':
        df = df.Define('ptPrefsrMuon', 'genl.pt()')
        df = df.Define('etaPrefsrMuon', 'genl.eta()')
        print("gen info created")
    if dataset.name in ['WplusmunuPostVFP', 'WminusmunuPostVFP']:
        cols_l_gen = ['etaPrefsrMuon', 'ptPrefsrMuon']
        nominal_cols = [*nominal_cols, *cols_l_gen]
        nominal_axes = [*nominal_axes, axis_l_eta_gen, axis_l_pt_gen]
        print("gen info accessed")
    nominal_gen = df.HistoBoost("nominal_gen", nominal_axes, [*nominal_cols, "weight"])
    results.append(nominal_gen)

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
