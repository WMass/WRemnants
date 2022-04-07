import argparse
from wremnants import theory_tools
parser = argparse.ArgumentParser()
parser.add_argument("--nThreads", type=int, help="number of threads", default=None)
parser.add_argument("--maxFiles", type=int, help="Max number of files (per dataset)", default=-1)
parser.add_argument("--filterProcs", type=str, nargs="*", help="Only run over processes matched by (subset) of name", default=["Wplus", "Wminus", "Zmumu", "Ztautau","data"])
parser.add_argument("--pdfs", type=str, nargs="*", default=["nnpdf31"], choices=theory_tools.pdfMap.keys(), help="PDF sets to produce error hists for")
args = parser.parse_args()

import ROOT
ROOT.gInterpreter.ProcessLine(".O3")
if not args.nThreads:
    ROOT.ROOT.EnableImplicitMT()
elif args.nThreads != 1:
    ROOT.ROOT.EnableImplicitMT(args.nThreads)

import time
time_start = time.time()
import narf
import wremnants
import pickle
import gzip
import hist
import lz4.frame
import logging
import math

filt = lambda x,filts=args.filterProcs: any([f in x.name for f in filts])
datasets = wremnants.datasets2016.getDatasets(maxFiles=args.maxFiles, filt=filt if args.filterProcs else None)

for dataset in datasets:
    print(dataset.name)

axis_massWgen = hist.axis.Variable([0., math.inf], name="massVgen")

axis_massZgen = hist.axis.Regular(12, 60., 120., name="massVgen")

axis_absYVgen = hist.axis.Variable(
    [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 10], name = "absYVgen"
)
axis_ptVgen = hist.axis.Variable(
    list(range(41))+[45, 50, 55, 60, 75, 100], name = "ptVgen"
)

#axis_ptVgen = hist.axis.Regular(120, 0., 120., name="ptVgen")

axis_chargeWgen = hist.axis.Regular(
    2, -2, 2, name="chargeVgen", underflow=False, overflow=False
)

axis_chargeZgen = hist.axis.Integer(
    0, 1, name="chargeVgen", underflow=False, overflow=False
)

axis_l_eta_gen = hist.axis.Regular(48, -2.4, 2.4, name = "prefsr_lepton_eta_gen")
axis_l_pt_gen = hist.axis.Regular(29, 26., 55., name = "prefsr_lepton_pt_gen")
axes_l_gen = [axis_l_eta_gen, axis_l_pt_gen]

wprocs_bugged = [
    "WplusmunuPostVFP", 
    "WminusmunuPostVFP",
    "WminustaunuPostVFP",
    "WplustaunuPostVFP"
]
wprocs_bugfix = [
    "WplusmunuPostVFP_bugfix", 
    "WminusmunuPostVFP_bugfix",
    "WminusmunuPostVFP_bugfix_slc7",
]
wprocs = [*wprocs_bugged, *wprocs_bugfix]
wprocs_bugged_to_check = [ # mu only for making comparison plots between bugged and bugfix samples
    "WplusmunuPostVFP", 
    "WminusmunuPostVFP",
]
zprocs_bugged = ["ZmumuPostVFP", "ZtautauPostVFP"]
zprocs_bugfix = ["ZmumuPostVFP_bugfix"]
zprocs = [*zprocs_bugged, *zprocs_bugfix]
zprocs_bugged_to_check = ["ZmumuPostVFP"]

time_loading_finished = time.time()
loading_time = time_loading_finished - time_start
print("time it takes to load datasets: ", loading_time)

def build_graph(df, dataset):
    time_process_start = time.time()

    print("build graph")
    print(dataset.name)
    results = []
    
    if dataset.is_data:
        df = df.DefinePerSample("weight", "1.0")
    else:
        df = df.Define("weight", "std::copysign(1.0, genWeight)")
    weightsum = df.SumAndCount("weight")


    if dataset.name in zprocs:
        nominal_axes = [axis_massZgen, axis_absYVgen, axis_ptVgen, axis_chargeZgen]
    else:
        nominal_axes = [axis_massWgen, axis_absYVgen, axis_ptVgen, axis_chargeWgen]

    nominal_cols = ["massVgen", "absYVgen", "ptVgen", "chargeVgen"]
    #results.append(theory_tools.make_scale_hist(df, nominal_axes, nominal_cols))
#    for pdf in args.pdfs:
#        results.extend(theory_tools.define_and_make_pdf_hists(df, nominal_axes, nominal_cols, pdfset=pdf))

    if not dataset.is_data:
        df = wremnants.define_prefsr_vars(df)
        df = df.Define("helicity_moments_scale_tensor", "wrem::makeHelicityMomentScaleTensor(csSineCosThetaPhi, scaleWeights_tensor, weight)")
        helicity_moments_scale = df.HistoBoost("helicity_moments_scale", nominal_axes, [*nominal_cols, "helicity_moments_scale_tensor"], tensor_axes = [wremnants.axis_helicity, *wremnants.scale_tensor_axes])

        results.append(helicity_moments_scale)

    if 'WplusmunuPostVFP' in dataset.name:
        print("w process: ", dataset.name)
        df = df.Define('ptPrefsrMuon', 'genlanti.pt()')
        df = df.Define('etaPrefsrMuon', 'genlanti.eta()')
    elif 'WminusmunuPostVFP' in dataset.name or 'ZmumuPostVFP' in dataset.name:
        df = df.Define('ptPrefsrMuon', 'genl.pt()')
        df = df.Define('etaPrefsrMuon', 'genl.eta()')
    if any([mc in dataset.name for mc in ['WplusmunuPostVFP', 'WminusmunuPostVFP', 'ZmumuPostVFP']]):
        nominal_cols = [*nominal_cols, 'etaPrefsrMuon', 'ptPrefsrMuon']
        nominal_axes = [*nominal_axes, axis_l_eta_gen, axis_l_pt_gen]
        print("mc process: ", dataset.name)
    if not dataset.is_data:
        nominal_gen = df.HistoBoost("nominal_gen", nominal_axes, [*nominal_cols, "weight"])
        results.append(nominal_gen)

    time_process_end = time.time()
    process_time = time_process_end - time_process_start
    print("Time it takes to process ", dataset.name, ": ", process_time)

    return results, weightsum

resultdict = narf.build_and_run(datasets, build_graph)

fname = "w_z_gen_dists_slc7.pkl.lz4"

print("writing output")
with lz4.frame.open(fname, "wb") as f:
    pickle.dump(resultdict, f, protocol = pickle.HIGHEST_PROTOCOL)

'''
print("computing angular coefficients")

z_moments_bugged = None
z_moments_bugfix = None
w_moments_bugged = None
w_moments_bugfix = None

not_data = lambda x: "data" not in x[0]

for key, val in filter(not_data, resultdict.items()):
    moments = val["output"]["helicity_moments_scale"]
    if key in zprocs_bugged_to_check:
        if z_moments_bugged is None:
            z_moments_bugged = moments
            print("the integral of ", key, "is ", moments.sum())
        else:
            # gen level kinematics, stack tau to mu channel to increase stats
            z_moments_bugged += moments
            print("the integral of ", key, "is ", moments.sum())
    elif key in wprocs_bugged_to_check:
        if w_moments_bugged is None:
            w_moments_bugged = moments
        else:
            w_moments_bugged += moments
    elif key in zprocs_bugfix:
        if z_moments_bugfix is None:
            z_moments_bugfix = moments
        else:
            z_moments_bugfix += moments
    elif key in wprocs_bugfix:
        if w_moments_bugfix is None:
            w_moments_bugfix = moments
        else:
            w_moments_bugfix += moments

z_coeffs_bugged = wremnants.moments_to_angular_coeffs(z_moments_bugged)
w_coeffs_bugged = wremnants.moments_to_angular_coeffs(w_moments_bugged)
z_coeffs_bugfix = wremnants.moments_to_angular_coeffs(z_moments_bugfix)
w_coeffs_bugfix = wremnants.moments_to_angular_coeffs(w_moments_bugfix)

with lz4.frame.open("z_coeffs_bugged.pkl.lz4", "wb") as f:
    pickle.dump(z_coeffs_bugged, f, protocol = pickle.HIGHEST_PROTOCOL)

with lz4.frame.open("w_coeffs_bugged.pkl.lz4", "wb") as f:
    pickle.dump(w_coeffs_bugged, f, protocol = pickle.HIGHEST_PROTOCOL)

with lz4.frame.open("z_coeffs_bugfix.pkl.lz4", "wb") as f:
    pickle.dump(z_coeffs_bugfix, f, protocol = pickle.HIGHEST_PROTOCOL)

with lz4.frame.open("w_coeffs_bugfix.pkl.lz4", "wb") as f:

with lz4.frame.open("z_coeffs_bugged_fine_bin.pkl.lz4", "wb") as f:
    pickle.dump(z_coeffs_bugged, f, protocol = pickle.HIGHEST_PROTOCOL)

with lz4.frame.open("w_coeffs_bugged_fine_bin.pkl.lz4", "wb") as f:
    pickle.dump(w_coeffs_bugged, f, protocol = pickle.HIGHEST_PROTOCOL)

with lz4.frame.open("z_coeffs_bugfix_fine_bin.pkl.lz4", "wb") as f:
    pickle.dump(z_coeffs_bugfix, f, protocol = pickle.HIGHEST_PROTOCOL)

with lz4.frame.open("w_coeffs_bugfix_fine_bin.pkl.lz4", "wb") as f:
    pickle.dump(w_coeffs_bugfix, f, protocol = pickle.HIGHEST_PROTOCOL)
'''
