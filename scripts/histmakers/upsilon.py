import math
import os

from wremnants.utilities import common, parsing
from wums import logging

analysis_label = common.analysis_label(os.path.basename(__file__))
parser, initargs = parsing.common_parser(analysis_label)

args = parser.parse_args()

logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)

import hist

import narf
from wremnants.production.datasets.dataset_tools import getDatasets
from wremnants.production.histmaker_tools import write_analysis_output

datasets = getDatasets(
    maxFiles=args.maxFiles,
    filt=args.filterProcs,
    excl=args.excludeProcs,
    base_path=args.dataPath,
    era=args.era,
    data_tags=[""],
    mc_tags=[""],
)

# define histogram axes, see: https://hist.readthedocs.io/en/latest/index.html
axis_nLepton = hist.axis.Integer(0, 5, name="nLepton", underflow=False)

all_axes = {
    "mll": hist.axis.Regular(20, 8.5, 11.5, name="mll"),
    "yll": hist.axis.Regular(50, -2.5, 2.5, name="yll"),
    "ptll": hist.axis.Regular(30, 5, 20, name="ptll", underflow=False),
    "pt": hist.axis.Regular(50, 0, 50, name="pt"),
    "eta": hist.axis.Regular(48, -2.4, 2.4, name="eta"),
    "phi": hist.axis.Regular(20, -math.pi, math.pi, circular=True, name="phi"),
}

cuts = {}


def build_graph(df, dataset):
    logger.info(f"build graph for dataset: {dataset.name}")

    results = []

    if dataset.is_data:
        df = df.DefinePerSample("weight", "1.0")
    else:
        df = df.Define("weight", "std::copysign(1.0, genWeight)")

    weightsum = df.SumAndCount("weight")
    # cuts["initial"] = weightsum

    df = df.Define(
        "Muon_isGoodGlobal", f"Muon_isGlobal && Muon_pt > 4 && abs(Muon_eta) < 2.4"
    )

    df = df.Define("Muon_isGoodPositive", f"Muon_isGoodGlobal && Muon_charge==1")
    df = df.Define("Muon_isGoodNegative", f"Muon_isGoodGlobal && Muon_charge==-1")

    # filter events
    df = df.Filter("Sum(Muon_isGoodNegative) >= 1 && Sum(Muon_isGoodPositive) >= 1 ")

    # cuts["dimuon_OS"] = df.SumAndCount("weight")

    df = df.Define("muon_pt", "Muon_pt[Muon_isGoodPositive][0]")
    df = df.Define("muon_eta", "Muon_eta[Muon_isGoodPositive][0]")
    df = df.Define("muon_phi", "Muon_phi[Muon_isGoodPositive][0]")

    df = df.Define("antimuon_pt", "Muon_pt[Muon_isGoodNegative][0]")
    df = df.Define("antimuon_eta", "Muon_eta[Muon_isGoodNegative][0]")
    df = df.Define("antimuon_phi", "Muon_phi[Muon_isGoodNegative][0]")

    df = df.Define(
        f"antimuon_mom4",
        f"ROOT::Math::PtEtaPhiMVector(muon_pt, muon_eta, muon_phi, wrem::muon_mass)",
    )
    df = df.Define(
        f"muon_mom4",
        f"ROOT::Math::PtEtaPhiMVector(antimuon_pt, antimuon_eta, antimuon_phi, wrem::muon_mass)",
    )

    df = df.Define(
        "ll_mom4",
        f"ROOT::Math::PxPyPzEVector(muon_mom4)+ROOT::Math::PxPyPzEVector(antimuon_mom4)",
    )

    df = df.Define("mll", "ll_mom4.mass()")
    df = df.Filter(f"mll >= 8.5 && mll < 11.5")

    # cuts["dimuon_mass"] = df.SumAndCount("weight")

    df = df.Define("ptll", "ll_mom4.mass()")
    df = df.Define("phill", "ll_mom4.phi()")
    df = df.Define("yll", "ll_mom4.Rapidity()")

    df = df.Define("leading_pt", "muon_pt > antimuon_pt ? muon_pt : antimuon_pt")
    df = df.Define("trailing_pt", "muon_pt > antimuon_pt ? antimuon_pt : muon_pt")

    # fill histograms
    for hlt in ["HLT_Dimuon24_Upsilon_noCorrL1", "HLT_Dimuon12_Upsilon_y1p4"]:

        df_sel = df.Filter(hlt)

        # cuts[hlt] = df_sel.SumAndCount("weight")

        results.append(df_sel.HistoBoost(f"{hlt}_mll", [all_axes["mll"]], ["mll"]))
        results.append(df_sel.HistoBoost(f"{hlt}_ptll", [all_axes["ptll"]], ["ptll"]))
        results.append(df_sel.HistoBoost(f"{hlt}_phill", [all_axes["phi"]], ["phill"]))
        results.append(df_sel.HistoBoost(f"{hlt}_yll", [all_axes["yll"]], ["yll"]))

        results.append(
            df_sel.HistoBoost(f"{hlt}_muon_pt", [all_axes["pt"]], ["muon_pt"])
        )
        results.append(
            df_sel.HistoBoost(f"{hlt}_muon_eta", [all_axes["eta"]], ["muon_eta"])
        )
        results.append(
            df_sel.HistoBoost(f"{hlt}_muon_phi", [all_axes["phi"]], ["muon_phi"])
        )

        results.append(
            df_sel.HistoBoost(f"{hlt}_antimuon_pt", [all_axes["pt"]], ["antimuon_pt"])
        )
        results.append(
            df_sel.HistoBoost(
                f"{hlt}_antimuon_eta", [all_axes["eta"]], ["antimuon_eta"]
            )
        )
        results.append(
            df_sel.HistoBoost(
                f"{hlt}_antimuon_phi", [all_axes["phi"]], ["antimuon_phi"]
            )
        )

        results.append(
            df_sel.HistoBoost(f"{hlt}_leading_pt", [all_axes["pt"]], ["leading_pt"])
        )
        results.append(
            df_sel.HistoBoost(f"{hlt}_trailing_pt", [all_axes["pt"]], ["trailing_pt"])
        )

    return results, weightsum


resultdict = narf.build_and_run(datasets, build_graph)

for name, nevents in cuts.items():
    print(f"{name}: {nevents}")

fout = f"{os.path.basename(__file__).replace('py', 'hdf5')}"
write_analysis_output(resultdict, fout, args)
