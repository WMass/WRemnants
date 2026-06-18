import math
import os

from wremnants.utilities import common, parsing, samples
from wums import logging

analysis_label = common.analysis_label(os.path.basename(__file__))
parser, initargs = parsing.common_parser(analysis_label)
parser.add_argument(
    "--muonCorr",
    default="none",
    choices=["none", "rochester", "scarekit"],
    help="Muon momentum correction to apply",
)
parser.add_argument(
    "--corrStep",
    default="1234",
    choices=["0", "1", "123", "1234"],
    help="Scarekit calibration stage (only with --muonCorr scarekit): "
    "0 = no correction, 1 = scale only, 123 = scale+smearing, 1234 = full",
)

args = parser.parse_args()

logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)

import hist

import narf
from wremnants.production import (
    generator_level_definitions,
    systematics,
    theory_corrections,
)
from wremnants.production.datasets.dataset_tools import getDatasets
from wremnants.production.histmaker_tools import write_analysis_output

if args.muonCorr == "rochester":
    narf.clingutils.Load("libPhysics")
    narf.clingutils.Load("libROOTVecOps")
    narf.clingutils.Load("libROOTDataFrame")
    narf.clingutils.Declare('#include "lowpu_rochester.hpp"')
elif args.muonCorr == "scarekit":
    narf.clingutils.Load("libROOTDataFrame")
    narf.clingutils.Declare('#include "lowpu_muonscarekit.hpp"')

datasets = getDatasets(
    maxFiles=args.maxFiles,
    filt=args.filterProcs,
    excl=args.excludeProcs,
    base_path=args.dataPath,
    era=args.era,
)

import pickle

import lz4.frame

theory_corrs = args.theoryCorr
theory_corr_base = f"{common.data_dir}/TheoryCorrections/5020GeV"


def load_corr_hist_5020(filename, proc, histname):
    """5020 GeV pickles use ZMUMU5020GEV keys and legacy hist names."""
    with lz4.frame.open(filename) as f:
        corr = pickle.load(f)
    key = histname.replace("scetlib_dyturbo_LatticeNP", "scetlib_dyturboLatticeNP")
    key = key.replace("_minnlo_ratio", "__minnlo_ratio")
    return corr["ZMUMU5020GEV"][key]


theory_corrections.load_corr_hist = load_corr_hist_5020
corr_helpers = theory_corrections.load_corr_helpers(
    [d.name for d in datasets if d.name in samples.zprocs],
    theory_corrs,
    base_dir=theory_corr_base,
)

# define histogram axes, see: https://hist.readthedocs.io/en/latest/index.html
axis_nLepton = hist.axis.Integer(0, 5, name="nLepton", underflow=False)
axis_mll = hist.axis.Regular(60, 76, 106, name="mll")
dilepton_ptV_binning = [
    0,
    1,
    1.5,
    2,
    2.5,
    3,
    3.5,
    4,
    4.5,
    5,
    5.5,
    6,
    6.5,
    7,
    7.5,
    8,
    8.5,
    9,
    9.5,
    10,
    10.5,
    11,
    11.5,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    22,
    24,
    26,
    28,
    30,
    33,
    37,
    44,
    100,
]
axis_ptll = hist.axis.Variable(
    dilepton_ptV_binning, name="ptll", underflow=False, overflow=True
)
yll_10quantiles_binning = [-2.5, -1.5, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 1.5, 2.5]
axis_yll = hist.axis.Variable(
    yll_10quantiles_binning, name="yll", underflow=True, overflow=True
)
absYll_binning = [0.0, 0.25, 0.5, 1.0, 1.5, 2.5]
axis_absYll = hist.axis.Variable(
    absYll_binning, name="absYll", underflow=False, overflow=True
)
axis_mu_pt = hist.axis.Regular(60, 25, 150, name="mu_pt")
axis_mu_eta = hist.axis.Regular(48, -2.4, 2.4, name="mu_eta")
axis_mu_phi = hist.axis.Regular(32, -math.pi, math.pi, circular=True, name="mu_phi")
axis_mu_oneOverPt = hist.axis.Regular(50, 0.005, 0.04, name="mu_oneOverPt")
axis_mu_charge = hist.axis.Variable([-1.5, -0.5, 0.5, 1.5], name="mu_charge")
axis_mu_nl = hist.axis.Variable(
    [6.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 17.5], name="mu_nl"
)
axis_mu_masspt = hist.axis.Regular(100, 0, 1e4, name="mu_masspt")
axis_cosThetaStarll = hist.axis.Regular(
    200, -1.0, 1.0, name="cosThetaStarll", underflow=False, overflow=False
)
axis_phiStarll = hist.axis.Regular(
    20, -math.pi, math.pi, circular=True, name="phiStarll"
)
axis_phill = hist.axis.Regular(50, -math.pi, math.pi, circular=True, name="phill")

axis_prefire_tensor = hist.axis.Integer(
    0, 2, name="prefire_variation", underflow=False, overflow=False
)


def build_graph(df, dataset):
    logger.info(f"build graph for dataset: {dataset.name}")

    results = []

    if dataset.is_data:
        df = df.DefinePerSample("weight", "1.0")
    else:
        df = df.Define("weight", "std::copysign(1.0, genWeight)")

    weightsum = df.SumAndCount("weight")

    df = df.Define("isEvenEvent", f"event % 2 == 0")

    # apply muon momentum corrections before selection
    if args.muonCorr == "rochester":
        if dataset.is_data:
            df = df.Define(
                "Muon_pt_corr",
                "wrem::applyRochesterData(Muon_pt, Muon_eta, Muon_phi, ROOT::VecOps::RVec<float>(Muon_charge.begin(), Muon_charge.end()))",
            )
        else:
            df = df.Define(
                "Muon_pt_corr",
                "wrem::applyRochesterMC(Muon_pt, Muon_eta, Muon_phi, ROOT::VecOps::RVec<float>(Muon_charge.begin(), Muon_charge.end()), Muon_genPartIdx, GenPart_pt, Muon_nTrackerLayers)",
            )
    elif args.muonCorr == "scarekit":
        if args.corrStep == "0":
            df = df.Alias("Muon_pt_corr", "Muon_pt")
        elif args.corrStep == "1":
            if dataset.is_data:
                df = df.Define(
                    "Muon_pt_corr",
                    "wrem::applyMuonScarekitData(Muon_pt, Muon_eta, Muon_phi, Muon_charge)",
                )
            else:
                df = df.Define(
                    "Muon_pt_corr",
                    "wrem::applyMuonScarekitMC_scaleOnly(Muon_pt, Muon_eta, Muon_phi, Muon_charge)",
                )
        elif args.corrStep == "123":
            if dataset.is_data:
                df = df.Define(
                    "Muon_pt_corr",
                    "wrem::applyMuonScarekitData(Muon_pt, Muon_eta, Muon_phi, Muon_charge)",
                )
            else:
                df = df.Define(
                    "Muon_pt_corr",
                    "wrem::applyMuonScarekitMC_noKFactor(Muon_pt, Muon_eta, Muon_phi, Muon_charge, Muon_nTrackerLayers)",
                )
        else:  # "1234"
            if dataset.is_data:
                df = df.Define(
                    "Muon_pt_corr",
                    "wrem::applyMuonScarekitData(Muon_pt, Muon_eta, Muon_phi, Muon_charge)",
                )
            else:
                df = df.Define(
                    "Muon_pt_corr",
                    "wrem::applyMuonScarekitMC(Muon_pt, Muon_eta, Muon_phi, Muon_charge, Muon_nTrackerLayers, run, luminosityBlock)",
                )
    else:  # "none"
        df = df.Alias("Muon_pt_corr", "Muon_pt")

    # filter events
    df = df.Filter("HLT_HIMu17")

    # available columns, see: https://cms-xpog.docs.cern.ch/autoDoc/

    # define new columns
    df = df.Define("nLepton", "nElectron + nMuon")

    # ---- Good muons (for Z->mumu selection) ----
    df = df.Define(
        "goodMu",
        "Muon_pt_corr > 18 && abs(Muon_eta) < 2.4 && Muon_mediumId && Muon_isGlobal",
    )
    df = df.Define("goodMu_idx", "ROOT::VecOps::Nonzero(goodMu)")
    df = df.Filter("goodMu_idx.size() == 2", "Exactly two good muons")

    # ---- Filter out events with extra electrons ----
    df = df.Filter("nElectron == 0", "No electrons in the event")

    # Opposite sign
    df = df.Define("i0", "int(goodMu_idx[0])").Define("i1", "int(goodMu_idx[1])")
    df = df.Filter("Muon_charge[i0] * Muon_charge[i1] < 0", "Opposite-sign muons")

    # ---- Build dimuon kinematics ----
    MU_MASS = 0.105658
    df = (
        df.Define(
            "mu0_p4",
            f"ROOT::Math::PtEtaPhiMVector(Muon_pt_corr[i0], Muon_eta[i0], Muon_phi[i0], {MU_MASS})",
        )
        .Define(
            "mu1_p4",
            f"ROOT::Math::PtEtaPhiMVector(Muon_pt_corr[i1], Muon_eta[i1], Muon_phi[i1], {MU_MASS})",
        )
        .Define("dimu_p4", "mu0_p4 + mu1_p4")
        .Define("mll", "dimu_p4.M()")
        .Define("ptll", "dimu_p4.Pt()")
        .Define("yll", "dimu_p4.Rapidity()")
        .Define("absYll", "std::fabs(yll)")
        .Define("phill", "dimu_p4.Phi()")
    )
    df = df.Filter("mll > 76 && mll < 106", "Z mass window")

    # ---- Rank muons: leading/trailing by pT; positive/negative by charge ----
    df = (
        df.Define("i_lead", "Muon_pt_corr[i0] >= Muon_pt_corr[i1] ? i0 : i1")
        .Define("i_trail", "Muon_pt_corr[i0] >= Muon_pt_corr[i1] ? i1 : i0")
        .Define("i_pos", "Muon_charge[i0] > 0 ? i0 : i1")
        .Define("i_neg", "Muon_charge[i0] > 0 ? i1 : i0")
        .Define("muleadpt", "Muon_pt_corr[i_lead]")
        .Define("mutrailpt", "Muon_pt_corr[i_trail]")
        .Define("muleadeta", "Muon_eta[i_lead]")
        .Define("mutraileta", "Muon_eta[i_trail]")
        .Define("mupospt", "Muon_pt_corr[i_pos]")
        .Define("munegpt", "Muon_pt_corr[i_neg]")
        .Define("muposeta", "Muon_eta[i_pos]")
        .Define("munegeta", "Muon_eta[i_neg]")
        .Define("muposphi", "Muon_phi[i_pos]")
        .Define("munegphi", "Muon_phi[i_neg]")
        .Define("mupos_oneOverPt", "1.0/Muon_pt_corr[i_pos]")
        .Define("muneg_oneOverPt", "1.0/Muon_pt_corr[i_neg]")
        .Define("muposcharge", "(double)Muon_charge[i_pos]")
        .Define("munegcharge", "(double)Muon_charge[i_neg]")
        .Define("mupos_nl", "(double)Muon_nTrackerLayers[i_pos]")
        .Define("muneg_nl", "(double)Muon_nTrackerLayers[i_neg]")
        .Define("mupos_masspt", "mll * Muon_pt_corr[i_pos]")
        .Define("muneg_masspt", "mll * Muon_pt_corr[i_neg]")
    )

    # ---- Build CS angles ----
    df = (
        df.Define(
            "mupos_p4",
            f"ROOT::Math::PtEtaPhiMVector(Muon_pt_corr[i_pos], Muon_eta[i_pos], Muon_phi[i_pos], {MU_MASS})",
        )
        .Define(
            "muneg_p4",
            f"ROOT::Math::PtEtaPhiMVector(Muon_pt_corr[i_neg], Muon_eta[i_neg], Muon_phi[i_neg], {MU_MASS})",
        )
        .Define("csSineCosThetaPhill", "wrem::csSineCosThetaPhi(mupos_p4, muneg_p4)")
    )
    df = df.Define("cosThetaStarll", "csSineCosThetaPhill.costheta")
    df = df.Define("phiStarll", "csSineCosThetaPhill.phi()")

    # prefiring
    if dataset.is_data:
        df = df.Define("nominal_weight", "1.0")
    else:
        df = df.Define("exp_weight", "weight*L1PreFiringWeight_Nom")

        df = generator_level_definitions.define_prefsr_vars(df)
        df = df.DefinePerSample("central_pdf_weight", "1.0")
        df = df.Alias("nominal_weight_uncorr", "exp_weight")
        df = df.DefinePerSample("theory_weight_truncate", "10.0")
        for theory_corr_name in theory_corrs:
            if theory_corr_name not in corr_helpers[dataset.name]:
                continue
            df = theory_corrections.define_theory_corr_weight_column(
                df, theory_corr_name
            )
            df = df.Define(
                f"{theory_corr_name}Weight_tensor",
                corr_helpers[dataset.name][theory_corr_name],
                [
                    "massVgen",
                    "absYVgen",
                    "ptVgen",
                    "chargeVgen",
                    f"{theory_corr_name}_corr_weight",
                ],
            )

        theory_corr_name = theory_corrs[0]
        df = df.Define("nominal_weight", f"{theory_corr_name}Weight_tensor[0]")

    # ---- Fill histograms ----
    hist_nLepton = df.HistoBoost(
        "nLepton", [axis_nLepton], ["nLepton", "nominal_weight"]
    )
    hist_mll = df.HistoBoost("mll", [axis_mll], ["mll", "nominal_weight"])
    hist_ptll = df.HistoBoost("ptll", [axis_ptll], ["ptll", "nominal_weight"])
    hist_yll = df.HistoBoost("yll", [axis_yll], ["yll", "nominal_weight"])
    hist_phill = df.HistoBoost("phill", [axis_phill], ["phill", "nominal_weight"])

    # Leading/trailing
    hist_mu_lead_pt = df.HistoBoost(
        "muleadpt", [axis_mu_pt], ["muleadpt", "nominal_weight"]
    )
    hist_mu_trail_pt = df.HistoBoost(
        "mutrailpt", [axis_mu_pt], ["mutrailpt", "nominal_weight"]
    )
    hist_mu_lead_eta = df.HistoBoost(
        "muleadeta", [axis_mu_eta], ["muleadeta", "nominal_weight"]
    )
    hist_mu_trail_eta = df.HistoBoost(
        "mutraileta", [axis_mu_eta], ["mutraileta", "nominal_weight"]
    )

    # Positive/negative
    hist_mu_pos_pt = df.HistoBoost(
        "mupospt", [axis_mu_pt], ["mupospt", "nominal_weight"]
    )
    hist_mu_neg_pt = df.HistoBoost(
        "munegpt", [axis_mu_pt], ["munegpt", "nominal_weight"]
    )
    hist_mu_pos_eta = df.HistoBoost(
        "muposeta", [axis_mu_eta], ["muposeta", "nominal_weight"]
    )
    hist_mu_neg_eta = df.HistoBoost(
        "munegeta", [axis_mu_eta], ["munegeta", "nominal_weight"]
    )
    hist_mu_pos_phi = df.HistoBoost(
        "muposphi", [axis_mu_phi], ["muposphi", "nominal_weight"]
    )
    hist_mu_neg_phi = df.HistoBoost(
        "munegphi", [axis_mu_phi], ["munegphi", "nominal_weight"]
    )
    hist_mu_pos_oneOverPt = df.HistoBoost(
        "mupos_oneOverPt", [axis_mu_oneOverPt], ["mupos_oneOverPt", "nominal_weight"]
    )
    hist_mu_neg_oneOverPt = df.HistoBoost(
        "muneg_oneOverPt", [axis_mu_oneOverPt], ["muneg_oneOverPt", "nominal_weight"]
    )
    hist_mu_pos_charge = df.HistoBoost(
        "muposcharge", [axis_mu_charge], ["muposcharge", "nominal_weight"]
    )
    hist_mu_neg_charge = df.HistoBoost(
        "munegcharge", [axis_mu_charge], ["munegcharge", "nominal_weight"]
    )
    hist_mu_pos_nl = df.HistoBoost(
        "mupos_nl", [axis_mu_nl], ["mupos_nl", "nominal_weight"]
    )
    hist_mu_neg_nl = df.HistoBoost(
        "muneg_nl", [axis_mu_nl], ["muneg_nl", "nominal_weight"]
    )
    hist_mu_pos_masspt = df.HistoBoost(
        "mupos_masspt", [axis_mu_masspt], ["mupos_masspt", "nominal_weight"]
    )
    hist_mu_neg_masspt = df.HistoBoost(
        "muneg_masspt", [axis_mu_masspt], ["muneg_masspt", "nominal_weight"]
    )

    # CS angles
    hist_cosThetaStarll = df.HistoBoost(
        "cosThetaStarll", [axis_cosThetaStarll], ["cosThetaStarll", "nominal_weight"]
    )
    hist_phiStarll = df.HistoBoost(
        "phiStarll", [axis_phiStarll], ["phiStarll", "nominal_weight"]
    )

    # 2D histograms
    hist_ptll_vs_yll = df.HistoBoost(
        "ptll_vs_yll", [axis_ptll, axis_yll], ["ptll", "yll", "nominal_weight"]
    )
    # MINIMUM BIN CONTENT: 95.79483724339086 at bin (ptll index 35, yll index 6) → ptll ∈ [28, 30) GeV, yll ∈ [0.25, 0.5)
    # DATA MINIMUM BIN CONTENT: 88.0 at bin (ptll index 35, yll index 3) → ptll ∈ [28, 30) GeV, yll ∈ [-0.5, -0.25)

    if not dataset.is_data:
        df = df.Define(
            "prefire_vector",
            """
        auto res = std::vector<double>{L1PreFiringWeight_Muon_StatUp/L1PreFiringWeight_Muon_Nom, L1PreFiringWeight_Muon_StatDn/L1PreFiringWeight_Muon_Nom};
        res[0] = nominal_weight * res[0];
        res[1] = nominal_weight * res[1];
        return res;
        """,
        )

        df = df.Define(
            "prefire_vector_weight", "wrem::vec_to_tensor<2>(prefire_vector)"
        )

        hist_prefire_tensor = df.HistoBoost(
            "nominal_prefiring",
            [axis_ptll, axis_absYll, axis_cosThetaStarll],
            ["ptll", "absYll", "cosThetaStarll", "prefire_vector_weight"],
            tensor_axes=[axis_prefire_tensor],
        )
        results.append(hist_prefire_tensor)

        hist_muleadeta_prefire = df.HistoBoost(
            "muleadeta_prefiring",
            [axis_mu_eta],
            ["muleadeta", "prefire_vector_weight"],
            tensor_axes=[axis_prefire_tensor],
        )
        results.append(hist_muleadeta_prefire)

        hist_mutraileta_prefire = df.HistoBoost(
            "mutraileta_prefiring",
            [axis_mu_eta],
            ["mutraileta", "prefire_vector_weight"],
            tensor_axes=[axis_prefire_tensor],
        )
        results.append(hist_mutraileta_prefire)

        systematics.add_theory_corr_hists(
            results,
            df,
            [axis_ptll, axis_absYll, axis_cosThetaStarll],
            ["ptll", "absYll", "cosThetaStarll"],
            corr_helpers[dataset.name],
            theory_corrs,
            modify_central_weight=True,
            isW=False,
            base_name="ptll",
        )

    results += [
        hist_mll,
        hist_ptll,
        hist_yll,
        hist_phill,
        hist_nLepton,
        hist_mu_lead_pt,
        hist_mu_trail_pt,
        hist_mu_lead_eta,
        hist_mu_trail_eta,
        hist_mu_pos_pt,
        hist_mu_neg_pt,
        hist_mu_pos_eta,
        hist_mu_neg_eta,
        hist_mu_pos_phi,
        hist_mu_neg_phi,
        hist_mu_pos_oneOverPt,
        hist_mu_neg_oneOverPt,
        hist_mu_pos_charge,
        hist_mu_neg_charge,
        hist_mu_pos_nl,
        hist_mu_neg_nl,
        hist_mu_pos_masspt,
        hist_mu_neg_masspt,
        hist_cosThetaStarll,
        hist_phiStarll,
        hist_ptll_vs_yll,
    ]

    return results, weightsum


resultdict = narf.build_and_run(datasets, build_graph)

fout = f"{os.path.basename(__file__).replace('py', 'hdf5')}"
write_analysis_output(resultdict, fout, args)
