import os
import math

from utilities import common, parsing
from wremnants.datasets.datagroups import Datagroups
from wremnants.histmaker_tools import make_quantile_helper
from wums import logging

analysis_label = Datagroups.analysisLabel(os.path.basename(__file__))
parser, initargs = parsing.common_parser(analysis_label)

args = parser.parse_args()

logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)

import hist

import narf
from wremnants.datasets.dataset_tools import getDatasets
from wremnants.histmaker_tools import write_analysis_output

datasets = getDatasets(
    maxFiles=args.maxFiles,
    filt=args.filterProcs,
    excl=args.excludeProcs,
    base_path=args.dataPath,
    mode=analysis_label,
    era=args.era,
)

quantile_file = "histmaker_test_scetlib_dyturboCorr.hdf5"

quantile_helper_cosThetaStarll = None
if quantile_file and os.path.exists(quantile_file):
    try:
        process_name = None
        for d in datasets:
            if not d.is_data and d.name:
                process_name = d.name
                break
        if process_name:
            quantile_helper_cosThetaStarll = make_quantile_helper(
                quantile_file,
                ["cosThetaStarll"],  # 1D quantiles for cosThetaStarll
                ["ptll", "absYll"],  # Dependent axes
                name="nominal_csQuantiles",
                processes=[process_name],
                n_quantiles=[8],  # 8 quantiles for cosThetaStarll
            )[0]  # Returns a list, take first element for 1D
            logger.info(f"Loaded quantile helper from {quantile_file}")
        else:
            logger.warning("No MC process found, skipping quantile helper")
    except Exception as e:
        logger.warning(f"Failed to load quantile helper: {e}")
        quantile_helper_cosThetaStarll = None
else:
    logger.info("Quantile file not specified or not found, quantile binning disabled")

# define histogram axes, see: https://hist.readthedocs.io/en/latest/index.html
axis_nLepton = hist.axis.Integer(0, 5, name="nLepton", underflow=False)
axis_mll  = hist.axis.Regular(60, 76, 106, name="mll")
dilepton_ptV_binning = [0, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 24, 26, 28, 30, 33, 37, 44, 100]
axis_ptll = hist.axis.Variable(dilepton_ptV_binning, name="ptll", underflow=False, overflow=True)
yll_10quantiles_binning = [-2.5, -1.5, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 1.5, 2.5]
axis_yll = hist.axis.Variable(yll_10quantiles_binning, name="yll", underflow=True, overflow=True)
absYll_binning = [0.0, 0.25, 0.5, 1.0, 1.5, 2.5]
axis_absYll = hist.axis.Variable(absYll_binning, name="absYll", underflow=False, overflow=True)
axis_mu_pt  = hist.axis.Regular(60, 25, 150, name="mu_pt")
axis_mu_eta = hist.axis.Regular(48, -2.4, 2.4, name="mu_eta")
axis_cosThetaStarll = hist.axis.Regular(200, -1.0, 1.0, name="cosThetaStarll", underflow=False, overflow=False)
axis_phiStarll = hist.axis.Regular(20, -math.pi, math.pi, circular=True, name="phiStarll")
axis_phill = hist.axis.Regular(50, -math.pi, math.pi, circular=True, name="phill")

axis_cosThetaStarll_quantile = hist.axis.Integer(0, 8, name="cosThetaStarll_qbin", underflow=False, overflow=False)
axis_prefire_tensor = hist.axis.Integer(0, 2, name="prefire_variation", underflow=False, overflow=False)

def build_graph(df, dataset):
    logger.info(f"build graph for dataset: {dataset.name}")

    results = []

    if dataset.is_data:
        df = df.DefinePerSample("weight", "1.0")
    else:
        df = df.Define("weight", "std::copysign(1.0, genWeight)")

    weightsum = df.SumAndCount("weight")


    # filter events
    df = df.Filter("HLT_HIMu17")

    # available columns, see: https://cms-xpog.docs.cern.ch/autoDoc/

    # define new columns
    df = df.Define("nLepton", "nElectron + nMuon")

    # ---- Good muons (for Z->mumu selection) ----
    df = df.Define(
        "goodMu",
        "Muon_pt > 18 && abs(Muon_eta) < 2.4 && Muon_mediumId && Muon_isGlobal"
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
        df.Define("mu0_p4", f"ROOT::Math::PtEtaPhiMVector(Muon_pt[i0], Muon_eta[i0], Muon_phi[i0], {MU_MASS})")
          .Define("mu1_p4", f"ROOT::Math::PtEtaPhiMVector(Muon_pt[i1], Muon_eta[i1], Muon_phi[i1], {MU_MASS})")
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
        df.Define("i_lead",  "Muon_pt[i0] >= Muon_pt[i1] ? i0 : i1")
          .Define("i_trail", "Muon_pt[i0] >= Muon_pt[i1] ? i1 : i0")
          .Define("i_pos", "Muon_charge[i0] > 0 ? i0 : i1")
          .Define("i_neg", "Muon_charge[i0] > 0 ? i1 : i0")

          .Define("muleadpt",  "Muon_pt[i_lead]")
          .Define("mutrailpt", "Muon_pt[i_trail]")
          .Define("muleadeta", "Muon_eta[i_lead]")
          .Define("mutraileta","Muon_eta[i_trail]")

          .Define("mupospt",  "Muon_pt[i_pos]")
          .Define("munegpt",  "Muon_pt[i_neg]")
          .Define("muposeta", "Muon_eta[i_pos]")
          .Define("munegeta", "Muon_eta[i_neg]")
    )

    # ---- Build CS angles ----
    df = (
        df.Define("mupos_p4", f"ROOT::Math::PtEtaPhiMVector(Muon_pt[i_pos], Muon_eta[i_pos], Muon_phi[i_pos], {MU_MASS})")
          .Define("muneg_p4", f"ROOT::Math::PtEtaPhiMVector(Muon_pt[i_neg], Muon_eta[i_neg], Muon_phi[i_neg], {MU_MASS})")
          .Define("csSineCosThetaPhill", "wrem::csSineCosThetaPhi(mupos_p4, muneg_p4)")
    )
    df = df.Define("cosThetaStarll", "csSineCosThetaPhill.costheta")
    df = df.Define("phiStarll", "csSineCosThetaPhill.phi()")

    # Compute quantile for cosThetaStarll (if helper is available)
    if quantile_helper_cosThetaStarll is not None:
        df = df.Define("cosThetaStarll_q", quantile_helper_cosThetaStarll,
               ["cosThetaStarll", "ptll", "absYll"])

        df = df.Define("cosThetaStarll_qbin", "int(std::floor(8.0*cosThetaStarll_q))")
        df = df.Redefine("cosThetaStarll_qbin", "std::min(7, std::max(0, cosThetaStarll_qbin))")


    # prefiring
    if dataset.is_data:
        df = df.Define("nominal_weight", "1.0")
    else:
        df = df.Define("nominal_weight", "weight*L1PreFiringWeight_Nom")

    # ---- Fill histograms ----
    hist_nLepton = df.HistoBoost("nLepton", [axis_nLepton], ["nLepton", "nominal_weight"])
    hist_mll  = df.HistoBoost("mll",  [axis_mll],  ["mll", "nominal_weight"])
    hist_ptll = df.HistoBoost("ptll", [axis_ptll], ["ptll", "nominal_weight"])
    hist_yll  = df.HistoBoost("yll",  [axis_yll],  ["yll", "nominal_weight"])
    hist_phill = df.HistoBoost("phill", [axis_phill], ["phill", "nominal_weight"])

    # Leading/trailing
    hist_mu_lead_pt   = df.HistoBoost("muleadpt",   [axis_mu_pt],  ["muleadpt", "nominal_weight"])
    hist_mu_trail_pt  = df.HistoBoost("mutrailpt",  [axis_mu_pt],  ["mutrailpt", "nominal_weight"])
    hist_mu_lead_eta  = df.HistoBoost("muleadeta",  [axis_mu_eta], ["muleadeta", "nominal_weight"])
    hist_mu_trail_eta = df.HistoBoost("mutraileta", [axis_mu_eta], ["mutraileta", "nominal_weight"])

    # Positive/negative
    hist_mu_pos_pt  = df.HistoBoost("mupospt",  [axis_mu_pt],  ["mupospt", "nominal_weight"])
    hist_mu_neg_pt  = df.HistoBoost("munegpt",  [axis_mu_pt],  ["munegpt", "nominal_weight"])
    hist_mu_pos_eta = df.HistoBoost("muposeta", [axis_mu_eta], ["muposeta", "nominal_weight"])
    hist_mu_neg_eta = df.HistoBoost("munegeta", [axis_mu_eta], ["munegeta", "nominal_weight"])

    # CS angles
    hist_cosThetaStarll = df.HistoBoost("cosThetaStarll", [axis_cosThetaStarll], ["cosThetaStarll", "nominal_weight"])
    hist_phiStarll = df.HistoBoost("phiStarll", [axis_phiStarll], ["phiStarll", "nominal_weight"])

    hist_ptll_absYll_byQ = []
    if quantile_helper_cosThetaStarll is not None:
        for q in range(8):
            dfq = df.Filter(f"cosThetaStarll_qbin == {q}")
            hist_ptll_absYll_byQ.append(
                dfq.HistoBoost(
                    f"ptll_vs_absYll_csQ{q}",
                    [axis_ptll, axis_absYll],
                    ["ptll", "absYll", "nominal_weight"],
                )
            )
        results += hist_ptll_absYll_byQ

    # 2D histograms
    hist_ptll_vs_yll = df.HistoBoost("ptll_vs_yll", [axis_ptll, axis_yll], ["ptll", "yll", "nominal_weight"])
    # MINIMUM BIN CONTENT: 95.79483724339086 at bin (ptll index 35, yll index 6) → ptll ∈ [28, 30) GeV, yll ∈ [0.25, 0.5)
    # DATA MINIMUM BIN CONTENT: 88.0 at bin (ptll index 35, yll index 3) → ptll ∈ [28, 30) GeV, yll ∈ [-0.5, -0.25)

    # Create histogram for quantile computation (needed to create quantile file)
    hist_csQuantiles = df.HistoBoost(
        "nominal",
        [axis_ptll, axis_absYll, axis_cosThetaStarll],
        ["ptll", "absYll", "cosThetaStarll", "nominal_weight"],
    )

    if not dataset.is_data:
        df = df.Define("prefire_vector", """
        auto res = std::vector<double>{L1PreFiringWeight_Muon_StatUp/L1PreFiringWeight_Muon_Nom, L1PreFiringWeight_Muon_StatDn/L1PreFiringWeight_Muon_Nom}; 
        res[0] = nominal_weight * res[0];
        res[1] = nominal_weight * res[1];
        return res;
        """)

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
    
    results += hist_ptll_absYll_byQ
    results += [
        hist_mll, hist_ptll, hist_yll, hist_phill, hist_nLepton,
        hist_mu_lead_pt, hist_mu_trail_pt, hist_mu_lead_eta, hist_mu_trail_eta,
        hist_mu_pos_pt, hist_mu_neg_pt, hist_mu_pos_eta, hist_mu_neg_eta,
        hist_cosThetaStarll, hist_phiStarll,
        hist_ptll_vs_yll, hist_csQuantiles,
    ]

    return results , weightsum


resultdict = narf.build_and_run(datasets, build_graph)

fout = f"{os.path.basename(__file__).replace('py', 'hdf5')}"
write_analysis_output(resultdict, fout, args)
