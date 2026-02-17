import os

from utilities import common, parsing
from wremnants.datasets.datagroups import Datagroups
from wums import logging

analysis_label = Datagroups.analysisLabel(os.path.basename(__file__))
parser, initargs = parsing.common_parser(analysis_label)
parser.add_argument(
    "--flavor",
    type=str,
    choices=["ee", "mumu"],
    help="Flavor (ee or mumu)",
    default="mumu",
)
parser = parsing.set_parser_default(parser, "met", "RawPFMET")
parser = parsing.set_parser_default(
    parser, "aggregateGroups", ["Diboson", "Top", "Wtaunu", "Wmunu", "Wenu"]
)
parser = parsing.set_parser_default(parser, "era", "2017H")

args = parser.parse_args()

logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)

import hist

import narf
import wremnants.lowpu as lowpu
from wremnants import (
    muon_selections,
    syst_tools,
    theory_corrections,
    theory_tools,
    unfolding_tools,
)
from wremnants.datasets.dataset_tools import getDatasets
from wremnants.histmaker_tools import (
    aggregate_groups,
    scale_to_data,
    write_analysis_output,
)

###################################
flavor = args.flavor  # mumu, ee
base_group = f"Z{flavor}"

# dilepton invariant mass cuts
mass_min = 60
mass_max = 120

# lepton cuts
lep_pt_min = 25
lep_pt_max = 9e99

datasets = getDatasets(
    maxFiles=args.maxFiles,
    filt=args.filterProcs,
    excl=list(
        set(
            args.excludeProcs + [f"SingleMuon_{args.era}"]
            if flavor == "ee"
            else [f"HighEGJet_{args.era}"]
        )
    ),
    aux=args.auxiliaryProcs,
    base_path=args.dataPath,
    era=args.era,
    nanoVersion="v12",
)

for d in datasets:
    logger.info(f"Dataset {d.name}")


# standard regular axes
axis_eta = hist.axis.Regular(48, -2.4, 2.4, name="eta")
axis_ptl = hist.axis.Regular(100, 0.0, 100.0, name="ptl")
axis_mll = hist.axis.Regular(60, 60, 120, name="mll")
axis_yll = hist.axis.Regular(50, -2.5, 2.5, name="yll")
axis_ptll = hist.axis.Regular(300, 0, 300, name="ptll")

axis_ptl = hist.axis.Regular(100, 0.0, 200.0, name="ptl")
axis_etal = hist.axis.Regular(50, -2.5, 2.5, name="etal")
axis_lin = hist.axis.Regular(5, 0, 5, name="lin")

# axes for final cards/fitting
nominal_axes = [
    hist.axis.Variable(
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 25, 30, 40, 50, 60, 75, 90, 150],
        name="ptll",
        underflow=False,
        overflow=True,
    ),
    hist.axis.Regular(20, -2.5, 2.5, name="yll", overflow=True, underflow=True),
    common.axis_charge,
]

# corresponding columns
nominal_cols = ["ptll", "yll", "TrigLep_charge"]

axis_mt = hist.axis.Regular(200, 0.0, 200, name="mt", underflow=False)
axis_met = hist.axis.Regular(200, 0, 200, name="MET")
axis_wlike_met = hist.axis.Regular(200, 0, 200, name="WlikeMET")
axes_mt = [axis_mt]
cols_mt = ["transverseMass"]

theory_helpers_procs = theory_corrections.make_theory_helpers(
    args.pdfs, args.theoryCorr
)
axis_ptVgen = theory_helpers_procs["Z"]["qcdScale"].hist.axes["ptVgen"]
axis_chargeVgen = theory_helpers_procs["Z"]["qcdScale"].hist.axes["chargeVgen"]

if args.unfolding:

    if args.unfoldingInclusive:
        cutsmap = {"fiducial": "masswindow"}
    else:
        cutsmap = {
            "pt_min": lep_pt_min,
            "pt_max": lep_pt_max,
            "mass_min": mass_min,
            "mass_max": mass_max,
        }

    unfolder_z = unfolding_tools.UnfolderZ(
        reco_axes_edges={"ptll": nominal_axes[0].edges, "yll": nominal_axes[1].edges},
        unfolding_axes_names=args.unfoldingAxes,
        unfolding_levels=args.unfoldingLevels,
        poi_as_noi=args.poiAsNoi,
        cutsmap=cutsmap,
        low_pu=True,
    )

    if not args.poiAsNoi:
        datasets = unfolding_tools.add_out_of_acceptance(datasets, group=base_group)


theory_corrs = [*args.theoryCorr, *args.ewTheoryCorr]
corr_helpers = theory_corrections.load_corr_helpers(
    [d.name for d in datasets if d.name in common.vprocs], theory_corrs
)

# recoil initialization
if not args.noRecoil:
    from wremnants import recoil_tools

    recoilHelper = recoil_tools.Recoil("lowPU", args, flavor)


def build_graph(df, dataset):
    logger.info(f"build graph for dataset: {dataset.name}")

    results = []

    theory_helpers = None
    if dataset.name in common.vprocs:
        theory_helpers = theory_helpers_procs[dataset.name[0]]

    if dataset.is_data:
        df = df.DefinePerSample("weight", "1.0")
    else:
        df = df.Define("weight", "std::copysign(1.0, genWeight)")
    df = df.Define("isEvenEvent", "event % 2 == 0")

    weightsum = df.SumAndCount("weight")

    axes = nominal_axes
    cols = nominal_cols

    if args.unfolding and dataset.group == base_group:
        df = unfolder_z.add_gen_histograms(
            args, df, results, dataset, corr_helpers, theory_helpers
        )

        if not unfolder_z.poi_as_noi:
            axes = [
                *nominal_axes,
                *unfolder_z.unfolding_axes[unfolder_z.unfolding_levels[-1]],
            ]
            cols = [
                *nominal_cols,
                *unfolder_z.unfolding_cols[unfolder_z.unfolding_levels[-1]],
            ]

    df = df.Define("TrigLep_charge", "isEvenEvent ? -1 : 1")  # wlike charge

    if flavor == "mumu":
        if not dataset.is_data:
            df = df.Define(
                "Muon_pt_corr",
                "wrem::applyRochesterMC(Muon_pt, Muon_eta, Muon_phi, Muon_charge, Muon_genPartIdx, GenPart_pt, Muon_nTrackerLayers)",
            )
            df = df.Filter("HLT_Mu17")
        else:
            df = df.Define(
                "Muon_pt_corr",
                "wrem::applyRochesterData(Muon_pt, Muon_eta, Muon_phi, Muon_charge)",
            )
            df = df.Filter("HLT_HIMu17")

        df = df.Define(
            "vetoMuons",
            "Muon_pt_corr > 10 && Muon_looseId && abs(Muon_eta) < 2.4 && abs(Muon_dxybs) < 0.05",
        )
        df = df.Filter("Sum(vetoMuons) == 2")

        df = df.Define(
            "vetoElectrons",
            "Electron_pt > 10 && Electron_cutBased > 0 && abs(Electron_eta) < 2.4",
        )
        df = df.Filter("Sum(vetoElectrons) == 0")

        df = df.Define(
            "goodLeptons",
            f"vetoMuons && Muon_pt_corr > {lep_pt_min} && Muon_pt_corr < {lep_pt_max} && Muon_mediumId && Muon_pfRelIso04_all < 0.15",
        )
        df = df.Define("goodLeptonsPlus", "goodLeptons && Muon_charge > 0")
        df = df.Define("goodLeptonsMinus", "goodLeptons && Muon_charge < 0")
        df = df.Filter("Sum(goodLeptons) == 2")
        df = df.Filter(
            "(Muon_charge[goodLeptons][0] + Muon_charge[goodLeptons][1]) == 0"
        )

        df = df.Define(
            "goodTrigObjs",
            "wrem::goodMuonTriggerCandidateLowPU(TrigObj_id, TrigObj_pt, TrigObj_l1pt, TrigObj_l2pt, TrigObj_filterBits)",
        )

        df = df.Define("Lep_pt_uncorr", "Muon_pt[goodLeptons]")
        df = df.Define("Lep_pt", "Muon_pt_corr[goodLeptons]")
        df = df.Define("Lep_eta", "Muon_eta[goodLeptons]")
        df = df.Define("Lep_phi", "Muon_phi[goodLeptons]")
        df = df.Define("Lep_charge", "Muon_charge[goodLeptons]")
        df = df.Define("Lep_mass", "Muon_mass[goodLeptons]")

    else:
        # undo the scale/smearing corrections, needed to correct RawMET
        df = df.Define(
            "Electron_pt_uncorr",
            "wrem::Egamma_undoCorrection(Electron_pt, Electron_eta, Electron_ecalCorr)",
        )

        if not dataset.is_data:
            df = df.Define(
                "Electron_pt_corr",
                "wrem::applyEGammaScaleSmearingUnc(0, Electron_pt, Electron_eta, Electron_dEscaleUp, Electron_dEscaleDown, Electron_dEsigmaUp, Electron_dEsigmaDown, 0)",
            )
            df = df.Filter("HLT_Ele20_WPLoose_Gsf")
        else:
            df = df.Define(
                "Electron_pt_corr",
                "wrem::applyEGammaScaleSmearingUnc(1, Electron_pt, Electron_eta, Electron_dEscaleUp, Electron_dEscaleDown, Electron_dEsigmaUp, Electron_dEsigmaDown, 0)",
            )
            df = df.Filter("HLT_HIEle20_WPLoose_Gsf")

        df = df.Define(
            "vetoElectrons",
            "Electron_pt_corr > 10 && Electron_cutBased > 0 && abs(Electron_eta) < 2.4",
        )
        df = df.Filter("Sum(vetoElectrons)==2")

        df = df.Define(
            "vetoMuons",
            "Muon_pt > 10 && Muon_looseId && abs(Muon_eta) < 2.4 && abs(Muon_dxybs) < 0.05 && abs(Muon_dz)< 0.2",
        )
        df = df.Filter("Sum(vetoMuons) == 0")

        df = df.Define(
            "goodLeptons",
            f"vetoElectrons && Electron_cutBased >= 3 && !(abs(Electron_eta) > 1.4442 && abs(Electron_eta) < 1.566) && Electron_pt_corr > {lep_pt_min} && Electron_pt_corr < {lep_pt_max}",
        )
        # df = df.Define("Electron_MediumID", "wrem::electron_id::pass_cutbased<3>(Electron_vidNestedWPBitmap)")
        # df = df.Define("goodLeptons", f"vetoElectrons && Electron_MediumID > 0 && Electron_pt_corr > {lep_pt_min} && Electron_pt_corr < {lep_pt_max}")
        df = df.Filter("Sum(goodLeptons)==2")

        df = df.Define("goodLeptonsPlus", "goodLeptons && Electron_charge > 0")
        df = df.Define("goodLeptonsMinus", "goodLeptons && Electron_charge < 0")

        df = df.Filter(
            "(Electron_charge[goodLeptons][0] + Electron_charge[goodLeptons][1]) == 0"
        )
        df = df.Define(
            "goodTrigObjs",
            "wrem::goodElectronTriggerCandidateLowPU(TrigObj_id, TrigObj_pt, TrigObj_l1pt, TrigObj_l2pt, TrigObj_filterBits)",
        )

        df = df.Define("Lep_pt_uncorr", "Electron_pt_uncorr[goodLeptons]")
        df = df.Define("Lep_pt", "Electron_pt_corr[goodLeptons]")
        df = df.Define("Lep_eta", "Electron_eta[goodLeptons]")
        df = df.Define("Lep_phi", "Electron_phi[goodLeptons]")
        df = df.Define("Lep_charge", "Electron_charge[goodLeptons]")
        df = df.Define("Lep_mass", "Electron_mass[goodLeptons]")

    df = df.Define(
        "trigMatch",
        "wrem::hasTriggerMatchLowPU(Lep_eta, Lep_phi, TrigObj_eta[goodTrigObjs], TrigObj_phi[goodTrigObjs])",
    )
    df = df.Define("nonTrigMatch", "wrem::inverse(trigMatch)")
    df = df.Filter("Sum(trigMatch) > 0")

    df = muon_selections.apply_met_filters(df)

    df = df.Define(
        "Lep1_mom4",
        "ROOT::Math::PtEtaPhiMVector(Lep_pt[0], Lep_eta[0], Lep_phi[0], Lep_mass[0])",
    )
    df = df.Define(
        "Lep2_mom4",
        "ROOT::Math::PtEtaPhiMVector(Lep_pt[1], Lep_eta[1], Lep_phi[1], Lep_mass[0])",
    )
    df = df.Define(
        "ll_mom4",
        "ROOT::Math::PxPyPzEVector(Lep1_mom4) + ROOT::Math::PxPyPzEVector(Lep2_mom4)",
    )
    df = df.Define("mll", "ll_mom4.mass()")
    df = df.Filter(f"mll > {mass_min} && mll < {mass_max}")

    df = df.Define(
        "ptll", "ll_mom4.pt()"
    )  # corrected ptll to be checked, the uncorrected ptll seems better/smoother
    df = df.Define("yll", "ll_mom4.Rapidity()")
    df = df.Define("absYll", "std::fabs(yll)")

    if not dataset.is_data:
        if flavor == "mumu":
            df = df.Define(
                "lepSF_ISO",
                "wrem::lepSF(Muon_pt_corr[goodLeptons], Muon_eta[goodLeptons], Muon_charge[goodLeptons], 1)",
            )
            df = df.Define(
                "lepSF_IDIP",
                "wrem::lepSF(Muon_pt_corr[goodLeptons], Muon_eta[goodLeptons], Muon_charge[goodLeptons], 2)",
            )  # largest effect
            df = df.Define(
                "lepSF_HLT",
                "wrem::lepSF_HLT_q(Muon_pt_corr[goodLeptons], Muon_eta[goodLeptons], Muon_charge[goodLeptons], 13)",
            )
            df = df.Define(
                "prefireCorr",
                "wrem::prefireCorr(0, Jet_pt, Jet_eta, Jet_phi, Jet_muEF, Jet_neEmEF, Jet_chEmEF, Photon_pt, Photon_eta, Photon_phi, Muon_pt_corr[goodLeptons], Muon_eta[goodLeptons], Muon_phi[goodLeptons])",
            )
            df = df.Define("SFMC", "lepSF_IDIP*lepSF_ISO*lepSF_HLT*prefireCorr")
        else:
            df = df.Define(
                "lepSF_IDISO",
                "wrem::lepSF(Electron_pt_corr[goodLeptons], Electron_eta[goodLeptons], Electron_charge[goodLeptons], 3)",
            )
            df = df.Define(
                "lepSF_HLT",
                "wrem::lepSF_HLT_q(Electron_pt_corr[goodLeptons], Electron_eta[goodLeptons], Electron_charge[goodLeptons], 11)",
            )
            df = df.Define(
                "prefireCorr",
                "wrem::prefireCorr(0, Jet_pt, Jet_eta, Jet_phi, Jet_muEF, Jet_neEmEF, Jet_chEmEF, Photon_pt, Photon_eta, Photon_phi, Electron_pt_corr[goodLeptons], Electron_eta[goodLeptons], Electron_phi[goodLeptons])",
            )
            df = df.Define("SFMC", "lepSF_IDISO*lepSF_HLT*prefireCorr")

        df = df.Define("exp_weight", "SFMC")
        df = theory_tools.define_theory_weights_and_corrs(
            df, dataset.name, corr_helpers, args, theory_helpers=theory_helpers
        )
    else:
        df = df.DefinePerSample("nominal_weight", "1.0")

    # lepton kinematics: leading/subleading, triggered/nontriggered
    df = df.Define("Lep_pt_leading", "Lep_pt[0] > Lep_pt[1] ? Lep_pt[0] : Lep_pt[1]")
    df = df.Define("Lep_pt_subleading", "Lep_pt[0] > Lep_pt[1] ? Lep_pt[1] : Lep_pt[0]")
    df = df.Define("Lep_pt_plus", "Lep_charge[0] > 0 ? Lep_pt[0] : Lep_pt[1]")
    df = df.Define("Lep_pt_minus", "Lep_charge[0] > 0? Lep_pt[1] : Lep_pt[0]")
    df = df.Define("Lep_pt_trg", "Lep_pt[trigMatch]")
    df = df.Define("Lep_pt_nontrg", "Lep_pt[nonTrigMatch]")

    df = df.Define("Lep_eta_leading", "Lep_pt[0] > Lep_pt[1] ? Lep_eta[0] : Lep_eta[1]")
    df = df.Define(
        "Lep_eta_subleading", "Lep_pt[0] > Lep_pt[1] ? Lep_eta[1] : Lep_eta[0]"
    )
    df = df.Define("Lep_eta_plus", "Lep_charge[0] > 0 ? Lep_eta[0] : Lep_eta[1]")
    df = df.Define("Lep_eta_minus", "Lep_charge[0] > 0? Lep_eta[1] : Lep_eta[0]")
    df = df.Define("Lep_eta_trg", "Lep_eta[trigMatch]")
    df = df.Define("Lep_eta_nontrg", "Lep_eta[nonTrigMatch]")

    results.append(df.HistoBoost("lep_pt", [axis_ptl], ["Lep_pt", "nominal_weight"]))
    results.append(
        df.HistoBoost(
            "lep_pt_leading", [axis_ptl], ["Lep_pt_leading", "nominal_weight"]
        )
    )
    results.append(
        df.HistoBoost(
            "lep_pt_subleading", [axis_ptl], ["Lep_pt_subleading", "nominal_weight"]
        )
    )
    results.append(
        df.HistoBoost("lep_pt_plus", [axis_ptl], ["Lep_pt_plus", "nominal_weight"])
    )
    results.append(
        df.HistoBoost("lep_pt_minus", [axis_ptl], ["Lep_pt_minus", "nominal_weight"])
    )
    results.append(
        df.HistoBoost("lep_pt_trg", [axis_ptl], ["Lep_pt_trg", "nominal_weight"])
    )
    results.append(
        df.HistoBoost("lep_pt_nontrg", [axis_ptl], ["Lep_pt_nontrg", "nominal_weight"])
    )

    results.append(df.HistoBoost("lep_eta", [axis_etal], ["Lep_eta", "nominal_weight"]))
    results.append(
        df.HistoBoost(
            "lep_eta_leading", [axis_etal], ["Lep_eta_leading", "nominal_weight"]
        )
    )
    results.append(
        df.HistoBoost(
            "lep_eta_subleading", [axis_etal], ["Lep_eta_subleading", "nominal_weight"]
        )
    )
    results.append(
        df.HistoBoost("lep_eta_plus", [axis_etal], ["Lep_eta_plus", "nominal_weight"])
    )
    results.append(
        df.HistoBoost("lep_eta_minus", [axis_etal], ["Lep_eta_minus", "nominal_weight"])
    )
    results.append(
        df.HistoBoost("lep_eta_trg", [axis_etal], ["Lep_eta_trg", "nominal_weight"])
    )
    results.append(
        df.HistoBoost(
            "lep_eta_nontrg", [axis_etal], ["Lep_eta_nontrg", "nominal_weight"]
        )
    )

    df = df.Define("noTrigMatch", "Sum(trigMatch)")
    results.append(
        df.HistoBoost("noTrigMatch", [axis_lin], ["noTrigMatch", "nominal_weight"])
    )

    # W-like
    df = df.Define("NonTrigLep_charge", "-TrigLep_charge")
    df = df.Define("trigLeps", "Lep_charge == TrigLep_charge")
    df = df.Define("nonTrigLeps", "Lep_charge == NonTrigLep_charge")
    df = df.Define("TrigLep_pt", "Lep_pt[trigLeps][0]")
    df = df.Define("TrigLep_eta", "Lep_eta[trigLeps][0]")
    df = df.Define("TrigLep_phi", "Lep_phi[trigLeps][0]")

    df = df.Define("NonTrigLep_pt", "Lep_pt[nonTrigLeps][0]")
    df = df.Define("NonTrigLep_eta", "Lep_eta[nonTrigLeps][0]")
    df = df.Define("NonTrigLep_phi", "Lep_phi[nonTrigLeps][0]")

    # uncorrected leptons
    df = df.Define("TrigLep_pt_uncorr", "Lep_pt_uncorr[trigLeps][0]")
    df = df.Define("NonTrigLep_pt_uncorr", "Lep_pt_uncorr[nonTrigLeps][0]")

    # Recoil calibrations
    if not args.noRecoil:
        leps_uncorr = [
            "TrigLep_pt_uncorr",
            "TrigLep_eta",
            "TrigLep_phi",
            "TrigLep_charge",
            "NonTrigLep_pt_uncorr",
            "NonTrigLep_eta",
            "NonTrigLep_phi",
            "NonTrigLep_charge",
        ]
        leps_corr = [
            "TrigLep_pt",
            "TrigLep_eta",
            "TrigLep_phi",
            "TrigLep_charge",
            "NonTrigLep_pt",
            "NonTrigLep_eta",
            "NonTrigLep_phi",
            "NonTrigLep_charge",
        ]
        df = recoilHelper.recoil_Z(
            df, results, dataset, common.zprocs_recoil_lowpu, leps_uncorr, leps_corr
        )  # produces corrected MET as MET_corr_rec_pt/phi
    else:
        df = df.Alias("MET_corr_rec_pt", "MET_pt")
        df = df.Alias("MET_corr_rec_phi", "MET_phi")

    df = df.Define(
        "transverseMass",
        f"wrem::get_mt_wlike(TrigLep_pt, TrigLep_phi, NonTrigLep_pt, NonTrigLep_phi, MET_corr_rec_pt, MET_corr_rec_phi)",
    )
    df = df.Define(
        "met_wlike_TV2",
        "wrem::get_met_wlike(NonTrigLep_pt, NonTrigLep_phi, MET_corr_rec_pt, MET_corr_rec_phi)",
    )
    df = df.Define("met_wlike_TV2_pt", "met_wlike_TV2.Mod()")

    results.append(df.HistoBoost("mll", [axis_mll], ["mll", "nominal_weight"]))
    results.append(df.HistoBoost("yll", [axis_yll], ["yll", "nominal_weight"]))
    results.append(df.HistoBoost("ptll", [axis_ptll], ["ptll", "nominal_weight"]))

    results.append(df.HistoBoost("nominal", axes, [*cols, "nominal_weight"]))
    results.append(
        df.HistoBoost("transverseMass", axes_mt, [*cols_mt, "nominal_weight"])
    )
    results.append(
        df.HistoBoost(
            "WlikeMET", [axis_wlike_met], ["met_wlike_TV2_pt", "nominal_weight"]
        )
    )
    results.append(
        df.HistoBoost("MET", [axis_met], ["MET_corr_rec_pt", "nominal_weight"])
    )

    results.append(df.HistoBoost("nominal", axes, [*cols, "nominal_weight"]))
    results.append(
        df.HistoBoost("transverseMass", axes_mt, [*cols_mt, "nominal_weight"])
    )

    if not dataset.is_data:
        # prefire
        df = df.Define(
            "prefireCorr_syst",
            "wrem::prefireCorr_syst(Jet_pt, Jet_eta, Jet_phi, Jet_muEF, Jet_neEmEF, Jet_chEmEF, Photon_pt, Photon_eta, Photon_phi, Lep_pt, Lep_eta, Lep_phi)",
        )
        df = df.Define(
            "prefireCorr_syst_tensor",
            "Eigen::TensorFixedSize<double, Eigen::Sizes<2>> res; auto w = nominal_weight*prefireCorr_syst; std::copy(std::begin(w), std::end(w), res.data()); return res;",
        )

        for n, c, a in (("nominal", cols, axes), ("transverseMass", cols_mt, axes_mt)):

            results.append(
                df.HistoBoost(
                    f"{n}_prefireCorr",
                    [*a],
                    [*c, "prefireCorr_syst_tensor"],
                    tensor_axes=[common.down_up_axis],
                )
            )

            if dataset.name in common.vprocs:
                df = syst_tools.add_theory_hists(
                    results,
                    df,
                    args,
                    dataset.name,
                    corr_helpers,
                    theory_helpers,
                    a,
                    c,
                    base_name=n,
                    for_wmass=False,
                )

            # lepton efficiencies
            if flavor == "mumu":
                df = lowpu.lepSF_systs(
                    df,
                    results,
                    "muSF_HLT_DATA_stat",
                    120,
                    "wrem::lepSF_HLT_var_mu(1, Lep_pt, Lep_eta, Lep_charge)",
                    n,
                    a,
                    c,
                )
                df = lowpu.lepSF_systs(
                    df,
                    results,
                    "muSF_HLT_DATA_syst",
                    120,
                    "wrem::lepSF_HLT_var_mu(2, Lep_pt, Lep_eta, Lep_charge)",
                    n,
                    a,
                    c,
                )
                df = lowpu.lepSF_systs(
                    df,
                    results,
                    "muSF_HLT_MC_stat",
                    120,
                    "wrem::lepSF_HLT_var_mu(-1, Lep_pt, Lep_eta, Lep_charge)",
                    n,
                    a,
                    c,
                )
                df = lowpu.lepSF_systs(
                    df,
                    results,
                    "muSF_HLT_MC_syst",
                    120,
                    "wrem::lepSF_HLT_var_mu(-2, Lep_pt, Lep_eta, Lep_charge)",
                    n,
                    a,
                    c,
                )
                df = lowpu.lepSF_systs(
                    df,
                    results,
                    "muSF_ISO_stat",
                    36,
                    "wrem::lepSF_ISO_var_mu(1, Lep_pt, Lep_eta, Lep_charge)",
                    n,
                    a,
                    c,
                )
                df = lowpu.lepSF_systs(
                    df,
                    results,
                    "muSF_ISO_DATA_syst",
                    36,
                    "wrem::lepSF_ISO_var_mu(2, Lep_pt, Lep_eta, Lep_charge)",
                    n,
                    a,
                    c,
                )
                df = lowpu.lepSF_systs(
                    df,
                    results,
                    "muSF_ISO_MC_syst",
                    36,
                    "wrem::lepSF_ISO_var_mu(-2, Lep_pt, Lep_eta, Lep_charge)",
                    n,
                    a,
                    c,
                )
                df = lowpu.lepSF_systs(
                    df,
                    results,
                    "muSF_IDIP_stat",
                    36,
                    "wrem::lepSF_IDIP_var_mu(1, Lep_pt, Lep_eta, Lep_charge)",
                    n,
                    a,
                    c,
                )
                df = lowpu.lepSF_systs(
                    df,
                    results,
                    "muSF_IDIP_DATA_syst",
                    36,
                    "wrem::lepSF_IDIP_var_mu(2, Lep_pt, Lep_eta, Lep_charge)",
                    n,
                    a,
                    c,
                )
                df = lowpu.lepSF_systs(
                    df,
                    results,
                    "muSF_IDIP_MC_syst",
                    36,
                    "wrem::lepSF_IDIP_var_mu(-2, Lep_pt, Lep_eta, Lep_charge)",
                    n,
                    a,
                    c,
                )
            # electron efficiency uncertainties currently don't work
            # else:
            #     df = lowPUcfg.lepSF_systs(df, results, "elSF_HLT_syst",      120, "wrem::lepSF_el_HLT_syst(Lep_pt, Lep_eta, Lep_charge)", n, a, c)
            #     df = lowPUcfg.lepSF_systs(df, results, "elSF_IDISO_syst",    36,  "wrem::lepSF_el_IDISO_syst(Lep_pt, Lep_eta, Lep_charge)", n, a, c)

            if not args.noRecoil and args.recoilUnc:
                df = recoilHelper.add_recoil_unc_Z(df, results, dataset, c, a, n)

    if args.unfolding and args.poiAsNoi and dataset.group == base_group:
        unfolder_z.add_poi_as_noi_histograms(
            df,
            results,
            axes,
            cols,
        )

    if hasattr(dataset, "out_of_acceptance"):
        # Rename dataset to not overwrite the original one
        dataset.name = dataset.name + "OOA"

    return results, weightsum


resultdict = narf.build_and_run(datasets, build_graph)

if not args.noScaleToData:
    scale_to_data(resultdict)
    aggregate_groups(datasets, resultdict, args.aggregateGroups)

write_analysis_output(resultdict, f"mz_lowPU_{flavor}.hdf5", args)
