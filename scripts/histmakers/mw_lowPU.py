import os

from utilities import common, differential, parsing
from wremnants.datasets.datagroups import Datagroups

analysis_label = Datagroups.analysisLabel(os.path.basename(__file__))
parser, initargs = parsing.common_parser(analysis_label)
parser.add_argument(
    "--noGenMatchMC",
    action="store_true",
    help="Don't use gen match filter for prompt muons with MC samples (note: QCD MC never has it anyway)",
)
parser.add_argument(
    "--flavor", type=str, choices=["e", "mu"], help="Flavor (e or mu)", default="mu"
)

parser = parsing.set_parser_default(parser, "pt", [34, 25, 1000])
parser = parsing.set_parser_default(parser, "met", "RawPFMET")
parser = parsing.set_parser_default(parser, "era", "2017H")

args = parser.parse_args()
isUnfolding = args.analysisMode == "unfolding"


import math

import hist

import narf
import wremnants
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
from wums import logging

###################################
flavor = args.flavor  # mu, e
if flavor == "mu":
    sigProcs = ["Wminusmunu", "Wplusmunu"]
    base_group = "Wmunu"
else:
    sigProcs = ["Wminusenu", "Wplusenu"]
    base_group = "Wenu"

datasets = getDatasets(
    maxFiles=args.maxFiles,
    filt=args.filterProcs,
    excl=list(
        set(args.excludeProcs + ["singlemuon"] if flavor == "e" else ["singleelectron"])
    ),
    base_path=args.dataPath,
    extended="msht20an3lo" not in args.pdfs,
    mode=analysis_label,
    era=args.era,
    nanoVersion="v12",
)

logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)

for d in datasets:
    logger.info(f"Dataset {d.name}")

mtw_min = 40  # for Wmass (roughly half the boson mass)

# lepton cuts
lep_pt_min = 25

# axes used in fakerate calculation
axis_fakes_pt = hist.axis.Variable(
    [25, 30, 35, 40, 45, 50, 55], name="pt", underflow=False, overflow=True
)
axis_fakes_eta = hist.axis.Regular(
    4, -2.4, 2.4, name="eta", underflow=False, overflow=False
)

# standard regular axes
axis_eta = hist.axis.Regular(24, -2.4, 2.4, name="eta", underflow=False, overflow=False)
axis_pt = hist.axis.Regular(
    56 - lep_pt_min, lep_pt_min, 56, name="pt", underflow=False, overflow=True
)
axis_phi = hist.axis.Regular(50, -math.pi, math.pi, name="phi", circular=True)
axis_iso = hist.axis.Regular(50, 0, 1, underflow=False, overflow=True, name="iso")
axis_ptW = hist.axis.Variable(
    [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 25, 30, 40, 50, 60, 75, 90, 150],
    name="ptW",
    underflow=False,
    overflow=True,
)
axis_mt = hist.axis.Variable(
    [0] + list(range(mtw_min, 150, 1)) + [150],
    name="mt",
    underflow=False,
    overflow=True,
)
axis_lin = hist.axis.Regular(5, 0, 5, name="lin")

qcdScaleByHelicity_helper = (
    wremnants.theory_corrections.make_qcd_uncertainty_helper_by_helicity()
)
axis_ptVgen = qcdScaleByHelicity_helper.hist.axes["ptVgen"]
axis_chargeVgen = qcdScaleByHelicity_helper.hist.axes["chargeVgen"]

gen_axes = {
    "ptVGen": hist.axis.Variable(
        [0, 8, 14, 20, 30, 40, 50, 60, 75, 90, 150],
        name="ptVGen",
        underflow=False,
        overflow=False,
    ),
}

groups_to_aggregate = args.aggregateGroups

if isUnfolding:
    unfolding_axes, unfolding_cols, unfolding_selections = (
        differential.get_dilepton_axes(args.genAxes, gen_axes)
    )
    datasets = unfolding_tools.add_out_of_acceptance(datasets, group=base_group)
    groups_to_aggregate.append(f"{base_group}OOA")

# axes/columns for unfolding ptW
nominal_axes = [
    axis_fakes_pt,
    axis_fakes_eta,
    common.axis_charge,
    axis_ptW,
    common.axis_passIso,
    common.axis_passMT,
]
nominal_cols = ["lep_pt", "lep_eta", "lep_charge", "ptW", "passIso", "passMT"]

# axes/columns for mW measurement using mT
axes_mt = [
    axis_fakes_pt,
    axis_fakes_eta,
    common.axis_charge,
    axis_mt,
    common.axis_passIso,
]
cols_mt = ["lep_pt", "lep_eta", "lep_charge", "transverseMass", "passIso"]

# axes only needed to compute fakerate in non-mt histograms
axes_fakerate = [
    axis_fakes_pt,
    axis_fakes_eta,
    common.axis_charge,
    common.axis_passIso,
    axis_mt,
]  ## was axis_mt
columns_fakerate = [
    "lep_pt",
    "lep_eta",
    "lep_charge",
    "passIso",
    "transverseMass",
]  ## was transverseMass


# extra axes which can be used to label tensor_axes
theory_corrs = [*args.theoryCorr, *args.ewTheoryCorr]
corr_helpers = theory_corrections.load_corr_helpers(
    [d.name for d in datasets if d.name in common.vprocs_lowpu], theory_corrs
)

# recoil initialization
if not args.noRecoil:
    from wremnants import recoil_tools

    recoilHelper = recoil_tools.Recoil("lowPU", args, flavor)


def build_graph(df, dataset):
    logger.info(f"build graph for dataset: {dataset.name}")
    results = []
    isQCDMC = dataset.group == "QCD"

    if dataset.is_data:
        df = df.DefinePerSample("weight", "1.0")
    else:
        df = df.Define("weight", "std::copysign(1.0, genWeight)")
    weightsum = df.SumAndCount("weight")
    df = df.Define("isEvenEvent", "event % 2 == 0")

    axes = nominal_axes
    cols = nominal_cols

    if isUnfolding and dataset.name in sigProcs:
        df = unfolding_tools.define_gen_level(
            df, args.genLevel, dataset.name, mode=analysis_label
        )

        if hasattr(dataset, "out_of_acceptance"):
            df = unfolding_tools.select_fiducial_space(
                df,
                mode="wmass",
                pt_min=args.pt[1],
                pt_max=args.pt[2],
                mtw_min=mtw_min,
                selections=unfolding_selections,
                accept=False,
            )
        else:
            df = unfolding_tools.select_fiducial_space(
                df,
                mode="wmass",
                pt_min=args.pt[1],
                pt_max=args.pt[2],
                mtw_min=mtw_min,
                selections=unfolding_selections,
                accept=True,
            )

            unfolding_tools.add_xnorm_histograms(
                results,
                df,
                args,
                dataset.name,
                corr_helpers,
                qcdScaleByHelicity_helper,
                unfolding_axes,
                unfolding_cols,
            )
            axes = [*axes, *unfolding_axes]
            cols = [*cols, *unfolding_cols]

    if flavor == "mu":
        if not dataset.is_data:
            df = df.Define(
                "Muon_pt_corr",
                "wrem::applyRochesterMC(Muon_pt, Muon_eta, Muon_phi, Muon_charge, Muon_genPartIdx, GenPart_pt, Muon_nTrackerLayers)",
            )
            # df = df.Alias("Muon_pt_corr", "Muon_pt")
            df = df.Filter("HLT_Mu17")
        else:
            df = df.Define(
                "Muon_pt_corr",
                "wrem::applyRochesterData(Muon_pt, Muon_eta, Muon_phi, Muon_charge)",
            )
            # df = df.Alias("Muon_pt_corr", "Muon_pt")
            df = df.Filter("HLT_HIMu17")

        df = df.Define(
            "vetoMuons",
            "Muon_pt_corr > 10 && Muon_looseId && abs(Muon_eta) < 2.4 && abs(Muon_dxybs) < 0.05",
        )
        df = df.Filter("Sum(vetoMuons) == 1")

        df = df.Define(
            "vetoElectrons",
            "Electron_pt > 10 && Electron_cutBased > 0 && abs(Electron_eta) < 2.4",
        )
        df = df.Filter("Sum(vetoElectrons) == 0")

        df = df.Define(
            "goodLeptons", f"vetoMuons && Muon_pt_corr > {lep_pt_min} && Muon_mediumId"
        )  # ISO requirement comes later && Muon_pfRelIso04_all < 0.15
        df = df.Define("goodLeptonsPlus", "goodLeptons && Muon_charge > 0")
        df = df.Define("goodLeptonsMinus", "goodLeptons && Muon_charge < 0")
        df = df.Filter("Sum(goodLeptons) == 1")

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
        df = df.Define("Lep_iso", "Muon_pfRelIso04_all[goodLeptons]")

        df = df.Define("passIso", "Lep_iso < 0.15")

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
        df = df.Filter("Sum(vetoElectrons)==1")

        df = df.Define(
            "vetoMuons",
            "Muon_pt > 10 && Muon_looseId && abs(Muon_eta) < 2.4 && abs(Muon_dxybs) < 0.05 && abs(Muon_dz)< 0.2",
        )
        df = df.Filter("Sum(vetoMuons) == 0")

        # df = df.Define("goodLeptons", f"vetoElectrons && Electron_cutBased >= 3 && !(abs(Electron_eta) > 1.4442 && abs(Electron_eta) < 1.566) && Electron_pt_corr > {lep_pt_min}")
        df = df.Define(
            "Electron_MediumID",
            "wrem::electron_id::pass_cutbased_noiso<3>(Electron_vidNestedWPBitmap)",
        )
        # df = df.Define("goodLeptons", "Electron_MediumID > 0")
        df = df.Define(
            "goodLeptons",
            f"vetoElectrons && Electron_MediumID > 0 && !(abs(Electron_eta) > 1.4442 && abs(Electron_eta) < 1.566) && Electron_pt_corr > {lep_pt_min}",
        )

        df = df.Filter("Sum(goodLeptons)==1")

        df = df.Define("goodLeptonsPlus", "goodLeptons && Electron_charge > 0")
        df = df.Define("goodLeptonsMinus", "goodLeptons && Electron_charge < 0")

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
        # df = df.Define("Lep_iso", "Electron_pfRelIso03_all[goodLeptons]") # Electron_miniPFRelIso_all Electron_pfRelIso03_all
        # df = df.Define("passIso", "Lep_iso < 0.15")
        df = df.Define(
            "passIso",
            "wrem::electron_id::pass_iso<3>(Electron_vidNestedWPBitmap[goodLeptons])[0] > 0",
        )

    df = df.Define(
        "trigMatch",
        "wrem::hasTriggerMatchLowPU(Lep_eta, Lep_phi, TrigObj_eta[goodTrigObjs], TrigObj_phi[goodTrigObjs])",
    )
    df = df.Define("nonTrigMatch", "wrem::inverse(trigMatch)")
    df = df.Filter("Sum(trigMatch) > 0")

    df = df.Define("lep_pt_uncorr", "Lep_pt_uncorr[0]")
    df = df.Define("lep_pt", "Lep_pt[0]")
    df = df.Define("lep_eta", "Lep_eta[0]")
    df = df.Define("lep_abseta", "abs(lep_eta)")
    df = df.Define("lep_phi", "Lep_phi[0]")
    df = df.Define("lep_charge", "Lep_charge[0]")
    df = df.Define("lep_mass", "Lep_mass[0]")
    # df = df.Define("lep_iso", "Lep_iso[0]")

    df = muon_selections.apply_met_filters(df)

    if not dataset.is_data:
        if flavor == "mu":
            df = df.Define("lepSF_ISO", "wrem::lepSF(Lep_pt, Lep_eta, Lep_charge, 1)")
            df = df.Define(
                "lepSF_IDIP", "wrem::lepSF(Lep_pt, Lep_eta, Lep_charge, 2)"
            )  # largest effect
            df = df.Define(
                "lepSF_HLT", "wrem::lepSF_HLT_q(Lep_pt, Lep_eta, Lep_charge, 13)"
            )
            df = df.Define(
                "prefireCorr",
                "wrem::prefireCorr(0, Jet_pt, Jet_eta, Jet_phi, Jet_muEF, Jet_neEmEF, Jet_chEmEF, Photon_pt, Photon_eta, Photon_phi, Lep_pt, Lep_eta, Lep_phi)",
            )
            df = df.Define("SFMC", "lepSF_IDIP*lepSF_ISO*lepSF_HLT*prefireCorr")
        else:
            df = df.Define("lepSF_IDISO", "wrem::lepSF(Lep_pt, Lep_eta, Lep_charge, 3)")
            df = df.Define(
                "lepSF_HLT", "wrem::lepSF_HLT_q(Lep_pt, Lep_eta, Lep_charge, 11)"
            )
            df = df.Define(
                "prefireCorr",
                "wrem::prefireCorr(0, Jet_pt, Jet_eta, Jet_phi, Jet_muEF, Jet_neEmEF, Jet_chEmEF, Photon_pt, Photon_eta, Photon_phi, Lep_pt, Lep_eta, Lep_phi)",
            )
            df = df.Define("SFMC", "lepSF_IDISO*lepSF_HLT*prefireCorr")

        df = df.Define("exp_weight", "SFMC")
        df = theory_tools.define_theory_weights_and_corrs(
            df, dataset.name, corr_helpers, args
        )
    else:
        df = df.DefinePerSample("nominal_weight", "1.0")

    # gen match to bare muons to select only prompt muons from MC processes, but also including tau decays
    if not dataset.is_data and not isQCDMC and not args.noGenMatchMC:
        df = theory_tools.define_postfsr_vars(df)
        postFSRLeps = "postfsrMuons" if flavor == "mu" else "postfsrElectrons"
        df = df.Filter(
            f"wrem::hasMatchDR2(lep_eta,lep_phi,GenPart_eta[{postFSRLeps}],GenPart_phi[{postFSRLeps}],0.09)"
        )

    df = df.Define("noTrigMatch", "Sum(trigMatch)")
    results.append(
        df.HistoBoost("noTrigMatch", [axis_lin], ["noTrigMatch", "nominal_weight"])
    )

    # Recoil calibrations
    if not args.noRecoil:
        leps_uncorr = ["lep_pt_uncorr", "lep_eta", "lep_phi", "lep_charge"]
        leps_corr = ["lep_pt", "lep_eta", "lep_phi", "lep_charge"]
        df = recoilHelper.recoil_W(
            df,
            results,
            dataset,
            common.vprocs_lowpu,
            leps_uncorr,
            leps_corr,
            cols_fakerate=columns_fakerate,
            axes_fakerate=axes_fakerate,
            mtw_min=mtw_min,
        )  # produces corrected MET as MET_corr_rec_pt/phi
    else:
        df = df.Alias("MET_corr_rec_pt", "MET_pt")
        df = df.Alias("MET_corr_rec_phi", "MET_phi")

    df = df.Define(
        "transverseMass",
        "wrem::mt_2(lep_pt, lep_phi, MET_corr_rec_pt, MET_corr_rec_phi)",
    )
    df = df.Define("passMT", f"transverseMass > {mtw_min}")

    results.append(
        df.HistoBoost(
            "lep_pt_eta_phi",
            [
                axis_pt,
                axis_eta,
                axis_phi,
                common.axis_charge,
                common.axis_passMT,
                common.axis_passIso,
            ],
            [
                "lep_pt",
                "lep_eta",
                "lep_phi",
                "lep_charge",
                "passMT",
                "passIso",
                "nominal_weight",
            ],
        )
    )

    # df = df.Define("iso_tmp", "if(lep_iso > 0.15) { std::cout << lep_iso << std::endl; } return lep_iso;")
    # results.append(df.HistoBoost("lep_iso", [axis_iso], ["iso_tmp", "nominal_weight"]))

    # results.append(df.HistoBoost("qcd_space", [axis_pt, axis_eta, axis_iso, common.axis_charge, axis_mT], ["lep_pt", "lep_eta", "lep_iso", "lep_charge", "transverseMass", "nominal_weight"]))

    df = df.Define(
        "ptW", "wrem::pt_2(lep_pt, lep_phi, MET_corr_rec_pt, MET_corr_rec_phi)"
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

            if dataset.name in common.vprocs_lowpu:
                df = syst_tools.add_theory_hists(
                    results,
                    df,
                    args,
                    dataset.name,
                    corr_helpers,
                    qcdScaleByHelicity_helper,
                    a,
                    c,
                    base_name=n,
                )

            # lepton efficiencies
            if flavor == "mu" and True:
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
                df = recoilHelper.add_recoil_unc_W(df, results, dataset, c, a, n)

    if dataset.name in sigProcs:
        # dummy lepton momentum scale
        netabins = 1
        nweights = 21
        mag = 1.0e-4
        df = df.Define(
            f"leptonScaleDummy{netabins}Bins",
            f"wrem::dummyScaleFromMassWeights<{netabins}, {nweights}>(nominal_weight, massWeight_tensor, lep_abseta, {mag})",
        )
        scale_etabins_axis = hist.axis.Regular(
            netabins, -2.4, 2.4, name="scaleEtaSlice", underflow=False, overflow=False
        )
        leptonMuonScaleSyst = df.HistoBoost(
            "transverseMass_leptonScaleSyst",
            axes_mt,
            [*cols_mt, f"leptonScaleDummy{netabins}Bins"],
            tensor_axes=[common.down_up_axis, scale_etabins_axis],
        )
        results.append(leptonMuonScaleSyst)

    if hasattr(dataset, "out_of_acceptance"):
        # Rename dataset to not overwrite the original one
        dataset.name = dataset.name + "OOA"

    return results, weightsum


resultdict = narf.build_and_run(datasets, build_graph)

if not args.noScaleToData:
    scale_to_data(resultdict)
    aggregate_groups(datasets, resultdict, groups_to_aggregate)

write_analysis_output(resultdict, f"mw_lowPU_{flavor}.hdf5", args)
