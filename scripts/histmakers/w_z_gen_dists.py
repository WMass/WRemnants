import os

import hist
import numpy as np

import narf
from utilities import common, differential, parsing
from wremnants import (
    helicity_utils,
    syst_tools,
    theory_corrections,
    theory_tools,
    unfolding_tools,
)
from wremnants.datasets.datagroups import Datagroups
from wremnants.datasets.dataset_tools import getDatasets
from wremnants.histmaker_tools import write_analysis_output
from wums import boostHistHelpers as hh
from wums import logging

analysis_label = Datagroups.analysisLabel(os.path.basename(__file__))
parser, initargs = parsing.common_parser(analysis_label)

parser.add_argument(
    "--skipHelicityXsecs",
    action="store_true",
    help="Skip the conversion of helicity helicity_xsecs to angular coeff fractions",
)
parser.add_argument(
    "--propagatePDFstoHelicity",
    action="store_true",
    help="Propagate PDF uncertainties to helicity xsecs",
)
parser.add_argument(
    "--useTheoryAgnosticBinning",
    action="store_true",
    help="Use theory agnostic binning (coarser) to produce the gen results",
)
parser.add_argument(
    "--useUnfoldingBinning",
    action="store_true",
    help="Use unfolding binning to produce the gen results",
)
parser.add_argument(
    "--singleLeptonHists",
    action="store_true",
    help="Also store single lepton kinematics",
)
parser.add_argument(
    "--photonHists", action="store_true", help="Also store photon kinematics"
)
parser.add_argument(
    "--skipEWHists",
    action="store_true",
    help="Also store histograms for EW reweighting. Use with --filter horace",
)
parser.add_argument("--signedY", action="store_true", help="use signed Y")
parser.add_argument(
    "--fiducial",
    choices=["masswindow", "dilepton", "singlelep"],
    help="Apply selection on leptons (No argument for inclusive)",
)
parser.add_argument(
    "--auxiliaryHistograms",
    action="store_true",
    help="Safe auxiliary histograms (mainly for ew analysis)",
)
parser.add_argument(
    "--ptqVgen",
    action="store_true",
    help="To store qt by Q variable instead of ptVgen, GEN only ",
    default=None,
)
parser.add_argument(
    "--helicity", action="store_true", help="Make qcdScaleByHelicity hist"
)
parser.add_argument(
    "--theoryCorrections", action="store_true", help="Apply default theory corrections"
)
parser.add_argument(
    "--addHelicityAxis", action="store_true", help="Add helicity to nominal axes"
)
parser.add_argument(
    "--addCharmAxis",
    action="store_true",
    help="Add axis to store info if the event has an outgoing charm quark",
)
parser.add_argument(
    "--finePtVBinning",
    action="store_true",
    help="Use 1 GeV binning for ptVgen (e.g., for theory corrections)",
)

parser = parsing.set_parser_default(parser, "filterProcs", common.vprocs)
args = parser.parse_args()

if not args.theoryCorrections:
    parser = parsing.set_parser_default(parser, "theoryCorr", [])
    parser = parsing.set_parser_default(parser, "ewTheoryCorr", [])
args = parser.parse_args()

logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)

datasets = getDatasets(
    maxFiles=args.maxFiles,
    filt=args.filterProcs,
    excl=args.excludeProcs,
    extended="msht20an3lo" not in args.pdfs,
    nanoVersion="v9",
    base_path=args.dataPath,
    mode=analysis_label,
)

logger.debug(f"Will process samples {[d.name for d in datasets]}")

axis_ygen = hist.axis.Regular(10, -5.0, 5.0, name="y")
col_rapidity = "yVgen" if args.signedY else "absYVgen"

axis_ptqVgen = hist.axis.Variable(
    [round(x, 4) for x in list(np.arange(0, 0.1 + 0.0125, 0.0125))]
    + [round(x, 4) for x in list(np.arange(0.1 + 0.025, 0.5 + 0.025, 0.025))],
    name="ptqVgen",
    underflow=False,
)

axis_chargeWgen = hist.axis.Regular(
    2, -2, 2, name="chargeVgen", underflow=False, overflow=False
)

axis_chargeZgen = hist.axis.Integer(
    0, 1, name="chargeVgen", underflow=False, overflow=False
)


axis_absetal_gen = hist.axis.Regular(24, 0, 2.4, name="absEtaGen")
axis_ptl_gen = hist.axis.Regular(34, 26.0, 60.0, name="ptGen")
axis_chargel_gen = hist.axis.Regular(
    2, -2.0, 2.0, underflow=False, overflow=False, name=f"qGen"
)

theory_corrs = [*args.theoryCorr, *args.ewTheoryCorr]
corr_helpers = theory_corrections.load_corr_helpers(common.vprocs, theory_corrs)


def build_graph(df, dataset):
    logger.info("build graph")
    logger.info(dataset.name)
    results = []

    if dataset.is_data:
        raise RuntimeError("Running GEN analysis over data is not supported")

    isW = dataset.name.startswith("W") and dataset.name[1] not in [
        "W",
        "Z",
    ]  # in common.wprocs
    isZ = dataset.name.startswith("Z") and dataset.name[1] not in [
        "W",
        "Z",
    ]  # in common.zprocs

    if args.addCharmAxis:
        axis_massWgen = hist.axis.Variable(
            [4.0, 13000.0], name="massVgen", underflow=True, overflow=False
        )
    else:
        axis_massWgen = hist.axis.Regular(
            120, 0, 120.0, name="massVgen", underflow=True, overflow=True
        )

    axis_massZgen = hist.axis.Regular(12, 60.0, 120.0, name="massVgen")

    theoryAgnostic_axes, _ = differential.get_theoryAgnostic_axes(
        ptV_flow=True, absYV_flow=True, wlike="Z" in dataset.name
    )
    axis_ptV_thag = theoryAgnostic_axes[0]
    axis_yV_thag = theoryAgnostic_axes[1]

    if args.useUnfoldingBinning:
        unfolding_axes, unfolding_cols, unfolding_selections = (
            differential.get_dilepton_axes(
                ["ptVGen", "absYVGen"],
                {
                    "ptll": common.get_dilepton_ptV_binning(fine=False),
                    "yll": common.yll_10quantiles_binning,
                },
                "prefsr",
                add_out_of_acceptance_axis=False,
            )
        )
        axis_absYVgen = hist.axis.Variable(
            unfolding_axes[1].edges,
            name="absYVgen",
            underflow=False,
        )
        axis_ptVgen = hist.axis.Variable(
            unfolding_axes[0].edges,
            name="ptVgen",
            underflow=False,
        )

        axis_massZgen = hist.axis.Regular(1, 60.0, 120.0, name="massVgen")

    elif args.useTheoryAgnosticBinning:
        axis_absYVgen = hist.axis.Variable(
            axis_yV_thag.edges,  # same axis as theory agnostic norms
            name="absYVgen",
            underflow=False,
        )
        axis_ptVgen = hist.axis.Variable(
            axis_ptV_thag.edges,  # same axis as theory agnostic norms,
            name="ptVgen",
            underflow=False,
        )
    else:
        axis_absYVgen = hist.axis.Variable(
            [
                0.0,
                0.25,
                0.5,
                0.75,
                1.0,
                1.25,
                1.5,
                1.75,
                2.0,
                2.25,
                2.5,
                2.75,
                3.0,
                3.25,
                3.5,
                4.0,
                5.0,
            ],  # this is the same binning as hists from theory corrections
            name="absYVgen",
            underflow=False,
        )
        axis_ptVgen = hist.axis.Variable(
            (*common.get_dilepton_ptV_binning(fine=args.finePtVBinning), 13000.0),
            name="ptVgen",
            underflow=False,
        )

    axis_rapidity = axis_ygen if args.signedY else axis_absYVgen

    weight_expr = "std::copysign(1.0, genWeight)"

    if "reweight_h2" in dataset.name:
        weight_expr = f"{weight_expr}*H2BugFixWeight[0]"
    elif "NNLOPS" in dataset.name:
        weight_expr = f"{weight_expr}*LHEScaleWeightAltSet1[4]"

    df = df.Define("weight", weight_expr)
    df = df.DefinePerSample("unity", "1.")
    # This sum should happen before any change of the weight
    weightsum = df.SumAndCount("weight")
    df = df.Define("isEvenEvent", "event % 2 == 0")

    df = theory_tools.define_theory_weights_and_corrs(
        df, dataset.name, corr_helpers, args
    )

    if isZ:
        nominal_axes = [
            axis_massZgen,
            axis_rapidity,
            axis_ptqVgen if args.ptqVgen else axis_ptVgen,
            axis_chargeZgen,
        ]
    else:
        nominal_axes = [
            axis_massWgen,
            axis_rapidity,
            axis_ptqVgen if args.ptqVgen else axis_ptVgen,
            axis_chargeWgen,
        ]

    nominal_cols = [
        "massVgen",
        col_rapidity,
        "ptqVgen" if args.ptqVgen else "ptVgen",
        "chargeVgen",
    ]

    if args.addCharmAxis:
        axis_charm = hist.axis.Regular(
            2, -0.5, 1.5, underflow=False, overflow=False, name="charm"
        )
        nominal_axes = [*nominal_axes, axis_charm]
        nominal_cols = [*nominal_cols, "charm"]

        df = df.Define(
            "charm",
            "Sum(abs(LHEPart_pdgId[LHEPart_status==1])==4) == 1 && Sum(abs(LHEPart_pdgId[LHEPart_status==1])==5) != 1",
        )

    if args.addHelicityAxis:
        # add helicity axis, indices, and weights

        # make a new axis here to avoid name collision with histograms which otherwise already
        # have a helicity axis
        axis_helicitygen = hist.axis.Integer(
            -1, 8, name="helicity", overflow=False, underflow=False
        )

        # since mutliple tensor weights are not currently supported, convert the helicity tensor to a std::array which
        # will be looped over during the filling (together with the helicity indices)
        df = df.DefinePerSample("helicity_idxs", "wrem::seq_idxs_array<9>(-1)")
        df = df.Define(
            "helicity_moments",
            "wrem::tensor_to_array(wrem::csAngularMoments(csSineCosThetaPhigen))",
        )

        nominal_axes += [axis_helicitygen]
        nominal_cols += ["helicity_idxs", "helicity_moments"]

    mode = f'{"z" if isZ else "w"}_{analysis_label}'
    if args.fiducial is not None:
        if isZ and args.fiducial == "singlelep":
            mode += "_wlike"

        df = unfolding_tools.select_fiducial_space(
            df,
            mode=mode,
            fiducial=args.fiducial,
            unfolding=True,
            selections=unfolding_selections,
        )

    if args.singleLeptonHists and (isW or isZ):
        df = unfolding_tools.define_gen_level(
            df, dataset.name, ["prefsr"], mode="w_mass" if isW else "z_wlike"
        )

        lep_cols = ["prefsrLep_absEta", "prefsrLep_pt", "prefsrLep_charge"]
        lep_axes = [axis_absetal_gen, axis_ptl_gen, axis_chargel_gen]

        if args.addCharmAxis:
            lep_axes = [*lep_axes, axis_charm]
            lep_cols = [*lep_cols, "charm"]

        results.append(
            df.HistoBoost(
                "prefsr_lep",
                lep_axes,
                [*lep_cols, "nominal_weight"],
                storage=hist.storage.Weight(),
            )
        )

    if not args.skipEWHists and (isW or isZ) and "Zmumu_powheg-weak" in dataset.name:
        if isZ:
            massBins = theory_tools.make_ew_binning(
                mass=91.1535,
                width=2.4932,
                initialStep=0.10,
                bin_edges_low=[0, 46, 50, 60, 70, 80],
                bin_edges_high=[100, 110, 120, 140, 160, 200],
            )
        else:
            massBins = theory_tools.make_ew_binning(
                mass=80.3815, width=2.0904, initialStep=0.010
            )

        # LHE level
        df = syst_tools.define_weak_weights(df, dataset.name)
        axis_lheMV = hist.axis.Variable(massBins, name="massVlhe", underflow=False)
        axis_lhePtV = hist.axis.Variable(
            common.ptV_binning, underflow=False, name="ptVlhe"
        )
        axis_lheAbsYV = hist.axis.Regular(50, 0, 5, underflow=False, name="absYVlhe")
        axis_lheYV = hist.axis.Regular(100, -5.0, 5.0, name="YVlhe")
        axis_lhechargeZ = hist.axis.Integer(
            0, 1, underflow=False, overflow=False, name="chargeVlhe"
        )
        axis_lhechargeW = hist.axis.Regular(
            2, -2.0, 2.0, underflow=False, overflow=False, name="chargeVlhe"
        )
        axis_lhechargeV = axis_lhechargeZ if isZ else axis_lhechargeW
        axis_lheCosThetaStar = hist.axis.Regular(50, -1, 1, name="cosThetaStarlhe")
        axis_lhePhiStar = hist.axis.Regular(
            8, -np.pi, np.pi, circular=True, name="phiStarlhe"
        )
        axis_weak = hist.axis.StrCategory(syst_tools.weakWeightNames(), name="weak")
        axis_helicity = helicity_utils.axis_helicity

        results.append(
            df.HistoBoost(
                "lhe_massVptV",
                [axis_lheMV, axis_lhePtV],
                ["massVlhe", "ptVlhe", "nominal_weight"],
                storage=hist.storage.Weight(),
            )
        )
        results.append(
            df.HistoBoost(
                "lhe_absYVptV",
                [axis_lheAbsYV, axis_lhePtV],
                ["absYVlhe", "ptVlhe", "nominal_weight"],
                storage=hist.storage.Weight(),
            )
        )
        results.append(
            df.HistoBoost(
                "lhe_absYVmassV",
                [axis_lheAbsYV, axis_lheMV],
                ["absYVlhe", "massVlhe", "nominal_weight"],
                storage=hist.storage.Weight(),
            )
        )
        results.append(
            df.HistoBoost(
                "lhe_massVcosTheta",
                [axis_lheMV, axis_lheCosThetaStar],
                ["massVlhe", "csCosThetalhe", "nominal_weight"],
                storage=hist.storage.Weight(),
            )
        )
        syst_tools.add_weakweights_hist(
            results,
            df,
            [axis_lheMV, axis_lheCosThetaStar],
            ["massVlhe", "csCosThetalhe"],
            proc=dataset.name,
            base_name="lhe_massVcosTheta",
        )

        results.append(
            df.HistoBoost(
                "lhe",
                [axis_lheMV, axis_lheAbsYV, axis_lhePtV, axis_lhechargeV],
                ["massVlhe", "absYVlhe", "ptVlhe", "chargeVlhe", "nominal_weight"],
                storage=hist.storage.Weight(),
            )
        )

        results.append(
            df.HistoBoost(
                "lhe_angular",
                [
                    axis_lheMV,
                    axis_lheAbsYV,
                    axis_lhePtV,
                    axis_lhechargeV,
                    axis_lheCosThetaStar,
                    axis_lhePhiStar,
                ],
                [
                    "massVlhe",
                    "absYVlhe",
                    "ptVlhe",
                    "chargeVlhe",
                    "csCosThetalhe",
                    "csPhilhe",
                    "nominal_weight",
                ],
                storage=hist.storage.Weight(),
            )
        )

        hist_lhe_helicity = df.HistoBoost(
            "lhe_helicity",
            [axis_lheMV, axis_lheAbsYV, axis_lhePtV, axis_lhechargeV],
            [
                "massVlhe",
                "absYVlhe",
                "ptVlhe",
                "chargeVlhe",
                "csAngularMomentslhe_wnom",
            ],
            tensor_axes=[axis_helicity],
        )
        results.append(hist_lhe_helicity)

        hist_lhe_weak_helicity = df.HistoBoost(
            "lhe_weak_helicity",
            [axis_lheMV, axis_lheAbsYV, axis_lhePtV, axis_lhechargeV],
            [
                "massVlhe",
                "absYVlhe",
                "ptVlhe",
                "chargeVlhe",
                "weakWeight_tensor_helicity",
            ],
            tensor_axes=[axis_helicity, axis_weak],
        )
        results.append(hist_lhe_weak_helicity)

        results.append(
            df.HistoBoost(
                "lhe_weak_angular",
                [
                    axis_lheMV,
                    axis_lheAbsYV,
                    axis_lhePtV,
                    axis_lhechargeV,
                    axis_lheCosThetaStar,
                    axis_lhePhiStar,
                ],
                [
                    "massVlhe",
                    "absYVlhe",
                    "ptVlhe",
                    "chargeVlhe",
                    "csCosThetalhe",
                    "csPhilhe",
                    "weakWeight_tensor_wnom",
                ],
                storage=hist.storage.Weight(),
                tensor_axes=[axis_weak],
            )
        )

    if (
        not args.skipEWHists
        and (isW or isZ)
        and "GenPart_status" in df.GetColumnNames()
    ):
        if isZ:
            massBins = theory_tools.make_ew_binning(
                mass=91.1535,
                width=2.4932,
                initialStep=0.010,
                bin_edges_low=[0, 50, 60],
                bin_edges_high=[120],
            )
        else:
            massBins = theory_tools.make_ew_binning(
                mass=80.3815, width=2.0904, initialStep=0.010
            )

        # pre FSR
        axis_genMV = hist.axis.Variable(massBins, name="massVgen", underflow=False)
        axis_genPtV = hist.axis.Variable(
            common.ptV_binning, underflow=False, name="ptVgen"
        )
        axis_genAbsYV = hist.axis.Regular(50, 0, 5, name="absYVgen")
        results.append(
            df.HistoBoost(
                "preFSR_massVptV",
                [axis_genMV, axis_genPtV],
                ["massVgen", "ptVgen", "nominal_weight"],
                storage=hist.storage.Weight(),
            )
        )
        results.append(
            df.HistoBoost(
                "preFSR_absYVptV",
                [axis_genAbsYV, axis_genPtV],
                ["absYVgen", "ptVgen", "nominal_weight"],
                storage=hist.storage.Weight(),
            )
        )
        results.append(
            df.HistoBoost(
                "preFSR_absYVmassV",
                [axis_genAbsYV, axis_genMV],
                ["absYVgen", "massVgen", "nominal_weight"],
                storage=hist.storage.Weight(),
            )
        )

        # post FSR, pre tau decay
        axis_ewMll = hist.axis.Variable(massBins, name="ewMll", underflow=False)
        axis_ewPtll = hist.axis.Variable(
            common.ptV_binning, underflow=False, name="ewPTll"
        )
        axis_ewAbsYll = hist.axis.Regular(50, 0, 5, name="ewAbsYll")
        results.append(
            df.HistoBoost(
                "ew_MllPTll",
                [axis_ewMll, axis_ewPtll],
                ["ewMll", "ewPTll", "nominal_weight"],
                storage=hist.storage.Weight(),
            )
        )
        results.append(
            df.HistoBoost(
                "ew_YllPTll",
                [axis_ewAbsYll, axis_ewPtll],
                ["ewAbsYll", "ewPTll", "nominal_weight"],
                storage=hist.storage.Weight(),
            )
        )
        results.append(
            df.HistoBoost(
                "ew_YllMll",
                [axis_ewAbsYll, axis_ewMll],
                ["ewAbsYll", "ewMll", "nominal_weight"],
                storage=hist.storage.Weight(),
            )
        )

        # dressed
        axis_ewMll = hist.axis.Variable(massBins, name="ewMll", underflow=False)
        axis_ewPtll = hist.axis.Variable(
            common.ptV_binning, underflow=False, name="ewPTll"
        )
        axis_ewAbsYll = hist.axis.Regular(50, 0, 5, name="ewAbsYll")
        df = theory_tools.define_dressed_vars(df, mode=mode)
        results.append(
            df.HistoBoost(
                "dressed_MllPTll",
                [axis_ewMll, axis_ewPtll],
                ["dressed_MV", "dressed_PTV", "nominal_weight"],
                storage=hist.storage.Weight(),
            )
        )
        results.append(
            df.HistoBoost(
                "dressed_YllPTll",
                [axis_ewAbsYll, axis_ewPtll],
                ["dressed_absYV", "dressed_PTV", "nominal_weight"],
                storage=hist.storage.Weight(),
            )
        )
        results.append(
            df.HistoBoost(
                "dressed_YllMll",
                [axis_ewAbsYll, axis_ewMll],
                ["dressed_absYV", "dressed_MV", "nominal_weight"],
                storage=hist.storage.Weight(),
            )
        )

        if args.auxiliaryHistograms:
            axis_ewMlly = hist.axis.Variable(massBins, name="ewMlly")
            results.append(
                df.HistoBoost(
                    "nominal_ewMlly",
                    [axis_ewMlly],
                    ["ewMlly", "nominal_weight"],
                    storage=hist.storage.Weight(),
                )
            )
            # coarse binning
            axis_Mll = hist.axis.Regular(100, 50, 150, name="Mll")
            results.append(
                df.HistoBoost(
                    "nominal_Mll",
                    [axis_Mll],
                    ["ewMll", "nominal_weight"],
                    storage=hist.storage.Weight(),
                )
            )
            axis_Mlly = hist.axis.Regular(100, 50, 150, name="Mlly")
            results.append(
                df.HistoBoost(
                    "nominal_Mlly",
                    [axis_Mlly],
                    ["ewMlly", "nominal_weight"],
                    storage=hist.storage.Weight(),
                )
            )

            axis_PTll = hist.axis.Regular(100, 0, 100, name="PTll")
            axis_PTlly = hist.axis.Regular(100, 0, 100, name="PTlly")
            axis_Yll = hist.axis.Regular(100, -5, 5, name="Yll")
            axis_Ylly = hist.axis.Regular(100, -5, 5, name="Ylly")
            results.append(
                df.HistoBoost(
                    "nominal_PTll",
                    [axis_PTll],
                    ["ewPTll", "nominal_weight"],
                    storage=hist.storage.Weight(),
                )
            )
            results.append(
                df.HistoBoost(
                    "nominal_PTlly",
                    [axis_PTlly],
                    ["ewPTlly", "nominal_weight"],
                    storage=hist.storage.Weight(),
                )
            )
            results.append(
                df.HistoBoost(
                    "nominal_Yll",
                    [axis_Yll],
                    ["ewYll", "nominal_weight"],
                    storage=hist.storage.Weight(),
                )
            )
            results.append(
                df.HistoBoost(
                    "nominal_Ylly",
                    [axis_Ylly],
                    ["ewYlly", "nominal_weight"],
                    storage=hist.storage.Weight(),
                )
            )

            # single lepton hists
            if args.singleLeptonHists:
                if isZ:
                    # first lepton is leading in pT
                    df = df.Define("ewLepPt1", "ewLeptons[0].pt()")
                    df = df.Define("ewLepPt2", "ewLeptons[1].pt()")
                    df = df.Define("ewLepEta1", "ewLeptons[0].eta()")
                    df = df.Define("ewLepEta2", "ewLeptons[1].eta()")
                if isW:
                    # first lepton is charged
                    df = df.Define(
                        "ewLepPt1",
                        "ewLeptons[0].mass() == 0 ? ewLeptons[1].pt() : ewLeptons[0].pt()",
                    )
                    df = df.Define(
                        "ewLepPt2",
                        "ewLeptons[0].mass() == 0 ? ewLeptons[0].pt() : ewLeptons[1].pt()",
                    )
                    df = df.Define(
                        "ewLepEta1",
                        "ewLeptons[0].mass() == 0 ? ewLeptons[1].eta() : ewLeptons[0].eta()",
                    )
                    df = df.Define(
                        "ewLepEta2",
                        "ewLeptons[0].mass() == 0 ? ewLeptons[0].eta() : ewLeptons[1].eta()",
                    )

                axis_ewLepPt = hist.axis.Regular(100, 0, 100, name="pt")
                results.append(
                    df.HistoBoost(
                        "nominal_ewLepPt1",
                        [axis_ewLepPt],
                        ["ewLepPt1", "nominal_weight"],
                        storage=hist.storage.Weight(),
                    )
                )
                results.append(
                    df.HistoBoost(
                        "nominal_ewLepPt2",
                        [axis_ewLepPt],
                        ["ewLepPt2", "nominal_weight"],
                        storage=hist.storage.Weight(),
                    )
                )
                axis_ewLepEta = hist.axis.Regular(100, -5, 5, name="eta")
                results.append(
                    df.HistoBoost(
                        "nominal_ewLepEta1",
                        [axis_ewLepEta],
                        ["ewLepEta1", "nominal_weight"],
                        storage=hist.storage.Weight(),
                    )
                )
                results.append(
                    df.HistoBoost(
                        "nominal_ewLepEta2",
                        [axis_ewLepEta],
                        ["ewLepEta2", "nominal_weight"],
                        storage=hist.storage.Weight(),
                    )
                )

            if args.photonHists:
                # photon distributions
                df = df.Define("nPhotons", "ewPhotons.size()")
                df = df.Define(
                    "leadPhotonPt",
                    "ewPhotons.size() > 0 ? log10(ewPhotons[0].pt()) : -99",
                )
                df = df.Define(
                    "leadPhotonEta", "ewPhotons.size() > 0 ? ewPhotons[0].eta() : -99"
                )
                df = df.Define(
                    "sublPhotonPt",
                    "ewPhotons.size() > 1 ? log10(ewPhotons[1].pt()) : -99",
                )
                df = df.Define(
                    "sublPhotonEta", "ewPhotons.size() > 1 ? ewPhotons[1].eta() : -99"
                )
                df = df.Define(
                    "trailPhotonPt",
                    "ewPhotons.size() > 2 ? log10(ewPhotons[2].pt()) : -99",
                )
                df = df.Define(
                    "trailPhotonEta", "ewPhotons.size() > 2 ? ewPhotons[2].eta() : -99"
                )

                axis_ewNPhotons = hist.axis.Regular(5, 0, 5, name="n")
                results.append(
                    df.HistoBoost(
                        "nominal_ewPhotons",
                        [axis_ewNPhotons],
                        ["nPhotons", "nominal_weight"],
                        storage=hist.storage.Weight(),
                    )
                )

                axis_photonPt = hist.axis.Regular(100, -5, 5, name="pt")
                axis_photonEta = hist.axis.Regular(100, -5, 5, name="eta")
                results.append(
                    df.HistoBoost(
                        "nominal_leadPhoton",
                        [axis_photonPt, axis_photonEta],
                        ["leadPhotonPt", "leadPhotonEta", "nominal_weight"],
                        storage=hist.storage.Weight(),
                    )
                )
                results.append(
                    df.HistoBoost(
                        "nominal_sublPhoton",
                        [axis_photonPt, axis_photonEta],
                        ["sublPhotonPt", "sublPhotonEta", "nominal_weight"],
                        storage=hist.storage.Weight(),
                    )
                )
                results.append(
                    df.HistoBoost(
                        "nominal_trailPhoton",
                        [axis_photonPt, axis_photonEta],
                        ["trailPhotonPt", "trailPhotonEta", "nominal_weight"],
                        storage=hist.storage.Weight(),
                    )
                )

                # postfsr definition
                axis_eta = hist.axis.Regular(
                    25, 0, 2.5, name="postfsrLep_absEta", overflow=True, underflow=False
                )
                axis_pt = hist.axis.Regular(
                    50, 20, 70, name="postfsrLep_pt", overflow=True, underflow=True
                )
                results.append(
                    df.HistoBoost(
                        "nominal_postfsr",
                        [axis_eta, axis_pt],
                        ["postfsrLep_absEta", "postfsrLep_pt", "nominal_weight"],
                        storage=hist.storage.Weight(),
                    )
                )

    if "powheg" in dataset.name:
        return results, weightsum

    if (
        "horace" not in dataset.name
        and "winhac" not in dataset.name
        and "LHEScaleWeight" in df.GetColumnNames()
        and "LHEPdfWeight" in df.GetColumnNames()
    ):

        qcdScaleByHelicity_helper = (
            theory_corrections.make_qcd_uncertainty_helper_by_helicity(
                is_z=dataset.name[0] != "W"
            )
            if args.helicity
            else None
        )

        df = syst_tools.add_theory_hists(
            results,
            df,
            args,
            dataset.name,
            corr_helpers,
            qcdScaleByHelicity_helper,
            nominal_axes,
            nominal_cols,
            base_name="nominal_gen",
            propagateToHelicity=args.propagatePDFstoHelicity,
        )

        helicity_axes = nominal_axes[:-1] if args.addHelicityAxis else nominal_axes
        helicity_cols = nominal_cols[:-2] if args.addHelicityAxis else nominal_cols

        if args.addCharmAxis:
            helicity_axes = helicity_axes[:-1]
            helicity_cols = helicity_cols[:-1]

        df = syst_tools.add_helicity_hists(
            results,
            df,
            dataset.name,
            helicity_axes,
            helicity_cols,
            base_name="nominal_gen",
            storage=hist.storage.Weight(),
        )

    nominal_gen = df.HistoBoost(
        "nominal_gen",
        nominal_axes,
        [*nominal_cols, "nominal_weight"],
        storage=hist.storage.Weight(),
    )

    results.append(nominal_gen)

    return results, weightsum


resultdict = narf.build_and_run(datasets, build_graph)
write_analysis_output(
    resultdict, f"{os.path.basename(__file__).replace('py', 'hdf5')}", args
)

# FIXME currently the addHelicityAxis functionality interferes with the prouduction
# of these histograms, so skip them in that case
if not args.addHelicityAxis and not args.skipHelicityXsecs:
    logger.info("Writing out helicity cross sections")

    helicity_xsecs_out = {}
    for dataset in datasets:
        name = dataset.name
        if "nominal_gen_helicity_xsecs_scale" not in resultdict[name]["output"]:
            logger.warning(
                f"Failed to find nominal_gen_helicity_xsecs_scale hist for proc {name}. Skipping!"
            )
            continue
        for var in ["", "lhe", "hardProcess", "postShower", "postBeamRemnants"]:
            if name not in [
                "ZmumuPostVFP",
                "Zee_MiNNLO",
                "Zmumu_MiNNLO",
                "WplusmunuPostVFP",
                "WminusmunuPostVFP",
            ]:
                continue

            if var == "":
                suffix = ""
            else:
                suffix = f"_{var}"

            helicity_xsecs = resultdict[name]["output"][
                f"nominal_gen_helicity_xsecs_scale{suffix}"
            ].get()

            key = f"{name[0]}{suffix}"

            if key not in helicity_xsecs_out.keys():
                helicity_xsecs_out[key] = helicity_xsecs
            else:
                helicity_xsecs_out[key] = hh.addHists(
                    helicity_xsecs_out[key], helicity_xsecs, createNew=False
                )

        # Different PDF set predictions with alphaS variations
        for var in [
            "CT18ZalphaS002",
            "MSHT20alphaS002",
            "NNPDF31alphaS002",
            "HERAPDF20extalphaS002",
            "MSHT20an3loalphaS002",
            "PDF4LHC21alphaS001",
            "HERAPDF20alphaS002",
            "CT18alphaS002",
            "NNPDF40alphaS001",
        ]:
            if name not in [
                "ZmumuPostVFP",
                "Zee_MiNNLO",
                "Zmumu_MiNNLO",
                "WplusmunuPostVFP",
                "WminusmunuPostVFP",
            ]:
                continue

            histname = f"nominal_gen_helicity_nominal_gen_pdf{var}"
            if histname not in resultdict[name]["output"].keys():
                continue

            helicity_xsecs = resultdict[name]["output"][histname].get()

            key = f"{name[0]}_{var}"

            if key not in helicity_xsecs_out.keys():
                helicity_xsecs_out[key] = helicity_xsecs
            else:
                helicity_xsecs_out[key] = hh.addHists(
                    helicity_xsecs_out[key], helicity_xsecs, createNew=False
                )

    if helicity_xsecs_out:
        outfname = "w_z_helicity_xsecs"
        if args.signedY:
            outfname += "_signedY"
        if args.useTheoryAgnosticBinning:
            outfname += "_theoryAgnosticBinning"
        outfname += ".hdf5"
        write_analysis_output(helicity_xsecs_out, outfname, args)
