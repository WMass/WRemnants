#!/usr/bin/env python3
import argparse
import math

import hist
import numpy as np

import rabbit.debugdata
import rabbit.io_tools
from rabbit import tensorwriter
from utilities import common, parsing
from wremnants import (
    combine_helpers,
    combine_theory_helper,
    combine_theoryAgnostic_helper,
    syst_tools,
    theory_corrections,
    theory_tools,
)
from wremnants.datasets import datagroups
from wremnants.datasets.datagroups import Datagroups
from wremnants.histselections import FakeSelectorSimpleABCD
from wremnants.regression import Regressor
from wremnants.syst_tools import (
    massWeightNames,
    scale_hist_up_down,
    scale_hist_up_down_corr_from_file,
    widthWeightNames,
)
from wums import boostHistHelpers as hh
from wums import logging, output_tools


def make_subparsers(parser):

    parser.add_argument(
        "--analysisMode",
        type=str,
        default=None,
        choices=["unfolding", "theoryAgnosticNormVar", "theoryAgnosticPolVar"],
        help="Select analysis mode to run. Default is the traditional analysis",
    )

    tmpKnownArgs, _ = parser.parse_known_args()
    subparserName = tmpKnownArgs.analysisMode
    if subparserName is None:
        return parser

    parser.add_argument(
        "--poiAsNoi",
        action="store_true",
        help="Make histogram to do the POIs as NOIs trick (some postprocessing will happen later in CardTool.py)",
    )
    parser.add_argument(
        "--forceRecoChargeAsGen",
        action="store_true",
        help="Force gen charge to match reco charge in CardTool, this only works when the reco charge is used to define the channel",
    )
    parser.add_argument(
        "--genAxes",
        type=str,
        default=[],
        nargs="+",
        help="Specify which gen axes should be used in unfolding/theory agnostic, if 'None', use all (inferred from metadata).",
    )
    parser.add_argument(
        "--priorNormXsec",
        type=float,
        default=1,
        help=r"Prior for shape uncertainties on cross sections for theory agnostic or unfolding analysis with POIs as NOIs (1 means 100%). If negative, it will use shapeNoConstraint in the fit",
    )
    parser.add_argument(
        "--scaleNormXsecHistYields",
        type=float,
        default=None,
        help="Scale yields of histogram with cross sections variations for theory agnostic analysis with POIs as NOIs. Can be used together with --priorNormXsec",
    )

    if "theoryAgnostic" in subparserName:
        if subparserName == "theoryAgnosticNormVar":
            parser.add_argument(
                "--theoryAgnosticBandSize",
                type=float,
                default=1.0,
                help="Multiplier for theory-motivated band in theory agnostic analysis with POIs as NOIs.",
            )
            parser.add_argument(
                "--helicitiesToInflate",
                type=int,
                nargs="*",
                default=[],
                help="Select which helicities you want to scale",
            )
        elif subparserName == "theoryAgnosticPolVar":
            parser.add_argument(
                "--noPolVarOnFake",
                action="store_true",
                help="Do not propagate POI variations to fakes",
            )
            parser.add_argument(
                "--symmetrizePolVar",
                action="store_true",
                help="Symmetrize up/Down variations in CardTool (using average)",
            )
    elif "unfolding" in subparserName:
        parser.add_argument(
            "--unfoldingLevel",
            type=str,
            default="prefsr",
            choices=["prefsr", "postfsr"],
            help="Definition for unfolding",
        )
        parser.add_argument(
            "--unfoldingScalemap",
            type=str,
            default=[],
            nargs="+",
            help="Read parameter uncertainties from fitresult to assign proper NOI variations",
        )
        parser.add_argument(
            "--unfoldingWithFlow",
            action="store_true",
            help="Include underflow/overflow in masked channels (for iterative unfolding)",
        )
        parser.add_argument(
            "--unfoldSimultaneousWandZ",
            action="store_true",
            help="Simultaneously unfold W and Z and correlate Z background in W channel",
        )
        parser.add_argument(
            "--constrainNOIs",
            action="store_true",
            help="Constrain NOI variation",
        )

        parser = parsing.set_parser_default(parser, "massVariation", 10)

    return parser


def make_parser(parser=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--outfolder",
        type=str,
        default=".",
        help="Output folder with all the outputs of this script (subfolder WMass or ZMassWLike is created automatically inside)",
    )
    parser.add_argument("-i", "--inputFile", nargs="+", type=str)
    parser.add_argument(
        "-p", "--postfix", type=str, help="Postfix for output file name", default=None
    )
    parser.add_argument(
        "-v",
        "--verbose",
        type=int,
        default=3,
        choices=[0, 1, 2, 3, 4],
        help="Set verbosity level with logging, the larger the more verbose",
    )
    parser.add_argument(
        "--noColorLogger", action="store_true", help="Do not use logging with colors"
    )
    parser.add_argument(
        "--sparse",
        action="store_true",
        help="Write out datacard in sparse mode",
    )
    parser.add_argument(
        "--excludeProcGroups",
        type=str,
        nargs="*",
        help="Don't run over processes belonging to these groups (only accepts exact group names)",
        default=["QCD"],
    )
    parser.add_argument(
        "--filterProcGroups",
        type=str,
        nargs="*",
        help="Only run over processes belonging to these groups",
        default=[],
    )
    parser.add_argument(
        "--addBSMProcess",
        type=str,
        default=None,
        help="Add BSM as independent process, not propagating the effect into the fakes",
        choices=[
            "WtoNMuMass5",
            "WtoNMuMass10",
            "WtoNMuMass30",
            "WtoNMuMass50",
            "WtoMuNuSMEFT",
        ],
    )
    parser.add_argument(
        "--addBSMMixing",
        nargs=2,
        default=None,
        help="""
        Add BSM as systematic variation on SM signal by mixing (i.e. reduce SM with by amount of BSM), the effect is also propagated into the fakes. 
        First argument is the BSM sample name, second is the mixing (number from 0 to 1).
        """,
    )
    parser.add_argument(
        "-x",
        "--excludeNuisances",
        type=str,
        default="",
        help="Regular expression to exclude some systematics from the datacard",
    )
    parser.add_argument(
        "-k",
        "--keepNuisances",
        type=str,
        default="",
        help="Regular expression to keep some systematics, overriding --excludeNuisances. Can be used to keep only some systs while excluding all the others with '.*'",
    )
    parser.add_argument(
        "--absorbNuisancesInCovariance",
        type=str,
        default="",
        help="Regular expression to absorb some systematics in the data covariance rather than keep them as explicit nuisance parameters",
    )
    parser.add_argument(
        "--keepExplicitNuisances",
        type=str,
        default="",
        help="Regular expression to keep some systematics as explicit nuisance parameters, overriding --absorbNuisancesInCovariance.  Can be used to keep only some systs as nuisances while absorbing all the others into the covariance with '.*'",
    )
    parser.add_argument(
        "-n",
        "--baseName",
        type=str,
        nargs="+",
        default=["nominal"],
        help="Histogram name in the file (e.g., 'nominal')",
    )
    parser.add_argument(
        "--qcdProcessName",
        type=str,
        default=None,
        help="Name for QCD process (by default taken from datagroups object)",
    )
    # setting on the fit behaviour
    parser.add_argument(
        "--realData", action="store_true", help="Store real data in datacards"
    )
    parser.add_argument(
        "--fitvar", nargs="+", help="Variable to fit", default=["eta-pt-charge"]
    )
    parser.add_argument(
        "--rebin",
        type=parsing.str_to_list_or_int,
        nargs="*",
        default=[],
        help="""
        Rebin axis by this value (default, 1, does nothing); 
        use integer 'n' for merging 'n' bins;
        use comma separated list with new edges, use a leading space in case the bin edge is negative (e.g. " -2.4")
        """,
    )
    parser.add_argument(
        "--absval",
        type=int,
        nargs="*",
        default=[],
        help="Take absolute value of axis if 1 (default, 0, does nothing)",
    )
    parser.add_argument(
        "--axlim",
        type=parsing.str_to_complex_or_int,
        default=[],
        nargs="*",
        help="""
        Restrict axis to this range or these bins (assumes pairs of values by axis, with trailing axes optional).
        Arguments must be pure real or pure imaginary numbers to select bin indices or values, respectively.
        """,
    )
    parser.add_argument(
        "--rebinBeforeSelection",
        action="store_true",
        help="Rebin before the selection operation (e.g. before fake rate computation), default is after",
    )
    parser.add_argument(
        "--lumiUncertainty",
        type=float,
        help=r"Uncertainty for luminosity in excess to 1 (e.g. 1.012 means 1.2%); automatic by default",
        default=None,
    )
    parser.add_argument(
        "--lumiScale",
        type=float,
        nargs="+",
        default=[1.0],
        help="Rescale equivalent luminosity by this value (e.g. 10 means ten times more data and MC)",
    )
    parser.add_argument(
        "--lumiScaleVarianceLinearly",
        type=str,
        nargs="*",
        default=[],
        choices=["data", "mc"],
        help="""
            When using --lumiScale, scale variance linearly instead of quadratically, to pretend there is really more data or MC (can specify both as well). 
            Note that statistical fluctuations in histograms cannot be lifted, so this option can lead to spurious constraints of systematic uncertainties 
            when the argument of lumiScale is larger than unity, because bin-by-bin fluctuations will not be covered by the assumed uncertainty. 
            For data, this only has an effect for the data-driven estimate of the QCD multijet background through the uncertainty propagation from them data-MC subtraction.
            """,
    )
    parser.add_argument(
        "--procsWithoutLumiNorm",
        type=str,
        nargs="*",
        help="Do not apply luminosity norm uncertainty on these processes (Data, Fake, and QCD are already automatically excluded)",
        default=[],
    )
    parser.add_argument(
        "--noi",
        type=str,
        nargs="+",
        choices=[
            "wmass",
            "alphaS",
            "zmass",
            "sin2thetaW",
            "wwidth",
            "xsec",
            "massdiffW",
            "massdiffZ",
        ],
        default=["wmass"],
        help="Select which nuisance(s) of interest to fit. Default: (%default)s",
    )
    parser.add_argument(
        "--massDiffWVar",
        type=str,
        default=None,
        choices=[
            "charge",
            "cosThetaStarll",
            "eta-sign",
            "eta-range",
            "etaRegion",
            "etaRegionSign",
            "etaRegionRange",
        ],
        help="For use with --noi massDiffW, select the variable to define the different mass differences",
    )
    parser.add_argument(
        "--massDiffZVar",
        type=str,
        default=None,
        choices=[
            "charge",
            "cosThetaStarll",
            "eta-sign",
            "eta-range",
            "etaRegion",
            "etaRegionSign",
            "etaRegionRange",
        ],
        help="For use with --noi massDiffZ, select the variable to define the different mass differences",
    )
    parser.add_argument(
        "--fitMassDecorr",
        type=str,
        default=[],
        nargs="*",
        help="Decorrelate POI for given axes, fit multiple POIs for the different POIs",
    )
    parser.add_argument(
        "--decorrRebin",
        type=int,
        nargs="*",
        default=[],
        help="Rebin axis by this value (default, 1, does nothing)",
    )
    parser.add_argument(
        "--decorrAbsval",
        type=int,
        nargs="*",
        default=[],
        help="Take absolute value of axis if 1 (default, 0, does nothing)",
    )
    parser.add_argument(
        "--decorrAxlim",
        type=float,
        default=[],
        nargs="*",
        help="Restrict axis to this range (assumes pairs of values by axis, with trailing axes optional)",
    )
    parser.add_argument(
        "--decorrSystByVar",
        type=str,
        nargs="*",
        default=[],
        choices=[
            "run",
            "phi",
            "nRecoVtx",
            "prefire",
            "effi",
            "lumi",
            "fakenorm",
            "effisyst",
            "decornorm",
            "ptscale",
        ],
        help="""
        Customize what uncertainties to decorrelate by a specific variable (the first string passed to this option),
        to facilitate tests (note: effi is for both effStat and effSyst, while effisyst is only for effSyst).""",
    )
    parser.add_argument(
        "--residualEffiSFasUncertainty",
        type=int,
        default=0,
        help="When decorrelating by N run bins (specify N), add custom systematic uncertainty for residual efficiency scale factors.",
    )
    parser.add_argument(
        "--fitresult",
        type=str,
        nargs="+",
        default=None,
        help="""
        Use data and covariance matrix from fitresult (e.g. for making a theory fit). 
        Following the fitresult filename, a list of channels can be provided to only take the covariance across these channels (default is all channels).
        """,
    )
    parser.add_argument(
        "--fitresultResult",
        type=str,
        default="asimov",
        help="Use fit result from this file (e.g. for making a theory fit).",
    )
    parser.add_argument(
        "--fakerateAxes",
        nargs="+",
        help="Axes for the fakerate binning",
        default=["eta", "pt", "charge"],
    )
    parser.add_argument(
        "--fakeEstimation",
        type=str,
        help="Set the mode for the fake estimation",
        default="extended1D",
        choices=["mc", "closure", "simple", "extrapolate", "extended1D", "extended2D"],
    )
    parser.add_argument(
        "--forceGlobalScaleFakes",
        default=None,
        type=float,
        help="Scale the fakes  by this factor (overriding any custom one implemented in datagroups.py in the fakeSelector).",
    )
    parser.add_argument(
        "--fakeMCCorr",
        type=str,
        default=[None],
        nargs="*",
        choices=["none", "pt", "eta", "mt"],
        help="axes to apply nonclosure correction from QCD MC. Leave empty for inclusive correction, use'none' for no correction",
    )
    parser.add_argument(
        "--fakeSmoothingMode",
        type=str,
        default="full",
        choices=FakeSelectorSimpleABCD.smoothing_modes,
        help="Smoothing mode for fake estimate.",
    )
    parser.add_argument(
        "--fakeSmoothingOrder",
        type=int,
        default=3,
        help="Order of the polynomial for the smoothing of the application region or full prediction, depending on the smoothing mode",
    )
    parser.add_argument(
        "--fakeSmoothingPolynomial",
        type=str,
        default="chebyshev",
        choices=Regressor.polynomials,
        help="Type of polynomial for the smoothing of the application region or full prediction, depending on the smoothing mode",
    )
    parser.add_argument(
        "--ABCDedgesByAxis",
        type=str,
        nargs="+",
        default=[],
        help="""
        Edges for ABCD method given an axis. Syntax is --ABCDedgesByAxis 'nameX=x1,x2,x3' 'nameY=y1,y2,y3'
        Values after = are converted into float internally.
        Can specify only one axis or two (potentially more).
        """,
    )
    parser.add_argument(
        "--allowNegativeExpectation",
        action="store_true",
        help="Allow processes to have negative contributions",
    )
    # settings on the nuisances itself
    parser.add_argument(
        "--doStatOnly",
        action="store_true",
        default=False,
        help="Set up fit to get stat-only uncertainty",
    )
    parser.add_argument(
        "--doStatOnlyMasked",
        action="store_true",
        help="Masked channel with no systematic uncertainties",
    )
    parser.add_argument(
        "--noTheoryUnc",
        action="store_true",
        default=False,
        help="Set up fit without theory uncertainties",
    )
    parser.add_argument(
        "--addMCStatToCovariance",
        action="store_true",
        help="Add the MC statistical uncertainty to the data covariance (as an alternative to Barlow-Beeston lite)",
    )
    parser.add_argument(
        "--correlateSignalMCstat",
        action="store_true",
        help="Use explicit parameters for signal MC stat uncertainty. Introduces one nuisance parameter per reco bin.",
    )
    parser.add_argument(
        "--minnloScaleUnc",
        choices=[
            "byHelicityPt",
            "byHelicityPtCharge",
            "byHelicityCharge",
            "byPtCharge",
            "byPt",
            "byCharge",
            "integrated",
            "none",
        ],
        default="byHelicityPt",
        help="Decorrelation for QCDscale",
    )
    parser.add_argument(
        "--resumUnc",
        default="tnp",
        type=str,
        choices=["scale", "binned_scale", "tnp", "tnp_minnlo", "minnlo", "none"],
        help="Include SCETlib uncertainties",
    )
    parser.add_argument(
        "--noTransitionUnc",
        action="store_true",
        help="Do not include matching transition parameter variations.",
    )
    parser.add_argument(
        "--npUnc",
        default="Delta_Lambda",
        type=str,
        choices=combine_theory_helper.TheoryHelper.valid_np_models,
        help="Nonperturbative uncertainty model",
    )
    parser.add_argument(
        "--scaleTNP",
        default=1,
        type=float,
        help="Scale the TNP uncertainties by this factor",
    )
    parser.add_argument(
        "--scalePdf",
        default=-1.0,
        type=float,
        help="Scale the PDF hessian uncertainties by this factor (by default take the value in the pdfInfo map)",
    )
    parser.add_argument(
        "--pdfUncFromCorr",
        action="store_true",
        help="Take PDF uncertainty from correction hist (requires having run that correction)",
    )
    parser.add_argument(
        "--asUncFromUncorr",
        action="store_true",
        help="Take alpha_S uncertainty from uncorrected hist (by default it reads it from the correction hist, but requires having run that correction)",
    )
    parser.add_argument(
        "--scaleMinnloScale",
        default=1.0,
        type=float,
        help="Scale the minnlo qcd scale uncertainties by this factor",
    )
    parser.add_argument(
        "--symmetrizeTheoryUnc",
        default="quadratic",
        type=str,
        help="Symmetrization type for minnlo scale variations",
    )
    parser.add_argument(
        "--symmetrizePdfUnc",
        default="quadratic",
        type=str,
        help="Symmetrization type for PDF (and alphas) variations",
    )
    parser.add_argument(
        "--massVariation", type=float, default=100, help="Variation of boson mass"
    )
    parser.add_argument(
        "--ewUnc",
        type=str,
        nargs="*",
        default=["renesanceEW", "powhegFOEW"],
        help="Include EW uncertainty (other than pure ISR or FSR)",
        choices=[
            x
            for x in theory_corrections.valid_theory_corrections()
            if ("ew" in x or "EW" in x) and "ISR" not in x and "FSR" not in x
        ],
    )
    parser.add_argument(
        "--isrUnc",
        type=str,
        nargs="*",
        default=[
            "pythiaew_ISR",
        ],
        help="Include ISR uncertainty",
        choices=[
            x
            for x in theory_corrections.valid_theory_corrections()
            if "ew" in x and "ISR" in x
        ],
    )
    parser.add_argument(
        "--fsrUnc",
        type=str,
        nargs="*",
        default=["horaceqedew_FSR", "horacelophotosmecoffew_FSR"],
        help="Include FSR uncertainty",
        choices=[
            x
            for x in theory_corrections.valid_theory_corrections()
            if "ew" in x and "FSR" in x
        ],
    )
    parser.add_argument(
        "--skipSignalSystOnFakes",
        action="store_true",
        help="Do not propagate signal uncertainties on fakes, mainly for checks.",
    )
    parser.add_argument(
        "--noQCDscaleFakes",
        action="store_true",
        help="Do not apply QCd scale uncertainties on fakes, mainly for debugging",
    )
    parser.add_argument(
        "--addQCDMC",
        action="store_true",
        help="Include QCD MC when making datacards (otherwise by default it will always be excluded)",
    )
    parser.add_argument(
        "--muonScaleVariation",
        choices=["smearingWeights", "massWeights", "manualShift"],
        default="smearingWeights",
        help="the method with which the muon scale variation histograms are derived",
    )
    parser.add_argument(
        "--scaleMuonCorr",
        type=float,
        default=1.0,
        help="Scale up/down dummy muon scale uncertainty by this factor",
    )
    parser.add_argument(
        "--correlatedNonClosureNuisances",
        action="store_true",
        help="get systematics from histograms for the Z non-closure nuisances without decorrelation in eta and pt",
    )
    parser.add_argument(
        "--calibrationStatScaling",
        type=float,
        default=2.1,
        help="scaling of calibration statistical uncertainty",
    )
    parser.add_argument(
        "--resolutionStatScaling",
        type=float,
        default=10.0,
        help="scaling of resolution statistical uncertainty",
    )
    parser.add_argument(
        "--correlatedAdHocA",
        type=float,
        default=0.0,
        help="fully correlated ad-hoc uncertainty on b-field term A (in addition to Z pdg mass)",
    )
    parser.add_argument(
        "--correlatedAdHocM",
        type=float,
        default=0.0,
        help="fully correlated ad-hoc uncertainty on alignment term M",
    )
    parser.add_argument(
        "--noEfficiencyUnc",
        action="store_true",
        help="Skip efficiency uncertainty (useful for tests, because it's slow). Equivalent to --excludeNuisances '.*effSystTnP|.*effStatTnP' ",
    )
    parser.add_argument(
        "--effStatLumiScale",
        type=float,
        default=None,
        help="Rescale equivalent luminosity for efficiency stat uncertainty by this value (e.g. 10 means ten times more data from tag and probe)",
    )
    parser.add_argument(
        "--binnedScaleFactors",
        action="store_true",
        help="Use binned scale factors (different helpers and nuisances)",
    )
    parser.add_argument(
        "--isoEfficiencySmoothing",
        action="store_true",
        help="If isolation SF was derived from smooth efficiencies instead of direct smoothing",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Add normalization uncertainty fully constrained across processes",
    )
    parser.add_argument(
        "--logNormalWmunu",
        default=0,
        type=float,
        help=r"""Add normalization uncertainty for W signal. 
            If negative, treat as free floating with the absolute being the size of the variation (e.g. -1.01 means +/-1% of the nominal is varied). 
            If 0 nothing is added""",
    )
    parser.add_argument(
        "--logNormalWtaunu",
        default=0,
        type=float,
        help=r"""Add normalization uncertainty for W->tau,nu process. 
            If negative, treat as free floating with the absolute being the size of the variation (e.g. -1.01 means +/-1% of the nominal is varied). 
            If 0 nothing is added""",
    )
    parser.add_argument(
        "--logNormalFake",
        default=1.05,
        type=float,
        help="Specify normalization uncertainty for Fake background (for W analysis). If negative, treat as free floating, if 0 nothing is added",
    )
    # pseudodata
    parser.add_argument(
        "--pseudoData", type=str, nargs="+", help="Histograms to use as pseudodata"
    )
    parser.add_argument(
        "--pseudoDataAxes",
        type=str,
        nargs="+",
        default=[None],
        help="Variation axes to use as pseudodata for each of the histograms",
    )
    parser.add_argument(
        "--pseudoDataIdxs",
        type=str,
        nargs="+",
        default=[None],
        help="Variation indices to use as pseudodata for each of the histograms",
    )
    parser.add_argument(
        "--pseudoDataFile",
        type=str,
        help="Input file for pseudodata (if it should be read from a different file)",
        default=None,
    )
    parser.add_argument(
        "--pseudoDataFitInputFile",
        type=str,
        help="Input file for pseudodata (if it should be read from a fit input file)",
        default=None,
    )
    parser.add_argument(
        "--pseudoDataFitInputChannel",
        type=str,
        help="Input chnnel name for pseudodata (if it should be read from a fit input file)",
        default="ch0",
    )
    parser.add_argument(
        "--pseudoDataFitInputDownUp",
        type=str,
        help="DownUp variation for pseudodata (if it should be read from a fit input file)",
        default="Up",
    )
    parser.add_argument(
        "--pseudoDataProcsRegexp",
        type=str,
        default=".*",
        help="Regular expression for processes taken from pseudodata file (all other processes are automatically got from the nominal file). Data is excluded automatically as usual",
    )
    parser.add_argument(
        "--pseudoDataFakes",
        type=str,
        nargs="+",
        default=[],
        choices=[
            "truthMC",
            "closure",
            "simple",
            "extrapolate",
            "extended1D",
            "extended2D",
            "dataClosure",
            "mcClosure",
            "simple-binned",
            "extended1D-binned",
            "extended1D-fakerate",
        ],
        help="Pseudodata for fakes are using QCD MC (closure), or different estimation methods (simple, extended1D, extended2D)",
    )
    parser.add_argument(
        "--addTauToSignal",
        action="store_true",
        help="Events from the same process but from tau final states are added to the signal",
    )
    parser.add_argument(
        "--helicityFitTheoryUnc",
        action="store_true",
        help="Removes PDF and theory uncertainties on signal processes",
    )
    parser.add_argument(
        "--recoCharge",
        type=str,
        default=["plus", "minus"],
        nargs="+",
        choices=["plus", "minus"],
        help="Specify reco charge to use, default uses both. This is a workaround for unfolding/theory-agnostic fit when running a single reco charge, as gen bins with opposite gen charge have to be filtered out",
    )
    parser.add_argument(
        "--massConstraintModeW",
        choices=["automatic", "constrained", "unconstrained"],
        default="automatic",
        help="Whether W mass is constrained within PDG value and uncertainty or unconstrained in the fit",
    )
    parser.add_argument(
        "--massConstraintModeZ",
        choices=["automatic", "constrained", "unconstrained"],
        default="automatic",
        help="Whether Z mass is constrained within PDG value and uncertainty or unconstrained in the fit",
    )
    parser.add_argument(
        "--decorMassWidth",
        action="store_true",
        help="remove width variations from mass variations",
    )
    parser.add_argument(
        "--muRmuFPolVar",
        action="store_true",
        help="Use polynomial variations (like in theoryAgnosticPolVar) instead of binned variations for muR and muF (of course in setupRabbit these are still constrained nuisances)",
    )
    parser.add_argument(
        "--binByBinStatScaleForMW",
        type=float,
        default=1.26,
        help="scaling of bin by bin statistical uncertainty for W mass analysis",
    )
    parser.add_argument(
        "--binByBinStatScaleForDilepton",
        type=float,
        default=1.0,
        help="scaling of bin by bin statistical uncertainty for Z-dilepton analysis",
    )
    parser.add_argument(
        "--angularCoeffs",
        action="store_true",
        help="convert helicity cross sections to angular coefficients",
    )
    parser.add_argument(
        "--systematicType",
        choices=["log_normal", "normal"],
        default="log_normal",
        help="probability density for systematic variations",
    )
    parser.add_argument(
        "--select",
        nargs="+",
        dest="selection",
        type=str,
        default=None,
        help="Apply a selection to the histograms, if the axis exists."
        "This option can be applied to any of the axis, not necessarily one of the fitaxes, unlike --axlim."
        "Use complex numbers for axis value, integers for bin number."
        "e.g. --select 'ptll 0 10"
        "e.g. --select 'ptll 0j 10j",
    )
    parser.add_argument(
        "--noTheoryCorrsViaHelicities",
        action="store_true",
        help="Don't use theory correction histograms produced via smoothing through helicites.",
    )
    parser.add_argument(
        "--breitwignerWMassWeights",
        action="store_true",
        help="Use the Breit-Wigner mass wights for mW.",
    )
    parser = make_subparsers(parser)

    return parser


def setup(
    writer,
    args,
    inputFile,
    inputBaseName,
    inputLumiScale,
    fitvar,
    stat_only=False,
    genvar=None,
    channel="ch0",
    fitresult_data=None,
    unfolding_scalemap=None,
    base_group=None,
):
    isUnfolding = args.analysisMode == "unfolding"
    isTheoryAgnostic = args.analysisMode in [
        "theoryAgnosticNormVar",
        "theoryAgnosticPolVar",
    ]
    isTheoryAgnosticPolVar = args.analysisMode == "theoryAgnosticPolVar"
    isPoiAsNoi = (isUnfolding or isTheoryAgnostic) and args.poiAsNoi

    decorr_syst_var = None
    if len(args.decorrSystByVar) >= 2:
        decorr_syst_var = args.decorrSystByVar[0]
        if decorr_syst_var not in fitvar:
            raise ValueError(
                f"Inconsistent variable {decorr_syst_var} passed to --decorrSystByVar: fit variables are {fitvar}"
            )
    elif len(args.decorrSystByVar) == 1:
        raise ValueError(
            "Option --decorrSystByVar requires at least two arguments, the first one is the name of the decorrelation variable"
        )

    # NOTE: args.filterProcGroups and args.excludeProcGroups should in principle not be used together
    #       (because filtering is equivalent to exclude something), however the exclusion is also meant to skip
    #       processes which are defined in the original process dictionary but are not supposed to be (always) run on
    if args.addQCDMC or "QCD" in args.filterProcGroups:
        logger.warning("Adding QCD MC to list of processes for the fit setup")
    elif "QCD" not in args.excludeProcGroups:
        logger.warning(
            "Automatic removal of QCD MC from list of processes. Use --filterProcGroups 'QCD' or --addQCDMC to keep it"
        )
        args.excludeProcGroups.append("QCD")
    filterGroup = args.filterProcGroups if args.filterProcGroups else None
    excludeGroup = args.excludeProcGroups if args.excludeProcGroups else None

    logger.debug(f"Filtering these groups of processes: {args.filterProcGroups}")
    logger.debug(f"Excluding these groups of processes: {args.excludeProcGroups}")

    datagroups = Datagroups(
        inputFile,
        excludeGroups=excludeGroup,
        filterGroups=filterGroup,
        xnorm=any(
            inputBaseName.startswith(x) for x in ["gen", "xnorm", "prefsr", "postfsr"]
        ),
    )

    bsm_signals = []
    if args.addBSMProcess:
        combine_helpers.add_bsm_process(datagroups, args.addBSMProcess)
        bsm_signals.append(args.addBSMProcess)

    datagroups.fit_axes = fitvar
    datagroups.channel = channel

    if args.selection:
        for sel in args.selection:
            sel_ax, sel_lb, sel_ub = sel.split()
            sel_lb = parsing.str_to_complex_or_int(sel_lb)
            sel_ub = parsing.str_to_complex_or_int(sel_ub)
            datagroups.setGlobalAction(
                lambda h: (
                    h[{sel_ax: slice(sel_lb, sel_ub, hist.sum)}]
                    if sel_ax in h.axes.name
                    else h
                ),
            )

    if args.angularCoeffs:
        datagroups.setGlobalAction(
            lambda h: theory_tools.helicity_xsec_to_angular_coeffs(
                h, helicity_axis_name="helicitygen"
            )
        )

    if args.axlim or args.rebin or args.absval:
        datagroups.set_rebin_action(
            fitvar,
            args.axlim,
            args.rebin,
            args.absval,
            args.rebinBeforeSelection,
            rename=False,
        )

    wmass = datagroups.mode[0] == "w"
    wlike = "wlike" in datagroups.mode
    lowPU = "lowpu" in datagroups.mode
    # Detect lowpu dilepton
    dilepton = "dilepton" in datagroups.mode or any(
        x in ["ptll", "mll"] for x in fitvar
    )
    genfit = datagroups.mode == "vgen"

    if genfit:
        hasw = any("W" in x for x in args.filterProcGroups)
        hasz = any("Z" in x for x in args.filterProcGroups)
        if hasw and hasz:
            raise ValueError("Only W or Z processes are permitted in the gen fit")
        wmass = hasw

    massConstraintMode = args.massConstraintModeW if wmass else args.massConstraintModeZ

    if massConstraintMode == "automatic":
        constrainMass = (
            "xsec" in args.noi or (dilepton and not "mll" in fitvar) or genfit
        )
    else:
        constrainMass = True if massConstraintMode == "constrained" else False
    logger.debug(f"constrainMass = {constrainMass}")

    if base_group is None:
        if wmass:
            base_group = "Wenu" if datagroups.flavor == "e" else "Wmunu"
        else:
            base_group = "Zee" if datagroups.flavor == "ee" else "Zmumu"

    if args.addTauToSignal:
        # add tau signal processes to signal group
        datagroups.groups[base_group].addMembers(
            datagroups.groups[base_group.replace("mu", "tau")].members
        )
        datagroups.deleteGroup(base_group.replace("mu", "tau"))

    if "xsec" in args.noi:
        datagroups.unconstrainedProcesses.append(base_group)
    if args.logNormalFake < 0.0:
        datagroups.unconstrainedProcesses.append(datagroups.fakeName)

    if (
        lowPU
        and not datagroups.xnorm
        and ((args.fakeEstimation != "simple") or (args.fakeSmoothingMode != "binned"))
    ):
        logger.error(
            f"When running lowPU mode, fakeEstimation should be set to 'simple' and fakeSmoothingMode set to 'binned'."
        )

    if dilepton and "run" in fitvar:
        # in case fit is split by runs/ cumulated lumi
        # run axis only exists for data, add it for MC, and scale the MC according to the luminosity fractions
        run_edges = common.run_edges
        run_edges_lumi = common.run_edges_lumi
        lumis = np.diff(run_edges_lumi) / run_edges_lumi[-1]

        datagroups.setGlobalAction(
            lambda h: (
                h
                if "run" in h.axes.name
                else hh.scaleHist(
                    hh.addGenericAxis(
                        h,
                        hist.axis.Variable(
                            run_edges + 0.5, name="run", underflow=False, overflow=False
                        ),
                        add_trailing=False,
                    ),
                    lumis[:, *[np.newaxis for a in h.axes]],
                )
            )
        )

    if datagroups.xnorm:
        datagroups.select_xnorm_groups([base_group, *bsm_signals], inputBaseName)

    if datagroups.xnorm or isUnfolding or isPoiAsNoi:
        datagroups.setGenAxes(
            sum_gen_axes=[a for a in datagroups.gen_axes_names if a not in fitvar],
            base_group=base_group,
            histToReadAxes=args.unfoldingLevel if isUnfolding else inputBaseName,
        )

    if isPoiAsNoi:
        poi_axes = datagroups.gen_axes_names if genvar is None else genvar
        # remove specified gen axes from set of gen axes in datagroups so that those are integrated over
        datagroups.setGenAxes(
            sum_gen_axes=[a for a in datagroups.gen_axes_names if a not in poi_axes],
            base_group=base_group,
            histToReadAxes=args.unfoldingLevel if isUnfolding else inputBaseName,
        )
        # FIXME: temporary customization of signal and out-of-acceptance process names for theory agnostic with POI as NOI
        # There might be a better way to do it more homogeneously with the rest.
        if isTheoryAgnostic:
            constrainMass = False
            hasSeparateOutOfAcceptanceSignal = False
            for g in datagroups.groups.keys():
                logger.debug(f"{g}: {[m.name for m in datagroups.groups[g].members]}")
            # check if the out-of-acceptance signal process exists as an independent process
            if any(
                m.name.endswith("OOA") for m in datagroups.groups[base_group].members
            ):
                hasSeparateOutOfAcceptanceSignal = True
                if wmass:
                    # out of acceptance contribution
                    datagroups.copyGroup(
                        base_group,
                        f"{base_group}OOA",
                        member_filter=lambda x: x.name.endswith("OOA"),
                    )
                    datagroups.groups[base_group].deleteMembers(
                        [
                            m
                            for m in datagroups.groups[base_group].members
                            if m.name.endswith("OOA")
                        ]
                    )
                else:
                    # out of acceptance contribution
                    datagroups.copyGroup(
                        base_group,
                        f"{base_group}OOA",
                        member_filter=lambda x: x.name.endswith("OOA"),
                    )
                    datagroups.groups[base_group].deleteMembers(
                        [
                            m
                            for m in datagroups.groups[base_group].members
                            if m.name.endswith("OOA")
                        ]
                    )
            if (
                any(x.endswith("OOA") for x in args.excludeProcGroups)
                and hasSeparateOutOfAcceptanceSignal
            ):
                datagroups.deleteGroup(
                    f"{base_group}OOA"
                )  # remove out of acceptance signal
        else:
            constrainMass = True
    elif isUnfolding or isTheoryAgnostic:
        constrainMass = False if isTheoryAgnostic else True
        datagroups.sum_gen_axes = [
            n for n in datagroups.sum_gen_axes if n not in fitvar
        ]

        datagroups.defineSignalBinsUnfolding(
            base_group,
            base_group[0],
            member_filter=lambda x: not x.name.endswith("OOA"),
            fitvar=fitvar,
            histToReadAxes=args.unfoldingLevel,
            disable_flow_fit_axes=not (datagroups.xnorm and args.unfoldingWithFlow),
        )

        # out of acceptance contribution
        to_del = [
            m
            for m in datagroups.groups[base_group].members
            if not m.name.endswith("OOA")
        ]
        if len(datagroups.groups[base_group].members) == len(to_del):
            datagroups.deleteGroup(base_group)
        else:
            datagroups.groups[base_group].deleteMembers(to_del)

    if args.qcdProcessName:
        datagroups.fakeName = args.qcdProcessName

    abcdExplicitAxisEdges = {}
    if len(args.ABCDedgesByAxis):
        for item in args.ABCDedgesByAxis:
            ax_name, ax_edges = item.split("=")
            abcdExplicitAxisEdges[ax_name] = [float(x) for x in ax_edges.split(",")]

    if wmass and not datagroups.xnorm:
        datagroups.fakerate_axes = args.fakerateAxes
        histselector_kwargs = dict(
            mode=args.fakeEstimation,
            smoothing_mode=args.fakeSmoothingMode,
            smoothingOrderSpectrum=args.fakeSmoothingOrder,
            smoothingPolynomialSpectrum=args.fakeSmoothingPolynomial,
            mcCorr=args.fakeMCCorr,
            integrate_x="mt" not in fitvar,
            forceGlobalScaleFakes=args.forceGlobalScaleFakes,
            abcdExplicitAxisEdges=abcdExplicitAxisEdges,
        )
        datagroups.set_histselectors(
            datagroups.getNames(), inputBaseName, **histselector_kwargs
        )

    logger.debug(f"Making datacards with these processes: {datagroups.getProcesses()}")

    era = datagroups.args_from_metadata("era")

    datagroups.nominalName = inputBaseName
    label = "W" if wmass else "Z"
    datagroups.setCustomSystGroupMapping(
        {
            "theoryTNP": f".*resum.*|.*TNP.*|mass.*{label}.*",
            "resumTheory": f".*scetlib.*|.*resum.*|.*TNP.*|mass.*{label}.*",
            "allTheory": f".*scetlib.*|pdf.*|.*QCD.*|.*resum.*|.*TNP.*|mass.*{label}.*",
            "ptTheory": f".*QCD.*|.*resum.*|.*TNP.*|mass.*{label}.*",
        }
    )
    datagroups.setCustomSystForCard(
        args.excludeNuisances,
        args.keepNuisances,
        args.absorbNuisancesInCovariance,
        args.keepExplicitNuisances,
    )

    datagroups.lumiScale = inputLumiScale
    datagroups.lumiScaleVarianceLinearly = args.lumiScaleVarianceLinearly

    if not isTheoryAgnostic:
        logger.info(f"datagroups.allMCProcesses(): {datagroups.allMCProcesses()}")

    passSystToFakes = (
        wmass
        and not (datagroups.xnorm or args.skipSignalSystOnFakes)
        and datagroups.fakeName != "QCD"
        and (excludeGroup != None and datagroups.fakeName not in excludeGroup)
        and (filterGroup is None or datagroups.fakeName in filterGroup)
    )

    dibosonMatch = ["WW", "WZ", "ZZ"]
    WMatch = [
        "W"
    ]  # TODO: the name of out-of-acceptance might be changed at some point, maybe to WmunuOutAcc, so W will match it as well (and can exclude it using "OutAcc" if needed)
    ZMatch = ["Z"]
    signalMatch = WMatch if wmass else ZMatch
    nonSignalMatch = ZMatch if wmass else WMatch

    wlike_vetoValidation = wlike and datagroups.args_from_metadata("validateVetoSF")
    datagroups.addProcessGroup(
        "single_v_samples", startsWith=[*WMatch, *ZMatch], excludeMatch=dibosonMatch
    )
    # TODO consistently treat low mass drell yan as signal across full analysis
    datagroups.addProcessGroup(
        "z_samples",
        startsWith=ZMatch,
        excludeMatch=dibosonMatch,
    )
    if wmass or wlike_vetoValidation:
        datagroups.addProcessGroup(
            "Zveto_samples",
            startsWith=[*ZMatch, "DYlowMass"],
            excludeMatch=dibosonMatch,
        )
    if wmass:
        datagroups.addProcessGroup(
            "w_samples",
            startsWith=WMatch,
            excludeMatch=dibosonMatch,
        )
        datagroups.addProcessGroup("wtau_samples", startsWith=["Wtaunu"])
        if not datagroups.xnorm:
            datagroups.addProcessGroup(
                "single_v_nonsig_samples",
                startsWith=ZMatch,
                excludeMatch=dibosonMatch,
            )
    datagroups.addProcessGroup(
        "single_vmu_samples",
        startsWith=[*WMatch, *ZMatch],
        excludeMatch=[*dibosonMatch, "tau"],
    )
    datagroups.addProcessGroup(
        "signal_samples", startsWith=signalMatch, excludeMatch=[*dibosonMatch, "tau"]
    )
    datagroups.addProcessGroup(
        "signal_samples_inctau",
        startsWith=signalMatch,
        excludeMatch=dibosonMatch,
    )
    datagroups.addProcessGroup(
        "nonsignal_samples_inctau",
        startsWith=nonSignalMatch,
        excludeMatch=dibosonMatch,
    )
    datagroups.addProcessGroup(
        "MCnoQCD",
        excludeMatch=["QCD", "Data", "Fake"],
    )
    procsWithoutLumiNorm = ["QCD", "Data", "Fake"] + args.procsWithoutLumiNorm
    datagroups.addProcessGroup(
        "MCwithLumiNorm",
        excludeMatch=procsWithoutLumiNorm,
    )
    # FIXME/FOLLOWUP: the following groups may actually not exclude the OOA when it is not defined as an independent process with specific name
    datagroups.addProcessGroup(
        "signal_samples_noOutAcc",
        startsWith=signalMatch,
        excludeMatch=[*dibosonMatch, "tau", "OOA"],
    )
    datagroups.addProcessGroup(
        "nonsignal_samples_noOutAcc",
        startsWith=nonSignalMatch,
        excludeMatch=[*dibosonMatch, "tau", "OOA"],
    )
    datagroups.addProcessGroup(
        "signal_samples_inctau_noOutAcc",
        startsWith=signalMatch,
        excludeMatch=[*dibosonMatch, "OOA"],
    )
    datagroups.addProcessGroup(
        "nonsignal_samples_inctau_noOutAcc",
        startsWith=nonSignalMatch,
        excludeMatch=[*dibosonMatch, "OOA"],
    )

    if not (isTheoryAgnostic or isUnfolding):
        logger.info(f"All MC processes {datagroups.procGroups['MCnoQCD']}")
        logger.info(f"Single V samples: {datagroups.procGroups['single_v_samples']}")
        if wmass and not datagroups.xnorm:
            logger.info(
                f"Single V no signal samples: {datagroups.procGroups['single_v_nonsig_samples']}"
            )
        logger.info(f"Signal samples: {datagroups.procGroups['signal_samples']}")

    signal_samples_forMass = ["signal_samples_inctau"]

    datagroups.writer = writer

    for pseudodata in args.pseudoDataFakes:
        if pseudodata in ["closure", "truthMC"]:
            pseudodataGroups = Datagroups(
                args.pseudoDataFile if args.pseudoDataFile else inputFile,
                filterGroups=["QCD"],
            )
            pseudodataGroups.fakerate_axes = args.fakerateAxes
            pseudodataGroups.copyGroup("QCD", "QCDTruth")
            if pseudodata == "truthMC":
                pseudodataGroups.deleteGroup("QCD")
            pseudodataGroups.set_histselectors(
                pseudodataGroups.getNames(),
                inputBaseName,
                fake_processes=[
                    "QCD",
                ],
                **histselector_kwargs,
            )
        else:
            pseudodataGroups = Datagroups(
                args.pseudoDataFile if args.pseudoDataFile else inputFile,
                excludeGroups=excludeGroup,
                filterGroups=filterGroup,
            )
            pseudodataGroups.fakerate_axes = args.fakerateAxes

        datagroups.addPseudodataHistogramFakes(pseudodata, pseudodataGroups)
    if args.pseudoData and not datagroups.xnorm:
        if args.pseudoDataFitInputFile:
            indata = rabbit.debugdata.FitInputData(args.pseudoDataFitInputFile)
            debugdata = rabbit.debugdata.FitDebugData(indata)
            datagroups.addPseudodataHistogramsFitInput(
                debugdata,
                args.pseudoData,
                args.pseudoDataFitInputChannel,
                args.pseudoDataFitInputDownUp,
            )
        else:
            if args.pseudoDataFile:
                # FIXME: should make sure to apply the same customizations as for the nominal datagroups so far
                pseudodataGroups = Datagroups(
                    args.pseudoDataFile,
                    excludeGroups=excludeGroup,
                    filterGroups=filterGroup,
                )

                if wmass and not datagroups.xnorm:
                    pseudodataGroups.fakerate_axes = args.fakerateAxes
                    pseudodataGroups.set_histselectors(
                        pseudodataGroups.getNames(),
                        inputBaseName,
                        **histselector_kwargs,
                    )
            else:
                pseudodataGroups = datagroups

            datagroups.addPseudodataHistograms(
                pseudodataGroups,
                args.pseudoData,
                args.pseudoDataAxes,
                args.pseudoDataIdxs,
                args.pseudoDataProcsRegexp,
            )

    if args.correlateSignalMCstat and datagroups.xnorm:
        masked_flow_axes = (
            ["ptGen", "ptVGen"]
            if (datagroups.xnorm and isUnfolding and args.unfoldingWithFlow)
            else []
        )
        combine_helpers.add_nominal_with_correlated_BinByBinStat(
            datagroups,
            wmass,
            base_name=inputBaseName,
            masked=datagroups.xnorm and fitresult_data is None,
            masked_flow_axes=masked_flow_axes,
        )
    else:
        datagroups.addNominalHistograms(
            real_data=args.realData,
            bin_by_bin_stat_scale=(
                args.binByBinStatScaleForMW
                if wmass
                else args.binByBinStatScaleForDilepton if dilepton else 1.0
            ),
            fitresult_data=fitresult_data,
            masked=datagroups.xnorm and fitresult_data is None,
            masked_flow_axes=(
                ["ptGen", "ptVGen"]
                if (datagroups.xnorm and isUnfolding and args.unfoldingWithFlow)
                else []
            ),
        )

    if stat_only and isUnfolding and not isPoiAsNoi:
        # At least one nuisance parameter is needed to run combine impacts (e.g. needed for unfolding postprocessing chain)
        # TODO: fix Rabbit to run w/o nuisances
        datagroups.addNormSystematic(
            name="dummy",
            processes=["MCnoQCD"],
            norm=1.0001,
        )

    if args.normalize:
        name = f"normalization_{datagroups.channel}"
        datagroups.writer.add_norm_systematic(
            name,
            datagroups.predictedProcesses(),
            datagroups.channel,
            uncertainty=1.01,
            noi=False,
            constrained=False,
            groups="Normalization",
            add_to_data_covariance=datagroups.isAbsorbedNuisance(name),
        )

    decorwidth = args.decorMassWidth or ("wwidth" in args.noi)
    if not (stat_only and constrainMass) and args.massVariation != 0:
        massVariation = 2.1 if (not wmass and constrainMass) else args.massVariation
        massWeightName = (
            f"massWeight_widthdecor{label}" if decorwidth else f"massWeight{label}"
        )
        mass_info = dict(
            processes=signal_samples_forMass,
            group=f"massShift",
            noi=not constrainMass,
            skipEntries=massWeightNames(proc=label, exclude=massVariation),
            mirror=False,
            noConstraint=not constrainMass,
            systAxes=["massShift"],
            passToFakes=passSystToFakes,
        )

        if args.breitwignerWMassWeights and label == "W":
            preOpMap = {}
            for group in ["Wmunu", "Wtaunu"]:
                if group not in datagroups.groups.keys():
                    continue
                for member in datagroups.groups[group].members:
                    h_ref = datagroups.readHist(
                        datagroups.nominalName, member, massWeightName
                    )
                    preOpMap[member.name] = (
                        lambda h, h_ref=h_ref: syst_tools.correct_bw_xsec(h, h_ref)
                    )

            datagroups.addSystematic(
                histname=f"breitwigner_{massWeightName}",
                name=f"massWeight{label}",
                preOpMap=preOpMap,
                **mass_info,
            )
        else:
            if len(args.fitMassDecorr) == 0:
                datagroups.addSystematic(
                    massWeightName,
                    **mass_info,
                )
            else:
                suffix = "".join([a.capitalize() for a in args.fitMassDecorr])
                new_names = [f"{a}_decorr" for a in args.fitMassDecorr]
                datagroups.addSystematic(
                    histname=massWeightName,
                    processes=signal_samples_forMass,
                    name=f"massDecorr{suffix}{label}",
                    group=f"massDecorr{label}",
                    # systNameReplace=[("Shift",f"Diff{suffix}")],
                    skipEntries=[
                        (x, *[-1] * len(args.fitMassDecorr))
                        for x in massWeightNames(proc=label, exclude=args.massVariation)
                    ],
                    noi=not constrainMass,
                    noConstraint=not constrainMass,
                    mirror=False,
                    systAxes=["massShift", *new_names],
                    passToFakes=passSystToFakes,
                    # isPoiHistDecorr is a special flag to deal with how the massShift variations are internally formed
                    isPoiHistDecorr=len(args.fitMassDecorr),
                    actionRequiresNomi=True,
                    action=syst_tools.decorrelateByAxes,
                    actionArgs=dict(
                        axesToDecorrNames=args.fitMassDecorr,
                        newDecorrAxesNames=new_names,
                        axlim=args.decorrAxlim,
                        rebin=args.decorrRebin,
                        absval=args.decorrAbsval,
                    ),
                )

            if "massdiffW" in args.noi:
                suffix = "".join([a.capitalize() for a in args.massDiffWVar.split("-")])
                combine_helpers.add_mass_diff_variations(
                    datagroups,
                    args.massDiffWVa,
                    name=massWeightName,
                    processes=signal_samples_forMass,
                    constrain=constrainMass,
                    suffix=suffix,
                    label=label,
                    passSystToFakes=passSystToFakes,
                )

    # this appears within doStatOnly because technically these nuisances should be part of it
    if isPoiAsNoi:
        if isTheoryAgnostic:
            theoryAgnostic_helper = combine_theoryAgnostic_helper.TheoryAgnosticHelper(
                datagroups, externalArgs=args
            )
            if isTheoryAgnosticPolVar:
                theoryAgnostic_helper.configure_polVar(
                    label,
                    passSystToFakes,
                    hasSeparateOutOfAcceptanceSignal,
                )
            else:
                theoryAgnostic_helper.configure_normVar(
                    label,
                    passSystToFakes,
                    poi_axes,
                )
            theoryAgnostic_helper.add_theoryAgnostic_uncertainty()

        elif isUnfolding:
            combine_helpers.add_noi_unfolding_variations(
                datagroups,
                label,
                passSystToFakes,
                poi_axes,
                prior_norm=args.priorNormXsec,
                scale_norm=args.scaleNormXsecHistYields,
                gen_level=args.unfoldingLevel,
                fitresult=unfolding_scalemap,
                constrained=args.constrainNOIs,
            )

    if args.muRmuFPolVar and not isTheoryAgnosticPolVar:
        muRmuFPolVar_helper = combine_theoryAgnostic_helper.TheoryAgnosticHelper(
            datagroups, externalArgs=args
        )
        muRmuFPolVar_helper.configure_polVar(
            label,
            passSystToFakes,
            False,
        )
        muRmuFPolVar_helper.add_theoryAgnostic_uncertainty()

    if args.correlateSignalMCstat and datagroups.xnorm and args.fitresult is None:
        # use variations from reco histogram and apply them to xnorm
        source = ("nominal", f"{inputBaseName}_yieldsUnfolding_theory_weight")
        # need to find the reco variables that correspond to the reco fit, reco fit must be done with variables in same order as gen bins
        gen2reco = {
            "qGen": "charge",
            "ptGen": "pt",
            "absEtaGen": "eta",
            "qVGen": "charge",
            "ptVGen": "ptll",
            "absYVGen": "yll",
        }
        recovar = [gen2reco[v] for v in fitvar]

        combine_helpers.add_explicit_BinByBinStat(
            datagroups,
            recovar,
            samples="signal_samples",
            wmass=wmass,
            source=source,
            label=label,
        )

    if args.addBSMMixing is not None and wmass:
        # add BSM sample as variation on top of SM W
        combine_helpers.add_bsm_mixing(
            datagroups,
            inputBaseName,
            args.addBSMMixing[0],
            mixing=float(args.addBSMMixing[1]),
            passSystToFakes=passSystToFakes,
        )

    if ("wwidth" in args.noi and not wmass) or (
        not datagroups.xnorm and not stat_only and not args.noTheoryUnc
    ):
        # Experimental range
        # widthVars = (42, ['widthW2p043GeV', 'widthW2p127GeV']) if wmass else (2.3, ['widthZ2p4929GeV', 'widthZ2p4975GeV'])
        # Variation from EW fit (mostly driven by alphas unc.)
        datagroups.addSystematic(
            "widthWeightZ",
            name="WidthZ0p8MeV",
            processes=["single_v_nonsig_samples"] if wmass else signal_samples_forMass,
            skipEntries=widthWeightNames(proc="Z", exclude=(2.49333, 2.49493)),
            groups=["ZmassAndWidth" if wmass else "widthZ", "theory"],
            mirror=False,
            noi="wwidth" in args.noi if not wmass else False,
            noConstraint="wwidth" in args.noi if not wmass else False,
            systAxes=["width"],
            systNameReplace=[["2p49333GeV", "Down"], ["2p49493GeV", "Up"]],
            passToFakes=passSystToFakes,
        )

    if wmass and ("wwidth" in args.noi or (not stat_only and not args.noTheoryUnc)):
        width_info = dict(
            name="WidthW0p6MeV",
            processes=signal_samples_forMass,
            groups=["widthW", "theory"],
            mirror=False,
            noi="wwidth" in args.noi,
            noConstraint="wwidth" in args.noi,
            skipEntries=widthWeightNames(proc="W", exclude=(2.09053, 2.09173)),
            systAxes=["width"],
            systNameReplace=[["2p09053GeV", "Down"], ["2p09173GeV", "Up"]],
            passToFakes=passSystToFakes,
        )
        widthWeightName = f"widthWeight{label}"
        if args.breitwignerWMassWeights:
            preOpMap = {}
            for group in ["Wmunu", "Wtaunu"]:
                if group not in datagroups.groups.keys():
                    continue
                for member in datagroups.groups[group].members:
                    h_ref = datagroups.readHist(
                        datagroups.nominalName, member, widthWeightName
                    )
                    preOpMap[member.name] = (
                        lambda h, h_ref=h_ref: syst_tools.correct_bw_xsec(h, h_ref)
                    )
            datagroups.addSystematic(
                histname=f"breitwigner_{widthWeightName}",
                preOpMap=preOpMap,
                **width_info,
            )
        else:
            datagroups.addSystematic(
                widthWeightName,
                **width_info,
            )

    if "sin2thetaW" in args.noi or (not stat_only and not args.noTheoryUnc):
        datagroups.addSystematic(
            "sin2thetaWeightZ",
            name=f"Sin2thetaZ0p00003",
            processes=["z_samples"],
            action=lambda h: h[
                {"sin2theta": ["sin2thetaZ0p23151", "sin2thetaZ0p23157"]}
            ],
            group=f"sin2thetaZ",
            mirror=False,
            noi="sin2thetaW" in args.noi,
            noConstraint="sin2thetaW" in args.noi,
            systAxes=["sin2theta"],
            outNames=[f"sin2thetaZDown", f"sin2thetaZUp"],
            passToFakes=passSystToFakes,
        )

    if "alphaS" in args.noi or (not stat_only and not args.noTheoryUnc):
        theorySystSamples = ["signal_samples_inctau"]
        if wmass:
            if args.helicityFitTheoryUnc:
                theorySystSamples = ["wtau_samples"]
            theorySystSamples.append("single_v_nonsig_samples")
        elif wlike:
            if args.helicityFitTheoryUnc:
                theorySystSamples = []
        if datagroups.xnorm:
            theorySystSamples = ["signal_samples"]

        theory_helper = combine_theory_helper.TheoryHelper(
            label, datagroups, args, hasNonsigSamples=(wmass and not datagroups.xnorm)
        )
        theory_helper.configure(
            resumUnc=args.resumUnc,
            transitionUnc=not args.noTransitionUnc,
            propagate_to_fakes=passSystToFakes
            and not args.noQCDscaleFakes
            and not datagroups.xnorm,
            np_model=args.npUnc,
            tnp_scale=args.scaleTNP,
            mirror_tnp=False,
            pdf_from_corr=args.pdfUncFromCorr,
            as_from_corr=not args.asUncFromUncorr,
            scale_pdf_unc=args.scalePdf,
            samples=theorySystSamples,
            minnlo_unc=args.minnloScaleUnc,
            minnlo_scale=args.scaleMinnloScale,
            from_hels=not args.noTheoryCorrsViaHelicities,
            theory_symmetrize=args.symmetrizeTheoryUnc,
            pdf_symmetrize=args.symmetrizePdfUnc,
        )

        theory_helper.add_pdf_alphas_variation(
            noi="alphaS" in args.noi,
        )

        if not stat_only and not args.noTheoryUnc:
            theory_helper.add_all_theory_unc(
                helicity_fit_unc=args.helicityFitTheoryUnc,
            )

    if stat_only:
        # print a card with only mass weights
        logger.info(
            "Using option --doStatOnly: the card was created with only mass nuisance parameter"
        )
        return datagroups

    if not args.noTheoryUnc:
        if wmass and not datagroups.xnorm:
            if args.massConstraintModeZ == "automatic":
                constrainMassZ = True
            else:
                constrainMassZ = (
                    True if args.massConstraintModeZ == "constrained" else False
                )

            massVariationZ = 2.1 if constrainMassZ else args.massVariation

            datagroups.addSystematic(
                f"massWeightZ",
                processes=["single_v_nonsig_samples"],
                groups=["ZmassAndWidth", "theory"],
                skipEntries=massWeightNames(proc="Z", exclude=massVariationZ),
                mirror=False,
                noi=not constrainMassZ,
                noConstraint=not constrainMassZ,
                systAxes=["massShift"],
                passToFakes=passSystToFakes,
            )

            if "massDiffZ" in args.noi:
                suffix = "".join([a.capitalize() for a in args.massDiffZVar.split("-")])
                combine_helpers.add_mass_diff_variations(
                    datagroups,
                    args.massDiffZVar,
                    name=f"{massWeightName}Z",
                    processes=["single_v_nonsig_samples"],
                    constrain=constrainMass,
                    suffix=suffix,
                    label="Z",
                    passSystToFakes=passSystToFakes,
                )

        if inputBaseName != "prefsr":
            # make prefsr ane EW free definition
            combine_helpers.add_electroweak_uncertainty(
                datagroups,
                [*args.ewUnc, *args.fsrUnc, *args.isrUnc],
                samples="single_v_samples",
                flavor=datagroups.flavor,
                passSystToFakes=passSystToFakes,
            )

    if datagroups.xnorm or genfit:
        return datagroups

    # Below: experimental uncertainties

    if wmass:
        # mirror hist in linear scale, this was done in the old definition of luminosity uncertainty from a histogram
        if "lumi" in args.decorrSystByVar and decorr_syst_var in fitvar:
            datagroups.addSystematic(
                name="lumi",
                processes=["MCwithLumiNorm"],
                groups=[f"luminosity", "experiment", "expNoCalib"],
                passToFakes=passSystToFakes,
                baseName="lumi_",
                systAxes=[f"{decorr_syst_var}_", "downUpVar"],
                labelsByAxis=[decorr_syst_var, "downUpVar"],
                actionRequiresNomi=True,
                action=syst_tools.decorrelateByAxes,
                actionArgs=dict(
                    axesToDecorrNames=[decorr_syst_var],
                    newDecorrAxesNames=[f"{decorr_syst_var}_"],
                ),
                preOp=scale_hist_up_down,
                preOpArgs={
                    "scale": (
                        datagroups.lumi_uncertainty
                        if args.lumiUncertainty is None
                        else args.lumiUncertainty
                    )
                },
            )
        else:
            datagroups.addSystematic(
                name="lumi",
                processes=["MCwithLumiNorm"],
                groups=[f"luminosity", "experiment", "expNoCalib"],
                passToFakes=passSystToFakes,
                outNames=["lumiDown", "lumiUp"],
                systAxes=["downUpVar"],
                labelsByAxis=["downUpVar"],
                preOp=scale_hist_up_down,
                preOpArgs={
                    "scale": (
                        datagroups.lumi_uncertainty
                        if args.lumiUncertainty is None
                        else args.lumiUncertainty
                    )
                },
            )
    else:
        datagroups.addNormSystematic(
            name="lumi",
            processes=["MCwithLumiNorm"],
            groups=[f"luminosity", "experiment", "expNoCalib"],
            passToFakes=passSystToFakes,
            norm=(
                datagroups.lumi_uncertainty
                if args.lumiUncertainty is None
                else args.lumiUncertainty
            ),
        )

    # add norm variations for decorrelated variable bins on each process
    if "decornorm" in args.decorrSystByVar and decorr_syst_var in fitvar:
        datagroups.addSystematic(
            name=f"{decorr_syst_var}DecorrNorm",
            processes=["MCnoQCD"],
            groups=[
                f"{decorr_syst_var}DecorrNorm",
                "experiment",
                "expNoLumi",
                "expNoCalib",
            ],
            passToFakes=passSystToFakes,
            baseName=f"{decorr_syst_var}DecorrNorm_",
            systAxes=[f"{decorr_syst_var}_", "downUpVar"],
            labelsByAxis=[decorr_syst_var, "downUpVar"],
            actionRequiresNomi=True,
            action=syst_tools.decorrelateByAxes,
            actionArgs=dict(
                axesToDecorrNames=[decorr_syst_var],
                newDecorrAxesNames=[f"{decorr_syst_var}_"],
            ),
            preOp=scale_hist_up_down,
            preOpArgs={"scale": 1.05},
        )

    # lowPU does not include PhotonInduced as a process. skip it:
    if not lowPU and "PhotonInduced" in datagroups.groups:
        datagroups.addNormSystematic(
            name="CMS_PhotonInduced",
            processes=["PhotonInduced"],
            groups=[f"CMS_background", "experiment", "expNoLumi", "expNoCalib"],
            passToFakes=passSystToFakes,
            norm=2.0,
        )
    if wmass:
        if args.logNormalWmunu != 0:
            datagroups.addNormSystematic(
                name="CMS_Wmunu",
                processes=["Wmunu"],
                groups=[
                    f"CMS_background",
                    *(
                        ["experiment", "expNoLumi", "expNoCalib"]
                        if args.logNormalWmunu > 0
                        else []
                    ),
                ],
                passToFakes=passSystToFakes,
                noi=args.logNormalWmunu < 0,
                noConstraint=args.logNormalWmunu < 0,
                norm=abs(args.logNormalWmunu),
            )
        if args.logNormalWtaunu != 0:
            datagroups.addNormSystematic(
                name="CMS_Wtaunu",
                processes=["Wtaunu"],
                groups=[
                    f"CMS_background",
                    *(
                        ["experiment", "expNoLumi", "expNoCalib"]
                        if args.logNormalWmunu > 0
                        else []
                    ),
                ],
                passToFakes=passSystToFakes,
                noi=args.logNormalWtaunu < 0,
                noConstraint=args.logNormalWtaunu < 0,
                norm=abs(args.logNormalWtaunu),
            )

        if args.logNormalFake > 0.0 and datagroups.fakeName in datagroups.groups.keys():
            if "fakenorm" in args.decorrSystByVar and decorr_syst_var in fitvar:
                datagroups.addSystematic(
                    name=f"CMS_{datagroups.fakeName}",
                    processes=[datagroups.fakeName],
                    groups=["Fake", "experiment", "expNoLumi", "expNoCalib"],
                    passToFakes=False,
                    baseName=f"CMS_{datagroups.fakeName}_",
                    systAxes=[f"{decorr_syst_var}_", "downUpVar"],
                    labelsByAxis=[decorr_syst_var, "downUpVar"],
                    actionRequiresNomi=True,
                    action=syst_tools.decorrelateByAxes,
                    actionArgs=dict(
                        axesToDecorrNames=[decorr_syst_var],
                        newDecorrAxesNames=[f"{decorr_syst_var}_"],
                    ),
                    preOp=scale_hist_up_down,
                    preOpArgs={"scale": args.logNormalFake},
                )
            else:
                datagroups.addNormSystematic(
                    name=f"CMS_{datagroups.fakeName}",
                    processes=[datagroups.fakeName],
                    groups=["Fake", "experiment", "expNoLumi", "expNoCalib"],
                    passToFakes=False,
                    norm=args.logNormalFake,
                )

        if "Top" in datagroups.groups:
            datagroups.addNormSystematic(
                name="CMS_Top",
                processes=["Top"],
                groups=[f"CMS_background", "experiment", "expNoLumi", "expNoCalib"],
                passToFakes=passSystToFakes,
                norm=1.06,
            )
        if "Diboson" in datagroups.groups:
            datagroups.addNormSystematic(
                name="CMS_VV",
                processes=["Diboson"],
                groups=[f"CMS_background", "experiment", "expNoLumi", "expNoCalib"],
                passToFakes=passSystToFakes,
                norm=1.16,
            )
    elif "Other" in datagroups.groups:
        datagroups.addNormSystematic(
            name="CMS_background",
            processes=["Other"],
            groups=[f"CMS_background", "experiment", "expNoLumi", "expNoCalib"],
            norm=1.15,
        )

    if (
        (datagroups.fakeName != "QCD" and args.qcdProcessName != "QCD")
        and datagroups.fakeName in datagroups.groups.keys()
        and not datagroups.xnorm
        and (
            args.fakeSmoothingMode != "binned"
            or (args.fakeEstimation in ["extrapolate"] and "mt" in fitvar)
        )
    ):

        fakeselector = datagroups.groups[datagroups.fakeName].histselector

        syst_axes = (
            [f"_{x}" for x in args.fakerateAxes if x != "pt"]
            if (
                args.fakeSmoothingMode != "binned"
                or args.fakeEstimation not in ["extrapolate"]
            )
            else [f"_{x}" for x in args.fakerateAxes]
        )
        info = dict(
            histname=inputBaseName,
            processes=datagroups.fakeName,
            noConstraint=False,
            mirror=False,
            scale=1,
            applySelection=False,  # don't apply selection, all regions will be needed for the action
            action=fakeselector.get_hist,
            systAxes=syst_axes + ["_param", "downUpVar"],
        )
        if args.fakeSmoothingMode in ["hybrid", "full"]:
            subgroup = f"{datagroups.fakeName}Smoothing"
            datagroups.addSystematic(
                **info,
                name=subgroup,
                baseName=subgroup,
                groups=[subgroup, "Fake", "experiment", "expNoLumi", "expNoCalib"],
                actionArgs=dict(variations_smoothing=True),
            )

        if args.fakeSmoothingMode in ["fakerate", "hybrid"]:
            subgroup = f"{datagroups.fakeName}Rate"
            datagroups.addSystematic(
                **info,
                name=subgroup,
                baseName=subgroup,
                groups=[subgroup, "Fake", "experiment", "expNoLumi", "expNoCalib"],
                actionArgs=dict(variations_frf=True),
            )

        if (
            args.fakeEstimation
            in [
                "extended2D",
            ]
            and args.fakeSmoothingMode != "full"
        ):
            subgroup = f"{datagroups.fakeName}Shape"
            datagroups.addSystematic(
                **info,
                name=subgroup,
                baseName=subgroup,
                groups=[subgroup, "Fake", "experiment", "expNoLumi", "expNoCalib"],
                actionArgs=dict(variations_scf=True),
            )

        if args.fakeSmoothingMode in ["hybrid", "full"] and args.fakeSmoothingOrder > 0:
            # add systematic of explicit parameter variation
            fakeSmoothingOrder = args.fakeSmoothingOrder

            def fake_nonclosure(
                h,
                axesToDecorrNames,
                param_idx=1,
                variation_size=0.5,
                normalize=False,
                *args,
                **kwargs,
            ):
                # apply variation by adding parameter value (assumes log space, e.g. in full smoothing)
                fakeselector.spectrum_regressor.external_params = np.zeros(
                    fakeSmoothingOrder + 1
                )
                fakeselector.spectrum_regressor.external_params[param_idx] = (
                    variation_size
                )
                hvar = fakeselector.get_hist(h, *args, **kwargs)
                # reset external parameters
                fakeselector.spectrum_regressor.external_params = None

                hnom = fakeselector.get_hist(h, *args, **kwargs)

                if normalize:
                    # normalize variation histogram to have the same integral as nominal
                    hScale = hh.divideHists(
                        hnom[{"pt": hist.sum}], hvar[{"pt": hist.sum}]
                    )
                    hvar = hh.multiplyHists(hvar, hScale)

                if len(axesToDecorrNames) == 0:
                    # inclusive
                    hvar = hist.Hist(
                        *hvar.axes,
                        hist.axis.Integer(
                            0, 1, name="var", underflow=False, overflow=False
                        ),
                        storage=hist.storage.Double(),
                        data=hvar.values(flow=True)[..., np.newaxis],
                    )
                else:
                    hvar = syst_tools.decorrelateByAxes(hvar, hnom, axesToDecorrNames)

                return hvar

            for axesToDecorrNames in [
                [],
            ]:
                for idx, mag in [
                    (1, 0.1),
                    (2, 0.1),
                ]:
                    subgroup = f"{datagroups.fakeName}Param{idx}"
                    datagroups.addSystematic(
                        inputBaseName,
                        groups=[
                            subgroup,
                            "Fake",
                            "experiment",
                            "expNoLumi",
                            "expNoCalib",
                        ],
                        name=subgroup
                        + (
                            f"_{'_'.join(axesToDecorrNames)}"
                            if len(axesToDecorrNames)
                            else ""
                        ),
                        baseName=subgroup,
                        processes=datagroups.fakeName,
                        noConstraint=False,
                        mirror=True,
                        scale=1,
                        applySelection=False,  # don't apply selection, external parameters need to be added
                        action=fake_nonclosure,
                        actionArgs=dict(
                            axesToDecorrNames=axesToDecorrNames,
                            param_idx=idx,
                            variation_size=mag,
                        ),
                        systAxes=(
                            ["var"]
                            if len(axesToDecorrNames) == 0
                            else [f"{n}_decorr" for n in axesToDecorrNames]
                        ),
                    )

    if not args.noEfficiencyUnc:

        if not lowPU:

            chargeDependentSteps = common.muonEfficiency_chargeDependentSteps
            effTypesNoIso = ["reco", "tracking", "idip", "trigger"]
            effStatTypes = [x for x in effTypesNoIso]
            if args.binnedScaleFactors or not args.isoEfficiencySmoothing:
                effStatTypes.extend(["iso"])
            else:
                effStatTypes.extend(["iso_effData", "iso_effMC"])
            allEffTnP = [f"effStatTnP_sf_{eff}" for eff in effStatTypes] + [
                "effSystTnP"
            ]
            for name in allEffTnP:
                if "Syst" in name:
                    axes = ["reco-tracking-idip-trigger-iso", "n_syst_variations"]
                    axlabels = ["WPSYST", "_etaDecorr"]
                    nameReplace = [
                        ("WPSYST0", "reco"),
                        ("WPSYST1", "tracking"),
                        ("WPSYST2", "idip"),
                        ("WPSYST3", "trigger"),
                        ("WPSYST4", "iso"),
                        ("effSystTnP", "effSyst"),
                        ("etaDecorr0", "fullyCorr"),
                    ]
                    scale = 1
                    mirror = True
                    groupName = "muon_eff_syst"
                    splitGroupDict = {
                        f"{groupName}_{x}": f".*effSyst.*{x}"
                        for x in list(effTypesNoIso + ["iso"])
                    }
                    actionSF = None
                    effActionArgs = {}
                    if (
                        any(x in args.decorrSystByVar for x in ["effi", "effisyst"])
                        and decorr_syst_var in fitvar
                    ):
                        axes = [
                            "reco-tracking-idip-trigger-iso",
                            "n_syst_variations",
                            f"{decorr_syst_var}_",
                        ]
                        axlabels = ["WPSYST", "_etaDecorr", decorr_syst_var]
                        actionSF = syst_tools.decorrelateByAxes
                        effActionArgs = dict(
                            axesToDecorrNames=[decorr_syst_var],
                            newDecorrAxesNames=[f"{decorr_syst_var}_"],
                        )
                else:
                    nameReplace = (
                        []
                        if any(x in name for x in chargeDependentSteps)
                        else [("q0", "qall")]
                    )  # for iso change the tag id with another sensible label
                    mirror = True
                    if args.binnedScaleFactors:
                        axes = ["SF eta", "nPtBins", "SF charge"]
                    else:
                        axes = ["SF eta", "nPtEigenBins", "SF charge"]
                    axlabels = ["eta", "pt", "q"]
                    nameReplace = nameReplace + [("effStatTnP_sf_", "effStat_")]
                    scale = 1
                    groupName = "muon_eff_stat"
                    splitGroupDict = {
                        f"{groupName}_{x}": f".*effStat.*{x}" for x in effStatTypes
                    }
                    actionSF = None
                    effActionArgs = {}
                    if "effi" in args.decorrSystByVar and decorr_syst_var in fitvar:
                        axes = [
                            "SF eta",
                            "nPtEigenBins",
                            "SF charge",
                            f"{decorr_syst_var}_",
                        ]
                        axlabels = ["eta", "pt", "q", decorr_syst_var]
                        actionSF = syst_tools.decorrelateByAxes
                        effActionArgs = dict(
                            axesToDecorrNames=[decorr_syst_var],
                            newDecorrAxesNames=[f"{decorr_syst_var}_"],
                        )
                if args.effStatLumiScale and "Syst" not in name:
                    scale /= math.sqrt(args.effStatLumiScale)

                datagroups.addSystematic(
                    name,
                    mirror=mirror,
                    groups=[
                        groupName,
                        "muon_eff_all",
                        "experiment",
                        "expNoLumi",
                        "expNoCalib",
                    ],
                    splitGroup=splitGroupDict,
                    systAxes=axes,
                    labelsByAxis=axlabels,
                    actionRequiresNomi=True,
                    action=actionSF,
                    actionArgs=effActionArgs,
                    baseName=name + "_",
                    processes=["MCnoQCD"],
                    passToFakes=passSystToFakes,
                    systNameReplace=nameReplace,
                    scale=scale,
                )
                # now add other systematics if present
                if name == "effSystTnP":
                    for es in common.muonEfficiency_altBkgSyst_effSteps:
                        datagroups.addSystematic(
                            f"effSystTnP_altBkg_{es}",
                            mirror=mirror,
                            groups=[
                                f"muon_eff_syst_{es}_altBkg",
                                groupName,
                                "muon_eff_all",
                                "experiment",
                                "expNoLumi",
                                "expNoCalib",
                            ],
                            systAxes=["n_syst_variations"],
                            labelsByAxis=[f"{es}_altBkg_etaDecorr"],
                            baseName=name + "_",
                            processes=["MCnoQCD"],
                            passToFakes=passSystToFakes,
                            systNameReplace=[
                                ("effSystTnP", "effSyst"),
                                ("etaDecorr0", "fullyCorr"),
                            ],
                            scale=scale,
                        )
            if (
                wmass and not datagroups.args_from_metadata("noVetoSF")
            ) or wlike_vetoValidation:
                useGlobalOrTrackerVeto = datagroups.args_from_metadata(
                    "useGlobalOrTrackerVeto"
                )
                useRefinedVeto = datagroups.args_from_metadata("useRefinedVeto")
                allEffTnP_veto = ["effStatTnP_veto_sf", "effSystTnP_veto"]
                for name in allEffTnP_veto:
                    if "Syst" in name:
                        if useGlobalOrTrackerVeto:
                            axes = [
                                "veto_reco-veto_tracking-veto_idip-veto_trackerreco-veto_trackertracking",
                                "n_syst_variations",
                            ]
                        else:
                            if useRefinedVeto:
                                axes = [
                                    "vetoreco-vetotracking-vetoidip",
                                    "n_syst_variations",
                                ]
                            else:
                                axes = [
                                    "veto_reco-veto_tracking-veto_idip",
                                    "n_syst_variations",
                                ]
                        axlabels = ["WPSYST", "_etaDecorr"]
                        if useGlobalOrTrackerVeto:
                            nameReplace = [
                                ("WPSYST0", "reco"),
                                ("WPSYST1", "tracking"),
                                ("WPSYST2", "idip"),
                                ("WPSYST3", "trackerreco"),
                                ("WPSYST4", "trackertracking"),
                                ("effSystTnP_veto", "effSyst_veto"),
                                ("etaDecorr0", "fullyCorr"),
                            ]
                        else:
                            nameReplace = [
                                ("WPSYST0", "reco"),
                                ("WPSYST1", "tracking"),
                                ("WPSYST2", "idip"),
                                ("effSystTnP_veto", "effSyst_veto"),
                                ("etaDecorr0", "fullyCorr"),
                            ]
                        scale = 1.0
                        mirror = True
                        groupName = "muon_eff_syst_veto"
                        if useGlobalOrTrackerVeto:
                            splitGroupDict = {
                                f"{groupName}{x}": f".*effSyst_veto.*{x}"
                                for x in list(
                                    [
                                        "reco",
                                        "tracking",
                                        "idip",
                                        "trackerreco",
                                        "trackertracking",
                                    ]
                                )
                            }
                        else:
                            splitGroupDict = {
                                f"{groupName}{x}": f".*effSyst_veto.*{x}"
                                for x in list(["reco", "tracking", "idip"])
                            }
                    else:
                        nameReplace = []
                        mirror = True
                        if useRefinedVeto:
                            axes = [
                                "vetoreco-vetotracking-vetoidip",
                                "SF eta",
                                "nPtEigenBins",
                                "SF charge",
                            ]
                            axlabels = ["WPSTEP", "eta", "pt", "q"]
                            nameReplace = nameReplace + [
                                ("effStatTnP_veto_sf_", "effStat_veto_"),
                                ("WPSTEP0", "reco"),
                                ("WPSTEP1", "tracking"),
                                ("WPSTEP2", "idip"),
                            ]
                        else:
                            axes = ["SF eta", "nPtEigenBins", "SF charge"]
                            axlabels = ["eta", "pt", "q"]
                            nameReplace = nameReplace + [
                                ("effStatTnP_veto_sf_", "effStat_veto_")
                            ]
                        scale = 1.0
                        groupName = "muon_eff_stat_veto"
                        splitGroupDict = {
                            f"{groupName}{x}": f".*effStat_veto.*{x}"
                            for x in list(["reco", "tracking", "idip"])
                        }
                    if args.effStatLumiScale and "Syst" not in name:
                        scale /= math.sqrt(args.effStatLumiScale)

                    datagroups.addSystematic(
                        name,
                        mirror=mirror,
                        groups=[
                            groupName,
                            "muon_eff_all",
                            "experiment",
                            "expNoLumi",
                            "expNoCalib",
                        ],
                        splitGroup=splitGroupDict,
                        systAxes=axes,
                        labelsByAxis=axlabels,
                        baseName=name + "_",
                        processes=["Zveto_samples"],
                        passToFakes=passSystToFakes if wmass else False,
                        systNameReplace=nameReplace,
                        scale=scale,
                    )

        else:
            if datagroups.flavor in ["mu", "mumu"]:
                lepEffs = [
                    "muSF_HLT_DATA_stat",
                    "muSF_HLT_DATA_syst",
                    "muSF_HLT_MC_stat",
                    "muSF_HLT_MC_syst",
                    "muSF_ISO_stat",
                    "muSF_ISO_DATA_syst",
                    "muSF_ISO_MC_syst",
                    "muSF_IDIP_stat",
                    "muSF_IDIP_DATA_syst",
                    "muSF_IDIP_MC_syst",
                ]
            else:
                lepEffs = []  # ["elSF_HLT_syst", "elSF_IDISO_stat"]

            for lepEff in lepEffs:
                datagroups.addSystematic(
                    lepEff,
                    processes=datagroups.allMCProcesses(),
                    mirror=True,
                    groups=["CMS_lepton_eff", "experiment", "expNoLumi", "expNoCalib"],
                    baseName=lepEff,
                    systAxes=["tensor_axis_0"],
                    labelsByAxis=[""],
                )

    if (wmass or wlike) and datagroups.args_from_metadata("recoilUnc"):
        combine_helpers.add_recoil_uncertainty(
            datagroups,
            ["signal_samples"],
            passSystToFakes=passSystToFakes,
            flavor=datagroups.flavor if datagroups.flavor else "mu",
            pu_type="lowPU" if lowPU else "highPU",
        )

    if lowPU:
        if datagroups.flavor in ["e", "ee"] and False:
            # disable, prefiring for muons currently broken? (fit fails)
            datagroups.addSystematic(
                "prefireCorr",
                processes=datagroups.allMCProcesses(),
                mirror=False,
                groups=["CMS_prefire17", "experiment", "expNoLumi", "expNoCalib"],
                baseName="CMS_prefire17",
                systAxes=["downUpVar"],
                labelsByAxis=["downUpVar"],
            )

        return datagroups

    # add dedicated uncertainties from residual corrections read from a file
    # implemented by modifying the nominal histogram
    if decorr_syst_var in fitvar and args.residualEffiSFasUncertainty > 0:
        ## action to apply corrections and move from nominal to alternate histogram in input
        corr_era = "2016" if era == "2016PostVFP" else era
        corr_input_path = f"{common.data_dir}/muonSF/corrections/{corr_era}/"
        preOpCorrAction = scale_hist_up_down_corr_from_file
        preOpCorrActionArgs = dict(
            corr_file=f"{corr_input_path}/dataMC_ZmumuEffCorr_eta_{args.residualEffiSFasUncertainty}{decorr_syst_var}Bins.pkl.lz4",
            corr_hist=f"dataMC_ZmumuEffCorr_eta_{decorr_syst_var}Bin",
        )
        #
        logger.warning(
            f"Adding uncertainty for residual efficiency corrections decorrelated by {decorr_syst_var} and eta"
        )
        #
        datagroups.addSystematic(
            name="residualEffiSF",
            processes=["MCnoQCD"],
            groups=["residualEffiSF", "experiment", "expNoLumi", "expNoCalib"],
            baseName="residualEffiSF_",
            systAxes=["eta_", f"{decorr_syst_var}_", "downUpVar"],
            labelsByAxis=["eta", decorr_syst_var, "downUpVar"],
            passToFakes=passSystToFakes,
            preOp=preOpCorrAction,
            preOpArgs=preOpCorrActionArgs,
            action=syst_tools.decorrelateByAxes,
            actionArgs=dict(
                axesToDecorrNames=["eta", decorr_syst_var],
                newDecorrAxesNames=["eta_", f"{decorr_syst_var}_"],
            ),
            actionRequiresNomi=True,
        )
        #
        logger.warning(
            f"Adding uncertainty for residual efficiency corrections decorrelated by {decorr_syst_var} inclusive in eta"
        )
        #
        datagroups.addSystematic(
            name="residualEffiSF",
            processes=["MCnoQCD"],
            groups=["residualEffiSF", "experiment", "expNoLumi", "expNoCalib"],
            baseName="residualEffiSF_",
            systAxes=[f"{decorr_syst_var}_", "downUpVar"],
            labelsByAxis=[decorr_syst_var, "downUpVar"],
            passToFakes=passSystToFakes,
            preOp=preOpCorrAction,
            preOpArgs=preOpCorrActionArgs,
            action=syst_tools.decorrelateByAxes,
            actionArgs=dict(
                axesToDecorrNames=[decorr_syst_var],
                newDecorrAxesNames=[f"{decorr_syst_var}_"],
            ),
            actionRequiresNomi=True,
        )

    # Below: all that is highPU specific

    # msv_config_dict = {
    #     "smearingWeights":{
    #         "hist_name": "muonScaleSyst_responseWeights",
    #         "syst_axes": ["unc", "downUpVar"],
    #         "syst_axes_labels": ["unc", "downUpVar"]
    #     },
    #     "massWeights":{
    #         "hist_name": "muonScaleSyst",
    #         "syst_axes": ["downUpVar", "scaleEtaSlice"],
    #         "syst_axes_labels": ["downUpVar", "ieta"]
    #     },
    #     "manualShift":{
    #         "hist_name": "muonScaleSyst_manualShift",
    #         "syst_axes": ["downUpVar"],
    #         "syst_axes_labels": ["downUpVar"]
    #     }
    # }

    # msv_config = msv_config_dict[args.muonScaleVariation]

    # datagroups.addSystematic(msv_config['hist_name'],
    #     processes=['single_v_samples' if wmass else 'single_vmu_samples'],
    #     group="muonCalibration",
    #     baseName="CMS_scale_m_",
    #     systAxes=msv_config['syst_axes'],
    #     labelsByAxis=msv_config['syst_axes_labels'],
    #     passToFakes=passSystToFakes,
    #     scale = args.scaleMuonCorr,
    # )
    prefireSystAxes = ["downUpVar"]
    prefireSystLabels = ["downUpVar"]
    prefireSystAction = None
    prefireSystActionArgs = {}
    if "prefire" in args.decorrSystByVar and decorr_syst_var in fitvar:
        prefireSystAxes = [f"{decorr_syst_var}_"] + prefireSystAxes
        prefireSystLabels = [decorr_syst_var] + prefireSystLabels
        prefireSystAction = syst_tools.decorrelateByAxes
        prefireSystActionArgs = dict(
            axesToDecorrNames=[decorr_syst_var],
            newDecorrAxesNames=[f"{decorr_syst_var}_"],
        )
    datagroups.addSystematic(
        "muonL1PrefireSyst",
        processes=["MCnoQCD"],
        groups=["muonPrefire", "prefire", "experiment", "expNoLumi", "expNoCalib"],
        baseName="CMS_prefire_syst_m",
        systAxes=prefireSystAxes,
        labelsByAxis=prefireSystLabels,
        passToFakes=passSystToFakes,
        action=prefireSystAction,
        actionArgs=prefireSystActionArgs,
        actionRequiresNomi=True,
    )

    prefireStatAxes = (
        ["etaPhiRegion", "downUpVar"] if era == "2016PostVFP" else ["downUpVar"]
    )
    prefireStatLabels = (
        ["etaPhiReg", "downUpVar"] if era == "2016PostVFP" else ["downUpVar"]
    )
    prefireStatAction = None
    prefireStatActionArgs = {}
    if "prefire" in args.decorrSystByVar and decorr_syst_var in fitvar:
        prefireStatAxes = [f"{decorr_syst_var}_"] + prefireStatAxes
        prefireStatLabels = [decorr_syst_var] + prefireStatLabels
        prefireStatAction = syst_tools.decorrelateByAxes
        prefireStatActionArgs = dict(
            axesToDecorrNames=[decorr_syst_var],
            newDecorrAxesNames=[f"{decorr_syst_var}_"],
        )

    datagroups.addSystematic(
        "muonL1PrefireStat",
        processes=["MCnoQCD"],
        groups=["muonPrefire", "prefire", "experiment", "expNoLumi", "expNoCalib"],
        baseName="CMS_prefire_stat_m_",
        passToFakes=passSystToFakes,
        systAxes=prefireStatAxes,
        labelsByAxis=prefireStatLabels,
        action=prefireStatAction,
        actionArgs=prefireStatActionArgs,
        actionRequiresNomi=True,
    )
    datagroups.addSystematic(
        "ecalL1Prefire",
        processes=["MCnoQCD"],
        groups=["ecalPrefire", "prefire", "experiment", "expNoLumi", "expNoCalib"],
        baseName="CMS_prefire_ecal",
        systAxes=["downUpVar"],
        labelsByAxis=["downUpVar"],
        passToFakes=passSystToFakes,
    )

    ## decorrelated momentum scale and resolution, when requested
    if not dilepton and "ptscale" in args.decorrSystByVar and decorr_syst_var in fitvar:
        datagroups.addSystematic(
            "muonScaleSyst_responseWeights",
            name="muonScaleSyst_responseWeightsDecorr",
            processes=["single_v_samples"],
            groups=["scaleCrctn", "muonCalibration", "experiment", "expNoLumi"],
            baseName="Scale_correction_",
            systAxes=["unc", f"{decorr_syst_var}_", "downUpVar"],
            passToFakes=passSystToFakes,
            scale=args.calibrationStatScaling,
            actionRequiresNomi=True,
            action=syst_tools.decorrelateByAxes,
            actionArgs=dict(
                axesToDecorrNames=[decorr_syst_var],
                newDecorrAxesNames=[f"{decorr_syst_var}_"],
            ),
        )

        datagroups.addSystematic(
            "muonScaleClosSyst_responseWeights",
            name="muonScaleClosSyst_responseWeightsDecorr",
            processes=["single_v_samples"],
            groups=["scaleClosCrctn", "muonCalibration", "experiment", "expNoLumi"],
            baseName="ScaleClos_correction_",
            systAxes=["unc", f"{decorr_syst_var}_", "downUpVar"],
            passToFakes=passSystToFakes,
            actionRequiresNomi=True,
            action=syst_tools.decorrelateByAxes,
            actionArgs=dict(
                axesToDecorrNames=[decorr_syst_var],
                newDecorrAxesNames=[f"{decorr_syst_var}_"],
            ),
        )
    else:
        datagroups.addSystematic(
            "muonScaleSyst_responseWeights",
            processes=["single_v_samples"],
            groups=["scaleCrctn", "muonCalibration", "experiment", "expNoLumi"],
            baseName="Scale_correction_",
            systAxes=["unc", "downUpVar"],
            passToFakes=passSystToFakes,
            scale=args.calibrationStatScaling,
        )
        datagroups.addSystematic(
            "muonScaleClosSyst_responseWeights",
            processes=["single_v_samples"],
            groups=["scaleClosCrctn", "muonCalibration", "experiment", "expNoLumi"],
            baseName="ScaleClos_correction_",
            systAxes=["unc", "downUpVar"],
            passToFakes=passSystToFakes,
        )

    mzerr = 2.1e-3
    mz0 = 91.18
    adhocA = args.correlatedAdHocA
    nomvarA = common.correlated_variation_base_size["A"]
    scaleA = math.sqrt((mzerr / mz0) ** 2 + adhocA**2) / nomvarA

    adhocM = args.correlatedAdHocM
    nomvarM = common.correlated_variation_base_size["M"]
    scaleM = adhocM / nomvarM

    datagroups.addSystematic(
        "muonScaleClosASyst_responseWeights",
        processes=["single_v_samples"],
        groups=["scaleClosACrctn", "muonCalibration", "experiment", "expNoLumi"],
        baseName="ScaleClosA_correction_",
        systAxes=["unc", "downUpVar"],
        passToFakes=passSystToFakes,
        scale=scaleA,
    )
    if abs(scaleM) > 0.0:
        datagroups.addSystematic(
            "muonScaleClosMSyst_responseWeights",
            processes=["single_v_samples"],
            groups=["scaleClosMCrctn", "muonCalibration", "experiment", "expNoLumi"],
            baseName="ScaleClosM_correction_",
            systAxes=["unc", "downUpVar"],
            passToFakes=passSystToFakes,
            scale=scaleM,
        )
    if not datagroups.args_from_metadata("noSmearing"):
        if (
            not dilepton
            and "ptscale" in args.decorrSystByVar
            and decorr_syst_var in fitvar
        ):
            datagroups.addSystematic(
                "muonResolutionSyst_responseWeights",
                name="muonResolutionSyst_responseWeightsDecorr",
                mirror=True,
                processes=["single_v_samples"],
                groups=[
                    "resolutionCrctn",
                    "muonCalibration",
                    "experiment",
                    "expNoLumi",
                ],
                baseName="Resolution_correction_",
                systAxes=["smearing_variation", f"{decorr_syst_var}_"],
                passToFakes=passSystToFakes,
                scale=args.resolutionStatScaling,
                actionRequiresNomi=True,
                action=syst_tools.decorrelateByAxes,
                actionArgs=dict(
                    axesToDecorrNames=[decorr_syst_var],
                    newDecorrAxesNames=[f"{decorr_syst_var}_"],
                ),
            )
        else:
            datagroups.addSystematic(
                "muonResolutionSyst_responseWeights",
                mirror=True,
                processes=["single_v_samples"],
                groups=[
                    "resolutionCrctn",
                    "muonCalibration",
                    "experiment",
                    "expNoLumi",
                ],
                baseName="Resolution_correction_",
                systAxes=["smearing_variation"],
                passToFakes=passSystToFakes,
                scale=args.resolutionStatScaling,
            )

    datagroups.addSystematic(
        "pixelMultiplicitySyst",
        mirror=True,
        processes=["single_v_samples"],
        groups=["pixelMultiplicitySyst", "muonCalibration", "experiment", "expNoLumi"],
        baseName="pixel_multiplicity_syst_",
        systAxes=["var"],
        passToFakes=passSystToFakes,
    )

    if datagroups.args_from_metadata("pixelMultiplicityStat"):
        datagroups.addSystematic(
            "pixelMultiplicityStat",
            mirror=True,
            processes=["single_v_samples"],
            groups=[
                "pixelMultiplicityStat",
                "muonCalibration",
                "experiment",
                "expNoLumi",
            ],
            baseName="pixel_multiplicity_stat_",
            systAxes=["var"],
            passToFakes=passSystToFakes,
        )

    if dilepton and "run" in fitvar:
        # add ad-hoc normalization uncertainty uncorrelated across run bins
        #   accounting for time instability (e.g. reflecting the corrections applied as average like pileup, prefiring, ...)
        datagroups.addSystematic(
            name="timeStability",
            processes=["MCnoQCD"],
            groups=["timeStability", "experiment", "expNoLumi", "expNoCalib"],
            passToFakes=passSystToFakes,
            mirror=True,
            labelsByAxis=[f"run"],
            systAxes=["run_"],
            action=lambda h: hh.addHists(
                h, hh.expand_hist_by_duplicate_axis(h, "run", "run_"), scale2=0.01
            ),
        )

        # add additional scale and resolution uncertainty uncorrelated across run slices
        datagroups.addSystematic(
            "muonScaleSyst_responseWeights",
            name="muonScaleSyst_responseWeightsDecorr",
            processes=["single_v_samples"],
            groups=["scaleCrctn", "muonCalibration", "experiment", "expNoLumi"],
            baseName="Scale_correction_",
            systAxes=["unc", "run_", "downUpVar"],
            passToFakes=passSystToFakes,
            scale=args.calibrationStatScaling,
            actionRequiresNomi=True,
            action=syst_tools.decorrelateByAxes,
            actionArgs=dict(axesToDecorrNames=["run"], newDecorrAxesNames=["run_"]),
        )

        datagroups.addSystematic(
            "muonScaleClosSyst_responseWeights",
            name="muonScaleClosSyst_responseWeightsDecorr",
            processes=["single_v_samples"],
            groups=["scaleClosCrctn", "muonCalibration", "experiment", "expNoLumi"],
            baseName="ScaleClos_correction_",
            systAxes=["unc", "run_", "downUpVar"],
            passToFakes=passSystToFakes,
            actionRequiresNomi=True,
            action=syst_tools.decorrelateByAxes,
            actionArgs=dict(axesToDecorrNames=["run"], newDecorrAxesNames=["run_"]),
        )

        if not datagroups.args_from_metadata("noSmearing"):
            datagroups.addSystematic(
                "muonResolutionSyst_responseWeights",
                name="muonResolutionSyst_responseWeightsDecorr",
                mirror=True,
                processes=["single_v_samples"],
                groups=[
                    "resolutionCrctn",
                    "muonCalibration",
                    "experiment",
                    "expNoLumi",
                ],
                baseName="Resolution_correction_",
                systAxes=["smearing_variation", "run_"],
                passToFakes=passSystToFakes,
                scale=args.resolutionStatScaling,
                actionRequiresNomi=True,
                action=syst_tools.decorrelateByAxes,
                actionArgs=dict(axesToDecorrNames=["run"], newDecorrAxesNames=["run_"]),
            )

    # Previously we had a QCD uncertainty for the mt dependence on the fakes, see: https://github.com/WMass/WRemnants/blob/f757c2c8137a720403b64d4c83b5463a2b27e80f/scripts/combine/setupRabbitWMass.py#L359

    return datagroups


def analysis_label(datagroups):
    analysis_name_map = {
        "w_mass": "WMass",
        "vgen": (
            "ZGen"
            if len(datagroups.getProcesses()) > 0
            and datagroups.getProcesses()[0][0] == "Z"
            else "WGen"
        ),
        "z_wlike": "ZMassWLike",
        "z_dilepton": "ZMassDilepton",
        "w_lowpu": "WMass_lowPU",
        "z_lowpu": "ZMass_lowPU",
    }

    if datagroups.mode not in analysis_name_map:
        raise ValueError(f"Invalid datagroups mode {datagroups.mode}")

    return analysis_name_map[datagroups.mode]


def outputFolderName(outfolder, datagroups, doStatOnly, postfix):
    to_join = [analysis_label(datagroups)] + datagroups.fit_axes

    if doStatOnly:
        to_join.append("statOnly")
    if datagroups.flavor:
        to_join.append(datagroups.flavor)
    if postfix is not None:
        to_join.append(postfix)

    return f"{outfolder}/{'_'.join(to_join)}/"


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()

    logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)

    isUnfolding = args.analysisMode == "unfolding"
    isTheoryAgnostic = args.analysisMode in [
        "theoryAgnosticNormVar",
        "theoryAgnosticPolVar",
    ]
    isTheoryAgnosticPolVar = args.analysisMode == "theoryAgnosticPolVar"
    isPoiAsNoi = (isUnfolding or isTheoryAgnostic) and args.poiAsNoi

    if isUnfolding and "xsec" in args.noi:
        raise ValueError(
            "Options unfolding and fitting the xsec are incompatible. Please choose one or the other"
        )

    if isTheoryAgnostic:
        if len(args.genAxes) == 0:
            args.genAxes = ["ptVgenSig-absYVgenSig-helicitySig"]
            logger.warning(
                f"Automatically setting '--genAxes {' '.join(args.genAxes)}' for theory agnostic analysis"
            )
            if args.poiAsNoi:
                logger.warning(
                    "This is only needed to properly get the systematic axes"
                )

    if len(args.inputFile) > 1 and ("wwidth" in args.noi or args.decorMassWidth):
        raise ValueError(
            "Fitting multiple channels with 'wwidth' or decorMassWidth is not currently supported since this can lead to inconsistent treatment of mass variations between channels."
        )

    writer = tensorwriter.TensorWriter(
        sparse=args.sparse,
        allow_negative_expectation=args.allowNegativeExpectation,
        systematic_type=args.systematicType,
        add_bin_by_bin_stat_to_data_cov=args.addMCStatToCovariance,
    )

    if args.fitresult is not None:
        # set data from external fitresult file
        if len(args.inputFile) > 1:
            logger.warning(
                "Theoryfit for more than one channels is currently experimental"
            )

        if args.fitresultResult is not None:
            result_key = None if args.realData else "asimov"
        else:
            result_key = args.fitresultResult

        fitresult, fitresult_meta = rabbit.io_tools.get_fitresult(
            args.fitresult[0], meta=True, result=result_key
        )

        if len(args.fitresult) > 1:
            mapping = args.fitresult[1]
        else:
            mapping = "BaseMapping"

        if len(args.fitresult) > 2:
            channels = args.fitresult[2:]
        else:
            channels = None

        fitresult_hist, fitresult_cov, fitresult_channels = (
            rabbit.io_tools.get_postfit_hist_cov(
                fitresult, mapping=mapping, channels=channels
            )
        )

        writer.add_data_covariance(fitresult_cov)

    dgs = {}  # keep datagroups for across channel definitions
    outnames = []
    # loop over all files
    for i, ifile in enumerate(args.inputFile):
        fitvar = args.fitvar[i].split("-")
        genvar = (
            args.genAxes[i].split("-")
            if hasattr(args, "genAxes") and len(args.genAxes)
            else None
        )
        iBaseName = args.baseName[0] if len(args.baseName) == 1 else args.baseName[i]
        iLumiScale = (
            args.lumiScale[0] if len(args.lumiScale) == 1 else args.lumiScale[i]
        )

        channel = f"ch{i}"

        if args.fitresult is not None:
            fitresult_data = fitresult_hist[i]
        else:
            fitresult_data = None

        if args.analysisMode == "unfolding" and len(args.unfoldingScalemap) > i:
            unfolding_scalemap = args.unfoldingScalemap[i]
        else:
            unfolding_scalemap = None

        datagroups = setup(
            writer,
            args,
            ifile,
            iBaseName,
            iLumiScale,
            fitvar,
            genvar=genvar,
            stat_only=args.doStatOnly,
            channel=channel,
            fitresult_data=fitresult_data,
            unfolding_scalemap=unfolding_scalemap,
        )

        if args.addBSMMixing is not None:
            # add masked channel for SM inclusive cross section with BSM variation
            datagroups_xnorm = setup(
                writer,
                args,
                ifile,
                "gen",
                iLumiScale,
                ["count"],
                genvar=["count"],
                stat_only=args.doStatOnly or args.doStatOnlyMasked,
                channel=f"Wmunu_BSM_masked",
                base_group="Wmunu",
            )
            # and without BSM contribution
            tmp_mixing = args.addBSMMixing[1]
            args.addBSMMixing[1] = 0
            datagroups_xnorm = setup(
                writer,
                args,
                ifile,
                "gen",
                iLumiScale,
                ["count"],
                genvar=["count"],
                stat_only=args.doStatOnly or args.doStatOnlyMasked,
                channel=f"Wmunu_masked",
                base_group="Wmunu",
            )
            args.addBSMMixing[1] = tmp_mixing

        for bsm_signal in filter(
            lambda x: x.startswith("WtoNMu"), datagroups.allMCProcesses()
        ):
            # add masked channel for inclusive cross section on BSM signal
            datagroups_xnorm = setup(
                writer,
                args,
                ifile,
                "gen",
                iLumiScale,
                ["count"],
                genvar=["count"],
                stat_only=args.doStatOnly or args.doStatOnlyMasked,
                channel=f"{bsm_signal}_masked",
                base_group=bsm_signal,
            )

        if isUnfolding:
            # add masked channel
            datagroups_xnorm = setup(
                writer,
                args,
                ifile,
                args.unfoldingLevel,
                iLumiScale,
                genvar,
                genvar=genvar,
                stat_only=args.doStatOnly or args.doStatOnlyMasked,
                channel=f"{channel}_masked",
                unfolding_scalemap=unfolding_scalemap,
            )

            if args.unfoldSimultaneousWandZ and datagroups.mode == "w_mass":
                # for simultaneous unfolding of W and Z we need to add the noi variations on the Z background in the single lepton channel

                if "z_dilepton" not in dgs:
                    raise RuntimeError(
                        "Datagroup 'z_dilepton' not found but required for unfoldSimultaneousWandZ (CLA order matters: specify dilepton first and then single lepton)"
                    )

                poi_axes = ["ptVGen", "absYVGen", "helicitySig"]

                # we have to use the same scalemap as in the Z channel
                scalemap = combine_helpers.get_scalemap(
                    dgs["z_dilepton"],
                    poi_axes,
                    gen_level=args.unfoldingLevel,
                )

                combine_helpers.add_noi_unfolding_variations(
                    datagroups,
                    "Z",
                    True,
                    False,
                    poi_axes=poi_axes,
                    prior_norm=args.priorNormXsec,
                    scale_norm=args.scaleNormXsecHistYields,
                    gen_level=args.unfoldingLevel,
                    process="Zmumu",
                    scalemap=scalemap,
                )

        outnames.append(
            (
                outputFolderName(
                    args.outfolder, datagroups, args.doStatOnly, args.postfix
                ),
                analysis_label(datagroups),
            )
        )

        dgs[datagroups.mode] = datagroups

    if len(outnames) == 1:
        outfolder, outfile = outnames[0]
    else:
        dir_append = "_".join(
            [
                "",
                *filter(
                    lambda x: x,
                    ["statOnly" if args.doStatOnly else "", args.postfix],
                ),
            ]
        )
        unique_names = list(dict.fromkeys([o[1] for o in outnames]))
        outfolder = f"{args.outfolder}/Combination_{''.join(unique_names)}{dir_append}/"
        outfile = "Combination"
    logger.info(f"Writing output to {outfile}")

    # propagate meta info into result file
    meta = {
        "meta_info": output_tools.make_meta_info_dict(
            args=args,
            wd=common.base_dir,
        ),
        "meta_info_input": datagroups.getMetaInfo(),
    }

    writer.write(outfolder=outfolder, outfilename=outfile, meta_data_dict=meta)

    logging.summary()
