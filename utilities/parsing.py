import argparse
import os

from utilities import common, logging

# choices for legend padding
choices_padding = ["auto", "lower left", "lower right", "upper left", "upper right"]


def set_parser_attribute(parser, argument, attribute, newValue):
    # change an argument of the parser, must be called before parse_arguments
    logger = logging.child_logger(__name__)
    f = next((x for x in parser._actions if x.dest == argument), None)
    if f:
        if hasattr(f, attribute):
            logger.info(
                f" Modifying {attribute} of {f.dest} from {getattr(f, attribute)} to {newValue}"
            )
            setattr(f, attribute, newValue)
        else:
            logger.warning(f" Parser argument {argument} has no attribute {attribute}!")
    else:
        logger.warning(f" Parser argument {argument} not found!")
    return parser


def set_parser_default(parser, argument, newDefault):
    # change the default argument of the parser, must be called before parse_arguments
    return set_parser_attribute(parser, argument, "default", newDefault)


def set_subparsers(subparser, name, analysis_label):

    if name is None:
        return subparser

    # options in common between unfolding/theoryAgnostic but not known to the main parser
    subparser.add_argument(
        "--poiAsNoi",
        action="store_true",
        help="Make histogram to do the POIs as NOIs trick (some postprocessing will happen later in CardTool.py)",
    )

    if name == "unfolding":
        # specific for unfolding
        axmap = {
            "w_lowpu": ["ptVGen"],
            "w_mass": ["ptGen", "absEtaGen"],
            "z_dilepton": ["ptVGen", "absYVGen"],
        }
        axmap["z_lowpu"] = axmap["w_lowpu"]
        axmap["z_wlike"] = ["qGen", *axmap["w_mass"]]
        if analysis_label not in axmap:
            raise ValueError(f"Unknown analysis {analysis_label}!")
        subparser.add_argument(
            "--genAxes",
            type=str,
            nargs="+",
            default=axmap[analysis_label],
            choices=["qGen", "ptGen", "absEtaGen", "ptVGen", "absYVGen", "helicitySig"],
            help="Generator level variable",
        )
        subparser.add_argument(
            "--genLevel",
            type=str,
            default="postFSR",
            choices=["preFSR", "postFSR"],
            help="Generator level definition for unfolding",
        )
        subparser.add_argument(
            "--genBins",
            type=int,
            nargs="+",
            default=[18, 0] if "wlike" in analysis_label else [16, 0],
            help="Number of generator level bins",
        )
        subparser.add_argument(
            "--fitresult",
            type=str,
            help="Fitresult to be used to reweight the gen distribution (e.g. for iterative POI as NOI unfolding)",
        )
        subparser.add_argument(
            "--inclusive",
            action="store_true",
            help="No fiducial selection (mass window only)",
        )
    elif "theoryAgnostic" in name:
        # specific for theory agnostic
        subparser.add_argument(
            "--genAxes",
            type=str,
            nargs="+",
            default=["ptVgenSig", "absYVgenSig", "helicitySig"],
            choices=["qGen", "ptVgenSig", "absYVgenSig", "helicitySig"],
            help="Generator level variable",
        )
        subparser.add_argument(
            "--genPtVbinEdges",
            type=float,
            nargs="*",
            default=[],
            help="Bin edges of gen ptV axis for theory agnostic",
        )
        subparser.add_argument(
            "--genAbsYVbinEdges",
            type=float,
            nargs="*",
            default=[],
            help="Bin edges of gen |yV| axis for theory agnostic",
        )
        if name == "theoryAgnosticPolVar":
            subparser.add_argument(
                "--theoryAgnosticFilePath",
                type=str,
                default=".",
                help="Path where input files are stored",
            )
            subparser.add_argument(
                "--theoryAgnosticFileTag",
                type=str,
                default="x0p30_y3p00_V10",
                choices=[
                    "x0p30_y3p00_V4",
                    "x0p30_y3p00_V5",
                    "x0p40_y3p50_V6",
                    "x0p30_y3p00_V7",
                    "x0p30_y3p00_V8",
                    "x0p30_y3p00_V9",
                    "x0p30_y3p00_V10",
                ],
                help="Tag for input files",
            )
            subparser.add_argument(
                "--theoryAgnosticSplitOOA",
                action="store_true",
                help="Define out-of-acceptance signal template as an independent process",
            )

    else:
        raise NotImplementedError(f"Subparser {name} is not defined. Please check!")

    return subparser


def common_histmaker_subparsers(parser, analysis_label):

    parser.add_argument(
        "--analysisMode",
        type=str,
        default=None,
        choices=["unfolding", "theoryAgnosticNormVar", "theoryAgnosticPolVar"],
        help="Select analysis mode to run. Default is the traditional analysis",
    )

    tmpKnownArgs, _ = parser.parse_known_args()
    unfolding = tmpKnownArgs.analysisMode == "unfolding"
    parser.add_argument(
        "--eta",
        nargs=3,
        type=float,
        help="Eta binning as 'nbins min max' (only uniform for now)",
        default=common.get_default_etabins(analysis_label),
    )
    parser.add_argument(
        "--pt",
        nargs=3,
        type=float,
        help="Pt binning as 'nbins,min,max' (only uniform for now)",
        default=common.get_default_ptbins(analysis_label, unfolding=unfolding),
    )
    parser = set_subparsers(parser, tmpKnownArgs.analysisMode, analysis_label)

    return parser


def base_parser():
    parser = argparse.ArgumentParser()
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
    return parser


def common_parser(analysis_label=""):
    for_reco_highPU = "gen" not in analysis_label and "lowpu" not in analysis_label
    parser = base_parser()
    parser.add_argument(
        "-j",
        "--nThreads",
        type=int,
        default=0,
        help="number of threads (0 or negative values use all available threads)",
    )
    initargs, _ = parser.parse_known_args()

    # initName for this internal logger is needed to avoid conflicts with the main logger named "wremnants" by default,
    # otherwise the logger is apparently propagated back to the root logger causing each following message to be printed twice
    common_logger = logging.setup_logger(
        __file__,
        initargs.verbose,
        initargs.noColorLogger,
        initName="common_logger_wremnants",
    )

    import ROOT

    ROOT.ROOT.EnableImplicitMT(max(0, initargs.nThreads))
    from wremnants import theory_corrections, theory_tools

    class PDFFilterAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            # Filter unique values, but keep first item in its position
            if "herapdf20" in values:
                values.append("herapdf20ext")
            unique_values = (
                [values[0], *set([x for x in values[1:]])] if len(values) >= 1 else []
            )
            setattr(namespace, self.dest, unique_values)

    class NoneFilterAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            # Filter unique values, but keep first item in its position
            filtered_values = [x for x in values if x not in ["none", None]]
            setattr(namespace, self.dest, filtered_values)

    parser.add_argument(
        "--pdfs",
        type=str,
        nargs="*",
        default=["ct18z", "msht20mcrange_renorm", "msht20mbrange_renorm"],
        choices=theory_tools.pdfMap.keys(),
        help="PDF sets to produce error hists for. If empty, use PDF set used in production (weight=1).",
        action=PDFFilterAction,
    )
    parser.add_argument(
        "--altPdfOnlyCentral",
        action="store_true",
        help="Only store central value for alternate PDF sets",
    )
    parser.add_argument(
        "--maxFiles", type=int, help="Max number of files (per dataset)", default=None
    )
    parser.add_argument(
        "--filterProcs",
        type=str,
        nargs="*",
        help="Only run over processes matched by group name or (subset) of name",
        default=[],
    )
    parser.add_argument(
        "--excludeProcs",
        type=str,
        nargs="*",
        help="Exclude processes matched by group name or (subset) of name",
        default=[],
    )  # no need to exclude QCD MC here, histograms can always be made, they are fast and light, so they are always available for tests
    parser.add_argument(
        "-p", "--postfix", type=str, help="Postfix for output file name", default=None
    )
    parser.add_argument(
        "--forceDefaultName",
        action="store_true",
        help="Don't modify the name of the output file with some default strings",
    )
    parser.add_argument(
        "--theoryCorr",
        nargs="*",
        type=str,
        action=NoneFilterAction,
        default=[
            "scetlib_dyturbo",
            "scetlib_dyturboCT18ZVars",
            "scetlib_dyturboCT18Z_pdfas",
        ],
        choices=theory_corrections.valid_theory_corrections(),
        help="Apply corrections from indicated generator. First will be nominal correction.",
    )
    parser.add_argument(
        "--theoryCorrAltOnly",
        action="store_true",
        help="Save hist for correction hists but don't modify central weight",
    )
    parser.add_argument(
        "--ewTheoryCorr",
        nargs="*",
        type=str,
        action=NoneFilterAction,
        choices=theory_corrections.valid_ew_theory_corrections(),
        default=[
            "renesanceEW",
            "powhegFOEW",
            "pythiaew_ISR",
            "horaceqedew_FSR",
            "horacelophotosmecoffew_FSR",
        ],
        help="Add EW theory corrections without modifying the default theoryCorr list. Will be appended to args.theoryCorr",
    )
    parser.add_argument(
        "--skipHelicity",
        action="store_true",
        help="Skip the qcdScaleByHelicity histogram (it can be huge)",
    )
    parser.add_argument(
        "--noRecoil", action="store_true", help="Don't apply recoild correction"
    )
    parser.add_argument(
        "--recoilHists",
        action="store_true",
        help="Save all recoil related histograms for calibration and validation",
    )
    parser.add_argument(
        "--recoilUnc",
        action="store_true",
        help="Run the recoil calibration with uncertainties (slower)",
    )
    parser.add_argument(
        "--highptscales",
        action="store_true",
        help="Apply highptscales option in MiNNLO for better description of data at high pT",
    )
    parser.add_argument(
        "--dataPath",
        type=str,
        default=None,
        help="Access samples from this path (default reads from local machine), for eos use 'root://eoscms.cern.ch//store/cmst3/group/wmass/w-mass-13TeV/NanoAOD/'",
    )
    parser.add_argument(
        "--noVertexWeight",
        action="store_true",
        help="Do not apply reweighting of vertex z distribution in MC to match data",
    )
    parser.add_argument(
        "--validationHists",
        action="store_true",
        help="make histograms used only for validations",
    )
    parser.add_argument(
        "--onlyMainHistograms",
        action="store_true",
        help="Only produce some histograms, skipping (most) systematics to run faster when those are not needed",
    )
    parser.add_argument(
        "--met",
        type=str,
        choices=[
            "DeepMETReso",
            "DeepMETResp",
            "RawPFMET",
            "DeepMETPVRobust",
            "DeepMETPVRobustNoPUPPI",
        ],
        help="Choice of MET",
        default="DeepMETPVRobust",
    )
    parser.add_argument("-o", "--outfolder", type=str, default="", help="Output folder")
    parser.add_argument(
        "--appendOutputFile",
        type=str,
        default="",
        help="Append analysis output to specified output file",
    )
    parser.add_argument(
        "--sequentialEventLoops",
        action="store_true",
        help="Run event loops sequentially for each process to reduce memory usage",
    )
    parser.add_argument(
        "-e",
        "--era",
        type=str,
        choices=[
            "2016PreVFP",
            "2016PostVFP",
            "2017",
            "2018",
            "2017H",
            "2023_PUAVE1",
            "2023_PUAVE2",
            "2023_PUAVE5",
            "2023_PUAVE10",
        ],
        help="Data set to process",
        default="2016PostVFP",
    )
    parser.add_argument(
        "--scale_A",
        default=1.0,
        type=float,
        help="scaling of the uncertainty on the b-field scale parameter A",
    )
    parser.add_argument(
        "--scale_e",
        default=1.0,
        type=float,
        help="scaling of the uncertainty on the material scale parameter e",
    )
    parser.add_argument(
        "--scale_M",
        default=1.0,
        type=float,
        help="scaling of the uncertainty on the alignment scale parameter M",
    )
    parser.add_argument(
        "--nonClosureScheme",
        type=str,
        default="A-M-combined",
        choices=[
            "none",
            "A-M-separated",
            "A-M-combined",
            "binned",
            "binned-plus-M",
            "A-only",
            "M-only",
        ],
        help="source of the Z non-closure nuisances",
    )
    parser.add_argument(
        "--correlatedNonClosureNP",
        action="store_false",
        help="disable the de-correlation of Z non-closure nuisance parameters after the jpsi massfit",
    )
    parser.add_argument(
        "--dummyNonClosureA",
        action="store_true",
        help="read values for the magnetic part of the Z non-closure from a file",
    )
    parser.add_argument(
        "--dummyNonClosureAMag",
        default=6.8e-5,
        type=float,
        help="magnitude of the dummy value for the magnetic part of the Z non-closure",
    )
    parser.add_argument(
        "--dummyNonClosureM",
        action="store_true",
        help="use a dummy value for the alignment part of the Z non-closure",
    )
    parser.add_argument(
        "--dummyNonClosureMMag",
        default=0.0,
        type=float,
        help="magnitude of the dummy value for the alignment part of the Z non-closure",
    )
    parser.add_argument(
        "--noScaleToData",
        action="store_true",
        help="Do not scale the MC histograms with xsec*lumi/sum(gen weights) in the postprocessing step",
    )
    parser.add_argument(
        "--aggregateGroups",
        type=str,
        nargs="*",
        default=["Diboson", "Top"],
        help="Sum up histograms from members of given groups in the postprocessing step",
    )
    parser.add_argument(
        "--muRmuFPolVarFilePath",
        type=str,
        default=f"{common.data_dir}/MiNNLOmuRmuFPolVar/",
        help="Path where input files are stored",
    )
    parser.add_argument(
        "--muRmuFPolVarFileTag",
        type=str,
        default="x0p50_y4p00_ConstrPol5ExtYdep_Trad",
        choices=[
            "x0p50_y4p00_ConstrPol5ExtYdep_Trad",
            "x0p50_y4p00_ConstrPol5Ext_Trad",
        ],
        help="Tag for input files",
    )
    parser.add_argument(
        "--nToysMC", type=int, help="random toys for data and MC", default=-1
    )
    parser.add_argument(
        "--varianceScalingForToys",
        type=int,
        default=1,
        help="Scaling of variance for toys (effective mc statistics corresponds to 1./scaling)",
    )
    parser.add_argument(
        "--randomSeedForToys", type=int, default=0, help="random seed for toys"
    )

    if for_reco_highPU:
        # additional arguments specific for histmaker of reconstructed objects at high pileup (mw, mz_wlike, and mz_dilepton)
        parser.add_argument(
            "--dphiMuonMetCut",
            type=float,
            help="Threshold to cut |deltaPhi| > thr*np.pi between muon and met",
            default=0.0,
        )
        parser.add_argument(
            "--muonCorrMC",
            type=str,
            default="idealMC_lbltruth",
            choices=[
                "none",
                "trackfit_only",
                "trackfit_only_idealMC",
                "lbl",
                "idealMC_lbltruth",
                "idealMC_massfit",
                "idealMC_lbltruth_massfit",
            ],
            help="Type of correction to apply to the muons in simulation",
        )
        parser.add_argument(
            "--muonCorrData",
            type=str,
            default="lbl_massfit",
            choices=["none", "trackfit_only", "lbl", "massfit", "lbl_massfit"],
            help="Type of correction to apply to the muons in data",
        )
        parser.add_argument(
            "--muScaleBins",
            type=int,
            default=1,
            help="Number of bins for muon scale uncertainty",
        )
        parser.add_argument(
            "--muonScaleVariation",
            choices=["smearingWeightsGaus", "smearingWeightsSplines", "massWeights"],
            default="smearingWeightsSplines",
            help="method to generate nominal muon scale variation histograms",
        )
        parser.add_argument(
            "--dummyMuScaleVar",
            action="store_true",
            help="Use a dummy 1e-4 variation on the muon scale instead of reading from the calibration file",
        )
        parser.add_argument(
            "--muonCorrMag",
            default=1.0e-4,
            type=float,
            help="Magnitude of dummy muon momentum calibration uncertainty",
        )
        parser.add_argument(
            "--muonCorrEtaBins",
            default=1,
            type=int,
            help="Number of eta bins for dummy muon momentum calibration uncertainty",
        )
        parser.add_argument(
            "--biasCalibration",
            type=str,
            default=None,
            choices=["binned", "parameterized", "A", "M"],
            help="Adjust central value by calibration bias hist for simulation",
        )
        parser.add_argument(
            "--noSmearing", action="store_true", help="Disable resolution corrections"
        )
        # options for efficiencies
        parser.add_argument(
            "--trackerMuons",
            action="store_true",
            help="Use tracker muons instead of global muons (need appropriate scale factors too). This is obsolete",
        )
        parser.add_argument(
            "--binnedScaleFactors",
            action="store_true",
            help="Use binned scale factors (different helpers)",
        )
        parser.add_argument(
            "--noSmooth3dsf",
            dest="smooth3dsf",
            action="store_false",
            help="If true (default) use smooth 3D scale factors instead of the original 2D ones (but eff. systs are still obtained from 2D version)",
        )
        parser.add_argument(
            "--isoEfficiencySmoothing",
            action="store_true",
            help="If isolation SF was derived from smooth efficiencies instead of direct smoothing",
        )
        parser.add_argument(
            "--noScaleFactors",
            action="store_true",
            help="Don't use scale factors for efficiency (legacy option for tests)",
        )
        parser.add_argument(
            "--isolationDefinition",
            choices=["iso04vtxAgn", "iso04", "iso04chg", "iso04chgvtxAgn"],
            default="iso04vtxAgn",
            help="Isolation type (and corresponding scale factors)",
        )
        parser.add_argument(
            "--isolationThreshold",
            default=0.15,
            type=float,
            help="Threshold for isolation cut",
        )
        parser.add_argument(
            "--reweightPixelMultiplicity",
            action="store_true",
            help="Reweight events based on number of valid pixel hits for the muons",
        )
        parser.add_argument(
            "--requirePixelHits",
            action="store_true",
            help="Require good muons to have at least one valid pixel hit used in the track refit.",
        )
        parser.add_argument(
            "--pixelMultiplicityStat",
            action="store_true",
            help="Include (very small) statistical uncertainties for pixel multiplicity variation",
        )
        parser.add_argument(
            "--vetoRecoPt",
            default=15,
            type=float,
            help="Lower threshold for muon pt in the veto definition",
        )

    commonargs, _ = parser.parse_known_args()

    if for_reco_highPU:
        if commonargs.trackerMuons:
            common_logger.warning(
                "Using tracker muons, but keep in mind that scale factors are obsolete and not recommended."
            )
            sfFile = "scaleFactorProduct_16Oct2022_TrackerMuonsHighPurity_vertexWeight_OSchargeExceptTracking.root"
        else:
            # note: for trigger and isolation one would actually use 3D SF vs eta-pt-ut.
            # However, even when using the 3D SF one still needs the 2D ones to read the syst/nomi ratio,
            # since the dataAltSig tag-and-probe fits were not run in 3D (it is assumed for simplicity that the syst/nomi ratio is independent from uT)
            #
            # 2D SF without ut-dependence, still needed to compute systematics when uing 3D SF
            if commonargs.era == "2016PostVFP":
                sfFile = (
                    "allSmooth_GtoHout.root"
                    if commonargs.isolationDefinition == "iso04"
                    else "allSmooth_GtoHout_vtxAgnIso.root"
                )
            elif commonargs.era == "2018":
                if commonargs.isolationDefinition == "iso04":
                    raise NotImplementedError(
                        f"For Era {commonargs.era} Isolation Definition {commonargs.isolationDefinition} is not supported"
                    )
                else:
                    sfFile = "allSmooth_2018_vtxAgnIso.root"
            elif commonargs.era == "2017":
                if commonargs.isolationDefinition == "iso04":
                    raise NotImplementedError(
                        f"For Era {commonargs.era} Isolation Definition {commonargs.isolationDefinition} is not supported"
                    )
                else:
                    sfFile = "allSmooth_2017_vtxAgnIso.root"
            else:
                raise NotImplementedError(f"Era {commonargs.era} is not yet supported")

        sfFile = f"{common.data_dir}/muonSF/{sfFile}"
    else:
        sfFile = ""

    parser.add_argument(
        "--sfFile", type=str, help="File with muon scale factors", default=sfFile
    )
    parser = common_histmaker_subparsers(parser, analysis_label)

    class PrintParserAction(argparse.Action):
        def __init__(self, option_strings, dest, nargs=0, **kwargs):
            if nargs != 0:
                raise ValueError(
                    "nargs for PrintParserAction must be 0 since it does not require any argument"
                )
            super().__init__(option_strings, dest, nargs=nargs, **kwargs)

        def __call__(self, parser, namespace, values, option_string=None):
            # meant to substitute the native help message of the parser printing the whole parser with its arguments
            # needed because when we call parse_args only the options defined until there will fall in the help message
            thisLogger = logging.child_logger(__name__)
            thisLogger.warning("Printing parser with all its arguments")
            thisLogger.warning("")
            thisLogger.warning(namespace)
            thisLogger.warning("")

    parser.add_argument(
        "--printParser",
        action=PrintParserAction,
        help="Print the whole parser with its arguments (use it as the last argument or default values might not be displayed correctly)",
    )

    return parser, initargs


def plot_parser():
    parser = base_parser()
    parser.add_argument(
        "-o",
        "--outpath",
        type=str,
        default=os.path.expanduser("~/www/WMassAnalysis"),
        help="Base path for output",
    )
    parser.add_argument(
        "-f", "--outfolder", type=str, default="./test", help="Subfolder for output"
    )
    parser.add_argument(
        "-p", "--postfix", type=str, help="Postfix for output file name"
    )
    parser.add_argument(
        "--eoscp",
        action="store_true",
        help="Override use of xrdcp and use the mount instead",
    )
    parser.add_argument(
        "--lumi",
        type=float,
        default=16.8,
        help="Luminosity used in the fit, needed to get the absolute cross section",
    )
    parser.add_argument(
        "--cmsDecor",
        default="Preliminary",
        nargs="?",
        type=str,
        choices=[
            None,
            " ",
            "Preliminary",
            "Work in progress",
            "Internal",
            "Supplementary",
        ],
        help="CMS label",
    )
    parser.add_argument("--logoPos", type=int, default=2, help="CMS logo position")
    parser.add_argument(
        "--legPos", type=str, default="upper right", help="Set legend position"
    )
    parser.add_argument(
        "--legSize",
        type=str,
        default="small",
        help="Legend text size (small: axis ticks size, large: axis label size, number)",
    )
    parser.add_argument(
        "--legCols", type=int, default=2, help="Number of columns in legend"
    )
    parser.add_argument(
        "--legPadding",
        type=str,
        default="auto",
        choices=choices_padding,
        help="Where to put empty entries in legend",
    )
    parser.add_argument(
        "--lowerLegPos",
        type=str,
        default="upper left",
        help="Set lower legend position",
    )
    parser.add_argument(
        "--lowerLegCols", type=int, default=2, help="Number of columns in lower legend"
    )
    parser.add_argument(
        "--lowerLegPadding",
        type=str,
        default="auto",
        choices=choices_padding,
        help="Where to put empty entries in lower legend",
    )
    parser.add_argument(
        "--noSciy",
        action="store_true",
        help="Don't allow scientific notation for y axis",
    )
    parser.add_argument(
        "--yscale",
        type=float,
        help="Scale the upper y axis by this factor (useful when auto scaling cuts off legend)",
    )
    parser.add_argument(
        "--ylim",
        type=float,
        nargs=2,
        help="Min and max values for y axis (if not specified, range set automatically)",
    )
    parser.add_argument("--xlim", type=float, nargs=2, help="min and max for x axis")
    parser.add_argument(
        "--rrange",
        type=float,
        nargs=2,
        default=[0.9, 1.1],
        help="y range for ratio plot",
    )
    parser.add_argument(
        "--scaleTextSize",
        type=float,
        default=1.0,
        help="Scale all text sizes by this number",
    )
    parser.add_argument(
        "--customFigureWidth",
        type=float,
        default=None,
        help="Use a custom figure width, otherwise chosen automatic",
    )

    return parser
