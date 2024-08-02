import hist
import pathlib
import argparse
import numpy as np
import os
from utilities import logging
from enum import Enum
import re

base_dir = f"{pathlib.Path(__file__).parent}/../"
wremnants_dir = f"{pathlib.Path(__file__).parent}/../wremnants"
data_dir =  f"{pathlib.Path(__file__).parent}/../wremnants-data/data/"

BR_TAUToMU = 0.1739
BR_TAUToE = 0.1782
# cross sections in pb
xsec_ZmmPostVFP = 2001.9
xsec_WpmunuPostVFP = 11765.9
xsec_WmmunuPostVFP = 8703.87
xsec_ZmmMass10to50PostVFP = 6997.0
Z_TAU_TO_LEP_RATIO = (1.-(1. - BR_TAUToMU - BR_TAUToE)**2)

wprocs = ["WplusmunuPostVFP", "WminusmunuPostVFP", "WminustaunuPostVFP", "WplustaunuPostVFP", 
    'Wplusmunu_MiNNLO-noqedisr', 'Wminusmunu_MiNNLO-noqedisr',
    'Wplusmunu_horace-lo-photos', 'Wplusmunu_horace-lo-photos-mecoff', 'Wplusmunu_horace-nlo', 'Wplusmunu_horace-lo', 'Wplusmunu_horace-qed', 
    'Wminusmunu_horace-lo-photos', 'Wminusmunu_horace-lo-photos-mecoff', 'Wminusmunu_horace-nlo', 'Wminusmunu_horace-lo', 'Wminusmunu_horace-qed',
    'Wplusmunu_winhac-lo-photos', 'Wplusmunu_winhac-lo', 'Wplusmunu_winhac-nlo', 
    'Wminusmunu_winhac-lo-photos', 'Wminusmunu_winhac-lo', 'Wminusmunu_winhac-nlo']
zprocs = ["ZmumuPostVFP", "ZtautauPostVFP", "ZmumuMiNLO", "ZmumuNNLOPS", 'Zmumu_MiNNLO-noqedisr',
    'Zmumu_horace-lo-photos', 'Zmumu_horace-lo-photos-isroff', 'Zmumu_horace-lo-photos-mecoff', 'Zmumu_horace-nlo', 'Zmumu_horace-lo', 'Zmumu_horace-new', 'Zmumu_horace-qed',
    'Zmumu_horace-alpha-fsr-off-isr-off', 'Zmumu_horace-alpha-old-fsr-off-isr-off', 'Zmumu_horace-alpha-old-fsr-off-isr-pythia',
    'Zmumu_renesance-lo', 'Zmumu_renesance-nlo',
          'Zmumu_powheg-lo', 'Zmumu_powheg-nloew-qedveto', 'Zmumu_powheg-nloew',
    ]

vprocs = wprocs+zprocs
zprocs_recoil = ["ZmumuPostVFP"]
wprocs_recoil = ["WplusmunuPostVFP", "WminusmunuPostVFP"]

wprocs_lowpu = ["Wminusmunu", "Wminusenu", "Wminustaunu", "Wplusmunu", "Wplusenu", "Wplustaunu"]
zprocs_lowpu = ["Zmumu", "Zee", "Ztautau"]
vprocs_lowpu = wprocs_lowpu+zprocs_lowpu
zprocs_recoil_lowpu = ["Zmumu", "Zee"]
wprocs_recoil_lowpu = ["Wminusmunu", "Wminusenu", "Wplusmunu", "Wplusenu"]

background_MCprocs = ["Top", "Diboson", "QCD", "DYlowMass"]
zprocs_all = zprocs_lowpu+zprocs
wprocs_all = wprocs_lowpu+wprocs
vprocs_all = vprocs_lowpu+vprocs

# input files for muon momentum scale nuisances
calib_dir = f"{data_dir}/calibration/"
closure_dir = f"{data_dir}/closure/"
calib_filepaths = {
    'mc_corrfile': {
        'idealMC_massfit': f"{calib_dir}/calibrationJMC_smeared_v718_nominal.root",
        'idealMC_lbltruth_massfit': f"{calib_dir}/calibrationJMC_smeared_v718_nominalLBL.root"
    },
    'data_corrfile': {
        'massfit': f"{calib_dir}/calibrationJDATA_ideal.root",
        'lbl_massfit': f"{calib_dir}/calibrationJDATA_MCstat_inclusive_binsel.root"
        # 'lbl_massfit': f"{calib_dir}/calibrationJZ_DATA_MCstat_binsel.root"
    },
    'mc_resofile': f"{calib_dir}/sigmaMC_LBL_JYZ.root",
    'data_resofile': f"{calib_dir}/sigmaDATA_LBL_JYZ.root",
    'tflite_file': f"{calib_dir}/muon_response.tflite"
    # 'tflite_file': f"{calib_dir}/muon_response_nosmearing.tflite"
}
closure_filepaths = {
    'parametrized': f"{closure_dir}/parametrizedClosureZ_ORkinweight_binsel_MCstat_fullres.root",
    # 'parametrized': f"{closure_dir}/parametrizedClosureZ_ORkinweight_binsel_MCstat_simul.root",
    'binned': f"{closure_dir}/closureZ_LBL_smeared_v721.root"
}

# some constants for momentum scale uncertainties
correlated_variation_base_size = {
    "A" : 1e-5,
    "M" : 1e-6,
    }

## 5% quantiles from aMC@NLO used in SMP-18-012
#ptV_5quantiles_binning = [0.0, 1.971, 2.949, 3.838, 4.733, 5.674, 6.684, 7.781, 8.979, 10.303, 11.777, 13.435, 15.332, 17.525, 20.115, 23.245, 27.173, 32.414, 40.151, 53.858, 13000.0]
## 10% quantiles from aMC@NLO used in SMP-18-012 with some rounding <== This one worked fine with toys
ptV_10quantiles_binning = [0.0, 2.95, 4.73, 6.68, 8.98, 11.78, 15.33, 20.11, 27.17, 40.15, 13000.]
# Integer rounded version of the 5% quantiles h[::hist.rebin(2)] for 10% quantiles
ptV_binning = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 20, 23, 27, 32, 40, 54, 13000]
ptV_corr_binning = ptV_binning[:-4]+list(range(30,110,10))
absYV_binning = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4]

# categorical axes in python bindings always have an overflow bin, so use a regular axis for the charge
axis_charge = hist.axis.Regular(2, -2., 2., underflow=False, overflow=False, name = "charge")

down_up_axis = hist.axis.Regular(2, -2., 2., underflow=False, overflow=False, name = "downUpVar")
down_nom_up_axis = hist.axis.Regular(3, -1.5, 1.5, underflow=False, overflow=False, name = "downNomUpVar")

# run edges chosen to separate eras (era F post VFP: [278769, 278808], era G [278820, 280385], era F [281613, 284044])
run_edges = np.array([278768, 278808, 279588, 279767, 280017, 280385, 282037, 283270, 283478, 283934, 284044])
run_edges_lumi = np.array([0.0, 0.419, 2.332, 4.329, 6.247, 8.072, 10.152, 12.265, 14.067, 15.994, 16.812])

# for fake estimation
# binary categories for simple ABCD method
passIsoName = "passIso"
passMTName = "passMT"

passIso = {passIsoName: True}
failIso = {passIsoName: False}
passMT = {passMTName: True}
failMT = {passMTName: False}

axis_passIso = hist.axis.Boolean(name = passIsoName)
axis_passMT = hist.axis.Boolean(name = passMTName)

# axes with only a few bins for beyond simple ABCD methods
axis_isoCat = hist.axis.Variable([0,4,8], name = "iso",underflow=False, overflow=True)
axis_relIsoCat = hist.axis.Variable([0,0.15,0.3], name = "relIso",underflow=False, overflow=True)


def get_binning_fakes_pt(min_pt, max_pt):
    edges = np.arange(min_pt,32,1)
    edges = np.append(edges, [e for e in [33,36,40,46,56] if e<max_pt][:-1])
    edges = np.append(edges, [max_pt])
    ## the following lines are used to replace the previous ones when studying different pT binning and the MC stat
    #edges = np.arange(min_pt,32.1,1.2)  
    #edges = np.append(edges, [e for e in [34.4, 38, 44, 56] if e<max_pt][:-1])
    #edges = np.append(edges, [max_pt])
    #edges = np.arange(min_pt,32,2)
    #edges = np.append(edges, [e for e in [32, 36, 40, 46, 56] if e<max_pt][:-1])
    #edges = np.append(edges, [max_pt])
    return edges


def get_binning_fakes_mt(mt_cut=40, high_mt_bins=False):
    edges = np.array([0, int(mt_cut/2.), mt_cut])
    if high_mt_bins:
        # needed for extended 2D method
        edges = np.append(edges, [e for e in [30,32,34,36,38,40,44,49,55,62] if e>mt_cut])
    return edges


def get_binning_fakes_relIso(high_iso_bins=False):
    edges = [0,0.15]
    if high_iso_bins:
        # needed for extended 2D method
        edges.append(0.3)
    return edges


def get_dilepton_ptV_binning(fine=False):
    return [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 20, 23, 27, 32, 40, 54, 100] if not fine else range(60)


def get_gen_axes(flow=False, dilepton_ptV_binning=None, inclusive=False):
    if dilepton_ptV_binning is None:
        dilepton_ptV_binning = get_dilepton_ptV_binning()

    gen_axes = {
        "ptVGen": hist.axis.Variable(dilepton_ptV_binning, name = "ptVGen", underflow=False, overflow=flow),
        "absYVGen": hist.axis.Regular(10, 0, 2.5, name = "absYVGen", underflow=False, overflow=flow)
    }
    if inclusive:
        binning = (*gen_axes["absYVGen"].edges[:-1], 5.)
        gen_axes["absYVGen"] = hist.axis.Variable(binning, name="absYVGen", underflow=False, overflow=flow)
    return gen_axes


def get_default_ptbins(analysis_label, unfolding=False, gen=False):
    vals = [30,26.,56.] if analysis_label[0] == "w" else [34,26.,60.]
    if unfolding and gen:
        raise ValueError("Inconsistent arguments for 'unfolding' and 'gen.' Must be unique")

    if unfolding:
        vals[0] += 2
        vals[2] += 2
    elif gen:
        vals[0] -= 2
        vals[1] += 2
    return vals


def get_default_etabins(analysis_label=None):
    return (48,-2.4,2.4)


def get_default_mtcut(analysis_label=None):
    return 40. if analysis_label[0] == "w" else 45.


def get_default_mz_window():
    return 60, 120


# following list is used in other scripts to track what steps are charge dependent
# but assumes the corresponding efficiencies were made that way
muonEfficiency_chargeDependentSteps = ["reco", "tracking", "idip", "trigger", "antitrigger"] # antitrigger = P(failTrig|IDIP), similar to antiiso = P(failIso|trigger)
muonEfficiency_altBkgSyst_effSteps = ["reco", "tracking"]
muonEfficiency_standaloneNumberOfValidHits = 1 # to use as "var >= this" (if this=0 the define for the cut is not used at all)


def getIsoMtRegionID(passIso=True, passMT=True):
    return passIso * 1 + passMT * 2


def getIsoMtRegionFromID(regionID):
    return {passIsoName : regionID & 1,
            passMTName  : regionID & 2}


def set_parser_default(parser, argument, newDefault):
    # change the default argument of the parser, must be called before parse_arguments
    logger = logging.child_logger(__name__)
    f = next((x for x in parser._actions if x.dest ==argument), None)
    if f:
        logger.info(f" Modifying default of {f.dest} from {f.default} to {newDefault}")
        f.default = newDefault
    else:
        logger.warning(f" Parser argument {argument} not found!")
    return parser


def set_subparsers(subparser, name, analysis_label):

    if name is None:
        return subparser

    # options in common between unfolding/theoryAgnostic but not known to the main parser
    subparser.add_argument("--poiAsNoi", action='store_true',
                           help="Make histogram to do the POIs as NOIs trick (some postprocessing will happen later in CardTool.py)")

    if name == "unfolding":
        # specific for unfolding
        axmap = {
            "w_lowpu" : ["ptVGen"],
            "w_mass" : ["ptGen", "absEtaGen"],
            "z_dilepton" : ["ptVGen", "absYVGen"],
        }
        axmap["z_lowpu"] = axmap["w_lowpu"]
        axmap["z_wlike"] = ["qGen", *axmap["w_mass"]]
        if analysis_label not in axmap:
            raise ValueError(f"Unknown analysis {analysis_label}!")
        subparser.add_argument("--genAxes", type=str, nargs="+", 
                               default=axmap[analysis_label], choices=["qGen", "ptGen", "absEtaGen", "ptVGen", "absYVGen", "helicitySig"],
                               help="Generator level variable")
        subparser.add_argument("--genLevel", type=str, default='postFSR', choices=["preFSR", "postFSR"],
                               help="Generator level definition for unfolding")
        subparser.add_argument("--genBins", type=int, nargs="+", default=[18, 0] if "wlike" in analysis_label else [16, 0],
                               help="Number of generator level bins")
        subparser.add_argument("--inclusive", action='store_true', help="No fiducial selection (mass window only)")
    elif "theoryAgnostic" in name:
        # specific for theory agnostic
        subparser.add_argument("--genAxes", type=str, nargs="+", default=["ptVgenSig", "absYVgenSig", "helicitySig"], choices=["qGen", "ptVgenSig", "absYVgenSig", "helicitySig"], help="Generator level variable")
        subparser.add_argument("--genPtVbinEdges", type=float, nargs="*", default=[],
                               help="Bin edges of gen ptV axis for theory agnostic")
        subparser.add_argument("--genAbsYVbinEdges", type=float, nargs="*", default=[],
                               help="Bin edges of gen |yV| axis for theory agnostic")
        if name == "theoryAgnosticPolVar":
            subparser.add_argument("--theoryAgnosticFilePath", type=str, default=".",
                                   help="Path where input files are stored")
            subparser.add_argument("--theoryAgnosticFileTag", type=str, default="x0p30_y3p00_V9", choices=["x0p30_y3p00_V4", "x0p30_y3p00_V5", "x0p40_y3p50_V6", "x0p30_y3p00_V7", "x0p30_y3p00_V8", "x0p30_y3p00_V9"],
                                   help="Tag for input files")
            subparser.add_argument("--theoryAgnosticSplitOOA", action='store_true',
                                   help="Define out-of-acceptance signal template as an independent process")

    else:
        raise NotImplementedError(f"Subparser {name} is not defined. Please check!")

    return subparser


def common_histmaker_subparsers(parser, analysis_label):

    parser.add_argument("--analysisMode", type=str, default=None,
                        choices=["unfolding", "theoryAgnosticNormVar", "theoryAgnosticPolVar"],
                        help="Select analysis mode to run. Default is the traditional analysis")
    
    tmpKnownArgs,_ = parser.parse_known_args()
    unfolding = tmpKnownArgs.analysisMode == "unfolding"
    parser.add_argument("--eta", nargs=3, type=float, help="Eta binning as 'nbins min max' (only uniform for now)", default=get_default_etabins(analysis_label))
    parser.add_argument("--pt", nargs=3, type=float, help="Pt binning as 'nbins,min,max' (only uniform for now)", default=get_default_ptbins(analysis_label, unfolding=unfolding))
    parser = set_subparsers(parser, tmpKnownArgs.analysisMode, analysis_label)

    return parser


def base_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", type=int, default=3, choices=[0,1,2,3,4],
                        help="Set verbosity level with logging, the larger the more verbose")
    parser.add_argument("--noColorLogger", action="store_true", help="Do not use logging with colors")
    return parser


def common_parser(analysis_label=""):
    for_reco_highPU = "gen" not in analysis_label and "lowpu" not in analysis_label
    parser = base_parser()
    parser.add_argument("-j", "--nThreads", type=int, default=0, help="number of threads (0 or negative values use all available threads)")
    initargs,_ = parser.parse_known_args()

    # initName for this internal logger is needed to avoid conflicts with the main logger named "wremnants" by default,
    # otherwise the logger is apparently propagated back to the root logger causing each following message to be printed twice 
    common_logger = logging.setup_logger(__file__, initargs.verbose, initargs.noColorLogger, initName="common_logger_wremnants")
    
    import ROOT
    ROOT.ROOT.EnableImplicitMT(max(0,initargs.nThreads))
    import narf
    import wremnants
    from wremnants import theory_corrections,theory_tools

    class PDFFilterAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            # Filter unique values, but keep first item in its position
            if "herapdf20" in values:
                values.append("herapdf20ext")
            unique_values = [values[0], *set([x for x in values[1:]])] if len(values) >= 1 else []
            setattr(namespace, self.dest, unique_values)

    class NoneFilterAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            # Filter unique values, but keep first item in its position
            filtered_values = [x for x in values if x not in ["none", None]]
            setattr(namespace, self.dest, filtered_values)

    parser.add_argument("--pdfs", type=str, nargs="*", default=["ct18z", "msht20mcrange_renorm", "msht20mbrange_renorm"], 
        choices=theory_tools.pdfMap.keys(), help="PDF sets to produce error hists for. If empty, use PDF set used in production (weight=1).", action=PDFFilterAction)
    parser.add_argument("--altPdfOnlyCentral", action='store_true', help="Only store central value for alternate PDF sets")
    parser.add_argument("--maxFiles", type=int, help="Max number of files (per dataset)", default=None)
    parser.add_argument("--filterProcs", type=str, nargs="*", help="Only run over processes matched by group name or (subset) of name", default=[])
    parser.add_argument("--excludeProcs", type=str, nargs="*", help="Exclude processes matched by group name or (subset) of name", default=[])  # no need to exclude QCD MC here, histograms can always be made, they are fast and light, so they are always available for tests
    parser.add_argument("-p", "--postfix", type=str, help="Postfix for output file name", default=None)
    parser.add_argument("--forceDefaultName", action='store_true', help="Don't modify the name of the output file with some default strings")
    parser.add_argument("--theoryCorr", nargs="*", type=str, action=NoneFilterAction,
        default=["scetlib_dyturbo", "scetlib_dyturboCT18ZVars", "scetlib_dyturboCT18Z_pdfas"], choices=theory_corrections.valid_theory_corrections(), 
        help="Apply corrections from indicated generator. First will be nominal correction.")
    parser.add_argument("--theoryCorrAltOnly", action='store_true', help="Save hist for correction hists but don't modify central weight")
    parser.add_argument("--ewTheoryCorr", nargs="*", type=str, action=NoneFilterAction, choices=theory_corrections.valid_ew_theory_corrections(), 
        default=["renesanceEW", "powhegFOEW", "pythiaew_ISR", "horaceqedew_FSR", "horacelophotosmecoffew_FSR", ],
        help="Add EW theory corrections without modifying the default theoryCorr list. Will be appended to args.theoryCorr")
    parser.add_argument("--skipHelicity", action='store_true', help="Skip the qcdScaleByHelicity histogram (it can be huge)")
    parser.add_argument("--noRecoil", action='store_true', help="Don't apply recoild correction")
    parser.add_argument("--recoilHists", action='store_true', help="Save all recoil related histograms for calibration and validation")
    parser.add_argument("--recoilUnc", action='store_true', help="Run the recoil calibration with uncertainties (slower)")
    parser.add_argument("--highptscales", action='store_true', help="Apply highptscales option in MiNNLO for better description of data at high pT")
    parser.add_argument("--dataPath", type=str, default=None, help="Access samples from this path (default reads from local machine), for eos use 'root://eoscms.cern.ch//store/cmst3/group/wmass/w-mass-13TeV/NanoAOD/'")
    parser.add_argument("--noVertexWeight", action='store_true', help="Do not apply reweighting of vertex z distribution in MC to match data")
    parser.add_argument("--validationHists", action='store_true', help="make histograms used only for validations")
    parser.add_argument("--onlyMainHistograms", action='store_true', help="Only produce some histograms, skipping (most) systematics to run faster when those are not needed")
    parser.add_argument("--met", type=str, choices=["DeepMETReso", "DeepMETResp", "RawPFMET", "DeepMETPVRobust", "DeepMETPVRobustNoPUPPI"], help="Choice of MET", default="DeepMETPVRobust")
    parser.add_argument("-o", "--outfolder", type=str, default="", help="Output folder")
    parser.add_argument("--appendOutputFile", type=str, default="", help="Append analysis output to specified output file")
    parser.add_argument("--sequentialEventLoops", action='store_true', help="Run event loops sequentially for each process to reduce memory usage")
    parser.add_argument("-e", "--era", type=str, choices=["2016PreVFP","2016PostVFP", "2017", "2018", "2017H", "2023_PUAVE1", "2023_PUAVE2", "2023_PUAVE5", "2023_PUAVE10"], help="Data set to process", default="2016PostVFP")
    parser.add_argument("--scale_A", default=1.0, type=float, help="scaling of the uncertainty on the b-field scale parameter A")
    parser.add_argument("--scale_e", default=1.0, type=float, help="scaling of the uncertainty on the material scale parameter e")
    parser.add_argument("--scale_M", default=1.0, type=float, help="scaling of the uncertainty on the alignment scale parameter M")
    parser.add_argument("--nonClosureScheme", type=str, default = "A-M-combined", choices=["none", "A-M-separated", "A-M-combined", "binned", "binned-plus-M", "A-only", "M-only"], help = "source of the Z non-closure nuisances")
    parser.add_argument("--correlatedNonClosureNP", action="store_false", help="disable the de-correlation of Z non-closure nuisance parameters after the jpsi massfit")
    parser.add_argument("--dummyNonClosureA", action="store_true", help="read values for the magnetic part of the Z non-closure from a file")
    parser.add_argument("--dummyNonClosureAMag", default=6.8e-5, type=float, help="magnitude of the dummy value for the magnetic part of the Z non-closure")
    parser.add_argument("--dummyNonClosureM", action="store_true", help="use a dummy value for the alignment part of the Z non-closure")
    parser.add_argument("--dummyNonClosureMMag", default=0., type=float, help="magnitude of the dummy value for the alignment part of the Z non-closure")
    parser.add_argument("--noScaleToData", action="store_true", help="Do not scale the MC histograms with xsec*lumi/sum(gen weights) in the postprocessing step")
    parser.add_argument("--aggregateGroups", type=str, nargs="*", default=["Diboson", "Top"], help="Sum up histograms from members of given groups in the postprocessing step")
    parser.add_argument("--muRmuFPolVarFilePath", type=str, default=f"{data_dir}/MiNNLOmuRmuFPolVar/", help="Path where input files are stored")
    parser.add_argument("--muRmuFPolVarFileTag", type=str, default="x0p50_y4p00_Fix_Trad", choices=["x0p50_y4p00_Fix_Trad","x0p50_y4p00_FixYdep_Trad"],help="Tag for input files")
    parser.add_argument("--nToysMC", type=int, help="random toys for data and MC", default=-1)
    parser.add_argument("--varianceScalingForToys", type=int, default=1, help="Scaling of variance for toys (effective mc statistics corresponds to 1./scaling)")
    parser.add_argument("--randomSeedForToys", type=int, default=0, help="random seed for toys")

    if for_reco_highPU:
        # additional arguments specific for histmaker of reconstructed objects at high pileup (mw, mz_wlike, and mz_dilepton)
        parser.add_argument("--dphiMuonMetCut", type=float, help="Threshold to cut |deltaPhi| > thr*np.pi between muon and met", default=0.0)
        parser.add_argument("--muonCorrMC", type=str, default="idealMC_lbltruth", 
            choices=["none", "trackfit_only", "trackfit_only_idealMC", "lbl", "idealMC_lbltruth", "idealMC_massfit", "idealMC_lbltruth_massfit"], 
            help="Type of correction to apply to the muons in simulation")
        parser.add_argument("--muonCorrData", type=str, default="lbl_massfit", 
            choices=["none", "trackfit_only", "lbl", "massfit", "lbl_massfit"], 
            help="Type of correction to apply to the muons in data")
        parser.add_argument("--muScaleBins", type=int, default=1, help="Number of bins for muon scale uncertainty")
        parser.add_argument("--muonScaleVariation", choices=["smearingWeightsGaus", "smearingWeightsSplines", "massWeights"], default="smearingWeightsSplines",  help="method to generate nominal muon scale variation histograms")
        parser.add_argument("--dummyMuScaleVar", action='store_true', help='Use a dummy 1e-4 variation on the muon scale instead of reading from the calibration file')
        parser.add_argument("--muonCorrMag", default=1.e-4, type=float, help="Magnitude of dummy muon momentum calibration uncertainty")
        parser.add_argument("--muonCorrEtaBins", default=1, type=int, help="Number of eta bins for dummy muon momentum calibration uncertainty")
        parser.add_argument("--excludeFlow", action='store_true', help="Excludes underflow and overflow bins in main axes")
        parser.add_argument("--biasCalibration", type=str, default=None, choices=["binned","parameterized", "A", "M"], help="Adjust central value by calibration bias hist for simulation")
        parser.add_argument("--noSmearing", action='store_true', help="Disable resolution corrections")
        # options for efficiencies
        parser.add_argument("--trackerMuons", action='store_true', help="Use tracker muons instead of global muons (need appropriate scale factors too). This is obsolete")
        parser.add_argument("--binnedScaleFactors", action='store_true', help="Use binned scale factors (different helpers)")
        parser.add_argument("--noSmooth3dsf", dest="smooth3dsf", action='store_false', help="If true (default) use smooth 3D scale factors instead of the original 2D ones (but eff. systs are still obtained from 2D version)")
        parser.add_argument("--isoEfficiencySmoothing", action='store_true', help="If isolation SF was derived from smooth efficiencies instead of direct smoothing") 
        parser.add_argument("--noScaleFactors", action="store_true", help="Don't use scale factors for efficiency (legacy option for tests)")
        parser.add_argument("--isolationDefinition", choices=["iso04vtxAgn", "iso04", "iso04chg", "iso04chgvtxAgn"], default="iso04vtxAgn",  help="Isolation type (and corresponding scale factors)")
        parser.add_argument("--isolationThreshold", default=0.15, type=float, help="Threshold for isolation cut")
        parser.add_argument("--reweightPixelMultiplicity", action='store_true', help="Reweight events based on number of valid pixel hits for the muons")
        parser.add_argument("--requirePixelHits", action='store_true', help="Require good muons to have at least one valid pixel hit used in the track refit.")
        parser.add_argument("--pixelMultiplicityStat", action='store_true', help="Include (very small) statistical uncertainties for pixel multiplicity variation")
        parser.add_argument("--vetoRecoPt", default=15, type=float, help="Lower threshold for muon pt in the veto definition")


    commonargs,_ = parser.parse_known_args()

    if for_reco_highPU:
        if commonargs.trackerMuons:
            common_logger.warning("Using tracker muons, but keep in mind that scale factors are obsolete and not recommended.")
            sfFile = "scaleFactorProduct_16Oct2022_TrackerMuonsHighPurity_vertexWeight_OSchargeExceptTracking.root"
        else:
            # note: for trigger and isolation one would actually use 3D SF vs eta-pt-ut.
            # However, even when using the 3D SF one still needs the 2D ones to read the syst/nomi ratio,
            # since the dataAltSig tag-and-probe fits were not run in 3D (it is assumed for simplicity that the syst/nomi ratio is independent from uT)
            #
            # 2D SF without ut-dependence, still needed to compute systematics when uing 3D SF
            if commonargs.era == "2016PostVFP":
                sfFile = "allSmooth_GtoHout.root" if commonargs.isolationDefinition == "iso04" else "allSmooth_GtoHout_vtxAgnIso.root"
            elif commonargs.era == "2018":
                if commonargs.isolationDefinition == "iso04":
                    raise NotImplementedError(f"For Era {commonargs.era} Isolation Definition {commonargs.isolationDefinition} is not supported")
                else:
                    sfFile = "allSmooth_2018_vtxAgnIso.root"
            elif commonargs.era == "2017": 
                if commonargs.isolationDefinition == "iso04":
                    raise NotImplementedError(f"For Era {commonargs.era} Isolation Definition {commonargs.isolationDefinition} is not supported")
                else:
                    sfFile = "allSmooth_2017_vtxAgnIso.root"
            else:
                raise NotImplementedError(f"Era {commonargs.era} is not yet supported")

        sfFile = f"{data_dir}/muonSF/{sfFile}"
    else:
        sfFile = ""

    parser.add_argument("--sfFile", type=str, help="File with muon scale factors", default=sfFile)
    parser = common_histmaker_subparsers(parser, analysis_label)

    class PrintParserAction(argparse.Action):                                            
        def __init__(self, option_strings, dest, nargs=0, **kwargs):
            if nargs != 0:
                raise ValueError('nargs for PrintParserAction must be 0 since it does not require any argument')
            super().__init__(option_strings, dest, nargs=nargs, **kwargs)
        def __call__(self, parser, namespace, values, option_string=None):
            # meant to substitute the native help message of the parser printing the whole parser with its arguments
            # needed because when we call parse_args only the options defined until there will fall in the help message
            thisLogger = logging.child_logger(__name__)
            thisLogger.warning("Printing parser with all its arguments")
            thisLogger.warning("")
            thisLogger.warning(namespace)
            thisLogger.warning("")

    parser.add_argument("--printParser", action=PrintParserAction, help="Print the whole parser with its arguments (use it as the last argument or default values might not be displayed correctly)")
    
    return parser,initargs


def plot_parser():
    parser = base_parser()
    parser.add_argument("-o", "--outpath", type=str, default=os.path.expanduser("~/www/WMassAnalysis"), help="Base path for output")
    parser.add_argument("-f", "--outfolder", type=str, default="./test", help="Subfolder for output")
    parser.add_argument("-p", "--postfix", type=str, help="Postfix for output file name")
    parser.add_argument("--cmsDecor", default="Work in progress", type=str, choices=[None,"Preliminary", "Work in progress", "Internal"], help="CMS label")
    parser.add_argument("--lumi", type=float, default=16.8, help="Luminosity used in the fit, needed to get the absolute cross section")
    parser.add_argument("--eoscp", action='store_true', help="Override use of xrdcp and use the mount instead")
    parser.add_argument("--scaleleg", type=float, default=1.0, help="Scale legend text")

    return parser


def natural_sort_key(s):
    # Sort string in a number aware way by plitting the string into alphabetic and numeric parts
    parts = re.split(r'(\d+)', s)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def natural_sort(strings):
    return sorted(strings, key=natural_sort_key)


def natural_sort_dict(dictionary):
    sorted_keys = natural_sort(dictionary.keys())
    sorted_dict = {key: dictionary[key] for key in sorted_keys}
    return sorted_dict


'''
INPUT -------------------------------------------------------------------------
|* (str) string: the string to be converted to list
|
ROUTINE -----------------------------------------------------------------------
|* converts a string to a string element in a list
|  - if not comma-separated, then the whole string becomes one single element
OUTPUT ------------------------------------------------------------------------
|* (float) string: the list-lized string
+------------------------------------------------------------------------------
'''
def string_to_list(string):
	if type(string) == str:
		string = string.split(",") # items have to be comma-separated 
		return string
	elif type(string) == list:
		return string
	else:
		raise TypeError(
            "string_to_list(): cannot convert an input that is"
            "neither a single string nor a list of strings to a list"
        )


'''
INPUT -------------------------------------------------------------------------
|* list(str): a list of strings
|
ROUTINE -----------------------------------------------------------------------
|* convert the list of string to a single string by join()
|
OUTPUT ------------------------------------------------------------------------
|* (str): the resulted string
+------------------------------------------------------------------------------
'''
def list_to_string(list_str):
	if type(list_str) == str:
		return list_str
	elif type(list_str) == list:
		string = ""
		return string.join(list_str)
	else:
		raise TypeError(
            "list_to_string(): cannot convert an input that is"
            " neither a single string or a list of strings"
        )
