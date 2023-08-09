import hist
import pathlib
import argparse
import numpy as np
import os
from utilities import logging
from enum import Enum

wremnants_dir = f"{pathlib.Path(__file__).parent}/../wremnants"
data_dir =  f"{pathlib.Path(__file__).parent}/../wremnants-data/data/"

wprocs = ["WplusmunuPostVFP", "WminusmunuPostVFP", "WminustaunuPostVFP", "WplustaunuPostVFP", 'WplusToMuNu_horace-lo-photos', 'WplusToMuNu_horace-qed', 'WplusToMuNu_horace-nlo', 'WminusToMuNu_horace-lo-photos', 'WminusToMuNu_horace-qed', 'WminusToMuNu_horace-nlo']
zprocs = ["ZmumuPostVFP", "ZtautauPostVFP", "ZmumuMiNLO", "ZmumuNNLOPS", 'ZToMuMu_horace-lo-photos', 'ZToMuMu_horace-qed', 'ZToMuMu_horace-nlo']
vprocs = wprocs+zprocs
zprocs_recoil = ["ZmumuPostVFP"]
wprocs_recoil = ["WplusmunuPostVFP", "WminusmunuPostVFP"]

wprocs_lowpu = ["WminusJetsToMuNu", "WminusJetsToENu", "WminusJetsToTauNu", "WplusJetsToMuNu", "WplusJetsToENu", "WplusJetsToTauNu"]
zprocs_lowpu = ["Zmumu", "Zee", "Ztautau"]
vprocs_lowpu = wprocs_lowpu+zprocs_lowpu
zprocs_recoil_lowpu = ["Zmumu", "Zee"]
wprocs_recoil_lowpu = ["WminusJetsToMuNu", "WminusJetsToENu", "WplusJetsToMuNu", "WplusJetsToENu"]

background_MCprocs = ["Top", "Diboson", "QCD"]
zprocs_all = zprocs_lowpu+zprocs
wprocs_all = wprocs_lowpu+wprocs
vprocs_all = vprocs_lowpu+vprocs

# unfolding axes for low pu
axis_recoil_reco_ptZ_lowpu = hist.axis.Variable([0, 5, 10, 15, 20, 30, 40, 50, 60, 75, 90, 150], name = "recoil_reco", underflow=False, overflow=True)
axis_recoil_gen_ptZ_lowpu = hist.axis.Variable([0.0, 10.0, 20.0, 40.0, 60.0, 90.0, 150], name = "recoil_gen", underflow=False, overflow=True)
axis_recoil_reco_ptW_lowpu = hist.axis.Variable([0, 5, 10, 15, 20, 30, 40, 50, 60, 75, 90, 150], name = "recoil_reco", underflow=False, overflow=True)
axis_recoil_gen_ptW_lowpu = hist.axis.Variable([0.0, 10.0, 20.0, 40.0, 60.0, 90.0, 150], name = "recoil_gen", underflow=False, overflow=True)
axis_mll_lowpu = hist.axis.Variable([60, 75] + list(range(80, 100, 2)) + [100, 105, 110, 120], name = "mll", underflow=False, overflow=False)
axis_mt_lowpu = hist.axis.Variable([0, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 110, 120, 130, 150], name = "mt", underflow=False, overflow=True)


# standard regular axes
axis_eta = hist.axis.Regular(48, -2.4, 2.4, name = "eta")
axis_pt = hist.axis.Regular(29, 26., 55., name = "pt")
#ptV_binning = [0, 2, 3, 4, 4.75, 5.5, 6.5, 8, 9, 10, 12, 14, 16, 18, 20, 23, 27, 32, 40, 55, 100]
ptV_binning = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 23, 27, 32, 40, 55, 100]
## 5% quantiles from aMC@NLO used in SMP-18-012
#ptV_10quantiles_binning = [0.0, 1.971, 2.949, 3.838, 4.733, 5.674, 6.684, 7.781, 8.979, 10.303, 11.777, 13.435, 15.332, 17.525, 20.115, 23.245, 27.173, 32.414, 40.151, 53.858, 13000.0]
## 10% quantiles from aMC@NLO used in SMP-18-012 with some rounding <== This one worked fine with toys
ptV_10quantiles_binning = [0.0, 2.95, 4.73, 6.68, 8.98, 11.78, 15.33, 20.11, 27.17, 40.15, np.inf]
absYV_binning = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4]

# categorical axes in python bindings always have an overflow bin, so use a regular
# axis for the charge
axis_charge = hist.axis.Regular(2, -2., 2., underflow=False, overflow=False, name = "charge")

down_up_axis = hist.axis.Regular(2, -2., 2., underflow=False, overflow=False, name = "downUpVar")
down_nom_up_axis = hist.axis.Regular(3, -1.5, 1.5, underflow=False, overflow=False, name = "downNomUpVar")

passIsoName = "passIso"
passMTName = "passMT"

passIso = {passIsoName: True}
failIso = {passIsoName: False}
passMT = {passMTName: True}
failMT = {passMTName: False}

axis_passIso = hist.axis.Boolean(name = passIsoName)
axis_passMT = hist.axis.Boolean(name = passMTName)

nominal_axes = [axis_eta, axis_pt, axis_charge, axis_passIso, axis_passMT]
    
# following list is used in other scripts to track what steps are charge dependent
# but assumes the corresponding efficiencies were made that way
muonEfficiency_chargeDependentSteps = ["reco", "tracking", "idip", "trigger", "antitrigger"] # antitrigger = P(failTrig|IDIP), similar to antiiso = P(failIso|trigger)
muonEfficiency_standaloneNumberOfValidHits = 1 # to use as "var >= this" (if this=0 the define for the cut is not used at all)

def getIsoMtRegionID(passIso=True, passMT=True):
    return passIso * 1 + passMT * 2

def getIsoMtRegionFromID(regionID):
    return {passIsoName : regionID & 1,
            passMTName  : regionID & 2}

def common_parser(for_reco_highPU=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("-j", "--nThreads", type=int, help="number of threads")
    parser.add_argument("-v", "--verbose", type=int, default=3, choices=[0,1,2,3,4],
                        help="Set verbosity level with logging, the larger the more verbose")
    parser.add_argument("--noColorLogger", action="store_true", help="Do not use logging with colors")
    initargs,_ = parser.parse_known_args()

    import ROOT
    if not initargs.nThreads:
        ROOT.ROOT.EnableImplicitMT()
    elif initargs.nThreads != 1:
        ROOT.ROOT.EnableImplicitMT(initargs.nThreads)
    import narf
    import wremnants
    from wremnants import theory_corrections,theory_tools

    parser.add_argument("--pdfs", type=str, nargs="*", default=["msht20"], choices=theory_tools.pdfMapExtended.keys(), help="PDF sets to produce error hists for")
    parser.add_argument("--altPdfOnlyCentral", action='store_true', help="Only store central value for alternate PDF sets")
    parser.add_argument("--maxFiles", type=int, help="Max number of files (per dataset)", default=-1)
    parser.add_argument("--filterProcs", type=str, nargs="*", help="Only run over processes matched by group name or (subset) of name", default=[])
    parser.add_argument("--excludeProcs", type=str, nargs="*", help="Exclude processes matched by group name or (subset) of name", default=[])  # no need to exclude QCD MC here, histograms can always be made, they are fast and light, so they are always available for tests
    parser.add_argument("--v8", action='store_true', help="Use NanoAODv8. Default is v9")
    parser.add_argument("-p", "--postfix", type=str, help="Postfix for output file name", default=None)
    parser.add_argument("--forceDefaultName", help="Don't modify the name", action='store_true')
    parser.add_argument("--theoryCorr", nargs="*", default=["scetlib_dyturbo"], choices=theory_corrections.valid_theory_corrections(),
        help="Apply corrections from indicated generator. First will be nominal correction.")
    parser.add_argument("--theoryCorrAltOnly", action='store_true', help="Save hist for correction hists but don't modify central weight")
    parser.add_argument("--widthVariations", action='store_true', help="Store variations of W and Z widths.")   
    parser.add_argument("--skipHelicity", action='store_true', help="Skip the qcdScaleByHelicity histogram (it can be huge)")
    parser.add_argument("--eta", nargs=3, type=float, help="Eta binning as 'nbins min max' (only uniform for now)", default=[48,-2.4,2.4])
    parser.add_argument("--pt", nargs=3, type=float, help="Pt binning as 'nbins,min,max' (only uniform for now)", default=[29,26.,55.])
    parser.add_argument("--noRecoil", action='store_true', help="Don't apply recoild correction")
    parser.add_argument("--recoilHists", action='store_true', help="Save all recoil related histograms for calibration and validation")
    parser.add_argument("--highptscales", action='store_true', help="Apply highptscales option in MiNNLO for better description of data at high pT")
    parser.add_argument("--dataPath", type=str, default=None, help="Access samples from eos")
    parser.add_argument("--noVertexWeight", action='store_true', help="Do not apply reweighting of vertex z distribution in MC to match data")
    parser.add_argument("--validationHists", action='store_true', help="make histograms used only for validations")
    parser.add_argument("--onlyMainHistograms", action='store_true', help="Only produce some histograms, skipping (most) systematics to run faster when those are not needed")
    parser.add_argument("--met", type=str, choices=["DeepMETReso", "RawPFMET"], help="MET (DeepMETReso or RawPFMET)", default="RawPFMET")                    
    parser.add_argument("-o", "--outfolder", type=str, default="", help="Output folder")
    parser.add_argument("-e", "--era", type=str, choices=["2016PreVFP","2016PostVFP"], help="Data set to process", default="2016PostVFP")
    parser.add_argument("--nonClosureScheme", type=str, default = "A-M-separated", choices=["none", "A-M-separated", "A-M-combined", "binned", "binned-plus-M"], help = "source of the Z non-closure nuisances")
    parser.add_argument("--correlatedNonClosureNP", action="store_true", help="disable the de-correlation of Z non-closure nuisance parameters after the jpsi massfit")
    parser.add_argument("--noScaleToData", action="store_true", help="Do not scale the MC histograms with xsec*lumi/sum(gen weights) in the postprocessing step")
    parser.add_argument("--aggregateGroups", type=str, nargs="*", default=["Diboson", "Top", "Wtaunu"], help="Sum up histograms from members of given groups in the postprocessing step")
    
    if for_reco_highPU:
        # additional arguments specific for histmaker of reconstructed objects at high pileup (mw, mz_wlike, and mz_dilepton)
        parser.add_argument("--dphiMuonMetCut", type=float, help="Threshold to cut |deltaPhi| > thr*np.pi between muon and met", default=0.25)
        parser.add_argument("--muonCorrMC", type=str, default="idealMC_lbltruth", 
            choices=["none", "trackfit_only", "trackfit_only_idealMC", "lbl", "idealMC_lbltruth", "idealMC_massfit", "idealMC_lbltruth_massfit"], 
            help="Type of correction to apply to the muons in simulation")
        parser.add_argument("--muonCorrData", type=str, default="lbl_massfit", 
            choices=["none", "trackfit_only", "lbl", "massfit", "lbl_massfit"], 
            help="Type of correction to apply to the muons in data")
        parser.add_argument("--muScaleBins", type=int, default=1, help="Number of bins for muon scale uncertainty")
        parser.add_argument("--muonScaleVariation", choices=["smearingWeightsGaus", "smearingWeightsSplines", "massWeights", "manualShift"], default="smearingWeightsSplines",  help="method to generate nominal muon scale variation histograms")
        parser.add_argument("--muonCorrMag", default=1.e-4, type=float, help="Magnitude of dummy muon momentum calibration uncertainty")
        parser.add_argument("--muonCorrEtaBins", default=1, type=int, help="Number of eta bins for dummy muon momentum calibration uncertainty")
        parser.add_argument("--excludeFlow", action='store_true', help="Excludes underflow and overflow bins in main axes")
        parser.add_argument("--biasCalibration", type=str, default=None, choices=["binned","parameterized", "A", "M"], help="Adjust central value by calibration bias hist for simulation")
        parser.add_argument("--smearing", action='store_true', help="Smear pT such that resolution matches data") #TODO change to --no-smearing once smearing is final
        parser.add_argument("--unfolding", action='store_true', help="Add information needed for unfolding")
        parser.add_argument("--genLevel", type=str, default='postFSR', choices=["preFSR", "postFSR"], help="Generator level definition for unfolding")
        parser.add_argument("--genBins", type=int, nargs="+", default=[3, 2], help="Number of generator level bins")
        # options for efficiencies
        parser.add_argument("--trackerMuons", action='store_true', help="Use tracker muons instead of global muons (need appropriate scale factors too). This is obsolete")
        parser.add_argument("--binnedScaleFactors", action='store_true', help="Use binned scale factors (different helpers)")
        parser.add_argument("--noSmooth3dsf", dest="smooth3dsf", action='store_false', help="If true (defaul) use smooth 3D scale factors instead of the original 2D ones (but eff. systs are still obtained from 2D version)")
        parser.add_argument("--sf2DnoUt", action='store_true', help="Use older smooth 2D scale factors with no ut dependence")
        parser.add_argument("--isoEfficiencySmoothing", action='store_true', help="If isolation SF was derived from smooth efficiencies instead of direct smoothing") 

    commonargs,_ = parser.parse_known_args()

    if for_reco_highPU:
        if commonargs.trackerMuons:
            logger.warning("Using tracker muons, but keep in mind that scale factors are obsolete and not recommended.")
            sfFile = "scaleFactorProduct_16Oct2022_TrackerMuonsHighPurity_vertexWeight_OSchargeExceptTracking.root"
        else:
            # note: any of the following file is fine for reco, tracking, and IDIP.
            # Instead, for trigger and isolation one would actually use 3D SF vs eta-pt-ut.
            # However, even when using the 3D SF one still needs the 2D ones to read the syst/nomi ratio,
            # since the dataAltSig tag-and-probe fits were not run in 3D (it is assumed for simplicity that the syst/nomi ratio is independent from uT)
            # the syst variations are the same in both files also for trigger/isolation (since they had been copied over)
            if commonargs.sf2DnoUt:
                sfFile = "allSmooth_GtoHout.root" # 2D SF without ut integration
            else:
                sfFile = "allSmooth_GtoH3Dout.root" # 2D SF from 3D with ut-integration

        sfFile = f"{data_dir}/testMuonSF/{sfFile}"
    else:
        sfFile = ""

    parser.add_argument("--sfFile", type=str, help="File with muon scale factors", default=sfFile)
        
    return parser,initargs

def common_parser_combine():
    from wremnants import theory_tools,combine_theory_helper

    parser = argparse.ArgumentParser()
    parser.add_argument("--wlike", action='store_true', help="Run W-like analysis of mZ")
    parser.add_argument("-o", "--outfolder", type=str, default=".", help="Output folder with the root file storing all histograms and datacards for single charge (subfolder WMass or ZMassWLike is created automatically inside)")
    parser.add_argument("-i", "--inputFile", type=str)
    parser.add_argument("--absolutePathInCard", action="store_true", help="In the datacard, set Absolute path for the root file where shapes are stored")
    parser.add_argument("--minnloScaleUnc", choices=["byHelicityPt", "byHelicityPtCharge", "byHelicityCharge", "byPtCharge", "byPt", "byCharge", "integrated",], default="byHelicityPt",
            help="Decorrelation for QCDscale")
    parser.add_argument("--rebin", type=int, nargs='*', default=[], help="Rebin axis by this value (default does nothing)")
    parser.add_argument("--resumUnc", default="tnp", type=str, choices=["scale", "tnp", "none"], help="Include SCETlib uncertainties")
    parser.add_argument("--npUnc", default="Delta_Lambda", type=str, choices=combine_theory_helper.TheoryHelper.valid_np_models, help="Nonperturbative uncertainty model")
    parser.add_argument("--tnpMagnitude", default=1, type=float, help="Variation size for the TNP")
    parser.add_argument("--scaleTNP", default=1, type=float, help="Scale the TNP uncertainties by this factor")
    parser.add_argument("--scalePdf", default=1, type=float, help="Scale the PDF hessian uncertainties by this factor")
    parser.add_argument("--pdfUncFromCorr", action='store_true', help="Take PDF uncertainty from correction hist (Requires having run that correction)")
    parser.add_argument("--qcdProcessName" , type=str, default="Fake",   help="Name for QCD process")
    parser.add_argument("--noStatUncFakes" , action="store_true",   help="Set bin error for QCD background templates to 0, to check MC stat uncertainties for signal only")
    parser.add_argument("--skipSignalSystOnFakes" , action="store_true", help="Do not propagate signal uncertainties on fakes, mainly for checks.")
    parser.add_argument("--noQCDscaleFakes", action="store_true",   help="Do not apply QCd scale uncertainties on fakes, mainly for debugging")
    parser.add_argument("--doStatOnly", action="store_true", default=False, help="Set up fit to get stat-only uncertainty (currently combinetf with -S 0 doesn't work)")
    parser.add_argument("-p", "--postfix", type=str, help="Postfix for output file name", default=None)
    parser.add_argument("-v", "--verbose", type=int, default=3, choices=[0,1,2,3,4],
                        help="Set verbosity level with logging, the larger the more verbose")
    parser.add_argument("--noColorLogger", action="store_true", help="Do not use logging with colors")
    parser.add_argument("--sumChannels", action='store_true', help="Only use one channel")
    parser.add_argument("--lumiScale", type=float, default=1.0, help="Rescale equivalent luminosity by this value (e.g. 10 means ten times more data and MC)")
    parser.add_argument("--addQCDMC", action="store_true", help="Include QCD MC when making datacards (otherwise by default it will always be excluded)")
    parser.add_argument("--excludeProcGroups", type=str, nargs="*", help="Don't run over processes belonging to these groups (only accepts exact group names)", default=["QCD"])
    parser.add_argument("--filterProcGroups", type=str, nargs="*", help="Only run over processes belonging to these groups", default=[])
    parser.add_argument("--muonScaleVariation", choices=["smearingWeights", "massWeights", "manualShift"], default="smearingWeights", help="the method with which the muon scale variation histograms are derived")
    return parser

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
