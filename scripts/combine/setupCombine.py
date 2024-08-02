#!/usr/bin/env python3
from wremnants import CardTool,combine_helpers, combine_theory_helper, combine_theoryAgnostic_helper, HDF5Writer, syst_tools, theory_corrections
from wremnants.syst_tools import massWeightNames
from wremnants.datasets.datagroups import Datagroups

from utilities import common, logging, boostHistHelpers as hh
from utilities.io_tools import input_tools
import argparse
import hist
import math, copy
import h5py
import narf.ioutils
import numpy as np

def make_subparsers(parser):

    parser.add_argument("--analysisMode", type=str, default=None,
                        choices=["unfolding", "theoryAgnosticNormVar", "theoryAgnosticPolVar"],
                        help="Select analysis mode to run. Default is the traditional analysis")

    tmpKnownArgs,_ = parser.parse_known_args()
    subparserName = tmpKnownArgs.analysisMode
    if subparserName is None:
        return parser

    parser.add_argument("--poiAsNoi", action='store_true', help="Make histogram to do the POIs as NOIs trick (some postprocessing will happen later in CardTool.py)")
    parser.add_argument("--forceRecoChargeAsGen", action="store_true", help="Force gen charge to match reco charge in CardTool, this only works when the reco charge is used to define the channel")
    parser.add_argument("--genAxes", type=str, default=[], nargs="+", help="Specify which gen axes should be used in unfolding/theory agnostic, if 'None', use all (inferred from metadata).")
    parser.add_argument("--priorNormXsec", type=float, default=1, help="Prior for shape uncertainties on cross sections for theory agnostic or unfolding analysis with POIs as NOIs (1 means 100\%). If negative, it will use shapeNoConstraint in the fit")
    parser.add_argument("--scaleNormXsecHistYields", type=float, default=None, help="Scale yields of histogram with cross sections variations for theory agnostic analysis with POIs as NOIs. Can be used together with --priorNormXsec")

    if "theoryAgnostic" in subparserName:
        if subparserName == "theoryAgnosticNormVar":
            parser.add_argument("--theoryAgnosticBandSize", type=float, default=1., help="Multiplier for theory-motivated band in theory agnostic analysis with POIs as NOIs.")
        elif subparserName == "theoryAgnosticPolVar":
            parser.add_argument("--noPolVarOnFake", action="store_true", help="Do not propagate POI variations to fakes")
            parser.add_argument("--symmetrizePolVar", action='store_true', help="Symmetrize up/Down variations in CardTool (using average)")
    elif "unfolding" in subparserName:
        parser = common.set_parser_default(parser, "massVariation", 10)
        parser = common.set_parser_default(parser, "hdf5", True)

    return parser


def make_parser(parser=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--outfolder", type=str, default=".", help="Output folder with the root file storing all histograms and datacards for single charge (subfolder WMass or ZMassWLike is created automatically inside)")
    parser.add_argument("-i", "--inputFile", nargs="+", type=str)
    parser.add_argument("-p", "--postfix", type=str, help="Postfix for output file name", default=None)
    parser.add_argument("-v", "--verbose", type=int, default=3, choices=[0,1,2,3,4],
                        help="Set verbosity level with logging, the larger the more verbose")
    parser.add_argument("--noColorLogger", action="store_true", help="Do not use logging with colors")
    parser.add_argument("--hdf5", action="store_true", help="Write out datacard in hdf5")
    parser.add_argument("--sparse", action="store_true", help="Write out datacard in sparse mode (only for when using hdf5)")
    parser.add_argument("--excludeProcGroups", type=str, nargs="*", help="Don't run over processes belonging to these groups (only accepts exact group names)", default=["QCD"])
    parser.add_argument("--filterProcGroups", type=str, nargs="*", help="Only run over processes belonging to these groups", default=[])
    parser.add_argument("-x", "--excludeNuisances", type=str, default="", help="Regular expression to exclude some systematics from the datacard")
    parser.add_argument("-k", "--keepNuisances", type=str, default="", help="Regular expression to keep some systematics, overriding --excludeNuisances. Can be used to keep only some systs while excluding all the others with '.*'")
    parser.add_argument("--absolutePathInCard", action="store_true", help="In the datacard, set Absolute path for the root file where shapes are stored")
    parser.add_argument("-n", "--baseName", type=str, nargs='+', default=["nominal"], help="Histogram name in the file (e.g., 'nominal')")
    parser.add_argument("--noHist", action='store_true', help="Skip the making of 2D histograms (root file is left untouched if existing)")
    parser.add_argument("--qcdProcessName" , type=str, default=None, help="Name for QCD process (by default taken from datagroups object")
    # setting on the fit behaviour
    parser.add_argument("--realData", action='store_true', help="Store real data in datacards")
    parser.add_argument("--fitvar", nargs="+", help="Variable to fit", default=["eta-pt-charge"])
    parser.add_argument("--rebin", type=int, nargs='*', default=[], help="Rebin axis by this value (default, 1, does nothing)")
    parser.add_argument("--absval", type=int, nargs='*', default=[], help="Take absolute value of axis if 1 (default, 0, does nothing)")
    parser.add_argument("--axlim", type=float, default=[], nargs='*', help="Restrict axis to this range (assumes pairs of values by axis, with trailing axes optional)")
    parser.add_argument("--rebinBeforeSelection", action='store_true', help="Rebin before the selection operation (e.g. before fake rate computation), default if after")
    parser.add_argument("--lumiScale", type=float, nargs='+', default=[1.0], help="Rescale equivalent luminosity by this value (e.g. 10 means ten times more data and MC)")
    parser.add_argument("--lumiScaleVarianceLinearly", type=str, nargs='*', default=[], choices=["data", "mc"], help="When using --lumiScale, scale variance linearly instead of quadratically, to pretend there is really more data or MC (can specify both as well). Note that statistical fluctuations in histograms cannot be lifted, so this option can lead to spurious constraints of systematic uncertainties when the argument of lumiScale is larger than unity, because bin-by-bin fluctuations will not be covered by the assumed uncertainty.")
    parser.add_argument("--sumChannels", action='store_true', help="Only use one channel")
    parser.add_argument("--fitXsec", action='store_true', help="Fit signal inclusive cross section")
    parser.add_argument("--fitWidth", action='store_true', help="Fit boson width")
    parser.add_argument("--fitMassDiff", type=str, default=None, choices=["charge", "cosThetaStarll", "eta-sign", "eta-range", "etaRegion", "etaRegionSign", "etaRegionRange"], help="Fit an additional POI for the difference in the boson mass")
    parser.add_argument("--fitMassDecorr", type=str, default=[], nargs='*', help="Decorrelate POI for given axes, fit multiple POIs for the different POIs")
    parser.add_argument("--decorrRebin", type=int, nargs='*', default=[], help="Rebin axis by this value (default, 1, does nothing)")
    parser.add_argument("--decorrAbsval", type=int, nargs='*', default=[], help="Take absolute value of axis if 1 (default, 0, does nothing)")
    parser.add_argument("--decorrAxlim", type=float, default=[], nargs='*', help="Restrict axis to this range (assumes pairs of values by axis, with trailing axes optional)")
    parser.add_argument("--fitresult", type=str, default=None ,help="Use data and covariance matrix from fitresult (for making a theory fit)")
    parser.add_argument("--noMCStat", action='store_true', help="Do not include MC stat uncertainty in covariance for theory fit (only when using --fitresult)")
    parser.add_argument("--fakerateAxes", nargs="+", help="Axes for the fakerate binning", default=["eta","pt","charge"])
    parser.add_argument("--fakeEstimation", type=str, help="Set the mode for the fake estimation", default="extended1D", choices=["closure", "simple", "extrapolate", "extended1D", "extended2D"])
    parser.add_argument("--fakeSmoothingMode", type=str, default="full", choices=["binned", "fakerate", "full"], help="Smoothing mode for fake estimate.")
    parser.add_argument("--forceGlobalScaleFakes", default=None, type=float, help="Scale the fakes  by this factor (overriding any custom one implemented in datagroups.py in the fakeSelector).")
    parser.add_argument("--fakeMCCorr", type=str, default=[None], nargs="*", choices=["none", "pt", "eta", "mt"], help="axes to apply nonclosure correction from QCD MC. Leave empty for inclusive correction, use'none' for no correction")
    parser.add_argument("--fakeSmoothingOrder", type=int, default=2, help="Order of the polynomial for the smoothing of the fake rate or full prediction, depending on the smoothing mode")
    parser.add_argument("--simultaneousABCD", action="store_true", help="Produce datacard for simultaneous fit of ABCD regions")
    parser.add_argument("--skipSumGroups", action="store_true", help="Don't add sum groups to the output to save time e.g. when computing impacts")
    parser.add_argument("--allowNegativeExpectation", action="store_true", help="Allow processes to have negative contributions")
    # settings on the nuisances itself
    parser.add_argument("--doStatOnly", action="store_true", default=False, help="Set up fit to get stat-only uncertainty (currently combinetf with -S 0 doesn't work)")
    parser.add_argument("--noTheoryUnc", action="store_true", default=False, help="Set up fit without theory uncertainties")
    parser.add_argument("--explicitSignalMCstat", action='store_true', help="Use explicit parameters for signal MC stat uncertainty. Introduces one nuisance parameter per reco bin. When unfolding, correlate bin by bin statistical uncertainty in reco bins with masked channel.")
    parser.add_argument("--minnloScaleUnc", choices=["byHelicityPt", "byHelicityPtCharge", "byHelicityCharge", "byPtCharge", "byPt", "byCharge", "integrated", "none"], default="byHelicityPt",
            help="Decorrelation for QCDscale")
    parser.add_argument("--resumUnc", default="tnp", type=str, choices=["scale", "tnp", "tnp_minnlo", "minnlo",  "none"], help="Include SCETlib uncertainties")
    parser.add_argument("--noTransitionUnc", action="store_true", help="Do not include matching transition parameter variations.")
    parser.add_argument("--npUnc", default="Delta_Lambda", type=str, choices=combine_theory_helper.TheoryHelper.valid_np_models, help="Nonperturbative uncertainty model")
    parser.add_argument("--scaleTNP", default=1, type=float, help="Scale the TNP uncertainties by this factor")
    parser.add_argument("--scalePdf", default=-1., type=float, help="Scale the PDF hessian uncertainties by this factor (by default take the value in the pdfInfo map)")
    parser.add_argument("--pdfUncFromCorr", action='store_true', help="Take PDF uncertainty from correction hist (Requires having run that correction)")
    parser.add_argument("--massVariation", type=float, default=100, help="Variation of boson mass")
    parser.add_argument("--ewUnc", type=str, nargs="*", default=["renesanceEW", "powhegFOEW"], help="Include EW uncertainty (other than pure ISR or FSR)",
        choices=[x for x in theory_corrections.valid_theory_corrections() if ("ew" in x or "EW" in x) and "ISR" not in x and "FSR" not in x])
    parser.add_argument("--isrUnc", type=str, nargs="*", default=["pythiaew_ISR",], help="Include ISR uncertainty",
        choices=[x for x in theory_corrections.valid_theory_corrections() if "ew" in x and "ISR" in x])
    parser.add_argument("--fsrUnc", type=str, nargs="*", default=["horaceqedew_FSR", "horacelophotosmecoffew_FSR"], help="Include FSR uncertainty",
        choices=[x for x in theory_corrections.valid_theory_corrections() if "ew" in x and "FSR" in x])
    parser.add_argument("--skipSignalSystOnFakes" , action="store_true", help="Do not propagate signal uncertainties on fakes, mainly for checks.")
    parser.add_argument("--noQCDscaleFakes", action="store_true", help="Do not apply QCd scale uncertainties on fakes, mainly for debugging")
    parser.add_argument("--addQCDMC", action="store_true", help="Include QCD MC when making datacards (otherwise by default it will always be excluded)")
    parser.add_argument("--muonScaleVariation", choices=["smearingWeights", "massWeights", "manualShift"], default="smearingWeights", help="the method with which the muon scale variation histograms are derived")
    parser.add_argument("--scaleMuonCorr", type=float, default=1.0, help="Scale up/down dummy muon scale uncertainty by this factor")
    parser.add_argument("--correlatedNonClosureNuisances", action='store_true', help="get systematics from histograms for the Z non-closure nuisances without decorrelation in eta and pt")
    parser.add_argument("--calibrationStatScaling", type=float, default=2.1, help="scaling of calibration statistical uncertainty")
    parser.add_argument("--resolutionStatScaling", type=float, default=10.0, help="scaling of resolution statistical uncertainty")
    parser.add_argument("--correlatedAdHocA", type=float, default=0.0, help="fully correlated ad-hoc uncertainty on b-field term A (in addition to Z pdg mass)")
    parser.add_argument("--correlatedAdHocM", type=float, default=0.0, help="fully correlated ad-hoc uncertainty on alignment term M")
    parser.add_argument("--noEfficiencyUnc", action='store_true', help="Skip efficiency uncertainty (useful for tests, because it's slow). Equivalent to --excludeNuisances '.*effSystTnP|.*effStatTnP' ")
    parser.add_argument("--effStatLumiScale", type=float, default=None, help="Rescale equivalent luminosity for efficiency stat uncertainty by this value (e.g. 10 means ten times more data from tag and probe)")
    parser.add_argument("--binnedScaleFactors", action='store_true', help="Use binned scale factors (different helpers and nuisances)")
    parser.add_argument("--isoEfficiencySmoothing", action='store_true', help="If isolation SF was derived from smooth efficiencies instead of direct smoothing")
    parser.add_argument("--logNormalWmunu", default=-1, type=float, help="Add lnN uncertainty for W signal (mainly for tests wifakes in control regions, where W is a subdominant background). If negative nothing is added")
    parser.add_argument("--logNormalFake", default=1.15, type=float, help="Specify lnN uncertainty for Fake background (for W analysis). If negative nothing is added")
    # pseudodata
    parser.add_argument("--pseudoData", type=str, nargs="+", help="Histograms to use as pseudodata")
    parser.add_argument("--pseudoDataAxes", type=str, nargs="+", default=[None], help="Variation axes to use as pseudodata for each of the histograms")
    parser.add_argument("--pseudoDataIdxs", type=str, nargs="+", default=[None], help="Variation indices to use as pseudodata for each of the histograms")
    parser.add_argument("--pseudoDataFile", type=str, help="Input file for pseudodata (if it should be read from a different file)", default=None)
    parser.add_argument("--pseudoDataProcsRegexp", type=str, default=".*", help="Regular expression for processes taken from pseudodata file (all other processes are automatically got from the nominal file). Data is excluded automatically as usual")
    parser.add_argument("--pseudoDataFakes", type=str, nargs="+", choices=["truthMC", "closure", "simple", "extrapolate", "extended1D", "extended2D"],
        help="Pseudodata for fakes are using QCD MC (closure), or different estimation methods (simple, extended1D, extended2D)")
    parser.add_argument("--addTauToSignal", action='store_true', help="Events from the same process but from tau final states are added to the signal")
    parser.add_argument("--noPDFandQCDtheorySystOnSignal", action='store_true', help="Removes PDF and theory uncertainties on signal processes")
    parser.add_argument("--recoCharge", type=str, default=["plus", "minus"], nargs="+", choices=["plus", "minus"], help="Specify reco charge to use, default uses both. This is a workaround for unfolding/theory-agnostic fit when running a single reco charge, as gen bins with opposite gen charge have to be filtered out")
    parser.add_argument("--forceConstrainMass", action='store_true', help="force mass to be constrained in fit")
    parser.add_argument("--decorMassWidth", action='store_true', help="remove width variations from mass variations")
    parser.add_argument("--muRmuFPolVar", action="store_true", help="Use polynomial variations (like in theoryAgnosticPolVar) instead of binned variations for muR and muF (of course in setupCombine these are still constrained nuisances)")
    parser.add_argument("--muRmuFFullyCorr", action="store_true", help="Use the fully correlated muR/muF variations derived from theoryAgnosticPolVar polynomial variations instead of the binned ones (by default you take the binned fully correlated variations even if you are using the polynomial variations for muR/muF, they just take care of the \"uncorrelated\" part) (this option only works with --muRmuFPolVar, it doesn't work on its own)")
    parser = make_subparsers(parser)

    return parser


def setup(args, inputFile, inputBaseName, inputLumiScale, fitvar, genvar=None, xnorm=False):

    isUnfolding = args.analysisMode == "unfolding"
    isTheoryAgnostic = args.analysisMode in ["theoryAgnosticNormVar", "theoryAgnosticPolVar"]
    isTheoryAgnosticPolVar = args.analysisMode == "theoryAgnosticPolVar"
    isPoiAsNoi = (isUnfolding or isTheoryAgnostic) and args.poiAsNoi
    isFloatingPOIsTheoryAgnostic = isTheoryAgnostic and not isPoiAsNoi
    isFloatingPOIs = (isUnfolding or isTheoryAgnostic) and not isPoiAsNoi

    # NOTE: args.filterProcGroups and args.excludeProcGroups should in principle not be used together
    #       (because filtering is equivalent to exclude something), however the exclusion is also meant to skip
    #       processes which are defined in the original process dictionary but are not supposed to be (always) run on
    if args.addQCDMC or "QCD" in args.filterProcGroups:
        logger.warning("Adding QCD MC to list of processes for the fit setup")
    elif "QCD" not in args.excludeProcGroups:
        logger.warning("Automatic removal of QCD MC from list of processes. Use --filterProcGroups 'QCD' or --addQCDMC to keep it")
        args.excludeProcGroups.append("QCD")
    filterGroup = args.filterProcGroups if args.filterProcGroups else None
    excludeGroup = args.excludeProcGroups if args.excludeProcGroups else None

    logger.debug(f"Filtering these groups of processes: {args.filterProcGroups}")
    logger.debug(f"Excluding these groups of processes: {args.excludeProcGroups}")

    datagroups = Datagroups(inputFile, excludeGroups=excludeGroup, filterGroups=filterGroup)

    if not xnorm and (args.axlim or args.rebin or args.absval):
        datagroups.set_rebin_action(fitvar, args.axlim, args.rebin, args.absval, args.rebinBeforeSelection, rename=False)

    wmass = datagroups.mode[0] == "w"
    wlike = "wlike" in datagroups.mode
    lowPU = "lowpu" in datagroups.mode
    # Detect lowpu dilepton
    dilepton = "dilepton" in datagroups.mode or any(x in ["ptll", "mll"] for x in fitvar)
    genfit = datagroups.mode == "vgen"

    if genfit:
        hasw = any("W" in x for x in args.filterProcGroups)
        hasz = any("Z" in x for x in args.filterProcGroups)
        if hasw and hasz:
            raise ValueError("Only W or Z processes are permitted in the gen fit")
        wmass = hasw

    simultaneousABCD = wmass and args.simultaneousABCD and not xnorm
    constrainMass = args.forceConstrainMass or args.fitXsec or (dilepton and not "mll" in fitvar) or genfit
    logger.debug(f"constrainMass = {constrainMass}")

    if wmass:
        base_group = "Wenu" if datagroups.flavor == "e" else "Wmunu"
    else:
        base_group = "Zee" if datagroups.flavor == "ee" else "Zmumu"

    if args.addTauToSignal:
        # add tau signal processes to signal group
        datagroups.groups[base_group].addMembers(datagroups.groups[base_group.replace("mu","tau")].members)
        datagroups.deleteGroup(base_group.replace("mu","tau"))

    if args.fitXsec:
        datagroups.unconstrainedProcesses.append(base_group)

    if lowPU and ((args.fakeEstimation != 'simple') or (args.fakeSmoothingMode != "binned")):
        logger.error(f"When running lowPU mode, fakeEstimation should be set to 'simple' and fakeSmoothingMode set to 'binned'.")

    if "run" in fitvar:
        # in case fit is split by runs/ cumulated lumi
        # run axis only exists for data, add it for MC, and scale the MC according to the luminosity fractions
        run_edges = common.run_edges
        run_edges_lumi = common.run_edges_lumi
        lumis = np.diff(run_edges_lumi)/run_edges_lumi[-1]

        datagroups.setGlobalAction(
            lambda h: h if "run" in h.axes.name else
                hh.scaleHist(
                    hh.addGenericAxis(h, hist.axis.Variable(run_edges+0.5, name = "run", underflow=False, overflow=False), add_trailing=False),
                    lumis[:,*[np.newaxis for a in h.axes]],
                )
            )

    if xnorm:
        datagroups.select_xnorm_groups(base_group)
        datagroups.globalAction = None # reset global action in case of rebinning or such
        if not isUnfolding:
            # creating the xnorm model (e.g. for the theory fit)
            if wmass and "qGen" in fitvar:
                # add gen charge as additional axis
                datagroups.groups[base_group].memberOp = [ (lambda h, m=member: hh.addGenChargeAxis(h,
                    idx=0 if "minus" in m.name else 1)) for member in datagroups.groups[base_group].members]
                xnorm_axes = ["qGen", *datagroups.gen_axes_names]
            else:
                xnorm_axes = datagroups.gen_axes_names[:]
            datagroups.setGenAxes(sum_gen_axes=[a for a in xnorm_axes if a not in fitvar])

    if isPoiAsNoi:
        constrainMass = False if isTheoryAgnostic else True
        poi_axes = datagroups.gen_axes_names if genvar is None else genvar
        # remove specified gen axes from set of gen axes in datagroups so that those are integrated over
        datagroups.setGenAxes(sum_gen_axes=[a for a in datagroups.gen_axes_names if a not in poi_axes])

        # FIXME: temporary customization of signal and out-of-acceptance process names for theory agnostic with POI as NOI
        # There might be a better way to do it more homogeneously with the rest.
        if isTheoryAgnostic:
            constrainMass = False
            hasSeparateOutOfAcceptanceSignal = False
            for g in datagroups.groups.keys():
                logger.debug(f"{g}: {[m.name for m in datagroups.groups[g].members]}")
            # check if the out-of-acceptance signal process exists as an independent process
            if any(m.name.endswith("OOA") for m in datagroups.groups[base_group].members):
                hasSeparateOutOfAcceptanceSignal = True
                if wmass:
                    # out of acceptance contribution
                    datagroups.copyGroup(base_group, f"{base_group}OOA", member_filter=lambda x: x.name.endswith("OOA"))
                    datagroups.groups[base_group].deleteMembers([m for m in datagroups.groups[base_group].members if m.name.endswith("OOA")])
                else:
                    # out of acceptance contribution
                    datagroups.copyGroup(base_group, f"{base_group}OOA", member_filter=lambda x: x.name.endswith("OOA"))
                    datagroups.groups[base_group].deleteMembers([m for m in datagroups.groups[base_group].members if m.name.endswith("OOA")])
            if any(x.endswith("OOA") for x in args.excludeProcGroups) and hasSeparateOutOfAcceptanceSignal:
                datagroups.deleteGroup(f"{base_group}OOA") # remove out of acceptance signal
    elif isUnfolding or isTheoryAgnostic:
        constrainMass = False if isTheoryAgnostic else True
        datagroups.setGenAxes(genvar)
        logger.info(f"GEN axes are {genvar}")
        if wmass and "qGen" in datagroups.gen_axes_names:
            # gen level bins, split by charge
            if "minus" in args.recoCharge:
                datagroups.defineSignalBinsUnfolding(base_group, f"W_qGen0", member_filter=lambda x: x.name.startswith("Wminus") and not x.name.endswith("OOA"), axesToRead=[ax for ax in datagroups.gen_axes_names if ax!="qGen"])
            if "plus" in args.recoCharge:
                datagroups.defineSignalBinsUnfolding(base_group, f"W_qGen1", member_filter=lambda x: x.name.startswith("Wplus") and not x.name.endswith("OOA"), axesToRead=[ax for ax in datagroups.gen_axes_names if ax!="qGen"])
        else:
            datagroups.defineSignalBinsUnfolding(base_group, base_group[0], member_filter=lambda x: not x.name.endswith("OOA"))

        # out of acceptance contribution
        to_del = [m for m in datagroups.groups[base_group].members if not m.name.endswith("OOA")]
        if len(datagroups.groups[base_group].members) == len(to_del):
            datagroups.deleteGroup(base_group)
        else:
            datagroups.groups[base_group].deleteMembers(to_del)

    if args.qcdProcessName:
        datagroups.fakeName = args.qcdProcessName

    if wmass and not xnorm:
        datagroups.fakerate_axes=args.fakerateAxes
        datagroups.set_histselectors(
            datagroups.getNames(), inputBaseName, mode=args.fakeEstimation,
            smoothing_mode=args.fakeSmoothingMode, smoothingOrderFakerate=args.fakeSmoothingOrder,
            mcCorr=args.fakeMCCorr,
            integrate_x="mt" not in fitvar,
            simultaneousABCD=simultaneousABCD, forceGlobalScaleFakes=args.forceGlobalScaleFakes,
            )

    # Start to create the CardTool object, customizing everything
    cardTool = CardTool.CardTool(xnorm=xnorm, simultaneousABCD=simultaneousABCD, real_data=args.realData)
    cardTool.setDatagroups(datagroups)

    logger.debug(f"Making datacards with these processes: {cardTool.getProcesses()}")
    if args.absolutePathInCard:
        cardTool.setAbsolutePathShapeInCard()

    if simultaneousABCD:
        # In case of ABCD we need to have different fake processes for e and mu to have uncorrelated uncertainties
        cardTool.setFakeName(datagroups.fakeName + (datagroups.flavor if datagroups.flavor else ""))
        cardTool.unroll=True

        # add ABCD regions to fit
        mtName = "mt" if "mt" in fitvar else common.passMTName
        if common.passIsoName not in fitvar:
            fitvar = [*fitvar, common.passIsoName]
        if mtName not in fitvar:
            fitvar = [*fitvar, mtName]

    cardTool.setFitAxes(fitvar)

    if args.sumChannels or xnorm or dilepton or simultaneousABCD or "charge" not in fitvar:
        cardTool.setWriteByCharge(False)
    else:
        cardTool.setChannels(args.recoCharge)
        if (isUnfolding or isTheoryAgnostic) and args.forceRecoChargeAsGen:
            cardTool.setExcludeProcessForChannel("plus", ".*qGen0")
            cardTool.setExcludeProcessForChannel("minus", ".*qGen1")

    if xnorm:
        histName = "xnorm"
        cardTool.setHistName(histName)
        cardTool.setNominalName(histName)
    else:
        cardTool.setHistName(inputBaseName)
        cardTool.setNominalName(inputBaseName)
        
    # define sumGroups for integrated cross section
    if isFloatingPOIs and not args.skipSumGroups:
        # TODO: make this less hardcoded to filter the charge (if the charge is not present this will duplicate things)
        if isTheoryAgnostic and wmass and "qGen" in datagroups.gen_axes:
            if "plus" in args.recoCharge:
                cardTool.addPOISumGroups(genCharge="qGen1")
            if "minus" in args.recoCharge:
                cardTool.addPOISumGroups(genCharge="qGen0")
        else:
            cardTool.addPOISumGroups()

    if args.noHist:
        cardTool.skipHistograms()
    cardTool.setSpacing(28)
    label = 'W' if wmass else 'Z'
    cardTool.setCustomSystGroupMapping({
        "theoryTNP" : f".*resum.*|.*TNP.*|mass.*{label}.*",
        "resumTheory" : f".*scetlib.*|.*resum.*|.*TNP.*|mass.*{label}.*",
        "allTheory" : f".*scetlib.*|pdf.*|.*QCD.*|.*resum.*|.*TNP.*|mass.*{label}.*",
        "ptTheory" : f".*QCD.*|.*resum.*|.*TNP.*|mass.*{label}.*",
    })
    cardTool.setCustomSystForCard(args.excludeNuisances, args.keepNuisances)

    if args.pseudoData:
        cardTool.setPseudodata(args.pseudoData, args.pseudoDataAxes, args.pseudoDataIdxs, args.pseudoDataProcsRegexp)
        if args.pseudoDataFile:
            # FIXME: should make sure to apply the same customizations as for the nominal datagroups so far
            pseudodataGroups = Datagroups(args.pseudoDataFile, excludeGroups=excludeGroup, filterGroups=filterGroup)
            if not xnorm and (args.axlim or args.rebin or args.absval):
                pseudodataGroups.set_rebin_action(fitvar, args.axlim, args.rebin, args.absval, rename=False)

            if wmass and not xnorm:
                pseudodataGroups.fakerate_axes=args.fakerateAxes
                pseudodataGroups.set_histselectors(
                    pseudodataGroups.getNames(), inputBaseName, mode=args.fakeEstimation,
                    smoothing_mode=args.fakeSmoothingMode, smoothingOrderFakerate=args.fakeSmoothingOrder,
                    mcCorr=args.fakeMCCorr,
                    integrate_x="mt" not in fitvar,
                    simultaneousABCD=simultaneousABCD, forceGlobalScaleFakes=args.forceGlobalScaleFakes,
                )

            cardTool.setPseudodataDatagroups(pseudodataGroups)
    if args.pseudoDataFakes:
        cardTool.setPseudodata(args.pseudoDataFakes)
        # pseudodata for fakes, either using data or QCD MC
        if "closure" in args.pseudoDataFakes or "truthMC" in args.pseudoDataFakes:
            filterGroupFakes = ["QCD"]
            pseudodataGroups = Datagroups(args.pseudoDataFile if args.pseudoDataFile else inputFile, filterGroups=filterGroupFakes)
            pseudodataGroups.fakerate_axes=args.fakerateAxes
            pseudodataGroups.copyGroup("QCD", "QCDTruth")
            pseudodataGroups.set_histselectors(
                pseudodataGroups.getNames(), inputBaseName, mode=args.fakeEstimation, fake_processes=["QCD",],
                smoothing_mode=args.fakeSmoothingMode, smoothingOrderFakerate=args.fakeSmoothingOrder,
                mcCorr=args.fakeMCCorr,
                simultaneousABCD=simultaneousABCD, forceGlobalScaleFakes=args.forceGlobalScaleFakes,
                )
        else:
            pseudodataGroups = Datagroups(args.pseudoDataFile if args.pseudoDataFile else inputFile, excludeGroups=excludeGroup, filterGroups=filterGroup)
            pseudodataGroups.fakerate_axes=args.fakerateAxes
        if args.axlim or args.rebin or args.absval:
            pseudodataGroups.set_rebin_action(fitvar, args.axlim, args.rebin, args.absval, rename=False)

        cardTool.setPseudodataDatagroups(pseudodataGroups)

    cardTool.setLumiScale(inputLumiScale, args.lumiScaleVarianceLinearly)

    if not isTheoryAgnostic:
        logger.info(f"cardTool.allMCProcesses(): {cardTool.allMCProcesses()}")

    passSystToFakes = wmass and not (simultaneousABCD or xnorm or args.skipSignalSystOnFakes) and cardTool.getFakeName() != "QCD" and (excludeGroup != None and cardTool.getFakeName() not in excludeGroup) and (filterGroup == None or cardTool.getFakeName() in filterGroup)

    # TODO: move to a common place if it is  useful
    def assertSample(name, startsWith=["W", "Z"], excludeMatch=[]):
        return any(name.startswith(init) for init in startsWith) and all(excl not in name for excl in excludeMatch)

    def assertSampleWithOOA(name, startsWith=["W", "Z"], excludeMatch=[]):
        return any(name.startswith(init) for init in startsWith) and name.endswith("OOA") and all(excl not in name for excl in excludeMatch)

    dibosonMatch = ["WW", "WZ", "ZZ"]
    WMatch = ["W"] # TODO: the name of out-of-acceptance might be changed at some point, maybe to WmunuOutAcc, so W will match it as well (and can exclude it using "OutAcc" if needed)
    ZMatch = ["Z"]
    signalMatch = WMatch if wmass else ZMatch
    nonSignalMatch = ZMatch if wmass else WMatch

    wlike_vetoValidation = wlike and input_tools.args_from_metadata(cardTool, "validateVetoSF")
    cardTool.addProcessGroup("single_v_samples", lambda x: assertSample(x, startsWith=[*WMatch, *ZMatch], excludeMatch=dibosonMatch))
    # TODO consistently treat low mass drell yan as signal across full analysis
    cardTool.addProcessGroup("z_samples", lambda x: assertSample(x, startsWith=ZMatch, excludeMatch=dibosonMatch))
    if wmass or wlike_vetoValidation:
        cardTool.addProcessGroup("Zveto_samples", lambda x: assertSample(x, startsWith=[*ZMatch, "DYlowMass"], excludeMatch=dibosonMatch))
    if wmass:
        cardTool.addProcessGroup("w_samples", lambda x: assertSample(x, startsWith=WMatch, excludeMatch=dibosonMatch))
        cardTool.addProcessGroup("wtau_samples", lambda x: assertSample(x, startsWith=["Wtaunu"]))
        if not xnorm:
            cardTool.addProcessGroup("single_v_nonsig_samples", lambda x: assertSample(x, startsWith=ZMatch, excludeMatch=dibosonMatch))
    cardTool.addProcessGroup("single_vmu_samples",    lambda x: assertSample(x, startsWith=[*WMatch, *ZMatch], excludeMatch=[*dibosonMatch, "tau"]))
    cardTool.addProcessGroup("signal_samples",        lambda x: assertSample(x, startsWith=signalMatch,        excludeMatch=[*dibosonMatch, "tau"]))
    cardTool.addProcessGroup("signal_samples_inctau", lambda x: assertSample(x, startsWith=signalMatch,        excludeMatch=[*dibosonMatch]))
    cardTool.addProcessGroup("nonsignal_samples_inctau", lambda x: assertSample(x, startsWith=nonSignalMatch,        excludeMatch=[*dibosonMatch]))
    cardTool.addProcessGroup("MCnoQCD", lambda x: x not in ["QCD", "Data"] + (["Fake"] if simultaneousABCD else []) )
    # FIXME/FOLLOWUP: the following groups may actually not exclude the OOA when it is not defined as an independent process with specific name
    cardTool.addProcessGroup("signal_samples_noOutAcc",        lambda x: assertSample(x, startsWith=signalMatch, excludeMatch=[*dibosonMatch, "tau", "OOA"]))
    cardTool.addProcessGroup("nonsignal_samples_noOutAcc",     lambda x: assertSample(x, startsWith=nonSignalMatch, excludeMatch=[*dibosonMatch, "tau", "OOA"]))
    cardTool.addProcessGroup("signal_samples_inctau_noOutAcc", lambda x: assertSample(x, startsWith=signalMatch, excludeMatch=[*dibosonMatch, "OOA"]))
    cardTool.addProcessGroup("nonsignal_samples_inctau_noOutAcc", lambda x: assertSample(x, startsWith=nonSignalMatch, excludeMatch=[*dibosonMatch, "OOA"]))
    # FIX for theoryAgnosticPolVar when using MiNNLO polynomial variations (we would still need these for the OOA)
    cardTool.addProcessGroup("signal_samples_OOA",        lambda x: assertSampleWithOOA(x, startsWith=signalMatch, excludeMatch=[*dibosonMatch, "tau"]))
    cardTool.addProcessGroup("nonsignal_samples_OOA",     lambda x: assertSampleWithOOA(x, startsWith=nonSignalMatch, excludeMatch=[*dibosonMatch, "tau"]))
    cardTool.addProcessGroup("signal_samples_inctau_OOA", lambda x: assertSampleWithOOA(x, startsWith=signalMatch, excludeMatch=[*dibosonMatch]))
    cardTool.addProcessGroup("nonsignal_samples_inctau_OOA", lambda x: assertSampleWithOOA(x, startsWith=nonSignalMatch, excludeMatch=[*dibosonMatch]))

    if not (isTheoryAgnostic or isUnfolding) :
        logger.info(f"All MC processes {cardTool.procGroups['MCnoQCD']}")
        logger.info(f"Single V samples: {cardTool.procGroups['single_v_samples']}")
        if wmass and not xnorm:
            logger.info(f"Single V no signal samples: {cardTool.procGroups['single_v_nonsig_samples']}")
        logger.info(f"Signal samples: {cardTool.procGroups['signal_samples']}")

    signal_samples_forMass = ["signal_samples_inctau"]
    if isFloatingPOIsTheoryAgnostic:
        logger.error("Temporarily not using mass weights for Wtaunu. Please update when possible")
        signal_samples_forMass = ["signal_samples"]

    if simultaneousABCD:
        # Fakerate A/B = C/D
        fakerate_axes_syst = [f"_{n}" for n in args.fakerateAxes]
        cardTool.addSystematic(
            name="nominal",
            rename=f"{cardTool.getFakeName()}Rate",
            processes=cardTool.getFakeName(),
            group="Fake",
            splitGroup = {"experiment": ".*"},
            systNamePrepend=f"{cardTool.getFakeName()}Rate",
            noConstraint=True,
            mirror=True,
            systAxes=fakerate_axes_syst,
            action=syst_tools.make_fakerate_variation,
            actionArgs=dict(
                fakerate_axes=args.fakerateAxes,
                fakerate_axes_syst=fakerate_axes_syst),
        )
        # Normalization parameters
        fakenorm_axes = [*args.fakerateAxes, mtName]
        fakenorm_axes_syst = [f"_{n}" for n in fakenorm_axes]
        cardTool.addSystematic(
            name="nominal",
            rename=f"{cardTool.getFakeName()}Norm",
            processes=cardTool.getFakeName(),
            group="Fake",
            splitGroup = {"experiment": ".*"},
            systNamePrepend=f"{cardTool.getFakeName()}Norm",
            noConstraint=True,
            mirror=True,
            systAxes=fakenorm_axes_syst,
            action=lambda h:
                hh.addHists(h,
                    hh.expand_hist_by_duplicate_axes(h, fakenorm_axes, fakenorm_axes_syst),
                    scale2=0.1)
        )

    decorwidth = args.decorMassWidth or args.fitWidth
    massWeightName = "massWeight_widthdecor" if decorwidth else "massWeight"
    if not (args.doStatOnly and constrainMass):
        if args.massVariation != 0:
            if len(args.fitMassDecorr)==0:
                cardTool.addSystematic(f"{massWeightName}{label}",
                                    processes=signal_samples_forMass,
                                    group=f"massShift",
                                    noi=not constrainMass,
                                    skipEntries=massWeightNames(proc=label, exclude=args.massVariation),
                                    mirror=False,
                                    noConstraint=not constrainMass,
                                    systAxes=["massShift"],
                                    passToFakes=passSystToFakes
                )
            else:
                suffix = "".join([a.capitalize() for a in args.fitMassDecorr])
                new_names = [f"{a}_decorr" for a in args.fitMassDecorr]
                cardTool.addSystematic(
                    name=f"{massWeightName}{label}",
                    processes=signal_samples_forMass,
                    rename=f"massDecorr{suffix}{label}",
                    group=f"massDecorr{label}",
                    # systNameReplace=[("Shift",f"Diff{suffix}")],
                    skipEntries=[(x, *[-1]*len(args.fitMassDecorr)) for x in massWeightNames(proc=label, exclude=args.massVariation)],
                    noi=not constrainMass,
                    noConstraint=not constrainMass,
                    mirror=False,
                    systAxes=["massShift", *new_names],
                    passToFakes=passSystToFakes,
                    actionRequiresNomi=True,
                    action=syst_tools.decorrelateByAxes,
                    actionArgs=dict(
                        axesToDecorrNames=args.fitMassDecorr, newDecorrAxesNames=new_names, axlim=args.decorrAxlim, rebin=args.decorrRebin, absval=args.decorrAbsval)
                )

        if args.fitMassDiff:
            suffix = "".join([a.capitalize() for a in args.fitMassDiff.split("-")])
            combine_helpers.add_mass_diff_variations(
                cardTool, 
                args.fitMassDiff,
                name=f"{massWeightName}{label}",
                processes=signal_samples_forMass, 
                constrain=constrainMass,
                suffix=suffix, 
                label=label,
                passSystToFakes=passSystToFakes,
            )

    # this appears within doStatOnly because technically these nuisances should be part of it
    if isPoiAsNoi:
        if isTheoryAgnostic:
            theoryAgnostic_helper = combine_theoryAgnostic_helper.TheoryAgnosticHelper(cardTool, externalArgs=args)
            if isTheoryAgnosticPolVar:
                theoryAgnostic_helper.configure_polVar(label,
                                                       passSystToFakes,
                                                       hasSeparateOutOfAcceptanceSignal,
                                                       )
            else:
                theoryAgnostic_helper.configure_normVar(label,
                                                        passSystToFakes,
                                                        poi_axes,
                                                        )
            theoryAgnostic_helper.add_theoryAgnostic_uncertainty()

        elif isUnfolding:
            noi_args = dict(
                group=f"normXsec{label}",
                passToFakes=passSystToFakes,
                name=f"yieldsUnfolding",
                systAxes=poi_axes,
                processes=["signal_samples"],
                noConstraint=True,
                noi=True,
                mirror=True,
                scale=1 if args.priorNormXsec < 0 else args.priorNormXsec, # histogram represents an (args.priorNormXsec*100)% prior
                systAxesFlow=[a for a in poi_axes if a in ["ptGen"]], # use underflow/overflow bins for ptGen
                labelsByAxis=poi_axes,
            )
            if wmass:
                # add two sets of systematics, one for each charge
                cardTool.addSystematic(**noi_args,
                    rename=f"noiWminus",
                    baseName=f"W_qGen0",
                    preOpMap={
                        m.name: (lambda h: hh.addHists(h[{ax: hist.tag.Slicer()[::hist.sum] for ax in poi_axes}], h, scale2=args.scaleNormXsecHistYields))
                            if "minus" in m.name else (lambda h: h[{ax: hist.tag.Slicer()[::hist.sum] for ax in poi_axes}])
                        for g in cardTool.procGroups["signal_samples"] for m in cardTool.datagroups.groups[g].members},
                )
                cardTool.addSystematic(**noi_args,
                    rename=f"noiWplus",
                    baseName=f"W_qGen1",
                    preOpMap={
                        m.name: (lambda h: hh.addHists(h[{ax: hist.tag.Slicer()[::hist.sum] for ax in poi_axes}], h, scale2=args.scaleNormXsecHistYields))
                            if "plus" in m.name else (lambda h: h[{ax: hist.tag.Slicer()[::hist.sum] for ax in poi_axes}])
                        for g in cardTool.procGroups["signal_samples"] for m in cardTool.datagroups.groups[g].members},
                )
            else:
                cardTool.addSystematic(**noi_args,
                    baseName=f"{label}_",
                    preOpMap={
                        m.name: (lambda h: hh.addHists(
                            h[{**{ax: hist.tag.Slicer()[::hist.sum] for ax in poi_axes}, "acceptance": hist.tag.Slicer()[::hist.sum]}], h[{"acceptance":True}], scale2=args.scaleNormXsecHistYields))
                        for g in cardTool.procGroups["signal_samples"] for m in cardTool.datagroups.groups[g].members},
                )

    if args.muRmuFPolVar and not isPoiAsNoi:
        muRmuFPolVar_helper = combine_theoryAgnostic_helper.TheoryAgnosticHelper(cardTool, externalArgs=args)
        muRmuFPolVar_helper.configure_polVar(label,
                                             passSystToFakes,
                                             False,
                                             )
        muRmuFPolVar_helper.add_theoryAgnostic_uncertainty()

    if args.explicitSignalMCstat:
        if not xnorm:
            recovar = fitvar
        else:
            # need to find the reco variables that correspond to the masked channel
            idx = args.inputFile.index(inputFile) 
            recovar = args.fitvar[idx].split("-")

        if xnorm and not args.fitresult:
            # use variations from reco histogram and apply them to xnorm 
            source = ("nominal", "yieldsUnfolding")
        else:
            None

        combine_helpers.add_explicit_MCstat(cardTool, recovar, samples="signal_samples", wmass=wmass, source=source)

    if args.doStatOnly:
        # print a card with only mass weights
        logger.info("Using option --doStatOnly: the card was created with only mass nuisance parameter")
        return cardTool

    if not args.noTheoryUnc:
        if wmass and not xnorm:
            cardTool.addSystematic(f"massWeightZ",
                                    processes=['single_v_nonsig_samples'],
                                    group="ZmassAndWidth",
                                    splitGroup = {"theory": ".*"},
                                    skipEntries=massWeightNames(proc="Z", exclude=2.1),
                                    mirror=False,
                                    noConstraint=False,
                                    systAxes=["massShift"],
                                    passToFakes=passSystToFakes,
            )

        # Experimental range
        #widthVars = (42, ['widthW2p043GeV', 'widthW2p127GeV']) if wmass else (2.3, ['widthZ2p4929GeV', 'widthZ2p4975GeV'])
        # Variation from EW fit (mostly driven by alphas unc.)
        cardTool.addSystematic("widthWeightZ",
                                rename="WidthZ0p8MeV",
                                processes= ['single_v_nonsig_samples'] if wmass else signal_samples_forMass,
                                action=lambda h: h[{"width" : ['widthZ2p49333GeV', 'widthZ2p49493GeV']}],
                                group="ZmassAndWidth" if wmass else "widthZ",
                                splitGroup = {"theory": ".*"},
                                mirror=False,
                                noi=args.fitWidth if not wmass else False,
                                noConstraint=args.fitWidth if not wmass else False,
                                systAxes=["width"],
                                outNames=["widthZDown", "widthZUp"],
                                passToFakes=passSystToFakes,
        )
        if wmass:
            cardTool.addSystematic("widthWeightW",
                                    rename="WidthW0p6MeV",
                                    processes=signal_samples_forMass,
                                    action=lambda h: h[{"width" : ['widthW2p09053GeV', 'widthW2p09173GeV']}],
                                    group="widthW",
                                    splitGroup = {"theory": ".*"},
                                    mirror=False,
                                    noi=args.fitWidth,
                                    noConstraint=args.fitWidth,
                                    systAxes=["width"],
                                    outNames=["widthWDown", "widthWUp"],
                                    passToFakes=passSystToFakes,
            )

        cardTool.addSystematic(f"sin2thetaWeightZ",
                                rename=f"Sin2thetaZ0p00003",
                                processes= ['z_samples'],
                                action=lambda h: h[{"sin2theta" : ['sin2thetaZ0p23151', 'sin2thetaZ0p23157']}],
                                group=f"sin2thetaZ",
                                mirror=False,
                                systAxes=["sin2theta"],
                                outNames=[f"sin2thetaZDown", f"sin2thetaZUp"],
                                passToFakes=passSystToFakes,
        )

        combine_helpers.add_electroweak_uncertainty(cardTool, [*args.ewUnc, *args.fsrUnc, *args.isrUnc],
            samples="single_v_samples", flavor=datagroups.flavor, passSystToFakes=passSystToFakes)

        to_fakes = passSystToFakes and not args.noQCDscaleFakes and not xnorm

        theory_helper = combine_theory_helper.TheoryHelper(cardTool, args, hasNonsigSamples=(wmass and not xnorm))
        theory_helper.configure(resumUnc=args.resumUnc,
            transitionUnc = not args.noTransitionUnc,
            propagate_to_fakes=to_fakes,
            np_model=args.npUnc,
            tnp_scale = args.scaleTNP,
            mirror_tnp=False,
            pdf_from_corr=args.pdfUncFromCorr,
            scale_pdf_unc=args.scalePdf,
            minnlo_unc=args.minnloScaleUnc,
        )

        theorySystSamples = ["signal_samples_inctau"]
        if wmass:
            if args.noPDFandQCDtheorySystOnSignal:
                theorySystSamples = ["wtau_samples"]
            theorySystSamples.append("single_v_nonsig_samples")
        if xnorm:
            theorySystSamples = ["signal_samples"]

        theory_helper.add_all_theory_unc(theorySystSamples, skipFromSignal=args.noPDFandQCDtheorySystOnSignal)

    if xnorm or genfit:
        return cardTool

    # Below: experimental uncertainties
    if not lowPU: # lowPU does not include PhotonInduced as a process. skip it:
        cardTool.addLnNSystematic("CMS_PhotonInduced", processes=["PhotonInduced"], size=2.0, group="CMS_background", splitGroup = {"experiment": ".*"},)
    if wmass:
        if args.logNormalWmunu > 0.0:
            cardTool.addLnNSystematic(f"CMS_Wmunu", processes=["Wmunu"], size=args.logNormalWmunu, group="CMS_background", splitGroup = {"experiment": ".*"})
        if args.logNormalFake > 0.0:
            cardTool.addLnNSystematic(f"CMS_{cardTool.getFakeName()}", processes=[cardTool.getFakeName()], size=args.logNormalFake, group="Fake", splitGroup = {"experiment": ".*"})
        cardTool.addLnNSystematic("CMS_Top", processes=["Top"], size=1.06, group="CMS_background", splitGroup = {"experiment": ".*"})
        cardTool.addLnNSystematic("CMS_VV", processes=["Diboson"], size=1.16, group="CMS_background", splitGroup = {"experiment": ".*"})
        cardTool.addSystematic("luminosity",
                                processes=['MCnoQCD'],
                                outNames=["lumiDown", "lumiUp"],
                                group="luminosity",
                                splitGroup = {"experiment": ".*"},
                                systAxes=["downUpVar"],
                                labelsByAxis=["downUpVar"],
                                passToFakes=passSystToFakes)
    else:
        cardTool.addLnNSystematic("CMS_background", processes=["Other"], size=1.15, group="CMS_background", splitGroup = {"experiment": ".*"},)
        cardTool.addLnNSystematic("lumi", processes=['MCnoQCD'], size=1.017 if lowPU else 1.012, group="luminosity", splitGroup = {"experiment": ".*"},)

    if cardTool.getFakeName() != "QCD" and cardTool.getFakeName() in datagroups.groups.keys() and not xnorm and (args.fakeSmoothingMode != "binned" or (args.fakeEstimation in ["extrapolate"] and "mt" in fitvar)):
        syst_axes = ["eta", "charge"] if (args.fakeSmoothingMode != "binned" or args.fakeEstimation not in ["extrapolate"]) else ["eta", "pt", "charge"]
        info=dict(
            name=inputBaseName, 
            group="Fake",
            processes=cardTool.getFakeName(),
            noConstraint=False,
            mirror=False,
            scale=1,
            applySelection=False, # don't apply selection, all regions will be needed for the action
            action=cardTool.datagroups.groups[cardTool.getFakeName()].histselector.get_hist,
            systAxes=[f"_{x}" for x in syst_axes if x in args.fakerateAxes]+["_param", "downUpVar"])
        subgroup = f"{cardTool.getFakeName()}Rate"
        cardTool.addSystematic(**info,
            rename=subgroup,
            splitGroup = {subgroup: f".*", "experiment": ".*"},
            systNamePrepend=subgroup,
            actionArgs=dict(variations_smoothing=True),
        )
        if args.fakeEstimation in ["extended2D",]:
            subgroup = f"{cardTool.getFakeName()}Shape"
            cardTool.addSystematic(**info,
                rename=subgroup,
                splitGroup = {subgroup: f".*", "experiment": ".*"},
                systNamePrepend=subgroup,
                actionArgs=dict(variations_scf=True),
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
            allEffTnP = [f"effStatTnP_sf_{eff}" for eff in effStatTypes] + ["effSystTnP"]
            for name in allEffTnP:
                if "Syst" in name:
                    axes = ["reco-tracking-idip-trigger-iso", "n_syst_variations"]
                    axlabels = ["WPSYST", "_etaDecorr"]
                    nameReplace = [("WPSYST0", "reco"), ("WPSYST1", "tracking"), ("WPSYST2", "idip"), ("WPSYST3", "trigger"), ("WPSYST4", "iso"), ("effSystTnP", "effSyst"), ("etaDecorr0", "fullyCorr") ]
                    scale = 1
                    mirror = True
                    mirrorDownVarEqualToNomi=False
                    groupName = "muon_eff_syst"
                    splitGroupDict = {f"{groupName}_{x}" : f".*effSyst.*{x}" for x in list(effTypesNoIso + ["iso"])}
                    splitGroupDict["muon_eff_all"] = ".*"
                else:
                    nameReplace = [] if any(x in name for x in chargeDependentSteps) else [("q0", "qall")] # for iso change the tag id with another sensible label
                    mirror = True
                    mirrorDownVarEqualToNomi=False
                    if args.binnedScaleFactors:
                        axes = ["SF eta", "nPtBins", "SF charge"]
                    else:
                        axes = ["SF eta", "nPtEigenBins", "SF charge"]
                    axlabels = ["eta", "pt", "q"]
                    nameReplace = nameReplace + [("effStatTnP_sf_", "effStat_")]
                    scale = 1
                    groupName = "muon_eff_stat"
                    splitGroupDict = {f"{groupName}_{x}" : f".*effStat.*{x}" for x in effStatTypes}
                    splitGroupDict["muon_eff_all"] = ".*"
                if args.effStatLumiScale and "Syst" not in name:
                    scale /= math.sqrt(args.effStatLumiScale)

                cardTool.addSystematic(
                    name,
                    mirror=mirror,
                    mirrorDownVarEqualToNomi=mirrorDownVarEqualToNomi,
                    group=groupName,
                    splitGroup={**splitGroupDict, "experiment": ".*"},
                    systAxes=axes,
                    labelsByAxis=axlabels,
                    baseName=name+"_",
                    processes=['MCnoQCD'],
                    passToFakes=passSystToFakes,
                    systNameReplace=nameReplace,
                    scale=scale,
                )
                # now add other systematics if present
                if name=="effSystTnP":
                    for es in common.muonEfficiency_altBkgSyst_effSteps:
                        cardTool.addSystematic(
                            f"effSystTnP_altBkg_{es}",
                            mirror=mirror,
                            mirrorDownVarEqualToNomi=mirrorDownVarEqualToNomi,
                            group=f"muon_eff_syst_{es}_altBkg",
                            splitGroup={groupName: ".*", "muon_eff_all" : ".*", "experiment": ".*"},
                            systAxes = ["n_syst_variations"],
                            labelsByAxis = [f"{es}_altBkg_etaDecorr"],
                            baseName=name+"_",
                            processes=['MCnoQCD'],
                            passToFakes=passSystToFakes,
                            systNameReplace=[("effSystTnP", "effSyst"), ("etaDecorr0", "fullyCorr")],
                            scale=scale,
                        )
            if wmass or wlike_vetoValidation:
                useGlobalOrTrackerVeto = input_tools.args_from_metadata(cardTool, "useGlobalOrTrackerVeto")
                useRefinedVeto = input_tools.args_from_metadata(cardTool, "useRefinedVeto")
                allEffTnP_veto = ["effStatTnP_veto_sf", "effSystTnP_veto"]
                for name in allEffTnP_veto:
                    if "Syst" in name:
                        if useGlobalOrTrackerVeto:
                            axes = ["veto_reco-veto_tracking-veto_idip-veto_trackerreco-veto_trackertracking", "n_syst_variations"]
                        else:
                            if useRefinedVeto:
                                axes = ["vetoreco-vetotracking-vetoidip", "n_syst_variations"]
                            else:
                                axes = ["veto_reco-veto_tracking-veto_idip", "n_syst_variations"]
                        axlabels = ["WPSYST", "_etaDecorr"]
                        if useGlobalOrTrackerVeto:
                            nameReplace = [("WPSYST0", "reco"), ("WPSYST1", "tracking"), ("WPSYST2", "idip"), ("WPSYST3", "trackerreco"), ("WPSYST4", "trackertracking"), ("effSystTnP_veto", "effSyst_veto"), ("etaDecorr0", "fullyCorr") ]
                        else:
                            nameReplace = [("WPSYST0", "reco"), ("WPSYST1", "tracking"), ("WPSYST2", "idip"), ("effSystTnP_veto", "effSyst_veto"), ("etaDecorr0", "fullyCorr") ]
                        scale = 1.0
                        mirror = True
                        mirrorDownVarEqualToNomi=False
                        groupName = "muon_eff_syst_veto"
                        if useGlobalOrTrackerVeto:
                            splitGroupDict = {f"{groupName}{x}" : f".*effSyst_veto.*{x}" for x in list(["reco","tracking","idip","trackerreco","trackertracking"])}
                        else:
                            splitGroupDict = {f"{groupName}{x}" : f".*effSyst_veto.*{x}" for x in list(["reco","tracking","idip"])}
                        splitGroupDict["muon_eff_all"] = ".*"
                    else:
                        nameReplace = []
                        mirror = True
                        mirrorDownVarEqualToNomi=False
                        if useRefinedVeto:
                            axes = ["vetoreco-vetotracking-vetoidip", "SF eta", "nPtEigenBins", "SF charge"]
                            axlabels = ["WPSTEP", "eta", "pt", "q"]
                            nameReplace = nameReplace + [("effStatTnP_veto_sf_", "effStat_veto_"), ("WPSTEP0", "reco"), ("WPSTEP1", "tracking"), ("WPSTEP2", "idip")]
                        else:
                            axes = ["SF eta", "nPtEigenBins", "SF charge"]
                            axlabels = ["eta", "pt", "q"]
                            nameReplace = nameReplace + [("effStatTnP_veto_sf_", "effStat_veto_")]
                        scale = 1.0
                        groupName = "muon_eff_stat_veto"
                        splitGroupDict = {f"{groupName}{x}" : f".*effStat_veto.*{x}" for x in list(["reco","tracking","idip"])}
                        splitGroupDict["muon_eff_all"] = ".*"
                    if args.effStatLumiScale and "Syst" not in name:
                        scale /= math.sqrt(args.effStatLumiScale)

                    cardTool.addSystematic(
                        name,
                        mirror=mirror,
                        mirrorDownVarEqualToNomi=mirrorDownVarEqualToNomi,
                        group=groupName,
                        splitGroup={**splitGroupDict, "experiment": ".*"},
                        systAxes=axes,
                        labelsByAxis=axlabels,
                        baseName=name+"_",
                        processes=['Zveto_samples'],
                        passToFakes=passSystToFakes if wmass else False,
                        systNameReplace=nameReplace,
                        scale=scale,
                    )

        else:
            if datagroups.flavor in ["mu", "mumu"]:
                lepEffs = ["muSF_HLT_DATA_stat", "muSF_HLT_DATA_syst", "muSF_HLT_MC_stat", "muSF_HLT_MC_syst", "muSF_ISO_stat", "muSF_ISO_DATA_syst", "muSF_ISO_MC_syst", "muSF_IDIP_stat", "muSF_IDIP_DATA_syst", "muSF_IDIP_MC_syst"]
            else:
                lepEffs = [] # ["elSF_HLT_syst", "elSF_IDISO_stat"]

            for lepEff in lepEffs:
                cardTool.addSystematic(lepEff,
                    processes=cardTool.allMCProcesses(),
                    mirror = True,
                    group="CMS_lepton_eff",
                    splitGroup = {"experiment": ".*"},
                    baseName=lepEff,
                    systAxes = ["tensor_axis_0"],
                    labelsByAxis = [""],
                )

    if (wmass or wlike) and input_tools.args_from_metadata(cardTool, "recoilUnc"):
        combine_helpers.add_recoil_uncertainty(cardTool, ["signal_samples"],
            passSystToFakes=passSystToFakes,
            flavor=datagroups.flavor if datagroups.flavor else "mu",
            pu_type="lowPU" if lowPU else "highPU")

    if lowPU:
        if datagroups.flavor in ["e", "ee"] and False:
            # disable, prefiring for muons currently broken? (fit fails)
            cardTool.addSystematic("prefireCorr",
                processes=cardTool.allMCProcesses(),
                mirror = False,
                group="CMS_prefire17",
                splitGroup = {"experiment": ".*"},
                baseName="CMS_prefire17",
                systAxes = ["downUpVar"],
                labelsByAxis = ["downUpVar"],
            )

        return cardTool

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

    # cardTool.addSystematic(msv_config['hist_name'],
    #     processes=['single_v_samples' if wmass else 'single_vmu_samples'],
    #     group="muonCalibration",
    #     baseName="CMS_scale_m_",
    #     systAxes=msv_config['syst_axes'],
    #     labelsByAxis=msv_config['syst_axes_labels'],
    #     passToFakes=passSystToFakes,
    #     scale = args.scaleMuonCorr,
    # )
    cardTool.addSystematic("muonL1PrefireSyst",
        processes=['MCnoQCD'],
        group="muonPrefire",
        splitGroup = {f"prefire" : f".*", "experiment": ".*"},
        baseName="CMS_prefire_syst_m",
        systAxes=["downUpVar"],
        labelsByAxis=["downUpVar"],
        passToFakes=passSystToFakes,
    )
    cardTool.addSystematic("muonL1PrefireStat",
        processes=['MCnoQCD'],
        group="muonPrefire",
        splitGroup = {f"prefire" : f".*", "experiment": ".*"},
        baseName="CMS_prefire_stat_m_",
        systAxes=["downUpVar", "etaPhiRegion"],
        labelsByAxis=["downUpVar", "etaPhiReg"],
        passToFakes=passSystToFakes,
    )
    cardTool.addSystematic("ecalL1Prefire",
        processes=['MCnoQCD'],
        group="ecalPrefire",
        splitGroup = {f"prefire" : f".*", "experiment": ".*"},
        baseName="CMS_prefire_ecal",
        systAxes=["downUpVar"],
        labelsByAxis=["downUpVar"],
        passToFakes=passSystToFakes,
    )

    cardTool.addSystematic("muonScaleSyst_responseWeights",
        processes=['single_v_samples'],
        group="scaleCrctn",
        splitGroup={f"muonCalibration" : f".*"},
        baseName="Scale_correction_",
        systAxes=["unc", "downUpVar"],
        passToFakes=passSystToFakes,
        scale = args.calibrationStatScaling,
    )
    cardTool.addSystematic("muonScaleClosSyst_responseWeights",
        processes=['single_v_samples'],
        group="scaleClosCrctn",
        splitGroup={f"muonCalibration" : f".*"},
        baseName="ScaleClos_correction_",
        systAxes=["unc", "downUpVar"],
        passToFakes=passSystToFakes,
    )

    mzerr = 2.1e-3
    mz0 = 91.18
    adhocA = args.correlatedAdHocA
    nomvarA = common.correlated_variation_base_size["A"]
    scaleA = math.sqrt( (mzerr/mz0)**2 + adhocA**2 )/nomvarA

    adhocM = args.correlatedAdHocM
    nomvarM = common.correlated_variation_base_size["M"]
    scaleM = adhocM/nomvarM

    cardTool.addSystematic("muonScaleClosASyst_responseWeights",
        processes=['single_v_samples'],
        group="scaleClosACrctn",
        splitGroup={f"muonCalibration" : f".*"},
        baseName="ScaleClosA_correction_",
        systAxes=["unc", "downUpVar"],
        passToFakes=passSystToFakes,
        scale = scaleA,
    )
    if abs(scaleM) > 0.:
        cardTool.addSystematic("muonScaleClosMSyst_responseWeights",
            processes=['single_v_samples'],
            group="scaleClosMCrctn",
            splitGroup={f"muonCalibration" : f".*"},
            baseName="ScaleClosM_correction_",
            systAxes=["unc", "downUpVar"],
            passToFakes=passSystToFakes,
            scale = scaleM,
        )
    if not input_tools.args_from_metadata(cardTool, "noSmearing"):
        cardTool.addSystematic("muonResolutionSyst_responseWeights",
            mirror = True,
            processes=['single_v_samples'],
            group="resolutionCrctn",
            splitGroup={f"muonCalibration" : f".*"},
            baseName="Resolution_correction_",
            systAxes=["smearing_variation"],
            passToFakes=passSystToFakes,
            scale = args.resolutionStatScaling,
        )

    cardTool.addSystematic("pixelMultiplicitySyst",
        mirror = True,
        processes=['single_v_samples'],
        group="pixelMultiplicitySyst",
        splitGroup={f"muonCalibration" : f".*"},
        baseName="pixel_multiplicity_syst_",
        systAxes=["var"],
        passToFakes=passSystToFakes,
    )

    if input_tools.args_from_metadata(cardTool, "pixelMultiplicityStat"):
        cardTool.addSystematic("pixelMultiplicityStat",
            mirror = True,
            processes=['single_v_samples'],
            group="pixelMultiplicityStat",
            splitGroup={f"muonCalibration" : f".*"},
            baseName="pixel_multiplicity_stat_",
            systAxes=["var"],
            passToFakes=passSystToFakes,
        )

    if "run" in fitvar:
        # add ad-hoc normalization uncertainty uncorrelated across run bins
        #   accounting for time instability (e.g. reflecting the corrections applied as average like pileup, prefiring, ...)
        cardTool.addSystematic(
            name=cardTool.nominalName,
            rename="timeStability",
            processes=['MCnoQCD'],
            group=f"timeStability",
            splitGroup = {"experiment": ".*"},
            passToFakes=passSystToFakes,
            mirror=True,
            labelsByAxis=[f"run"],
            systAxes=["run_"],
            action=lambda h:
                hh.addHists(h,
                    hh.expand_hist_by_duplicate_axis(h, "run", "run_"),
                    scale2=0.01)
        )

        # add additional scale and resolution uncertainty uncorrelated across run slices
        cardTool.addSystematic("muonScaleSyst_responseWeights",
            rename="muonScaleSyst_responseWeightsDecorr",
            processes=['single_v_samples'],
            group="scaleCrctn",
            splitGroup={f"muonCalibration" : f".*"},
            baseName="Scale_correction_",
            systAxes=["unc", "downUpVar", "run_"],
            passToFakes=passSystToFakes,
            scale = args.calibrationStatScaling,
            actionRequiresNomi=True,
            action=syst_tools.decorrelateByAxes,
            actionArgs=dict(
                axesToDecorrNames=["run"], newDecorrAxesNames=["run_"])
        )

        cardTool.addSystematic("muonScaleClosSyst_responseWeights",
            rename="muonScaleClosSyst_responseWeightsDecorr",
            processes=['single_v_samples'],
            group="scaleClosCrctn",
            splitGroup={f"muonCalibration" : f".*"},
            baseName="ScaleClos_correction_",
            systAxes=["unc", "downUpVar", "run_"],
            passToFakes=passSystToFakes,
            actionRequiresNomi=True,
            action=syst_tools.decorrelateByAxes,
            actionArgs=dict(
                axesToDecorrNames=["run"], newDecorrAxesNames=["run_"])
        )

        if not input_tools.args_from_metadata(cardTool, "noSmearing"):
            cardTool.addSystematic("muonResolutionSyst_responseWeights",
                rename="muonResolutionSyst_responseWeightsDecorr",
                mirror = True,
                processes=['single_v_samples'],
                group="resolutionCrctn",
                splitGroup={f"muonCalibration" : f".*"},
                baseName="Resolution_correction_",
                systAxes=["smearing_variation", "run_"],
                passToFakes=passSystToFakes,
                scale = args.resolutionStatScaling,
                actionRequiresNomi=True,
                action=syst_tools.decorrelateByAxes,
                actionArgs=dict(
                    axesToDecorrNames=["run"], newDecorrAxesNames=["run_"])
            )

    # Previously we had a QCD uncertainty for the mt dependence on the fakes, see: https://github.com/WMass/WRemnants/blob/f757c2c8137a720403b64d4c83b5463a2b27e80f/scripts/combine/setupCombineWMass.py#L359

    return cardTool

def analysis_label(card_tool):
    analysis_name_map = {
        "w_mass" : "WMass",
        "vgen" : "ZGen" if len(card_tool.getProcesses()) > 0 and card_tool.getProcesses()[0][0] == "Z" else "WGen",
        "z_wlike" : "ZMassWLike",
        "z_dilepton" : "ZMassDilepton",
        "w_lowpu" : "WMass_lowPU",
        "z_lowpu" : "ZMass_lowPU",
    }

    if card_tool.datagroups.mode not in analysis_name_map:
        raise ValueError(f"Invalid datagroups mode {card_tool.datagroups.mode}")

    return analysis_name_map[card_tool.datagroups.mode]

def outputFolderName(outfolder, card_tool, doStatOnly, postfix):
    to_join = [analysis_label(card_tool)]+card_tool.fit_axes

    if doStatOnly:
        to_join.append("statOnly")
    if card_tool.datagroups.flavor:
        to_join.append(card_tool.datagroups.flavor)
    if postfix is not None:
        to_join.append(postfix)

    return f"{outfolder}/{'_'.join(to_join)}/"

def main(args, xnorm=False):
    forceNonzero = False #args.analysisMode == None
    checkSysts = forceNonzero

    fitvar = args.fitvar[0].split("-") if not xnorm else ["count"]
    genvar = args.genAxes[0].split("-") if hasattr(args,"genAxes") and len(args.genAxes) else None
    iBaseName = args.baseName[0]
    iLumiScale = args.lumiScale[0]
    cardTool = setup(args, args.inputFile[0], iBaseName, iLumiScale, fitvar, genvar, xnorm=xnorm)
    cardTool.setOutput(outputFolderName(args.outfolder, cardTool, args.doStatOnly, args.postfix), analysis_label(cardTool))
    cardTool.writeOutput(args=args, forceNonzero=forceNonzero, check_systs=checkSysts)
    return

if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()

    logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)

    isUnfolding = args.analysisMode == "unfolding"
    isTheoryAgnostic = args.analysisMode in ["theoryAgnosticNormVar", "theoryAgnosticPolVar"]
    isTheoryAgnosticPolVar = args.analysisMode == "theoryAgnosticPolVar"
    isPoiAsNoi = (isUnfolding or isTheoryAgnostic) and args.poiAsNoi
    isFloatingPOIsTheoryAgnostic = isTheoryAgnostic and not isPoiAsNoi
    isFloatingPOIs = (isUnfolding or isTheoryAgnostic) and not isPoiAsNoi

    if args.noHist and args.noStatUncFakes:
        raise ValueError("Option --noHist would override --noStatUncFakes. Please select only one of them")
    if isUnfolding and args.fitXsec:
        raise ValueError("Options unfolding and --fitXsec are incompatible. Please choose one or the other")

    if isTheoryAgnostic:
        if len(args.genAxes) == 0:
            args.genAxes = ["ptVgenSig-absYVgenSig-helicitySig"]
            logger.warning(f"Automatically setting '--genAxes {' '.join(args.genAxes)}' for theory agnostic analysis")
            if args.poiAsNoi:
                logger.warning("This is only needed to properly get the systematic axes")

    if isFloatingPOIsTheoryAgnostic:
        # The following is temporary, just to avoid passing the option explicitly
        logger.warning("For now setting theory agnostic without POI as NOI activates --doStatOnly")
        args.doStatOnly = True

    if args.hdf5:
        writer = HDF5Writer.HDF5Writer(sparse=args.sparse)

        if args.baseName == "xnorm":
            writer.theoryFit = True

        # loop over all files
        outnames = []
        for i, ifile in enumerate(args.inputFile):
            fitvar = args.fitvar[i].split("-")
            genvar = args.genAxes[i].split("-") if hasattr(args,"genAxes") and len(args.genAxes) else None
            iBaseName = args.baseName[0] if len(args.baseName)==1 else args.baseName[i]
            iLumiScale = args.lumiScale[0] if len(args.lumiScale)==1 else args.lumiScale[i]
            
            cardTool = setup(args, ifile, iBaseName, iLumiScale, fitvar, xnorm=args.fitresult is not None)
            outnames.append( (outputFolderName(args.outfolder, cardTool, args.doStatOnly, args.postfix), analysis_label(cardTool)) )

            writer.add_channel(cardTool)
            if isFloatingPOIs or isUnfolding:
                fitvar = genvar if isPoiAsNoi else ["count"] 
                cardTool = setup(args, ifile, iBaseName, iLumiScale, fitvar, xnorm=True)
                writer.add_channel(cardTool)
        if args.fitresult:
            writer.set_fitresult(args.fitresult, mc_stat=not (args.noMCStat or args.explicitSignalMCstat))

        if len(outnames) == 1:
            outfolder, outfile = outnames[0]
        else:
            dir_append = '_'.join(['', *filter(lambda x: x, ['statOnly' if args.doStatOnly else '', args.postfix])])
            unique_names = list(dict.fromkeys([o[1] for o in outnames]))
            outfolder = f"{args.outfolder}/Combination_{''.join(unique_names)}{dir_append}/"
            outfile = "Combination"
        logger.info(f"Writing HDF5 output to {outfile}")
        writer.write(args, outfolder, outfile, allowNegativeExpectation=args.allowNegativeExpectation)
    else:
        if len(args.inputFile) > 1:
            raise IOError(f"Multiple input files only supported within --hdf5 mode")

        main(args)
        if isFloatingPOIs:
            logger.warning("Now running with xnorm = True")
            # in case of unfolding and hdf5, the xnorm histograms are directly written into the hdf5
            main(args, xnorm=True)

    logging.summary()
