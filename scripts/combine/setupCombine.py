#!/usr/bin/env python3
from wremnants import CardTool,combine_helpers,combine_theory_helper, HDF5Writer
from wremnants.datasets.datagroups import Datagroups
from utilities import common, logging, input_tools, boostHistHelpers as hh
import itertools
import argparse
import hist
import math

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
    parser.add_argument("-n", "--baseName", type=str, help="Histogram name in the file (e.g., 'nominal')", default="nominal")
    parser.add_argument("--noHist", action='store_true', help="Skip the making of 2D histograms (root file is left untouched if existing)")
    parser.add_argument("--qcdProcessName" , type=str, default=None, help="Name for QCD process")
    # setting on the fit behaviour
    parser.add_argument("--fitvar", nargs="+", help="Variable to fit", default=["eta-pt-charge"])
    parser.add_argument("--rebin", type=int, nargs='*', default=[], help="Rebin axis by this value (default, 1, does nothing)")
    parser.add_argument("--absval", type=int, nargs='*', default=[], help="Take absolute value of axis if 1 (default, 0, does nothing)")
    parser.add_argument("--axlim", type=float, default=[], nargs='*', help="Restrict axis to this range (assumes pairs of values by axis, with trailing axes optional)")
    parser.add_argument("--lumiScale", type=float, default=1.0, help="Rescale equivalent luminosity by this value (e.g. 10 means ten times more data and MC)")
    parser.add_argument("--sumChannels", action='store_true', help="Only use one channel")
    parser.add_argument("--fitXsec", action='store_true', help="Fit signal inclusive cross section")
    parser.add_argument("--fitresult", type=str, default=None ,help="Use data and covariance matrix from fitresult (for making a theory fit)")
    parser.add_argument("--fakerateAxes", nargs="+", help="Axes for the fakerate binning", default=["eta","pt","charge"])
    parser.add_argument("--ABCD", action="store_true", help="Produce datacard for simultaneous fit of ABCD regions")
    # settings on the nuisances itself
    parser.add_argument("--doStatOnly", action="store_true", default=False, help="Set up fit to get stat-only uncertainty (currently combinetf with -S 0 doesn't work)")
    parser.add_argument("--minnloScaleUnc", choices=["byHelicityPt", "byHelicityPtCharge", "byHelicityCharge", "byPtCharge", "byPt", "byCharge", "integrated",], default="byHelicityPt",
            help="Decorrelation for QCDscale")
    parser.add_argument("--resumUnc", default="tnp", type=str, choices=["scale", "tnp", "none"], help="Include SCETlib uncertainties")
    parser.add_argument("--npUnc", default="Delta_Lambda", type=str, choices=combine_theory_helper.TheoryHelper.valid_np_models, help="Nonperturbative uncertainty model")
    parser.add_argument("--tnpMagnitude", default=1, type=float, help="Variation size for the TNP")
    parser.add_argument("--scaleTNP", default=5, type=float, help="Scale the TNP uncertainties by this factor")
    parser.add_argument("--scalePdf", default=1, type=float, help="Scale the PDF hessian uncertainties by this factor")
    parser.add_argument("--pdfUncFromCorr", action='store_true', help="Take PDF uncertainty from correction hist (Requires having run that correction)")
    parser.add_argument("--ewUnc", type=str, nargs="*", default=["horacenloew"], choices=["horacenloew", "winhacnloew"], help="Include EW uncertainty")
    parser.add_argument("--widthUnc", action='store_true', help="Include uncertainty on W and Z width")
    parser.add_argument("--noStatUncFakes" , action="store_true",   help="Set bin error for QCD background templates to 0, to check MC stat uncertainties for signal only")
    parser.add_argument("--skipSignalSystOnFakes" , action="store_true", help="Do not propagate signal uncertainties on fakes, mainly for checks.")
    parser.add_argument("--noQCDscaleFakes", action="store_true",   help="Do not apply QCd scale uncertainties on fakes, mainly for debugging")
    parser.add_argument("--addQCDMC", action="store_true", help="Include QCD MC when making datacards (otherwise by default it will always be excluded)")
    parser.add_argument("--muonScaleVariation", choices=["smearingWeights", "massWeights", "manualShift"], default="smearingWeights", help="the method with which the muon scale variation histograms are derived")
    parser.add_argument("--scaleMuonCorr", type=float, default=1.0, help="Scale up/down dummy muon scale uncertainty by this factor")
    parser.add_argument("--correlatedNonClosureNuisances", action='store_true', help="get systematics from histograms for the Z non-closure nuisances without decorrelation in eta and pt")
    parser.add_argument("--sepImpactForNC", action="store_true", help="use a dedicated impact gropu for non closure nuisances, instead of putting them in muonScale")
    parser.add_argument("--noEfficiencyUnc", action='store_true', help="Skip efficiency uncertainty (useful for tests, because it's slow). Equivalent to --excludeNuisances '.*effSystTnP|.*effStatTnP' ")
    parser.add_argument("--effStatLumiScale", type=float, default=None, help="Rescale equivalent luminosity for efficiency stat uncertainty by this value (e.g. 10 means ten times more data from tag and probe)")
    parser.add_argument("--binnedScaleFactors", action='store_true', help="Use binned scale factors (different helpers and nuisances)")
    parser.add_argument("--isoEfficiencySmoothing", action='store_true', help="If isolation SF was derived from smooth efficiencies instead of direct smoothing")
    # pseudodata
    parser.add_argument("--pseudoData", type=str, help="Hist to use as pseudodata")
    parser.add_argument("--pseudoDataIdx", type=str, default="0", help="Variation index to use as pseudodata")
    parser.add_argument("--pseudoDataFile", type=str, help="Input file for pseudodata (if it should be read from a different file)", default=None)
    parser.add_argument("--pseudoDataProcsRegexp", type=str, default=".*", help="Regular expression for processes taken from pseudodata file (all other processes are automatically got from the nominal file). Data is excluded automatically as usual")
    # unfolding/differential/theory agnostic
    parser.add_argument("--unfolding", action='store_true', help="Prepare datacard for unfolding")
    parser.add_argument("--genAxes", type=str, default=None, nargs="+", help="Specify which gen axis should be used in unfolding, if 'None', use all (inferred from metadata).")
    parser.add_argument("--theoryAgnostic", action='store_true', help="Prepare datacard for theory agnostic analysis, similar to unfolding but different axis and possibly other differences")
    # utility options to deal with charge when relevant, mainly for theory agnostic but also unfolding
    parser.add_argument("--recoCharge", type=str, default=["plus", "minus"], nargs="+", choices=["plus", "minus"], help="Specify reco charge to use, default uses both. This is a workaround for unfolding/theory-agnostic fit when running a single reco charge, as gen bins with opposite gen charge have to be filtered out")
    parser.add_argument("--forceRecoChargeAsGen", action="store_true", help="Force gen charge to match reco charge in CardTool, this only works when the reco charge is used to define the channel")

    return parser

def setup(args, inputFile, fitvar, xnorm=False):

    # NOTE: args.filterProcGroups and args.excludeProcGroups should in principle not be used together
    #       (because filtering is equivalent to exclude something), however the exclusion is also meant to skip
    #       processes which are defined in the original process dictionary but are not supposed to be (always) run on
    if args.addQCDMC or "QCD" in args.filterProcGroups:
        logger.warning("Adding QCD MC to list of processes for the fit setup")
    else:
        if "QCD" not in args.excludeProcGroups:
            logger.warning("Automatic removal of QCD MC from list of processes. Use --filterProcGroups 'QCD' or --addQCDMC to keep it")
            args.excludeProcGroups.append("QCD")
    filterGroup = args.filterProcGroups if args.filterProcGroups else None
    excludeGroup = args.excludeProcGroups if args.excludeProcGroups else None
    if args.ABCD and (excludeGroup is None or "Fake" not in excludeGroup):
        excludeGroup.append("Fake")
    logger.debug(f"Filtering these groups of processes: {args.filterProcGroups}")
    logger.debug(f"Excluding these groups of processes: {args.excludeProcGroups}")

    datagroups = Datagroups(inputFile, excludeGroups=excludeGroup, filterGroups=filterGroup, applySelection= not xnorm and not args.ABCD)

    if not xnorm and (args.axlim or args.rebin or args.absval):
        if len(args.axlim) % 2 or len(args.axlim)/2 > len(fitvar) or len(args.rebin) > len(fitvar):
            raise ValueError("Inconsistent rebin or axlim arguments. axlim must be at most two entries per axis, and rebin at most one")

        sel = {}
        for var,low,high,rebin in itertools.zip_longest(fitvar, args.axlim[::2], args.axlim[1::2], args.rebin):
            s = hist.tag.Slicer()
            if low is not None and high is not None:
                logger.info(f"Restricting the axis '{var}' to range [{low}, {high}]")
                sel[var] = s[complex(0, low):complex(0, high):hist.rebin(rebin) if rebin else None]
            elif rebin:
                sel[var] = s[hist.rebin(rebin)]
            if rebin:
                logger.info(f"Rebinning the axis '{var}' by [{rebin}]")

        if len(sel) > 0:
            logger.info(f"Will apply the global selection {sel}")
            datagroups.setGlobalAction(lambda h: h[sel])

        for i, (var, absval) in enumerate(itertools.zip_longest(fitvar, args.absval)):
            if absval:
                logger.info(f"Taking the absolute value of axis '{var}'")
                datagroups.setGlobalAction(lambda h, ax=var: hh.makeAbsHist(h, ax))
                fitvar[i] = f"abs{var}"

    wmass = datagroups.wmass
    wlike = datagroups.wlike
    lowPU = datagroups.lowPU
    dilepton = datagroups.dilepton

    constrainMass = (dilepton and not "mll" in fitvar) or args.fitXsec

    if wmass:
        base_group = "Wenu" if datagroups.flavor == "e" else "Wmunu"
    else:
        base_group = "Zee" if datagroups.flavor == "ee" else "Zmumu"

    if args.unfolding and args.fitXsec:
        raise ValueError("Options --unfolding and --fitXsec are incompatible. Please choose one or the other")
    elif args.fitXsec:
        datagroups.unconstrainedProcesses.append(base_group)
    elif args.unfolding:
        constrainMass = False if args.theoryAgnostic else True
        datagroups.setGenAxes(args.genAxes)
        if wmass:
            # gen level bins, split by charge
            if "minus" in args.recoCharge:
                datagroups.defineSignalBinsUnfolding(base_group, f"W_qGen0", member_filter=lambda x: x.name.startswith("Wminus"))
            if "plus" in args.recoCharge:
                datagroups.defineSignalBinsUnfolding(base_group, f"W_qGen1", member_filter=lambda x: x.name.startswith("Wplus"))
            # out of acceptance contribution
            datagroups.groups[base_group].deleteMembers([m for m in datagroups.groups[base_group].members if not m.name.startswith("Bkg")])
        else:
            datagroups.defineSignalBinsUnfolding(base_group, "Z", member_filter=lambda x: x.name.startswith(base_group))
            # out of acceptance contribution
            datagroups.groups[base_group].deleteMembers([m for m in datagroups.groups[base_group].members if not m.name.startswith("Bkg")])

 
    if args.noHist and args.noStatUncFakes:
        raise ValueError("Option --noHist would override --noStatUncFakes. Please select only one of them")

    if "BkgWmunu" in args.excludeProcGroups:
        datagroups.deleteGroup("Wmunu") # remove out of acceptance signal

    # Start to create the CardTool object, customizing everything
    cardTool = CardTool.CardTool(xnorm=xnorm, ABCD=wmass and args.ABCD)
    cardTool.setDatagroups(datagroups)
    if args.qcdProcessName:
        cardTool.setFakeName(args.qcdProcessName)
    logger.debug(f"Making datacards with these processes: {cardTool.getProcesses()}")
    if args.absolutePathInCard:
        cardTool.setAbsolutePathShapeInCard()
    cardTool.setProjectionAxes(fitvar)
    cardTool.setFakerateAxes(args.fakerateAxes)
    if wmass and args.ABCD:
        # In case of ABCD we need to have different fake processes fir e and mu to have uncorrelated uncertainties
        cardTool.setFakeName(datagroups.fakeName + (datagroups.flavor if datagroups.flavor else "")) 
        cardTool.unroll=True
    if args.sumChannels or xnorm or dilepton or (wmass and args.ABCD):
        cardTool.setChannels(["inclusive"])
        cardTool.setWriteByCharge(False)
    else:
        cardTool.setChannels(args.recoCharge)
        if args.forceRecoChargeAsGen:
            cardTool.setExcludeProcessForChannel("plus", ".*qGen0")
            cardTool.setExcludeProcessForChannel("minus", ".*qGen1")
    if xnorm:
        histName = "xnorm"
        cardTool.setHistName(histName)
        cardTool.setNominalName(histName)
        datagroups.select_xnorm_groups()
        if args.unfolding:
            cardTool.setProjectionAxes(["count"])
        else:
            if wmass:
                # add gen charge as additional axis
                datagroups.groups[base_group].add_member_axis("qGen", datagroups.results, 
                    member_filters={-1: lambda x: x.name.startswith("Wminus"), 1: lambda x: x.name.startswith("Wplus")}, 
                    hist_filter=lambda x: x.startswith("xnorm"))
                datagroups.gen_axes = ["qGen", *datagroups.gen_axes]
            else:
                datagroups.gen_axes = datagroups.gen_axes[:]

            cardTool.unroll = True
            # remove projection axes from gen axes, otherwise they will be integrated before
            
            if datagroups.gen_axes != cardTool.project:
                raise NotImplementedError(f"The gen axes of the model {datagroups.gen_axes} do not agree with the ones requested {cardTool.project}")
            datagroups.setGenAxes([]) # [a for a in datagroups.gen_axes if a not in cardTool.project])
    else:
        cardTool.setHistName(args.baseName)
        cardTool.setNominalName(args.baseName)
        
    # define sumGroups for integrated cross section
    if args.unfolding:
        # TODO: make this less hardcoded to filter the charge (if the charge is not present this will duplicate things)
        if wmass:
            if "plus" in args.recoCharge:
                cardTool.addPOISumGroups(genCharge="qGen1")
            if "minus" in args.recoCharge:
                cardTool.addPOISumGroups(genCharge="qGen0")
        else:
            cardTool.addPOISumGroups()

    if args.noHist:
        cardTool.skipHistograms()
    cardTool.setSpacing(28)
    if args.noStatUncFakes:
        cardTool.setProcsNoStatUnc(procs=args.qcdProcessName, resetList=False)
    cardTool.setCustomSystForCard(args.excludeNuisances, args.keepNuisances)
    if args.pseudoData:
        cardTool.setPseudodata(args.pseudoData, args.pseudoDataIdx, args.pseudoDataProcsRegexp)
        if args.pseudoDataFile:
            cardTool.setPseudodataDatagroups(make_datagroup(args.pseudoDataFile,
                                                                  excludeGroups=excludeGroup,
                                                                  filterGroups=filterGroup,
                                                                  applySelection= not xnorm)
            )
    cardTool.setLumiScale(args.lumiScale)

    if not args.theoryAgnostic:
        logger.info(f"cardTool.allMCProcesses(): {cardTool.allMCProcesses()}")
        
    passSystToFakes = wmass and not args.skipSignalSystOnFakes and args.qcdProcessName not in excludeGroup and (filterGroup == None or args.qcdProcessName in filterGroup) and not xnorm

    cardTool.addProcessGroup("single_v_samples", lambda x: x[0] in ["W", "Z"] and x[1] not in ["W","Z"])
    if wmass:
        cardTool.addProcessGroup("w_samples", lambda x: x[0] == "W" and x[1] not in ["W","Z"])
        if not xnorm:
            cardTool.addProcessGroup("single_v_nonsig_samples", lambda x: x[0] == "Z" and x[1] not in ["W","Z"])

    cardTool.addProcessGroup("single_vmu_samples", lambda x: x[0] in ["W", "Z"] and x[1] not in ["W","Z"] and "tau" not in x)
    cardTool.addProcessGroup("signal_samples", lambda x: ((x[0] == "W" and wmass) or (x[0] == "Z" and not wmass)) and x[1] not in ["W","Z"] and "tau" not in x)
    cardTool.addProcessGroup("signal_samples_inctau", lambda x: ((x[0] == "W" and wmass) or (x[0] == "Z" and not wmass)) and x[1] not in ["W","Z"])
    cardTool.addProcessGroup("MCnoQCD", lambda x: x not in ["QCD", "Data"])

    if not (args.theoryAgnostic or args.unfolding) :
        logger.info(f"All MC processes {cardTool.procGroups['MCnoQCD']}")
        logger.info(f"Single V samples: {cardTool.procGroups['single_v_samples']}")
        if wmass and not xnorm:
            logger.info(f"Single V no signal samples: {cardTool.procGroups['single_v_nonsig_samples']}")
        logger.info(f"Signal samples: {cardTool.procGroups['signal_samples']}")

    constrainedZ = constrainMass and not wmass
    label = 'W' if wmass else 'Z'
    massSkip = [(f"^massShift[W|Z]{i}MeV.*",) for i in range(0, 110 if constrainedZ else 100, 10)]
    if wmass and not xnorm and not args.doStatOnly:
        cardTool.addSystematic(f"massWeightZ",
                                processes=['single_v_nonsig_samples'],
                                group=f"massShiftZ",
                                skipEntries=massSkip[:]+[("^massShiftZ100MeV.*",)],
                                mirror=False,
                                noConstraint=False,
                                systAxes=["massShift"],
                                passToFakes=passSystToFakes,
        )

    if not (constrainMass or wmass):
        massSkip.append(("^massShift.*2p1MeV.*",))

    signal_samples_forMass = ["signal_samples_inctau"]
    if args.theoryAgnostic:
        logger.error("Temporarily not using mass weights for Wtaunu. Please update when possible")
        signal_samples_forMass = ["signal_samples"]
    if constrainMass and args.doStatOnly:
        logger.info("Using option --doStatOnly: the card was created without nuisance parameters")
        return cardTool

    if args.doStatOnly and constrainMass:
        # no mass weight uncertainty for stat only fits if mass weight is a nuisance (e.g. unfolding, xsec, ...)
        return cardTool

    cardTool.addSystematic(f"massWeight{label}",
                           processes=signal_samples_forMass,
                           group=f"massShift{label}",
                           noi=not constrainMass,
                           skipEntries=massSkip,
                           mirror=False,
                           #TODO: Name this
                           noConstraint=not constrainMass,
                           systAxes=["massShift"],
                           passToFakes=passSystToFakes,
    )

    if args.doStatOnly:
        # print a card with only mass weights
        logger.info("Using option --doStatOnly: the card was created with only mass nuisance parameter")
        return cardTool
    
    if args.widthUnc:
        widthSkipZ = [("widthZ2p49333GeV",), ("widthZ2p49493GeV",), ("widthZ2p4952GeV",)] 
        widthSkipW = [("widthW2p09053GeV",), ("widthW2p09173GeV",), ("widthW2p085GeV",)]
        if wmass:
            cardTool.addSystematic(f"widthWeightZ",
                                    processes=["single_v_nonsig_samples"],
                                    group=f"widthZ",
                                    skipEntries=widthSkipZ[:],
                                    mirror=False,
                                    systAxes=["width"],
                                    passToFakes=passSystToFakes,
            )
        cardTool.addSystematic(f"widthWeight{label}",
                                processes=["signal_samples_inctau"],
                                skipEntries=widthSkipZ[:] if label=="Z" else widthSkipW[:],
                                group=f"width{label}",
                                mirror=False,
                                #TODO: Name this
                                systAxes=["width"],
                                passToFakes=passSystToFakes,
        )



    for ewUnc in args.ewUnc:
        if ewUnc=="winhacnloew" and (not wmass or datagroups.flavor == "e"):
            logger.warning("Winhac is not implemented for any other process than W, proceed w/o winhac EW uncertainty")
            continue

        cardTool.addSystematic(f"{ewUnc}Corr", 
            processes=['w_samples'] if ewUnc=="winhacnloew" else ['single_v_samples'],
            mirror=True,
            group="theory_ew",
            systAxes=["systIdx"],
            labelsByAxis=[f"{ewUnc}Corr"],
            skipEntries=[(0, -1), (1, -1)],
            passToFakes=passSystToFakes,
        )

    to_fakes = passSystToFakes and not args.noQCDscaleFakes and not xnorm
    
    theory_helper = combine_theory_helper.TheoryHelper(cardTool)
    theory_helper.configure(resumUnc=args.resumUnc, 
        propagate_to_fakes=to_fakes,
        np_model=args.npUnc,
        tnp_magnitude=args.tnpMagnitude,
        tnp_scale = args.scaleTNP,
        mirror_tnp=True,
        pdf_from_corr=args.pdfUncFromCorr,
        scale_pdf_unc=args.scalePdf,
        minnloScaleUnc=args.minnloScaleUnc,
    )
    theory_helper.add_all_theory_unc()

    if xnorm:
        return cardTool

    # Below: experimental uncertainties

    if wmass:
        #cardTool.addLnNSystematic("CMS_Fakes", processes=[args.qcdProcessName], size=1.05, group="MultijetBkg")
        cardTool.addLnNSystematic("CMS_Top", processes=["Top"], size=1.06, group="CMS_background")
        cardTool.addLnNSystematic("CMS_VV", processes=["Diboson"], size=1.16, group="CMS_background")
        cardTool.addSystematic("luminosity",
                                processes=['MCnoQCD'],
                                outNames=["lumiDown", "lumiUp"],
                                group="luminosity",
                                systAxes=["downUpVar"],
                                labelsByAxis=["downUpVar"],
                                passToFakes=passSystToFakes)
    else:
        cardTool.addLnNSystematic("CMS_background", processes=["Other"], size=1.15, group="CMS_background")
        cardTool.addLnNSystematic("luminosity", processes=['MCnoQCD'], size=1.017 if lowPU else 1.012, group="luminosity")

    if not args.noEfficiencyUnc:

        if not lowPU:

            ## this is only needed when using 2D SF from 3D with ut-integration, let's comment for now
            # if wmass:
            #     cardTool.addSystematic("sf2d", 
            #         processes=['MCnoQCD'],
            #         outNames=["sf2dDown","sf2dUp"],
            #         group="SF3Dvs2D",
            #         scale = 1.0,
            #         mirror = True,
            #         mirrorDownVarEqualToNomi=False, # keep False, True is pathological
            #         noConstraint=False,
            #         systAxes=[],
            #         #labelsByAxis=["downUpVar"],
            #         passToFakes=passSystToFakes,
            #     )

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
                    scale = 1.0
                    mirror = True
                    mirrorDownVarEqualToNomi=False
                    groupName = "muon_eff_syst"
                    splitGroupDict = {f"{groupName}_{x}" : f".*effSyst.*{x}" for x in list(effTypesNoIso + ["iso"])}
                    splitGroupDict[groupName] = ".*effSyst.*" # add also the group with everything
                    # decorrDictEff = {                        
                    #     "x" : {
                    #         "label" : "eta",
                    #         "edges": [round(-2.4+i*0.1,1) for i in range(49)]
                    #     }
                    # }

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
                    scale = 1.0
                    groupName = "muon_eff_stat"
                    splitGroupDict = {f"{groupName}_{x}" : f".*effStat.*{x}" for x in effStatTypes}
                    splitGroupDict[groupName] = ".*effStat.*" # add also the group with everything
                if args.effStatLumiScale and "Syst" not in name:
                    scale /= math.sqrt(args.effStatLumiScale)

                cardTool.addSystematic(
                    name, 
                    mirror=mirror,
                    mirrorDownVarEqualToNomi=mirrorDownVarEqualToNomi,
                    group=groupName,
                    systAxes=axes,
                    labelsByAxis=axlabels,
                    baseName=name+"_",
                    processes=['MCnoQCD'],
                    passToFakes=passSystToFakes,
                    systNameReplace=nameReplace,
                    scale=scale,
                    splitGroup=splitGroupDict,
                    decorrelateByBin = {}
                )
                # if "Syst" in name and decorrDictEff != {}:
                #     # add fully correlated version again
                #     cardTool.addSystematic(
                #         name,
                #         rename=f"{name}_EtaDecorr",
                #         mirror=mirror,
                #         mirrorDownVarEqualToNomi=mirrorDownVarEqualToNomi,
                #         group=groupName,
                #         systAxes=axes,
                #         labelsByAxis=axlabels,
                #         baseName=name+"_",
                #         processes=['MCnoQCD'],
                #         passToFakes=passSystToFakes,
                #         systNameReplace=nameReplace,
                #         scale=scale,
                #         splitGroup=splitGroupDict,
                #         decorrelateByBin = decorrDictEff
                #     )
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
                    baseName=lepEff,
                    systAxes = ["tensor_axis_0"],
                    labelsByAxis = [""], 
                )

    # if (wmass or wlike) and not input_tools.args_from_metadata(cardTool, "noRecoil"):
    #     combine_helpers.add_recoil_uncertainty(cardTool, ["signal_samples"], 
    #         passSystToFakes=passSystToFakes, 
    #         flavor=datagroups.flavor if datagroups.flavor else "mu",
    #         pu_type="lowPU" if lowPU else "highPU")

    if lowPU:
        if datagroups.flavor in ["e", "ee"]:
            # disable, prefiring for muons currently broken? (fit fails)
            cardTool.addSystematic("prefireCorr",
                processes=cardTool.allMCProcesses(),
                mirror = False,
                group="CMS_prefire17",
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
    #     group="muonScale",
    #     baseName="CMS_scale_m_",
    #     systAxes=msv_config['syst_axes'],
    #     labelsByAxis=msv_config['syst_axes_labels'],
    #     passToFakes=passSystToFakes,
    #     scale = args.scaleMuonCorr,
    # )
    cardTool.addSystematic("muonL1PrefireSyst", 
        processes=['MCnoQCD'],
        group="muonPrefire",
        baseName="CMS_prefire_syst_m",
        systAxes=["downUpVar"],
        labelsByAxis=["downUpVar"],
        passToFakes=passSystToFakes,
    )
    cardTool.addSystematic("muonL1PrefireStat", 
        processes=['MCnoQCD'],
        group="muonPrefire",
        baseName="CMS_prefire_stat_m_",
        systAxes=["downUpVar", "etaPhiRegion"],
        labelsByAxis=["downUpVar", "etaPhiReg"],
        passToFakes=passSystToFakes,
    )
    cardTool.addSystematic("ecalL1Prefire", 
        processes=['MCnoQCD'],
        group="ecalPrefire",
        baseName="CMS_prefire_ecal",
        systAxes=["downUpVar"],
        labelsByAxis=["downUpVar"],
        passToFakes=passSystToFakes,
    )

    non_closure_scheme = input_tools.args_from_metadata(cardTool, "nonClosureScheme")
    correlated_non_closure = input_tools.args_from_metadata(cardTool, "correlatedNonClosureNP")
    if non_closure_scheme in ["A-M-separated", "A-only"]:
        cardTool.addSystematic("Z_non_closure_parametrized_A", 
            processes=['single_v_samples'],
            group="nonClosure" if args.sepImpactForNC else "muonScale",
            baseName="Z_nonClosure_parametrized_A_",
            systAxes=["unc", "downUpVar"] if not correlated_non_closure else ["downUpVar"],
            labelsByAxis=["unc", "downUpVar"] if not correlated_non_closure else ["downUpVar"],
            passToFakes=passSystToFakes
        )
    if non_closure_scheme in ["A-M-separated", "M-only", "binned-plus-M"]:
        cardTool.addSystematic("Z_non_closure_parametrized_M", 
            processes=['single_v_samples'],
            group="nonClosure" if args.sepImpactForNC else "muonScale",
            baseName="Z_nonClosure_parametrized_M_",
            systAxes=["unc", "downUpVar"] if not correlated_non_closure else ["downUpVar"],
            labelsByAxis=["unc", "downUpVar"] if not correlated_non_closure else ["downUpVar"],
            passToFakes=passSystToFakes
        )            
    if non_closure_scheme == "A-M-combined":
        cardTool.addSystematic("Z_non_closure_parametrized", 
            processes=['single_v_samples'],
            group="nonClosure" if args.sepImpactForNC else "muonScale",
            baseName="Z_nonClosure_parametrized_",
            systAxes=["unc", "downUpVar"] if not correlated_non_closure else ["downUpVar"],
            labelsByAxis=["unc", "downUpVar"] if not correlated_non_closure else ["downUpVar"],
            passToFakes=passSystToFakes
        )
    if non_closure_scheme in ["binned", "binned-plus-M"]:
        cardTool.addSystematic("Z_non_closure_binned", 
            processes=['single_v_samples'],
            group="nonClosure" if args.sepImpactForNC else "muonScale",
            baseName="Z_nonClosure_binned_",
            systAxes=["unc_ieta", "unc_ipt", "downUpVar"] if not correlated_non_closure else ["downUpVar"],
            labelsByAxis=["unc_ieta", "unc_ipt", "downUpVar"] if not correlated_non_closure else ["downUpVar"],
            passToFakes=passSystToFakes
        )
    if not input_tools.args_from_metadata(cardTool, "noSmearing"):
        cardTool.addSystematic("muonResolutionSyst_responseWeights", 
            mirror = True,
            processes=['single_v_samples'],
            group="resolutionCorr",
            baseName="Resolution_correction_",
            systAxes=["smearing_variation"],
            passToFakes=passSystToFakes
        )
       
    
    # Previously we had a QCD uncertainty for the mt dependence on the fakes, see: https://github.com/WMass/WRemnants/blob/f757c2c8137a720403b64d4c83b5463a2b27e80f/scripts/combine/setupCombineWMass.py#L359

    return cardTool

def main(args, xnorm=False):
    fitvar = args.fitvar[0].split("-")
    cardTool = setup(args, inputFile=args.inputFile[0], fitvar=fitvar, xnorm=xnorm)

    cardTool.setOutput(args.outfolder, fitvars=fitvar, doStatOnly=args.doStatOnly, postfix=args.postfix)

    cardTool.writeOutput(args=args, forceNonzero=not args.unfolding, check_systs=not args.unfolding)
    return

if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()
    
    logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)
                                  
    if args.theoryAgnostic:
        args.unfolding = True
        logger.warning("For now setting --theoryAgnostic activates --unfolding, they should do the same things")
        if args.genAxis is None:
            args.genAxis = ["absYVgenSig", "ptVgenSig", "helicity"]
            logger.warning("Automatically setting '--genAxis absYVgenSig ptVgenSig helicity' for theory agnostic analysis")
        # The following is temporary, just to avoid passing the option explicitly
        logger.warning("For now setting --theoryAgnostic activates --doStatOnly")
        args.doStatOnly = True
    
    if args.hdf5: 
        writer = HDF5Writer.HDF5Writer()

        # loop over all files
        for i, ifile in enumerate(args.inputFile):
            fitvar = args.fitvar[i].split("-")
            cardTool = setup(args, ifile, fitvar, xnorm=args.fitresult is not None)
            writer.add_channel(cardTool)
            if args.unfolding:
                cardTool = setup(args, ifile, fitvar, xnorm=True)
                writer.add_channel(cardTool)
        if args.fitresult:
            writer.set_fitresult(args.fitresult)
        writer.write(args)
    else:
        if len(args.inputFile) > 1:
            raise IOError(f"Multiple input files only supported within --hdf5 mode")

        main(args)
        if args.unfolding:
            # in case of unfolding and hdf5, the xnorm histograms are directly written into the hdf5
            main(args, xnorm=True)

    logging.summary()
