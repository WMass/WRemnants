#!/usr/bin/env python3
from wremnants import CardTool,theory_tools,syst_tools,combine_helpers
from wremnants import histselections as sel
from wremnants.datasets.datagroups2016 import make_datagroups_2016
from utilities import common, logging, input_tools
import argparse
import os
import pathlib
import hist
import copy
import math
import time

scriptdir = f"{pathlib.Path(__file__).parent}"
data_dir = f"{pathlib.Path(__file__).parent}/../../wremnants/data/"

def make_parser(parser=None):
    if not parser:
        parser = common.common_parser_combine()
    parser.add_argument("--fitvar", help="Variable to fit", default="pt-eta")
    parser.add_argument("--noEfficiencyUnc", action='store_true', help="Skip efficiency uncertainty (useful for tests, because it's slow). Equivalent to --excludeNuisances '.*effSystTnP|.*effStatTnP' ")
    parser.add_argument("--ewUnc", action='store_true', help="Include EW uncertainty")
    parser.add_argument("--pseudoData", type=str, help="Hist to use as pseudodata")
    parser.add_argument("--pseudoDataIdx", type=int, default=0, help="Variation index to use as pseudodata")
    parser.add_argument("--pseudoDataFile", type=str, help="Input file for pseudodata (if it should be read from a different file)", default=None)
    parser.add_argument("--pseudoDataProcsRegexp", type=str, default=".*", help="Regular expression for processes taken from pseudodata file (all other processes are automatically got from the nominal file). Data is excluded automatically as usual")
    parser.add_argument("-x",  "--excludeNuisances", type=str, default="", help="Regular expression to exclude some systematics from the datacard")
    parser.add_argument("-k",  "--keepNuisances", type=str, default="", help="Regular expression to keep some systematics, overriding --excludeNuisances. Can be used to keep only some systs while excluding all the others with '.*'")
    parser.add_argument("--scaleMuonCorr", type=float, default=1.0, help="Scale up/down dummy muon scale uncertainty by this factor")
    parser.add_argument("--noHist", action='store_true', help="Skip the making of 2D histograms (root file is left untouched if existing)")
    parser.add_argument("--effStatLumiScale", type=float, default=None, help="Rescale equivalent luminosity for efficiency stat uncertainty by this value (e.g. 10 means ten times more data from tag and probe)")
    parser.add_argument("--binnedScaleFactors", action='store_true', help="Use binned scale factors (different helpers and nuisances)")
    parser.add_argument("--isoEfficiencySmoothing", action='store_true', help="If isolation SF was derived from smooth efficiencies instead of direct smoothing")
    parser.add_argument("--xlim", type=float, nargs=2, default=None, help="Restrict x axis to this range")
    parser.add_argument("--unfolding", action='store_true', help="Prepare datacard for unfolding")
    parser.add_argument("--genAxis", type=str, default=None, nargs="+", help="Specify which gen axis should be used in unfolding, if 'None', use all (inferred from metadata).")
    parser.add_argument("--fitXsec", action='store_true', help="Fit signal inclusive cross section")
    parser.add_argument("--correlatedNonClosureNuisances", action='store_true', help="get systematics from histograms for the Z non-closure nuisances without decorrelation in eta and pt")
    parser.add_argument("--sepImpactForNC", action="store_true", help="use a dedicated impact gropu for non closure nuisances, instead of putting them in muonScale")
    parser.add_argument("--genModel", action="store_true", help="Produce datacard with the xnorm as model (binned according to axes defined in --fitvar)")
    # TODO: move next option in common.py? 
    parser.add_argument("--absolutePathInCard", action="store_true", help="In the datacard, set Absolute path for the root file where shapes are stored")
    return parser

def main(args,xnorm=False):   

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
    logger.debug(f"Filtering these groups of processes: {args.filterProcGroups}")
    logger.debug(f"Excluding these groups of processes: {args.excludeProcGroups}")
    
    datagroups = make_datagroups_2016(args.inputFile, excludeGroups=excludeGroup, filterGroups=filterGroup, applySelection= not xnorm)

    if args.xlim:
        if len(args.fitvar.split("-")) > 1:
            raise ValueError("Restricting the x axis not supported for 2D hist")
        s = hist.tag.Slicer()
        datagroups.setGlobalAction(lambda h: h[{args.fitvar : s[complex(0, args.xlim[0]):complex(0, args.xlim[1])]}])

    wmass = datagroups.wmass
    wlike = datagroups.wlike

    constrainMass = False

    if wmass:
        name = "WMass"
    elif wlike:
        name = "ZMassWLike"
    else:
        name = "ZMassDilepton"
        constrainMass = "mll" not in args.fitvar

    tag = name+"_"+args.fitvar.replace("-","_")
    if args.doStatOnly:
        tag += "_statOnly"
    if args.postfix:
        tag += f"_{args.postfix}"

    outfolder = f"{args.outfolder}/{tag}/"
    if not os.path.isdir(outfolder):
        os.makedirs(outfolder)

    templateDir = f"{scriptdir}/Templates/WMass"

    if args.unfolding and args.fitXsec:
        raise ValueError("Options --unfolding and --fitXsec are incompatible. Please choose one or the other")
    elif args.fitXsec:
        datagroups.unconstrainedProcesses.append("Wmunu" if wmass else "Zmumu")
    elif args.unfolding:
        constrainMass = True
        datagroups.setGenAxes(args.genAxis)
        
        if wmass:
            # gen level bins, split by charge
            datagroups.defineSignalBinsUnfolding("Wmunu", "Wmunu_qGen0", member_filter=lambda x: x.name.startswith("Wminusmunu"))
            datagroups.defineSignalBinsUnfolding("Wmunu", "Wmunu_qGen1", member_filter=lambda x: x.name.startswith("Wplusmunu"))
            # out of acceptance contribution
            datagroups.groups["Wmunu"].deleteMembers([m for m in datagroups.groups["Wmunu"].members if not m.name.startswith("Bkg")])
        else:
            datagroups.defineSignalBinsUnfolding("Zmumu", member_filter=lambda x: x.name.startswith("Zmumu"))
            # out of acceptance contribution
            datagroups.groups["Zmumu"].deleteMembers([m for m in datagroups.groups["Zmumu"].members if not m.name.startswith("Bkg")])

    if args.noHist and args.noStatUncFakes:
        raise ValueError("Option --noHist would override --noStatUncFakes. Please select only one of them")

    suffix = '_xnorm' if xnorm else ''

    # Start to create the CardTool object, customizing everything
    cardTool = CardTool.CardTool(f"{outfolder}/{name}_{{chan}}{suffix}.txt")
    cardTool.setDatagroups(datagroups)
    logger.debug(f"Making datacards with these processes: {cardTool.getProcesses()}")
    cardTool.setNominalTemplate(f"{templateDir}/main.txt")
    cardTool.setProjectionAxes(args.fitvar.split("-"))
    if args.absolutePathInCard:
        cardTool.setAbsolutePathShapeInCard()
    if args.sumChannels or xnorm or name in ["ZMassDilepton"]:
        cardTool.setChannels(["inclusive"])
        cardTool.setWriteByCharge(False)
    if xnorm:
        histName = "xnorm"
        cardTool.setWriteByCharge(False)
        cardTool.setHistName(histName)
        cardTool.setNominalName(histName)
        datagroups.select_xnorm_groups() # only keep processes where xnorm is defined
        if args.unfolding:
            cardTool.setProjectionAxes(["count"])
        else:
            if wmass:
                # add gen charge as additional axis
                datagroups.groups["Wmunu"].add_member_axis("qGen", datagroups.results, 
                    member_filters={-1: lambda x: x.name.startswith("Wminusmunu"), 1: lambda x: x.name.startswith("Wplusmunu")}, 
                    hist_filter=lambda x: x.startswith("xnorm"))
                datagroups.deleteGroup("Fake")
            cardTool.unroll = True
            # remove projection axes from gen axes, otherwise they will be integrated before
            datagroups.setGenAxes([a for a in datagroups.gen_axes if a not in cardTool.project])
    if args.unfolding:
        cardTool.addPOISumGroups()
    if args.noHist:
        cardTool.skipHistograms()
    cardTool.setOutfile(os.path.abspath(f"{outfolder}/{name}CombineInput{suffix}.root"))
    cardTool.setFakeName(args.qcdProcessName)
    cardTool.setSpacing(52)
    if args.noStatUncFakes:
        cardTool.setProcsNoStatUnc(procs=args.qcdProcessName, resetList=False)
    cardTool.setCustomSystForCard(args.excludeNuisances, args.keepNuisances)
    if args.pseudoData:
        cardTool.setPseudodata(args.pseudoData, args.pseudoDataIdx, args.pseudoDataProcsRegexp)
        if args.pseudoDataFile:
            cardTool.setPseudodataDatagroups(make_datagroups_2016(args.pseudoDataFile,
                                                                  excludeGroups=excludeGroup,
                                                                  filterGroups=filterGroup)
            )
    cardTool.setLumiScale(args.lumiScale)

    logger.info(f"cardTool.allMCProcesses(): {cardTool.allMCProcesses()}")
        
    passSystToFakes = wmass and not args.skipSignalSystOnFakes and args.qcdProcessName not in excludeGroup and (filterGroup == None or args.qcdProcessName in filterGroup) and not xnorm

    single_v_samples = cardTool.filteredProcesses(lambda x: x[0] in ["W", "Z"] and ("mu" in x or "tau" in x))
    single_v_nonsig_samples = cardTool.filteredProcesses(lambda x: x[0] == ("Z" if wmass else "W"))
    single_vmu_samples = list(filter(lambda x: "mu" in x, single_v_samples))
    signal_samples = list(filter(lambda x: x[0] == ("W" if wmass else "Z"), single_vmu_samples))
    signal_samples_inctau = list(filter(lambda x: x[0] == ("W" if wmass else "Z"), single_v_samples))

    allMCprocesses_noQCDMC = [x for x in cardTool.allMCProcesses() if x != "QCD"]
    
    logger.info(f"All MC processes {allMCprocesses_noQCDMC}")
    logger.info(f"Single V samples: {single_v_samples}")
    logger.info(f"Single V no signal samples: {single_v_nonsig_samples}")
    logger.info(f"Signal samples: {signal_samples}")

    constrainedZ = constrainMass and not wmass
    massSkip = [(f"^massShift{i}MeV.*",) for i in range(0, 110 if constrainedZ else 100, 10)]
    if not (constrainMass or wmass):
        massSkip.append(("^massShift2p1MeV.*",))

    cardTool.addSystematic("massWeight", 
                            processes=signal_samples_inctau,
                            group=f"massShift{'W' if wmass else 'Z'}",
                            skipEntries=massSkip,
                            mirror=False,
                            systNamePrepend="W" if wmass else "Z",
                            #TODO: Name this
                            noConstraint=not constrainMass,
                            systAxes=["massShift"],
                            passToFakes=passSystToFakes,
    )
    
    if args.doStatOnly:
        # print a card with only mass weights, no longer need a dummy syst since combinetf is fixed now
        #cardTool.addLnNSystematic("dummy", processes=["Top", "Diboson"] if wmass else ["Other"], size=1.001, group="dummy")
        cardTool.writeOutput(args=args, xnorm=xnorm)
        logger.info("Using option --doStatOnly: the card was created with only mass nuisance parameter")
        return

    if not xnorm:
        if wmass:
            cardTool.addSystematic("luminosity",
                                   processes=allMCprocesses_noQCDMC,
                                   outNames=["lumiDown", "lumiUp"],
                                   group="luminosity",
                                   systAxes=["downUpVar"],
                                   labelsByAxis=["downUpVar"],
                                   passToFakes=passSystToFakes)

        else:
            # TOCHECK: no fakes here, most likely
            cardTool.addLnNSystematic("luminosity", processes=allMCprocesses_noQCDMC, size=1.012, group="luminosity")
    else:
        pass

    if args.ewUnc:
        cardTool.addSystematic(f"horacenloewCorr", 
            processes=single_v_samples,
            mirror=True,
            group="theory_ew",
            systAxes=["systIdx"],
            labelsByAxis=["horacenloewCorr"],
            skipEntries=[(0, -1), (2, -1)],
            passToFakes=passSystToFakes,
        )

    if not args.noEfficiencyUnc and not xnorm:

        ## this is only needed when using 2D SF from 3D with ut-integration, let's comment for now
        # if wmass:
        #     cardTool.addSystematic("sf2d", 
        #         processes=allMCprocesses_noQCDMC,
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
                processes=allMCprocesses_noQCDMC,
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
            #         processes=allMCprocesses_noQCDMC,
            #         passToFakes=passSystToFakes,
            #         systNameReplace=nameReplace,
            #         scale=scale,
            #         splitGroup=splitGroupDict,
            #         decorrelateByBin = decorrDictEff
            #     )

    to_fakes = passSystToFakes and not args.noQCDscaleFakes and not xnorm
    combine_helpers.add_pdf_uncertainty(cardTool, single_v_samples, passSystToFakes, from_corr=args.pdfUncFromCorr, scale=args.scalePdf)
    combine_helpers.add_modeling_uncertainty(cardTool, args.minnloScaleUnc, signal_samples_inctau, 
        single_v_nonsig_samples if not xnorm else [], to_fakes, args.resumUnc, wmass, scaleTNP=args.scaleTNP, rebin_pt=args.rebinPtV)

    if not xnorm:
        msv_config_dict = {
            "smearingWeights":{
                "hist_name": "muonScaleSyst_responseWeights",
                "syst_axes": ["unc", "downUpVar"],
                "syst_axes_labels": ["unc", "downUpVar"]
            },
            "massWeights":{
                "hist_name": "muonScaleSyst",
                "syst_axes": ["downUpVar", "scaleEtaSlice"],
                "syst_axes_labels": ["downUpVar", "ieta"]
            },
            "manualShift":{
                "hist_name": "muonScaleSyst_manualShift",
                "syst_axes": ["downUpVar"],
                "syst_axes_labels": ["downUpVar"]
            }
        }

        # FIXME: remove this once msv from smearing weights is implemented for the Z
        msv_config = msv_config_dict[args.muonScaleVariation] if wmass else msv_config_dict["massWeights"]

        cardTool.addSystematic(msv_config['hist_name'], 
            processes=single_v_samples if wmass else single_vmu_samples,
            group="muonScale",
            baseName="CMS_scale_m_",
            systAxes=msv_config['syst_axes'],
            labelsByAxis=msv_config['syst_axes_labels'],
            passToFakes=passSystToFakes,
            scale = args.scaleMuonCorr
        )
        cardTool.addSystematic("muonL1PrefireSyst", 
            processes=allMCprocesses_noQCDMC,
            group="muonPrefire",
            baseName="CMS_prefire_syst_m",
            systAxes=["downUpVar"],
            labelsByAxis=["downUpVar"],
            passToFakes=passSystToFakes,
        )
        cardTool.addSystematic("muonL1PrefireStat", 
            processes=allMCprocesses_noQCDMC,
            group="muonPrefire",
            baseName="CMS_prefire_stat_m_",
            systAxes=["downUpVar", "etaPhiRegion"],
            labelsByAxis=["downUpVar", "etaPhiReg"],
            passToFakes=passSystToFakes,
        )
        cardTool.addSystematic("ecalL1Prefire", 
            processes=allMCprocesses_noQCDMC,
            group="ecalPrefire",
            baseName="CMS_prefire_ecal",
            systAxes=["downUpVar"],
            labelsByAxis=["downUpVar"],
            passToFakes=passSystToFakes,
        )
        if wmass or wlike:
            combine_helpers.add_recoil_uncertainty(cardTool, signal_samples, passSystToFakes=passSystToFakes, flavor="mu")

        if wmass:
            non_closure_scheme = input_tools.args_from_metadata(cardTool, "nonClosureScheme")
            if non_closure_scheme == "A-M-separated":
                cardTool.addSystematic("Z_non_closure_parametrized_A", 
                    processes=single_v_samples,
                    group="nonClosure" if args.sepImpactForNC else "muonScale",
                    baseName="Z_nonClosure_parametrized_A_",
                    systAxes=["unc", "downUpVar"] if not (args.correlatedNonClosureNuisances) else ["downUpVar"],
                    labelsByAxis=["unc", "downUpVar"] if not (args.correlatedNonClosureNuisances) else ["downUpVar"],
                    passToFakes=passSystToFakes
                )
            if non_closure_scheme in ["A-M-separated", "binned-plus-M"]:
                cardTool.addSystematic("Z_non_closure_parametrized_M", 
                    processes=single_v_samples,
                    group="nonClosure" if args.sepImpactForNC else "muonScale",
                    baseName="Z_nonClosure_parametrized_M_",
                    systAxes=["unc", "downUpVar"] if not (args.correlatedNonClosureNuisances) else ["downUpVar"],
                    labelsByAxis=["unc", "downUpVar"] if not (args.correlatedNonClosureNuisances) else ["downUpVar"],
                    passToFakes=passSystToFakes
                )            
            if non_closure_scheme == "A-M-combined":
                cardTool.addSystematic("Z_non_closure_parametrized", 
                    processes=single_v_samples,
                    group="nonClosure" if args.sepImpactForNC else "muonScale",
                    baseName="Z_nonClosure_parametrized_",
                    systAxes=["unc", "downUpVar"] if not (args.correlatedNonClosureNuisances) else ["downUpVar"],
                    labelsByAxis=["unc", "downUpVar"] if not (args.correlatedNonClosureNuisances) else ["downUpVar"],
                    passToFakes=passSystToFakes
                )
            if non_closure_scheme in ["binned", "binned-plus-M"]:
                cardTool.addSystematic("Z_non_closure_binned", 
                    processes=single_v_samples,
                    group="nonClosure" if args.sepImpactForNC else "muonScale",
                    baseName="Z_nonClosure_binned_",
                    systAxes=["unc_ieta", "unc_ipt", "downUpVar"] if not (args.correlatedNonClosureNuisances) else ["downUpVar"],
                    labelsByAxis=["unc_ieta", "unc_ipt", "downUpVar"] if not (args.correlatedNonClosureNuisances) else ["downUpVar"],
                    passToFakes=passSystToFakes
                )

            #cardTool.addLnNSystematic("CMS_Fakes", processes=[args.qcdProcessName], size=1.05, group="MultijetBkg")
            cardTool.addLnNSystematic("CMS_Top", processes=["Top"], size=1.06)
            cardTool.addLnNSystematic("CMS_VV", processes=["Diboson"], size=1.16)
    
            ## FIXME 1: with the jet cut removed this syst is probably no longer needed, but one could still consider
            ## it to cover for how much the fake estimate changes when modifying the composition of the QCD region
            ## FIXME 2: it doesn't really make sense to mirror this one since the systematic goes only in one direction
            # cardTool.addSystematic(f"qcdJetPt30", 
            #                        processes=["Fake"],
            #                        mirror=True,
            #                        group="MultijetBkg",
            #                        systAxes=[],
            #                        outNames=["qcdJetPt30Down", "qcdJetPt30Up"],
            #                        passToFakes=passSystToFakes,
            # )
            #

            ## Remove for now since it seems redundant after adding the dphi cut
            ## keep in case it is needed again in the near future (we still have to test deepmet)
            # if "Fake" not in excludeGroup:
            #     for charge in ["plus", "minus"]:
            #         chargeId = "q1" if charge == "plus" else "q0"
            #         decorrDict = {}
            #         # decorrDict = {                        
            #         #     "xy" : {
            #         #         "label" : ["eta", "pt"],
            #         #         "edges": [[round(-2.4+i*0.4,1) for i in range(13)], [round(26.0+i*2,1) for i in range(16)]]
            #         #     }
            #         # }
            #         outnames = [f"mtCorrFakes_{chargeId}{upd}" for upd in ["Up", "Down"]]
            #         cardTool.addSystematic(f"nominal", # this is the histogram to read
            #                                systAxes=[],
            #                                processes=["Fake"],
            #                                mirror=True,
            #                                group="MultijetBkg",
            #                                outNames=outnames, # actual names for nuisances
            #                                rename=f"mtCorrFakes_{chargeId}", # this name is used only to identify the syst in CardTool's syst list
            #                                action=sel.applyCorrection,
            #                                doActionBeforeMirror=True, # to mirror after the histogram has been created
            #                                actionArgs={"scale": 1.0,
            #                                            "corrFile" : f"{data_dir}/fakesWmass/fakerateFactorMtBasedCorrection_vsEtaPt.root",
            #                                            "corrHist": f"etaPtCharge_mtCorrection_{charge}",
            #                                            "offsetCorr": 1.0,
            #                                            "createNew": True},
            #                                decorrelateByBin=decorrDict
            #         )
        else:
            cardTool.addLnNSystematic("CMS_background", processes=["Other"], size=1.15)

    cardTool.writeOutput(args=args, xnorm=xnorm, forceNonzero=not args.unfolding, check_systs=not args.unfolding)
    logger.info(f"Output stored in {outfolder}")
    
if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()

    logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)
    
    time0 = time.time()

    if args.genModel:
        main(args, xnorm=True)
    else:
        main(args)
        if args.unfolding:
            main(args, xnorm=True)

    logger.info(f"Running time: {time.time()-time0}")
