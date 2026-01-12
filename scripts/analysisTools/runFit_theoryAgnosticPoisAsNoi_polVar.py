import os

# for setup and fit
skipSetup = 0
skipFit = 1
# for plots
skipImpacts = 1
skipCorrelation = 1
skipCorrLine = 1
skipDiffnuis = 1
skipCompareDiffnuis = 1
skipTemplates = 1  # check settings
#
justPrint = 1

foldEtaIntoAbsEta = True

## histmaker
# /usr/bin/time -v python scripts/histmakers/mw_with_mu_eta_pt.py -o /scratch/ciprianm/CombineStudies/theoryAgnostic_pol/January2025/x0p50_y3p00_THAGNV0/ --maxFiles -1 --postfix x0p50_y3p00_THAGNV0_utSignAxis --addAxisSignUt --analysisMode theoryAgnosticPolVar --poiAsNoi --theoryAgnosticFilePath /scratch/ciprianm/TheoryAgnosticInputFiles/ --theoryAgnosticFileTag x0p50_y3p00_THAGNV0 --theoryAgnosticSplitOOA -v 4

# /usr/bin/time -v python scripts/histmakers/mw_with_mu_eta_pt.py -o /scratch/mciprian/CombineStudies/theoryAgnostic_pol/x0p40_y3p50_V4/ -v 4  --dataPath root://eoscms.cern.ch//store/cmst3/group/wmass/w-mass-13TeV/NanoAOD/ --filterProcs Data Wmunu --maxFiles -1  [ -p splitOOA|splitOOA_oneMCfileEvery2 ] [ --oneMCfileEveryN 2 ] theoryAgnosticPolVar --theoryAgnosticFilePath /path/to/files/  --theoryAgnosticFileTag x0p40_y3p50_V4  --theoryAgnosticSplitOOA
# --theoryAgnosticSplitOOA should be used by default

splitOOA = True  # use out-of-acceptance as a different process (it assumes the histograms were created accordingly)
onlySignal = False  #  out-of-acceptance is excluded when splitOOA = True
onlySignalAndOOA = False  # (requires onlySignal=True to be effective) signal only but keep OOA as background, with all uncertainties if applied
doStatOnly = False
noFake = False  # irrelevant when onlySignal=True
noPDFandQCDtheorySystOnSignal = False  # irrelevant when doStatOnly=True
tag = "x0p50_y3p00_THAGNV0"  # "x0p40_y3p50_V6" # "x0p40_y3p50_V6" # "x0p40_y3p50_V4" # "x0p30_y3p00_V4"
oneMCfileEveryN = 2
testFolder = f"oneMCfileEvery{oneMCfileEveryN}" if oneMCfileEveryN > 1 else "fullStat"
testFolder += "_utSignAxis"
projectToNewLumi = -1.0  # set negative to skip it.
lumiScaleVarianceLinearly = [
    "Data"
]  # [], ["Data", "MC"], use in conjunction with projectToNewLumi when it is not negative

splitOOAtag = ""
setupLumiOption = ""
# setupFakeOption = " --fakeEstimation simple --fakeSmoothingMode binned"
setupFakeOption = (
    " --rebinBeforeSelection --noPolVarOnFake  --fakeSmoothingMode fakerate "
)
if splitOOA:
    splitOOAtag = "_splitOOA"
    testFolder = f"{testFolder}{splitOOAtag}"
if projectToNewLumi > 0.0:
    lumiFactor = str(round(projectToNewLumi, 2)).replace(".", "p")
    testFolder = (
        f"{testFolder}_projectLumiX{lumiFactor}{''.join(lumiScaleVarianceLinearly)}"
    )
    setupLumiOption = (
        f" --effStatLumiScale {projectToNewLumi} --lumiScale {projectToNewLumi}"
    )
    if len(lumiScaleVarianceLinearly):
        setupLumiOption += f" --lumiScaleVarianceLinearly {' '.join(x.lower() for x in lumiScaleVarianceLinearly)}"

if doStatOnly:
    doSystTests = {"tag": "noSysts", "exclude": None}
else:
    # doSystTests = {"tag"  : "allSysts_noQCD_keepResumNonpert",
    #               "exclude" : '.*recoil|.*QCDscale|.*resum'
    #               }
    doSystTests = {"tag": "allSysts", "exclude": ".*recoil"}

# baseOutdir = f"/scratch/mciprian/CombineStudies/theoryAgnostic_pol/fromTanmay_09Apr2024/{tag}/"
# baseOutdir = f"/scratch/mciprian/CombineStudies/theoryAgnostic_pol/{tag}/"
baseOutdir = f"/scratch/ciprianm/CombineStudies/theoryAgnostic_pol/{tag}/"
basePlotDir = "scripts/analysisTools/plots/fromMyWremnants/fitResults/theoryAgnostic_polVar/pdfCT18Z_April2024/"

inputFileHDF5 = f"{baseOutdir}/mw_with_mu_eta_pt_scetlib_dyturboCorr_maxFiles_m1_{tag}_utSignAxis.hdf5"
if oneMCfileEveryN > 1:
    inputFileHDF5 = f"{baseOutdir}/mw_with_mu_eta_pt_scetlib_dyturboCorr_maxFiles_m1_{tag}_utSignAxis_oneMCfileEvery{oneMCfileEveryN}.hdf5"

if noPDFandQCDtheorySystOnSignal and not doStatOnly:
    testFolder += "/noPDFandQCDtheorySystOnSignal/"

procFolder = "onlySignal" if onlySignal else "allProcsNoFake" if noFake else "allProcs"
if onlySignal and onlySignalAndOOA:
    procFolder = "onlySignalAndOOA"

outdir = f"{baseOutdir}/{testFolder}/{procFolder}/"

theoryAgnosticOptions = " --analysisMode theoryAgnosticPolVar --poiAsNoi"

setupRabbitOptions = setupLumiOption + setupFakeOption
if doStatOnly:
    setupRabbitOptions += " --doStatOnly"
elif noPDFandQCDtheorySystOnSignal:
    setupRabbitOptions += " --noPDFandQCDtheorySystOnSignal"

if onlySignal:
    setupRabbitOptions += " --filterProcGroups Data Wmunu"
    if not onlySignalAndOOA:
        setupRabbitOptions += " --excludeProcGroups WmunuOOA"
elif noFake:
    setupRabbitOptions += " --excludeProcGroups Fake"

setupRabbitOptionsTraditional = setupRabbitOptions.replace("--noPolVarOnFake", "")
setupRabbitOptions += f" {theoryAgnosticOptions}"

baseCoeffs = ["UL", "A0", "A1", "A2", "A3", "A4"]
# coeffs = [x for x in baseCoeffs]
# coeffs = ["UL"]
# coeffs = [f"ULand{a}" for a in ["A0", "A1", "A2", "A3", "A4"]]
# coeffs = [f"ULandA0andA4and{a}" for a in ["A1", "A2", "A3"]]
# coeffs = ["and".join(x for x in baseCoeffs), "and".join(x for x in baseCoeffs if x != "A3")]
coeffs = [
    "traditional",
    "UL",
    "ULandA4",
    "ULandA0andA4",
    "and".join(x for x in baseCoeffs if x not in ["A1", "A2"]),
    "and".join(x for x in baseCoeffs),
]
# coeffs = ["and".join(x for x in baseCoeffs if x not in ["A1", "A3"])]
# coeffs = ["and".join(x for x in baseCoeffs)]
# coeffs = ["and".join(x for x in baseCoeffs if x not in ["A1", "A2"])]
# coeffs =["ULandA3"]


def safeSystem(cmd, dryRun=False, quitOnFail=True):
    print(cmd)
    if not dryRun:
        res = os.system(cmd)
        if res:
            print("-" * 30)
            print(
                "safeSystem(): error occurred when executing the following command. Aborting"
            )
            print(cmd)
            print("-" * 30)
            if quitOnFail:
                quit()
        return res
    else:
        return 0


for c in coeffs:

    if c == "traditional":
        subFolder = "traditionalFit"
        toReject = None
    elif "and" in c:
        if all(x in c for x in baseCoeffs):
            subFolder = "allCoeff"
            toReject = None
        else:
            subFolder = f"{c}"
            toReject = "|".join([x for x in baseCoeffs if x not in c])
    else:
        subFolder = f"only{c}"
        toReject = "|".join([x for x in baseCoeffs if x != c])

    subFolderBase = subFolder  # backup since it is modified later

    customOptExclude = None
    if doSystTests:
        subFolder += f"_{doSystTests['tag']}"
        customOptExclude = doSystTests["exclude"]
    if toReject != None:
        if customOptExclude != None and customOptExclude != "":
            customOptExclude = f"{customOptExclude}|.*(theory|pol).*({toReject})"
        else:
            customOptExclude = f".*(theory|pol).*({toReject})"

    customOpt = f" -x '{customOptExclude}' "
    if customOptExclude is None or customOptExclude == "":
        customOpt = ""

    if foldEtaIntoAbsEta and c != "traditional":
        # customOpt += " --foldEtaIntoAbsEta"
        # subFolder += "_absEta"
        customOpt += " --absval 1"

    mainOutputFolder = f"{outdir}/{subFolder}"

    setupRabbitOpts = (
        setupRabbitOptionsTraditional if c == "traditional" else setupRabbitOptions
    )
    cmdCard = f"/usr/bin/time -v python scripts/combine/setupRabbit.py -i {inputFileHDF5} -o {mainOutputFolder}/ --absolutePathInCard {setupRabbitOpts} {customOpt}"

    # etaVar = "abseta" if foldEtaIntoAbsEta else "eta"
    etaVar = "eta"  # patch, it seems the folder name stays eta also with --absval 1
    analysisFolderBase = f"WMass_{etaVar}_pt_charge"
    fitFolder = analysisFolderBase
    if doStatOnly:
        fitFolder += "_statOnly"

    cmdFit = f"python WRemnants/scripts/combine/fitManager.py -i {mainOutputFolder}/{fitFolder}/ --skip-fit-data --comb"

    if not skipSetup:
        safeSystem(cmdCard, dryRun=justPrint)
    if not skipFit:
        safeSystem(cmdFit, dryRun=justPrint)
    print()

    # next part for plotting
    fullStatFolder = mainOutputFolder.replace(testFolder, f"fullStat{splitOOAtag}")
    filesFraction = oneMCfileEveryN if oneMCfileEveryN > 1 else 2
    halfStatFolder = mainOutputFolder.replace(
        testFolder, f"oneMCfileEvery{filesFraction}{splitOOAtag}"
    )

    traditionalFitFolder = mainOutputFolder.replace(subFolderBase, "traditionalFit")

    impact_postfix = f"asimov_{procFolder}_{subFolder}"

    fitResultFile = f"{fullStatFolder}/{fitFolder}/nominal/fit/hessian/fitresults_123456789_Asimov_bbb1_cxs0.root"
    baseOutputPlotPath = f"{basePlotDir}/checkFullStat{splitOOAtag}/{tag}/{procFolder}/{analysisFolderBase}"

    cmdImp = f"python scripts/analysisTools/w_mass_13TeV/makeImpactsOnMW.py {fitResultFile} -o {baseOutputPlotPath}/makeImpactsOnMW/ --scaleToMeV --showTotal --postfix {impact_postfix}  -x '.*eff_(stat|syst)_|.*AlphaS|.*nonClosure|.*resolutionCrctn|.*scaleCrctn|.*scaleClos|.*polVar|.*QCDscale$|.*resum$|.*(muon|ecal)Prefire'  --compareFile {halfStatFolder}/{fitFolder}/nominal/fit/hessian/fitresults_123456789_Asimov_bbb1_cxs0.root --printAltVal --legendEntries 'Full MC stat' '1/2 MC stat'"

    if not skipImpacts:
        safeSystem(cmdImp, dryRun=justPrint)
        print()

    corr_postfix = f"{subFolder}"
    corr_stat = "_statOnly" if doStatOnly else ""

    cmdCorr = f"python scripts/analysisTools/w_mass_13TeV/subMatrix.py {fitResultFile} -o {baseOutputPlotPath}/subMatrix/ --postfix {corr_postfix}_withMW -p '.*polVar|.*massShiftW' --uniqueString polVar{corr_stat} --title 'Th. agn., {subFolder}' --noTextMatrix"

    if not skipCorrelation:
        safeSystem(cmdCorr, dryRun=justPrint)
        print()

    nCorrLinePar = 30
    cmdCorrLine = f"python scripts/analysisTools/w_mass_13TeV/getCorrelationLine.py {fitResultFile} -o {baseOutputPlotPath}/getCorrelationLine/ --postfix {subFolder} -p massShiftW100MeV -m none --title '{subFolder}: first {nCorrLinePar} parameters more correlated with m_{{W}}' -n {nCorrLinePar}"

    if not skipCorrLine:
        safeSystem(cmdCorrLine, dryRun=justPrint)
        print()

    diffnuis_postfix = f"{subFolder}"

    cmdDiff = f"python scripts/analysisTools/w_mass_13TeV/diffNuisances.py {fitResultFile} -o {baseOutputPlotPath}/diffNuisances/ --postfix {diffnuis_postfix} --pois '.*polVar' --uniqueString polVar --y-setting -0.5 -0.25 0 0.25 0.5 --defaultYmax 0.75"

    if not skipDiffnuis:
        safeSystem(cmdDiff, dryRun=justPrint)
        print()

    if doSystTests["tag"] == "allSysts":

        systToPlot = {
            "resumAndSCETlib": " --pois '.*resum|.*scetlib' ",
            "PDFandAlphaS": " --pois '.*pdf' ",
            # "QCDscaleW"            : " --pois '.*QCDscaleW.*helicity_[0-4]+' --bm 0.45 "
        }
        # for hel in range(5):
        #    systToPlot[f"QCDscaleW_A{hel}"] = f" --pois '.*QCDscaleW.*helicity_{hel}_' --bm 0.45 "

        for s in systToPlot.keys():
            expr = systToPlot[s]

            cmdCompareDiff = f"python scripts/analysisTools/w_mass_13TeV/diffNuisances.py {fitResultFile} -o {baseOutputPlotPath}/diffNuisances/ --postfix {diffnuis_postfix}_compareTraditional {expr} --uniqueString {s} --y-setting -1.5 -0.5 0 0.5 1.5 --defaultYmax 1.5 --expected-infile {traditionalFitFolder}/{fitFolder}/nominal/fit/hessian/fitresults_123456789_Asimov_bbb1_cxs0.root --postfitLegendLabelObs 'Theory agnostic fit' --postfitLegendLabelExp 'Traditional fit' "

            if not skipCompareDiffnuis:
                safeSystem(cmdCompareDiff, dryRun=justPrint)
                print()

# This is partially standalone since it uses the base coefficients,
# it can run using the settings of the last item of the previous loop
if oneMCfileEveryN == 1 and not skipTemplates:

    charges = ["plus", "minus"]

    for charge in charges:
        for c in baseCoeffs:

            cmdTempl = f"python scripts/analysisTools/w_mass_13TeV/makeSystRatios.py {fullStatFolder}/{fitFolder}/WMassCombineInput.root {baseOutputPlotPath}/makeSystRatios/muonRecoCharge_{charge}/{c}/Wmunu/ -s '.*polVarW_{c}_{charge}' --systPostfix 'polVarW_{c}_{charge}' -p 'Wmunu' -c {charge} --plot 2D --compareSingleSystToNomi"

            safeSystem(cmdTempl, dryRun=justPrint)
            print()


## other utility commands
# python scripts/analysisTools/w_mass_13TeV/getCorrelationLine.py /scratch/mciprian/CombineStudies/theoryAgnostic_pol/x0p40_y3p50_V4//fullStat_splitOOA/onlySignal//allCoeff_noSysts/WMass_eta_pt_charge_statOnly/nominal/fit/hessian/fitresults_123456789_Asimov_bbb1_cxs0.root -o scripts/analysisTools/plots/fromMyWremnants/fitResults/TESTS/theoryAgnostic_polVar//checkFullStat_splitOOA/x0p40_y3p50_V4/onlySignal/WMass_eta_pt_charge/getCorrelationLine/ --postfix allCoeff_noSysts -p massShiftW100MeV -m none -n -1 --title "Correlation of signal variations with m_{W}: fit with all coeffs"


# fit setup traditonal (here for 1/2 stat) with eta or abseta
# /usr/bin/time -v python scripts/combine/setupRabbit.py -i /scratch/mciprian/CombineStudies/theoryAgnostic_pol/x0p30_y3p00_V8//mw_with_mu_eta_pt_scetlib_dyturboCorr_maxFiles_m1_testTApolvar_x0p30_y3p00_V8_oneMCfileEvery2.hdf5 -o /scratch/mciprian/CombineStudies/theoryAgnostic_pol/x0p30_y3p00_V8//oneMCfileEvery2_splitOOA/allProcs//traditionalFit_allSysts_abseta/ --absolutePathInCard  -x '.*recoil'  --absval 1
# /usr/bin/time -v python scripts/combine/setupRabbit.py -i /scratch/mciprian/CombineStudies/theoryAgnostic_pol/x0p30_y3p00_V8//mw_with_mu_eta_pt_scetlib_dyturboCorr_maxFiles_m1_testTApolvar_x0p30_y3p00_V8_oneMCfileEvery2.hdf5 -o /scratch/mciprian/CombineStudies/theoryAgnostic_pol/x0p30_y3p00_V8//oneMCfileEvery2_splitOOA/allProcs//traditionalFit_allSysts/ --absolutePathInCard  -x '.*recoil'

# Fit traditional (full and 1/2 stat), eta and abseta
# python WRemnants/scripts/combine/fitManager.py -i /scratch/mciprian/CombineStudies/theoryAgnostic_pol/x0p30_y3p00_V8//fullStat_splitOOA/allProcs/traditionalFit_allSysts_abseta/WMass_eta_pt_charge/ --skip-fit-data --comb
# python WRemnants/scripts/combine/fitManager.py -i /scratch/mciprian/CombineStudies/theoryAgnostic_pol/x0p30_y3p00_V8//fullStat_splitOOA/allProcs/traditionalFit_allSysts/WMass_eta_pt_charge/ --skip-fit-data --comb
