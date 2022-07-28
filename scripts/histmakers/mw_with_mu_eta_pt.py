import argparse
import pickle
import gzip
import ROOT

parser = argparse.ArgumentParser()
parser.add_argument("-j", "--nThreads", type=int, help="number of threads", default=None)
initargs,_ = parser.parse_known_args()

ROOT.gInterpreter.ProcessLine(".O3")
if not initargs.nThreads:
    ROOT.ROOT.EnableImplicitMT()
elif init.args.nThreads != 1:
    ROOT.ROOT.EnableImplicitMT(initargs.nThreads)
import narf
import wremnants
from wremnants import theory_tools,syst_tools,scetlib_corrections
import hist
import lz4.frame
import logging
import math
import time

logging.basicConfig(level=logging.INFO)

parser.add_argument("-e", "--era", type=str, choices=["2016PreVFP","2016PostVFP"], help="Data set to process", default="2016PostVFP")
parser.add_argument("--pdfs", type=str, nargs="*", default=["nnpdf31"], choices=theory_tools.pdfMapExtended.keys(), help="PDF sets to produce error hists for (first is central set)")
parser.add_argument("--altPdfOnlyCentral", action='store_true', help="Only store central value for alternate PDF sets")
parser.add_argument("--maxFiles", type=int, help="Max number of files (per dataset)", default=-1)
parser.add_argument("--filterProcs", type=str, nargs="*", help="Only run over processes matched by (subset) of name")
parser.add_argument("--skipHelicity", action='store_true', help="Skip the qcdScaleByHelicity histogram (it can be huge)")
parser.add_argument("--scetlibCorr", choices=["altHist", "noUnc", "full", "altHistNoUnc"], help="Save hist for SCETlib correction with/without uncertainties, with/without modifying central weight")
parser.add_argument("--noMuonCorr", action="store_true", help="Don't use corrected pt-eta-phi-charge")
parser.add_argument("--noScaleFactors", action="store_true", help="Don't use scale factors for efficiency")
parser.add_argument("--muonCorrMag", default=1.e-4, type=float, help="Magnitude of dummy muon momentum calibration uncertainty")
parser.add_argument("--muonCorrEtaBins", default=1, type=int, help="Number of eta bins for dummy muon momentum calibration uncertainty")
parser.add_argument("-p", "--postfix", type=str, help="Postfix for output file name", default=None)
parser.add_argument("--eta", nargs=3, type=float, help="Eta binning as 'nbins min max' (only uniform for now)", default=[48,-2.4,2.4])
parser.add_argument("--pt", nargs=3, type=float, help="Pt binning as 'nbins,min,max' (only uniform for now)", default=[29,26.,55.])
parser.add_argument("--lumiUncertainty", type=float, help="Uncertainty for luminosity in excess to 1 (e.g. 1.012 means 1.2\%)", default=1.012)
args = parser.parse_args()

filt = lambda x,filts=args.filterProcs: any([f in x.name for f in filts]) 
datasets = wremnants.datasets2016.getDatasets(maxFiles=args.maxFiles, filt=filt if args.filterProcs else None)

era = args.era
noMuonCorr = args.noMuonCorr

muon_prefiring_helper, muon_prefiring_helper_stat, muon_prefiring_helper_syst = wremnants.make_muon_prefiring_helpers(era = era)
scetlibCorrZ_helper = scetlib_corrections.make_corr_helper(isW=False)
scetlibCorrW_helper = scetlib_corrections.make_corr_helper(isW=True)
qcdScaleByHelicity_Zhelper = wremnants.makeQCDScaleByHelicityHelper(is_w_like = True)
qcdScaleByHelicity_Whelper = wremnants.makeQCDScaleByHelicityHelper()

wprocs = ["WplusmunuPostVFP", "WminusmunuPostVFP", "WminustaunuPostVFP", "WplustaunuPostVFP"]
# for tests of NanoAOD compression need to add ["WminusmunuPostVFP_LZMA_9", "WminusmunuPostVFP_LZ4_4"]
zprocs = ["ZmumuPostVFP", "ZtautauPostVFP"]

# custom template binning
template_neta = int(args.eta[0])
template_mineta = args.eta[1]
template_maxeta = args.eta[2]
print(f"Eta binning: {template_neta} bins from {template_mineta} to {template_maxeta}")
template_npt = int(args.pt[0])
template_minpt = args.pt[1]
template_maxpt = args.pt[2]
print(f"Pt binning: {template_npt} bins from {template_minpt} to {template_maxpt}")

# standard regular axes
axis_eta = hist.axis.Regular(template_neta, template_mineta, template_maxeta, name = "eta")
axis_pt = hist.axis.Regular(template_npt, template_minpt, template_maxpt, name = "pt")

# categorical axes in python bindings always have an overflow bin, so use a regular
# axis for the charge
axis_charge = hist.axis.Regular(2, -2., 2., underflow=False, overflow=False, name = "charge")

axis_passIso = hist.axis.Boolean(name = "passIso")
axis_passMT = hist.axis.Boolean(name = "passMT")

nominal_axes = [axis_eta, axis_pt, axis_charge, axis_passIso, axis_passMT]

axis_ptVgen = qcdScaleByHelicity_Whelper.hist.axes["ptVgen"]
axis_chargeVgen = qcdScaleByHelicity_Whelper.hist.axes["chargeVgen"]

# extra axes which can be used to label tensor_axes

down_up_axis = hist.axis.Regular(2, -2., 2., underflow=False, overflow=False, name = "downUpVar")

down_nom_up_axis = hist.axis.Regular(3, -1.5, 1.5, underflow=False, overflow=False, name = "downNomUpVar")


muon_efficiency_helper, muon_efficiency_helper_stat, muon_efficiency_helper_syst = wremnants.make_muon_efficiency_helpers(era = era, max_pt = axis_pt.edges[-1])

pileup_helper = wremnants.make_pileup_helper(era = era)

calibration_helper, calibration_uncertainty_helper = wremnants.make_muon_calibration_helpers()

def build_graph(df, dataset):
    print("build graph", dataset.name)
    results = []

    if dataset.is_data:
        df = df.DefinePerSample("weight", "1.0")
    else:
        df = df.Define("weight", "std::copysign(1.0, genWeight)")

    weightsum = df.SumAndCount("weight")

    df = df.Filter("HLT_IsoTkMu24 || HLT_IsoMu24")

    isW = dataset.name in wprocs
    isZ = dataset.name in zprocs
    if noMuonCorr:
        df = df.Alias("Muon_correctedPt", "Muon_pt")
        df = df.Alias("Muon_correctedEta", "Muon_eta")
        df = df.Alias("Muon_correctedPhi", "Muon_phi")
        df = df.Alias("Muon_correctedCharge", "Muon_charge")
    else:
        if dataset.is_data:
            #TODO corrections not available for data yet
            df = df.Alias("Muon_correctedPt", "Muon_cvhbsPt")
            df = df.Alias("Muon_correctedEta", "Muon_cvhbsEta")
            df = df.Alias("Muon_correctedPhi", "Muon_cvhbsPhi")
            df = df.Alias("Muon_correctedCharge", "Muon_cvhbsCharge")
        elif isW or isZ:
            df = wremnants.define_corrected_muons(df, calibration_helper)
        else:
            # no track refit available for background monte carlo samples and this is "good enough"
            df = df.Alias("Muon_correctedPt", "Muon_pt")
            df = df.Alias("Muon_correctedEta", "Muon_eta")
            df = df.Alias("Muon_correctedPhi", "Muon_phi")
            df = df.Alias("Muon_correctedCharge", "Muon_charge")

    # n.b. charge = -99 is a placeholder for invalid track refit/corrections (mostly just from tracks below
    # the pt threshold of 8 GeV in the nano production)
    df = df.Define("vetoMuonsPre", "Muon_looseId && abs(Muon_dxybs) < 0.05 && Muon_correctedCharge != -99")
    df = df.Define("vetoMuons", "vetoMuonsPre && Muon_correctedPt > 10. && abs(Muon_correctedEta) < 2.4")
    df = df.Filter("Sum(vetoMuons) == 1")
    df = df.Define("goodMuons", "vetoMuons && Muon_mediumId && Muon_isGlobal")
    df = df.Filter("Sum(goodMuons) == 1")

    df = df.Define("goodMuons_pt0", "Muon_correctedPt[goodMuons][0]")
    df = df.Define("goodMuons_eta0", "Muon_correctedEta[goodMuons][0]")
    df = df.Define("goodMuons_phi0", "Muon_correctedPhi[goodMuons][0]")
    df = df.Define("goodMuons_charge0", "Muon_correctedCharge[goodMuons][0]")

    df = df.Define("goodMuons_pfRelIso04_all0", "Muon_pfRelIso04_all[goodMuons][0]")

    #TODO improve this to include muon mass?
    df = df.Define("transverseMass", "wrem::mt_2(goodMuons_pt0, goodMuons_phi0, DeepMETResolutionTune_pt, DeepMETResolutionTune_phi)")

    df = df.Define("vetoElectrons", "Electron_pt > 10 && Electron_cutBased > 0 && abs(Electron_eta) < 2.4 && abs(Electron_dxy) < 0.05 && abs(Electron_dz)< 0.2")

    df = df.Filter("Sum(vetoElectrons) == 0")

    df = df.Define("goodCleanJets", "Jet_jetId >= 6 && (Jet_pt > 50 || Jet_puId >= 4) && Jet_pt > 30 && abs(Jet_eta) < 2.4 && wrem::cleanJetsFromLeptons(Jet_eta,Jet_phi,Muon_correctedEta[vetoMuons],Muon_correctedPhi[vetoMuons],Electron_eta[vetoElectrons],Electron_phi[vetoElectrons])")

    df = df.Define("passMT", "transverseMass >= 40.0")
    df = df.Filter("passMT || Sum(goodCleanJets)>=1")
    df = df.Define("passIso", "goodMuons_pfRelIso04_all0 < 0.15")

    df = df.Define("goodTrigObjs", "wrem::goodMuonTriggerCandidate(TrigObj_id,TrigObj_pt,TrigObj_l1pt,TrigObj_l2pt,TrigObj_filterBits)")
    df = df.Filter("wrem::hasTriggerMatch(goodMuons_eta0,goodMuons_phi0,TrigObj_eta[goodTrigObjs],TrigObj_phi[goodTrigObjs])")
    df = df.Filter("Flag_globalSuperTightHalo2016Filter && Flag_EcalDeadCellTriggerPrimitiveFilter && Flag_goodVertices && Flag_HBHENoiseIsoFilter && Flag_HBHENoiseFilter && Flag_BadPFMuonFilter")

    nominal_cols = ["goodMuons_eta0", "goodMuons_pt0", "goodMuons_charge0", "passIso", "passMT"]

    if dataset.is_data:
        nominal = df.HistoBoost("nominal", nominal_axes, nominal_cols)
        results.append(nominal)

    else:
        df = df.Define("weight_pu", pileup_helper, ["Pileup_nTrueInt"])
        df = df.Define("weight_fullMuonSF_withTrackingReco", muon_efficiency_helper, ["goodMuons_pt0", "goodMuons_eta0", "goodMuons_charge0", "passIso"])
        df = df.Define("weight_newMuonPrefiringSF", muon_prefiring_helper, ["Muon_correctedEta", "Muon_correctedPt", "Muon_correctedPhi", "Muon_looseId"])

        weight_expr = "weight*weight_pu*weight_newMuonPrefiringSF"
        if not args.noScaleFactors:
            weight_expr += "*weight_fullMuonSF_withTrackingReco"
        
        if isW or isZ:
            df = df.Define("nominal_pdf_cen", theory_tools.pdf_central_weight(dataset.name, args.pdfs[0]))
            weight_expr = f"{weight_expr}*nominal_pdf_cen"
            df = wremnants.define_prefsr_vars(df)

            if args.scetlibCorr:
                df = theory_tools.define_scetlib_corr(df, weight_expr, scetlibCorrZ_helper if isZ else scetlibCorrW_helper,
                                                      corr_type=args.scetlibCorr)
                results.extend(theory_tools.make_scetlibCorr_hists(df, "nominal", axes=nominal_axes, cols=nominal_cols, 
                                                                   helper=scetlibCorrZ_helper if isZ else scetlibCorrW_helper,
                                                                   corr_type=args.scetlibCorr))
            else:
                df = df.Define("nominal_weight", weight_expr)

            for i, pdf in enumerate(args.pdfs):
                withUnc = i == 0 or not args.altPdfOnlyCentral
                results.extend(theory_tools.define_and_make_pdf_hists(df, nominal_axes, nominal_cols, dataset.name, pdf, withUnc))
        else:
            df = df.Define("nominal_weight", weight_expr)

        nominal = df.HistoBoost("nominal", nominal_axes, [*nominal_cols, "nominal_weight"])
        results.append(nominal)

        df = df.Define("effStatTnP_tensor", muon_efficiency_helper_stat, ["goodMuons_pt0", "goodMuons_eta0", "goodMuons_charge0", "passIso", "nominal_weight"])

        effStatTnP = df.HistoBoost("effStatTnP", nominal_axes, [*nominal_cols, "effStatTnP_tensor"], tensor_axes = muon_efficiency_helper_stat.tensor_axes)
        results.append(effStatTnP)

        df = df.Define("effSystTnP_weight", muon_efficiency_helper_syst, ["goodMuons_pt0", "goodMuons_eta0", "goodMuons_charge0", "passIso", "nominal_weight"])

        effSystTnP = df.HistoBoost("effSystTnP", nominal_axes, [*nominal_cols, "effSystTnP_weight"], tensor_axes = muon_efficiency_helper_syst.tensor_axes)
        results.append(effSystTnP)

        #FIXME skipping EffTrackingRecoTnP_ since it's not consistently defined yet

        df = df.Define("muonL1PrefireStat_tensor", muon_prefiring_helper_stat, ["Muon_correctedEta", "Muon_correctedPt", "Muon_correctedPhi", "Muon_looseId", "nominal_weight"])

        muonL1PrefireStat = df.HistoBoost("muonL1PrefireStat", nominal_axes, [*nominal_cols, "muonL1PrefireStat_tensor"], tensor_axes = muon_prefiring_helper_stat.tensor_axes)
        results.append(muonL1PrefireStat)

        df = df.Define("muonL1PrefireSyst_tensor", muon_prefiring_helper_syst, ["Muon_correctedEta", "Muon_correctedPt", "Muon_correctedPhi", "Muon_looseId", "nominal_weight"])

        muonL1PrefireSyst = df.HistoBoost("muonL1PrefireSyst", nominal_axes, [*nominal_cols, "muonL1PrefireSyst_tensor"], tensor_axes = [down_up_axis])
        results.append(muonL1PrefireSyst)

        # luminosity, done here as shape variation despite being a flat scaling so to facilitate propagating to fakes afterwards
        df = df.Define("luminosityScaling", f"wrem::dummyScaling(nominal_weight, {args.lumiUncertainty})")
        luminosity = df.HistoBoost("luminosity", nominal_axes, [*nominal_cols, "luminosityScaling"], tensor_axes = [down_up_axis])
        results.append(luminosity)
                
        # n.b. this is the W analysis so mass weights shouldn't be propagated
        # on the Z samples (but can still use it for dummy muon scale)
        if isW or isZ:

            df = theory_tools.define_scale_tensor(df)
            results.append(theory_tools.make_scale_hist(df, [*nominal_axes, axis_ptVgen, axis_chargeVgen], [*nominal_cols, "ptVgen", "chargeVgen"]))

            # currently SCETLIB corrections are applicable to W-only, and helicity-split scales are only valid for one of W or Z at a time
            # TODO make this work for both simultaneously as needed
            if not args.skipHelicity:
                helicity_helper = qcdScaleByHelicity_Zhelper if isZ else qcdScaleByHelicity_Whelper
                # TODO: Should have consistent order here with the scetlib correction function
                df = df.Define("helicityWeight_tensor", helicity_helper, ["massVgen", "absYVgen", "ptVgen", "chargeVgen", "csSineCosThetaPhi", "scaleWeights_tensor", "nominal_weight"])
                qcdScaleByHelicityUnc = df.HistoBoost("qcdScaleByHelicity", [*nominal_axes, axis_ptVgen, axis_chargeVgen], [*nominal_cols, "ptVgen", "chargeVgen", "helicityWeight_tensor"], tensor_axes=helicity_helper.tensor_axes)
                results.append(qcdScaleByHelicityUnc)

            masswargs = (nominal_axes, nominal_cols) if isW else (None, None)
            df, masswhist = syst_tools.define_mass_weights(df, isW, *masswargs)
            if masswhist:
                results.append(masswhist)

            # Don't think it makes sense to apply the mass weights to scale leptons from tau decays
            if not "tau" in dataset.name:
                # TODO: Move to syst_tools
                netabins = args.muonCorrEtaBins
                nweights = 21
                mag = args.muonCorrMag
                df = df.Define(f"muonScaleDummy{netabins}Bins", f"wrem::dummyScaleFromMassWeights<{netabins}, {nweights}>(nominal_weight, massWeight_tensor, goodMuons_eta0, {mag}, {str(isW).lower()})")
                scale_etabins_axis = hist.axis.Regular(netabins, -2.4, 2.4, name="scaleEtaSlice", underflow=False, overflow=False)
                dummyMuonScaleSyst = df.HistoBoost("muonScaleSyst", nominal_axes, [*nominal_cols, f"muonScaleDummy{netabins}Bins"],
                                                   tensor_axes=[down_up_axis, scale_etabins_axis])

                results.append(dummyMuonScaleSyst)

            df = df.Define("Muon_cvhbsMomCov", "wrem::splitNestedRVec(Muon_cvhbsMomCov_Vals, Muon_cvhbsMomCov_Counts)")

            df = df.Define("muonScaleSyst_responseWeights_tensor", calibration_uncertainty_helper,
                           ["Muon_correctedPt",
                            "Muon_correctedEta",
                            "Muon_correctedPhi",
                            "Muon_correctedCharge",
                            "Muon_genPartIdx",
                            "Muon_cvhbsMomCov",
                            "vetoMuonsPre",
                            "GenPart_pt",
                            "GenPart_eta",
                            "GenPart_phi",
                            "GenPart_pdgId",
                            "GenPart_statusFlags",
                            "nominal_weight"])

            dummyMuonScaleSyst_responseWeights = df.HistoBoost("muonScaleSyst_responseWeights", nominal_axes, [*nominal_cols, "muonScaleSyst_responseWeights_tensor"], tensor_axes = calibration_uncertainty_helper.tensor_axes)
            results.append(dummyMuonScaleSyst_responseWeights)



    return results, weightsum

resultdict = narf.build_and_run(datasets, build_graph)
print("-"*30)
for key in resultdict.keys():
    print(f"Dataset {key}: unweighted events (before cut) = {resultdict[key]['event_count']}")
    print("-"*30)

fname = "mw_with_mu_eta_pt.pkl.lz4"
if args.postfix:
    fname = fname.replace(".pkl.lz4", f"_{args.postfix}.pkl.lz4")

time0 = time.time()
print("writing output...")
with lz4.frame.open(fname, "wb") as f:
    pickle.dump(resultdict, f, protocol = pickle.HIGHEST_PROTOCOL)
print("Output", time.time()-time0)
