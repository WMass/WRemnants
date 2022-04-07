import argparse
import pickle
import gzip

import narf
import wremnants
import hist
import lz4.frame
import logging
import math
from wremnants import theory_tools

parser = argparse.ArgumentParser()
parser.add_argument("--nThreads", type=int, help="number of threads", default=None)
parser.add_argument("--maxFiles", type=int, help="Max number of files (per dataset)", default=-1)
parser.add_argument("--filterProcs", type=str, nargs="*", help="Only run over processes matched by (subset) of name", default=None)
parser.add_argument("--muScaleMag", type=float, default=1e-4, help="Magnitude of dummy muon scale uncertainty")
parser.add_argument("--muScaleBins", type=int, default=1, help="Number of bins for muon scale uncertainty")
args = parser.parse_args()

import ROOT
ROOT.gInterpreter.ProcessLine(".O3")
if not args.nThreads:
    ROOT.ROOT.EnableImplicitMT()
elif args.nThreads != 1:
    ROOT.ROOT.EnableImplicitMT(args.nThreads)

filt = lambda x,filts=args.filterProcs: any([f in x.name for f in filts]) 
datasets = wremnants.datasets2016.getDatasets(maxFiles=args.maxFiles, filt=filt if args.filterProcs else None)

era = "2016PostVFP"

muon_prefiring_helper, muon_prefiring_helper_stat, muon_prefiring_helper_syst = wremnants.make_muon_prefiring_helpers(era = era)

scetlibCorrZ_helper = wremnants.makeScetlibCorrHelper(isW=False)
scetlibCorrW_helper = wremnants.makeScetlibCorrHelper(isW=True)

qcdScaleByHelicity_helper = wremnants.makeQCDScaleByHelicityHelper(is_w_like = True)
axis_ptVgen = qcdScaleByHelicity_helper.hist.axes["ptVgen"]
axis_chargeVgen = qcdScaleByHelicity_helper.hist.axes["chargeVgen"]

wprocs = ["WplusmunuPostVFP", "WminusmunuPostVFP", "WminustaunuPostVFP", "WplustaunuPostVFP"]
zprocs = ["ZmumuPostVFP", "ZtautauPostVFP"]

# standard regular axes
axis_eta = hist.axis.Regular(48, -2.4, 2.4, name = "eta")
axis_pt = hist.axis.Regular(29, 26., 55., name = "pt")

# categorical axes in python bindings always have an overflow bin, so use a regular
# axis for the charge
axis_charge = hist.axis.Regular(2, -2., 2., underflow=False, overflow=False, name = "charge")

nominal_axes = [axis_eta, axis_pt, axis_charge]


# extra axes for dilepton validation plots
axis_mll = hist.axis.Regular(24, 60., 120., name = "mll")
axis_yll = hist.axis.Regular(40, -4.0, 4.0, name = "yll")

axis_ptll = hist.axis.Variable(
    [0, 2, 3, 4, 4.75, 5.5, 6.5, 8, 9, 10, 12, 14, 16, 18, 20, 23, 27, 32, 40, 55, 100], name = "ptll"
)

axis_costhetastarll = hist.axis.Regular(20, -1., 1., name = "costhetastarll")
axis_phistarll = hist.axis.Regular(20, -math.pi, math.pi, circular = True, name = "phistarll")

dilepton_axes = [axis_mll, axis_yll, axis_ptll, axis_costhetastarll, axis_phistarll]

# extra axes which can be used to label tensor_axes

down_up_axis = hist.axis.Regular(2, -2., 2., underflow=False, overflow=False, name = "downUpVar")

muon_efficiency_helper, muon_efficiency_helper_stat, muon_efficiency_helper_syst = wremnants.make_muon_efficiency_helpers(era = era, max_pt = axis_pt.edges[-1], is_w_like = True)

pileup_helper = wremnants.make_pileup_helper(era = era)

calibration_helper, calibration_uncertainty_helper = wremnants.make_muon_calibration_helpers()

#def add_plots_with_systematics

def build_graph(df, dataset):
    print("build graph")
    results = []

    if dataset.is_data:
        df = df.DefinePerSample("weight", "1.0")
    else:
        df = df.Define("weight", "std::copysign(1.0, genWeight)")

    weightsum = df.SumAndCount("weight")

    df = df.Filter("HLT_IsoTkMu24 || HLT_IsoMu24")

    if dataset.is_data:
        #TODO corrections not available for data yet
        df = df.Alias("Muon_correctedPt", "Muon_cvhbsPt")
        df = df.Alias("Muon_correctedEta", "Muon_cvhbsEta")
        df = df.Alias("Muon_correctedPhi", "Muon_cvhbsPhi")
        df = df.Alias("Muon_correctedCharge", "Muon_cvhbsCharge")
    elif dataset.name in wprocs or dataset.name in zprocs:
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
    df = df.Filter("Sum(vetoMuons) == 2")

    df = df.Define("goodMuons", "vetoMuons && Muon_mediumId && Muon_isGlobal && Muon_pfRelIso04_all < 0.15")
    df = df.Filter("Sum(goodMuons) == 2")

    # mu- for even event numbers, mu+ for odd event numbers
    df = df.Define("TrigMuon_charge", "event % 2 == 0 ? -1 : 1")
    df = df.Define("NonTrigMuon_charge", "-TrigMuon_charge")

    df = df.Define("trigMuons", "goodMuons && Muon_correctedCharge == TrigMuon_charge")
    df = df.Define("nonTrigMuons", "goodMuons && Muon_correctedCharge == NonTrigMuon_charge")

    df = df.Filter("Sum(trigMuons) == 1 && Sum(nonTrigMuons) == 1")

    df = df.Define("TrigMuon_pt", "Muon_correctedPt[trigMuons][0]")
    df = df.Define("TrigMuon_eta", "Muon_correctedEta[trigMuons][0]")
    df = df.Define("TrigMuon_phi", "Muon_correctedPhi[trigMuons][0]")

    df = df.Define("NonTrigMuon_pt", "Muon_correctedPt[nonTrigMuons][0]")
    df = df.Define("NonTrigMuon_eta", "Muon_correctedEta[nonTrigMuons][0]")
    df = df.Define("NonTrigMuon_phi", "Muon_correctedPhi[nonTrigMuons][0]")

    df = df.Filter("NonTrigMuon_pt > 26.")

    df = df.Define("vetoElectrons", "Electron_pt > 10 && Electron_cutBased > 0 && abs(Electron_eta) < 2.4 && abs(Electron_dxy) < 0.05 && abs(Electron_dz)< 0.2")

    df = df.Filter("Sum(vetoElectrons) == 0")

    df = df.Define("goodTrigObjs", "wrem::goodMuonTriggerCandidate(TrigObj_id,TrigObj_pt,TrigObj_l1pt,TrigObj_l2pt,TrigObj_filterBits)")
    df = df.Filter("wrem::hasTriggerMatch(TrigMuon_eta,TrigMuon_phi,TrigObj_eta[goodTrigObjs],TrigObj_phi[goodTrigObjs])")
    df = df.Filter("Flag_globalSuperTightHalo2016Filter && Flag_EcalDeadCellTriggerPrimitiveFilter && Flag_goodVertices && Flag_HBHENoiseIsoFilter && Flag_HBHENoiseFilter && Flag_BadPFMuonFilter")


    df = df.Define("TrigMuon_mom4", "ROOT::Math::PtEtaPhiMVector(TrigMuon_pt, TrigMuon_eta, TrigMuon_phi, wrem::muon_mass)")
    df = df.Define("NonTrigMuon_mom4", "ROOT::Math::PtEtaPhiMVector(NonTrigMuon_pt, NonTrigMuon_eta, NonTrigMuon_phi, wrem::muon_mass)")
    df = df.Define("Z_mom4", "ROOT::Math::PxPyPzEVector(TrigMuon_mom4)+ROOT::Math::PxPyPzEVector(NonTrigMuon_mom4)")
    df = df.Define("ptZ", "Z_mom4.pt()")
    df = df.Define("massZ", "Z_mom4.mass()")
    df = df.Define("yZ", "Z_mom4.Rapidity()")
    df = df.Define("absYZ", "std::fabs(yZ)")

    df = df.Define("csSineCosThetaPhiZ", "TrigMuon_charge == -1 ? wrem::csSineCosThetaPhi(TrigMuon_mom4, NonTrigMuon_mom4) : wrem::csSineCosThetaPhi(NonTrigMuon_mom4, TrigMuon_mom4)")

    df = df.Define("cosThetaStarZ", "csSineCosThetaPhiZ.costheta")
    df = df.Define("phiStarZ", "std::atan2(csSineCosThetaPhiZ.sinphi, csSineCosThetaPhiZ.cosphi)")


    isW = dataset.name in wprocs
    isZ = dataset.name in zprocs

    if not dataset.is_data:
        df = df.Define("weight_pu", pileup_helper, ["Pileup_nTrueInt"])
        df = df.Define("weight_fullMuonSF_withTrackingReco", muon_efficiency_helper, ["TrigMuon_pt", "TrigMuon_eta", "TrigMuon_charge", "NonTrigMuon_pt", "NonTrigMuon_eta", "NonTrigMuon_charge"])
        df = df.Define("weight_newMuonPrefiringSF", muon_prefiring_helper, ["Muon_correctedEta", "Muon_correctedPt", "Muon_correctedPhi", "Muon_looseId"])

        if dataset.name in wprocs or dataset.name in zprocs:
            df = wremnants.define_prefsr_vars(df)

        applyScetlibCorr = True
        weight_expr = "weight*weight_pu*weight_fullMuonSF_withTrackingReco*weight_newMuonPrefiringSF"
        if isZ or isW and applyScetlibCorr:
            df = theory_tools.define_scetlib_corr(df, weight_expr, scetlibCorrZ_helper if isZ else scetlibCorrW_helper)
        else:
            df = df.Define("nominal_weight", weight_expr)

    else:
        df = df.DefinePerSample("nominal_weight", "1.0")

    # dilepton plots go here, before mass or transverse mass cuts
    df_dilepton = df
    df_dilepton = df_dilepton.Filter("TrigMuon_pt > 26.")

    dilepton_cols = ["massZ", "yZ", "ptZ", "cosThetaStarZ", "phiStarZ"]
    dilepton = df_dilepton.HistoBoost("dilepton", dilepton_axes, [*dilepton_cols, "nominal_weight"])
    results.append(dilepton)

    if isW or isZ:
        results.extend(theory_tools.make_scetlibCorr_hists(df_dilepton, "dilepton", dilepton_axes, dilepton_cols, 
            scetlibCorrZ_helper if isZ else scetlibCorrW_helper)
        )

    df = df.Filter("massZ >= 60. && massZ < 120.")

    #TODO improve this to include muon mass?
    df = df.Define("transverseMass", "wrem::mt_wlike_nano(TrigMuon_pt, TrigMuon_phi, NonTrigMuon_pt, NonTrigMuon_phi, MET_pt, MET_phi)")

    df = df.Filter("transverseMass >= 40.")

    nominal_cols = ["TrigMuon_eta", "TrigMuon_pt", "TrigMuon_charge"]

    nominal = df.HistoBoost("nominal", nominal_axes, [*nominal_cols, "nominal_weight"])
    results.append(nominal)

    if not dataset.is_data:
        # TODO fix the helpers for w-like
        df = df.Define("effStatTnP_tensor", muon_efficiency_helper_stat, ["TrigMuon_pt", "TrigMuon_eta", "TrigMuon_charge", "NonTrigMuon_pt", "NonTrigMuon_eta", "NonTrigMuon_charge", "nominal_weight"])

        effStatTnP = df.HistoBoost("effStatTnP", nominal_axes, [*nominal_cols, "effStatTnP_tensor"], tensor_axes = muon_efficiency_helper_stat.tensor_axes)
        results.append(effStatTnP)

        df = df.Define("effSystIsoTnP_weight", muon_efficiency_helper_syst, ["TrigMuon_pt", "TrigMuon_eta", "TrigMuon_charge", "NonTrigMuon_pt", "NonTrigMuon_eta", "NonTrigMuon_charge", "nominal_weight"])

        effSystIsoTnP = df.HistoBoost("effSystIsoTnP", nominal_axes, [*nominal_cols, "effSystIsoTnP_weight"], tensor_axes = muon_efficiency_helper_syst.tensor_axes)
        results.append(effSystIsoTnP)

        #FIXME skipping EffTrackingRecoTnP_ since it's not consistently defined yet

        df = df.Define("muonL1PrefireStat_tensor", muon_prefiring_helper_stat, ["Muon_correctedEta", "Muon_correctedPt", "Muon_correctedPhi", "Muon_looseId", "nominal_weight"])

        muonL1PrefireStat = df.HistoBoost("muonL1PrefireStat", nominal_axes, [*nominal_cols, "muonL1PrefireStat_tensor"], tensor_axes = muon_prefiring_helper_stat.tensor_axes)
        results.append(muonL1PrefireStat)

        df = df.Define("muonL1PrefireSyst_tensor", muon_prefiring_helper_syst, ["Muon_correctedEta", "Muon_correctedPt", "Muon_correctedPhi", "Muon_looseId", "nominal_weight"])

        muonL1PrefireSyst = df.HistoBoost("muonL1PrefireSyst", nominal_axes, [*nominal_cols, "muonL1PrefireSyst_tensor"], tensor_axes = [down_up_axis])
        results.append(muonL1PrefireSyst)

        # n.b. this is the W analysis so mass weights shouldn't be propagated
        # on the Z samples (but can still use it for dummy muon scale)
        if isW or isZ:
            results.extend(theory_tools.make_scetlibCorr_hists(df, "nominal", axes=nominal_axes, cols=nominal_cols, 
                helper=scetlibCorrZ_helper if isZ else scetlibCorrW_helper))

            df = theory_tools.define_scale_tensor(df)
            results.append(theory_tools.make_scale_hist(df, [*nominal_axes, axis_ptVgen], [*nominal_cols, "ptVgen"]))

            if isZ:
                df = df.Define("helicityWeight_tensor", qcdScaleByHelicity_helper, ["massVgen", "absYVgen", "ptVgen", "chargeVgen", "csSineCosThetaPhi", "scaleWeights_tensor", "nominal_weight"])
                qcdScaleByHelicityUnc = df.HistoBoost("qcdScaleByHelicity", nominal_axes+[axis_ptVgen, axis_chargeVgen], [*nominal_cols, "ptVgen", "chargeVgen", "helicityWeight_tensor"], tensor_axes=qcdScaleByHelicity_helper.tensor_axes)
                results.append(qcdScaleByHelicityUnc)

            results.extend(theory_tools.define_and_make_pdf_hists(df, nominal_axes, nominal_cols))

            nweights = 21 if isW else 23
            df = df.Define("massWeight_tensor", f"auto res = wrem::vec_to_tensor_t<double, {nweights}>(MEParamWeight); res = nominal_weight*res; return res;")
            df = df.Define("massWeight_tensor_wnom", f"auto res = massWeight_tensor; res = nominal_weight*res; return res;")

            if isZ:
                massWeight = df.HistoBoost("massWeight", nominal_axes, [*nominal_cols, "massWeight_tensor_wnom"])
                results.append(massWeight)

            # Don't think it makes sense to apply the mass weights to scale leptons from tau decays
            if not "tau" in dataset.name:
                df = df.Define("muonScaleDummy", f"wrem::dummyScaleFromMassWeights<{args.muScaleBins}, {nweights}>(nominal_weight, massWeight_tensor, TrigMuon_eta, {args.muScaleMag}, {str(isW).lower()})")
                scale_etabins_axis = hist.axis.Regular(args.muScaleBins, -2.4, 2.4, name="scaleEtaSlice", underflow=False, overflow=False)
                dummyMuonScaleSyst = df.HistoBoost("muonScaleSyst", nominal_axes, [*nominal_cols, "muonScaleDummy"], 
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

fname = "mz_wlike_with_mu_eta_pt.pkl.lz4"

print("writing output")
with lz4.frame.open(fname, "wb") as f:
    pickle.dump(resultdict, f, protocol = pickle.HIGHEST_PROTOCOL)
