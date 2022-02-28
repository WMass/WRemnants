import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-j", "--nThreads", type=int, help="number of threads", default=None)
parser.add_argument("--maxFiles", type=int, help="Max number of files (per dataset)", default=-1)
parser.add_argument("--filterProcs", type=str, nargs="*", help="Only run over processes matched by (subset) of name", default=None)
args = parser.parse_args()

import ROOT
ROOT.gInterpreter.ProcessLine(".O3")
if not args.nThreads:
    ROOT.ROOT.EnableImplicitMT()
elif args.nThreads != 1:
    ROOT.ROOT.EnableImplicitMT(args.nThreads)

import pickle
import gzip

import narf
import wremnants
import hist
import lz4.frame
import logging

filt = lambda x,filts=args.filterProcs: any([f in x.name for f in filts]) 
datasets = wremnants.datasets2016.getDatasets(maxFiles=args.maxFiles, filt=filt if args.filterProcs else None)

era = "2016PostVFP"

muon_prefiring_helper, muon_prefiring_helper_stat, muon_prefiring_helper_syst = wremnants.make_muon_prefiring_helpers(era = era)
scetlibCorr_helper = wremnants.makeScetlibCorrHelper()
qcdScaleByHelicity_helper = wremnants.makeQCDScaleByHelicityHelper()

wprocs = ["WplusmunuPostVFP", "WminusmunuPostVFP", "WminustaunuPostVFP", "WplustaunuPostVFP"]
zprocs = ["ZmumuPostVFP", "ZtautauPostVFP"]

# standard regular axes
axis_eta = hist.axis.Regular(48, -2.4, 2.4, name = "eta")
axis_pt = hist.axis.Regular(29, 26., 55., name = "pt")
axis_gen_eta = hist.axis.Regular(48, -2.4, 2.4, name = "gen_eta")
axis_gen_pt = hist.axis.Regular(29, 26., 55., name = "gen_pt")

# categorical axes in python bindings always have an overflow bin, so use a regular
# axis for the charge
axis_charge = hist.axis.Regular(2, -2., 2., underflow=False, overflow=False, name = "charge")

axis_passIso = hist.axis.Boolean(name = "passIso")
axis_passMT = hist.axis.Boolean(name = "passMT")

nominal_axes = [axis_eta, axis_pt, axis_charge, axis_passIso, axis_passMT]
gen_axes = [axis_gen_eta, axis_gen_pt]

print(qcdScaleByHelicity_helper.tensor_axes)
axis_ptVgen = qcdScaleByHelicity_helper.hist.axes["ptVgen"]
axis_chargeVgen = qcdScaleByHelicity_helper.hist.axes["chargeVgen"]

# extra axes which can be used to label tensor_axes

down_up_axis = hist.axis.Regular(2, -2., 2., underflow=False, overflow=False, name = "downUpVar")

down_nom_up_axis = hist.axis.Regular(3, -1.5, 1.5, underflow=False, overflow=False, name = "downNomUpVar")


muon_efficiency_helper, muon_efficiency_helper_stat, muon_efficiency_helper_syst = wremnants.make_muon_efficiency_helpers(era = era, max_pt = axis_pt.edges[-1])

pileup_helper = wremnants.make_pileup_helper(era = era)

def build_graph(df, dataset):
    print("build graph")
    results = []

    if dataset.is_data:
        df = df.DefinePerSample("weight", "1.0")
    else:
        df = df.Define("weight", "std::copysign(1.0, genWeight)")

    weightsum = df.SumAndCount("weight")

    df = df.Filter("HLT_IsoTkMu24 || HLT_IsoMu24")

    df = df.Define("vetoMuons", "Muon_pt > 10 && Muon_looseId && abs(Muon_eta) < 2.4 && abs(Muon_dxybs) < 0.05")
    df = df.Filter("Sum(vetoMuons) == 1")
    df = df.Define("goodMuons", "vetoMuons && Muon_mediumId && Muon_isGlobal")
    df = df.Filter("Sum(goodMuons) == 1")

    df = df.Define("goodMuons_pt0", "Muon_pt[goodMuons][0]")
    df = df.Define("goodMuons_eta0", "Muon_eta[goodMuons][0]")
    df = df.Define("goodMuons_phi0", "Muon_phi[goodMuons][0]")
    df = df.Define("goodMuons_charge0", "Muon_charge[goodMuons][0]")

    df = df.Define("goodMuons_pfRelIso04_all0", "Muon_pfRelIso04_all[goodMuons][0]")

    #TODO improve this to include muon mass?
    df = df.Define("transverseMass", "wrem::mt_2(goodMuons_pt0, goodMuons_phi0, MET_pt, MET_phi)")

    df = df.Define("vetoElectrons", "Electron_pt > 10 && Electron_cutBased > 0 && abs(Electron_eta) < 2.4 && abs(Electron_dxy) < 0.05 && abs(Electron_dz)< 0.2")

    df = df.Filter("Sum(vetoElectrons) == 0")

    df = df.Define("goodCleanJets", "Jet_jetId >= 6 && (Jet_pt > 50 || Jet_puId >= 4) && Jet_pt > 30 && abs(Jet_eta) < 2.4 && wrem::cleanJetsFromLeptons(Jet_eta,Jet_phi,Muon_eta[vetoMuons],Muon_phi[vetoMuons],Electron_eta[vetoElectrons],Electron_phi[vetoElectrons])")

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
        df = df.Define("weight_newMuonPrefiringSF", muon_prefiring_helper, ["Muon_eta", "Muon_pt", "Muon_phi", "Muon_looseId"])

        df = df.Define("nominal_weight", "weight*weight_pu*weight_fullMuonSF_withTrackingReco*weight_newMuonPrefiringSF")

        nominal = df.HistoBoost("nominal", nominal_axes, [*nominal_cols, "nominal_weight"])
        results.append(nominal)

        df = df.Define("effStatTnP_tensor", muon_efficiency_helper_stat, ["goodMuons_pt0", "goodMuons_eta0", "goodMuons_charge0", "passIso", "nominal_weight"])

        effStatTnP = df.HistoBoost("effStatTnP", nominal_axes, [*nominal_cols, "effStatTnP_tensor"], tensor_axes = muon_efficiency_helper_stat.tensor_axes)
        results.append(effStatTnP)

        df = df.Define("effSystIsoTnP_weight", muon_efficiency_helper_syst, ["goodMuons_pt0", "goodMuons_eta0", "goodMuons_charge0", "passIso", "nominal_weight"])

        effSystIsoTnP = df.HistoBoost("effSystIsoTnP", nominal_axes, [*nominal_cols, "effSystIsoTnP_weight"], tensor_axes = muon_efficiency_helper_syst.tensor_axes)
        results.append(effSystIsoTnP)

        #FIXME skipping EffTrackingRecoTnP_ since it's not consistently defined yet

        df = df.Define("muonL1PrefireStat_tensor", muon_prefiring_helper_stat, ["Muon_eta", "Muon_pt", "Muon_phi", "Muon_looseId", "nominal_weight"])

        muonL1PrefireStat = df.HistoBoost("muonL1PrefireStat", nominal_axes, [*nominal_cols, "muonL1PrefireStat_tensor"], tensor_axes = muon_prefiring_helper_stat.tensor_axes)
        results.append(muonL1PrefireStat)

        df = df.Define("muonL1PrefireSyst_tensor", muon_prefiring_helper_syst, ["Muon_eta", "Muon_pt", "Muon_phi", "Muon_looseId", "nominal_weight"])

        muonL1PrefireSyst = df.HistoBoost("muonL1PrefireSyst", nominal_axes, [*nominal_cols, "muonL1PrefireSyst_tensor"], tensor_axes = [down_up_axis])
        results.append(muonL1PrefireSyst)

        # n.b. this is the W analysis so mass weights shouldn't be propagated
        # on the Z samples (but can still use it for dummy muon scale)
        if dataset.name in wprocs or dataset.name in zprocs:

            isW = dataset.name in wprocs

            df = wremnants.define_prefsr_vars(df)

            scaleHist = df.HistoBoost("qcdScale", nominal_axes+[axis_ptVgen], [*nominal_cols, "ptVgen", "scaleWeights_tensor"], tensor_axes = wremnants.scale_tensor_axes)
            results.append(scaleHist)

            # currently SCETLIB corrections are applicable to W-only, and helicity-split scales are only valid for one of W or Z at a time
            # TODO make this work for both simultaneously as needed
            if isW:
                df = df.Define("scetlibWeight_tensor", scetlibCorr_helper, ["massVgen", "yVgen", "ptVgen", "nominal_weight"])
                scetlibUnc = df.HistoBoost("scetlibUnc", nominal_axes, [*nominal_cols, "scetlibWeight_tensor"], tensor_axes=scetlibCorr_helper.tensor_axes)
                results.append(scetlibUnc)

                df = df.Define("helicityWeight_tensor", qcdScaleByHelicity_helper, ["massVgen", "absYVgen", "ptVgen", "chargeVgen", "csSineCosThetaPhi", "scaleWeights_tensor", "nominal_weight"])
                qcdScaleByHelicityUnc = df.HistoBoost("qcdScaleByHelicity", nominal_axes+[axis_ptVgen, axis_chargeVgen], [*nominal_cols, "ptVgen", "chargeVgen", "helicityWeight_tensor"], tensor_axes=qcdScaleByHelicity_helper.tensor_axes)
                results.append(qcdScaleByHelicityUnc)

            # slice 101 elements starting from 0 and clip values at += 10.0
            df = df.Define("pdfWeights_tensor", "auto res = wrem::clip_tensor(wrem::vec_to_tensor_t<double, 101>(LHEPdfWeight), 10.); res = nominal_weight*res; return res;")

            pdfNNPDF31 = df.HistoBoost("pdfNNPDF31", nominal_axes, [*nominal_cols, "pdfWeights_tensor"])
            results.append(pdfNNPDF31)

            # slice 2 elements starting from 101
            df = df.Define("pdfWeightsAS_tensor", "auto res = wrem::vec_to_tensor_t<double, 2>(LHEPdfWeight, 101); res = nominal_weight*res; return res;")

            alphaS002NNPDF31 = df.HistoBoost("alphaS002NNPDF31", nominal_axes, [*nominal_cols, "pdfWeightsAS_tensor"])
            results.append(alphaS002NNPDF31)

            # Don't think it makes sense to apply the mass weights to scale leptons from tau decays, and it doesn't have MEParamWeight for now anyway
            if not "tau" in dataset.name:
                nweights = 21 if isW else 23
                df = df.Define("massWeight_tensor", f"wrem::vec_to_tensor_t<double, {nweights}>(MEParamWeight)")
                df = df.Define("massWeight_tensor_wnom", f"auto res = massWeight_tensor; res = nominal_weight*res; return res;")

                if isW:
                    massWeight = df.HistoBoost("massWeight", nominal_axes, [*nominal_cols, "massWeight_tensor_wnom"])
                    results.append(massWeight)

                netabins = 4
                df = df.Define("muonScaleDummy4Bins2e4", f"wrem::dummyScaleFromMassWeights<{netabins}, {nweights}>(nominal_weight, massWeight_tensor, goodMuons_eta0, 2.e-4, {str(isW).lower()})")
                scale_etabins_axis = hist.axis.Regular(4, -2.4, 2.4, name="scaleEtaSlice", underflow=False, overflow=False)
                dummyMuonScaleSyst = df.HistoBoost("muonScaleSyst", nominal_axes, [*nominal_cols, "muonScaleDummy4Bins2e4"], 
                    tensor_axes=[down_up_axis, scale_etabins_axis])
                results.append(dummyMuonScaleSyst)
            if dataset.name == 'WplusmunuPostVFP':
                df = df.Define('ptPrefsrMuon', 'genlanti.pt()')
                df = df.Define('etaPrefsrMuon', 'genlanti.eta()')
                print("gen info created")
            elif dataset.name == 'WminusmunuPostVFP':
                df = df.Define('ptPrefsrMuon', 'genl.pt()')
                df = df.Define('etaPrefsrMuon', 'genl.eta()')
                print("gen info created")
            if dataset.name in ['WplusmunuPostVFP', 'WminusmunuPostVFP']:
                gen_cols = ['etaPrefsrMuon', 'ptPrefsrMuon']
                gen = df.HistoBoost("gen", gen_axes, [*gen_cols, "nominal_weight"])
                results.append(gen)
                print("gen info accessed")
    return results, weightsum

resultdict = narf.build_and_run(datasets, build_graph)

fname = "mw_with_mu_eta_pt.pkl.lz4"

print("writing output")
with lz4.frame.open(fname, "wb") as f:
    pickle.dump(resultdict, f, protocol = pickle.HIGHEST_PROTOCOL)
