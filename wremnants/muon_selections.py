from utilities import common
from wremnants import muon_calibration, theory_tools
from wums import logging

logger = logging.child_logger(__name__)


def getIsoBranchBySpec(vertexAgnostic=True, coneSize="04", charged=False):
    root = "vtxAgnPfRelIso" if vertexAgnostic else "pfRelIso"
    component = "chg" if charged else "all"
    cone = coneSize.replace(".", "")
    return f"Muon_{root}{cone}_{component}"


def getIsoBranch(isoDefinition="iso04vtxAgn"):
    if isoDefinition == "iso04vtxAgn":
        return getIsoBranchBySpec()
    elif isoDefinition == "iso04":
        return getIsoBranchBySpec(vertexAgnostic=False)
    elif isoDefinition == "iso04chg":
        return getIsoBranchBySpec(vertexAgnostic=False, charged=True)
    elif isoDefinition == "iso04chgvtxAgn":
        return getIsoBranchBySpec(charged=True)
    else:
        raise NotImplementedError(
            f"Isolation definition {isoDefinition} not implemented"
        )


def apply_iso_muons(
    df,
    iso_first,
    iso_second,
    isoBranch,
    isoThreshold=0.15,
    name_first="trigMuons",
    name_second="nonTrigMuons",
):
    # iso_first/iso_second are integers with values -1/1 for fail/pass isolation, or 0 if not cut has to be applied
    if iso_first:
        isoCond0 = "<" if iso_first == 1 else ">"
        df = df.Filter(f"{isoBranch}[{name_first}][0] {isoCond0} {isoThreshold}")
    if iso_second:
        isoCond1 = "<" if iso_second == 1 else ">"
        df = df.Filter(f"{isoBranch}[{name_second}][0] {isoCond1} {isoThreshold}")
    return df


def apply_met_filters(df):
    df = df.Filter(
        "Flag_globalSuperTightHalo2016Filter && Flag_EcalDeadCellTriggerPrimitiveFilter && Flag_goodVertices && Flag_HBHENoiseIsoFilter && Flag_HBHENoiseFilter && Flag_BadPFMuonFilter"
    )

    return df


def select_good_secondary_vertices(df, dlenSig=4.0, ntracks=0):
    df = df.Define(f"goodSV", f"SV_dlenSig > {dlenSig} && SV_ntracks >= {ntracks}")
    return df


def select_veto_muons(
    df,
    nMuons=1,
    condition="==",
    ptCut=15.0,
    staPtCut=15.0,
    etaCut=2.4,
    useGlobalOrTrackerVeto=False,
    tightGlobalOrTracker=True,
):

    # n.b. charge = -99 is a placeholder for invalid track refit/corrections (mostly just from tracks below
    # the pt threshold of 8 GeV in the nano production)
    # tightGlobalOrTracker relevant only when useGlobalOrTrackerVeto = True
    df = df.Define(
        "vetoMuonsPre",
        "Muon_looseId && abs(Muon_dxybs) < 0.05 && Muon_correctedCharge != -99",
    )
    df = df.Define(
        "Muon_isGoodGlobal",
        f"Muon_isGlobal && Muon_highPurity && Muon_standalonePt > {staPtCut} && Muon_standaloneNumberOfValidHits > 0 && wrem::vectDeltaR2(Muon_standaloneEta, Muon_standalonePhi, Muon_correctedEta, Muon_correctedPhi) < 0.09",
    )
    if useGlobalOrTrackerVeto:
        if tightGlobalOrTracker:
            df = df.Define(
                "Muon_isGoodTracker",
                "Muon_highPurity && Muon_isTracker && Muon_innerTrackOriginalAlgo != 13 && Muon_innerTrackOriginalAlgo != 14",
            )
            df = df.Define(
                "vetoMuonsPre2",
                "vetoMuonsPre && (Muon_isGoodGlobal || Muon_isGoodTracker)",
            )
        else:
            df = df.Alias("vetoMuonsPre2", "vetoMuonsPre")
    else:
        df = df.Define("vetoMuonsPre2", "vetoMuonsPre && Muon_isGoodGlobal")
    df = df.Define(
        "vetoMuons",
        f"vetoMuonsPre2 && Muon_correctedPt > {ptCut} && abs(Muon_correctedEta) < {etaCut}",
    )
    if nMuons >= 0:
        df = df.Filter(f"Sum(vetoMuons) {condition} {nMuons}")

    return df


def select_good_muons(
    df,
    ptLow,
    ptHigh,
    datasetGroup,
    nMuons=1,
    use_trackerMuons=False,
    use_isolation=False,
    isoBranch="Muon_vtxAgnPfRelIso04_all",
    isoThreshold=0.15,
    condition="==",
    nonPromptFromSV=False,
    nonPromptFromLighMesonDecay=False,
    requirePixelHits=False,
    requireID=True,
):

    # requireID can be set to False to remove ID from the selection (it doesn't override the nonprompt control regions with light mesons decay though)
    if use_trackerMuons:
        df = df.Define(
            "Muon_category",
            "Muon_isTracker && Muon_innerTrackOriginalAlgo != 13 && Muon_innerTrackOriginalAlgo != 14 && Muon_highPurity",
        )
    else:
        df = df.Define("Muon_category", "Muon_isGlobal && Muon_highPurity")

    goodMuonsSelection = f"Muon_correctedPt > {ptLow} && Muon_correctedPt < {ptHigh} && vetoMuons && Muon_category"

    if nonPromptFromSV:
        # medium ID added afterwards
        df = select_good_secondary_vertices(df)
        # match by index
        # FIXME: result is not as expected, somthing might be wrong here (either in nanoAOD or in accessing it) disabled for now
        # df = df.Define("Muon_goodSV", "ROOT::VecOps::Take(goodSV, Muon_svIdx, 0)")
        # goodMuonsSelection += " && Muon_sip3d > 4.0 && Muon_goodSV"

        goodMuonsSelection += " && Muon_sip3d > 4.0 && wrem::hasMatchDR2(Muon_correctedEta,Muon_correctedPhi,SV_eta[goodSV],SV_phi[goodSV], 0.01)"

    if nonPromptFromLighMesonDecay:
        # looseID should be part of veto, but just in case, the global condition should also already exist
        goodMuonsSelection += (
            " && Muon_looseId && Muon_isGlobal && !Muon_mediumId && Muon_trkKink > 20."
        )
    elif requireID:
        goodMuonsSelection += " && Muon_mediumId"

    if use_isolation:
        goodMuonsSelection += f" && {isoBranch} < {isoThreshold}"

    if requirePixelHits:
        goodMuonsSelection += " && Muon_cvhNValidPixelHits > 0"

    df = df.Define("goodMuons", goodMuonsSelection)
    if nMuons >= 0:
        df = df.Filter(f"Sum(goodMuons) {condition} {nMuons}")

    return df


def define_trigger_muons(
    df, name_first="trigMuons", name_second="nonTrigMuons", dilepton=False
):
    if dilepton:
        # by convention define first as negative charge, but actually both leptons could be triggering here
        logger.debug(
            f"Using dilepton trigger selection, the negative (positive) muon collection is named {name_first} ({name_second})"
        )
        df = df.DefinePerSample(f"{name_first}_charge0", "-1")
        df = df.DefinePerSample(f"{name_second}_charge0", "1")
    else:
        # mu- for even event numbers, mu+ for odd event numbers
        logger.debug(
            f"Using w-like trigger selection, the trigger (non trigger) muon collection is named {name_first} ({name_second})"
        )
        df = df.Define(f"{name_first}_charge0", "isEvenEvent ? -1 : 1")
        df = df.Define(f"{name_second}_charge0", "isEvenEvent ? 1 : -1")

    df = df.Define(
        name_first, f"goodMuons && Muon_correctedCharge == {name_first}_charge0"
    )
    df = df.Define(
        name_second, f"goodMuons && Muon_correctedCharge == {name_second}_charge0"
    )

    df = muon_calibration.define_corrected_reco_muon_kinematics(
        df, name_first, ["pt", "eta", "phi"]
    )
    df = muon_calibration.define_corrected_reco_muon_kinematics(
        df, name_second, ["pt", "eta", "phi"]
    )
    return df


def define_muon_uT_variable(
    df,
    isWorZ,
    smooth3dsf=False,
    colNamePrefix="goodMuons",
    addWithTnpMuonVar=False,
):
    # TODO: instead of having "addWithTnpMuonVar" it might be better to call this function twice
    # TODO: possibily specifying the target variable name (_uT0 or _tnpUT0) and the input variables
    if smooth3dsf:
        if isWorZ:
            df = theory_tools.define_prefsr_vars(df)
            df = df.Define(
                f"{colNamePrefix}_uT0",
                f"wrem::zqtproj0_boson({colNamePrefix}_pt0, {colNamePrefix}_phi0, ptVgen, phiVgen)",
            )
            if addWithTnpMuonVar:
                df = df.Define(
                    f"{colNamePrefix}_tnpUT0",
                    f"wrem::zqtproj0_boson(Muon_pt[{colNamePrefix}][0], Muon_phi[{colNamePrefix}][0], ptVgen, phiVgen)",
                )
        else:
            # for background processes (Top and Diboson, since Wtaunu and Ztautau are part of isW or isZ)
            # sum all gen e, mu, tau, or neutrinos to define the boson proxy
            # choose particles with status 1 (stable) and statusFlag & 1 (prompt) or taus with status 2 (decayed)
            # there is no double counting for leptons from tau decays, since they have status 1 but not statusFlag & 1
            if "GenPart_leptonAndPhoton" not in df.GetColumnNames():
                df = df.Define(
                    "GenPart_leptonAndPhoton",
                    "(GenPart_status == 1 || (GenPart_status == 2 && abs(GenPart_pdgId) == 15)) && (GenPart_statusFlags & 1) && (abs(GenPart_pdgId) == 22 || (abs(GenPart_pdgId) >= 11 && abs(GenPart_pdgId) <= 16 ) )",
                )
                df = df.Define(
                    "vecSumLeptonAndPhoton_TV2",
                    f"wrem::transverseVectorSum(GenPart_pt[GenPart_leptonAndPhoton],GenPart_phi[GenPart_leptonAndPhoton])",
                )
            df = df.Define(
                f"{colNamePrefix}_uT0",
                f"wrem::zqtproj0_boson({colNamePrefix}_pt0, {colNamePrefix}_phi0, vecSumLeptonAndPhoton_TV2)",
            )
            if addWithTnpMuonVar:
                df = df.Define(
                    f"{colNamePrefix}_tnpUT0",
                    f"wrem::zqtproj0_boson(Muon_pt[{colNamePrefix}][0], Muon_phi[{colNamePrefix}][0], vecSumLeptonAndPhoton_TV2)",
                )
    else:
        # this is a dummy, the uT axis when present will have a single bin
        df = df.Define(f"{colNamePrefix}_uT0", "0.0f")
        if addWithTnpMuonVar:
            df = df.Define(f"{colNamePrefix}_tnpUT0", "0.0f")

    return df


def select_z_candidate(
    df,
    mass_min=60,
    mass_max=120,
    name_first="trigMuons",
    name_second="nonTrigMuons",
    mass="wrem::muon_mass",
):

    df = df.Filter(f"Sum({name_first}) == 1 && Sum({name_second}) == 1")

    df = df.Define(
        f"{name_first}_mom4",
        f"ROOT::Math::PtEtaPhiMVector({name_first}_pt0, {name_first}_eta0, {name_first}_phi0, {mass})",
    )
    df = df.Define(
        f"{name_second}_mom4",
        f"ROOT::Math::PtEtaPhiMVector({name_second}_pt0, {name_second}_eta0, {name_second}_phi0, {mass})",
    )
    df = df.Define(
        "ll_mom4",
        f"ROOT::Math::PxPyPzEVector({name_first}_mom4)+ROOT::Math::PxPyPzEVector({name_second}_mom4)",
    )
    df = df.Define("mll", "ll_mom4.mass()")

    df = df.Filter(f"mll >= {mass_min} && mll < {mass_max}")

    return df


def apply_triggermatching_muon(
    df, dataset, muon, otherMuon=None, era="2016PostVFP", idx=0
):
    df = df.Define(
        "goodTrigObjs",
        f"wrem::goodMuonTriggerCandidate<wrem::Era::Era_{era}>(TrigObj_id,TrigObj_filterBits)",
    )
    if otherMuon is None:
        df = df.Filter(
            f"wrem::hasTriggerMatch({muon}_eta{idx},{muon}_phi{idx},TrigObj_eta[goodTrigObjs],TrigObj_phi[goodTrigObjs])"
        )
    else:
        # implement OR of trigger matching condition (for dilepton), also create corresponding flags
        df = df.Define(
            f"{muon}_passTrigger{idx}",
            f"wrem::hasTriggerMatch({muon}_eta{idx},{muon}_phi{idx},TrigObj_eta[goodTrigObjs],TrigObj_phi[goodTrigObjs])",
        )
        df = df.Define(
            f"{otherMuon}_passTrigger{idx}",
            f"wrem::hasTriggerMatch({otherMuon}_eta{idx},{otherMuon}_phi{idx},TrigObj_eta[goodTrigObjs],TrigObj_phi[goodTrigObjs])",
        )
        df = df.Filter(f"{muon}_passTrigger{idx} || {otherMuon}_passTrigger{idx}")

    return df


def veto_electrons(df):

    df = df.Define(
        "vetoElectrons",
        "Electron_pt > 10 && Electron_cutBased > 0 && abs(Electron_eta) < 2.4 && abs(Electron_dxy) < 0.05 && abs(Electron_dz)< 0.2",
    )
    df = df.Filter("Sum(vetoElectrons) == 0")

    return df


def select_standalone_muons(
    df, dataset, use_trackerMuons=False, muons="goodMuons", idx=0
):

    nHitsSA = common.muonEfficiency_standaloneNumberOfValidHits

    if use_trackerMuons:
        # try to use standalone variables when possible
        df = df.Define(
            f"{muons}_SApt{idx}",
            f"Muon_isStandalone[{muons}][{idx}] ? Muon_standalonePt[{muons}][{idx}] : {muons}_pt{idx}",
        )
        df = df.Define(
            f"{muons}_SAeta{idx}",
            f"Muon_isStandalone[{muons}][{idx}] ? Muon_standaloneEta[{muons}][{idx}] : {muons}_eta{idx}",
        )
        df = df.Define(
            f"{muons}_SAphi{idx}",
            f"Muon_isStandalone[{muons}][{idx}] ? Muon_standalonePhi[{muons}][{idx}] : {muons}_phi{idx}",
        )
    else:
        df = df.Define(f"{muons}_SApt{idx}", f"Muon_standalonePt[{muons}][{idx}]")
        df = df.Define(f"{muons}_SAeta{idx}", f"Muon_standaloneEta[{muons}][{idx}]")
        df = df.Define(f"{muons}_SAphi{idx}", f"Muon_standalonePhi[{muons}][{idx}]")

    # the next cuts are mainly needed for consistency with the reco efficiency measurement for the case with global muons
    # note, when SA does not exist this cut is still fine because of how we define these variables
    df = df.Filter(
        f"{muons}_SApt{idx} > 15.0 && wrem::deltaR2({muons}_SAeta{idx}, {muons}_SAphi{idx}, {muons}_eta{idx}, {muons}_phi{idx}) < 0.09"
    )
    if nHitsSA > 0 and not use_trackerMuons:
        df = df.Filter(f"Muon_standaloneNumberOfValidHits[{muons}][{idx}] >= {nHitsSA}")

    return df


def hlt_string(era="2016PostVFP"):
    match era:
        case "2016PostVFP":
            hltString = "HLT_IsoTkMu24 || HLT_IsoMu24"
        case "2017":
            hltString = "HLT_IsoMu24"  # to be potentially replaced by HLT_IsoMu27
        case "2018":
            hltString = "HLT_IsoMu24"
        case _:
            hltString = "HLT_IsoMu24"
    return hltString
