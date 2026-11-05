"""Compatibility between the versions of the custom W-mass NanoAOD.

The analysis code is written for the NanoV9-based production (TrackFitV722_NanoProdv6).
The NanoV15 production (WMassNanoProduction tag NanoV15_vX.Y, CMSSW_15_0) renames some
columns and stores others with narrower integer types. define_nano_compat maps them back
onto the NanoV9 names and types at the start of the graph, so the rest of the code can stay
unchanged. Jet_jetId and Jet_puId (the NanoV15 jets are AK4 PUPPI jets without these
flags) and the Electron_dEscale/dEsigma columns have no NanoV15 counterpart and are not
provided.
"""

from wremnants.utilities import samples
from wums import logging

logger = logging.child_logger(__name__)

# NanoV9 name: NanoV15 name
renamed_columns_v15 = {
    "MET_pt": "PFMET_pt",
    "MET_phi": "PFMET_phi",
    "MET_sumEt": "PFMET_sumEt",
    "RawMET_pt": "RawPFMET_pt",
    "RawMET_phi": "RawPFMET_phi",
    "RawMET_sumEt": "RawPFMET_sumEt",
    "MET_fiducialGenPt": "FiducialMET_pt",
    "MET_fiducialGenPhi": "FiducialMET_phi",
    "fixedGridRhoFastjetAll": "Rho_fixedGridRhoFastjetAll",
    "fixedGridRhoFastjetCentral": "Rho_fixedGridRhoFastjetCentral",
    "fixedGridRhoFastjetCentralCalo": "Rho_fixedGridRhoFastjetCentralCalo",
    "fixedGridRhoFastjetCentralChargedPileUp": "Rho_fixedGridRhoFastjetCentralChargedPileUp",
    "fixedGridRhoFastjetCentralNeutral": "Rho_fixedGridRhoFastjetCentralNeutral",
}

# columns with a different (narrower or unsigned) type in NanoV15: their NanoV9 type
retyped_columns_v15 = {
    "nMuon": "unsigned int",
    "nElectron": "unsigned int",
    "PV_npvsGood": "int",
    "Muon_genPartIdx": "ROOT::VecOps::RVec<int>",
    "Muon_jetIdx": "ROOT::VecOps::RVec<int>",
    "Muon_svIdx": "ROOT::VecOps::RVec<int>",
    "Muon_nTrackerLayers": "ROOT::VecOps::RVec<int>",
    "Electron_cutBased": "ROOT::VecOps::RVec<int>",
    "GenPart_genPartIdxMother": "ROOT::VecOps::RVec<int>",
    "GenPart_statusFlags": "ROOT::VecOps::RVec<int>",
    "TrigObj_id": "ROOT::VecOps::RVec<int>",
    # the muon bits used in the analysis (IsoMu24 = 16, IsoTkMu24 = 32) are the same in both
    "TrigObj_filterBits": "ROOT::VecOps::RVec<int>",
}


def define_nano_compat(df, nano_version="v9"):
    """Expose the columns of a NanoV15 file under the NanoV9 names and types."""
    if nano_version == "v9":
        return df
    if nano_version != "v15":
        raise ValueError(
            f"Unknown NanoAOD version {nano_version}, supported are {samples.nano_versions}"
        )

    columns = set(df.GetColumnNames())
    for name, name_v15 in renamed_columns_v15.items():
        if name not in columns and name_v15 in columns:
            df = df.Alias(name, name_v15)

    for name, cpp_type in retyped_columns_v15.items():
        if name not in columns:
            continue
        if cpp_type.startswith("ROOT::VecOps::RVec"):
            expr = f"{cpp_type}({name}.begin(), {name}.end())"
        else:
            expr = f"static_cast<{cpp_type}>({name})"
        df = df.Redefine(name, expr)

    return df
