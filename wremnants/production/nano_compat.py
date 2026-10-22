"""Compatibility between the versions of the custom W-mass NanoAOD.

The analysis code is written for the NanoV9-based production (TrackFitV722_NanoProdv6).
The NanoV15 production (WMassNanoProduction tag NanoV15_vX.Y, CMSSW_15_0) renames some
columns and stores others with narrower integer types. define_nano_compat maps them back
onto the NanoV9 names and types at the start of the graph, so the rest of the code can stay
unchanged.

The jets of NanoV15 are AK4 PUPPI jets (NanoV9: CHS), and their jet ID and pileup ID flags
are not stored. Jet_jetId is rebuilt with the NanoAODv15 recipe of the JetMET TWiki
(JetID13TeVUL, 2016), Jet_puId from Jet_puIdDisc (133X Winter24 PUPPI training) with the
CMSSW working points of that training, see nano_compat.hpp. The Electron_dEscale/dEsigma
columns have no NanoV15 counterpart and are not provided.
"""

import narf
from wremnants.utilities import samples
from wums import logging

logger = logging.child_logger(__name__)

narf.clingutils.Declare('#include "nano_compat.hpp"')

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

    jet_id_inputs = [
        "Jet_eta",
        "Jet_chHEF",
        "Jet_neHEF",
        "Jet_chEmEF",
        "Jet_neEmEF",
        "Jet_muEF",
        "Jet_chMultiplicity",
        "Jet_neMultiplicity",
    ]
    if "Jet_jetId" not in columns and all(c in columns for c in jet_id_inputs):
        df = df.Define("Jet_jetId", f"wrem::jetIdUL16Puppi({', '.join(jet_id_inputs)})")
    if "Jet_puId" not in columns and "Jet_puIdDisc" in columns:
        df = df.Define(
            "Jet_puId", "wrem::puIdWinter24Puppi(Jet_pt, Jet_eta, Jet_puIdDisc)"
        )

    for name, cpp_type in retyped_columns_v15.items():
        if name not in columns:
            continue
        if cpp_type.startswith("ROOT::VecOps::RVec"):
            expr = f"{cpp_type}({name}.begin(), {name}.end())"
        else:
            expr = f"static_cast<{cpp_type}>({name})"
        df = df.Redefine(name, expr)

    return df
