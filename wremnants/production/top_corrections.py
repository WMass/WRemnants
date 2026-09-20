from wums import logging

logger = logging.child_logger(__name__)

# The corrections are derived for ttbar and must not be applied to single top or to
# ttX, see https://twiki.cern.ch/twiki/bin/view/CMS/TopPtReweighting
ttbar_procs = ["TTLeptonic", "TTSemileptonic"]

# Ratio of the NNLO QCD + NLO EW calculation to POWHEG+Pythia8 for the top quark pt
# spectrum, parametrised by the TOP PAG (JHEP 1710 (2017) 186). It has to be evaluated
# with the parton level top quark, 'isLastCopy' (after radiation and before decay), and
# is frozen above 500 GeV, beyond the range in which it was derived.
top_pt_sf = "(0.103*std::exp(-0.0118*std::min<double>({pt}, 500.)) - 0.000134*std::min<double>({pt}, 500.) + 0.973)"


def define_top_pt_weight(df, dataset_name):
    """NNLO-NLO top pt reweighting, unity for the processes it does not apply to"""
    if not any(dataset_name.startswith(p) for p in ttbar_procs):
        return df.DefinePerSample("topPtWeight", "1.0")

    logger.debug(f"Define the NNLO-NLO top pt reweighting for {dataset_name}")
    # bit 13 of the status flags is 'isLastCopy'
    df = df.Define(
        "topQuark", "GenPart_pdgId == 6 && (GenPart_statusFlags & (1 << 13))"
    )
    df = df.Define(
        "antiTopQuark", "GenPart_pdgId == -6 && (GenPart_statusFlags & (1 << 13))"
    )
    df = df.Define("topPt", "GenPart_pt[topQuark][0]")
    df = df.Define("antiTopPt", "GenPart_pt[antiTopQuark][0]")
    df = df.Define(
        "topPtWeight",
        f"std::sqrt({top_pt_sf.format(pt='topPt')}*{top_pt_sf.format(pt='antiTopPt')})",
    )
    return df
