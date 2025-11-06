from utilities import common
from wums import logging

logger = logging.child_logger(__name__)
data_dir = common.data_dir


def define_jpsi_triggers(df, trigger_name="", cutflow={}):
    if trigger_name == "":
        logger.error("no trigger name provided, cannot filter DataFrame")
        return df, None
    logger.info(f"HLT selection: {trigger_name}")
    df = df.Filter(f"HLT_{trigger_name}")
    cutflow = df.SumAndCount("weight")
    return df, cutflow


def bkmm_selections(df, dataset_name, selections):
    cutflow = {}
    dfs_per_cut = []

    logger.info("Enforce at least 1 bkmm dimuon candidate")
    df = df.Filter("nbkmm > 0")
    cutflow["bkmm dimuon cands > 0"] = df.SumAndCount("weight")
    dfs_per_cut.append(df)

    if dataset_name == "signalBuToJpsiK":
        logger.debug("Defining signal specific mask")
        df = df.Define("bkmm_passes", "bkmm_gen_pdgId != 0")
    else:
        df = df.Define(
            "bkmm_passes", "ROOT::VecOps::RVec<bool>(bkmm_mm_index.size(), true)"
        )

    for cutflow_name, description, selection_func in selections:
        logger.info(description)
        df = selection_func(df)
        df = _apply_filter(df, cutflow_name)
        cutflow[cutflow_name] = df.SumAndCount("weight")
        dfs_per_cut.append(df)

    return df, cutflow, dfs_per_cut


def select_only_passing_bkmm_candidates(
    df, signal: bool = False, select_first: bool = False
):
    logger.info("Selecting passing bkmm candidates")

    # Signal - gen match (and only that one!)
    if signal:
        logger.info("Requiring gen-matched candidate for signal")

        # find the first (and only) gen-matched candidate, if passing
        df = df.Define(
            "bkmm_gen_match",
            """
            int gen_idx = -1;
            for (size_t i = 0; i < bkmm_passes.size(); ++i) {
                if (bkmm_passes[i] && bkmm_gen_pdgId[i] != 0) {
                    gen_idx = static_cast<int>(i);
                    break;
                }
            }
            return gen_idx;
            """,
        )

        # filter out events where no gen-matched candidate passed all selections
        df = df.Filter("bkmm_gen_match >= 0")

        # get corresponding dimuon index
        df = df.Define("mm_gen_match", "bkmm_mm_index[bkmm_gen_match]")

        # Extract only gen-matched candidate (vector to scalar)
        all_columns = df.GetColumnNames()
        for col in all_columns:
            col_name = str(col)
            col_type = df.GetColumnType(col_name)
            if "RVec" in col_type or "vector<" in col_type:
                if col_name.startswith("bkmm_"):
                    df = df.Redefine(col_name, f"{col_name}[bkmm_gen_match]")
                if col_name.startswith("mm_"):
                    df = df.Redefine(col_name, f"{col_name}[mm_gen_match]")

        df = df.Redefine("nbkmm", "1")

        return df

    # Select only first passing candidate
    if select_first:
        logger.info("Selecting first passing bkmm candidate")

        # get first candidate that passed all selections (NO CRITERIA)
        df = df.Define(
            "bkmm_best_idx",
            """
            int best_idx = -1;
            for (size_t i = 0; i < bkmm_passes.size(); ++i) {
                if (bkmm_passes[i]) {
                    best_idx = static_cast<int>(i);
                    break;
                }
            }
            return best_idx;
            """,
        )

        # Get corresponding dimuon index
        df = df.Define("mm_best_idx", "bkmm_mm_index[bkmm_best_idx]")

        # vector columns to scalars w first passing cand value
        all_columns = df.GetColumnNames()
        for col in all_columns:
            col_name = str(col)
            col_type = df.GetColumnType(col_name)
            if "RVec" in col_type or "vector<" in col_type:
                if col_name.startswith("bkmm_"):
                    df = df.Redefine(col_name, f"{col_name}[bkmm_best_idx]")
                if col_name.startswith("mm_"):
                    df = df.Redefine(col_name, f"{col_name}[mm_best_idx]")

        df = df.Redefine("nbkmm", "1")

        return df

    # else: keep all passing candidates
    logger.info("Keeping all passing bkmm candidates")

    # Update scalar
    df = df.Redefine(
        "nbkmm",
        """
        int count = 0;
        for (bool pass : bkmm_passes) {
            if (pass) count++;
        }
        return count;
        """,
    )

    # mask for mm candidates corresponding to passing bkmm
    df = df.Define(
        "mm_passes",
        """
        ROOT::VecOps::RVec<bool> mm_mask(mm_kin_pt.size(), false);
        for (size_t i = 0; i < bkmm_passes.size(); ++i) {
            if (bkmm_passes[i]) {
                int mm_idx = bkmm_mm_index[i];
                if (mm_idx >= 0 && mm_idx < mm_mask.size()) {
                    mm_mask[mm_idx] = true;
                }
            }
        }
        return mm_mask;
        """,
    )

    # filter all bkmm_ and mm_ vectors to only passing candidates
    all_columns = df.GetColumnNames()
    for col in all_columns:
        col_name = str(col)
        col_type = df.GetColumnType(col_name)

        if "RVec" in col_type or "vector<" in col_type:
            if col_name.startswith("bkmm_") and col_name != "bkmm_passes":
                df = df.Redefine(col_name, f"{col_name}[bkmm_passes]")
            elif col_name.startswith("mm_") and col_name != "mm_passes":
                df = df.Redefine(col_name, f"{col_name}[mm_passes]")

    return df


def select_opposite_sign_dimuon(df):
    """Require opposite-sign dimuon candidates."""
    condition = """
    ROOT::VecOps::RVec<bool> passes = bkmm_passes;
    for (size_t i = 0; i < bkmm_mm_index.size(); ++i) {
        if (!passes[i]) continue;
        int idx = bkmm_mm_index[i];
        if (idx < 0 || idx >= mm_mu1_index.size() || idx >= mm_mu2_index.size()) {
            passes[i] = false;
            continue;
        }
        int mu1_idx = mm_mu1_index[idx];
        int mu2_idx = mm_mu2_index[idx];
        if (mu1_idx < 0 || mu1_idx >= Muon_charge.size() ||
            mu2_idx < 0 || mu2_idx >= Muon_charge.size() ||
            Muon_charge[mu1_idx] * Muon_charge[mu2_idx] >= 0) {
            passes[i] = false;
        }
    }
    return passes;
    """
    return df.Redefine("bkmm_passes", condition)


def select_muon_eta(df, max_eta):
    """Require |eta| < max_eta for both muons."""
    condition = _generate_abs_muon_pair_condition("Muon_eta", "<", max_eta)
    return df.Redefine("bkmm_passes", condition)


def select_muon_pt(df, min_pt):
    """Require pT > min_pt GeV for both muons."""
    condition = _generate_muon_pair_condition("Muon_pt", ">", min_pt)
    return df.Redefine("bkmm_passes", condition)


def select_muon_softmva(df, min_mva):
    """Require soft MVA > min_mva for both muons."""
    condition = _generate_muon_pair_condition("Muon_softMva", ">", min_mva)
    return df.Redefine("bkmm_passes", condition)


def select_dimuon_pt(df, min_pt):
    """Require dimuon pT > min_pt GeV."""
    condition = _generate_dimuon_condition("mm_kin_pt", ">", min_pt)
    return df.Redefine("bkmm_passes", condition)


def select_dimuon_alphabs(df, max_alphabs):
    """Require dimuon alphaBS < max_alphabs."""
    condition = _generate_dimuon_condition("mm_kin_alphaBS", "<", max_alphabs)
    return df.Redefine("bkmm_passes", condition)


def select_dimuon_vtx_prob(df, min_prob):
    """Require dimuon vertex probability > min_prob."""
    condition = _generate_dimuon_condition("mm_kin_vtx_prob", ">", min_prob)
    return df.Redefine("bkmm_passes", condition)


def select_dimuon_sl3d(df, min_sl3d):
    """Require dimuon 3D significance > min_sl3d."""
    condition = _generate_dimuon_condition("mm_kin_sl3d", ">", min_sl3d)
    return df.Redefine("bkmm_passes", condition)


def select_bkmm_vtx_prob(df, min_prob):
    """Require bkmm J/psi+MC vertex probability > min_prob."""
    condition = _generate_bkmm_condition("bkmm_jpsimc_vtx_prob", ">", min_prob)
    return df.Redefine("bkmm_passes", condition)


def select_bkmm_mass_window(df, center, width):
    """Require |bkmm mass - center| < width GeV."""
    condition = _generate_mass_window_condition("bkmm_jpsimc_mass", center, width)
    return df.Redefine("bkmm_passes", condition)


# def select_vtx_constraints(df, cand, constraint, dim):
#    """Require cand (B or dimuon) vtx_{dim}"""
#    condition =
#    return df.Redefine("bkmm_passes", condition)


def select_bkmm_bmm_bdt(df, value):
    """Select greater than value on bkmm bmm bdt variable"""  # NOTE: NEED TO CONFIRM THIS DOESN'T USE KAON AT ALL (variable name suggests it...)
    condition = _generate_bdt_condition("bkmm_bmm", value)
    return df.Redefine("bkmm_passes", condition)


def _generate_muon_pair_condition(variable, operator, threshold):
    """Generate C++ code for conditions on both muons in a dimuon pair."""
    ops = {">": ">", "<": "<", ">=": ">=", "<=": "<="}
    op = ops[operator]

    return f"""
    ROOT::VecOps::RVec<bool> passes = bkmm_passes;
    for (size_t i = 0; i < bkmm_mm_index.size(); ++i) {{
        if (!passes[i]) continue;
        int idx = bkmm_mm_index[i];
        if (idx < 0 || idx >= mm_mu1_index.size() || idx >= mm_mu2_index.size()) {{
            passes[i] = false;
            continue;
        }}
        int mu1_idx = mm_mu1_index[idx];
        int mu2_idx = mm_mu2_index[idx];
        if (mu1_idx < 0 || mu1_idx >= {variable}.size() ||
            mu2_idx < 0 || mu2_idx >= {variable}.size() ||
            !({variable}[mu1_idx] {op} {threshold} && {variable}[mu2_idx] {op} {threshold})) {{
            passes[i] = false;
        }}
    }}
    return passes;
    """


def _generate_dimuon_condition(variable, operator, threshold):
    """Generate C++ code for conditions on dimuon properties."""
    ops = {">": ">", "<": "<", ">=": ">=", "<=": "<="}
    op = ops[operator]

    return f"""
    ROOT::VecOps::RVec<bool> passes = bkmm_passes;
    for (size_t i = 0; i < bkmm_mm_index.size(); ++i) {{
        if (!passes[i]) continue;
        int idx = bkmm_mm_index[i];
        if (idx < 0 || idx >= {variable}.size() || !({variable}[idx] {op} {threshold})) {{
            passes[i] = false;
        }}
    }}
    return passes;
    """


def _generate_bkmm_condition(variable, operator, threshold):
    """Generate C++ code for conditions on bkmm candidate properties."""
    ops = {">": ">", "<": "<", ">=": ">=", "<=": "<="}
    op = ops[operator]

    return f"""
    ROOT::VecOps::RVec<bool> passes = bkmm_passes;
    for (size_t i = 0; i < {variable}.size(); ++i) {{
        if (!passes[i]) continue;
        if (!({variable}[i] {op} {threshold})) {{
            passes[i] = false;
        }}
    }}
    return passes;
    """


def _generate_abs_muon_pair_condition(variable, operator, threshold):
    """Generate C++ code for conditions on abs() of muon properties."""
    ops = {">": ">", "<": "<", ">=": ">=", "<=": "<="}
    op = ops[operator]

    return f"""
    ROOT::VecOps::RVec<bool> passes = bkmm_passes;
    for (size_t i = 0; i < bkmm_mm_index.size(); ++i) {{
        if (!passes[i]) continue;
        int idx = bkmm_mm_index[i];
        if (idx < 0 || idx >= mm_mu1_index.size() || idx >= mm_mu2_index.size()) {{
            passes[i] = false;
            continue;
        }}
        int mu1_idx = mm_mu1_index[idx];
        int mu2_idx = mm_mu2_index[idx];
        if (mu1_idx < 0 || mu1_idx >= {variable}.size() ||
            mu2_idx < 0 || mu2_idx >= {variable}.size() ||
            !(abs({variable}[mu1_idx]) {op} {threshold} && abs({variable}[mu2_idx]) {op} {threshold})) {{
            passes[i] = false;
        }}
    }}
    return passes;
    """


def _generate_mass_window_condition(variable, center, width):
    """Generate C++ code for mass window cut."""
    return f"""
    ROOT::VecOps::RVec<bool> passes = bkmm_passes;
    for (size_t i = 0; i < {variable}.size(); ++i) {{
        if (!passes[i]) continue;
        if (!(abs({variable}[i] - {center}) < {width})) {{
            passes[i] = false;
        }}
    }}
    return passes;
    """


def _generate_bdt_condition(fit, value):
    """Code for bdt cut"""
    variable = f"{fit}_bdt"
    return f"""
    ROOT::VecOps::RVec<bool> passes = bkmm_passes;
    for (size_t i = 0; i < {variable}.size(); ++i) {{
        if (!passes[i]) continue;
        if (!({variable}[i] > {value})) {{
            passes[i] = false;
        }}
    }}
    return passes
    """


def _apply_filter(df, cutflow_name):
    """Apply filter based on bkmm_passes mask."""
    filter_name = f"has_passing_{cutflow_name.replace(' ', '_').replace('|', '').replace('<', '').replace('>', '').replace('.', 'p')}"
    df = df.Define(
        filter_name,
        """
        bool result = false;
        for (bool pass : bkmm_passes) {
            if (pass) { result = true; break; }
        }
        return result;
        """,
    )
    return df.Filter(filter_name)


def analyze_candidate_multiplicity(df):
    logger.info("Analyzing candidate multiplicity")

    df_with_counts = df.Define("n_bkmm_candidates", "bkmm_mm_index.size()")

    total_events = df_with_counts.SumAndCount("weight")[0].GetValue()
    events_with_1 = (
        df_with_counts.Filter("n_bkmm_candidates == 1")
        .SumAndCount("weight")[0]
        .GetValue()
    )
    events_with_2 = (
        df_with_counts.Filter("n_bkmm_candidates == 2")
        .SumAndCount("weight")[0]
        .GetValue()
    )
    events_with_3 = (
        df_with_counts.Filter("n_bkmm_candidates == 3")
        .SumAndCount("weight")[0]
        .GetValue()
    )
    events_with_4plus = (
        df_with_counts.Filter("n_bkmm_candidates >= 4")
        .SumAndCount("weight")[0]
        .GetValue()
    )

    logger.info(f"Candidate multiplicity:")
    logger.info(f"  Total events: {total_events}")
    logger.info(
        f"  Events with 1 candidate: {events_with_1} ({100*events_with_1/total_events:.1f}%)"
    )
    logger.info(
        f"  Events with 2 candidates: {events_with_2} ({100*events_with_2/total_events:.1f}%)"
    )
    logger.info(
        f"  Events with 3 candidates: {events_with_3} ({100*events_with_3/total_events:.1f}%)"
    )
    logger.info(
        f"  Events with 4+ candidates: {events_with_4plus} ({100*events_with_4plus/total_events:.1f}%)"
    )

    return df


def inspect_dataframe(df):
    cols = df.GetColumnNames()
    for col in cols:
        if any(x in col for x in ["bkmm", "mm_", "Muon"]):
            col_type = df.GetColumnType(col)
            logger.info(f"Column {col}: type = {col_type}")
