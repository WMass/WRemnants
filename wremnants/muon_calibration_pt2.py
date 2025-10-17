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


def bkmm_selections(df):
    logger.info("Applying bkmm (additional) selections from BPH-21-006")
    cutflow = {}
    dfs_per_cut = []

    logger.info("Enfore at least 1 bkmm dimuon candidate")
    df = df.Filter("nbkmm > 0")
    cutflow["bkmm dimuon cands > 0"] = df.SumAndCount("weight")
    dfs_per_cut.append(df)

    logger.info("Require at least one opposite-sign dimuon candidate")
    df = df.Define(
        "has_opposite_sign",
        """
        bool result = false;
        for (size_t i = 0; i < bkmm_mm_index.size(); ++i) {
            int idx = bkmm_mm_index[i];
            if (idx >= 0 && idx < mm_mu1_index.size() && idx < mm_mu2_index.size()) {
                int mu1_idx = mm_mu1_index[idx];
                int mu2_idx = mm_mu2_index[idx];
                if (mu1_idx >= 0 && mu1_idx < Muon_charge.size() &&
                    mu2_idx >= 0 && mu2_idx < Muon_charge.size()) {
                    if (Muon_charge[mu1_idx] * Muon_charge[mu2_idx] < 0) {
                        result = true;
                        break;
                    }
                }
            }
        }
        return result;
    """,
    )
    # selection should already be made for bkmm_* variables...
    df = df.Filter("has_opposite_sign")
    cutflow["dimuon cand neutral"] = df.SumAndCount("weight")
    dfs_per_cut.append(df)

    # Muon eta cuts
    logger.info("Require |eta| < 1.4 for both muons")
    df = df.Define(
        "muons_eta_pass",
        """
        bool result = false;
        for (size_t i = 0; i < bkmm_mm_index.size(); ++i) {
            int idx = bkmm_mm_index[i];
            if (idx >= 0 && idx < mm_mu1_index.size() && idx < mm_mu2_index.size()) {
                int mu1_idx = mm_mu1_index[idx];
                int mu2_idx = mm_mu2_index[idx];
                if (mu1_idx >= 0 && mu1_idx < Muon_eta.size() &&
                    mu2_idx >= 0 && mu2_idx < Muon_eta.size()) {
                    if (abs(Muon_eta[mu1_idx]) < 1.4 && abs(Muon_eta[mu2_idx]) < 1.4) {
                        result = true;
                        break;
                    }
                }
            }
        }
        return result;
    """,
    )
    df = df.Filter("muons_eta_pass")
    cutflow["muon |eta| < 1.4"] = df.SumAndCount("weight")
    dfs_per_cut.append(df)

    # Muon pt cuts
    logger.info("Require pT > 4 GeV for both muons")
    df = df.Define(
        "muons_pt_pass",
        """
        bool result = false;
        for (size_t i = 0; i < bkmm_mm_index.size(); ++i) {
            int idx = bkmm_mm_index[i];
            if (idx >= 0 && idx < mm_mu1_index.size() && idx < mm_mu2_index.size()) {
                int mu1_idx = mm_mu1_index[idx];
                int mu2_idx = mm_mu2_index[idx];
                if (mu1_idx >= 0 && mu1_idx < Muon_pt.size() &&
                    mu2_idx >= 0 && mu2_idx < Muon_pt.size()) {
                    if (Muon_pt[mu1_idx] > 4 && Muon_pt[mu2_idx] > 4) {
                        result = true;
                        break;
                    }
                }
            }
        }
        return result;
    """,
    )
    df = df.Filter("muons_pt_pass")
    cutflow["muon pT > 4"] = df.SumAndCount("weight")
    dfs_per_cut.append(df)

    # Muon soft MVA cuts
    logger.info("Require soft MVA > 0.45 for both muons")
    df = df.Define(
        "muons_softmva_pass",
        """
        bool result = false;
        for (size_t i = 0; i < bkmm_mm_index.size(); ++i) {
            int idx = bkmm_mm_index[i];
            if (idx >= 0 && idx < mm_mu1_index.size() && idx < mm_mu2_index.size()) {
                int mu1_idx = mm_mu1_index[idx];
                int mu2_idx = mm_mu2_index[idx];
                if (mu1_idx >= 0 && mu1_idx < Muon_softMva.size() &&
                    mu2_idx >= 0 && mu2_idx < Muon_softMva.size()) {
                    if (Muon_softMva[mu1_idx] > 0.45 && Muon_softMva[mu2_idx] > 0.45) {
                        result = true;
                        break;
                    }
                }
            }
        }
        return result;
    """,
    )
    df = df.Filter("muons_softmva_pass")
    cutflow["muon softMVA > 0.45"] = df.SumAndCount("weight")
    dfs_per_cut.append(df)

    # Dimuon pT cut
    logger.info("Require dimuon pT > 7 GeV")
    df = df.Define(
        "mm_pt_pass",
        """
        bool result = false;
        for (size_t i = 0; i < bkmm_mm_index.size(); ++i) {
            int idx = bkmm_mm_index[i];
            if (idx >= 0 && idx < mm_kin_pt.size()) {
                if (mm_kin_pt[idx] > 7.0) {
                    result = true;
                    break;
                }
            }
        }
        return result;
    """,
    )
    df = df.Filter("mm_pt_pass")
    cutflow["dimuon pT > 7"] = df.SumAndCount("weight")
    dfs_per_cut.append(df)

    # Dimuon alphaBS cut
    logger.info("Require dimuon alphaBS < 0.4")
    df = df.Define(
        "mm_alphabs_pass",
        """
        bool result = false;
        for (size_t i = 0; i < bkmm_mm_index.size(); ++i) {
            int idx = bkmm_mm_index[i];
            if (idx >= 0 && idx < mm_kin_alphaBS.size()) {
                if (mm_kin_alphaBS[idx] < 0.4) {
                    result = true;
                    break;
                }
            }
        }
        return result;
    """,
    )
    df = df.Filter("mm_alphabs_pass")
    cutflow["dimuon alphaBS < 0.4"] = df.SumAndCount("weight")
    dfs_per_cut.append(df)

    # Dimuon vertex probability cut
    logger.info("Require dimuon vtx prob > 0.1")
    df = df.Define(
        "mm_vtxprob_pass",
        """
        bool result = false;
        for (size_t i = 0; i < bkmm_mm_index.size(); ++i) {
            int idx = bkmm_mm_index[i];
            if (idx >= 0 && idx < mm_kin_vtx_prob.size()) {
                if (mm_kin_vtx_prob[idx] > 0.1) {
                    result = true;
                    break;
                }
            }
        }
        return result;
    """,
    )
    df = df.Filter("mm_vtxprob_pass")
    cutflow["dimuon vtx prob > 0.1"] = df.SumAndCount("weight")
    dfs_per_cut.append(df)

    # Dimuon 3D significance cut
    logger.info("Require dimuon 3D significance > 4")
    df = df.Define(
        "mm_sl3d_pass",
        """
        bool result = false;
        for (size_t i = 0; i < bkmm_mm_index.size(); ++i) {
            int idx = bkmm_mm_index[i];
            if (idx >= 0 && idx < mm_kin_sl3d.size()) {
                if (mm_kin_sl3d[idx] > 4) {
                    result = true;
                    break;
                }
            }
        }
        return result;
    """,
    )
    df = df.Filter("mm_sl3d_pass")
    cutflow["dimuon sl3d > 4"] = df.SumAndCount("weight")
    dfs_per_cut.append(df)

    # Bkmm J/psi+MC vertex probability cut
    logger.info("Require bkmm J/psi+MC vtx prob > 0.025")
    df = df.Define(
        "bkmm_vtxprob_pass",
        """
        bool result = false;
        for (size_t i = 0; i < bkmm_jpsimc_vtx_prob.size(); ++i) {
            if (bkmm_jpsimc_vtx_prob[i] > 0.025) {
                result = true;
                break;
            }
        }
        return result;
    """,
    )
    df = df.Filter("bkmm_vtxprob_pass")
    cutflow["bkmm vtx prob > 0.025"] = df.SumAndCount("weight")
    dfs_per_cut.append(df)

    # Bkmm mass window cut
    logger.info("Require |bkmm mass - 5.4| < 0.5 GeV")
    df = df.Define(
        "bkmm_mass_pass",
        """
        bool result = false;
        for (size_t i = 0; i < bkmm_jpsimc_mass.size(); ++i) {
            if (abs(bkmm_jpsimc_mass[i] - 5.4) < 0.5) {
                result = true;
                break;
            }
        }
        return result;
    """,
    )
    df = df.Filter("bkmm_mass_pass")
    cutflow["bkmm mass window"] = df.SumAndCount("weight")
    dfs_per_cut.append(df)

    return df, cutflow, dfs_per_cut


def select_first_bkmm_candidate(df):
    logger.info("Selecting first bkmm candidate")

    logger.info("Enfore at least 1 bkmm dimuon candidate")
    df = df.Filter("nbkmm > 0")

    df = df.Define("bkmm_best_idx", "0")

    df = df.Redefine("bkmm_jpsimc_mass", "bkmm_jpsimc_mass[bkmm_best_idx]")
    df = df.Redefine("bkmm_jpsimc_massErr", "bkmm_jpsimc_massErr[bkmm_best_idx]")

    return df


def bkmm_selections2(df):
    logger.info("Applying bkmm (additional) selections from BPH-21-006")
    cutflow = {}
    dfs_per_cut = []

    logger.info("Enforce at least 1 bkmm dimuon candidate")
    df = df.Filter("nbkmm > 0")
    cutflow["bkmm dimuon cands > 0"] = df.SumAndCount("weight")
    dfs_per_cut.append(df)

    # Initialize mask - all candidates start as passing
    df = df.Define(
        "bkmm_passes",
        """
        std::vector<bool> passes(bkmm_mm_index.size(), true);
        return passes;
    """,
    )

    logger.info("Require at least one opposite-sign dimuon candidate")
    df = df.Redefine(
        "bkmm_passes",
        """
        std::vector<bool> passes = bkmm_passes;
        for (size_t i = 0; i < bkmm_mm_index.size(); ++i) {
            if (!passes[i]) continue;  // Skip already failed candidates
            
            int idx = bkmm_mm_index[i];
            if (idx >= 0 && idx < mm_mu1_index.size() && idx < mm_mu2_index.size()) {
                int mu1_idx = mm_mu1_index[idx];
                int mu2_idx = mm_mu2_index[idx];
                if (mu1_idx >= 0 && mu1_idx < Muon_charge.size() &&
                    mu2_idx >= 0 && mu2_idx < Muon_charge.size()) {
                    if (Muon_charge[mu1_idx] * Muon_charge[mu2_idx] >= 0) {
                        passes[i] = false;
                    }
                } else {
                    passes[i] = false;
                }
            } else {
                passes[i] = false;
            }
        }
        return passes;
    """,
    )
    df = df.Define(
        "has_passing_opposite_sign",
        """
        bool result = false;
        for (bool pass : bkmm_passes) {
            if (pass) { result = true; break; }
        }
        return result;
    """,
    )
    df = df.Filter("has_passing_opposite_sign")
    cutflow["dimuon cand neutral"] = df.SumAndCount("weight")
    dfs_per_cut.append(df)

    # Muon eta cuts
    logger.info("Require |eta| < 1.4 for both muons")
    df = df.Redefine(
        "bkmm_passes",
        """
        std::vector<bool> passes = bkmm_passes;
        for (size_t i = 0; i < bkmm_mm_index.size(); ++i) {
            if (!passes[i]) continue;
            
            int idx = bkmm_mm_index[i];
            if (idx >= 0 && idx < mm_mu1_index.size() && idx < mm_mu2_index.size()) {
                int mu1_idx = mm_mu1_index[idx];
                int mu2_idx = mm_mu2_index[idx];
                if (mu1_idx >= 0 && mu1_idx < Muon_eta.size() &&
                    mu2_idx >= 0 && mu2_idx < Muon_eta.size()) {
                    if (!(abs(Muon_eta[mu1_idx]) < 1.4 && abs(Muon_eta[mu2_idx]) < 1.4)) {
                        passes[i] = false;
                    }
                } else {
                    passes[i] = false;
                }
            } else {
                passes[i] = false;
            }
        }
        return passes;
    """,
    )
    df = df.Define(
        "has_passing_eta",
        """
        bool result = false;
        for (bool pass : bkmm_passes) {
            if (pass) { result = true; break; }
        }
        return result;
    """,
    )
    df = df.Filter("has_passing_eta")
    cutflow["muon |eta| < 1.4"] = df.SumAndCount("weight")
    dfs_per_cut.append(df)

    # Muon pt cuts
    logger.info("Require pT > 4 GeV for both muons")
    df = df.Redefine(
        "bkmm_passes",
        """
        std::vector<bool> passes = bkmm_passes;
        for (size_t i = 0; i < bkmm_mm_index.size(); ++i) {
            if (!passes[i]) continue;
            
            int idx = bkmm_mm_index[i];
            if (idx >= 0 && idx < mm_mu1_index.size() && idx < mm_mu2_index.size()) {
                int mu1_idx = mm_mu1_index[idx];
                int mu2_idx = mm_mu2_index[idx];
                if (mu1_idx >= 0 && mu1_idx < Muon_pt.size() &&
                    mu2_idx >= 0 && mu2_idx < Muon_pt.size()) {
                    if (!(Muon_pt[mu1_idx] > 4 && Muon_pt[mu2_idx] > 4)) {
                        passes[i] = false;
                    }
                } else {
                    passes[i] = false;
                }
            } else {        
                passes[i] = false;
            }
        }
        return passes;
    """,
    )
    df = df.Define(
        "has_passing_pt",
        """
        bool result = false;
        for (bool pass : bkmm_passes) {
            if (pass) { result = true; break; }
        }
        return result;
    """,
    )
    df = df.Filter("has_passing_pt")
    cutflow["muon pT > 4"] = df.SumAndCount("weight")
    dfs_per_cut.append(df)

    # Muon soft MVA cuts
    logger.info("Require soft MVA > 0.45 for both muons")
    df = df.Redefine(
        "bkmm_passes",
        """
        std::vector<bool> passes = bkmm_passes;
        for (size_t i = 0; i < bkmm_mm_index.size(); ++i) {
            if (!passes[i]) continue;
            
            int idx = bkmm_mm_index[i];
            if (idx >= 0 && idx < mm_mu1_index.size() && idx < mm_mu2_index.size()) {
                int mu1_idx = mm_mu1_index[idx];
                int mu2_idx = mm_mu2_index[idx];
                if (mu1_idx >= 0 && mu1_idx < Muon_softMva.size() &&
                    mu2_idx >= 0 && mu2_idx < Muon_softMva.size()) {
                    if (!(Muon_softMva[mu1_idx] > 0.45 && Muon_softMva[mu2_idx] > 0.45)) {
                        passes[i] = false;
                    }
                } else {
                    passes[i] = false;
                }
            } else {
                passes[i] = false;
            }
        }
        return passes;
    """,
    )
    df = df.Define(
        "has_passing_softmva",
        """
        bool result = false;
        for (bool pass : bkmm_passes) {
            if (pass) { result = true; break; }
        }
        return result;
    """,
    )
    df = df.Filter("has_passing_softmva")
    cutflow["muon softMVA > 0.45"] = df.SumAndCount("weight")
    dfs_per_cut.append(df)

    # Dimuon pT cut
    logger.info("Require dimuon pT > 7 GeV")
    df = df.Redefine(
        "bkmm_passes",
        """
        std::vector<bool> passes = bkmm_passes;
        for (size_t i = 0; i < bkmm_mm_index.size(); ++i) {
            if (!passes[i]) continue;
            
            int idx = bkmm_mm_index[i];
            if (idx >= 0 && idx < mm_kin_pt.size()) {
                if (!(mm_kin_pt[idx] > 7.0)) {
                    passes[i] = false;
                }
            } else {
                passes[i] = false;
            }
        }
        return passes;
    """,
    )
    df = df.Define(
        "has_passing_mm_pt",
        """
        bool result = false;
        for (bool pass : bkmm_passes) {
            if (pass) { result = true; break; }
        }
        return result;
    """,
    )
    df = df.Filter("has_passing_mm_pt")
    cutflow["dimuon pT > 7"] = df.SumAndCount("weight")
    dfs_per_cut.append(df)

    # Dimuon alphaBS cut
    logger.info("Require dimuon alphaBS < 0.4")
    df = df.Redefine(
        "bkmm_passes",
        """
        std::vector<bool> passes = bkmm_passes;
        for (size_t i = 0; i < bkmm_mm_index.size(); ++i) {
            if (!passes[i]) continue;
            
            int idx = bkmm_mm_index[i];
            if (idx >= 0 && idx < mm_kin_alphaBS.size()) {
                if (!(mm_kin_alphaBS[idx] < 0.4)) {
                    passes[i] = false;
                }
            } else {
                passes[i] = false;
            }
        }
        return passes;
    """,
    )
    df = df.Define(
        "has_passing_alphabs",
        """
        bool result = false;
        for (bool pass : bkmm_passes) {
            if (pass) { result = true; break; }
        }
        return result;
    """,
    )
    df = df.Filter("has_passing_alphabs")
    cutflow["dimuon alphaBS < 0.4"] = df.SumAndCount("weight")
    dfs_per_cut.append(df)

    # Dimuon vertex probability cut
    logger.info("Require dimuon vtx prob > 0.1")
    df = df.Redefine(
        "bkmm_passes",
        """
        std::vector<bool> passes = bkmm_passes;
        for (size_t i = 0; i < bkmm_mm_index.size(); ++i) {
            if (!passes[i]) continue;
            
            int idx = bkmm_mm_index[i];
            if (idx >= 0 && idx < mm_kin_vtx_prob.size()) {
                if (!(mm_kin_vtx_prob[idx] > 0.1)) {
                    passes[i] = false;
                }
            } else {
                passes[i] = false;
            }
        }
        return passes;
    """,
    )
    df = df.Define(
        "has_passing_vtxprob",
        """
        bool result = false;
        for (bool pass : bkmm_passes) {
            if (pass) { result = true; break; }
        }
        return result;
    """,
    )
    df = df.Filter("has_passing_vtxprob")
    cutflow["dimuon vtx prob > 0.1"] = df.SumAndCount("weight")
    dfs_per_cut.append(df)

    # Dimuon 3D significance cut
    logger.info("Require dimuon 3D significance > 4")
    df = df.Redefine(
        "bkmm_passes",
        """
        std::vector<bool> passes = bkmm_passes;
        for (size_t i = 0; i < bkmm_mm_index.size(); ++i) {
            if (!passes[i]) continue;
            
            int idx = bkmm_mm_index[i];
            if (idx >= 0 && idx < mm_kin_sl3d.size()) {
                if (!(mm_kin_sl3d[idx] > 4)) {
                    passes[i] = false;
                }
            } else {
                passes[i] = false;
            }
        }
        return passes;
    """,
    )
    df = df.Define(
        "has_passing_sl3d",
        """
        bool result = false;
        for (bool pass : bkmm_passes) {
            if (pass) { result = true; break; }
        }
        return result;
    """,
    )
    df = df.Filter("has_passing_sl3d")
    cutflow["dimuon sl3d > 4"] = df.SumAndCount("weight")
    dfs_per_cut.append(df)

    # Bkmm J/psi+MC vertex probability cut
    logger.info("Require bkmm J/psi+MC vtx prob > 0.025")
    df = df.Redefine(
        "bkmm_passes",
        """
        std::vector<bool> passes = bkmm_passes;
        for (size_t i = 0; i < bkmm_jpsimc_vtx_prob.size(); ++i) {
            if (!passes[i]) continue;
            
            if (!(bkmm_jpsimc_vtx_prob[i] > 0.025)) {
                passes[i] = false;
            }
        }
        return passes;
    """,
    )
    df = df.Define(
        "has_passing_bkmm_vtxprob",
        """
        bool result = false;
        for (bool pass : bkmm_passes) {
            if (pass) { result = true; break; }
        }
        return result;
    """,
    )
    df = df.Filter("has_passing_bkmm_vtxprob")
    cutflow["bkmm vtx prob > 0.025"] = df.SumAndCount("weight")
    dfs_per_cut.append(df)

    # Bkmm mass window cut
    logger.info("Require |bkmm mass - 5.4| < 0.5 GeV")
    df = df.Redefine(
        "bkmm_passes",
        """
        std::vector<bool> passes = bkmm_passes;
        for (size_t i = 0; i < bkmm_jpsimc_mass.size(); ++i) {
            if (!passes[i]) continue;
            
            if (!(abs(bkmm_jpsimc_mass[i] - 5.4) < 0.5)) {
                passes[i] = false;
            }
        }
        return passes;
    """,
    )
    df = df.Define(
        "has_passing_mass",
        """
        bool result = false;
        for (bool pass : bkmm_passes) {
            if (pass) { result = true; break; }
        }
        return result;
    """,
    )
    df = df.Filter("has_passing_mass")
    cutflow["bkmm mass window"] = df.SumAndCount("weight")
    dfs_per_cut.append(df)

    return df, cutflow, dfs_per_cut


def select_first_bkmm_candidate2(df):
    logger.info("Selecting first passing bkmm candidate")

    # Select the first candidate that passed all selections
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

    df = df.Redefine("bkmm_jpsimc_mass", "bkmm_jpsimc_mass[bkmm_best_idx]")
    df = df.Redefine("bkmm_jpsimc_massErr", "bkmm_jpsimc_massErr[bkmm_best_idx]")
    df = df.Redefine("bkmm_jpsimc_pt", "bkmm_jpsimc_pt[bkmm_best_idx]")
    df = df.Redefine("bkmm_jpsimc_phi", "bkmm_jpsimc_phi[bkmm_best_idx]")
    df = df.Redefine("bkmm_jpsimc_eta", "bkmm_jpsimc_eta[bkmm_best_idx]")
    df = df.Redefine("bkmm_kaon_pt", "bkmm_kaon_pt[bkmm_best_idx]")
    df = df.Redefine("bkmm_kaon_phi", "bkmm_kaon_phi[bkmm_best_idx]")
    df = df.Redefine("bkmm_kaon_eta", "bkmm_kaon_eta[bkmm_best_idx]")
    df = df.Redefine("bkmm_kaon_charge", "bkmm_kaon_charge[bkmm_best_idx]")
    return df


def inspect_dataframe(df):
    cols = df.GetColumnNames()
    for col in cols:
        if any(x in col for x in ["bkmm", "mm_", "Muon"]):
            col_type = df.GetColumnType(col)
            logger.info(f"Column {col}: type = {col_type}")


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
