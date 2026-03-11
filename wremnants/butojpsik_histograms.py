"""
Too many histograms to leave in the histmaker :)
"""

import math

import hist

all_butojpsik_axes = {
    # random
    "bkmm_kaon_shit_response_weights": hist.axis.Regular(
        1000,
        -1000,
        1000,
        name="bkmm_kaon_response_weights",
        underflow=True,
        overflow=True,
    ),
    # Muon variables
    "Muon_eta": hist.axis.Regular(
        50, -2.7, 2.7, name="Muon_eta", underflow=False, overflow=False
    ),
    "Muon_pt": hist.axis.Regular(
        20, 0.0, 100.0, name="Muon_pt", underflow=False, overflow=False
    ),
    "Muon_phi": hist.axis.Regular(
        50, -math.pi, math.pi, name="Muon_phi", underflow=False, overflow=False
    ),
    # Kaon kinematic variables (no jpsimc)
    "bkmm_kaon_eta": hist.axis.Regular(
        28, -1.4, 1.4, name="bkmm_kaon_eta", underflow=False, overflow=False
    ),
    "bkmm_kaon_pt": hist.axis.Variable(
        [1, 2, 3, 8], name="bkmm_kaon_pt", underflow=False, overflow=False
    ),
    "bkmm_kaon_curvature": hist.axis.Regular(
        5, 0, 1, name="bkmm_kaon_curvature", underflow=False, overflow=False
    ),
    "bkmm_kaon_shit_recoPt": hist.axis.Variable(
        [1, 2, 3, 8], name="bkmm_kaon_pt", underflow=False, overflow=False
    ),
    "bkmm_kaon_shit_genPt": hist.axis.Variable(
        [1, 2, 3, 8], name="bkmm_kaon_pt", underflow=False, overflow=False
    ),
    "bkmm_kaon_phi": hist.axis.Regular(
        50, -math.pi, math.pi, name="bkmm_kaon_phi", underflow=False, overflow=False
    ),
    "bkmm_kaon_charge": hist.axis.Regular(
        2, -2, 2, name="bkmm_kaon_charge", underflow=False, overflow=False
    ),
    # Kaon impact parameter variables
    "bkmm_kaon_dxy_bs": hist.axis.Regular(
        50, -0.5, 0.5, name="bkmm_kaon_dxy_bs", underflow=False, overflow=False
    ),
    "bkmm_kaon_sdxy_bs": hist.axis.Regular(
        50, -10.0, 10.0, name="bkmm_kaon_sdxy_bs", underflow=False, overflow=False
    ),
    "bkmm_kaon_mu1_doca": hist.axis.Regular(
        50, 0.0, 0.1, name="bkmm_kaon_mu1_doca", underflow=False, overflow=False
    ),
    "bkmm_kaon_mu2_doca": hist.axis.Regular(
        50, 0.0, 0.1, name="bkmm_kaon_mu2_doca", underflow=False, overflow=False
    ),
    # B kinematic variables with jpsi mass constraint
    "bkmm_jpsimc_eta": hist.axis.Regular(
        50, -2.7, 2.7, name="bkmm_jpsimc_eta", underflow=False, overflow=False
    ),
    "bkmm_jpsimc_pt": hist.axis.Regular(
        100, 0.0, 70.0, name="bkmm_jpsimc_pt", underflow=False, overflow=False
    ),
    "bkmm_jpsimc_phi": hist.axis.Regular(
        50, -math.pi, math.pi, name="bkmm_jpsimc_phi", underflow=False, overflow=False
    ),
    "bkmm_jpsimc_mass": hist.axis.Regular(
        100, 5.2, 5.4, name="bkmm_jpsimc_mass", underflow=False, overflow=False
    ),
    "bkmm_jpsimc_massErr": hist.axis.Regular(
        30, 0.0, 0.15, name="bkmm_jpsimc_massErr", underflow=False, overflow=False
    ),
    # kaon of bkmm jpsimc candidate kinematic variables
    "bkmm_jpsimc_kaon1eta": hist.axis.Regular(
        28, -1.4, 1.4, name="bkmm_jpsimc_kaon1eta", underflow=False, overflow=False
    ),
    "bkmm_jpsimc_kaon1phi": hist.axis.Regular(
        50,
        -math.pi,
        math.pi,
        name="bkmm_jpsimc_kaon1phi",
        underflow=False,
        overflow=False,
    ),
    "bkmm_jpsimc_kaon1pt": hist.axis.Regular(
        80, 0.0, 8.0, name="bkmm_jpsimc_kaon1pt", underflow=False, overflow=False
    ),
    # "bkmm_jpsimc_kaon1pt": hist.axis.Variable(
    #    [1,2,3,8], name="bkmm_jpsimc_kaon1pt", underflow=False, overflow=False
    # ),
    # bkmm jpimc vtx
    "bkmm_jpsimc_vtx_chi2dof": hist.axis.Regular(
        50, 0.0, 5.0, name="bkmm_jpsimc_vtx_chi2dof", underflow=False, overflow=False
    ),
    "bkmm_jpsimc_vtx_prob": hist.axis.Regular(
        50, 0.0, 1.0, name="bkmm_jpsimc_vtx_prob", underflow=False, overflow=False
    ),
    "bkmm_jpsimc_vtx_x": hist.axis.Regular(
        50, -0.5, 0.5, name="bkmm_jpsimc_vtx_x", underflow=False, overflow=False
    ),
    "bkmm_jpsimc_vtx_y": hist.axis.Regular(
        50, -0.5, 0.5, name="bkmm_jpsimc_vtx_y", underflow=False, overflow=False
    ),
    "bkmm_jpsimc_vtx_z": hist.axis.Regular(
        50, -15.0, 15.0, name="bkmm_jpsimc_vtx_z", underflow=False, overflow=False
    ),
    "bkmm_jpsimc_vtx_xErr": hist.axis.Regular(
        50, 0.0, 0.01, name="bkmm_jpsimc_vtx_xErr", underflow=False, overflow=False
    ),
    "bkmm_jpsimc_vtx_yErr": hist.axis.Regular(
        50, 0.0, 0.01, name="bkmm_jpsimc_vtx_yErr", underflow=False, overflow=False
    ),
    "bkmm_jpsimc_vtx_zErr": hist.axis.Regular(
        50, 0.0, 0.01, name="bkmm_jpsimc_vtx_zErr", underflow=False, overflow=False
    ),
    "bkmm_jpsimc_l3d": hist.axis.Regular(
        50, 0.0, 2.0, name="bkmm_jpsimc_l3d", underflow=False, overflow=False
    ),
    "bkmm_jpsimc_lxy": hist.axis.Regular(
        50, 0.0, 2.0, name="bkmm_jpsimc_lxy", underflow=False, overflow=False
    ),
    "bkmm_jpsimc_sl3d": hist.axis.Regular(
        50, 0.0, 50.0, name="bkmm_jpsimc_sl3d", underflow=False, overflow=False
    ),
    "bkmm_jpsimc_slxy": hist.axis.Regular(
        50, 0.0, 50.0, name="bkmm_jpsimc_slxy", underflow=False, overflow=False
    ),
    # B pointing angle variables
    "bkmm_jpsimc_alpha": hist.axis.Regular(
        50, 0.0, 0.2, name="bkmm_jpsimc_alpha", underflow=False, overflow=False
    ),
    "bkmm_jpsimc_alphaErr": hist.axis.Regular(
        50, 0.0, 0.05, name="bkmm_jpsimc_alphaErr", underflow=False, overflow=False
    ),
    "bkmm_jpsimc_alphaBS": hist.axis.Regular(
        50, 0.0, 0.2, name="bkmm_jpsimc_alphaBS", underflow=False, overflow=False
    ),
    "bkmm_jpsimc_alphaBSErr": hist.axis.Regular(
        50, 0.0, 0.05, name="bkmm_jpsimc_alphaBSErr", underflow=False, overflow=False
    ),
    # J/psi MC impact parameter variables
    "bkmm_jpsimc_pvip": hist.axis.Regular(
        50, 0.0, 0.1, name="bkmm_jpsimc_pvip", underflow=False, overflow=False
    ),
    "bkmm_jpsimc_pvipErr": hist.axis.Regular(
        50, 0.0, 0.01, name="bkmm_jpsimc_pvipErr", underflow=False, overflow=False
    ),
    "bkmm_jpsimc_spvip": hist.axis.Regular(
        50, 0.0, 20.0, name="bkmm_jpsimc_spvip", underflow=False, overflow=False
    ),
    "bkmm_jpsimc_pvlip": hist.axis.Regular(
        50, 0.0, 2.0, name="bkmm_jpsimc_pvlip", underflow=False, overflow=False
    ),
    "bkmm_jpsimc_pvlipErr": hist.axis.Regular(
        50, 0.0, 0.1, name="bkmm_jpsimc_pvlipErr", underflow=False, overflow=False
    ),
    "bkmm_jpsimc_pvlipSig": hist.axis.Regular(
        50, 0.0, 100.0, name="bkmm_jpsimc_pvlipSig", underflow=False, overflow=False
    ),
    # J/psi MC alternative PV variables
    "bkmm_jpsimc_pv2ip": hist.axis.Regular(
        50, 0.0, 0.1, name="bkmm_jpsimc_pv2ip", underflow=False, overflow=False
    ),
    "bkmm_jpsimc_pv2ipErr": hist.axis.Regular(
        50, 0.0, 0.01, name="bkmm_jpsimc_pv2ipErr", underflow=False, overflow=False
    ),
    "bkmm_jpsimc_spv2ip": hist.axis.Regular(
        50, 0.0, 20.0, name="bkmm_jpsimc_spv2ip", underflow=False, overflow=False
    ),
    "bkmm_jpsimc_pv2lip": hist.axis.Regular(
        50, 0.0, 2.0, name="bkmm_jpsimc_pv2lip", underflow=False, overflow=False
    ),
    "bkmm_jpsimc_pv2lipErr": hist.axis.Regular(
        50, 0.0, 0.1, name="bkmm_jpsimc_pv2lipErr", underflow=False, overflow=False
    ),
    "bkmm_jpsimc_pv2lipSig": hist.axis.Regular(
        50, 0.0, 100.0, name="bkmm_jpsimc_pv2lipSig", underflow=False, overflow=False
    ),
    # J/psi MC primary vertex position
    "bkmm_jpsimc_pv_z": hist.axis.Regular(
        50, -15.0, 15.0, name="bkmm_jpsimc_pv_z", underflow=False, overflow=False
    ),
    "bkmm_jpsimc_pv_zErr": hist.axis.Regular(
        50, 0.0, 0.05, name="bkmm_jpsimc_pv_zErr", underflow=False, overflow=False
    ),
    # J/psi MC lifetime variables
    "bkmm_jpsimc_tau": hist.axis.Regular(
        50, 0.0, 0.01, name="bkmm_jpsimc_tau", underflow=False, overflow=False
    ),
    "bkmm_jpsimc_taue": hist.axis.Regular(
        50, 0.0, 0.005, name="bkmm_jpsimc_taue", underflow=False, overflow=False
    ),
    "bkmm_jpsimc_tauxy": hist.axis.Regular(
        50, 0.0, 0.01, name="bkmm_jpsimc_tauxy", underflow=False, overflow=False
    ),
    "bkmm_jpsimc_tauxye": hist.axis.Regular(
        50, 0.0, 0.005, name="bkmm_jpsimc_tauxye", underflow=False, overflow=False
    ),
    # No MC fit kinematic variables
    "bkmm_nomc_eta": hist.axis.Regular(
        50, -2.7, 2.7, name="bkmm_nomc_eta", underflow=False, overflow=False
    ),
    "bkmm_nomc_pt": hist.axis.Regular(
        100, 0.0, 70.0, name="bkmm_nomc_pt", underflow=False, overflow=False
    ),
    "bkmm_nomc_phi": hist.axis.Regular(
        50, -math.pi, math.pi, name="bkmm_nomc_phi", underflow=False, overflow=False
    ),
    "bkmm_nomc_mass": hist.axis.Regular(
        100, 5.2, 5.4, name="bkmm_nomc_mass", underflow=False, overflow=False
    ),
    "bkmm_nomc_massErr": hist.axis.Regular(
        30, 0.0, 0.15, name="bkmm_nomc_massErr", underflow=False, overflow=False
    ),
    # No MC kaon kinematic variables
    "bkmm_nomc_kaon1eta": hist.axis.Regular(
        50, -2.7, 2.7, name="bkmm_nomc_kaon1eta", underflow=False, overflow=False
    ),
    "bkmm_nomc_kaon1phi": hist.axis.Regular(
        50,
        -math.pi,
        math.pi,
        name="bkmm_nomc_kaon1phi",
        underflow=False,
        overflow=False,
    ),
    "bkmm_nomc_kaon1pt": hist.axis.Regular(
        100, 0.0, 20.0, name="bkmm_nomc_kaon1pt", underflow=False, overflow=False
    ),
    # No MC vertex quality variables
    "bkmm_nomc_vtx_chi2dof": hist.axis.Regular(
        50, 0.0, 5.0, name="bkmm_nomc_vtx_chi2dof", underflow=False, overflow=False
    ),
    "bkmm_nomc_vtx_prob": hist.axis.Regular(
        50, 0.0, 1.0, name="bkmm_nomc_vtx_prob", underflow=False, overflow=False
    ),
    # No MC vertex position variables
    "bkmm_nomc_vtx_x": hist.axis.Regular(
        50, -0.5, 0.5, name="bkmm_nomc_vtx_x", underflow=False, overflow=False
    ),
    "bkmm_nomc_vtx_y": hist.axis.Regular(
        50, -0.5, 0.5, name="bkmm_nomc_vtx_y", underflow=False, overflow=False
    ),
    "bkmm_nomc_vtx_z": hist.axis.Regular(
        50, -15.0, 15.0, name="bkmm_nomc_vtx_z", underflow=False, overflow=False
    ),
    "bkmm_nomc_vtx_xErr": hist.axis.Regular(
        50, 0.0, 0.01, name="bkmm_nomc_vtx_xErr", underflow=False, overflow=False
    ),
    "bkmm_nomc_vtx_yErr": hist.axis.Regular(
        50, 0.0, 0.01, name="bkmm_nomc_vtx_yErr", underflow=False, overflow=False
    ),
    "bkmm_nomc_vtx_zErr": hist.axis.Regular(
        50, 0.0, 0.05, name="bkmm_nomc_vtx_zErr", underflow=False, overflow=False
    ),
    # No MC displacement variables
    "bkmm_nomc_l3d": hist.axis.Regular(
        50, 0.0, 2.0, name="bkmm_nomc_l3d", underflow=False, overflow=False
    ),
    "bkmm_nomc_lxy": hist.axis.Regular(
        50, 0.0, 2.0, name="bkmm_nomc_lxy", underflow=False, overflow=False
    ),
    "bkmm_nomc_sl3d": hist.axis.Regular(
        50, 0.0, 100.0, name="bkmm_nomc_sl3d", underflow=False, overflow=False
    ),
    "bkmm_nomc_slxy": hist.axis.Regular(
        50, 0.0, 100.0, name="bkmm_nomc_slxy", underflow=False, overflow=False
    ),
    # No MC pointing angle variables
    "bkmm_nomc_alpha": hist.axis.Regular(
        50, 0.0, 0.2, name="bkmm_nomc_alpha", underflow=False, overflow=False
    ),
    "bkmm_nomc_alphaErr": hist.axis.Regular(
        50, 0.0, 0.05, name="bkmm_nomc_alphaErr", underflow=False, overflow=False
    ),
    "bkmm_nomc_alphaBS": hist.axis.Regular(
        50, 0.0, 0.2, name="bkmm_nomc_alphaBS", underflow=False, overflow=False
    ),
    "bkmm_nomc_alphaBSErr": hist.axis.Regular(
        50, 0.0, 0.05, name="bkmm_nomc_alphaBSErr", underflow=False, overflow=False
    ),
    # No MC impact parameter variables
    "bkmm_nomc_pvip": hist.axis.Regular(
        50, 0.0, 0.1, name="bkmm_nomc_pvip", underflow=False, overflow=False
    ),
    "bkmm_nomc_pvipErr": hist.axis.Regular(
        50, 0.0, 0.01, name="bkmm_nomc_pvipErr", underflow=False, overflow=False
    ),
    "bkmm_nomc_spvip": hist.axis.Regular(
        50, 0.0, 20.0, name="bkmm_nomc_spvip", underflow=False, overflow=False
    ),
    "bkmm_nomc_pvlip": hist.axis.Regular(
        50, 0.0, 2.0, name="bkmm_nomc_pvlip", underflow=False, overflow=False
    ),
    "bkmm_nomc_pvlipErr": hist.axis.Regular(
        50, 0.0, 0.1, name="bkmm_nomc_pvlipErr", underflow=False, overflow=False
    ),
    "bkmm_nomc_pvlipSig": hist.axis.Regular(
        50, 0.0, 100.0, name="bkmm_nomc_pvlipSig", underflow=False, overflow=False
    ),
    # No MC alternative PV variables
    "bkmm_nomc_pv2ip": hist.axis.Regular(
        50, 0.0, 0.1, name="bkmm_nomc_pv2ip", underflow=False, overflow=False
    ),
    "bkmm_nomc_pv2ipErr": hist.axis.Regular(
        50, 0.0, 0.01, name="bkmm_nomc_pv2ipErr", underflow=False, overflow=False
    ),
    "bkmm_nomc_spv2ip": hist.axis.Regular(
        50, 0.0, 20.0, name="bkmm_nomc_spv2ip", underflow=False, overflow=False
    ),
    "bkmm_nomc_pv2lip": hist.axis.Regular(
        50, 0.0, 2.0, name="bkmm_nomc_pv2lip", underflow=False, overflow=False
    ),
    "bkmm_nomc_pv2lipErr": hist.axis.Regular(
        50, 0.0, 0.1, name="bkmm_nomc_pv2lipErr", underflow=False, overflow=False
    ),
    "bkmm_nomc_pv2lipSig": hist.axis.Regular(
        50, 0.0, 100.0, name="bkmm_nomc_pv2lipSig", underflow=False, overflow=False
    ),
    # No MC primary vertex position
    "bkmm_nomc_pv_z": hist.axis.Regular(
        50, -15.0, 15.0, name="bkmm_nomc_pv_z", underflow=False, overflow=False
    ),
    "bkmm_nomc_pv_zErr": hist.axis.Regular(
        50, 0.0, 0.05, name="bkmm_nomc_pv_zErr", underflow=False, overflow=False
    ),
    # No MC lifetime variables
    "bkmm_nomc_tau": hist.axis.Regular(
        50, 0.0, 0.01, name="bkmm_nomc_tau", underflow=False, overflow=False
    ),
    "bkmm_nomc_taue": hist.axis.Regular(
        50, 0.0, 0.005, name="bkmm_nomc_taue", underflow=False, overflow=False
    ),
    "bkmm_nomc_tauxy": hist.axis.Regular(
        50, 0.0, 0.01, name="bkmm_nomc_tauxy", underflow=False, overflow=False
    ),
    "bkmm_nomc_tauxye": hist.axis.Regular(
        50, 0.0, 0.005, name="bkmm_nomc_tauxye", underflow=False, overflow=False
    ),
    # bmm-specific discriminant variables
    "bkmm_bmm_bdt": hist.axis.Regular(
        50, -1.0, 1.0, name="bkmm_bmm_bdt", underflow=False, overflow=False
    ),
    "bkmm_bmm_docatrk": hist.axis.Regular(
        50, 0.0, 0.5, name="bkmm_bmm_docatrk", underflow=False, overflow=False
    ),
    "bkmm_bmm_iso": hist.axis.Regular(
        50, 0.0, 1.0, name="bkmm_bmm_iso", underflow=False, overflow=False
    ),
    "bkmm_bmm_m1iso": hist.axis.Regular(
        50, 0.0, 1.0, name="bkmm_bmm_m1iso", underflow=False, overflow=False
    ),
    "bkmm_bmm_m2iso": hist.axis.Regular(
        50, 0.0, 1.0, name="bkmm_bmm_m2iso", underflow=False, overflow=False
    ),
    "bkmm_bmm_mva": hist.axis.Regular(
        50, -1.0, 1.0, name="bkmm_bmm_mva", underflow=False, overflow=False
    ),
    "bkmm_bmm_otherVtxMaxProb": hist.axis.Regular(
        50, 0.0, 1.0, name="bkmm_bmm_otherVtxMaxProb", underflow=False, overflow=False
    ),
    "bkmm_bmm_otherVtxMaxProb1": hist.axis.Regular(
        50, 0.0, 1.0, name="bkmm_bmm_otherVtxMaxProb1", underflow=False, overflow=False
    ),
    "bkmm_bmm_otherVtxMaxProb2": hist.axis.Regular(
        50, 0.0, 1.0, name="bkmm_bmm_otherVtxMaxProb2", underflow=False, overflow=False
    ),
    # Track multiplicity variables
    "bkmm_bmm_closetrk": hist.axis.Regular(
        20, 0, 20, name="bkmm_bmm_closetrk", underflow=False, overflow=False
    ),
    "bkmm_bmm_closetrks1": hist.axis.Regular(
        20, 0, 20, name="bkmm_bmm_closetrks1", underflow=False, overflow=False
    ),
    "bkmm_bmm_closetrks2": hist.axis.Regular(
        20, 0, 20, name="bkmm_bmm_closetrks2", underflow=False, overflow=False
    ),
    "bkmm_bmm_closetrks3": hist.axis.Regular(
        20, 0, 20, name="bkmm_bmm_closetrks3", underflow=False, overflow=False
    ),
    "bkmm_bmm_nBMTrks": hist.axis.Regular(
        20, 0, 20, name="bkmm_bmm_nBMTrks", underflow=False, overflow=False
    ),
    "bkmm_bmm_nDisTrks": hist.axis.Regular(
        50, 0, 50, name="bkmm_bmm_nDisTrks", underflow=False, overflow=False
    ),
    "bkmm_bmm_nTrks": hist.axis.Regular(
        100, 0, 100, name="bkmm_bmm_nTrks", underflow=False, overflow=False
    ),
    # Index and validity variables
    "bkmm_jpsimc_valid": hist.axis.Regular(
        2, 0, 2, name="bkmm_jpsimc_valid", underflow=False, overflow=False
    ),
    "bkmm_nomc_valid": hist.axis.Regular(
        2, 0, 2, name="bkmm_nomc_valid", underflow=False, overflow=False
    ),
    "bkmm_mm_index": hist.axis.Regular(
        12, -2, 10, name="bkmm_mm_index", underflow=False, overflow=False
    ),
    # number of bkmm candidates
    "nbkmm": hist.axis.Regular(6, 0, 6, name="nbkmm", underflow=False, overflow=False),
    # dimuon variables (hardcoded bkmm candidate filtering, so trying to select jpsis)
    # Angular variables
    "mm_kin_alpha": hist.axis.Regular(
        50, 0.0, 1.0, name="mm_kin_alpha", underflow=False, overflow=False
    ),
    "mm_kin_alphaBS": hist.axis.Regular(
        50, 0.0, 1.0, name="mm_kin_alphaBS", underflow=False, overflow=False
    ),
    "mm_kin_eta": hist.axis.Regular(
        50, -2.7, 2.7, name="mm_kin_eta", underflow=False, overflow=False
    ),
    "mm_kin_mu1eta": hist.axis.Regular(
        50, -2.7, 2.7, name="mm_kin_mu1eta", underflow=False, overflow=False
    ),
    "mm_kin_mu2eta": hist.axis.Regular(
        50, -2.7, 2.7, name="mm_kin_mu2eta", underflow=False, overflow=False
    ),
    "mm_kin_phi": hist.axis.Regular(
        50, -math.pi, math.pi, name="mm_kin_phi", underflow=False, overflow=False
    ),
    "mm_kin_mu1phi": hist.axis.Regular(
        50, -math.pi, math.pi, name="mm_kin_mu1phi", underflow=False, overflow=False
    ),
    "mm_kin_mu2phi": hist.axis.Regular(
        50, -math.pi, math.pi, name="mm_kin_mu2phi", underflow=False, overflow=False
    ),
    # Mass
    "mm_kin_mass": hist.axis.Regular(
        50, 2.8, 3.4, name="mm_kin_mass", underflow=False, overflow=False
    ),
    # Transverse momentum
    "mm_kin_pt": hist.axis.Regular(
        100, 0.0, 50.0, name="mm_kin_pt", underflow=False, overflow=False
    ),
    "mm_kin_mu1pt": hist.axis.Regular(
        100, 0.0, 50.0, name="mm_kin_mu1pt", underflow=False, overflow=False
    ),
    "mm_kin_mu2pt": hist.axis.Regular(
        100, 0.0, 20.0, name="mm_kin_mu2pt", underflow=False, overflow=False
    ),
    # Decay lengths (3D and transverse)
    "mm_kin_l3d": hist.axis.Regular(
        100, 0.0, 1.0, name="mm_kin_l3d", underflow=False, overflow=False
    ),
    "mm_kin_lxy": hist.axis.Regular(
        100, 0.0, 1.0, name="mm_kin_lxy", underflow=False, overflow=False
    ),
    "mm_kin_sl3d": hist.axis.Regular(
        100, 0.0, 100.0, name="mm_kin_sl3d", underflow=False, overflow=False
    ),
    "mm_kin_slxy": hist.axis.Regular(
        100, 0.0, 100.0, name="mm_kin_slxy", underflow=False, overflow=False
    ),
    # Impact parameters
    "mm_kin_pv2ip": hist.axis.Regular(
        50, 0, 0.5, name="mm_kin_pv2ip", underflow=False, overflow=False
    ),
    "mm_kin_pv2lip": hist.axis.Regular(
        100, -5.0, 5.0, name="mm_kin_pv2lip", underflow=False, overflow=False
    ),
    "mm_kin_pvip": hist.axis.Regular(
        50, 0, 0.1, name="mm_kin_pvip", underflow=False, overflow=False
    ),
    "mm_kin_pvlip": hist.axis.Regular(
        40, -0.02, 0.02, name="mm_kin_pvlip", underflow=False, overflow=False
    ),
    # Impact parameter significance
    "mm_kin_pv2lipSig": hist.axis.Regular(
        100, 0.0, 50.0, name="mm_kin_pv2ipSig", underflow=False, overflow=False
    ),
    "mm_kin_pvlipSig": hist.axis.Regular(
        100, 0.0, 100.0, name="mm_kin_pvlipSig", underflow=False, overflow=False
    ),
    "mm_kin_spv2ip": hist.axis.Regular(
        100, 0.0, 50.0, name="mm_kin_spv2ip", underflow=False, overflow=False
    ),
    "mm_kin_spvip": hist.axis.Regular(
        100, 0.0, 50.0, name="mm_kin_spvip", underflow=False, overflow=False
    ),
    "mm_kin_tau": hist.axis.Regular(
        50, -0.1, 0.1, name="mm_kin_tau", underflow=False, overflow=False
    ),
    "mm_kin_tauxy": hist.axis.Regular(
        50, -0.1, 0.1, name="mm_kin_tauxy", underflow=False, overflow=False
    ),
    # Vertex positions
    "mm_kin_pv_z": hist.axis.Regular(
        100, -30.0, 30.0, name="mm_kin_pv_z", underflow=False, overflow=False
    ),
    "mm_kin_vtx_x": hist.axis.Regular(
        100, -0.5, 0.5, name="mm_kin_vtx_x", underflow=False, overflow=False
    ),
    "mm_kin_vtx_y": hist.axis.Regular(
        100, -0.5, 0.5, name="mm_kin_vtx_y", underflow=False, overflow=False
    ),
    "mm_kin_vtx_z": hist.axis.Regular(
        100, -30.0, 30.0, name="mm_kin_vtx_z", underflow=False, overflow=False
    ),
    # Vertex quality
    "mm_kin_vtx_chi2dof": hist.axis.Regular(
        100, 0.0, 10.0, name="mm_kin_vtx_chi2dof", underflow=False, overflow=False
    ),
    "mm_kin_vtx_prob": hist.axis.Regular(
        50, 0.0, 1.0, name="mm_kin_vtx_prob", underflow=False, overflow=False
    ),
    # Errors/uncertainties (logarithmic scale might be better, but using linear here)
    "mm_kin_alphaBSErr": hist.axis.Regular(
        50, 0.0, 0.5, name="mm_kin_alphaBSErr", underflow=False, overflow=False
    ),
    "mm_kin_alphaErr": hist.axis.Regular(
        50, 0.0, 0.5, name="mm_kin_alphaErr", underflow=False, overflow=False
    ),
    "mm_kin_massErr": hist.axis.Regular(
        50, 0.0, 0.1, name="mm_kin_massErr", underflow=False, overflow=False
    ),
    "mm_kin_pv2ipErr": hist.axis.Regular(
        50, 0.0, 0.05, name="mm_kin_pv2ipErr", underflow=False, overflow=False
    ),
    "mm_kin_pv2lipErr": hist.axis.Regular(
        50, 0.0, 0.1, name="mm_kin_pv2lipErr", underflow=False, overflow=False
    ),
    "mm_kin_pv_zErr": hist.axis.Regular(
        50, 0.0, 0.1, name="mm_kin_pv_zErr", underflow=False, overflow=False
    ),
    "mm_kin_pvipErr": hist.axis.Regular(
        20, 0.0, 0.01, name="mm_kin_pvipErr", underflow=False, overflow=False
    ),
    "mm_kin_pvlipErr": hist.axis.Regular(
        50, 0.0, 0.1, name="mm_kin_pvlipErr", underflow=False, overflow=False
    ),
    "mm_kin_taue": hist.axis.Regular(
        50, 0.0, 1.0, name="mm_kin_taue", underflow=False, overflow=False
    ),
    "mm_kin_tauxye": hist.axis.Regular(
        50, 0.0, 1.0, name="mm_kin_tauxye", underflow=False, overflow=False
    ),
    "mm_kin_vtx_xErr": hist.axis.Regular(
        50, 0.0, 0.05, name="mm_kin_vtx_xErr", underflow=False, overflow=False
    ),
    "mm_kin_vtx_yErr": hist.axis.Regular(
        50, 0.0, 0.05, name="mm_kin_vtx_yErr", underflow=False, overflow=False
    ),
    "mm_kin_vtx_zErr": hist.axis.Regular(
        50, 0.0, 0.1, name="mm_kin_vtx_zErr", underflow=False, overflow=False
    ),
    # gen bkmm and mm
    # "bkmm_gen_kaon_pt": hist.axis.Regular(
    #    100, 0.0, 20.0, name="bkmm_gen_kaon_pt", underflow=False, overflow=False
    # ),
    # "bkmm_gen_l3d": hist.axis.Regular(
    #    50, 0.0, 2.0, name="bkmm_jpsimc_l3d", underflow=False, overflow=False
    # ),
    # "bkmm_gen_lxy": hist.axis.Regular(
    #    50, 0.0, 2.0, name="bkmm_jpsimc_lxy", underflow=False, overflow=False
    # ),
    # "bkmm_gen_mass": hist.axis.Regular(
    #    100, 4.8, 6.0, name="bkmm_jpsimc_mass", underflow=False, overflow=False
    # ),
    #    "bkmm_gen_prod_x": hist.axis.Regular(
    #    50, -0.5, 0.5, name="bkmm_gen_prod_x", underflow=False, overflow=False
    # ),
    # "bkmm_gen_prod_y": hist.axis.Regular(
    #    50, -0.5, 0.5, name="bkmm_gen_prod_y", underflow=False, overflow=False
    # ),
    # "bkmm_gen_prod_z": hist.axis.Regular(
    #    50, -15.0, 15.0, name="bkmm_gen_prod_z", underflow=False, overflow=False
    # ),
    # "bkmm_gen_pt": hist.axis.Regular(
    #    100, 0.0, 70.0, name="bkmm_gen_pt", underflow=False, overflow=False
    # ),
    # "bkmm_gen_tau": hist.axis.Regular(
    #    50, 0.0, 0.01, name="bkmm_gen_tau", underflow=False, overflow=False
    # ),
    # "mm_gen_alpha_ip": hist.axis.Regular(
    #    50, 0.0, 1.0, name="mm_gen_alpha_ip", underflow=False, overflow=False
    # ),
    # "mm_gen_alpha_p_phi": hist.axis.Regular(
    #    50, -math.pi, math.pi, name="mm_gen_alpha_p_phi", underflow=False, overflow=False
    # ),
    # "mm_gen_alpha_p_theta": hist.axis.Regular(
    #    50, 0.0, math.pi, name="mm_gen_alpha_p_theta", underflow=False, overflow=False
    # ),
    # "mm_gen_alpha_vtx": hist.axis.Regular(
    #    100, 0.0, 1.0, name="mm_gen_alpha_vtx", underflow=False, overflow=False
    # ),
    # "mm_gen_doca": hist.axis.Regular(
    #    50, 0.0, 0.1, name="mm_gen_doca", underflow=False, overflow=False
    # ),
    # "mm_gen_l3d": hist.axis.Regular(
    #    100, 0.0, 1.0, name="mm_gen_l3d", underflow=False, overflow=False
    # ),
    # "mm_gen_lxy": hist.axis.Regular(
    #    100, 0.0, 1.0, name="mm_gen_lxy", underflow=False, overflow=False
    # ),
    # "mm_gen_mass": hist.axis.Regular(
    #    50, 2.8, 3.4, name="mm_gen_mass", underflow=False, overflow=False
    # ),
    # "mm_gen_mu1_pt": hist.axis.Regular(
    #    100, 0.0, 50.0, name="mm_gen_mu1_pt", underflow=False, overflow=False
    # ),
    # "mm_gen_mu2_pt": hist.axis.Regular(
    #    100, 0.0, 20.0, name="mm_gen_mu2_pt", underflow=False, overflow=False
    # ),
    # "mm_gen_prod_z": hist.axis.Regular(
    #    100, -30.0, 30.0, name="mm_gen_prod_z", underflow=False, overflow=False
    # ),
    # "mm_gen_pt": hist.axis.Regular(
    #    100, 0.0, 50.0, name="mm_gen_pt", underflow=False, overflow=False
    # ),
    # "mm_gen_tau": hist.axis.Regular(
    #    50, -0.1, 0.1, name="mm_gen_tau", underflow=False, overflow=False
    # ),
    # "mm_gen_vtx_x": hist.axis.Regular(
    #    100, -0.5, 0.5, name="mm_gen_vtx_x", underflow=False, overflow=False
    # ),
    # "mm_gen_vtx_y": hist.axis.Regular(
    #    100, -0.5, 0.5, name="mm_gen_vtx_y", underflow=False, overflow=False
    # ),
    # "mm_gen_vtx_z": hist.axis.Regular(
    #    100, -30.0, 30.0, name="mm_gen_vtx_z", underflow=False, overflow=False
    # ),
}
