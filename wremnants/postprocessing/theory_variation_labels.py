STANDARD_CORRELATED_NP_UNCERTAINTIES = [
    ["Lambda20.25", "Lambda2-0.25", "chargeVgenNP0scetlibNPZLambda2"],
    ["Lambda4.16", "Lambda4.01", "chargeVgenNP0scetlibNPZLambda4"],
    [
        "Delta_Lambda20.02",
        "Delta_Lambda2-0.02",
        "chargeVgenNP0scetlibNPZDelta_Lambda2",
    ],
]

LATTICE_CORRELATED_NP_UNCERTAINTIES = [
    ["lambda20.5", "lambda20.0", "chargeVgenNP0scetlibNPLambda2"],
    ["lambda40.16", "lambda40.01", "chargeVgenNP0scetlibNPLambda4"],
    [
        "delta_lambda20.105",
        "delta_lambda20.145",
        "chargeVgenNP0scetlibNPDelta_Lambda2",
    ],
]

STANDARD_GAMMA_NP_UNCERTAINTIES = [
    ["omega_nu0.5", "c_nu-0.1-omega_nu0.5", "scetlibNPgamma"],
]

LATTICE_GAMMA_NP_UNCERTAINTIES = [
    [
        "lambda2_nu0.0696-lambda4_nu0.0122-lambda_inf_nu1.1Ext",
        "lambda2_nu0.1044-lambda4_nu0.0026-lambda_inf_nu2.1Ext",
        "scetlibNPgammaEigvar1",
    ],
    [
        "lambda2_nu0.1153-lambda4_nu0.0032-lambda_inf_nu1.6Ext",
        "lambda2_nu0.0587-lambda4_nu0.0116-lambda_inf_nu1.6Ext",
        "scetlibNPgammaEigvar2",
    ],
    [
        "lambda2_nu0.0873-lambda4_nu0.0092",
        "lambda2_nu0.0867-lambda4_nu0.0056",
        "scetlibNPgammaEigvar3",
    ],
]

TNP_UNCERTAINTIES = [
    ["gamma_cusp1.", "gamma_cusp-1."],
    ["gamma_mu_q1.", "gamma_mu_q-1."],
    ["gamma_nu1.", "gamma_nu-1."],
    ["h_qqV1.", "h_qqV-1."],
    ["s1.", "s-1."],
    ["b_qqV0.5", "b_qqV-0.5"],
    ["b_qqbarV0.5", "b_qqbarV-0.5"],
    ["b_qqS0.5", "b_qqS-0.5"],
    ["b_qqDS0.5", "b_qqDS-0.5"],
    ["b_qg0.5", "b_qg-0.5"],
]

TRANSITION_FO_UNCERTAINTIES = [
    [
        "transition_points0.2_0.75_1.0",
        "transition_points0.2_0.35_1.0",
        "resumTransitionZ",
    ],
    [
        "renorm_scale_pt20_envelope_Up",
        "renorm_scale_pt20_envelope_Down",
        "resumFOScaleZ",
    ],
]

BC_QUARK_MASS_VARIATIONS = [
    (
        "scetlib_dyturbo_LatticeNP_MSHT20mbrange_N3p0LL_N2LO_pdfvars",
        "pdfMSHT20mbrange",
        "pdf1",
        "pdf6",
    ),
    (
        "scetlib_dyturbo_LatticeNP_MSHT20mcrange_N3p0LL_N2LO_pdfvars",
        "pdfMSHT20mcrange",
        "pdf1",
        "pdf8",
    ),
]
