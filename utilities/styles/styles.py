
process_colors = {
    "Data": "black",
    "Zmumu": "#5790FC",
    "Z": "#5790FC",
    "Zll": "#5790FC",
    "Zee": "#5790FC",
    "Ztautau": "#7A21DD",
    "Wmunu": "#E42536",
    "Wenu": "#E42536",
    "Wtaunu": "#F89C20",
    "DYlowMass": "deepskyblue",
    "PhotonInduced": "gold",
    "Top": "green",
    "Diboson": "#964A8B",
    "Rare": "#964A8B",
    "QCD": "#9C9CA1",
    "Other": "#9C9CA1",
    "Fake": "#9C9CA1",
    "Fake_e": "#9C9CA1",
    "Fake_mu": "#9C9CA1",
}

process_supergroups = {
    "wmass":{
        "Z": ["Ztautau", "Zmumu", "DYlowMass"],
        "Rare": ["PhotonInduced", "Top", "Diboson"],
    },
    "dilepton":{
        "Other": ["Other","PhotonInduced", "Ztautau"],
    },
    "lowpu_w":{
        "Z": ["Ztautau", "Zmumu", "Zee", "DYlowMass"],
        "Rare": ["PhotonInduced", "Top", "Diboson"],
    },
}
process_supergroups["wlike"]=process_supergroups["dilepton"]
process_supergroups["lowpu_z"]=process_supergroups["dilepton"]

process_labels = {
    "Data": "Data",
    "Zmumu": r"Z$\to\mu\mu$",
    "Zee": r"Z$\to ee$",
    "Zll": r"Z$\to\ell\ell$",
    "Z": r"Z",
    "Ztautau": r"Z$\to\tau\tau$",
    "Wmunu":  r"W$^{\pm}\to\mu\nu$",
    "Wenu": r"W$^{\pm}\to e\nu$",
    "Wtaunu": r"W$^{\pm}\to\tau\nu$",
    "DYlowMass": r"Z$\to\mu\mu$, $10<m<50$ GeV",
    "PhotonInduced": r"$\gamma$-induced",
    "Top": "Top",
    "Diboson": "Diboson",
    "QCD": "QCD MC",
    "Other": "Other",
    "Fake": "Nonprompt",
    "Fake_e": "Nonprompt (e)",
    "Fake_mu": r"Nonprompt (\mu)",
}

xlabels = {
    "pt" : r"p$_{T}^{\ell}$ (GeV)",
    "ptGen" : r"p$_{T}^{\ell}$ (GeV)",
    "ptW" : r"p$_{T}^{\ell+p_{\mathrm{T}}^{miss}}$ (GeV)",
    "ptVGen" : r"p$_{T}^\mathrm{V}$ (GeV)",
    "muonJetPt": r"p$_{T}^\mathrm{jet[\ell]}$ (GeV)",
    "eta" : r"$\eta^{\ell}$",
    "etaGen" : r"$\eta^{\ell}$",
    "abseta" : r"$|\eta^{\ell}|$",
    "absEta" : r"$|\eta^{\ell}|$",
    "absEtaGen" : r"$|\eta^{\ell}|$",
    "ptll" : r"p$_{\mathrm{T}}^{\ell\ell}$ (GeV)",
    "yll" : r"y$^{\ell\ell}$",
    "absYVGen" : r"|Y$^\mathrm{V}$|",
    "mll" : r"m$_{\ell\ell}$ (GeV)",
    "ewMll" : r"m$^{\mathrm{EW}}_{\ell\ell}$ (GeV)",
    "costhetastarll" : r"$\cos{\theta^{\star}_{\ell\ell}}$",
    "cosThetaStarll" : r"$\cos{\theta^{\star}_{\ell\ell}}$",
    "phistarll" : r"$\phi^{\star}_{\ell\ell}$",
    "phiStarll" : r"$\phi^{\star}_{\ell\ell}$",
    "MET_pt" : r"p$_{\mathrm{T}}^{miss}$ (GeV)",
    "MET" : r"p$_{\mathrm{T}}^{miss}$ (GeV)",
    "met" : r"p$_{\mathrm{T}}^{miss}$ (GeV)",
    "mt" : r"m$_{T}^{\ell\nu}$ (GeV)",
    "mtfix" : r"m$_{T}^\mathrm{fix}$ (GeV)",
    "etaPlus" : r"$\eta^{\ell(+)}$",
    "etaMinus" : r"$\eta^{\ell(-)}$",
    "ptPlus" : r"p$_{\mathrm{T}}^{\ell(+)}$ (GeV)",
    "ptMinus" : r"p$_{\mathrm{T}}^{\ell(-)}$ (GeV)",
    "etaSum":r"$\eta^{\ell(+)} + \eta^{\ell(-)}$",
    "etaDiff":r"$\eta^{\ell(+)} - \eta^{\ell(-)}$",
    "etaDiff":r"$\eta^{\ell(+)} - \eta^{\ell(-)}$",
    "etaAbsEta": r"$\eta^{\ell[\mathrm{argmax(|\eta^{\ell}|)}]}$",
    "ewMll": "ewMll",
    "ewMlly": "ewMlly",
    "ewLogDeltaM": "ewLogDeltaM",
    "dxy":r"$d_\mathrm{xy}$ (cm)",
    "iso": r"$I$ (GeV)",
    "relIso": "$I_\mathrm{rel}$",
}

# uncertainties
common_groups = [
    "Total",
    "stat",
    "binByBinStat",
    "luminosity",
    "recoil",
    "CMS_background",
    "theory_ew",
    "normXsecW",
    "width"
]
nuisance_groupings = {
    "max": common_groups + [
        "massShift",
        "QCDscale", 
        "pdfCT18Z",
        "resum",
        "muon_eff_syst",
        "muon_eff_stat",
        "prefire",
        "muonCalibration",
        "Fake",
        "bcQuarkMass"
    ],
    "min": common_groups + [
        "massShiftW", "massShiftZ",
        "QCDscalePtChargeMiNNLO", "QCDscaleZPtChargeMiNNLO", "QCDscaleWPtChargeMiNNLO", "QCDscaleZPtHelicityMiNNLO", "QCDscaleWPtHelicityMiNNLO", "QCDscaleZPtChargeHelicityMiNNLO", "QCDscaleWPtChargeHelicityMiNNLO",
        "pdfCT18ZNoAlphaS", "pdfCT18ZAlphaS",
        "resumTNP", "resumNonpert", "resumTransition", "resumScale", "bcQuarkMass",
        "muon_eff_stat_reco", "muon_eff_stat_trigger", "muon_eff_stat_iso", "muon_eff_stat_idip",
        "muon_eff_syst_reco", "muon_eff_syst_trigger", "muon_eff_syst_iso", "muon_eff_syst_idip",
        "muonPrefire", "ecalPrefire",
        "nonClosure", "resolutionCrctn",
        "FakeRate", "FakeShape", "FakeeRate", "FakeeShape", "FakemuRate", "FakemuShape"
    ],
    "unfolding_max": [
        "Total",
        "QCDscale", 
        "pdfCT18Z",
        "resum",
        "theory_ew",
    ],
    "unfolding_min": [
        "Total",
        "QCDscalePtChargeMiNNLO", "QCDscaleZPtChargeMiNNLO", "QCDscaleWPtChargeMiNNLO", "QCDscaleZPtHelicityMiNNLO", "QCDscaleWPtHelicityMiNNLO", "QCDscaleZPtChargeHelicityMiNNLO", "QCDscaleWPtChargeHelicityMiNNLO",
        "QCDscaleZMiNNLO", "QCDscaleWMiNNLO",
        "pdfCT18ZNoAlphaS", "pdfCT18ZAlphaS",
        "resumTNP", "resumNonpert", "resumTransition", "resumScale", "bcQuarkMass",
        "theory_ew",
    ]
}

text_dict = {
    "Zmumu": r"$\mathrm{Z}\rightarrow\mu\mu$",
    "ZToMuMu": r"$\mathrm{Z}\rightarrow\mu\mu$",
    "Wplusmunu": r"$\mathrm{W}^+\rightarrow\mu\nu$",
    "Wminusmunu": r"$\mathrm{W}^-\rightarrow\mu\nu$",
    "WplusToMuNu": r"$\mathrm{W}^+\rightarrow\mu\nu$",
    "WminusToMuNu": r"$\mathrm{W}^-\rightarrow\mu\nu$"
}

axis_labels = {
    "ewPTll": r"$\mathrm{Post\ FSR}\ p_\mathrm{T}^{\ell\ell}$",
    "ewMll": r"$\mathrm{Post\ FSR}\ m^{\ell\ell}$", 
    "ewYll": r"$\mathrm{Post\ FSR}\ Y^{\ell\ell}$",
    "ewAbsYll": r"$\mathrm{Post\ FSR}\ |Y^{\ell\ell}|$",
    "ptgen": r"$\mathrm{Pre\ FSR}\ p_\mathrm{T}^{\ell}$",
    "etagen": r"$\mathrm{Pre\ FSR}\ \eta^{\ell}$", 
    "ptVgen": r"$\mathrm{Pre\ FSR}\ p_\mathrm{T}^{\ell\ell}$",
    "absYVgen": r"$\mathrm{Pre\ FSR}\ |Y^{\ell\ell}|$", 
    "massVgen": r"$\mathrm{Pre\ FSR}\ m^{\ell\ell}$", 
    "qT" : r"$\mathrm{Pre\ FSR}\ p_\mathrm{T}^{\ell\ell}$",
    "Q" : r"$\mathrm{Pre\ FSR}\ m^{\ell\ell}$", 
    "absY" : r"$\mathrm{Pre\ FSR}\ Y^{\ell\ell}$",
    "charge" : r"$\mathrm{Pre\ FSR\ charge}$", 
}

syst_labels = {
    "powhegnloewew" : {0: "nominal", 1: "powheg EW NLO / LO"},
    "powhegnloewew_ISR" : {0: "nominal", 1: "powheg EW NLO / NLO QED veto"},
    "horacenloew" : {0: "nominal", 1: "horace EW NLO / LO", 2: "horace EW NLO / LO doubled", },
    "winhacnloew" : {0: "nominal", 1: "winhac EW NLO / LO", 2: "winhac EW NLO / LO doubled", },
    "matrix_radish" : "MATRIX+RadISH",
    "virtual_ew" : {
        0: r"NLOEW + HOEW, CMS, ($G_\mu, m_\mathrm{Z}, \mathrm{sin}^2\Theta_\mathrm{eff}$) scheme",
        1: r"NLOEW + HOEW, PS, ($G_\mu, m_\mathrm{Z}, \mathrm{sin}^2\Theta_\mathrm{eff}$) scheme", 
        2: r"NLOEW + HOEW, CMS, ($\alpha(m_\mathrm{Z}),m _\mathrm{Z}, \mathrm{sin}^2\Theta_\mathrm{eff}$) scheme", }
}

syst_labels["virtual_ew_wlike"] = syst_labels["virtual_ew"]

poi_types = {
    "mu": "$\mu$",
    "nois": "$\mathrm{NOI}$",
    "pmaskedexp": "d$\sigma$ [pb]",
    "sumpois": "d$\sigma$ [pb]",
    "pmaskedexpnorm": "1/$\sigma$ d$\sigma$",
    "sumpoisnorm": "1/$\sigma$ d$\sigma$",
}
