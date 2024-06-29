from utilities import logging

logger = logging.child_logger(__name__)


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
    "w_mass":{
        "Z": ["Ztautau", "Zmumu", "DYlowMass"],
        "Rare": ["PhotonInduced", "Top", "Diboson"],
    },
    "z_dilepton":{
        "Other": ["Other","PhotonInduced", "Ztautau"],
    },
    "w_lowpu":{
        "Z": ["Ztautau", "Zmumu", "Zee", "DYlowMass"],
        "Rare": ["PhotonInduced", "Top", "Diboson"],
    },
}
process_supergroups["z_wlike"]=process_supergroups["z_dilepton"]
process_supergroups["z_lowpu"]=process_supergroups["z_dilepton"]

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
    "super":[
        "Total",
        "stat",
        "binByBinStat",
        "theory", 
        "experiment",
        "muonCalibration",
    ],
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

poi_types = {
    "mu": "$\mu$",
    "nois": "$\mathrm{NOI}$",
    "pmaskedexp": "d$\sigma$ [pb]",
    "sumpois": "d$\sigma$ [pb]",
    "pmaskedexpnorm": "1/$\sigma$ d$\sigma$",
    "sumpoisnorm": "1/$\sigma$ d$\sigma$",
}

axis_labels = {
    "ewPTll": r"$\mathrm{Post\ FSR}\ p_\mathrm{T}^{\ell\ell}$",
    "ewMll": r"$\mathrm{Post\ FSR}\ m^{\ell\ell}$", 
    "ewYll": r"$\mathrm{Post\ FSR}\ Y^{\ell\ell}$",
    "ewAbsYll": r"$\mathrm{Post\ FSR}\ |Y^{\ell\ell}|$",
    "csCosTheta" : r"$\mathrm{Post\ FSR\ \cos{\theta^{\star}_{\ell\ell}}}$", 
    "ptgen": r"$\mathrm{Pre\ FSR}\ p_\mathrm{T}^{\ell}$",
    "etagen": r"$\mathrm{Pre\ FSR}\ \eta^{\ell}$", 
    "ptVgen": r"$\mathrm{Pre\ FSR}\ p_\mathrm{T}^{\ell\ell}$",
    "absYVgen": r"$\mathrm{Pre\ FSR}\ |Y^{\ell\ell}|$", 
    "massVgen": r"$\mathrm{Pre\ FSR}\ m^{\ell\ell}$", 
    "csCosThetagen" : r"$\mathrm{Pre\ FSR\ \cos{\theta^{\star}_{\ell\ell}}}$", 
    "ptlhe": r"$\mathrm{LHE}\ p_\mathrm{T}^{\ell}$",
    "etalhe": r"$\mathrm{LHE}\ \eta^{\ell}$", 
    "ptVlhe": r"$\mathrm{LHE}\ p_\mathrm{T}^{\ell\ell}$",
    "absYVlhe": r"$\mathrm{LHE}\ |Y^{\ell\ell}|$", 
    "massVlhe": r"$\mathrm{LHE}\ m^{\ell\ell}$", 
    "cosThetaStarlhe" : r"$\mathrm{LHE\ \cos{\theta^{\star}_{\ell\ell}}}$", 
    "qT" : r"$\mathrm{Pre\ FSR}\ p_\mathrm{T}^{\ell\ell}$",
    "Q" : r"$\mathrm{Pre\ FSR}\ m^{\ell\ell}$", 
    "absY" : r"$\mathrm{Pre\ FSR}\ Y^{\ell\ell}$",
    "charge" : r"$\mathrm{Pre\ FSR\ charge}$", 
}

systematics_labels = {
    "massShiftZ100MeV": '$\Delta m_\mathrm{Z} = \pm 100\mathrm{MeV}$',
    "massShiftW100MeV": '$\Delta m_\mathrm{W} = \pm 100\mathrm{MeV}$',
    "widthZ": '$\Delta \Gamma_\mathrm{Z} = \pm 0.8\mathrm{MeV}$',
    "widthW": '$\Delta \Gamma_\mathrm{W} = \pm 0.6\mathrm{MeV}$',
    # powhegFOEW variations
    'weak_no_ew': "no EW", 
    'weak_no_ho': "no HO", 
    'weak_default': "nominal", 
    'weak_ps': "PS", 
    'weak_mt_dn': '$m_\mathrm{t}^\mathrm{down}$', 
    'weak_mt_up': '$m_\mathrm{t}^\mathrm{up}$', 
    'weak_mz_dn': '$m_\mathrm{Z}^\mathrm{down}$', 
    'weak_mz_up': '$m_\mathrm{Z}^\mathrm{up}$', 
    'weak_gmu_dn': '$G_\mu^\mathrm{up}$', 
    'weak_gmu_up': '$G_\mu^\mathrm{down}$', 
    'weak_aem': r'$\alpha_\mathrm{EM}$',  
    'weak_fs': 'FS',  
    'weak_mh_dn': '$m_\mathrm{H}^\mathrm{down}$',  
    'weak_mh_up': '$m_\mathrm{H}^\mathrm{up}$',   
    'weak_s2eff_0p23125': '$\mathrm{sin}^2_\mathrm{eff}=0.23125$',  
    'weak_s2eff_0p23105': '$\mathrm{sin}^2_\mathrm{eff}=0.23105$',   
    'weak_s2eff_0p22155': '$\mathrm{sin}^2_\mathrm{eff}=0.22155$',  
    'weak_s2eff_0p23185': '$\mathrm{sin}^2_\mathrm{eff}=0.23185$',  
    'weak_s2eff_0p23205': '$\mathrm{sin}^2_\mathrm{eff}=0.23205$', 
    'weak_s2eff_0p23255': '$\mathrm{sin}^2_\mathrm{eff}=0.23255$',  
    'weak_s2eff_0p23355': '$\mathrm{sin}^2_\mathrm{eff}=0.23355$',  
    'weak_s2eff_0p23455': '$\mathrm{sin}^2_\mathrm{eff}=0.23455$',  
    'weak_s2eff_0p22955': '$\mathrm{sin}^2_\mathrm{eff}=0.22955$',  
    'weak_s2eff_0p22655': '$\mathrm{sin}^2_\mathrm{eff}=0.22655$',
    # EW
    'pythiaew_ISRCorr1': 'Pythia ISR on / off',
    'horacelophotosmecoffew_FSRCorr1': 'Photos MEC off / on',
    'horaceqedew_FSRCorr1': 'Horace FSR / Photos',
    'nlo_ew_virtual': 'EW virtual',
    'weak_default': 'EW virtual',
    # alternative generators
    "matrix_radish" : "MATRIX+RadISH",
}

systematics_labels_idxs = {
    "powhegnloewew" : {0: "nominal", 1: "powheg EW NLO / LO"},
    "powhegnloewew_ISR" : {0: "nominal", 1: "powheg EW NLO / NLO QED veto"},
    "pythiaew" : {0: "nominal", 1: "pythia ISR EW on / off"},
    "horaceqedew" : {0: "nominal", 1: "Horace / Photos", },
    "horacenloew" : {0: "nominal", 1: "Horace EW NLO / LO", 2: "Horace EW NLO / LO doubled", },
    "winhacnloew" : {0: "nominal", 1: "Winhac EW NLO / LO", 2: "Wnhac EW NLO / LO doubled", },
    "horacelophotosmecoffew": {0: "nominal", 1: "Photos MEC off / on"},
    "virtual_ew" : {
        0: r"NLOEW + HOEW, CMS, ($G_\mu, m_\mathrm{Z}, \mathrm{sin}^2\Theta_\mathrm{eff}$) scheme",
        1: r"NLOEW + HOEW, PS, ($G_\mu, m_\mathrm{Z}, \mathrm{sin}^2\Theta_\mathrm{eff}$) scheme", 
        2: r"NLOEW + HOEW, CMS, ($\alpha(m_\mathrm{Z}),m _\mathrm{Z}, \mathrm{sin}^2\Theta_\mathrm{eff}$) scheme", }
}
systematics_labels_idxs["virtual_ew_wlike"] = systematics_labels_idxs["virtual_ew"]


def get_systematics_label(key, idx=0):
    if key in systematics_labels:
        return systematics_labels[key]
    
    # custom formatting
    if key in systematics_labels_idxs:
        return systematics_labels_idxs[key][idx]

    # default return key
    logger.info(f"No label found for {key}")
    return key


def get_labels_colors_procs_sorted(procs):
    # order of the processes in the plots
    procs_sort = ["Wmunu", "Fake", "Zmumu", "Wtaunu", "Top", "DYlowMass", "Other", "Ztautau", "Diboson", "PhotonInduced"][::-1]

    procs = sorted(procs, key=lambda x: procs_sort.index(x) if x in procs_sort else len(procs_sort))
    logger.info(f"Found processes {procs} in fitresult")
    labels = [process_labels.get(p, p) for p in procs]
    colors = [process_colors.get(p, "red") for p in procs]
    return labels, colors, procs