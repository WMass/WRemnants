from utilities import boostHistHelpers as hh
from utilities import logging

logger = logging.child_logger(__name__)

# colors from CAT (https://cms-analysis.docs.cern.ch/guidelines/plotting/colors/)
# #5790fc blue
# #f89c20 orange
# #e42536 red
# #964a8b light purple
# #9c9ca1 grey
# #7a21dd dark purple

process_colors = {
    "Data": "black",
    "Zmumu": "#5790FC",
    "Z": "#5790FC",
    "Zll": "#5790FC",
    "Zee": "#5790FC",
    "Ztautau": "#964a8b",
    "Wmunu": "#E42536",
    "Wenu": "#E42536",
    "Wtaunu": "#F89C20",
    "DYlowMass": "deepskyblue",
    "PhotonInduced": "gold",
    "Top": "green",
    "Diboson": "#7A21DD",
    "Rare": "#7A21DD",
    "Other": "#7A21DD",
    "QCD": "#964A8B",
    "Fake": "#964A8B",
    "Fake_e": "#964A8B",
    "Fake_mu": "#964A8B",
    "Prompt": "#E42536",
}

process_supergroups = {
    "sv": {
        "Prompt": [
            "Wmunu",
            "Wtaunu",
            "Ztautau",
            "Zmumu",
            "DYlowMass",
            "PhotonInduced",
            "Top",
            "Diboson",
        ],
        "Fake": ["Fake"],
        "QCD": ["QCD"],
    },
    "w_mass": {
        "Wmunu": ["Wmunu"],
        "Wtaunu": ["Wtaunu"],
        "Z": ["Ztautau", "Zmumu", "DYlowMass"],
        "Fake": ["Fake"],
        "Rare": ["PhotonInduced", "Top", "Diboson"],
    },
    "z_dilepton": {
        "Z": ["Zmumu", "Ztautau"],
        "Other": ["Other", "PhotonInduced"],
    },
    "w_lowpu": {
        "Z": ["Ztautau", "Zmumu", "Zee", "DYlowMass"],
        "Rare": ["PhotonInduced", "Top", "Diboson"],
    },
}
process_supergroups["z_wlike"] = process_supergroups["z_dilepton"]
process_supergroups["z_lowpu"] = process_supergroups["z_dilepton"]

process_labels = {
    "Data": "Data",
    "Zmumu": r"Z/$\gamma^{\star}\to\mu\mu$",
    "Zee": r"Z/$\gamma^{\star}\to ee$",
    "Zll": r"Z/$\gamma^{\star}\to\mu\mu$",
    "Z": r"Z/$\gamma^{\star}\to\mu\mu/\tau\tau$",
    "Ztautau": r"Z/$\gamma^{\star}\to\tau\tau$",
    "Wmunu": r"W$^{\pm}\to\mu\nu$",
    "Wenu": r"W$^{\pm}\to e\nu$",
    "Wtaunu": r"W$^{\pm}\to\tau\nu$",
    "DYlowMass": r"Z/$\gamma^{\star}\to\mu\mu$, $10<m<50$ GeV",
    "PhotonInduced": r"$\gamma$-induced",
    "Top": "Top",
    "Diboson": "Diboson",
    "QCD": "QCD MC (predicted)",
    "Other": "Other",
    "Fake": "Nonprompt",
    "Fake_e": "Nonprompt (e)",
    "Fake_mu": r"Nonprompt (\mu)",
    "Prompt": "Prompt",
}

xlabels = {
    "pt": r"$\mathit{p}_{T}^{\mu}$ (GeV)",
    "ptGen": r"$\mathit{p}_{T}^{\mu}$ (GeV)",
    "ptW": r"$\mathit{p}_{T}^{\mu+p_{\mathrm{T}}^{miss}}$ (GeV)",
    "ptVGen": r"$\mathit{p}_{T}^\mathrm{V}$ (GeV)",
    "ptVgen": r"$\mathit{p}_{T}^\mathrm{V}$ (GeV)",
    "ptWgen": r"$\mathit{p}_{T}^\mathrm{W}$ (GeV)",
    "ptZgen": r"$\mathit{p}_{T}^\mathrm{Z}$ (GeV)",
    "muonJetPt": r"$\mathit{p}_{T}^\mathrm{jet[\mu]}$ (GeV)",
    "eta": r"$\mathit{\eta}^{\mu}$",
    "etaGen": r"$\mathit{\eta}^{\mu}$",
    "abseta": r"$|\mathit{\eta}^{\mu}|$",
    "absEta": r"$|\mathit{\eta}^{\mu}|$",
    "absEtaGen": r"$|\mathit{\eta}^{\mu}|$",
    "ptll": r"$\mathit{p}_{\mathrm{T}}^{\mu\mu}$ (GeV)",
    "yll": r"$\mathit{y}^{\mu\mu}$",
    "absYVGen": r"|$\mathit{Y}^\mathrm{V}$|",
    "mll": r"$\mathit{m}_{\mu\mu}$ (GeV)",
    "ewMll": r"$\mathit{m}^{\mathrm{EW}}_{\mu\mu}$ (GeV)",
    "costhetastarll": r"$\cos{\mathit{\theta}^{\star}_{\mu\mu}}$",
    "cosThetaStarll": r"$\cos{\mathit{\theta}^{\star}_{\mu\mu}}$",
    "phistarll": r"$\mathit{\phi}^{\star}_{\mu\mu}$",
    "phiStarll": r"$\mathit{\phi}^{\star}_{\mu\mu}$",
    "MET_pt": r"$\mathit{p}_{\mathrm{T}}^{miss}$ (GeV)",
    "MET": r"$\mathit{p}_{\mathrm{T}}^{miss}$ (GeV)",
    "met": r"$\mathit{p}_{\mathrm{T}}^{miss}$ (GeV)",
    "mt": r"$\mathit{m}_{T}^{\mu\nu}$ (GeV)",
    "mtfix": r"$\mathit{m}_{T}^\mathrm{fix}$ (GeV)",
    "etaPlus": r"$\mathit{\eta}^{\mu(+)}$",
    "etaMinus": r"$\mathit{\eta}^{\mu(-)}$",
    "ptPlus": r"$\mathit{p}_{\mathrm{T}}^{\mu(+)}$ (GeV)",
    "ptMinus": r"$\mathit{p}_{\mathrm{T}}^{\mu(-)}$ (GeV)",
    "etaSum": r"$\mathit{\eta}^{\mu(+)} + \mathit{\eta}^{\mu(-)}$",
    "etaDiff": r"$\mathit{\eta}^{\mu(+)} - \mathit{\eta}^{\mu(-)}$",
    "etaDiff": r"$\mathit{\eta}^{\mu(+)} - \mathit{\eta}^{\mu(-)}$",
    "etaAbsEta": r"$\mathit{\eta}^{\mu[\mathrm{argmax(|\mathit{\eta}^{\mu}|)}]}$",
    "ewMll": "ewMll",
    "ewMlly": "ewMlly",
    "ewLogDeltaM": "ewLogDeltaM",
    "dxy": r"$\mathit{d}_\mathrm{xy}$ (cm)",
    "iso": r"$I$ (GeV)",
    "relIso": r"$I_\mathrm{rel}$",
}

legend_labels = {
    "gamma_cusp-1.": r"$\mathit{\Gamma}_{cusp}$",
    "gamma_cusp1.": r"$\mathit{\Gamma}_{cusp}$",
    "gamma_mu_q-1.": r"$\mathit{\gamma}_{\mu}$",
    "gamma_mu_q1.": r"$\mathit{\gamma}_{\mu}$",
    "gamma_nu-1.": r"$\mathit{\gamma}_{\nu}$",
    "gamma_nu1.": r"$\mathit{\gamma}_{\nu}$",
    "Lambda20.25": r"$\mathit{\Lambda}_{2}$",
    "Lambda2-0.25": r"$\mathit{\Lambda}_{2}$",
    "h_qqV-1.": "Hard func.",
    "h_qqV1.": "Hard func.",
    "s-1.": "Soft func.",
    "s1.": "Soft func.",
    "b_qqV-0.5": r"$qqV$ BF",
    "b_qqV0.5": r"$qqV$ BF",
    "b_qqbarV-0.5": r"$q\bar{q}V$ BF",
    "b_qqbarV0.5": r"$q\bar{q}V$ BF",
    "b_qqS-0.5": r"$qqS$ BF",
    "b_qqS0.5": r"$qqS$ BF",
    "b_qqDS-0.5": r"$qq\Delta S$ BF",
    "b_qqDS0.5": r"$qq\Delta S$ BF",
    "b_qg-0.5": r"$qg$ BF",
    "b_qg0.5": r"$qg$ BF",
}

legend_labels_combine = {
    "massShiftW100MeV": r"$\mathit{m}_\mathrm{W} \pm 100\,\mathrm{MeV}$",
    "massShiftZ100MeV": r"$\mathit{m}_\mathrm{Z} \pm 100\,\mathrm{MeV}$",
    "QCDscaleWinclusive_PtV0_13000helicity_0_SymAvg": r"$\mathit{A}_0$",
    "QCDscaleWinclusive_PtV0_13000helicity_1_SymAvg": r"$\mathit{A}_1$",
    "QCDscaleWinclusive_PtV0_13000helicity_2_SymAvg": r"$\mathit{A}_2$",
    "QCDscaleWinclusive_PtV0_13000helicity_3_SymAvg": r"$\mathit{A}_3$",
    "resumTNP_gamma_nu": r"$\mathit{\gamma}_{\nu}$",
    "chargeVgenNP0scetlibNPWLambda2": r"$\mathit{\Lambda}_{2}$",
    "pythia_shower_kt": r"Pythia shower $\mathit{k}_T$",
    "nlo_ew_virtual": "EW virtual",
    "weak_default": "EW virtual",
    "virtual_ewCorr0": "EW virtual",
    "horacelophotosmecoffew_FSRCorr0": "FSR MEC off",
    "horaceqedew_FSRCorr0": "FSR horace",
    "pythiaew_ISRCorr0": "ISR off",
    "horacelophotosmecoffew_FSRCorr1": "FSR MEC off",
    "horaceqedew_FSRCorr1": "FSR horace",
    "pythiaew_ISRCorr1": "ISR off",
    "pdfMSHT20mbrangeSymAvg": r"$\mathit{m}_b + 1.25\, GeV$",
    "pdfMSHT20mcrangeSymAvg": r"$\mathit{m}_c + 0.2\, GeV$",
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
    "width",
    "ZmassAndWidth",
    "massAndWidth",
    "normXsecZ",
]
nuisance_groupings = {
    "super": [
        "Total",
        "stat",
        "binByBinStat",
        "theory",
        "expNoCalib",
        "muonCalibration",
    ],
    "max": common_groups
    + [
        "angularCoeffs",
        "pdfCT18Z",
        "pTModeling",
        "muon_eff_syst",
        "muon_eff_stat",
        "prefire",
        "muonCalibration",
        "Fake",
        "normWplus_Helicity-1",
        "normWplus_Helicity0",
        "normWplus_Helicity1",
        "normWplus_Helicity2",
        "normWplus_Helicity3",
        "normWplus_Helicity4",
        "normWminus_Helicity-1",
        "normWminus_Helicity0",
        "normWminus_Helicity1",
        "normWminus_Helicity2",
        "normWminus_Helicity3",
        "normWminus_Helicity4",
        "normW_Helicity-1",
        "normW_Helicity0",
        "normW_Helicity1",
        "normW_Helicity2",
        "normW_Helicity3",
        "normW_Helicity4",
        "normZ",
        "normZ_Helicity-1",
        "normZ_Helicity0",
        "normZ_Helicity1",
        "normZ_Helicity2",
        "normZ_Helicity3",
        "normZ_Helicity4",
    ],
    "min": common_groups
    + [
        "massShiftW",
        "massShiftZ",
        "QCDscalePtChargeMiNNLO",
        "QCDscaleZPtChargeMiNNLO",
        "QCDscaleWPtChargeMiNNLO",
        "QCDscaleZPtHelicityMiNNLO",
        "QCDscaleWPtHelicityMiNNLO",
        "QCDscaleZPtChargeHelicityMiNNLO",
        "QCDscaleWPtChargeHelicityMiNNLO",
        "pythia_shower_kt",
        "pdfCT18ZNoAlphaS",
        "pdfCT18ZAlphaS",
        "resumTNP",
        "resumNonpert",
        "resumTransition",
        "resumScale",
        "bcQuarkMass",
        "muon_eff_stat_reco",
        "muon_eff_stat_trigger",
        "muon_eff_stat_iso",
        "muon_eff_stat_idip",
        "muon_eff_syst_reco",
        "muon_eff_syst_trigger",
        "muon_eff_syst_iso",
        "muon_eff_syst_idip",
        "muonPrefire",
        "ecalPrefire",
        "nonClosure",
        "resolutionCrctn",
        "FakeRate",
        "FakeShape",
        "FakeeRate",
        "FakeeShape",
        "FakemuRate",
        "FakemuShape",
    ],
    "unfolding_max": [
        "Total",
        "stat",
        "binByBinStat",
        "binByBinStatW",
        "binByBinStatZ",
        "experiment",
        "angularCoeffs",
        "pdfCT18Z",
        "pTModeling",
        "theory_ew",
    ],
    "unfolding_min": [
        "Total",
        "stat",
        "binByBinStatW",
        "binByBinStat",
        "binByBinStatZ",
        "experiment",
        "QCDscalePtChargeMiNNLO",
        "QCDscaleZPtChargeMiNNLO",
        "QCDscaleWPtChargeMiNNLO",
        "QCDscaleZPtHelicityMiNNLO",
        "QCDscaleWPtHelicityMiNNLO",
        "QCDscaleZPtChargeHelicityMiNNLO",
        "QCDscaleWPtChargeHelicityMiNNLO",
        "QCDscaleZMiNNLO",
        "QCDscaleWMiNNLO",
        "pythia_shower_kt",
        "pdfCT18ZNoAlphaS",
        "pdfCT18ZAlphaS",
        "resumTNP",
        "resumNonpert",
        "resumTransition",
        "resumScale",
        "bcQuarkMass",
        "theory_ew",
    ],
}

text_dict = {
    "Zmumu": r"$\mathrm{Z}\rightarrow\mu\mu$",
    "ZToMuMu": r"$\mathrm{Z}\rightarrow\mu\mu$",
    "Wplusmunu": r"$\mathrm{W}^+\rightarrow\mu\nu$",
    "Wminusmunu": r"$\mathrm{W}^-\rightarrow\mu\nu$",
    "WplusToMuNu": r"$\mathrm{W}^+\rightarrow\mu\nu$",
    "WminusToMuNu": r"$\mathrm{W}^-\rightarrow\mu\nu$",
}

poi_types = {
    "mu": r"$\mu$",
    "nois": r"$\mathrm{NOI}$",
    "pmaskedexp": r"d$\sigma$ [pb]",
    "sumpois": r"d$\sigma$ [pb]",
    "pmaskedexpnorm": r"1/$\sigma$ d$\sigma$",
    "sumpoisnorm": r"1/$\sigma$ d$\sigma$",
    "ratiometapois": r"$\sigma(W^{+})/\sigma(W^{-})$",
    "helpois": "Ai",
    "helmetapois": "Ai",
}

axis_labels = {
    "ewPTll": r"$\mathrm{Post\ FSR}\ p_\mathrm{T}^{\mu\mu}$",
    "ewMll": r"$\mathrm{Post\ FSR}\ m^{\mu\mu}$",
    "ewYll": r"$\mathrm{Post\ FSR}\ Y^{\mu\mu}$",
    "ewAbsYll": r"$\mathrm{Post\ FSR}\ |Y^{\mu\mu}|$",
    "csCosTheta": r"$\mathrm{Post\ FSR\ \cos{\theta^{\star}_{\mu\mu}}}$",
    "ptgen": r"$\mathrm{Pre\ FSR}\ p_\mathrm{T}^{\mu}$",
    "etagen": r"$\mathrm{Pre\ FSR}\ \eta^{\mu}$",
    "ptVgen": r"$\mathrm{Pre\ FSR}\ p_\mathrm{T}^{\mu\mu}$",
    "absYVgen": r"$\mathrm{Pre\ FSR}\ |Y^{\mu\mu}|$",
    "massVgen": r"$\mathrm{Pre\ FSR}\ m^{\mu\mu}$",
    "csCosThetagen": r"$\mathrm{Pre\ FSR\ \cos{\theta^{\star}_{\mu\mu}}}$",
    "ptlhe": r"$\mathrm{LHE}\ p_\mathrm{T}^{\mu}$",
    "etalhe": r"$\mathrm{LHE}\ \eta^{\mu}$",
    "ptVlhe": r"$\mathrm{LHE}\ p_\mathrm{T}^{\mu\mu}$",
    "absYVlhe": r"$\mathrm{LHE}\ |Y^{\mu\mu}|$",
    "massVlhe": r"$\mathrm{LHE}\ m^{\mu\mu}$",
    "cosThetaStarlhe": r"$\mathrm{LHE\ \cos{\theta^{\star}_{\mu\mu}}}$",
    "qT": r"$\mathrm{Pre\ FSR}\ p_\mathrm{T}^{\mu\mu}$",
    "Q": r"$\mathrm{Pre\ FSR}\ m^{\mu\mu}$",
    "absY": r"$\mathrm{Pre\ FSR}\ Y^{\mu\mu}$",
    "charge": r"$\mathrm{Pre\ FSR\ charge}$",
}

systematics_labels = {
    "massShiftZ100MeV": r"$\Delta m_\mathrm{Z} = \pm 100\mathrm{MeV}$",
    "massShiftW100MeV": r"$\Delta m_\mathrm{W} = \pm 100\mathrm{MeV}$",
    "widthZ": r"$\Delta \Gamma_\mathrm{Z} = \pm 0.8\mathrm{MeV}$",
    "widthW": r"$\Delta \Gamma_\mathrm{W} = \pm 0.6\mathrm{MeV}$",
    # powhegFOEW variations
    "weak_no_ew": "no EW",
    "weak_no_ho": "no HO",
    "weak_default": "nominal",
    "weak_ps": "PS",
    "weak_mt_dn": r"$m_\mathrm{t}^\mathrm{down}$",
    "weak_mt_up": r"$m_\mathrm{t}^\mathrm{up}$",
    "weak_mz_dn": r"$m_\mathrm{Z}^\mathrm{down}$",
    "weak_mz_up": r"$m_\mathrm{Z}^\mathrm{up}$",
    "weak_gmu_dn": r"$G_\mu^\mathrm{up}$",
    "weak_gmu_up": r"$G_\mu^\mathrm{down}$",
    "weak_aem": r"$\alpha_\mathrm{EM}$",
    "weak_fs": "FS",
    "weak_mh_dn": r"$m_\mathrm{H}^\mathrm{down}$",
    "weak_mh_up": r"$m_\mathrm{H}^\mathrm{up}$",
    "weak_s2eff_0p23125": r"$\mathrm{sin}^2_\mathrm{eff}=0.23125$",
    "weak_s2eff_0p23105": r"$\mathrm{sin}^2_\mathrm{eff}=0.23105$",
    "weak_s2eff_0p22155": r"$\mathrm{sin}^2_\mathrm{eff}=0.22155$",
    "weak_s2eff_0p23185": r"$\mathrm{sin}^2_\mathrm{eff}=0.23185$",
    "weak_s2eff_0p23205": r"$\mathrm{sin}^2_\mathrm{eff}=0.23205$",
    "weak_s2eff_0p23255": r"$\mathrm{sin}^2_\mathrm{eff}=0.23255$",
    "weak_s2eff_0p23355": r"$\mathrm{sin}^2_\mathrm{eff}=0.23355$",
    "weak_s2eff_0p23455": r"$\mathrm{sin}^2_\mathrm{eff}=0.23455$",
    "weak_s2eff_0p22955": r"$\mathrm{sin}^2_\mathrm{eff}=0.22955$",
    "weak_s2eff_0p22655": r"$\mathrm{sin}^2_\mathrm{eff}=0.22655$",
    # EW
    "pythiaew_ISRCorr1": "Pythia ISR on / off",
    "horacelophotosmecoffew_FSRCorr1": "Photos MEC off / on",
    "horaceqedew_FSRCorr1": "Horace FSR / Photos",
    "nlo_ew_virtual": "EW virtual",
    "weak_default": "EW virtual",
    # alternative generators
    "matrix_radish": "MATRIX+RadISH",
}

systematics_labels_idxs = {
    "powhegnloewew": {0: "nominal", 1: "powheg EW NLO / LO"},
    "powhegnloewew_ISR": {0: "nominal", 1: "powheg EW NLO / NLO QED veto"},
    "pythiaew": {0: "nominal", 1: "pythia ISR EW on / off"},
    "horaceqedew": {
        0: "nominal",
        1: "Horace / Photos",
    },
    "horacenloew": {
        0: "nominal",
        1: "Horace EW NLO / LO",
        2: "Horace EW NLO / LO doubled",
    },
    "winhacnloew": {
        0: "nominal",
        1: "Winhac EW NLO / LO",
        2: "Wnhac EW NLO / LO doubled",
    },
    "horacelophotosmecoffew": {0: "nominal", 1: "Photos MEC off / on"},
    "virtual_ew": {
        0: r"NLOEW + HOEW, CMS, ($G_\mu, m_\mathrm{Z}, \mathrm{sin}^2\Theta_\mathrm{eff}$) scheme",
        1: r"NLOEW + HOEW, PS, ($G_\mu, m_\mathrm{Z}, \mathrm{sin}^2\Theta_\mathrm{eff}$) scheme",
        2: r"NLOEW + HOEW, CMS, ($\alpha(m_\mathrm{Z}),m _\mathrm{Z}, \mathrm{sin}^2\Theta_\mathrm{eff}$) scheme",
    },
}
systematics_labels_idxs["virtual_ew_wlike"] = systematics_labels_idxs["virtual_ew"]


def get_systematics_label(key, idx=0):
    if key in systematics_labels:
        return systematics_labels[key]

    # custom formatting
    if key in systematics_labels_idxs:
        return systematics_labels_idxs[key][idx]

    if "helicity" in key.split("_")[-1]:
        idx = int(key.split("_")[-1][-1])
        if idx == 0:
            label = "UL"
        else:
            label = str(idx - 1)

        return rf"$\pm\sigma_\mathrm{{{label}}}$"

    # default return key
    logger.info(f"No label found for {key}")
    return key


def get_labels_colors_procs_sorted(procs):
    # order of the processes in the plots by this list
    procs_sort = [
        "Wmunu",
        "Fake",
        "QCD",
        "Z",
        "Zmumu",
        "Wtaunu",
        "Top",
        "DYlowMass",
        "Other",
        "Ztautau",
        "Diboson",
        "PhotonInduced",
        "Prompt",
        "Rare",
    ][::-1]

    procs = sorted(
        procs, key=lambda x: procs_sort.index(x) if x in procs_sort else len(procs_sort)
    )
    logger.info(f"Found processes {procs} in fitresult")
    labels = [process_labels.get(p, p) for p in procs]
    colors = [process_colors.get(p, "red") for p in procs]
    return labels, colors, procs


def process_grouping(grouping, hist_stack, procs):
    if grouping in process_supergroups.keys():
        new_stack = {}
        for new_name, old_procs in process_supergroups[grouping].items():
            stacks = [hist_stack[procs.index(p)] for p in old_procs if p in procs]
            if len(stacks) == 0:
                continue
            new_stack[new_name] = hh.sumHists(stacks)
    else:
        new_stack = hist_stack
        logger.warning(
            f"No supergroups found for input file with mode {grouping}, proceed without merging groups"
        )

    labels, colors, procs = get_labels_colors_procs_sorted(
        [k for k in new_stack.keys()]
    )
    hist_stack = [new_stack[p] for p in procs]

    return hist_stack, labels, colors, procs
