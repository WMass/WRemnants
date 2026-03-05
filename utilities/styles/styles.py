import copy

import matplotlib.cm as cm

from wums import boostHistHelpers as hh
from wums import logging

logger = logging.child_logger(__name__)


def translate_html_to_latex(n):
    # transform html style formatting into latex style
    if "</" in n:
        n = (
            n.replace("<i>", r"\mathit{")
            .replace("</i>", "}")
            .replace("<sub>", "_{")
            .replace("<sup>", "^{")
            .replace("</sub>", "}")
            .replace("</sup>", "}")
            .replace(r"μ", r"\mu")
            .replace(r"ε", r"\epsilon")
            .replace(" ", r"\ ")
        )
    return n


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
    "WtoNMu_5": "#006400",
    "WtoNMu_10": "#009933",
    "WtoNMu_30": "#00CC99",
    "WtoNMu_50": "#00FFFF",
    "BSM": "#409C3D",
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
        "Zmumu": ["Zmumu"],
        "Other": ["Other", "Ztautau", "PhotonInduced"],
    },
    "w_lowpu": {
        "Zll": ["Ztautau", "Zmumu", "Zee", "DYlowMass"],
        "Rare": ["PhotonInduced", "Top", "Diboson"],
    },
}
process_supergroups["z_wlike"] = process_supergroups["z_dilepton"]
process_supergroups["w_mass_npmc"] = copy.deepcopy(process_supergroups["w_mass"])
process_supergroups["w_mass_npmc"]["Fake"] = ["QCD"]
process_supergroups["z_lowpu"] = process_supergroups["z_dilepton"]
process_supergroups["bsm"] = process_supergroups["w_mass"]
process_supergroups["bsm"]["BSM"] = ["WtoNMu_5", "WtoNMu_10", "WtoNMu_50"]

process_labels = {
    "Data": "Data",
    "Zmumu": r"Z/$\gamma^{\star}\to\mu\mu$",
    "Zee": r"Z/$\gamma^{\star}\to ee$",
    "Zll": r"Z/$\gamma^{\star}\to\ell\ell$",
    "Z": r"Z/$\gamma^{\star}\to\mu\mu/\tau\tau$",
    "Ztautau": r"Z/$\gamma^{\star}\to\tau\tau$",
    "Wmunu": r"W$^{\pm}\to\mu\nu$",
    "Wenu": r"W$^{\pm}\to e\nu$",
    "Wtaunu": r"W$^{\pm}\to\tau\nu$",
    "DYlowMass": r"Z/$\gamma^{\star}\to\mu\mu$, $10<m<50$ GeV",
    "PhotonInduced": r"$\gamma$-induced",
    "Top": "Top",
    "Diboson": "Diboson",
    "QCD": "QCD MC",
    "Other": "Other",
    "Fake": "Nonprompt",
    "Fake_e": "Nonprompt (e)",
    "Fake_mu": r"Nonprompt ($\mu$)",
    "Prompt": "Prompt",
    "WtoNMu_5": r"W$^{\pm}\to\mu\mathrm{N} (5GeV)$",
    "WtoNMu_10": r"W$^{\pm}\to\mu\mathrm{N} (10GeV)$",
    "WtoNMu_30": r"W$^{\pm}\to\mu\mathrm{N} (30GeV)$",
    "WtoNMu_50": r"W$^{\pm}\to\mu\mathrm{N} (50GeV)$",
}

axis_labels = {
    "pt": {"label": r"$\mathit{p}_{T}^{\mu}$", "unit": "GeV"},
    "ptGen": {"label": r"$\mathit{p}_{T}^{\mu}$", "unit": "GeV"},
    "ptW": {"label": r"$\mathit{p}_{T}^{\mu+MET}$", "unit": "GeV"},
    "ptVGen": {"label": r"$\mathit{p}_{T}^\mathrm{V}$", "unit": "GeV"},
    "ptVgen": {"label": r"$\mathit{p}_{T}^\mathrm{V}$", "unit": "GeV"},
    "ptWgen": {"label": r"$\mathit{p}_{T}^\mathrm{W}$", "unit": "GeV"},
    "ptZgen": {"label": r"$\mathit{p}_{T}^\mathrm{Z}$", "unit": "GeV"},
    "muonJetPt": {"label": r"$\mathit{p}_{T}^\mathrm{jet[\mu]}$", "unit": "GeV"},
    "recoil_perp": {"label": r"$\it{u}_{\mathrm{T}}^{\perp}$", "unit": "GeV"},
    "recoil_para": {"label": r"$\it{u}_{\mathrm{T}}^{\parallel}$", "unit": "GeV"},
    "qGen": r"$\mathit{q}^{\mu}$",
    "eta": r"$\mathit{\eta}^{\mu}$",
    "etaGen": r"$\mathit{\eta}^{\mu}$",
    "abseta": r"$|\mathit{\eta}^{\mu}|$",
    "absEta": r"$|\mathit{\eta}^{\mu}|$",
    "absEtaGen": r"$|\mathit{\eta}^{\mu}|$",
    "mtGen": {"label": r"$\mathit{m}_{T}^{\mu+MET}$", "unit": "GeV"},
    "ptll": {"label": r"$\mathit{p}_{\mathrm{T}}^{\mu\mu}$", "unit": "GeV"},
    "yll": r"$\mathit{y}^{\mu\mu}$",
    "absYVGen": r"|$\mathit{Y}^\mathrm{V}$|",
    "mll": {"label": r"$\mathit{m}_{\mu\mu}$", "unit": "GeV"},
    "ewMll": {"label": r"$\mathit{m}^{\mathrm{EW}}_{\mu\mu}$", "unit": "GeV"},
    "ewMlly": {"label": r"$\mathit{m}^{\mathrm{EW}}_{\mu\mu\gamma}$", "unit": "GeV"},
    "costhetastarll": r"$\cos{\mathit{\theta}^{\star}_{\mu\mu}}$",
    "cosThetaStarll": r"$\cos{\mathit{\theta}^{\star}_{\mu\mu}}$",
    "cosThetaStarll_quantile": {
        "label": r"$\cos{\mathit{\theta}^{\star}_{\mu\mu}}$",
        "unit": "quantile",
    },
    "absCosThetaStarll": r"$|\cos{\mathit{\theta}^{\star}_{\mu\mu}}|$",
    "phistarll": r"$\mathit{\phi}^{\star}_{\mu\mu}$",
    "phiStarll": r"$\mathit{\phi}^{\star}_{\mu\mu}$",
    "phiStarll_quantile": {
        "label": r"$\mathit{\phi}^{\star}_{\mu\mu}$",
        "unit": "quantile",
    },
    "absPhiStarll": r"$|\mathit{\phi}^{\star}_{\mu\mu}|$",
    "MET_pt": {"label": r"$\mathit{p}_{\mathrm{T}}^{miss}$", "unit": "GeV"},
    "MET": {"label": r"$\mathit{p}_{\mathrm{T}}^{miss}$", "unit": "GeV"},
    "met": {"label": r"$\mathit{p}_{\mathrm{T}}^{miss}$", "unit": "GeV"},
    "mt": {"label": r"$\mathit{m}_{T}$", "unit": "GeV"},
    "mtfix": {"label": r"$\mathit{m}_{T}^\mathrm{fix}$", "unit": "GeV"},
    "etaPlus": r"$\mathit{\eta}^{\mu(+)}$",
    "etaMinus": r"$\mathit{\eta}^{\mu(-)}$",
    "ptPlus": {"label": r"$\mathit{p}_{\mathrm{T}}^{\mu(+)}$", "unit": "GeV"},
    "ptMinus": {"label": r"$\mathit{p}_{\mathrm{T}}^{\mu(-)}$", "unit": "GeV"},
    "etaSum": r"$\mathit{\eta}^{\mu(+)} + \mathit{\eta}^{\mu(-)}$",
    "etaDiff": r"$\mathit{\eta}^{\mu(+)} - \mathit{\eta}^{\mu(-)}$",
    "etaDiff": r"$\mathit{\eta}^{\mu(+)} - \mathit{\eta}^{\mu(-)}$",
    "etaAbsEta": r"$\mathit{\eta}^{\mu[\mathrm{argmax(|\mathit{\eta}^{\mu}|)}]}$",
    "ewLogDeltaM": "ewLogDeltaM",
    "dxy": r"$\mathit{d}_\mathrm{xy}$ (cm)",
    "iso": {"label": r"$I$", "unit": "GeV"},
    "relIso": r"$I_\mathrm{rel}$",
    "run": r"Run range",
    "nRecoVtx": r"Number of reconstructed vertices",
    "PV_npvsGood": r"Number of reconstructed vertices",
    # "ewPTll": r"$\mathrm{Post\ FSR}\ p_\mathrm{T}^{\mu\mu}$",
    # "ewMll": r"$\mathrm{Post\ FSR}\ m^{\mu\mu}$",
    # "ewYll": r"$\mathrm{Post\ FSR}\ Y^{\mu\mu}$",
    # "ewAbsYll": r"$\mathrm{Post\ FSR}\ |Y^{\mu\mu}|$",
    # "csCosTheta": r"$\mathrm{Post\ FSR\ \cos{\theta^{\star}_{\mu\mu}}}$",
    # "ptgen": r"$\mathrm{Pre\ FSR}\ p_\mathrm{T}^{\mu}$",
    # "etagen": r"$\mathrm{Pre\ FSR}\ \eta^{\mu}$",
    # "ptVgen": r"$\mathrm{Pre\ FSR}\ p_\mathrm{T}^{\mu\mu}$",
    # "absYVgen": r"$\mathrm{Pre\ FSR}\ |Y^{\mu\mu}|$",
    # "massVgen": r"$\mathrm{Pre\ FSR}\ m^{\mu\mu}$",
    # "csCosThetagen": r"$\mathrm{Pre\ FSR\ \cos{\theta^{\star}_{\mu\mu}}}$",
    # "ptlhe": r"$\mathrm{LHE}\ p_\mathrm{T}^{\mu}$",
    # "etalhe": r"$\mathrm{LHE}\ \eta^{\mu}$",
    # "ptVlhe": r"$\mathrm{LHE}\ p_\mathrm{T}^{\mu\mu}$",
    # "absYVlhe": r"$\mathrm{LHE}\ |Y^{\mu\mu}|$",
    # "massVlhe": r"$\mathrm{LHE}\ m^{\mu\mu}$",
    # "cosThetaStarlhe": r"$\mathrm{LHE\ \cos{\theta^{\star}_{\mu\mu}}}$",
    # "qT": r"$\mathrm{Pre\ FSR}\ p_\mathrm{T}^{\mu\mu}$",
    # "Q": r"$\mathrm{Pre\ FSR}\ m^{\mu\mu}$",
    # "absY": r"$\mathrm{Pre\ FSR}\ Y^{\mu\mu}$",
    # "charge": r"$\mathrm{Pre\ FSR\ charge}$",
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
nuisance_grouping = {
    "super": [
        "Total",
        "stat",
        "binByBinStat",
        "theory",
        "expNoCalib",
        "muonCalibration",
    ],
    "splitAi": common_groups
    + [
        "angularCoeffs",
        "angularCoeffs_A0",
        "angularCoeffs_A1",
        "angularCoeffs_A2",
        "angularCoeffs_A3",
        "angularCoeffs_A4",
        "angularCoeffs_A5",
        "angularCoeffs_A6",
        "angularCoeffs_A7",
        "pdfCT18Z",
        "pTModeling",
        "muon_eff_syst",
        "muon_eff_stat",
        "prefire",
        "muonCalibration",
        "Fake",
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
        "binByBinStatDYlowMass",
        "binByBinStatDiboson",
        "binByBinStatPhotonInduced",
        "binByBinStatTop",
        "binByBinStatWmunu",
        "binByBinStatWtaunu",
        "binByBinStatZmumu",
        "binByBinStatZtautau",
        "binByBinStatWtoNMu_5",
        "binByBinStatWtoNMu_10",
        "binByBinStatWtoNMu_30",
        "binByBinStatWtoNMu_50",
    ],
    "unfolding": [
        "Total",
        "stat",
        "binByBinStat",
        "theory",
        "expNoLumi",
        "luminosity",
    ],
    "unfolding_prefsr": [
        "Total",
        "stat",
        "binByBinStat",
        "theory_qcd",
        "theory_ew",
        "expNoLumi",
        "luminosity",
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
        "massShift",
        "widthW",
        "widthZ",
        "sin2thetaZ",
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
    "xsecs": [
        "Total",
        "stat",
        "binByBinStat",
        "binByBinStatW",
        "binByBinStatZ",
        # "expNoLumi",
        "luminosity",
        "angularCoeffs",
        "pdfCT18Z",
        "pTModeling",
        "theory_ew",
        "massShift",
        "widthW",
        "widthZ",
        "sin2thetaZ",
        "muon_eff_syst",
        "muon_eff_stat",
        "prefire",
        "muonCalibration",
        "Fake",
        "recoil",
        "CMS_background",
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

translate_selection = {
    "charge": lambda x: rf"$\mathit{{q}}^\mu = {int(x)}$",
    "qGen": lambda x: rf"$\mathit{{q}}^\mu = {int(x)}$",
    "absYVGen": lambda l, h: rf"${round(l,3)} < |Y| < {round(h,3)}$",
    "helicitySig": lambda x: rf"$\sigma_{{{'UL' if x==-1 else int(x)}}}$",
    "ai": lambda x: rf"$A_{int(x)}$",
}

impact_labels = {
    "WtoNMu_5": "<i>μ</i><sub>W→Nμ(5GeV)</sub>",
    "WtoNMu_10": "<i>μ</i><sub>W→Nμ(10GeV)</sub>",
    "WtoNMu_30": "<i>μ</i><sub>W→Nμ(30GeV)</sub>",
    "WtoNMu_50": "<i>μ</i><sub>W→Nμ(50GeV)</sub>",
    "massShiftZ100MeV": "<i>m</i><sub>Z</sub>",
    "massShiftW100MeV": "<i>m</i><sub>W</sub>",
    "widthZ": "Γ<i>m</i><sub>Z</sub>",
    "widthW": "Γ<i>m</i><sub>W</sub>",
    "angularCoeffs": "Angular coefficients",
    "QCDscale": "<i>μ</i><sub>R </sub> <i>μ</i><sub>F </sub> scale",
    "QCDscaleZMiNNLO": "<i>μ</i><sub>R </sub> <i>μ</i><sub>F </sub> scale (Z)",
    "QCDscaleWMiNNLO": "<i>μ</i><sub>R </sub> <i>μ</i><sub>F </sub> scale (W)",
    "QCDscalePtChargeMiNNLO": "<i>μ</i><sub>R </sub> <i>μ</i><sub>F </sub> scale",
    "QCDscaleZPtChargeMiNNLO": "<i>μ</i><sub>R </sub> <i>μ</i><sub>F </sub> scale (Z)",
    "QCDscaleWPtChargeMiNNLO": "<i>μ</i><sub>R </sub> <i>μ</i><sub>F </sub> scale (W)",
    "QCDscaleZPtHelicityMiNNLO": "<i>μ</i><sub>R </sub> <i>μ</i><sub>F </sub> scale (Z)",
    "QCDscaleWPtHelicityMiNNLO": "<i>μ</i><sub>R </sub> <i>μ</i><sub>F </sub> scale (W)",
    "QCDscaleZPtChargeHelicityMiNNLO": "<i>μ</i><sub>R </sub> <i>μ</i><sub>F </sub> scale (Z)",
    "QCDscaleWPtChargeHelicityMiNNLO": "<i>μ</i><sub>R </sub> <i>μ</i><sub>F </sub> scale (W)",
    "binByBinStat": "Simulation stat.",
    "binByBinStatW": "Bin-by-bin stat. (W)",
    "binByBinStatZ": "Bin-by-bin stat. (Z)",
    "binByBinStatWmunu": "Bin-by-bin stat. (W→μν)",
    "binByBinStatWtaunu": "Bin-by-bin stat. (W→τν)",
    "binByBinStatDYlowMass": "Bin-by-bin stat. (Z→μμ, m<50GeV)",
    "binByBinStatZmumu": "Bin-by-bin stat. (Z→μμ)",
    "binByBinStatZtautau": "Bin-by-bin stat. (Z→ττ)",
    "binByBinStatDiboson": "Bin-by-bin stat. (VV)",
    "binByBinStatPhotonInduced": "Bin-by-bin stat. (γ-induced)",
    "binByBinStatTop": "Bin-by-bin stat. (top)",
    "binByBinStatWtoNMu_5": "Bin-by-bin stat. (BSM)",
    "binByBinStatWtoNMu_10": "Bin-by-bin stat. (BSM)",
    "binByBinStatWtoNMu_30": "Bin-by-bin stat. (BSM)",
    "binByBinStatWtoNMu_50": "Bin-by-bin stat. (BSM)",
    "recoil": "recoil",
    "CMS_background": "Other bkg.",
    "FakeHighMT": "FakeHighMT",
    "FakeLowMT": "FakeLowMT",
    "rFake": "fakerate",
    "rFakemu": "fakerate",
    "rFakee": "fakerate",
    "FakemuHighMT": "FakeHighMT",
    "FakemuLowMT": "FakeLowMT",
    "FakeeHighMT": "FakeHighMT",
    "FakeeLowMT": "FakeLowMT",
    "massShiftZ": "Z boson mass",
    "massShiftW": "W boson mass",
    "pdfMSHT20": "PDF",
    "pdfCT18Z": "PDF",
    "pdfMSHT20NoAlphaS": "PDF",
    "pdfMSHT20AlphaS": "<i>α</i><sub>S</sub> PDF",
    "pdfCT18ZAlphaS": "<i>α</i><sub>S</sub> PDF",
    "pdfCT18ZNoAlphaS": "PDF",
    "pTModeling": "<i>p</i><sub>T</sub><sup>V</sup> modelling",
    "resum": "Resummation",
    "resumTNP": "TNP",
    "resumNonpert": "Non pert.",
    "muonCalibration": "Muon calibration",
    "muonScale": "Muon scale",
    "nonClosure": "Muon scale",
    "resolutionCrctn": "Muon resolution",
    "muon_eff_stat": "<i>ε</i><sup>μ</sup><sub>stat</sub>",
    "muon_eff_syst": "<i>ε</i><sup>μ</sup><sub>syst</sub>",
    "prefire": "L1 prefire",
    "muonPrefire": "L1 muon prefire",
    "ecalPrefire": "L1 ecal prefire",
    "stat": "Data stat.",
    "luminosity": "Luminosity",
    "FakeRate": "Fake rate factors",
    "FakeShape": "Fake shape corrections",
    "Fake": "Fakes",
    "ZmassAndWidth": "Z mass & width",
    "bcQuarkMass": "b,c quark masses",
    "experiment": "Experiment",
    "expNoLumi": "Experiment",
    "expNoCalib": "Experiment (excl. calib.)",
    "theory": "Theory",
    "theory_qcd": "Theory (QCD)",
    "theory_ew": "Theory (EW)",
    "nlo_ew_virtual": "EW (virtual)",
    "pythia_shower_kt": "Pythia shower <i>k</i><sub>T</sub>",
    "Scale_correction_unc117": "<i>p</i><sub>T</sub><sup>μ</sup> calib. J/ψ stat. (117)",
    "Scale_correction_unc128": "<i>p</i><sub>T</sub><sup>μ</sup> calib. J/ψ stat. (128)",
    "Scale_correction_unc129": "<i>p</i><sub>T</sub><sup>μ</sup> calib. J/ψ stat. (129)",
    "Scale_correction_unc137": "<i>p</i><sub>T</sub><sup>μ</sup> calib. J/ψ stat. (137)",
    "Scale_correction_unc138": "<i>p</i><sub>T</sub><sup>μ</sup> calib. J/ψ stat. (138)",
    "Scale_correction_unc139": "<i>p</i><sub>T</sub><sup>μ</sup> calib. J/ψ stat. (139)",
    "Scale_correction_unc140": "<i>p</i><sub>T</sub><sup>μ</sup> calib. J/ψ stat. (140)",
    "Scale_correction_unc141": "<i>p</i><sub>T</sub><sup>μ</sup> calib. J/ψ stat. (141)",
    "Scale_correction_unc142": "<i>p</i><sub>T</sub><sup>μ</sup> calib. J/ψ stat. (142)",
    "Scale_correction_unc143": "<i>p</i><sub>T</sub><sup>μ</sup> calib. J/ψ stat. (143)",
    "ScaleClosA_correction_unc0": "<i>p</i><sub>T</sub><sup>μ</sup> calib. Δ<i>m</i><sub>Z</sub><sup>PDG</sup>",
    "ScaleClos_correction_unc48": "<i>p</i><sub>T</sub><sup>μ</sup> calib. Z closure stat. (48)",
    "FakeParam0var0": "Nonprompt syst. (0)",
    "FakeParam1var0": "Nonprompt syst. (1)",
    "FakeParam2var0": "Nonprompt syst. (2)",
    "pdf1CT18ZSymAvg": "PDF (1) [avg.]",
    "pdf2CT18ZSymAvg": "PDF (2) [avg.]",
    "pdf3CT18ZSymAvg": "PDF (3) [avg.]",
    "pdf4CT18ZSymAvg": "PDF (4) [avg.]",
    "pdf5CT18ZSymAvg": "PDF (5) [avg.]",
    "pdf6CT18ZSymAvg": "PDF (6) [avg.]",
    "pdf7CT18ZSymAvg": "PDF (7) [avg.]",
    "pdf8CT18ZSymAvg": "PDF (8) [avg.]",
    "pdf9CT18ZSymAvg": "PDF (9) [avg.]",
    "pdf10CT18ZSymAvg": "PDF (10) [avg.]",
    "pdf11CT18ZSymAvg": "PDF (11) [avg.]",
    "pdf12CT18ZSymAvg": "PDF (12) [avg.]",
    "pdf13CT18ZSymAvg": "PDF (13) [avg.]",
    "pdf14CT18ZSymAvg": "PDF (14) [avg.]",
    "pdf15CT18ZSymAvg": "PDF (15) [avg.]",
    "pdf16CT18ZSymAvg": "PDF (16) [avg.]",
    "pdf17CT18ZSymAvg": "PDF (17) [avg.]",
    "pdf18CT18ZSymAvg": "PDF (18) [avg.]",
    "pdf19CT18ZSymAvg": "PDF (19) [avg.]",
    "pdf20CT18ZSymAvg": "PDF (20) [avg.]",
    "pdf21CT18ZSymAvg": "PDF (21) [avg.]",
    "pdf22CT18ZSymAvg": "PDF (22) [avg.]",
    "pdf23CT18ZSymAvg": "PDF (23) [avg.]",
    "pdf24CT18ZSymAvg": "PDF (24) [avg.]",
    "pdf25CT18ZSymAvg": "PDF (25) [avg.]",
    "pdf26CT18ZSymAvg": "PDF (26) [avg.]",
    "pdf27CT18ZSymAvg": "PDF (27) [avg.]",
    "pdf28CT18ZSymAvg": "PDF (28) [avg.]",
    "pdf29CT18ZSymAvg": "PDF (29) [avg.]",
    "pdf1CT18ZSymDiff": "PDF (1) [diff.]",
    "pdf2CT18ZSymDiff": "PDF (2) [diff.]",
    "pdf3CT18ZSymDiff": "PDF (3) [diff.]",
    "pdf4CT18ZSymDiff": "PDF (4) [diff.]",
    "pdf5CT18ZSymDiff": "PDF (5) [diff.]",
    "pdf6CT18ZSymDiff": "PDF (6) [diff.]",
    "pdf7CT18ZSymDiff": "PDF (7) [diff.]",
    "pdf8CT18ZSymDiff": "PDF (8) [diff.]",
    "pdf9CT18ZSymDiff": "PDF (9) [diff.]",
    "pdf10CT18ZSymDiff": "PDF (10) [diff.]",
    "pdf11CT18ZSymDiff": "PDF (11) [diff.]",
    "pdf12CT18ZSymDiff": "PDF (12) [diff.]",
    "pdf13CT18ZSymDiff": "PDF (13) [diff.]",
    "pdf14CT18ZSymDiff": "PDF (14) [diff.]",
    "pdf15CT18ZSymDiff": "PDF (15) [diff.]",
    "pdf16CT18ZSymDiff": "PDF (16) [diff.]",
    "pdf17CT18ZSymDiff": "PDF (17) [diff.]",
    "pdf18CT18ZSymDiff": "PDF (18) [diff.]",
    "pdf19CT18ZSymDiff": "PDF (19) [diff.]",
    "pdf20CT18ZSymDiff": "PDF (20) [diff.]",
    "pdf21CT18ZSymDiff": "PDF (21) [diff.]",
    "pdf22CT18ZSymDiff": "PDF (22) [diff.]",
    "pdf23CT18ZSymDiff": "PDF (23) [diff.]",
    "pdf24CT18ZSymDiff": "PDF (24) [diff.]",
    "pdf25CT18ZSymDiff": "PDF (25) [diff.]",
    "pdf26CT18ZSymDiff": "PDF (26) [diff.]",
    "pdf27CT18ZSymDiff": "PDF (27) [diff.]",
    "pdf28CT18ZSymDiff": "PDF (28) [diff.]",
    "pdf29CT18ZSymDiff": "PDF (29) [diff.]",
    "pdfAlphaSSymAvg": "PDF <i>α</i><sub>S</sub> [avg.]",
    "pdfAlphaSSymDiff": "PDF <i>α</i><sub>S</sub> [diff.]",
    "pdfMSHT20mcrangeSymAvg": "PDF Δ<i>m</i><sub>c</sub> [avg.]",
    "pdfMSHT20mcrangeSymDiff": "PDF Δ<i>m</i><sub>c</sub> [diff.]",
    "pdfMSHT20mbrangeSymAvg": "PDF Δ<i>m</i><sub>b</sub> [avg.]",
    "pdfMSHT20mbrangeSymDiff": "PDF Δ<i>m</i><sub>b</sub> [diff.]",
    "QCDscaleWinclusive_PtV0_13000helicity_0_SymAvg": "<i>A</i><sub>0</sub> angular coeff., W, inc.",
    "QCDscaleWinclusive_PtV0_13000helicity_2_SymAvg": "<i>A</i><sub>2</sub> angular coeff., W, inc.",
    "scetlibNPgamma": "SCETLib γ",
    "chargeVgenNP0scetlibNPZLambda2": "SCETLib λ²(Z)",
    "chargeVgenNP1scetlibNPWLambda2": "SCETLib λ²(W⁻)",
    "chargeVgenNP0scetlibNPWLambda2": "SCETLib λ²(W⁺)",
    "chargeVgenNP0scetlibNPWDelta_Lambda2": "SCETLib Δλ²(W⁻)",
    "chargeVgenNP1scetlibNPWDelta_Lambda2": "SCETLib Δλ²(W⁺)",
    "chargeVgenNP0scetlibNPZDelta_Lambda2": "SCETLib Δλ²(Z)",
    "chargeVgenNP0scetlibNPWLambda4": "SCETLib λ⁴(W⁻)",
    "chargeVgenNP1scetlibNPWLambda4": "SCETLib λ⁴(W⁺)",
    "chargeVgenNP0scetlibNPZLambda4": "SCETLib λ⁴(Z)",
    "resumTransitionWSymDiff": "resum. transition W [diff.]",
    "resumTransitionZSymDiff": "resum. transition Z [diff.]",
    "resumTransitionZSymAvg": "resum. transition W [avg.]",
    "resumTransitionWSymAvg": "resum. transition Z [avg.]",
    "resumFOScaleWSymAvg": "resum. FO scale W [avg.]",
    "resumFOScaleWSymDiff": "resum. FO scale W [diff.]",
    "resumFOScaleZSymAvg": "resum. FO scale Z [avg.]",
    "resumFOScaleZSymDiff": "resum. FO scale Z [diff.]",
    "resumTNP_b_qqV": "resum. TNP BF qqV",
    "resumTNP_b_qg": "resum. TNP BF qg",
    "resumTNP_b_qqS": "resum. TNP BF qqS",
    "resumTNP_b_qqbarV": "resum. TNP BF q$\\bar{q}$V",
    "resumTNP_b_qqDS": "resum. TNP BF qqΔS",
    "resumTNP_gamma_nu": "resum. TNP γ<sub>ν</sub>",
    "resumTNP_gamma_mu_q": "resum. TNP γ<sub>μ</sub>",
    "resumTNP_gamma_cusp": "resum. TNP Γ<sub>cusp</sub>",
    "resumTNP_h_qqV": "resum. TNP Hard func.",
    "resumTNP_s": "resum. TNP Soft func.",
}


# same as impact labels but in latex format
systematics_labels = {k: translate_html_to_latex(v) for k, v in impact_labels.items()}


# systematics_labels = {

#     # powhegFOEW variations
#     "weak_no_ew": "no EW",
#     "weak_no_ho": "no HO",
#     "weak_ps": "PS",
#     "weak_mt_dn": r"$m_\mathrm{t}^\mathrm{down}$",
#     "weak_mt_up": r"$m_\mathrm{t}^\mathrm{up}$",
#     "weak_mz_dn": r"$m_\mathrm{Z}^\mathrm{down}$",
#     "weak_mz_up": r"$m_\mathrm{Z}^\mathrm{up}$",
#     "weak_gmu_dn": r"$G_\mu^\mathrm{up}$",
#     "weak_gmu_up": r"$G_\mu^\mathrm{down}$",
#     "weak_aem": r"$\alpha_\mathrm{EM}$",
#     "weak_fs": "FS",
#     "weak_mh_dn": r"$m_\mathrm{H}^\mathrm{down}$",
#     "weak_mh_up": r"$m_\mathrm{H}^\mathrm{up}$",
#     "weak_s2eff_0p23125": r"$\mathrm{sin}^2_\mathrm{eff}=0.23125$",
#     "weak_s2eff_0p23105": r"$\mathrm{sin}^2_\mathrm{eff}=0.23105$",
#     "weak_s2eff_0p22155": r"$\mathrm{sin}^2_\mathrm{eff}=0.22155$",
#     "weak_s2eff_0p23185": r"$\mathrm{sin}^2_\mathrm{eff}=0.23185$",
#     "weak_s2eff_0p23205": r"$\mathrm{sin}^2_\mathrm{eff}=0.23205$",
#     "weak_s2eff_0p23255": r"$\mathrm{sin}^2_\mathrm{eff}=0.23255$",
#     "weak_s2eff_0p23355": r"$\mathrm{sin}^2_\mathrm{eff}=0.23355$",
#     "weak_s2eff_0p23455": r"$\mathrm{sin}^2_\mathrm{eff}=0.23455$",
#     "weak_s2eff_0p22955": r"$\mathrm{sin}^2_\mathrm{eff}=0.22955$",
#     "weak_s2eff_0p22655": r"$\mathrm{sin}^2_\mathrm{eff}=0.22655$",
#     # EW
#     "pythiaew_ISRCorr1": "Pythia ISR on / off",
#     "horacelophotosmecoffew_FSRCorr1": "Photos MEC off / on",
#     "horaceqedew_FSRCorr1": "Horace FSR / Photos",
#     "nlo_ew_virtual": "EW virtual",
#     "weak_default": "EW virtual",
#     # alternative generators
#     "matrix_radish": "MATRIX+RadISH",
# }

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

    cmap = cm.get_cmap("tab10")

    procs = sorted(
        procs, key=lambda x: procs_sort.index(x) if x in procs_sort else len(procs_sort)
    )
    logger.debug(f"Found processes {procs} in fitresult")
    labels = [process_labels.get(p, p) for p in procs]
    colors = [process_colors.get(p, cmap(i % cmap.N)) for i, p in enumerate(procs)]
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
