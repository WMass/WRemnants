## eras
eras_run2 = ["2016PreVFP", "2016PostVFP", "2017", "2018", "13TeVGen"]

supported_eras = eras_run2 + [
    "2016PostVFP",
    "2017G",
    "2017H",
    "2023_PUAVE1",
    "2023_PUAVE2",
    "2023_PUAVE5",
    "2023_PUAVE10",
    "13TeVGen",
]

## Samples with sqrt{S} = 13 TeV
# central MiNNLO samples with muon decay
wprocs_mu_minnlo_run2 = [f"Wplusmunu_{e}" for e in eras_run2] + [
    f"Wminusmunu_{e}" for e in eras_run2
]
zprocs_mu_minnlo_run2 = [
    f"Zmumu_{e}" for e in eras_run2
]  # + [f"DYJetsToMuMuMass10to50_{e}" for e in eras_run2]

# central MiNNLO samples with muon or e decay
wprocs_emu_minnlo_2017H = [
    "Wplusmunu_2017H",
    "Wminusmunu_2017H",
    "Wplusenu_2017H",
    "Wminusenu_2017H",
]
zprocs_emu_minnlo_2017H = ["Zmumu_2017H", "Zee_2017H"]
vprocs_emu_minnlo_2017H = wprocs_emu_minnlo_2017H + zprocs_emu_minnlo_2017H

wprocs_emu_minnlo = wprocs_mu_minnlo_run2 + wprocs_emu_minnlo_2017H
zprocs_emu_minnlo = zprocs_mu_minnlo_run2 + zprocs_emu_minnlo_2017H
vprocs_emu_minnlo = wprocs_emu_minnlo + zprocs_emu_minnlo

# central MiNNLO samples with tau
wprocs_tau_minnlo_run2 = [f"Wplustaunu_{e}" for e in eras_run2] + [
    f"Wminustaunu_{e}" for e in eras_run2
]
zprocs_tau_minnlo_run2 = [f"Ztautau_{e}" for e in eras_run2] + [
    f"Ztautau10to50_{e}" for e in eras_run2
]

wprocs_tau_minnlo_2017H = [
    "Wplustaunu_2017H",
    "Wminustaunu_2017H",
]
zprocs_tau_minnlo_2017H = [
    "Ztautau_2017H",
]

wprocs_tau_minnlo = wprocs_tau_minnlo_run2 + wprocs_tau_minnlo_2017H
zprocs_tau_minnlo = zprocs_tau_minnlo_run2 + zprocs_tau_minnlo_2017H
vprocs_tau_minnlo = wprocs_tau_minnlo + zprocs_tau_minnlo

wprocs_minnlo = wprocs_emu_minnlo + wprocs_tau_minnlo
zprocs_minnlo = zprocs_emu_minnlo + zprocs_tau_minnlo
vprocs_minnlo = wprocs_minnlo + zprocs_minnlo

wprocs_2017H = wprocs_emu_minnlo_2017H + wprocs_tau_minnlo_2017H
zprocs_2017H = zprocs_emu_minnlo_2017H + zprocs_tau_minnlo_2017H
vprocs_2017H = wprocs_2017H + zprocs_2017H

# alternative gen samples at sqrt{s} = 13
wprocs_alt = [
    "Wplusmunu_MiNNLO",
    "Wminusmunu_MiNNLO",
    "Wplusmunu_MiNNLO-noqedisr",
    "Wminusmunu_MiNNLO-noqedisr",
    "Wplusmunu_horace-lo-photos",
    "Wplusmunu_horace-lo-photos-mecoff",
    "Wplusmunu_horace-nlo",
    "Wplusmunu_horace-lo",
    "Wplusmunu_horace-qed",
    "Wminusmunu_horace-lo-photos",
    "Wminusmunu_horace-lo-photos-mecoff",
    "Wminusmunu_horace-nlo",
    "Wminusmunu_horace-lo",
    "Wminusmunu_horace-qed",
    "Wplusmunu_winhac-lo-photos",
    "Wplusmunu_winhac-lo",
    "Wplusmunu_winhac-nlo",
    "Wminusmunu_winhac-lo-photos",
    "Wminusmunu_winhac-lo",
    "Wminusmunu_winhac-nlo",
    "WplusCharmToMuNu",
    "WminusCharmToMuNu",
]
zprocs_alt = [
    "Zmumu_MiNNLO",
    "Zmumu_13TeVGen",
    "ZmumuMiNLO",
    "ZmumuNNLOPS",
    "Zmumu_MiNNLO-noqedisr",
    "Zmumu_horace-lo-photos",
    "Zmumu_horace-lo-photos-isroff",
    "Zmumu_horace-lo-photos-mecoff",
    "Zmumu_horace-nlo",
    "Zmumu_horace-lo",
    "Zmumu_horace-new",
    "Zmumu_horace-qed",
    "Zmumu_horace-alpha-fsr-off-isr-off",
    "Zmumu_horace-alpha-old-fsr-off-isr-off",
    "Zmumu_horace-alpha-old-fsr-off-isr-pythia",
    "Zmumu_renesance-lo",
    "Zmumu_renesance-nlo",
    "Zmumu_powheg-lo",
    "Zmumu_powheg-nloew-qedveto",
    "Zmumu_powheg-nloew",
]

wprocs_bsm = [
    "WtoNMuMass5_2016PostVFP",
    "WtoNMuMass10_2016PostVFP",
    "WtoNMuMass30_2016PostVFP",
    "WtoNMuMass50_2016PostVFP",
    "WtoMuNuSMEFT_2016PostVFP",
]

## Samples with sqrt{S} = 5020GeV
wprocs_emu_minnlo_2017G = [
    "Wplusmunu_2017G",
    "Wminusmunu_2017G",
    "Wplusenu_2017G",
    "Wminusenu_2017G",
]
zprocs_emu_minnlo_2017G = ["Zmumu_2017G", "Zee_2017G"]
vprocs_emu_minnlo_2017G = wprocs_emu_minnlo_2017G + zprocs_emu_minnlo_2017G

wprocs_tau_minnlo_2017G = [
    "Wplustaunu_2017G",
    "Wminustaunu_2017G",
]
zprocs_tau_minnlo_2017G = [
    "Ztautau_2017G",
]
vprocs_tau_minnlo_2017G = wprocs_tau_minnlo_2017G + zprocs_tau_minnlo_2017G

wprocs_minnlo_2017G = wprocs_emu_minnlo_2017G + wprocs_tau_minnlo_2017G
zprocs_minnlo_2017G = zprocs_emu_minnlo_2017G + zprocs_tau_minnlo_2017G
vprocs_minnlo_2017G = wprocs_minnlo_2017G + zprocs_minnlo_2017G

# all W and Z samples
wprocs = wprocs_minnlo + wprocs_alt + wprocs_bsm + wprocs_minnlo_2017G
zprocs = zprocs_minnlo + zprocs_alt + zprocs_minnlo_2017G
vprocs = wprocs + zprocs + vprocs_minnlo_2017G

zprocs_recoil = ["Zmumu_2016PostVFP"]
wprocs_recoil = ["Wplusmunu_2016PostVFP", "Wminusmunu_2016PostVFP"]

zprocs_recoil_lowpu = ["Zmumu", "Zee"]
wprocs_recoil_lowpu = ["Wminusmunu", "Wminusenu", "Wplusmunu", "Wplusenu"]
