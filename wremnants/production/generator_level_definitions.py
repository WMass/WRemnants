import narf
from wums import logging

narf.clingutils.Declare('#include "theoryTools.hpp"')

logger = logging.child_logger(__name__)


def define_dressed_vars(df, mode, flavor="mu"):
    if "dressedGenV_mom4" in df.GetColumnNames():
        logger.debug("LHE variables are already defined, do nothing here.")
        return df

    logger.info(f"Defining dressed variables for mode '{mode}' and flavor '{flavor}'")

    # use postfsr neutrinos
    df = define_postfsr_vars(df, mode)

    lep_pdgId = 13 if flavor == "mu" else 11

    if mode[0] == "z":
        df = df.Define("dressedLep", f"GenDressedLepton_pdgId=={lep_pdgId}")
        df = df.Define("dressedAntiLep", f"GenDressedLepton_pdgId==-{lep_pdgId}")

        df = df.Define("hasDressedLep", "ROOT::VecOps::Any(dressedLep)")
        df = df.Define("hasDressedAntiLep", "ROOT::VecOps::Any(dressedAntiLep)")

        df = df.Define(
            "dressedLep_idx", "ROOT::VecOps::ArgMax(GenDressedLepton_pt[dressedLep])"
        )
        df = df.Define(
            "dressedAntiLep_idx",
            "ROOT::VecOps::ArgMax(GenDressedLepton_pt[dressedAntiLep])",
        )

        df = df.Define(
            "dressedLep_pt",
            "hasDressedLep ? static_cast<double>(GenDressedLepton_pt[dressedLep][dressedLep_idx]) : 0",
        )
        df = df.Define(
            "dressedLep_eta",
            "hasDressedLep ? GenDressedLepton_eta[dressedLep][dressedLep_idx] : 0",
        )
        df = df.Define(
            "dressedLep_phi",
            "hasDressedLep ? GenDressedLepton_phi[dressedLep][dressedLep_idx] : 0",
        )
        df = df.Define(
            "dressedLep_mass",
            "hasDressedLep ? GenDressedLepton_mass[dressedLep][dressedLep_idx] : 0",
        )

        df = df.Define(
            "dressedAntiLep_pt",
            "hasDressedAntiLep ? static_cast<double>(GenDressedLepton_pt[dressedAntiLep][dressedAntiLep_idx]) : 0",
        )
        df = df.Define(
            "dressedAntiLep_eta",
            "hasDressedAntiLep ? GenDressedLepton_eta[dressedAntiLep][dressedAntiLep_idx] : 0",
        )
        df = df.Define(
            "dressedAntiLep_phi",
            "hasDressedAntiLep ? GenDressedLepton_phi[dressedAntiLep][dressedAntiLep_idx] : 0",
        )
        df = df.Define(
            "dressedAntiLep_mass",
            "hasDressedAntiLep ? GenDressedLepton_mass[dressedAntiLep][dressedAntiLep_idx] : 0",
        )

        df = df.Define(
            "dressedLep_mom4",
            "ROOT::Math::PtEtaPhiMVector(dressedLep_pt, dressedLep_eta, dressedLep_phi, dressedLep_mass)",
        )
        df = df.Define(
            "dressedAntiLep_mom4",
            "ROOT::Math::PtEtaPhiMVector(dressedAntiLep_pt, dressedAntiLep_eta, dressedAntiLep_phi, dressedAntiLep_mass)",
        )

        df = df.Define(
            "dressedGenV_mom4",
            "dressedLep_mom4 + dressedAntiLep_mom4 + postfsrNeutrinos_mom4",
        )
    else:
        df = df.Define("dressedLep", f"abs(GenDressedLepton_pdgId)=={lep_pdgId}")
        df = df.Define("hasDressedLep", "ROOT::VecOps::Any(dressedLep)")
        df = df.Define(
            "dressedLep_idx", "ROOT::VecOps::ArgMax(GenDressedLepton_pt[dressedLep])"
        )

        df = df.Define(
            "dressedLep_pt",
            "hasDressedLep ? static_cast<double>(GenDressedLepton_pt[dressedLep][dressedLep_idx]) : 0",
        )
        df = df.Define(
            "dressedLep_eta",
            "hasDressedLep ? GenDressedLepton_eta[dressedLep][dressedLep_idx] : 0",
        )
        df = df.Define(
            "dressedLep_phi",
            "hasDressedLep ? GenDressedLepton_phi[dressedLep][dressedLep_idx] : 0",
        )
        df = df.Define(
            "dressedLep_mass",
            "hasDressedLep ? GenDressedLepton_mass[dressedLep][dressedLep_idx] : 0",
        )

        df = df.Define(
            "dressedLep_mom4",
            "ROOT::Math::PtEtaPhiMVector(dressedLep_pt, dressedLep_eta, dressedLep_phi, dressedLep_mass)",
        )

        df = df.Define("dressedGenV_mom4", "dressedLep_mom4 + postfsrNeutrinos_mom4")

    df = df.Define("dressed_MV", "dressedGenV_mom4.mass()")
    df = df.Define("dressed_absYV", "std::fabs(dressedGenV_mom4.Rapidity())")
    df = df.Define("dressed_PTV", "dressedGenV_mom4.pt()")

    return df


def define_lhe_vars(df):
    if "lheLeps" in df.GetColumnNames():
        logger.debug("LHE leptons are already defined, do nothing here.")
        return df

    logger.info("Defining LHE variables")

    # SM leptons (11-16) or BSM neutrino (9900012)
    df = df.Define(
        "lheLeps",
        "LHEPart_status == 1 && LHEPart_pdgId >= 11 && LHEPart_pdgId <= 16",
    )
    df = df.Define(
        "lheAntiLeps",
        "LHEPart_status == 1 && LHEPart_pdgId <= -11 && LHEPart_pdgId >= -16",
    )
    df = df.Define(
        "lheBSM",
        "LHEPart_pdgId == 9900012 && LHEPart_status == 1",
    )

    df = df.Define("lheLep", "Sum(lheLeps) == 1 ? lheLeps : lheBSM")
    df = df.Define("lheAntiLep", "Sum(lheAntiLeps) == 1 ? lheAntiLeps : lheBSM")
    df = df.Define(
        "lheLep_idx",
        'if (Sum(lheLep) != 1) throw std::runtime_error("lhe lepton not found, sum = " + std::to_string(Sum(lheAntiLep))); return ROOT::VecOps::ArgMax(lheLep);',
    )
    df = df.Define(
        "lheAntiLep_idx",
        'if (Sum(lheAntiLep) != 1) throw std::runtime_error("lhe anti-lepton not found, sum = " + std::to_string(Sum(lheAntiLep))); return ROOT::VecOps::ArgMax(lheAntiLep);',
    )

    df = df.Define("lheVs", "abs(LHEPart_pdgId) >=23 && abs(LHEPart_pdgId)<=24")
    df = df.Define(
        "lheV_idx",
        'if (Sum(lheVs) != 1) throw std::runtime_error("LHE V not found."); return ROOT::VecOps::ArgMax(lheVs);',
    )
    df = df.Define("lheV_pdgId", "LHEPart_pdgId[lheV_idx]")
    df = df.Define("lheV_pt", "LHEPart_pt[lheV_idx]")

    df = df.Define(
        "lheLep_mom",
        "ROOT::Math::PtEtaPhiMVector(LHEPart_pt[lheLep_idx], LHEPart_eta[lheLep_idx], LHEPart_phi[lheLep_idx], LHEPart_mass[lheLep_idx])",
    )
    df = df.Define(
        "lheAntiLep_mom",
        "ROOT::Math::PtEtaPhiMVector(LHEPart_pt[lheAntiLep_idx], LHEPart_eta[lheAntiLep_idx], LHEPart_phi[lheAntiLep_idx], LHEPart_mass[lheAntiLep_idx])",
    )
    df = df.Define(
        "lheV",
        "ROOT::Math::PxPyPzEVector(lheLep_mom)+ROOT::Math::PxPyPzEVector(lheAntiLep_mom)",
    )
    df = df.Define("ptVlhe", "lheV.pt()")
    df = df.Define("massVlhe", "lheV.mass()")
    df = df.Define("ptqVlhe", "lheV.pt()/lheV.mass()")
    df = df.Define("yVlhe", "lheV.Rapidity()")
    df = df.Define("phiVlhe", "lheV.Phi()")
    df = df.Define("absYVlhe", "std::fabs(yVlhe)")
    df = df.Define(
        "chargeVlhe", "LHEPart_pdgId[lheLep_idx] + LHEPart_pdgId[lheAntiLep_idx]"
    )
    df = df.Define(
        "csSineCosThetaPhilhe", "wrem::csSineCosThetaPhi(lheAntiLep_mom, lheLep_mom)"
    )
    df = df.Define("csCosThetalhe", "csSineCosThetaPhilhe.costheta")
    df = df.Define("csPhilhe", "csSineCosThetaPhilhe.phi()")
    df = df.Define(
        "csAngularMomentslhe", "wrem::csAngularMoments(csSineCosThetaPhilhe)"
    )

    if "LHEWeight_originalXWGTUP" in df.GetColumnNames():
        df = df.Define(
            "csAngularMomentslhe_wnom",
            "auto res = csAngularMomentslhe; res = LHEWeight_originalXWGTUP*res; return res;",
        )
    else:
        df = df.Alias("csAngularMomentslhe_wnom", "csAngularMomentslhe")

    return df


def define_prefsr_vars(df):
    if "prefsrLeps" in df.GetColumnNames():
        logger.debug("PreFSR leptons are already defined, do nothing here.")
        return df

    logger.info("Defining preFSR variables")

    df = df.Define(
        "prefsrLeps",
        "wrem::prefsrLeptons(GenPart_status, GenPart_statusFlags, GenPart_pdgId, GenPart_genPartIdxMother)",
    )
    df = df.Define(
        "genl",
        "ROOT::Math::PtEtaPhiMVector(GenPart_pt[prefsrLeps[0]], GenPart_eta[prefsrLeps[0]], GenPart_phi[prefsrLeps[0]], GenPart_mass[prefsrLeps[0]])",
    )
    df = df.Define(
        "genlanti",
        "ROOT::Math::PtEtaPhiMVector(GenPart_pt[prefsrLeps[1]], GenPart_eta[prefsrLeps[1]], GenPart_phi[prefsrLeps[1]], GenPart_mass[prefsrLeps[1]])",
    )
    df = df.Define(
        "genV", "ROOT::Math::PxPyPzEVector(genl)+ROOT::Math::PxPyPzEVector(genlanti)"
    )
    df = df.Define("ptVgen", "genV.pt()")
    df = df.Define("massVgen", "genV.mass()")
    df = df.Define("ptqVgen", "genV.pt()/genV.mass()")
    df = df.Define("yVgen", "genV.Rapidity()")
    df = df.Define("phiVgen", "genV.Phi()")
    df = df.Define("absYVgen", "std::fabs(yVgen)")
    df = df.Define(
        "chargeVgen",
        "-1 * (GenPart_pdgId[prefsrLeps[0]] % 2 + GenPart_pdgId[prefsrLeps[1]] % 2)",
    )
    df = df.Define("csSineCosThetaPhigen", "wrem::csSineCosThetaPhi(genlanti, genl)")
    df = df.Define("csCosThetagen", "csSineCosThetaPhigen.costheta")
    df = df.Define("csPhigen", "csSineCosThetaPhigen.phi()")

    # define w and w-like variables
    df = df.Define("qgen", "isEvenEvent ? -1 : 1")
    df = df.Define("ptgen", "isEvenEvent ? genl.pt() : genlanti.pt()")
    df = df.Define("etagen", "isEvenEvent ? genl.eta() : genlanti.eta()")
    df = df.Define("absetagen", "std::fabs(etagen)")
    df = df.Define("ptOthergen", "isEvenEvent ? genlanti.pt() : genl.pt()")
    df = df.Define("etaOthergen", "isEvenEvent ? genlanti.eta() : genl.eta()")
    df = df.Define("absetaOthergen", "std::fabs(etaOthergen)")
    df = df.Define(
        "mTVgen", "wrem::mt_2(genl.pt(), genl.phi(), genlanti.pt(), genlanti.phi())"
    )

    return df


def define_intermediate_gen_vars(df, label, statusMin, statusMax):
    # define additional variables corresponding to intermediate states in the pythia history
    df = df.Define(
        f"idxV{label}",
        f"wrem::selectGenPart(GenPart_status, GenPart_pdgId, 23, 24, {statusMin}, {statusMax})",
    )
    df = df.Define(
        f"mom4V{label}",
        f"ROOT::Math::PtEtaPhiMVector(GenPart_pt[idxV{label}], GenPart_eta[idxV{label}], GenPart_phi[idxV{label}], GenPart_mass[idxV{label}])",
    )
    df = df.Define(f"ptV{label}", f"mom4V{label}.pt()")
    df = df.Define(f"massV{label}", f"mom4V{label}.mass()")
    df = df.Define(f"ptqV{label}", f"mom4V{label}.pt()/mom4V{label}.mass()")
    df = df.Define(f"yV{label}", f"mom4V{label}.Rapidity()")
    df = df.Define(f"phiV{label}", f"mom4V{label}.Phi()")
    df = df.Define(f"absYV{label}", f"std::fabs(yV{label})")
    df = df.Define(f"chargeV{label}", "chargeVgen")
    df = df.Define(
        f"csSineCosThetaPhi{label}",
        f"wrem::csSineCosThetaPhiTransported(genlanti, genl, mom4V{label})",
    )

    return df


def define_postfsr_vars(df, mode=None):
    if "postfsrLeptons" in df.GetColumnNames():
        logger.debug("PostFSR leptons are already defined, do nothing here.")
        return df

    logger.info(f"Defining postFSR variables for mode '{mode}'")

    # status flags in NanoAOD: https://cms-nanoaod-integration.web.cern.ch/autoDoc/NanoAODv9/2016ULpostVFP/doc_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1.html
    # post fsr definition: is stable && (isPrompt or isDirectPromptTauDecayProduct) && is lepton
    df = df.Define(
        "postfsrLeptons",
        "GenPart_status == 1 && (GenPart_statusFlags & 1 || GenPart_statusFlags & (1 << 5)) && abs(GenPart_pdgId) >= 11 && abs(GenPart_pdgId) <= 16",
    )
    df = df.Define("postfsrElectrons", "postfsrLeptons && abs(GenPart_pdgId) == 11")
    df = df.Define("postfsrMuons", "postfsrLeptons && abs(GenPart_pdgId) == 13")
    df = df.Define(
        "postfsrNeutrinos",
        "postfsrLeptons && (abs(GenPart_pdgId)==12 || abs(GenPart_pdgId)==14 || abs(GenPart_pdgId)==16)",
    )

    df = df.Define(
        "postfsrNeutrinos_mom4",
        """wrem::Sum4Vec(
            GenPart_pt[postfsrNeutrinos], GenPart_eta[postfsrNeutrinos], GenPart_phi[postfsrNeutrinos])""",
    )

    if mode is not None:
        # defition of more complex postfsr object
        # use fiducial gen met, see: https://twiki.cern.ch/twiki/bin/viewauth/CMS/ParticleLevelProducer
        if mode[0] == "z":
            # find the leading charged lepton and antilepton idx
            df = df.Define(
                "postfsrLep",
                "postfsrLeptons && (GenPart_pdgId==11 || GenPart_pdgId==13)",
            )
            df = df.Define(
                "postfsrAntiLep",
                "postfsrLeptons && (GenPart_pdgId==-11 || GenPart_pdgId==-13)",
            )

            df = df.Define(
                "postfsrLep_idx", "ROOT::VecOps::ArgMax(GenPart_pt[postfsrLep])"
            )
            df = df.Define(
                "postfsrAntiLep_idx", "ROOT::VecOps::ArgMax(GenPart_pt[postfsrAntiLep])"
            )

            df = df.Define(
                "postfsrLep_pt",
                "isEvenEvent ? static_cast<double>(GenPart_pt[postfsrLep][postfsrLep_idx]) : static_cast<double>(GenPart_pt[postfsrAntiLep][postfsrAntiLep_idx])",
            )
            df = df.Define(
                "postfsrLep_eta",
                "isEvenEvent ? GenPart_eta[postfsrLep][postfsrLep_idx] : GenPart_eta[postfsrAntiLep][postfsrAntiLep_idx]",
            )
            df = df.Define(
                "postfsrLep_phi",
                "isEvenEvent ? GenPart_phi[postfsrLep][postfsrLep_idx] : GenPart_phi[postfsrAntiLep][postfsrAntiLep_idx]",
            )
            df = df.Define(
                "postfsrLep_mass",
                "isEvenEvent ? wrem::get_pdgid_mass(GenPart_pdgId[postfsrLep][postfsrLep_idx]) : wrem::get_pdgid_mass(GenPart_pdgId[postfsrAntiLep][postfsrAntiLep_idx])",
            )
            df = df.Define("postfsrLep_charge", "isEvenEvent ? 1 : -1")

            df = df.Define(
                "postfsrOtherLep_pt",
                "isEvenEvent ? GenPart_pt[postfsrAntiLep][postfsrAntiLep_idx] : GenPart_pt[postfsrLep][postfsrLep_idx]",
            )
            df = df.Define(
                "postfsrOtherLep_eta",
                "isEvenEvent ? GenPart_eta[postfsrAntiLep][postfsrAntiLep_idx] : GenPart_eta[postfsrLep][postfsrLep_idx]",
            )
            df = df.Define(
                "postfsrOtherLep_phi",
                "isEvenEvent ? GenPart_phi[postfsrAntiLep][postfsrAntiLep_idx] : GenPart_phi[postfsrLep][postfsrLep_idx]",
            )
            df = df.Define(
                "postfsrOtherLep_mass",
                "isEvenEvent ? wrem::get_pdgid_mass(GenPart_pdgId[postfsrAntiLep][postfsrAntiLep_idx]) : wrem::get_pdgid_mass(GenPart_pdgId[postfsrLep][postfsrLep_idx])",
            )

            df = df.Define(
                "postfsrOtherLep_absEta",
                "static_cast<double>(std::fabs(postfsrOtherLep_eta))",
            )
        else:
            # find the leading charged lepton or antilepton idx
            df = df.Define(
                "postfsrLep",
                "postfsrLeptons && (abs(GenPart_pdgId)==11 || abs(GenPart_pdgId)==13)",
            )
            df = df.Define(
                "postfsrLep_idx", "ROOT::VecOps::ArgMax(GenPart_pt[postfsrLep])"
            )

            df = df.Define(
                "postfsrLep_pt",
                "static_cast<double>(GenPart_pt[postfsrLep][postfsrLep_idx])",
            )
            df = df.Define("postfsrLep_eta", "GenPart_eta[postfsrLep][postfsrLep_idx]")
            df = df.Define("postfsrLep_phi", "GenPart_phi[postfsrLep][postfsrLep_idx]")
            df = df.Define(
                "postfsrLep_mass",
                "wrem::get_pdgid_mass(GenPart_pdgId[postfsrLep][postfsrLep_idx])",
            )
            df = df.Define(
                "postfsrLep_charge",
                "GenPart_pdgId[postfsrLep][postfsrLep_idx] > 0 ? -1 : 1",
            )

        df = df.Define(
            "postfsrLep_absEta", "static_cast<double>(std::fabs(postfsrLep_eta))"
        )

        if mode[0] == "w" or "wlike" in mode:
            if "wlike" in mode:
                # for wlike selection
                df = df.Define(
                    "postfsrMET_wlike",
                    "wrem::get_met_wlike(postfsrOtherLep_pt, postfsrOtherLep_phi, MET_fiducialGenPt, MET_fiducialGenPhi)",
                )
                df = df.Define("postfsrMET_pt", "postfsrMET_wlike.Mod()")
                df = df.Define("postfsrMET_phi", "postfsrMET_wlike.Phi()")
            else:
                df = df.Alias("postfsrMET_pt", "MET_fiducialGenPt")
                df = df.Alias("postfsrMET_phi", "MET_fiducialGenPhi")
                df = df.Define(
                    "postfsrPTV",
                    "wrem::pt_2(postfsrLep_pt, postfsrLep_phi, postfsrMET_pt, postfsrMET_phi)",
                )

            df = df.Define(
                "postfsrMT",
                "wrem::mt_2(postfsrLep_pt, postfsrLep_phi, postfsrMET_pt, postfsrMET_phi)",
            )
            df = df.Define(
                "postfsrDeltaPhiMuonMet",
                "std::fabs(wrem::deltaPhi(postfsrLep_phi, postfsrMET_phi))",
            )

        # definition of boson kinematics
        if mode[0] == "z":
            # four vectors
            df = df.Define(
                "postfsrLep_mom4",
                "ROOT::Math::PtEtaPhiMVector(postfsrLep_pt, postfsrLep_eta, postfsrLep_phi, postfsrLep_mass)",
            )
            df = df.Define(
                "postfsrAntiLep_mom4",
                "ROOT::Math::PtEtaPhiMVector(postfsrOtherLep_pt, postfsrOtherLep_eta, postfsrOtherLep_phi, postfsrOtherLep_mass)",
            )

            df = df.Define("postfsrGenV_mom4", "postfsrLep_mom4 + postfsrAntiLep_mom4")
            df = df.Define("postfsrMV", "postfsrGenV_mom4.mass()")
            df = df.Define("postfsrYV", "postfsrGenV_mom4.Rapidity()")
            df = df.Define("postfsrabsYV", "std::fabs(postfsrYV)")
            df = df.Define("postfsrPTV", "postfsrGenV_mom4.pt()")
            df = df.DefinePerSample("postfsrChargeV", "0")
        else:
            df = df.Define("postfsrChargeV", "postfsrLep_charge")

    return df


def define_ew_vars(df):
    if "ewLeptons" in df.GetColumnNames():
        logger.debug("EW leptons are already defined, do nothing here.")
        return df

    df = df.Define(
        "ewLeptons",
        "wrem::ewLeptons(GenPart_status, GenPart_statusFlags, GenPart_pdgId, GenPart_pt, GenPart_eta, GenPart_phi)",
    )
    df = df.Define(
        "ewPhotons",
        "wrem::ewPhotons(GenPart_status, GenPart_statusFlags, GenPart_pdgId, GenPart_pt, GenPart_eta, GenPart_phi)",
    )
    df = df.Define("ewGenV", "wrem::ewGenVPhos(ewLeptons, ewPhotons)")
    df = df.Define("ewMll", "(ewLeptons[0]+ewLeptons[1]).mass()")
    df = df.Define("ewMlly", "ewGenV.mass()")
    df = df.Define("ewLogDeltaM", "log10(ewMlly-ewMll)")

    df = df.Define("ewPTll", "(ewLeptons[0]+ewLeptons[1]).pt()")
    df = df.Define("ewPTlly", "ewGenV.pt()")
    df = df.Define("ewYll", "(ewLeptons[0]+ewLeptons[1]).Rapidity()")
    df = df.Define("ewAbsYll", "std::fabs(ewYll)")
    df = df.Define("ewYlly", "ewGenV.Rapidity()")

    return df
