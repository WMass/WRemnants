#ifndef WREMNANTS_LOWPU_EFFICIENCIES_H
#define WREMNANTS_LOWPU_EFFICIENCIES_H

#include "TFile.h"
#include "defines.hpp"
#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>

namespace wrem {

// Muon SF
TFile *_leptonSF = new TFile("wremnants-data/data/lowPU/efficiencies/"
                             "2021-10-22_allSFs_nodz_dxybs_lowPU.root",
                             "READ");
TH2 *lepSF_mu_iso = (TH2F *)(_leptonSF->Get(
    "mu_SF2D_nominal_iso_lowPU_both")); // already the ratio DATA/MC
TH2 *lepSF_mu_idip = (TH2F *)(_leptonSF->Get(
    "mu_SF2D_nominal_idip_lowPU_both")); // already the ratio DATA/MC
TH2 *lepSF_mu_trg_DATA = (TH2F *)(_leptonSF->Get(
    "mu_effData_trigger_lowPU_both")); // separate DATA/MC
TH2 *lepSF_mu_trg_MC =
    (TH2F *)(_leptonSF->Get("mu_effMC_trigger_lowPU_both")); // separate DATA/MC
TH2 *lepSF_mu_trg_plus_DATA =
    (TH2F *)(_leptonSF->Get("mu_effData_trigger_lowPU_plus"));
TH2 *lepSF_mu_trg_plus_MC =
    (TH2F *)(_leptonSF->Get("mu_effMC_trigger_lowPU_plus"));
TH2 *lepSF_mu_trg_minus_DATA =
    (TH2F *)(_leptonSF->Get("mu_effData_trigger_lowPU_minus"));
TH2 *lepSF_mu_trg_minus_MC =
    (TH2F *)(_leptonSF->Get("mu_effMC_trigger_lowPU_minus"));

// Variations (for each SF, the nominal is varied with alternative data and MC
// shapes)
TH2 *lepSF_mu_iso_altDATA =
    (TH2F *)(_leptonSF->Get("mu_SF2D_dataAltSig_iso_lowPU_both"));
TH2 *lepSF_mu_iso_altMC =
    (TH2F *)(_leptonSF->Get("mu_SF2D_MCAltSig_iso_lowPU_both"));
TH2 *lepSF_mu_idip_altDATA =
    (TH2F *)(_leptonSF->Get("mu_SF2D_dataAltSig_idip_lowPU_both"));
TH2 *lepSF_mu_idip_altMC =
    (TH2F *)(_leptonSF->Get("mu_SF2D_MCAltSig_idip_lowPU_both"));

TH2 *lepSF_mu_trg_DATA_alt =
    (TH2F *)(_leptonSF->Get("mu_effData_altSig_trigger_lowPU_both"));
TH2 *lepSF_mu_trg_MC_alt =
    (TH2F *)(_leptonSF->Get("mu_effMC_altSig_trigger_lowPU_both"));
TH2 *lepSF_mu_trg_plus_DATA_alt =
    (TH2F *)(_leptonSF->Get("mu_effData_altSig_trigger_lowPU_plus"));
TH2 *lepSF_mu_trg_plus_MC_alt =
    (TH2F *)(_leptonSF->Get("mu_effMC_altSig_trigger_lowPU_plus"));
TH2 *lepSF_mu_trg_minus_DATA_alt =
    (TH2F *)(_leptonSF->Get("mu_effData_altSig_trigger_lowPU_minus"));
TH2 *lepSF_mu_trg_minus_MC_alt =
    (TH2F *)(_leptonSF->Get("mu_effMC_altSig_trigger_lowPU_minus"));

// Electron SF
TH2 *lepSF_el_idiso =
    (TH2F *)(_leptonSF->Get("el_SF2D_nominal_idiso_lowPU_both"));
TH2 *lepSF_el_trg_DATA =
    (TH2F *)(_leptonSF->Get("el_effData_trigger_lowPU_both"));
TH2 *lepSF_el_trg_MC = (TH2F *)(_leptonSF->Get("el_effMC_trigger_lowPU_both"));
TH2 *lepSF_el_trg_plus_DATA =
    (TH2F *)(_leptonSF->Get("el_effData_trigger_lowPU_plus"));
TH2 *lepSF_el_trg_plus_MC =
    (TH2F *)(_leptonSF->Get("el_effMC_trigger_lowPU_plus"));
TH2 *lepSF_el_trg_minus_DATA =
    (TH2F *)(_leptonSF->Get("el_effData_trigger_lowPU_minus"));
TH2 *lepSF_el_trg_minus_MC =
    (TH2F *)(_leptonSF->Get("el_effMC_trigger_lowPU_minus"));

// Variations (for each SF, the nominal is varied with alternative data and MC
// shapes)
TH2 *lepSF_el_idiso_altDATA =
    (TH2F *)(_leptonSF->Get("el_SF2D_dataAltSig_idiso_lowPU_both"));
TH2 *lepSF_el_idiso_altMC =
    (TH2F *)(_leptonSF->Get("el_SF2D_MCAltSig_idiso_lowPU_both"));
// TH2 * lepSF_el_idiso_alt =
// (TH2F*)(_leptonSF->Get("el_SF2D_dataMCAltSig_idiso_lowPU_both"));

TH2 *lepSF_el_trg_DATA_alt =
    (TH2F *)(_leptonSF->Get("el_effData_altSig_trigger_lowPU_both"));
TH2 *lepSF_el_trg_MC_alt =
    (TH2F *)(_leptonSF->Get("el_effMC_altSig_trigger_lowPU_both"));
TH2 *lepSF_el_trg_plus_DATA_alt =
    (TH2F *)(_leptonSF->Get("el_effData_altSig_trigger_lowPU_plus"));
TH2 *lepSF_el_trg_plus_MC_alt =
    (TH2F *)(_leptonSF->Get("el_effMC_altSig_trigger_lowPU_plus"));
TH2 *lepSF_el_trg_minus_DATA_alt =
    (TH2F *)(_leptonSF->Get("el_effData_altSig_trigger_lowPU_minus"));
TH2 *lepSF_el_trg_minus_MC_alt =
    (TH2F *)(_leptonSF->Get("el_effMC_altSig_trigger_lowPU_minus"));

double lepSF_HLT_q(Vec_f pt, Vec_f eta, Vec_i q, int flavorType,
                   int varType = 0, int dEta = 0, int dpT = 0) {

  // flavorType: 11 or 13
  // varType: 0=nominal, 1=stat DATA, 2=syst DATA, -1=stat MC, -2=syst MC

  TH2 *lepSF_trg_DATA;
  TH2 *lepSF_trg_MC;
  TH2 *lepSF_alt_DATA; // alternative correction
  TH2 *lepSF_alt_MC;   // alternative correction

  unsigned int size = pt.size();
  double corr = 1.0;

  double trgSF_DATA = 1.0;
  double trgSF_MC = 1.0;

  for (unsigned int i = 0; i < size; ++i) {

    double eta_ = eta.at(i);
    double pt_ = pt.at(i);
    double q_ = q.at(i);

    if (flavorType == 11) {
      if (q_ > 0) {
        lepSF_trg_DATA = lepSF_el_trg_plus_DATA;
        lepSF_trg_MC = lepSF_el_trg_plus_MC;
        lepSF_alt_DATA = lepSF_el_trg_plus_DATA_alt;
        lepSF_alt_MC = lepSF_el_trg_plus_MC_alt;
      } else {
        lepSF_trg_DATA = lepSF_el_trg_minus_DATA;
        lepSF_trg_MC = lepSF_el_trg_minus_MC;
        lepSF_alt_DATA = lepSF_el_trg_minus_DATA_alt;
        lepSF_alt_MC = lepSF_el_trg_minus_MC_alt;
      }
    } else if (flavorType == 13) {
      if (q_ > 0) {
        lepSF_trg_DATA = lepSF_mu_trg_plus_DATA;
        lepSF_trg_MC = lepSF_mu_trg_plus_MC;
        lepSF_alt_DATA = lepSF_mu_trg_plus_DATA_alt;
        lepSF_alt_MC = lepSF_mu_trg_plus_MC_alt;
      } else {
        lepSF_trg_DATA = lepSF_mu_trg_minus_DATA;
        lepSF_trg_MC = lepSF_mu_trg_minus_MC;
        lepSF_alt_DATA = lepSF_mu_trg_minus_DATA_alt;
        lepSF_alt_MC = lepSF_mu_trg_minus_MC_alt;
      }
    } else
      return 1.0;

    int etabin =
        std::max(1, std::min(lepSF_trg_DATA->GetNbinsX(),
                             lepSF_trg_DATA->GetXaxis()->FindBin(eta_)));
    int ptbin = std::max(1, std::min(lepSF_trg_DATA->GetNbinsY(),
                                     lepSF_trg_DATA->GetYaxis()->FindBin(pt_)));

    double SF_DATA;
    double SF_MC;
    if (varType == 1 and etabin == dEta and ptbin == dpT) { // STAT DATA
      SF_DATA = lepSF_trg_DATA->GetBinContent(etabin, ptbin) +
                lepSF_trg_DATA->GetBinError(etabin, ptbin);
      SF_MC = lepSF_trg_MC->GetBinContent(etabin, ptbin);
    } else if (varType == -1 and etabin == dEta and ptbin == dpT) { // STAT MC

      SF_DATA = lepSF_trg_DATA->GetBinContent(etabin, ptbin);
      SF_MC = lepSF_trg_MC->GetBinContent(etabin, ptbin) +
              lepSF_trg_MC->GetBinError(etabin, ptbin);
    } else if (varType == 2 and etabin == dEta and ptbin == dpT) { // SYST DATA

      SF_DATA = lepSF_alt_DATA->GetBinContent(etabin, ptbin);
      SF_MC = lepSF_trg_MC->GetBinContent(etabin, ptbin);
    } else if (varType == -2 and etabin == dEta and ptbin == dpT) { // SYST MC

      SF_DATA = lepSF_trg_DATA->GetBinContent(etabin, ptbin);
      SF_MC = lepSF_alt_MC->GetBinContent(etabin, ptbin);
    } else { // nominal

      SF_DATA = lepSF_trg_DATA->GetBinContent(etabin, ptbin);
      SF_MC = lepSF_trg_MC->GetBinContent(etabin, ptbin);
    }

    trgSF_DATA *= 1. - SF_DATA;
    trgSF_MC *= 1. - SF_MC;
  }

  corr *= (1. - trgSF_DATA) / (1. - trgSF_MC);
  return corr;
}

double lepSF_HLT(Vec_f pt, Vec_f eta, Vec_i q, int flavorType, int varType = 0,
                 int dEta = 0, int dpT = 0) {

  // flavorType: 11 or 13
  // varType: 0=nominal, 1=stat DATA, 2=syst DATA, -1=stat MC, -2=syst MC

  TH2 *lepSF_trg_DATA;
  TH2 *lepSF_trg_MC;
  TH2 *lepSF_alt_DATA; // alternative correction
  TH2 *lepSF_alt_MC;   // alternative correction

  if (flavorType == 11) {

    lepSF_trg_DATA = lepSF_el_trg_DATA;
    lepSF_trg_MC = lepSF_el_trg_MC;
    lepSF_alt_DATA = lepSF_el_trg_DATA_alt;
    lepSF_alt_MC = lepSF_el_trg_MC_alt;
  } else if (flavorType == 13) {

    lepSF_trg_DATA = lepSF_mu_trg_DATA;
    lepSF_trg_MC = lepSF_mu_trg_MC;
    lepSF_alt_DATA = lepSF_mu_trg_DATA_alt;
    lepSF_alt_MC = lepSF_mu_trg_MC_alt;
  } else
    return 1.0;

  unsigned int size = pt.size();
  double corr = 1.0;

  double trgSF_DATA = 1.0;
  double trgSF_MC = 1.0;

  for (unsigned int i = 0; i < size; ++i) {

    double eta_ = eta.at(i);
    double pt_ = pt.at(i);

    int etabin =
        std::max(1, std::min(lepSF_trg_DATA->GetNbinsX(),
                             lepSF_trg_DATA->GetXaxis()->FindBin(eta_)));
    int ptbin = std::max(1, std::min(lepSF_trg_DATA->GetNbinsY(),
                                     lepSF_trg_DATA->GetYaxis()->FindBin(pt_)));

    double SF_DATA;
    double SF_MC;
    if (varType == 1 and etabin == dEta and ptbin == dpT) { // STAT DATA
      SF_DATA = lepSF_trg_DATA->GetBinContent(etabin, ptbin) +
                lepSF_trg_DATA->GetBinError(etabin, ptbin);
      SF_MC = lepSF_trg_MC->GetBinContent(etabin, ptbin);
    } else if (varType == -1 and etabin == dEta and ptbin == dpT) { // STAT MC

      SF_DATA = lepSF_trg_DATA->GetBinContent(etabin, ptbin);
      SF_MC = lepSF_trg_MC->GetBinContent(etabin, ptbin) +
              lepSF_trg_MC->GetBinError(etabin, ptbin);
    } else if (varType == 2 and etabin == dEta and ptbin == dpT) { // SYST DATA

      SF_DATA = lepSF_alt_DATA->GetBinContent(etabin, ptbin);
      SF_MC = lepSF_trg_MC->GetBinContent(etabin, ptbin);
    } else if (varType == -2 and etabin == dEta and ptbin == dpT) { // SYST MC

      SF_DATA = lepSF_trg_DATA->GetBinContent(etabin, ptbin);
      SF_MC = lepSF_alt_MC->GetBinContent(etabin, ptbin);
    } else { // nominal

      SF_DATA = lepSF_trg_DATA->GetBinContent(etabin, ptbin);
      SF_MC = lepSF_trg_MC->GetBinContent(etabin, ptbin);
    }

    trgSF_DATA *= 1. - SF_DATA;
    trgSF_MC *= 1. - SF_MC;
  }

  corr *= (1. - trgSF_DATA) / (1. - trgSF_MC);
  return corr;
}

double lepSF(Vec_f pt, Vec_f eta, Vec_i q, int corrType, int varType = 0,
             int dEta = 0, int dpT = 0) {

  TH2 *lepSF;
  TH2 *lepSF_alt_data; // alternative correction
  TH2 *lepSF_alt_mc;   // alternative correction

  if (corrType == 1) { // muon ISO
    lepSF = lepSF_mu_iso;
    lepSF_alt_data = lepSF_mu_iso_altDATA;
    lepSF_alt_mc = lepSF_mu_iso_altMC;
  } else if (corrType == 2) { // muon ID/IP
    lepSF = lepSF_mu_idip;
    lepSF_alt_data = lepSF_mu_idip_altDATA;
    lepSF_alt_mc = lepSF_mu_idip_altMC;
  } else if (corrType == 3) { // electron ID/ISO
    lepSF = lepSF_el_idiso;
    lepSF_alt_data = lepSF_el_idiso_altDATA;
    lepSF_alt_mc = lepSF_el_idiso_altMC;
  } else
    return 1.0;

  unsigned int size = pt.size();
  double corr = 1.0;

  for (unsigned int i = 0; i < size; ++i) {

    double eta_ = eta.at(i);
    double pt_ = pt.at(i);

    int etabin = std::max(
        1, std::min(lepSF->GetNbinsX(), lepSF->GetXaxis()->FindBin(eta_)));
    int ptbin = std::max(
        1, std::min(lepSF->GetNbinsY(), lepSF->GetYaxis()->FindBin(pt_)));

    double SF;
    if (varType == 1 and etabin == dEta and
        ptbin == dpT) { // STAT DATA == STAT MC

      SF = lepSF->GetBinContent(etabin, ptbin) +
           lepSF->GetBinError(etabin, ptbin);
    } else if (varType == -1 and etabin == dEta and
               ptbin == dpT) { // STAT MC == STAT DATA

      SF = lepSF->GetBinContent(etabin, ptbin) +
           lepSF->GetBinError(etabin, ptbin);
    } else if (varType == 2 and etabin == dEta and ptbin == dpT) { // SYST DATA

      SF = lepSF_alt_data->GetBinContent(etabin, ptbin);
    } else if (varType == -2 and etabin == dEta and ptbin == dpT) { // SYST MC

      SF = lepSF_alt_mc->GetBinContent(etabin, ptbin);
    } else { // nominal

      SF = lepSF->GetBinContent(etabin, ptbin);
    }

    corr *= SF;
  }

  return corr;
}

Vec_f lepSF_HLT_var_mu(int varType, Vec_f pt, Vec_f eta, Vec_i q) {

  // Varitation of the muon HLT efficiencies (12 eta, 10 pT -> 120 in total)
  // varType: 1=stat DATA, 2=syst DATA, -1=stat MC, -2=syst MC

  Vec_f res(120, 0);
  double nom = lepSF_HLT_q(pt, eta, q, 13, 0, 0, 0); // nominal

  int c = 0;
  for (int iEta = 1; iEta <= 12; iEta++) {

    for (int iPt = 1; iPt <= 10; iPt++) {

      // if(iEta == 1 and iPt == 1) cout << lepSF_HLT(pt, eta, q, 13, varType,
      // iEta, iPt)/nom << endl;
      res[c] = lepSF_HLT(pt, eta, q, 13, varType, iEta, iPt) / nom;
      c++;
    }
  }

  return res;
}

Vec_f lepSF_ISO_var_mu(int varType, Vec_f pt, Vec_f eta, Vec_i q) {

  // Varitation of the muon ISO efficiencies (12 eta, 3 pT -> 36 in total)
  // varType: 1=stat DATA, 2=syst DATA, -1=stat MC, -2=syst MC

  Vec_f res(36, 0);
  double nom = lepSF(pt, eta, q, 1, 0, 0, 0); // nominal

  int c = 0;
  for (int iEta = 1; iEta <= 12; iEta++) {

    for (int iPt = 1; iPt <= 3; iPt++) {

      res[c] = lepSF(pt, eta, q, 1, varType, iEta, iPt) / nom;
      c++;
    }
  }

  return res;
}

Vec_f lepSF_IDIP_var_mu(int varType, Vec_f pt, Vec_f eta, Vec_i q) {

  // Varitation of the IDIP efficiencies (12 eta, 3 pT -> 36 in total)
  // varType: 1=stat DATA, 2=syst DATA, -1=stat MC, -2=syst MC

  Vec_f res(36, 0);
  double nom = lepSF(pt, eta, q, 2, 0, 0, 0); // nominal

  int c = 0;
  for (int iEta = 1; iEta <= 12; iEta++) {

    for (int iPt = 1; iPt <= 3; iPt++) {

      res[c] = lepSF(pt, eta, q, 2, varType, iEta, iPt) / nom;
      c++;
    }
  }

  return res;
}

// OLDER FUNCTIONS

double applyElectronSFSingleLeg(Vec_f pt, Vec_f eta, Vec_i q) {

  unsigned int size = pt.size();
  double corr = 1.0;

  double trgSF_DATA = 1.0;
  double trgSF_MC = 1.0;

  bool trg_applied = false;
  for (unsigned int i = 0; i < size; ++i) {

    double eta_ = eta.at(i);
    double pt_ = pt.at(i);
    int q_ = q.at(i);

    int etabin_idiso =
        std::max(1, std::min(lepSF_el_idiso->GetNbinsX(),
                             lepSF_el_idiso->GetXaxis()->FindBin(eta_)));
    int ptbin_idiso =
        std::max(1, std::min(lepSF_el_idiso->GetNbinsY(),
                             lepSF_el_idiso->GetYaxis()->FindBin(pt_)));

    int etabin_HLT =
        std::max(1, std::min(lepSF_el_trg_DATA->GetNbinsX(),
                             lepSF_el_trg_DATA->GetXaxis()->FindBin(eta_)));
    int ptbin_HLT =
        std::max(1, std::min(lepSF_el_trg_DATA->GetNbinsY(),
                             lepSF_el_trg_DATA->GetYaxis()->FindBin(pt_)));

    corr *= lepSF_el_idiso->GetBinContent(etabin_idiso, ptbin_idiso);

    if (q_ < 0) {

      trgSF_DATA *=
          1. - lepSF_el_trg_DATA->GetBinContent(etabin_HLT, ptbin_HLT);
      trgSF_MC *= 1. - lepSF_el_trg_MC->GetBinContent(etabin_HLT, ptbin_HLT);
    } else {

      // trgSF_DATA *= 1. - lepSF_el_trg_DATA->GetBinContent(etabin_HLT,
      // ptbin_HLT); trgSF_MC *= 1. - lepSF_el_trg_MC->GetBinContent(etabin_HLT,
      // ptbin_HLT);
    }
  }
  // cout << (1. - trgSF_DATA) / (1. - trgSF_MC) << endl;
  corr *= (1. - trgSF_DATA) / (1. - trgSF_MC);

  return corr;
}

double applyElectronSF(Vec_f pt, Vec_f eta, Vec_i q) {

  unsigned int size = pt.size();
  double corr = 1.0;

  double trgSF_DATA = 1.0;
  double trgSF_MC = 1.0;

  bool trg_applied = false;
  for (unsigned int i = 0; i < size; ++i) {

    double eta_ = eta.at(i);
    double pt_ = pt.at(i);
    int q_ = q.at(i);

    int etabin_idiso =
        std::max(1, std::min(lepSF_el_idiso->GetNbinsX(),
                             lepSF_el_idiso->GetXaxis()->FindBin(eta_)));
    int ptbin_idiso =
        std::max(1, std::min(lepSF_el_idiso->GetNbinsY(),
                             lepSF_el_idiso->GetYaxis()->FindBin(pt_)));

    int etabin_HLT =
        std::max(1, std::min(lepSF_el_trg_DATA->GetNbinsX(),
                             lepSF_el_trg_DATA->GetXaxis()->FindBin(eta_)));
    int ptbin_HLT =
        std::max(1, std::min(lepSF_el_trg_DATA->GetNbinsY(),
                             lepSF_el_trg_DATA->GetYaxis()->FindBin(pt_)));

    corr *= lepSF_el_idiso->GetBinContent(etabin_idiso, ptbin_idiso);

    if (q_ < 0) {

      trgSF_DATA *=
          1. - lepSF_el_trg_DATA->GetBinContent(etabin_HLT, ptbin_HLT);
      trgSF_MC *= 1. - lepSF_el_trg_MC->GetBinContent(etabin_HLT, ptbin_HLT);
    } else {

      trgSF_DATA *=
          1. - lepSF_el_trg_DATA->GetBinContent(etabin_HLT, ptbin_HLT);
      trgSF_MC *= 1. - lepSF_el_trg_MC->GetBinContent(etabin_HLT, ptbin_HLT);
    }
  }
  corr *= (1. - trgSF_DATA) / (1. - trgSF_MC);

  return corr;
}

double lepSF_el_HLT(Vec_f pt, Vec_f eta, Vec_i q, int dEta = 0, int dpT = 0) {

  unsigned int size = pt.size();
  double corr = 1.0;

  double trgSF_DATA = 1.0;
  double trgSF_MC = 1.0;

  for (unsigned int i = 0; i < size; ++i) {

    double eta_ = eta.at(i);
    double pt_ = pt.at(i);

    int etabin_HLT =
        std::max(1, std::min(lepSF_el_trg_DATA->GetNbinsX(),
                             lepSF_el_trg_DATA->GetXaxis()->FindBin(eta_)));
    int ptbin_HLT =
        std::max(1, std::min(lepSF_el_trg_DATA->GetNbinsY(),
                             lepSF_el_trg_DATA->GetYaxis()->FindBin(pt_)));

    double nom_DATA = lepSF_el_trg_DATA->GetBinContent(etabin_HLT, ptbin_HLT);
    double var_DATA = lepSF_el_trg_DATA->GetBinError(etabin_HLT, ptbin_HLT);
    double nom_MC = lepSF_el_trg_MC->GetBinContent(etabin_HLT, ptbin_HLT);
    double var_MC = lepSF_el_trg_MC->GetBinError(etabin_HLT, ptbin_HLT);

    if (dEta > 0 and dpT > 0 and etabin_HLT == dEta and
        ptbin_HLT == dpT) { // up

      trgSF_DATA *= 1. - (nom_DATA + var_DATA);
      trgSF_MC *= 1. - (nom_MC + var_MC);

    } else if (dEta < 0 and dpT < 0 and etabin_HLT == -dEta and
               ptbin_HLT == -dpT) { // down

      trgSF_DATA *= 1. - (nom_DATA - var_DATA);
      trgSF_MC *= 1. - (nom_MC - var_MC);
    } else { // nominal

      trgSF_DATA *= 1. - nom_DATA;
      trgSF_MC *= 1. - nom_MC;
    }
  }

  corr *= (1. - trgSF_DATA) / (1. - trgSF_MC);

  return corr;
}

double lepSF_el_IDISO(Vec_f pt, Vec_f eta, Vec_i q, int dEta = 0, int dpT = 0) {

  unsigned int size = pt.size();
  double corr = 1.0;

  for (unsigned int i = 0; i < size; ++i) {

    double eta_ = eta.at(i);
    double pt_ = pt.at(i);

    int etabin_idiso =
        std::max(1, std::min(lepSF_el_idiso->GetNbinsX(),
                             lepSF_el_idiso->GetXaxis()->FindBin(eta_)));
    int ptbin_idiso =
        std::max(1, std::min(lepSF_el_idiso->GetNbinsY(),
                             lepSF_el_idiso->GetYaxis()->FindBin(pt_)));

    double idiso_nom = lepSF_el_idiso->GetBinContent(etabin_idiso, ptbin_idiso);
    double idiso_var = lepSF_el_idiso->GetBinError(etabin_idiso, ptbin_idiso);

    if (dEta > 0 and dpT > 0 and etabin_idiso == dEta and ptbin_idiso == dpT)
      corr *= (idiso_nom + idiso_var); // up
    else if (dEta < 0 and dpT < 0 and etabin_idiso == -dEta and
             ptbin_idiso == -dpT)
      corr *= (idiso_nom - idiso_var); // down
    else
      corr *= idiso_nom; // nominal
  }

  return corr;
}

Vec_f lepSF_el_HLT_syst(Vec_f pt, Vec_f eta, Vec_i q) {

  // 12 eta, 10 pT,
  // 2*120 in total

  Vec_f res(2 * 120, 0);
  double nom = lepSF_el_HLT(pt, eta, q, 0, 0); // nominal

  int c = 0;
  for (int iEta = 1; iEta <= 12; iEta++) {

    for (int iPt = 1; iPt <= 10; iPt++) {

      res[c] = lepSF_el_HLT(pt, eta, q, iEta, iPt) / nom; // up
      c++;

      res[c] = lepSF_el_HLT(pt, eta, q, -iEta, -iPt) / nom; // down
      c++;
    }
  }

  return res;
}

Vec_f lepSF_el_IDISO_syst(Vec_f pt, Vec_f eta, Vec_i q) {

  // 12 eta, 3 pT,
  // 2*36 in total

  Vec_f res(2 * 36, 0);
  double nom = lepSF_el_IDISO(pt, eta, q, 0, 0); // nominal

  int c = 0;
  for (int iEta = 1; iEta <= 12; iEta++) {

    for (int iPt = 1; iPt <= 3; iPt++) {

      res[c] = lepSF_el_IDISO(pt, eta, q, iEta, iPt) / nom;
      ; // up
      c++;

      res[c] = lepSF_el_IDISO(pt, eta, q, -iEta, -iPt) / nom;
      ; // down
      c++;
    }
  }

  return res;
}

double lepSF_mu_HLT(Vec_f pt, Vec_f eta, Vec_i q, int varType = 0, int dEta = 0,
                    int dpT = 0) {

  // varType: 0=nominal, 1=stat DATA, 2=syst DATA, -1=stat MC, -2=syst MC

  unsigned int size = pt.size();
  double corr = 1.0;

  double trgSF_DATA = 1.0;
  double trgSF_MC = 1.0;

  for (unsigned int i = 0; i < size; ++i) {

    double eta_ = eta.at(i);
    double pt_ = pt.at(i);

    int etabin_HLT =
        std::max(1, std::min(lepSF_mu_trg_DATA->GetNbinsX(),
                             lepSF_mu_trg_DATA->GetXaxis()->FindBin(eta_)));
    int ptbin_HLT =
        std::max(1, std::min(lepSF_mu_trg_DATA->GetNbinsY(),
                             lepSF_mu_trg_DATA->GetYaxis()->FindBin(pt_)));

    double SF_DATA;
    double SF_MC;
    if (varType == 1 and etabin_HLT == dEta and ptbin_HLT == dpT) { // STAT DATA

      SF_DATA = lepSF_mu_trg_DATA->GetBinContent(etabin_HLT, ptbin_HLT) +
                lepSF_mu_trg_DATA->GetBinError(etabin_HLT, ptbin_HLT);
      SF_MC = lepSF_mu_trg_MC->GetBinContent(etabin_HLT, ptbin_HLT);
    } else if (varType == -1 and etabin_HLT == dEta and
               ptbin_HLT == dpT) { // STAT MC

      SF_DATA = lepSF_mu_trg_DATA->GetBinContent(etabin_HLT, ptbin_HLT);
      SF_MC = lepSF_mu_trg_MC->GetBinContent(etabin_HLT, ptbin_HLT) +
              lepSF_mu_trg_MC->GetBinError(etabin_HLT, ptbin_HLT);
    } else if (varType == 2 and etabin_HLT == dEta and
               ptbin_HLT == dpT) { // SYST DATA

      SF_DATA = lepSF_mu_trg_DATA_alt->GetBinContent(etabin_HLT, ptbin_HLT);
      SF_MC = lepSF_mu_trg_MC->GetBinContent(etabin_HLT, ptbin_HLT);
    } else if (varType == -2 and etabin_HLT == dEta and
               ptbin_HLT == dpT) { // SYST MC

      SF_DATA = lepSF_mu_trg_DATA->GetBinContent(etabin_HLT, ptbin_HLT);
      SF_MC = lepSF_mu_trg_MC_alt->GetBinContent(etabin_HLT, ptbin_HLT);
    } else { // nominal

      SF_DATA = lepSF_mu_trg_DATA->GetBinContent(etabin_HLT, ptbin_HLT);
      SF_MC = lepSF_mu_trg_MC->GetBinContent(etabin_HLT, ptbin_HLT);
    }

    trgSF_DATA *= 1. - SF_DATA;
    trgSF_MC *= 1. - SF_MC;

    // double nom_DATA = lepSF_mu_trg_DATA->GetBinContent(etabin_HLT,
    // ptbin_HLT); double var_DATA = lepSF_mu_trg_DATA->GetBinError(etabin_HLT,
    // ptbin_HLT); // STAT uncertainty on the efficiencies
    // double var_DATA = lepSF_mu_trg_DATA_alt->GetBinContent(etabin_HLT,
    // ptbin_HLT);

    // double var_MC = lepSF_mu_trg_MC->GetBinError(etabin_HLT, ptbin_HLT); //
    // STAT uncertainty on the efficiencies
    // double var_MC = lepSF_mu_trg_MC_alt->GetBinContent(etabin_HLT,
    // ptbin_HLT);
    /*
            if(dEta > 0 and dpT > 0 and etabin_HLT == dEta and ptbin_HLT == dpT)
       { // up

                    //trgSF_DATA *= 1. - (nom_DATA + var_DATA);
                    //trgSF_MC *= 1. - (nom_MC + var_MC);
        trgSF_DATA *= 1. -  var_DATA;
                    trgSF_MC *= 1. - var_MC;

            }
            else if(dEta < 0 and dpT < 0 and etabin_HLT == -dEta and ptbin_HLT
       == -dpT) { // down

                    trgSF_DATA *= 1. - (nom_DATA - var_DATA);
                    trgSF_MC *= 1. - (nom_MC - var_MC);
            }
            else { // nominal

                    trgSF_DATA *= 1. - nom_DATA;
                    trgSF_MC *= 1. - nom_MC;
            }
    */
  }

  corr *= (1. - trgSF_DATA) / (1. - trgSF_MC);

  return corr;
}

double lepSF_mu_ISO(Vec_f pt, Vec_f eta, Vec_i q, int varType = 0, int dEta = 0,
                    int dpT = 0) {

  unsigned int size = pt.size();
  double corr = 1.0;

  for (unsigned int i = 0; i < size; ++i) {

    double eta_ = eta.at(i);
    double pt_ = pt.at(i);

    int etabin_idiso =
        std::max(1, std::min(lepSF_mu_iso->GetNbinsX(),
                             lepSF_mu_iso->GetXaxis()->FindBin(eta_)));
    int ptbin_idiso =
        std::max(1, std::min(lepSF_mu_iso->GetNbinsY(),
                             lepSF_mu_iso->GetYaxis()->FindBin(pt_)));

    double SF;
    if (varType == 1 and etabin_idiso == dEta and
        ptbin_idiso == dpT) { // STAT DATA == STAT MC

      SF = lepSF_mu_iso->GetBinContent(etabin_idiso, ptbin_idiso) +
           lepSF_mu_iso->GetBinError(etabin_idiso, ptbin_idiso);
    } else if (varType == -1 and etabin_idiso == dEta and
               ptbin_idiso == dpT) { // STAT MC == STAT DATA

      SF = lepSF_mu_iso->GetBinContent(etabin_idiso, ptbin_idiso) +
           lepSF_mu_iso->GetBinError(etabin_idiso, ptbin_idiso);
    } else if (varType == 2 and etabin_idiso == dEta and
               ptbin_idiso == dpT) { // SYST DATA

      SF = lepSF_mu_iso_altDATA->GetBinContent(etabin_idiso, ptbin_idiso);
    } else if (varType == -2 and etabin_idiso == dEta and
               ptbin_idiso == dpT) { // SYST MC

      SF = lepSF_mu_iso_altMC->GetBinContent(etabin_idiso, ptbin_idiso);
    } else { // nominal

      SF = lepSF_mu_iso->GetBinContent(etabin_idiso, ptbin_idiso);
    }

    corr *= SF;

    /*
    double idiso_nom = lepSF_mu_iso->GetBinContent(etabin_idiso, ptbin_idiso);
            double idiso_var = lepSF_mu_iso->GetBinError(etabin_idiso,
    ptbin_idiso);

            if(dEta > 0 and dpT > 0 and etabin_idiso == dEta and ptbin_idiso ==
    dpT) corr *= (idiso_nom + idiso_var); // up else if(dEta < 0 and dpT < 0 and
    etabin_idiso == -dEta and ptbin_idiso == -dpT) corr *= (idiso_nom -
    idiso_var); // down else corr *= idiso_nom; // nominal
    */
  }

  return corr;
}

double lepSF_mu_IDIP(Vec_f pt, Vec_f eta, Vec_i q, int varType = 0,
                     int dEta = 0, int dpT = 0) {

  unsigned int size = pt.size();
  double corr = 1.0;

  for (unsigned int i = 0; i < size; ++i) {

    double eta_ = eta.at(i);
    double pt_ = pt.at(i);

    int etabin_idiso =
        std::max(1, std::min(lepSF_mu_idip->GetNbinsX(),
                             lepSF_mu_idip->GetXaxis()->FindBin(eta_)));
    int ptbin_idiso =
        std::max(1, std::min(lepSF_mu_idip->GetNbinsY(),
                             lepSF_mu_idip->GetYaxis()->FindBin(pt_)));

    double SF;
    if (varType == 1 and etabin_idiso == dEta and
        ptbin_idiso == dpT) { // STAT DATA == STAT MC

      SF = lepSF_mu_idip->GetBinContent(etabin_idiso, ptbin_idiso) +
           lepSF_mu_idip->GetBinError(etabin_idiso, ptbin_idiso);
    } else if (varType == -1 and etabin_idiso == dEta and
               ptbin_idiso == dpT) { // STAT MC == STAT DATA

      SF = lepSF_mu_idip->GetBinContent(etabin_idiso, ptbin_idiso) +
           lepSF_mu_idip->GetBinError(etabin_idiso, ptbin_idiso);
    } else if (varType == 2 and etabin_idiso == dEta and
               ptbin_idiso == dpT) { // SYST DATA

      SF = lepSF_mu_idip_altDATA->GetBinContent(etabin_idiso, ptbin_idiso);
    } else if (varType == -2 and etabin_idiso == dEta and
               ptbin_idiso == dpT) { // SYST MC

      SF = lepSF_mu_idip_altMC->GetBinContent(etabin_idiso, ptbin_idiso);
    } else { // nominal

      SF = lepSF_mu_idip->GetBinContent(etabin_idiso, ptbin_idiso);
    }

    corr *= SF;

    /*
    double idiso_nom = lepSF_mu_idip->GetBinContent(etabin_idiso, ptbin_idiso);
            double idiso_var = lepSF_mu_idip->GetBinError(etabin_idiso,
    ptbin_idiso);

            if(dEta > 0 and dpT > 0 and etabin_idiso == dEta and ptbin_idiso ==
    dpT) corr *= (idiso_nom + idiso_var); // up else if(dEta < 0 and dpT < 0 and
    etabin_idiso == -dEta and ptbin_idiso == -dpT) corr *= (idiso_nom -
    idiso_var); // down else corr *= idiso_nom; // nominal
    */
  }

  return corr;
}

Vec_f lepSF_mu_HLT_syst(int varType, Vec_f pt, Vec_f eta, Vec_i q) {

  // Varitation of the HLT efficiencies (2 eta, 10 pT -> 120 in total)
  // varType: 1=stat DATA, 2=syst DATA, -1=stat MC, -2=syst MC

  Vec_f res(120, 0);
  double nom = lepSF_mu_HLT(pt, eta, q, 0, 0, 0); // nominal

  int c = 0;
  for (int iEta = 1; iEta <= 12; iEta++) {

    for (int iPt = 1; iPt <= 10; iPt++) {

      // if(iEta == 1 and iPt == 1) cout << lepSF_mu_HLT(pt, eta, q, varType,
      // iEta, iPt)/nom << endl;
      res[c] = lepSF_mu_HLT(pt, eta, q, varType, iEta, iPt) / nom;
      c++;
    }
  }

  return res;
}

Vec_f lepSF_mu_ISO_syst(int varType, Vec_f pt, Vec_f eta, Vec_i q) {

  // Varitation of the ISO efficiencies (12 eta, 3 pT -> 36 in total)
  // varType: 1=stat DATA, 2=syst DATA, -1=stat MC, -2=syst MC

  Vec_f res(36, 0);
  double nom = lepSF_mu_ISO(pt, eta, q, 0, 0, 0); // nominal

  int c = 0;
  for (int iEta = 1; iEta <= 12; iEta++) {

    for (int iPt = 1; iPt <= 3; iPt++) {

      // if(iEta == 6 and iPt == 1) cout << lepSF_mu_ISO(pt, eta, q, varType,
      // iEta, iPt)/nom << endl;
      res[c] = lepSF_mu_ISO(pt, eta, q, varType, iEta, iPt) / nom;
      c++;
    }
  }

  return res;
}

Vec_f lepSF_mu_IDIP_syst(int varType, Vec_f pt, Vec_f eta, Vec_i q) {

  // Varitation of the ISO efficiencies (12 eta, 3 pT -> 36 in total)
  // varType: 1=stat DATA, 2=syst DATA, -1=stat MC, -2=syst MC

  Vec_f res(36, 0);
  double nom = lepSF_mu_IDIP(pt, eta, q, 0, 0, 0); // nominal

  int c = 0;
  for (int iEta = 1; iEta <= 12; iEta++) {

    for (int iPt = 1; iPt <= 3; iPt++) {

      res[c] = lepSF_mu_IDIP(pt, eta, q, varType, iEta, iPt) / nom;
      ; // up
      c++;
    }
  }

  return res;
}

Vec_f applyElectronSFTot(Vec_f pt, Vec_f eta, Vec_i q) {

  // IDISO: 12 eta, 3 pT, 36 in total
  // HLT:	  12 eta, 10 pT, 120 in total
  // TOTAL: 2*156 = 312 variations (up and down)

  Vec_f res(312 + 1, 0);
  /*
  res[0] = applyElectronSF(pt, eta, q); // nominal
  //res[1] = applyElectronSF(pt, eta, q); // up
  //res[2] = applyElectronSF(pt, eta, q); // down


  // variations on
  int c = 1;
  for(int iEta=1; iEta <= 12; iEta++) { // IDISO

          for(int iPt=1; iPt <= 3; iPt++) {

                  res[c] = applyElectronSF_IDISO(pt, eta, q, iEta, iPt); // up
                  c++;

                  res[c] = applyElectronSF_IDISO(pt, eta, q, -iEta, -iPt); //
  down c++;
          }
  }
  */

  return res;
}

Vec_f applyElectronSFSingleLegTot(Vec_f pt, Vec_f eta, Vec_i q) {

  Vec_f res(3, 0);

  res[0] = applyElectronSFSingleLeg(pt, eta, q);
  res[1] = applyElectronSFSingleLeg(pt, eta, q); // up
  res[2] = applyElectronSFSingleLeg(pt, eta, q); // down

  return res;
}

double applyMuonSFSingleLeg(Vec_f pt, Vec_f eta, Vec_i q) {

  unsigned int size = pt.size();
  double corr = 1.0;

  double trgSF_DATA = 1.0;
  double trgSF_MC = 1.0;

  for (unsigned int i = 0; i < size; ++i) {

    double eta_ = eta.at(i);
    double pt_ = pt.at(i);
    int q_ = q.at(i);

    int etabin_iso =
        std::max(1, std::min(lepSF_mu_iso->GetNbinsX(),
                             lepSF_mu_iso->GetXaxis()->FindBin(eta_)));
    int ptbin_iso =
        std::max(1, std::min(lepSF_mu_iso->GetNbinsY(),
                             lepSF_mu_iso->GetYaxis()->FindBin(pt_)));

    int etabin_idip =
        std::max(1, std::min(lepSF_mu_idip->GetNbinsX(),
                             lepSF_mu_idip->GetXaxis()->FindBin(eta_)));
    int ptbin_idip =
        std::max(1, std::min(lepSF_mu_idip->GetNbinsY(),
                             lepSF_mu_idip->GetYaxis()->FindBin(pt_)));

    int etabin_HLT =
        std::max(1, std::min(lepSF_mu_trg_DATA->GetNbinsX(),
                             lepSF_mu_trg_DATA->GetXaxis()->FindBin(eta_)));
    int ptbin_HLT =
        std::max(1, std::min(lepSF_mu_trg_DATA->GetNbinsY(),
                             lepSF_mu_trg_DATA->GetYaxis()->FindBin(pt_)));

    corr *= lepSF_mu_iso->GetBinContent(etabin_iso, ptbin_iso);
    corr *= lepSF_mu_idip->GetBinContent(etabin_idip, ptbin_idip);

    if (q_ > 0) {

      trgSF_DATA *=
          1. - lepSF_mu_trg_DATA->GetBinContent(etabin_HLT, ptbin_HLT);
      trgSF_MC *= 1. - lepSF_mu_trg_MC->GetBinContent(etabin_HLT, ptbin_HLT);
    } else {

      // trgSF_DATA *= 1. - lepSF_mu_trg_neg_DATA->GetBinContent(etabin_HLT,
      // ptbin_HLT); trgSF_MC *= 1. -
      // lepSF_mu_trg_neg_MC->GetBinContent(etabin_HLT, ptbin_HLT);
    }
  }
  // cout << (1. - trgSF_DATA) / (1. - trgSF_MC) << endl;
  corr *= (1. - trgSF_DATA) / (1. - trgSF_MC);

  return corr;
}

} // namespace wrem

#endif
