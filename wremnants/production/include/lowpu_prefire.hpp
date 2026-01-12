#ifndef WREMNANTS_LOWPU_PREFIRE_H
#define WREMNANTS_LOWPU_PREFIRE_H

#include "TFile.h"
#include "defines.hpp"
#include "utils.hpp"
#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>

namespace wrem {

TFile *_preFire = new TFile(
    "wremnants-data/data/lowPU/prefire/All2017Gand2017HPrefiringMaps.root",
    "READ");
TH2 *preFire_jet = (TH2F *)(_preFire->Get("L1prefiring_jetempt_2017H"));
TH2 *preFire_photon = (TH2F *)(_preFire->Get("L1prefiring_photonpt_2017H"));

// TFile *_preFire = new
// TFile("analyses/lowPU/weights/preFire/L1prefiring_combined_2017BtoF.root",
// "READ"); TH2 * preFire_jet =
// (TH2F*)(_preFire->Get("L1prefiring_jetempt_2017BtoF")); TH2 * preFire_photon
// = (TH2F*)(_preFire->Get("L1prefiring_photonpt_2017BtoF"));

double prefireCorr_getPrefireProbability(int type, int fluctuation, double eta,
                                         double pt, double maxpt) {

  double pref_prob = 0.0;
  double statunct = 0.0;
  double pt_ = std::min(pt, maxpt - 0.01);

  if (type == 0) { // jet map
    int etabin = std::max(1, std::min(preFire_jet->GetNbinsX(),
                                      preFire_jet->GetXaxis()->FindBin(eta)));
    int ptbin = std::max(1, std::min(preFire_jet->GetNbinsY(),
                                     preFire_jet->GetYaxis()->FindBin(pt_)));
    pref_prob = preFire_jet->GetBinContent(etabin, ptbin);
    statunct = preFire_jet->GetBinError(etabin, ptbin);
  } else { // photon map

    int etabin =
        std::max(1, std::min(preFire_photon->GetNbinsX(),
                             preFire_photon->GetXaxis()->FindBin(eta)));
    int ptbin = std::max(1, std::min(preFire_photon->GetNbinsY(),
                                     preFire_photon->GetYaxis()->FindBin(pt_)));
    pref_prob = preFire_photon->GetBinContent(etabin, ptbin);
    statunct = preFire_photon->GetBinError(etabin, ptbin);
  }

  double systunct = 0.2 * pref_prob;
  if (fluctuation == 1)
    pref_prob =
        std::min(1., pref_prob + sqrt(pow(statunct, 2) + pow(systunct, 2)));
  if (fluctuation == -1)
    pref_prob =
        std::max(0., pref_prob - sqrt(pow(statunct, 2) + pow(systunct, 2)));

  return pref_prob;
}

double prefireCorr(int fluctuation, Vec_f jet_pt, Vec_f jet_eta, Vec_f jet_phi,
                   Vec_i jet_muef, Vec_i jet_chemef, Vec_i jet_neemef,
                   Vec_f photon_pt, Vec_f photon_eta, Vec_f photon_phi,
                   Vec_f tag_pt, Vec_f tag_eta, Vec_f tag_phi) {

  // https://lathomas.web.cern.ch/lathomas/TSGStuff/L1Prefiring/PrefiringMaps_2017GandH/
  // https://github.com/lathomas/cmssw/blob/L1Prefiring_9_4_9/L1Prefiring/EventWeightProducer/plugins/L1ECALPrefiringWeightProducer.cc#L126

  // PAT implementation
  // https://github.com/cms-sw/cmssw/blob/master/PhysicsTools/PatUtils/plugins/L1PrefiringWeightProducer.cc

  // tag_[pt,eta,phi] the object to be filtered from the jets

  double nonPrefiringProba = 1.0; // final one
  double nonPrefiringProbaECAL = 1.0;
  double nonPrefiringProbaMuon = 1.0;

  double JetMinPt = 20;
  double JetMaxPt = 500;
  double JetMinEta = 2.0;
  double JetMaxEta = 3.0;
  double PhotonMinPt = 10;
  double PhotonMaxPt = 500;
  double PhotonMinEta = 2.0;
  double PhotonMaxEta = 3.0;

  // JetMinPt = 10; // for emPT

  unsigned njets = jet_pt.size();

  // loop over photons
  for (unsigned pid = 0; pid < photon_pt.size(); ++pid) {

    double p_pt = photon_pt.at(pid);
    double p_eta = photon_eta.at(pid);

    if (p_pt >= PhotonMinPt and std::abs(p_eta) <= PhotonMaxEta and
        std::abs(p_eta) >= PhotonMinEta) {

      double prefiringprob_gam = prefireCorr_getPrefireProbability(
          1, fluctuation, photon_eta.at(pid), photon_pt.at(pid), PhotonMaxPt);
      nonPrefiringProbaECAL *= (1.0 - prefiringprob_gam);
    }
  }

  // loop over jets
  for (unsigned jid = 0; jid < jet_pt.size(); ++jid) {

    double j_pt = jet_pt.at(jid);
    double j_eta = jet_eta.at(jid);
    double j_phi = jet_phi.at(jid);
    double j_muef = jet_muef.at(jid);
    double j_emef = jet_chemef.at(jid) + jet_neemef.at(jid);
    j_pt *= j_emef;

    // check if jet is in the tag (= muon or electron) collection
    bool skipJet = false;
    for (unsigned tid = 0; tid < tag_pt.size(); ++tid) {

      double t_eta = tag_eta.at(tid);
      double t_phi = tag_phi.at(tid);
      // if(deltaR(j_eta, j_phi, t_eta, t_phi) < 0.4) {
      if (deltaR2(j_eta, j_phi, t_eta, t_phi) < 0.4 * 0.4) {
        skipJet = true;
        break;
      }
    }
    if (skipJet)
      continue;

    double nonprefiringprobfromoverlappingphotons = 1.;
    bool foundOverlappingPhotons = false;

    if (j_pt < JetMinPt or std::abs(j_eta) > JetMaxEta or
        std::abs(j_eta) < JetMinEta or j_muef > 0.5)
      continue;

    // loop over photons to remove overlap
    for (unsigned pid = 0; pid < photon_pt.size(); ++pid) {

      double p_pt = photon_pt.at(pid);
      double p_eta = photon_eta.at(pid);
      double p_phi = photon_phi.at(pid);

      if (p_pt < PhotonMinPt and std::abs(p_eta) > PhotonMaxEta and
          std::abs(p_eta) < PhotonMinEta)
        continue;

      // check if photon in jet
      // if(deltaR(j_eta, j_phi, p_eta, p_phi) > 0.4) continue; // overlap
      // criteria (was 0.16?)
      if (deltaR2(j_eta, j_phi, p_eta, p_phi) > 0.4 * 0.4)
        continue;

      double prefiringprob_gam = prefireCorr_getPrefireProbability(
          1, fluctuation, p_eta, p_pt, PhotonMaxPt);
      nonprefiringprobfromoverlappingphotons *= (1.0 - prefiringprob_gam);
      foundOverlappingPhotons = true;
    }

    double nonprefiringprobfromoverlappingjet =
        1.0 - prefireCorr_getPrefireProbability(0, fluctuation, j_eta, j_pt,
                                                JetMaxPt);

    if (!foundOverlappingPhotons) {

      nonPrefiringProbaECAL *= nonprefiringprobfromoverlappingjet;
    } else if (nonprefiringprobfromoverlappingphotons >
               nonprefiringprobfromoverlappingjet) {

      // if overlapping photons have a non prefiring rate larger than the jet,
      // then replace these weights by the jet one i.e. select the maximum
      // prefiring probability (or minimum non-prefireing probability)
      if (nonprefiringprobfromoverlappingphotons > 0.) {

        nonPrefiringProbaECAL *= nonprefiringprobfromoverlappingjet /
                                 nonprefiringprobfromoverlappingphotons;
      } else {

        // nonPrefiringProbaECAL = 0.;
      }
    } else {

      // if overlapping photons have a non prefiring rate smaller than the jet,
      // don't consider the jet in the event weight, and do nothing.
    }
  }

  nonPrefiringProba = nonPrefiringProbaECAL;

  return nonPrefiringProba;
}

Vec_f prefireCorr_syst(Vec_f jet_pt, Vec_f jet_eta, Vec_f jet_phi,
                       Vec_i jet_muef, Vec_i jet_chemef, Vec_i jet_neemef,
                       Vec_f photon_pt, Vec_f photon_eta, Vec_f photon_phi,
                       Vec_f tag_pt, Vec_f tag_eta, Vec_f tag_phi) {

  Vec_f res(2, 1);

  double nom =
      prefireCorr(0, jet_pt, jet_eta, jet_phi, jet_muef, jet_chemef, jet_neemef,
                  photon_pt, photon_eta, photon_phi, tag_pt, tag_eta, tag_phi);
  res[0] =
      prefireCorr(1, jet_pt, jet_eta, jet_phi, jet_muef, jet_chemef, jet_neemef,
                  photon_pt, photon_eta, photon_phi, tag_pt, tag_eta, tag_phi) /
      nom; // up
  res[1] = prefireCorr(-1, jet_pt, jet_eta, jet_phi, jet_muef, jet_chemef,
                       jet_neemef, photon_pt, photon_eta, photon_phi, tag_pt,
                       tag_eta, tag_phi) /
           nom; // down

  return res;
}

} // namespace wrem

#endif
