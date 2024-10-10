#ifndef WREMNANTS_LOWPU_UTILS_H
#define WREMNANTS_LOWPU_UTILS_H

#include "defines.hpp"
#include "utils.hpp"
#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <iostream>

namespace wrem {

Vec_i inverse(Vec_b tag) {

  Vec_i res(tag.size(), 0);
  for (unsigned int i = 0; i < res.size(); ++i) {

    if (tag[i] == 0)
      res[i] = 1;
  }
  // res[i] = !tag[i];
  return res;
}

Vec_i hasTriggerMatchLowPU(const Vec_f &eta, const Vec_f &phi,
                           const Vec_f &TrigObj_eta, const Vec_f &TrigObj_phi) {

  Vec_i res(eta.size(), 0); // initialize to 0
  for (unsigned int i = 0; i < res.size(); ++i) {
    for (unsigned int jtrig = 0; jtrig < TrigObj_eta.size(); ++jtrig) {
      // use deltaR*deltaR < 0.3*0.3, to be faster
      if (deltaR2(eta[i], phi[i], TrigObj_eta[jtrig], TrigObj_phi[jtrig]) <
          0.09) {
        res[i] = 1;
        break; // exit loop on trigger objects, and go to next muon
      }
    }
  }
  // res will be triggerMatchedMuons in RDF, like
  // RDF::Define("triggerMatchedMuons","hasTriggerMatch(Muon_eta,Muon_phi,TrigObj_eta[goodTrigObjs],TrigObj_phi[goodTrigObjs])")
  return res;
}

Vec_b goodMuonTriggerCandidateLowPU(const Vec_i &TrigObj_id,
                                    const Vec_f &TrigObj_pt,
                                    const Vec_f &TrigObj_l1pt,
                                    const Vec_f &TrigObj_l2pt,
                                    const Vec_i &TrigObj_filterBits) {

  Vec_b res(TrigObj_id.size(), false); // initialize to 0
  for (unsigned int i = 0; i < res.size(); ++i) {

    if (TrigObj_id[i] != 13)
      continue;
    // if (TrigObj_pt[i]   < 17.) continue;
    // if (TrigObj_l1pt[i] < 17.) continue; // no impact...
    // if (TrigObj_l1pt[i] < 16.) continue;
    // if (! (( TrigObj_filterBits[i] & 8) || (TrigObj_l2pt[i] > 10. &&
    // (TrigObj_filterBits[i] & 2) )) ) continue;
    if (!(TrigObj_filterBits[i] & 1))
      continue;
    // if(!(TrigObj_filterBits[i] & 8)) continue;
    res[i] = true;
  }

  return res;
}

Vec_b goodElectronTriggerCandidateLowPU(const Vec_i &TrigObj_id,
                                        const Vec_f &TrigObj_pt,
                                        const Vec_f &TrigObj_l1pt,
                                        const Vec_f &TrigObj_l2pt,
                                        const Vec_i &TrigObj_filterBits) {

  Vec_b res(TrigObj_id.size(), false); // initialize to 0
  for (unsigned int i = 0; i < res.size(); ++i) {
    if (TrigObj_id[i] != 11)
      continue;
    // if (TrigObj_pt[i]   < 20.) continue;
    // if (TrigObj_pt[i]   < 18.) continue;
    // if (TrigObj_l1pt[i] < 16.) continue;
    // if (! (( TrigObj_filterBits[i] & 8) || (TrigObj_l2pt[i] > 10. &&
    // (TrigObj_filterBits[i] & 2) )) ) continue; if (! (( TrigObj_filterBits[i]
    // & 8) || (TrigObj_l2pt[i] > 10. && (TrigObj_filterBits[i] & 2) )) )
    // continue;
    if (!(TrigObj_filterBits[i] & 1))
      continue;
    res[i] = true;
  }
  // res will be goodTrigObjs in RDF
  // e.g.
  // RDF::Define("goodTrigObjs","goodMuonTriggerCandidate(TrigObj_id,TrigObj_pt,TrigObj_l1pt,TrigObj_l2pt,TrigObj_filterBits)")
  return res;
}

Vec_f applyEGammaScaleSmearingUnc(int isData, Vec_f pt, Vec_f eta,
                                  Vec_f dEscaleUp, Vec_f dEscaleDown,
                                  Vec_f dEsigmaUp, Vec_f dEsigmaDown,
                                  int fluctuation = 0) {

  unsigned int size = pt.size();
  Vec_f res(size, 0.0);

  for (unsigned int i = 0; i < size; ++i) {

    if (fluctuation == 0)
      res[i] = pt[i];
    else if (fluctuation == 1) {

      double E = pt[i] * std::cosh(eta[i]); // compute energy
      if (isData)
        E += dEscaleUp[i];
      else
        E += dEsigmaUp[i];
      res[i] = E / std::cosh(eta[i]); // convert to pT
    } else if (fluctuation == -1) {

      double E = pt[i] * std::cosh(eta[i]); // compute energy
      if (isData)
        E += dEscaleDown[i];
      else
        E += dEsigmaDown[i];
      res[i] = E / std::cosh(eta[i]); // convert to pT
    }
  }

  return res;
}

Vec_f Egamma_undoCorrection(Vec_f pt, Vec_f eta, Vec_f Electron_ecalCorr) {

  // Electron_ecalCorr = calibrated ecal energy / miniaod ecal energy
  // we want the miniaod ecal energy
  unsigned int size = pt.size();
  Vec_f res(size, 0.0);

  for (unsigned int i = 0; i < size; ++i) {

    res[i] = pt[i] / Electron_ecalCorr[i];
  }

  return res;
}

bool debug(Vec_i TrigObj_filterBits) {

  // std::cout << goodLeptons.size() << " " << goodLeptonsPlus.size() << " " <<
  // goodLeptonsMinus.size() << " " << trgMatch.size() << std::endl;
  // if(goodLeptons.size() != 2) return true;
  // if(goodLeptons.size() <= 2) return true;
  std::cout << TrigObj_filterBits.size() << std::endl;

  for (int k = 0; k < TrigObj_filterBits.size(); k++) {

    // if(goodLeptons.at(k) and trigMatch.at(k)) lol = true;
    std::cout << " TrigObj_filterBits=" << TrigObj_filterBits.at(k)
              << std::endl;
    // std::cout << " goodLeptons=" << goodLeptons.at(k) << " goodLeptonsPlus="
    // << goodLeptonsPlus.at(k) << " goodLeptonsMinus=" <<
    // goodLeptonsMinus.at(k) << " trgMatch=" << trgMatch.at(k) << std::endl;
  }

  return true;
}

Vec_f trigFilter(Vec_f tag, Vec_f trigMatch) {

  Vec_f res;
  for (int k = 0; k < tag.size(); k++) {

    if (trigMatch.at(k))
      res.push_back(tag[k]);
  }
  return res;
}

Vec_i indices_(const Vec_f &vec, const int &start = 0) {
  Vec_i res(vec.size(), 0);
  std::iota(std::begin(res), std::end(res), start);
  return res;
}

Vec_i indices_(const int &size, const int &start = 0) {
  Vec_i res(size, 0);
  std::iota(std::begin(res), std::end(res), start);
  return res;
}

} // namespace wrem

#endif
