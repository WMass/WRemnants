#ifndef WREMNANTS_NANO_COMPAT_H
#define WREMNANTS_NANO_COMPAT_H

#include <array>
#include <cmath>

#include "ROOT/RVec.hxx"

namespace wrem {

// Jet ID (tight, tightLepVeto) of the AK4 PUPPI jets of NanoV15 for 2016: the
// NanoAODv15 recipe of the JetMET TWiki JetID13TeVUL, i.e. the UL2016 PUPPI
// criteria with the PUPPI-weighted Jet_chMultiplicity and Jet_neMultiplicity
// (CMSSW PFJetIDSelectionFunctor RUN2UL16PUPPI requires > 1 instead of >= 1
// neutral particles for 2.7 < |eta| <= 3). Encoded like the NanoV9 Jet_jetId:
// tight adds 2, tightLepVeto adds 4.
template <typename F, typename M>
ROOT::VecOps::RVec<int> jetIdUL16Puppi(
    const ROOT::VecOps::RVec<F> &eta, const ROOT::VecOps::RVec<F> &chHEF,
    const ROOT::VecOps::RVec<F> &neHEF, const ROOT::VecOps::RVec<F> &chEmEF,
    const ROOT::VecOps::RVec<F> &neEmEF, const ROOT::VecOps::RVec<F> &muEF,
    const ROOT::VecOps::RVec<M> &chMultiplicity,
    const ROOT::VecOps::RVec<M> &neMultiplicity) {
  ROOT::VecOps::RVec<int> res(eta.size(), 0);
  for (std::size_t i = 0; i < eta.size(); ++i) {
    const double aeta = std::abs(eta[i]);
    const int nch = chMultiplicity[i];
    const int nneutrals = neMultiplicity[i];
    bool tight = false;
    bool lepveto = true;
    if (aeta <= 2.4) {
      tight = chHEF[i] > 0. && nch > 0 && neHEF[i] < 0.9 && neEmEF[i] < 0.9 &&
              nch + nneutrals > 1;
      lepveto = chEmEF[i] < 0.8 && muEF[i] < 0.8;
    } else if (aeta <= 2.7) {
      tight = neHEF[i] < 0.98 && neEmEF[i] < 0.99;
    } else if (aeta <= 3.0) {
      tight = nneutrals >= 1;
    } else {
      tight = neEmEF[i] < 0.9 && nneutrals > 2;
    }
    res[i] = (tight ? 2 : 0) + (tight && lepveto ? 4 : 0);
  }
  return res;
}

// Pileup jet ID flags from the discriminant of NanoV15 (Jet_puIdDisc, 133X
// Winter24 PUPPI training) with the working points of that training in CMSSW
// (full_133x_Winter24_puppiv18_wp in PileupJetIDCutParams_cfi), binned as in
// PileupJetIdAlgo. Encoded like the 2016 NanoV9 Jet_puId, whose tight and loose
// bits are flipped (TWiki PileupJetIDUL): 4 = tight, 2 = medium, 1 = loose, so
// that Jet_puId >= 4 keeps its NanoV9 2016 meaning (tight). The training and
// working points are for Run 3 (TWiki PileupJetIDRun3); there are none for 2016
// PUPPI jets. The pileup ID is only meant for jets with pT < 50 GeV.
template <typename F>
ROOT::VecOps::RVec<int> puIdWinter24Puppi(const ROOT::VecOps::RVec<F> &pt,
                                          const ROOT::VecOps::RVec<F> &eta,
                                          const ROOT::VecOps::RVec<F> &disc) {
  // [pt bin: <10, 10-20, 20-30, 30-40, >=40][|eta| bin: <2.5, 2.5-2.75,
  // 2.75-3, >=3]
  using Table = std::array<std::array<double, 4>, 5>;
  static const Table tight = {{{-1.00, -1.00, -1.00, -1.00},
                               {0.038, 0.219, -0.220, -0.254},
                               {0.033, 0.060, -0.154, -0.154},
                               {0.056, 0.103, -0.159, -0.109},
                               {0.043, 0.127, -0.067, -0.059}}};
  static const Table medium = {{{-1.00, -1.00, -1.00, -1.00},
                                {-0.200, -0.068, -0.158, -0.384},
                                {-0.109, -0.179, -0.293, -0.322},
                                {-0.043, -0.124, -0.259, -0.286},
                                {-0.034, -0.071, -0.198, -0.235}}};
  static const Table loose = {{{-1.00, -1.00, -1.00, -1.00},
                               {-0.723, -0.392, -0.277, -0.516},
                               {-0.548, -0.347, -0.313, -0.489},
                               {-0.413, -0.289, -0.322, -0.438},
                               {-0.279, -0.219, -0.279, -0.384}}};
  ROOT::VecOps::RVec<int> res(pt.size(), 0);
  for (std::size_t i = 0; i < pt.size(); ++i) {
    const int ipt = pt[i] < 10   ? 0
                    : pt[i] < 20 ? 1
                    : pt[i] < 30 ? 2
                    : pt[i] < 40 ? 3
                                 : 4;
    const double aeta = std::abs(eta[i]);
    const int ieta = aeta < 2.5 ? 0 : aeta < 2.75 ? 1 : aeta < 3.0 ? 2 : 3;
    res[i] = (disc[i] > tight[ipt][ieta] ? 4 : 0) +
             (disc[i] > medium[ipt][ieta] ? 2 : 0) +
             (disc[i] > loose[ipt][ieta] ? 1 : 0);
  }
  return res;
}

} // namespace wrem

#endif
