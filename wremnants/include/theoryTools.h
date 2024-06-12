#ifndef WREMNANTS_THEORYTOOLS_H
#define WREMNANTS_THEORYTOOLS_H

#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <ROOT/RVec.hxx>
#include "utils.h"

namespace wrem {


ROOT::Math::PxPyPzEVector Sum4Vec(
  const ROOT::VecOps::RVec<double>& pt, 
  const ROOT::VecOps::RVec<double>& eta, 
  const ROOT::VecOps::RVec<double>& phi,
  const ROOT::VecOps::RVec<double>& mass  
  ) {
  const std::size_t ngenparts = pt.size();
  ROOT::Math::PxPyPzEVector full_system(0,0,0,0);
  for (std::size_t i = 0; i < ngenparts; ++i) {   
    ROOT::Math::PtEtaPhiMVector p(pt[i], eta[i], phi[i], mass[i]);
    full_system += p;
  }
  return full_system;
}
// overload method for massless particles
ROOT::Math::PxPyPzEVector Sum4Vec(
  const ROOT::VecOps::RVec<double>& pt, 
  const ROOT::VecOps::RVec<double>& eta, 
  const ROOT::VecOps::RVec<double>& phi
  ) {
  const std::size_t ngenparts = pt.size();
  ROOT::Math::PxPyPzEVector full_system(0,0,0,0);
  for (std::size_t i = 0; i < ngenparts; ++i) {   
    ROOT::Math::PtEtaPhiMVector p(pt[i], eta[i], phi[i], 0);
    full_system += p;
  }
  return full_system;
}



Eigen::TensorFixedSize<int, Eigen::Sizes<2>> prefsrLeptons(const ROOT::VecOps::RVec<int>& status,
        const ROOT::VecOps::RVec<int>& statusFlags, const ROOT::VecOps::RVec<int>& pdgId, const ROOT::VecOps::RVec<int>& motherIdx) {

  const std::size_t ngenparts = status.size();


  std::array<std::size_t, 2> photos_idxs;
  std::array<std::size_t, 2> all_idxs;
  std::size_t nphotos = 0;
  std::size_t nall = 0;
  for (std::size_t i = 0; i < ngenparts; ++i) {
    const int &istatus = status[i];
    const int &istatusFlags = statusFlags[i];
    const int &ipdgId = pdgId[i];
    const int &imotherIdx = motherIdx[i];

    const int absPdgId = std::abs(ipdgId);

    const bool is_lepton = absPdgId >= 11 && absPdgId <= 16;
    const bool is_status746 = istatus == 746;
    const bool is_status23 = istatus == 23;
    const int &motherPdgId = pdgId[imotherIdx];
    const bool is_motherV = motherPdgId == 23 || std::abs(motherPdgId) == 24;

    const bool is_photos = is_lepton && is_status746 && is_motherV;


    const bool is_fromHardProcess = istatusFlags & ( 1 << 8 );

    // TODO: Is there a way to relax the fromHardProcess condition?
    const bool is_other = is_lepton && (is_motherV || is_status23) && is_fromHardProcess;

    // If there are status = 746 leptons, they came from photos and are pre-FSR
    // (but still need to check the mother in case photos was applied to other particles in the
    // event, and in case of radiation from taus and their decay products)
    const bool is_all = is_photos || is_other;

    if (is_photos) {
      if (nphotos < 2) {
        photos_idxs[nphotos] = i;
      }
      ++nphotos;
    }

    if (is_all) {
      if (nall < 2) {
        all_idxs[nall] = i;
      }
      ++nall;
    }
  }

  std::array<std::size_t, 2> selected_idxs;
  if (nphotos == 2) {
    selected_idxs = photos_idxs;
  }
  else if (nall == 2) {
    selected_idxs = all_idxs;
  }
  else {
    throw std::range_error("Expected to find 2 pre-FSR leptons, but found " + std::to_string(nall) + ", " + std::to_string(nphotos));
  }

  std::array<int, 2> selected_pdgids = { pdgId[selected_idxs[0]], pdgId[selected_idxs[1]] };
  const bool partIdx = selected_pdgids[0] > 0;
  Eigen::TensorFixedSize<int, Eigen::Sizes<2>> out;
  out(0) = selected_idxs[!partIdx];
  out(1) = selected_idxs[partIdx];
  return out;

}

  constexpr size_t NHELICITY = 9;
  using helicity_tensor = Eigen::TensorFixedSize<double, Eigen::Sizes<NHELICITY>> ;

  helicity_tensor csAngularFactors(const CSVars &csvars)
  {
    const double sinThetaCS = csvars.sintheta;
    const double cosThetaCS = csvars.costheta;
    const double sinPhiCS = csvars.sinphi;
    const double cosPhiCS = csvars.cosphi;

    const double sin2ThetaCS = 2. * sinThetaCS * cosThetaCS;
    const double sin2PhiCS = 2. * sinPhiCS * cosPhiCS;
    const double cos2ThetaCS = 1. - 2. * sinThetaCS * sinThetaCS;
    const double cos2PhiCS = 1. - 2. * sinPhiCS * sinPhiCS;
    helicity_tensor angular;
    angular(0) = 1. + cosThetaCS * cosThetaCS;
    angular(1) = 0.5 * (1. - 3. * cosThetaCS * cosThetaCS);
    angular(2) = sin2ThetaCS * cosPhiCS;
    angular(3) = 0.5 * sinThetaCS * sinThetaCS * cos2PhiCS;
    angular(4) = sinThetaCS * cosPhiCS;
    angular(5) = cosThetaCS;
    angular(6) = sinThetaCS * sinThetaCS * sin2PhiCS;
    angular(7) = sin2ThetaCS * sinPhiCS;
    angular(8) = sinThetaCS * sinPhiCS;
    return angular;
  }

  helicity_tensor csAngularMoments(const CSVars &csvars) {
    const helicity_tensor &angular = csAngularFactors(csvars);

    // using definition from arxiv:1606.00689 Eq. 1 and 5 to align with ATLAS
    helicity_tensor scales;
    scales.setValues({ 0., 20./3., 5., 20., 4., 4., 5., 5., 4. });

    helicity_tensor offsets;
    offsets.setValues({ 1., 2./3., 0., 0., 0., 0., 0., 0., 0. });

    const helicity_tensor moments = scales*angular + offsets;

    return moments;
  }




using scale_tensor_t = Eigen::TensorFixedSize<double, Eigen::Sizes<3, 3>>;

scale_tensor_t makeScaleTensor(const Vec_f &scale_weights, double thres) {
  // from nanoaod doc lines (do I trust them?)
//[0] is mur=0.5 muf=0.5; [1] is mur=0.5 muf=1; [2] is mur=0.5 muf=2; [3] is mur=1 muf=0.5 ; [4] is mur=1 muf=1; [5] is mur=1 muf=2; [6] is mur=2 muf=0.5; [7] is mur=2 muf=1 ; [8] is mur=2 muf=2)*

  // ordering of the tensor axes are mur, muf with elements ordered 0.5, 1.0, 2.0
  // clip large weights when filling
  
  scale_tensor_t res;
  res(0, 0) = std::clamp<double>(scale_weights[0], -thres, thres); //mur=0.5 muf=0.5;
  res(0, 1) = std::clamp<double>(scale_weights[1], -thres, thres); //mur=0.5 muf=1.0;
  res(0, 2) = std::clamp<double>(scale_weights[2], -thres, thres); //mur=0.5 muf=2.0;
  res(1, 0) = std::clamp<double>(scale_weights[3], -thres, thres); //mur=1.0 muf=0.5;
  res(1, 1) = std::clamp<double>(scale_weights[4], -thres, thres); //mur=1.0 muf=1.0;
  res(1, 2) = std::clamp<double>(scale_weights[5], -thres, thres); //mur=1.0 muf=2.0;
  res(2, 0) = std::clamp<double>(scale_weights[6], -thres, thres); //mur=2.0 muf=0.5;
  res(2, 1) = std::clamp<double>(scale_weights[7], -thres, thres); //mur=2.0 muf=1.0;
  res(2, 2) = std::clamp<double>(scale_weights[8], -thres, thres); //mur=2.0 muf=2.0;

  return res;
}

using helicity_scale_tensor_t = Eigen::TensorFixedSize<double, Eigen::Sizes<NHELICITY, 3, 3>>;

  helicity_scale_tensor_t makeHelicityMomentScaleTensor(const CSVars &csvars, const scale_tensor_t &scale_tensor, double original_weight = 1.0)
  {

    constexpr Eigen::Index nhelicity = NHELICITY;
    constexpr Eigen::Index nmur = 3;
    constexpr Eigen::Index nmuf = 3;

    constexpr std::array<Eigen::Index, 3> broadcastscales = {1, nmur, nmuf};
    constexpr std::array<Eigen::Index, 3> broadcasthelicities = {nhelicity, 1, 1};
    constexpr std::array<Eigen::Index, 3> reshapescale = {1, nmur, nmuf};

    Eigen::TensorFixedSize<double, Eigen::Sizes<nhelicity, 1, 1>> moments = csAngularMoments(csvars).reshape(broadcasthelicities);

    return original_weight * scale_tensor.reshape(reshapescale).broadcast(broadcasthelicities) * moments.broadcast(broadcastscales);
  }

  template <Eigen::Index Npdfs>
  class makeHelicityMomentPdfTensor
  {
  public:
    makeHelicityMomentPdfTensor() {}

    using helicity_pdf_tensor_t = Eigen::TensorFixedSize<double, Eigen::Sizes<NHELICITY, Npdfs>>;
    using pdf_tensor_t = Eigen::TensorFixedSize<double, Eigen::Sizes<Npdfs>>;

    auto operator()(const CSVars &csvars, const pdf_tensor_t &pdf_tensor, double original_weight = 1.0)
    {

      constexpr Eigen::Index nhelicity = NHELICITY;

      constexpr std::array<Eigen::Index, 2> broadcastpdf = {1, Npdfs};
      constexpr std::array<Eigen::Index, 2> broadcasthelicities = {nhelicity, 1};
      constexpr std::array<Eigen::Index, 2> reshapepdf = {1, Npdfs};
      const auto moments = csAngularMoments(csvars).reshape(broadcasthelicities);

      helicity_pdf_tensor_t helicity_pdf_tensor;
      helicity_pdf_tensor = original_weight * moments.broadcast(broadcastpdf) * pdf_tensor.reshape(reshapepdf).broadcast(broadcasthelicities);

      return helicity_pdf_tensor;
    }
  };

double get_pdgid_mass(const int& pdgId){
  switch(std::abs(pdgId)) {
    case 11:
      return electron_mass;
    case 13:
      return muon_mass;
    case 15:
      return tau_mass;
  }
  return 0;
}

ROOT::VecOps::RVec<ROOT::Math::PxPyPzEVector> ewLeptons(
  const ROOT::VecOps::RVec<int>& status,
  const ROOT::VecOps::RVec<int>& statusFlags, 
  const ROOT::VecOps::RVec<int>& pdgId, 
  const ROOT::VecOps::RVec<double>& pt, 
  const ROOT::VecOps::RVec<double>& eta, 
  const ROOT::VecOps::RVec<double>& phi
) {

  const std::size_t ngenparts = status.size();
  ROOT::VecOps::RVec<ROOT::Math::PxPyPzEVector> leptons;

  for (std::size_t i = 0; i < ngenparts; ++i) {
    const int &istatus = status[i];
    const int &istatusFlags = statusFlags[i];
    const int &ipdgId = pdgId[i];

    const int absPdgId = std::abs(ipdgId);

    const bool is_lepton = absPdgId >= 11 && absPdgId <= 16;
    const bool is_status1 = istatus == 1;
    const bool is_status2 = istatus == 2;
    const bool is_tau = is_status2 && absPdgId == 15;

    const bool is_fromHardProcess = istatusFlags & ( 1 << 8 );
    const bool is_prompt = istatusFlags & ( 1 << 0 );

    const bool is_selected = is_lepton && (is_status1 || is_tau) && is_prompt && is_fromHardProcess;

    if (is_selected) {
      const double mass = get_pdgid_mass(ipdgId);
      ROOT::Math::PtEtaPhiMVector p4(pt[i], eta[i], phi[i], mass);
      leptons.emplace_back(p4);
    }
  }

  int nleptons = leptons.size();
  if (nleptons < 2) {
    throw std::range_error("Expected to find at least 2 prompt bare leptons from hard process, but found " + std::to_string(nleptons));
  }

  std::sort(leptons.begin(), leptons.end(), [](const auto a, const auto b) {return a.pt() > b.pt(); });
  assert(leptons[0].pt() > leptons[1].pt());

  return leptons;

}

ROOT::VecOps::RVec<ROOT::Math::PxPyPzEVector> ewPhotons(
  const ROOT::VecOps::RVec<int>& status,
  const ROOT::VecOps::RVec<int>& statusFlags, 
  const ROOT::VecOps::RVec<int>& pdgId, 
  const ROOT::VecOps::RVec<double>& pt, 
  const ROOT::VecOps::RVec<double>& eta, 
  const ROOT::VecOps::RVec<double>& phi
) {

  const std::size_t ngenparts = status.size();
  ROOT::VecOps::RVec<ROOT::Math::PxPyPzEVector> photons;

  for (std::size_t i = 0; i < ngenparts; ++i) {
    const int &istatus = status[i];
    const int &istatusFlags = statusFlags[i];
    const int &ipdgId = pdgId[i];

    const int absPdgId = std::abs(ipdgId);

    const bool is_photon = absPdgId == 22;
    const bool is_status1 = istatus == 1;

    const bool is_fromHardProcess = istatusFlags & ( 1 << 8 );
    const bool is_prompt = istatusFlags & ( 1 << 0 );

    const bool is_selected   = is_photon && is_status1 && is_prompt;

    if (is_selected) {
      ROOT::Math::PtEtaPhiMVector p4(pt[i], eta[i], phi[i], 0.);
      photons.emplace_back(p4);
    }
  }

  return photons;

}

ROOT::Math::PxPyPzEVector ewGenVPhos(const ROOT::VecOps::RVec<PxPyPzEVector>& leptons, const ROOT::VecOps::RVec<PxPyPzEVector>& photons) {

  ROOT::Math::PxPyPzEVector full_system(0,0,0,0);
  for (auto &p : leptons) {
    full_system += p;
  }
  for (auto &p : photons) {
    full_system += p;
  }

  return full_system;
}


int selectGenPart(const ROOT::VecOps::RVec<int>& status, const ROOT::VecOps::RVec<int>& pdgId, const int absPdgIdMin, const int absPdgIdMax, const int statusMin, const int statusMax, bool require = true) {


  const std::size_t ngenparts = status.size();

  int selidx = -1;
  int selstatus = -1;
  for (std::size_t i = 0; i < ngenparts; ++i) {
    const int &istatus = status[i];
    const int &iabsPdgId = std::abs(pdgId[i]);

    if (iabsPdgId >= absPdgIdMin && iabsPdgId <= absPdgIdMax && istatus >= statusMin && istatus <= statusMax && istatus > selstatus) {
      selidx = i;
      selstatus = istatus;
    }
  }

  if (require and selidx == -1) {
    throw std::runtime_error("Expected to find a gen particle matching the criteria, but did not.");
  }

  return selidx;

}


} 

#endif
