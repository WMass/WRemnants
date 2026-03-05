#ifndef WREMNANTS_UTILS_H
#define WREMNANTS_UTILS_H

#include <Math/Vector4D.h>
#include <ROOT/RVec.hxx>
#include <TVector2.h>
#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>

#include "defines.hpp"
#include <algorithm>
#include <vector>

using namespace ROOT;

namespace wrem {

// pdg value
constexpr double electron_mass = 5.110e-04;
constexpr double muon_mass = 0.1056583745;
constexpr double tau_mass = 1.77682;

template <typename T>
ROOT::VecOps::RVec<T> absVal(const ROOT::VecOps::RVec<T> &val) {

  ROOT::VecOps::RVec<T> res(val.size(), 0.0); // initialize to 0
  for (unsigned int i = 0; i < res.size(); ++i) {
    res[i] = std::abs(val[i]);
  }
  return res;
}

template <typename T> bool printVar(const T &var) {
  std::cout << var << std::endl;
  return 1;
}

float pt_2(float pt1, float phi1, float pt2, float phi2) {

  TVector2 p1 = TVector2();
  p1.SetMagPhi(pt1, phi1);

  TVector2 sum = TVector2();
  sum.SetMagPhi(pt2, phi2);
  sum = p1 + sum;

  return sum.Mod();
}

float mt_2(float pt1, float phi1, float pt2, float phi2) {
  return std::sqrt(2 * pt1 * pt2 * (1 - std::cos(phi1 - phi2)));
}

TVector2 get_met_wlike(float ptOther, float phiOther, float met, float phimet) {

  TVector2 pl = TVector2();
  pl.SetMagPhi(ptOther, phiOther);

  TVector2 met_wlike = TVector2();
  met_wlike.SetMagPhi(met, phimet);
  met_wlike = pl + met_wlike;

  return met_wlike;
}

float get_mt_wlike(float pt, float phi, const TVector2 &met_wlike) {

  return mt_2(pt, phi, met_wlike.Mod(), met_wlike.Phi());
}

float get_mt_wlike(float pt, float phi, float ptOther, float phiOther,
                   float met, float phimet) {

  const TVector2 met_wlike = get_met_wlike(ptOther, phiOther, met, phimet);
  return get_mt_wlike(pt, phi, met_wlike);
}

double deltaPhi(float phi1, float phi2) {
  double result = phi1 - phi2;
  while (result > M_PI)
    result -= 2.0 * M_PI;
  while (result <= -1.0 * M_PI)
    result += 2.0 * M_PI;
  return result;
}

double deltaR2(float eta1, float phi1, float eta2, float phi2) {
  double deta = eta1 - eta2;
  double dphi = deltaPhi(phi1, phi2);
  return deta * deta + dphi * dphi;
}

RVec<double> vectDeltaR2(RVec<float> eta1, RVec<float> phi1, RVec<float> eta2,
                         RVec<float> phi2) {
  RVec<double> vect;
  for (unsigned int i = 0U; i != eta1.size(); i++) {
    vect.push_back(deltaR2(eta1[i], phi1[i], eta2[i], phi2[i]));
  }
  return vect;
}

Vec_i cleanJetsFromLeptons(const Vec_f &Jet_eta, const Vec_f &Jet_phi,
                           const Vec_f &Muon_eta, const Vec_f &Muon_phi,
                           const Vec_f &Electron_eta,
                           const Vec_f &Electron_phi) {

  Vec_i res(Jet_eta.size(), 1); // initialize to true and set to false whenever
                                // the jet overlaps with a muon

  for (unsigned int ij = 0; ij < res.size(); ++ij) {

    for (unsigned int im = 0; im < Muon_eta.size(); ++im) {
      if (deltaR2(Jet_eta[ij], Jet_phi[ij], Muon_eta[im], Muon_phi[im]) <
          0.16) { // cone DR = 0.4
        res[ij] = 0;
        break;
      }
    }

    if (res[ij]) {
      for (unsigned int ie = 0; ie < Electron_eta.size(); ++ie) {
        if (deltaR2(Jet_eta[ij], Jet_phi[ij], Electron_eta[ie],
                    Electron_phi[ie]) < 0.16) { // cone DR = 0.4
          res[ij] = 0;
          break;
        }
      }
    }
  }

  return res;
}

template <Era era>
Vec_i goodMuonTriggerCandidate(const Vec_i &TrigObj_id, const Vec_f &TrigObj_pt,
                               const Vec_f &TrigObj_l1pt,
                               const Vec_f &TrigObj_l2pt,
                               const Vec_i &TrigObj_filterBits) {
  Vec_i res(TrigObj_id.size(), 0); // initialize to 0
  for (unsigned int i = 0; i < res.size(); ++i) {
    if (TrigObj_id[i] != 13)
      continue;
    if (TrigObj_pt[i] < 24.)
      continue;
    if (TrigObj_l1pt[i] < 22.)
      continue;
    if constexpr (era == Era::Era_2016PostVFP) {
      if (!((TrigObj_filterBits[i] & 8) ||
            (TrigObj_l2pt[i] > 10. && (TrigObj_filterBits[i] & 2))))
        continue;
    } else {
      if (!(TrigObj_l2pt[i] > 10. && (TrigObj_filterBits[i] & 2)))
        continue;
    }
    res[i] = 1;
  }
  // res will be goodTrigObjs in RDF
  // e.g.
  // RDF::Define("goodTrigObjs","goodMuonTriggerCandidate(TrigObj_id,TrigObj_pt,TrigObj_l1pt,TrigObj_l2pt,TrigObj_filterBits)")
  return res;
}

// new overloaded function to be used with new ntuples having additional trigger
// bits
template <Era era>
Vec_i goodMuonTriggerCandidate(const Vec_i &TrigObj_id,
                               const Vec_i &TrigObj_filterBits) {
  Vec_i res(TrigObj_id.size(), 0); // initialize to 0
  for (unsigned int i = 0; i < res.size(); ++i) {
    if (TrigObj_id[i] != 13)
      continue;
    if constexpr (era == Era::Era_2016PostVFP) {
      if (!((TrigObj_filterBits[i] & 16) || (TrigObj_filterBits[i] & 32)))
        continue;
    } else {
      if (!(TrigObj_filterBits[i] & 4096))
        continue; // add 8192 later?
    }
    res[i] = 1;
  }
  return res;
}

Vec_i hasTriggerMatch(const Vec_f &eta, const Vec_f &phi,
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

bool hasTriggerMatch(const float &eta, const float &phi,
                     const Vec_f &TrigObj_eta, const Vec_f &TrigObj_phi) {

  for (unsigned int jtrig = 0; jtrig < TrigObj_eta.size(); ++jtrig) {
    if (deltaR2(eta, phi, TrigObj_eta[jtrig], TrigObj_phi[jtrig]) < 0.09)
      return true;
  }
  return false;
}

bool hasMatchDR2(const float &eta, const float &phi, const Vec_f &vec_eta,
                 const Vec_f &vec_phi, const float dr2 = 0.09) {

  for (unsigned int jvec = 0; jvec < vec_eta.size(); ++jvec) {
    if (deltaR2(eta, phi, vec_eta[jvec], vec_phi[jvec]) < dr2)
      return true;
  }
  return false;
}

RVec<Int_t> hasMatchDR2(const Vec_f &eta, const Vec_f &phi,
                        const Vec_f &vec_eta, const Vec_f &vec_phi,
                        const float dr2 = 0.09) {

  Vec_i hasMatch(eta.size(), 0);
  for (unsigned int ivec = 0; ivec < eta.size(); ++ivec) {
    for (unsigned int jvec = 0; jvec < vec_eta.size(); ++jvec) {
      if (deltaR2(eta[ivec], phi[ivec], vec_eta[jvec], vec_phi[jvec]) < dr2) {
        // found at least one object passing condition, go to next element of
        // first collection (saves time)
        hasMatch[ivec] = 1;
        continue;
      }
    }
  }
  return hasMatch;
}

RVec<Int_t> hasMatchDR2collWithSingle(const Vec_f &coll1_eta,
                                      const Vec_f &coll1_phi,
                                      const Float_t &eta, const Float_t &phi,
                                      const Float_t dr2 = 0.09) {
  Vec_i resDR(coll1_eta.size(), 0);
  float tmp_dr = 999.;
  for (int ic1 = 0; ic1 < coll1_eta.size(); ic1++) {
    tmp_dr = deltaR2(coll1_eta.at(ic1), coll1_phi.at(ic1), eta, phi);
    if (tmp_dr < dr2)
      resDR[ic1] = 1;
  }
  return resDR;
}

int hasMatchDR2idx(const float &eta, const float &phi, const Vec_f &vec_eta,
                   const Vec_f &vec_phi, const float dr2 = 0.09) {

  for (unsigned int jvec = 0; jvec < vec_eta.size(); ++jvec) {
    if (deltaR2(eta, phi, vec_eta[jvec], vec_phi[jvec]) < dr2)
      return jvec;
  }
  return -1;
}

int hasMatchDR2idx_closest(const float &eta, const float &phi,
                           const Vec_f &vec_eta, const Vec_f &vec_phi,
                           const float dr2 = 0.09) {

  double minDR2 = 1000.0;
  int ret = -1;
  for (unsigned int jvec = 0; jvec < vec_eta.size(); ++jvec) {
    double thisDR2 = deltaR2(eta, phi, vec_eta[jvec], vec_phi[jvec]);
    if (thisDR2 < dr2 and thisDR2 < minDR2) {
      minDR2 = thisDR2;
      ret = jvec;
    }
  }
  return ret;
}

Vec_i charge_from_pdgid(const Vec_i &pdgid) {

  // start by assigning negative charge, and set to +1 for negative pdgId
  // for neutral particles this might not be the desired choice, might implement
  // specific exceptions but for now this function is used for charged particles
  Vec_i res(pdgid.size(), -1);
  for (unsigned int i = 0; i < res.size(); ++i) {
    if (pdgid[i] < 0)
      res[i] = 1;
  }
  return res;
}

template <typename T>
T unmatched_postfsrMuon_var(const ROOT::VecOps::RVec<T> &var, const Vec_f &pt,
                            int hasMatchDR2idx) {

  T retVar = -99;
  if (hasMatchDR2idx < 0) {
    // std::cout << "Warning: no gen-reco match found" << std::endl;
    return retVar;
  }

  if (var.size() < 2) {
    // std::cout << "Warning: only one matched postFSR muon found" << std::endl;
    return retVar;
  }

  float maxPt = -1;
  // no need to require OS charge for the matched and unmatched muons (with
  // Z->4mu due to PHOTOS one can get same charge, but it is fine for the veto)
  for (unsigned int i = 0; i < var.size(); i++) {
    if (i != hasMatchDR2idx and pt[i] > maxPt) {
      retVar = var[i];
      maxPt = pt[i];
    }
  }
  return retVar;
}

template <typename T>
T unmatched_postfsrMuon_var_withCharge(const ROOT::VecOps::RVec<T> &var,
                                       const Vec_f &pt, const Vec_i &charge,
                                       int hasMatchDR2idx) {

  T retVar = -99;
  if (hasMatchDR2idx < 0) {
    // std::cout << "Warning: no gen-reco match found" << std::endl;
    return retVar;
  }

  if (var.size() < 2) {
    // std::cout << "Warning: only one matched postFSR muon found" << std::endl;
    return retVar;
  }

  float maxPt = -1;
  int matchedCharge = charge[hasMatchDR2idx];
  for (unsigned int i = 0; i < var.size(); i++) {
    if (i != hasMatchDR2idx and ((charge[i] + matchedCharge) == 0) and
        pt[i] > maxPt) {
      retVar = var[i];
      maxPt = pt[i];
    }
  }
  return retVar;
}

RVec<int> postFSRLeptonsIdx(RVec<bool> postFSRleptons) {
  RVec<int> v;
  for (unsigned int i = 0; i < postFSRleptons.size(); i++) {
    if (postFSRleptons[i])
      v.push_back(i);
  }
  return v;
}

float zqtproj0(const float &goodMuons_pt0, const float &goodMuons_eta0,
               const float &goodMuons_phi0, RVec<float> &GenPart_pt,
               RVec<float> &GenPart_eta, RVec<float> &GenPart_phi,
               RVec<int> &postFSRnusIdx) {
  ROOT::Math::PtEtaPhiMVector muon(goodMuons_pt0, goodMuons_eta0,
                                   goodMuons_phi0, muon_mass);
  ROOT::Math::PtEtaPhiMVector neutrino(GenPart_pt[postFSRnusIdx[0]],
                                       GenPart_eta[postFSRnusIdx[0]],
                                       GenPart_phi[postFSRnusIdx[0]], 0.);
  TVector2 Muon(muon.X(), muon.Y()), Neutrino(neutrino.X(), neutrino.Y());
  return (Muon * ((Muon + Neutrino))) / sqrt(Muon * Muon);
}

float zqtproj0(float pt, float phi, float ptOther, float phiOther) {
  TVector2 lep, boson;
  lep.SetMagPhi(pt, phi);
  boson.SetMagPhi(ptOther, phiOther);
  boson += lep;
  return (lep * boson) / pt;
}

int zqtproj0_angleSign(float pt, float phi, float ptOther, float phiOther) {
  TVector2 lep, boson;
  lep.SetMagPhi(pt, phi);
  boson.SetMagPhi(ptOther, phiOther);
  boson += lep;
  return std::copysign(1, lep * boson);
}

float zqtproj0_angleCosine(float pt, float phi, float ptOther, float phiOther) {
  TVector2 lep, boson;
  lep.SetMagPhi(pt, phi);
  boson.SetMagPhi(ptOther, phiOther);
  boson += lep;
  return (lep * boson) / (pt * boson.Mod());
}

float zqtproj0_boson(float pt, float phi, float bosonPt, float bosonPhi) {
  TVector2 lep, boson;
  lep.SetMagPhi(pt, phi);
  boson.SetMagPhi(bosonPt, bosonPhi);
  return (lep * boson) / pt;
}

float zqtproj0_boson(float pt, float phi, const TVector2 &boson) {
  TVector2 lep;
  lep.SetMagPhi(pt, phi);
  return (lep * boson) / pt;
}

float zqtproj0_boson(const TVector2 &lep, const TVector2 &boson) {
  return (lep * boson) / lep.Mod();
}

TVector2 transverseVectorSum(const Vec_f &pt, const Vec_f &phi) {

  TVector2 sum = TVector2();
  for (unsigned int i = 0; i < pt.size(); ++i) {
    if (i == 0) {
      sum.SetMagPhi(pt[i], phi[i]);
    } else {
      TVector2 part = TVector2();
      part.SetMagPhi(pt[i], phi[i]);
      sum += part;
    }
  }
  return sum;
}

Vec_f slice_vec(const Vec_f &vec, int start, int end) {
  Vec_f res(end - start, 0);
  std::copy(vec.begin() + start, vec.begin() + end, res.begin());
  return res;
}

template <std::ptrdiff_t N, typename V>
auto vec_to_tensor(const V &vec, std::size_t start = 0) {
  Eigen::TensorFixedSize<typename V::value_type, Eigen::Sizes<N>> res;
  std::copy(vec.begin() + start, vec.begin() + start + N, res.data());
  return res;
}

template <typename T, std::ptrdiff_t N, typename V>
auto vec_to_tensor_t(const V &vec, std::size_t start = 0) {
  Eigen::TensorFixedSize<T, Eigen::Sizes<N>> res;
  std::copy(vec.begin() + start, vec.begin() + start + N, res.data());
  return res;
}

template <typename T, std::ptrdiff_t N>
std::array<T, N>
tensor_to_array(const Eigen::TensorFixedSize<T, Eigen::Sizes<N>> &tensor) {
  std::array<T, N> res;
  std::copy(tensor.data(), tensor.data() + N, res.data());
  return res;
}

template <typename V> auto array_view(const V &vec, std::size_t start = 0) {
  return Eigen::Map<
      const Eigen::Array<typename V::value_type, Eigen::Dynamic, 1>>(
      vec.data() + start, vec.size() - start);
}

template <typename V> auto array_view(V &vec, std::size_t start = 0) {
  return Eigen::Map<Eigen::Array<typename V::value_type, Eigen::Dynamic, 1>>(
      vec.data() + start, vec.size() - start);
}

template <typename V> auto tensor_view(const V &vec, std::size_t start = 0) {
  return Eigen::TensorMap<const Eigen::Tensor<typename V::value_type, 1>>(
      vec.data() + start, vec.size() - start);
}

template <typename V> auto tensor_view(V &vec, std::size_t start = 0) {
  return Eigen::TensorMap<Eigen::Tensor<typename V::value_type, 1>>(
      vec.data() + start, vec.size() - start);
}

template <typename T>
T clip_tensor(const T &tensor, const typename T::Scalar &thres) {
  return tensor.cwiseMax(-thres).cwiseMin(thres);
}

// like std::make_shared but detach the TH1-derived object from the current
// directory
template <class T, class... Args>
std::shared_ptr<T> make_shared_TH1(Args &&...args) {
  using hist_t = std::decay_t<T>;
  hist_t *hist = new hist_t(std::forward<Args>(args)...);
  hist->SetDirectory(nullptr);
  return std::shared_ptr<T>(hist);
}

template <typename ArgType, typename = std::enable_if_t<
                                std::is_same_v<typename ArgType::Scalar, bool>>>
auto tensor_count(const ArgType &arg) {
  return arg.template cast<std::size_t>().sum();
}

template <typename ArgType, typename = std::enable_if_t<
                                std::is_same_v<typename ArgType::Scalar, bool>>>
std::size_t tensor_count_eval(const ArgType &arg) {
  return Eigen::TensorFixedSize<std::size_t, Eigen::Sizes<>>(
      tensor_count(arg))();
}

template <class ArgType> struct nonzero_helper {
  using ArrayType =
      Eigen::Array<typename Eigen::Index, Eigen::Dynamic, 1, Eigen::ColMajor,
                   ArgType::MaxSizeAtCompileTime, 1>;
};

template <class ArgType> class nonzero_functor {
  const ArgType &m_vec;

public:
  nonzero_functor(const ArgType &arg) : m_vec(arg) {}

  typename Eigen::Index operator()(Eigen::Index row) const {
    if (row == lastrow_) {
      return lastidx_;
    }
    const bool cached = lastrow_ == (row - 1);
    lastrow_ = row;
    if (cached) {
      for (Eigen::Index i = lastidx_ + 1; i < m_vec.rows(); ++i) {
        if (m_vec[i] != 0) {
          lastidx_ = i;
          return i;
        }
      }
    } else {
      for (Eigen::Index i = 0, count = 0; i < m_vec.rows(); ++i) {
        if (m_vec[i] != 0) {
          if (count++ == row) {
            lastidx_ = i;
            return i;
          }
        }
      }
    }
    return -1;
  }

private:
  mutable Eigen::Index lastrow_ = -1;
  mutable Eigen::Index lastidx_ = -1;
};

template <class ArgType> class nonzero_tensor_functor {
public:
  nonzero_tensor_functor(const ArgType &arg) : arg_(arg) {}

  typename Eigen::Index operator()(Eigen::Index row) const {
    if (row == lastrow_) {
      return lastidx_;
    }
    const bool cached = lastrow_ == (row - 1);
    lastrow_ = row;
    if (cached) {
      for (Eigen::Index i = lastidx_ + 1; i < arg_.size(); ++i) {
        if (arg_(i)) {
          lastidx_ = i;
          return i;
        }
      }
    } else {
      for (Eigen::Index i = 0, count = 0; i < arg_.size(); ++i) {
        if (arg_(i)) {
          if (count++ == row) {
            lastidx_ = i;
            return i;
          }
        }
      }
    }
    lastidx_ = -1;
    return -1;
  }

private:
  const Eigen::TensorRef<Eigen::Tensor<typename ArgType::Scalar, 1>> arg_;
  mutable Eigen::Index lastrow_ = -1;
  mutable Eigen::Index lastidx_ = -1;
};

template <class ArgType>
Eigen::CwiseNullaryOp<nonzero_functor<ArgType>,
                      typename nonzero_helper<ArgType>::ArrayType>
make_nonzero(const Eigen::ArrayBase<ArgType> &arg) {
  using ArrayType = typename nonzero_helper<ArgType>::ArrayType;
  static_assert(ArrayType::ColsAtCompileTime == 1);
  std::size_t size;
  if constexpr (std::is_same_v<typename ArrayType::Scalar, bool>) {
    size = arg.count();
  } else {
    size = (arg != 0).count();
  }
  return ArrayType::NullaryExpr(size, 1,
                                nonzero_functor<ArgType>(arg.derived()));
}

template <typename ArgType> auto make_nonzero_tensor(const ArgType &arg) {
  if constexpr (std::is_same_v<typename ArgType::Scalar, bool>) {
    auto asints = arg.template cast<Eigen::Index>();
    auto size = asints.sum();
    Eigen::TensorFixedSize<Eigen::Index, Eigen::Sizes<>> sizetensor = size;
    const Eigen::array<Eigen::Index, 1> offsets = {0};
    const Eigen::array<Eigen::Index, 1> extents = {sizetensor()};
    auto slice = asints.slice(offsets, extents);
    return slice.nullaryExpr(nonzero_tensor_functor(arg));
  } else {
    auto notzero = arg != static_cast<typename ArgType::Scalar>(0);
    auto asints = notzero.template cast<Eigen::Index>();
    auto size = asints.sum();
    Eigen::TensorFixedSize<Eigen::Index, Eigen::Sizes<>> sizetensor = size;
    const Eigen::array<Eigen::Index, 1> offsets = {0};
    const Eigen::array<Eigen::Index, 1> extents = {sizetensor()};
    auto slice = asints.slice(offsets, extents);
    return slice.nullaryExpr(nonzero_tensor_functor(notzero));
  }
}

template <class ArgType, class IndexType> class fancy_index_tensor_functor {
public:
  fancy_index_tensor_functor(const ArgType &arg, const IndexType &idxs)
      : arg_(arg), idxs_(idxs) {}

  typename Eigen::Index operator()(Eigen::Index row) const {
    return arg_(idxs_(row));
  }

private:
  const Eigen::TensorRef<Eigen::Tensor<typename ArgType::Scalar, 1>> arg_;
  const Eigen::TensorRef<Eigen::Tensor<typename IndexType::Scalar, 1>> idxs_;
};

template <class ArgType, class IndexType>
auto fancy_index(const ArgType &arg, const IndexType &idxs) {
  return idxs.template cast<typename ArgType::Scalar>().nullaryExpr(
      fancy_index_tensor_functor(arg, idxs));
}

template <class ArgType, class MaskType>
auto bool_index(const ArgType &arg, const MaskType &mask) {
  return fancy_index(arg, make_nonzero_tensor(mask));
}

template <typename ArgTypeIf, typename ArgTypeThen, typename ArgTypeElse>
auto scalar_select(const ArgTypeIf &cond, const ArgTypeThen &arg0,
                   const ArgTypeElse &arg1) {
  Eigen::TensorRef<Eigen::Tensor<typename ArgTypeThen::Scalar, 1>> arg0ref(
      arg0);
  Eigen::array<Eigen::Index, 1> shape{1};
  Eigen::array<Eigen::Index, 1> broadcast{arg0ref.size()};
  return cond.reshape(shape).broadcast(broadcast).select(arg0, arg1);
}

// Breit-Wigner mass weights
// values in running width scheme
const double MZ_GEN_ = 911876;
const double GAMMAZ_GEN_ = 24941;
const double MW_GEN_ = 80379;
const double GAMMAW_GEN_ = 2091.1;

double convertRunningToFixedWidth(const double param, const double mass,
                                  const double width) {
  // converts mass or width from running width scheme to fixed width scheme
  //
  // parameters:
  // param = mass or width in running width scheme
  // mass = reference mass in running width scheme
  // width = reference width in running width scheme

  return param / std::sqrt(1 + std::pow(width / mass, 2));
}

double computeBreitWigner(const double massVgen, const double ref_mass,
                          const double ref_Gamma) {
  // Calculates the Breit-Wigner function at a given mass point
  //
  // formula:
  // BW(m,mGen,Gamma) = k / ( s^2 + Gamma^2 M^2 )
  // where
  // s = m^2 - mGen^2
  // k = 2*sqrt(2)/pi * M*Gamma*gamma / sqrt(M^2 + gamma)
  // gamma = sqrt(mGen^2(mGen^2 + Gamma^2))
  // m = mass of the V boson in GeV (can be offshell)
  // mGen = true mass parameters of the V boson in GeV

  const double gamma =
      ref_mass * std::sqrt(ref_mass * ref_mass + ref_Gamma * ref_Gamma);
  const double k = (2 * std::sqrt(2) / M_PI) * ref_mass * ref_Gamma * gamma /
                   std::sqrt(ref_mass * ref_mass + gamma);
  const double s_hat = massVgen * massVgen * 1000 * 1000;
  const double offshell = s_hat - ref_mass * ref_mass;
  const double bw =
      k / (offshell * offshell + ref_Gamma * ref_Gamma * ref_mass * ref_mass);

  return bw;
}

template <int Type>
double computeBreitWignerWeight(const double massVgen, const double offset_mass,
                                const double offset_width) {
  // applies offset to the reference mass and width of the V boson in the
  // running width scheme converts offset and reference mass and width to fixed
  // width scheme and calculates ratio of BW(offset)/BW(reference)
  //
  // parameters:
  // massVgen = generated mass of the V boson in GeV
  // offset = mass shift in GeV
  // type = 0 for Z, 1 for W
  //
  // returns:
  // weight = BW(offset mass)/BW(reference mass)
  //
  // Notes:
  // Gamma = width of V boson, which scales like mGen^3 at LO
  // so Gamma(offset mass) = Gamma(reference mass)*(offset mass/reference
  // mass)^3

  double MV_GEN_ = 0;
  double GAMMAV_GEN_ = 0;
  if constexpr (Type == 0) {
    MV_GEN_ = MZ_GEN_;
    GAMMAV_GEN_ = GAMMAZ_GEN_;
  } else if constexpr (Type == 1) {
    MV_GEN_ = MW_GEN_;
    GAMMAV_GEN_ = GAMMAW_GEN_;
  }

  // convert reference mass and width to fixed width scheme and compute BW
  // weight
  const double ref_fixed_mass =
      convertRunningToFixedWidth(MV_GEN_, MV_GEN_, GAMMAV_GEN_);
  const double ref_fixed_Gamma =
      convertRunningToFixedWidth(GAMMAV_GEN_, MV_GEN_, GAMMAV_GEN_);
  const double bw =
      computeBreitWigner(massVgen, ref_fixed_mass, ref_fixed_Gamma);

  // shift the mass and compute the corresponding width in the running width
  // scheme
  const double target_mass = MV_GEN_ + offset_mass;
  const double target_Gamma =
      (GAMMAW_GEN_ + offset_width) * std::pow(target_mass / MW_GEN_, 3);
  // convert shifted mass and width to fixed width scheme
  const double target_fixed_mass =
      convertRunningToFixedWidth(target_mass, target_mass, target_Gamma);
  const double target_fixed_Gamma =
      convertRunningToFixedWidth(target_Gamma, target_mass, target_Gamma);
  // compute BW weight for shifted mass and width
  const double bw_offset =
      computeBreitWigner(massVgen, target_fixed_mass, target_fixed_Gamma);

  const double weight = bw_offset / bw;
  return weight;
}

template <int Type> Vec_f breitWignerMassWeights(const double massVgen) {

  // Z -> type=0
  // W -> type=1

  constexpr int N = (Type == 0) ? 23 : 21;
  Vec_f res(N, 1);
  double offset = -100;
  for (int i = 0; i < 21; i++) {
    offset = -100 + i * 10;
    res[i] = computeBreitWignerWeight<Type>(massVgen, offset, 0.0);
  }

  if constexpr (Type == 0) {
    res[21] = computeBreitWignerWeight<Type>(massVgen, -2.1, 0.0);
    res[22] = computeBreitWignerWeight<Type>(massVgen, 2.1, 0.0);
  }

  return res;
}

template <int Type> Vec_f breitWignerWidthWeights(const double massVgen) {

  // Z -> type=0
  // W -> type=1

  constexpr std::size_t N = 5;

  constexpr std::array<float, N> offsets =
      (Type == 0) ? std::array<float, N>{-0.77f, 0.83f, -1.2f, 1.1f, 3.4f}
                  : std::array<float, N>{-0.57f, 0.63f, -48.1f, -6.1f, 35.9f};

  Vec_f res(N, 1);
  for (std::size_t i = 0; i < N; ++i) {
    res[i] = computeBreitWignerWeight<Type>(massVgen, 0.0, offsets[i]);
  }

  return res;
}

// remove width dependence from mass weights
template <std::size_t N> class MassWeightHelper {

public:
  using tensor_t = Eigen::TensorFixedSize<double, Eigen::Sizes<N>>;

  MassWeightHelper(const double m0, const double gamma0,
                   const std::vector<double> &massVals,
                   const std::vector<double> &widthVals) {

    if (massVals.size() != N) {
      throw std::runtime_error("Invalid massVals, the length must match the "
                               "number of mass weights.");
    }

    // pre-compute which width weight to use and to what power it needs to be
    // raised to compensate the width change which accompanies the default mass
    // weights
    for (std::size_t i = 0; i < N; ++i) {
      const double m = massVals[i];
      const double gammaSource = std::pow(m / m0, 3) * gamma0;

      const double logGammaTargetRatio = std::log(gamma0 / gammaSource);

      // find closest width weight to compensate for change in width which
      // accompanies the original mass weight
      double mindistance = std::numeric_limits<double>::infinity();
      std::size_t widthIdx = 0;
      double widthPower = 0.;
      for (std::size_t j = 0; j < widthVals.size(); ++j) {
        const double logGammaRatio = std::log(widthVals[j] / gamma0);
        const double distance = std::fabs(logGammaRatio - logGammaTargetRatio);
        if (distance < mindistance) {
          widthIdx = j;
          widthPower = logGammaTargetRatio / logGammaRatio;
        }
      }

      widthIdxs_[i] = widthIdx;
      widthPowers_[i] = widthPower;
    }
  }

  tensor_t operator()(const ROOT::VecOps::RVec<float> &massWeights,
                      const ROOT::VecOps::RVec<float> &widthWeights) const {
    tensor_t res;
    for (std::size_t i = 0; i < N; ++i) {
      res(i) = massWeights[i] *
               std::pow(widthWeights[widthIdxs_[i]], widthPowers_[i]);
      // protect against rare pathological cases where width weight is zero
      if (!std::isfinite(res(i))) {
        res(i) = massWeights[i];
      }
    }

    return res;
  }

private:
  std::array<std::size_t, N> widthIdxs_;
  std::array<double, N> widthPowers_;
};

// take elements from a 1d tensor by index
template <typename tensor_t, std::size_t N> class index_taker {
public:
  using idxs_type = std::array<std::ptrdiff_t, N>;
  using out_tensor_t =
      Eigen::TensorFixedSize<typename tensor_t::Scalar, Eigen::Sizes<N>>;

  index_taker(const idxs_type &idxs) : idxs_(idxs) {}

  index_taker(const std::vector<std::ptrdiff_t> &idxs) {
    if (idxs.size() != N) {
      throw std::runtime_error("Mismatched indexes size");
    }

    for (std::size_t i = 0; i < N; ++i) {
      idxs_[i] = idxs[i];
    }
  }

  out_tensor_t operator()(const tensor_t &tensor) const {
    out_tensor_t res;

    for (std::size_t i = 0; i < N; ++i) {
      res(i) = tensor(idxs_[i]);
    }

    return res;
  }

private:
  idxs_type idxs_;
};

enum class TriggerCat { nonTriggering = 0, triggering = 1 };

std::vector<int> seq_idxs(const int size, const int start = 0) {
  std::vector<int> res(size);
  std::iota(res.begin(), res.end(), start);
  return res;
}

template <int N> std::array<int, N> seq_idxs_array(const int start = 0) {
  std::array<int, N> res;
  std::iota(res.begin(), res.end(), start);
  return res;
}

class ToyHelper {

public:
  ToyHelper(const std::size_t ntoys, const std::size_t seed = 0,
            const unsigned int var_scaling = 1, const unsigned int nslots = 1)
      : ntoys_(ntoys), var_scaling_(var_scaling) {
    const unsigned int nslotsactual = std::max(nslots, 1U);
    rng_.reserve(nslotsactual);
    auto const hash = std::hash<std::string>()("ToyHelper");
    for (std::size_t islot = 0; islot < nslotsactual; ++islot) {
      std::seed_seq seq{hash, seed, islot};
      rng_.emplace_back(seq);
    }
  }

  std::vector<int> operator()(const unsigned int slot) {

    std::poisson_distribution pois(1. / double(var_scaling_));

    std::vector<int> res;
    res.reserve(2 * ntoys_);

    // index 0 is the nominal, so just one entry, not randomized)
    res.emplace_back(0);

    auto &rngslot = rng_[slot];

    for (std::size_t itoy = 1; itoy < ntoys_; ++itoy) {
      const std::size_t nsamples = var_scaling_ * pois(rngslot);
      for (std::size_t isample = 0; isample < nsamples; ++isample) {
        res.emplace_back(itoy);
      }
    }

    return res;
  }

private:
  std::size_t ntoys_;
  double var_scaling_;
  std::vector<std::mt19937> rng_;
};

class RandomUniformHelper {

public:
  RandomUniformHelper(const std::size_t nsplits, const std::size_t seed = 0,
                      const unsigned int nslots = 1)
      : nsplits_(nsplits) {
    const unsigned int nslotsactual = std::max(nslots, 1U);
    rng_.reserve(nslotsactual);
    auto const hash = std::hash<std::string>()("RandomUniformHelper");
    for (std::size_t islot = 0; islot < nslotsactual; ++islot) {
      std::seed_seq seq{hash, seed, islot};
      rng_.emplace_back(seq);
    }
  }

  int operator()(const unsigned int slot) {
    std::uniform_int_distribution<int> dist(0, nsplits_ - 1);
    int res;
    auto &rngslot = rng_[slot];
    res = dist(rngslot);
    return res;
  }

private:
  std::size_t nsplits_;
  std::vector<std::mt19937> rng_;
};

class JackknifeHelper {

public:
  JackknifeHelper(const std::size_t nsplits, const float eff = 0.5,
                  const std::size_t seed = 0, const unsigned int nslots = 1)
      : nsplits_(nsplits), eff_(eff) {
    const unsigned int nslotsactual = std::max(nslots, 1U);
    rng_.reserve(nslotsactual);
    auto const hash = std::hash<std::string>()("JackknifeHelper");
    for (std::size_t islot = 0; islot < nslotsactual; ++islot) {
      std::seed_seq seq{hash, seed, islot};
      rng_.emplace_back(seq);
    }
  }

  std::vector<int> operator()(const unsigned int slot) {
    std::uniform_real_distribution<float> dist(0, 1);
    std::vector<int> res;
    res.reserve(2 * nsplits_);
    auto &rngslot = rng_[slot];

    // index 0 is the nominal, so just one entry, not randomized)
    res.emplace_back(0);

    for (std::size_t itoy = 1; itoy < nsplits_; ++itoy) {
      const float p = dist(rngslot);
      if (p < eff_) {
        res.emplace_back(itoy);
      }
    }

    return res;
  }

private:
  std::size_t nsplits_;
  const float eff_;
  std::vector<std::mt19937> rng_;
};
// function to do stuff with run splitting in MC should define a helper
// to have flexibility to set the internal arrays once at run-time
unsigned int get_dummy_run_by_lumi_quantile(const unsigned int run,
                                            const unsigned int lumi,
                                            const unsigned long long event,
                                            const Vec_d lumi_edges,
                                            const Vec_ui run_vals) {

  std::seed_seq seq{std::size_t(run), std::size_t(lumi), std::size_t(event)};
  std::mt19937 rng(seq);

  int bin = 0;
  // generate uniformly distributed double in [0,1)
  std::uniform_real_distribution<> dis(0.0, 1.0);
  double rn = dis(rng);
  // std::cout << "rn = " << rn << std::endl;
  // loop from second edge
  for (int i = 1; i < lumi_edges.size(); i++) {
    if (rn < lumi_edges[i]) {
      bin = i - 1;
      break;
    }
  }
  // std::cout << "rn, bin = " << rn << ", " << ret << std::endl;
  return run_vals[bin];
}

template <typename TensorType>
TensorType clamp_tensor_safe(const TensorType &tensor, double min_val,
                             double max_val) {
  static_assert(std::is_same_v<typename TensorType::Scalar, double>,
                "Tensor must have double scalar type");

  // Create result tensor with same dimensions
  TensorType result = tensor;

  // Apply element-wise operation: only clamp, preserve NaN
  result = result.unaryExpr([min_val, max_val](double x) {
    if (std::isnan(x)) {
      return x; // Keep NaN as is
    }
    return std::max(min_val, std::min(max_val, x));
  });

  return result;
}

// Overload with NaN replacement
template <typename TensorType>
TensorType clamp_tensor_safe(const TensorType &tensor, double min_val,
                             double max_val, double nan_replacement) {
  static_assert(std::is_same_v<typename TensorType::Scalar, double>,
                "Tensor must have double scalar type");

  // Create result tensor with same dimensions
  TensorType result = tensor;

  // Apply element-wise operation: replace NaN, then clamp
  result = result.unaryExpr([min_val, max_val, nan_replacement](double x) {
    if (std::isnan(x)) {
      return nan_replacement;
    }
    return std::max(min_val, std::min(max_val, x));
  });

  return result;
}

double get_differential_norm_weight(const double var_value,
                                    const Vec_d &axis_edges,
                                    const Vec_d &weight_list,
                                    bool flows_to_unit = false) {

  const size_t imax = axis_edges.size() - 1;

  // Below lower bound
  if (var_value < axis_edges.front())
    return flows_to_unit ? 1.0 : weight_list.front();

  // Above upper bound
  if (var_value >= axis_edges.back())
    return flows_to_unit ? 1.0 : weight_list.back();

  // Binary search for correct interval
  auto it = std::lower_bound(axis_edges.begin(), axis_edges.end(), var_value);
  size_t idx = std::distance(axis_edges.begin(), it) - 1;

  return weight_list[idx];
}

} // namespace wrem

#endif
