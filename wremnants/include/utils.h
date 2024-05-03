#ifndef WREMNANTS_UTILS_H
#define WREMNANTS_UTILS_H


#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include "defines.h"
#include "ROOT/RVec.hxx"
#include <Math/Vector4D.h>
#include "TVector2.h"

using namespace ROOT;

namespace wrem {

// pdg value
constexpr double electron_mass = 5.110e-04;
constexpr double muon_mass = 0.1056583745;
constexpr double tau_mass = 1.77682;

template <typename T>
ROOT::VecOps::RVec<T> absVal(const ROOT::VecOps::RVec<T> & val) {

    ROOT::VecOps::RVec<T> res(val.size(), 0.0); // initialize to 0
    for (unsigned int i = 0; i < res.size(); ++i) {
        res[i] = std::abs(val[i]);
    }
    return res;

}

template <typename T>
bool printVar(const T& var) {
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
    return std::sqrt(2*pt1*pt2*(1-std::cos(phi1-phi2)));
}

TVector2 get_met_wlike(float ptOther, float phiOther, float met, float phimet) {

  TVector2 pl = TVector2();
  pl.SetMagPhi(ptOther,phiOther);

  TVector2 met_wlike = TVector2();
  met_wlike.SetMagPhi(met,phimet);
  met_wlike = pl + met_wlike;

  return met_wlike;
  
}

float get_mt_wlike(float pt, float phi, const TVector2& met_wlike) {

    return mt_2(pt, phi, met_wlike.Mod(), met_wlike.Phi());

}
    
float get_mt_wlike(float pt, float phi, float ptOther, float phiOther, float met, float phimet) {

    const TVector2 met_wlike = get_met_wlike(ptOther, phiOther, met, phimet);
    return get_mt_wlike(pt, phi, met_wlike);

}
    
double deltaPhi(float phi1, float phi2) {
    double result = phi1 - phi2;
    while (result > M_PI) result -= 2.0*M_PI;
    while (result <= -1.0*M_PI) result += 2.0*M_PI;
    return result;
}

double deltaR2(float eta1, float phi1, float eta2, float phi2) {
    double deta = eta1-eta2;
    double dphi = deltaPhi(phi1,phi2);
    return deta*deta + dphi*dphi;
}

RVec<double> vectDeltaR2(RVec<float> eta1, RVec<float> phi1, RVec<float> eta2, RVec<float> phi2) {
	RVec<double> vect;
	for (unsigned int i=0U; i!=eta1.size(); i++) {
    	vect.push_back(deltaR2(eta1[i],phi1[i],eta2[i],phi2[i]));
	}
	return vect;
}

Vec_i cleanJetsFromLeptons(const Vec_f& Jet_eta, const Vec_f& Jet_phi, const Vec_f& Muon_eta, const Vec_f& Muon_phi, const Vec_f& Electron_eta, const Vec_f& Electron_phi) {

   Vec_i res(Jet_eta.size(), 1); // initialize to true and set to false whenever the jet overlaps with a muon

   for (unsigned int ij = 0; ij < res.size(); ++ij) {

     for (unsigned int im = 0; im < Muon_eta.size(); ++im) {
         if (deltaR2(Jet_eta[ij], Jet_phi[ij], Muon_eta[im], Muon_phi[im]) < 0.16) { // cone DR = 0.4
             res[ij] = 0;
             break;
         }
     }

     if (res[ij]) {
         for (unsigned int ie = 0; ie < Electron_eta.size(); ++ie) {
             if (deltaR2(Jet_eta[ij], Jet_phi[ij], Electron_eta[ie], Electron_phi[ie]) < 0.16) { // cone DR = 0.4
                 res[ij] = 0;
                 break;
             }
         }
     }

   }

   return res;
}
    
template <Era era>
Vec_i goodMuonTriggerCandidate(const Vec_i& TrigObj_id, const Vec_f& TrigObj_pt, const Vec_f& TrigObj_l1pt, const Vec_f& TrigObj_l2pt, const Vec_i& TrigObj_filterBits) {
   Vec_i res(TrigObj_id.size(), 0); // initialize to 0
   for (unsigned int i = 0; i < res.size(); ++i) {
       if (TrigObj_id[i]  != 13 ) continue;
       if (TrigObj_pt[i]   < 24.) continue;
       if (TrigObj_l1pt[i] < 22.) continue;
       if constexpr (era == Era::Era_2016PostVFP){
	 if (! (( TrigObj_filterBits[i] & 8) || (TrigObj_l2pt[i] > 10. && (TrigObj_filterBits[i] & 2) ))) continue;
       } else {
	 if (! (TrigObj_l2pt[i] > 10. && (TrigObj_filterBits[i] & 2) )) continue;
       }
       res[i] = 1;
   }
   // res will be goodTrigObjs in RDF
   // e.g. RDF::Define("goodTrigObjs","goodMuonTriggerCandidate(TrigObj_id,TrigObj_pt,TrigObj_l1pt,TrigObj_l2pt,TrigObj_filterBits)")
   return res;
}

// new overloaded function to be used with new ntuples having additional trigger bits
template <Era era>
Vec_i goodMuonTriggerCandidate(const Vec_i& TrigObj_id, const Vec_i& TrigObj_filterBits) {
  Vec_i res(TrigObj_id.size(), 0); // initialize to 0
  for (unsigned int i = 0; i < res.size(); ++i) {
        if (TrigObj_id[i]  != 13 ) continue;
	if constexpr (era == Era::Era_2016PostVFP){
	  if (! ( (TrigObj_filterBits[i] & 16) || (TrigObj_filterBits[i] & 32) ))  continue;
	} else {
	  if (! (TrigObj_filterBits[i] & 4096) )  continue;//add 8192 later?
	}
        res[i] = 1;
    }
    return res;
}

Vec_i hasTriggerMatch(const Vec_f& eta, const Vec_f& phi, const Vec_f& TrigObj_eta, const Vec_f& TrigObj_phi) {

   Vec_i res(eta.size(), 0); // initialize to 0
   for (unsigned int i = 0; i < res.size(); ++i) {
      for (unsigned int jtrig = 0; jtrig < TrigObj_eta.size(); ++jtrig) {
	  // use deltaR*deltaR < 0.3*0.3, to be faster
          if (deltaR2(eta[i], phi[i], TrigObj_eta[jtrig], TrigObj_phi[jtrig]) < 0.09) {
              res[i] = 1;
              break; // exit loop on trigger objects, and go to next muon
          }
      }
   }
   // res will be triggerMatchedMuons in RDF, like
   // RDF::Define("triggerMatchedMuons","hasTriggerMatch(Muon_eta,Muon_phi,TrigObj_eta[goodTrigObjs],TrigObj_phi[goodTrigObjs])")
   return res;

}

bool hasTriggerMatch(const float& eta, const float& phi, const Vec_f& TrigObj_eta, const Vec_f& TrigObj_phi) {

  for (unsigned int jtrig = 0; jtrig < TrigObj_eta.size(); ++jtrig) {
    if (deltaR2(eta, phi, TrigObj_eta[jtrig], TrigObj_phi[jtrig]) < 0.09) return true;
  }
  return false;

}

bool hasMatchDR2(const float& eta, const float& phi, const Vec_f& vec_eta, const Vec_f& vec_phi, const float dr2 = 0.09) {

  for (unsigned int jvec = 0; jvec < vec_eta.size(); ++jvec) {
    if (deltaR2(eta, phi, vec_eta[jvec], vec_phi[jvec]) < dr2) return true;
  }
  return false;

}
    
RVec<Int_t> hasMatchDR2(const Vec_f& eta, const Vec_f& phi, const Vec_f& vec_eta, const Vec_f& vec_phi, const float dr2 = 0.09) {

  Vec_i hasMatch(eta.size(), 0);
  for (unsigned int ivec = 0; ivec < eta.size(); ++ivec) {
      for (unsigned int jvec = 0; jvec < vec_eta.size(); ++jvec) {
          if (deltaR2(eta[ivec], phi[ivec], vec_eta[jvec], vec_phi[jvec]) < dr2) {
              // found at least one object passing condition, go to next element of first collection (saves time)
              hasMatch[ivec] = 1;
              continue;
          }
      }
  }
  return hasMatch;

}

RVec<Int_t> hasMatchDR2collWithSingle(const Vec_f &coll1_eta, const Vec_f &coll1_phi,
                                      const Float_t &eta, const Float_t &phi,
                                      const Float_t dr2 = 0.09)
{
    Vec_i resDR(coll1_eta.size(), 0);
    float tmp_dr = 999.;
    for(int ic1 = 0; ic1 < coll1_eta.size(); ic1++){
        tmp_dr = deltaR2(coll1_eta.at(ic1), coll1_phi.at(ic1), eta, phi);
        if (tmp_dr < dr2) resDR[ic1] = 1;
    }
    return resDR;
}

int hasMatchDR2idx(const float& eta, const float& phi, const Vec_f& vec_eta, const Vec_f& vec_phi, const float dr2 = 0.09) {

  for (unsigned int jvec = 0; jvec < vec_eta.size(); ++jvec) {
    if (deltaR2(eta, phi, vec_eta[jvec], vec_phi[jvec]) < dr2) return jvec;
  }
  return -1;

}

bool veto_vector_sorting(std::pair<float,float> i,std::pair<float,float> j) { return (i.first>j.first); }

float unmatched_postfsrMuon_var(const Vec_f& var, const Vec_f& pt, int hasMatchDR2idx) {

  std::vector<std::pair<float,float> > ptsortingvector;
  for (unsigned int i = 0; i < var.size(); i++) {
    if (i!=hasMatchDR2idx) ptsortingvector.push_back(std::pair(pt[i],var[i]));
  }
  if (ptsortingvector.size() == 0) return -99;
  std::sort(ptsortingvector.begin(),ptsortingvector.end(),veto_vector_sorting);
  return ptsortingvector[0].second;

}

bool veto_vector_sorting_charge(std::pair<float,int> i,std::pair<float,int> j) { return (i.first>j.first); }

int unmatched_postfsrMuon_charge(const Vec_i& var, const Vec_f& pt, int hasMatchDR2idx) {

  std::vector<std::pair<float,int> > ptsortingvector;
  for (unsigned int i = 0; i < var.size(); i++) {
    if (i!=hasMatchDR2idx) ptsortingvector.push_back(std::pair(pt[i],var[i]));
  }
  if (ptsortingvector.size() == 0) return -99;
  std::sort(ptsortingvector.begin(),ptsortingvector.end(),veto_vector_sorting_charge);
  return ptsortingvector[0].second;

}

RVec<int> postFSRLeptonsIdx(RVec<bool> postFSRleptons) {
  RVec<int> v;
  for (unsigned int i=0; i<postFSRleptons.size(); i++) {
    if (postFSRleptons[i]) v.push_back(i);
  }
  return v;
}

float zqtproj0(const float &goodMuons_pt0, const float &goodMuons_eta0, const float &goodMuons_phi0, RVec<float> &GenPart_pt, RVec<float> &GenPart_eta, RVec<float> &GenPart_phi, RVec<int> &postFSRnusIdx) {
    ROOT::Math::PtEtaPhiMVector muon(goodMuons_pt0, goodMuons_eta0, goodMuons_phi0, muon_mass);
    ROOT::Math::PtEtaPhiMVector neutrino(GenPart_pt[postFSRnusIdx[0]], GenPart_eta[postFSRnusIdx[0]], GenPart_phi[postFSRnusIdx[0]], 0.);
    TVector2 Muon(muon.X(), muon.Y()), Neutrino(neutrino.X(), neutrino.Y());
    return (Muon*((Muon+Neutrino)))/sqrt(Muon*Muon);
}
    
float zqtproj0(float pt, float phi, float ptOther, float phiOther) {
    TVector2 lep, boson;
    lep.SetMagPhi(pt,phi);
    boson.SetMagPhi(ptOther,phiOther);
    boson += lep;
    return (lep*boson)/pt;
}

float zqtproj0_boson(float pt, float phi, float bosonPt, float bosonPhi) {
    TVector2 lep, boson;
    lep.SetMagPhi(pt,phi);
    boson.SetMagPhi(bosonPt, bosonPhi);
    return (lep*boson)/pt;
}

float zqtproj0_boson(float pt, float phi, const TVector2& boson) {
    TVector2 lep;
    lep.SetMagPhi(pt,phi);
    return (lep*boson)/pt;
}

float zqtproj0_boson(const TVector2& lep, const TVector2& boson) {
    return (lep*boson)/lep.Mod();
}
    
TVector2 transverseVectorSum(const Vec_f& pt, const Vec_f& phi) {

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
    
template<std::ptrdiff_t N, typename V>
auto vec_to_tensor(const V &vec, std::size_t start = 0) {
  Eigen::TensorFixedSize<typename V::value_type, Eigen::Sizes<N>> res;
  std::copy(vec.begin() + start, vec.begin() + start + N, res.data());
  return res;
}

template<typename T, std::ptrdiff_t N, typename V>
auto vec_to_tensor_t(const V &vec, std::size_t start = 0) {
  Eigen::TensorFixedSize<T, Eigen::Sizes<N>> res;
  std::copy(vec.begin() + start, vec.begin() + start + N, res.data());
  return res;
}

template<typename V>
auto array_view(const V &vec, std::size_t start = 0) {
  return Eigen::Map<const Eigen::Array<typename V::value_type, Eigen::Dynamic, 1>>(vec.data() + start, vec.size() - start);
}

template<typename V>
auto array_view(V &vec, std::size_t start = 0) {
  return Eigen::Map<Eigen::Array<typename V::value_type, Eigen::Dynamic, 1>>(vec.data() + start, vec.size() - start);
}

template<typename V>
auto tensor_view(const V &vec, std::size_t start = 0) {
  return Eigen::TensorMap<const Eigen::Tensor<typename V::value_type, 1>>(vec.data() + start, vec.size() - start);
}

template<typename V>
auto tensor_view(V &vec, std::size_t start = 0) {
  return Eigen::TensorMap<Eigen::Tensor<typename V::value_type, 1>>(vec.data() + start, vec.size() - start);
}

template<typename T>
T clip_tensor(const T &tensor, const typename T::Scalar &thres) {
  return tensor.cwiseMax(-thres).cwiseMin(thres);
}

// like std::make_shared but detach the TH1-derived object from the current directory
template< class T, class... Args >
std::shared_ptr<T> make_shared_TH1( Args&&... args ) {
  using hist_t = std::decay_t<T>;
  hist_t *hist = new hist_t(std::forward<Args>(args)...);
  hist->SetDirectory(nullptr);
  return std::shared_ptr<T>(hist);
}

template <typename ArgType, typename = std::enable_if_t<std::is_same_v<typename ArgType::Scalar, bool>>>
auto tensor_count(const ArgType &arg) {
  return arg.template cast<std::size_t>().sum();
}

template <typename ArgType, typename = std::enable_if_t<std::is_same_v<typename ArgType::Scalar, bool>>>
std::size_t tensor_count_eval(const ArgType &arg) {
  return Eigen::TensorFixedSize<std::size_t, Eigen::Sizes<>>(tensor_count(arg))();
}


template<class ArgType>
struct nonzero_helper {
  using ArrayType =  Eigen::Array<typename Eigen::Index,
                 Eigen::Dynamic,
                 1,
                 Eigen::ColMajor,
                 ArgType::MaxSizeAtCompileTime,
                 1>;
};

template<class ArgType>
class nonzero_functor {
  const ArgType &m_vec;
public:

  nonzero_functor(const ArgType& arg) : m_vec(arg) {}

  typename Eigen::Index operator() (Eigen::Index row) const {
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
    }
    else {
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

template<class ArgType>
class nonzero_tensor_functor {
public:

  nonzero_tensor_functor(const ArgType& arg) : arg_(arg) {}

  typename Eigen::Index operator() (Eigen::Index row) const {
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
    }
    else {
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
Eigen:: CwiseNullaryOp<nonzero_functor<ArgType>, typename nonzero_helper<ArgType>::ArrayType>
make_nonzero(const Eigen::ArrayBase<ArgType>& arg)
{
  using ArrayType = typename nonzero_helper<ArgType>::ArrayType;
  static_assert(ArrayType::ColsAtCompileTime == 1);
  std::size_t size;
  if constexpr (std::is_same_v<typename ArrayType::Scalar, bool>) {
    size = arg.count();
  }
  else {
    size = (arg != 0).count();
  }
  return ArrayType::NullaryExpr(size, 1, nonzero_functor<ArgType>(arg.derived()));
}

template <typename ArgType>
auto make_nonzero_tensor(const ArgType &arg) {
  if constexpr (std::is_same_v<typename ArgType::Scalar, bool>) {
    auto asints = arg.template cast<Eigen::Index>();
    auto size = asints.sum();
    Eigen::TensorFixedSize<Eigen::Index, Eigen::Sizes<>> sizetensor = size;
    const Eigen::array<Eigen::Index, 1> offsets = {0};
    const Eigen::array<Eigen::Index, 1> extents = {sizetensor()};
    auto slice = asints.slice(offsets, extents);
    return slice.nullaryExpr(nonzero_tensor_functor(arg));
  }
  else {
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

template<class ArgType, class IndexType>
class fancy_index_tensor_functor {
public:
  fancy_index_tensor_functor(const ArgType &arg, const IndexType &idxs) : arg_(arg), idxs_(idxs) {}

  typename Eigen::Index operator() (Eigen::Index row) const {
    return arg_(idxs_(row));
  }


private:
  const Eigen::TensorRef<Eigen::Tensor<typename ArgType::Scalar, 1>> arg_;
  const Eigen::TensorRef<Eigen::Tensor<typename IndexType::Scalar, 1>> idxs_;
};

template<class ArgType, class IndexType>
auto fancy_index(const ArgType &arg, const IndexType &idxs) {
  return idxs.template cast<typename ArgType::Scalar>().nullaryExpr(fancy_index_tensor_functor(arg, idxs));
}

template<class ArgType, class MaskType>
auto bool_index(const ArgType &arg, const MaskType &mask) {
  return fancy_index(arg, make_nonzero_tensor(mask));
}

template<typename ArgTypeIf, typename ArgTypeThen, typename ArgTypeElse>
auto scalar_select(const ArgTypeIf &cond, const ArgTypeThen &arg0, const ArgTypeElse &arg1) {
  Eigen::TensorRef<Eigen::Tensor<typename ArgTypeThen::Scalar, 1>> arg0ref(arg0);
  Eigen::array<Eigen::Index, 1> shape{1};
  Eigen::array<Eigen::Index, 1> broadcast{arg0ref.size()};
  return cond.reshape(shape).broadcast(broadcast).select(arg0, arg1);
}


// Breit-Wigner mass weights
const double MZ_GEN_ = 91153.509740726733;
const double GAMMAZ_GEN_ = 2493.2018986110700;
const double MW_GEN_ = 80351.812293789408;
const double GAMMAW_GEN_ = 2090.4310808144846;

double computeBreitWignerWeight(double massVgen, double offset, int type) {
    
    double MV_GEN_ = 0;
    double GAMMAV_GEN_ = 0;
    if(type == 0) {
        MV_GEN_ = MZ_GEN_;
        GAMMAV_GEN_ = GAMMAZ_GEN_;
    }
    else {
        MV_GEN_ = MW_GEN_;
        GAMMAV_GEN_ = GAMMAW_GEN_;
    }

    double targetMass = MV_GEN_ + offset;
    //double gamma_cen = std::sqrt(MV_GEN_*MV_GEN_*(MV_GEN_*MV_GEN_+GAMMAV_GEN_*GAMMAV_GEN_));
    //double gamma = std::sqrt(targetMass*targetMass*(targetMass*targetMass+GAMMAV_GEN_*GAMMAV_GEN_));
    double s_hat = massVgen*massVgen*1000*1000;
    double offshell = s_hat - MV_GEN_*MV_GEN_;
    double offshellOffset = s_hat - targetMass*targetMass;
    double weight = (offshell*offshell + GAMMAV_GEN_*GAMMAV_GEN_*MV_GEN_*MV_GEN_) / (offshellOffset*offshellOffset + GAMMAV_GEN_*GAMMAV_GEN_*targetMass*targetMass);
    return weight;
}

Vec_f breitWignerWeights(double massVgen, int type=0) {
    
    // Z -> type=0
    // W -> type=1
    
    Vec_f res(21, 1);
    double offset = -100;
    for(int i=0; i<= 21; i++) {
        
        offset = -100 + i*10;
        res[i] = computeBreitWignerWeight(massVgen, offset, type);
        //cout << i << " " << offset << " " << res[i] << endl;
    }

    return res;
}

// take elements from a 1d tensor by index
template<typename tensor_t, std::size_t N>
class index_taker {
public:
    using idxs_type = std::array<std::ptrdiff_t, N>;
    using out_tensor_t = Eigen::TensorFixedSize<typename tensor_t::Scalar, Eigen::Sizes<N>>;

    index_taker(const idxs_type &idxs) : idxs_(idxs) {}

    index_taker(const std::vector<std::ptrdiff_t> &idxs) {
        if (idxs.size() != N) {
            throw std::runtime_error("Mismatched indexes size");
        }

        for (std::size_t i = 0; i < N; ++i) {
            idxs_[i] = idxs[i];
        }
    }

    out_tensor_t operator() (const tensor_t &tensor) const {
        out_tensor_t res;

        for (std::size_t i = 0; i < N; ++i) {
            res(i) = tensor(idxs_[i]);
        }

        return res;

    }
private:
    idxs_type idxs_;
};

}


#endif
