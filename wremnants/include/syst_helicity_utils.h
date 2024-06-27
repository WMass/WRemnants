#ifndef WREMNANTS_SYST_HELICITY_UTILS_H
#define WREMNANTS_SYST_HELICITY_UTILS_H

#include "TH2D.h"
#include "utils.h"
#include "theory_corrections.h"

namespace wrem {

  //class to expand the muon eff stat var tensor with helcitity weights
template <std::size_t nEta, std::size_t nPt, std::size_t ch, std::size_t nhelicity>
class muon_eff_helper_stat_helicity {  
 public:
  using eff_t = Eigen::TensorFixedSize<double, Eigen::Sizes<nEta, nPt, ch>>;
  using hel_t = Eigen::TensorFixedSize<double, Eigen::Sizes<nhelicity>>;
  using value_t = Eigen::TensorFixedSize<double, Eigen::Sizes<nhelicity, nEta, nPt, ch>>;
  
  muon_eff_helper_stat_helicity() {}
  
  value_t operator() (const eff_t& et, const hel_t& ht) {
    constexpr std::array<Eigen::Index, 4> broadcastEff = { 1, nEta, nPt, ch};
    constexpr std::array<Eigen::Index, 4> broadcasthelicities = { nhelicity, 1, 1, 1};
    auto shape3 = ht.reshape(broadcasthelicities).broadcast(broadcastEff);
    auto shape4 = et.reshape(broadcastEff).broadcast(broadcasthelicities);
    return shape3*shape4;
  }
};
 
//class to expand a rank 1 tensor with helicity weights
template <std::size_t nSize, std::size_t nhelicity>
class tensor1D_helper_helicity {  
 public:
  using tensor1D_t = Eigen::TensorFixedSize<double, Eigen::Sizes<nSize>>;
  using hel_t = Eigen::TensorFixedSize<double, Eigen::Sizes<nhelicity>>;
  using value_t = Eigen::TensorFixedSize<double, Eigen::Sizes<nhelicity, nSize>>;
  
  tensor1D_helper_helicity() {}
  
  value_t operator() (const tensor1D_t& et, const hel_t& ht) {
    constexpr std::array<Eigen::Index, 2> broadcastEff = { 1, nSize };
    constexpr std::array<Eigen::Index, 2> broadcasthelicities = { nhelicity, 1};
    auto shape3 = ht.reshape(broadcasthelicities).broadcast(broadcastEff);
    auto shape4 = et.reshape(broadcastEff).broadcast(broadcasthelicities);
    return shape3*shape4;
  }
};

//class to expand a rank 2 tensor of type <NSize,UP/DOWN> by helicity
 template <std::size_t nsize, std::size_t nhelicity>
class tensorupdownvar_helper_helicity {
 public:
  using value_type = Eigen::TensorFixedSize<double, Eigen::Sizes<nhelicity, nsize, 2>>;
  using pref_tensor = Eigen::TensorFixedSize<double, Eigen::Sizes<nsize, 2>>;
  using hel_tensor = Eigen::TensorFixedSize<double, Eigen::Sizes<nhelicity>>;
  tensorupdownvar_helper_helicity() {}
  value_type operator() (const pref_tensor &pf, const hel_tensor &ht) const {
    constexpr std::array<Eigen::Index, 3> broadcasthelicities = { nhelicity, 1, 1};
    constexpr std::array<Eigen::Index, 3> broadcastpref = { 1, nsize,2};   
    auto shape5 = pf.reshape(broadcastpref).broadcast(broadcasthelicities);
    auto shape4 = ht.reshape(broadcasthelicities).broadcast(broadcastpref);
    return shape4*shape5;
  }
}; 


//Expand rank two tensor by helicity
 template <std::size_t dim1, std::size_t dim2, std::size_t nhelicity>
class tensorRank2_helper_helicity {
 public:
  using value_type = Eigen::TensorFixedSize<double, Eigen::Sizes<nhelicity, dim1, dim2>>;
  using pref_tensor = Eigen::TensorFixedSize<double, Eigen::Sizes<dim1, dim2>>;
  using hel_tensor = Eigen::TensorFixedSize<double, Eigen::Sizes<nhelicity>>;
  tensorRank2_helper_helicity() {}
  value_type operator() (const pref_tensor &pf, const hel_tensor &ht) const {
    constexpr std::array<Eigen::Index, 3> broadcasthelicities = { nhelicity, 1, 1};
    constexpr std::array<Eigen::Index, 3> broadcastpref = { 1, dim1,dim2};   
    auto shape5 = pf.reshape(broadcastpref).broadcast(broadcasthelicities);
    auto shape4 = ht.reshape(broadcasthelicities).broadcast(broadcastpref);
    return shape4*shape5;
  }
}; 

Eigen::TensorFixedSize<double, Eigen::Sizes<NHELICITY>> scalarmultiplyHelWeightTensor(double wt, Eigen::TensorFixedSize<double, Eigen::Sizes<NHELICITY>>& helTensor) {
  return wt*helTensor;
}

template <typename T>
class WeightByHelicityHelper : public TensorCorrectionsHelper<T> {
   using base_t = TensorCorrectionsHelper<T>;
   using tensor_t = typename T::storage_type::value_type::tensor_t;
   static constexpr auto sizes = narf::tensor_traits<tensor_t>::sizes;
   static constexpr auto NHELICITY_WEIGHTS = NHELICITY;
   // TODO: Can presumably get the double type from the template param
   using helweight_tensor_t = Eigen::TensorFixedSize<double, Eigen::Sizes<NHELICITY_WEIGHTS>>;

 public:
   using base_t::base_t;

   helweight_tensor_t operator() (double mV, double yV, double ptV, int qV, const CSVars &csvars, double nominal_weight) {
     //static_assert(nhelicity == NHELICITY);
     const auto moments = csAngularFactors(csvars);
     const auto coeffs = base_t::get_tensor(mV, yV, ptV, qV);
     helweight_tensor_t helWeights;
     double sum = 0.;
     for(unsigned int i = 0; i < NHELICITY; i++) {
       helWeights(i) = coeffs(i) * moments(i);
       sum += coeffs(i) * moments(i);//full sum of all components
     }
     double factor = 1./sum;
     helweight_tensor_t helWeights_tensor = factor*helWeights;
     return helWeights_tensor;
  }
};

}

#endif
