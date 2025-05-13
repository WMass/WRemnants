#ifndef WREMNANTS_SYST_HELICITY_UTILS_H
#define WREMNANTS_SYST_HELICITY_UTILS_H

#include "theory_corrections.hpp"
#include "utils.hpp"
#include <TH2D.h>

namespace wrem {

// class to expand a rank N tensor with helicity weights
template <std::size_t Rank, std::size_t nhelicity, std::size_t... Dims>
class tensor_helper_helicity {
public:
  using value_type =
      Eigen::TensorFixedSize<double, Eigen::Sizes<nhelicity, Dims...>>;
  using pref_tensor = Eigen::TensorFixedSize<double, Eigen::Sizes<Dims...>>;
  using hel_tensor = Eigen::TensorFixedSize<double, Eigen::Sizes<nhelicity>>;

  tensor_helper_helicity() {}

  value_type operator()(const pref_tensor &et, const hel_tensor &ht) {
    constexpr std::array<Eigen::Index, Rank + 1> broadcastpref = {1, Dims...};
    std::array<Eigen::Index, Rank + 1> broadcasthelicities;
    broadcasthelicities[0] = nhelicity;
    std::fill(broadcasthelicities.begin() + 1, broadcasthelicities.end(), 1);

    auto shape3 = ht.reshape(broadcasthelicities).broadcast(broadcastpref);
    auto shape4 = et.reshape(broadcastpref).broadcast(broadcasthelicities);
    return shape3 * shape4;
  }
};

Eigen::TensorFixedSize<double, Eigen::Sizes<NHELICITY>>
scalarmultiplyHelWeightTensor(
    double wt,
    Eigen::TensorFixedSize<double, Eigen::Sizes<NHELICITY>> &helTensor) {
  return wt * helTensor;
}

template <typename T>
class WeightByHelicityHelper : public TensorCorrectionsHelper<T> {
  using base_t = TensorCorrectionsHelper<T>;
  using tensor_t = typename T::storage_type::value_type::tensor_t;
  static constexpr auto sizes = narf::tensor_traits<tensor_t>::sizes;
  static constexpr auto NHELICITY_WEIGHTS = NHELICITY;
  // TODO: Can presumably get the double type from the template param
  using helweight_tensor_t =
      Eigen::TensorFixedSize<double, Eigen::Sizes<NHELICITY_WEIGHTS>>;

public:
  using base_t::base_t;

  helweight_tensor_t operator()(double mV, double yV, double ptV, int qV,
                                const CSVars &csvars) {
    // static_assert(nhelicity == NHELICITY);
    const auto angular = csAngularFactors(csvars);
    const auto coeffs = base_t::get_tensor(mV, yV, ptV, qV);
    helweight_tensor_t helWeights;
    double sum = 0.;
    for (unsigned int i = 0; i < NHELICITY; i++) {
      helWeights(i) = coeffs(i) * angular(i);
      sum += coeffs(i) * angular(i); // full sum of all components
    }
    double factor = 1.; // protection against non-covered gen phase space
    if (sum != 0.)
      factor = 1. / sum;
    helweight_tensor_t helWeights_tensor = factor * helWeights;
    return helWeights_tensor;
  }
};

} // namespace wrem

#endif
