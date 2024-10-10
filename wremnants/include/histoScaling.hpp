#ifndef WREMNANTS_HISTO_SCALING_H
#define WREMNANTS_HISTO_SCALING_H

#include "utils.hpp"

namespace wrem {

Eigen::TensorFixedSize<double, Eigen::Sizes<2>>
constantScaling(double nominal_weight, double scale) {

  Eigen::TensorFixedSize<double, Eigen::Sizes<2>> outWeights;

  // Down weight, then up weight
  outWeights(0) = nominal_weight / scale;
  outWeights(1) = nominal_weight * scale;
  return outWeights;
}

Eigen::TensorFixedSize<double, Eigen::Sizes<2>>
twoPointScaling(double nominal_weight, double scaleDown, double scaleUp) {

  Eigen::TensorFixedSize<double, Eigen::Sizes<2>> outWeights;

  // Down weight, then up weight, nominal_weight should not already include a
  // centralScale weight if any
  outWeights(0) = nominal_weight * scaleDown;
  outWeights(1) = nominal_weight * scaleUp;
  return outWeights;
}

} // namespace wrem

#endif
