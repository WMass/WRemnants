#ifndef WREMNANTS_SYST_HELICITY_UTILS_POLVAR_H
#define WREMNANTS_SYST_HELICITY_UTILS_POLVAR_H

#include "defines.hpp"
#include "theory_corrections.hpp"
#include "utils.hpp"
#include <boost/histogram/axis.hpp>

namespace wrem {

template <int GenCharge, int HelicityCoeffIndex, int NVars, typename HIST_VAR,
          typename HIST_NOM>
class WeightByHelicityHelper_polvar {

public:
  WeightByHelicityHelper_polvar(HIST_VAR &&hvar, HIST_NOM &&hnom)
      : hvar_(std::make_shared<const HIST_VAR>(std::move(hvar))),
        hnom_(std::make_shared<const HIST_NOM>(std::move(hnom))) {}

  using helWeights_tensor_t =
      Eigen::TensorFixedSize<double, Eigen::Sizes<NVars, 2>>; // 2 for Up/Down

  // gen-level qt/Q, |yV|, charge
  helWeights_tensor_t operator()(double qToverQ, double yV, int chargeV,
                                 const CSVars &csvars, double nominal_weight) {

    helWeights_tensor_t helWeights;

    if (GenCharge != chargeV) {
      helWeights.setConstant(nominal_weight);
      return helWeights;
    }

    // read index of the input histogram
    auto const ptV_idx = std::clamp(hnom_->template axis<0>().index(qToverQ), 0,
                                    sizeAxis0 - 1); // qT/Q, basically ptV/mV
    auto const yV_idx =
        std::clamp(hnom_->template axis<1>().index(yV), 0,
                   sizeAxis1 - 1); // yV is actually already |yV|
    auto const nHelCoeffs = hnom_->template axis<2>().size();

    const auto moments = csAngularFactors(csvars);
    std::array<double, NHELICITY> nomiCoeffs;
    for (int hel_idx = 0; hel_idx < nHelCoeffs; hel_idx++) {
      nomiCoeffs[hel_idx] = hnom_->at(ptV_idx, yV_idx, hel_idx);
    }

    // UL is a special case, no need to compute the sums, the ratio
    // hvar/hnom(qT/Q, yV) is already the weight we need
    if (HelicityCoeffIndex == 0) {
      for (unsigned int ivar_idx = 0; ivar_idx < NVars; ivar_idx++) {
        for (unsigned int iDownUp_idx = 0; iDownUp_idx < 2; iDownUp_idx++) {
          helWeights(ivar_idx, iDownUp_idx) =
              (nomiCoeffs[HelicityCoeffIndex] == 0.0)
                  ? nominal_weight
                  : nominal_weight *
                        hvar_->at(ptV_idx, yV_idx, ivar_idx, iDownUp_idx) /
                        nomiCoeffs[HelicityCoeffIndex];
        }
      }
      return helWeights;
    }

    // // compute nominal sum only once
    double sumNomi =
        moments(0); // based on how the input coefficients in the files are
                    // defined, here we only need (1 + cos^2(theta))
    for (int ihel = 1; ihel < nHelCoeffs; ihel++) {
      sumNomi += nomiCoeffs[ihel] * moments(ihel);
    }

    double sumAlt = 0.;
    double coeffAlt = 0.0;
    double factor = nominal_weight / sumNomi;

    for (unsigned int ivar_idx = 0; ivar_idx < NVars; ivar_idx++) {
      for (unsigned int iDownUp_idx = 0; iDownUp_idx < 2; iDownUp_idx++) {
        coeffAlt = hvar_->at(ptV_idx, yV_idx, ivar_idx, iDownUp_idx);
        sumAlt = sumNomi + (coeffAlt - nomiCoeffs[HelicityCoeffIndex]) *
                               moments(HelicityCoeffIndex);
        helWeights(ivar_idx, iDownUp_idx) = sumAlt * factor;
      }
    }

    return helWeights;
  }

protected:
  std::shared_ptr<const HIST_NOM>
      hnom_; // values of all nominal Ai coefficients vs ptV * yV for a given
             // charge
  std::shared_ptr<const HIST_VAR>
      hvar_; // for a given coefficients Aj and charge, values of alternate Aj
             // vs ptV * yV
  int sizeAxis0 = hnom_->template axis<0>().size();
  int sizeAxis1 = hnom_->template axis<1>().size();
};

} // namespace wrem

#endif
