#ifndef WREMNANTS_THEORY_CORRECTIONS_H
#define WREMNANTS_THEORY_CORRECTIONS_H

#include <boost/histogram.hpp>
#include "utils.h"
#include <cmath>
#include "traits.h"
#include "csVariables.h"

namespace wrem {

template <typename T>
class TensorCorrectionsHelper3D {

typedef typename T::storage_type::value_type::tensor_t tensor_t;

public:
    TensorCorrectionsHelper3D(T&& corrections) :
        correctionHist_(std::make_shared<const T>(std::move(corrections))) {}

    tensor_t operator() (double x, double y, double z, double nominal_weight = 1.0) {
        auto xbin = correctionHist_->template axis<0>().index(x);
        auto ybin = correctionHist_->template axis<1>().index(y);
        auto zbin = correctionHist_->template axis<2>().index(z);
        return nominal_weight*correctionHist_->at(xbin, ybin, zbin).data();
    }
private:
    std::shared_ptr<const T> correctionHist_;
};

template <typename T>
class QCDScaleByHelicityCorrectionsHelper : public TensorCorrectionsHelper3D<T> {

using tensor_t = typename T::storage_type::value_type::tensor_t;
static constexpr auto sizes = narf::tensor_traits<tensor_t>::sizes;

using small_tensor_t = Eigen::TensorFixedSize<double, Eigen::Sizes<1, 1, sizes.back()>>;

public:
    QCDScaleByHelicityCorrectionsHelper(T&& corrections) : TensorCorrectionsHelper3D<T>(std::move(corrections)) {}

    small_tensor_t csAngularFactors(const CSVars &csvars) {

        const double sinThetaCS = csvars.sintheta;
        const double cosThetaCS = csvars.costheta;
        const double sinPhiCS = csvars.sinphi;
        const double cosPhiCS = csvars.cosphi;

        const double sin2ThetaCS = 2.*sinThetaCS*cosThetaCS;
        const double cos2ThetaCS = 1. - 2.*sinThetaCS*sinThetaCS;
        const double cos2PhiCS= 1. - 2.*sinPhiCS*sinPhiCS;
        small_tensor_t angular;
        angular.setValues({{{(1.+cosThetaCS*cosThetaCS),
            0.5*(1. - 3.*cosThetaCS*cosThetaCS),
            sin2ThetaCS*cosPhiCS, 
            0.5*sinThetaCS*sinThetaCS*cos2PhiCS,
            sinThetaCS*cosPhiCS,
            cosThetaCS,
            sinThetaCS*sinThetaCS*sin2ThetaCS,
            sin2ThetaCS*sinPhiCS,
            sinThetaCS*sinPhiCS
        }}});
        return angular;
    }

    tensor_t operator() (double ptV, double yV, double qV, const CSVars &csvars, const scale_tensor_t &scale_tensor, double nominal_weight = 1.0) {

        // pure angular terms without angular coeffs multiplied through
        const auto angular = csAngularFactors(csvars);

        static_assert(sizes.size() == 3);

        constexpr std::array<Eigen::Index, 3> broadcastscales = { sizes[0], sizes[1], 1 };
        // now multiplied through by angular coefficients (1.0 for 1+cos^2theta term)
        const tensor_t angular_with_coeffs = angular.broadcast(broadcastscales)*TensorCorrectionsHelper3D<T>::operator()(ptV, yV, qV);

        // denominator for each scale combination
        constexpr std::array<Eigen::Index, 2> helicitydims = { 2 };
        constexpr std::array<Eigen::Index, 3> broadcasthelicities = { 1, 1, sizes[2] };
        constexpr std::array<Eigen::Index, 3> reshapeden = { sizes[0], sizes[1], 1 };
        auto denominator = angular_with_coeffs.sum(helicitydims).reshape(reshapeden).broadcast(broadcasthelicities);

        constexpr std::array<Eigen::Index, 3> reshapescale = { sizes[0], sizes[1], 1 };
        auto scale = scale_tensor.reshape(reshapescale).broadcast(broadcasthelicities);

        return nominal_weight*scale*angular_with_coeffs/denominator;
    }

};

}

#endif
