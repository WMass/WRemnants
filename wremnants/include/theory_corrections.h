#ifndef WREMNANTS_THEORY_CORRECTIONS_H
#define WREMNANTS_THEORY_CORRECTIONS_H

#include <boost/histogram.hpp>
#include "utils.h"
#include <cmath>
#include "traits.h"
#include "csVariables.h"
#include "theoryTools.h"

namespace wrem {

template <typename T>
class TensorCorrectionsHelper3D {

typedef typename T::storage_type::value_type::tensor_t tensor_t;

public:
    TensorCorrectionsHelper3D(T&& corrections) :
        correctionHist_(std::make_shared<const T>(std::move(corrections))) {}

    template<typename X, typename Y, typename Z>
    const tensor_t &get_tensor(const X &x, const Y &y, const Z &z) {
        auto xbin = correctionHist_->template axis<0>().index(x);
        auto ybin = correctionHist_->template axis<1>().index(y);
        auto zbin = correctionHist_->template axis<2>().index(z);
        return correctionHist_->at(xbin, ybin, zbin).data();
    }

    tensor_t operator() (double x, double y, double z, double nominal_weight = 1.0) {
        return nominal_weight*get_tensor(x, y, z);
    }
private:
    std::shared_ptr<const T> correctionHist_;
};

template <typename T>
class QCDScaleByHelicityCorrectionsHelper : public TensorCorrectionsHelper3D<T> {

using tensor_t = typename T::storage_type::value_type::tensor_t;
static constexpr auto sizes = narf::tensor_traits<tensor_t>::sizes;
static constexpr auto nhelicity = sizes[0];
static constexpr auto nmur = sizes[1];
static constexpr auto nmuf = sizes[2];

using small_tensor_t = Eigen::TensorFixedSize<double, Eigen::Sizes<nhelicity, 1, 1>>;

public:
    QCDScaleByHelicityCorrectionsHelper(T&& corrections) : TensorCorrectionsHelper3D<T>(std::move(corrections)) {}

    small_tensor_t csAngularFactors(const CSVars &csvars) {

        const double sinThetaCS = csvars.sintheta;
        const double cosThetaCS = csvars.costheta;
        const double sinPhiCS = csvars.sinphi;
        const double cosPhiCS = csvars.cosphi;

        const double sin2ThetaCS = 2.*sinThetaCS*cosThetaCS;
        const double sin2PhiCS = 2.*sinPhiCS*cosPhiCS;
        const double cos2ThetaCS = 1. - 2.*sinThetaCS*sinThetaCS;
        const double cos2PhiCS= 1. - 2.*sinPhiCS*sinPhiCS;
        small_tensor_t angular;
        // n.b. setValues is error-prone for multi-dim tensors
        angular(0, 0, 0) = 1.+cosThetaCS*cosThetaCS;
        angular(1, 0, 0) = 0.5*(1. - 3.*cosThetaCS*cosThetaCS);
        angular(2, 0, 0) = sin2ThetaCS*cosPhiCS;
        angular(3, 0, 0) = 0.5*sinThetaCS*sinThetaCS*cos2PhiCS;
        angular(4, 0, 0) = sinThetaCS*cosPhiCS;
        angular(5, 0, 0) = cosThetaCS;
        angular(6, 0, 0) = sinThetaCS*sinThetaCS*sin2ThetaCS; //WRONG
        angular(7, 0, 0) = sin2ThetaCS*sinPhiCS;
        angular(8, 0, 0) = sinThetaCS*sinPhiCS;

        return angular;
    }

    tensor_t operator() (double yV, double ptV, int qV, const CSVars &csvars, const scale_tensor_t &scale_tensor, double nominal_weight = 1.0) {

        // pure angular terms without angular coeffs multiplied through
        const auto angular = csAngularFactors(csvars);

        static_assert(sizes.size() == 3);

        constexpr std::array<Eigen::Index, 3> broadcastscales = { 1, nmur, nmuf };
        // now multiplied through by angular coefficients (1.0 for 1+cos^2theta term)
        const tensor_t angular_with_coeffs = angular.broadcast(broadcastscales)*TensorCorrectionsHelper3D<T>::get_tensor(yV, ptV, qV);

        // denominator for each scale combination
        constexpr std::array<Eigen::Index, 1> helicitydims = { 0 };
        constexpr std::array<Eigen::Index, 3> broadcasthelicities = { nhelicity, 1, 1 };
        constexpr std::array<Eigen::Index, 3> reshapeden = { 1, nmur, nmuf };
        auto denominator = angular_with_coeffs.sum(helicitydims).reshape(reshapeden).broadcast(broadcasthelicities);

        constexpr std::array<Eigen::Index, 3> reshapescale = { 1, nmur, nmuf };
        auto scale = scale_tensor.reshape(reshapescale).broadcast(broadcasthelicities);

        return nominal_weight*scale*angular_with_coeffs/denominator;
    }

};

}

#endif
