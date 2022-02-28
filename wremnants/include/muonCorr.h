#ifndef WREMNANTS_MUONCORR_H
#define WREMNANTS_MUONCORR_H


#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>

namespace wrem {

double scaleWeight(double weight, double scale) {
    return std::exp(scale*std::log(std::abs(weight)))*std::copysign(1., weight);
}

template <size_t ETABINS, size_t T>
Eigen::TensorFixedSize<double, Eigen::Sizes<2, ETABINS>> dummyScaleFromMassWeights(double nominal_weight,
    Eigen::TensorFixedSize<double, Eigen::Sizes<T>>& weights, double eta, double scale, bool isW=true) {
    const double refMass = isW ? 80351.81229 : 91153.50974;
    const size_t centralIdx = 10;
    const double scaleMeV = refMass*scale;
    const int step10MeV = std::floor(scaleMeV/10.)+1;
    if (centralIdx-step10MeV < 0)
        throw std::out_of_range("Maximum allowed range for momentum scale uncertainty is 100 MeV!");
    const double scaleFac = scaleMeV/(10.*step10MeV);

    Eigen::TensorFixedSize<double, Eigen::Sizes<2, ETABINS>> outWeights;
    outWeights.setConstant(nominal_weight);

    const double etaStep = 2*2.4/ETABINS;
    size_t ieta = (std::clamp(eta, -2.4, 2.4)+2.4)/etaStep;
    
    // Down weight, then up weight
    outWeights(0, ieta) = scaleWeight(weights[centralIdx-step10MeV], scaleFac)*nominal_weight;
    outWeights(1, ieta) = scaleWeight(weights[centralIdx+step10MeV], scaleFac)*nominal_weight;
    return outWeights;
}

}

#endif
