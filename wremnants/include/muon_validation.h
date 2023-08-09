#include <boost/histogram.hpp>
#include <stdlib.h>
#include <defines.h>
#include "muonCorr.h"
#include <cmath>

namespace wrem {

template <typename T, size_t N_MASSWEIGHTS>
class JpsiCorrectionsUncHelper_massWeights {

public:
    using hist_t = T;
    using tensor_t = typename T::storage_type::value_type::tensor_t;
    static constexpr auto sizes = narf::tensor_traits<tensor_t>::sizes;
    static constexpr auto nUnc = sizes[sizes.size() - 1]; // 1 for central value
    using out_tensor_t = Eigen::TensorFixedSize<double, Eigen::Sizes<nUnc, 2>>;

    JpsiCorrectionsUncHelper_massWeights(T&& corrections) :
        correctionHist_(std::make_shared<const T>(std::move(corrections))) {}

    // helper for bin lookup which implements the compile-time loop over axes
    template<typename... Xs, std::size_t... Idxs>
    const tensor_t &get_tensor_impl(std::index_sequence<Idxs...>, const Xs&... xs) {
        return correctionHist_->at(correctionHist_->template axis<Idxs>().index(xs)...).data();
    }

    // variadic templated bin lookup
    template<typename... Xs>
    const tensor_t &get_tensor(const Xs&... xs) {
        return get_tensor_impl(std::index_sequence_for<Xs...>{}, xs...);
    }

    // for smearing weights derived from propagating uncs on A, e, M to uncs on qop
    // the massWeights procedure doesn't work with multiple muons
    out_tensor_t operator() (
        float recoEta,
        int recoCharge,
        float recoPt,
        Eigen::TensorFixedSize<double, Eigen::Sizes<N_MASSWEIGHTS>> &weights,
        double nominal_weight = 1.0,
        bool isW = true
    ) {
        double scaleCut = 0.001;
        out_tensor_t res;
        res.setConstant(1.0);
        const auto &params = get_tensor(recoEta);

        for (std::ptrdiff_t ivar = 0; ivar < nUnc; ++ivar) {
            const double AUnc = params(0, ivar);
            const double eUnc = params(1, ivar);
            const double MUnc = params(2, ivar);
            double recoK = 1.0 /recoPt;
            double recoKUnc = (AUnc - eUnc * recoK) * recoK + recoCharge * MUnc;
            double scale = (
                abs(recoKUnc / recoK) > scaleCut ? 
                (std::copysign(1., recoKUnc / recoK) * scaleCut) : (recoKUnc / recoK)
            );
            auto scaleShiftWeights = scaleShiftWeightsFromMassWeights<N_MASSWEIGHTS>(
                nominal_weight, weights, scale, isW
            );
            res(ivar, 0) = scaleShiftWeights(0);
            res(ivar, 1) = scaleShiftWeights(1);
        }
        return res;
    }

private:
    std::shared_ptr<const T> correctionHist_;

};

}
