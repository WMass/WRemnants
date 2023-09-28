#include <ROOT/RVec.hxx>
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <cmath>
#include "defines.h"
#include "tfliteutils.h"


namespace wrem {

using ROOT::VecOps::RVec;

class ResponseCorrector {

public:

    ResponseCorrector(const std::string &filename, const std::string &func_name) :
        helper_(std::make_shared<narf::tflite_helper>(filename, func_name, ROOT::GetThreadPoolSize())) {}

    double operator() (const double &pt) {

        Eigen::TensorFixedSize<double, Eigen::Sizes<>> pt_tensor;
        Eigen::TensorFixedSize<double, Eigen::Sizes<>> pt_corr_tensor;
        pt_tensor(0) = pt;

        auto const inputs = std::tie(pt_tensor);
        auto outputs = std::tie(pt_corr_tensor);
        (*helper_)(inputs, outputs);
        return pt_corr_tensor(0);
    }


private:
    std::shared_ptr<narf::tflite_helper> helper_;
    double ut_max;
    double ut_min;
};



template <size_t N_UNC>
class RecoilCalibrationHelper {

public:

    struct scalar_tensor {
        using out_tensor_t = Eigen::TensorFixedSize<double, Eigen::Sizes<N_UNC>>;
        using out_scalar_t = Eigen::TensorFixedSize<double, Eigen::Sizes<>>;

        out_scalar_t ut_corr;
        out_tensor_t unc_weights;
    };

    using out_t = scalar_tensor;

    RecoilCalibrationHelper(const std::string &filename, const std::string &func_name, const double ut_min_, const double ut_max_, bool do_unc_) :
        helper_(std::make_shared<narf::tflite_helper>(filename, func_name, ROOT::GetThreadPoolSize())), 
        helper_no_unc_(std::make_shared<narf::tflite_helper>(filename, func_name + "_no_unc", ROOT::GetThreadPoolSize())) {
            do_unc = do_unc_;
            ut_max = ut_max_;
            ut_min = ut_min_;
    }

    out_t operator() (const double &pt, const double &ut) {

        Eigen::TensorFixedSize<double, Eigen::Sizes<>> pt_tensor;
        Eigen::TensorFixedSize<double, Eigen::Sizes<>> ut_tensor;
        pt_tensor(0) = pt;
        ut_tensor(0) = ut;

        out_t ret;
        if(ut < ut_max and ut > ut_min) {
            auto const inputs = std::tie(pt_tensor, ut_tensor);
            if(do_unc) {
                auto outputs = std::tie(ret.ut_corr, ret.unc_weights);
                (*helper_)(inputs, outputs);
            }
            else {
                auto outputs = std::tie(ret.ut_corr);
                (*helper_no_unc_)(inputs, outputs);
                ret.unc_weights.setConstant(1.);
            }
        }
        else {
            ret.ut_corr = ut_tensor;
            ret.unc_weights.setConstant(1.);
        }
        return ret;
    }


private:
    std::shared_ptr<narf::tflite_helper> helper_;
    std::shared_ptr<narf::tflite_helper> helper_no_unc_;
    bool do_unc;
    double ut_max;
    double ut_min;
};


}


