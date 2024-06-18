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

class METXYCorrectionHelper {

public:

    using c_t = std::vector<double>;

    METXYCorrectionHelper(const c_t &coeff_x_, const c_t &coeff_y_) {
        coeff_x = coeff_x_;
        coeff_y = coeff_y_;
    }

    Vec_d operator() (const double &met_pt, const double &met_phi, const int &npv) {

        double delta;
        double pUx = met_pt*cos(met_phi);
        double pUy = met_pt*sin(met_phi);

        Vec_d res(2, 0);
        res[0] = hypot(pUx, pUy);
        res[1] = atan2(pUy, pUx);

        // x
        delta = 0;
        for(int k=0; k<coeff_x.size(); k++) delta += coeff_x.at(k)*std::pow(npv, k);
        if(std::abs(delta) > deltaMax) delta = 0;
        pUx -= delta;

        // y
        delta = 0;
        for(int k=0; k<coeff_y.size(); k++) delta += coeff_y.at(k)*std::pow(npv, k);
        if(std::abs(delta) > deltaMax) delta = 0;
        pUy -= delta;

        res[0] = hypot(pUx, pUy); // preserves the MET magnitude??
        res[1] = atan2(pUy, pUx);
        return res;
    }


private:
    c_t coeff_x;
    c_t coeff_y;
    double deltaMax = 10; // protection for high/nonphysical corrections
};


class VPTReweightHelper {

public:

    VPTReweightHelper(const std::vector<double> &bins_, const std::vector<double> &weights_, const double pt_min_, const double pt_max_) {
        bins = bins_;
        weights = weights_;
        pt_min = pt_min_;
        pt_max = pt_max_;
    }

    double operator() (const double &pt) {
        if(pt > pt_max or pt < pt_min) {
            return 1.;
        }
        int idx = bins.size()-2;
        for(unsigned int i=0; i < bins.size()-1; i++) {
            if(pt >= bins.at(i) and pt < bins.at(i+1)) {
                idx = i;
                break;
            }
        }
        return weights.at(idx);
    }


private:
    std::vector<double> bins;
    std::vector<double> weights;
    double pt_min;
    double pt_max;
};



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

        out_scalar_t ut_para_corr;
        out_scalar_t ut_perp_corr;
        out_tensor_t unc_weights;
    };

    using out_t = scalar_tensor;

    RecoilCalibrationHelper(const std::string &filename, const std::string &func_name, bool do_unc_) :
        helper_(std::make_shared<narf::tflite_helper>(filename, func_name, ROOT::GetThreadPoolSize())), 
        helper_no_unc_(std::make_shared<narf::tflite_helper>(filename, func_name + "_no_unc", ROOT::GetThreadPoolSize())) {
            do_unc = do_unc_;
    }

    out_t operator() (const double &pt, const double &ut_para, const double &ut_perp) {

        Eigen::TensorFixedSize<double, Eigen::Sizes<>> pt_tensor;
        Eigen::TensorFixedSize<double, Eigen::Sizes<>> ut_para_tensor;
        Eigen::TensorFixedSize<double, Eigen::Sizes<>> ut_perp_tensor;
        pt_tensor(0) = pt;
        ut_para_tensor(0) = ut_para;
        ut_perp_tensor(0) = ut_perp;

        out_t ret;
        auto const inputs = std::tie(pt_tensor, ut_para_tensor, ut_perp_tensor);
        if(do_unc) {
            auto outputs = std::tie(ret.ut_para_corr, ret.ut_perp_corr, ret.unc_weights);
            (*helper_)(inputs, outputs);
        }
        else {
            auto outputs = std::tie(ret.ut_para_corr, ret.ut_perp_corr);
            (*helper_no_unc_)(inputs, outputs);
            ret.unc_weights.setConstant(1.);
        }
        for(int k=0; k<N_UNC; k++) {
        //if(ret.unc_weights[k] < 0.99999 or ret.unc_weights[k] > 1.00001)
            if(std::isnan(ret.unc_weights[k]) or std::isinf(ret.unc_weights[k])) {
            cout << k << " INF/NAN " << std::isnan(ret.unc_weights[k]) << " " << std::isinf(ret.unc_weights[k]) << "  " <<  pt << " " << ut_para << " " << ut_perp << endl;
            }
        }
        return ret;
    }


private:
    std::shared_ptr<narf::tflite_helper> helper_;
    std::shared_ptr<narf::tflite_helper> helper_no_unc_;
    bool do_unc;
};



class RecoilCalibrationUncertaintyHelper {

public:

    RecoilCalibrationUncertaintyHelper(const std::string &filename, const std::string &func_name) :
        helper_(std::make_shared<narf::tflite_helper>(filename, func_name, ROOT::GetThreadPoolSize())) {}


    double operator() (const double &pt, const double &ut) {

        Eigen::TensorFixedSize<double, Eigen::Sizes<>> pt_tensor;
        Eigen::TensorFixedSize<double, Eigen::Sizes<>> ut_tensor;
        pt_tensor(0) = pt;
        ut_tensor(0) = ut;

        Eigen::TensorFixedSize<double, Eigen::Sizes<>> ret;
        auto const inputs = std::tie(pt_tensor, ut_tensor);
        auto outputs = std::tie(ret);
        (*helper_)(inputs, outputs);
        return ret(0);
    }


private:
    std::shared_ptr<narf::tflite_helper> helper_;
};





}


