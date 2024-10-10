#ifndef WREMNANTS_MUON_EFFICIENCIES_VETO_H
#define WREMNANTS_MUON_EFFICIENCIES_VETO_H

#include "defines.hpp"
#include <array>
#include <boost/histogram/axis.hpp>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>

namespace wrem {

template <typename HIST_SF, int NEtaBins, int NPtEigenBins, int NSysts,
          int Steps>
class muon_efficiency_veto_helper_base {

public:
  muon_efficiency_veto_helper_base(HIST_SF &&sf_veto)
      : sf_veto_(std::make_shared<const HIST_SF>(std::move(sf_veto))) {}

  double scale_factor(float pt, float eta, int charge) const {

    if (charge <= -99)
      return 1.0;

    const int ieta = sf_veto_->template axis<0>().index(eta);
    const int ipt = sf_veto_->template axis<1>().index(pt);
    const int icharge = sf_veto_->template axis<2>().index(charge);
    const double sf = sf_veto_->at(ieta, ipt, icharge, 0).value();

    return sf;
  }

  using syst_tensor_t =
      Eigen::TensorFixedSize<double, Eigen::Sizes<Steps, NSysts>>;

  syst_tensor_t sf_syst_var(float pt, float eta, int charge) const {

    syst_tensor_t res;
    res.setConstant(1.0);
    if (charge <= -99)
      return res;

    const int ieta = sf_veto_->template axis<0>().index(eta);
    const int ipt = sf_veto_->template axis<1>().index(pt);
    const int icharge = sf_veto_->template axis<2>().index(charge);
    const double sf_nomi = sf_veto_->at(ieta, ipt, icharge, 0).value();
    for (int step = 0; step < Steps; step++) {
      for (int ns = 0; ns < NSysts; ns++) {
        // TODO: logic of next line only works if the decorrelation uses the
        // same binning as the eta axis it is done like that because the
        // histogram doesn't already store the SF for the eta decorrelated syst
        if ((ns == 0) || (ns == (ieta + 1)))
          res(step, ns) =
              sf_veto_->at(ieta, ipt, icharge, 1 + NPtEigenBins + step)
                  .value() /
              sf_nomi;
        else
          res(step, ns) = 1.;
      }
    }

    return res;
  }

  using stat_tensor_t =
      Eigen::TensorFixedSize<double, Eigen::Sizes<NEtaBins, NPtEigenBins, 2>>;

  stat_tensor_t sf_stat_var(float pt, float eta, int charge) const {
    stat_tensor_t res;
    res.setConstant(1.0);
    if (charge <= -99)
      return res;

    const int ieta = sf_veto_->template axis<0>().index(eta);
    const int ipt = sf_veto_->template axis<1>().index(pt);
    const int icharge = sf_veto_->template axis<2>().index(charge);
    const double sf_nomi = sf_veto_->at(ieta, ipt, icharge, 0).value();
    for (int tensor_eigen_idx = 1; tensor_eigen_idx <= NPtEigenBins;
         tensor_eigen_idx++) {
      res(ieta, tensor_eigen_idx - 1, icharge) *=
          sf_veto_->at(ieta, ipt, icharge, tensor_eigen_idx).value() / sf_nomi;
    }

    return res;
  }

protected:
  std::shared_ptr<const HIST_SF> sf_veto_;
};

template <typename HIST_SF, int NEtaBins, int NPtEigenBins, int NSysts,
          int Steps>
class muon_efficiency_veto_helper
    : public muon_efficiency_veto_helper_base<HIST_SF, NEtaBins, NPtEigenBins,
                                              NSysts, Steps> {

public:
  using base_t = muon_efficiency_veto_helper_base<HIST_SF, NEtaBins,
                                                  NPtEigenBins, NSysts, Steps>;

  using base_t::base_t;

  muon_efficiency_veto_helper(const base_t &other) : base_t(other) {}

  double operator()(float pt, float eta, int charge) {
    return base_t::scale_factor(pt, eta, charge);
  }
};

template <typename HIST_SF, int NEtaBins, int NPtEigenBins, int NSysts,
          int Steps>
class muon_efficiency_veto_helper_syst
    : public muon_efficiency_veto_helper_base<HIST_SF, NEtaBins, NPtEigenBins,
                                              NSysts, Steps> {

public:
  using base_t = muon_efficiency_veto_helper_base<HIST_SF, NEtaBins,
                                                  NPtEigenBins, NSysts, Steps>;
  using tensor_t = typename base_t::syst_tensor_t;

  using base_t::base_t;

  muon_efficiency_veto_helper_syst(const base_t &other) : base_t(other) {}

  tensor_t operator()(float pt, float eta, int charge,
                      double nominal_weight = 1.0) {
    return nominal_weight * base_t::sf_syst_var(pt, eta, charge);
  }
};

template <typename HIST_SF, int NEtaBins, int NPtEigenBins, int NSysts,
          int Steps>
class muon_efficiency_veto_helper_stat
    : public muon_efficiency_veto_helper_base<HIST_SF, NEtaBins, NPtEigenBins,
                                              NSysts, Steps> {

public:
  using base_t = muon_efficiency_veto_helper_base<HIST_SF, NEtaBins,
                                                  NPtEigenBins, NSysts, Steps>;
  using tensor_t = typename base_t::stat_tensor_t;

  using base_t::base_t;

  muon_efficiency_veto_helper_stat(const base_t &other) : base_t(other) {}

  tensor_t operator()(float pt, float eta, int charge,
                      double nominal_weight = 1.0) {
    return nominal_weight * base_t::sf_stat_var(pt, eta, charge);
  }
};

} // namespace wrem

#endif
