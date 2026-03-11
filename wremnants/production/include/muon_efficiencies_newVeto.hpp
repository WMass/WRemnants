#ifndef WREMNANTS_MUON_EFFICIENCIES_NEWVETO_H
#define WREMNANTS_MUON_EFFICIENCIES_NEWVETO_H

#include "defines.hpp"
#include <array>
#include <boost/histogram/axis.hpp>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>

namespace wrem {

// syst and stat are stacked in the syst axis, usually it is 1 (nominal) + 1
// (global syst) + 48 (eta-decorr syst) + 4 (pt eigen stat) NSysts is only for
// the syst part, NEtaBins and NPtEigenBins are only needed for the stat part
template <typename HIST_SF, int NSteps, int NSysts, int NEtaBins,
          int NPtEigenBins>
class muon_efficiency_newVeto_helper_base {

public:
  muon_efficiency_newVeto_helper_base(HIST_SF &&sf_all)
      : sf_all_(std::make_shared<const HIST_SF>(std::move(sf_all))) {}

  // Note that the overall SF is read using the inner track variables, even if
  // the tracking part should have used the standalone ones This is because in
  // order to derive antiveto SF one has to invert the final product, it can't
  // be done from a single step

  // the nominal SF stored in the first bin of the syst axis is the same for all
  // steps because it is already the product of all steps, while the splitting
  // by step is only really relevant for the stat/syst variations, where each
  // step produces its own uncertainty

  std::array<int, 3> pt_eta_charge_idxs_fromValues(float pt, float eta,
                                                   int charge) const {

    // clamping here should not be really necessary since the overflow bins
    // already have the SF set equal to the closest inner bins of the SF
    // histograms
    const int eta_idx =
        std::clamp(sf_all_->template axis<0>().index(eta), 0, hsfNeta_ - 1);
    const int pt_idx =
        std::clamp(sf_all_->template axis<1>().index(pt), 0, hsfNpt_ - 1);
    const int charge_idx = std::clamp(sf_all_->template axis<2>().index(charge),
                                      0, hsfNcharge_ - 1);
    std::array<int, 3> ret = {pt_idx, eta_idx, charge_idx};
    return ret;
  }

  double scale_factor_byIndex(int pt_idx, int eta_idx, int charge_idx,
                              int idx_step, int idx_nom_alt) const {

    // std::cout << "pt_idx, eta_idx, charge_idx, idx_step, idx_nom_alt = " <<
    // pt_idx << ", " << eta_idx << ", " << charge_idx << ", " << idx_step << ",
    // " << idx_nom_alt << std::endl;
    const double sf =
        sf_all_->at(eta_idx, pt_idx, charge_idx, idx_step, idx_nom_alt).value();
    return sf;
  }

  double scale_factor(float pt, float eta, int charge, int idx_step,
                      int idx_nom_alt) const {

    std::array<int, 3> pt_eta_charge_idxs =
        pt_eta_charge_idxs_fromValues(pt, eta, charge);
    auto const pt_idx = pt_eta_charge_idxs[0];
    auto const eta_idx = pt_eta_charge_idxs[1];
    auto const charge_idx = pt_eta_charge_idxs[2];

    const double sf = scale_factor_byIndex(pt_idx, eta_idx, charge_idx,
                                           idx_step, idx_nom_alt);
    return sf;
  }

  double scale_factor_nomi_byIndex(int pt_idx, int eta_idx,
                                   int charge_idx) const {

    // can read the bin correspondign to the first step, since the nominal is
    // the same for all steps (it is the product already)
    const double sf =
        scale_factor_byIndex(pt_idx, eta_idx, charge_idx, 0, idx_nom_);

    return sf;
  }

  double scale_factor_nomi(float pt, float eta, int charge) const {

    if (charge <= -99)
      return 1.0;
    std::array<int, 3> pt_eta_charge_idxs =
        pt_eta_charge_idxs_fromValues(pt, eta, charge);
    auto const pt_idx = pt_eta_charge_idxs[0];
    auto const eta_idx = pt_eta_charge_idxs[1];
    auto const charge_idx = pt_eta_charge_idxs[2];

    // can read the bin corresponding to the first step, since the nominal is
    // the same for all steps (it is the product already)
    const double sf = scale_factor_nomi_byIndex(pt_idx, eta_idx, charge_idx);
    return sf;
  }

  using syst_tensor_t =
      Eigen::TensorFixedSize<double, Eigen::Sizes<NSteps, NSysts>>;

  syst_tensor_t sf_syst_var(float pt, float eta, int charge) const {

    syst_tensor_t res;
    res.setConstant(1.0);
    if (charge <= -99)
      return res;

    std::array<int, 3> pt_eta_charge_idxs =
        pt_eta_charge_idxs_fromValues(pt, eta, charge);
    auto const pt_idx = pt_eta_charge_idxs[0];
    auto const eta_idx = pt_eta_charge_idxs[1];
    auto const charge_idx = pt_eta_charge_idxs[2];

    const double sf_nomi =
        scale_factor_nomi_byIndex(pt_idx, eta_idx, charge_idx);

    for (int nstep = 0; nstep < NSteps; nstep++) {
      for (int nsyst = 0; nsyst < NSysts; nsyst++) {
        const double sf_alt =
            scale_factor_byIndex(pt_idx, eta_idx, charge_idx, nstep,
                                 sf_all_->template axis<4>().index(nsyst + 1));
        res(nstep, nsyst) = sf_alt / sf_nomi;
      }
    }

    return res;
  }

  // usually 3 steps * 48 eta bins * 4 pt eigen bins * 2 charges (all steps are
  // currently chharge dependent
  using stat_tensor_t =
      Eigen::TensorFixedSize<double,
                             Eigen::Sizes<NSteps, NEtaBins, NPtEigenBins, 2>>;

  stat_tensor_t sf_stat_var(float pt, float eta, int charge) const {
    stat_tensor_t res;
    res.setConstant(1.0);
    if (charge <= -99)
      return res;

    std::array<int, 3> pt_eta_charge_idxs =
        pt_eta_charge_idxs_fromValues(pt, eta, charge);
    auto const pt_idx = pt_eta_charge_idxs[0];
    auto const eta_idx = pt_eta_charge_idxs[1];
    auto const charge_idx = pt_eta_charge_idxs[2];

    auto const eigen_axis = sf_all_->template axis<4>();

    // overflow/underflow are attributed to adjacent bin
    auto const tensor_eta_idx = std::clamp(eta_idx, 0, NEtaBins - 1);

    const double sf_nomi =
        scale_factor_nomi_byIndex(pt_idx, eta_idx, charge_idx);
    // loop on dimension with the parameters variations (the histogram already
    // has the alternate SF) start from 1 + NSysts because these first bins
    // contain the nominal SF + the syst variations
    int start_idx = 1 + NSysts;
    // int end_idx   = start_idx + NPtEigenBins;

    for (int nstep = 0; nstep < NSteps; nstep++) {
      for (int tensor_eigen_idx = 0; tensor_eigen_idx < NPtEigenBins;
           tensor_eigen_idx++) {

        const int eigen_axis_idx =
            eigen_axis.index(tensor_eigen_idx + start_idx);

        const double sf_stat = scale_factor_byIndex(pt_idx, eta_idx, charge_idx,
                                                    nstep, eigen_axis_idx);

        res(nstep, tensor_eta_idx, tensor_eigen_idx, charge_idx) =
            sf_stat / sf_nomi;
      }
    }
    return res;
  }

protected:
  std::shared_ptr<const HIST_SF> sf_all_;
  // cache the bin indices since the string category lookup is slow
  int idx_nom_ = sf_all_->template axis<4>().index(0);
  int hsfNeta_ = sf_all_->template axis<0>().size();
  int hsfNpt_ = sf_all_->template axis<1>().size();
  int hsfNcharge_ = sf_all_->template axis<2>().size();
};

/////
template <typename HIST_SF, int NSteps, int NSysts, int NEtaBins,
          int NPtEigenBins>
class muon_efficiency_newVeto_helper
    : public muon_efficiency_newVeto_helper_base<HIST_SF, NSteps, NSysts,
                                                 NEtaBins, NPtEigenBins> {

public:
  using base_t = muon_efficiency_newVeto_helper_base<HIST_SF, NSteps, NSysts,
                                                     NEtaBins, NPtEigenBins>;

  using base_t::base_t;

  muon_efficiency_newVeto_helper(const base_t &other) : base_t(other) {}

  double operator()(float pt, float eta, int charge) {
    return base_t::scale_factor_nomi(pt, eta, charge);
  }
};

template <typename HIST_SF, int NSteps, int NSysts, int NEtaBins,
          int NPtEigenBins>
class muon_efficiency_newVeto_helper_syst
    : public muon_efficiency_newVeto_helper_base<HIST_SF, NSteps, NSysts,
                                                 NEtaBins, NPtEigenBins> {

public:
  using base_t = muon_efficiency_newVeto_helper_base<HIST_SF, NSteps, NSysts,
                                                     NEtaBins, NPtEigenBins>;
  using tensor_t = typename base_t::syst_tensor_t;

  using base_t::base_t;

  muon_efficiency_newVeto_helper_syst(const base_t &other) : base_t(other) {}

  tensor_t operator()(float pt, float eta, int charge,
                      double nominal_weight = 1.0) {
    return nominal_weight * base_t::sf_syst_var(pt, eta, charge);
  }
};

template <typename HIST_SF, int NSteps, int NSysts, int NEtaBins,
          int NPtEigenBins>
class muon_efficiency_newVeto_helper_stat
    : public muon_efficiency_newVeto_helper_base<HIST_SF, NSteps, NSysts,
                                                 NEtaBins, NPtEigenBins> {

public:
  using base_t = muon_efficiency_newVeto_helper_base<HIST_SF, NSteps, NSysts,
                                                     NEtaBins, NPtEigenBins>;
  using tensor_t = typename base_t::stat_tensor_t;

  using base_t::base_t;

  muon_efficiency_newVeto_helper_stat(const base_t &other) : base_t(other) {}

  tensor_t operator()(float pt, float eta, int charge,
                      double nominal_weight = 1.0) {
    return nominal_weight * base_t::sf_stat_var(pt, eta, charge);
  }
};

} // namespace wrem

#endif
