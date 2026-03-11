#ifndef WREMNANTS_MUON_EFFICIENCIES_SMOOTH_H
#define WREMNANTS_MUON_EFFICIENCIES_SMOOTH_H

#include "defines.hpp"
#include <array>
#include <boost/histogram/axis.hpp>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>

namespace wrem {

// TODO use enums for integer/boolean/category axes so that the code is less
// error-prone?

template <int NSysts, typename HIST_SF>
class muon_efficiency_smooth_helper_base {
public:
  muon_efficiency_smooth_helper_base(HIST_SF &&sf_all)
      : sf_all_(std::make_shared<const HIST_SF>(std::move(sf_all))) {}

  std::array<double, 5>
  scale_factor_array(int pt_idx, int eta_idx, int sapt_idx, int saeta_idx,
                     int charge_idx, bool pass_iso, bool pass_trigger,
                     bool iso_with_trigger, int idx_nom_alt) const {

    auto const eff_type_idx_reco = idx_reco_;
    auto const eff_type_idx_tracking = idx_tracking_;
    auto const eff_type_idx_idip = idx_idip_;
    auto const eff_type_idx_trig = pass_trigger ? idx_trig_ : idx_antitrig_;
    auto const eff_type_idx_iso_pass =
        iso_with_trigger
            ? (pass_trigger ? idx_iso_triggering_ : idx_iso_antitriggering_)
            : idx_iso_nontriggering_;
    auto const eff_type_idx_iso_fail =
        idx_antiiso_triggering_; // for now we never consider the case with
                                 // failing isolation and failed trigger
                                 // (antiiso only needed for Wmass)
    auto const eff_type_idx_iso =
        pass_iso ? eff_type_idx_iso_pass : eff_type_idx_iso_fail;

    const double reco =
        sf_all_->at(eta_idx, pt_idx, charge_idx, eff_type_idx_reco, idx_nom_alt)
            .value();
    const double tracking = sf_all_
                                ->at(saeta_idx, sapt_idx, charge_idx,
                                     eff_type_idx_tracking, idx_nom_alt)
                                .value();
    const double idip =
        sf_all_->at(eta_idx, pt_idx, charge_idx, eff_type_idx_idip, idx_nom_alt)
            .value();
    double trig = 1.0;
    if (iso_with_trigger)
      trig =
          sf_all_
              ->at(eta_idx, pt_idx, charge_idx, eff_type_idx_trig, idx_nom_alt)
              .value();
    const double iso =
        sf_all_->at(eta_idx, pt_idx, charge_idx, eff_type_idx_iso, idx_nom_alt)
            .value();

    std::array<double, 5> ret = {reco, tracking, idip, trig, iso};
    // for(int i = 0; i < ret.size(); i++)
    //     std::cout << "Scale factor i = " << i << " --> " << ret[i] <<
    //     std::endl;

    return ret;
  }

  double scale_factor_product(float pt, float eta, float sapt, float saeta,
                              int charge, bool pass_iso, bool pass_trigger,
                              bool iso_with_trigger, int idx_nom_alt) const {

    auto const eta_idx = sf_all_->template axis<0>().index(eta);
    auto const pt_idx = sf_all_->template axis<1>().index(pt);
    auto const charge_idx = sf_all_->template axis<2>().index(charge);
    auto const saeta_idx = sf_all_->template axis<0>().index(saeta);
    auto const sapt_idx = sf_all_->template axis<1>().index(sapt);

    std::array<double, 5> allSF = scale_factor_array(
        pt_idx, eta_idx, sapt_idx, saeta_idx, charge_idx, pass_iso,
        pass_trigger, iso_with_trigger, idx_nom_alt);
    double sf = 1.0;
    for (int i = 0; i < allSF.size(); i++) {
      // std::cout << "Scale factor i = " << i << " --> " << allSF[i] <<
      // std::endl;
      sf *= allSF[i];
    }
    // std::cout << "Scale factor product " << sf << std::endl;
    return sf;
  }

  using syst_tensor_t = Eigen::TensorFixedSize<
      double, Eigen::Sizes<5, NSysts>>; // 5 bins for reco, tracking, idip,
                                        // trigger, iso(notrig) in this order

  syst_tensor_t sf_syst_var(float pt, float eta, float sapt, float saeta,
                            int charge, bool pass_iso, bool pass_trigger,
                            bool iso_with_trigger) const {

    syst_tensor_t res;

    auto const eta_idx = sf_all_->template axis<0>().index(eta);
    auto const pt_idx = sf_all_->template axis<1>().index(pt);
    auto const saeta_idx = sf_all_->template axis<0>().index(saeta);
    auto const sapt_idx = sf_all_->template axis<1>().index(sapt);
    auto const charge_idx = sf_all_->template axis<2>().index(charge);

    std::array<double, 5> allSF_nomi =
        scale_factor_array(pt_idx, eta_idx, sapt_idx, saeta_idx, charge_idx,
                           pass_iso, pass_trigger, iso_with_trigger, idx_nom_);

    for (int ns = 0; ns < NSysts; ns++) {

      std::array<double, 5> allSF_alt = scale_factor_array(
          pt_idx, eta_idx, sapt_idx, saeta_idx, charge_idx, pass_iso,
          pass_trigger, iso_with_trigger,
          sf_all_->template axis<4>().index(
              ns + 1)); // 0 is the nominal, systs starts from 1

      // anticorrelation between iso and antiiso already embedded in the numbers
      // stored in the histograms also the alternate comes from data efficiency
      // variation only, so the anticorrelation in the efficiencies is preserved
      // in the scale factors

      // order is reco-tracking-idip-trigger-iso
      for (int i = 0; i < allSF_nomi.size(); i++) {
        // if (allSF_nomi[i] <= 0)
        //     std::cout << "allSF_nomi/alt[" << i << "] = " << allSF_nomi[i] <<
        //     "/" << allSF_alt[i] << " --> pt/eta/charge = " << pt << "/" <<
        //     eta << "/" << charge << std::endl;
        res(i, ns) = allSF_alt[i] / allSF_nomi[i];
      }
    }

    return res;
  }

protected:
  std::shared_ptr<const HIST_SF> sf_all_;
  // cache the bin indices since the string category lookup is slow
  int idx_reco_ = sf_all_->template axis<3>().index("reco");
  int idx_tracking_ = sf_all_->template axis<3>().index("tracking");
  int idx_idip_ = sf_all_->template axis<3>().index("idip");
  int idx_trig_ = sf_all_->template axis<3>().index("trigger");
  int idx_antitrig_ = sf_all_->template axis<3>().index("antitrigger");
  int idx_iso_triggering_ = sf_all_->template axis<3>().index("iso");
  int idx_antiiso_triggering_ = sf_all_->template axis<3>().index("antiiso");
  int idx_iso_nontriggering_ = sf_all_->template axis<3>().index("isonotrig");
  int idx_iso_antitriggering_ =
      sf_all_->template axis<3>().index("isoantitrig");

  int idx_nom_ = sf_all_->template axis<4>().index(0);
  // int idx_alt_ = sf_all_->template axis<4>().index(NSysts);
};

// base template for one-lepton case
template <AnalysisType analysisType, int NSysts, typename HIST_SF>
class muon_efficiency_smooth_helper
    : public muon_efficiency_smooth_helper_base<NSysts, HIST_SF> {

public:
  using base_t = muon_efficiency_smooth_helper_base<NSysts, HIST_SF>;
  // inherit constructor
  using base_t::base_t;

  muon_efficiency_smooth_helper(const base_t &other) : base_t(other) {}

  double operator()(float pt, float eta, float sapt, float saeta, int charge,
                    bool pass_iso) {
    constexpr bool iso_with_trigger = true;
    constexpr bool pass_trigger = true;
    return base_t::scale_factor_product(pt, eta, sapt, saeta, charge, pass_iso,
                                        pass_trigger, iso_with_trigger,
                                        base_t::idx_nom_);
  }
};

// specialization for two-lepton case Wlike
template <int NSysts, typename HIST_SF>
class muon_efficiency_smooth_helper<AnalysisType::Wlike, NSysts, HIST_SF>
    : public muon_efficiency_smooth_helper_base<NSysts, HIST_SF> {

public:
  using base_t = muon_efficiency_smooth_helper_base<NSysts, HIST_SF>;
  // inherit constructor
  using base_t::base_t;

  muon_efficiency_smooth_helper(const base_t &other) : base_t(other) {}

  double operator()(float trig_pt, float trig_eta, float trig_sapt,
                    float trig_saeta, int trig_charge, bool trig_passiso,
                    float nontrig_pt, float nontrig_eta, float nontrig_sapt,
                    float nontrig_saeta, int nontrig_charge,
                    bool nontrig_passiso) {
    constexpr bool iso_with_trigger = true;
    constexpr bool pass_trigger =
        true; // can be true also for second lepton, since it will be overridden
              // by iso_without_trigger anyway
    constexpr bool iso_without_trigger = false;
    const double sftrig = base_t::scale_factor_product(
        trig_pt, trig_eta, trig_sapt, trig_saeta, trig_charge, trig_passiso,
        pass_trigger, iso_with_trigger, base_t::idx_nom_);
    const double sfnontrig = base_t::scale_factor_product(
        nontrig_pt, nontrig_eta, nontrig_sapt, nontrig_saeta, nontrig_charge,
        nontrig_passiso, pass_trigger, iso_without_trigger, base_t::idx_nom_);
    return sftrig * sfnontrig;
  }
};

// specialization for two-lepton case Dilepton
template <int NSysts, typename HIST_SF>
class muon_efficiency_smooth_helper<AnalysisType::Dilepton, NSysts, HIST_SF>
    : public muon_efficiency_smooth_helper_base<NSysts, HIST_SF> {

public:
  using base_t = muon_efficiency_smooth_helper_base<NSysts, HIST_SF>;
  // inherit constructor
  using base_t::base_t;

  muon_efficiency_smooth_helper(const base_t &other) : base_t(other) {}

  // may also assume the first is the one passing the trigger for sure, but it
  // depends on how these operators are called in the loop keeping both flags is
  // redundant but more flexible since there is no assumption on the sorting
  double operator()(float first_pt, float first_eta, float first_sapt,
                    float first_saeta, int first_charge, bool first_passiso,
                    bool first_passtrigger, float second_pt, float second_eta,
                    float second_sapt, float second_saeta, int second_charge,
                    bool second_passiso, bool second_passtrigger) {
    constexpr bool iso_with_trigger =
        true; // will be P(iso|passTrigger) or P(iso|failTrigger) depending on
              // first_passtrigger and second_passtrigger
    const double sftrig = base_t::scale_factor_product(
        first_pt, first_eta, first_sapt, first_saeta, first_charge,
        first_passiso, first_passtrigger, iso_with_trigger, base_t::idx_nom_);
    const double sfnontrig = base_t::scale_factor_product(
        second_pt, second_eta, second_sapt, second_saeta, second_charge,
        second_passiso, second_passtrigger, iso_with_trigger, base_t::idx_nom_);
    return sftrig * sfnontrig;
  }
};

// Now the syst, which is similar to the nominal
//
// base template for one lepton case
template <AnalysisType analysisType, int NSysts, typename HIST_SF>
class muon_efficiency_smooth_helper_syst
    : public muon_efficiency_smooth_helper_base<NSysts, HIST_SF> {

public:
  using base_t = muon_efficiency_smooth_helper_base<NSysts, HIST_SF>;
  using tensor_t = typename base_t::syst_tensor_t;

  // inherit constructor
  using base_t::base_t;

  muon_efficiency_smooth_helper_syst(const base_t &other) : base_t(other) {}

  tensor_t operator()(float pt, float eta, float sapt, float saeta, int charge,
                      bool pass_iso, double nominal_weight = 1.0) {
    constexpr bool iso_with_trigger = true;
    constexpr bool pass_trigger = true;
    return nominal_weight * base_t::sf_syst_var(pt, eta, sapt, saeta, charge,
                                                pass_iso, pass_trigger,
                                                iso_with_trigger);
  }
};

// specialization for two-lepton case Wlike
template <int NSysts, typename HIST_SF>
class muon_efficiency_smooth_helper_syst<AnalysisType::Wlike, NSysts, HIST_SF>
    : public muon_efficiency_smooth_helper_base<NSysts, HIST_SF> {

public:
  using base_t = muon_efficiency_smooth_helper_base<NSysts, HIST_SF>;
  using tensor_t = typename base_t::syst_tensor_t;

  // inherit constructor
  using base_t::base_t;

  muon_efficiency_smooth_helper_syst(const base_t &other) : base_t(other) {}

  tensor_t operator()(float trig_pt, float trig_eta, float trig_sapt,
                      float trig_saeta, int trig_charge, bool trig_passiso,
                      float nontrig_pt, float nontrig_eta, float nontrig_sapt,
                      float nontrig_saeta, int nontrig_charge,
                      bool nontrig_passiso, double nominal_weight = 1.0) {
    constexpr bool iso_with_trigger = true;
    constexpr bool pass_trigger =
        true; // can be true also for second lepton, since it will be overridden
              // by iso_without_trigger anyway
    constexpr bool iso_without_trigger = false;
    const tensor_t variation_trig = base_t::sf_syst_var(
        trig_pt, trig_eta, trig_sapt, trig_saeta, trig_charge, trig_passiso,
        pass_trigger, iso_with_trigger);
    const tensor_t variation_nontrig = base_t::sf_syst_var(
        nontrig_pt, nontrig_eta, nontrig_sapt, nontrig_saeta, nontrig_charge,
        nontrig_passiso, pass_trigger, iso_without_trigger);
    return nominal_weight * variation_trig * variation_nontrig;
  }
};

// specialization for two-lepton case Dilepton
template <int NSysts, typename HIST_SF>
class muon_efficiency_smooth_helper_syst<AnalysisType::Dilepton, NSysts,
                                         HIST_SF>
    : public muon_efficiency_smooth_helper_base<NSysts, HIST_SF> {

public:
  using base_t = muon_efficiency_smooth_helper_base<NSysts, HIST_SF>;
  using tensor_t = typename base_t::syst_tensor_t;

  // inherit constructor
  using base_t::base_t;

  muon_efficiency_smooth_helper_syst(const base_t &other) : base_t(other) {}

  // may also assume the first is the one passing the trigger for sure, but it
  // depends on how these operators are called in the loop keeping both flags is
  // redundant but more flexible since there is no assumption on the sorting
  tensor_t operator()(float first_pt, float first_eta, float first_sapt,
                      float first_saeta, int first_charge, bool first_passiso,
                      bool first_passtrigger, float second_pt, float second_eta,
                      float second_sapt, float second_saeta, int second_charge,
                      bool second_passiso, bool second_passtrigger,
                      double nominal_weight = 1.0) {
    constexpr bool iso_with_trigger =
        true; // will be P(iso|passTrigger) or P(iso|failTrigger) depending on
              // first_passtrigger and second_passtrigger
    const tensor_t variation_trig = base_t::sf_syst_var(
        first_pt, first_eta, first_sapt, first_saeta, first_charge,
        first_passiso, first_passtrigger, iso_with_trigger);
    const tensor_t variation_nontrig = base_t::sf_syst_var(
        second_pt, second_eta, second_sapt, second_saeta, second_charge,
        second_passiso, second_passtrigger, iso_with_trigger);
    return nominal_weight * variation_trig * variation_nontrig;
  }
};

//////////////////
//
// for 3D smoothed SF (iso/trigger), keep separate from original version with
// only 2D SF
//
//////////////////
template <int NSysts, typename HIST_SF, typename HIST_SF3D>
class muon_efficiency_smooth3D_helper_base {
public:
  muon_efficiency_smooth3D_helper_base(HIST_SF &&sf_all, HIST_SF3D &&sf3D_all)
      : sf_all_(std::make_shared<const HIST_SF>(std::move(sf_all))),
        sf3D_all_(std::make_shared<const HIST_SF3D>(std::move(sf3D_all))) {}

  std::array<double, 5> scale_factor_array(int pt_idx, int eta_idx,
                                           int sapt_idx, int saeta_idx,
                                           int ut_idx, int charge_idx,
                                           bool pass_iso, bool pass_trigger,
                                           bool iso_with_trigger,
                                           int idx_nom_alt) const {

    auto const eff_type_idx_reco = idx_reco_;
    auto const eff_type_idx_tracking = idx_tracking_;
    auto const eff_type_idx_idip = idx_idip_;
    auto const eff_type_idx_trig = pass_trigger ? idx3D_trig_ : idx3D_antitrig_;
    auto const eff_type_idx_iso_pass =
        iso_with_trigger
            ? (pass_trigger ? idx3D_iso_triggering_ : idx3D_iso_antitriggering_)
            : idx3D_iso_nontriggering_;
    auto const eff_type_idx_iso_fail =
        idx3D_antiiso_triggering_; // for now we never consider the case with
                                   // failing isolation and failed trigger
                                   // (antiiso only needed for Wmass)
    auto const eff_type_idx_iso =
        pass_iso ? eff_type_idx_iso_pass : eff_type_idx_iso_fail;

    const double reco =
        sf_all_->at(eta_idx, pt_idx, charge_idx, eff_type_idx_reco, idx_nom_alt)
            .value();
    const double tracking = sf_all_
                                ->at(saeta_idx, sapt_idx, charge_idx,
                                     eff_type_idx_tracking, idx_nom_alt)
                                .value();
    const double idip =
        sf_all_->at(eta_idx, pt_idx, charge_idx, eff_type_idx_idip, idx_nom_alt)
            .value();
    double trig = 1.0;
    if (iso_with_trigger)
      trig = sf3D_all_
                 ->at(eta_idx, pt_idx, charge_idx, eff_type_idx_trig,
                      idx_nom_alt, ut_idx)
                 .value();
    const double iso = sf3D_all_
                           ->at(eta_idx, pt_idx, charge_idx, eff_type_idx_iso,
                                idx_nom_alt, ut_idx)
                           .value();
    std::array<double, 5> ret = {reco, tracking, idip, trig, iso};

    return ret;
  }

  double scale_factor_product(float pt, float eta, float sapt, float saeta,
                              float ut, int charge, bool pass_iso,
                              bool pass_trigger, bool iso_with_trigger,
                              int idx_nom_alt) const {

    // FIXME: this assumes the same eta and pt binning, for 2D and 3D SF, which
    // for now is verified
    //        Keeping same binning is surely better to have, but be careful
    auto const eta_idx = sf_all_->template axis<0>().index(eta);
    auto const pt_idx = sf_all_->template axis<1>().index(pt);
    auto const charge_idx = sf_all_->template axis<2>().index(charge);
    auto const saeta_idx = sf_all_->template axis<0>().index(saeta);
    auto const sapt_idx = sf_all_->template axis<1>().index(sapt);
    auto const ut_idx = sf3D_all_->template axis<5>().index(ut);

    std::array<double, 5> allSF = scale_factor_array(
        pt_idx, eta_idx, sapt_idx, saeta_idx, ut_idx, charge_idx, pass_iso,
        pass_trigger, iso_with_trigger, idx_nom_alt);
    double sf = 1.0;
    for (int i = 0; i < allSF.size(); i++) {
      sf *= allSF[i];
    }
    return sf;
  }

  using syst_tensor_t = Eigen::TensorFixedSize<
      double, Eigen::Sizes<5, NSysts>>; // 5 bins for reco, tracking, idip,
                                        // trigger, iso(notrig) in this order

  syst_tensor_t sf_syst_var(float pt, float eta, float sapt, float saeta,
                            float ut, int charge, bool pass_iso,
                            bool pass_trigger, bool iso_with_trigger) const {

    syst_tensor_t res;

    auto const eta_idx = sf_all_->template axis<0>().index(eta);
    auto const pt_idx = sf_all_->template axis<1>().index(pt);
    auto const saeta_idx = sf_all_->template axis<0>().index(saeta);
    auto const sapt_idx = sf_all_->template axis<1>().index(sapt);
    auto const charge_idx = sf_all_->template axis<2>().index(charge);
    auto const ut_idx = sf3D_all_->template axis<5>().index(ut);

    std::array<double, 5> allSF_nomi = scale_factor_array(
        pt_idx, eta_idx, sapt_idx, saeta_idx, ut_idx, charge_idx, pass_iso,
        pass_trigger, iso_with_trigger, idx_nom_);

    for (int ns = 0; ns < NSysts; ns++) {

      std::array<double, 5> allSF_alt = scale_factor_array(
          pt_idx, eta_idx, sapt_idx, saeta_idx, ut_idx, charge_idx, pass_iso,
          pass_trigger, iso_with_trigger,
          sf_all_->template axis<4>().index(
              ns + 1)); // 0 is the nominal, systs starts from 1

      // anticorrelation between iso and antiiso already embedded in the numbers
      // stored in the histograms also the alternate comes from data efficiency
      // variation only, so the anticorrelation in the efficiencies is preserved
      // in the scale factors

      // order is reco-tracking-idip-trigger-iso
      for (int i = 0; i < allSF_nomi.size(); i++) {
        res(i, ns) = allSF_alt[i] / allSF_nomi[i];
      }
    }

    return res;
  }

protected:
  std::shared_ptr<const HIST_SF> sf_all_;
  // cache the bin indices since the string category lookup is slow
  int idx_reco_ = sf_all_->template axis<3>().index("reco");
  int idx_tracking_ = sf_all_->template axis<3>().index("tracking");
  int idx_idip_ = sf_all_->template axis<3>().index("idip");
  int idx_nom_ = sf_all_->template axis<4>().index(0);
  // now 3D SF
  // note that ut is put on the last axis to maintain the same structure of axes
  // as in the 2D case
  std::shared_ptr<const HIST_SF3D> sf3D_all_;
  int idx3D_trig_ = sf3D_all_->template axis<3>().index("trigger");
  int idx3D_antitrig_ = sf3D_all_->template axis<3>().index("antitrigger");
  int idx3D_iso_triggering_ = sf3D_all_->template axis<3>().index("iso");
  int idx3D_antiiso_triggering_ =
      sf3D_all_->template axis<3>().index("antiiso");
  int idx3D_iso_nontriggering_ =
      sf3D_all_->template axis<3>().index("isonotrig");
  int idx3D_iso_antitriggering_ =
      sf3D_all_->template axis<3>().index("isoantitrig");
};

// base template for one-lepton case
template <AnalysisType analysisType, int NSysts, typename HIST_SF,
          typename HIST_SF3D>
class muon_efficiency_smooth3D_helper
    : public muon_efficiency_smooth3D_helper_base<NSysts, HIST_SF, HIST_SF3D> {

public:
  using base_t =
      muon_efficiency_smooth3D_helper_base<NSysts, HIST_SF, HIST_SF3D>;
  // inherit constructor
  using base_t::base_t;

  muon_efficiency_smooth3D_helper(const base_t &other) : base_t(other) {}

  double operator()(float pt, float eta, float sapt, float saeta, float ut,
                    int charge, bool pass_iso) {
    constexpr bool iso_with_trigger = true;
    constexpr bool pass_trigger = true;
    return base_t::scale_factor_product(pt, eta, sapt, saeta, ut, charge,
                                        pass_iso, pass_trigger,
                                        iso_with_trigger, base_t::idx_nom_);
  }
};

// specialization for two-lepton case Wlike
template <int NSysts, typename HIST_SF, typename HIST_SF3D>
class muon_efficiency_smooth3D_helper<AnalysisType::Wlike, NSysts, HIST_SF,
                                      HIST_SF3D>
    : public muon_efficiency_smooth3D_helper_base<NSysts, HIST_SF, HIST_SF3D> {

public:
  using base_t =
      muon_efficiency_smooth3D_helper_base<NSysts, HIST_SF, HIST_SF3D>;
  // inherit constructor
  using base_t::base_t;

  muon_efficiency_smooth3D_helper(const base_t &other) : base_t(other) {}

  double operator()(float trig_pt, float trig_eta, float trig_sapt,
                    float trig_saeta, float trig_ut, int trig_charge,
                    bool trig_passiso, float nontrig_pt, float nontrig_eta,
                    float nontrig_sapt, float nontrig_saeta, float nontrig_ut,
                    int nontrig_charge, bool nontrig_passiso) {
    constexpr bool iso_with_trigger = true;
    constexpr bool pass_trigger =
        true; // can be true also for second lepton, since it will be overridden
              // by iso_without_trigger anyway
    constexpr bool iso_without_trigger = false;
    const double sftrig = base_t::scale_factor_product(
        trig_pt, trig_eta, trig_sapt, trig_saeta, trig_ut, trig_charge,
        trig_passiso, pass_trigger, iso_with_trigger, base_t::idx_nom_);
    const double sfnontrig = base_t::scale_factor_product(
        nontrig_pt, nontrig_eta, nontrig_sapt, nontrig_saeta, nontrig_ut,
        nontrig_charge, nontrig_passiso, pass_trigger, iso_without_trigger,
        base_t::idx_nom_);
    return sftrig * sfnontrig;
  }
};

// specialization for two-lepton case Dilepton
template <int NSysts, typename HIST_SF, typename HIST_SF3D>
class muon_efficiency_smooth3D_helper<AnalysisType::Dilepton, NSysts, HIST_SF,
                                      HIST_SF3D>
    : public muon_efficiency_smooth3D_helper_base<NSysts, HIST_SF, HIST_SF3D> {

public:
  using base_t =
      muon_efficiency_smooth3D_helper_base<NSysts, HIST_SF, HIST_SF3D>;
  // inherit constructor
  using base_t::base_t;

  muon_efficiency_smooth3D_helper(const base_t &other) : base_t(other) {}

  double operator()(float first_pt, float first_eta, float first_sapt,
                    float first_saeta, float first_ut, int first_charge,
                    bool first_passiso, bool first_passtrigger, float second_pt,
                    float second_eta, float second_sapt, float second_saeta,
                    float second_ut, int second_charge, bool second_passiso,
                    bool second_passtrigger) {
    constexpr bool iso_with_trigger =
        true; // will be P(iso|passTrigger) or P(iso|failTrigger) depending on
              // first_passtrigger and second_passtrigger

    const double sftrig = base_t::scale_factor_product(
        first_pt, first_eta, first_sapt, first_saeta, first_ut, first_charge,
        first_passiso, first_passtrigger, iso_with_trigger, base_t::idx_nom_);
    const double sfnontrig = base_t::scale_factor_product(
        second_pt, second_eta, second_sapt, second_saeta, second_ut,
        second_charge, second_passiso, second_passtrigger, iso_with_trigger,
        base_t::idx_nom_);
    return sftrig * sfnontrig;
  }
};

// base template for one lepton case
template <AnalysisType analysisType, int NSysts, typename HIST_SF,
          typename HIST_SF3D>
class muon_efficiency_smooth3D_helper_syst
    : public muon_efficiency_smooth3D_helper_base<NSysts, HIST_SF, HIST_SF3D> {

public:
  using base_t =
      muon_efficiency_smooth3D_helper_base<NSysts, HIST_SF, HIST_SF3D>;
  using tensor_t = typename base_t::syst_tensor_t;

  // inherit constructor
  using base_t::base_t;

  muon_efficiency_smooth3D_helper_syst(const base_t &other) : base_t(other) {}

  tensor_t operator()(float pt, float eta, float sapt, float saeta, float ut,
                      int charge, bool pass_iso, double nominal_weight = 1.0) {
    constexpr bool iso_with_trigger = true;
    constexpr bool pass_trigger = true;
    return nominal_weight * base_t::sf_syst_var(pt, eta, sapt, saeta, ut,
                                                charge, pass_iso, pass_trigger,
                                                iso_with_trigger);
  }
};

// specialization for two-lepton case Wlike
template <int NSysts, typename HIST_SF, typename HIST_SF3D>
class muon_efficiency_smooth3D_helper_syst<AnalysisType::Wlike, NSysts, HIST_SF,
                                           HIST_SF3D>
    : public muon_efficiency_smooth3D_helper_base<NSysts, HIST_SF, HIST_SF3D> {

public:
  using base_t =
      muon_efficiency_smooth3D_helper_base<NSysts, HIST_SF, HIST_SF3D>;
  using tensor_t = typename base_t::syst_tensor_t;

  // inherit constructor
  using base_t::base_t;

  muon_efficiency_smooth3D_helper_syst(const base_t &other) : base_t(other) {}

  tensor_t operator()(float trig_pt, float trig_eta, float trig_sapt,
                      float trig_saeta, float trig_ut, int trig_charge,
                      bool trig_passiso, float nontrig_pt, float nontrig_eta,
                      float nontrig_sapt, float nontrig_saeta, float nontrig_ut,
                      int nontrig_charge, bool nontrig_passiso,
                      double nominal_weight = 1.0) {
    constexpr bool iso_with_trigger = true;
    constexpr bool pass_trigger =
        true; // can be true also for second lepton, since it will be overridden
              // by iso_without_trigger anyway
    constexpr bool iso_without_trigger = false;
    const tensor_t variation_trig = base_t::sf_syst_var(
        trig_pt, trig_eta, trig_sapt, trig_saeta, trig_ut, trig_charge,
        trig_passiso, pass_trigger, iso_with_trigger);
    const tensor_t variation_nontrig = base_t::sf_syst_var(
        nontrig_pt, nontrig_eta, nontrig_sapt, nontrig_saeta, nontrig_ut,
        nontrig_charge, nontrig_passiso, pass_trigger, iso_without_trigger);
    return nominal_weight * variation_trig * variation_nontrig;
  }
};

// specialization for two-lepton case Dilepton
template <int NSysts, typename HIST_SF, typename HIST_SF3D>
class muon_efficiency_smooth3D_helper_syst<AnalysisType::Dilepton, NSysts,
                                           HIST_SF, HIST_SF3D>
    : public muon_efficiency_smooth3D_helper_base<NSysts, HIST_SF, HIST_SF3D> {

public:
  using base_t =
      muon_efficiency_smooth3D_helper_base<NSysts, HIST_SF, HIST_SF3D>;
  using tensor_t = typename base_t::syst_tensor_t;

  // inherit constructor
  using base_t::base_t;

  muon_efficiency_smooth3D_helper_syst(const base_t &other) : base_t(other) {}

  tensor_t operator()(float first_pt, float first_eta, float first_sapt,
                      float first_saeta, float first_ut, int first_charge,
                      bool first_passiso, bool first_passtrigger,
                      float second_pt, float second_eta, float second_sapt,
                      float second_saeta, float second_ut, int second_charge,
                      bool second_passiso, bool second_passtrigger,
                      double nominal_weight = 1.0) {
    constexpr bool iso_with_trigger =
        true; // will be P(iso|passTrigger) or P(iso|failTrigger) depending on
              // first_passtrigger and second_passtrigger
    const tensor_t variation_trig = base_t::sf_syst_var(
        first_pt, first_eta, first_sapt, first_saeta, first_ut, first_charge,
        first_passiso, first_passtrigger, iso_with_trigger);
    const tensor_t variation_nontrig = base_t::sf_syst_var(
        second_pt, second_eta, second_sapt, second_saeta, second_ut,
        second_charge, second_passiso, second_passtrigger, iso_with_trigger);
    return nominal_weight * variation_trig * variation_nontrig;
  }
};

////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////

////
// STAT UNCERTAINTY
// this is now an independent class with respect to the previous one which only
// deals with nominal and statistical variations
////
//// BASE CLASS FOR HELPER_STAT
template <int NEtaBins, int NPtEigenBins, int NCharges, typename HIST_SF>
class muon_efficiency_smooth_helper_stat_base {
public:
  muon_efficiency_smooth_helper_stat_base(HIST_SF &&sf_type)
      : sf_type_(std::make_shared<const HIST_SF>(std::move(sf_type))) {}

  // number of eta bins, number of eigen variations for pt axis, then 2 charges
  using stat_tensor_t =
      Eigen::TensorFixedSize<double,
                             Eigen::Sizes<NEtaBins, NPtEigenBins, NCharges>>;

  int checkEffTypeInAxis(boost::histogram::axis::category<std::string> axis,
                         const std::string &match = "match") {
    int ret = -1;
    for (Int_t i = 0; i < axis.size(); i++) {
      if (match == axis.value(i)) {
        ret = i;
        break;
      }
    }
    return ret;
  }

  // general case with no isolation (and no uT dependence)
  // TODO: develop separate method for trigger specifically? Or even merge them
  // all again?
  stat_tensor_t sf_stat_var(float pt, float eta, int charge,
                            bool pass_trigger) const {
    stat_tensor_t res;
    res.setConstant(1.0);

    auto const eta_idx = sf_type_->template axis<0>().index(eta);
    auto const pt_idx = sf_type_->template axis<1>().index(pt);
    auto const charge_idx = sf_type_->template axis<2>().index(charge);
    auto const eff_type_idx =
        isTriggerStep_ ? (pass_trigger ? idx_trig_ : idx_antitrig_) : 0;
    auto const eigen_axis = sf_type_->template axis<4>();

    // overflow/underflow are attributed to adjacent bin
    auto const tensor_eta_idx = std::clamp(eta_idx, 0, NEtaBins - 1);

    auto const &cell_nomi =
        sf_type_->at(eta_idx, pt_idx, charge_idx, eff_type_idx, idx_nom_);

    const double sf_nomi = cell_nomi.value();
    // loop on dimension with the parameters variations (the histogram already
    // has the alternate SF) start from 1 because first bin contains the nominal
    // SF
    for (int tensor_eigen_idx = 1; tensor_eigen_idx <= NPtEigenBins;
         tensor_eigen_idx++) {

      auto const eigen_axis_idx = eigen_axis.index(tensor_eigen_idx);

      auto const &cell_stat = sf_type_->at(eta_idx, pt_idx, charge_idx,
                                           eff_type_idx, eigen_axis_idx);

      const double sf_stat = cell_stat.value();
      const double sf_stat_variation = sf_stat / sf_nomi;

      res(tensor_eta_idx, tensor_eigen_idx - 1, charge_idx) *=
          sf_stat_variation;
    }

    return res;
  }

  // general case with no isolation (with uT dependence)
  stat_tensor_t sf_stat_var(float pt, float eta, float ut, int charge,
                            bool pass_trigger) const {
    stat_tensor_t res;
    res.setConstant(1.0);

    auto const eta_idx = sf_type_->template axis<0>().index(eta);
    auto const pt_idx = sf_type_->template axis<1>().index(pt);
    auto const charge_idx = sf_type_->template axis<2>().index(charge);
    auto const eff_type_idx =
        isTriggerStep_ ? (pass_trigger ? idx_trig_ : idx_antitrig_) : 0;
    auto const eigen_axis = sf_type_->template axis<4>();
    auto const ut_idx = sf_type_->template axis<5>().index(ut);

    // overflow/underflow are attributed to adjacent bin
    auto const tensor_eta_idx = std::clamp(eta_idx, 0, NEtaBins - 1);

    auto const &cell_nomi = sf_type_->at(eta_idx, pt_idx, charge_idx,
                                         eff_type_idx, idx_nom_, ut_idx);

    const double sf_nomi = cell_nomi.value();
    // loop on dimension with the parameters variations (the histogram already
    // has the alternate SF) start from 1 because first bin contains the nominal
    // SF
    for (int tensor_eigen_idx = 1; tensor_eigen_idx <= NPtEigenBins;
         tensor_eigen_idx++) {

      auto const eigen_axis_idx = eigen_axis.index(tensor_eigen_idx);

      auto const &cell_stat = sf_type_->at(
          eta_idx, pt_idx, charge_idx, eff_type_idx, eigen_axis_idx, ut_idx);

      const double sf_stat = cell_stat.value();
      const double sf_stat_variation = sf_stat / sf_nomi;

      res(tensor_eta_idx, tensor_eigen_idx - 1, charge_idx) *=
          sf_stat_variation;
    }

    return res;
  }

  // special case for isolation (and no uT dependence)
  stat_tensor_t sf_stat_var_iso(float pt, float eta, int charge, bool pass_iso,
                                bool pass_trigger,
                                bool iso_with_trigger) const {
    stat_tensor_t res;
    res.setConstant(1.0);

    auto const eta_idx = sf_type_->template axis<0>().index(eta);
    auto const pt_idx = sf_type_->template axis<1>().index(pt);
    auto const charge_idx = sf_type_->template axis<2>().index(charge);
    auto const eigen_axis = sf_type_->template axis<4>();

    auto const eff_type_idx_iso_pass =
        iso_with_trigger
            ? (pass_trigger ? idx_iso_triggering_ : idx_iso_antitriggering_)
            : idx_iso_nontriggering_;
    auto const eff_type_idx_iso_fail = idx_antiiso_triggering_;
    auto const eff_type_idx_iso =
        pass_iso ? eff_type_idx_iso_pass : eff_type_idx_iso_fail;

    // overflow/underflow are attributed to adjacent bin
    auto const tensor_eta_idx = std::clamp(eta_idx, 0, NEtaBins - 1);

    auto const &cell_nomi =
        sf_type_->at(eta_idx, pt_idx, charge_idx, eff_type_idx_iso, idx_nom_);

    const double sf_nomi = cell_nomi.value();
    // loop on dimension with the parameters variations (the histogram already
    // has the alternate SF) start from 1 because first bin contains the nominal
    // SF
    for (int tensor_eigen_idx = 1; tensor_eigen_idx <= NPtEigenBins;
         tensor_eigen_idx++) {

      auto const eigen_axis_idx = eigen_axis.index(tensor_eigen_idx);

      auto const &cell_stat = sf_type_->at(eta_idx, pt_idx, charge_idx,
                                           eff_type_idx_iso, eigen_axis_idx);

      const double sf_stat = cell_stat.value();
      const double sf_stat_variation = sf_stat / sf_nomi;
      res(tensor_eta_idx, tensor_eigen_idx - 1, charge_idx) *=
          sf_stat_variation;
    }

    return res;
  }

  // special case for isolation with ut dependence
  stat_tensor_t sf_stat_var_iso(float pt, float eta, float ut, int charge,
                                bool pass_iso, bool pass_trigger,
                                bool iso_with_trigger) const {
    stat_tensor_t res;
    res.setConstant(1.0);

    auto const eta_idx = sf_type_->template axis<0>().index(eta);
    auto const pt_idx = sf_type_->template axis<1>().index(pt);
    auto const charge_idx = sf_type_->template axis<2>().index(charge);
    auto const eigen_axis = sf_type_->template axis<4>();
    auto const ut_idx = sf_type_->template axis<5>().index(ut);

    auto const eff_type_idx_iso_pass =
        iso_with_trigger
            ? (pass_trigger ? idx_iso_triggering_ : idx_iso_antitriggering_)
            : idx_iso_nontriggering_;
    auto const eff_type_idx_iso_fail = idx_antiiso_triggering_;
    auto const eff_type_idx_iso =
        pass_iso ? eff_type_idx_iso_pass : eff_type_idx_iso_fail;

    // overflow/underflow are attributed to adjacent bin
    auto const tensor_eta_idx = std::clamp(eta_idx, 0, NEtaBins - 1);

    auto const &cell_nomi = sf_type_->at(eta_idx, pt_idx, charge_idx,
                                         eff_type_idx_iso, idx_nom_, ut_idx);

    const double sf_nomi = cell_nomi.value();
    // loop on dimension with the parameters variations (the histogram already
    // has the alternate SF) start from 1 because first bin contains the nominal
    // SF
    for (int tensor_eigen_idx = 1; tensor_eigen_idx <= NPtEigenBins;
         tensor_eigen_idx++) {

      auto const eigen_axis_idx = eigen_axis.index(tensor_eigen_idx);

      auto const &cell_stat =
          sf_type_->at(eta_idx, pt_idx, charge_idx, eff_type_idx_iso,
                       eigen_axis_idx, ut_idx);

      const double sf_stat = cell_stat.value();
      const double sf_stat_variation = sf_stat / sf_nomi;
      res(tensor_eta_idx, tensor_eigen_idx - 1, charge_idx) *=
          sf_stat_variation;
    }

    return res;
  }

protected:
  std::shared_ptr<const HIST_SF> sf_type_;
  // cache the bin indices since the string category lookup is slow
  int idx_nom_ = sf_type_->template axis<4>().index(
      0); // input effStat axis is organized as nomi - UpVar, with nomi centered
          // at 0
  // int isTriggerStep_ = sf_type_->template axis<3>().value(0) == "trigger"; //
  // special treatment for the stat variation in 2 lepton case Wlike
  int isTriggerStep_ =
      (checkEffTypeInAxis(sf_type_->template axis<3>(), "trigger") >=
       0); // special treatment for the stat variation in 2 lepton case Wlike
  // check if axis name exists in histogram, return -1 (invalid index) if not
  // found
  int idx_trig_ = checkEffTypeInAxis(sf_type_->template axis<3>(), "trigger");
  int idx_antitrig_ =
      checkEffTypeInAxis(sf_type_->template axis<3>(), "antitrigger");
  int idx_iso_triggering_ =
      checkEffTypeInAxis(sf_type_->template axis<3>(), "iso");
  int idx_antiiso_triggering_ =
      checkEffTypeInAxis(sf_type_->template axis<3>(), "antiiso");
  int idx_iso_nontriggering_ =
      checkEffTypeInAxis(sf_type_->template axis<3>(), "isonotrig");
  int idx_iso_antitriggering_ =
      checkEffTypeInAxis(sf_type_->template axis<3>(), "isoantitrig");
};

////
//// General case no uT dependence (isolation is treated separately)
////

// base template for one lepton case
template <AnalysisType analysisType, int NEtaBins, int NPtEigenBins,
          int NCharges, typename HIST_SF>
class muon_efficiency_smooth_helper_stat
    : public muon_efficiency_smooth_helper_stat_base<NEtaBins, NPtEigenBins,
                                                     NCharges, HIST_SF> {

public:
  using stat_base_t =
      muon_efficiency_smooth_helper_stat_base<NEtaBins, NPtEigenBins, NCharges,
                                              HIST_SF>;
  using tensor_t = typename stat_base_t::stat_tensor_t;

  using stat_base_t::stat_base_t;

  tensor_t operator()(float pt, float eta, int charge,
                      double nominal_weight = 1.0) {
    constexpr bool pass_trigger = true;
    return nominal_weight *
           stat_base_t::sf_stat_var(pt, eta, charge, pass_trigger);
  }
};

// specialization for two-lepton case Wlike
template <int NEtaBins, int NPtEigenBins, int NCharges, typename HIST_SF>
class muon_efficiency_smooth_helper_stat<AnalysisType::Wlike, NEtaBins,
                                         NPtEigenBins, NCharges, HIST_SF>
    : public muon_efficiency_smooth_helper_stat_base<NEtaBins, NPtEigenBins,
                                                     NCharges, HIST_SF> {

public:
  using stat_base_t =
      muon_efficiency_smooth_helper_stat_base<NEtaBins, NPtEigenBins, NCharges,
                                              HIST_SF>;
  using tensor_t = typename stat_base_t::stat_tensor_t;

  using stat_base_t::stat_base_t;

  tensor_t operator()(float trig_pt, float trig_eta, int trig_charge,
                      float nontrig_pt, float nontrig_eta, int nontrig_charge,
                      double nominal_weight = 1.0) {

    constexpr bool pass_trigger = true; // can be used on second lepton since it
                                        // does nothing when step is not trigger
    const tensor_t variation_trig =
        stat_base_t::sf_stat_var(trig_pt, trig_eta, trig_charge, pass_trigger);
    if (stat_base_t::isTriggerStep_) {
      return nominal_weight * variation_trig;
    } else {
      const tensor_t variation_nontrig = stat_base_t::sf_stat_var(
          nontrig_pt, nontrig_eta, nontrig_charge, pass_trigger);
      return nominal_weight * variation_trig * variation_nontrig;
    }
  }
};

// specialization for two-lepton case Dilepton
template <int NEtaBins, int NPtEigenBins, int NCharges, typename HIST_SF>
class muon_efficiency_smooth_helper_stat<AnalysisType::Dilepton, NEtaBins,
                                         NPtEigenBins, NCharges, HIST_SF>
    : public muon_efficiency_smooth_helper_stat_base<NEtaBins, NPtEigenBins,
                                                     NCharges, HIST_SF> {

public:
  using stat_base_t =
      muon_efficiency_smooth_helper_stat_base<NEtaBins, NPtEigenBins, NCharges,
                                              HIST_SF>;
  using tensor_t = typename stat_base_t::stat_tensor_t;

  using stat_base_t::stat_base_t;

  tensor_t operator()(float first_pt, float first_eta, int first_charge,
                      bool first_passtrigger, float second_pt, float second_eta,
                      int second_charge, bool second_passtrigger,
                      double nominal_weight = 1.0) {

    const tensor_t variation_first = stat_base_t::sf_stat_var(
        first_pt, first_eta, first_charge, first_passtrigger);
    const tensor_t variation_second = stat_base_t::sf_stat_var(
        second_pt, second_eta, second_charge, second_passtrigger);
    return nominal_weight * variation_first * variation_second;
  }
};

////
//// Isolation no uT dependence
////

// base template for one lepton case
template <AnalysisType analysisType, int NEtaBins, int NPtEigenBins,
          int NCharges, typename HIST_SF>
class muon_efficiency_smooth_helper_stat_iso
    : public muon_efficiency_smooth_helper_stat_base<NEtaBins, NPtEigenBins,
                                                     NCharges, HIST_SF> {

public:
  using stat_base_t =
      muon_efficiency_smooth_helper_stat_base<NEtaBins, NPtEigenBins, NCharges,
                                              HIST_SF>;
  using tensor_t = typename stat_base_t::stat_tensor_t;

  using stat_base_t::stat_base_t;

  tensor_t operator()(float pt, float eta, int charge, bool pass_iso,
                      double nominal_weight = 1.0) {
    constexpr bool iso_with_trigger = true;
    constexpr bool pass_trigger = true;
    return nominal_weight * stat_base_t::sf_stat_var_iso(pt, eta, charge,
                                                         pass_iso, pass_trigger,
                                                         iso_with_trigger);
  }
};

// specialization for two-lepton case Wlike
template <int NEtaBins, int NPtEigenBins, int NCharges, typename HIST_SF>
class muon_efficiency_smooth_helper_stat_iso<AnalysisType::Wlike, NEtaBins,
                                             NPtEigenBins, NCharges, HIST_SF>
    : public muon_efficiency_smooth_helper_stat_base<NEtaBins, NPtEigenBins,
                                                     NCharges, HIST_SF> {

public:
  using stat_base_t =
      muon_efficiency_smooth_helper_stat_base<NEtaBins, NPtEigenBins, NCharges,
                                              HIST_SF>;
  using tensor_t = typename stat_base_t::stat_tensor_t;

  using stat_base_t::stat_base_t;

  tensor_t operator()(float trig_pt, float trig_eta, int trig_charge,
                      bool trig_passiso, float nontrig_pt, float nontrig_eta,
                      int nontrig_charge, bool nontrig_passiso,
                      double nominal_weight = 1.0) {
    constexpr bool iso_with_trigger = true;
    constexpr bool pass_trigger =
        true; // overridden by iso_without_trigger for nontrig lepton
    constexpr bool iso_without_trigger = false;

    const tensor_t variation_trig = stat_base_t::sf_stat_var_iso(
        trig_pt, trig_eta, trig_charge, trig_passiso, pass_trigger,
        iso_with_trigger);
    const tensor_t variation_nontrig = stat_base_t::sf_stat_var_iso(
        nontrig_pt, nontrig_eta, nontrig_charge, nontrig_passiso, pass_trigger,
        iso_without_trigger);
    return nominal_weight * variation_trig * variation_nontrig;
  }
};

// specialization for two-lepton case Dilepton
template <int NEtaBins, int NPtEigenBins, int NCharges, typename HIST_SF>
class muon_efficiency_smooth_helper_stat_iso<AnalysisType::Dilepton, NEtaBins,
                                             NPtEigenBins, NCharges, HIST_SF>
    : public muon_efficiency_smooth_helper_stat_base<NEtaBins, NPtEigenBins,
                                                     NCharges, HIST_SF> {

public:
  using stat_base_t =
      muon_efficiency_smooth_helper_stat_base<NEtaBins, NPtEigenBins, NCharges,
                                              HIST_SF>;
  using tensor_t = typename stat_base_t::stat_tensor_t;

  using stat_base_t::stat_base_t;

  tensor_t operator()(float first_pt, float first_eta, int first_charge,
                      bool first_passiso, bool first_passtrigger,
                      float second_pt, float second_eta, int second_charge,
                      bool second_passiso, bool second_passtrigger,
                      double nominal_weight = 1.0) {

    constexpr bool iso_with_trigger =
        true; // for both leptons, then each lepton can pass or fail the trigger

    const tensor_t variation_first = stat_base_t::sf_stat_var_iso(
        first_pt, first_eta, first_charge, first_passiso, first_passtrigger,
        iso_with_trigger);
    const tensor_t variation_second = stat_base_t::sf_stat_var_iso(
        second_pt, second_eta, second_charge, second_passiso,
        second_passtrigger, iso_with_trigger);
    return nominal_weight * variation_first * variation_second;
  }
};

/// case with ut dependence
// isolation has another special class, so this would be only trigger, but in
// fact it can be used for any step
//
// base template for one lepton case
template <AnalysisType analysisType, int NEtaBins, int NPtEigenBins,
          int NCharges, typename HIST_SF>
class muon_efficiency_smooth_helper_stat_utDep
    : public muon_efficiency_smooth_helper_stat_base<NEtaBins, NPtEigenBins,
                                                     NCharges, HIST_SF> {

public:
  using stat_base_t =
      muon_efficiency_smooth_helper_stat_base<NEtaBins, NPtEigenBins, NCharges,
                                              HIST_SF>;
  using tensor_t = typename stat_base_t::stat_tensor_t;

  using stat_base_t::stat_base_t;

  tensor_t operator()(float pt, float eta, float ut, int charge,
                      double nominal_weight = 1.0) {
    constexpr bool pass_trigger = true;
    return nominal_weight *
           stat_base_t::sf_stat_var(pt, eta, ut, charge, pass_trigger);
  }
};

// specialization for two-lepton case Wlike
template <int NEtaBins, int NPtEigenBins, int NCharges, typename HIST_SF>
class muon_efficiency_smooth_helper_stat_utDep<AnalysisType::Wlike, NEtaBins,
                                               NPtEigenBins, NCharges, HIST_SF>
    : public muon_efficiency_smooth_helper_stat_base<NEtaBins, NPtEigenBins,
                                                     NCharges, HIST_SF> {

public:
  using stat_base_t =
      muon_efficiency_smooth_helper_stat_base<NEtaBins, NPtEigenBins, NCharges,
                                              HIST_SF>;
  using tensor_t = typename stat_base_t::stat_tensor_t;

  using stat_base_t::stat_base_t;

  tensor_t operator()(float trig_pt, float trig_eta, float trig_ut,
                      int trig_charge, float nontrig_pt, float nontrig_eta,
                      float nontrig_ut, int nontrig_charge,
                      double nominal_weight = 1.0) {

    constexpr bool pass_trigger = true; // can be used on second lepton since it
                                        // does nothing when step is not trigger
    const tensor_t variation_trig = stat_base_t::sf_stat_var(
        trig_pt, trig_eta, trig_ut, trig_charge, pass_trigger);
    if (stat_base_t::isTriggerStep_) {
      return nominal_weight * variation_trig;
    } else {
      const tensor_t variation_nontrig = stat_base_t::sf_stat_var(
          nontrig_pt, nontrig_eta, nontrig_ut, nontrig_charge, pass_trigger);
      return nominal_weight * variation_trig * variation_nontrig;
    }
  }
};

// specialization for two-lepton case Dilepton
template <int NEtaBins, int NPtEigenBins, int NCharges, typename HIST_SF>
class muon_efficiency_smooth_helper_stat_utDep<AnalysisType::Dilepton, NEtaBins,
                                               NPtEigenBins, NCharges, HIST_SF>
    : public muon_efficiency_smooth_helper_stat_base<NEtaBins, NPtEigenBins,
                                                     NCharges, HIST_SF> {

public:
  using stat_base_t =
      muon_efficiency_smooth_helper_stat_base<NEtaBins, NPtEigenBins, NCharges,
                                              HIST_SF>;
  using tensor_t = typename stat_base_t::stat_tensor_t;

  using stat_base_t::stat_base_t;

  tensor_t operator()(float first_pt, float first_eta, float first_ut,
                      int first_charge, bool first_passtrigger, float second_pt,
                      float second_eta, float second_ut, int second_charge,
                      bool second_passtrigger, double nominal_weight = 1.0) {

    const tensor_t variation_first = stat_base_t::sf_stat_var(
        first_pt, first_eta, first_ut, first_charge, first_passtrigger);
    const tensor_t variation_second = stat_base_t::sf_stat_var(
        second_pt, second_eta, second_ut, second_charge, second_passtrigger);
    return nominal_weight * variation_first * variation_second;
  }
};

////
//// Isolation (with ut dependence)
////

// base template for one lepton case
template <AnalysisType analysisType, int NEtaBins, int NPtEigenBins,
          int NCharges, typename HIST_SF>
class muon_efficiency_smooth_helper_stat_iso_utDep
    : public muon_efficiency_smooth_helper_stat_base<NEtaBins, NPtEigenBins,
                                                     NCharges, HIST_SF> {

public:
  using stat_base_t =
      muon_efficiency_smooth_helper_stat_base<NEtaBins, NPtEigenBins, NCharges,
                                              HIST_SF>;
  using tensor_t = typename stat_base_t::stat_tensor_t;

  using stat_base_t::stat_base_t;

  tensor_t operator()(float pt, float eta, float ut, int charge, bool pass_iso,
                      double nominal_weight = 1.0) {
    constexpr bool iso_with_trigger = true;
    constexpr bool pass_trigger = true;
    return nominal_weight * stat_base_t::sf_stat_var_iso(pt, eta, ut, charge,
                                                         pass_iso, pass_trigger,
                                                         iso_with_trigger);
  }
};

// specialization for two-lepton case Wlike
template <int NEtaBins, int NPtEigenBins, int NCharges, typename HIST_SF>
class muon_efficiency_smooth_helper_stat_iso_utDep<
    AnalysisType::Wlike, NEtaBins, NPtEigenBins, NCharges, HIST_SF>
    : public muon_efficiency_smooth_helper_stat_base<NEtaBins, NPtEigenBins,
                                                     NCharges, HIST_SF> {

public:
  using stat_base_t =
      muon_efficiency_smooth_helper_stat_base<NEtaBins, NPtEigenBins, NCharges,
                                              HIST_SF>;
  using tensor_t = typename stat_base_t::stat_tensor_t;

  using stat_base_t::stat_base_t;

  tensor_t operator()(float trig_pt, float trig_eta, float trig_ut,
                      int trig_charge, bool trig_passiso, float nontrig_pt,
                      float nontrig_eta, float nontrig_ut, int nontrig_charge,
                      bool nontrig_passiso, double nominal_weight = 1.0) {
    constexpr bool iso_with_trigger = true;
    constexpr bool pass_trigger = true;
    constexpr bool iso_without_trigger = false;

    const tensor_t variation_trig = stat_base_t::sf_stat_var_iso(
        trig_pt, trig_eta, trig_ut, trig_charge, trig_passiso, pass_trigger,
        iso_with_trigger);
    const tensor_t variation_nontrig = stat_base_t::sf_stat_var_iso(
        nontrig_pt, nontrig_eta, nontrig_ut, nontrig_charge, nontrig_passiso,
        pass_trigger, iso_without_trigger);
    return nominal_weight * variation_trig * variation_nontrig;
  }
};

// specialization for two-lepton case Dilepton
template <int NEtaBins, int NPtEigenBins, int NCharges, typename HIST_SF>
class muon_efficiency_smooth_helper_stat_iso_utDep<
    AnalysisType::Dilepton, NEtaBins, NPtEigenBins, NCharges, HIST_SF>
    : public muon_efficiency_smooth_helper_stat_base<NEtaBins, NPtEigenBins,
                                                     NCharges, HIST_SF> {

public:
  using stat_base_t =
      muon_efficiency_smooth_helper_stat_base<NEtaBins, NPtEigenBins, NCharges,
                                              HIST_SF>;
  using tensor_t = typename stat_base_t::stat_tensor_t;

  using stat_base_t::stat_base_t;

  tensor_t operator()(float first_pt, float first_eta, float first_ut,
                      int first_charge, bool first_passiso,
                      bool first_passtrigger, float second_pt, float second_eta,
                      float second_ut, int second_charge, bool second_passiso,
                      bool second_passtrigger, double nominal_weight = 1.0) {
    constexpr bool iso_with_trigger = true;

    const tensor_t variation_first = stat_base_t::sf_stat_var_iso(
        first_pt, first_eta, first_ut, first_charge, first_passiso,
        first_passtrigger, iso_with_trigger);
    const tensor_t variation_second = stat_base_t::sf_stat_var_iso(
        second_pt, second_eta, second_ut, second_charge, second_passiso,
        second_passtrigger, iso_with_trigger);
    return nominal_weight * variation_first * variation_second;
  }
};

///////////
// ADDITIONAL UTILITY HELPERS TO MANAGE A SINGLE EFFICIENCY STEP
// originally conceived to deal with additional systematic uncertainties on the
// tracking step the "nominal" helper is also implemented so one can add special
// additional scale factors if ever useful
///////////
template <int NSysts, typename HIST_SF>
class muon_efficiency_smooth_helper_base_oneStep {
public:
  muon_efficiency_smooth_helper_base_oneStep(HIST_SF &&sf_type)
      : sf_type_(std::make_shared<const HIST_SF>(std::move(sf_type))) {}

  double scale_factor_byCell(int pt_idx, int eta_idx, int charge_idx,
                             int idx_nom_alt) const {
    const double sf = sf_type_->at(eta_idx, pt_idx, charge_idx, idx_nom_alt);
    // std::cout << "Scale factor " << sf << std::endl;
    return sf;
  }

  double scale_factor(float pt, float eta, int charge, int idx_nom_alt) const {

    auto const eta_idx = sf_type_->template axis<0>().index(eta);
    auto const pt_idx = sf_type_->template axis<1>().index(pt);
    auto const charge_idx = sf_type_->template axis<2>().index(charge);
    return scale_factor_byCell(pt_idx, eta_idx, charge_idx, idx_nom_alt);
  }

  using syst_tensor_t = Eigen::TensorFixedSize<double, Eigen::Sizes<NSysts>>;

  syst_tensor_t sf_syst_var(float pt, float eta, int charge) const {

    syst_tensor_t res;

    auto const eta_idx = sf_type_->template axis<0>().index(eta);
    auto const pt_idx = sf_type_->template axis<1>().index(pt);
    auto const charge_idx = sf_type_->template axis<2>().index(charge);

    double sf_nomi = scale_factor_byCell(pt_idx, eta_idx, charge_idx, idx_nom_);

    for (int ns = 0; ns < NSysts; ns++) {

      double sf_alt = scale_factor_byCell(
          pt_idx, eta_idx, charge_idx,
          sf_type_->template axis<3>().index(
              ns + 1)); // 0 is the nominal, systs starts from 1
      res(ns) = sf_alt / sf_nomi;
    }

    return res;
  }

protected:
  std::shared_ptr<const HIST_SF> sf_type_;
  // cache the bin indices since the string category lookup is slow
  int idx_nom_ = sf_type_->template axis<3>().index(
      0); // input effStat axis is organized as nomi - UpVar, with nomi centered
          // at 0
};

// base template for one-lepton case
template <AnalysisType analysisType, int NSysts, typename HIST_SF>
class muon_efficiency_smooth_helper_oneStep
    : public muon_efficiency_smooth_helper_base_oneStep<NSysts, HIST_SF> {

public:
  using base_t = muon_efficiency_smooth_helper_base_oneStep<NSysts, HIST_SF>;
  // inherit constructor
  using base_t::base_t;

  muon_efficiency_smooth_helper_oneStep(const base_t &other) : base_t(other) {}

  double operator()(float pt, float eta, int charge) {
    return base_t::scale_factor(pt, eta, charge, base_t::idx_nom_);
  }
};

// specialization for two-lepton case Wlike
template <int NSysts, typename HIST_SF>
class muon_efficiency_smooth_helper_oneStep<AnalysisType::Wlike, NSysts,
                                            HIST_SF>
    : public muon_efficiency_smooth_helper_base_oneStep<NSysts, HIST_SF> {

public:
  using base_t = muon_efficiency_smooth_helper_base_oneStep<NSysts, HIST_SF>;
  // inherit constructor
  using base_t::base_t;

  muon_efficiency_smooth_helper_oneStep(const base_t &other) : base_t(other) {}

  double operator()(float trig_pt, float trig_eta, int trig_charge,
                    float nontrig_pt, float nontrig_eta, int nontrig_charge) {
    const double sftrig =
        base_t::scale_factor(trig_pt, trig_eta, trig_charge, base_t::idx_nom_);
    const double sfnontrig = base_t::scale_factor(
        nontrig_pt, nontrig_eta, nontrig_charge, base_t::idx_nom_);
    return sftrig * sfnontrig;
  }
};

// specialization for two-lepton case Dilepton
template <int NSysts, typename HIST_SF>
class muon_efficiency_smooth_helper_oneStep<AnalysisType::Dilepton, NSysts,
                                            HIST_SF>
    : public muon_efficiency_smooth_helper_base_oneStep<NSysts, HIST_SF> {

public:
  using base_t = muon_efficiency_smooth_helper_base_oneStep<NSysts, HIST_SF>;
  // inherit constructor
  using base_t::base_t;

  muon_efficiency_smooth_helper_oneStep(const base_t &other) : base_t(other) {}

  // may also assume the first is the one passing the trigger for sure, but it
  // depends on how these operators are called in the loop keeping both flags is
  // redundant but more flexible since there is no assumption on the sorting
  double operator()(float first_pt, float first_eta, int first_charge,
                    float second_pt, float second_eta, int second_charge) {
    const double sftrig = base_t::scale_factor(first_pt, first_eta,
                                               first_charge, base_t::idx_nom_);
    const double sfnontrig = base_t::scale_factor(
        second_pt, second_eta, second_charge, base_t::idx_nom_);
    return sftrig * sfnontrig;
  }
};

// Now the syst, which is similar to the nominal
//
// base template for one lepton case
template <AnalysisType analysisType, int NSysts, typename HIST_SF>
class muon_efficiency_smooth_helper_syst_oneStep
    : public muon_efficiency_smooth_helper_base_oneStep<NSysts, HIST_SF> {

public:
  using base_t = muon_efficiency_smooth_helper_base_oneStep<NSysts, HIST_SF>;
  using tensor_t = typename base_t::syst_tensor_t;

  // inherit constructor
  using base_t::base_t;

  muon_efficiency_smooth_helper_syst_oneStep(const base_t &other)
      : base_t(other) {}

  tensor_t operator()(float pt, float eta, int charge,
                      double nominal_weight = 1.0) {
    return nominal_weight * base_t::sf_syst_var(pt, eta, charge);
  }
};

// specialization for two-lepton case Wlike
template <int NSysts, typename HIST_SF>
class muon_efficiency_smooth_helper_syst_oneStep<AnalysisType::Wlike, NSysts,
                                                 HIST_SF>
    : public muon_efficiency_smooth_helper_base_oneStep<NSysts, HIST_SF> {

public:
  using base_t = muon_efficiency_smooth_helper_base_oneStep<NSysts, HIST_SF>;
  using tensor_t = typename base_t::syst_tensor_t;

  // inherit constructor
  using base_t::base_t;

  muon_efficiency_smooth_helper_syst_oneStep(const base_t &other)
      : base_t(other) {}

  tensor_t operator()(float trig_pt, float trig_eta, int trig_charge,
                      float nontrig_pt, float nontrig_eta, int nontrig_charge,
                      double nominal_weight = 1.0) {
    const tensor_t variation_trig =
        base_t::sf_syst_var(trig_pt, trig_eta, trig_charge);
    const tensor_t variation_nontrig =
        base_t::sf_syst_var(nontrig_pt, nontrig_eta, nontrig_charge);
    return nominal_weight * variation_trig * variation_nontrig;
  }
};

// specialization for two-lepton case Dilepton
template <int NSysts, typename HIST_SF>
class muon_efficiency_smooth_helper_syst_oneStep<AnalysisType::Dilepton, NSysts,
                                                 HIST_SF>
    : public muon_efficiency_smooth_helper_base_oneStep<NSysts, HIST_SF> {

public:
  using base_t = muon_efficiency_smooth_helper_base_oneStep<NSysts, HIST_SF>;
  using tensor_t = typename base_t::syst_tensor_t;

  // inherit constructor
  using base_t::base_t;

  muon_efficiency_smooth_helper_syst_oneStep(const base_t &other)
      : base_t(other) {}

  // may also assume the first is the one passing the trigger for sure, but it
  // depends on how these operators are called in the loop keeping both flags is
  // redundant but more flexible since there is no assumption on the sorting
  tensor_t operator()(float first_pt, float first_eta, int first_charge,
                      float second_pt, float second_eta, int second_charge,
                      double nominal_weight = 1.0) {
    constexpr bool iso_with_trigger =
        true; // will be P(iso|passTrigger) or P(iso|failTrigger) depending on
              // first_passtrigger and second_passtrigger
    constexpr bool pass_iso = true;
    const tensor_t variation_trig =
        base_t::sf_syst_var(first_pt, first_eta, first_charge);
    const tensor_t variation_nontrig =
        base_t::sf_syst_var(second_pt, second_eta, second_charge);
    return nominal_weight * variation_trig * variation_nontrig;
  }
};

} // namespace wrem

#endif
