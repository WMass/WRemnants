#ifndef WREMNANTS_MUON_EFFICIENCIES_H
#define WREMNANTS_MUON_EFFICIENCIES_H

namespace wrem {

// TODO use enums for integer/boolean/category axes so that the code is less error-prone?

template<int NEtaBins, int NPtBins, typename HIST_IDIPTRIGISO, typename HIST_TRACKINGRECO>
class muon_efficiency_helper_base {
public:

  muon_efficiency_helper_base(HIST_IDIPTRIGISO &&sf_idip_trig_iso, HIST_TRACKINGRECO &&sf_tracking_reco) :
    sf_idip_trig_iso_(std::make_shared<const HIST_IDIPTRIGISO>(std::move(sf_idip_trig_iso))),
    sf_tracking_reco_(std::make_shared<const HIST_TRACKINGRECO>(std::move(sf_tracking_reco))) {
      std::cout << "idx check " << idip_idx_ << std::endl;

    }

  auto const &sf_trackingreco(float eta, int charge, int idx_nom_alt) const {
    //TODO index for nom_alt are in principle known at compile time

    return sf_tracking_reco_->at(sf_tracking_reco_->template axis<0>().index(eta),
                                sf_tracking_reco_->template axis<1>().index(charge),
                                idx_nom_alt);
  }

  double scale_factor_product(float pt, float eta, int charge, bool pass_iso, bool with_trigger, int idx_nom_alt) const {
    auto const eta_idx = sf_idip_trig_iso_->template axis<0>().index(eta);
    auto const pt_idx = sf_idip_trig_iso_->template axis<1>().index(pt);
    auto const charge_idx = sf_idip_trig_iso_->template axis<2>().index(charge);
    auto const eff_type_idx_idip_trig = with_trigger ? idx_idip_trig_ : idip_idx_;
    auto const eff_type_idx_iso_pass = with_trigger ? idx_iso_triggering_ : idx_iso_nontriggering_;
    auto const eff_type_idx_iso_fail = with_trigger ? idx_antiiso_triggering_ : idx_antiiso_nontriggering_;
    auto const eff_type_idx_iso = pass_iso ? eff_type_idx_iso_pass : eff_type_idx_iso_fail;

    const double idip = sf_idip_trig_iso_->at(eta_idx, pt_idx, charge_idx, eff_type_idx_idip_trig, idx_nom_alt).value();
    const double iso = sf_idip_trig_iso_->at(eta_idx, pt_idx, charge_idx, eff_type_idx_iso, idx_nom_alt).value();
    const double trackingreco = sf_trackingreco(eta, charge, idx_nom_alt).value();

    return idip*iso*trackingreco;
  }

  using syst_tensor_t = Eigen::TensorFixedSize<double, Eigen::Sizes<2>>;

  syst_tensor_t sf_idip_trig_iso_syst_var(float pt, float eta, int charge, bool pass_iso, bool with_trigger) const {
    syst_tensor_t res;

    auto const eta_idx = sf_idip_trig_iso_->template axis<0>().index(eta);
    auto const pt_idx = sf_idip_trig_iso_->template axis<1>().index(pt);
    auto const charge_idx = sf_idip_trig_iso_->template axis<2>().index(charge);
    auto const eff_type_idx_idip_trig = with_trigger ? idx_idip_trig_ : idip_idx_;
    auto const eff_type_idx_iso_pass = with_trigger ? idx_iso_triggering_ : idx_iso_nontriggering_;
    auto const eff_type_idx_iso_fail = with_trigger ? idx_antiiso_triggering_ : idx_antiiso_nontriggering_;
    auto const eff_type_idx_iso = pass_iso ? eff_type_idx_iso_pass : eff_type_idx_iso_fail;

    const double sf_idip_trig = sf_idip_trig_iso_->at(eta_idx,
                                                        pt_idx,
                                                        charge_idx,
                                                        eff_type_idx_idip_trig,
                                                        idx_nom_).value();

    const double sf_idip_trig_alt = sf_idip_trig_iso_->at(eta_idx,
                                                        pt_idx,
                                                        charge_idx,
                                                        eff_type_idx_idip_trig,
                                                        idx_alt_).value();

    const double variation_factor_idip_trig = sf_idip_trig_alt/sf_idip_trig;

    res(0) = variation_factor_idip_trig;

    const double sf_iso = sf_idip_trig_iso_->at(eta_idx,
                                           pt_idx,
                                           charge_idx,
                                           eff_type_idx_iso,
                                           idx_nom_).value();

    const double sf_iso_alt = sf_idip_trig_iso_->at(eta_idx,
                                           pt_idx,
                                           charge_idx,
                                           eff_type_idx_iso,
                                           idx_alt_).value();

    // anti-correlation between iso and anti-iso SF's is not exact, but an excellent
    // approximation
    const double variation_factor_iso = pass_iso ? sf_iso_alt/sf_iso : (sf_iso - (sf_iso_alt - sf_iso))/sf_iso;

    res(1) = variation_factor_iso;

    return res;

  }

  using stat_tensor_t = Eigen::TensorFixedSize<double, Eigen::Sizes<NEtaBins, NPtBins, 2, 2>>;

  //TODO avoid duplication of code above

  stat_tensor_t sf_idip_trig_iso_stat_var(float pt, float eta, int charge, bool pass_iso, bool with_trigger) const {
    stat_tensor_t res;
    res.setConstant(1.0);

    auto const eta_idx = sf_idip_trig_iso_->template axis<0>().index(eta);
    auto const pt_idx = sf_idip_trig_iso_->template axis<1>().index(pt);
    auto const charge_idx = sf_idip_trig_iso_->template axis<2>().index(charge);
    auto const eff_type_idx_idip_trig = with_trigger ? idx_idip_trig_ : idip_idx_;
    auto const eff_type_idx_iso_pass = with_trigger ? idx_iso_triggering_ : idx_iso_nontriggering_;
    auto const eff_type_idx_iso_fail = with_trigger ? idx_antiiso_triggering_ : idx_antiiso_nontriggering_;
    auto const eff_type_idx_iso = pass_iso ? eff_type_idx_iso_pass : eff_type_idx_iso_fail;

    // overflow/underflow are attributed to adjacent bin
    auto const tensor_eta_idx = std::clamp(eta_idx, 0, NEtaBins - 1);
    auto const tensor_pt_idx = std::clamp(pt_idx, 0, NPtBins - 1);

    auto const &cell_idip_trig = sf_idip_trig_iso_->at(eta_idx,
                                                        pt_idx,
                                                        charge_idx,
                                                        eff_type_idx_idip_trig,
                                                        idx_nom_);

    const double sf_idip_trig = cell_idip_trig.value();
    const double err_idip_trig = std::sqrt(cell_idip_trig.variance());

    const double variation_factor_idip_trig = (sf_idip_trig + err_idip_trig)/sf_idip_trig;

    res(tensor_eta_idx, tensor_pt_idx, charge_idx, 0) *= variation_factor_idip_trig;

    auto const &cell_iso = sf_idip_trig_iso_->at(eta_idx,
                                           pt_idx,
                                           charge_idx,
                                           eff_type_idx_iso,
                                           idx_nom_);

    const double sf_iso = cell_iso.value();
    const double err_iso = std::sqrt(cell_iso.variance());

    // anti-correlation between iso and anti-iso SF's is not exact, but an excellent
    // approximation
    const double variation_factor_iso = pass_iso ? (sf_iso + err_iso)/sf_iso : (sf_iso - err_iso)/sf_iso;

    res(tensor_eta_idx, tensor_pt_idx, charge_idx, 1) *= variation_factor_iso;

    return res;
  }



protected:

  std::shared_ptr<const HIST_IDIPTRIGISO> sf_idip_trig_iso_;
  std::shared_ptr<const HIST_TRACKINGRECO> sf_tracking_reco_;

  // cache the bin indices since the string category lookup is slow
  int idip_idx_ = sf_idip_trig_iso_->template axis<3>().index("idip");
  int idx_idip_trig_ = sf_idip_trig_iso_->template axis<3>().index("idip_trig");
  int idx_iso_triggering_ = sf_idip_trig_iso_->template axis<3>().index("iso_triggering");
  int idx_antiiso_triggering_ = sf_idip_trig_iso_->template axis<3>().index("antiiso_triggering");
  int idx_iso_nontriggering_ = sf_idip_trig_iso_->template axis<3>().index("iso_nontriggering");
  int idx_antiiso_nontriggering_ = sf_idip_trig_iso_->template axis<3>().index("antiiso_nontriggering");

  int idx_nom_ = sf_idip_trig_iso_->template axis<4>().index(0);
  int idx_alt_ = sf_idip_trig_iso_->template axis<4>().index(1);

};

// base template for one-lepton case
template<bool do_other, int NEtaBins, int NPtBins, typename HIST_IDIPTRIGISO, typename HIST_TRACKINGRECO>
class muon_efficiency_helper : public muon_efficiency_helper_base<NEtaBins, NPtBins, HIST_IDIPTRIGISO, HIST_TRACKINGRECO> {

public:

  using base_t = muon_efficiency_helper_base<NEtaBins, NPtBins, HIST_IDIPTRIGISO, HIST_TRACKINGRECO>;

  // inherit constructor
  using base_t::base_t;

  double operator() (float pt, float eta, int charge, bool pass_iso) {
    constexpr bool with_trigger = true;
    return base_t::scale_factor_product(pt, eta, charge, pass_iso, with_trigger, base_t::idx_nom_);
  }

};

// specialization for two-lepton case
template<int NEtaBins, int NPtBins, typename HIST_IDIPTRIGISO, typename HIST_TRACKINGRECO>
class muon_efficiency_helper<true, NEtaBins, NPtBins, HIST_IDIPTRIGISO, HIST_TRACKINGRECO> : public muon_efficiency_helper_base<NEtaBins, NPtBins, HIST_IDIPTRIGISO, HIST_TRACKINGRECO> {

public:

  using base_t = muon_efficiency_helper_base<NEtaBins, NPtBins, HIST_IDIPTRIGISO, HIST_TRACKINGRECO>;

  // inherit constructor
  using base_t::base_t;

  double operator() (float trig_pt, float trig_eta, int trig_charge,
                     float nontrig_pt, float nontrig_eta, int nontrig_charge) {
    constexpr bool with_trigger = true;
    constexpr bool without_trigger = false;
    constexpr bool pass_iso = true;

    const double sftrig = base_t::scale_factor_product(trig_pt, trig_eta, trig_charge, pass_iso, with_trigger, base_t::idx_nom_);

    const double sfnontrig = base_t::scale_factor_product(nontrig_pt, nontrig_eta, nontrig_charge, pass_iso, without_trigger, base_t::idx_nom_);

    return sftrig*sfnontrig;
  }



};

// base template for one lepton case
template<bool do_other, int NEtaBins, int NPtBins, typename HIST_IDIPTRIGISO, typename HIST_TRACKINGRECO>
class muon_efficiency_helper_stat : public muon_efficiency_helper_base<NEtaBins, NPtBins, HIST_IDIPTRIGISO, HIST_TRACKINGRECO> {

public:

  using base_t = muon_efficiency_helper_base<NEtaBins, NPtBins, HIST_IDIPTRIGISO, HIST_TRACKINGRECO>;
  using tensor_t = typename base_t::stat_tensor_t;

  muon_efficiency_helper_stat(const base_t &other) : base_t(other) {}

  tensor_t operator() (float pt, float eta, int charge, bool pass_iso, double nominal_weight = 1.0) {
    constexpr bool with_trigger = true;
    return nominal_weight*base_t::sf_idip_trig_iso_stat_var(pt, eta, charge, pass_iso, with_trigger);
  }

};

// specialization for two-lepton case
template<int NEtaBins, int NPtBins, typename HIST_IDIPTRIGISO, typename HIST_TRACKINGRECO>
class muon_efficiency_helper_stat<true, NEtaBins, NPtBins, HIST_IDIPTRIGISO, HIST_TRACKINGRECO> : public muon_efficiency_helper_base<NEtaBins, NPtBins, HIST_IDIPTRIGISO, HIST_TRACKINGRECO> {

public:

  using base_t = muon_efficiency_helper_base<NEtaBins, NPtBins, HIST_IDIPTRIGISO, HIST_TRACKINGRECO>;
  using tensor_t = typename base_t::stat_tensor_t;

  muon_efficiency_helper_stat(const base_t &other) : base_t(other) {}

  tensor_t operator() (float trig_pt, float trig_eta, int trig_charge,
                     float nontrig_pt, float nontrig_eta, int nontrig_charge, double nominal_weight = 1.0) {
    constexpr bool with_trigger = true;
    constexpr bool without_trigger = false;
    constexpr bool pass_iso = true;

    const tensor_t variation_trig = base_t::sf_idip_trig_iso_stat_var(trig_pt, trig_eta, trig_charge, pass_iso, with_trigger);

    const tensor_t variation_nontrig = base_t::sf_idip_trig_iso_stat_var(nontrig_pt, nontrig_eta, nontrig_charge, pass_iso, without_trigger);

    return nominal_weight*variation_trig*variation_nontrig;
  }

};

// base template for one lepton case
template<bool do_other, int NEtaBins, int NPtBins, typename HIST_IDIPTRIGISO, typename HIST_TRACKINGRECO>
class muon_efficiency_helper_syst : public muon_efficiency_helper_base<NEtaBins, NPtBins, HIST_IDIPTRIGISO, HIST_TRACKINGRECO> {

public:

  using base_t = muon_efficiency_helper_base<NEtaBins, NPtBins, HIST_IDIPTRIGISO, HIST_TRACKINGRECO>;
  using tensor_t = typename base_t::syst_tensor_t;

  muon_efficiency_helper_syst(const base_t &other) : base_t(other) {}

  tensor_t operator() (float pt, float eta, int charge, bool pass_iso, double nominal_weight = 1.0) {
    constexpr bool with_trigger = true;
    return nominal_weight*base_t::sf_idip_trig_iso_syst_var(pt, eta, charge, pass_iso, with_trigger);
  }

};

// specialization for two-lepton case
template<int NEtaBins, int NPtBins, typename HIST_IDIPTRIGISO, typename HIST_TRACKINGRECO>
class muon_efficiency_helper_syst<true, NEtaBins, NPtBins, HIST_IDIPTRIGISO, HIST_TRACKINGRECO> : public muon_efficiency_helper_base<NEtaBins, NPtBins, HIST_IDIPTRIGISO, HIST_TRACKINGRECO> {

public:

  using base_t = muon_efficiency_helper_base<NEtaBins, NPtBins, HIST_IDIPTRIGISO, HIST_TRACKINGRECO>;
  using tensor_t = typename base_t::syst_tensor_t;

  muon_efficiency_helper_syst(const base_t &other) : base_t(other) {}

  tensor_t operator() (float trig_pt, float trig_eta, int trig_charge,
                     float nontrig_pt, float nontrig_eta, int nontrig_charge, double nominal_weight = 1.0) {
    constexpr bool with_trigger = true;
    constexpr bool without_trigger = false;
    constexpr bool pass_iso = true;

    const tensor_t variation_trig = base_t::sf_idip_trig_iso_syst_var(trig_pt, trig_eta, trig_charge, pass_iso, with_trigger);

    const tensor_t variation_nontrig = base_t::sf_idip_trig_iso_syst_var(nontrig_pt, nontrig_eta, nontrig_charge, pass_iso, without_trigger);

    return nominal_weight*variation_trig*variation_nontrig;
  }

};


}

#endif
