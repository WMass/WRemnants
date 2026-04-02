#ifndef WREMNANTS_MUON_EFFICIENCIES_BINNED_VQT_REAL_H
#define WREMNANTS_MUON_EFFICIENCIES_BINNED_VQT_REAL_H

#include <TFile.h>
#include <TH3D.h>
#include <array>
#include <boost/histogram/axis.hpp>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <string>
#include <vector>

namespace wrem_vqt_real {

template <typename HIST_IDIPTRIGISO, typename HIST_TRACKING, typename HIST_RECO>
class muon_efficiency_binned_helper_base {
public:
  muon_efficiency_binned_helper_base(HIST_IDIPTRIGISO &&sf_idip_trig_iso,
                                     HIST_TRACKING &&sf_tracking,
                                     HIST_RECO &&sf_reco,
                                     std::vector<std::string> filenames,
                                     std::vector<std::string> histonames,
                                     bool error, int step)
      : sf_idip_trig_iso_(std::make_shared<const HIST_IDIPTRIGISO>(
            std::move(sf_idip_trig_iso))),
        sf_tracking_(
            std::make_shared<const HIST_TRACKING>(std::move(sf_tracking))),
        sf_reco_(std::make_shared<const HIST_RECO>(std::move(sf_reco))) {
    error_ = error;
    step_ = step;
    TFile *fileiso = new TFile(filenames[0].c_str());
    iso3d_ = (TH3D *)fileiso->Get(histonames[0].c_str())->Clone();
    iso3d_->SetName("iso3d");
    TFile *filetriggerminus = new TFile(filenames[1].c_str());
    trig3dminus_ =
        (TH3D *)filetriggerminus->Get(histonames[1].c_str())->Clone();
    trig3dminus_->SetName("triggerminus3d");
    TFile *filetriggerplus = new TFile(filenames[2].c_str());
    trig3dplus_ = (TH3D *)filetriggerplus->Get(histonames[2].c_str())->Clone();
    trig3dplus_->SetName("triggerplus3d");
  }

  std::array<double, 5> scale_factor_array(int pt_idx, int pt_idx_reco,
                                           int sapt_idx, int eta_idx,
                                           int saeta_idx, int charge_idx,
                                           bool pass_iso, bool with_trigger,
                                           int idx_nom_alt) const {

    auto const eff_type_idx_reco = idx_reco_;
    auto const eff_type_idx_tracking = idx_tracking_;
    auto const eff_type_idx_idip = idx_idip_;
    auto const eff_type_idx_trig = idx_trig_;
    auto const eff_type_idx_iso_pass =
        with_trigger ? idx_iso_triggering_ : idx_iso_nontriggering_;
    auto const eff_type_idx_iso_fail =
        with_trigger ? idx_antiiso_triggering_ : idx_antiiso_nontriggering_;
    auto const eff_type_idx_iso =
        pass_iso ? eff_type_idx_iso_pass : eff_type_idx_iso_fail;

    const double reco = sf_reco_
                            ->at(eta_idx, pt_idx_reco, charge_idx,
                                 eff_type_idx_reco, idx_nom_alt)
                            .value();
    const double tracking = sf_tracking_
                                ->at(saeta_idx, sapt_idx, charge_idx,
                                     eff_type_idx_tracking, idx_nom_alt)
                                .value();
    const double idip =
        sf_idip_trig_iso_
            ->at(eta_idx, pt_idx, charge_idx, eff_type_idx_idip, idx_nom_alt)
            .value();
    double trig = 1.0;
    if (with_trigger)
      trig =
          sf_idip_trig_iso_
              ->at(eta_idx, pt_idx, charge_idx, eff_type_idx_trig, idx_nom_alt)
              .value();
    const double iso =
        sf_idip_trig_iso_
            ->at(eta_idx, pt_idx, charge_idx, eff_type_idx_iso, idx_nom_alt)
            .value();

    std::array<double, 5> ret = {reco, tracking, idip, trig, iso};
    // for(int i = 0; i < ret.size(); i++)
    //     std::cout << "Scale factor i = " << i << " --> " << ret[i] <<
    //     std::endl;

    return ret;
  }

  double scale_factor_product(float pt, float eta, float sapt, float saeta,
                              int charge, bool pass_iso, bool with_trigger,
                              float vqt, int idx_nom_alt) const {

    auto const eta_idx = sf_idip_trig_iso_->template axis<0>().index(eta);
    auto const pt_idx = sf_idip_trig_iso_->template axis<1>().index(pt);
    auto const charge_idx = sf_idip_trig_iso_->template axis<2>().index(charge);
    auto const pt_idx_reco = sf_reco_->template axis<1>().index(pt);
    auto const saeta_idx = sf_tracking_->template axis<0>().index(saeta);
    auto const sapt_idx = sf_tracking_->template axis<1>().index(sapt);

    std::array<double, 5> allSF =
        scale_factor_array(pt_idx, pt_idx_reco, sapt_idx, eta_idx, saeta_idx,
                           charge_idx, pass_iso, with_trigger, idx_nom_alt);
    double sf = 1.0;
    if (step_ >= 0) {
      for (int i = 0; i < (allSF.size() - 2); i++) {
        // std::cout << "Scale factor i = " << i << " --> " << allSF[i] <<
        // std::endl;
        sf *= allSF[i];
      }
    }
    if (error_) {
      if (step_ == 1) {
        TH3D *chargehisto = charge > 0 ? trig3dplus_ : trig3dminus_;
        double binlowchargecenter = chargehisto->GetXaxis()->GetBinCenter(1),
               binupchargecenter = chargehisto->GetXaxis()->GetBinCenter(
                   chargehisto->GetXaxis()->GetNbins());
        int bincharge = chargehisto->FindFixBin(vqt, eta, pt),
            binlowcharge = chargehisto->FindFixBin(binlowchargecenter, eta, pt),
            binupcharge = chargehisto->FindFixBin(binupchargecenter, eta, pt);
        if ((vqt >= chargehisto->GetXaxis()->GetBinLowEdge(1)) &&
            (vqt <= chargehisto->GetXaxis()->GetBinUpEdge(
                        chargehisto->GetXaxis()->GetNbins())))
          sf *= chargehisto->GetBinError(bincharge);
        else {
          if (vqt < chargehisto->GetXaxis()->GetBinLowEdge(1))
            sf *= chargehisto->GetBinError(binlowcharge);
          else
            sf *= chargehisto->GetBinError(binupcharge);
        }
      }
      if (step_ == 2) {
        TH3D *chargehisto = charge > 0 ? trig3dplus_ : trig3dminus_;
        double binlowchargecenter = chargehisto->GetXaxis()->GetBinCenter(1),
               binupchargecenter = chargehisto->GetXaxis()->GetBinCenter(
                   chargehisto->GetXaxis()->GetNbins());
        int bincharge = chargehisto->FindFixBin(vqt, eta, pt),
            binlowcharge = chargehisto->FindFixBin(binlowchargecenter, eta, pt),
            binupcharge = chargehisto->FindFixBin(binupchargecenter, eta, pt);
        if ((vqt >= chargehisto->GetXaxis()->GetBinLowEdge(1)) &&
            (vqt <= chargehisto->GetXaxis()->GetBinUpEdge(
                        chargehisto->GetXaxis()->GetNbins())))
          sf *= chargehisto->GetBinContent(bincharge);
        else {
          if (vqt < chargehisto->GetXaxis()->GetBinLowEdge(1))
            sf *= chargehisto->GetBinContent(binlowcharge);
          else
            sf *= chargehisto->GetBinContent(binupcharge);
        }
        double binlowcenter = iso3d_->GetXaxis()->GetBinCenter(1),
               binupcenter = iso3d_->GetXaxis()->GetBinCenter(
                   iso3d_->GetXaxis()->GetNbins());
        int bin = iso3d_->FindFixBin(vqt, eta, pt),
            binlow = iso3d_->FindFixBin(binlowcenter, eta, pt),
            binup = iso3d_->FindFixBin(binupcenter, eta, pt);
        if ((vqt >= iso3d_->GetXaxis()->GetBinLowEdge(1)) &&
            (vqt <=
             iso3d_->GetXaxis()->GetBinUpEdge(iso3d_->GetXaxis()->GetNbins())))
          sf *= iso3d_->GetBinError(bin);
        else {
          if (vqt < iso3d_->GetXaxis()->GetBinLowEdge(1))
            sf *= iso3d_->GetBinError(binlow);
          else
            sf *= iso3d_->GetBinError(binup);
        }
      }
    } else {
      if (step_ >= 1) {
        TH3D *chargehisto = charge > 0 ? trig3dplus_ : trig3dminus_;
        double binlowchargecenter = chargehisto->GetXaxis()->GetBinCenter(1),
               binupchargecenter = chargehisto->GetXaxis()->GetBinCenter(
                   chargehisto->GetXaxis()->GetNbins());
        int bincharge = chargehisto->FindFixBin(vqt, eta, pt),
            binlowcharge = chargehisto->FindFixBin(binlowchargecenter, eta, pt),
            binupcharge = chargehisto->FindFixBin(binupchargecenter, eta, pt);
        if ((vqt >= chargehisto->GetXaxis()->GetBinLowEdge(1)) &&
            (vqt <= chargehisto->GetXaxis()->GetBinUpEdge(
                        chargehisto->GetXaxis()->GetNbins())))
          sf *= chargehisto->GetBinContent(bincharge);
        else {
          if (vqt < chargehisto->GetXaxis()->GetBinLowEdge(1))
            sf *= chargehisto->GetBinContent(binlowcharge);
          else
            sf *= chargehisto->GetBinContent(binupcharge);
        }
      }
      if (step_ >= 2) {
        double binlowcenter = iso3d_->GetXaxis()->GetBinCenter(1),
               binupcenter = iso3d_->GetXaxis()->GetBinCenter(
                   iso3d_->GetXaxis()->GetNbins());
        int bin = iso3d_->FindFixBin(vqt, eta, pt),
            binlow = iso3d_->FindFixBin(binlowcenter, eta, pt),
            binup = iso3d_->FindFixBin(binupcenter, eta, pt);
        if ((vqt >= iso3d_->GetXaxis()->GetBinLowEdge(1)) &&
            (vqt <=
             iso3d_->GetXaxis()->GetBinUpEdge(iso3d_->GetXaxis()->GetNbins())))
          sf *= iso3d_->GetBinContent(bin);
        else {
          if (vqt < iso3d_->GetXaxis()->GetBinLowEdge(1))
            sf *= iso3d_->GetBinContent(binlow);
          else
            sf *= iso3d_->GetBinContent(binup);
        }
      }
    }
    // std::cout << "Scale factor product " << sf << std::endl;
    return sf;
  }

  using syst_tensor_t = Eigen::TensorFixedSize<
      double, Eigen::Sizes<5>>; // 5 bins for reco, tracking, idip, trigger,
                                // iso(notrig) in this order

  syst_tensor_t sf_syst_var(float pt, float eta, float sapt, float saeta,
                            int charge, bool pass_iso,
                            bool with_trigger) const {

    syst_tensor_t res;

    auto const eta_idx = sf_idip_trig_iso_->template axis<0>().index(eta);
    auto const pt_idx = sf_idip_trig_iso_->template axis<1>().index(pt);
    auto const charge_idx = sf_idip_trig_iso_->template axis<2>().index(charge);
    auto const pt_idx_reco = sf_reco_->template axis<1>().index(pt);
    auto const saeta_idx = sf_tracking_->template axis<0>().index(saeta);
    auto const sapt_idx = sf_tracking_->template axis<1>().index(sapt);

    std::array<double, 5> allSF_nomi =
        scale_factor_array(pt_idx, pt_idx_reco, sapt_idx, eta_idx, saeta_idx,
                           charge_idx, pass_iso, with_trigger, idx_nom_);
    std::array<double, 5> allSF_alt =
        scale_factor_array(pt_idx, pt_idx_reco, sapt_idx, eta_idx, saeta_idx,
                           charge_idx, pass_iso, with_trigger, idx_alt_);

    // anticorrelation between iso and antiiso already embedded in the numbers
    // stored in the histograms also the alternate comes from data efficiency
    // variation only, so the anticorrelation in the efficiencies is preserved
    // in the scale factors

    // order is reco-tracking-idip-trigger-iso
    for (int i = 0; i < allSF_nomi.size(); i++) {
      res(i) = allSF_alt[i] / allSF_nomi[i];
    }

    return res;
  }

protected:
  std::shared_ptr<const HIST_IDIPTRIGISO> sf_idip_trig_iso_;
  std::shared_ptr<const HIST_TRACKING> sf_tracking_;
  std::shared_ptr<const HIST_RECO> sf_reco_;

  // hardcoded things to keep original pt binning for tnp when using smoothed
  // efficiencies
  // boost::histogram::axis::variable<double>
  // originalTnpPtBins{24., 26., 28., 30., 32., 34., 36., 38., 40., 42., 44., 47.,
  // 50., 55., 60., 65.};

  // cache the bin indices since the string category lookup is slow
  int idx_reco_ = sf_reco_->template axis<3>().index("reco");
  int idx_tracking_ = sf_tracking_->template axis<3>().index("tracking");
  int idx_idip_ = sf_idip_trig_iso_->template axis<3>().index("idip");
  int idx_trig_ = sf_idip_trig_iso_->template axis<3>().index("trigger");
  int idx_iso_triggering_ = sf_idip_trig_iso_->template axis<3>().index("iso");
  int idx_antiiso_triggering_ =
      sf_idip_trig_iso_->template axis<3>().index("antiiso");
  int idx_iso_nontriggering_ =
      sf_idip_trig_iso_->template axis<3>().index("isonotrig");
  int idx_antiiso_nontriggering_ =
      sf_idip_trig_iso_->template axis<3>().index("antiisonotrig");

  int idx_nom_ = sf_idip_trig_iso_->template axis<4>().index(0);
  int idx_alt_ = sf_idip_trig_iso_->template axis<4>().index(1);

  TH3D *iso3d_;
  TH3D *trig3dplus_;
  TH3D *trig3dminus_;
  int step_;
  bool error_;
};

// base template for one-lepton case
template <bool do_other, typename HIST_IDIPTRIGISO, typename HIST_TRACKING,
          typename HIST_RECO>
class muon_efficiency_binned_helper
    : public muon_efficiency_binned_helper_base<HIST_IDIPTRIGISO, HIST_TRACKING,
                                                HIST_RECO> {

public:
  using base_t = muon_efficiency_binned_helper_base<HIST_IDIPTRIGISO,
                                                    HIST_TRACKING, HIST_RECO>;
  // inherit constructor
  using base_t::base_t;

  muon_efficiency_binned_helper(const base_t &other) : base_t(other) {}

  double operator()(float pt, float eta, float sapt, float saeta, int charge,
                    bool pass_iso, float vqt) {
    constexpr bool with_trigger = true;
    return base_t::scale_factor_product(pt, eta, sapt, saeta, charge, pass_iso,
                                        with_trigger, vqt, base_t::idx_nom_);
  }
};

// specialization for two-lepton case
template <typename HIST_IDIPTRIGISO, typename HIST_TRACKING, typename HIST_RECO>
class muon_efficiency_binned_helper<true, HIST_IDIPTRIGISO, HIST_TRACKING,
                                    HIST_RECO>
    : public muon_efficiency_binned_helper_base<HIST_IDIPTRIGISO, HIST_TRACKING,
                                                HIST_RECO> {

public:
  using base_t = muon_efficiency_binned_helper_base<HIST_IDIPTRIGISO,
                                                    HIST_TRACKING, HIST_RECO>;
  // inherit constructor
  using base_t::base_t;

  muon_efficiency_binned_helper(const base_t &other) : base_t(other) {}

  double operator()(float trig_pt, float trig_eta, float trig_sapt,
                    float trig_saeta, int trig_charge, float nontrig_pt,
                    float nontrig_eta, float nontrig_sapt, float nontrig_saeta,
                    int nontrig_charge) {
    constexpr bool with_trigger = true;
    constexpr bool without_trigger = false;
    constexpr bool pass_iso = true;
    const double sftrig = base_t::scale_factor_product(
        trig_pt, trig_eta, trig_sapt, trig_saeta, trig_charge, pass_iso,
        with_trigger, base_t::idx_nom_);
    const double sfnontrig = base_t::scale_factor_product(
        nontrig_pt, nontrig_eta, nontrig_sapt, nontrig_saeta, nontrig_charge,
        pass_iso, without_trigger, base_t::idx_nom_);
    return sftrig * sfnontrig;
  }
};

// base template for one lepton case
template <bool do_other, typename HIST_IDIPTRIGISO, typename HIST_TRACKING,
          typename HIST_RECO>
class muon_efficiency_binned_helper_syst
    : public muon_efficiency_binned_helper_base<HIST_IDIPTRIGISO, HIST_TRACKING,
                                                HIST_RECO> {

public:
  using base_t = muon_efficiency_binned_helper_base<HIST_IDIPTRIGISO,
                                                    HIST_TRACKING, HIST_RECO>;

  using tensor_t = typename base_t::syst_tensor_t;

  // inherit constructor
  using base_t::base_t;

  muon_efficiency_binned_helper_syst(const base_t &other) : base_t(other) {}

  tensor_t operator()(float pt, float eta, float sapt, float saeta, int charge,
                      bool pass_iso, double nominal_weight = 1.0) {
    constexpr bool with_trigger = true;
    return nominal_weight * base_t::sf_syst_var(pt, eta, sapt, saeta, charge,
                                                pass_iso, with_trigger);
  }
};

// specialization for two-lepton case
template <typename HIST_IDIPTRIGISO, typename HIST_TRACKING, typename HIST_RECO>
class muon_efficiency_binned_helper_syst<true, HIST_IDIPTRIGISO, HIST_TRACKING,
                                         HIST_RECO>
    : public muon_efficiency_binned_helper_base<HIST_IDIPTRIGISO, HIST_TRACKING,
                                                HIST_RECO> {

public:
  using base_t = muon_efficiency_binned_helper_base<HIST_IDIPTRIGISO,
                                                    HIST_TRACKING, HIST_RECO>;
  using tensor_t = typename base_t::syst_tensor_t;

  // inherit constructor
  using base_t::base_t;

  muon_efficiency_binned_helper_syst(const base_t &other) : base_t(other) {}

  tensor_t operator()(float trig_pt, float trig_eta, float trig_sapt,
                      float trig_saeta, int trig_charge, float nontrig_pt,
                      float nontrig_eta, float nontrig_sapt,
                      float nontrig_saeta, int nontrig_charge,
                      double nominal_weight = 1.0) {
    constexpr bool with_trigger = true;
    constexpr bool without_trigger = false;
    constexpr bool pass_iso = true;
    const tensor_t variation_trig =
        base_t::sf_syst_var(trig_pt, trig_eta, trig_sapt, trig_saeta,
                            trig_charge, pass_iso, with_trigger);
    const tensor_t variation_nontrig = base_t::sf_syst_var(
        nontrig_pt, nontrig_eta, nontrig_sapt, nontrig_saeta, nontrig_charge,
        pass_iso, without_trigger);
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
template <int NEtaBins, int NPtBins, typename HIST_SF>
class muon_efficiency_binned_helper_stat_base {
public:
  muon_efficiency_binned_helper_stat_base(HIST_SF &&sf_type)
      : sf_type_(std::make_shared<const HIST_SF>(std::move(sf_type))) {}

  // number of eta bins, number of eigen variations for pt axis, then 2 charges
  using stat_tensor_t =
      Eigen::TensorFixedSize<double, Eigen::Sizes<NEtaBins, NPtBins, 2>>;

  int checkEffTypeInAxis(boost::histogram::axis::category<std::string> axis,
                         const std::string &match = "match") {
    int ret = -1;
    for (int i = 0; i < axis.size(); i++) {
      if (match == axis.value(i)) {
        ret = i;
        break;
      }
    }
    return ret;
  }

  // general case with no isolation
  stat_tensor_t sf_stat_var(float pt, float eta, int charge) const {
    stat_tensor_t res;
    res.setConstant(1.0);

    auto const eta_idx = sf_type_->template axis<0>().index(eta);
    auto const pt_idx = sf_type_->template axis<1>().index(pt);
    auto const charge_idx = sf_type_->template axis<2>().index(charge);
    auto const eff_type_idx =
        0; // TODO FIXME, use first (and only existent) bin

    // overflow/underflow are attributed to adjacent bin
    auto const tensor_eta_idx = std::clamp(eta_idx, 0, NEtaBins - 1);
    auto const tensor_pt_idx = std::clamp(pt_idx, 0, NPtBins - 1);

    auto const &cell_nomi =
        sf_type_->at(eta_idx, pt_idx, charge_idx, eff_type_idx);

    const double sf_value = cell_nomi.value();
    const double sf_statunc = std::sqrt(cell_nomi.variance());
    const double sf_stat_variation = (sf_value + sf_statunc) / sf_value;

    res(tensor_eta_idx, tensor_pt_idx, charge_idx) *= sf_stat_variation;

    return res;
  }

  // special case for isolation
  stat_tensor_t sf_stat_var_iso(float pt, float eta, int charge, bool pass_iso,
                                bool with_trigger) const {
    stat_tensor_t res;
    res.setConstant(1.0);

    auto const eta_idx = sf_type_->template axis<0>().index(eta);
    auto const pt_idx = sf_type_->template axis<1>().index(pt);
    auto const charge_idx = sf_type_->template axis<2>().index(charge);

    auto const eff_type_idx_iso_pass =
        with_trigger ? idx_iso_triggering_ : idx_iso_nontriggering_;
    auto const eff_type_idx_iso_fail =
        with_trigger ? idx_antiiso_triggering_ : idx_antiiso_nontriggering_;
    auto const eff_type_idx_iso =
        pass_iso ? eff_type_idx_iso_pass : eff_type_idx_iso_fail;

    // overflow/underflow are attributed to adjacent bin
    auto const tensor_eta_idx = std::clamp(eta_idx, 0, NEtaBins - 1);
    auto const tensor_pt_idx = std::clamp(pt_idx, 0, NPtBins - 1);

    auto const &cell_nomi =
        sf_type_->at(eta_idx, pt_idx, charge_idx, eff_type_idx_iso);

    const double sf_value = cell_nomi.value();
    const double sf_statunc = std::sqrt(cell_nomi.variance());
    // anti-correlation between iso and anti-iso SF's is not exact, but an
    // excellent approximation
    double sf_shifted = sf_value + sf_statunc;
    if (not pass_iso) {
      sf_shifted = sf_value - sf_statunc;
      sf_shifted = std::clamp(sf_shifted, 0.0, sf_shifted);
    }
    const double sf_stat_variation = sf_shifted / sf_value;

    res(tensor_eta_idx, tensor_pt_idx, charge_idx) *= sf_stat_variation;

    return res;
  }

protected:
  std::shared_ptr<const HIST_SF> sf_type_;
  // cache the bin indices since the string category lookup is slow
  int isTriggerStep_ =
      sf_type_->template axis<3>().value(0) ==
      "trigger"; // special treatment for the stat variation in 2 lepton case
  // check if axis name exists in histogram, return -1 (invalid index) if not
  // found
  int idx_iso_triggering_ =
      checkEffTypeInAxis(sf_type_->template axis<3>(), "iso");
  int idx_antiiso_triggering_ =
      checkEffTypeInAxis(sf_type_->template axis<3>(), "antiiso");
  int idx_iso_nontriggering_ =
      checkEffTypeInAxis(sf_type_->template axis<3>(), "isonotrig");
  int idx_antiiso_nontriggering_ =
      checkEffTypeInAxis(sf_type_->template axis<3>(), "antiisonotrig");
};

////
//// General case (isolation is treated separately)
////

// base template for one lepton case
template <bool do_other, int NEtaBins, int NPtBins, typename HIST_SF>
class muon_efficiency_binned_helper_stat
    : public muon_efficiency_binned_helper_stat_base<NEtaBins, NPtBins,
                                                     HIST_SF> {

public:
  using stat_base_t =
      muon_efficiency_binned_helper_stat_base<NEtaBins, NPtBins, HIST_SF>;
  using tensor_t = typename stat_base_t::stat_tensor_t;

  using stat_base_t::stat_base_t;

  tensor_t operator()(float pt, float eta, int charge,
                      double nominal_weight = 1.0) {
    return nominal_weight * stat_base_t::sf_stat_var(pt, eta, charge);
  }
};

// specialization for two-lepton case
template <int NEtaBins, int NPtBins, typename HIST_SF>
class muon_efficiency_binned_helper_stat<true, NEtaBins, NPtBins, HIST_SF>
    : public muon_efficiency_binned_helper_stat_base<NEtaBins, NPtBins,
                                                     HIST_SF> {

public:
  using stat_base_t =
      muon_efficiency_binned_helper_stat_base<NEtaBins, NPtBins, HIST_SF>;
  using tensor_t = typename stat_base_t::stat_tensor_t;

  using stat_base_t::stat_base_t;

  tensor_t operator()(float trig_pt, float trig_eta, int trig_charge,
                      float nontrig_pt, float nontrig_eta, int nontrig_charge,
                      double nominal_weight = 1.0) {

    const tensor_t variation_trig =
        stat_base_t::sf_stat_var(trig_pt, trig_eta, trig_charge);
    if (stat_base_t::isTriggerStep_) {
      return nominal_weight * variation_trig;
    } else {
      const tensor_t variation_nontrig =
          stat_base_t::sf_stat_var(nontrig_pt, nontrig_eta, nontrig_charge);
      return nominal_weight * variation_trig * variation_nontrig;
    }
  }
};

////
//// Isolation
////

// base template for one lepton case
template <bool do_other, int NEtaBins, int NPtBins, typename HIST_SF>
class muon_efficiency_binned_helper_stat_iso
    : public muon_efficiency_binned_helper_stat_base<NEtaBins, NPtBins,
                                                     HIST_SF> {

public:
  using stat_base_t =
      muon_efficiency_binned_helper_stat_base<NEtaBins, NPtBins, HIST_SF>;
  using tensor_t = typename stat_base_t::stat_tensor_t;

  using stat_base_t::stat_base_t;

  tensor_t operator()(float pt, float eta, int charge, bool pass_iso,
                      double nominal_weight = 1.0) {
    constexpr bool with_trigger = true;
    return nominal_weight * stat_base_t::sf_stat_var_iso(
                                pt, eta, charge, pass_iso, with_trigger);
  }
};

// specialization for two-lepton case
template <int NEtaBins, int NPtBins, typename HIST_SF>
class muon_efficiency_binned_helper_stat_iso<true, NEtaBins, NPtBins, HIST_SF>
    : public muon_efficiency_binned_helper_stat_base<NEtaBins, NPtBins,
                                                     HIST_SF> {

public:
  using stat_base_t =
      muon_efficiency_binned_helper_stat_base<NEtaBins, NPtBins, HIST_SF>;
  using tensor_t = typename stat_base_t::stat_tensor_t;

  using stat_base_t::stat_base_t;

  tensor_t operator()(float trig_pt, float trig_eta, int trig_charge,
                      float nontrig_pt, float nontrig_eta, int nontrig_charge,
                      double nominal_weight = 1.0) {
    constexpr bool with_trigger = true;
    constexpr bool without_trigger = false;
    constexpr bool pass_iso = true;

    const tensor_t variation_trig = stat_base_t::sf_stat_var_iso(
        trig_pt, trig_eta, trig_charge, pass_iso, with_trigger);
    const tensor_t variation_nontrig = stat_base_t::sf_stat_var_iso(
        nontrig_pt, nontrig_eta, nontrig_charge, pass_iso, without_trigger);
    return nominal_weight * variation_trig * variation_nontrig;
  }
};

} // namespace wrem_vqt_real

#endif
