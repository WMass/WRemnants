#ifndef WREMNANTS_MUON_PREFIRING_H
#define WREMNANTS_MUON_PREFIRING_H

#include "utils.hpp"
#include <TH2D.h>

namespace wrem {

class muon_prefiring_helper {
public:
  muon_prefiring_helper(const TH2D &parms, const TH2D &hotspotparms)
      : parameters_(make_shared_TH1<const TH2D>(parms)),
        hotspot_parameters_(make_shared_TH1<const TH2D>(hotspotparms)) {}

  static bool is_hotspot(float eta, float phi) {
    // region with problematic chambers where prefiring probabilities are
    // measured separately
    if (eta > 1.24 and eta < 1.6 and phi > 2.44346 and phi < 2.79253) {
      return true;
    }
    return false;
  }

  double operator()(const Vec_f &eta, const Vec_f &pt, const Vec_f &phi,
                    const Vec_i &charge, const Vec_b &looseId) const {
    double sf = 1.0;

    const TH2D &hprefire = *parameters_;
    const TH2D &hMuonPrefiringNew_hotspot = *hotspot_parameters_;

    int nBins = hprefire.GetNbinsX();
    int prefireBin = 0;
    double prefiringProbability = 0.0;
    for (unsigned int i = 0; i < eta.size(); ++i) {
      if (charge[i] == -99 || !looseId[i])
        continue;
      if (is_hotspot(eta[i], phi[i])) {
        const double plateau =
            std::clamp(hMuonPrefiringNew_hotspot.GetBinContent(1, 3), 0., 1.);
        prefiringProbability =
            plateau /
            (std::exp((pt[i] - hMuonPrefiringNew_hotspot.GetBinContent(1, 1)) /
                      hMuonPrefiringNew_hotspot.GetBinContent(1, 2)) +
             1);
      } else {
        prefireBin = std::clamp(
            hprefire.GetXaxis()->FindFixBin(std::fabs(eta[i])), 1, nBins);
        const double plateau =
            std::clamp(hprefire.GetBinContent(prefireBin, 3), 0., 1.);
        prefiringProbability =
            plateau /
            (std::exp((pt[i] - hprefire.GetBinContent(prefireBin, 1)) /
                      hprefire.GetBinContent(prefireBin, 2)) +
             1);
      }
      sf *= (1.0 - prefiringProbability);
    }
    return sf;
  }

  const std::shared_ptr<const TH2D> &parameters() const { return parameters_; }
  const std::shared_ptr<const TH2D> &hotspot_parameters() const {
    return hotspot_parameters_;
  }

private:
  std::shared_ptr<const TH2D> parameters_;
  std::shared_ptr<const TH2D> hotspot_parameters_;
};

template <std::size_t NEtaBins> class muon_prefiring_helper_stat {

public:
  static constexpr std::size_t NVar = NEtaBins + 1;
  using value_type = Eigen::TensorFixedSize<double, Eigen::Sizes<NVar, 2>>;

  muon_prefiring_helper_stat(const muon_prefiring_helper &other)
      : parameters_(other.parameters()),
        hotspot_parameters_(other.hotspot_parameters()) {}

  value_type operator()(const Vec_f &eta, const Vec_f &pt, const Vec_f &phi,
                        const Vec_i &charge, const Vec_b &looseId,
                        double nominal_weight = 1.0) const {

    value_type res;
    res.setConstant(nominal_weight);

    const TH2D &hprefire = *parameters_;
    const TH2D &hMuonPrefiringNew_hotspot = *hotspot_parameters_;

    const int nBins = hprefire.GetNbinsX();

    for (unsigned int i = 0; i < eta.size(); ++i) {

      if (charge[i] == -99 || !looseId[i])
        continue;

      int idx;
      double plateau_raw;
      double plateau_err;

      if (muon_prefiring_helper::is_hotspot(eta[i], phi[i])) {
        // hotspot index is the last one
        idx = NEtaBins;
        plateau_raw = hMuonPrefiringNew_hotspot.GetBinContent(1, 3);
        plateau_err = hMuonPrefiringNew_hotspot.GetBinError(1, 3);

      } else {
        const int prefireBin = std::clamp(
            hprefire.GetXaxis()->FindFixBin(std::fabs(eta[i])), 1, nBins);
        // standard case, index from histogram bin
        idx = prefireBin - 1;
        plateau_raw = hprefire.GetBinContent(prefireBin, 3);
        plateau_err = hprefire.GetBinError(prefireBin, 3);
      }

      // the vector bin corresponding to prefireBin will be multiplied by ratio
      // of (1 - prefiring probability) with respect to nominal
      const double plateau_nom = std::clamp(plateau_raw, 0., 1.);
      const double plateau_up = std::clamp(plateau_raw + plateau_err, 0., 1.);
      const double plateau_down = std::clamp(plateau_raw - plateau_err, 0., 1.);

      res(idx, 0) *= (1. - plateau_down) / (1. - plateau_nom);
      res(idx, 1) *= (1. - plateau_up) / (1. - plateau_nom);
    }
    return res;
  }

private:
  std::shared_ptr<const TH2D> parameters_;
  std::shared_ptr<const TH2D> hotspot_parameters_;
};

class muon_prefiring_helper_syst {

public:
  using value_type = Eigen::TensorFixedSize<double, Eigen::Sizes<2>>;

  muon_prefiring_helper_syst(const muon_prefiring_helper &other)
      : parameters_(other.parameters()),
        hotspot_parameters_(other.hotspot_parameters()) {}

  value_type operator()(const Vec_f &eta, const Vec_f &pt, const Vec_f &phi,
                        const Vec_i &charge, const Vec_b &looseId,
                        double nominal_weight = 1.0) const {

    value_type res;
    res.setConstant(nominal_weight);

    const TH2D &hprefire = *parameters_;
    const TH2D &hMuonPrefiringNew_hotspot = *hotspot_parameters_;

    const int nBins = hprefire.GetNbinsX();

    for (unsigned int i = 0; i < eta.size(); ++i) {

      if (charge[i] == -99 || !looseId[i])
        continue;

      double plateau_raw;

      if (muon_prefiring_helper::is_hotspot(eta[i], phi[i])) {
        plateau_raw = hMuonPrefiringNew_hotspot.GetBinContent(1, 3);

      } else {
        const int prefireBin = std::clamp(
            hprefire.GetXaxis()->FindFixBin(std::fabs(eta[i])), 1, nBins);
        plateau_raw = hprefire.GetBinContent(prefireBin, 3);
      }

      // the vector bin corresponding to prefireBin will be multiplied by ratio
      // of (1 - prefiring probability) with respect to nominal
      const double plateau_nom = std::clamp(plateau_raw, 0., 1.);
      const double plateau_up = std::clamp(1.11 * plateau_raw, 0., 1.);
      const double plateau_down = std::clamp(0.89 * plateau_raw, 0., 1.);

      res(0) *= (1. - plateau_down) / (1. - plateau_nom);
      res(1) *= (1. - plateau_up) / (1. - plateau_nom);
    }
    return res;
  }

private:
  std::shared_ptr<const TH2D> parameters_;
  std::shared_ptr<const TH2D> hotspot_parameters_;
};

} // namespace wrem

#endif
