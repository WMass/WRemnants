#ifndef WREM_SHIFT_SMEAR_REWEIGHT_HELPERS_H
#define WREM_SHIFT_SMEAR_REWEIGHT_HELPERS_H

// Drop-in replacements for the analytic JpsiCorrectionsUncHelperSplines
// and SmearingUncertaintyHelperParametrized that feed the per-muon
// (y, c, u, σ) tuple through the trained shift_smear_reweight model
// (combined ONNX: y_raw, c_raw, u_raw, σ_raw -> log_r). Both helpers
// preserve the legacy variation axes (η × {A, e, M} for the J/ψ scale,
// η × {a, c, b, d} for the resolution) so downstream nuisance
// bookkeeping is unchanged; what changes is how each variation's
// per-muon weight is computed -- from the analytic linearisation
// in (q/p) / σ² to the trained log W from the network. The legacy
// ``response_weights`` column is no longer consulted; the network
// already gives the exact reweight factor, so ``alt_weight =
// exp(log_r)`` is the full multiplicative correction.
//
// All preprocessing (target_mean / target_std on y, cond_mean /
// cond_std on c, target_std on u and σ) is baked into the ONNX graph,
// so the C++ side never has to read ``preproc.json``. Inputs to
// ``operator()`` carry physical units throughout:
//   y_raw  = (r_κ, δλ, δφ)        in raw r_κ / rad / rad
//   c_raw  = (log p_T, q, λ, sφ, cφ) in raw
//   u_raw  = (δr_κ, δλ, δφ)        the per-variation physical shift
//   σ_raw  = (σ_r_κ, σ_δλ, σ_δφ)   the per-variation physical width

#include <eigen3/Eigen/Dense>
#include <ROOT/RVec.hxx>
#include <array>
#include <cmath>
#include <memory>
#include <string>
#include <utility>

// ``muon_calibration.hpp`` (carrying ``calculateQopUnc``,
// ``SmearingHelperParametrized``, ``wrem::clip_tensor``, ``narf::get_value``)
// is intentionally not re-#included here -- it lacks a header guard and
// any re-include in the same cling TU redefines everything. Both
// headers are loaded side-by-side via ``narf.clingutils.Declare`` in
// muon_calibration.py (muon_calibration.hpp first, this header second),
// so the symbols are already in scope by the time we get here.
#include "onnxutils.hpp"

namespace wrem {

// Front matter:
//
//   F = 3 (r_kappa, dλ, dφ)
//   NCond = 6 (log_pt_gen, charge, λ_gen, sin φ_gen, cos φ_gen,
//              muon_source)
//
// matches scripts/corrections/muon_calibration/train_muon_response_flow.py
// :compute_targets_and_conditioning. ``muon_source`` is the per-muon
// integer class tag — 1 for prompt W/Z muons, 15 for τ-decay muons,
// 443 for J/ψ legs (not used in this helper). The ORT graph itself
// remaps these integer codes to the compact {-1, 0, +1} the network
// was trained on, so callers pass the raw integer (one of {1, 15, 443})
// as a plain float in c_raw[5] and no further conversion is needed.
namespace shift_smear_reweight {

constexpr std::size_t F = 3;
constexpr std::size_t NCond = 6;

// Pre-helper map: ``Muon_genPartFlav`` per-row to the integer code
// the model expects. The training schema uses 1 for prompt W/Z and
// 15 for τ-decay; any other ``genPartFlav`` value (0, 3, 4, 5...) is
// mapped to 1 (treated as the prompt class). The 443 sentinel for
// J/ψ legs is not produced by this helper — the wremnants helpers
// run on W/Z analysis MC.
inline int muon_source_from_gen_part_flav(int gen_part_flav) {
  return (gen_part_flav == 15) ? 15 : 1;
}

// Build the per-muon (y_raw, c_raw) pair from kinematics. ``q_*`` is
// the charge as a float (already signed) -- we don't need
// ``calculateTheta`` because the conditioning's ``λ_gen`` is
// ``atan(sinh(η_gen))``.
inline void
compute_y_raw(float kappa_reco, float eta_reco, float phi_reco, float kappa_gen,
              float eta_gen, float phi_gen, std::array<float, F> &y_out) {
  const float lam_r = std::atan(std::sinh(eta_reco));
  const float lam_g = std::atan(std::sinh(eta_gen));
  const float dphi_raw = phi_reco - phi_gen;
  // wrap to (-π, π].
  const float dphi = std::atan2(std::sin(dphi_raw), std::cos(dphi_raw));
  y_out[0] = kappa_reco / kappa_gen - 1.0f;
  y_out[1] = lam_r - lam_g;
  y_out[2] = dphi;
}

inline void compute_c_raw(float kappa_gen, float eta_gen, float phi_gen,
                          int muon_source,
                          std::array<float, NCond> &c_out) {
  c_out[0] = -std::log(std::fabs(kappa_gen) * std::cosh(eta_gen));
  c_out[1] = (kappa_gen >= 0.0f) ? 1.0f : -1.0f;  // sign(kappa) == sign(q)
  c_out[2] = std::atan(std::sinh(eta_gen));
  c_out[3] = std::sin(phi_gen);
  c_out[4] = std::cos(phi_gen);
  // Raw integer code; ORT does the {1, 15, 443} -> {-1, 0, +1} remap
  // and the pass-through standardisation inside the graph.
  c_out[5] = static_cast<float>(muon_source);
}

// Compile-time-sized ONNX runner for the combined
// (y_raw, c_raw, u_raw, σ_raw) -> log_r call at batch 1.  ``NVar`` is
// the number of (u, σ) variations baked into the inference at the call
// site (it's the second dim of the model's dynamic-axis input).
//
// Each ``operator()`` invocation issues one Ort::Session::Run, so an
// event with two muons triggers two calls; for J/ψ that's the same
// cardinality as the existing analytic helper's inner per-muon loop.
class ReweightModel {
public:
  ReweightModel(const std::string &onnx_path, unsigned int nslots = 1)
      : onnx_(onnx_path, nslots) {}

  // Static-shape entry: y is [1, F], c is [1, NCond], u and σ are
  // [1, NVar, F], output is [1, NVar].  Buffers are caller-owned so
  // they can live on the stack across the per-muon loop.
  template <std::size_t NVar>
  void
  run(const Eigen::TensorFixedSize<float, Eigen::Sizes<1, F>> &y,
      const Eigen::TensorFixedSize<float, Eigen::Sizes<1, NCond>> &c,
      const Eigen::TensorFixedSize<float, Eigen::Sizes<1, NVar, F>> &u_raw,
      const Eigen::TensorFixedSize<float, Eigen::Sizes<1, NVar, F>> &sigma_raw,
      Eigen::TensorFixedSize<float, Eigen::Sizes<1, NVar>> &log_r) {
    auto inputs = std::make_tuple(std::cref(y), std::cref(c),
                                  std::cref(u_raw), std::cref(sigma_raw));
    auto outputs = std::make_tuple(std::ref(log_r));
    onnx_(inputs, outputs);
  }

private:
  // onnx_helper_alloc creates Ort::Value views over the caller's tensor
  // data on every call, so static-shape Eigen tensors at the run site
  // are enough.  Slot selection inside onnx_helper_alloc is by TBB
  // thread index (see onnxutils.hpp), so the same model object is safe
  // under ``RunGraphs``.
  narf::onnx_helper_alloc onnx_;
};

}  // namespace shift_smear_reweight

// ---------------------------------------------------------------------------
// J/ψ scale-uncertainty drop-in
// ---------------------------------------------------------------------------
//
// Same signature, same tensor_axes, same downstream nuisance bookkeeping as
// JpsiCorrectionsUncHelperSplines.  Internally the analytic
// ``alt_weight = 1 + dwdqop · δqop`` is replaced by ``exp(log_r)`` from the
// trained network, with ``u_std = (δr_kappa) / target_std[0]`` and
// σ_std = 0 (scale variations don't smear).
//
// ``δr_kappa = q_gen · δκ_reco / κ_gen``.  With ``κ = q / (p_T · cosh η)``
// and assuming a charge match between reco and gen (the ~0.1% mismeasured
// tail is absorbed by the network's r_kappa ~ -2 mode), the conversion
// simplifies to ``δr_kappa = δqop_reco · p_gen``.

template <typename T> class JpsiCorrectionsUncReweightHelper {
public:
  using hist_t = T;
  using tensor_t = typename T::storage_type::value_type::tensor_t;
  static constexpr auto sizes = narf::tensor_traits<tensor_t>::sizes;
  static constexpr auto nUnc = sizes[sizes.size() - 1];
  // (u down, u up) per η × {A, e, M} variation.
  static constexpr std::size_t NVar = 2 * static_cast<std::size_t>(nUnc);
  using out_tensor_t = Eigen::TensorFixedSize<double, Eigen::Sizes<nUnc, 2>>;

  JpsiCorrectionsUncReweightHelper(T &&corrections,
                                   const std::string &onnx_path,
                                   unsigned int nslots = 1)
      : correctionHist_(std::make_shared<const T>(std::move(corrections))),
        model_(onnx_path, nslots) {}

  // Variadic templated bin lookup (lifted from JpsiCorrectionsUncHelperSplines
  // so the lookup interface stays identical).
  template <typename... Xs, std::size_t... Idxs>
  const tensor_t &get_tensor_impl(std::index_sequence<Idxs...>,
                                  const Xs &...xs) {
    return correctionHist_
        ->at(correctionHist_->template axis<Idxs>().index(xs)...)
        .data();
  }
  template <typename... Xs> const tensor_t &get_tensor(const Xs &...xs) {
    return get_tensor_impl(std::index_sequence_for<Xs...>{}, xs...);
  }

  out_tensor_t
  operator()(const RVec<float> &recPts, const RVec<float> &recEtas,
             const RVec<float> &recPhis, const RVec<int> &recCharges,
             const RVec<float> &genPts, const RVec<float> &genEtas,
             const RVec<float> &genPhis, const RVec<int> &genCharges,
             const RVec<int> &muonSources,
             const RVec<std::pair<double, double>> &response_weights,
             double nominal_weight = 1.0) {
    (void)response_weights;  // legacy linearisation column; not consulted
                             // -- the network gives the full weight directly.

    auto const nmuons = recPts.size();
    out_tensor_t alt_weights_all;
    alt_weights_all.setConstant(nominal_weight);

    // Per-muon scratch tensors (live on the stack).  Sigma stays zero
    // throughout for the scale-only variation.
    Eigen::TensorFixedSize<float, Eigen::Sizes<1, shift_smear_reweight::F>> y_t;
    Eigen::TensorFixedSize<float, Eigen::Sizes<1, shift_smear_reweight::NCond>>
        c_t;
    Eigen::TensorFixedSize<float,
                           Eigen::Sizes<1, NVar, shift_smear_reweight::F>>
        u_buf;
    Eigen::TensorFixedSize<float,
                           Eigen::Sizes<1, NVar, shift_smear_reweight::F>>
        sigma_buf;
    Eigen::TensorFixedSize<float, Eigen::Sizes<1, NVar>> log_r;
    sigma_buf.setZero();

    for (std::size_t i = 0; i < nmuons; ++i) {
      const float recPt = recPts[i];
      const float recEta = recEtas[i];
      const float recPhi = recPhis[i];
      const int recCharge = recCharges[i];
      const float genPt = genPts[i];
      const float genEta = genEtas[i];
      const float genPhi = genPhis[i];
      const int genCharge = genCharges[i];
      const int muonSource = shift_smear_reweight::muon_source_from_gen_part_flav(
          muonSources[i]);

      const float kappa_reco =
          static_cast<float>(recCharge) / (recPt * std::cosh(recEta));
      const float kappa_gen =
          static_cast<float>(genCharge) / (genPt * std::cosh(genEta));

      // Full (κ_reco, η_reco, φ_reco, κ_gen, η_gen, φ_gen) form for y
      // and (κ_gen, η_gen, φ_gen, muon_source) for c -- no φ ≈ 0 or
      // charge-match approximation. The κ ratio absorbs charge flips
      // natively: when sign(q_reco) ≠ sign(q_gen), κ_reco/κ_gen ≈
      // -p_gen/p_reco and r_kappa ≈ -2, which is exactly the mode the
      // network was trained on.
      std::array<float, shift_smear_reweight::F> y;
      std::array<float, shift_smear_reweight::NCond> c;
      shift_smear_reweight::compute_y_raw(kappa_reco, recEta, recPhi,
                                          kappa_gen, genEta, genPhi, y);
      shift_smear_reweight::compute_c_raw(kappa_gen, genEta, genPhi,
                                          muonSource, c);

      for (std::size_t k = 0; k < shift_smear_reweight::F; ++k) y_t(0, k) = y[k];
      for (std::size_t k = 0; k < shift_smear_reweight::NCond; ++k)
        c_t(0, k) = c[k];

      // δr_kappa per variation (in raw r_κ units; the ONNX preproc
      // divides by target_std internally):
      //   δr_kappa = δκ_reco / κ_gen = δqop_reco · sign(q_gen) · p_gen
      // ``calculateQopUnc`` returns the signed δqop_reco using the reco
      // charge internally (M term, leading sign); no further q_reco·q_gen
      // factor enters, so charge-flipped muons get the same conversion
      // as charge-matched ones.
      const auto &params = get_tensor(recEta);
      const double pgen = genPt * std::cosh(genEta);
      const double sign_qgen = (genCharge >= 0) ? 1.0 : -1.0;
      u_buf.setZero();

      for (std::ptrdiff_t ivar = 0; ivar < nUnc; ++ivar) {
        const double AUnc = params(0, ivar);
        const double eUnc = params(1, ivar);
        const double MUnc = params(2, ivar);
        const double recoQopUnc =
            calculateQopUnc(recPt, recEta, recCharge, AUnc, eUnc, MUnc);
        const float delta_r_kappa =
            static_cast<float>(recoQopUnc * pgen * sign_qgen);
        for (std::ptrdiff_t idownup = 0; idownup < 2; ++idownup) {
          const float dir = (idownup == 0) ? -1.0f : 1.0f;
          const std::size_t k =
              static_cast<std::size_t>(ivar) * 2 + idownup;
          u_buf(0, k, 0) = dir * delta_r_kappa;
        }
      }

      model_.template run<NVar>(y_t, c_t, u_buf, sigma_buf, log_r);

      out_tensor_t alt_weights;
      for (std::ptrdiff_t ivar = 0; ivar < nUnc; ++ivar) {
        for (std::ptrdiff_t idownup = 0; idownup < 2; ++idownup) {
          const std::size_t k =
              static_cast<std::size_t>(ivar) * 2 + idownup;
          alt_weights(ivar, idownup) =
              std::exp(static_cast<double>(log_r(0, k)));
        }
      }
      const out_tensor_t alt_weights_clamped = wrem::clip_tensor(alt_weights, 10.);
      alt_weights_all *= alt_weights_clamped;
    }
    return alt_weights_all;
  }

private:
  std::shared_ptr<const T> correctionHist_;
  shift_smear_reweight::ReweightModel model_;
};

// ---------------------------------------------------------------------------
// Resolution-uncertainty drop-in (signature extended with gen kinematics)
// ---------------------------------------------------------------------------
//
// Unlike the analytic SmearingUncertaintyHelperParametrized, which only
// needs (pt_reco, η_reco) plus a precomputed ``dweightdsigmasq``, the
// network-based reweight needs the full (κ_reco, κ_gen, η_reco, η_gen,
// φ_reco, φ_gen) tuple so we can form the model's y and c.  The signature
// is therefore extended with the gen columns; the one call site in
// ``wremnants.production.muon_calibration.add_resolution_uncertainty`` is
// updated accordingly.
//
// Per variation v we convert the analytic δσ²_rel(p, η) to an axis-aligned
// σ_std along r_kappa:
//   σ²_r = δσ²_rel · qop_reco² · p_gen²     (variance of δr_kappa from the
//                                            extra smear)
//   σ_std[0] = √max(0, σ²_r) / target_std[0]
// Variations with σ²_var < σ²_nom (rare; "negative smear") collapse to
// σ_std = 0 -- the model is even in σ so a small negative δσ² maps to no
// change in log W to lowest order, matching the analytic linearisation
// at leading order.

template <typename HISTNOM, typename HISTVAR, std::size_t NVar>
class SmearingUncertaintyReweightHelper
    : public SmearingHelperParametrized<HISTNOM> {
public:
  using base_t = SmearingHelperParametrized<HISTNOM>;
  using out_tensor_t = Eigen::TensorFixedSize<double, Eigen::Sizes<NVar>>;

  SmearingUncertaintyReweightHelper(const base_t &helper, HISTVAR &&hvar,
                                    const std::string &onnx_path,
                                    unsigned int nslots = 1)
      : base_t(helper),
        hvar_(std::make_shared<const HISTVAR>(std::move(hvar))),
        model_(onnx_path, nslots) {}

  out_tensor_t operator()(const RVec<float> &recPts, const RVec<float> &recEtas,
                          const RVec<float> &recPhis, const RVec<int> &recCharges,
                          const RVec<float> &genPts, const RVec<float> &genEtas,
                          const RVec<float> &genPhis, const RVec<int> &genCharges,
                          const RVec<int> &muonSources,
                          const RVec<std::pair<double, double>> &response_weights,
                          const double nominal_weight = 1.0) const {
    (void)response_weights;  // see Jpsi helper note.

    out_tensor_t res;
    res.setConstant(nominal_weight);

    Eigen::TensorFixedSize<float, Eigen::Sizes<1, shift_smear_reweight::F>> y_t;
    Eigen::TensorFixedSize<float, Eigen::Sizes<1, shift_smear_reweight::NCond>>
        c_t;
    Eigen::TensorFixedSize<float,
                           Eigen::Sizes<1, NVar, shift_smear_reweight::F>>
        u_buf;
    Eigen::TensorFixedSize<float,
                           Eigen::Sizes<1, NVar, shift_smear_reweight::F>>
        sigma_buf;
    Eigen::TensorFixedSize<float, Eigen::Sizes<1, NVar>> log_r;
    u_buf.setZero();  // resolution variations don't shift.

    for (std::size_t i = 0; i < recPts.size(); ++i) {
      const float recPt = recPts[i];
      const float recEta = recEtas[i];
      const float recPhi = recPhis[i];
      const int recCharge = recCharges[i];
      const float genPt = genPts[i];
      const float genEta = genEtas[i];
      const float genPhi = genPhis[i];
      const int genCharge = genCharges[i];
      const int muonSource = shift_smear_reweight::muon_source_from_gen_part_flav(
          muonSources[i]);

      const float kappa_reco =
          static_cast<float>(recCharge) / (recPt * std::cosh(recEta));
      const float kappa_gen =
          static_cast<float>(genCharge) / (genPt * std::cosh(genEta));
      // Full y / c with the real (φ_reco, φ_gen); no charge-match
      // assumption -- κ_reco/κ_gen folds the reco-charge sign in
      // natively. ``muonSource`` is the post-mapping integer code (1
      // or 15) the ORT graph will remap to {-1, 0} internally.
      std::array<float, shift_smear_reweight::F> y;
      std::array<float, shift_smear_reweight::NCond> c;
      shift_smear_reweight::compute_y_raw(kappa_reco, recEta, recPhi,
                                          kappa_gen, genEta, genPhi, y);
      shift_smear_reweight::compute_c_raw(kappa_gen, genEta, genPhi,
                                          muonSource, c);
      for (std::size_t k = 0; k < shift_smear_reweight::F; ++k) y_t(0, k) = y[k];
      for (std::size_t k = 0; k < shift_smear_reweight::NCond; ++k)
        c_t(0, k) = c[k];

      // Nominal σ²_rel(pt, η) from the base helper's histogram.
      auto const &resolution_parms = narf::get_value(base_t::hist(), recEta).data();
      const std::size_t idatamc = 0;
      const double anom = resolution_parms(0, idatamc);
      const double cnom = resolution_parms(1, idatamc);
      const double bnom = resolution_parms(2, idatamc);
      const double dnom = resolution_parms(3, idatamc);
      const double sigmasqnom_rel =
          anom + cnom * recPt * recPt + bnom / (1. + dnom / recPt / recPt);

      auto const &resolution_parms_var = narf::get_value(*hvar_, recEta).data();
      const double p_reco = recPt * std::cosh(recEta);
      const double qop_reco_sq = 1.0 / (p_reco * p_reco);
      const double pgen = genPt * std::cosh(genEta);
      const double pgen_sq = pgen * pgen;

      sigma_buf.setZero();
      for (std::size_t ivar = 0; ivar < NVar; ++ivar) {
        const double avar = resolution_parms_var(0, ivar);
        const double cvar = resolution_parms_var(1, ivar);
        const double bvar = resolution_parms_var(2, ivar);
        const double dvar = resolution_parms_var(3, ivar);
        const double sigmasqvar_rel =
            avar + cvar * recPt * recPt + bvar / (1. + dvar / recPt / recPt);
        // var(δr_kappa) from the extra smear above nominal.  Raw r_κ
        // units; the ONNX preproc divides by target_std internally.
        const double dsigmarelsq = sigmasqvar_rel - sigmasqnom_rel;
        const double var_r_kappa = dsigmarelsq * qop_reco_sq * pgen_sq;
        const double sigma_r_kappa =
            (var_r_kappa > 0.0) ? std::sqrt(var_r_kappa) : 0.0;
        sigma_buf(0, ivar, 0) = static_cast<float>(sigma_r_kappa);
      }

      model_.template run<NVar>(y_t, c_t, u_buf, sigma_buf, log_r);

      out_tensor_t alt_weights;
      for (std::size_t ivar = 0; ivar < NVar; ++ivar) {
        alt_weights(ivar) = std::exp(static_cast<double>(log_r(0, ivar)));
      }
      const out_tensor_t alt_weights_clamped =
          wrem::clip_tensor(alt_weights, 10.);
      res *= alt_weights_clamped;
    }
    return res;
  }

private:
  std::shared_ptr<const HISTVAR> hvar_;
  mutable shift_smear_reweight::ReweightModel model_;
};

}  // namespace wrem

#endif
