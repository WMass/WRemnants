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
      : onnx_(std::make_shared<narf::onnx_helper_alloc>(onnx_path, nslots)) {}

  // Static-shape entry: y is [1, F], c is [1, NCond], u and σ are
  // [1, NVar, F], output is [1, NVar].  Buffers are caller-owned so
  // they can live on the stack across the per-muon loop.
  template <std::size_t NVar>
  void
  run(Eigen::TensorFixedSize<float, Eigen::Sizes<1, F>> &y,
      Eigen::TensorFixedSize<float, Eigen::Sizes<1, NCond>> &c,
      Eigen::TensorFixedSize<float, Eigen::Sizes<1, NVar, F>> &u_raw,
      Eigen::TensorFixedSize<float, Eigen::Sizes<1, NVar, F>> &sigma_raw,
      Eigen::TensorFixedSize<float, Eigen::Sizes<1, NVar>> &log_r) {
    // ``narf::onnx_helper_alloc::operator()`` takes input/output tuples of
    // references-to-tensor (the lambdas inside use ``auto&...``) and
    // forwards ``.data()`` to ``Ort::Value::CreateTensor<T>(... T* ...)``,
    // which is non-const even for input tensors. So the input tensors
    // here are deliberately non-const (the caller's scratch buffers are
    // freshly allocated per muon and immediately discarded).
    //
    // ``std::tie`` makes a tuple of plain references; ``std::cref`` would
    // wrap in ``reference_wrapper<>``, which has no ``narf::tensor_traits``
    // specialisation and would break ``::get_sizes()``.
    auto inputs = std::tie(y, c, u_raw, sigma_raw);
    auto outputs = std::tie(log_r);
    (*onnx_)(inputs, outputs);
  }

private:
  // onnx_helper_alloc creates Ort::Value views over the caller's tensor
  // data on every call. Slot selection inside is by TBB thread index
  // (see onnxutils.hpp), so the same instance is safe under RunGraphs.
  // Held via shared_ptr so the type is copyable -- RDataFrame Define
  // needs to copy the helper into its internal storage.
  std::shared_ptr<narf::onnx_helper_alloc> onnx_;
};

// Per-muon scale-shift evaluator: encapsulates the kinematics ->
// (y_raw, c_raw) packing, the u_buf fill from a caller-supplied
// ``delta_r_kappa[NVar]`` array, the ONNX call, and the
// ``exp(log_r)`` + ``clip_tensor`` step.  All ONNX-backed scale-
// uncertainty helpers (J/psi-stats, Z non-closure parametrized, Z
// non-closure binned, including the correlated variants) compose
// one of these and supply their own per-muon ``delta_r_kappa``;
// the per-muon kinematics handling is shared in one place.
//
// ``NVar`` is the per-muon shift multiplicity (2 for Z-non-closure
// down/up; 2 * n_unc for J/psi stats; etc.).
template <std::size_t NVar>
class ShiftReweightEvaluator {
public:
  using delta_r_t = std::array<float, NVar>;
  using alt_weights_t = Eigen::TensorFixedSize<double, Eigen::Sizes<NVar>>;

  ShiftReweightEvaluator(const std::string &onnx_path,
                         unsigned int nslots = 1)
      : model_(std::make_shared<ReweightModel>(onnx_path, nslots)) {}

  // Per-muon evaluation. ``muon_source_raw`` is the raw
  // ``Muon_genPartFlav`` value; the ORT graph remaps it internally.
  // ``delta_r_kappa[k]`` is the signed per-variation shift in raw
  // r_kappa units (caller computes this from its own correction hist).
  alt_weights_t
  evaluate(float recPt, float recEta, float recPhi, int recCharge,
           float genPt, float genEta, float genPhi, int genCharge,
           int muonSource_raw,
           const delta_r_t &delta_r_kappa) const {
    const float kappa_reco =
        static_cast<float>(recCharge) / (recPt * std::cosh(recEta));
    const float kappa_gen =
        static_cast<float>(genCharge) / (genPt * std::cosh(genEta));
    const int muon_source =
        muon_source_from_gen_part_flav(muonSource_raw);

    Eigen::TensorFixedSize<float, Eigen::Sizes<1, F>> y_t;
    Eigen::TensorFixedSize<float, Eigen::Sizes<1, NCond>> c_t;
    Eigen::TensorFixedSize<float, Eigen::Sizes<1, NVar, F>> u_buf;
    Eigen::TensorFixedSize<float, Eigen::Sizes<1, NVar, F>> sigma_buf;
    Eigen::TensorFixedSize<float, Eigen::Sizes<1, NVar>> log_r;

    std::array<float, F> y;
    std::array<float, NCond> c;
    compute_y_raw(kappa_reco, recEta, recPhi,
                  kappa_gen, genEta, genPhi, y);
    compute_c_raw(kappa_gen, genEta, genPhi, muon_source, c);
    for (std::size_t k = 0; k < F; ++k) y_t(0, k) = y[k];
    for (std::size_t k = 0; k < NCond; ++k) c_t(0, k) = c[k];

    u_buf.setZero();
    for (std::size_t k = 0; k < NVar; ++k) {
      u_buf(0, k, 0) = delta_r_kappa[k];
    }
    sigma_buf.setZero();

    model_->template run<NVar>(y_t, c_t, u_buf, sigma_buf, log_r);

    alt_weights_t alt_weights;
    for (std::size_t k = 0; k < NVar; ++k) {
      alt_weights(k) = std::exp(static_cast<double>(log_r(0, k)));
    }
    return wrem::clip_tensor(alt_weights, 10.);
  }

private:
  std::shared_ptr<ReweightModel> model_;
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
  using evaluator_t = shift_smear_reweight::ShiftReweightEvaluator<NVar>;

  JpsiCorrectionsUncReweightHelper(T &&corrections,
                                   const std::string &onnx_path,
                                   unsigned int nslots = 1)
      : correctionHist_(std::make_shared<const T>(std::move(corrections))),
        evaluator_(onnx_path, nslots) {}

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

  // Note: no ``response_weights`` column -- the trained network supplies
  // the full multiplicative reweight, so the spline-based dweightdqop
  // linearisation isn't consulted (and the SplinesDifferentialWeightsHelper
  // doesn't need to be evaluated at all when this helper is in use).
  out_tensor_t
  operator()(const RVec<float> &recPts, const RVec<float> &recEtas,
             const RVec<float> &recPhis, const RVec<int> &recCharges,
             const RVec<float> &genPts, const RVec<float> &genEtas,
             const RVec<float> &genPhis, const RVec<int> &genCharges,
             const RVec<int> &muonSources,
             double nominal_weight = 1.0) {
    auto const nmuons = recPts.size();
    out_tensor_t alt_weights_all;
    alt_weights_all.setConstant(nominal_weight);

    for (std::size_t i = 0; i < nmuons; ++i) {
      const float recPt = recPts[i];
      const float recEta = recEtas[i];
      const int recCharge = recCharges[i];
      const float genPt = genPts[i];
      const float genEta = genEtas[i];
      const int genCharge = genCharges[i];

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

      typename evaluator_t::delta_r_t delta_r_kappa;
      for (std::ptrdiff_t ivar = 0; ivar < nUnc; ++ivar) {
        const double AUnc = params(0, ivar);
        const double eUnc = params(1, ivar);
        const double MUnc = params(2, ivar);
        const double recoQopUnc =
            calculateQopUnc(recPt, recEta, recCharge, AUnc, eUnc, MUnc);
        const float dr =
            static_cast<float>(recoQopUnc * pgen * sign_qgen);
        delta_r_kappa[ivar * 2 + 0] = -dr;
        delta_r_kappa[ivar * 2 + 1] = +dr;
      }

      const auto alt_weights_flat = evaluator_.evaluate(
          recPt, recEta, recPhis[i], recCharge,
          genPt, genEta, genPhis[i], genCharge,
          muonSources[i], delta_r_kappa);

      out_tensor_t alt_weights;
      for (std::ptrdiff_t ivar = 0; ivar < nUnc; ++ivar) {
        for (std::ptrdiff_t idownup = 0; idownup < 2; ++idownup) {
          alt_weights(ivar, idownup) =
              alt_weights_flat(ivar * 2 + idownup);
        }
      }
      alt_weights_all *= alt_weights;
    }
    return alt_weights_all;
  }

private:
  std::shared_ptr<const T> correctionHist_;
  evaluator_t evaluator_;
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
        model_(std::make_shared<shift_smear_reweight::ReweightModel>(
            onnx_path, nslots)) {}

  // No ``response_weights`` column -- see JpsiCorrectionsUncReweightHelper.
  out_tensor_t operator()(const RVec<float> &recPts, const RVec<float> &recEtas,
                          const RVec<float> &recPhis, const RVec<int> &recCharges,
                          const RVec<float> &genPts, const RVec<float> &genEtas,
                          const RVec<float> &genPhis, const RVec<int> &genCharges,
                          const RVec<int> &muonSources,
                          const double nominal_weight = 1.0) const {
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

      model_->template run<NVar>(y_t, c_t, u_buf, sigma_buf, log_r);

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
  std::shared_ptr<shift_smear_reweight::ReweightModel> model_;
};

// ---------------------------------------------------------------------------
// Z non-closure ONNX drop-ins
// ---------------------------------------------------------------------------
//
// Drop-in replacements for ZNonClosure{Parametrized,Binned}Helper{,Splines}{,Corl}.
// The non-closure shift source (per-η parameterised A/e/M; per-(η,pT)
// binned non-closure factor) is unchanged; only the per-muon evaluation
// is routed through ``shift_smear_reweight::ShiftReweightEvaluator``
// instead of the analytic splines-linearisation
// (``alt_weight = dweightdqop · δqop + 1``).  ``NVar = 2`` (down/up).
//
// The Splines and analytic variants take different column lists (the
// Splines ones consume the ``response_weight`` column, the analytic
// ones take genQops / recoQops / covs); the ONNX variants here align
// with the J/psi-style ONNX column list (no ``response_weight``;
// kinematics + ``muon_source``), so the histmaker call sites can share
// the same column-list builder for every J/psi-style ONNX helper.

namespace shift_smear_reweight {

// Internal helper: compute the per-muon (down, up) δr_kappa array for
// the Z-non-closure parametrized source.  ``params`` is the per-η
// tensor row [A, e, M], ``calVarFlags`` selects which subset of the
// three contributes (A=1, e=2, M=4 -- bit flags).
//
// Mirrors the body of ``ZNonClosureParametrizedHelperSplines::operator()``
// up to (and including) the ``recoQopUnc`` calculation; the (down, up)
// signed shift then converts to raw r_kappa units the same way the
// J/psi-stats helper does (``δr_kappa = δqop_reco · p_gen · sign_qgen``).
template <typename ParamsT>
inline std::array<float, 2> z_non_closure_param_delta_r_kappa(
    const ParamsT &params,
    float recPt, float recEta, int recCharge,
    float genPt, float genEta, int genCharge,
    int calVarFlags) {
  double recoK = 1.0 / recPt;
  double recoKUnc = 0.0;
  enum calVarFlagsScheme { AFlag = 1, eFlag = 2, MFlag = 4 };
  if (calVarFlags & AFlag) {
    const double AUnc = params(0);
    recoKUnc += AUnc * recoK;
  }
  if (calVarFlags & eFlag) {
    const double eUnc = params(1);
    recoKUnc += -1.0 * eUnc * recoK * recoK;
  }
  if (calVarFlags & MFlag) {
    const double MUnc = params(2);
    recoKUnc += static_cast<double>(recCharge) * MUnc;
  }
  const double recoQopUnc =
      static_cast<double>(recCharge) *
      std::sin(calculateTheta(recEta)) * recoKUnc;
  const double pgen = genPt * std::cosh(genEta);
  const double sign_qgen = (genCharge >= 0) ? 1.0 : -1.0;
  const float dr = static_cast<float>(recoQopUnc * pgen * sign_qgen);
  return {-dr, +dr};
}

// Same for the binned source: ``non_closure`` is the per-(η, pT) bin
// value of the non-closure factor (1.0 = no shift).
inline std::array<float, 2> z_non_closure_binned_delta_r_kappa(
    double non_closure,
    float recPt, float recEta, int recCharge,
    float genPt, float genEta, int genCharge) {
  const double recoKUnc = (non_closure - 1.0) * (1.0 / recPt);
  const double recoQopUnc = calculateQopUnc(recEta, recCharge, recoKUnc);
  const double pgen = genPt * std::cosh(genEta);
  const double sign_qgen = (genCharge >= 0) ? 1.0 : -1.0;
  const float dr = static_cast<float>(recoQopUnc * pgen * sign_qgen);
  return {-dr, +dr};
}

}  // namespace shift_smear_reweight

// Parametrized, correlated: one [down, up] tensor for the whole event.
template <typename T, std::size_t NEtaBins>
class ZNonClosureParametrizedReweightHelperCorl {
public:
  using hist_t = T;
  using out_tensor_t = Eigen::TensorFixedSize<double, Eigen::Sizes<2>>;
  using evaluator_t = shift_smear_reweight::ShiftReweightEvaluator<2>;

  ZNonClosureParametrizedReweightHelperCorl(T &&corrections,
                                            const std::string &onnx_path,
                                            unsigned int nslots = 1)
      : correctionHist_(std::make_shared<const T>(std::move(corrections))),
        evaluator_(onnx_path, nslots) {}

  out_tensor_t
  operator()(const RVec<float> &recPts, const RVec<float> &recEtas,
             const RVec<float> &recPhis, const RVec<int> &recCharges,
             const RVec<float> &genPts, const RVec<float> &genEtas,
             const RVec<float> &genPhis, const RVec<int> &genCharges,
             const RVec<int> &muonSources,
             double nominal_weight = 1.0,
             int calVarFlags = 7) {
    out_tensor_t res;
    res.setConstant(nominal_weight);
    for (std::size_t i = 0; i < recPts.size(); ++i) {
      const unsigned int iEta = std::clamp(
          correctionHist_->template axis<0>().index(recEtas[i]),
          0, int(NEtaBins) - 1);
      const auto &params = correctionHist_->at(iEta).data();
      const auto delta_r_kappa =
          shift_smear_reweight::z_non_closure_param_delta_r_kappa(
              params, recPts[i], recEtas[i], recCharges[i],
              genPts[i], genEtas[i], genCharges[i], calVarFlags);
      const auto alt_weights = evaluator_.evaluate(
          recPts[i], recEtas[i], recPhis[i], recCharges[i],
          genPts[i], genEtas[i], genPhis[i], genCharges[i],
          muonSources[i], delta_r_kappa);
      res(0) *= alt_weights(0);
      res(1) *= alt_weights(1);
    }
    return res;
  }

private:
  std::shared_ptr<const T> correctionHist_;
  evaluator_t evaluator_;
};

// Parametrized, decorrelated: one [down, up] tensor per η bin.
template <typename T, std::size_t NEtaBins>
class ZNonClosureParametrizedReweightHelper {
public:
  using hist_t = T;
  using out_tensor_t =
      Eigen::TensorFixedSize<double, Eigen::Sizes<NEtaBins, 2>>;
  using evaluator_t = shift_smear_reweight::ShiftReweightEvaluator<2>;

  ZNonClosureParametrizedReweightHelper(T &&corrections,
                                        const std::string &onnx_path,
                                        unsigned int nslots = 1)
      : correctionHist_(std::make_shared<const T>(std::move(corrections))),
        evaluator_(onnx_path, nslots) {}

  out_tensor_t
  operator()(const RVec<float> &recPts, const RVec<float> &recEtas,
             const RVec<float> &recPhis, const RVec<int> &recCharges,
             const RVec<float> &genPts, const RVec<float> &genEtas,
             const RVec<float> &genPhis, const RVec<int> &genCharges,
             const RVec<int> &muonSources,
             double nominal_weight = 1.0,
             int calVarFlags = 7) {
    out_tensor_t res;
    res.setConstant(nominal_weight);
    for (std::size_t i = 0; i < recPts.size(); ++i) {
      const unsigned int iEta = std::clamp(
          correctionHist_->template axis<0>().index(recEtas[i]),
          0, int(NEtaBins) - 1);
      const auto &params = correctionHist_->at(iEta).data();
      const auto delta_r_kappa =
          shift_smear_reweight::z_non_closure_param_delta_r_kappa(
              params, recPts[i], recEtas[i], recCharges[i],
              genPts[i], genEtas[i], genCharges[i], calVarFlags);
      const auto alt_weights = evaluator_.evaluate(
          recPts[i], recEtas[i], recPhis[i], recCharges[i],
          genPts[i], genEtas[i], genPhis[i], genCharges[i],
          muonSources[i], delta_r_kappa);
      res(iEta, 0) *= alt_weights(0);
      res(iEta, 1) *= alt_weights(1);
    }
    return res;
  }

private:
  std::shared_ptr<const T> correctionHist_;
  evaluator_t evaluator_;
};

// Binned, correlated.
template <typename T, std::size_t NEtaBins, std::size_t NPtBins>
class ZNonClosureBinnedReweightHelperCorl {
public:
  using hist_t = T;
  using out_tensor_t = Eigen::TensorFixedSize<double, Eigen::Sizes<2>>;
  using evaluator_t = shift_smear_reweight::ShiftReweightEvaluator<2>;

  ZNonClosureBinnedReweightHelperCorl(T &&corrections,
                                      const std::string &onnx_path,
                                      unsigned int nslots = 1)
      : correctionHist_(std::make_shared<const T>(std::move(corrections))),
        evaluator_(onnx_path, nslots) {}

  out_tensor_t
  operator()(const RVec<float> &recPts, const RVec<float> &recEtas,
             const RVec<float> &recPhis, const RVec<int> &recCharges,
             const RVec<float> &genPts, const RVec<float> &genEtas,
             const RVec<float> &genPhis, const RVec<int> &genCharges,
             const RVec<int> &muonSources,
             double nominal_weight = 1.0) {
    out_tensor_t res;
    res.setConstant(nominal_weight);
    for (std::size_t i = 0; i < recPts.size(); ++i) {
      const unsigned int iEta = std::clamp(
          correctionHist_->template axis<0>().index(recEtas[i]),
          0, int(NEtaBins) - 1);
      const unsigned int iPt = std::clamp(
          correctionHist_->template axis<1>().index(recPts[i]),
          0, int(NPtBins) - 1);
      const double non_closure = correctionHist_->at(iEta, iPt).value();
      const auto delta_r_kappa =
          shift_smear_reweight::z_non_closure_binned_delta_r_kappa(
              non_closure, recPts[i], recEtas[i], recCharges[i],
              genPts[i], genEtas[i], genCharges[i]);
      const auto alt_weights = evaluator_.evaluate(
          recPts[i], recEtas[i], recPhis[i], recCharges[i],
          genPts[i], genEtas[i], genPhis[i], genCharges[i],
          muonSources[i], delta_r_kappa);
      res(0) *= alt_weights(0);
      res(1) *= alt_weights(1);
    }
    return res;
  }

private:
  std::shared_ptr<const T> correctionHist_;
  evaluator_t evaluator_;
};

// Binned, decorrelated.
template <typename T, std::size_t NEtaBins, std::size_t NPtBins>
class ZNonClosureBinnedReweightHelper {
public:
  using hist_t = T;
  using out_tensor_t =
      Eigen::TensorFixedSize<double, Eigen::Sizes<NEtaBins, NPtBins, 2>>;
  using evaluator_t = shift_smear_reweight::ShiftReweightEvaluator<2>;

  ZNonClosureBinnedReweightHelper(T &&corrections,
                                  const std::string &onnx_path,
                                  unsigned int nslots = 1)
      : correctionHist_(std::make_shared<const T>(std::move(corrections))),
        evaluator_(onnx_path, nslots) {}

  out_tensor_t
  operator()(const RVec<float> &recPts, const RVec<float> &recEtas,
             const RVec<float> &recPhis, const RVec<int> &recCharges,
             const RVec<float> &genPts, const RVec<float> &genEtas,
             const RVec<float> &genPhis, const RVec<int> &genCharges,
             const RVec<int> &muonSources,
             double nominal_weight = 1.0) {
    out_tensor_t res;
    res.setConstant(nominal_weight);
    for (std::size_t i = 0; i < recPts.size(); ++i) {
      const unsigned int iEta = std::clamp(
          correctionHist_->template axis<0>().index(recEtas[i]),
          0, int(NEtaBins) - 1);
      const unsigned int iPt = std::clamp(
          correctionHist_->template axis<1>().index(recPts[i]),
          0, int(NPtBins) - 1);
      const double non_closure = correctionHist_->at(iEta, iPt).value();
      const auto delta_r_kappa =
          shift_smear_reweight::z_non_closure_binned_delta_r_kappa(
              non_closure, recPts[i], recEtas[i], recCharges[i],
              genPts[i], genEtas[i], genCharges[i]);
      const auto alt_weights = evaluator_.evaluate(
          recPts[i], recEtas[i], recPhis[i], recCharges[i],
          genPts[i], genEtas[i], genPhis[i], genCharges[i],
          muonSources[i], delta_r_kappa);
      res(iEta, iPt, 0) *= alt_weights(0);
      res(iEta, iPt, 1) *= alt_weights(1);
    }
    return res;
  }

private:
  std::shared_ptr<const T> correctionHist_;
  evaluator_t evaluator_;
};

}  // namespace wrem

#endif
