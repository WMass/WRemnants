#ifndef WREMNANTS_MUON_EFFICIENCIES_INSITU_H
#define WREMNANTS_MUON_EFFICIENCIES_INSITU_H

#include <array>
#include <boost/histogram/axis.hpp>
#include <cmath>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace wrem {

// In-situ muon efficiency helper for the W/Z muon analyses.
//
// Floats the ID (idip), HLT (trigger) and Iso efficiency data/MC scale
// factors as Chebyshev polynomials, decorrelated per NEta-bin probe-eta:
//   IDIP    : 1D in pt, charge-dependent  -> NCoeffPt per (eta, q)
//   Trigger : 2D in (pt, ut), charge-dep  -> NCoeffPt*NCoeffUt per (eta, q)
//   Iso     : 2D in (pt, ut), charge-incl -> NCoeffPt*NCoeffUt per (eta)
// matching the SMP-23-002 measurement variables.
//
//   xtilde_pt = 2*(clamp(pt,ptmin,ptmax)-ptmin)/(ptmax-ptmin) - 1
//   xtilde_ut = 2*(clamp(ut,utmin,utmax)-utmin)/(utmax-utmin) - 1
//   P_X = sum_k [, m] theta_{X,etaBin,[q],k[,m]} T_k(x_pt) [T_m(x_ut)]
//
// LINEAR parameterisation: the data/MC scale factor of a PASSING leg is the
// Chebyshev polynomial directly,
//   f_pass = 1 + P_X,
// UNRESTRICTED (P may push the implied data efficiency outside [0,1]). This
// makes the passing-leg weight independent of the MC efficiency, so the fitted
// coefficients are a pure data/MC ratio that transfers across analyses. With
// the MC efficiency eMC = effMC_X(pt,eta,q[,ut]) (clamped to (0, effMC_max] so
// 1-eMC > 0), the implied data efficiency is eMC*(1+P) and a FAILING leg
// contributes the ratio of fail efficiencies
//   f_fail = (1 - eMC*(1+P)) / (1 - eMC).
// 1+P is capped so that eMC*(1+P) <= effData_max < 1 (see fpass_capped): the
// unconstrained coefficients can otherwise put the linearisation point outside
// the physical region, where the fail probability would go negative.
// effMC therefore enters ONLY the fail factor: each histmaker uses its own MC
// (W effMC for the single-muon analysis, Z tag-and-probe effMC for the
// dilepton). The dilepton tag leg always passes idip & trigger, so its weight
// (1+P_idip)(1+P_trig) is MC-free -- there is a single effMC triplet, the
// probe's. At theta = 0: f_pass = f_fail = 1 -> nominal unchanged.
//
// NO safety net: where the data efficiency reaches/exceeds 1
// (1 - eMC*(1+P) <= 0) or 1 + P <= 0, the helper THROWS (the run aborts) rather
// than clamping -- a deliberate probe of whether the data are precise enough to
// keep P inside the physical region. The old logit form bounded eps in (0,1)
// and avoided this singularity at the cost of an eMC-dependent f_pass.
//
// rabbit treats every output-tensor bin as an independent linearised
// (log-normal) nuisance around the nominal. We store per coefficient the
// analytic first-order response:
//   res(c) = w_nom * exp( delta * d lnW / d theta_c )
// with, per step, d ln f / d theta_c = basis_c / (1+P) (pass) or
// -eMC*basis_c / (1 - eMC*(1+P)) (fail), basis_c = T_k(x_pt) [* T_m(x_ut)].
// The fail derivative ~ eMC/(1-eMC) grows at high efficiency (no longer bounded
// by the old |d ln f / d s| <= 1), so delta may need retuning. Coefficients are
// unconstrained, so delta only sets nuisance units (theta_c = delta * n_c).
//
// category: 0 = nominal, 1 = failIso, 2 = failHLT, 3 = failID.
//
// Two concrete helpers share the same implementation through a common base:
//   - muon_insitu_efficiency_helper          (two-leg, Z dilepton): probe +
//       tag muon variables; the tag leg always passes idip & trigger (the
//       dilepton tag-and-probe topology), iso is never tested on the tag.
//   - muon_insitu_efficiency_helper_singleleg (single-leg, W single muon):
//       only the probe (the single good muon) contributes; there is no tag
//       leg. Use category 0 (nominal) for signal muons that pass idip,
//       trigger and iso.
// Each derived class exposes a single operator() (distinct signatures) so the
// RDF backend dispatches unambiguously.

template <int NEta, int NCoeffPt, int NCoeffUt, typename HIST_IDIP,
          typename HIST_TRIG, typename HIST_ISO>
class muon_insitu_efficiency_helper_base {
public:
  static_assert(NCoeffPt <= 4, "Chebyshev pt order > 3 not implemented");
  static_assert(NCoeffUt <= 4, "Chebyshev ut order > 3 not implemented");

  // flat index layout: [ idip | trigger | iso ]
  static constexpr int NCoeff2D = NCoeffPt * NCoeffUt;
  static constexpr int nID = NEta * 2 * NCoeffPt;  // (eta, q) x pt
  static constexpr int nHLT = NEta * 2 * NCoeff2D; // (eta, q) x pt x ut
  static constexpr int nIso = NEta * NCoeff2D;     // eta x pt x ut
  static constexpr int NSF = nID + nHLT + nIso;

  using tensor_t = Eigen::TensorFixedSize<double, Eigen::Sizes<NSF>>;

  // theta_central (optional, last arg): the accumulated central Chebyshev
  // coefficients for the iterative fit. Empty -> all zero (SF=1, linearise
  // around the nominal == iteration 0). Otherwise the variations are
  // linearised, and the MC reweighted (via central_weight), around
  // f_pass = 1 + P(theta_central).
  //
  // A single effMC triplet (the probe's): it enters only the fail factor, and
  // the dilepton tag leg always passes (its weight 1+P is MC-free), so no
  // separate tag-side effMC is needed.
  muon_insitu_efficiency_helper_base(
      HIST_IDIP &&effMC_idip, HIST_TRIG &&effMC_trig, HIST_ISO &&effMC_iso,
      double ptmin, double ptmax, double utmin, double utmax,
      double delta = 0.01, double effMC_max = 0.9999,
      double effData_max = 0.9999,
      const std::vector<double> &theta_central = {})
      : effMC_idip_(std::make_shared<const HIST_IDIP>(std::move(effMC_idip))),
        effMC_trig_(std::make_shared<const HIST_TRIG>(std::move(effMC_trig))),
        effMC_iso_(std::make_shared<const HIST_ISO>(std::move(effMC_iso))),
        ptmin_(ptmin), ptmax_(ptmax), utmin_(utmin), utmax_(utmax),
        delta_(delta), effMC_max_(effMC_max), effData_max_(effData_max) {
    if (theta_central.empty()) {
      theta_central_.fill(0.0);
    } else {
      for (int i = 0; i < NSF; ++i)
        theta_central_[i] = theta_central[i];
    }
  }

protected:
  enum Step { IDIP = 0, TRIG = 1, ISO = 2 };

  // Shared response computation. WithTag adds the (always-passing idip &
  // trigger) tag leg at compile time; for the single-leg helper it is elided.
  template <bool WithTag>
  tensor_t compute(float probe_pt, float probe_eta, int probe_charge,
                   float probe_ut, float tag_pt, float tag_eta, int tag_charge,
                   float tag_ut, int category, double nominal_weight) const {

    tensor_t res;
    res.setConstant(nominal_weight);

    std::array<double, NSF> grad;
    grad.fill(0.0);

    // tag leg always passes idip and trigger (two-leg topology only); the
    // passing factor 1+P is MC-free, so no effMC is read for it
    if constexpr (WithTag) {
      accumulate(grad, tag_pt, tag_eta, tag_charge, tag_ut, IDIP, true);
      accumulate(grad, tag_pt, tag_eta, tag_charge, tag_ut, TRIG, true);
    }

    switch (category) {
    case 0: // nominal: probe passes idip, trigger, iso
      accumulate(grad, probe_pt, probe_eta, probe_charge, probe_ut, IDIP, true);
      accumulate(grad, probe_pt, probe_eta, probe_charge, probe_ut, TRIG, true);
      accumulate(grad, probe_pt, probe_eta, probe_charge, probe_ut, ISO, true);
      break;
    case 1: // failIso: probe passes idip, trigger, fails iso
      accumulate(grad, probe_pt, probe_eta, probe_charge, probe_ut, IDIP, true);
      accumulate(grad, probe_pt, probe_eta, probe_charge, probe_ut, TRIG, true);
      accumulate(grad, probe_pt, probe_eta, probe_charge, probe_ut, ISO, false);
      break;
    case 2: // failHLT: probe passes idip, fails trigger
      accumulate(grad, probe_pt, probe_eta, probe_charge, probe_ut, IDIP, true);
      accumulate(grad, probe_pt, probe_eta, probe_charge, probe_ut, TRIG,
                 false);
      break;
    case 3: // failID: probe fails idip
      accumulate(grad, probe_pt, probe_eta, probe_charge, probe_ut, IDIP,
                 false);
      break;
    default:
      break;
    }

    // store the per-coefficient log-normal response w_nom * exp(delta*dlnW/dt)
    for (int i = 0; i < NSF; ++i) {
      if (grad[i] != 0.0)
        res(i) = nominal_weight * std::exp(delta_ * grad[i]);
    }
    return res;
  }

  // Evaluate one leg/step: fill ``basis`` with the basis functions, set
  // ``ncoef`` to how many of them this step uses, ``off``
  // to the flat parameter offset of its block, and ``Pval`` to the polynomial
  // value P(theta_central) (the passing SF is 1+Pval). Shared by accumulate
  // (gradient) and central_factor (reweight) so both linearise around the SAME
  // point -> self-consistent fit.
  void eval_leg(float pt, float eta, int charge, float ut, Step step,
                double basis[NCoeff2D], int &ncoef, int &off,
                double &Pval) const {
    double p = pt;
    if (p < ptmin_)
      p = ptmin_;
    if (p > ptmax_)
      p = ptmax_;
    const double xpt = 2.0 * (p - ptmin_) / (ptmax_ - ptmin_) - 1.0;
    double Tpt[4];
    Tpt[0] = 1.0;
    Tpt[1] = xpt;
    Tpt[2] = 2.0 * xpt * xpt - 1.0;
    Tpt[3] = 4.0 * xpt * xpt * xpt - 3.0 * xpt;

    double Tut[4] = {1.0, 0.0, 0.0, 0.0};
    if (step != IDIP) {
      double u = ut;
      if (u < utmin_)
        u = utmin_;
      if (u > utmax_)
        u = utmax_;
      const double xut = 2.0 * (u - utmin_) / (utmax_ - utmin_) - 1.0;
      Tut[1] = xut;
      Tut[2] = 2.0 * xut * xut - 1.0;
      Tut[3] = 4.0 * xut * xut * xut - 3.0 * xut;
    }

    int b = effMC_idip_->template axis<0>().index(eta);
    if (b < 0)
      b = 0;
    if (b >= NEta)
      b = NEta - 1;
    const int qbit = (charge > 0) ? 1 : 0;

    // product basis and block offset
    if (step == IDIP) {
      ncoef = NCoeffPt;
      off = (qbit * NEta + b) * NCoeffPt;
      for (int k = 0; k < NCoeffPt; ++k)
        basis[k] = Tpt[k];
    } else {
      ncoef = NCoeff2D;
      if (step == TRIG) {
        off = nID + (qbit * NEta + b) * NCoeff2D;
      } else { // ISO (charge-inclusive)
        off = nID + nHLT + b * NCoeff2D;
      }
      for (int k = 0; k < NCoeffPt; ++k)
        for (int m = 0; m < NCoeffUt; ++m)
          basis[k * NCoeffUt + m] = Tpt[k] * Tut[m];
    }

    double s = 0.0;
    for (int j = 0; j < ncoef; ++j)
      s += theta_central_[off + j] * basis[j];
    Pval = s;
  }

  // effMC lookup for the probe leg; returns e clamped to (0, effMC_max_], or
  // 0.0 to signal "empty effMC -> skip this step". Only the fail factor reads
  // effMC.
  double lookup_effMC(Step step, float pt, float eta, int charge,
                      float ut) const {
    double e = 0.0;
    if (step == IDIP) {
      const int eta_i = effMC_idip_->template axis<0>().index(eta);
      const int pt_i = effMC_idip_->template axis<1>().index(pt);
      const int q_i = effMC_idip_->template axis<2>().index(charge);
      e = effMC_idip_->at(eta_i, pt_i, q_i).value();
    } else if (step == TRIG) {
      const int eta_i = effMC_trig_->template axis<0>().index(eta);
      const int pt_i = effMC_trig_->template axis<1>().index(pt);
      const int q_i = effMC_trig_->template axis<2>().index(charge);
      const int ut_i = effMC_trig_->template axis<3>().index(ut);
      e = effMC_trig_->at(eta_i, pt_i, q_i, ut_i).value();
    } else { // ISO (charge-inclusive)
      const int eta_i = effMC_iso_->template axis<0>().index(eta);
      const int pt_i = effMC_iso_->template axis<1>().index(pt);
      const int ut_i = effMC_iso_->template axis<2>().index(ut);
      e = effMC_iso_->at(eta_i, pt_i, ut_i).value();
    }
    if (!(e > 0.0))
      return 0.0;
    if (e > effMC_max_)
      e = effMC_max_;
    return e;
  }

  // Passing factor at the linearisation point, capped so the implied data
  // efficiency eMC*(1+P) stays <= effData_max_ < 1.
  //
  // The coefficients are unconstrained, so nothing stops the fit from leaving
  // the physical region where little data constrains it -- the corners of the
  // pt/ut window. A point with eMC*(1+P) >= 1 means a negative fail
  // probability: the fail factor's numerator 1 - eMC*(1+P) turns negative.
  // Capping projects such a point back onto the physical boundary so it can
  // still be used to linearise around. Value and derivative are both taken at
  // the capped point, so accumulate and central_factor stay consistent.
  //
  // This guards the linearisation point only; it does not constrain the fit,
  // which can still walk outside and needs a bounded parameterisation to stop
  // it at the source.
  double fpass_capped(double e, double Pval) const {
    const double fp = 1.0 + Pval;
    const double fp_max = effData_max_ / e;
    return fp < fp_max ? fp : fp_max;
  }

  // Accumulate the linearised gradient dlnW/dtheta_c at theta_central for one
  // leg/step. With basis_c = T_k(x_pt) [* T_m(x_ut)] and P = P(theta_central):
  //   pass: d ln(1+P)/dtheta_c               = basis_c / (1+P)   (MC-free)
  //   fail: d ln[(1-eMC(1+P))/(1-eMC)]/dt_c  = -eMC*basis_c / (1 - eMC*(1+P))
  // Throws if 1+P <= 0; an implied data efficiency eMC*(1+P) >= 1 is capped
  // back onto the physical boundary by fpass_capped rather than thrown on.
  void accumulate(std::array<double, NSF> &grad, float pt, float eta,
                  int charge, float ut, Step step, bool pass) const {
    double basis[NCoeff2D], Pval;
    int ncoef, off;
    eval_leg(pt, eta, charge, ut, step, basis, ncoef, off, Pval);

    double dfac;
    if (pass) {
      const double fp = 1.0 + Pval;
      if (fp <= 0.0)
        throw_unphysical(step, pt, eta, /*e=*/0.0, Pval, true);
      dfac = 1.0 / fp; // independent of effMC
    } else {
      const double e = lookup_effMC(step, pt, eta, charge, ut);
      if (e <= 0.0)
        return; // empty effMC -> no variation for this step
      const double den = 1.0 - e * fpass_capped(e, Pval);
      dfac = -e / den;
    }

    for (int j = 0; j < ncoef; ++j)
      grad[off + j] += basis[j] * dfac;
  }

  // Central per-leg/step factor f_X(theta_central):
  //   f_pass = 1 + P   (MC-free),   f_fail = (1 - eMC*(1+P)) / (1 - eMC).
  // Equals 1 at theta_central = 0. 1+P is capped by fpass_capped so the fail
  // numerator stays positive; 1+P <= 0 still throws.
  double central_factor(float pt, float eta, int charge, float ut, Step step,
                        bool pass) const {
    double basis[NCoeff2D], Pval;
    int ncoef, off;
    eval_leg(pt, eta, charge, ut, step, basis, ncoef, off, Pval);
    if (pass) {
      const double fp = 1.0 + Pval;
      if (fp <= 0.0)
        throw_unphysical(step, pt, eta, /*e=*/0.0, Pval, true);
      return fp;
    }
    const double e = lookup_effMC(step, pt, eta, charge, ut);
    if (e <= 0.0)
      return 1.0; // empty effMC -> no reweight
    const double den = 1.0 - e * fpass_capped(e, Pval);
    return den / (1.0 - e);
  }

  // Central MC reweight W(theta_central) = product of per-leg/step factors over
  // the tested steps of the category. 1.0 at theta_central = 0.
  template <bool WithTag>
  double central_weight(float probe_pt, float probe_eta, int probe_charge,
                        float probe_ut, float tag_pt, float tag_eta,
                        int tag_charge, float tag_ut, int category) const {
    double W = 1.0;
    if constexpr (WithTag) {
      // tag leg always passes: weight 1+P, MC-free
      W *= central_factor(tag_pt, tag_eta, tag_charge, tag_ut, IDIP, true);
      W *= central_factor(tag_pt, tag_eta, tag_charge, tag_ut, TRIG, true);
    }
    switch (category) {
    case 0: // nominal
      W *= central_factor(probe_pt, probe_eta, probe_charge, probe_ut, IDIP,
                          true);
      W *= central_factor(probe_pt, probe_eta, probe_charge, probe_ut, TRIG,
                          true);
      W *= central_factor(probe_pt, probe_eta, probe_charge, probe_ut, ISO,
                          true);
      break;
    case 1: // failIso
      W *= central_factor(probe_pt, probe_eta, probe_charge, probe_ut, IDIP,
                          true);
      W *= central_factor(probe_pt, probe_eta, probe_charge, probe_ut, TRIG,
                          true);
      W *= central_factor(probe_pt, probe_eta, probe_charge, probe_ut, ISO,
                          false);
      break;
    case 2: // failHLT
      W *= central_factor(probe_pt, probe_eta, probe_charge, probe_ut, IDIP,
                          true);
      W *= central_factor(probe_pt, probe_eta, probe_charge, probe_ut, TRIG,
                          false);
      break;
    case 3: // failID
      W *= central_factor(probe_pt, probe_eta, probe_charge, probe_ut, IDIP,
                          false);
      break;
    default:
      break;
    }
    // Each factor is checked positive in central_factor (throws otherwise), so
    // the product W stays finite and positive.
    return W;
  }

  // Fail-loudly on an unphysical linearisation point. Builds an informative
  // message and throws; never returns.
  [[noreturn]] void throw_unphysical(Step step, float pt, float eta, double e,
                                     double Pval, bool pass) const {
    const char *sname = (step == IDIP)   ? "idip"
                        : (step == TRIG) ? "trigger"
                                         : "iso";
    std::ostringstream os;
    os << "muon_insitu_efficiency: unphysical " << (pass ? "pass" : "fail")
       << " factor for step " << sname << " at pt=" << pt << " eta=" << eta
       << " effMC=" << e << " P=" << Pval << " -> "
       << (pass ? "1+P <= 0" : "data efficiency effMC*(1+P) >= 1")
       << " (in-situ SF outside the physical region)";
    throw std::runtime_error(os.str());
  }

  std::shared_ptr<const HIST_IDIP> effMC_idip_;
  std::shared_ptr<const HIST_TRIG> effMC_trig_;
  std::shared_ptr<const HIST_ISO> effMC_iso_;
  double ptmin_;
  double ptmax_;
  double utmin_;
  double utmax_;
  double delta_;
  double effMC_max_;
  double effData_max_;
  std::array<double, NSF> theta_central_;
};

// Two-leg helper (Z dilepton tag-and-probe).
template <int NEta, int NCoeffPt, int NCoeffUt, typename HIST_IDIP,
          typename HIST_TRIG, typename HIST_ISO>
class muon_insitu_efficiency_helper
    : public muon_insitu_efficiency_helper_base<
          NEta, NCoeffPt, NCoeffUt, HIST_IDIP, HIST_TRIG, HIST_ISO> {
public:
  using Base =
      muon_insitu_efficiency_helper_base<NEta, NCoeffPt, NCoeffUt, HIST_IDIP,
                                         HIST_TRIG, HIST_ISO>;
  using Base::Base;
  using typename Base::tensor_t;

  tensor_t operator()(float probe_pt, float probe_eta, int probe_charge,
                      float probe_ut, float tag_pt, float tag_eta,
                      int tag_charge, float tag_ut, int category,
                      double nominal_weight) const {
    return this->template compute<true>(probe_pt, probe_eta, probe_charge,
                                        probe_ut, tag_pt, tag_eta, tag_charge,
                                        tag_ut, category, nominal_weight);
  }
};

// Single-leg helper (W single muon): no tag leg.
template <int NEta, int NCoeffPt, int NCoeffUt, typename HIST_IDIP,
          typename HIST_TRIG, typename HIST_ISO>
class muon_insitu_efficiency_helper_singleleg
    : public muon_insitu_efficiency_helper_base<
          NEta, NCoeffPt, NCoeffUt, HIST_IDIP, HIST_TRIG, HIST_ISO> {
public:
  using Base =
      muon_insitu_efficiency_helper_base<NEta, NCoeffPt, NCoeffUt, HIST_IDIP,
                                         HIST_TRIG, HIST_ISO>;
  using Base::Base;
  using typename Base::tensor_t;

  tensor_t operator()(float probe_pt, float probe_eta, int probe_charge,
                      float probe_ut, int category,
                      double nominal_weight) const {
    return this->template compute<false>(probe_pt, probe_eta, probe_charge,
                                         probe_ut, 0.0f, 0.0f, 0, 0.0f,
                                         category, nominal_weight);
  }
};

// Central reweight functor (two-leg): returns the scalar W(theta_central) to
// fold into nominal_weight before the templates/tensors are filled. Shares the
// base (effMC + theta_central) with the tensor helper.
template <int NEta, int NCoeffPt, int NCoeffUt, typename HIST_IDIP,
          typename HIST_TRIG, typename HIST_ISO>
class muon_insitu_central_weight_helper
    : public muon_insitu_efficiency_helper_base<
          NEta, NCoeffPt, NCoeffUt, HIST_IDIP, HIST_TRIG, HIST_ISO> {
public:
  using Base =
      muon_insitu_efficiency_helper_base<NEta, NCoeffPt, NCoeffUt, HIST_IDIP,
                                         HIST_TRIG, HIST_ISO>;
  using Base::Base;

  double operator()(float probe_pt, float probe_eta, int probe_charge,
                    float probe_ut, float tag_pt, float tag_eta, int tag_charge,
                    float tag_ut, int category) const {
    return this->template central_weight<true>(
        probe_pt, probe_eta, probe_charge, probe_ut, tag_pt, tag_eta,
        tag_charge, tag_ut, category);
  }
};

// Central reweight functor (single-leg): no tag leg.
template <int NEta, int NCoeffPt, int NCoeffUt, typename HIST_IDIP,
          typename HIST_TRIG, typename HIST_ISO>
class muon_insitu_central_weight_helper_singleleg
    : public muon_insitu_efficiency_helper_base<
          NEta, NCoeffPt, NCoeffUt, HIST_IDIP, HIST_TRIG, HIST_ISO> {
public:
  using Base =
      muon_insitu_efficiency_helper_base<NEta, NCoeffPt, NCoeffUt, HIST_IDIP,
                                         HIST_TRIG, HIST_ISO>;
  using Base::Base;

  double operator()(float probe_pt, float probe_eta, int probe_charge,
                    float probe_ut, int category) const {
    return this->template central_weight<false>(probe_pt, probe_eta,
                                                probe_charge, probe_ut, 0.0f,
                                                0.0f, 0, 0.0f, category);
  }
};

} // namespace wrem

#endif
