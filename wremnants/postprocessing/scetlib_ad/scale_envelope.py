"""Frozen gen-level ENVELOPE nuisances for the scetlib_ad param model.

Why this exists
---------------
``set_diff_scales(1)`` makes SCETlib's profile scales differentiable, so the
param model *can* profile ``kappa_R`` and ``kappa_F`` directly. It must not,
and the reason is physics rather than convenience:

* the analysis does not treat them as parameters at all. AN-25-085
  (``uncerts.tex:173``) assigns the missing-higher-order uncertainty as "the
  envelope of the 7 variations obtained by varying each scale up and down by a
  factor of 2, while keeping 0.5 <= muR/muF <= 2", symmetrized "using the
  quadratic approach also used for PDF uncertainties";
* a profiled ``kappa_F`` is not even a well-posed nuisance. Its response is
  EVEN in its own theta (measured |even| = 62.9 sigma against |odd| = 17.7
  sigma on the Z, ``studies/scetlib-ad-param-model/260911-basin-nuisance-shapes``),
  because the prediction depends on ln^2(kappa_F) at leading log. A single
  symmetric Gaussian prior on such a direction has TWO mirror minima, and the
  bimodality that produced in alpha_s is what sent us looking;
* and 93 % of that free direction's pull on the fit sat at qT < 10 GeV, where
  the scale uncertainty is not where the analysis puts its uncertainty at all
  (resummation and the TNPs are).

An envelope fixes all three. It is a fixed per-gen-bin SHAPE, so its response is
linear in theta by construction and has exactly one minimum; it is the AN's own
prescription; and it can be zeroed where the AN says it does not belong.

What it is, precisely
---------------------
Build, ONCE, at the anchor:

1. evaluate sigma_gen at the 7 points (kappa_R, kappa_F) of :data:`SEVEN_POINT`;
2. per gen bin, take the largest UP and largest DOWN log-deviation from the
   nominal -- that is the envelope, and the nominal (1, 1) leg is what makes it
   contain zero, so an all-up or all-down bin still gets a one-sided envelope;
3. symmetrize into TWO nuisances with rabbit's own convention
   (``rabbit/tensorwriter.py:376-383``, ``symmetrize="quadratic"``)::

       logkup   =  max_legs ln(sigma_leg / sigma_nom)      >= 0
       logkdown = -min_legs ln(sigma_leg / sigma_nom)      >= 0
       k_avg    = 0.5 * (logkup + logkdown)
       k_diff   = 0.5 * sqrt(3) * (logkup - logkdown)

   (``logkdown`` carries rabbit's sign flip, so both are "up-like": a perfectly
   symmetric envelope gives ``k_avg = e``, ``k_diff = 0``.)
4. HARD-ZERO every gen bin below :attr:`EnvelopeSpec.qt_min`.

The two nuisances then enter as an ordinary gen-level template,

    sigma_gen -> sigma_gen * (1 + theta_avg * k_avg + theta_diff * k_diff)

folded through the response matrix like any other gen-level shape.

FROZEN is a requirement, not an optimisation: a per-bin ``max`` over legs is not
differentiable, and re-deriving it at every minimiser step would hand rabbit a
kinked prediction wherever the winning leg changes.

Two deliberate choices, so they are arguable rather than buried
--------------------------------------------------------------
* **linear, not log-normal.** rabbit applies a card nuisance as
  ``exp(theta * logk)``; this applies ``1 + theta * k``. They agree to O(k^2),
  and ``k`` here is a per-cent-scale number, so the difference is ~1e-4 of the
  response. Linear is what makes the direction *exactly* odd in theta, which is
  the whole point of replacing the profiled scale.
* **the 20 GeV cut is a CHOICE, and it is not literally the AN's.** The AN says
  the scale uncertainty has "a negligible effect at ptll < 20 GeV" *by
  construction* -- an observation, not a cut. Imposing it as a hard zero is a
  decision (Luca, 2026-09-11) taken because the free-parameter treatment put
  almost all of its leverage exactly there. Set ``qt_min=None`` to switch it off
  and see the difference.

What production actually does, which is NOT the AN's 7 points
-------------------------------------------------------------
Worth knowing before comparing to a card. The production correction file carries
BOTH envelopes, precomputed with exactly this prescription
(``wremnants/production/theory_corrections.py:918-1026``, ``compute_envelope``):

* ``renorm_fact_scale_pt20_envelope``  -- the 7 points, i.e. :data:`SEVEN_POINT`;
* ``renorm_scale_pt20_envelope``       -- only THREE: nominal, (0.5, 1), (2, 1),
  i.e. mu_R alone.

and the datacard nuisance ``resumFOScaleZSymAvg`` / ``SymDiff`` is built from the
**three-point** one (``theory_variation_labels.TRANSITION_FO_UNCERTAINTIES[1]``
names ``renorm_scale_pt20_envelope_Up/Down``). So the card's scale uncertainty is
mu_R-only, while AN-25-085 describes the 7-point envelope. Both are available
here -- :func:`seven_point_scale_spec` and :func:`three_point_mur_scale_spec` --
and the three-point one is what a like-for-like closure against an old card has
to use. It matters: the 7-point envelope is ~45 % larger (yield-weighted, above
20 GeV, at reco level; ``studies/scetlib-ad-param-model/260911-scale-envelope``).

Production's own conventions, which this module matches deliberately:

* the min/max is taken over the entries INCLUDING the central, so the envelope
  brackets the nominal by construction (``compute_envelope``, lines 925-929);
* the pt20 restriction REPLACES every bin below the threshold with the nominal,
  which is the same thing as a zero log-shift, and the threshold is resolved as
  ``axes["qT"].index(20.0)`` -- a half-open bin lookup, so the bin starting at
  20 keeps the envelope and everything strictly below it does not. That is the
  "lower edge >= qt_min" convention in :meth:`FrozenEnvelope._qt_mask`.
  Production's comment for why: "restricted to qT>20GeV to capture only the
  fixed order part of the variation and neglect the part at low pt which should
  be redundant with the TNPs".

The spec is a small named object rather than inline constants so another
direction family can be enveloped later without rewriting the model. Note that
``resumTransition2`` deliberately is NOT one of them: the transition points are
profiled directly (Luca, 2026-09-11).
"""

import math

import numpy as np
import tensorflow as tf

SQRT3 = math.sqrt(3.0)

# (kappa_R, kappa_F) over {0.5, 1, 2}^2 minus the two with ratio 4 or 1/4, i.e.
# the AN's "each scale up and down by a factor of 2, keeping 0.5 <= muR/muF <= 2".
# The nominal is FIRST and is the anchor itself; it is not re-evaluated.
SEVEN_POINT = (
    (1.0, 1.0),
    (2.0, 1.0),
    (0.5, 1.0),
    (1.0, 2.0),
    (1.0, 0.5),
    (2.0, 2.0),
    (0.5, 0.5),
)

# Rabbit-facing names of the two SCETlib parameters the legs move. SCETlib's
# own ``set_muR_factor`` scales mu_R at FIXED mu_F (``kappaFO *= f;
# kappaf /= f``, DrellYan.cpp:260-270, header DrellYan.hpp:171-173, since
# muF = kappaf * kappaFO * Q), and the differentiable ``scale_kappa_R`` enters
# the same way (``L_R = 2 ln(kappa_R muFO / mu_ref)``, DrellYanAD.cpp:2201-2225)
# while ``scale_kappa_F`` drives the muF member pair only. So these two ARE
# (kappa_R, kappa_F) with the other scale held fixed, and the production
# template named ``kappaFO2.-kappaf0.5`` is the (2, 1) leg.
KAPPA_R = "resumScaleMuR"
KAPPA_F = "resumScaleMuF"


class EnvelopeSpec:
    """A named family of gen-level legs to be enveloped and symmetrized.

    Parameters
    ----------
    name
        Stem of the two rabbit-facing nuisance names (``<name>SymAvg`` /
        ``<name>SymDiff``).
    legs
        Physical parameter overrides per leg, ``{rabbit_name: value}``. The
        FIRST entry must be the nominal, i.e. the anchor's own values; it is
        verified against the anchor and then contributes ``ln ratio = 0``
        without being evaluated.
    qt_min
        Gen bins whose qT range is not entirely at or above this are zeroed.
        ``None`` applies the envelope everywhere.
    qt_axis_index
        Which gen axis is qT. The model's gen axes are ``(qT, |Y|)``.
    diff_fact
        ``sqrt(3)`` for rabbit's ``symmetrize="quadratic"`` (the AN's choice),
        ``1.0`` for ``"linear"``.
    """

    RESPONSES = ("linear", "lognormal")

    def __init__(
        self,
        name,
        legs,
        qt_min=None,
        qt_axis_index=0,
        diff_fact=SQRT3,
        response="linear",
    ):
        if len(legs) < 2:
            raise ValueError(
                "EnvelopeSpec: need a nominal leg plus at least one variation"
            )
        if response not in self.RESPONSES:
            raise ValueError(
                f"EnvelopeSpec: response must be one of {self.RESPONSES}, "
                f"got {response!r}"
            )
        self.name = str(name)
        self.legs = tuple(dict(leg) for leg in legs)
        self.qt_min = None if qt_min is None else float(qt_min)
        self.qt_axis_index = int(qt_axis_index)
        self.diff_fact = float(diff_fact)
        self.response = str(response)

    @property
    def param_names(self):
        """The two rabbit-facing nuisance names, in rabbit's layout order."""
        return (self.name + "SymAvg", self.name + "SymDiff")

    @property
    def leg_labels(self):
        return tuple(
            "nominal" if not leg else ",".join(f"{k}={v:g}" for k, v in leg.items())
            for leg in self.legs
        )

    def __repr__(self):
        cut = "no qT cut" if self.qt_min is None else f"qT >= {self.qt_min:g} GeV"
        return f"EnvelopeSpec({self.name}, {len(self.legs)} legs, {cut})"


# The three points the production DATACARD's resumFOScaleZ actually uses:
# mu_R alone (theory_corrections.renorm_scale_vars). Kept as its own constant so
# the difference from SEVEN_POINT is visible rather than a slice index.
THREE_POINT_MUR = ((1.0, 1.0), (0.5, 1.0), (2.0, 1.0))

POINT_SETS = {7: SEVEN_POINT, 3: THREE_POINT_MUR}


def scale_spec(points=3, qt_min=20.0, name="resumFOScaleEnv", response="linear"):
    """Envelope spec over a named (mu_R, mu_F) point set.

    ``points=3`` (DEFAULT) is mu_R alone -- what the production datacard's
    ``resumFOScaleZ`` is actually built from. ``points=7`` is the envelope the
    AN TEXT describes, over (mu_R, mu_F); the correction file computes it too
    but no datacard consumes it. See the module docstring for why they differ
    and by how much.
    """
    if points not in POINT_SETS:
        raise ValueError(
            f"scale_spec: points must be one of {sorted(POINT_SETS)}, got {points!r}"
        )
    legs = [{KAPPA_R: kr, KAPPA_F: kf} for kr, kf in POINT_SETS[points]]
    return EnvelopeSpec(name, legs, qt_min=qt_min, response=response)


def seven_point_scale_spec(qt_min=20.0, name="resumFOScaleEnv", response="linear"):
    """The AN's 7-point (mu_R, mu_F) envelope as an :class:`EnvelopeSpec`."""
    return scale_spec(7, qt_min=qt_min, name=name, response=response)


def three_point_mur_scale_spec(qt_min=20.0, name="resumFOScaleEnv", response="linear"):
    """The production datacard's mu_R-only 3-point envelope."""
    return scale_spec(3, qt_min=qt_min, name=name, response=response)


class FrozenEnvelope:
    """The evaluated envelope: two constant per-gen-bin shapes, plus provenance.

    ``sigma_fn(overrides)`` must return sigma_gen FLAT over the gen grid at the
    anchor with those PHYSICAL overrides applied -- the model's
    ``sigma_gen_at`` does exactly that.
    """

    def __init__(self, spec, sigma_fn, gen_axes, sigma_nom=None, dtype=tf.float64):
        self.spec = spec
        self.gen_axes = [(n, np.asarray(e, dtype=np.float64)) for n, e in gen_axes]
        self.gen_shape = tuple(len(e) - 1 for _, e in self.gen_axes)

        nominal = spec.legs[0]
        sig = np.asarray(
            sigma_nom if sigma_nom is not None else sigma_fn(nominal), dtype=np.float64
        ).reshape(-1)
        if not np.all(sig > 0):
            raise ValueError(
                f"scale_envelope[{spec.name}]: {int(np.sum(sig <= 0))} of {sig.size} "
                "gen bins have non-positive sigma at the nominal point, so the "
                "log-ratio envelope is undefined."
            )
        self.sigma_nom = sig

        # ln(sigma_leg / sigma_nom) per variation leg. The nominal leg is not
        # evaluated: it is exactly 0 by construction, and it is what guarantees
        # the envelope brackets the nominal (a bin whose legs all go the same way
        # gets a one-sided envelope rather than a spurious two-sided one).
        d = np.zeros((len(spec.legs), sig.size), dtype=np.float64)
        for i, leg in enumerate(spec.legs[1:], start=1):
            s = np.asarray(sigma_fn(leg), dtype=np.float64).reshape(-1)
            if s.shape != sig.shape:
                raise ValueError(
                    f"scale_envelope[{spec.name}]: leg {spec.leg_labels[i]} returned "
                    f"{s.shape} bins, nominal has {sig.shape}"
                )
            if not np.all(s > 0):
                raise ValueError(
                    f"scale_envelope[{spec.name}]: leg {spec.leg_labels[i]} has "
                    f"{int(np.sum(s <= 0))} non-positive gen bins."
                )
            d[i] = np.log(s / sig)
        self.log_ratio = d

        self.up_leg = np.argmax(d, axis=0)
        self.down_leg = np.argmin(d, axis=0)
        logkup = d[self.up_leg, np.arange(sig.size)]
        logkdown = -d[self.down_leg, np.arange(sig.size)]

        # rabbit/tensorwriter.py:376-383, symmetrize="quadratic".
        k_avg = 0.5 * (logkup + logkdown)
        k_diff = 0.5 * spec.diff_fact * (logkup - logkdown)

        self.qt_mask = self._qt_mask()
        self.k_avg_raw, self.k_diff_raw = k_avg.copy(), k_diff.copy()
        self.k_avg = np.where(self.qt_mask, k_avg, 0.0)
        self.k_diff = np.where(self.qt_mask, k_diff, 0.0)

        self._k = tf.constant(np.stack([self.k_avg, self.k_diff], axis=0), dtype=dtype)
        self.dtype = dtype

    # -- construction helpers -------------------------------------------------

    def _qt_mask(self):
        """True where the envelope applies.

        Convention, stated because the threshold need not be a bin edge: a gen
        bin carries the uncertainty only if it lies ENTIRELY at or above
        ``qt_min``, i.e. its LOWER edge is >= qt_min. A bin straddling the
        threshold is zeroed. That is the conservative reading of "does not apply
        below 20 GeV" -- it never applies the uncertainty to phase space the
        prescription excludes -- and it is exact when 20 GeV is an edge, which
        is the case the model checks for and reports.
        """
        n = int(np.prod(self.gen_shape))
        if self.spec.qt_min is None:
            return np.ones(n, dtype=bool)
        edges = self.gen_axes[self.spec.qt_axis_index][1]
        lo = edges[:-1] >= self.spec.qt_min
        # Broadcast the qT decision over the remaining gen axes, in the model's
        # own flatten order (C order over self.gen_shape).
        shape = [1] * len(self.gen_shape)
        shape[self.spec.qt_axis_index] = -1
        return np.broadcast_to(lo.reshape(shape), self.gen_shape).reshape(n).copy()

    # -- use -----------------------------------------------------------------

    @property
    def param_names(self):
        return self.spec.param_names

    @property
    def nparams(self):
        return 2

    def factor_tf(self, theta):
        """The multiplicative gen-bin factor, shape (n_gen,).

        ``linear``    : ``1 + theta_avg * k_avg + theta_diff * k_diff``
        ``lognormal`` : ``exp(theta_avg * k_avg + theta_diff * k_diff)``

        LINEAR is the default on purpose (see the module docstring): the point of
        the envelope is a direction that is exactly odd in its own nuisance.
        ``lognormal`` is rabbit's own card convention and is offered so the two
        can be compared rather than argued about -- they differ by k^2/2, which
        on this card's largest envelope bin (|k| = 0.074) is 0.3 % of the yield
        at |theta| = 1, i.e. NOT negligible at the edge of the range even though
        it is well below the response itself.
        """
        t = tf.reshape(tf.cast(theta, self.dtype), [1, 2])
        e = tf.reshape(tf.matmul(t, self._k), [-1])
        if self.spec.response == "lognormal":
            return tf.exp(e)
        return tf.ones([], dtype=self.dtype) + e

    def factor_np(self, theta):
        t = np.asarray(theta, dtype=np.float64).reshape(2)
        e = t[0] * self.k_avg + t[1] * self.k_diff
        return np.exp(e) if self.spec.response == "lognormal" else 1.0 + e

    # -- provenance ----------------------------------------------------------

    def describe(self):
        n_on = int(self.qt_mask.sum())
        cut = (
            "everywhere"
            if self.spec.qt_min is None
            else (
                f"qT >= {self.spec.qt_min:g} GeV ({n_on}/{self.qt_mask.size} gen bins)"
            )
        )
        return (
            f"{self.spec.name}: {len(self.spec.legs)}-point envelope, {cut}, "
            f"{self.spec.response} response, "
            f"max|k_avg| = {np.abs(self.k_avg).max():.4f}, "
            f"max|k_diff| = {np.abs(self.k_diff).max():.4f}"
        )

    def to_dict(self):
        """Everything a diagnostic plot or a closure test needs."""
        return dict(
            name=self.spec.name,
            param_names=np.array(self.param_names),
            leg_labels=np.array(self.spec.leg_labels),
            kappas=np.array(
                [
                    [leg.get(KAPPA_R, np.nan), leg.get(KAPPA_F, np.nan)]
                    for leg in self.spec.legs
                ]
            ),
            gen_shape=np.array(self.gen_shape),
            gen_axis_names=np.array([n for n, _ in self.gen_axes]),
            sigma_nom=self.sigma_nom,
            log_ratio=self.log_ratio,
            up_leg=self.up_leg,
            down_leg=self.down_leg,
            k_avg=self.k_avg,
            k_diff=self.k_diff,
            k_avg_raw=self.k_avg_raw,
            k_diff_raw=self.k_diff_raw,
            qt_mask=self.qt_mask,
            qt_min=np.array(np.nan if self.spec.qt_min is None else self.spec.qt_min),
            diff_fact=np.array(self.spec.diff_fact),
            response=np.array(self.spec.response),
            **{f"edges_{i}": e for i, (_, e) in enumerate(self.gen_axes)},
        )
