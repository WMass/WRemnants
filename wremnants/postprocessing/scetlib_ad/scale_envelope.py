"""Frozen gen-level scale-envelope nuisances for the scetlib_ad param model.

The mu_R/mu_F missing-higher-order uncertainty is not profiled through kappa_R and
kappa_F (a profiled kappa_F responds as ln^2 kappa_F, i.e. evenly, and a symmetric
prior then has two mirror minima). As in AN-25-085 it is an envelope of scale
variations, symmetrised quadratically and applied as a fixed gen-level shape.
Built once, at the anchor:

1. evaluate sigma_gen at each (kappa_R, kappa_F) leg of the point set:
   ``points=7`` is the AN's 7-point set :data:`SEVEN_POINT`; ``points=3`` (the
   default) is mu_R alone, :data:`THREE_POINT_MUR`, from which the production
   datacard's ``resumFOScaleZ`` nuisance is built;
2. per gen bin, take the largest up and down log-deviation from the nominal
   (the nominal leg is included, so the envelope always brackets it);
3. symmetrise into two shapes with rabbit's ``symmetrize="quadratic"``::

       logkup   =  max_legs ln(sigma_leg / sigma_nom)      >= 0
       logkdown = -min_legs ln(sigma_leg / sigma_nom)      >= 0
       k_avg    = 0.5 * (logkup + logkdown)
       k_diff   = 0.5 * sqrt(3) * (logkup - logkdown)

4. zero every gen bin whose lower qT edge is below ``qt_min`` (default 20 GeV,
   as the production pt20 envelopes; ``None`` disables the cut).

The two nuisances ``<name>SymAvg`` / ``<name>SymDiff`` then act as

    linear (default): sigma_gen -> sigma_gen * (1 + theta_avg k_avg + theta_diff k_diff)
    lognormal:        sigma_gen -> sigma_gen * exp(theta_avg k_avg + theta_diff k_diff)

and are folded through the response like any gen-level shape. Linear makes the
direction exactly odd in theta; lognormal is rabbit's card convention; they
differ at O(k^2). The envelope must stay frozen: a per-bin max over legs is not
differentiable.
"""

import math

import numpy as np
import tensorflow as tf

SQRT3 = math.sqrt(3.0)

# (kappa_R, kappa_F): each scale x2 and /2 with 0.5 <= muR/muF <= 2. The nominal
# is first and is the anchor itself; it is not re-evaluated.
SEVEN_POINT = (
    (1.0, 1.0),
    (2.0, 1.0),
    (0.5, 1.0),
    (1.0, 2.0),
    (1.0, 0.5),
    (2.0, 2.0),
    (0.5, 0.5),
)

# Rabbit-facing names of the parameters the legs move. Each scales its own scale
# at fixed other, so the production template ``kappaFO2.-kappaf0.5`` is (2, 1).
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


# mu_R alone: the points the production datacard's resumFOScaleZ uses.
THREE_POINT_MUR = ((1.0, 1.0), (0.5, 1.0), (2.0, 1.0))

POINT_SETS = {7: SEVEN_POINT, 3: THREE_POINT_MUR}


def scale_spec(points=3, qt_min=20.0, name="resumFOScaleEnv", response="linear"):
    """Envelope spec over a named (mu_R, mu_F) point set.

    ``points=3`` (default) is mu_R alone, as in the production datacard;
    ``points=7`` is the AN's (mu_R, mu_F) envelope.
    """
    if points not in POINT_SETS:
        raise ValueError(
            f"scale_spec: points must be one of {sorted(POINT_SETS)}, got {points!r}"
        )
    legs = [{KAPPA_R: kr, KAPPA_F: kf} for kr, kf in POINT_SETS[points]]
    return EnvelopeSpec(name, legs, qt_min=qt_min, response=response)


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

        # ln(sigma_leg / sigma_nom) per leg; the nominal row stays 0, so the
        # envelope always brackets the nominal.
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

        # rabbit's symmetrize="quadratic".
        k_avg = 0.5 * (logkup + logkdown)
        k_diff = 0.5 * spec.diff_fact * (logkup - logkdown)

        self.qt_mask = self._qt_mask()
        self.k_avg_raw, self.k_diff_raw = k_avg.copy(), k_diff.copy()
        self.k_avg = np.where(self.qt_mask, k_avg, 0.0)
        self.k_diff = np.where(self.qt_mask, k_diff, 0.0)

        self._k = tf.constant(np.stack([self.k_avg, self.k_diff], axis=0), dtype=dtype)
        self.dtype = dtype

    def _qt_mask(self):
        """True where the envelope applies: gen bins with lower qT edge >= qt_min.

        A bin straddling the threshold is zeroed.
        """
        n = int(np.prod(self.gen_shape))
        if self.spec.qt_min is None:
            return np.ones(n, dtype=bool)
        edges = self.gen_axes[self.spec.qt_axis_index][1]
        lo = edges[:-1] >= self.spec.qt_min
        # Broadcast over the other gen axes, C order over self.gen_shape.
        shape = [1] * len(self.gen_shape)
        shape[self.spec.qt_axis_index] = -1
        return np.broadcast_to(lo.reshape(shape), self.gen_shape).reshape(n).copy()

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
        """
        t = tf.reshape(tf.cast(theta, self.dtype), [1, 2])
        e = tf.reshape(tf.matmul(t, self._k), [-1])
        if self.spec.response == "lognormal":
            return tf.exp(e)
        return tf.ones([], dtype=self.dtype) + e

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
