"""Probability that the track of a muon has no valid pixel hit, as a function of its
generated pt and of qopr = (q/p)_reco / (q/p)_gen, for the response-based pixel
multiplicity weights (wremnants.production.muon_calibration.make_pixel_multiplicity_helpers).

Muons without a valid pixel hit differ from the others by their resolution only, so
the probability follows from the response of the two classes, which the dilepton
histmaker books for the Z -> mumu MC (hPixelResponseTrig, hPixelResponseNonTrig):

    f0(genPt, qopr) = pi R0 / ((1 - pi) R1 + pi R0)

with R1 (R0) the response of the muons with (without) a valid pixel hit and pi the
fraction of muons without one. Per eta bin the responses are described by

  - a standardized shape of R1, z = (qopr - 1 - m1) / sigma1(genPt), common to all
    generated pt, with sigma1^2 = a1^2 + b1^2 genPt^2;
  - R0 = R1 convolved with a Gaussian of mean mu and width delta(genPt),
    delta^2 = a^2 + b^2 genPt^2.

They are fitted (binned Poisson likelihood) in the generated-pt bins whose response
the pt cuts of the histmaker do not truncate. The probability itself needs no such
restriction: at given generated pt and qopr a cut on the reconstructed pt removes both
classes alike. pi is set per trigger category, eta and charge so that the probability
averages to the fraction of muons without a valid pixel hit in the histograms.
"""

import hist
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq, least_squares, nnls

from wums import logging

logger = logging.child_logger(__name__)

# probabilists' Gauss-Hermite nodes for the convolution with a Gaussian
_GH_NODES, _GH_WEIGHTS = np.polynomial.hermite_e.hermegauss(61)
_GH_WEIGHTS = _GH_WEIGHTS / _GH_WEIGHTS.sum()

# grid of the standardized response, dense in the core
_Z_GRID = 120.0 * np.sinh(np.linspace(-5.0, 5.0, 6001)) / np.sinh(5.0)


def _cdf_at_edges(counts):
    """Cumulative distribution at the bin edges, of the non-negative part of `counts`"""
    c = np.clip(counts, 0.0, None)
    return np.concatenate([[0.0], np.cumsum(c)]) / c.sum()


def _location_width(counts, edges):
    """Median and half of the central 68.27% interval of a binned distribution"""
    cdf = _cdf_at_edges(counts)
    lo, med, hi = np.interp([0.158655, 0.5, 0.841345], cdf, edges)
    return med, 0.5 * (hi - lo)


def _poisson_deviance_residuals(n, e, floor=1e-2):
    """Signed square roots of the Poisson deviance terms of the observed numbers of
    events n against the expected e: their sum of squares is -2 log of the likelihood
    ratio, which, unlike a chi2, is valid for the sparse bins in the tails of the
    response. Expectations below `floor` events are set to it, so that single muons far
    in the tails, which the model does not describe, cannot dominate a fit.
    """
    e = np.clip(e, floor, None)
    n = np.clip(n, 0.0, None)
    dev = 2.0 * (e - n + n * np.log(np.where(n > 0.0, n, e) / e))
    return np.sign(n - e) * np.sqrt(np.clip(dev, 0.0, None))


def _fit_quadrature_sum(pt, var, err):
    """var = a^2 + b^2 pt^2 with a^2, b^2 >= 0, weighted least squares; returns (a^2, b^2)"""
    design = np.stack([np.ones_like(pt), pt**2], axis=1) / err[:, None]
    coeff, _ = nnls(design, var / err)
    return coeff


class ResponseModel:
    """Response of the muons with (R1) and without (R0) a valid pixel hit in one eta bin.

    counts: [genPt, qopr, class] with class 0 = no valid pixel hit, 1 = at least one;
    genpt_edges, qopr_edges: bin edges; fit_bins: generated-pt bins to fit, with the
    qopr range that the pt cuts leave untruncated in each of them.
    """

    def __init__(self, counts, variances, genpt_edges, qopr_edges, fit_bins):
        self.qopr_edges = qopr_edges
        self.fit_pt = np.array(
            [0.5 * (genpt_edges[g] + genpt_edges[g + 1]) for g, _ in fit_bins]
        )
        centers = 0.5 * (qopr_edges[1:] + qopr_edges[:-1])

        # --- muons with a valid pixel hit: location, width and standardized shape
        loc, width, nev, cdfs = [], [], [], []
        for g, _ in fit_bins:
            c1 = counts[g, :, 1]
            m, s = _location_width(c1, qopr_edges)
            loc.append(m - 1.0)
            width.append(s)
            # effective number of events, for the weights of the fits below
            nev.append(c1.sum() ** 2 / max(variances[g, :, 1].sum(), 1e-30))
            # monotone cubic interpolation of the cumulative distribution: a smooth density
            cdf = PchipInterpolator(qopr_edges, _cdf_at_edges(c1), extrapolate=False)(
                _Z_GRID * s + m
            )
            x = _Z_GRID * s + m
            cdfs.append(
                np.where(x < qopr_edges[0], 0.0, np.where(x > qopr_edges[-1], 1.0, cdf))
            )
        loc, width, nev = np.array(loc), np.array(width), np.array(nev)
        self.m1 = np.average(loc, weights=nev)
        self.cdf = np.average(np.array(cdfs), axis=0, weights=nev)
        var_err = 2.0 * width**2 * 1.25 / np.sqrt(2.0 * nev)
        self.sigma1_coeff = _fit_quadrature_sum(self.fit_pt, width**2, var_err)
        self.sigma1_bins = width

        # --- muons without a valid pixel hit: Gaussian convolution of the above
        mu, delta, norm, n0_eff, chi2, ndf = [], [], [], [], [], []
        for i, (g, (xlo, xhi)) in enumerate(fit_bins):
            sel = (centers > xlo) & (centers < xhi)
            c0 = counts[g, :, 0]
            # effective weight of an event, to turn sums of weights into numbers of events
            wgt = variances[g, :, 0].sum() / max(c0.sum(), 1e-30)
            n0 = c0[sel] / wgt

            def expected(par, pt=self.fit_pt[i], sel=sel, wgt=wgt):
                return (
                    np.exp(par[0])
                    * self._prob(pt, 0, mu=par[1], delta=np.exp(par[2]))[sel]
                    / wgt
                )

            def residuals(par, n0=n0, expected=expected):
                return _poisson_deviance_residuals(n0, expected(par))

            start = [np.log(max(c0[sel].sum(), 1e-9)), 0.0, np.log(max(width[i], 1e-4))]
            best = min(
                (
                    least_squares(residuals, [start[0], 0.0, start[2] + d])
                    for d in (-1.0, 0.0, 1.0)
                ),
                key=lambda r: r.cost,
            )
            norm.append(np.exp(best.x[0]))
            mu.append(best.x[1])
            delta.append(np.exp(best.x[2]))
            n0_eff.append(max(np.exp(best.x[0]) / wgt, 1.0))
            # goodness of the fit from the bins with enough expected events for a chi2
            e = expected(best.x)
            ok = e >= 5.0
            chi2.append(np.sum((n0[ok] - e[ok]) ** 2 / e[ok]))
            ndf.append(max(ok.sum() - 3, 0))
        mu, delta, norm, n0_eff = (np.array(x) for x in (mu, delta, norm, n0_eff))
        self.mu = np.average(mu, weights=n0_eff)
        # uncertainty of delta^2 = (variance without) - (variance with a valid pixel hit),
        # from the first: it does not vanish with delta
        var_err = 2.0 * (width**2 + delta**2) * 1.25 / np.sqrt(2.0 * n0_eff)
        self.delta_coeff = _fit_quadrature_sum(self.fit_pt, delta**2, var_err)
        self.delta_bins = delta
        self.mu_bins = mu
        self.chi2 = np.array(chi2)
        self.ndf = np.array(ndf)
        # fraction of muons without a valid pixel hit from the fitted normalisation
        self.frac0_bins = norm / (
            norm + np.array([counts[g, :, 1].sum() for g, _ in fit_bins])
        )

    def sigma1(self, pt):
        return np.sqrt(self.sigma1_coeff[0] + self.sigma1_coeff[1] * pt**2)

    def delta(self, pt):
        return np.sqrt(self.delta_coeff[0] + self.delta_coeff[1] * pt**2)

    def _prob(self, pt, cls, mu=None, delta=None, edges=None):
        """Probabilities of the qopr bins for a muon of generated pt `pt` and class `cls`"""
        edges = self.qopr_edges if edges is None else edges
        z = (edges - 1.0 - self.m1) / self.sigma1(pt)
        if cls == 1:
            return np.diff(np.interp(z, _Z_GRID, self.cdf))
        mu = self.mu if mu is None else mu
        delta = self.delta(pt) if delta is None else delta
        shift = (mu + delta * _GH_NODES) / self.sigma1(pt)
        cdf = np.interp(z[:, None] - shift[None, :], _Z_GRID, self.cdf) @ _GH_WEIGHTS
        return np.diff(cdf)

    def probability(self, pt, frac0, edges=None):
        """Probability of no valid pixel hit in the qopr bins, for the fraction `frac0`
        of muons without one"""
        p1 = self._prob(pt, 1, edges=edges)
        p0 = self._prob(pt, 0, edges=edges)
        den = (1.0 - frac0) * p1 + frac0 * p0
        return np.where(den > 0.0, frac0 * p0 / np.where(den > 0.0, den, 1.0), 1.0)


def untruncated_qopr_range(genpt_lo, genpt_hi, pt_min, pt_max):
    """qopr range in which no muon of a generated-pt bin is cut by pt_min < pt < pt_max
    (qopr ~ genPt / pt)"""
    return genpt_hi / pt_max, genpt_lo / pt_min


def make_zero_pixel_probability(
    hresponse,
    pt_range,
    genpt_fit_range=(34.0, 50.0),
    genpt_map_edges=np.linspace(20.0, 70.0, 26),
    qopr_map_core=(0.9, 1.1),
    qopr_map_merge=4,
):
    """Probability map from the response histograms.

    hresponse: hist with the axes (nontrig_trig, eta, genPt, charge, qopr, nvalidpixel),
    nvalidpixel with the two bins "no valid pixel hit" and "at least one".
    pt_range: (min, max) of the reconstructed pt of the muons in the histograms.
    genpt_fit_range: generated-pt bins inside this range are used to fit the response.
    genpt_map_edges: generated-pt bins of the map; its qopr bins are those of the
    histogram inside qopr_map_core and qopr_map_merge of them merged outside.
    Returns the map, a hist with the axes (nontrig_trig, eta, charge, genPt, qopr)
    without flow bins, and a dictionary with the parameters and checks of the model.
    """
    names = [ax.name for ax in hresponse.axes]
    if names != ["nontrig_trig", "eta", "genPt", "charge", "qopr", "nvalidpixel"]:
        raise ValueError(f"Unexpected axes of the response histogram: {names}")
    ax_trig, ax_eta, ax_genpt, ax_charge, ax_qopr, _ = hresponse.axes
    genpt_edges = np.asarray(ax_genpt.edges)
    qopr_edges = np.asarray(ax_qopr.edges)
    # [trig, eta, genPt, charge, qopr, class], without flow bins
    vals = hresponse.values()
    variances = hresponse.variances()
    if variances is None:
        variances = np.abs(vals)

    fit_bins = []
    for g in range(len(genpt_edges) - 1):
        if (
            genpt_edges[g] >= genpt_fit_range[0]
            and genpt_edges[g + 1] <= genpt_fit_range[1]
        ):
            xlo, xhi = untruncated_qopr_range(
                genpt_edges[g], genpt_edges[g + 1], *pt_range
            )
            fit_bins.append((g, (max(xlo, qopr_edges[0]), min(xhi, qopr_edges[-1]))))
    if len(fit_bins) < 2:
        raise ValueError(f"Fewer than two generated-pt bins within {genpt_fit_range}")
    logger.info(
        "Response fitted in the generated-pt bins "
        + ", ".join(
            f"[{genpt_edges[g]:g}, {genpt_edges[g + 1]:g}] (qopr {lo:.2f}-{hi:.2f})"
            for g, (lo, hi) in fit_bins
        )
    )

    # qopr binning of the map: the histogram bins, merged outside the core
    core_lo, core_hi = qopr_map_core
    merge = qopr_map_merge
    idx = np.arange(len(qopr_edges))
    in_core = (qopr_edges > core_lo - 1e-9) & (qopr_edges < core_hi + 1e-9)
    keep = in_core | (((idx - idx[in_core][0]) % merge) == 0)
    keep[0] = keep[-1] = True
    map_edges = qopr_edges[keep]
    owner = np.searchsorted(map_edges, 0.5 * (qopr_edges[1:] + qopr_edges[:-1])) - 1

    axis_genpt_map = hist.axis.Variable(
        genpt_map_edges, name="genPt", underflow=False, overflow=False
    )
    axis_qopr_map = hist.axis.Variable(
        map_edges, name="qopr", underflow=False, overflow=False
    )
    axes_out = []
    for ax in (ax_trig, ax_eta, ax_charge):
        axes_out.append(
            hist.axis.Variable(ax.edges, name=ax.name, underflow=False, overflow=False)
        )
    hprob = hist.Hist(
        *axes_out[:1],
        axes_out[1],
        axes_out[2],
        axis_genpt_map,
        axis_qopr_map,
        name="hZeroPixelProb",
    )

    genpt_centers = 0.5 * (genpt_edges[1:] + genpt_edges[:-1])
    map_centers = 0.5 * (genpt_map_edges[1:] + genpt_map_edges[:-1])
    neta = ax_eta.size
    info = {
        k: np.zeros(neta)
        for k in (
            "m1",
            "sigma1_a2",
            "sigma1_b2",
            "mu",
            "delta_a2",
            "delta_b2",
            "chi2_fit",
            "ndf_fit",
            "chi2_all",
            "ndf_all",
        )
    }
    info["sigma1_bins"] = np.zeros((neta, len(fit_bins)))
    info["delta_bins"] = np.zeros((neta, len(fit_bins)))
    info["frac0_bins"] = np.zeros((neta, len(fit_bins)))
    info["frac0"] = np.zeros((ax_trig.size, neta, ax_charge.size))
    info["frac0_observed"] = np.zeros((ax_trig.size, neta, ax_charge.size))
    info["fit_genpt"] = np.array([genpt_centers[g] for g, _ in fit_bins])
    info["eta_edges"] = np.asarray(ax_eta.edges)

    for ie in range(neta):
        # shapes from both trigger categories and charges
        counts = vals[:, ie].sum(axis=(0, 2))  # [genPt, qopr, class]
        var = variances[:, ie].sum(axis=(0, 2))
        model = ResponseModel(counts, var, genpt_edges, qopr_edges, fit_bins)
        info["m1"][ie] = model.m1
        info["sigma1_a2"][ie], info["sigma1_b2"][ie] = model.sigma1_coeff
        info["mu"][ie] = model.mu
        info["delta_a2"][ie], info["delta_b2"][ie] = model.delta_coeff
        info["chi2_fit"][ie], info["ndf_fit"][ie] = model.chi2.sum(), model.ndf.sum()
        info["sigma1_bins"][ie] = model.sigma1_bins
        info["delta_bins"][ie] = model.delta_bins
        info["frac0_bins"][ie] = model.frac0_bins

        # probabilities of the two classes in the histogram's generated-pt bins (for the
        # normalisation and the check) and in those of the map
        p1_hist = np.array(
            [
                np.bincount(owner, model._prob(pt, 1), minlength=len(map_edges) - 1)
                for pt in genpt_centers
            ]
        )
        p0_hist = np.array(
            [
                np.bincount(owner, model._prob(pt, 0), minlength=len(map_edges) - 1)
                for pt in genpt_centers
            ]
        )
        p1_map = np.array([model._prob(pt, 1, edges=map_edges) for pt in map_centers])
        p0_map = np.array([model._prob(pt, 0, edges=map_edges) for pt in map_centers])

        def prob(p0, p1, frac0):
            den = (1.0 - frac0) * p1 + frac0 * p0
            return np.where(den > 0.0, frac0 * p0 / np.where(den > 0.0, den, 1.0), 1.0)

        chi2 = ndf = 0.0
        for it in range(ax_trig.size):
            for iq in range(ax_charge.size):
                c = vals[it, ie, :, iq]  # [genPt, qopr, class]
                # merged to the qopr bins of the map
                n_all = np.array(
                    [
                        np.bincount(owner, row, minlength=len(map_edges) - 1)
                        for row in c.sum(axis=-1)
                    ]
                )
                n_zero = np.array(
                    [
                        np.bincount(owner, row, minlength=len(map_edges) - 1)
                        for row in c[..., 0]
                    ]
                )
                v_all = np.array(
                    [
                        np.bincount(owner, row, minlength=len(map_edges) - 1)
                        for row in variances[it, ie, :, iq].sum(axis=-1)
                    ]
                )
                n_all = np.clip(n_all, 0.0, None)
                target = n_zero.sum() / n_all.sum()
                if target <= 1e-6:
                    frac0 = 1e-6
                else:
                    frac0 = brentq(
                        lambda f: np.sum(n_all * prob(p0_hist, p1_hist, f))
                        / n_all.sum()
                        - target,
                        1e-6,
                        0.9,
                    )
                info["frac0"][it, ie, iq] = frac0
                info["frac0_observed"][it, ie, iq] = target
                hprob.values()[it, ie, iq] = prob(p0_map, p1_map, frac0)
                # check: observed against expected numbers of muons without a valid pixel hit
                f = prob(p0_hist, p1_hist, frac0)
                expected_var = f * (1.0 - f) * v_all
                ok = expected_var > 25.0 * (v_all.sum() / max(n_all.sum(), 1e-30)) ** 2
                chi2 += np.sum((n_zero[ok] - n_all[ok] * f[ok]) ** 2 / expected_var[ok])
                ndf += ok.sum()
        info["chi2_all"][ie], info["ndf_all"][ie] = chi2, ndf

    return hprob, info
