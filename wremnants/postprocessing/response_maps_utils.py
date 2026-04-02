"""Helpers for building response-map TFLite models from qopr histograms.

Workflow:
1. Load and merge response histograms from one or more processes.
2. Project them to (genCharge, qopr, genPt, genEta).
3. Build fixed CDF interpolation points and convert histograms to quantiles.
4. Interpolate quantiles in (genPt, genEta) and derive CDF/PDF/dweight helpers.
5. Run optional debug and monotonicity diagnostics.
6. Export the TFLite model.
"""

import os

import h5py
import numpy as np
import scipy
import tensorflow as tf
import tensorflow_probability as tfp

import wums.fitutils
import wums.ioutils
import wums.tfutils
from wums import logging

logger = logging.child_logger(__name__)

DEFAULT_PROCS = {
    "kaon": ["BuToJpsiK_2018"],
    "muon": [
        "Zmumu_2016PostVFP",
        "Ztautau_2016PostVFP",
        "Wplusmunu_2016PostVFP",
        "Wminusmunu_2016PostVFP",
        "Wplustaunu_2016PostVFP",
        "Wminustaunu_2016PostVFP",
    ],
}

DEFAULT_OUTPUT_NAMES = {
    "kaon": "kaon_response.tflite",
    "muon": "muon_response.tflite",
}


def resolve_procs(args):
    return args.procs or list(DEFAULT_PROCS[args.particleType])


def default_output_name(particle_type):
    return DEFAULT_OUTPUT_NAMES[particle_type]


def output_name_with_postfix(particle_type, postfix):
    output_name = default_output_name(particle_type)
    if not postfix:
        return output_name

    stem, suffix = os.path.splitext(output_name)
    return f"{stem}_{postfix}{suffix}"


def load_merged_response_histograms(input_file, procs):
    hist_response = None
    hist_response_scaled = None
    hist_response_smeared = None

    with h5py.File(input_file, "r") as h5file:
        for proc in procs:
            logger.info("Loading response histograms for proc %s", proc)
            results = wums.ioutils.pickle_load_h5py(h5file[proc])
            hist_response_proc = results["output"]["hist_qopr"].get()
            hist_response_scaled_proc = results["output"]["hist_qopr_shifted"].get()
            hist_response_smeared_proc = results["output"][
                "hist_qopr_smearedmulti"
            ].get()

            if hist_response is None:
                hist_response = hist_response_proc
                hist_response_scaled = hist_response_scaled_proc
                hist_response_smeared = hist_response_smeared_proc
            else:
                hist_response += hist_response_proc
                hist_response_scaled += hist_response_scaled_proc
                hist_response_smeared += hist_response_smeared_proc

    if hist_response is None:
        raise ValueError(f"No response histograms were loaded from {input_file}")

    return hist_response, hist_response_scaled, hist_response_smeared


def project_response_histograms(
    hist_response, hist_response_scaled, hist_response_smeared
):
    axes = ("genCharge", "qopr", "genPt", "genEta")
    return (
        hist_response.project(*axes),
        hist_response_scaled.project(*axes),
        hist_response_smeared.project(*axes),
    )


def make_interp_cdfvals(interp_sigma_min, interp_sigma_max, interp_sigma_steps):
    interp_sigmas = np.linspace(interp_sigma_min, interp_sigma_max, interp_sigma_steps)
    interp_cdfvals = scipy.stats.norm.cdf(interp_sigmas)
    return np.concatenate([[0.0], interp_cdfvals, [1.0]])


def make_quant_cdfvals(interp_cdfvals):
    quant_cdfvals = tf.constant(interp_cdfvals, tf.float64)
    return quant_cdfvals[None, :, None, None]


def histograms_to_quantiles(
    hist_response, hist_response_scaled, hist_response_smeared, quant_cdfvals
):
    quants, _ = wums.fitutils.hist_to_quantiles(hist_response, quant_cdfvals, axis=1)
    quants_scaled, _ = wums.fitutils.hist_to_quantiles(
        hist_response_scaled, quant_cdfvals, axis=1
    )
    quants_smeared, _ = wums.fitutils.hist_to_quantiles(
        hist_response_smeared, quant_cdfvals, axis=1
    )
    return quants, quants_scaled, quants_smeared


def make_grid_points(hist_response):
    return tuple(tf.constant(axis.centers) for axis in hist_response.axes[2:])


def make_axis_bounds(hist_response):
    qopr_edges_flat = np.reshape(hist_response.axes[1].edges, [-1])
    pt_edges_flat = np.reshape(hist_response.axes[2].edges, [-1])
    eta_edges_flat = np.reshape(hist_response.axes[3].edges, [-1])
    return {
        "qopr_low": tf.constant(qopr_edges_flat[0], tf.float64),
        "qopr_high": tf.constant(qopr_edges_flat[-1], tf.float64),
        "pt_low": tf.constant(pt_edges_flat[0], tf.float64),
        "pt_high": tf.constant(pt_edges_flat[-1], tf.float64),
        "eta_low": tf.constant(eta_edges_flat[0], tf.float64),
        "eta_high": tf.constant(eta_edges_flat[-1], tf.float64),
    }


def make_dsigmasq(hist_response_smeared):
    dsigma = hist_response_smeared.metadata["sigmarel"]
    return tf.constant(dsigma**2, tf.float64)


class ResponseMapInterpolator:
    def __init__(
        self,
        quants,
        quants_scaled,
        quants_smeared,
        quant_cdfvals,
        grid_points,
        bounds,
        dsigmasq,
    ):
        self.quants = tf.constant(quants, tf.float64)
        self.quants_scaled = tf.constant(quants_scaled, tf.float64)
        self.quants_smeared = tf.constant(quants_smeared, tf.float64)
        self.quant_cdfvals = quant_cdfvals
        self.grid_points = grid_points
        self.qopr_low = bounds["qopr_low"]
        self.qopr_high = bounds["qopr_high"]
        self.pt_low = bounds["pt_low"]
        self.pt_high = bounds["pt_high"]
        self.eta_low = bounds["eta_low"]
        self.eta_high = bounds["eta_high"]
        self.dsigmasq = dsigmasq

    def _interp_quantiles(self, quants_sel, gen_pt, gen_eta, gen_charge):
        charge_idx = tf.where(gen_charge > 0.0, 1, 0)
        quants_charge = quants_sel[charge_idx]

        x = tf.stack([gen_pt, gen_eta], axis=0)
        x = x[None, :]
        quants_interp = tfp.math.batch_interp_rectilinear_nd_grid(
            x, x_grid_points=self.grid_points, y_ref=quants_charge, axis=1
        )

        return tf.reshape(quants_interp, [-1])

    def interp_cdf(self, quants_sel, gen_pt, gen_eta, gen_charge, qopr):
        quants_interp = self._interp_quantiles(quants_sel, gen_pt, gen_eta, gen_charge)
        quant_cdfvals_interp = tf.reshape(self.quant_cdfvals, [-1])

        qopr = tf.clip_by_value(qopr, self.qopr_low, self.qopr_high)
        qopr = tf.reshape(qopr, [-1])

        cdf = wums.fitutils.cubic_spline_interpolate(
            xi=quants_interp[..., None],
            yi=quant_cdfvals_interp[..., None],
            x=qopr[..., None],
            axis=0,
        )
        return cdf[..., 0]

    def interp_pdf(self, quants_sel, gen_pt, gen_eta, gen_charge, qopr):
        with tf.GradientTape() as tape:
            tape.watch(qopr)
            cdf = self.interp_cdf(quants_sel, gen_pt, gen_eta, gen_charge, qopr)
        pdf = tape.gradient(cdf, qopr)
        return cdf, pdf

    def interp_dpdf(self, quants_sel, gen_pt, gen_eta, gen_charge, qopr):
        with tf.GradientTape() as outer_tape:
            outer_tape.watch(qopr)
            with tf.GradientTape() as inner_tape:
                inner_tape.watch(qopr)
                cdf = self.interp_cdf(quants_sel, gen_pt, gen_eta, gen_charge, qopr)
            pdf = inner_tape.gradient(cdf, qopr)
        dpdf = outer_tape.gradient(pdf, qopr)
        return cdf, pdf, dpdf

    def interp_dweight(self, gen_pt, gen_eta, gen_charge, qopr):
        _, pdf, dpdf = self.interp_dpdf(self.quants, gen_pt, gen_eta, gen_charge, qopr)
        _, pdf_smeared = self.interp_pdf(
            self.quants_smeared, gen_pt, gen_eta, gen_charge, qopr
        )

        dweightdscale = -dpdf / pdf
        dweightdsigmasq = (pdf_smeared - pdf) / pdf / self.dsigmasq

        in_range = (
            (qopr > self.qopr_low)
            & (qopr < self.qopr_high)
            & (gen_pt > self.pt_low)
            & (gen_pt < self.pt_high)
            & (gen_eta > self.eta_low)
            & (gen_eta < self.eta_high)
        )

        dweightdscale = tf.where(in_range, dweightdscale, tf.zeros_like(dweightdscale))
        dweightdsigmasq = tf.where(
            in_range, dweightdsigmasq, tf.zeros_like(dweightdsigmasq)
        )

        dweightdscale = tf.where(
            tf.math.is_finite(dweightdscale),
            dweightdscale,
            tf.zeros_like(dweightdscale),
        )
        dweightdsigmasq = tf.where(
            tf.math.is_finite(dweightdsigmasq),
            dweightdsigmasq,
            tf.zeros_like(dweightdsigmasq),
        )

        return dweightdscale, dweightdsigmasq


def build_response_map_interpolator(
    hist_response, hist_response_scaled, hist_response_smeared, interp_cdfvals
):
    quant_cdfvals = make_quant_cdfvals(interp_cdfvals)
    quants, quants_scaled, quants_smeared = histograms_to_quantiles(
        hist_response, hist_response_scaled, hist_response_smeared, quant_cdfvals
    )
    grid_points = make_grid_points(hist_response)
    bounds = make_axis_bounds(hist_response)
    dsigmasq = make_dsigmasq(hist_response_smeared)
    return ResponseMapInterpolator(
        quants=quants,
        quants_scaled=quants_scaled,
        quants_smeared=quants_smeared,
        quant_cdfvals=quant_cdfvals,
        grid_points=grid_points,
        bounds=bounds,
        dsigmasq=dsigmasq,
    )


def log_histogram_summary(
    hist_response, hist_response_scaled, hist_response_smeared, interp_cdfvals
):
    dscale = hist_response_scaled.metadata["scalerel"]
    dsigma = hist_response_smeared.metadata["sigmarel"]
    logger.info("Projected response histogram shape: %s", hist_response.shape)
    logger.info("Scale relative shift: %s", dscale)
    logger.info("Sigma relative shift: %s", dsigma)
    logger.info("Interpolation cdf values: %s", interp_cdfvals)


def log_quantile_diagnostics(response_map):
    quants = response_map.quants.numpy()
    quants_scaled = response_map.quants_scaled.numpy()
    dquants = np.sum((quants_scaled - quants) ** 2)
    non_finite = np.count_nonzero(~np.isfinite(quants))
    logger.info("Quantile delta sum: %s", dquants)
    logger.info("Non-finite quantiles: %s", non_finite)


def log_interpolator_bounds(response_map):
    logger.info(
        "qopr bounds: [%s, %s]",
        response_map.qopr_low.numpy(),
        response_map.qopr_high.numpy(),
    )
    logger.info(
        "pt bounds: [%s, %s]",
        response_map.pt_low.numpy(),
        response_map.pt_high.numpy(),
    )
    logger.info(
        "eta bounds: [%s, %s]",
        response_map.eta_low.numpy(),
        response_map.eta_high.numpy(),
    )


def run_debug_check(response_map, gen_pt, gen_eta, gen_charge, qopr):
    gen_pt = tf.constant(gen_pt, tf.float64)
    gen_eta = tf.constant(gen_eta, tf.float64)
    gen_charge = tf.constant(gen_charge, tf.float64)
    qopr = tf.constant(qopr, tf.float64)

    cdf = response_map.interp_cdf(
        response_map.quants, gen_pt, gen_eta, gen_charge, qopr
    )
    dweightdscale, dweightdsigmasq = response_map.interp_dweight(
        gen_pt, gen_eta, gen_charge, qopr
    )

    logger.info(
        "Debug check at genPt=%s genEta=%s genCharge=%s qopr=%s",
        gen_pt.numpy(),
        gen_eta.numpy(),
        gen_charge.numpy(),
        qopr.numpy(),
    )
    logger.info("Interpolated cdf: %s", cdf.numpy())
    logger.info("dweight/dscale: %s", dweightdscale.numpy())
    logger.info("dweight/dsigmasq: %s", dweightdsigmasq.numpy())


def make_debug_plots(
    response_map,
    plot_dir,
    gen_pt,
    gen_eta,
    gen_charge,
    n_qopr_points,
):
    import matplotlib.pyplot as plt

    os.makedirs(plot_dir, exist_ok=True)

    gen_pt_tensor = tf.constant(gen_pt, tf.float64)
    gen_eta_tensor = tf.constant(gen_eta, tf.float64)
    gen_charge_tensor = tf.constant(gen_charge, tf.float64)

    qopr_scan = np.linspace(
        response_map.qopr_low.numpy(),
        response_map.qopr_high.numpy(),
        n_qopr_points,
    )
    qopr_tensor = tf.constant(qopr_scan, tf.float64)

    cdf, pdf = response_map.interp_pdf(
        response_map.quants,
        gen_pt_tensor,
        gen_eta_tensor,
        gen_charge_tensor,
        qopr_tensor,
    )
    _, pdf_smeared = response_map.interp_pdf(
        response_map.quants_smeared,
        gen_pt_tensor,
        gen_eta_tensor,
        gen_charge_tensor,
        qopr_tensor,
    )
    dweightdscale, dweightdsigmasq = response_map.interp_dweight(
        gen_pt_tensor, gen_eta_tensor, gen_charge_tensor, qopr_tensor
    )

    quant_cdfvals = tf.reshape(response_map.quant_cdfvals, [-1]).numpy()
    quants_nominal = response_map._interp_quantiles(
        response_map.quants, gen_pt_tensor, gen_eta_tensor, gen_charge_tensor
    ).numpy()
    quants_scaled = response_map._interp_quantiles(
        response_map.quants_scaled, gen_pt_tensor, gen_eta_tensor, gen_charge_tensor
    ).numpy()
    quants_smeared = response_map._interp_quantiles(
        response_map.quants_smeared, gen_pt_tensor, gen_eta_tensor, gen_charge_tensor
    ).numpy()

    fig, ax = plt.subplots()
    ax.plot(quant_cdfvals, quants_nominal, label="nominal")
    ax.plot(quant_cdfvals, quants_scaled, label="scaled")
    ax.plot(quant_cdfvals, quants_smeared, label="smeared")
    ax.set_xlabel("cdf")
    ax.set_ylabel("qopr quantile")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "quantiles.png"))
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(qopr_scan, cdf.numpy(), label="cdf")
    ax.plot(qopr_scan, pdf.numpy(), label="pdf")
    ax.plot(qopr_scan, pdf_smeared.numpy(), label="pdf_smeared")
    ax.set_xlabel("qopr")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "cdf_pdf_scan.png"))
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(qopr_scan, dweightdscale.numpy(), label="dweight/dscale")
    ax.plot(qopr_scan, dweightdsigmasq.numpy(), label="dweight/dsigmasq")
    ax.set_xlabel("qopr")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "dweights_scan.png"))
    plt.close(fig)

    logger.info("Wrote debug plots to %s", plot_dir)


def _save_plot(fig, path):
    fig.tight_layout()
    fig.savefig(path)
    fig.clf()


def _plot_hist_with_overlay(
    hist_response_sel, xvals, yvals, path, xlim=None, yscale=None
):
    import matplotlib.pyplot as plt

    fig = plt.figure()
    hist_response_sel.plot()
    plt.plot(xvals, yvals)
    if xlim is not None:
        plt.xlim(xlim)
    if yscale is not None:
        plt.yscale(yscale)
    _save_plot(fig, path)
    plt.close(fig)


def _plot_quantiles(hist_response_sel, quantiles, cdf_values, path, xlim=None):
    import matplotlib.pyplot as plt

    fig = plt.figure()
    hist_response_sel.plot()
    plt.scatter(
        quantiles,
        np.zeros_like(quantiles),
        c=cdf_values,
        cmap="viridis",
        s=50,
        zorder=5,
        label="Quantiles",
    )
    plt.colorbar(label="CDF value")
    plt.legend()
    if xlim is not None:
        plt.xlim(xlim)
    _save_plot(fig, path)
    plt.close(fig)


def _plot_line(xvals, yvals, path, xlim=None):
    import matplotlib.pyplot as plt

    fig = plt.figure()
    plt.plot(xvals, yvals)
    if xlim is not None:
        plt.xlim(xlim)
    _save_plot(fig, path)
    plt.close(fig)


def make_bin_inputs(hist_response):
    pt_centers = np.reshape(hist_response.axes[2].centers, [-1])
    eta_centers = np.reshape(hist_response.axes[3].centers, [-1])

    for charge_idx in range(hist_response.axes[0].size):
        gen_charge = 1.0 if charge_idx == 1 else -1.0
        for pt_idx, gen_pt in enumerate(pt_centers):
            for eta_idx, gen_eta in enumerate(eta_centers):
                yield {
                    "charge_idx": charge_idx,
                    "pt_idx": pt_idx,
                    "eta_idx": eta_idx,
                    "gen_charge": gen_charge,
                    "gen_pt": gen_pt,
                    "gen_eta": gen_eta,
                }


def compute_quantile_checks(response_map, bin_inputs):
    quants = response_map.quants[
        bin_inputs["charge_idx"], :, bin_inputs["pt_idx"], bin_inputs["eta_idx"]
    ].numpy()
    quant_diffs = np.diff(quants)

    gen_pt_tensor = tf.constant(bin_inputs["gen_pt"], tf.float64)
    gen_eta_tensor = tf.constant(bin_inputs["gen_eta"], tf.float64)
    gen_charge_tensor = tf.constant(bin_inputs["gen_charge"], tf.float64)

    quants_interp = response_map._interp_quantiles(
        response_map.quants,
        gen_pt_tensor,
        gen_eta_tensor,
        gen_charge_tensor,
    ).numpy()
    quants_interp_diffs = np.diff(quants_interp)

    return {
        "quants": quants,
        "quants_interp": quants_interp,
        "quants_monotone": bool(np.all(quant_diffs > 0)) if quant_diffs.size else False,
        "quants_min_diff": float(np.min(quant_diffs)) if quant_diffs.size else np.nan,
        "quants_interp_monotone": (
            bool(np.all(quants_interp_diffs > 0)) if quants_interp_diffs.size else False
        ),
        "quants_interp_min_diff": (
            float(np.min(quants_interp_diffs)) if quants_interp_diffs.size else np.nan
        ),
        "gen_pt_tensor": gen_pt_tensor,
        "gen_eta_tensor": gen_eta_tensor,
        "gen_charge_tensor": gen_charge_tensor,
    }


def compute_cdf_pdf_checks(
    response_map,
    gen_pt_tensor,
    gen_eta_tensor,
    gen_charge_tensor,
    qopr_tensor,
    qoprvals,
):
    cdf_vals, pdf_vals, dpdf_vals = response_map.interp_dpdf(
        response_map.quants,
        gen_pt_tensor,
        gen_eta_tensor,
        gen_charge_tensor,
        qopr_tensor,
    )
    _, pdf_smeared_vals = response_map.interp_pdf(
        response_map.quants_smeared,
        gen_pt_tensor,
        gen_eta_tensor,
        gen_charge_tensor,
        qopr_tensor,
    )

    cdf_vals = cdf_vals.numpy()
    pdf_vals = pdf_vals.numpy()
    dpdf_vals = dpdf_vals.numpy()
    d2pdf_vals = ((pdf_smeared_vals - pdf_vals) / response_map.dsigmasq).numpy()
    cdf_diffs = np.diff(cdf_vals)

    with np.errstate(divide="ignore", invalid="ignore"):
        dweightdscale_vals = -dpdf_vals / pdf_vals
        dweightdscale_vals = np.where(
            np.isfinite(dweightdscale_vals), dweightdscale_vals, 0.0
        )

    return {
        "cdf_vals": cdf_vals,
        "pdf_vals": pdf_vals,
        "dpdf_vals": dpdf_vals,
        "d2pdf_vals": d2pdf_vals,
        "dweightdscale_vals": dweightdscale_vals,
        "cdf_monotone": bool(np.all(cdf_diffs >= 0)) if cdf_diffs.size else False,
        "cdf_min_diff": float(np.min(cdf_diffs)) if cdf_diffs.size else np.nan,
        "pdf_min": float(np.min(pdf_vals)),
        "pdf_has_neg": float(np.min(pdf_vals)) < 0,
        "pdf_integral": float(np.sum(pdf_vals) * (qoprvals[1] - qoprvals[0])),
    }


def compute_histogram_checks(hist_response, bin_inputs):
    hist_response_sel = hist_response[
        bin_inputs["charge_idx"], :, bin_inputs["pt_idx"], bin_inputs["eta_idx"]
    ]
    hist_vals = hist_response_sel.values()
    hist_sum = float(np.sum(hist_vals))
    hist_min = float(np.min(hist_vals))
    hist_nonzero_bins = int(np.sum(hist_vals > 0))
    hist_max_bin = float(np.max(hist_vals))
    hist_cdf = (
        np.cumsum(hist_vals) / hist_sum if hist_sum > 0 else np.zeros_like(hist_vals)
    )

    return {
        "hist_response_sel": hist_response_sel,
        "hist_sum": hist_sum,
        "hist_min": hist_min,
        "hist_nonzero_bins": hist_nonzero_bins,
        "hist_max_over_sum": hist_max_bin / hist_sum if hist_sum > 0 else 0.0,
        "flat_regions": int(np.sum(np.diff(hist_cdf) == 0)) if hist_cdf.size else 0,
    }


def classify_bin_issue(quantile_checks, cdf_pdf_checks):
    return (
        (not quantile_checks["quants_monotone"])
        or (not quantile_checks["quants_interp_monotone"])
        or (not cdf_pdf_checks["cdf_monotone"])
        or cdf_pdf_checks["pdf_has_neg"]
    )


def make_bin_diagnostics(
    response_map, hist_response, bin_inputs, qopr_tensor, qoprvals
):
    quantile_checks = compute_quantile_checks(response_map, bin_inputs)
    cdf_pdf_checks = compute_cdf_pdf_checks(
        response_map,
        quantile_checks["gen_pt_tensor"],
        quantile_checks["gen_eta_tensor"],
        quantile_checks["gen_charge_tensor"],
        qopr_tensor,
        qoprvals,
    )
    histogram_checks = compute_histogram_checks(hist_response, bin_inputs)

    return {
        "bin_inputs": bin_inputs,
        "has_issue": classify_bin_issue(quantile_checks, cdf_pdf_checks),
        **quantile_checks,
        **cdf_pdf_checks,
        **histogram_checks,
    }


def log_bin_diagnostics(result):
    log_method = logger.warning if result["has_issue"] else logger.debug
    log_method(
        "Monotonicity check chargeIdx=%s ptIdx=%s etaIdx=%s hasIssue=%s "
        "quantsMonotone=%s quantsInterpMonotone=%s cdfMonotone=%s "
        "pdfHasNeg=%s pdfIntegral=%s flatRegions=%s",
        result["bin_inputs"]["charge_idx"],
        result["bin_inputs"]["pt_idx"],
        result["bin_inputs"]["eta_idx"],
        result["has_issue"],
        result["quants_monotone"],
        result["quants_interp_monotone"],
        result["cdf_monotone"],
        result["pdf_has_neg"],
        result["pdf_integral"],
        result["flat_regions"],
    )


def plot_bin_diagnostics(result, plot_dir, qoprvals, quant_cdfvals, qopr_bin_width):
    name = (
        f"charge{result['bin_inputs']['charge_idx']}_"
        f"ptidx{result['bin_inputs']['pt_idx']}_"
        f"etaidx{result['bin_inputs']['eta_idx']}"
    )
    pdfvals_scaled = result["pdf_vals"] * result["hist_sum"] * qopr_bin_width

    _plot_hist_with_overlay(
        result["hist_response_sel"],
        qoprvals,
        pdfvals_scaled,
        os.path.join(plot_dir, f"{name}.png"),
        xlim=[0.9, 1.1],
    )
    _plot_hist_with_overlay(
        result["hist_response_sel"],
        qoprvals,
        pdfvals_scaled,
        os.path.join(plot_dir, f"{name}_log.png"),
        xlim=[0.9, 1.1],
        yscale="log",
    )
    _plot_quantiles(
        result["hist_response_sel"],
        result["quants_interp"],
        quant_cdfvals,
        os.path.join(plot_dir, f"{name}_quantiles.png"),
    )
    _plot_line(
        qoprvals,
        result["cdf_vals"],
        os.path.join(plot_dir, f"{name}_cdf.png"),
        xlim=[0.9, 1.1],
    )
    _plot_line(
        qoprvals,
        result["dpdf_vals"],
        os.path.join(plot_dir, f"{name}_d.png"),
        xlim=[0.9, 1.1],
    )
    _plot_line(
        qoprvals,
        result["d2pdf_vals"],
        os.path.join(plot_dir, f"{name}_d2.png"),
        xlim=[0.9, 1.1],
    )
    dweight_mask = np.abs(result["dweightdscale_vals"]) < 1e4
    _plot_line(
        qoprvals[dweight_mask],
        result["dweightdscale_vals"][dweight_mask],
        os.path.join(plot_dir, f"{name}_dweightdscale.png"),
        xlim=[0.9, 1.1],
    )


def run_monotonicity_checks(
    response_map,
    hist_response,
    plot_dir=None,
    plot_all_checks=False,
    plot_only_issues=False,
    qopr_min=0.0,
    qopr_max=2.0,
    n_qopr_points=1000,
):
    quant_cdfvals = tf.reshape(response_map.quant_cdfvals, [-1]).numpy()
    qoprvals = np.linspace(qopr_min, qopr_max, n_qopr_points)
    qopr_tensor = tf.constant(qoprvals, tf.float64)
    qopr_bin_width = hist_response.axes[1].centers[1] - hist_response.axes[1].centers[0]

    if plot_all_checks or plot_only_issues:
        if plot_dir is None:
            raise ValueError(
                "plot_dir must be provided when making monotonicity check plots"
            )
        os.makedirs(plot_dir, exist_ok=True)

    results = []
    n_issues = 0

    for bin_inputs in make_bin_inputs(hist_response):
        result = make_bin_diagnostics(
            response_map, hist_response, bin_inputs, qopr_tensor, qoprvals
        )
        results.append(result)
        log_bin_diagnostics(result)

        if result["has_issue"]:
            n_issues += 1

        if plot_all_checks or (plot_only_issues and result["has_issue"]):
            plot_bin_diagnostics(
                result, plot_dir, qoprvals, quant_cdfvals, qopr_bin_width
            )

    logger.info(
        "Monotonicity checks completed for %s bins, issues found in %s bins",
        len(results),
        n_issues,
    )
    if plot_all_checks or plot_only_issues:
        logger.info("Monotonicity check plots written to %s", plot_dir)
    return results


def make_tflite_model(response_map):
    scalar_spec = tf.TensorSpec([], tf.float64)
    input_signature = 4 * [scalar_spec]
    return wums.tfutils.function_to_tflite(response_map.interp_dweight, input_signature)


def write_tflite_model(output_dir, output_name, tflite_model):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_name)
    with open(output_path, "wb") as output_file:
        output_file.write(tflite_model)
    logger.info("Wrote response map TFLite model to %s", output_path)
    return output_path
