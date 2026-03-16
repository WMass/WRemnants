import os

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy
import tensorflow as tf
import tensorflow_probability as tfp

import wums.fitutils
import wums.ioutils
import wums.tfutils

mpl.rcParams["figure.dpi"] = 300

# TODO migrate butojpsik stuff to separate file and leave this untouched or generalize correctly...
BuToJpsiK = None

infile = "w_z_muonresponse_scetlib_dyturboCorr_maxFiles_m1.hdf5"
postfix = ""
if BuToJpsiK:
    postfix = "debug"
    infile = "/ceph/submit/data/user/p/pmlugato/mz/calibration/kaonresponse_scetlib_dyturboCorr_debug.hdf5"

hist_response = None
hist_response_scaled = None
hist_response_smeared = None

procs = []
procs.append("Zmumu_2016PostVFP")
procs.append("Ztautau_2016PostVFP")
procs.append("Wplusmunu_2016PostVFP")
procs.append("Wminusmunu_2016PostVFP")
procs.append("Wplustaunu_2016PostVFP")
procs.append("Wminustaunu_2016PostVFP")

if BuToJpsiK:
    procs = ["BuToJpsiK"]


with h5py.File(infile, "r") as f:
    for proc in procs:
        results = wums.ioutils.pickle_load_h5py(f[proc])
        hist_response_proc = results["output"]["hist_qopr"].get()
        hist_response_scaled_proc = results["output"]["hist_qopr_shifted"].get()
        hist_response_smeared_proc = results["output"]["hist_qopr_smearedmulti"].get()
        if hist_response is None:
            hist_response = hist_response_proc
            hist_response_scaled = hist_response_scaled_proc
            hist_response_smeared = hist_response_smeared_proc
        else:
            hist_response += hist_response_proc
            hist_response_scaled += hist_response_scaled_proc
            hist_response_smeared += hist_response_smeared_proc


print(hist_response)

dscale = hist_response_scaled.metadata["scalerel"]
dsigma = hist_response_smeared.metadata["sigmarel"]
dsigmasq = dsigma**2

dscale = tf.constant(dscale, tf.float64)
dsigmasq = tf.constant(dsigmasq, tf.float64)


hist_response = hist_response.project("genCharge", "qopr", "genPt", "genEta")
hist_response_scaled = hist_response_scaled.project(
    "genCharge", "qopr", "genPt", "genEta"
)
hist_response_smeared = hist_response_smeared.project(
    "genCharge", "qopr", "genPt", "genEta"
)

print(hist_response)

interp_sigmas = np.linspace(-5.0, 5.0, 21)
# interp_sigmas = np.linspace(-3.0, 3.0, 15)
interp_cdfvals = scipy.stats.norm.cdf(interp_sigmas)
# interp_cdfvals = np.linspace(0.0, 1.0, 20)

print("interp_cdfvals", interp_cdfvals)

interp_cdfvals = np.concatenate([[0.0], interp_cdfvals, [1.0]])

# trying clamping because cdf flat in tails from low stats
# pmin = 1e-3 # without is like O(10^-7)
# interp_cdfvals = interp_cdfvals[(interp_cdfvals >= pmin) & (interp_cdfvals <= 1.0 - pmin)]

quant_cdfvals = tf.constant(interp_cdfvals, tf.float64)

quant_cdfvals = quant_cdfvals[None, :, None, None]
quant_cdfvals_interp = tf.reshape(quant_cdfvals, [-1])


print("quant_cdfvals", quant_cdfvals)

quants, _ = wums.fitutils.hist_to_quantiles(hist_response, quant_cdfvals, axis=1)
quants_scaled, _ = wums.fitutils.hist_to_quantiles(
    hist_response_scaled, quant_cdfvals, axis=1
)
quants_smeared, _ = wums.fitutils.hist_to_quantiles(
    hist_response_smeared, quant_cdfvals, axis=1
)


# print("quants", quants)
# print("quants_scaled", quants_scaled)
# print("quants_smeared", quants_smeared)
dquants = np.sum((quants_scaled - quants) ** 2)
print("dquants", dquants)

print("non-finite quants:", np.count_nonzero(np.invert(np.isfinite(quants))))


quants = tf.constant(quants, tf.float64)
quants_scaled = tf.constant(quants_scaled, tf.float64)
quants_smeared = tf.constant(quants_smeared, tf.float64)

grid_points = [tf.constant(axis.centers) for axis in hist_response.axes]
grid_points = grid_points[2:]
grid_points = tuple(grid_points)


qopr_edges_flat = np.reshape(hist_response.axes[1].edges, [-1])

qopr_low = qopr_edges_flat[0]
qopr_high = qopr_edges_flat[-1]

qopr_low = tf.constant(qopr_low, tf.float64)
qopr_high = tf.constant(qopr_high, tf.float64)

pt_edges_flat = np.reshape(hist_response.axes[2].edges, [-1])
pt_low = tf.constant(pt_edges_flat[0], tf.float64)
pt_high = tf.constant(pt_edges_flat[-1], tf.float64)

eta_edges_flat = np.reshape(hist_response.axes[3].edges, [-1])
eta_low = tf.constant(eta_edges_flat[0], tf.float64)
eta_high = tf.constant(eta_edges_flat[-1], tf.float64)

print("qopr bounds:", qopr_low, qopr_high)
print("pt bounds:", pt_low, pt_high)
print("eta bounds:", eta_low, eta_high)


def interp_cdf(quants_sel, genPt, genEta, genCharge, qopr):
    chargeIdx = tf.where(genCharge > 0.0, 1, 0)
    quants_charge = quants_sel[chargeIdx]

    x = tf.stack([genPt, genEta], axis=0)
    x = x[None, :]
    quants_interp = tfp.math.batch_interp_rectilinear_nd_grid(
        x, x_grid_points=grid_points, y_ref=quants_charge, axis=1
    )

    quants_interp = tf.reshape(quants_interp, [-1])
    quant_cdfvals_interp = tf.reshape(quant_cdfvals, [-1])

    qopr = tf.clip_by_value(qopr, qopr_low, qopr_high)

    qopr = tf.reshape(qopr, [-1])

    # cdf = wums.fitutils.pchip_interpolate(xi = quants_interp, yi = quant_cdfvals_interp, x = qopr)
    cdf = wums.fitutils.cubic_spline_interpolate(
        xi=quants_interp[..., None],
        yi=quant_cdfvals_interp[..., None],
        x=qopr[..., None],
        axis=0,
    )
    cdf = cdf[..., 0]  # for cubic interp

    return cdf


def interp_dpdf(quants_sel, genPt, genEta, genCharge, qopr):
    with tf.GradientTape() as t0:
        t0.watch(qopr)
        with tf.GradientTape() as t1:
            t1.watch(qopr)
            cdf = interp_cdf(quants_sel, genPt, genEta, genCharge, qopr)
        pdf = t1.gradient(cdf, qopr)
    dpdf = t0.gradient(pdf, qopr)

    return cdf, pdf, dpdf


def interp_pdf(quants_sel, genPt, genEta, genCharge, qopr):

    with tf.GradientTape() as t0:
        t0.watch(qopr)
        cdf = interp_cdf(quants_sel, genPt, genEta, genCharge, qopr)
    pdf = t0.gradient(cdf, qopr)

    return cdf, pdf


def interp_dweight(genPt, genEta, genCharge, qopr):
    cdf, pdf, dpdf = interp_dpdf(quants, genPt, genEta, genCharge, qopr)
    cdf_smeared, pdf_smeared = interp_pdf(
        quants_smeared, genPt, genEta, genCharge, qopr
    )

    dweightdscale = -dpdf / pdf
    dweightdsigmasq = (pdf_smeared - pdf) / pdf / dsigmasq

    in_range = (
        (qopr > qopr_low)
        & (qopr < qopr_high)
        & (genPt > pt_low)
        & (genPt < pt_high)
        & (genEta > eta_low)
        & (genEta < eta_high)
    )

    dweightdscale = tf.where(in_range, dweightdscale, tf.zeros_like(dweightdscale))
    dweightdsigmasq = tf.where(
        in_range, dweightdsigmasq, tf.zeros_like(dweightdsigmasq)
    )

    dweightdscale = tf.where(
        tf.math.is_finite(dweightdscale), dweightdscale, tf.zeros_like(dweightdscale)
    )
    dweightdsigmasq = tf.where(
        tf.math.is_finite(dweightdsigmasq),
        dweightdsigmasq,
        tf.zeros_like(dweightdsigmasq),
    )

    return dweightdscale, dweightdsigmasq


genPt_test = tf.constant(25.0, tf.float64)
# genPt_test = tf.constant(5.0, tf.float64)
genEta_test = tf.constant(0.1, tf.float64)
genCharge_test = tf.constant(1.0, tf.float64)
qopr_test = tf.constant(1.002, tf.float64)

res = interp_cdf(quants, genPt_test, genEta_test, genCharge_test, qopr_test)
res2a, res2b = interp_dweight(genPt_test, genEta_test, genCharge_test, qopr_test)

print("res", res)
print("res2a", res2a)
print("res2b", res2b)

scalar_spec = tf.TensorSpec([], tf.float64)
input_signature = 4 * [scalar_spec]

tflite_model = wums.tfutils.function_to_tflite(interp_dweight, input_signature)

output_filename = "muon_response.tflite"
if BuToJpsiK:
    # output_filename = "kaon_response.tflite"
    output_filename = f"kaon_response_{postfix}.tflite"

with open(output_filename, "wb") as f:
    f.write(tflite_model)


###########################

# plotting and debugging below

###########################


qoprvals = np.linspace(0.0, 2.0, 1000)  # og 1000
# qoprvals = np.linspace(0.9, 1.1, 500)

centers_flat = [np.reshape(center, [-1]) for center in hist_response.axes.centers]

# outdir = "/home/submit/pmlugato/public_html/mz/2_12_jpsimc_kpteta/splines/pchip/"
outdir = "/home/submit/pmlugato/public_html/mz/2_12_jpsimc_kpteta/splines/cubic/"
os.makedirs(outdir, exist_ok=True)

plot_all = True
plot_only_issues = False
show_plots = False


def _save_plot(fig, name):
    path = os.path.join(outdir, name)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_hist_and_smooth(
    x_centers, hist_vals, smooth_x, smooth_y, title, xlabel, ylabel, name, xlim=None
):
    fig, ax = plt.subplots()
    ax.step(x_centers, hist_vals, where="mid", label="hist", color="black")
    ax.plot(smooth_x, smooth_y, label="interp", color="C0")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xlim:
        ax.set_xlim(left=xlim[0], right=xlim[1])
    ax.legend()
    _save_plot(fig, name)


def _plot_line(x, y, title, xlabel, ylabel, name, xlim=None, color="C0"):
    fig, ax = plt.subplots()
    ax.plot(x, y, color=color)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xlim:
        ax.set_xlim(left=xlim[0], right=xlim[1])
    _save_plot(fig, name)


n_pt = len(centers_flat[2])
n_eta = len(centers_flat[3])
charge_idx = 0
ptidx = 0
etaidx = 5

testpt = tf.constant(centers_flat[2][ptidx], tf.float64)
testeta = tf.constant(centers_flat[3][etaidx], tf.float64)
testcharge = tf.constant(1.0 if charge_idx == 1 else -1.0, tf.float64)

print(f"Output dir: {outdir}")
print(f"Selected bin: ptidx={ptidx}, etaidx={etaidx}, charge={int(testcharge.numpy())}")

hist_response_sel = hist_response[charge_idx, :, ptidx, etaidx]
hist_sum = hist_response_sel.sum().value
hist_vals = hist_response_sel.values()
hist_vals_norm = hist_vals / hist_sum if hist_sum > 0 else hist_vals
qopr_centers = hist_response_sel.axes[0].centers
qopr_edges = hist_response_sel.axes[0].edges
qopr_widths = np.diff(qopr_edges)
hist_density = hist_vals_norm / qopr_widths if hist_sum > 0 else hist_vals

# Empirical CDF from histogram
print("Computing histogram CDF")
hist_cdf = np.cumsum(hist_vals_norm)
hist_cdf = np.clip(hist_cdf, 0.0, 1.0)

# Interpolated CDF/PDF and derivatives
print("Computing interpolated CDF/PDF and derivatives")
qopr_grid = np.linspace(qopr_edges[0], qopr_edges[-1], 500)
qopr_grid_tf = tf.constant(qopr_grid, tf.float64)

cdf_vals = interp_cdf(quants, testpt, testeta, testcharge, qopr_grid_tf).numpy()
with tf.GradientTape() as t0:
    t0.watch(qopr_grid_tf)
    with tf.GradientTape() as t1:
        t1.watch(qopr_grid_tf)
        cdf_tf = interp_cdf(quants, testpt, testeta, testcharge, qopr_grid_tf)
    pdf_tf = t1.gradient(cdf_tf, qopr_grid_tf)
dpdf_tf = t0.gradient(pdf_tf, qopr_grid_tf)

pdf_vals = pdf_tf.numpy()
dpdf_vals = dpdf_tf.numpy()

# Normalize interpolated PDF for consistent overlay
pdf_area = np.trapezoid(np.clip(pdf_vals, 0.0, None), qopr_grid)
if pdf_area > 0:
    pdf_vals_norm = pdf_vals / pdf_area
else:
    pdf_vals_norm = pdf_vals
print(
    f"pdf_area={pdf_area:.6g} hist_sum={hist_sum:.6g} "
    f"hist_norm_sum={hist_vals_norm.sum():.6g}"
)

# dweight/dscale = -dpdf/pdf
print("Computing dweight/dscale")
with np.errstate(divide="ignore", invalid="ignore"):
    dweightdscale_vals = -dpdf_vals / pdf_vals
    dweightdscale_vals = np.where(
        np.isfinite(dweightdscale_vals), dweightdscale_vals, 0.0
    )

title_prefix = f"ptidx{ptidx} etaidx{etaidx} q{int(testcharge.numpy())}"

# qopr histogram + pdf overlay
print("Saving qopr pdf overlay")
_plot_hist_and_smooth(
    qopr_centers,
    hist_density,
    qopr_grid,
    np.clip(pdf_vals, 0.0, None),
    f"{title_prefix} PDF overlay",
    r"$(q/p)_{reco} \quad / \quad (q/p)_{gen}$",
    "pdf",
    f"qopr_pdf_overlay_pt{ptidx}_eta{etaidx}_q{int(testcharge.numpy())}.png",
    xlim=[0.8, 1.2],
)

print("Save qopr distribution w/out PDF overlay")
_plot_line(
    qopr_centers,
    hist_density,
    f"{title_prefix} qopr",
    r"$(q/p)_{reco} \quad / \quad (q/p)_{gen}$",
    "",
    f"qopr_hist_pt{ptidx}_eta{etaidx}_q{int(testcharge.numpy())}.png",
    xlim=[0.8, 1.2],
    color="black",
)

# CDF overlay (hist CDF + interpolated CDF)
print("Saving qopr cdf overlay")
_plot_hist_and_smooth(
    qopr_centers,
    hist_cdf,
    qopr_grid,
    cdf_vals,
    f"{title_prefix} qopr cdf",
    "qopr",
    "cdf",
    f"qopr_cdf_overlay_pt{ptidx}_eta{etaidx}_q{int(testcharge.numpy())}.png",
)

# First derivative of CDF = pdf
print("Saving pdf")
_plot_line(
    qopr_grid,
    pdf_vals_norm,
    f"{title_prefix} pdf",
    "qopr",
    "pdf",
    f"qopr_pdf_pt{ptidx}_eta{etaidx}_q{int(testcharge.numpy())}.png",
)

# Second derivative of CDF = dpdf/dqopr
print("Saving dpdf/dqopr")
_plot_line(
    qopr_grid,
    dpdf_vals,
    f"{title_prefix} dpdf/dqopr",
    "qopr",
    "d(pdf)/d(qopr)",
    f"qopr_dpdf_pt{ptidx}_eta{etaidx}_q{int(testcharge.numpy())}.png",
)

# dweight/dscale
print("Saving dweight/dscale")
_plot_line(
    qopr_grid,
    dweightdscale_vals,
    f"{title_prefix} dweight/dscale",
    "qopr",
    "dweight/dscale",
    f"dweightdscale_pt{ptidx}_eta{etaidx}_q{int(testcharge.numpy())}.png",
)

# Quantiles plot
print("Saving quantiles")
q = quants[charge_idx, :, ptidx, etaidx].numpy()
_plot_line(
    quant_cdfvals_interp.numpy(),
    q,
    f"{title_prefix} quantiles",
    "cdf",
    "qopr",
    f"qopr_quantiles_pt{ptidx}_eta{etaidx}_q{int(testcharge.numpy())}.png",
)

debug = False
if debug:
    for ptidx in range(n_pt):
        for etaidx in range(n_eta):
            testpt = tf.constant(centers_flat[2][ptidx], tf.float64)
            testeta = tf.constant(centers_flat[3][etaidx], tf.float64)
            testcharge = tf.constant(1.0 if charge_idx == 1 else -1.0, tf.float64)

            q = quants[charge_idx, :, ptidx, etaidx].numpy()
            q_diffs = np.diff(q)
            quants_monotone = np.all(q_diffs > 0) if q_diffs.size else False
            quants_min_diff = np.min(q_diffs) if q_diffs.size else np.nan

            x = tf.stack([testpt, testeta], axis=0)[None, :]
            quants_charge = quants[charge_idx]
            quants_interp = tfp.math.batch_interp_rectilinear_nd_grid(
                x, x_grid_points=grid_points, y_ref=quants_charge, axis=1
            )
            quants_interp = tf.reshape(quants_interp, [-1]).numpy()
            quants_interp_diffs = np.diff(quants_interp)
            quants_interp_monotone = (
                np.all(quants_interp_diffs > 0) if quants_interp_diffs.size else False
            )
            quants_interp_min_diff = (
                np.min(quants_interp_diffs) if quants_interp_diffs.size else np.nan
            )

            cdf_vals = []
            for qoprval in qoprvals:
                cdf_val = interp_cdf(
                    quants,
                    testpt,
                    testeta,
                    testcharge,
                    tf.constant(qoprval, tf.float64),
                ).numpy()
                cdf_vals.append(cdf_val)

            cdf_vals = np.squeeze(np.array(cdf_vals))
            print()
            print()
        cdf_diffs = np.diff(cdf_vals, axis=0)
        # print(cdf_diffs)
        print()
        cdf_monotone = np.all(cdf_diffs >= 0) if cdf_diffs.size else False
        cdf_min_diff = np.min(cdf_diffs) if cdf_diffs.size else np.nan
        if cdf_diffs.size == 0:
            print(
                "empty cdf diffs at",
                "ptidx",
                ptidx,
                "etaidx",
                etaidx,
                "cdf_vals_len",
                len(cdf_vals),
            )

        hist_response_sel = hist_response[charge_idx, :, ptidx, etaidx]
        hist_sum = hist_response_sel.sum().value
        hist_min = np.min(hist_response_sel.values())

        hist_nonzero_bins = np.sum(hist_response_sel.values() > 0)
        hist_max_bin = np.max(hist_response_sel.values())

        print(
            "hist_nonzero_bins",
            hist_nonzero_bins,
            "hist_max/sum",
            hist_max_bin / hist_sum if hist_sum > 0 else 0,
        )

        pdfvals_sel = []
        dpdfvals_sel = []
        d2pdfvals_sel = []
        dweightdscale_vals_sel = []

        for qoprval in qoprvals:
            testqopr = tf.constant(qoprval, tf.float64)
            cdf, pdf, dpdf = interp_dpdf(quants, testpt, testeta, testcharge, testqopr)
            cdf_smeared, pdf_smeared = interp_pdf(
                quants_smeared, testpt, testeta, testcharge, testqopr
            )
            d2pdf = (pdf_smeared - pdf) / dsigmasq
            pdfvals_sel.append(pdf.numpy())
            dpdfvals_sel.append(dpdf.numpy())
            d2pdfvals_sel.append(d2pdf.numpy())
            if pdf.numpy() != 0.0:
                dweightdscale_vals_sel.append((-dpdf / pdf).numpy())
            else:
                dweightdscale_vals_sel.append(np.nan)

        pdfvals_sel = np.array(pdfvals_sel)
        dpdfvals_sel = np.array(dpdfvals_sel)
        d2pdfvals_sel = np.array(d2pdfvals_sel)
        dweightdscale_vals_sel = np.array(dweightdscale_vals_sel)

        pdf_min = np.min(pdfvals_sel)
        pdf_has_neg = pdf_min < 0
        integral = np.sum(pdfvals_sel) * (qoprvals[1] - qoprvals[0])

        print(
            "ptidx",
            ptidx,
            "etaidx",
            etaidx,
            "\nhist_sum",
            hist_sum,
            "hist_min",
            hist_min,
            "\nqopr quants",
            q,
            "\nquants_mono",
            quants_monotone,
            "quants_min_diff",
            quants_min_diff,
            "\nqinterp_mono",
            quants_interp_monotone,
            "qinterp_min_diff",
            quants_interp_min_diff,
            "\ncdf_mono",
            cdf_monotone,
            "cdf_min_diff",
            cdf_min_diff,
            "\npdf_min",
            pdf_min,
            "pdf_has_neg",
            pdf_has_neg,
            "pdf_int",
            integral,
        )

        # After computing CDF for the histogram
        hist_cdf = np.cumsum(hist_response_sel.values()) / hist_sum
        flat_regions = np.sum(np.diff(hist_cdf) == 0)
        print("CDF flat regions:", flat_regions)

        has_issue = (
            (not quants_monotone)
            or (not quants_interp_monotone)
            or (not cdf_monotone)
            or pdf_has_neg
        )

        print("HAS ISSUE?", has_issue)
        print()

        if not (plot_all or (plot_only_issues and has_issue)):
            continue

        pdfvals_scaled = (
            pdfvals_sel * hist_sum * (centers_flat[1][1] - centers_flat[1][0])
        )

        name = f"ptidx{ptidx}_etaidx{etaidx}"
        plot = plt.figure()
        hist_response_sel.plot()
        plt.plot(qoprvals, pdfvals_scaled)
        plt.xlim([0.9, 1.1])
        plot.savefig(outdir + f"{name}_{postfix}.png")
        plt.yscale("log")
        plot.savefig(outdir + f"{name}_{postfix}_log.png")
        if show_plots:
            plt.show()
        plt.close(plot)

        plot = plt.figure()
        hist_response_sel.plot()
        # Overlay the quantile positions
        q_interp = quants_interp
        cdf_values_to_plot = interp_cdfvals  # your 23 CDF values
        plt.scatter(
            q_interp,
            np.zeros_like(q_interp),
            c=cdf_values_to_plot,
            cmap="viridis",
            s=50,
            zorder=5,
            label="Quantiles",
        )
        plt.colorbar(label="CDF value")
        # plt.xlim([0.9, 1.1])
        plt.legend()
        plot.savefig(outdir + f"{name}_{postfix}_quantiles.png")
        plt.close(plot)

        plot = plt.figure()
        plt.plot(qoprvals, cdf_vals)
        plt.xlim([0.9, 1.1])
        plot.savefig(outdir + f"{name}_{postfix}_cdf.png")
        if show_plots:
            plt.show()
        plt.close(plot)

        plot = plt.figure()
        plt.plot(qoprvals, dpdfvals_sel)
        plt.xlim([0.9, 1.1])
        plot.savefig(outdir + f"{name}_{postfix}_d.png")
        if show_plots:
            plt.show()
        plt.close(plot)

        plot = plt.figure()
        plt.plot(qoprvals, d2pdfvals_sel)
        plt.xlim([0.9, 1.1])
        plot.savefig(outdir + f"{name}_{postfix}_d2.png")
        if show_plots:
            plt.show()
        plt.close(plot)

        plot = plt.figure()
        mask = np.abs(dweightdscale_vals_sel) < 1e4
        plt.plot(qoprvals[mask], dweightdscale_vals_sel[mask])
        plt.xlim([0.9, 1.1])
        plot.savefig(outdir + f"{name}_{postfix}_dweightdscale.png")
        if show_plots:
            plt.show()
        plt.close(plot)
