import narf
import narf.fitutils
import h5py
import numpy as np
import scipy
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import narf.tfutils
import math

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300


infile = "w_z_muonresponse_scetlib_dyturboCorr_maxFiles_m1.hdf5"

hist_response = None
hist_response_scaled = None
hist_response_smeared = None

procs = []
procs.append("ZmumuPostVFP")
procs.append("ZtautauPostVFP")
procs.append("WplusmunuPostVFP")
procs.append("WminusmunuPostVFP")
procs.append("WplustaunuPostVFP")
procs.append("WminustaunuPostVFP")


with h5py.File(infile, 'r') as f:
    results = narf.ioutils.pickle_load_h5py(f["results"])
    for proc in procs:
        hist_response_proc = results[proc]["output"]["hist_qopr"].get()
        hist_response_scaled_proc = results[proc]["output"]["hist_qopr_shifted"].get()
        hist_response_smeared_proc = results[proc]["output"]["hist_qopr_smearedmulti"].get()
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
hist_response_scaled = hist_response_scaled.project("genCharge", "qopr", "genPt", "genEta")
hist_response_smeared = hist_response_smeared.project("genCharge", "qopr", "genPt", "genEta")

print(hist_response)

interp_sigmas = np.linspace(-5., 5., 21)
interp_cdfvals = scipy.stats.norm.cdf(interp_sigmas)
interp_cdfvals = np.concatenate([[0.], interp_cdfvals, [1.]])


quant_cdfvals = tf.constant(interp_cdfvals, tf.float64)

quant_cdfvals = quant_cdfvals[None, :, None, None]
quant_cdfvals_interp = tf.reshape(quant_cdfvals, [-1])


print("quant_cdfvals", quant_cdfvals)


quants, _ = narf.fitutils.hist_to_quantiles(hist_response, quant_cdfvals, axis = 1)
quants_scaled, _ = narf.fitutils.hist_to_quantiles(hist_response_scaled, quant_cdfvals, axis = 1)
quants_smeared, _ = narf.fitutils.hist_to_quantiles(hist_response_smeared, quant_cdfvals, axis = 1)


# print("quants", quants, quants_scaled, quants_smeared)
dquants = np.sum((quants_scaled-quants)**2)
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
    chargeIdx = tf.where(genCharge > 0., 1, 0)
    quants_charge = quants_sel[chargeIdx]

    x = tf.stack([genPt, genEta], axis=0)
    x = x[None, :]
    quants_interp = tfp.math.batch_interp_rectilinear_nd_grid(x, x_grid_points = grid_points, y_ref = quants_charge, axis = 1)

    quants_interp = tf.reshape(quants_interp, [-1])
    quant_cdfvals_interp = tf.reshape(quant_cdfvals, [-1])

    qopr = tf.clip_by_value(qopr, qopr_low, qopr_high)

    qopr = tf.reshape(qopr, [-1])

    # cdf = narf.fitutils.pchip_interpolate(xi = quants_interp, yi = quant_cdfvals_interp, x = qopr)
    cdf = narf.fitutils.cubic_spline_interpolate(xi = quants_interp[..., None], yi = quant_cdfvals_interp[..., None], x = qopr[..., None], axis=0)
    cdf = cdf[..., 0]

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
    cdf_smeared, pdf_smeared= interp_pdf(quants_smeared, genPt, genEta, genCharge, qopr)

    dweightdscale = -dpdf/pdf
    dweightdsigmasq = (pdf_smeared - pdf)/pdf/dsigmasq

    in_range = (qopr > qopr_low) & (qopr < qopr_high)  & (genPt > pt_low) & (genPt < pt_high) & (genEta > eta_low) & (genEta < eta_high)

    dweightdscale = tf.where(in_range, dweightdscale, tf.zeros_like(dweightdscale))
    dweightdsigmasq = tf.where(in_range, dweightdsigmasq, tf.zeros_like(dweightdsigmasq))

    dweightdscale = tf.where(tf.math.is_finite(dweightdscale), dweightdscale, tf.zeros_like(dweightdscale))
    dweightdsigmasq = tf.where(tf.math.is_finite(dweightdsigmasq), dweightdsigmasq, tf.zeros_like(dweightdsigmasq))

    return dweightdscale, dweightdsigmasq


genPt_test = tf.constant(25., tf.float64)
genEta_test = tf.constant(0.1, tf.float64)
genCharge_test = tf.constant(1., tf.float64)
qopr_test = tf.constant(1.002, tf.float64)

res = interp_cdf(quants, genPt_test, genEta_test, genCharge_test, qopr_test)
res2a, res2b = interp_dweight(genPt_test, genEta_test, genCharge_test, qopr_test)

print("res", res)
print("res2a", res2a)
print("res2b", res2b)

scalar_spec = tf.TensorSpec([], tf.float64)
input_signature = 4*[scalar_spec]

tflite_model = narf.tfutils.function_to_tflite(interp_dweight, input_signature)

output_filename = "muon_response.tflite"

with open(output_filename, 'wb') as f:
    f.write(tflite_model)

#this is just for plotting
def func_pdf(h):
    dtype = tf.float64
    xvals = [tf.constant(center, dtype=dtype) for center in h.axes.centers]
    xedges = [tf.constant(edge, dtype=dtype) for edge in h.axes.edges]
    axis=1

    # cdf = narf.fitutils.pchip_interpolate(xi = quants, yi = quant_cdfvals, x = xedges[axis], axis=axis)
    cdf = narf.fitutils.cubic_spline_interpolate(xi = quants, yi = quant_cdfvals, x = xedges[axis], axis=axis)

    pdf = cdf[:,1:] - cdf[:,:-1]
    # pdf = tf.maximum(pdf, tf.zeros_like(pdf))

    return pdf

qoprvals = np.linspace(0., 2., 1000)
# qoprvals = np.linspace(0.9, 1.1, 1000)

# etaidx = 47
# ptidx = 20

ptidx = 20
etaidx = 24

# ptidx = 0
# etaidx = 47


centers_flat = [np.reshape(center, [-1]) for center in hist_response.axes.centers]

testpt = centers_flat[2][ptidx]
testeta = centers_flat[3][etaidx]

testpt = tf.constant(testpt, tf.float64)
testeta = tf.constant(testeta, tf.float64)

testcharge = tf.constant(1.0, tf.float64)

print("testpt, eta", testpt, testeta)

hist_response_sel = hist_response[1, :, ptidx, etaidx]


pdfvals_sel = []
dpdfvals_sel = []
d2pdfvals_sel = []

for qoprval in qoprvals:
    testqopr = tf.constant(qoprval, tf.float64)
    cdf, pdf, dpdf = interp_dpdf(quants, testpt, testeta, testcharge, testqopr)
    cdf_smeared, pdf_smeared = interp_pdf(quants_smeared, testpt, testeta, testcharge, testqopr)
    d2pdf = (pdf_smeared - pdf)/dsigmasq
    pdfvals_sel.append(pdf.numpy())
    dpdfvals_sel.append(dpdf.numpy())
    d2pdfvals_sel.append(d2pdf.numpy())

pdfvals_sel = np.array(pdfvals_sel)
dpdfvals_sel = np.array(dpdfvals_sel)
d2pdfvals_sel = np.array(d2pdfvals_sel)


print("pdfvals_sel", pdfvals_sel)

integral = np.sum(pdfvals_sel)*(qoprvals[1]-qoprvals[0])

pdfvals_sel *= hist_response_sel.sum().value*(centers_flat[1][1] - centers_flat[1][0])


plot = plt.figure()
hist_response_sel.plot()
# plt.plot(hist_response_sel.axes[0].centers, pdfvals_sel)
plt.plot(qoprvals, pdfvals_sel)
# plt.show()
plt.xlim([0.9, 1.1])
# plt.xlim([0.8, 1.2])
plot.savefig("test.png")
plt.yscale("log")
plot.savefig("test_log.png")


plot = plt.figure()
plt.plot(qoprvals, dpdfvals_sel)
plt.xlim([0.9, 1.1])
plot.savefig("testd.png")


plot = plt.figure()
plt.plot(qoprvals, d2pdfvals_sel)
plt.xlim([0.9, 1.1])
plot.savefig("testd2.png")

