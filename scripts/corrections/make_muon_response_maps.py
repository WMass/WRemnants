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


infile = "w_z_muonresponse_scetlib_dyturboCorr_NonClosureCorl.hdf5"

hist_response = None

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
        if hist_response is None:
            hist_response = hist_response_proc
        else:
            hist_response += hist_response_proc


print(hist_response)


hist_response = hist_response.project("genCharge", "qopr", "genPt", "genEta")

print(hist_response)

interp_sigmas = np.linspace(-5., 5., 21)
interp_cdfvals = scipy.stats.norm.cdf(interp_sigmas)
interp_cdfvals = np.concatenate([[0.], interp_cdfvals, [1.]])


quant_cdfvals = tf.constant(interp_cdfvals, tf.float64)

quant_cdfvals = quant_cdfvals[None, :, None, None]
quant_cdfvals_interp = tf.reshape(quant_cdfvals, [-1])


print("quant_cdfvals", quant_cdfvals)


quants, errs = narf.fitutils.hist_to_quantiles(hist_response, quant_cdfvals, axis = 1)


print("non-finite quants:", np.count_nonzero(np.invert(np.isfinite(quants))))



quants = tf.constant(quants, tf.float64)

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

# first use a cubic spline to estimate the gradients at the knot points
knot_slopes = np.zeros_like(quants)
for i0 in range(knot_slopes.shape[0]):
    for i2 in range(knot_slopes.shape[2]):
        for i3 in range(knot_slopes.shape[3]):
            spline = scipy.interpolate.make_interp_spline(x = quants[i0,:,i2,i3], y = quant_cdfvals_interp, k=3,  bc_type="natural")
            grad = spline(quants[i0,:,i2,i3], nu=1)
            knot_slopes[i0, :, i2, i3] = grad

knot_slopes = tf.constant(knot_slopes, tf.float64)

# now the final interpolation in tensorflow, where the C2 continuity of the cubic spline guarantees continuous second derivatives needed for the resolution variation weights

# n.b. in an ideal world we would be able to do this directly in one step with a C3 continuous monotonic spline
# but an implementation of this doesn't currently exist in tensorflow

def interp_pdf(genPt, genEta, genCharge, qopr):
    chargeIdx = tf.where(genCharge > 0., 1, 0)
    quants_charge = quants[chargeIdx]

    qopr = tf.reshape(qopr, [-1])


    x = tf.stack([genPt, genEta], axis=0)
    x = x[None, :]
    quants_interp = tfp.math.batch_interp_rectilinear_nd_grid(x, x_grid_points = grid_points, y_ref = quants_charge, axis = 1)
    quants_interp = tf.reshape(quants_interp, [-1])

    knot_slopes_charge = knot_slopes[chargeIdx]
    knot_slopes_interp = tfp.math.batch_interp_rectilinear_nd_grid(x, x_grid_points = grid_points, y_ref = knot_slopes_charge, axis = 1)
    knot_slopes_interp = tf.reshape(knot_slopes_interp, [-1])

    pdf = narf.fitutils.cubic_spline_interpolate(xi = quants_interp[None, :], yi = knot_slopes_interp[None, :], x = qopr[None, :])
    pdf = tf.reshape(pdf, [])


    # neither monotonicity of the initial spline estimating the gradients, nor positivity of the second spline is
    # otherwise enforced so we need to explicitly clip the pdf to prevent negative values
    pdf = tf.maximum (pdf, tf.constant(0., tf.float64))

    return pdf


def interp_grads(genPt, genEta, genCharge, qopr):
    with tf.GradientTape() as t0:
        t0.watch(qopr)
        with tf.GradientTape() as t1:
            t1.watch(qopr)
            pdf = interp_pdf(genPt, genEta, genCharge, qopr)
        dpdf = t1.gradient(pdf, qopr)
    d2pdf = t0.gradient(dpdf, qopr)

    return pdf, dpdf, d2pdf

def interp_dweight(genPt, genEta, genCharge, qopr):

    with tf.GradientTape() as t0:
        t0.watch(qopr)
        with tf.GradientTape() as t1:
            t1.watch(qopr)
            pdf = interp_pdf(genPt, genEta, genCharge, qopr)
        dpdf = t1.gradient(pdf, qopr)
    d2pdf = t0.gradient(dpdf, qopr)

    pdf, dpdf, d2pdf = interp_grads(genPt, genEta, genCharge, qopr)

    dweightdscale = -dpdf/pdf
    dweightdsigmasq = 0.5*d2pdf/pdf

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

res = interp_pdf(genPt_test, genEta_test, genCharge_test, qopr_test)
res2 = interp_dweight(genPt_test, genEta_test, genCharge_test, qopr_test)

print("res", res)
print("res2", res2)

scalar_spec = tf.TensorSpec([], tf.float64)
input_signature = 4*[scalar_spec]

tflite_model = narf.tfutils.function_to_tflite(interp_dweight, input_signature)

output_filename = "muon_response.tflite"

with open(output_filename, 'wb') as f:
    f.write(tflite_model)

#
# qoprvals = np.linspace(0., 2., 1000)
qoprvals = np.linspace(0.9, 1.1, 1000)
pttf = tf.constant(40., tf.float64)
etatf = tf.constant(0., tf.float64)
# pttf = tf.constant(9.5, tf.float64)
# etatf = tf.constant(2.35, tf.float64)
chargetf = tf.constant(1., tf.float64)

pdfs = []
dpdfs = []
d2pdfs = []

for qoprval in qoprvals:
    qopr = tf.constant(qoprval, tf.float64)
    pdf, dpdf, d2pdf = interp_grads(pttf,etatf, chargetf, qopr)

    pdfs.append(pdf)
    dpdfs.append(dpdf)
    d2pdfs.append(d2pdf)

plt.figure()
plt.plot(qoprvals, pdfs)
plt.savefig("pdf.png")

plt.figure()
plt.plot(qoprvals, dpdfs)
plt.savefig("dpdf.png")

plt.figure()
plt.plot(qoprvals, d2pdfs)
plt.savefig("d2pdf.png")


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

for qoprval in qoprvals:
    testqopr = tf.constant(qoprval, tf.float64)
    pdf = interp_pdf(testpt, testeta, testcharge, testqopr)
    pdfvals_sel.append(pdf.numpy())

pdfvals_sel = np.array(pdfvals_sel)


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

