import narf
import narf.fitutils
import h5py
import numpy as np
import scipy
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import narf.tfutils


infile = "w_z_muonresponse_scetlib_dyturboCorr.hdf5"

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
interp_cdfvals = 0.5*(1. + scipy.special.erf(interp_sigmas/np.sqrt(2.)))
interp_cdfvals = np.concatenate([[0.], interp_cdfvals, [1.]])

# interp_cdfvals = np.linspace(0., 1., 101)

quant_cdfvals = tf.constant(interp_cdfvals, tf.float64)

quant_cdfvals = quant_cdfvals[None, :, None, None]

quants, errs = narf.fitutils.hist_to_quantiles(hist_response, quant_cdfvals, axis = 1)

print(quants[0, :, 0, 0])

quants = tf.constant(quants, tf.float64)
# quants = quants[..., None]

grid_points = [tf.constant(axis.centers) for axis in hist_response.axes]
grid_points = grid_points[2:]
grid_points = tuple(grid_points)



qopr_edges_flat = np.reshape(hist_response.axes[1].edges, [-1])
qopr_low = tf.constant(qopr_edges_flat[0], tf.float64)
qopr_high = tf.constant(qopr_edges_flat[-1], tf.float64)

pt_edges_flat = np.reshape(hist_response.axes[2].edges, [-1])
pt_low = tf.constant(pt_edges_flat[0], tf.float64)
pt_high = tf.constant(pt_edges_flat[-1], tf.float64)

eta_edges_flat = np.reshape(hist_response.axes[3].edges, [-1])
eta_low = tf.constant(eta_edges_flat[0], tf.float64)
eta_high = tf.constant(eta_edges_flat[-1], tf.float64)

print("qopr bounds:", qopr_low, qopr_high)
print("pt bounds:", pt_low, pt_high)
print("eta bounds:", eta_low, eta_high)

def interp_cdf(genPt, genEta, genCharge, qopr):
    chargeIdx = tf.where(genCharge > 0., 1, 0)
    quants_charge = quants[chargeIdx]

    x = tf.stack([genPt, genEta], axis=0)
    x = x[None, :]
    quants_interp = tfp.math.batch_interp_rectilinear_nd_grid(x, x_grid_points = grid_points, y_ref = quants_charge, axis = 1)

    quants_interp = tf.reshape(quants_interp, [-1])
    quant_cdfvals_interp = tf.reshape(quant_cdfvals, [-1])

    qopr = tf.clip_by_value(qopr, qopr_low, qopr_high)

    qopr = tf.reshape(qopr, [-1])

    cdf = narf.fitutils.pchip_interpolate(xi = quants_interp, yi = quant_cdfvals_interp, x = qopr)

    return cdf

def interp_dweight(genPt, genEta, genCharge, qopr):
    with tf.GradientTape() as t0:
        t0.watch(qopr)
        with tf.GradientTape() as t1:
            t1.watch(qopr)
            cdf = interp_cdf(genPt, genEta, genCharge, qopr)
        pdf = t1.gradient(cdf, qopr)
    dpdf = t0.gradient(pdf, qopr)
    dweight = dpdf/pdf

    dweight = tf.where(qopr <= qopr_low, tf.zeros_like(dweight), dweight)
    dweight = tf.where(qopr >= qopr_high, tf.zeros_like(dweight), dweight)

    dweight = tf.where(genPt < pt_low, tf.zeros_like(dweight), dweight)
    dweight = tf.where(genPt > pt_high, tf.zeros_like(dweight), dweight)
    #
    dweight = tf.where(genEta < eta_low, tf.zeros_like(dweight), dweight)
    dweight = tf.where(genEta > eta_high, tf.zeros_like(dweight), dweight)

    return dweight

genPt_test = tf.constant(25., tf.float64)
genEta_test = tf.constant(0.1, tf.float64)
genCharge_test = tf.constant(1., tf.float64)
qopr_test = tf.constant(1.002, tf.float64)

res = interp_cdf(genPt_test, genEta_test, genCharge_test, qopr_test)
res2 = interp_dweight(genPt_test, genEta_test, genCharge_test, qopr_test)

print("res", res)
print("res2", res2)

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

    cdf = narf.fitutils.pchip_interpolate(xi = quants, yi = quant_cdfvals, x = xedges[axis], axis=axis)

    pdf = cdf[:,1:] - cdf[:,:-1]
    # pdf = tf.maximum(pdf, tf.zeros_like(pdf))

    return pdf



# etaidx = 47
# ptidx = 20

ptidx = 20
etaidx = 24

pdfvals = func_pdf(hist_response)
pdfvals_sel = pdfvals[1, :, ptidx, etaidx]
hist_response_sel = hist_response[1, :, ptidx, etaidx]

pdfvals_sel *= hist_response_sel.sum().value/np.sum(pdfvals_sel)


# hplot = htest[5]

plot = plt.figure()
hist_response_sel.plot()
plt.plot(hist_response_sel.axes[0].centers, pdfvals_sel)
# plt.show()
plt.xlim([0.9, 1.1])
# plt.xlim([0.8, 1.2])
plot.savefig("test.png")
plt.yscale("log")
plot.savefig("test_log.png")

