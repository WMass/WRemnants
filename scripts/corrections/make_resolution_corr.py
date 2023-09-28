import wremnants
import narf
import numpy as np
import scipy
import ROOT
import hist
import matplotlib.pyplot as plt
import pickle
import lz4
import lz4.frame

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

fnamedata = f"{wremnants.data_dir}/calibration/sigmaDATA_LBL_JYZ.root"
fnamemc = f"{wremnants.data_dir}/calibration/sigmaMC_LBL_JYZ.root"

fdata = ROOT.TFile.Open(fnamedata)

sigmarelsq_data_root = fdata.Get("smearing")
cov_sigmarelsq_data_root = fdata.Get("covariance_matrix")

sigmarelsq_data = narf.root_to_hist(sigmarelsq_data_root)
cov_sigmarelsq_data = narf.root_to_hist(cov_sigmarelsq_data_root)

fdata.Close()


fmc = ROOT.TFile.Open(fnamemc)

sigmarelsq_mc_root = fmc.Get("smearing")
cov_sigmarelsq_mc_root = fmc.Get("covariance_matrix")

sigmarelsq_mc = narf.root_to_hist(sigmarelsq_mc_root)
cov_sigmarelsq_mc = narf.root_to_hist(cov_sigmarelsq_mc_root)

fmc.Close()

sigmarelsq_data = sigmarelsq_data[:, 25.j:]
sigmarelsq_mc = sigmarelsq_mc[:, 25.j:]

smearingrel = sigmarelsq_data + -1.*sigmarelsq_mc

centers_flat = [np.reshape(center, [-1]) for center in smearingrel.axes.centers]

neta_smooth = 96
npt_smooth = 240

axis_eta_fine = hist.axis.Regular(neta_smooth, -2.4, 2.4, underflow=True, overflow=True, name="eta")
axis_pt_fine = hist.axis.Regular(npt_smooth, 25., 85., underflow=True, overflow=True, name="pt")

smearingrel_smooth = hist.Hist(axis_eta_fine, axis_pt_fine)

spline = scipy.interpolate.RegularGridInterpolator(centers_flat, smearingrel.values(), method="cubic", bounds_error = False, fill_value = None)

centers_fine_flat = [np.reshape(center, [-1]) for center in smearingrel_smooth.axes.centers]
eval_points = np.meshgrid(*centers_fine_flat, indexing="ij")
eval_points = np.stack(eval_points, axis=-1)

smearingrel_smooth.values()[...] = spline(eval_points)

#set overflow in eta to boundary values (but leave overflow in pt as 0)
smearingrel_smooth.values(flow=True)[0, 1:-1] = smearingrel_smooth.values()[0]
smearingrel_smooth.values(flow=True)[-1, 1:-1] = smearingrel_smooth.values()[-1]

plt.figure()
# smearingrel.values()[...] = np.sqrt(smearingrel.values())
smearingrel.plot()
plt.ylim([25., 60.])
plt.savefig("smearingrel.png")
plt.close()

plt.figure()
# smearingrel_smooth.values()[...] = np.sqrt(smearingrel_smooth.values())
smearingrel_smooth.plot()
plt.ylim([25., 60.])
plt.savefig("smearingrel_smooth.png")
plt.close()

with lz4.frame.open("smearingrel_smooth.pkl.lz4", "wb") as fout:
    pickle.dump(smearingrel_smooth, fout, protocol=pickle.HIGHEST_PROTOCOL)

# verify consistency of histogram errors with covariance matrices
def check_errs(sigmarelsq, cov_sigmarelsq):
    sigmarelsqvals = sigmarelsq.values()
    variance_sigmarelsq = sigmarelsq.variances()

    variance_sigmarelsq_flat = np.reshape(variance_sigmarelsq, [-1])

    r = variance_sigmarelsq_flat/np.diag(cov_sigmarelsq.values())

    if not np.all(np.isclose(r, 1.0)):
        raise ValueError("Inconsistent variances between histogram and covariance matrix.")

    print(r)


check_errs(sigmarelsq_data, cov_sigmarelsq_data)
check_errs(sigmarelsq_mc, cov_sigmarelsq_mc)

cov_smearing = cov_sigmarelsq_data + cov_sigmarelsq_mc

# inflation factor for Z MC stat uncertainty which is not directly included
cov_smearing *= 1.47

e, v = np.linalg.eigh(cov_smearing.values())

neig = e.shape[0]

vscaled = np.sqrt(e[None, :])*v

print(vscaled.shape)

smearing_variations_values = np.reshape(vscaled, [*smearingrel.values().shape, neig])



# print(smearing_variations)

axis_var = hist.axis.Integer(0, neig, underflow=False, overflow=False, name="smearing_variation")

smearing_variations = hist.Hist(*smearingrel.axes, axis_var)
smearing_variations.values()[...] = smearing_variations_values

smearing_variations_smooth = hist.Hist(*smearingrel_smooth.axes, axis_var)

for ieig in range(neig):
    print("ieig", ieig)
    spline_var = scipy.interpolate.RegularGridInterpolator(centers_flat, smearing_variations.values()[..., ieig], method="cubic", bounds_error = False, fill_value = None)
    smearing_variations_smooth.values()[..., ieig] = spline_var(eval_points)


#set overflow in eta to boundary values (but leave overflow in pt as 0)
smearing_variations_smooth.values(flow=True)[0, 1:-1] = smearing_variations_smooth.values()[0]
smearing_variations_smooth.values(flow=True)[-1, 1:-1] = smearing_variations_smooth.values()[-1]

print(smearing_variations_smooth)
print(smearing_variations_smooth.values().shape)


for ieig in range(neig):
    plt.figure()
    smearing_variations[..., ieig].plot()
    plt.savefig(f"smearing_variation_{ieig}.png")
    plt.close()

    plt.figure()
    smearing_variations_smooth[..., ieig].plot()
    plt.savefig(f"smearing_variations_smooth_{ieig}.png")
    plt.close()

with lz4.frame.open("smearing_variations_smooth.pkl.lz4", "wb") as fout:
    pickle.dump(smearing_variations_smooth, fout, protocol=pickle.HIGHEST_PROTOCOL)
