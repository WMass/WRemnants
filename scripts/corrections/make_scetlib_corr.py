import numpy as np
import pickle
import lz4.frame
import logging
from wremnants import plot_tools, scetlib_corrections,theory_tools
from wremnants import boostHistHelpers as hh
import hist

logging.basicConfig(level=logging.INFO)

with lz4.frame.open("/home/k/kelong/work/WRemnants/w_z_gen_dists.pkl.lz4") as minnlof:
    minnlo = pickle.load(minnlof)

scetlibhZ_tot = scetlib_corrections.read_scetlib_hist("/home/k/kelong/work/Generators/scetlib-cms/prod/scetlib_run/helicityTest/Z/inclusive_Z.npz", add_nonsing=True, charge=0)

minnlohZ = minnlo["ZmumuPostVFP"]["output"]["nominal_gen"]*2001/minnlo["ZmumuPostVFP"]["weight_sum"]
scetlibh = hh.makeAbsHist(scetlibhZ_tot, "y")

corrh_unc  = scetlib_corrections.make_scetlib_minnlo_corr(minnlohZ, scetlibh)
corrh = hist.Hist(*corrh_unc.axes, name=corrh_unc.name, storage=hist.storage.Double(), data=corrh_unc.values(flow=True))
logging.info(f"Average correction is {corrh.sum()/np.ones_like(corrh).sum()}")

with lz4.frame.open("wremnants/data/N3LLCorrections/inclusive_pT_y_m.pkl.lz4", "wb") as f:
    pickle.dump({
            "Z" : {
                "inclusive_pT_y_m_ratio" : corrh,
                "inclusive_pT_y_m_scetlib" : scetlibh,
                "inclusive_pT_y_m_minnlo" : minnlohZ,
            },
            "W" : {
                "inclusive_pT_y_m_ratio" : corrh, 
                "inclusive_pT_y_m_scetlib" : scetlibh,
                "inclusive_pT_y_m_minnlo" : minnlohZ,
            },
        }, f, protocol = pickle.HIGHEST_PROTOCOL)
