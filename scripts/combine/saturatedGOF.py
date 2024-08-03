import ROOT
import scipy
import argparse
from utilities.io_tools import combinetf_input
from narf import ioutils
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("infile", type=str, help="combinetf output ROOT file")
args = parser.parse_args()

rtfile = ROOT.TFile(args.infile)
tree = rtfile.Get("fitresults")
tree.GetEntry(0)

fitresult_h5py = combinetf_input.get_fitresult(args.infile.replace(".root",".hdf5"))
meta = ioutils.pickle_load_h5py(fitresult_h5py["meta"])
nbins = sum([np.product([len(a) for a in info["axes"]]) for info in meta["channel_info"].values()])
ndf = nbins - tree.ndofpartial

print(f"nbins = {nbins}")
print(f"-loglikelihood_{{full}} = {tree.nllvalfull}")
print(f"-loglikelihood_{{saturated}} = {tree.satnllvalfull}")
print(f"2*(nllfull-nllsat)/ndf = {2*(tree.nllvalfull-tree.satnllvalfull)}/{ndf}")
print("chi2 probability =", scipy.stats.chi2.sf(2*(tree.nllvalfull-tree.satnllvalfull), ndf)*100, "%")