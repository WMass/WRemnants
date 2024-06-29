# plot goodness of fit from toys with data and/or pseudodata 
# use full nll as test statistic

import uproot
import os
import argparse
import mplhep as hep
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2
from utilities import common, logging
from utilities.io_tools import output_tools
from wremnants import plot_tools


import pdb

hep.style.use(hep.style.ROOT)

parser = common.plot_parser()
parser.add_argument("-i","--input", type=str, help="fitresult root file with toys")
parser.add_argument("-t","--testData", default=[], nargs="*" , help="fitresult root file with test data (data, pseudodata)")
parser.add_argument("-l","--testLabels", default=[], nargs="*" , help="labels for test data (data, pseudodata) for plotting")
parser.add_argument("-n","--nDoF", default=20,type=int, help="Number of degrees of freedom, for the chi2 function in the plot")
args = parser.parse_args()

logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)

outdir = output_tools.make_plot_dir(args.outpath, args.outfolder)

with uproot.open(f"{args.input}:fitresults") as utree:
    nll = utree['nllvalfull'].array(library="np")
    nll_saturated = utree['satnllvalfull'].array(library="np")
    # deviance
    toys = 2*(nll - nll_saturated)

data=[]
for tdata in args.testData:
    with uproot.open(f"{tdata}:fitresults") as utree:
        nll = utree['nllvalfull'].array(library="np")[0]
        nll_saturated = utree['satnllvalfull'].array(library="np")[0]
        data.append(2*(nll - nll_saturated))
    
data = np.array(data)

if args.testLabels ==[]:
    testlabels = ["" for _ in data]
else:
    testlabels = args.testLabels

# calculate p value as fraction of toys that have a higher likelihood than data
# pvalues = np.array([sum(toys>dt)/len(toys) for dt in data])
# p value from chi2
pvalues = np.array([(1-chi2.cdf(dt, args.nDoF)) for dt in data])

xlabel="$2 (\\mathrm{log}(\\mathcal{L}_\\mathrm{Full}) - \\mathrm{log}(\\mathcal{L}_\\mathrm{Saturated})) $"
colors = ["red", "blue", "green"]
saveas="png"
textsize = 16
labelsize = 12
nBins = 50

cm = mpl.colormaps["gist_rainbow"]

rangex = 0, max(max(np.append(toys,data)), args.nDoF)
rangex = 0, rangex[1]+(0.01*(rangex[1]-rangex[0]))

plt.clf()
fig = plt.figure()
fig.subplots_adjust(left=0.15, right=0.99, top=0.99, bottom=0.125)
ax = fig.add_subplot(111)

nEntries, bins, _ = ax.hist(toys, bins=nBins, range=rangex, color="yellow")
ax.step(bins, np.append(nEntries,nEntries[-1]), color="black", where="post", linewidth=0.5)

rangey = 0, max(nEntries)*1.2

for i, dt, l, p in zip(range(len(data)), data, testlabels, pvalues):
    ax.plot([dt,dt], rangey, color=cm(i/len(data)), label=f"{l} $p={round(p*100,1)}\%$")

norm = (rangex[1]-rangex[0])/nBins*len(toys)

x = np.linspace(*rangex, 100)
ax.plot(x, chi2.pdf(x, args.nDoF)*norm, 'k-', lw=1, label='chi2 pdf')

ax.set_ylabel("Number of toys", fontsize=textsize)

ax.set_xlabel(xlabel, fontsize=textsize)

ax.set_xlim(rangex)
ax.set_ylim(rangey)

plot_tools.addLegend(ax, ncols=2, text_size=25*args.scaleleg, loc="upper left")

plt.xticks(fontsize = labelsize)
plt.yticks(fontsize = labelsize)

scale = max(1, np.divide(*ax.get_figure().get_size_inches())*0.3)
hep.cms.label(ax=ax, lumi=float(f"{args.lumi:.3g}"), fontsize=20*args.scaleleg*scale, 
    label=args.cmsDecor, data=False)

histname = "hist_GoF"
plot_tools.save_pdf_and_png(outdir, histname)

if output_tools.is_eosuser_path(args.outpath) and args.eoscp:
    output_tools.copy_to_eos(outdir, args.outpath, args.outfolder)
