from wremnants import plot_tools,theory_tools
from utilities import boostHistHelpers as hh
from utilities.io_tools import input_tools, output_tools
import hist
import utilities.common
import matplotlib.pyplot as plt
import mplhep as hep
from matplotlib import cm
import numpy as np
import boost_histogram as bh
import argparse
import pathlib
import os
import pickle
import h5py
import narf.ioutils
import logging
import shutil
import copy
from utilities.differential import get_theoryAgnostic_axes
from wremnants.helicity_utils import axis_helicity_multidim

xlabels = {
    "pt" : r"p$_{T}^{\ell}$ (GeV)",
    "eta" : r"$\eta^{\ell}$",
    "unrolled" : r"(p$_{T}^{\ell}$, $\eta^{\ell}$) bin",
    "unrolled_gen" : r"($|\mathrm{y}^{Z}|$,p$_{T}^{Z}$) bin",
    "unrolled_gen_hel" : r"($|\mathrm{y}^{Z}|$,p$_{T}^{Z}$) bin",
    "ptVgen" : r"p$_{T}^{Z}$ (GeV)",
    "absYVgen" : r"$|\mathrm{y}^{Z}|$",
    "ptll" : r"p$_{\mathrm{T}}^{\ell\ell}$ (GeV)",
    "yll" : r"y$^{\ell\ell}$",
    "mll" : r"m$_{\ell\ell}$ (GeV)",
}

parser = argparse.ArgumentParser()
parser.add_argument("infile", help="Output file of the analysis stage, containing ND boost histograms")
parser.add_argument("--pdfs", type=str, nargs='+', help="List of histograms to plot", choices=theory_tools.pdfMap.keys(), required=True)
parser.add_argument("-c", "--channel", type=str, choices=["plus", "minus", "all"], default="all", help="Select channel to plot")
parser.add_argument("-p", "--outpath", type=str, default=os.path.expanduser("~/www/WMassAnalysis"), help="Base path for output")
parser.add_argument("-f", "--outfolder", type=str, default="test", help="Subfolder for output")
parser.add_argument("-r", "--rrange", type=float, nargs=2, default=[0.9, 1.1], help="y range for ratio plot")
parser.add_argument("-d", "--datasets", type=str, nargs="+", help="Dataset to plot", required=True)
parser.add_argument("--obs", type=str, nargs='+', choices=xlabels.keys(), help="Observable to plot", required=True)
parser.add_argument("--together", action='store_true', help="y range for ratio plot")
parser.add_argument("--baseName", type=str, help="Name of nominal hist")
parser.add_argument("--ymax", type=float, help="Max value for y axis (if not specified, range set automatically)")
args = parser.parse_args()

for pdf in args.pdfs:
    if pdf not in theory_tools.pdfMap:
        raise ValueError(f"pdf {pdf} is not a valid hist (not defined in theory_tools.pdfMap)")
    print(pdf)
# args.pdf.append("herapdf20ext")
band_hists = {}

for dataset in args.datasets:
    if "W" in args.datasets[0][0]:
        xlabels["ptVgen"] = xlabels["ptVgen"].replace("Z", "W")
        xlabels["absYVgen"] = xlabels["absYVgen"].replace("Z", "W")
        xlabels["unrolled_gen"] = xlabels["unrolled_gen"].replace("Z", "W")
        xlabels["unrolled_gen_hel"] = xlabels["unrolled_gen_hel"].replace("Z", "W")

    pdfInfo = theory_tools.pdfMap 
    pdfNames = [pdfInfo[pdf]["name"] for pdf in args.pdfs]
    histNames = pdfNames if not args.baseName else [f"{args.baseName}_{pdfName}" for pdfName in pdfNames]

    pdfHists = input_tools.read_all_and_scale(args.infile, args.datasets, histNames)
    axis_label = "pdfVar"

    uncType = [pdfInfo[pdf]["combine"] for pdf in args.pdfs]
    uncScale = [pdfInfo[pdf]["scale"] if "scale" in pdfInfo[pdf] else 1. for pdf in args.pdfs]
    uncHists = [[h[{axis_label : 0}], *theory_tools.hessianPdfUnc(h, axis_label, unc, scale)] for h,unc,scale in zip(pdfHists, uncType, uncScale)]
    names = [[pdfName+" $\pm1\sigma$", "", ""] for pdfName in pdfNames]
    cmap = cm.get_cmap("tab10")
    colors = [[cmap(i)]*3 for i in range(len(args.pdfs))]

    if "unrolled_gen_hel" in args.obs:
        print(dataset)
        moments = input_tools.read_all_and_scale(args.infile, [dataset], ["nominal_gen_helicity_moments_scale"])
        coeffs =  theory_tools.moments_to_helicities(moments[0].project('ptVgen','absYVgen','helicity','muRfact','muFfact'))
        
        moments_pdf = input_tools.read_all_and_scale(args.infile, [dataset], [f"helicity_{args.baseName}_{pdfName}" for pdfName in pdfNames])
        
        coeffs_pdf = []
        for moments in moments_pdf:
            coeffs_pdf.append(theory_tools.moments_to_helicities(moments.project('ptVgen','absYVgen','helicity',axis_label)))
        uncHists = [[h[{axis_label : 0}], *theory_tools.hessianPdfUnc(h, axis_label, unc, scale)] for h,unc,scale in zip(coeffs_pdf, uncType, uncScale)]

        coeffs_heraext = []
        moments_heraext = input_tools.read_all_and_scale(args.infile, [dataset], [f"helicity_{args.baseName}_pdfHERAPDF20ext"])
        for moments in moments_heraext:
            coeffs_heraext.append(theory_tools.moments_to_helicities(moments.project('ptVgen','absYVgen','helicity',axis_label)))
        
        uncType_hera = [pdfInfo["herapdf20ext"]["combine"]]
        uncScale_hera = [pdfInfo["herapdf20ext"]["scale"] if "scale" in pdfInfo["herapdf20ext"] else 1.]
        uncHists_hera = [[h[{axis_label : 0}], *theory_tools.hessianPdfUnc(h, axis_label, unc, scale)] for h,unc,scale in zip(coeffs_heraext, uncType_hera, uncScale_hera)]
        
        print(uncHists_hera)

        for ipdf,pdf in enumerate(args.pdfs):
            if "herapdf20" in pdf:
                nom = uncHists[ipdf][0]
                up = uncHists[ipdf][1]
                down = uncHists[ipdf][2]

                print(nom.shape)
                print(nom[0,0,0], up[0,0,0], down[0,0,0])
                print(uncHists_hera[0][0].values()[0,0,0],uncHists_hera[0][1].values()[0,0,0],uncHists_hera[0][2].values()[0,0,0])

                up[...] = nom.values() + np.sqrt(np.square(up.values() - nom.values())+np.square(uncHists_hera[0][1].values() - nom.values()))

                down[...] = nom.values() - np.sqrt(np.square(down.values() - nom.values())+np.square(uncHists_hera[0][2].values() - nom.values()))

                print(uncHists[ipdf][0].values()[0,0,0],up.values()[0,0,0],down.values()[0,0,0])

                uncHists[ipdf][1] = up
                uncHists[ipdf][2] = down

        # add alphaS
        # alphaNames = []
        # axis_label = "alphasVar"
        # for ipdf,pdf in enumerate(args.pdfs):
        #     if pdfInfo[pdf]["alphasRange"] == "001":
        #         alphaNames.append(f"helicity_{args.baseName}_{pdfNames[ipdf]}alphaS001")
        #     else:
        #         alphaNames.append(f"helicity_{args.baseName}_{pdfNames[ipdf]}alphaS002")
        #     alphaHists = input_tools.read_all_and_scale(args.infile, args.datasets, alphaNames)
        #     alphaHists_hel = []
        #     for alphaHist in alphaHists:
        #         alphaHists_hel.append(theory_tools.moments_to_angular_coeffs(alphaHist.project('helicity','absYVgen','ptVgen',axis_label)))
        #     uncHists[ipdf].extend([alphaHists_hel[ipdf][...,0],alphaHists_hel[ipdf][...,1]])
        #     names[ipdf].extend([pdfNames[ipdf]+"alpha $\pm1\sigma$",""])
        #     colors[ipdf].extend([[cmap(i)]*2 for i in range(len(args.pdfs),2*len(args.pdfs))][0])

        # add QCD scales
        uncHists.append([coeffs[{"muRfact" : 2.j, "muFfact" : 2.j}],coeffs[{"muRfact" : 0.5j, "muFfact" : 0.5j}],coeffs[{"muRfact" : 2.j, "muFfact" : 1.j}], coeffs[{"muRfact" : 0.5j, "muFfact" : 1.j}],coeffs[{"muRfact" : 1.j, "muFfact" : 2.j}],coeffs[{"muRfact" : 1.j, "muFfact" : 0.5j}]])
        names.append(["QCDscale_muRmuFUp","QCDscale_muRmuFDown","QCDscale_muRUp","QCDscale_muRDown","QCDscale_muFUp","QCDscale_muFDown"])
        colors.append([[cmap(i)]*6 for i in range(2+len(args.pdfs))][0])
        uncHists.append([coeffs[{'muRfact':1.j,'muFfact':1.j}]])
        names.append(["QCDscale_central"])
        colors.append([[cmap(i)]*1 for i in range(2*len(args.pdfs),2*len(args.pdfs)+1)][0])

    outdir = output_tools.make_plot_dir(args.outpath, args.outfolder)
    plot_names = copy.copy(args.pdfs)
    plot_names.append("QCD_scales")

    theoryAgnostic_axes, _ = get_theoryAgnostic_axes(ptV_flow=True, absYV_flow=True)
    axis_ptV = theoryAgnostic_axes[0]
    axis_yV = theoryAgnostic_axes[1]

    variations = np.zeros((len(axis_ptV.centers)+1,len(axis_yV.centers)+1,9))
    print(variations.shape)
    for ihel in coeffs.axes["helicity"].edges[:-1]:
        for obs in args.obs:
            all_hists = []
            all_colors = []
            all_names = []
            for name,color,labels,hists in zip(plot_names,colors, names, uncHists):
                # This is the reference
                if not "unrolled" in obs:
                    action = lambda x: x.project(obs)
                    hists1D = [action(x) for x in hists]
                else:
                    obs2unroll = ["ptVgen","absYVgen"] if "unrolled_gen" in obs else ["pt","eta"]
                    action = hh.unrolledHist
                    if not "hel" in obs:
                        hists1D = [action(x,obs2unroll,binwnorm=True) for x in hists]
                    else:
                        hists1D = [action(x[{'helicity': ihel*1.j}],obs2unroll,binwnorm=True) for x in hists]

                all_hists.extend(hists1D)
                all_colors.extend(color)
                all_names.extend(labels)
            
            # Add the nominal for reference
            
            hists1D = [all_hists[0], *all_hists]
            plot_cols = [colors[0][0], *all_colors]
            plot_labels = ["", *all_names]
            # print([h.values() for h in hists1D])

            lower = np.minimum.reduce([h.values() for h in hists1D])/np.abs(hists1D[0].values())
            upper = np.maximum.reduce([h.values() for h in hists1D])/np.abs(hists1D[0].values())
            
            symm = np.where(np.abs(lower-1.)>np.abs(upper-1.),lower,upper)
            if ihel==-1:
                symm = 0.5*np.ones_like(symm)
            elif ihel==4:
                symm = 2*np.ones_like(symm)

            fig = plot_tools.makePlotWithRatioToRef(hists1D, colors=plot_cols, labels=plot_labels, alpha=0.7,
                rrange=args.rrange, ylabel="$\sigma$/bin", xlabel=xlabels[obs], rlabel=f"x/{args.pdfs[0].upper()}", binwnorm=None, nlegcols=1, only_ratio=True)
            outfile = f"{name}Hist_{obs}_{dataset}_sigma{ihel}"
            ax2 = fig.axes
            # print(ihel,symm, (1-symm)+1)
            # ax1.fill_between(hists1D[0].axes[0].centers,np.minimum.reduce([h.values() for h in hists1D]),np.maximum.reduce([h.values() for h in hists1D]),color="grey",alpha=0.5, label="theory agnostic variation")
            ax2[0].fill_between(hists1D[0].axes[0].centers,symm, 2-symm,color="grey",alpha=0.3, label="theory agnostic variation",hatch="//")
            ax2[0].set_xticklabels([])
            ax2[0].set_xticks([])
            min_val = np.min(np.concatenate((symm,2-symm)))
            max_val = np.max(np.concatenate((symm,2-symm)))
            if not ihel ==-1:
                ax2[0].set_ylim(min_val,max_val)
            else:
                ax2[0].set_ylim(0,2)
            plot_tools.save_pdf_and_png(outdir, outfile)
            plot_tools.write_index_and_log(outdir, outfile)

            vars = np.ones((len(axis_ptV.centers)+1,len(axis_yV.centers)+1))
            # symm = symm-1
            print(symm)
            vars[:-1,:-1] = symm.reshape((len(axis_ptV.centers),len(axis_yV.centers)))
            # print(vars)
            variations[...,int(ihel)+1]=vars

    print(variations.shape)

    hvariations = hist.Hist(axis_ptV,axis_yV,axis_helicity_multidim, name=f"theorybands_{dataset}",data=variations)
    band_hists[dataset] = hvariations

outfile = "theoryband_variations.hdf5"
with h5py.File(outfile, 'w') as f:
    narf.ioutils.pickle_dump_h5py("theorybands", band_hists, f)

