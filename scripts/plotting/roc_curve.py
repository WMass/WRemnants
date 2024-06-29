from utilities import logging, common
from utilities.io_tools import output_tools
from utilities.styles import styles
from utilities import boostHistHelpers as hh

from wremnants import plot_tools
from wremnants import histselections as sel
from wremnants.datasets.datagroups import Datagroups

import hist
import numpy as np
import matplotlib as mpl
# from matplotlib import pyplot as plt
# import mplhep as hep
# import uncertainties as unc
# import uncertainties.unumpy as unp
# from scipy import stats

import pdb

def distance_point_to_line(p1, p2, points):
    # p1, p2 are each an array of one point, points can be an array of multiple points
    return np.cross(p2-p1,p1-points)/np.linalg.norm(p2-p1)

def select_max_distance_points(p1, p2, points, pnew="both"):
    dsts = distance_point_to_line(p1, p2, points)
    if np.max(dsts) <= 0:
        if pnew=="both":
            return [p1, p2]
        elif pnew=="left":
            return [p1]
        elif pnew=="right":
            return [p2]
    idx = np.argmax(dsts)
    pnew = points[idx]
    return [*select_max_distance_points(p1, pnew, points, pnew="right"), *select_max_distance_points(pnew, p2, points, pnew="left")]


if __name__ == '__main__':
    parser = common.plot_parser()
    parser.add_argument("infile", help="Output file of the analysis stage, containing ND boost histograms")
    parser.add_argument("-n", "--baseName", type=str, help="Histogram name in the file (e.g., 'nominal')", default="nominal")
    parser.add_argument("--logy", action='store_true', help="Enable log scale for y axis")

    args = parser.parse_args()
    logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)

    outdir = output_tools.make_plot_dir(args.outpath, args.outfolder, eoscp=args.eoscp)

    colors = mpl.colormaps["tab10"]

    groups = Datagroups(args.infile, excludeGroups=None, extendedABCD=True, integrateHighMT=True)

    groups.loadHistsForDatagroups("mt_met", syst="", procsToRead=["QCD", "Wmunu"], applySelection=False)


    groups.loadHistsForDatagroups("iso_dxy", syst="", procsToRead=["QCD", "Wmunu"], applySelection=False)
    histInfo = groups.getDatagroups()

    hSig = histInfo["Wmunu"].hists["mt_met"]
    hBkg = histInfo["QCD"].hists["mt_met"]
    hSig_mt = hSig.project("mt")
    hBkg_mt = hBkg.project("mt")
    hSig_met = hSig.project("met")
    hBkg_met = hBkg.project("met")
    hSig_mt_met = hSig.project("mt","met")
    hBkg_mt_met = hBkg.project("mt","met")

    hSig = histInfo["QCD"].hists["iso_dxy"]
    hBkg = histInfo["Wmunu"].hists["iso_dxy"]
    hSig_iso = hSig.project("iso")
    hBkg_iso = hBkg.project("iso")
    hSig_dxy = hSig.project("dxy")
    hBkg_dxy = hBkg.project("dxy")
    hSig_iso_dxy = hSig.project("iso","dxy")
    hBkg_iso_dxy = hBkg.project("iso","dxy")

    ylim = (0.01, 1) if args.logy else (0, 1)

    fig, ax1 = plot_tools.figure(None, xlabel="Prompt efficiency", ylabel="Nonprompt efficiency", cms_label=args.cmsDecor,
                                grid=True, automatic_scale=False, width_scale=1.2, xlim=(0, 1), ylim=ylim, logy=args.logy)    

    if not args.logy:
        ax1.plot([0,1],[0,1], linestyle="--", marker="none", color="black")

    wps = {
        "mt": [40,],
        "met": [22.5,],
        "iso": [0.15, 0.3],
        "dxy": [0.05, 0.01]
    }

    for i, (label, linestyle, hSig, hBkg) in enumerate((
        ("mt", "-", hSig_mt, hBkg_mt),
        ("met", "-", hSig_met, hBkg_met),
        ("mt-met", "--", hSig_mt_met, hBkg_mt_met),
        ("iso", "-", hSig_iso, hBkg_iso),
        ("dxy", "-", hSig_dxy, hBkg_dxy),
        ("iso-dxy", "--", hSig_iso_dxy, hBkg_iso_dxy),
    )):
        logger.info(f"Now at {label}")

        # all upper cuts, so we go from right to left/ or equivalent: switch fpr and tpr 
        ls = label.split("-")

        # normalize to unity
        hSig = hh.normalize(hSig, scale=1, createNew=False)
        hBkg = hh.normalize(hBkg, scale=1, createNew=False)

        if len(ls)==1:
            tpr = np.cumsum(hSig.values(flow=True))
            fpr = np.cumsum(hBkg.values(flow=True))
        else:
            # calculate all possible tpr and fpr
            logger.info("calculate all possible tpr and fpr for 2D histogram")
            size = hSig.axes.extent
            pts = np.empty((*size, 2))
            for n in range(size[0]):
                for m in range(size[1]):
                    pts[n,m,1] = hSig[slice(0,n+1,hist.sum), slice(0,m+1,hist.sum)]
                    pts[n,m,0] = hBkg[slice(0,n+1,hist.sum), slice(0,m+1,hist.sum)]

            logger.info("Select best tpr and fpr on 1D line")
            # select always the point that has the largest perpendicular distance to a line in direction of lower right tpr=100%, fpr=0% i.e. (1,0) 
            # 1) start with two points (0,0) and (1,1)
            # 2) follow prescription: make line between points and select third point 
            # 3) make two more lines from the selected point 
            # 4) for each line go to 2)
            # 5) finish once there is no more point in direction of (1,0)
            p1 = np.array([0,0])
            p2 = np.array([1,1])

            pts = np.reshape(pts, (np.product(size),2))

            # as a cross check- plot all tpr,fpr points
            # ax1.plot(pts[:,0], pts[:,1], linestyle="none", marker=".", color=colors(i))

            pts_sel = np.array(select_max_distance_points(p1, p2, pts))

            tpr = pts_sel[:,1]
            fpr = pts_sel[:,0]

        # plot working points
        idxs = [hSig.axes[label].index(w) for w in wps.get(label,[])]
        wpx = [fpr[i] for i in idxs]
        wpy = [tpr[i] for i in idxs]
        ax1.plot(wpx, wpy, linestyle="none", marker="o", color=colors(i))

        for w in wps.get(label,[]):
            idx = hSig.axes[label].index(w)
            logger.info(f"Working point {w} at bin {idx} with fpr={fpr[idx]} and tpr={tpr[idx] }")

        roc_auc = np.trapz(tpr, fpr)

        ax1.plot(fpr, tpr, linestyle=linestyle, marker="none", color=colors(i), 
            label=" & ".join([styles.xlabels.get(l,l).replace("(GeV)","") for l in ls])+f" (auc={round(roc_auc,2)})")

    plot_tools.addLegend(ax1, ncols=2, text_size=12, loc="upper left")
    plot_tools.fix_axes(ax1, logy=args.logy)

    outfile="roc"
    if args.postfix:
        outfile += f"_{args.postfix}"

    plot_tools.save_pdf_and_png(outdir, outfile)
    plot_tools.write_index_and_log(outdir, outfile, 
        analysis_meta_info={"AnalysisOutput" : groups.getMetaInfo()},
        args=args,
    )

    if output_tools.is_eosuser_path(args.outpath) and args.eoscp:
        output_tools.copy_to_eos(outdir, args.outpath, args.outfolder)
