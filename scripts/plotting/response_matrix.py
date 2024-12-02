import hist
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

from utilities import boostHistHelpers as hh
from utilities import logging, parsing
from utilities.io_tools import output_tools
from wremnants import plot_tools
from wremnants.datasets.datagroups import Datagroups

parser = parsing.plot_parser()
parser.add_argument(
    "infile", help="Output file of the analysis stage, containing ND boost histogrdams"
)
parser.add_argument(
    "--procFilters",
    type=str,
    nargs="*",
    default="Zmumu",
    help="Filter to plot (default no filter, only specify if you want a subset",
)
parser.add_argument(
    "--axes",
    type=str,
    nargs="+",
    default=["pt-ptGen", "abs(eta)-absEtaGen"],
    help="Define for which axes the response matrix to be plotted",
)
parser.add_argument(
    "-n",
    "--baseName",
    type=str,
    help="Histogram base name in the file (e.g., 'nominal')",
    default="nominal",
)
parser.add_argument(
    "--histName",
    type=str,
    help="Histogram name in the file (e.g., 'nominal')",
    default="nominal",
)
parser.add_argument(
    "-c",
    "--channels",
    type=str,
    nargs="+",
    choices=["plus", "minus", "all"],
    default=["all"],
    help="Select channel to plot",
)

args = parser.parse_args()

logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)

outdir = output_tools.make_plot_dir(args.outpath, args.outfolder, eoscp=args.eoscp)

groups = Datagroups(
    args.infile,
    filterGroups=args.procFilters,
    excludeGroups=None if args.procFilters else ["QCD"],
)

groups.setGenAxes(
    sum_gen_axes=[]
)  # set gen axes empty to not integrate over when loading

if "Wmunu" in groups.groups:
    groups.copyGroup(
        "Wmunu", "Wmunu_qGen0", member_filter=lambda x: x.name.startswith("Wminus")
    )
    groups.copyGroup(
        "Wmunu", "Wmunu_qGen1", member_filter=lambda x: x.name.startswith("Wplus")
    )

    groups.deleteGroup("Wmunu")

if "Zmumu" in groups.groups:
    groups.groups["Zmumu"].deleteMembers(
        [m for m in groups.groups["Zmumu"].members if "BkgZmumu" in m.name]
    )


datasets = groups.getNames()
logger.info(f"Will plot datasets {datasets}")

groups.loadHistsForDatagroups(args.baseName, syst=args.histName, procsToRead=datasets)

datagroups = groups.getDatagroups()

translate_label = {
    "pt": r"$\mathrm{Reco}\ p_\mathrm{T}\ [\mathrm{GeV}]$",
    "ptGen": r"$\mathrm{Gen}\ p_\mathrm{T}\ [\mathrm{GeV}]$",
    "eta": r"$\mathrm{Reco}\ \eta$",
    "abs(eta)": r"$\mathrm{Reco}\ |\eta|$",
    "absEtaGen": r"$\mathrm{Gen}\ |\eta|$",
    "ptll": r"$\mathrm{Reco}\ p_\mathrm{T}(\ell\ell)\ [\mathrm{GeV}]$",
    "ptW": r"$\mathrm{Reco}\ p_\mathrm{T}(\ell\nu)\ [\mathrm{GeV}]$",
    "ptVGen": r"$\mathrm{Gen}\ p_\mathrm{T}(V)\ [\mathrm{GeV}]$",
    "yll": r"$\mathrm{Reco}\ Y(\mathrm{V})$",
    "abs(yll)": r"$\mathrm{Reco}\ |Y(\mathrm{V})|$",
    "absYVGen": r"$\mathrm{Gen}\ |Y(\mathrm{V})|$",
    "cosThetaStarll": r"$\mathrm{Reco}\ \cos{\theta^{\star}_{\ell\ell}}$",
    "phiStarll": r"$\mathrm{Reco}\ \phi^{\star}_{\ell\ell}$",
    "helicitySig": r"Helicity",
}


def get_purity(matrix, xbins, ybins):

    centers = xbins[:-1] + (xbins[1:] - xbins[:-1]) / 2
    edges = ybins

    values = []
    for iBin, center in enumerate(centers):
        # find out in which gen bin(s) we are
        ilow = np.where(edges == edges[center > edges][-1])[0][0]
        ihigh = np.where(edges == edges[center < edges][0])[0][0]

        # sum corresponding diagonal element(s)
        diag = matrix[iBin, ilow:ihigh].sum()

        # sum reco bins
        reco = matrix[iBin, :].sum()

        values.append(diag / reco)

    return np.array(values)


def get_stability(matrix, xbins, ybins):
    # stability is same computation as purity with inverted axes
    return get_purity(matrix.T, ybins, xbins)


def plot_resolution(
    histo,
    axes_reco,
    axis_gen,
    selections_global,
    selections_slices,
    suffix=None,
    normalize=False,
):
    # plot slices of gen bins in 1D reco space

    if isinstance(axes_reco, str):
        axes_reco = [axes_reco]
    for i, a in enumerate(axes_reco):
        if a.startswith("abs("):
            axes_reco[i] = a[4:-1]
            histo = hh.makeAbsHist(histo, a[4:-1], rename=False)

    if len(axes_reco) == 1:
        xlabel = translate_label[axes_reco[0]]
    else:
        xlabel = "-".join(
            [translate_label[a].replace(r"[\mathrm{GeV}]", "") for a in axes_reco]
        )
        xlabel = xlabel.replace(r"-$\mathrm{Reco}", "-$") + " bin"

    for sel, idx in selections_global:
        if sel is not None:
            if histo.axes[sel].size - 1 < idx:
                continue
            h2d = histo[{sel: idx}].project(axis_gen, *axes_reco)
        else:
            h2d = histo.project(axis_gen, *axes_reco)

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot()
        ax.set_xlabel(xlabel)
        if normalize:
            ylabel = "Frequency"
        else:
            ylabel = "Events"
            if xlabel.endswith("bin"):
                ylabel += " / bin"
            else:
                ylabel += " / unit"

        ax.set_ylabel(ylabel)

        xedges = None
        for sel2, idx2 in selections_slices:
            if (
                idx2 not in [hist.underflow, hist.overflow]
                and h2d.axes[sel2].size - 1 < idx2
            ):
                continue

            if isinstance(h2d.axes[sel2], hist.axis.Integer):
                label = int(h2d.axes[sel2].edges[idx2])
                if sel2 == "helicitySig":
                    if idx2 == 0:
                        label = r"$\sigma_{\mathrm{UL}}$"
                    else:
                        label = rf"$\sigma_{label}$"
            else:
                edges = h2d.axes[sel2].edges
                var2 = translate_label[sel2].replace(r"[\mathrm{GeV}]", "")
                if idx2 == hist.overflow:
                    label = f"{var2} > {edges[-1]}"
                elif idx2 + 1 < len(edges):
                    lo, hi = edges[idx2], edges[idx2 + 1]
                    label = f"{lo} < {var2} < {hi}"
            h1d = h2d[{sel2: idx2}]
            if len(axes_reco) > 1:
                h1d = hh.unrolledHist(h1d, binwnorm=None, obs=axes_reco)

            values = h1d.values()
            if len(axes_reco) == 1:
                values /= np.diff(h1d.axes[0].edges)

            if normalize:
                values /= abs(values).sum()

            ax.stairs(values, edges=h1d.axes[0].edges, label=label)

            if xedges is None:
                xedges = h1d.axes[0].edges

        ax.set_xlim([min(xedges), max(xedges)])

        y_min, y_max = ax.get_ylim()
        ax.set_ylim([min(0, y_min), y_max * 1.5])

        # Use scientific notation
        ax.ticklabel_format(style="sci", axis="y", scilimits=(-2, 2))

        # move scientific notation (e.g. 10^5) a bit to the left
        offset_text = ax.get_yaxis().get_offset_text()
        offset_text.set_position((-0.08, 1.02))

        if sel is not None:
            lo, hi = histo.axes[sel].edges[idx], histo.axes[sel].edges[idx + 1]
            var = translate_label[sel].replace(r"[\mathrm{GeV}]", "")
            if sel.startswith("abs") and lo == 0:
                title = f"{var} < {hi}"
            else:
                title = f"{lo} < {var} < {hi}"

            plt.text(
                0.06,
                0.94,
                title,
                horizontalalignment="left",
                verticalalignment="top",
                transform=ax.transAxes,
                fontsize=20,
            )

        plot_tools.add_cms_decor(
            ax, args.cmsDecor, data=False, lumi=None, loc=args.logoPos
        )
        plot_tools.addLegend(
            ax, ncols=args.legCols, loc=args.legPos, text_size=args.legSize
        )

        outfile = f"resolution_{g_name}_{'_'.join(axes_reco)}"

        if sel is not None:
            outfile += f"_{sel}{idx}"
        if suffix:
            outfile += f"_{suffix}"
        plot_tools.save_pdf_and_png(outdir, outfile)

        plot_tools.write_index_and_log(
            outdir,
            outfile,
            analysis_meta_info={args.infile: groups.getMetaInfo()},
            args=args,
        )


for g_name, group in datagroups.items():
    histo = group.hists[args.histName]

    for channel in args.channels:
        select = (
            {}
            if channel == "all"
            else {"charge": -1.0j if channel == "minus" else 1.0j}
        )

        # plot slices of resolution
        if all(x in histo.axes.name for x in ["pt", "ptGen", "absEtaGen"]):
            plot_resolution(
                histo,
                axes_reco="pt",
                axis_gen="ptGen",
                selections_global=(("absEtaGen", 0), ("absEtaGen", 17)),
                selections_slices=(("ptGen", 0), ("ptGen", 6), ("ptGen", 13)),
            )
        if all(x in histo.axes.name for x in ["ptGen", "eta", "absEtaGen"]):
            plot_resolution(
                histo,
                axes_reco="abs(eta)",
                axis_gen="absEtaGen",
                selections_global=(("ptGen", 0), ("ptGen", 13)),
                selections_slices=(
                    ("absEtaGen", 0),
                    ("absEtaGen", 8),
                    ("absEtaGen", 17),
                ),
            )

        if all(x in histo.axes.name for x in ["ptll", "ptVGen", "absYVGen"]):
            plot_resolution(
                histo,
                axes_reco="ptll",
                axis_gen="ptVGen",
                selections_global=(
                    ("absYVGen", 0),
                    ("absYVGen", 3),
                ),
                selections_slices=(
                    ("ptVGen", 0),
                    ("ptVGen", 8),
                    ("ptVGen", hist.overflow),
                ),
            )
        if all(x in histo.axes.name for x in ["ptll", "ptVGen", "absYVGen"]):
            plot_resolution(
                histo,
                axes_reco="abs(yll)",
                axis_gen="absYVGen",
                selections_global=(
                    ("ptVGen", 0),
                    ("ptVGen", 10),
                ),
                selections_slices=(
                    ("absYVGen", 0),
                    ("absYVGen", 2),
                    ("absYVGen", hist.overflow),
                ),
            )

        if all(x in histo.axes.name for x in ["cosThetaStarll", "helicitySig"]):
            plot_resolution(
                histo,
                axes_reco="cosThetaStarll",
                axis_gen="helicitySig",
                selections_global=((None, None),),
                selections_slices=([("helicitySig", i) for i in range(0, 9)]),
                normalize=True,
            )
        if all(x in histo.axes.name for x in ["phiStarll", "helicitySig"]):
            plot_resolution(
                histo,
                axes_reco="phiStarll",
                axis_gen="helicitySig",
                selections_global=((None, None),),
                selections_slices=([("helicitySig", i) for i in range(0, 9)]),
                normalize=True,
            )

        if all(
            x in histo.axes.name for x in ["cosThetaStarll", "phiStarll", "helicitySig"]
        ):
            plot_resolution(
                histo,
                axes_reco=["cosThetaStarll", "phiStarll"],
                axis_gen="helicitySig",
                selections_global=((None, None),),
                selections_slices=([("helicitySig", i) for i in range(0, 9)]),
                normalize=True,
            )
            for i in range(0, 9):
                plot_resolution(
                    histo,
                    axes_reco=["cosThetaStarll", "phiStarll"],
                    axis_gen="helicitySig",
                    selections_global=((None, None),),
                    selections_slices=(("helicitySig", i),),
                    suffix=f"HelicityIdx{i}",
                    normalize=False,
                )
                plot_resolution(
                    histo,
                    axes_reco="ptll",
                    axis_gen="helicitySig",
                    selections_global=((None, None),),
                    selections_slices=(("helicitySig", i),),
                    suffix=f"HelicityIdx{i}",
                    normalize=False,
                )
                plot_resolution(
                    histo,
                    axes_reco="abs(yll)",
                    axis_gen="helicitySig",
                    selections_global=((None, None),),
                    selections_slices=(("helicitySig", i),),
                    suffix=f"HelicityIdx{i}",
                    normalize=False,
                )

        for axes_string in args.axes:
            axes = axes_string.split("-")

            if (
                (groups.mode[0] == "w" or "wlike" in groups.mode) and axes[1] == "ptGen"
            ) or ((groups.mode[0] == "z") and axes[1] == "ptVGen"):
                genFlow = True
            else:
                genFlow = False

            if axes[0].startswith("abs("):
                # mirror axis at half
                hist2d = histo[select].project(axes[0][4:-1], *axes[1:])
                nbins = len(hist2d.axes.edges[0]) - 1
                values = (
                    hist2d.values(flow=genFlow)[: int(nbins / 2)][::-1]
                    + hist2d.values(flow=genFlow)[int(nbins / 2) :]
                )
                xbins = hist2d.axes[0].edges[int(nbins / 2) :]
            else:
                hist2d = histo[select].project(*axes)
                if len(hist2d.axes[0]) == len(hist2d.axes[1]) + 1:
                    hist2d = hh.disableFlow(hist2d, hist2d.axes[0].name)
                values = hist2d.values(flow=genFlow)
                xbins = hist2d.axes[0].edges

            if np.sum(values < 0):
                logger.warning(
                    f"Found {np.sum(values<0)} negative entries {values[values<0]}. Setting to zero"
                )
                values[values < 0] = 0

            ybins = hist2d.axes[1].edges
            if genFlow and hist2d.axes[1].traits.underflow:
                ybins = np.array([xbins[0], *ybins])
            if genFlow and hist2d.axes[1].traits.overflow:
                ybins = np.array([*ybins, xbins[-1]])

            outname = (
                g_name
                + "_"
                + "_".join([a.replace("(", "").replace(")", "") for a in axes])
            )

            # plot purity
            fig = plt.figure(figsize=(8, 4))
            ax = fig.add_subplot()
            ax.set_xlabel(translate_label[axes[0]])
            ax.set_ylabel("Purity")

            purity = get_purity(values, xbins, ybins)

            hep.histplot(purity, xbins, color="blue")

            range_y = max(purity) - min(purity)
            min_y = min(purity) - range_y * 0.1
            max_y = max(purity) + range_y * 0.1

            ax.set_xlim([min(xbins), max(xbins)])
            ax.set_ylim([min_y, max_y])

            plot_tools.add_cms_decor(
                ax, args.cmsDecor, data=False, lumi=None, loc=args.logoPos
            )

            outfile = "purity_" + outname
            plot_tools.save_pdf_and_png(outdir, outfile)

            # plot stability
            fig = plt.figure(figsize=(8, 4))
            ax = fig.add_subplot()
            ax.set_xlabel(translate_label[axes[1]])
            ax.set_ylabel("Stability")

            stability = get_stability(values, xbins, ybins)

            hep.histplot(stability, ybins, color="red")

            range_y = max(stability) - min(stability)
            min_y = min(stability) - range_y * 0.1
            max_y = max(stability) + range_y * 0.1

            ax.set_xlim([min(ybins), max(ybins)])
            ax.set_ylim([min_y, max_y])

            plot_tools.add_cms_decor(
                ax, args.cmsDecor, data=False, lumi=None, loc=args.logoPos
            )

            outfile = "stability_" + outname
            plot_tools.save_pdf_and_png(outdir, outfile)

            # plot response matrix
            fig = plt.figure()  # figsize=(8*width,8))
            ax = fig.add_subplot()

            ax.set_xlabel(translate_label[axes[0]])
            ax.set_ylabel(translate_label[axes[1]])

            hep.hist2dplot(
                values, xbins=xbins, ybins=ybins, cmin=0
            )  # , labels=(xlabels,ylabels))

            # calculate condition number
            nans = np.isnan(values)
            nancount = nans.sum()
            if nancount:
                logger.warning(
                    f"Found {nancount} NaNs in positions {np.argwhere(nans)}. Setting to zero"
                )
                values[nans] = 0

            cond = np.linalg.cond(values)
            logger.info(f"Condition number: {cond}")
            plt.text(
                0.2,
                0.94,
                round(cond, 1),
                horizontalalignment="right",
                verticalalignment="top",
                transform=ax.transAxes,
                color="white",
            )

            # ax.set_xticks(np.arange(len(xlabels))+0.5)
            # ax.set_yticks(np.arange(len(xlabels))+0.5)
            # ax.set_xticklabels(xlabels, rotation = 90)
            # ax.set_yticklabels(xlabels)

            hep.cms.label(ax=ax, fontsize=20, label=args.cmsDecor, data=False)

            outfile = "responce_matrix_" + outname

            outfile += f"_{channel}" if channel != "all" else ""

            plot_tools.save_pdf_and_png(outdir, outfile)

            plot_tools.write_index_and_log(
                outdir,
                outfile,
                #     yield_tables={"Values" : cov_mat}, nround=2 if "correlation" in matrix else 10,
                analysis_meta_info={args.infile: groups.getMetaInfo()},
                args=args,
            )

if output_tools.is_eosuser_path(args.outpath) and args.eoscp:
    output_tools.copy_to_eos(outdir, args.outpath, args.outfolder)
