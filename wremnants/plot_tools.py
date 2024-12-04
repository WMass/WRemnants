import datetime
import json
import math
import pathlib
import shutil
import socket
import sys
import textwrap

import hist
import matplotlib as mpl
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from matplotlib import patches, ticker
from matplotlib.legend_handler import HandlerLine2D, HandlerPatch
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

# for setting number of decimal places on tick labels
from matplotlib.ticker import StrMethodFormatter

import narf
from utilities import boostHistHelpers as hh
from utilities import common, logging

hep.style.use(hep.style.ROOT)

logger = logging.child_logger(__name__)


def cfgFigure(
    href, xlim=None, bin_density=300, width_scale=1, automatic_scale=True, base_size=8
):
    hax = href.axes[0]
    if not xlim:
        xlim = [hax.edges[0], hax.edges[-1]]
    if not automatic_scale:
        return plt.figure(figsize=(width_scale * base_size, base_size)), xlim
    xlim_range = float(xlim[1] - xlim[0])
    original_xrange = float(hax.edges[-1] - hax.edges[0])
    raw_width = (hax.size / float(bin_density)) * (xlim_range / original_xrange)
    width = math.ceil(raw_width)
    return (
        plt.figure(figsize=(width_scale * base_size * width, width_scale * base_size)),
        xlim,
    )


def figure(
    href,
    xlabel,
    ylabel,
    ylim=None,
    xlim=None,
    grid=False,
    plot_title=None,
    title_padding=0,
    bin_density=300,
    logy=False,
    logx=False,
    width_scale=1,
    height=8,
    automatic_scale=True,
):
    if isinstance(href, hist.Hist):
        fig, xlim = cfgFigure(href, xlim, bin_density, width_scale, automatic_scale)
    else:
        if automatic_scale:
            raw_width = len(href) / float(bin_density)
            width = math.ceil(raw_width)
        else:
            width = 1
        fig = plt.figure(figsize=(width_scale * height * width, height))

    ax1 = fig.add_subplot()

    ax1.set_xlabel(xlabel)
    if ylabel is not None:
        ax1.set_ylabel(ylabel, labelpad=40, va="center", multialignment="center")
    ax1.set_xlim(xlim)

    if ylim is not None:
        ax1.set_ylim(ylim)
    else:
        ax1.autoscale(axis="y")

    if logy:
        ax1.set_yscale("log")
    if logx:
        ax1.set_xscale("log")

    if grid:
        ax1.grid(which="both")
    if plot_title:
        ax1.set_title(plot_title, pad=title_padding)
    return fig, ax1


def figureWithRatio(
    href,
    xlabel,
    ylabel,
    ylim,
    rlabel,
    rrange,
    xlim=None,
    grid_on_main_plot=False,
    grid_on_ratio_plot=False,
    plot_title=None,
    title_padding=0,
    x_ticks_ndp=None,
    bin_density=300,
    logy=False,
    logx=False,
    width_scale=1,
    automatic_scale=True,
    base_size=8,
    only_ratio=False,
    subplotsizes=[4, 2],
):
    fig, xlim = cfgFigure(
        href, xlim, bin_density, width_scale, automatic_scale, base_size
    )

    ratio_axes = []

    if len(subplotsizes) == 2 and len(rrange) == 2 and isinstance(rlabel, str):
        rrange = [rrange]
        rlabel = [rlabel]

    if not only_ratio:
        ax1 = fig.add_subplot(sum(subplotsizes), 1, (1, subplotsizes[0]))
        ax1.set_xlabel(" ")
        ax1.set_ylabel(ylabel, labelpad=40, va="center", multialignment="center")
        ax1.set_xlim(xlim)

        if ylim:
            ax1.set_ylim(ylim)
        else:
            ax1.autoscale(axis="y")

        if logy:
            ax1.set_yscale("log")
        if grid_on_main_plot:
            ax1.grid(which="both")
        if plot_title:
            ax1.set_title(plot_title, pad=title_padding)

    for i, (ax, rr, rl) in enumerate(zip(subplotsizes[1:], rrange, rlabel)):
        xax = fig.add_subplot(
            sum(subplotsizes),
            1,
            (sum(subplotsizes[: i + 1]) + 1, sum(subplotsizes[: i + 2])),
        )
        xax.set_xlabel(" ")
        xax.set_ylabel(rl, labelpad=40, va="center", multialignment="center")
        xax.set_ylim(rr)
        ratio_axes.append(xax)

    for ax in ratio_axes:
        ax.set_xlim(xlim)

        if x_ticks_ndp:
            ax.xaxis.set_major_formatter(
                StrMethodFormatter("{x:." + str(x_ticks_ndp) + "f}")
            )
        if logx:
            ax.set_xscale("log")

        if grid_on_ratio_plot:
            ax.grid(which="both")

    ratio_axes[-1].set_xlabel(xlabel)

    if logx and not only_ratio:
        ax1.set_xscale("log")

    if not only_ratio:
        return fig, ax1, ratio_axes
    else:
        return fig, xax


class StackedLineHandler:
    def legend_artist(
        self, legend, orig_handle, fontsize, handlebox, linewidth_scale=1.0
    ):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        line1 = Line2D(
            [x0, x0 + width],
            [y0 + height * 0.75, y0 + height * 0.75],
            color=orig_handle.get_color(),
            lw=linewidth_scale * orig_handle.get_linewidth(),
            linestyle="-",
        )
        line2 = Line2D(
            [x0, x0 + width],
            [y0 + height * 0.25, y0 + height * 0.25],
            color=orig_handle.get_color(),
            lw=linewidth_scale * orig_handle.get_linewidth(),
            linestyle="--",
        )
        handlebox.add_artist(line1)
        handlebox.add_artist(line2)
        return [line1, line2]


class StackFilledHandler:
    def legend_artist(
        self, legend, orig_handle, fontsize, handlebox, linewidth_scale=1.0
    ):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        line0 = Line2D(
            [x0, x0 + width],
            [y0 + height * 0.5, y0 + height * 0.5],
            color=orig_handle.get_edgecolor(),
            lw=linewidth_scale * orig_handle.get_linewidth(),
            linestyle="-",
        )
        line1 = Line2D(
            [x0, x0 + width],
            [y0 + height, y0 + height],
            color=orig_handle.get_edgecolor(),
            lw=linewidth_scale * orig_handle.get_linewidth(),
            linestyle=orig_handle.get_linestyle(),
        )
        line2 = Line2D(
            [x0, x0 + width],
            [y0, y0],
            color=orig_handle.get_edgecolor(),
            lw=linewidth_scale * orig_handle.get_linewidth(),
            linestyle=orig_handle.get_linestyle(),
        )
        # Create the filled area between the lines using a polygon
        fill_coords = [
            [x0, y0],
            [x0 + width, y0],
            [x0 + width, y0 + height],
            [x0, y0 + height],
        ]
        fill = Polygon(fill_coords, color=orig_handle.get_facecolor(), alpha=0.3)

        handlebox.add_artist(fill)
        handlebox.add_artist(line0)
        handlebox.add_artist(line1)
        handlebox.add_artist(line2)
        return [line0, line1, line2, fill]


class BandFilledHandler:
    def legend_artist(
        self, legend, orig_handle, fontsize, handlebox, linewidth_scale=1.0
    ):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        line1 = Line2D(
            [x0, x0 + width],
            [y0 + height, y0 + height],
            color=orig_handle.get_edgecolor(),
            lw=linewidth_scale * orig_handle.get_linewidth(),
            linestyle=orig_handle.get_linestyle(),
        )
        line2 = Line2D(
            [x0, x0 + width],
            [y0, y0],
            color=orig_handle.get_edgecolor(),
            lw=linewidth_scale * orig_handle.get_linewidth(),
            linestyle=orig_handle.get_linestyle(),
        )
        # Create the filled area between the lines using a polygon
        fill_coords = [
            [x0, y0],
            [x0 + width, y0],
            [x0 + width, y0 + height],
            [x0, y0 + height],
        ]
        fill = Polygon(fill_coords, color=orig_handle.get_facecolor(), alpha=0.3)

        handlebox.add_artist(fill)
        handlebox.add_artist(line1)
        handlebox.add_artist(line2)
        return [line1, line2, fill]


class DoubleBandHandler(HandlerPatch):
    def create_artists(
        self,
        legend,
        orig_handle,
        xdescent,
        ydescent,
        width,
        height,
        fontsize,
        trans,
        linewidth_scale=1.0,
    ):
        # Create the outer polygon
        polygon_outer = Polygon(
            [
                [xdescent, ydescent],
                [xdescent + width, ydescent],
                [xdescent + width, ydescent + height],
                [xdescent, ydescent + height],
            ],
            color=orig_handle.get_facecolor(),
            alpha=orig_handle.get_alpha(),
        )
        # Create the inner polygon (narrower)
        margin = 0.25 * width
        polygon_inner = Polygon(
            [
                [xdescent + margin, ydescent],
                [xdescent + width - margin, ydescent],
                [xdescent + width - margin, ydescent + height],
                [xdescent + margin, ydescent + height],
            ],
            color=orig_handle.get_facecolor(),
            alpha=orig_handle.get_alpha(),
        )
        line = Line2D(
            [xdescent + width / 2, xdescent + width / 2],
            [ydescent, ydescent + height],
            color="black",
            lw=linewidth_scale * orig_handle.get_linewidth(),
            linestyle=orig_handle.get_linestyle(),
        )

        # Set the transformations
        polygon_outer.set_transform(trans)
        polygon_inner.set_transform(trans)
        line.set_transform(trans)

        return [polygon_outer, polygon_inner, line]


class TripleBandHandler(HandlerPatch):
    def create_artists(
        self,
        legend,
        orig_handle,
        xdescent,
        ydescent,
        width,
        height,
        fontsize,
        trans,
        linewidth_scale=1.0,
    ):
        # Create the outer polygon
        polygon_outer = Polygon(
            [
                [xdescent, ydescent],
                [xdescent + width, ydescent],
                [xdescent + width, ydescent + height],
                [xdescent, ydescent + height],
            ],
            color=orig_handle.get_facecolor(),
            alpha=orig_handle.get_alpha(),
        )
        # Create the inner polygon (narrower)
        margin = 0.15 * width
        polygon_inner = Polygon(
            [
                [xdescent + margin, ydescent],
                [xdescent + width - margin, ydescent],
                [xdescent + width - margin, ydescent + height],
                [xdescent + margin, ydescent + height],
            ],
            color=orig_handle.get_facecolor(),
            alpha=orig_handle.get_alpha(),
        )
        # Create the inner polygon (narrower)
        margin = 0.3 * width
        polygon_inner2 = Polygon(
            [
                [xdescent + margin, ydescent],
                [xdescent + width - margin, ydescent],
                [xdescent + width - margin, ydescent + height],
                [xdescent + margin, ydescent + height],
            ],
            color=orig_handle.get_facecolor(),
            alpha=orig_handle.get_alpha(),
        )
        line = Line2D(
            [xdescent + width / 2, xdescent + width / 2],
            [ydescent, ydescent + height],
            color="black",
            lw=linewidth_scale * orig_handle.get_linewidth(),
            linestyle=orig_handle.get_linestyle(),
        )

        # Set the transformations
        polygon_outer.set_transform(trans)
        polygon_inner.set_transform(trans)
        polygon_inner2.set_transform(trans)
        line.set_transform(trans)

        return [polygon_outer, polygon_inner, polygon_inner2, line]


def get_custom_handler_map(keys):
    if len(keys) == 0:
        return None
    handler_map = {}
    for key in keys:
        if key == "stacked":
            handler_map[Line2D] = StackedLineHandler()
        elif key == "stackfilled":
            handler_map[Polygon] = StackFilledHandler()
        elif key == "bandfilled":
            handler_map[Polygon] = BandFilledHandler()
        elif key == "verticleline":
            handler_map[Line2D] = HandlerLine2D(update_func=update_prop)
        elif key == "doubleband":
            handler_map[Polygon] = DoubleBandHandler()
        elif key == "tripleband":
            handler_map[Polygon] = TripleBandHandler()
    return handler_map


# https://stackoverflow.com/questions/53122592/legend-with-vertical-line
def update_prop(handle, orig):
    handle.update_from(orig)
    x, y = handle.get_data()
    handle.set_data([np.mean(x)] * 2, [0, 2 * y[0]])


def padding(ncols, labels, handles, loc="auto"):
    if len(labels) % ncols:
        # if not all legend places are filled, add empty legend entries towards the center of the figure
        rest = ncols - len(labels) % ncols
        nrows = int(np.ceil(len(labels) / ncols))
        for i in range(rest):
            if loc == "upper right":
                idx = (i + 1) * nrows - 1
            elif loc == "lower right":
                idx = i * nrows
            elif loc == "upper left":
                idx = len(labels) - i * nrows
            elif loc == "lower left":
                idx = len(labels) - (i + 1) * nrows + 1
            else:
                raise ValueError(f"Invalid option {loc} for legend padding location")
            handles.insert(idx, patches.Patch(color="none", label=" "))
            labels.insert(idx, " ")
    return labels, handles


def addLegend(
    ax,
    ncols=2,
    extra_text=None,
    extra_text_loc=None,
    text_size=None,
    loc="upper right",
    bbox_to_anchor=None,
    extra_handles=[],
    extra_labels=[],
    extra_entries_first=True,
    custom_handlers=[],
    markerfirst=True,
    reverse=True,
    labelcolor=None,
    padding_loc="auto",
):
    handles, labels = ax.get_legend_handles_labels()
    if extra_entries_first:
        handles.extend(extra_handles)
        labels.extend(extra_labels)
    else:
        handles = [*extra_handles, *handles]
        labels = [*extra_labels, *labels]

    if len(labels) == 0:
        return

    if padding_loc == "auto":
        legend_to_padding_loc = {
            "upper right": "lower left",
            "1": "lower left",
            "upper left": "lower right",
            "2": "lower right",
            "lower left": "upper right",
            "3": "upper right",
            "lower right": "upper left",
            "4": "upper left",
        }
        padding_loc = legend_to_padding_loc.get(str(loc), "lower right")

    labels, handles = padding(ncols, labels, handles, padding_loc)

    text_size = get_textsize(ax, text_size)
    handler_map = get_custom_handler_map(custom_handlers)
    leg = ax.legend(
        handles=handles,
        labels=labels,
        prop={"size": text_size},
        ncol=ncols,
        loc=loc,
        handler_map=handler_map,
        reverse=reverse,
        bbox_to_anchor=bbox_to_anchor,
        markerfirst=markerfirst,
        labelcolor=labelcolor,
    )

    if extra_text is not None:
        if extra_text_loc is None:
            # Add text to the left of the legend
            # Get the bounding box of the legend
            bbox = leg.get_window_extent()

            # Convert the bbox to display coordinates (relative to the figure)
            bbox_transform = plt.gcf().transFigure.inverted()
            bbox_disp = bbox_transform.transform(bbox)

            # Adjust the x position by moving it to the left
            extra_text_loc = bbox_disp[0, 0] - 0.25, bbox_disp[1, 1] - 0.01

            transform = plt.gcf().transFigure
        else:
            transform = None

        wrap_text(
            extra_text,
            ax,
            *extra_text_loc,
            text_size=text_size,
            ha="left",
            va="top",
            transform=transform,
        )


def get_textsize(ax, text_size):
    if text_size == "large" or text_size is None:
        # legend size same as axis label size
        return ax.yaxis.label.get_size()
    elif text_size == "small":
        # legend size same as axis ticklabel size (numbers)
        return ax.yaxis.get_ticklabels()[0].get_fontsize()
    elif text_size == "verysmall":
        # legend size same as axis ticklabel size (numbers)
        return ax.yaxis.get_ticklabels()[0].get_fontsize() * 0.6
    else:
        return int(text_size)


def wrap_text(
    text,
    ax,
    lower_x,
    y,
    upper_x=None,
    text_size=None,
    transform=None,
    ha=None,
    va="center",
):
    # wrap text within lower_x and upper_x,
    #  if text is already given as pieces in a list, use these pieces,
    #  otherwise calculate the pieces automatically
    text_size = get_textsize(ax, text_size)

    if isinstance(text, str):
        # Get the width of the text in data coordinates
        bbox = ax.get_window_extent().transformed(ax.transData.inverted())
        width_data = upper_x - lower_x
        width_display = bbox.width * (
            width_data / (ax.get_xlim()[1] - ax.get_xlim()[0])
        )
        # Estimate the number of characters that fit in this width
        # This is an approximation and may need adjustment
        char_width = text_size * 0.2  # Approximate width of a character in inches
        max_chars = int(width_display / char_width)
        wrapped_text = "\n".join(textwrap.wrap(text, width=max_chars))
    else:
        wrapped_text = "\n".join(text)

    if ha is not None:
        x = lower_x
    elif upper_x is not None:
        x = (lower_x + upper_x) / 2
        ha = "center"
    else:
        x = lower_x
        ha = "left"
    ax.text(
        x,
        y,
        wrapped_text,
        ha=ha,
        va=va,
        transform=transform if transform is not None else ax.transAxes,
        fontsize=text_size,
        wrap=True,
    )


def add_cms_decor(
    ax, label=None, lumi=None, loc=2, data=True, text_size=None, no_energy=False
):
    text_size = get_textsize(ax, text_size)
    if no_energy:
        hep.cms.text(ax=ax, text=label, loc=loc, fontsize=text_size)
    else:
        hep.cms.label(
            ax=ax,
            lumi=lumi,
            lumi_format="{0:.3g}",
            fontsize=text_size,
            label=label,
            data=data,
            loc=loc,
        )


def makeStackPlotWithRatio(
    histInfo,
    stackedProcs,
    histName="nominal",
    unstacked=None,
    fitresult=None,
    prefit=False,
    xlabel="",
    ylabel=None,
    rlabel="Data/Pred.",
    rrange=[0.9, 1.1],
    ylim=None,
    xlim=None,
    nlegcols=2,
    binwnorm=None,
    select={},
    action=(lambda x: x),
    extra_text=None,
    extra_text_loc=(0.8, 0.7),
    grid=False,
    plot_title=None,
    title_padding=0,
    yscale=None,
    logy=False,
    logx=False,
    fill_between=False,
    ratio_to_data=False,
    baseline=True,
    legtext_size=20,
    cms_decor="Preliminary",
    lumi=16.8,
    no_fill=False,
    no_stack=False,
    no_ratio=False,
    density=False,
    flow="none",
    bin_density=300,
    unstacked_linestyles=[],
    double_lines=False,
    ratio_error=True,
    normalize_to_data=False,
    cutoff=1e-6,
    noSci=False,
    logoPos=2,
    width_scale=1.0,
    linewidth=2,
    alpha=0.7,
    legPos="upper right",
    leg_padding="auto",
    lowerLegCols=2,
    lowerLegPos="upper right",
    lower_panel_variations=0,
    lower_leg_padding="auto",
    scaleRatioUnstacked=[],
    subplotsizes=[4, 2],
):
    add_ratio = not (no_stack or no_ratio)
    if ylabel is None:
        ylabel = "Events/bin" if not density else "density"

    colors = [histInfo[k].color for k in stackedProcs if histInfo[k].hists[histName]]
    labels = [histInfo[k].label for k in stackedProcs if histInfo[k].hists[histName]]

    to_read = stackedProcs[:]
    if "Data" in histInfo:
        to_read.append("Data")

    stack = []
    data_hist = None
    for k in to_read:
        if histName not in histInfo[k].hists or not histInfo[k].hists[histName]:
            logger.warning(f"Failed to find hist {histName} for proc {k}")
            continue
        h = action(histInfo[k].hists[histName])[select]

        # Use this if the hist has been rebinned for combine
        if xlim:
            h = h[complex(0, xlim[0]) : complex(0, xlim[1])]

        # If plotting from combine, apply the action to the underlying hist.
        # Don't do this for the generic case, as it screws up the ability to make multiple plots
        if fitresult:
            histInfo[k].hists[histName] = h

        if k != "Data":
            stack.append(h)
        else:
            data_hist = h

    if add_ratio:
        fig, ax1, ratio_axes = figureWithRatio(
            stack[0],
            xlabel,
            ylabel,
            ylim,
            rlabel,
            rrange,
            xlim=xlim,
            logy=logy,
            logx=logx,
            grid_on_ratio_plot=grid,
            plot_title=plot_title,
            title_padding=title_padding,
            bin_density=bin_density,
            width_scale=width_scale,
            subplotsizes=subplotsizes,
        )
        ax2 = ratio_axes[-1]
    else:
        fig, ax1 = figure(
            stack[0],
            xlabel,
            ylabel,
            ylim,
            xlim=xlim,
            logy=logy,
            logx=logx,
            plot_title=plot_title,
            title_padding=title_padding,
            bin_density=bin_density,
            width_scale=width_scale,
        )
        ratio_axes = None
        ax2 = None

    if fitresult:
        import uproot

        combine_result = uproot.open(fitresult)

        fittype = "prefit" if prefit else "postfit"

        # set histograms to prefit/postfit values
        for p in to_read:

            hname = f"expproc_{p}_{fittype}" if p != "Data" else "obs"
            vals = combine_result[hname].to_hist().values()
            if len(histInfo[p].hists[histName].values()) != len(vals):
                raise ValueError(
                    f"The size of the combine histogram ({(vals.shape)}) is not consistent with the xlim or input hist ({histInfo[p].hists[histName].shape})"
                )

            histInfo[p].hists[histName].values()[...] = vals
            if p == "Data":
                histInfo[p].hists[histName].variances()[...] = vals

        # for postfit uncertaity bands
        axis = histInfo[to_read[0]].hists[histName].axes[0].edges

        # need to divide by bin width
        binwidth = axis[1:] - axis[:-1]
        hexp = combine_result[f"expfull_{fittype}"].to_hist()
        if hexp.storage_type != hist.storage.Weight:
            raise ValueError(
                f"Did not find uncertainties in {fittype} hist. Make sure you run combinetf with --computeHistErrors!"
            )
        nom = hexp.values() / binwidth
        std = np.sqrt(hexp.variances()) / binwidth

        hatchstyle = "///"
        ax1.fill_between(
            axis,
            np.append(nom + std, (nom + std)[-1]),
            np.append(nom - std, (nom - std)[-1]),
            step="post",
            facecolor="none",
            zorder=2,
            hatch=hatchstyle,
            edgecolor="k",
            linewidth=0.0,
            label="Uncertainty",
        )

        if add_ratio:
            ax2.fill_between(
                axis,
                np.append((nom + std) / nom, ((nom + std) / nom)[-1]),
                np.append((nom - std) / nom, ((nom - std) / nom)[-1]),
                step="post",
                facecolor="none",
                zorder=2,
                hatch=hatchstyle,
                edgecolor="k",
                linewidth=0.0,
            )

    opts = dict(stack=not no_stack, flow=flow)
    optsr = opts.copy()  # no binwnorm for ratio axis
    optsr["density"] = density
    if density:
        opts["density"] = True
    else:
        opts["binwnorm"] = binwnorm

    if type(unstacked) == str:
        unstacked = unstacked.split(",")

    scale = 1.0
    if normalize_to_data:
        if "Data" not in histInfo:
            raise ValueError("Can't normalize to data without a data histogram!")

        vals = [
            x.value if hasattr(x, "value") else x
            for x in (data_hist.sum(), hh.sumHists(stack).sum())
        ]
        varis = [
            x.variance if hasattr(x, "variance") else x ** 0.5
            for x in (data_hist.sum(), hh.sumHists(stack).sum())
        ]
        scale = vals[0] / vals[1]
        unc = scale * (varis[0] / vals[0] ** 2 + varis[1] / vals[1] ** 2) ** 0.5
        ndigits = -math.floor(math.log10(abs(unc))) + 1
        logger.info(
            f"Rescaling all processes by {round(scale,ndigits)} +/- {round(unc,ndigits)} to match data norm"
        )
        stack = [s * scale for s in stack]

    hep.histplot(
        stack,
        histtype="fill" if not no_fill else "step",
        color=colors,
        label=labels,
        ax=ax1,
        zorder=1,
        **opts,
    )

    if "Data" in histInfo and ratio_to_data and add_ratio:
        hep.histplot(
            hh.divideHists(
                hh.sumHists(stack), data_hist, cutoff=cutoff, by_ax_name=False
            ),
            histtype="step",
            color=histInfo[stackedProcs[-1]].color,
            label=histInfo[stackedProcs[-1]].label,
            yerr=False,
            ax=ax2,
            zorder=3,
            **optsr,
        )

    extra_handles = []
    extra_labels = []
    if unstacked:
        linestyles = ["solid"] * len(unstacked)
        data_idx = -1
        if "Data" in unstacked:
            data_idx = unstacked.index("Data")
            linestyles[data_idx] = "None"
        linestyles = np.array(linestyles, dtype=object)
        logger.debug("Number of linestyles", len(linestyles))
        logger.debug("Length of unstacked", len(unstacked))
        if unstacked_linestyles:
            linestyles[data_idx + 1 : data_idx + 1 + len(unstacked_linestyles)] = (
                unstacked_linestyles
            )
        elif double_lines:
            linestyles[data_idx + 1 :: 2] = ["solid"] * len(
                linestyles[data_idx + 1 :: 2]
            )
            linestyles[data_idx + 2 :: 2] = ["dashed"] * len(
                linestyles[data_idx + 2 :: 2]
            )

        ratio_ref = data_hist if ratio_to_data else hh.sumHists(stack)
        if baseline and add_ratio:
            hep.histplot(
                hh.divideHists(
                    ratio_ref,
                    ratio_ref,
                    cutoff=cutoff,
                    rel_unc=True,
                    flow=False,
                    by_ax_name=False,
                ),
                histtype="step",
                color="grey",
                alpha=0.5,
                yerr=(
                    ratio_error
                    if ratio_ref.storage_type == hist.storage.Weight
                    else False
                ),
                ax=ax2,
                linewidth=linewidth,
                **optsr,
            )

        if fill_between and add_ratio:
            fill_procs = [x for x in unstacked if x != "Data"]
            if fill_between < 0:
                fill_between = len(fill_procs) + 1
            logger.debug(f"Filling first {fill_between}")
            for up, down in zip(
                fill_procs[:fill_between:2], fill_procs[1:fill_between:2]
            ):
                unstack_up = action(histInfo[up].hists[histName]) * scale
                unstack_down = action(histInfo[down].hists[histName]) * scale
                unstack_upr = hh.divideHists(
                    unstack_up, ratio_ref, 1e-6, flow=False, by_ax_name=False
                ).values()
                unstack_downr = hh.divideHists(
                    unstack_down, ratio_ref, 1e-6, flow=False, by_ax_name=False
                ).values()
                ax2.fill_between(
                    unstack_up.axes[0].edges,
                    np.insert(unstack_upr, 0, unstack_upr[0]),
                    np.insert(unstack_downr, 0, unstack_downr[0]),
                    step="pre",
                    color=histInfo[up].color,
                    alpha=0.5,
                )

        for i, (proc, style) in enumerate(zip(unstacked, linestyles)):
            unstack = histInfo[proc].hists[histName]
            if not fitresult or proc not in to_read:
                unstack = action(unstack)[select]
            if proc != "Data":
                unstack = unstack * scale
            if len(scaleRatioUnstacked) > i:
                hdiff = hh.addHists(unstack, ratio_ref, scale2=-1)
                hdiff = hh.scaleHist(hdiff, scaleRatioUnstacked[i])
                unstack = hh.addHists(hdiff, ratio_ref)

            if i >= lower_panel_variations or proc == "Data":
                # unstacked that are filled between are only plot in the lower panel
                hep.histplot(
                    unstack,
                    yerr=True if style == "None" else False,
                    histtype="errorbar" if style == "None" else "step",
                    color=histInfo[proc].color,
                    label=histInfo[proc].label,
                    ax=ax1,
                    alpha=alpha if style != "None" else 1.0,
                    linestyle=style,
                    linewidth=linewidth,
                    **opts,
                )
            elif histInfo[proc].label and histInfo[proc].label not in extra_labels:
                if fill_between is not None and i < fill_between:
                    extra_handles.append(
                        Polygon(
                            [[0, 0], [0, 0], [0, 0], [0, 0]],
                            color=histInfo[proc].color,
                            linestyle=style,
                            linewidth=linewidth,
                            alpha=alpha,
                        )
                    )
                else:
                    extra_handles.append(
                        Line2D(
                            [0],
                            [0],
                            color=histInfo[proc].color,
                            linestyle=style,
                            linewidth=linewidth,
                        )
                    )

                extra_labels.append(histInfo[proc].label)
            if ratio_to_data and proc == "Data" or not add_ratio:
                continue
            stack_ratio = hh.divideHists(
                unstack,
                ratio_ref,
                cutoff=cutoff,
                rel_unc=True,
                flow=False,
                by_ax_name=False,
            )
            hep.histplot(
                stack_ratio,
                histtype="errorbar" if style == "None" else "step",
                color=histInfo[proc].color,
                yerr=(
                    True
                    if (
                        style == "None"
                        and stack_ratio.storage_type == hist.storage.Weight
                    )
                    else False
                ),
                linewidth=linewidth,
                linestyle=style,
                ax=ax2,
                **optsr,
            )

    addLegend(
        ax1,
        nlegcols,
        loc=legPos,
        extra_text=extra_text,
        extra_text_loc=extra_text_loc,
        text_size=legtext_size,
        padding_loc=leg_padding,
    )
    if add_ratio:
        addLegend(
            ax2,
            lowerLegCols,
            text_size=legtext_size,
            loc=lowerLegPos,
            extra_handles=extra_handles,
            extra_labels=extra_labels,
            custom_handlers=(
                ["bandfilled"]
                if fill_between is not None
                else ["stacked"] if double_lines else []
            ),
            padding_loc=lower_leg_padding,
        )

    fix_axes(ax1, ratio_axes, fig, yscale=yscale, logy=logy, noSci=noSci)

    lumi = float(f"{lumi:.3g}") if not density else None
    if cms_decor:
        add_cms_decor(ax1, cms_decor, data="Data" in histInfo, lumi=lumi, loc=logoPos)

    return fig


def makePlotWithRatioToRef(
    hists,
    labels,
    colors,
    hists_ratio=None,
    midratio_idxs=None,
    linestyles=[],
    xlabel="",
    ylabel="Events/bin",
    rlabel=["x/nominal"],
    rrange=[[0.9, 1.1]],
    ylim=None,
    xlim=None,
    nlegcols=2,
    lowerLegPos="upper right",
    lowerLegCols=2,
    binwnorm=None,
    alpha=1.0,
    baseline=True,
    dataIdx=None,
    ratio_to_data=False,
    autorrange=None,
    grid=False,
    extra_text=None,
    extra_text_loc=(0.8, 0.7),
    yerr=False,
    legtext_size=20,
    plot_title=None,
    x_ticks_ndp=None,
    bin_density=300,
    yscale=None,
    logoPos=2,
    logy=False,
    logx=False,
    fill_between=0,
    title_padding=0,
    cms_label=None,
    cutoff=1e-6,
    only_ratio=False,
    width_scale=1,
    automatic_scale=True,
    base_size=8,
    linewidth=2,
    leg_padding="auto",
    lower_leg_padding="auto",
    subplotsizes=[4, 2],
    no_sci=False,
    lumi=None,
    center_rlabels=False,
    swap_ratio_panels=False,
):
    if hists_ratio is None:
        hists_ratio = hists

    if len(hists_ratio) != len(labels) or len(hists_ratio) != len(colors):
        raise ValueError(
            f"Number of hists ({len(hists_ratio)}), colors ({len(colors)}), and labels ({len(labels)}) must agree!"
        )
    ratio_hists = [
        hh.divideHists(
            h,
            hists[dataIdx if ratio_to_data else 0],
            cutoff=cutoff,
            flow=False,
            rel_unc=True,
            by_ax_name=False,
        )
        for h in hists_ratio[not baseline :]
    ]

    ratio_axes_idx = -1
    midratio_axes_idx = 0
    midratio_hists = None
    if len(subplotsizes) == 3:
        if midratio_idxs is None:
            raise ValueError("Found two ratio axes but no midratio hist indices!")

        midratio_hists = [
            hh.divideHists(
                hists_ratio[i],
                hists_ratio[midratio_idxs[0]],
                cutoff=cutoff,
                flow=False,
                rel_unc=True,
                by_ax_name=False,
            )
            for i in midratio_idxs
        ]

        if swap_ratio_panels:
            rlabel = rlabel[::-1]
            rrange = rrange[::-1]
            ratio_axes_idx = 0
            midratio_axes_idx = -1

    if not only_ratio:
        fig, ax1, ratio_axes = figureWithRatio(
            hists[0],
            xlabel,
            ylabel,
            ylim,
            rlabel,
            rrange,
            xlim=xlim,
            grid_on_ratio_plot=grid,
            plot_title=plot_title,
            title_padding=title_padding,
            bin_density=bin_density,
            logy=logy,
            logx=logx,
            only_ratio=only_ratio,
            width_scale=width_scale,
            subplotsizes=subplotsizes,
            automatic_scale=automatic_scale,
            base_size=base_size,
        )
        ax2 = ratio_axes[ratio_axes_idx]
    else:
        fig, ax2 = figureWithRatio(
            hists[0],
            xlabel,
            ylabel,
            ylim,
            rlabel,
            rrange,
            xlim=xlim,
            grid_on_ratio_plot=grid,
            plot_title=plot_title,
            title_padding=title_padding,
            bin_density=bin_density,
            logy=logy,
            logx=logx,
            only_ratio=only_ratio,
            width_scale=width_scale,
            automatic_scale=automatic_scale,
            base_size=base_size,
        )

    linestyles = linestyles + ["solid"] * (len(hists_ratio) - len(linestyles))

    exclude_data = lambda x: [j for i, j in enumerate(x) if i != dataIdx]

    if dataIdx is not None:
        hep.histplot(
            hists[dataIdx],
            histtype="errorbar",
            color=colors[dataIdx],
            label=labels[dataIdx],
            stack=False,
            ax=ax1,
            binwnorm=binwnorm,
            alpha=alpha,
            flow="none",
            zorder=4,
        )

    hists_noData = exclude_data(hists)
    if not only_ratio:
        hep.histplot(
            hists_noData,
            histtype="step",
            color=exclude_data(colors)[: len(hists_noData)],
            label=exclude_data(labels)[: len(hists_noData)],
            linestyle=exclude_data(linestyles)[: len(hists_noData)],
            linewidth=linewidth,
            stack=False,
            ax=ax1,
            yerr=yerr,
            binwnorm=binwnorm,
            alpha=alpha,
            flow="none",
            zorder=3,
        )

    if len(ratio_hists) > 1:
        plotRatio(
            ax2,
            ratio_hists=ratio_hists,
            labels=labels,
            colors=colors,
            linestyles=linestyles,
            linewidth=linewidth,
            lowerLegPos=lowerLegPos,
            lowerLegCols=lowerLegCols,
            legtext_size=legtext_size,
            lower_leg_padding=lower_leg_padding,
            alpha=alpha,
            yerr=False,
            fill_between=fill_between,
            dataIdx=dataIdx,
            baseline=baseline,
            add_legend=not only_ratio,
        )
        if midratio_hists:
            plotRatio(
                ratio_axes[midratio_axes_idx],
                ratio_hists=midratio_hists,
                labels=[labels[i] for i in midratio_idxs],
                colors=[colors[i] for i in midratio_idxs],
                linestyles=[linestyles[i] for i in midratio_idxs],
                linewidth=linewidth,
                lowerLegPos=lowerLegCols,
                lowerLegCols=lowerLegCols,
                legtext_size=legtext_size,
                lower_leg_padding=lower_leg_padding,
                alpha=alpha,
                yerr=False,
                fill_between=fill_between,
                dataIdx=(
                    midratio_idxs.index(dataIdx) if dataIdx in midratio_idxs else None
                ),
                baseline=baseline,
                add_legend=False,
            )

    if not only_ratio:
        addLegend(
            ax1,
            nlegcols,
            extra_text=extra_text,
            extra_text_loc=extra_text_loc,
            text_size=legtext_size,
            padding_loc=leg_padding,
        )

        fix_axes(
            ax1,
            ratio_axes,
            fig,
            x_ticks_ndp=x_ticks_ndp,
            yscale=yscale,
            logy=logy,
            noSci=no_sci,
            center_rlabels=center_rlabels,
        )

    if cms_label:
        add_cms_decor(ax1, cms_label, lumi=lumi, loc=logoPos)

    return fig


def plotRatio(
    ax,
    ratio_hists,
    labels,
    colors,
    linestyles=[],
    linewidth=2,
    lowerLegPos="upper right",
    lowerLegCols=2,
    legtext_size=None,
    alpha=1.0,
    yerr=False,
    fill_between=0,
    dataIdx=-1,
    baseline=True,
    add_legend=True,
    lower_leg_padding="auto",
):
    if fill_between != 0:
        for upr, downr, color in zip(
            ratio_hists[-fill_between::2],
            ratio_hists[-fill_between + 1 :: 2],
            colors[-fill_between::2],
        ):
            ax.fill_between(
                upr.axes[0].edges,
                np.append(upr.values(), upr.values()[-1]),
                np.append(downr.values(), downr.values()[-1]),
                step="post",
                color=color,
                alpha=0.5,
            )

    exclude_data = lambda x: [j for i, j in enumerate(x) if i != dataIdx]

    hep.histplot(
        exclude_data(ratio_hists)[not baseline :],
        histtype="step",
        color=exclude_data(colors)[not baseline :],
        linestyle=exclude_data(linestyles)[not baseline :],
        linewidth=linewidth,
        yerr=yerr,
        stack=False,
        ax=ax,
        alpha=alpha,
        flow="none",
    )
    if dataIdx is not None:
        hep.histplot(
            ratio_hists[dataIdx],
            histtype="errorbar",
            color=colors[dataIdx],
            xerr=False,
            yerr=True,
            stack=False,
            ax=ax,
            alpha=alpha,
            flow="none",
        )

    extra_handles = [
        Polygon(
            [[0, 0], [0, 0], [0, 0], [0, 0]],
            color=c,
            linestyle=l,
            linewidth=linewidth,
            alpha=alpha,
        )
        for c, l in zip(colors[-fill_between::2], linestyles[-fill_between::2])
    ]
    extra_labels = exclude_data(labels)[: len(extra_handles)]

    if add_legend:
        addLegend(
            ax,
            lowerLegCols,
            loc=lowerLegPos,
            text_size=legtext_size,
            extra_handles=extra_handles,
            extra_labels=extra_labels,
            custom_handlers=["stackfilled"],
            padding_loc=lower_leg_padding,
        )


def makeHistPlot2D(h2d, flow=False, **kwargs):
    if flow:
        xedges, yedges = extendEdgesByFlow(h2d)
    else:
        edges = h2d.axes.edges
        xedges = np.reshape(edges[0], len(edges[0]))
        yedges = edges[1][0]
    values = h2d.values(flow=flow)
    variances = h2d.variances(flow=flow)
    makePlot2D(values, variances, xedges, yedges, **kwargs)


def makePlot2D(
    values,
    variances=None,
    xedges=None,
    yedges=None,
    density=False,
    plot_uncertainties=False,
    xlabel="",
    ylabel="",
    zlabel="",
    colormap="RdBu",
    plot_title=None,
    ylim=None,
    xlim=None,
    zlim=None,
    zsymmetrize=None,
    logz=False,  # logy=False, logx=False, #TODO implement
    logoPos=2,
    cms_label="Work in progress",
    has_data=False,
    scaleleg=1.0,
    automatic_scale=False,
    width_scale=1.2,
):
    if xedges is None or yedges is None:
        xbins, ybins = values.shape
        if xedges is None:
            xedges = np.arange(xbins)
        if yedges is None:
            yedges = np.arange(ybins)
    # if variances is None:
    #     logger.warning("No variances given, assume")
    #     variances = values

    if density:
        xbinwidths = np.diff(xedges)
        ybinwidths = np.diff(yedges)
        binwidths = np.outer(xbinwidths, ybinwidths)
        values /= binwidths
        variances /= binwidths
    elif plot_uncertainties:
        # plot relative uncertainties instead
        values = np.sqrt(hh.relVariance(values, variances, fillOnes=True))

    if xlim is None:
        xlim = (xedges[0], xedges[-1])
    if ylim is None:
        ylim = (yedges[0], yedges[-1])

    fig, ax = figure(
        values,
        xlabel=xlabel,
        ylabel=ylabel,
        automatic_scale=automatic_scale,
        width_scale=width_scale,
        xlim=xlim,
        ylim=ylim,
    )

    if zlim is None:
        if logz:
            zmin = min(values[values > 0])  # smallest value that is not 0
        else:
            zmin = values.min()
        zmax = values.max()
        zlim = (zmin, zmax)

    # make symmetric range around value of zsymmetrize
    if zsymmetrize is not None:
        zrange = max((zmin - zsymmetrize), (zsymmetrize - zmax))
        zlim = [zsymmetrize - zrange, zsymmetrize + zrange]

    if plot_title:
        ax.text(
            1.0,
            1.003,
            plot_title,
            transform=ax.transAxes,
            fontsize=30,
            verticalalignment="bottom",
            horizontalalignment="right",
        )

    scale = max(1, np.divide(*ax.get_figure().get_size_inches()) * 0.3)
    hep.cms.label(
        ax=ax,
        lumi=None,
        fontsize=20 * scaleleg * scale,
        label=cms_label,
        data=has_data,
        loc=logoPos,
    )

    return fig


def extendEdgesByFlow(href, bin_flow_width=0.02):
    # add extra bin with bin wdith of a fraction of the total width
    all_edges = []
    for axis in href.axes:
        edges = axis.edges
        axis_range = edges[-1] - edges[0]
        if axis.traits.underflow:
            edges = np.insert(edges, 0, edges[0] - axis_range * bin_flow_width)
        if axis.traits.overflow:
            edges = np.append(edges, edges[-1] + axis_range * bin_flow_width)
        all_edges.append(edges)
    if len(all_edges) == 1:
        return all_edges[0]
    else:
        return all_edges


def fix_axes(
    ax1,
    ratio_axes=None,
    fig=None,
    x_ticks_ndp=None,
    yscale=None,
    logy=False,
    noSci=False,
    center_rlabels=False,
):
    if yscale:
        ymin, ymax = ax1.get_ylim()
        ax1.set_ylim(ymin, ymax * yscale)

    ax1.tick_params(axis="y", pad=5)  # Set distance to axis for y-axis numbers
    redo_axis_ticks(ax1, "x")

    if noSci and not logy:
        redo_axis_ticks(ax1, "y")
    elif not logy:
        ax1.ticklabel_format(style="sci", useMathText=True, axis="y", scilimits=(0, 0))

    if ratio_axes is not None:
        if not isinstance(ratio_axes, (list, tuple, np.ndarray)):
            ratio_axes = [ratio_axes]

        ax1.set_xticklabels([])
        for i, ax in enumerate(ratio_axes):
            if i == len(ratio_axes) - 1:
                redo_axis_ticks(ax, "x")
            else:
                ax.set_xticklabels([])

        # Function to get the position of the ylabel in axes coordinates
        def get_ylabel_position(ax):
            label = ax.get_yaxis().get_label()
            fig.canvas.draw()  # This is necessary to update the figure
            return ax.transAxes.inverted().transform(
                label.get_window_extent().get_points()
            )[0, 0]

        # Get the leftmost position of the y-axis labels
        y_label_pos = min(*[get_ylabel_position(ax) for ax in [ax1, *ratio_axes]])

        # Set all labels to the leftmost position
        ax1.yaxis.set_label_coords(y_label_pos * 0.7, 1.0)

        for i, ax in enumerate(ratio_axes):
            ax.tick_params(axis="y", pad=5)  # Set distance to axis for y-axis numbers
            ax.yaxis.set_label_coords(y_label_pos * 0.7, 0.5 if center_rlabels else 1.0)

        if x_ticks_ndp:
            ax.xaxis.set_major_formatter(
                StrMethodFormatter("{x:." + str(x_ticks_ndp) + "f}")
            )


def redo_axis_ticks(ax, axlabel, no_labels=False):
    autoloc = ticker.AutoLocator()
    # Need this to avoid a warning when you set the axis values manually
    fixedloc = ticker.FixedLocator(
        autoloc.tick_values(*getattr(ax, f"get_{axlabel}lim")())
    )
    getattr(ax, f"{axlabel}axis").set_major_locator(fixedloc)
    ticks = getattr(ax, f"get_{axlabel}ticks")()
    labels = [format_axis_num(x, ticks[-1]) for x in ticks] if not no_labels else []
    getattr(ax, f"set_{axlabel}ticklabels")(labels)


def format_axis_num(val, maxval):
    if type(val) == int or val.is_integer():
        # This is kinda dumb and I might change it
        return f"{val:.0f}" if maxval > 5 else f"{val:0.1f}"
    return f"{val:0.3g}" if maxval > 10 else f"{val:0.2g}"


def save_pdf_and_png(outdir, basename, fig=None):
    fname = f"{outdir}/{basename}.pdf"
    if fig:
        fig.savefig(fname, bbox_inches="tight")
        fig.savefig(fname.replace(".pdf", ".png"), bbox_inches="tight")
    else:
        plt.savefig(fname, bbox_inches="tight")
        plt.savefig(fname.replace(".pdf", ".png"), bbox_inches="tight")
    logger.info(f"Wrote file(s) {fname}(.png)")
    logger.info(f"Wrote file(s) {fname}(.png)")


def write_index_and_log(
    outpath,
    logname,
    template_dir=f"{pathlib.Path(__file__).parent}/Templates",
    yield_tables=None,
    analysis_meta_info=None,
    args={},
    nround=2,
):
    indexname = "index.php"
    if "mit.edu" in socket.gethostname() and not (
        hasattr(args, "eoscp") and args.eoscp
    ):
        indexname = "index_mit.php"

    shutil.copyfile(f"{template_dir}/{indexname}", f"{outpath}/index.php")
    logname = f"{outpath}/{logname}.log"

    with open(logname, "w") as logf:
        meta_info = (
            "-" * 80
            + "\n"
            + f"Script called at {datetime.datetime.now()}\n"
            + f"The command was: {narf.ioutils.script_command_to_str(sys.argv, args)}\n"
            + "-" * 80
            + "\n"
        )
        logf.write(meta_info)
        meta_info = narf.ioutils.make_meta_info_dict(
            "notebooks", args=args, wd=common.base_dir
        )
        logf.write(f"git hash: {meta_info['git_hash']}\n")
        logf.write(f"git diff: {meta_info['git_diff']}\n")

        if yield_tables:
            for k, v in yield_tables.items():
                logf.write(f"Yield information for {k}\n")
                logf.write("-" * 80 + "\n")
                logf.write(str(v.round(nround)) + "\n\n")

            if (
                "Unstacked processes" in yield_tables
                and "Stacked processes" in yield_tables
            ):
                if "Data" in yield_tables["Unstacked processes"]["Process"].values:
                    unstacked = yield_tables["Unstacked processes"]
                    data_yield = unstacked[unstacked["Process"] == "Data"][
                        "Yield"
                    ].iloc[0]
                    ratio = (
                        float(
                            yield_tables["Stacked processes"]["Yield"].sum()
                            / data_yield
                        )
                        * 100
                    )
                    logf.write(f"===> Sum unstacked to data is {ratio:.2f}%")

        if analysis_meta_info:
            for k, analysis_info in analysis_meta_info.items():
                logf.write("\n" + "-" * 80 + "\n")
                logf.write(f"Meta info from input file {k}\n")
                logf.write("\n" + "-" * 80 + "\n")
                logf.write(json.dumps(analysis_info, indent=5).replace("\\n", "\n"))
        logger.info(f"Writing file {logname}")


def make_summary_plot(
    centerline,
    center_unc,
    center_unc_part,
    center_label,
    df,
    colors,
    xlim,
    xlabel,
    ylim=None,
    legend_loc="upper right",
    double_colors=False,
    capsize=10,
    width_scale=1.5,
    center_color="black",
    label_points=True,
    legtext_size=None,
    lumi=None,
    bbox_to_anchor=None,
    markers=None,
    top_offset=0,
    bottom_offset=0,
    padding=4,
    point_size=0.24,
    point_center_colors=None,
    cms_label="Preliminary",
    logoPos=0,
    leg_padding="auto",
):
    nentries = len(df) + (bottom_offset - top_offset)

    # This code makes me feel like an idiot by I can't think of a better way to do it
    if type(colors) == str and colors == "auto":
        cmap = mpl.cm.get_cmap("tab10")
        colors = [cmap(i) for i in range(len(df))]

    if len(colors) < len(df):
        raise ValueError(
            f"Length of values ({nentries}) must be equal or smaller than colors!"
        )
    if ylim is None:
        ylim = [0.2, nentries + 1.5]

    fig, ax1 = figure(
        None,
        xlabel=xlabel,
        ylabel="",
        grid=False,
        automatic_scale=False,
        width_scale=width_scale,
        height=padding + point_size * nentries,
        xlim=xlim,
        ylim=ylim,
    )

    ax1.plot(
        [centerline, centerline],
        ylim,
        linestyle="solid",
        marker="none",
        color=center_color,
        linewidth=2,
    )
    if center_unc_part is not None:
        ax1.fill_between(
            [centerline - center_unc_part, centerline + center_unc_part],
            *ylim,
            color="silver",
            alpha=0.6,
        )
    ax1.fill_between(
        [centerline - center_unc, centerline + center_unc],
        *ylim,
        color="silver",
        alpha=0.6,
    )

    if center_unc_part is None:
        extra_handles = [
            (
                Polygon(
                    [[0, 0], [0, 0], [0, 0], [0, 0]],
                    color="silver",
                    linestyle="solid",
                    alpha=0.6,
                ),
                Line2D(
                    [0, 0], [0, 0], color=center_color, linestyle="solid", linewidth=2
                ),
            )
        ]
    else:
        extra_handles = [
            (
                Polygon(
                    [[0, 0], [0, 0], [0, 0], [0, 0]],
                    facecolor="silver",
                    linestyle="solid",
                    edgecolor="black",
                    linewidth=2,
                    alpha=0.6,
                ),
            )
        ]

    extra_labels = [center_label]

    if markers is None:
        markers = ["o"] * len(df)

    textsize = get_textsize(ax1, legtext_size)

    for i, (x, row) in enumerate(df.iterrows()):
        # Use for spacing purposes
        vals = row.iloc[1:].values
        xpos = vals[0]
        u = vals[1:]
        pos = nentries - i - top_offset
        ax1.errorbar(
            [xpos],
            [pos],
            xerr=u[0],
            linestyle="",
            linewidth=3,
            marker=markers[i],
            color=colors[i],
            capsize=capsize,
            # label=row.loc["Name"] if label_points else None,
        )
        if len(u) > 1:
            ax1.errorbar(
                [xpos],
                [pos],
                xerr=u[1],
                linestyle="",
                linewidth=3,
                marker=markers[i],
                color=colors[i] if not point_center_colors else point_center_colors[i],
                capsize=capsize,
            )
        if label_points:
            ax1.annotate(
                row["Name"],
                (xlim[0] + 0.01 * (xlim[1] - xlim[0]), pos),
                fontsize=textsize,
                ha="left",
                va="center",
                # color="dimgrey",
                annotation_clip=False,
                style=None,
            )

    if cms_label:
        add_cms_decor(
            ax1,
            cms_label,
            loc=logoPos,
            lumi=lumi,
            no_energy=lumi is None,
            text_size=textsize,
        )

    if legend_loc is not None or bbox_to_anchor is not None:
        # Assume these are data coords, and convert to figure coords
        # Probably should be an option instead of guessing
        if bbox_to_anchor is not None and any(abs(x) > 1 for x in bbox_to_anchor):
            data_to_figure = ax1.transData + ax1.transAxes.inverted()
            bbox_to_anchor = data_to_figure.transform(bbox_to_anchor)

        addLegend(
            ax1,
            ncols=1,
            text_size=legtext_size,
            bbox_to_anchor=bbox_to_anchor,
            loc=legend_loc,
            reverse=False,
            markerfirst=True,
            labelcolor=center_color,
            extra_handles=extra_handles,
            extra_labels=extra_labels,
            extra_entries_first=False,
            custom_handlers=(
                ["verticleline"] if center_unc_part is None else ["doubleband"]
            ),
            padding_loc=leg_padding,
        )

    ax1.minorticks_off()
    ax1.set_yticklabels([])
    ax1.xaxis.set_major_locator(ticker.LinearLocator(numticks=5))

    return fig
