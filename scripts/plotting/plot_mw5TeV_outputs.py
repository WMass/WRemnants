import argparse
import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mplhep as hep

from wremnants.utilities.io_tools import input_tools
from wums import plot_tools

import hist
import numpy as np

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import LogNorm

MC_GROUPS = {
    r"$Z/\gamma^*\rightarrow\mu\mu$": ["Zmumu_2017G", "Zmumu"],
    r"$Z/\gamma^*\rightarrow\tau\tau$": ["Ztautau_2017G", "Ztautau"],
    r"$W\rightarrow\tau\nu$": ["Wplustaunu_2017G", "Wminustaunu_2017G", "Wtaunu"],
    r"$W\rightarrow\mu\nu$": ["Wplusmunu_2017G", "Wminusmunu_2017G", "Wmunu"],
}

MC_COLORS = [
    "#C44E52",  # red
    "#F4A259",  # orange
    "#7BC67B",  # green
    "#7DB7E8",  # blue
]


def load_results(infile):
    obj = input_tools.read_infile(infile)
    return obj[0] if isinstance(obj, tuple) else obj


def read_hist(results, sample, histname):
    return results[sample]["output"][histname].get()


def samples_in_file(results):
    return [k for k in results.keys() if k != "meta_info"]


def get_data_hist(results, histname):
    data_samples = [s for s in samples_in_file(results) if s.startswith("SingleMuon")]
    if len(data_samples) != 1:
        raise RuntimeError(f"Expected one SingleMuon sample, found {data_samples}")

    h = read_hist(results, data_samples[0], histname)
    y = h.sum().value if hasattr(h.sum(), "value") else h.sum()
    print(f"{data_samples[0]:25s} {histname:15s} yield = {y}")
    return h


def sum_hists(results, samples, histname):
    hsum = None

    for sample in samples:
        if sample not in results:
            print(f"Skipping missing sample: {sample}")
            continue

        h = read_hist(results, sample, histname)
        y = h.sum().value if hasattr(h.sum(), "value") else h.sum()
        print(f"{sample:25s} {histname:15s} yield = {y}")

        hsum = h.copy() if hsum is None else hsum + h

    return hsum


def get_mc_stack(results, histname):
    hists = []
    labels = []

    for label, samples in MC_GROUPS.items():
        h = sum_hists(results, samples, histname)
        if h is None:
            continue

        hists.append(h)
        labels.append(label)

    if not hists:
        raise RuntimeError(f"No MC histograms found for {histname}")

    return hists, labels


def select_w_mt_bin(h, w_mt_bin):
    if w_mt_bin is None:
        return h

    if "w_mt" not in h.axes.name:
        raise RuntimeError(
            f"Requested w_mt bin {w_mt_bin}, but histogram axes are "
            f"{list(h.axes.name)}"
        )

    axis = h.axes["w_mt"]

    if w_mt_bin < 0 or w_mt_bin >= len(axis):
        raise RuntimeError(
            f"w_mt bin {w_mt_bin} is outside the valid range "
            f"[0, {len(axis) - 1}]"
        )

    low = axis.edges[w_mt_bin]
    high = axis.edges[w_mt_bin + 1]

    h_out = h[{"w_mt": w_mt_bin}]

    print(
        f"Selected w_mt bin {w_mt_bin}: "
        f"[{low}, {high}) GeV; remaining axes = {list(h_out.axes.name)}"
    )

    return h_out


def make_plot(
    infile,
    outdir,
    histname,
    xlabel,
    xlim,
    lumi,
    com,
    logy,
    wMtBin=None,
):
    os.makedirs(outdir, exist_ok=True)

    results = load_results(infile)
    print("Available samples:", samples_in_file(results))

    data_hist = get_data_hist(results, histname)
    mc_hists, mc_labels = get_mc_stack(results, histname)

    data_hist = select_w_mt_bin(data_hist, wMtBin)
    mc_hists = [
        select_w_mt_bin(h, wMtBin)
        for h in mc_hists
    ]

    if len(data_hist.axes) != 1:
        raise RuntimeError(
            f"Histogram {histname} has axes {list(data_hist.axes.name)} "
            "after the w_mt selection; dataMC mode expects one axis."
        )

    for h in mc_hists:
        if len(h.axes) != 1:
            raise RuntimeError(
                f"Histogram {histname} has axes {list(h.axes.name)} "
                "after the w_mt selection; dataMC mode expects one axis."
            )

    fig, ax = plt.subplots(figsize=(10.5, 8.0))

    colors = MC_COLORS[:len(mc_hists)]

    hep.histplot(
        mc_hists,
        ax=ax,
        stack=True,
        histtype="fill",
        color=colors[:len(mc_hists)],
        alpha=0.75,
        linewidth=0,
        label=mc_labels,
        flow="none",
    )

    hep.histplot(
        mc_hists,
        ax=ax,
        stack=True,
        histtype="step",
        color="black",
        linewidth=1.0,
        flow="none",
    )

    hep.histplot(
        data_hist,
        ax=ax,
        histtype="errorbar",
        color="black",
        label="Data",
        flow="none",
    )

    if histname == "met_pt":
        for cut in [20, 25, 30, 35, 40]:
            ax.axvline(cut, color="black", linestyle="--", linewidth=0.8, alpha=0.35)
        ax.axvline(25, color="black", linewidth=1.5, label=r"$p_T^{miss}>25$ GeV")

    if histname == "w_mt":
        ax.axvline(40, color="black", linewidth=1.5, label=r"$m_T>40$ GeV")

    if histname == "mu_abs_dxy":
        ax.axvline(
            0.015,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label=r"$|d_{xy}|=0.015$ cm",
        )

    if histname == "mu_relIso":
        ax.axvline(
            0.4,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label=r"$\mathrm{RelIso}=0.4$",
        )

    ax.set_xlabel(xlabel if xlabel else histname, fontsize=14)
    ax.set_ylabel("Events/bin", fontsize=14)

    if xlim is not None:
        ax.set_xlim(*xlim)

    if logy:
        ax.set_yscale("log")

    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
    ax.minorticks_on()

    plot_tools.addLegend(ax, ncols=1, loc="upper right", text_size="small")

    hep.cms.label(
        ax=ax,
        label="Preliminary",
        data=True,
        lumi=lumi,
        com=com,
        loc=0,
    )

    fig.tight_layout()

    wmt_suffix = f"_wmt{wMtBin}" if wMtBin is not None else ""
    basename = f"{histname}_nominal{wmt_suffix}"
    plot_tools.save_pdf_and_png(outdir, basename, fig=fig)
    plt.close(fig)


def draw_abcd_regions(ax, dxyCut=None, relIsoCut=None):
    if dxyCut is None or relIsoCut is None:
        return

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    # Cut lines
    ax.axvline(
        dxyCut,
        color="black",
        linestyle="--",
        linewidth=1.8,
    )

    ax.axhline(
        relIsoCut,
        color="black",
        linestyle="--",
        linewidth=1.8,
    )

    # Text positions: centers of the four visible regions
    x_low = 0.5 * (xmin + dxyCut)
    x_high = 0.5 * (dxyCut + xmax)
    y_low = 0.5 * (ymin + relIsoCut)
    y_high = 0.5 * (relIsoCut + ymax)

    text_style = dict(
        fontsize=18,
        fontweight="bold",
        color="black",
        ha="center",
        va="center",
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
    )

    ax.text(x_high, y_high, "A", **text_style)
    ax.text(x_high, y_low,  "B", **text_style)
    ax.text(x_low,  y_high, "C", **text_style)
    ax.text(x_low,  y_low,  "D / SR", **text_style)

    # Optional small labels for the cuts
    ax.text(
        dxyCut,
        ymax,
        rf"$|d_{{xy}}|={dxyCut:g}$ cm",
        ha="right",
        va="top",
        rotation=90,
        fontsize=11,
        bbox=dict(facecolor="white", alpha=0.65, edgecolor="none"),
    )

    ax.text(
        xmax,
        relIsoCut,
        rf"RelIso={relIsoCut:g}",
        ha="right",
        va="bottom",
        fontsize=11,
        bbox=dict(facecolor="white", alpha=0.65, edgecolor="none"),
    )

def make_2d_datamc_plot(
    infile,
    outdir,
    histname,
    xlabel,
    ylabel,
    xlim,
    ylim,
    lumi,
    com,
    logz,
    relIsoCut=None,
    dxyCut=None,
    wMtBin=None,
    makeRatio=True,
):
    os.makedirs(outdir, exist_ok=True)

    results = load_results(infile)
    print("Available samples:", samples_in_file(results))

    data_hist = get_data_hist(results, histname)
    mc_hists, mc_labels = get_mc_stack(results, histname)

    data_hist = select_w_mt_bin(data_hist, wMtBin)
    mc_hists = [select_w_mt_bin(h, wMtBin) for h in mc_hists]

    if len(data_hist.axes) != 2:
        raise RuntimeError(
            f"Histogram {histname} has {len(data_hist.axes)} axes, but dataMC2D expects 2D."
        )

    mc_hist = None
    for h in mc_hists:
        mc_hist = h.copy() if mc_hist is None else mc_hist + h

    xedges = np.asarray(data_hist.axes[0].edges, dtype=float)
    yedges = np.asarray(data_hist.axes[1].edges, dtype=float)

    data_vals = np.asarray(data_hist.values(flow=False), dtype=float)
    mc_vals = np.asarray(mc_hist.values(flow=False), dtype=float)

    ratio_vals = np.divide(
        data_vals,
        mc_vals,
        out=np.full_like(data_vals, np.nan, dtype=float),
        where=mc_vals != 0,
    )

    plots = [
        ("data", data_vals, "Data events/bin", logz),
        ("mc", mc_vals, "MC events/bin", logz),
    ]

    if makeRatio:
        plots.append(
            ("dataMC_ratio", ratio_vals, "Data/MC", False)
        )

    for tag, values, zlabel, use_log in plots:
        fig, ax = plt.subplots(figsize=(9.0, 8.0))

        if use_log:
            values_to_plot = np.ma.masked_where(values <= 0, values)
            norm = LogNorm()
        else:
            values_to_plot = np.ma.masked_invalid(values)
            norm = None

        mesh = ax.pcolormesh(
            xedges,
            yedges,
            values_to_plot.T,
            shading="auto",
            norm=norm,
        )

        cbar = fig.colorbar(mesh, ax=ax)
        cbar.set_label(zlabel, fontsize=14)

        ax.set_xlabel(xlabel if xlabel else data_hist.axes[0].name, fontsize=14)
        ax.set_ylabel(ylabel if ylabel else data_hist.axes[1].name, fontsize=14)

        if xlim is not None:
            ax.set_xlim(*xlim)

        if ylim is not None:
            ax.set_ylim(*ylim)

        draw_abcd_regions(
            ax,
            dxyCut=dxyCut,
            relIsoCut=relIsoCut,
        )


        if tag == "dataMC_ratio":
            mesh.set_clim(0.5, 1.5)

        ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
        ax.minorticks_on()

        hep.cms.label(
            ax=ax,
            label="Preliminary",
            data=True,
            lumi=lumi,
            com=com,
            loc=0,
        )

        fig.tight_layout()

        suffix = "nocuts"

        if dxyCut is not None and relIsoCut is not None:
            suffix = f"abcd_dxy{dxyCut:g}_relIso{relIsoCut:g}"
            suffix = suffix.replace(".", "p")

        if wMtBin is not None:
            suffix = f"{suffix}_wmt{wMtBin}"

        basename = f"{histname}_{tag}_2D_{suffix}"
        plot_tools.save_pdf_and_png(outdir, basename, fig=fig)

def resolve_variation_histname(histname, variation):
    suffix = f"_{variation}"

    if histname.endswith(suffix):
        return histname

    return f"{histname}{suffix}"


def variation_axis_name(variation):
    if variation == "alphaS":
        return "alphaS"

    if variation == "pdf":
        return "pdf"

    raise RuntimeError(f"Unknown variation type: {variation}")


def variation_axis_labels(h, axis_name):
    axis = None

    for ax in h.axes:
        if ax.name == axis_name:
            axis = ax
            break

    if axis is None:
        raise RuntimeError(f"Could not find axis '{axis_name}' in histogram with axes {[ax.name for ax in h.axes]}")

    try:
        return [axis.value(i) for i in range(axis.size)]
    except Exception:
        return list(axis)


def slice_variation(h, axis_name, label):
    return h[{axis_name: hist.loc(label)}]


def hist_integral(h):
    s = h.sum()
    return s.value if hasattr(s, "value") else s


def scale_hist_to_ref_integral(h, h_ref):
    h_sum = hist_integral(h)
    ref_sum = hist_integral(h_ref)

    if h_sum == 0:
        return h.copy()

    return h * (ref_sum / h_sum)


def sum_variation_hists(results, samples, histname, axis_name):
    labels = None
    sums = {}

    for sample in samples:
        if sample not in results:
            print(f"Skipping missing sample: {sample}")
            continue

        if histname not in results[sample]["output"]:
            print(f"Skipping missing hist {histname} for sample {sample}")
            continue

        h = read_hist(results, sample, histname)
        sample_labels = variation_axis_labels(h, axis_name)

        if labels is None:
            labels = sample_labels
            sums = {label: None for label in labels}
        elif sample_labels != labels:
            raise RuntimeError(
                f"Variation labels changed for sample {sample}. "
                f"Expected {labels}, got {sample_labels}"
            )

        print(f"{sample:25s} {histname:20s} {axis_name} variations = {len(labels)}")

        for label in labels:
            h_var = slice_variation(h, axis_name, label)
            sums[label] = h_var.copy() if sums[label] is None else sums[label] + h_var

    if labels is None:
        raise RuntimeError(f"No variation histograms found for {histname}")

    central_label = labels[0]
    print(f"Central variation label: {central_label}")
    print(f"Central integral: {hist_integral(sums[central_label])}")

    if len(labels) <= 6:
        for label in labels:
            print(f"  {label:20s} integral = {hist_integral(sums[label])}")
    else:
        print(f"Number of non-central variations: {len(labels) - 1}")
        print(f"First few labels: {labels[:6]}")

    return labels, sums


def make_ratio_values(h_var, h_central, shape_norm=True):
    h_use = scale_hist_to_ref_integral(h_var, h_central) if shape_norm else h_var

    num = np.asarray(h_use.values(flow=False), dtype=float)
    den = np.asarray(h_central.values(flow=False), dtype=float)

    return np.divide(
        num,
        den,
        out=np.full_like(den, np.nan, dtype=float),
        where=den != 0,
    )


def make_alphaS_plot(h_central, h_up, h_down, histname, outdir, xlabel, xlim, rrange, lumi, com, logy, shape_norm):
    hists = [h_central, h_up, h_down]

    if shape_norm:
        h_up_ratio = scale_hist_to_ref_integral(h_up, h_central)
        h_down_ratio = scale_hist_to_ref_integral(h_down, h_central)
    else:
        h_up_ratio = h_up
        h_down_ratio = h_down

    hists_ratio = [h_central, h_up_ratio, h_down_ratio]

    labels = [
        "central",
        r"$\alpha_s$ up",
        r"$\alpha_s$ down",
    ]

    colors = MC_COLORS[1:4]

    fig = plot_tools.makePlotWithRatioToRef(
        hists,
        labels,
        colors=colors,
        hists_ratio=hists_ratio,
        linestyles=["solid", "solid", "solid"],
        xlabel=xlabel if xlabel else histname,
        ylabel="Events/bin",
        rlabel=["Var./central"],
        rrange=[rrange],
        xlim=xlim,
        logy=logy,
        baseline=True,
        dataIdx=None,
        ratio_to_data=False,
        yerr=False,
        ratio_legend=False,
        nlegcols=1,
        legtext_size=20,
        cms_label=None,
        lumi=lumi,
        width_scale=1.0,
        linewidth=2,
    )

    hep.cms.label(
        ax=fig.axes[0],
        label="Preliminary",
        data=True,
        lumi=lumi,
        com=com,
        loc=0,
    )

    fig.axes[0].set_xlabel("")
    fig.axes[0].xaxis.label.set_visible(False)

    ratio_ax = fig.axes[-1]

    lower_handles = [
        Line2D([0], [0], color=colors[1], linestyle="solid", linewidth=2),
        Line2D([0], [0], color=colors[2], linestyle="solid", linewidth=2),
    ]

    lower_labels = [
        r"$\alpha_s$ up / central",
        r"$\alpha_s$ down / central",
    ]

    plot_tools.addLegend(
        ratio_ax,
        ncols=1,
        loc="upper right",
        text_size=18,
        extra_handles=lower_handles,
        extra_labels=lower_labels,
    )

    fig.tight_layout()

    basename = f"{histname}"
    plot_tools.save_pdf_and_png(outdir, basename, fig=fig)


def make_pdf_plot(h_central, h_variations, histname, outdir, xlabel, xlim, rrange, lumi, com, logy, shape_norm, pdf_lines=False):
    ratios = np.array([
        make_ratio_values(h_var, h_central, shape_norm=shape_norm)
        for h_var in h_variations
    ])

    ratio_min = np.nanmin(ratios, axis=0)
    ratio_max = np.nanmax(ratios, axis=0)

    edges = np.asarray(h_central.axes[0].edges, dtype=float)

    fig = plot_tools.makePlotWithRatioToRef(
        [h_central],
        ["central"],
        colors=[MC_COLORS[0]],
        hists_ratio=[h_central],
        linestyles=["solid"],
        xlabel=xlabel if xlabel else histname,
        ylabel="Events/bin",
        rlabel=["PDF/central"],
        rrange=[rrange],
        xlim=xlim,
        logy=logy,
        baseline=True,
        dataIdx=None,
        ratio_to_data=False,
        yerr=False,
        ratio_legend=False,
        nlegcols=1,
        legtext_size=18,
        cms_label=None,
        lumi=lumi,
        width_scale=1.0,
        linewidth=2,
    )

    hep.cms.label(
        ax=fig.axes[0],
        label="Preliminary",
        data=True,
        lumi=lumi,
        com=com,
        loc=0,
    )

    fig.axes[0].set_xlabel("")
    fig.axes[0].xaxis.label.set_visible(False)

    ratio_ax = fig.axes[-1]

    ratio_ax.axhline(1.0, color="black", linewidth=1.0, zorder=3)

    if pdf_lines:
        for ratio in ratios:
            ratio_ax.step(
                edges,
                np.r_[ratio, ratio[-1]],
                where="post",
                color="gray",
                alpha=0.25,
                linewidth=0.6,
                zorder=1,
            )

    ratio_ax.fill_between(
        edges,
        np.r_[ratio_min, ratio_min[-1]],
        np.r_[ratio_max, ratio_max[-1]],
        step="post",
        color=MC_COLORS[1],
        alpha=0.45,
    )

    ratio_ax.set_ylim(*rrange)

    lower_handles = []
    lower_labels = []

    if pdf_lines:
        lower_handles.append(
            Line2D([0], [0], color="gray", alpha=0.6, linewidth=1.0)
        )
        lower_labels.append("PDF members")

    lower_handles.append(
        Patch(facecolor=MC_COLORS[1], alpha=0.35 if pdf_lines else 0.45)
    )
    lower_labels.append("PDF envelope")

    plot_tools.addLegend(
        ratio_ax,
        ncols=1,
        loc="upper right",
        text_size=18.0,
        extra_handles=lower_handles,
        extra_labels=lower_labels,
    )

    fig.tight_layout()

    basename = f"{histname}"
    plot_tools.save_pdf_and_png(outdir, basename, fig=fig)


def make_variation_plot(infile, outdir, histname, samples, variation, xlabel, xlim, rrange, lumi, com, logy, shape_norm, pdf_lines=False):
    os.makedirs(outdir, exist_ok=True)

    results = load_results(infile)
    print("Available samples:", samples_in_file(results))
    print(f"Using {variation} samples:", samples)

    histname = resolve_variation_histname(histname, variation)
    axis_name = variation_axis_name(variation)

    labels, hists_by_label = sum_variation_hists(results, samples, histname, axis_name)

    central_label = labels[0]
    h_central = hists_by_label[central_label]

    if len(h_central.axes) != 1:
        raise RuntimeError(
            f"{histname} has {len(h_central.axes)} axes after variation slicing. "
            "Use an unrolled plotting script for 2D histograms."
        )

    if variation == "alphaS":
        required = ["central", "alphaSUp", "alphaSDown"]
        missing = [label for label in required if label not in hists_by_label]

        if missing:
            raise RuntimeError(f"Missing alphaS labels {missing}. Available labels are {labels}")

        make_alphaS_plot(
            hists_by_label["central"],
            hists_by_label["alphaSUp"],
            hists_by_label["alphaSDown"],
            histname,
            outdir,
            xlabel,
            xlim,
            rrange,
            lumi,
            com,
            logy,
            shape_norm,
        )

    elif variation == "pdf":
        h_variations = [hists_by_label[label] for label in labels[1:]]

        if len(h_variations) == 0:
            raise RuntimeError(f"No PDF variations found in {histname}. Labels are {labels}")

        make_pdf_plot(
            h_central,
            h_variations,
            histname,
            outdir,
            xlabel,
            xlim,
            rrange,
            lumi,
            com,
            logy,
            shape_norm,
            pdf_lines=pdf_lines,
        )



# ============================================================================
# Final-presentation Rabbit input plots
# ============================================================================
#
# These configurations intentionally use the same multidimensional histograms
# used to build the Rabbit tensors, then project them to the displayed axis.
#
# For w_pt, w_mt bin 0 is selected to match the tensor-preparation workflow.
RABBIT_INPUT_CONFIGS = {
    "mu_pt": {
        "nominal_hist": "mu_abcd",
        "alphaS_hist": "mu_abcd_minnlo_pdfas_Corr",
        "pdf_hist": "mu_abcd_minnlo_pdfvars_Corr",
        "x_axis": "pt",
        "selections": {},
        "xlabel": r"$p_T^\mu$ [GeV]",
        "xlim": [18.0, 44.0],
    },
    "mu_eta": {
        "nominal_hist": "mu_abcd",
        "alphaS_hist": "mu_abcd_minnlo_pdfas_Corr",
        "pdf_hist": "mu_abcd_minnlo_pdfvars_Corr",
        "x_axis": "eta",
        "selections": {},
        "xlabel": r"$\eta^\mu$",
        "xlim": [-2.4, 2.4],
    },
    "mu_charge": {
        "nominal_hist": "mu_abcd",
        "alphaS_hist": "mu_abcd_minnlo_pdfas_Corr",
        "pdf_hist": "mu_abcd_minnlo_pdfvars_Corr",
        "x_axis": "charge",
        "selections": {},
        "xlabel": "Muon charge",
        "xlim": [-2.0, 2.0],
    },
    "w_pt": {
        "nominal_hist": "wpt_abcd",
        "alphaS_hist": "wpt_abcd_minnlo_pdfas_Corr",
        "pdf_hist": "wpt_abcd_minnlo_pdfvars_Corr",
        "x_axis": "w_pt",
        "selections": {},
        "xlabel": r"$p_T^W$ [GeV]",
        "xlim": [0.0, 80.0],
    },
    "w_mt": {
        "nominal_hist": "w_mt",
        "alphaS_hist": "w_mt_minnlo_pdfas_Corr",
        "pdf_hist": "w_mt_minnlo_pdfvars_Corr",
        "x_axis": "w_mt",
        "selections": {},
        "xlabel": r"$m_T^W$ [GeV]",
        "xlim": [40.0, 160.0],
    },
    "met_pt": {
        "nominal_hist": "deepmet_reso_pt",
        "alphaS_hist": "deepmet_reso_pt_minnlo_pdfas_Corr",
        "pdf_hist": "deepmet_reso_pt_minnlo_pdfvars_Corr",
        "x_axis": "met_pt",
        "selections": {},
        "xlabel": r"$p_T^{\mathrm{miss}}$ [GeV]",
        "xlim": [18.0, 44.0],
    },
}

ALPHAS_UP_LABEL = "pdfCT18ZNNLO_as_0120"
ALPHAS_DOWN_LABEL = "pdfCT18ZNNLO_as_0116"

ALPHAS_UP_COLOR = "#2CA02C"
ALPHAS_DOWN_COLOR = "#D62728"
PDF_COLOR = "#1F77B4"


def find_matching_samples(results, filters):
    available = samples_in_file(results)
    return [
        sample
        for sample in available
        if any(sample == filt or sample.startswith(filt) for filt in filters)
    ]


def find_wmunu_samples(results):
    explicit = find_matching_samples(
        results,
        ["Wplusmunu_2017G", "Wminusmunu_2017G"],
    )
    if explicit:
        return explicit

    aggregate = find_matching_samples(results, ["Wmunu"])
    if aggregate:
        return aggregate

    raise RuntimeError(
        "Could not find Wplusmunu/Wminusmunu or aggregate Wmunu samples."
    )


def final_mc_groups(results):
    groups = []

    group_specs = [
        (
            r"$Z/\gamma^*\rightarrow\mu\mu$",
            ["Zmumu_2017G", "Zmumu"],
            None,
        ),
        (
            r"$Z/\gamma^*\rightarrow\tau\tau$",
            ["Ztautau_2017G", "Ztautau"],
            None,
        ),
        (
            r"$W\rightarrow\tau\nu$",
            ["Wplustaunu_2017G", "Wminustaunu_2017G"],
            ["Wtaunu"],
        ),
        (
            r"$W\rightarrow\mu\nu$",
            ["Wplusmunu_2017G", "Wminusmunu_2017G"],
            ["Wmunu"],
        ),
    ]

    for label, primary_filters, fallback_filters in group_specs:
        matched = find_matching_samples(results, primary_filters)

        if not matched and fallback_filters is not None:
            matched = find_matching_samples(results, fallback_filters)

        if matched:
            groups.append((label, matched))

    if not groups:
        raise RuntimeError("No nominal MC groups were found.")

    return groups

def select_and_project(h, x_axis, selections):
    h_use = h

    for axis_name, bin_index in selections.items():
        if axis_name not in h_use.axes.name:
            raise RuntimeError(
                f"Selection axis '{axis_name}' is missing from "
                f"histogram axes {list(h_use.axes.name)}"
            )
        h_use = h_use[{axis_name: int(bin_index)}]

    if x_axis not in h_use.axes.name:
        raise RuntimeError(
            f"Projection axis '{x_axis}' is missing from "
            f"histogram axes {list(h_use.axes.name)}"
        )

    if len(h_use.axes) > 1:
        h_use = h_use.project(x_axis)

    if list(h_use.axes.name) != [x_axis]:
        raise RuntimeError(
            f"Expected one projected axis '{x_axis}', got "
            f"{list(h_use.axes.name)}"
        )

    return h_use


def sum_projected_hists(
    results,
    samples,
    histname,
    x_axis,
    selections,
):
    hsum = None

    for sample in samples:
        if sample not in results:
            continue

        output = results[sample].get("output", {})
        if histname not in output:
            print(f"Skipping {sample}: missing histogram {histname}")
            continue

        h = read_hist(results, sample, histname)
        h = select_and_project(h, x_axis, selections)

        y = hist_integral(h)
        print(f"{sample:25s} {histname:35s} projected yield = {y}")

        hsum = h.copy() if hsum is None else hsum + h

    return hsum


def projected_variation_sums(
    results,
    samples,
    histname,
    x_axis,
    selections,
):
    labels = None
    sums = {}

    for sample in samples:
        if sample not in results:
            continue

        output = results[sample].get("output", {})
        if histname not in output:
            print(f"Skipping {sample}: missing histogram {histname}")
            continue

        h = read_hist(results, sample, histname)
        sample_labels = variation_axis_labels(h, "vars")

        if labels is None:
            labels = sample_labels
            sums = {label: None for label in labels}
        elif sample_labels != labels:
            raise RuntimeError(
                f"Variation labels changed for {sample}. "
                f"Expected {labels}, got {sample_labels}"
            )

        for label in labels:
            h_var = slice_variation(h, "vars", label)
            h_var = select_and_project(h_var, x_axis, selections)
            sums[label] = (
                h_var.copy()
                if sums[label] is None
                else sums[label] + h_var
            )

    if labels is None:
        raise RuntimeError(
            f"No projected variation histograms found for {histname}"
        )

    return labels, sums


def sum_hist_list(hists):
    total = None
    for h in hists:
        total = h.copy() if total is None else total + h
    return total


def ratio_array(h_num, h_den):
    num = np.asarray(h_num.values(flow=False), dtype=float)
    den = np.asarray(h_den.values(flow=False), dtype=float)

    return np.divide(
        num,
        den,
        out=np.full_like(den, np.nan, dtype=float),
        where=den != 0,
    )


def step_array(ax, edges, values, **kwargs):
    ax.step(
        edges,
        np.r_[values, values[-1]],
        where="post",
        **kwargs,
    )


def make_rabbit_input_plot(
    infile,
    outdir,
    input_name,
    xlabel,
    xlim,
    rrange,
    lumi,
    com,
    logy,
    shape_norm=False,
    pdf_lines=False,
):
    """
    Preserve the original filled, stacked Data/MC appearance and add the
    theory variations only in the lower ratio panel.

    The pdf_lines argument is retained for command-line compatibility, but
    every non-central PDF member is always plotted. No PDF envelope is made.
    """
    if input_name not in RABBIT_INPUT_CONFIGS:
        raise RuntimeError(
            f"Unknown Rabbit input '{input_name}'. "
            f"Choose from {list(RABBIT_INPUT_CONFIGS)}"
        )

    config = RABBIT_INPUT_CONFIGS[input_name]

    nominal_histname = config["nominal_hist"]
    alphaS_histname = config["alphaS_hist"]
    pdf_histname = config["pdf_hist"]
    x_axis = config["x_axis"]
    selections = config["selections"]

    xlabel = xlabel if xlabel is not None else config["xlabel"]
    xlim = xlim if xlim is not None else config["xlim"]

    os.makedirs(outdir, exist_ok=True)

    results = load_results(infile)
    print("Available samples:", samples_in_file(results))
    print("Rabbit input configuration:", config)

    data_hist = get_data_hist(results, nominal_histname)
    data_hist = select_and_project(data_hist, x_axis, selections)

    mc_group_hists = []
    mc_group_labels = []
    background_group_hists = []

    signal_samples = find_wmunu_samples(results)
    signal_sample_set = set(signal_samples)
    signal_nominal = None

    for label, group_samples in final_mc_groups(results):
        h_group = sum_projected_hists(
            results,
            group_samples,
            nominal_histname,
            x_axis,
            selections,
        )

        if h_group is None:
            continue

        mc_group_hists.append(h_group)
        mc_group_labels.append(label)

        is_signal_group = any(
            sample in signal_sample_set
            for sample in group_samples
        )

        if is_signal_group:
            signal_nominal = (
                h_group.copy()
                if signal_nominal is None
                else signal_nominal + h_group
            )
        else:
            background_group_hists.append(h_group)

    if not mc_group_hists:
        raise RuntimeError(
            f"No nominal MC histograms were found for {nominal_histname}"
        )

    if signal_nominal is None:
        raise RuntimeError(
            f"No nominal W->munu histogram found for {nominal_histname}"
        )

    if not background_group_hists:
        raise RuntimeError(
            f"No background MC histograms found for {nominal_histname}"
        )

    total_nominal = sum_hist_list(mc_group_hists)
    background_nominal = sum_hist_list(background_group_hists)

    data_yield = hist_integral(data_hist)
    mc_yield = hist_integral(total_nominal)

    if mc_yield <= 0:
        raise RuntimeError(
            f"Cannot normalize MC to data: total MC yield is {mc_yield}"
        )

    data_mc_norm = data_yield / mc_yield

    print(f"Projected data yield       = {data_yield}")
    print(f"Projected total MC yield   = {mc_yield}")
    print(f"Data/MC normalization      = {data_mc_norm}")

    # Preserve the normalization used by the previous presentation plots:
    #
    #     scale = projected data yield / projected total nominal MC yield
    #
    # Apply this ONE common scale factor to every nominal MC component and
    # every varied prediction. Data are not rescaled. Individual processes
    # and individual PDF members are never normalized separately.
    mc_group_hists = [h * data_mc_norm for h in mc_group_hists]
    total_nominal = total_nominal * data_mc_norm
    signal_nominal = signal_nominal * data_mc_norm
    background_nominal = background_nominal * data_mc_norm

    alphaS_labels, alphaS_sums = projected_variation_sums(
        results,
        signal_samples,
        alphaS_histname,
        x_axis,
        selections,
    )

    if ALPHAS_UP_LABEL not in alphaS_sums:
        raise RuntimeError(
            f"Missing alphaS-up label '{ALPHAS_UP_LABEL}'. "
            f"Available labels: {alphaS_labels}"
        )

    if ALPHAS_DOWN_LABEL not in alphaS_sums:
        raise RuntimeError(
            f"Missing alphaS-down label '{ALPHAS_DOWN_LABEL}'. "
            f"Available labels: {alphaS_labels}"
        )

    signal_alphaS_up = alphaS_sums[ALPHAS_UP_LABEL] * data_mc_norm
    signal_alphaS_down = alphaS_sums[ALPHAS_DOWN_LABEL] * data_mc_norm

    if shape_norm:
        signal_alphaS_up = scale_hist_to_ref_integral(
            signal_alphaS_up,
            signal_nominal,
        )
        signal_alphaS_down = scale_hist_to_ref_integral(
            signal_alphaS_down,
            signal_nominal,
        )

    total_alphaS_up = background_nominal + signal_alphaS_up
    total_alphaS_down = background_nominal + signal_alphaS_down

    pdf_labels, pdf_sums = projected_variation_sums(
        results,
        signal_samples,
        pdf_histname,
        x_axis,
        selections,
    )

    # The first entry is the central PDF member. Plot every remaining member.
    pdf_member_labels = pdf_labels[1:]
    if not pdf_member_labels:
        raise RuntimeError(
            f"No non-central PDF members found. Labels: {pdf_labels}"
        )

    total_pdf_members = []
    for label in pdf_member_labels:
        signal_pdf = pdf_sums[label] * data_mc_norm

        if shape_norm:
            signal_pdf = scale_hist_to_ref_integral(
                signal_pdf,
                signal_nominal,
            )

        total_pdf_members.append(background_nominal + signal_pdf)

    edges = np.asarray(total_nominal.axes[0].edges, dtype=float)

    alphaS_up_ratio = ratio_array(total_alphaS_up, total_nominal)
    alphaS_down_ratio = ratio_array(total_alphaS_down, total_nominal)

    pdf_ratios = np.asarray(
        [ratio_array(h, total_nominal) for h in total_pdf_members],
        dtype=float,
    )

    fig, (ax, ratio_ax) = plt.subplots(
        2,
        1,
        figsize=(10.5 * 1.25, 8.0),
        sharex=True,
        gridspec_kw={
            "height_ratios": [3.0, 1.15],
            "hspace": 0.05,
        },
    )

    colors = MC_COLORS[:len(mc_group_hists)]

    # Preserve the original filled, stacked MC appearance.
    hep.histplot(
        mc_group_hists,
        ax=ax,
        stack=True,
        histtype="step",
        color=colors,
        alpha=0.95,
        linewidth=0,
        label=mc_group_labels,
        flow="none",
    )

    # Preserve the thin black outlines between stacked components.
    hep.histplot(
        mc_group_hists,
        ax=ax,
        stack=True,
        histtype="step",
        color=colors,
        linewidth=2.0,
        flow="none",
    )

    hep.histplot(
        data_hist,
        ax=ax,
        histtype="errorbar",
        color="black",
        label="Data",
        flow="none",
    )

    # Do not draw a total-MC line or theory variations over the filled stack.
    ax.set_ylabel("Events/bin", fontsize=16)
    ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=True,
        right=True,
    )
    ax.minorticks_on()
    ax.tick_params(labelbottom=False)

    if xlim is not None:
        ax.set_xlim(*xlim)

    if logy:
        ax.set_yscale("log")

    ax.relim()
    ax.autoscale_view()

    plot_tools.addLegend(
        ax,
        ncols=1,
        loc="upper right",
        text_size=16,
    )

    hep.cms.label(
        ax=ax,
        label="Preliminary",
        data=True,
        lumi=lumi,
        com=com,
        loc=0,
    )
    # Lower panel: alpha_s up/down plus every PDF member. No envelope.
    ratio_ax.axhline(
        1.0,
        color="black",
        linewidth=1.0,
        zorder=1,
    )

    for ratio in pdf_ratios:
        step_array(
            ratio_ax,
            edges,
            ratio,
            color=PDF_COLOR,
            linewidth=0.65,
            alpha=0.50,
            zorder=2,
        )

    step_array(
        ratio_ax,
        edges,
        alphaS_up_ratio,
        color=ALPHAS_UP_COLOR,
        linewidth=2.0,
        zorder=4,
    )
    step_array(
        ratio_ax,
        edges,
        alphaS_down_ratio,
        color=ALPHAS_DOWN_COLOR,
        linewidth=2.0,
        zorder=4,
    )

    ratio_ax.set_ylabel("Var./nominal", fontsize=14)
    ratio_ax.set_xlabel(xlabel, fontsize=16)
    ratio_ax.set_ylim(*rrange)

    if xlim is not None:
        ratio_ax.set_xlim(*xlim)

    ratio_ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=True,
        right=True,
    )
    ratio_ax.minorticks_on()

    variation_handles = [
        Line2D(
            [0],
            [0],
            color=ALPHAS_UP_COLOR,
            linewidth=2.0,
        ),
        Line2D(
            [0],
            [0],
            color=ALPHAS_DOWN_COLOR,
            linewidth=2.0,
        ),
        Line2D(
            [0],
            [0],
            color=PDF_COLOR,
            linewidth=1.0,
            alpha=0.70,
        ),
    ]
    variation_labels = [
        r"$\alpha_s$ up / nominal",
        r"$\alpha_s$ down / nominal",
        f"PDF members ({len(pdf_member_labels)}) / nominal",
    ]

    plot_tools.addLegend(
        ratio_ax,
        ncols=1,
        loc="upper right",
        text_size=15,
        extra_handles=variation_handles,
        extra_labels=variation_labels,
    )

    fig.align_ylabels([ax, ratio_ax])
    fig.tight_layout()

    norm_tag = "shapeNorm" if shape_norm else "raw"
    basename = (
        f"{input_name}_rabbit_input_filledMC_allPDFs_normToData_{norm_tag}"
    )
    plot_tools.save_pdf_and_png(outdir, basename, fig=fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", help="HDF5 output from mw_5TeV.py")
    parser.add_argument("--hist", required=True, help="Histogram name, e.g. w_pt, w_mt, met_pt")
    parser.add_argument("-o", "--outdir", default="plots_mw_5TeV")
    parser.add_argument("--xlabel", default=None)
    parser.add_argument("--xlim", nargs=2, type=float, default=None)
    parser.add_argument("--rrange", nargs=2, type=float, default=[0.95, 1.05])
    parser.add_argument("--logy", action="store_true")
    parser.add_argument("--lumi", type=float, default=0.300)
    parser.add_argument("--com", default="5.02")
    parser.add_argument(
        "--mode",
        choices=["dataMC", "dataMC2D", "alphaS", "pdf", "rabbitInputs"],
        default="dataMC",
        help=("dataMC for nominal plots; alphaS/pdf for separate variation plots; ""rabbitInputs for the final line-only Data/MC plot with alphaS and PDF variations."),
    )
    parser.add_argument(
        "--samples",
        nargs="*",
        default=None,
        help="MC samples to use for variation plotting",
    )
    parser.add_argument(
        "--rawRatio",
        action="store_true",
        help="Use raw variation/central ratios instead of shape-normalized ratios",
    )
    parser.add_argument(
        "--pdfLines",
        action="store_true",
        help="Draw individual PDF-member ratio lines behind the PDF envelope",
    )

    parser.add_argument("--ylabel", default=None)
    parser.add_argument("--ylim", nargs=2, type=float, default=None)
    parser.add_argument("--logz", action="store_true")
    parser.add_argument("--relIsoCut", type=float, default=None)
    parser.add_argument("--dxyCut", type=float, default=None)
    parser.add_argument(
        "--wMtBin",
        type=int,
        choices=[0, 1],
        default=None,
        help=(
            "For dataMC or dataMC2D, select w_mt bin 0 ([0,40) GeV) "
            "or 1 ([40,160) GeV) before plotting."
        ),
    )
    parser.add_argument(
        "--skip2DRatio",
        action="store_true",
        help="For dataMC2D, save only the data and MC plots.",
    )

    args = parser.parse_args()

    if args.mode == "dataMC":
        make_plot(
            args.infile,
            args.outdir,
            args.hist,
            args.xlabel,
            args.xlim,
            args.lumi,
            args.com,
            args.logy,
            wMtBin=args.wMtBin,
        )

    elif args.mode == "rabbitInputs":
        make_rabbit_input_plot(
            args.infile,
            args.outdir,
            args.hist,
            args.xlabel,
            args.xlim,
            args.rrange,
            args.lumi,
            args.com,
            args.logy,
            shape_norm=not args.rawRatio,
            pdf_lines=args.pdfLines,
        )

    elif args.mode in ["alphaS", "pdf"]:
        samples_for_variation = args.samples

        if samples_for_variation is None or len(samples_for_variation) == 0:
            samples_for_variation = [
                "Wplusmunu_2017G",
                "Wminusmunu_2017G",
                "Wmunu",
            ]

        make_variation_plot(
            args.infile,
            args.outdir,
            args.hist,
            samples_for_variation,
            args.mode,
            args.xlabel,
            args.xlim,
            args.rrange,
            args.lumi,
            args.com,
            args.logy,
            shape_norm=not args.rawRatio,
            pdf_lines=args.pdfLines,
        )
    elif args.mode == "dataMC2D":
        make_2d_datamc_plot(
            args.infile,
            args.outdir,
            args.hist,
            args.xlabel,
            args.ylabel,
            args.xlim,
            args.ylim,
            args.lumi,
            args.com,
            args.logz,
            relIsoCut=args.relIsoCut,
            dxyCut=args.dxyCut,
            wMtBin=args.wMtBin,
            makeRatio=not args.skip2DRatio,
        )

if __name__ == "__main__":
    main()