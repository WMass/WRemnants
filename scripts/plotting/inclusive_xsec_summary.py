# On results of fiducial inclusive cross sections and their ratios
# make a summary plot with different theory predictions
# make a latex summary table with the breakdown of uncertainties

import math

import hist
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse, Polygon

import rabbit.io_tools
from utilities import parsing
from utilities.io_tools import tex_tools

from wums import logging, output_tools, plot_tools  # isort: skip

parser = parsing.plot_parser()
parser.add_argument("infile", type=str, help="Rabbit fitresult file")
parser.add_argument(
    "--pdfFiles",
    type=str,
    nargs="*",
    default=[],
    help="Rabbit fitresult files with alternative pdf predictions",
)
parser.add_argument(
    "--config",
    type=str,
    default=None,
    help="Path to config file for style formatting",
)
args = parser.parse_args()

logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)

config = plot_tools.load_config(args.config)

grouping = getattr(config, "nuisance_grouping", {}).get("xsecs", None)
translate_label = getattr(config, "systematics_labels", {})

fitresult, meta = rabbit.io_tools.get_fitresult(args.infile, meta=True)
result = fitresult["mappings"]

pdf_results = {}
comp_result = {}
for pdf_file in args.pdfFiles:
    pdf_name = pdf_file.split("/")[-2].split("_")[-1]
    pdf_name = pdf_name.upper().replace("AN3LO", "an3lo")

    logger.info(f"Now at PDF {pdf_name}")

    pdf_result, pdf_meta = rabbit.io_tools.get_fitresult(pdf_file, meta=True)

    pdf_mapping = pdf_result["mappings"]

    if "CompositeMapping" in pdf_mapping.keys():
        comp_result[pdf_name] = pdf_mapping["CompositeMapping"]
    else:
        pdf_results[pdf_name] = pdf_mapping

nPDFs = len(args.pdfFiles)
pdf_colors = {
    "CT18": "#2ca02c",
    "CT18Z": "#E42536",
    "PDF4LHC21": "#9467bd",
    "MSHT20": "#7f7f7f",
    "MSHT20an3lo": "#8c564b",
    "NNPDF31": "#e377c2",
    "NNPDF40": "#17becf",
}

# mapping, channel, bin
xsec_keys = [
    (r"$\mathrm{W}^{-}$", "Project ch1_masked qGen", "ch1_masked", {"qGen": 0}),
    (r"$\mathrm{W}^{+}$", "Project ch1_masked qGen", "ch1_masked", {"qGen": 1}),
    (r"$\mathrm{W}$", "Project ch1_masked", "ch1_masked", None),
    (r"$\mathrm{Z}$", "Project ch0_masked", "ch0_masked", None),
    (
        r"$\mathrm{W}^{+}/\mathrm{W}^{-}$",
        "Ratio ch1_masked ch1_masked qGen:1,ptGen:sum,absEtaGen:sum qGen:0,ptGen:sum,absEtaGen:sum",
        "ch1_masked",
        None,
    ),
    (
        r"$\mathrm{W/Z}$",
        "Ratio ch1_masked ch0_masked qGen:sum,ptGen:sum,absEtaGen:sum ptVGen:sum,absYVGen:sum",
        "ch1_masked_ch0_masked",
        None,
    ),
]

lumi = meta["meta_info_input"]["channel_info"]["ch0"]["lumi"]

custom_order = [
    "Total",
    "stat",
    "binByBinStat",
    "luminosity",
    "Fake",
    "CMS_background",
    "muon_eff_syst",
    "muon_eff_stat",
    "prefire",
    "muonCalibration",
    "recoil",
    "pdfCT18Z",
    "angularCoeffs",
    "pTModeling",
    "theory_ew",
    "massAndWidths",
]

dfs = []
for name, mapping, channel, selection in xsec_keys:
    hp = result[mapping]["channels"][channel]["hist_prefit_inclusive"].get()
    h1 = result[mapping]["channels"][channel]["hist_postfit_inclusive"].get()
    hi = result[mapping]["channels"][channel][
        "hist_postfit_inclusive_global_impacts_grouped"
    ].get()
    if selection is not None:
        hp = hp[selection]
        h1 = h1[selection]
        hi = hi[selection]
    if getattr(h1, "axes", False) and "yield" in h1.axes.name:
        hp = hp[{"yield": hist.sum}]
        h1 = h1[{"yield": hist.sum}]
        hi = hi[{"yield": hist.sum}]

    prefit = hp.value
    prefit_error = hp.variance**0.5

    value = h1.value
    error = h1.variance**0.5

    impacts = hi.values()

    labels = np.array(hi.axes["impacts"])
    mask = np.isin(labels, grouping)

    labels = labels[mask]
    impacts = impacts[mask]

    if np.sum(impacts**2) ** 0.5 / error - 1 > 10e-10:
        raise RuntimeError(
            f"Sources don't add up to total error, got a difference of {np.sum(impacts**2)**0.5/error - 1}"
        )

    labels = np.append(labels, "Total")
    impacts = np.append(impacts, error)

    df = pd.DataFrame(np.array(impacts, dtype=np.float64).T, columns=["impact"])

    df["label"] = labels

    for label, combine_labels in {
        "binByBinStat": ["binByBinStatW", "binByBinStatZ", "binByBinStat"],
        "theory_ew": ["theory_ew", "massShift", "sin2thetaZ", "widthW", "widthZ"],
    }.items():

        subset = df[df["label"].isin(combine_labels)]
        subset = subset.fillna(0)
        combined = np.sqrt((subset[["impact"]] ** 2).sum())

        # df = df.drop(columns=["binByBinStatW", "binByBinStatZ", "massShift", "sin2thetaZ", "widthW", "widthZ"])

        new_row = pd.DataFrame(
            {
                "label": [label],
                "impact": [combined["impact"]],
            }
        )

        # Remove old rows and append the new one
        df = df[~df["label"].isin(combine_labels)]
        df = pd.concat([df, new_row], ignore_index=True)

    df["name"] = name
    df["value"] = value

    df["prefit"] = prefit
    df["prefit_error"] = prefit_error

    for pdf_name, pdf_res in pdf_results.items():
        channel_mappings = pdf_res[mapping.replace("_masked", "")]["channels"][
            channel.replace("_masked", "")
        ]
        hr = channel_mappings["hist_prefit_inclusive"].get()
        hr_impacts = channel_mappings[
            "hist_prefit_inclusive_global_impacts_grouped"
        ].get()

        if selection is not None:
            hr = hr[selection]
            hr_impacts = hr_impacts[selection]
        if getattr(hr, "axes", False) and "yield" in hr.axes.name:
            hr = hr[{"yield": hist.sum}]
            hr_impacts = hr_impacts[{"yield": hist.sum}]

        df[pdf_name] = hr.value
        df[f"{pdf_name}_error"] = hr.variance**0.5
        df[f"{pdf_name}_pdf"] = hr_impacts[{"impacts": f"pdf{pdf_name}"}]

    # Convert 'labels' column to categorical with the custom order
    df["label"] = pd.Categorical(df["label"], categories=custom_order, ordered=True)

    df["source"] = df["label"].apply(lambda l: translate_label.get(l, l))

    df = df.sort_values("label", ascending=False)

    dfs.append(df)

df = pd.concat(dfs)

names = [k[0] for k in xsec_keys]

outdir = output_tools.make_plot_dir(args.outpath, eoscp=args.eoscp)

# make latex table
outname = "summary_table"
if args.postfix:
    outname += f"_{args.postfix}"

df_t = df.copy()
relative = True  # compute relative uncertainty
percentage = True  # numbers in percentage

if relative:
    df_t["impact"] /= df_t["value"]
if percentage:
    df_t["impact"] *= 100

# sorting
cat_dtype = pd.CategoricalDtype(categories=names, ordered=True)
df_t["name"] = df_t["name"].astype(cat_dtype)

tex_tools.make_latex_table(
    df_t,
    output_dir=outdir,
    output_name=outname,
    column_title=None,
    caption="Uncertainties in percentage.",
    label="",
    sublabel="",
    column_name="name",
    row_name="source",
    cell_columns=["impact"],
    cell_format=lambda x: f"${round(x,2)}$",
    sort="impact",
)

# make plot 1D
hep.style.use(hep.style.ROOT)

plt.clf()
fig = plt.figure()
fig.subplots_adjust(left=0.15, right=0.99, top=0.99, bottom=0.125)
ax = fig.add_subplot(111)

# x axis range
lo, hi = 0.925, 1.085

# totals = []
# stats = []
norms = []
for i, name in enumerate(names[::-1]):
    df_g = df.loc[df["name"] == name]

    norm = df_g["value"].values[0]
    total = df_g.loc[df_g["label"] == "Total"]["impact"].values[0]
    stat = df_g.loc[df_g["label"] == "stat"]["impact"].values[0]
    total_rel = total / norm
    stat_rel = stat / norm

    prefit = df_g["prefit"].values[0] / norm
    prefit_err = df_g["prefit_error"].values[0] / norm

    norms.append(norm)
    # totals.append(total)
    # stats.append(stat)

    x1 = ax.bar(
        1.0, height=1, bottom=i, width=2 * total_rel, color="silver"  # , label="Total"
    )
    x2 = ax.bar(
        1.0, height=1, bottom=i, width=2 * stat_rel, color="gold"  # , label="Stat"
    )

    # ax.errorbar([prefit], [i+0.5], xerr=prefit_err, color="red", marker="o", label="Prefit" if i ==0 else None)

    for j, pdf_name in enumerate(pdf_results.keys()):
        pdf_value = df_g[pdf_name].values[0] / norm
        pdf_error = df_g[f"{pdf_name}_error"].values[0] / norm
        ax.errorbar(
            [pdf_value],
            [i + 1 - (j + 1) / (nPDFs + 1)],
            xerr=pdf_error,
            color=pdf_colors[pdf_name],
            marker="o",
            label=pdf_name if i == 0 else None,
        )
        # # only plot PDF uncertainties
        # pdf_error_pdf = df_g[f"{pdf_name}_pdf"].values[0] / norm
        # ax.errorbar(
        #     [pdf_value],
        #     [i + 1 - (j + 1) / (nPDFs + 1)],
        #     xerr=pdf_error_pdf,
        #     color=pdf_colors[pdf_name],
        #     capsize=5,
        #     capthick=2,
        #     marker="o",
        # )

    # round to two significant digits in total uncertainty
    sig_digi = 2 - int(math.floor(math.log10(abs(total)))) - 1

    if sig_digi <= 0:
        norm = int(norm)
        total = int(total)
    else:
        norm = round(norm, sig_digi)
        total = round(total, sig_digi)

    ax.text(
        lo + 0.005,
        i + 0.5,
        name,
        fontsize=20,
        verticalalignment="bottom",
        horizontalalignment="left",
    )
    title = rf"${norm} \pm {total}"
    if "/" in name:
        title += "$"
    else:
        title += r"\,\mathrm{pb}$"
    ax.text(
        hi - 0.04,
        i + 0.5,
        title,
        fontsize=20,
        verticalalignment="bottom",
        horizontalalignment="left",
    )

# ax.text(
#     hi - 0.04,
#     len(names) + 0.5,
#     r"$\mathrm{Measured} \pm {unc}$",
#     fontsize=20,
#     verticalalignment="bottom",
#     horizontalalignment="left",
# )

x0 = ax.plot([1.0, 1.0], [0, len(norms)], color="black")
ax.plot([lo, hi], [len(norms), len(norms)], color="black")

p = Polygon(
    [[0, 0], [0, 0], [0, 0], [0, 0]],
    facecolor="silver",
    linestyle="solid",
    edgecolor="black",
    linewidth=2,
    alpha=0.6,
)

p.outer_color = "grey"
p.outer_alpha = 0.5
p.inner_color = "gold"
p.inner_alpha = 1

extra_handles = [(p,)]

extra_labels = ["Measurement"]

plot_tools.addLegend(
    ax,
    ncols=args.legCols,
    text_size="small",
    bbox_to_anchor=None,
    loc="upper left",
    reverse=False,
    markerfirst=True,
    labelcolor="black",
    extra_handles=extra_handles,
    extra_labels=extra_labels,
    extra_entries_first=False,
    custom_handlers=(["doubleband"]),
    padding_loc="auto",
)

ax.set_xlim([lo, hi])
ax.set_ylim([0, len(norms) + 2])

ax.set_xlabel("Prediction / Measurement")

# Disable ticks on the top and right axes
ax.tick_params(top=False)

# Disable y-axis labels and ticks
plt.gca().set_yticklabels([])
plt.gca().set_yticks([])

plot_tools.add_cms_decor(ax, args.cmsDecor, data=True, lumi=lumi, loc=args.logoPos)

outname = "summary"
if args.postfix:
    outname += f"_{args.postfix}"
plot_tools.save_pdf_and_png(outdir, outname)

output_tools.write_index_and_log(
    outdir,
    outname,
    analysis_meta_info={"CombinetfOutput": meta["meta_info"]},
    args=args,
)


# make plot 2D ellipses


def plot_cov_ellipse(cov, pos, nstd=1, **kwargs):
    """
    Plots an `nstd` sigma ellipse based on the covariance matrix (`cov`)
    centered at position `pos`.
    """
    # Eigenvalue decomposition
    eigvals, eigvecs = np.linalg.eigh(cov)

    # Sort eigenvalues and eigenvectors
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    # Compute angle in degrees
    theta = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))

    # Width and height are "2*nstd" standard deviations
    width, height = 2 * nstd * np.sqrt(eigvals)

    # Create ellipse
    ellipse = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)
    return ellipse


for name, channel0, channel1, unit in (
    ("WpWm", xsec_keys[0], xsec_keys[1], "pb"),
    ("WZ", xsec_keys[2], xsec_keys[3], "pb"),
    ("R", xsec_keys[4], xsec_keys[5], None),
):

    plt.clf()
    fig = plt.figure()
    fig.subplots_adjust(left=0.15, right=0.99, top=0.99, bottom=0.125)
    ax = fig.add_subplot(111)

    for pdf_name, result in comp_result.items():
        ckey0 = channel0[1] + " " + channel0[2]
        ckey1 = channel1[1] + " " + channel1[2]

        ibin = 0
        for k, r in result["channels"].items():
            fittype = "postfit" if f"hist_postfit_inclusive" in r.keys() else "prefit"

            hi = r[f"hist_{fittype}_inclusive"].get()
            if getattr(hi, "axes", False) and "yield" in hi.axes.name:
                hi = hi[{"yield": hist.sum}]

            if k == ckey0 or k == ckey0.replace("_masked", ""):
                sel = channel0[-1]

                if sel is not None:
                    x = hi[sel].value
                    ix = ibin + [i for i in sel.values()][0]
                else:
                    x = hi.value
                    ix = ibin

            if k == ckey1 or k == ckey1.replace("_masked", ""):
                sel = channel1[-1]
                if sel is not None:
                    y = hi[sel].value
                    iy = ibin + [i for i in sel.values()][0]
                else:
                    y = hi.value
                    iy = ibin

            ibin += hi.size if hasattr(hi, "size") else 1

        cov = result[f"hist_{fittype}_inclusive_cov"].get().values()
        cov = cov[np.ix_([ix, iy], [ix, iy])]

        # for pos, cov in zip(points, covs):
        if fittype == "postfit":
            ell = plot_cov_ellipse(
                cov,
                np.array([x, y]),
                nstd=1,
                edgecolor="none",
                facecolor="grey",
                label="Measurement",
            )
            ax.add_patch(ell)
            ax.plot(x, y, color="black", marker="P")
        else:
            icol = pdf_colors[pdf_name]
            ell = plot_cov_ellipse(
                cov,
                np.array([x, y]),
                nstd=1,
                edgecolor=icol,
                facecolor="none",
                linewidth=2,
                label=pdf_name,
            )
            ax.add_patch(ell)
            ax.plot(x, y, color=icol, marker="o", alpha=0)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    yrange = ylim[1] - ylim[0]

    ax.set_xlim(*xlim)
    ax.set_ylim(ylim[0], ylim[1] + yrange * 0.25)
    if unit is not None:
        plt.xlabel(rf"$\sigma({channel0[0].replace("$","")})$ [pb]")
        plt.ylabel(rf"$\sigma({channel1[0].replace("$","")})$ [pb]")
    else:
        plt.xlabel(channel0[0])
        plt.ylabel(channel1[0])
    # plt.title("2D Covariance Ellipses")
    # plt.grid(True)
    # plt.show()

    plot_tools.addLegend(
        ax,
        ncols=args.legCols,
        text_size="small",
        bbox_to_anchor=None,
        loc="upper left",
        reverse=False,
        # markerfirst=True,
        labelcolor="black",
        # extra_handles=extra_handles,
        # extra_labels=extra_labels,
        # extra_entries_first=False,
        # custom_handlers=(
        #     ["doubleband"]
        # ),
        padding_loc="auto",
    )

    plot_tools.add_cms_decor(ax, args.cmsDecor, data=True, loc=args.logoPos)

    outname = f"summary_2D_{name}"
    if args.postfix:
        outname += f"_{args.postfix}"
    plot_tools.save_pdf_and_png(outdir, outname)

    output_tools.write_index_and_log(
        outdir,
        outname,
        analysis_meta_info={"CombinetfOutput": meta["meta_info"]},
        args=args,
    )


if output_tools.is_eosuser_path(args.outpath) and args.eoscp:
    output_tools.copy_to_eos(args.outpath, args.outfolder)
