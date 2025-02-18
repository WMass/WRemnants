import argparse
import copy
import os

import h5py
import hist
import numpy as np
from matplotlib import cm

import wums.ioutils
from utilities.differential import get_theoryAgnostic_axes
from utilities.io_tools import input_tools, output_tools
from wremnants import plot_tools, theory_tools
from wremnants.helicity_utils import axis_helicity_multidim
from wums import boostHistHelpers as hh

xlabels = {
    "pt": r"p$_{T}^{\ell}$ (GeV)",
    "eta": r"$\eta^{\ell}$",
    "unrolled": r"(p$_{T}^{\ell}$, $\eta^{\ell}$) bin",
    "unrolled_gen": r"($|\mathrm{y}^{Z}|$,p$_{T}^{Z}$) bin",
    "unrolled_gen_hel": r"unrolled ($|\mathrm{y}^{Z}|$,p$_{T}^{Z}$) bin",
    "ptVgen": r"p$_{T}^{Z}$ (GeV)",
    "absYVgen": r"$|\mathrm{y}^{Z}|$",
    "ptll": r"p$_{\mathrm{T}}^{\ell\ell}$ (GeV)",
    "yll": r"y$^{\ell\ell}$",
    "mll": r"m$_{\ell\ell}$ (GeV)",
}

parser = argparse.ArgumentParser()
parser.add_argument(
    "infile", help="Output file of the analysis stage, containing ND boost histograms"
)
parser.add_argument(
    "--pdfs",
    type=str,
    nargs="+",
    help="List of histograms to plot",
    choices=theory_tools.pdfMap.keys(),
    required=True,
)
parser.add_argument(
    "-c",
    "--channel",
    type=str,
    choices=["plus", "minus", "all"],
    default="all",
    help="Select channel to plot",
)
parser.add_argument(
    "-p",
    "--outpath",
    type=str,
    default=os.path.expanduser("~/www/WMassAnalysis"),
    help="Base path for output",
)
parser.add_argument(
    "-f", "--outfolder", type=str, default="test", help="Subfolder for output"
)
parser.add_argument(
    "-r",
    "--rrange",
    type=float,
    nargs=2,
    default=[0.9, 1.1],
    help="y range for ratio plot",
)
parser.add_argument(
    "-d", "--datasets", type=str, nargs="+", help="Dataset to plot", required=True
)
parser.add_argument(
    "--obs",
    type=str,
    nargs="+",
    choices=xlabels.keys(),
    help="Observable to plot",
    required=True,
)
parser.add_argument("--together", action="store_true", help="y range for ratio plot")
parser.add_argument("--baseName", type=str, help="Name of nominal hist")
parser.add_argument(
    "--ymax",
    type=float,
    help="Max value for y axis (if not specified, range set automatically)",
)
args = parser.parse_args()

for pdf in args.pdfs:
    if pdf not in theory_tools.pdfMap:
        raise ValueError(
            f"pdf {pdf} is not a valid hist (not defined in theory_tools.pdfMap)"
        )
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

    axis_label = "pdfVar"

    uncType = [pdfInfo[pdf]["combine"] for pdf in args.pdfs]
    uncScale = [
        pdfInfo[pdf]["scale"] if "scale" in pdfInfo[pdf] else 1.0 for pdf in args.pdfs
    ]
    names = [[pdfName + r" $\pm1\sigma$", "", ""] for pdfName in pdfNames]
    cmap = cm.get_cmap("tab10")
    colors = [[cmap(i)] * 3 for i in range(len(args.pdfs))]

    if "unrolled_gen_hel" in args.obs:
        print(dataset)
        moments = input_tools.read_all_and_scale(
            args.infile, [dataset], [f"{args.baseName}_helicity_xsecs_scale"]
        )
        coeffs = moments[0].project(
            "ptVgen", "absYVgen", "helicity", "muRfact", "muFfact"
        )

        moments_lhe = input_tools.read_all_and_scale(
            args.infile, [dataset], [f"{args.baseName}_helicity_xsecs_scale_lhe"]
        )
        hel_lhe = moments_lhe[0].project(
            "ptVgen", "absYVgen", "helicity", "muRfact", "muFfact"
        )
        # ratioUL = hh.divideHists(coeffs[{'helicity':-1.j,'muRfact':1.j,'muFfact':1.j}],hel_lhe[{'helicity':-1.j,'muRfact':1.j,'muFfact':1.j}])
        # coeffs_lhe=hh.multiplyHists(hel_lhe,ratioUL)
        coeffs_lhe = hel_lhe

        moments_pdf = input_tools.read_all_and_scale(
            args.infile,
            [dataset],
            [f"{args.baseName}_helicity_{pdfName}" for pdfName in pdfNames],
        )

        coeffs_pdf = []
        for moment_pdf in moments_pdf:
            hel_pdf = moment_pdf.project("ptVgen", "absYVgen", "helicity", axis_label)
            # ratioUL = hh.divideHists(coeffs[{'helicity':-1.j,'muRfact':1.j,'muFfact':1.j}],hel_pdf[{'helicity':-1.j}])
            # coeffs_pdf.append(hh.multiplyHists(hel_pdf,ratioUL))
            coeffs_pdf.append(hel_pdf)

        uncHists = [
            [h[{axis_label: 0}], *theory_tools.hessianPdfUnc(h, axis_label, unc, scale)]
            for h, unc, scale in zip(coeffs_pdf, uncType, uncScale)
        ]

        coeffs_heraext = []
        moments_heraext = input_tools.read_all_and_scale(
            args.infile, [dataset], [f"{args.baseName}_helicity_pdfHERAPDF20ext"]
        )
        hel_heraext = moments_heraext[0].project(
            "ptVgen", "absYVgen", "helicity", axis_label
        )
        # ratioUL = hh.divideHists(coeffs[{'helicity':-1.j,'muRfact':1.j,'muFfact':1.j}],hel_heraext[{'helicity':-1.j}])
        # coeffs_heraext.append(hh.multiplyHists(hel_heraext,ratioUL))
        coeffs_heraext.append(hel_heraext)

        uncType_hera = [pdfInfo["herapdf20ext"]["combine"]]
        uncScale_hera = [
            (
                pdfInfo["herapdf20ext"]["scale"]
                if "scale" in pdfInfo["herapdf20ext"]
                else 1.0
            )
        ]
        uncHists_hera = [
            [h[{axis_label: 0}], *theory_tools.hessianPdfUnc(h, axis_label, unc, scale)]
            for h, unc, scale in zip(coeffs_heraext, uncType_hera, uncScale_hera)
        ]

        for ipdf, pdf in enumerate(args.pdfs):
            if "herapdf20" in pdf:
                nom = uncHists[ipdf][0]
                up = uncHists[ipdf][1]
                down = uncHists[ipdf][2]

                up[...] = nom.values() + np.sqrt(
                    np.square(up.values() - nom.values())
                    + np.square(uncHists_hera[0][1].values() - nom.values())
                )

                down[...] = nom.values() - np.sqrt(
                    np.square(down.values() - nom.values())
                    + np.square(uncHists_hera[0][2].values() - nom.values())
                )

                uncHists[ipdf][1] = up
                uncHists[ipdf][2] = down

        # add alphaS
        alphaNames = []
        coeffs_alpha = []
        axis_label_alpha = "alphasVar"
        for ipdf, pdf in enumerate(args.pdfs):

            has_as = "alphasRange" in pdfInfo[pdf]
            if has_as:
                asr = pdfInfo[pdf]["alphasRange"]
                scale_alpha = 0.75 if asr == "002" else 1.5
                alphaNames.append(
                    f"{args.baseName}_helicity_{args.baseName}_helicity_{pdfNames[ipdf]}alphaS{asr}"
                )
                # nominal_gen_pdfMSHT20alphaS002
                moments_alpha = input_tools.read_all_and_scale(
                    args.infile, [dataset], alphaNames
                )

                hel_alpha = moments_alpha[0].project(
                    "ptVgen", "absYVgen", "helicity", "alphasVar"
                )
                # ratioUL = hh.divideHists(coeffs[{'helicity':-1.j,'muRfact':1.j,'muFfact':1.j}],scale_alpha*hel_alpha[{'helicity':-1.j}])
                # coeffs_alpha.append(hh.multiplyHists(hel_alpha,ratioUL))
                coeffs_alpha.append(hel_alpha)
            uncHists[ipdf].extend(
                [
                    uncHists[ipdf][0]
                    + (coeffs_alpha[ipdf][..., 1] - uncHists[ipdf][0]) * scale_alpha,
                    uncHists[ipdf][0]
                    + (coeffs_alpha[ipdf][..., 2] - uncHists[ipdf][0]) * scale_alpha,
                ]
            )
            names[ipdf].extend([pdfNames[ipdf] + r"alpha $\pm1\sigma$", ""])

            colors[ipdf].extend([cmap(ipdf + 1)] * 2)

        # add QCD scales
        uncHists.append(
            [
                coeffs[{"muRfact": 1.0j, "muFfact": 1.0j}],
                *[
                    coeffs[{"muRfact": 2.0j, "muFfact": 2.0j}],
                    coeffs[{"muRfact": 0.5j, "muFfact": 0.5j}],
                    coeffs[{"muRfact": 2.0j, "muFfact": 1.0j}],
                    coeffs[{"muRfact": 0.5j, "muFfact": 1.0j}],
                    coeffs[{"muRfact": 1.0j, "muFfact": 2.0j}],
                    coeffs[{"muRfact": 1.0j, "muFfact": 0.5j}],
                ],
            ]
        )
        names.append(["QCDscale_central", "", "", "", "", "", ""])
        colors.extend(
            [[cmap(i)] * 7 for i in range(len(args.pdfs) + 1, len(args.pdfs) + 2)]
        )

        uncHists.append(
            [
                coeffs_lhe[{"muRfact": 1.0j, "muFfact": 1.0j}],
                *[
                    coeffs_lhe[{"muRfact": 2.0j, "muFfact": 2.0j}],
                    coeffs_lhe[{"muRfact": 0.5j, "muFfact": 0.5j}],
                    coeffs_lhe[{"muRfact": 2.0j, "muFfact": 1.0j}],
                    coeffs_lhe[{"muRfact": 0.5j, "muFfact": 1.0j}],
                    coeffs_lhe[{"muRfact": 1.0j, "muFfact": 2.0j}],
                    coeffs_lhe[{"muRfact": 1.0j, "muFfact": 0.5j}],
                ],
            ]
        )
        names.append(["QCDscalelhe_central", "", "", "", "", "", ""])
        colors.extend(
            [[cmap(i)] * 7 for i in range(len(args.pdfs) + 2, len(args.pdfs) + 3)]
        )

    outdir = output_tools.make_plot_dir(args.outpath, args.outfolder)
    plot_names = copy.copy(args.pdfs)
    plot_names.append("QCD_scales")
    plot_names.append("QCD_scales_lhe")

    theoryAgnostic_axes, _ = get_theoryAgnostic_axes(
        ptV_flow=True, absYV_flow=True, wlike="Z" in dataset
    )
    axis_ptV = theoryAgnostic_axes[0]
    axis_yV = theoryAgnostic_axes[1]

    variations = np.zeros((len(axis_ptV.centers) + 1, len(axis_yV.centers) + 1, 9))
    print(variations.shape)
    for ihel in coeffs.axes["helicity"].edges[:-1]:
        for obs in args.obs:
            all_hists = []
            all_hists_list = []
            all_colors = []
            all_colors_list = []
            all_names = []
            for name, color, labels, hists in zip(plot_names, colors, names, uncHists):
                # This is the reference
                if not "unrolled" in obs:
                    action = lambda x: x.project(obs)
                    hists1D = [action(x) for x in hists]
                else:
                    obs2unroll = (
                        ["ptVgen", "absYVgen"]
                        if "unrolled_gen" in obs
                        else ["pt", "eta"]
                    )
                    action = hh.unrolledHist
                    if not "hel" in obs:
                        hists1D = [action(x, obs2unroll, binwnorm=True) for x in hists]
                    else:
                        hists1D = [
                            action(
                                x[{"helicity": ihel * 1.0j}],
                                obs2unroll,
                                binwnorm=True,
                                add_flow_bins=True,
                            )
                            for x in hists
                        ]

                all_hists.extend(hists1D)
                all_hists_list.append(hists1D)
                all_colors.extend(color)
                all_colors_list.append(color)
                all_names.extend(labels)

            lower = np.minimum.reduce(
                [h.values(flow=True) for h in all_hists]
            ) / np.abs(all_hists[0].values(flow=True))
            upper = np.maximum.reduce(
                [h.values(flow=True) for h in all_hists]
            ) / np.abs(all_hists[0].values(flow=True))

            symm = np.where(np.abs(lower - 1.0) > np.abs(upper - 1.0), lower, upper)
            if ihel == -1:
                symm = 0.5 * np.ones_like(symm)
            elif ihel == 4:
                symm = 2 * np.ones_like(symm)

            fig = plot_tools.makePlotWithRatioToRef(
                all_hists,
                colors=all_colors,
                labels=all_names,
                alpha=0.4,
                rrange=args.rrange,
                ylabel=r"$\sigma$/bin",
                xlabel=xlabels[obs],
                rlabel=f"x/{args.pdfs[0].upper()}",
                binwnorm=None,
                nlegcols=1,
                only_ratio=False,
                width_scale=2,
            )
            outfile = f"{name}Hist_{obs}_{dataset}_sigma{round(ihel)}"
            ax1, ax2 = fig.axes

            for igroup, hgroup in enumerate(all_hists_list):
                hvalues = [h.values() for h in hgroup]
                max_envelope = np.max(np.array(hvalues), axis=0)
                min_envelope = np.min(np.array(hvalues), axis=0)
                ax1.fill_between(
                    all_hists[0].axes[0].centers,
                    min_envelope,
                    max_envelope,
                    color=all_colors_list[igroup][0],
                    alpha=0.2,
                    label="Envelope",
                    step="mid",
                )
                ax2.fill_between(
                    all_hists[0].axes[0].centers,
                    min_envelope / np.abs(all_hists[0].values(flow=True)),
                    max_envelope / np.abs(all_hists[0].values(flow=True)),
                    color=all_colors_list[igroup][0],
                    alpha=0.2,
                    label="Envelope",
                    step="mid",
                )

            ax2.fill_between(
                all_hists[0].axes[0].centers,
                symm,
                2 - symm,
                color="grey",
                alpha=0.3,
                label="theory agnostic variation",
                hatch="//",
                step="mid",
            )
            ax2.set_xticklabels([])
            ax2.set_xticks([])
            min_val = np.min(np.concatenate((symm, 2 - symm)))
            max_val = np.max(np.concatenate((symm, 2 - symm)))

            if not ihel == -1:
                ax2.set_ylim(min_val, max_val)
            else:
                ax2.set_ylim(0, 2)
            # plot_tools.save_pdf_and_png(outdir, outfile)
            # plot_tools.write_index_and_log(outdir, outfile)

            vars = np.ones((len(axis_ptV.centers) + 1, len(axis_yV.centers) + 1))
            min_arr = np.minimum(symm, 2 - symm)
            max_arr = np.maximum(symm, 2 - symm)
            max_arr = max_arr - 1

            vars = max_arr.reshape(
                (len(axis_ptV.centers) + 1, len(axis_yV.centers) + 1)
            )
            vars[-1:, :] = 0.5 * np.ones_like(vars[-1:, :])
            vars[:, -1:] = 0.5 * np.ones_like(vars[:, -1:])

            variations[..., int(ihel) + 1] = vars

    hvariations = hist.Hist(
        axis_ptV,
        axis_yV,
        axis_helicity_multidim,
        name=f"theorybands_{dataset}",
        data=variations,
    )
    band_hists[dataset] = hvariations

outfile = "theoryband_variations_corr.hdf5"
with h5py.File(outfile, "w") as f:
    wums.ioutils.pickle_dump_h5py("theorybands", band_hists, f)
