import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from scipy.stats import chi2

from utilities import parsing
from utilities.io_tools import input_tools, output_tools
from utilities.styles import styles
from wremnants import plot_tools, theory_tools
from wums import boostHistHelpers as hh
from wums import logging

if __name__ == "__main__":
    parser = parsing.plot_parser()
    parser.add_argument(
        "helicities",
        nargs="+",
        type=str,
        help="File `w_z_helicity_xsecs.hdf` with helicity cross sections produced in w_z_gen_dists.py histmaker",
    )
    parser.add_argument(
        "--labels", nargs="+", type=str, help="Labels for the different input files"
    )
    parser.add_argument(
        "--keys",
        nargs="+",
        type=str,
        default=["", "_lhe"],
        help="List of histogram keys to be loaded; For a given KEY he object '{PROCESS}{KEY}' will be searched",
    )
    parser.add_argument(
        "--process", default="Z", choices=["Z", "W"], help="Process to be plotted"
    )
    parser.add_argument(
        "--plotXsecs",
        action="store_true",
        help="Plot helicity cross sections instead of angular coefficients",
    )
    parser.add_argument(
        "--plotSum",
        action="store_true",
        help="Plot sum of histograms from different input files",
    )

    args = parser.parse_args()
    logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)

    outdir = output_tools.make_plot_dir(args.outpath, args.outfolder, eoscp=args.eoscp)

    linestyles = ["solid", "dashed"]
    suffixes = {"": "MiNNLO(pythia)", "lhe": "MiNNLO(LHE)"}

    helicities = {}
    helicity_sum = {}
    for helicity_file in args.helicities:
        with h5py.File(helicity_file, "r") as ff:
            out = input_tools.load_results_h5py(ff)
        for key in out.keys():
            if not key.startswith(args.process) or key in ["meta_info"]:
                continue
            if key.replace(args.process, "") not in args.keys:
                continue
            if key not in helicities:
                helicities[key] = []
                if args.plotSum:
                    helicity_sum[key] = None
            hhelicity = out[key]
            hhelicity = hhelicity[
                {"muRfact": 1j, "muFfact": 1j, "chargeVgen": 0, "massVgen": 0}
            ]
            hhelicity = hh.disableFlow(hhelicity, "absYVGen")

            if len(args.helicities) > 1 and args.plotSum:
                if helicity_sum[key] is None:
                    helicity_sum[key] = hhelicity.copy()
                else:
                    helicity_sum[key] = hh.addHists(helicity_sum[key], hhelicity)

            helicities[key].append(hhelicity)

    if args.plotXsecs:
        weight_sum_all = 0
        for i, helicity_file in enumerate(args.helicities):
            weight_sum = 0
            xsec = 0
            with h5py.File(
                helicity_file.replace("helicity_xsecs", "gen_dists"), "r"
            ) as ff:
                out = input_tools.load_results_h5py(ff)
            procs = [k for k in out.keys() if k.startswith(args.process)]
            if len(procs) > 1:
                logger.warning(
                    f"Number of matched processes should be 1 but found {len(procs)}"
                )
            xsec = out[procs[0]]["dataset"]["xsec"]
            weight_sum = out[procs[0]]["weight_sum"]
            weight_sum_all += weight_sum

            for hkey in helicities.keys():
                helicities[hkey][i] = hh.scaleHist(
                    helicities[hkey][i], xsec / weight_sum
                )

        for hkey in helicity_sum.keys():
            if helicity_sum.get(hkey, None) is not None:
                # under the assumption that xsec is the same for all processes
                helicity_sum[hkey] = hh.scaleHist(
                    helicity_sum[hkey], xsec / weight_sum_all
                )

        for k, h in helicities.items():
            if len(h) == 2:
                chi2_stat = np.sum(
                    (h[0].values(flow=True) - h[1].values(flow=True)) ** 2
                    / (h[0].variances(flow=True) + h[1].variances(flow=True))
                )
                dof = len(h[0].values(flow=True).flatten()) - 1
                p_val = chi2.sf(chi2_stat, dof)

                logger.info(f"For {k}; Chi2/ndf = {chi2_stat} with p-value = {p_val}")

    colors = mpl.colormaps["tab10"]

    for var in ("absYVGen", "ptVGen"):

        hists1d = {
            k: [h.project(var, "helicity") for h in v] for k, v in helicities.items()
        }
        if len(helicity_sum):
            h1d_sum = {k: v.project(var, "helicity") for k, v in helicity_sum.items()}

        if not args.plotXsecs:
            hists1d = {
                k: [theory_tools.helicity_xsec_to_angular_coeffs(h) for h in v]
                for k, v in hists1d.items()
            }
            if len(helicity_sum):
                h1d_sum = {
                    k: theory_tools.helicity_xsec_to_angular_coeffs(v)
                    for k, v in h1d_sum.items()
                }

        for i in next(iter(hists1d.values()))[0].axes["helicity"]:
            if i == -1 and not args.plotXsecs:
                continue

            h1ds = {
                k: [h[{"helicity": complex(0, i)}] for h in v]
                for k, v in hists1d.items()
            }

            ylabel = (
                rf"$\mathrm{{A}}_{i}$"
                if not args.plotXsecs
                else r"$\sigma_{"
                + (r"\mathrm{UL}" if i == -1 else str(i))
                + r"}\,[\mathrm{pb}]$"
            )
            fig, ax1 = plot_tools.figure(
                next(iter(h1ds.values()))[0],
                xlabel=styles.xlabels.get(var, var),
                ylabel=ylabel,
                grid=False,
                automatic_scale=False,
                width_scale=1.2,
                logy=False,
            )

            y_min = np.inf
            y_max = -np.inf

            if args.plotSum:
                for m, (k, hs_sum) in enumerate(h1d_sum.items()):
                    suffix = suffixes[k.replace(args.process, "").replace("_", "")]
                    if len(helicity_sum):
                        h = hs_sum[{"helicity": complex(0, i)}]
                        hep.histplot(
                            h,
                            histtype="step",
                            color="black",
                            label=suffix,
                            linestyle=linestyles[m],
                            yerr=False,
                            ax=ax1,
                            zorder=3,
                            density=False,
                            binwnorm=1 if args.plotXsecs else None,
                        )
                        y_min = min(y_min, min(h.values()))
                        y_max = max(y_max, max(h.values()))
            else:
                for m, (k, hs) in enumerate(h1ds.items()):
                    suffix = suffixes[k.replace(args.process, "").replace("_", "")]
                    for j, h in enumerate(hs):
                        hep.histplot(
                            h,
                            histtype="step",
                            color=colors(j),
                            label=f"{args.labels[j]} {suffix}",
                            linestyle=linestyles[m],
                            yerr=False,
                            ax=ax1,
                            zorder=3,
                            density=False,
                            binwnorm=1 if args.plotXsecs else None,
                        )
                        y_min = min(y_min, min(h.values()))
                        y_max = max(y_max, max(h.values()))

            if args.plotXsecs:
                y_min, y_max = ax1.get_ylim()
            yrange = y_max - y_min

            x_min, x_max = ax1.get_xlim()
            plt.plot([x_min, x_max], [0, 0], color="black", linestyle="--")

            ax1.set_ylim([y_min - yrange * 0.1, y_max + yrange * 0.2])

            plot_tools.add_cms_decor(
                ax1, args.cmsDecor, data=False, lumi=None, loc=args.logoPos
            )
            plot_tools.addLegend(
                ax1, ncols=args.legCols, loc=args.legPos, text_size=args.legSize
            )

            if args.plotXsecs:
                outfile = f"helicity_xsec_" + ("UL" if i == -1 else str(i))
            else:
                outfile = f"angular_coefficient_{i}"

            outfile += f"_{var}"
            if args.postfix:
                outfile += f"_{args.postfix}"
            plot_tools.save_pdf_and_png(outdir, outfile)
            plot_tools.write_index_and_log(outdir, outfile, args=args)

    if output_tools.is_eosuser_path(args.outpath) and args.eoscp:
        output_tools.copy_to_eos(outdir, args.outpath, args.outfolder)
