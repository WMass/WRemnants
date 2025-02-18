import combinetf2.io_tools
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import ticker

from utilities import parsing
from utilities.io_tools import hepdata_tools, output_tools
from wremnants import plot_tools

parser = parsing.plot_parser()
parser.add_argument(
    "-r",
    "--reffile",
    required=True,
    type=str,
    help="Combine fitresult file for nominal result",
)
parser.add_argument("--print", action="store_true", help="Print results")
parser.add_argument(
    "--diffToCentral", action="store_true", help="Show difference to central result"
)
parser.add_argument(
    "--saveForHepdata",
    action="store_true",
    help="Save output as ROOT to prepare HEPData",
)
args = parser.parse_args()

basename = args.reffile

dfs = combinetf2.io_tools.read_all_groupunc_df(
    [
        args.reffile.format(postfix=p)
        for p in [
            "",
            "_scetlib_dyturboN3p1LL",
            "_scetlib_dyturboN4p0LL",  # "_dyturboN3LLp",
            "_dataPtllRwgt",
        ]
    ],
    names=[
        # "SCETlib+DYTurbo N$^{3{+}0}$LL+NNLO",
        # "SCETlib+DYTurbo N$^{3{+}1}$LL+NNLO",
        # "SCETlib+DYTurbo N$^{4{+}0}$LL+NNLO",
        "N$^{3{+}0}$LL+NNLO",
        "N$^{3{+}1}$LL+NNLO",
        "N$^{4{+}0}$LL+NNLO",
        r"$\mathit{p}_{T}^{\ell\ell}$ rwgt.," + "\n N$^{3{+}0}$LL unc.",
    ],
    uncs=["standard_pTModeling"],
)

isW = "WMass" in args.reffile

if isW:
    combdf = combinetf2.io_tools.read_all_groupunc_df(
        [args.reffile.format(postfix="_CombinedPtll")],
        names=(
            [
                r"Combined $\mathit{p}_{T}^{\ell\ell}$ fit," + "\n N$^{3{+}0}$LL unc.",
            ]
            if isW
            else []
        ),
        uncs=["standard_pTModeling"],
    )
    dfs = pd.concat((dfs, combdf), ignore_index=True)

if isW:
    xlim = [80331, 80372]
else:
    xlim = [91160, 91280] if "flipEvenOdd" not in basename else [91170, 91290]

if args.print:
    for k, v in dfs.iterrows():
        print(v.iloc[0], round(v.iloc[1], 1), round(v.iloc[3], 1), round(v.iloc[2], 2))

central = dfs.iloc[0, :]

xlabel = r"$\mathit{m}_{" + ("W" if isW else "Z") + "}$ (MeV)"

central_val = central["value"]
if args.diffToCentral:
    if args.saveForHepdata:
        # save also the original absolute value
        dfs["absolute_value"] = dfs["value"].values
    dfs["value"] -= central_val
    xlim = [xlim[0] - central_val, xlim[1] - central_val]
    central_val = 0
    xlabel = r"$\Delta$" + xlabel

fig = plot_tools.make_summary_plot(
    central_val,
    central["err_total"],
    central["err_standard_pTModeling"],
    "N$^{3{+}0}$LL+NNLO\n (nominal)",
    dfs.iloc[1:, :],
    colors="auto",
    xlim=xlim,
    xlabel=xlabel,
    legend_loc="upper left",
    legtext_size="small",
    logoPos=0,
    cms_label=args.cmsDecor,
    lumi=16.8,
    padding=5,
)
ax = plt.gca()
ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
ax.xaxis.grid(False, which="both")
ax.yaxis.grid(False, which="both")

eoscp = output_tools.is_eosuser_path(args.outpath)
outdir = output_tools.make_plot_dir(args.outpath, args.outfolder, eoscp=eoscp)

outname = f"{'Wmass' if isW else 'Wlike'}_modeling_summary"
if args.postfix:
    outname += f"_{args.postfix}"

plot_tools.save_pdf_and_png(outdir, outname, fig)
plot_tools.write_index_and_log(outdir, outname)

if args.saveForHepdata:
    column_labels = [xlabel, "Total uncertainty", "Model uncertainty"]
    if args.diffToCentral:
        column_labels.append(xlabel.replace(r"$\Delta$", ""))

    hepdata_tools.make_mass_summary_histogram(
        dfs, f"{outdir}/{outname}.root", column_labels
    )

if eoscp:
    output_tools.copy_to_eos(outdir, args.outpath, args.outfolder)
