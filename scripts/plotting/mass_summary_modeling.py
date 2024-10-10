import argparse

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import ticker

from utilities import common
from utilities.io_tools import combinetf_input, output_tools
from wremnants import plot_tools

parser = argparse.ArgumentParser()
parser = common.plot_parser()
parser.add_argument(
    "-r",
    "--reffile",
    required=True,
    type=str,
    help="Combine fitresult file for nominal result",
)
parser.add_argument("--print", action="store_true", help="Print results")

args = parser.parse_args()

basename = args.reffile

dfs = combinetf_input.read_all_groupunc_df(
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
        "SCETlib+DYTurbo N$^{3+0}$LL+NNLO",
        "SCETlib+DYTurbo N$^{3+1}$LL+NNLO",
        "SCETlib+DYTurbo N$^{4+0}$LL+NNLO",
        "$p_{T}^{\\ell\\ell}$ rwgt., N$^{3+0}$LL unc.",
    ],
    uncs=["standard_pTModeling"],
)

isW = "WMass" in args.reffile

if isW:
    combdf = combinetf_input.read_all_groupunc_df(
        [args.reffile.format(postfix="_CombinedPtll")],
        names=(
            [
                "Combined $p_{T}^{\\ell\\ell}$ fit, N$^{3+0}$LL unc.",
            ]
            if isW
            else []
        ),
        uncs=["standard_pTModeling"],
    )
    dfs = pd.concat((dfs, combdf), ignore_index=True)

outname = f"{'Wmass' if isW else 'Wlike'}_modeling_summary"
if args.postfix:
    outname += f"_{args.postfix}"

if isW:
    xlim = [80260, 80410]
else:
    xlim = [91160, 91280] if "flipEvenOdd" not in basename else [91170, 91290]

if args.print:
    for k, v in dfs.iterrows():
        print(v.iloc[0], round(v.iloc[1], 1), round(v.iloc[3], 1), round(v.iloc[2], 2))

central = dfs.iloc[0, :]

eoscp = output_tools.is_eosuser_path(args.outpath)
outdir = output_tools.make_plot_dir(args.outpath, args.outfolder, eoscp=eoscp)

fig = plot_tools.make_summary_plot(
    central["value"],
    central["err_standard_pTModeling"],
    "Nominal result",
    dfs.iloc[1:, :],
    colors="auto",
    xlim=xlim,
    xlabel="$m_{W}$ (MeV)" if isW else "$m_{Z}$ (MeV)",
    legend_loc="upper left",
    legtext_size="small",
    cms_loc=0,
)
ax = plt.gca()
ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
ax.xaxis.set_major_locator(ticker.MultipleLocator(40))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(20))
ax.xaxis.grid(False, which="both")
ax.yaxis.grid(False, which="both")
plot_tools.save_pdf_and_png(outdir, outname, fig)
plot_tools.write_index_and_log(outdir, outname)
if eoscp:
    output_tools.copy_to_eos(outdir, args.outpath, args.outfolder)
