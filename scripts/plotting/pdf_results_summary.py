import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker

from utilities import parsing
from utilities.io_tools import hepdata_tools, rabbit_input
from wremnants import theory_tools
from wums import output_tools, plot_tools

parser = parsing.plot_parser()
parser.add_argument(
    "-r",
    "--reffile",
    required=True,
    type=str,
    help="Combine fitresult file for nominal result",
)
parser.add_argument(
    "-i",
    "--reffileinf",
    required=False,
    type=str,
    help="Combine fitresult file for inflated result",
)
# parser.add_argument("--pdfs", default=["ct18z", "ct18", "herapdf20",  "msht20",  "msht20an3lo",  "nnpdf31",  "nnpdf40",  "pdf4lhc21"],
parser.add_argument(
    "--pdfs",
    default=[
        "ct18z",
        "ct18",
        "pdf4lhc21",
        "msht20",
        "msht20an3lo",
        "nnpdf31",
        "nnpdf40",
    ],
    type=str,
    help="PDF to plot",
)
parser.add_argument(
    "--diffToCentral", action="store_true", help="Show difference to central result"
)
parser.add_argument(
    "--colors",
    type=str,
    nargs="+",
    help="Colors for PDFs",
    default=[
        "#E42536",
        "#2ca02c",
        "#9467bd",
        "#7f7f7f",
        "#8c564b",
        "#e377c2",
        "#17becf",
    ],
)
parser.add_argument("--print", action="store_true", help="Print results")
parser.add_argument(
    "--printAssym",
    action="store_true",
    help="Print results with asymmetric uncertainties",
)
parser.add_argument(
    "--impactType",
    choices=["traditional", "nonprofiled", "global"],
    default="traditional",
    help="Type of impacts to use for PDF group",
)
parser.add_argument(
    "--saveForHepdata",
    action="store_true",
    help="Save output as ROOT to prepare HEPData",
)
args = parser.parse_args()


def weave_vals(vals):
    return np.ravel(np.stack([list(v) for v in vals], axis=1))


isW = "WMass" in args.reffile

ref_mass = 80355 if isW else 91188
ref_unc = 6.0 if isW else 2.0

pdf_name = lambda p: theory_tools.pdfMap[p]["name"]

rename = {
    f"err_{pdf_name(pdf)}{suffix}": f"err_pdf{suffix}"
    for pdf in args.pdfs
    for suffix in ["", "_up", "_down"]
}

dfs = rabbit_input.read_all_groupunc_df(
    [args.reffile.format(pdf=pdf) for pdf in args.pdfs],
    rename_cols=rename,
    uncs=[pdf_name(pdf) for pdf in args.pdfs],
    names=[pdf_name(pdf)[3:] for pdf in args.pdfs],
    impact_type=args.impactType,
)

if args.print:
    if args.printAssym:
        print(
            "PDF           Mass PDF_unc_up PDF_unc_down  Prof. unc  Total_unc_up Total_unc_down"
        )
    else:
        print("PDF           Mass PDF_unc Total_unc")
    to_print = (
        ["value", "err_pdf", "err_Total"]
        if not args.printAssym
        else [
            "value",
            "err_pdf_up",
            "err_pdf_down",
            "err_Profiled",
            "err_Total_up",
            "err_Total_down",
        ]
    )
    for k, v in dfs.iterrows():
        print(v.loc["Name"].ljust(12), *(round(v.loc[x], 1) for x in to_print))

central = dfs.iloc[0, :]

xlabel = r"$\mathit{m}_{" + ("W" if isW else "Z") + "}$ (MeV)"

xlim = (
    args.xlim
    if args.xlim is not None
    else ([91160, 91220] if not isW else [80329, 80374])
)

central_val = central["value"]
if args.diffToCentral:
    if args.saveForHepdata:
        # save also the original absolute value
        dfs["absolute_value"] = dfs["value"].values
    dfs["value"] -= central_val
    xlim = [xlim[0] - central_val, xlim[1] - central_val]
    central_val = 0
    xlabel = r"$\Delta$" + xlabel

dfs["Name"] = dfs["Name"].replace("MSHT20an3lo", "MSHT20aN3LO")
dfs["Name"] = dfs["Name"].replace("NNPDF31", "NNPDF3.1")
dfs["Name"] = dfs["Name"].replace("NNPDF40", "NNPDF4.0")

dfs = dfs[["Name", "value", "err_pdf", "err_Total"]]

fig = plot_tools.make_summary_plot(
    central_val,
    central["err_Total"],
    central["err_pdf"],
    "CT18Z (nominal)",
    dfs.iloc[1:, :],
    colors=args.colors[1:] if args.colors[0] != "auto" else args.colors[0],
    center_color="black",
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
minor_loc = 5 if xlim[1] - xlim[0] < 100 else 10
ax.yaxis.set_major_locator(ticker.MultipleLocator(minor_loc * 2))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(minor_loc))
ax.xaxis.grid(False, which="both")
ax.yaxis.grid(False, which="both")

eoscp = output_tools.is_eosuser_path(args.outpath)
outdir = output_tools.make_plot_dir(args.outpath, args.outfolder, eoscp=eoscp)

outname = f"{'Wmass' if isW else 'Wlike'}_pdf_summary"
if args.postfix:
    outname += f"_{args.postfix}"

plot_tools.save_pdf_and_png(outdir, outname, fig)
output_tools.write_index_and_log(outdir, outname)

if args.saveForHepdata:
    column_labels = [xlabel, "Total uncertainty", "PDF uncertainty"]
    if args.diffToCentral:
        column_labels.append(xlabel.replace(r"$\Delta$", ""))

    hepdata_tools.make_mass_summary_histogram(
        dfs, f"{outdir}/{outname}.root", column_labels
    )

if eoscp:
    output_tools.copy_to_eos(outdir, args.outpath, args.outfolder)
