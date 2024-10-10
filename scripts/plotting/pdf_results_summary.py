import argparse

from utilities import common
from utilities.io_tools import combinetf_input
from wremnants import plot_tools, theory_tools

parser = argparse.ArgumentParser()
parser = common.plot_parser()
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
    default=["ct18z", "ct18", "nnpdf40", "msht20an3lo", "nnpdf31", "pdf4lhc21"],
    type=str,
    help="PDF to plot",
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

args = parser.parse_args()


def weave_vals(vals):
    return np.ravel(np.stack([list(v) for v in vals], axis=1))


isW = "WMass" in args.reffile

ref_mass = 80355 if isW else 91188
ref_unc = 6.0 if isW else 2.0

pdf_name = lambda p: theory_tools.pdfMap[p]["name"]

pdf_dfs = combinetf_input.read_all_groupunc_df(
    [args.reffile.format(pdf=pdf) for pdf in args.pdfs],
    rename_cols={f"err_{pdf_name(pdf)}": "err_pdf" for pdf in args.pdfs},
    uncs=[pdf_name(pdf) for pdf in args.pdfs],
    names=[pdf_name(pdf)[3:] for pdf in args.pdfs],
)
# pdf_infdfs = {pdf_name(pdf)[3:] : combinetf_input.read_groupunc_df(args.reffileinf.format(pdf=pdf), pdf_name(pdf)) for pdf in args.pdfs}

outname = f"{'Wmass' if isW else 'Wlike'}_pdf_summary"
if args.postfix:
    outname += f"_{args.postfix}"

if args.print:
    for k, v in pdf_dfs.iterrows():
        print(round(v.iloc[1], 1), round(v.iloc[3], 1), round(v.iloc[2], 1))

central = pdf_dfs.iloc[0, :]

plot_tools.make_summary_plot(
    central["value"],
    central["err_pdf"],
    central["Name"],
    pdf_dfs.iloc[1:, :],
    colors=args.colors[1:] if args.colors[0] != "auto" else args.colors[0],
    center_color="black" if args.colors[0] == "auto" else args.colors[0],
    xlim=[91160, 91220] if not isW else [80300, 80440],
    xlabel="$m_{Z}$ (MeV)" if not isW else "$m_{W}$ (MeV)",
    out=args.outpath,
    outfolder=args.outfolder,
    name=outname,
)
