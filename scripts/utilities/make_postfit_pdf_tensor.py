import argparse

import hist
import numpy as np

from rabbit import inputdata, tensorwriter
from wremnants import theory_tools

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output", default="./", help="output directory")
parser.add_argument("--outname", default="test_tensor", help="output file name")
parser.add_argument(
    "--postfix",
    default=None,
    type=str,
    help="Postfix to append on output file name",
)
parser.add_argument(
    "--sparse",
    default=False,
    action="store_true",
    help="Make sparse tensor",
)
parser.add_argument(
    "--rabbit-input",
    type=str,
    required=True,
    help="Rabbit input file for the reference fit",
)
parser.add_argument(
    "--proc",
    type=str,
    choices=["Zmumu", "Wmunu"],
    required=True,
    help="Process name to use for the PDF fit (should match the signal)",
)
args = parser.parse_args()

indata = inputdata.FitInputData(args.rabbit_input)

# Build tensor
writer = tensorwriter.TensorWriter(
    sparse=args.sparse,
)

chan, *_ = indata.channel_info.keys()

metadata = indata.metadata
pdf_input = indata.metadata["meta_info_input"]["args"]["pdfs"][0]
pdf_name = theory_tools.pdfMap[pdf_input]["lha_name"]

x_range = np.logspace(-4, -0.01, 201)
pdf_data = theory_tools.get_pdf_data(pdf_name, "uv", 80.360, x_range[:-1])
pdf_hist = hist.Hist(
    hist.axis.Variable(x_range, name="x"),
    hist.axis.StrCategory(
        [s for s in indata.systs if "pdfAlphaS" not in s.decode()], name="pdfVar"
    ),
    data=pdf_data.T,
)

writer.add_channel(pdf_hist.axes[:-1], chan)

if args.proc.encode("utf-8") not in indata.procs:
    raise ValueError(f"Process {args.proc} not found in input data")

writer.add_process(pdf_hist[..., 0], args.proc, chan, signal=False)
writer.add_data(pdf_hist[..., 0], chan)

for syst in pdf_hist.axes["pdfVar"]:
    writer.add_systematic(
        pdf_hist[..., syst],
        syst,
        args.proc,
        chan,
        mirror=True,
    )

directory = args.output
if directory == "":
    directory = "./"
filename = args.outname
if args.postfix:
    filename += f"_{args.postfix}"

writer.write(outfolder=directory, outfilename=filename)
