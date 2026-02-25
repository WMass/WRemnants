import argparse

import hist
import numpy as np

from rabbit import inputdata, tensorwriter
from wremnants import theory_tools
from wums import logging, output_tools

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
parser.add_argument(
    "--noColorLogger", action="store_true", help="Disable colored logging output."
)
parser.add_argument(
    "-l", "--fit-label", type=str, default="cmsmw", help="Label in the output PDF grids"
)
parser.add_argument(
    "-v", "--verbose", choices=[0, 1, 2, 3, 4], default=3, help="Set verbosity level."
)
args = parser.parse_args()

logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)

indata = inputdata.FitInputData(args.rabbit_input)

# Build tensor
writer = tensorwriter.TensorWriter(
    sparse=args.sparse,
)

metadata = indata.metadata

pdf_input = indata.metadata["meta_info_input"]["args"]["pdfs"][0]
pdf_scale = metadata["meta_info"]["args"]["scalePdf"]

pdfInfo = theory_tools.pdf_info_map("Zmumu_2016PostVFP", pdf_input)
pdf_name = pdfInfo["lha_name"]

if pdf_scale == -1:
    pdf_scale = theory_tools.pdf_inflation_factor(
        theory_tools.pdfMap[pdf_input], metadata["meta_info"]["args"]["noi"]
    )
    logger.info(f"Using default inflation factor from theory_tools: {pdf_scale}")

pdf_scale *= pdfInfo["scale"]
logger.info(f"Scaling PDF uncertainties by {pdf_scale}")

symHessian = pdfInfo["combine"] == "symHessian"
symmetrize = indata.metadata["meta_info"]["args"]["symmetrizePdfUnc"]

if not symHessian:
    logger.info(f"Applying {symmetrize} symmetrization procedure")

labels = np.array([s for s in indata.systs if "pdfAlphaS" not in s.decode()], dtype=str)
if symHessian:
    labels[::2] = [
        s.replace("SymAvg", "Down").replace("SymDiff", "Up") for s in labels[::2]
    ]
    labels[1::2] = [
        s.replace("SymAvg", "Down").replace("SymDiff", "Down") for s in labels[1::2]
    ]

x_range = np.logspace(-4, -0.01, 201)

for chan in ["u", "d", "uv", "dv"]:
    pdf_data = theory_tools.get_pdf_data(pdf_name, chan, 80.360, x_range[:-1])
    pdf_hist = hist.Hist(
        hist.axis.Variable(x_range, name="x"),
        hist.axis.StrCategory(labels, name="pdfVar"),
        data=pdf_data.T,
    )

    writer.add_channel(pdf_hist.axes[:-1], chan)

    if args.proc.encode("utf-8") not in indata.procs:
        raise ValueError(f"Process {args.proc} not found in input data")

    writer.add_process(pdf_hist[..., 0], args.proc, chan, signal=False)
    writer.add_data(pdf_hist[..., 0], chan)

    if symHessian:
        for syst in pdf_hist.axes["pdfVar"]:
            writer.add_systematic(
                pdf_hist[..., syst],
                syst,
                args.proc,
                chan,
            )
    else:
        systs = list(pdf_hist.axes["pdfVar"])
        for systUp, systDown in zip(systs[::2], systs[1::2]):
            writer.add_systematic(
                [pdf_hist[..., systUp], pdf_hist[..., systDown]],
                systUp.replace("Up", ""),
                args.proc,
                chan,
                symmetrize=symmetrize,
            )

directory = args.output
if directory == "":
    directory = "./"
filename = args.outname
if args.postfix:
    filename += f"_{args.postfix}"

meta_data = {"meta_info": output_tools.make_meta_info_dict()}
writer.write(outfolder=directory, outfilename=filename, meta_data_dict=meta_data)
