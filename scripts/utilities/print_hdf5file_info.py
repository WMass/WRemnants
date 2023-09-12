import argparse
from utilities import input_tools

parser = argparse.ArgumentParser()
parser.add_argument("infile", type=str, help="Input hdf5 file")
parsers = parser.add_subparsers(dest='mode')
histparser = parsers.add_parser("hists", help="Print info about histograms")
histparser.add_argument("-s", "--sample", type=str, required=True, help="Sample name (e.g., ZmumuPostVFP)")
histparser.add_argument("--hist", type=str, help="Print info about specific hist for sample")
sampleparser = parsers.add_parser("samples", help="Print info about samples")

args = parser.parse_args()

if args.mode == "hists":
    names = input_tools.read_hist_names(args.infile, args.sample)
    print("Valid names for process {args.sample} are:")
    print(names)

if args.mode == "samples":
    keys = input_tools.read_keys(args.infile)
    print(f"Valid samples in file are {[k for k in keys if k != 'meta_info']}")