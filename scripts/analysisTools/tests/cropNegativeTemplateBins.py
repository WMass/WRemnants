#!/usr/bin/env python

# from shutil import copyfile
# import numpy as np
# import root_numpy
import argparse
import os
import os.path
import re
import sys

import ROOT

# in the fit, bins with 0 content in the nominal are ignored.
# so, after we made the shape files, we can just set to 0 the negative content of bins in the
# nominal templates (no need to touch the alternate as well)
# note that, by default, this function modify the original input file


def cropNegativeContent(h, silent=False, cropError=False, cropValue=0.0001):

    hasCroppedBins = False
    if "THn" in h.ClassName():
        nbins = h.GetNbins() + 1
        for i in range(nbins):
            nom = h.GetBinContent(
                i, 0
            )  # apparently I need to pass 0 explicitly for the second argument when using THn, might be a bug
            if nom < 0.0:
                hasCroppedBins = True
                h.SetBinContent(i, cropValue)
                if cropError:
                    h.SetBinError(i, cropValue)
    else:
        nbins = h.GetNcells() + 1
        integral = h.Integral()
        for i in range(nbins):
            nom = h.GetBinContent(i)
            if nom < 0.0:
                hasCroppedBins = True
                h.SetBinContent(i, cropValue)
                if cropError:
                    h.SetBinError(i, cropValue)
        if not silent and hasCroppedBins:
            integralNonNeg = h.Integral()
            print(
                "{n}: original integral = {i} changed by {r} (new/old)".format(
                    n=h.GetName(), i=str(integral), r=str(integralNonNeg / integral)
                )
            )
    return hasCroppedBins


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--infile", type=str, default="", help="Input file name")
    parser.add_argument(
        "-o",
        "--outdir",
        type=str,
        default="./",
        help="Output folder (current one as default)",
    )
    parser.add_argument(
        "-p",
        "--processes",
        type=str,
        default="x_Z|x_W.*|x_Top.*|.*DiBosons.*|x_data_fakes.*",
        help="Regular expression for histograms to be affected",
    )
    parser.add_argument(
        "-a",
        "--all",
        dest="allHists",
        action="store_true",
        help="Crop all templates, not just those for systematics",
    )
    parser.add_argument(
        "--crop-error",
        dest="cropError",
        action="store_true",
        help="Crop error as well setting it to the cropped value",
    )
    parser.add_argument(
        "--crop-value",
        dest="cropValue",
        type=float,
        default=0.001,
        help="Crop negative bin content to this value",
    )
    parser.add_argument(
        "-s",
        "--silent",
        action="store_true",
        help="Print info when acting on histograms",
    )
    args = parser.parse_args()
    options = args

    if not len(args.infile):
        print("Warning: you must specify an input file name with option --infile")
        quit()

    # manage output folder
    outdir = args.outdir
    if not outdir.endswith("/"):
        outdir += "/"
    if outdir != "./":
        if not os.path.exists(outdir):
            print("Creating folder", outdir)
            os.system("mkdir -p " + outdir)

    basenameFile = os.path.basename(args.infile)

    infileCopy = outdir + basenameFile.replace(".root", "_ORIGINAL.root")
    print("Copying original file with shapes into a backup")
    cpcmd = "cp {f} {fnew}".format(f=args.infile, fnew=infileCopy)
    print(cpcmd)
    os.system(cpcmd)
    outfile = outdir + basenameFile
    if os.path.abspath(outfile) != os.path.abspath(args.infile):
        print("Copying original file with shapes again in output folder")
        cpcmd = "cp {f} {fnew}".format(f=args.infile, fnew=outfile)
        print(cpcmd)
        os.system(cpcmd)
    else:
        print(
            "Output folder is the same as the input file. I will update the input file"
        )

    tfno = ROOT.TFile(args.infileCopy, "READ")
    if not tfno or not tfno.IsOpen():
        raise (RuntimeError("Unable to open file {fn}".format(fn=args.infile)))
    of = ROOT.TFile(outfile, "UPDATE")
    if not of or not of.IsOpen():
        raise (RuntimeError("Unable to open file {fn}".format(fn=outfile)))
    nKeys = tfno.GetNkeys()
    nCopiedKeys = 0
    for ikey, e in enumerate(tfno.GetListOfKeys()):
        name = e.GetName()
        if not re.match(args.processes, name):
            continue
        if not args.allHists and re.match(".*(Up|Down)", name):
            continue
        obj = e.ReadObj()
        if not obj:
            raise (RuntimeError("Unable to read object {n}".format(n=name)))
        cropNegativeContent(
            obj, silent=args.silent, cropErrro=args.cropError, cropValue=args.cropValue
        )
        obj.Write(name, ROOT.TObject.kOverwrite)
        nCopiedKeys += 1
        if (ikey + 1) % 100 == 0:
            sys.stdout.write("Key {0:.2%}     \r".format(float(ikey + 1) / nKeys))
            sys.stdout.flush()

    print(
        "Overwrote {n}/{tot} from {fn}".format(
            n=str(nCopiedKeys), tot=str(nKeys), fn=args.infile
        )
    )
    print("Updated file saved in {of}".format(of=outfile))
    of.Close()
    tfno.Close()
