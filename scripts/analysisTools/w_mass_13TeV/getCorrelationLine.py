#!/usr/bin/env python3

# python w-helicity-13TeV/getCorrelationLine.py cards/diffXsec_mu_2019_04_09_newSystAndWtau_fixTriSF/fit/hessian/fitresults_123456789_Asimov_combinedLep_bbb1_cxs1.root -o plots/diffXsecAnalysis/muon/diffXsec_mu_2019_04_09_newSystAndWtau_fixTriSF/getCorrelationLine/ -p CMS_Wmu_sig_lepeff -m sumpoisnorm -n 50

import re
from array import array

# from subMatrix import niceName # skip for now, have to adapt the script
from operator import itemgetter

import utilitiesCMG

utilities = utilitiesCMG.util()

## safe batch mode
import sys

args = sys.argv[:]
sys.argv = ["-b"]
import ROOT

sys.argv = args
ROOT.gROOT.SetBatch(True)
ROOT.PyConfig.IgnoreCommandLineOptions = True

# sys.path.append(os.getcwd() + "/plotUtils/")
# from utility import *
from scripts.analysisTools.plotUtils.utility import (
    common_plot_parser,
    copyOutputToEos,
    createPlotDirAndCopyPhp,
)

if __name__ == "__main__":

    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetPaintTextFormat(".3f")

    # date = datetime.date.today().isoformat()

    parser = common_plot_parser()
    parser.add_argument(
        "fitresult", type=str, nargs=1, help="fitresult.root file from combinetf"
    )
    parser.add_argument(
        "-o",
        "--outdir",
        default=None,
        type=str,
        help="outdput directory to save the plot",
    )
    parser.add_argument(
        "-p",
        "--param",
        default="",
        type=str,
        help="parameter for which you want to show the correlation matrix. Must be a single object",
    )
    parser.add_argument(
        "-t",
        "--type",
        default="hessian",
        type=str,
        choices=["toys", "hessian"],
        help="which type of input file: toys or hessian (default)",
    )
    parser.add_argument(
        "-m",
        "--matrix",
        default="",
        type=str,
        help="matrix to be used (name is correlation_matrix_channel<matrix>)",
    )
    parser.add_argument(
        "--postfix",
        default="",
        type=str,
        help="Postfix for the plotted correlation matrix",
    )
    parser.add_argument(
        "--vertical-labels-X",
        dest="verticalLabelsX",
        action="store_true",
        help="Set labels on X axis vertically (sometimes they overlap if rotated)",
    )
    parser.add_argument(
        "--title",
        default="",
        type=str,
        help="Title for matrix. Use 0 to remove title. By default, string passed to option -p is used",
    )
    parser.add_argument(
        "-n",
        "--show-N",
        dest="showN",
        default=10,
        type=int,
        help="Show the N nuisances more correlated (in absolute value) with the parameter given with --param.",
    )
    args = parser.parse_args()

    ROOT.TColor.CreateGradientColorTable(
        3,
        array("d", [0.00, 0.50, 1.00]),
        ##array ("d", [1.00, 1.00, 0.00]),
        ##array ("d", [0.70, 1.00, 0.34]),
        ##array ("d", [0.00, 1.00, 0.82]),
        array("d", [0.00, 1.00, 1.00]),
        array("d", [0.34, 1.00, 0.65]),
        array("d", [0.82, 1.00, 0.00]),
        255,
        0.95,
    )

    if not args.matrix:
        print("Need to specify which matrix with option -m (e.g. -m 'channelnone')")
        quit()

    if not args.param:
        print("Need to specify a parameter with option -p")
        quit()

    outdir_original = args.outdir
    outdir = None
    if outdir_original:
        outdir = createPlotDirAndCopyPhp(outdir_original, eoscp=args.eoscp)

    param = args.param
    print(f"Will do parameter with the following name: {param}")

    corr = {}
    sign = {}
    index = 0

    ### GET LIST OF PARAMETERS THAT MATCH THE SPECIFIED OPTION IN THE TOYFILE
    if args.type == "toys":
        print("Toys not implemented, sorry. Only hessian for now")
        quit()
    elif args.type == "hessian":
        hessfile = ROOT.TFile(args.fitresult[0], "read")
        suffix = args.matrix
        corrmatrix = hessfile.Get("correlation_matrix_channel" + suffix)
        covmatrix = hessfile.Get("covariance_matrix_channel" + suffix)
        for ib in range(1 + corrmatrix.GetNbinsX() + 1):
            if re.match(param, corrmatrix.GetXaxis().GetBinLabel(ib)):
                ## store mean and rms into the dictionaries from before
                ## also keep a list of the parameter names, for sorting
                index = ib

    ## construct the covariances and the correlations in one go.
    for bin in range(1 + corrmatrix.GetNbinsX() + 1):
        label = corrmatrix.GetXaxis().GetBinLabel(bin)
        if label == param:
            continue
        # save absolute value, but keep track of the sign
        bincontent = corrmatrix.GetBinContent(index, bin)
        corr[label] = abs(bincontent)
        sign[label] = -1 if bincontent < 0 else 1

    sorted_keys = sorted(corr.items(), key=itemgetter(1), reverse=True)
    inum = 1
    nToShow = args.showN if args.showN > 0 else len(corr.keys())
    hist = ROOT.TH1D("hist", "", nToShow, 0, nToShow)
    for key, val in sorted_keys:
        print("%s   %s" % (key, val * sign[key]))
        # hist.GetXaxis().SetBinLabel(inum,niceName(key))
        hist.GetXaxis().SetBinLabel(inum, key)
        hist.SetBinContent(inum, val * sign[key])
        inum += 1
        if inum > nToShow:
            break

    c = ROOT.TCanvas("c", "", 1200, 800)
    c.SetTickx(1)
    c.SetTicky(1)

    c.SetLeftMargin(0.12)
    c.SetRightMargin(0.05)
    c.SetBottomMargin(0.3)

    if args.verticalLabelsX:
        hist.LabelsOption("v", "X")
    if hist.GetNbinsX() >= 20:
        hist.LabelsOption("v", "X")

    hist.SetTitle(
        "parameter: " + param + "    channel: " + args.matrix.replace("channel", "")
    )
    if len(args.title):
        if args.title == "0":
            hist.SetTitle("")
        else:
            hist.SetTitle(args.title)

    hist.GetYaxis().SetTitle("Correlation")
    hist.SetLineWidth(2)
    hist.SetLineColor(ROOT.kGreen + 2)
    hist.SetFillColor(ROOT.kGreen + 1)
    hist.SetFillColorAlpha(ROOT.kGreen + 1, 0.35)
    # hist.SetFillStyle(3001)
    hist.Draw("B")
    miny = hist.GetBinContent(hist.GetMinimumBin())
    maxy = hist.GetBinContent(hist.GetMaximumBin())
    maxval = max(abs(miny), abs(maxy))
    maxval *= 1.1
    hist.GetYaxis().SetRangeUser(-maxval, maxval)
    c.SetGridx(1)
    c.SetGridy(1)
    c.RedrawAxis("sameaxis")

    if outdir:
        for i in ["pdf", "png"]:
            pf = "" if not args.postfix else "_" + args.postfix
            c.SaveAs(
                outdir
                + "/corrLine{pf}_{pn}_{ch}.{i}".format(
                    pf=pf, i=i, pn=param, ch=args.matrix.replace("channel", "")
                )
            )
    copyOutputToEos(outdir, outdir_original, eoscp=args.eoscp)
