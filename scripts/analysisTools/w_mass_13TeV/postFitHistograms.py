#!/usr/bin/env python3

# example (only charge plus)
# python fitresults.root -o outputPath/postFitHistograms/ --suffix postVFP -l 16.8 [-c plus --no2Dplot]

import os, re
import argparse
from array import array
import scipy

import sys
args = sys.argv[:]
sys.argv = ['-b']
import ROOT
sys.argv = args
ROOT.gROOT.SetBatch(True)
ROOT.PyConfig.IgnoreCommandLineOptions = True

from copy import *
import wremnants

import utilitiesCMG
utilities = utilitiesCMG.util()
from utilities import logging

#sys.path.append(os.getcwd() + "/plotUtils/")
#from utility import *
from scripts.analysisTools.plotUtils.utility import *

def normalizeTH2byBinWidth(h2):
    for ix in range(1,1+h2.GetNbinsX()):
        for iy in range(1,1+h2.GetNbinsY()):
            binWidth = h2.GetXaxis().GetBinWidth(ix) * h2.GetYaxis().GetBinWidth(iy)
            h2.SetBinContent(ix,iy, h2.GetBinContent(ix,iy)/binWidth)
            h2.SetBinError(ix,iy, h2.GetBinError(ix,iy)/binWidth)


def normalizeTH1unrolledSingleChargebyBinWidth(h1, h2, unrollAlongX=True):
    # h2 is just used to retrieve the binning, the content is not used
    # unrollAlongX should be true when X is eta and we are taking slices at costant pt
    for ix in range(1,1+h2.GetNbinsX()):
        for iy in range(1,1+h2.GetNbinsY()):
            ibin = 0
            if  unrollAlongX:
                ibin = ix + (iy-1) * h2.GetNbinsX()
            else:
                ibin = iy + (ix-1) * h2.GetNbinsY()
            binWidth = h2.GetXaxis().GetBinWidth(ix) * h2.GetYaxis().GetBinWidth(iy)
            h1.SetBinContent(ibin, h1.GetBinContent(ibin)/binWidth)
            h1.SetBinError(ibin,   h1.GetBinError(ibin)/binWidth)



def dressed2DfromFit(h1d, binning, name, title='', shift=0,
                     nCharges=2, nMaskedCha=2, 
                     nRecoBins=0, invertXY=True):
    
    if len(binning) == 4:
        n1 = binning[0]; bins1 = array('d', binning[1])
        n2 = binning[2]; bins2 = array('d', binning[3])
        h2_1 = ROOT.TH2D(name, title, n1, bins1, n2, bins2 )
    else:
        n1 = binning[0]; min1 = binning[1]; max1 = binning[2]
        n2 = binning[3]; min2 = binning[4]; max2 = binning[5]
        h2_1 = ROOT.TH2D(name, title, n1, min1, max1, n2, min2, max2)
    h1d_shifted = singleChargeUnrolled(h1d, shift, nCharges,
                                       nMaskedCha, 
                                       nRecoBins=nRecoBins)
    h2_backrolled = roll1Dto2D(h1d_shifted, h2_1, invertXY)
    return h2_backrolled

# the expproc histograms have the unrolled histograms for both charges (if the charge-combined fit was made) and flavours (if the flavour combined fit was made)
# the order is charge plus, charge minus for charge combination
# if the flavour combinations was made, each charge sector is in turn divided in muons-electrons
# at the end, there are N bins where N is the number of charges in the fit times number of masked channels for each charge

def chargeUnrolledBinShifts(h1d, nCharges=2, nMaskedCha=2):
    nbins = int((h1d.GetNbinsX()-nCharges*nMaskedCha)/2)
    ret = {}
    if h1d.Integral(0,nbins)==0:
        ret = {'plus': nbins, 'minus': 0}
    else:
        ret = {'plus': 0, 'minus': nbins}
    return ret

def singleChargeUnrolled(h1d, shift, nCharges=2, nMaskedCha=2, name="shift", nRecoBins=0):
    extrabins = 0 if 'obs' in h1d.GetName() else nCharges*nMaskedCha
    nbins = int((h1d.GetNbinsX()-extrabins)/2)
    h1d_shifted = ROOT.TH1D(name,'',nbins,0,nbins)
    #h1d_shifted.SetBins(nbins,0.5,float(nbins)+0.5)
    for b in range(1, nbins+1):
        h1d_shifted.SetBinContent(b,h1d.GetBinContent(b+shift))
        h1d_shifted.SetBinError(b,h1d.GetBinError(b+shift))
    return h1d_shifted

if __name__ == "__main__":
    parser = common_plot_parser()
    parser.add_argument("rootfile", type=str, nargs=1, help="Input file with histograms")
    parser.add_argument('-o','--outdir', dest='outdir', default='.', type=str, help='output directory to save the plots')
    parser.add_argument('-m','--n-mask-chan', dest='nMaskedChannel', default=0, type=int, help='Number of masked channels in the fit for each charge (0 if not using masked channels because no signal POIs is used in the fit)')
    parser.add_argument('-c','--charges', dest='charges', choices=['plus', 'minus', 'both'], default='both', type=str, help='Charges to process')
    parser.add_argument(     '--no2Dplot', dest="no2Dplot", action='store_true', help="Do not plot 2D templates")
    parser.add_argument(     '--comparePrefitPostfit', action='store_true', help="Plot comparisons of postfit/prefit yields and uncertainties ")
    parser.add_argument(     '--wlike', dest="isWlike", action='store_true', help="Analysis is wlike")
    parser.add_argument('-n','--normWidth', action='store_true', help="Normalize histograms by bin area (mainly if non uniform binning is used)")
    parser.add_argument('-p', '--postfix', dest="postfix", default='', type=str, help="define postfix for each plot")
    parser.add_argument('-l','--lumi', default=16.8, type=float, help='Integrated luminosity to print in the plot')
    parser.add_argument('--filterProcesses', default="", type=str, help='If given, regexp to filter some processes')
    parser.add_argument('--dataTitle', default="Data", type=str, help='Title for data in legend (usually Data but could be Pseudodata)')
    parser.add_argument("--rrPre", dest="ratioRangePrefit", default=(0.9,1.1), type=float, nargs=2, help="Range for ratio plot prefit")
    parser.add_argument("--rrPost", dest="ratioRangePostfit", default=(0.99,1.01), type=float, nargs=2, help="Range for ratio plot postfit")
    parser.add_argument(     '--unrolledRatioRangeAsPrefit', action='store_true', help="For unrolled stack plot use same ratio range as prefit also in the postfit")
    parser.add_argument('--run', choices=["prefit", "postfit"], default=["prefit", "postfit"], nargs='+', type=str, help='What plots to produce')
    # FIXME: read binning from some input file
    parser.add_argument("--ptBins", default=None, type=float, nargs='*', help="Edges for pt bins (if not given it uses a default)")
    parser.add_argument("--etaBins", default=None, type=float, nargs='*', help="Edges for eta bins (if not given it uses a default)")
    args = parser.parse_args()

    logger = logging.setup_logger(os.path.basename(__file__), args.verbose)

    ROOT.gStyle.SetOptStat(0)
    ROOT.TH1.SetDefaultSumw2()

    adjustSettings_CMS_lumi()
    setTDRStyle()

    charges = ["plus", "minus"] if args.charges == "both" else [args.charges]
    nCharges = len(charges) 
    nMaskedChanPerCharge = args.nMaskedChannel # check if we actually have masked channels, we may not, default should be 0
    lep = "Muon"
    xaxisname2D = "{l} #eta".format(l=lep)
    yaxisname2D = "{l} p_{{T}} [GeV]".format(l=lep)

    #########
    # FIXME
    #########
    # hardcoded eta-pt reco binning for now
    etabins = [round(-2.4 + (0.1 * i), 1) for i in range(0,49)]
    ptbins = [round(26.0 + (1.0 * i), 1) for i in range(0, 35 if args.isWlike else 31)]
    if args.etaBins:
        etabins = args.etaBins
    if args.ptBins:
        ptbins = args.ptBins
    recoBins = templateBinning(etabins, ptbins)
    logger.warning("-"*30)
    logger.warning("USING THIS BINNING: PLEASE CHECK IF IT IS OK")
    logger.warning("-"*30)
    recoBins.printBinAll()
    logger.warning("-"*30)
    nRecoBins = recoBins.NTotBins
    #following array is used to call function dressed2DfromFit()
    binning = [recoBins.Neta, recoBins.etaBins, recoBins.Npt, recoBins.ptBins]

    outdir_original = args.outdir
    outdir = createPlotDirAndCopyPhp(outdir_original, eoscp=args.eoscp)
    if args.comparePrefitPostfit:
        outdirsub = outdir+'/postfit_over_prefit/'
        createPlotDirAndCopyPhp(outdirsub)

    savErrorLevel = ROOT.gErrorIgnoreLevel; ROOT.gErrorIgnoreLevel = ROOT.kError;

    infile = safeOpenFile(args.rootfile[0], mode="READ")
    h1d = safeGetObject(infile, "expfull_postfit")
    # for some postfit stuff to print in plots
    nbins = safeGetObject(infile, "obs").GetNbinsX()
    tree = safeGetObject(infile, "fitresults", detach=False)
    tree.GetEntry(0)
    postfit_2deltaLL = 2.0 * (tree.nllvalfull - tree.satnllvalfull)
    ndof = nbins - tree.nois_ndof
    postfit_chi2prob = scipy.stats.chi2.sf(postfit_2deltaLL, ndof)
    logger.warning(f"-loglikelihood_{{full}} = {tree.nllvalfull}")
    logger.warning(f"-loglikelihood_{{saturated}} = {tree.satnllvalfull}")
    logger.warning(f"2*(nllfull-nllsat) = {postfit_2deltaLL}")
    logger.warning(f"nbins = {nbins}")
    logger.warning(f"chi2 probability = {postfit_chi2prob}")
    #
    textForUnrolled = f"2 * (nllfull - nllsat) = {round(postfit_2deltaLL,1)}     ndof(+/-) = {ndof}     #chi^{{2}} prob = {round(100.0*postfit_chi2prob,1)}%::0.3,0.96,0.08,0.04"

    shifts = chargeUnrolledBinShifts(h1d, nCharges, nMaskedChanPerCharge)
    # get process names:
    predictedProcessNames = []
    for k in infile.GetListOfKeys():
        name = k.GetName()
        if "expproc" in name and "prefit" in name:
            #e.g. expproc_Wmunu_prefit, but might have a strange name as expproc_AAA_BBB_CCC_prefit
            predictedProcessNames.append("_".join(name.split("_")[1:-1]))
    logger.info(f"Found these predicted processes: {predictedProcessNames}")

    originalProcessNames = predictedProcessNames[:]
    gatherDict = {}
    if args.gatherProcesses:
        logger.warning("Gathering these processes together")
        gatherDict = gatherProcesses_[args.gatherProcesses]
        reducedProcs = []
        groupedProcs = []
        for k in gatherDict.keys():
            logger.warning(f"{k}: {gatherDict[k]}")
            groupedProcs.extend(gatherDict[k])
            reducedProcs.append(k)
        predictedProcessNames = [x for x in originalProcessNames if x not in groupedProcs]
        predictedProcessNames.extend(reducedProcs)
        logger.warning(f"Reduced predicted processes: {predictedProcessNames}")

    postfix = ""
    if args.postfix:
        postfix = f"_{args.postfix}"
    full_outfileName = f"{outdir}/plots{postfix}.root"
    outfile = ROOT.TFile(full_outfileName, "recreate")
    logger.info(f"Will save 2D templates in file --> {full_outfileName}")

    process_features = {p: {"color": colors_plots_[p], "title": legEntries_plots_[p]} for p in predictedProcessNames}

    canvas2D = ROOT.TCanvas("canvas2D","",800,700)

    verticalAxisName = "Events / bin [GeV^{-1 }]" if args.normWidth else "Events"
    verticalAxisNameProjX = "Events / bin" if args.normWidth else "Events"  # X is eta
    verticalAxisNameProjY = "Events / bin [GeV^{-1 }]" if args.normWidth else "Events"  # Y is pt

    cwide = ROOT.TCanvas("cwide","",2400,600)                      
    cnarrow = ROOT.TCanvas("cnarrow","",650,700)                      

    canvasRatio = ROOT.TCanvas("canvasRatio","",2400,600)

    # to draw panels in the unrolled plots
    ptBinRanges = []
    for ipt in range(0,recoBins.Npt):
        #ptBinRanges.append("p_{{T}} #in [{ptmin:3g}, {ptmax:.3g}]".format(ptmin=recoBins.ptBins[ipt], ptmax=recoBins.ptBins[ipt+1]))
        ptBinRanges.append("#splitline{{[{ptmin},{ptmax}]}}{{GeV}}".format(ptmin=int(recoBins.ptBins[ipt]), ptmax=int(recoBins.ptBins[ipt+1])))

    etaBinRanges = []
    for ieta in range(0,recoBins.Neta):
        etaBinRanges.append("[{etamin},{etamax}]".format(etamin=round(recoBins.etaBins[ieta],1), etamax=round(recoBins.etaBins[ieta+1],1)))
        

    for charge in charges:
        binshift = shifts[charge]

        logger.info("="*30)
        logger.info(f"charge: {charge}")
        logger.info("-"*30)
        all_procs = {}
        all_procs_unrolled = {}
        ratios_unrolled = {}
        ratios_unc_unrolled = {}
        unc_unrolled = {}
        #
        all_procs_unrolled_y = {}
        ratios_unrolled_y = {}
        ratios_unc_unrolled_y = {}
        unc_unrolled_y = {}

        # keep this order
        whatToRun = ["postfit", "prefit"] if len(args.run) == 2 else [args.run[0]]
        for prepost in whatToRun:

            suffix = f"_{prepost}{postfix}"
            canv = ROOT.TCanvas()
            chfl = charge
            logger.info("-"*30)
            logger.info(f"Doing {prepost}")
            logger.info("-"*30)

            procs = [str(key) for key in process_features.keys() if (args.filterProcesses == "" or re.match(args.filterProcesses, str(key)) )]
            titles = [process_features[p]["title"] for p in procs]
            procs += ["obs"]
            titles += [args.dataTitle]
            procsAndTitles = dict(zip(procs,titles))

            inputHists = {"obs": safeGetObject(infile, "obs")}
            # get original histograms and possibly group them if needed
            for i,p in enumerate(originalProcessNames):
                pname = f"{p}_{prepost}"
                hname = f"expproc_{p}_{prepost}"
                logger.debug(f"Hist {hname}")
                # check if name is already in list of reduced processes ...
                if p in procs:
                    inputHists[p] = safeGetObject(infile, hname)
                else:
                    # ... otherwise get group which contains it to index the dictionary
                    for k in gatherDict.keys():
                        if p in gatherDict[k]:
                            if k not in inputHists.keys():
                                # first time it appears, create the histogram
                                inputHists[k] = safeGetObject(infile, hname)
                            else:
                                # histogram with same key exists, sum the new one to it
                                inputHists[k].Add(safeGetObject(infile, hname))
                        else:
                            continue

            for i,p in enumerate(procs):
                if p == 'obs':
                    pname = p
                    hname = pname
                else:
                    pname = f"{p}_{prepost}"
                    hname = f"expproc_{p}_{prepost}"
                keyplot = f"{chfl}_{pname}"
                logger.debug(f"Hist {hname}")
                h1_1 = inputHists[p]
                h2_backrolled = dressed2DfromFit(h1_1, binning, pname, titles[i], binshift, nMaskedCha=nMaskedChanPerCharge,
                                                 nRecoBins=nRecoBins)
                h1_unrolled = unroll2Dto1D(h2_backrolled, newname=f"unroll_{pname}", cropNegativeBins=False)
                h1_unrolled_y = unroll2Dto1D(h2_backrolled, newname=f"unroll_y_{pname}", cropNegativeBins=False, invertUnroll=True)

                if args.normWidth:
                    normalizeTH1unrolledSingleChargebyBinWidth(h1_unrolled, h2_backrolled, unrollAlongX=True)
                    normalizeTH1unrolledSingleChargebyBinWidth(h1_unrolled_y, h2_backrolled, unrollAlongX=False)
                all_procs[keyplot] = h2_backrolled;
                all_procs_unrolled[keyplot] = h1_unrolled
                all_procs_unrolled_y[keyplot] = h1_unrolled_y
                all_procs[keyplot].SetDirectory(0); 
                all_procs_unrolled[keyplot].SetDirectory(0)
                all_procs_unrolled_y[keyplot].SetDirectory(0)
                if prepost == 'prefit' and p != 'obs' and args.comparePrefitPostfit:
                    postfitkey = keyplot.replace("prefit", "postfit")
                    if all_procs_unrolled[postfitkey] != None and all_procs_unrolled[keyplot] != None:
                        keyratio = f"{chfl}_{p}_ratio"
                        # yields
                        ratios_unrolled[keyratio] = copy.deepcopy(all_procs_unrolled[postfitkey].Clone(keyratio))
                        ratios_unrolled[keyratio].SetDirectory(0)
                        ratios_unrolled[keyratio].Divide(all_procs_unrolled[keyplot])
                        ratios_unrolled[keyratio].SetFillColor(process_features[p]["color"])
                        #
                        ratios_unrolled_y[keyratio] = copy.deepcopy(all_procs_unrolled_y[postfitkey].Clone(keyratio+"_y"))
                        ratios_unrolled_y[keyratio].SetDirectory(0)
                        ratios_unrolled_y[keyratio].Divide(all_procs_unrolled_y[keyplot])
                        ratios_unrolled_y[keyratio].SetFillColor(process_features[p]["color"])

                        for ibin in range(1,ratios_unrolled[keyratio].GetNbinsX()+1):
                            unc = 0.0 if all_procs_unrolled[keyplot].GetBinContent(ibin) == 0.0 else (all_procs_unrolled[postfitkey].GetBinError(ibin) / all_procs_unrolled[keyplot].GetBinContent(ibin))
                            ratios_unrolled[keyratio].SetBinError(ibin, unc)
                            #
                            unc = 0.0 if all_procs_unrolled_y[keyplot].GetBinContent(ibin) == 0.0 else (all_procs_unrolled_y[postfitkey].GetBinError(ibin) / all_procs_unrolled_y[keyplot].GetBinContent(ibin))
                            ratios_unrolled_y[keyratio].SetBinError(ibin, unc)
                        # uncertainties
                        ratios_unc_unrolled[keyratio] = copy.deepcopy(all_procs_unrolled[postfitkey].Clone(keyratio+"_unc"))
                        ratios_unc_unrolled[keyratio].SetDirectory(0)
                        ratios_unc_unrolled[keyratio].SetFillColor(process_features[p]["color"])
                        unc_unrolled[postfitkey] = copy.deepcopy(all_procs_unrolled[postfitkey].Clone(postfitkey+"_unc"))
                        unc_unrolled[keyplot] = copy.deepcopy(all_procs_unrolled[keyplot].Clone(keyplot+"_unc"))
                        unc_unrolled[postfitkey].Reset("ICESM")
                        unc_unrolled[keyplot].Reset("ICESM")
                        #
                        ratios_unc_unrolled_y[keyratio] = copy.deepcopy(all_procs_unrolled_y[postfitkey].Clone(keyratio+"_unc_y"))
                        ratios_unc_unrolled_y[keyratio].SetDirectory(0)
                        ratios_unc_unrolled_y[keyratio].SetFillColor(process_features[p]["color"])
                        unc_unrolled_y[postfitkey] = copy.deepcopy(all_procs_unrolled_y[postfitkey].Clone(postfitkey+"_unc_y"))
                        unc_unrolled_y[keyplot] = copy.deepcopy(all_procs_unrolled_y[keyplot].Clone(keyplot+"_unc_y"))
                        unc_unrolled_y[postfitkey].Reset("ICESM")
                        unc_unrolled_y[keyplot].Reset("ICESM")

                        for ibin in range(1,ratios_unc_unrolled[keyratio].GetNbinsX()+1):
                            val = 0.0 if all_procs_unrolled[keyplot].GetBinError(ibin) == 0.0 else (all_procs_unrolled[postfitkey].GetBinError(ibin) / all_procs_unrolled[keyplot].GetBinError(ibin))
                            ratios_unc_unrolled[keyratio].SetBinContent(ibin, val)
                            ratios_unc_unrolled[keyratio].SetBinError(ibin, 0.0)
                            unc_unrolled[postfitkey].SetBinContent(ibin, all_procs_unrolled[postfitkey].GetBinError(ibin))
                            unc_unrolled[keyplot].SetBinContent(ibin, all_procs_unrolled[keyplot].GetBinError(ibin))
                            #
                            val = 0.0 if all_procs_unrolled_y[keyplot].GetBinError(ibin) == 0.0 else (all_procs_unrolled_y[postfitkey].GetBinError(ibin) / all_procs_unrolled_y[keyplot].GetBinError(ibin))
                            ratios_unc_unrolled_y[keyratio].SetBinContent(ibin, val)
                            ratios_unc_unrolled_y[keyratio].SetBinError(ibin, 0.0)
                            unc_unrolled_y[postfitkey].SetBinContent(ibin, all_procs_unrolled_y[postfitkey].GetBinError(ibin))
                            unc_unrolled_y[keyplot].SetBinContent(ibin, all_procs_unrolled_y[keyplot].GetBinError(ibin))

                        # try plotting both
                        hists = [copy.deepcopy(all_procs_unrolled[postfitkey].Clone(f"postfit_{chfl}_{p}")),
                                 copy.deepcopy(all_procs_unrolled[keyplot].Clone(f"prefit_{chfl}_{p}"))]
                        vertLines="{a},{b}".format(a=recoBins.Npt,b=recoBins.Neta)
                        legs = ["postfit","prefit"]
                        # yields
                        unrollPfx = ""
                        drawNTH1(hists, legs, "unrolled lepton (#eta, p_{T}) bin", "Events", f"postfitAndprefit_yields_chan{chfl}_{p}{unrollPfx}", outdirsub, leftMargin=0.06, rightMargin=0.02, labelRatioTmp="postfit/prefit::0.8,1.2", legendCoords="0.45,0.8,0.92,1.0;2", passCanvas=cwide, drawLumiLatex=True, lumi=args.lumi, drawVertLines=vertLines, textForLines=ptBinRanges, yAxisExtendConstant=1.25, markerStyleFirstHistogram=1, fillStyleSecondHistogram=1001, colorVec=[ROOT.kGray], moreTextLatex=f"{process_features[p]['title']}::0.3,0.95,0.08,0.055")
                        # uncertainties
                        hists = [unc_unrolled[postfitkey], unc_unrolled[keyplot]]
                        drawNTH1(hists, legs, "unrolled lepton (#eta, p_{T}) bin", "Uncertainty", f"postfitAndprefit_uncertainty_chan{chfl}_{p}{unrollPfx}", outdirsub, leftMargin=0.06, rightMargin=0.02, labelRatioTmp="postfit/prefit", legendCoords="0.45,0.8,0.92,1.0;2", passCanvas=cwide, drawLumiLatex=True, lumi=args.lumi, drawVertLines=vertLines, textForLines=ptBinRanges, yAxisExtendConstant=1.25, markerStyleFirstHistogram=1, fillStyleSecondHistogram=1001, colorVec=[ROOT.kGray], moreTextLatex=f"{process_features[p]['title']}::0.3,0.95,0.08,0.055", setRatioRangeFromHisto=True, setOnlyLineRatio=True)
                        #
                        hists = [copy.deepcopy(all_procs_unrolled_y[postfitkey].Clone(f"postfit_{chfl}_{p}_y")),
                                 copy.deepcopy(all_procs_unrolled_y[keyplot].Clone(f"prefit_{chfl}_{p}_y"))]
                        unrollPfx = "_unrollY"
                        vertLines="{a},{b}".format(a=recoBins.Neta,b=recoBins.Npt)
                        drawNTH1(hists, legs, "unrolled lepton (#eta, p_{T}) bin", "Events", f"postfitAndprefit_yields_chan{chfl}_{p}{unrollPfx}", outdirsub, leftMargin=0.06, rightMargin=0.02, labelRatioTmp="postfit/prefit::0.8,1.2", legendCoords="0.45,0.8,0.92,1.0;2", passCanvas=cwide, drawLumiLatex=True, lumi=args.lumi, drawVertLines=vertLines, textForLines=etaBinRanges, yAxisExtendConstant=1.25, markerStyleFirstHistogram=1, fillStyleSecondHistogram=1001, colorVec=[ROOT.kGray], moreTextLatex=f"{process_features[p]['title']}::0.3,0.95,0.08,0.055")
                        # uncertainties
                        hists = [unc_unrolled_y[postfitkey], unc_unrolled_y[keyplot]]
                        drawNTH1(hists, legs, "unrolled lepton (#eta, p_{T}) bin", "Uncertainty", f"postfitAndprefit_uncertainty_chan{chfl}_{p}{unrollPfx}", outdirsub, leftMargin=0.06, rightMargin=0.02, labelRatioTmp="postfit/prefit", legendCoords="0.45,0.8,0.92,1.0;2", passCanvas=cwide, drawLumiLatex=True, lumi=args.lumi, drawVertLines=vertLines, textForLines=etaBinRanges, yAxisExtendConstant=1.25, markerStyleFirstHistogram=1, fillStyleSecondHistogram=1001, colorVec=[ROOT.kGray], moreTextLatex=f"{process_features[p]['title']}::0.3,0.95,0.08,0.055", setRatioRangeFromHisto=True, setOnlyLineRatio=True)



                    else:
                        logger.error("Error: something went wrong! Missing either {postfitkey} or {keyplot}")
                        quit()
                # end of postfit/prefit comparisons

                if not args.no2Dplot:
                    cname = f"{p}_{chfl}{suffix}"
                    h2_backrolled.Write(cname)
                    drawCorrelationPlot(h2_backrolled, xaxisname2D, yaxisname2D, verticalAxisName, cname, "", 
                                        outdir, 0,0, False, False, False, 1, palette=57, passCanvas=canvas2D)

            # now draw the 1D projections         
            sortedKeys = list(sorted(all_procs.keys(), key = lambda k: all_procs[k].Integral()))

            # this has the uncertainty propagated with the full covariance matrix
            h1_expfull = safeGetObject(infile, f"expfull_{prepost}")
            expfullName2D = f"expfull_{charge}_{prepost}"
            h2_expfull_backrolled = dressed2DfromFit(h1_expfull, binning, expfullName2D, expfullName2D,
                                                     binshift,
                                                     nMaskedCha=nMaskedChanPerCharge,
                                                     nRecoBins=nRecoBins)
            h1_expfull_unrolled = unroll2Dto1D(h2_expfull_backrolled, newname=f"unroll_{expfullName2D}", cropNegativeBins=False)
            h1_expfull_unrolled_y = unroll2Dto1D(h2_expfull_backrolled, newname=f"unroll_y_{expfullName2D}", cropNegativeBins=False, invertUnroll=True)

            h2_expfull_backrolled.Write(f"expfull_{prepost}_{charge}")
            # can normalize the unrolled, not the 2D
            if args.normWidth:
                normalizeTH1unrolledSingleChargebyBinWidth(h1_expfull_unrolled, h2_expfull_backrolled, unrollAlongX=True)
                normalizeTH1unrolledSingleChargebyBinWidth(h1_expfull_unrolled_y, h2_expfull_backrolled, unrollAlongX=False)

            for projection in ['X', 'Y']:
                if projection=='X':
                    nbinsProj = all_procs[f"{chfl}_obs"].GetNbinsY()
                    projName = f"expfull_{prepost}_{charge}_px"
                    projNameData = f"data_{prepost}_{charge}_px"
                    hexpfull = h2_expfull_backrolled.ProjectionX(projName,     1, nbinsProj, "e")
                    hdata = all_procs[f"{chfl}_obs"].ProjectionX(projNameData, 1, nbinsProj, "e")
                else:
                    nbinsProj = all_procs[f"{chfl}_obs"].GetNbinsX()
                    projName = f"expfull_{prepost}_{charge}_py"
                    projNameData = f"data_{prepost}_{charge}_py"
                    hexpfull = h2_expfull_backrolled.ProjectionY(projName, 1, nbinsProj, "e")
                    hdata = all_procs[f"{chfl}_obs"].ProjectionY(projNameData, 1, nbinsProj, "e")
                if args.normWidth:
                    hdata.Scale(1., "width")
                    hexpfull.Scale(1., "width")

                hdata.Write()
                hexpfull.Write()
                htot = hdata.Clone(f"tot_{charge}") 
                htot.Reset("ICES");
                htot.Sumw2()
                stack = ROOT.THStack(f"stack_{prepost}_{charge}_proj{projection}", "")

                leg = prepareLegend(0.2, 0.72, 0.9, 0.90, textSize=0.035, nColumns=3)

                listOfProj = []
                for key in sortedKeys:
                    histo = all_procs[key]
                    keycolor = key.replace('_prefit','').replace('_postfit','')
                    if keycolor.startswith(f"{charge}_"):
                        keycolor = "_".join(keycolor.split("_")[1:])
                    if 'obs' in key or prepost not in key: continue
                    logger.info(f"{projection} projection:  {key}   {histo.GetName()}   {keycolor}")
                    if  projection == 'X':
                        proj1d = all_procs[key].ProjectionX(all_procs[key].GetName() + charge+"_px", 0, -1, "e")
                    else:
                        proj1d = all_procs[key].ProjectionY(all_procs[key].GetName() + charge+"_py", 0, -1, "e")
                    if args.normWidth:
                        proj1d.Scale(1., "width")
                    proj1d.SetFillColor(process_features[keycolor]["color"])
                    stack.Add(proj1d)
                    proj1d.Write()
                    htot.Add(proj1d) 
                    listOfProj.append([proj1d, procsAndTitles[keycolor]])

                leg.AddEntry(hdata,args.dataTitle,'PE')
                for pair in reversed(listOfProj):
                    leg.AddEntry(pair[0], pair[1], 'F')

                xaxisProj = xaxisname2D if projection == "X" else yaxisname2D
                cnameProj = f"projection{projection}_{chfl}{suffix}"
                ratioYlabel = "data/pred::" + (f"{args.ratioRangePrefit[0]},{args.ratioRangePrefit[1]}" if prepost == "prefit" else f"{args.ratioRangePostfit[0]},{args.ratioRangePostfit[1]}")
                verticalAxisNameProj = verticalAxisNameProjX if projection == "X" else verticalAxisNameProjY
                drawTH1dataMCstack(hdata, stack, xaxisProj, verticalAxisNameProj, cnameProj, outdir, leg, ratioYlabel,
                                   1, passCanvas=cnarrow, hErrStack=hexpfull, lumi=args.lumi, drawLumiLatex=True,
                                   topMargin=0.06, noLegendRatio=True)

            # hdata_unrolled = singleChargeUnrolled(infile.Get('obs'), binshift, nCharges, nMaskedChanPerCharge, 
            #                                       name=f"unrolled_{charge}_data",
            #                                       nRecoBins=nRecoBins).Clone('unrolled_{ch}_data'.format(ch=charge))
            dataName2D = f"data_{charge}"
            h2_data_backrolled = dressed2DfromFit(safeGetObject(infile, "obs"), binning, dataName2D, dataName2D,
                                                  binshift,
                                                  nMaskedCha=nMaskedChanPerCharge,
                                                  nRecoBins=nRecoBins)
            hdata_unrolled = unroll2Dto1D(h2_data_backrolled, newname=f"unrolled_{charge}_data", cropNegativeBins=False)
            hdata_unrolled_y = unroll2Dto1D(h2_data_backrolled, newname=f"unrolled_{charge}_data", cropNegativeBins=False, invertUnroll=True)

            if args.normWidth:
                normalizeTH1unrolledSingleChargebyBinWidth(hdata_unrolled, h2_data_backrolled, unrollAlongX=True)
                normalizeTH1unrolledSingleChargebyBinWidth(hdata_unrolled_y, h2_data_backrolled, unrollAlongX=False)
            hdata_unrolled.SetDirectory(0)
            htot_unrolled  = hdata_unrolled.Clone(f"unrolled_{charge}_full")
            htot_unrolled.Reset("ICES")
            htot_unrolled.Sumw2()
            htot_unrolled.SetDirectory(0)
            hdata_unrolled_y.SetDirectory(0)
            htot_unrolled_y  = hdata_unrolled_y.Clone(f"unrolled_y_{charge}_full")
            htot_unrolled_y.Reset("ICES")
            htot_unrolled_y.Sumw2()
            htot_unrolled_y.SetDirectory(0)
            stack_unrolled = ROOT.THStack(f"stack_unrolled_{prepost}_{charge}", "")
            stack_unrolled_y = ROOT.THStack(f"stack_unrolled_y_{prepost}_{charge}", "")
            leg_unrolled = prepareLegend(0.08, 0.81, 0.95, 0.90, textSize=0.045, nColumns=min(9, len(predictedProcessNames)+1))
            listOfProj = []
            for key in sortedKeys:
                histo = all_procs[key]
                keycolor = key.replace('_prefit','').replace('_postfit','')
                if keycolor.startswith(f"{charge}_"):
                    keycolor = "_".join(keycolor.split("_")[1:])
                if key=='obs' or prepost not in key: continue
                logger.info("unrolled {: >35} {: >35}   {y} ".format(key, histo.GetName(), y=str("%.3f" % histo.Integral())) )
                proc_unrolled = all_procs_unrolled[key]
                proc_unrolled.SetFillColor(process_features[keycolor]["color"])
                stack_unrolled.Add(proc_unrolled)
                htot_unrolled.Add(proc_unrolled)
                listOfProj.append([proc_unrolled, procsAndTitles[keycolor]])
                #
                proc_unrolled_y = all_procs_unrolled_y[key]
                proc_unrolled_y.SetFillColor(process_features[keycolor]["color"])
                stack_unrolled_y.Add(proc_unrolled_y)
                htot_unrolled_y.Add(proc_unrolled_y)

                
            leg_unrolled.AddEntry(hdata_unrolled, args.dataTitle, 'PE')
            for pair in reversed(listOfProj):
                leg_unrolled.AddEntry(pair[0], pair[1], 'F')

            YlabelUnroll = verticalAxisName + "::%.2f,%.2f" % (0, 2.*hdata_unrolled.GetBinContent(hdata_unrolled.GetMaximumBin()))
            # for unrolled may use always the prefit ratio for better comparison
            ratioYlabel = "data/pred::" + (f"{args.ratioRangePrefit[0]},{args.ratioRangePrefit[1]}" if prepost == "prefit" or args.unrolledRatioRangeAsPrefit else f"{args.ratioRangePostfit[0]},{args.ratioRangePostfit[1]}")

            pp = prepost.capitalize()
            if prepost == "postfit":
                fullTextForUnrolled = f"#bf{{{pp}}}:     {textForUnrolled}"
            else:
                fullTextForUnrolled = f"#bf{{{pp}}}"
            logger.info(fullTextForUnrolled)
            
            cnameUnroll = f"unrolled_{chfl}{suffix}"
            XlabelUnroll = "unrolled template along #eta:  #eta #in [%.1f, %.1f]" % (recoBins.etaBins[0], recoBins.etaBins[-1])
            drawTH1dataMCstack(hdata_unrolled, stack_unrolled, XlabelUnroll, YlabelUnroll, cnameUnroll, outdir,
                               leg_unrolled, ratioYlabel, 1, passCanvas=cwide, hErrStack=h1_expfull_unrolled, lumi=args.lumi,
                               wideCanvas=True, leftMargin=0.05,rightMargin=0.01, topMargin=0.06,
                               drawVertLines="{a},{b}".format(a=recoBins.Npt,b=recoBins.Neta),
                               textForLines=ptBinRanges, etaptbinning=binning,
                               textSize=0.04, textAngle=0, textYheightOffset=0.65, moreTextLatex=fullTextForUnrolled,
                               noLegendRatio=True)
            #
            cnameUnroll = f"unrolled_{chfl}{suffix}_unrollY"
            XlabelUnroll = "unrolled template along p_{T}:  p_{T} #in [%.1f, %.1f]" % (recoBins.ptBins[0], recoBins.ptBins[-1])
            drawTH1dataMCstack(hdata_unrolled_y, stack_unrolled_y, XlabelUnroll, YlabelUnroll, cnameUnroll, outdir,
                               leg_unrolled, ratioYlabel, 1, passCanvas=cwide, hErrStack=h1_expfull_unrolled_y, lumi=args.lumi,
                               wideCanvas=True, leftMargin=0.05,rightMargin=0.01, topMargin=0.06,
                               drawVertLines="{a},{b}".format(a=recoBins.Neta,b=recoBins.Npt),
                               textForLines=etaBinRanges, etaptbinning=binning,
                               textSize=0.04, textAngle=0, textYheightOffset=0.65, moreTextLatex=fullTextForUnrolled,
                               noLegendRatio=True)

    outfile.Close()

    ROOT.gErrorIgnoreLevel = savErrorLevel;
    copyOutputToEos(outdir_original, eoscp=args.eoscp)

