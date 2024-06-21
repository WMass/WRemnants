#!/usr/bin/env python3

# plot impacts on mW or mZ from groups of nuisance parameters

# example for W
# only main groups
# python scripts/analysisTools/w_mass_13TeV/makeImpactsOnMW.py filesFromjosh/histmaker_output_Jun11_2f929ca/WMass_eta_pt_charge/fitresults_123456789.root -o scripts/analysisTools/plots/fromMyWremnants/fitResults/fromJosh/histmaker_output_Jun11_2f929ca/WMass_eta_pt_charge/makeImpactsOnMW/ --scaleToMeV --showTotal --postfix asimov -x ".*eff_(stat|syst)|.*AlphaS$|.*nonClosure|.*resolutionCrctn|.*scaleCrctn|.*scaleClos|.*polVar|.*QCDscale$|.*QCDscale(W|Z)|.*resum|.*(muon|ecal)Prefire|FakeRate|theory_ew_|.*pixel|theory$|experiment$|bcQuark|helicity_shower"
#
# only theory subgroups
# python scripts/analysisTools/w_mass_13TeV/makeImpactsOnMW.py filesFromjosh/histmaker_output_Jun11_2f929ca/WMass_eta_pt_charge/fitresults_123456789.root -o scripts/analysisTools/plots/fromMyWremnants/fitResults/fromJosh/histmaker_output_Jun11_2f929ca/WMass_eta_pt_charge/makeImpactsOnMW/ --scaleToMeV --postfix asimov_theorySplit -x ".*" -k ".*pdf.*Alpha|.*resum.+|.*QCDscale.+|bcQuark|theory_ew.+|sin2|Zmass|width|helicity_shower" --margin 0.55,0.12,0.1,0.1
#
# only efficiency subgroups
# python scripts/analysisTools/w_mass_13TeV/makeImpactsOnMW.py filesFromjosh/histmaker_output_Jun11_2f929ca/WMass_eta_pt_charge/fitresults_123456789.root -o scripts/analysisTools/plots/fromMyWremnants/fitResults/fromJosh/histmaker_output_Jun11_2f929ca/WMass_eta_pt_charge/makeImpactsOnMW/ --scaleToMeV --postfix asimov_efficiencySplit -x ".*" -k ".*eff_(all|stat|syst)" --margin 0.45,0.12,0.1,0.1

import os, datetime, re, operator, math
import argparse
import shutil

from utilities import logging
from array import array
from copy import *

## safe batch mode
import sys
args = sys.argv[:]
sys.argv = ['-b']
import ROOT
sys.argv = args
ROOT.gROOT.SetBatch(True)
ROOT.PyConfig.IgnoreCommandLineOptions = True

import utilitiesCMG
utilities = utilitiesCMG.util()

from scripts.analysisTools.plotUtils.utility import *

def getBetterLabel(k, isWlike):
    if k == "binByBinStat":
        label = "MC_stat" if isWlike else "MCandFakes_stat"
    elif k == "stat":
        label = "data_stat"
    # elif k == "muon_eff_stat":
    #     label = "Total eff_stat"
    # elif k == "muon_eff_syst":
    #     label = "Total eff_syst"
    elif "QCDscale" in k:
        if k == "QCDscaleWMiNNLO":
            label = "Angular coefficients W"
        elif k == "QCDscaleZMiNNLO":
            label = "Angular coefficients Z"
        else:
            label = "Angular coefficients" # "A_{i}"
    else:
        label = k
    return label

def readNuisances(args, infile=None, logger=None):

    if infile is None:
        infile = args.rootfile[0]

    if logger == None:
        logger = logging.setup_logger(os.path.basename(__file__), 3)

    logger.info(f"Starting with file {infile} ...")

    #massNuisanceName = "WmassShift{s}MeV".format(s=int(args.prefitUncertainty))
    massNuisanceName = ".*massShift.*{s}MeV$".format(s=int(args.prefitUncertainty))
    #massNuisanceName = "massShift{s}MeV".format(s=int(args.prefitUncertainty))
    valuesAndErrors = utilities.getFromHessian(infile,params=[massNuisanceName])
    if len(valuesAndErrors) == 1:
        masskey = list(valuesAndErrors.keys())[0]
        totalUncertainty = valuesAndErrors[masskey][1] - valuesAndErrors[masskey][0]
    else:
        error_msg = f"Found more parameters matching expected mass expression: {valuesAndErrors.keys()}"
        raise IOError(error_msg)
        #totalUncertainty = valuesAndErrors[massNuisanceName][1] - valuesAndErrors[massNuisanceName][0]

    if args.scaleToMeV:
        totalUncertainty *= args.prefitUncertainty
        logger.info("Total m%s uncertainty: %2.2f MeV" % (boson, totalUncertainty))
    else:
        logger.info("Total m%s uncertainty: %2.3f (\% of prefit)" % (boson, totalUncertainty))
        
    group = 'group_' if len(args.nuisgroups) else ''
    th2name = 'nuisance_{group}impact_nois'.format(group=group)
    hessfile = ROOT.TFile(infile,'read')
    impMat = hessfile.Get(th2name)
    if impMat == None:
        error_msg = f"Cannot find the impact TH2 named {th2name} in input file. Maybe you didn't run --doImpacts?"
        raise IOError(error_msg)

    if args.keepNuisgroups != None:
        matchKeep = re.compile(args.keepNuisgroups)
    if args.excludeNuisgroups:
        matchExclude = re.compile(args.excludeNuisgroups)
    
    logger.info("Histograms loaded successfully ...")
    nuisGroup_nameVal = {}
    for iy in range(1,impMat.GetNbinsY()+1):
        label = impMat.GetYaxis().GetBinLabel(iy)
        if args.keepNuisgroups and matchKeep.match(label):
            nuisGroup_nameVal[label] = impMat.GetBinContent(1,iy)
        elif args.excludeNuisgroups and matchExclude.match(label):
            continue
        nuisGroup_nameVal[label] = impMat.GetBinContent(1,iy)
    return totalUncertainty,nuisGroup_nameVal


if __name__ == "__main__":

    ROOT.gStyle.SetOptStat(0)

    parser = common_plot_parser()
    parser.add_argument("rootfile", type=str, nargs=1)
    parser.add_argument('-o','--outdir',     default='./makeImpactsOnMW/',   type=str, help='output directory to save the plot (not needed with --justPrint)')
    parser.add_argument(     '--nuisgroups', default='ALL',   type=str, help='nuis groups for which you want to show the impacts (can pass comma-separated list to make all of them one after the other). Use full name, no regular expressions. By default, all are made')
    parser.add_argument('-k',  '--keepNuisgroups', default=None,   type=str, help='nuis groups for which you want to show the impacts, using regular expressions')
    parser.add_argument('-x', '--excludeNuisgroups', default=".*eff_(stat|syst)|.*AlphaS$|.*nonClosure|.*resolutionCrctn|.*scaleCrctn|.*scaleClos|.*polVar|.*QCDscale$|.*QCDscale(W|Z)|.*resum|.*(muon|ecal)Prefire|FakeRate|theory_ew_|.*pixel|theory$|experiment$|bcQuark|helicity_shower",   type=str, help='Regular expression for nuisances to be excluded (note that it wins against --keepNuisgroups since evaluated before it')
    parser.add_argument(     '--setStat',   default=-1.0, type=float, help='If positive, use this value for stat (this is before scaling to MeV) until combinetf is fixed')
    parser.add_argument(     '--postfix',     default='',   type=str, help='postfix for the output name')
    parser.add_argument(     '--canvasSize', default='800,1200', type=str, help='Pass canvas dimensions as "width,height" ')
    # parser.add_argument(     '--draw-option', dest='drawOption', default='COLZ TEXT', type=str, help='Options for drawing TH2')
    parser.add_argument(     '--margin',   default='', type=str, help='Pass canvas margin as "left,right,top,bottom" ')
    parser.add_argument(     '--scaleToMeV', action='store_true',   help='Report numbers in terms of uncertainty on mW in MeV (default is to report percentage of prefit uncertainty)')
    parser.add_argument(     '--showTotal', action='store_true',   help='Show total uncertainty in plot')
    parser.add_argument(     '--prefitUncertainty',      default=100.0, type=float, help='prefit uncertainty on mW in MeV')
    parser.add_argument(     '--wlike', dest='isWlike', action="store_true", help="impacts for W-like analysis (it prints mZ accordingly). Default is Wmass");
    parser.add_argument(     '--compareFile', default='', type=str, help='Additional file to compare impacts with (must have the same impact labels)')
    parser.add_argument(     '--setStatAlt', default=-1.0, type=float, help='If positive, use this value for stat of the file passed with compareFile, otherwise use the same as the other file')
    parser.add_argument(     '--legendEntries', nargs=2, type=str, help="Legend entries when comparing files", default=["Nominal", "Alternate"])
    parser.add_argument(     '--printAltVal', action='store_true', help='When comparing to a second file, also print the values for the alternative')
    parser.add_argument(     '--justPrint', action='store_true', help='Print without plotting')
    args = parser.parse_args()

    logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)
    # palettes:
    # 69 + inverted, using TColor::InvertPalette(), kBeach
    # 70 + inverted, using TColor::InvertPalette(), kBlackBody
    # 109, kCool
    # 57 kBird
    # 55 kRainBow
    # 62 kColorPrintableOnGrey
    # 73 kCMYK
    # 58 kCubehelix
    # 68 kAvocado
    # 111 kGistEarth + inverted
    # 87 kLightTemperature
    # 85 kIsland
    # 56 kInvertedDarkBodyRadiator
    # 100 kSolar + inverted

    if "ZMassWLike" in args.rootfile[0] and not args.isWlike:
        logger.warning(f"ZMassWLike found in input path {args.rootfile[0]}, but option --wlike not specified, please check")
        quit()
    
    ROOT.TColor.CreateGradientColorTable(3,
                                         array ("d", [0.00, 0.50, 1.00]),
                                         ##array ("d", [1.00, 1.00, 0.00]),
                                         ##array ("d", [0.70, 1.00, 0.34]),
                                         ##array ("d", [0.00, 1.00, 0.82]),
                                         array ("d", [0.00, 1.00, 1.00]),
                                         array ("d", [0.34, 1.00, 0.65]),
                                         array ("d", [0.82, 1.00, 0.00]),
                                         255,  0.95)


    absValue = False
    if len(args.nuisgroups): absValue = True

    if absValue:    
        ROOT.TColor.CreateGradientColorTable(2,
                                             array ("d", [0.00, 1.00]),
                                             ##array ("d", [1.00, 1.00, 0.00]),
                                             ##array ("d", [0.70, 1.00, 0.34]),
                                             ##array ("d", [0.00, 1.00, 0.82]),
                                             array ("d", [1.00, 1.00]),
                                             array ("d", [1.00, 0.65]),
                                             array ("d", [1.00, 0.00]),
                                             255,  0.95)

    boson = "Z" if args.isWlike else "W"
    
    compare = True if len(args.compareFile) else False
    if not compare and args.printAltVal:
        error_msg = "--printAltVal only works with --compareFile. Please try again"
        raise IOError(error_msg)
    
    totalUncertainty_mW, nuisGroup_nameVal = readNuisances(args, args.rootfile[0], logger=logger)
    if args.setStat > 0.0:
        nuisGroup_nameVal["stat"] = args.setStat

    if compare:
        totalUncertainty_mW_alt, nuisGroup_nameVal_alt = readNuisances(args, args.compareFile, logger=logger)
        if args.setStatAlt > 0.0:
            nuisGroup_nameVal_alt["stat"] = args.setStatAlt

    sortedGroups = sorted(nuisGroup_nameVal.keys(), key= lambda x: nuisGroup_nameVal[x])

    ROOT.gStyle.SetPaintTextFormat('2.1f' if args.scaleToMeV else '0.3f')

    logger.info("Creating output histogram ...")
    # add 1 more bin for total
    nbins = len(sortedGroups)
    if args.showTotal:
        nbins += 1

    # new version
    h1 = ROOT.TH1D("impactsOnMw_chart","impacts of nuisance groups on m_{%s}" % boson,nbins,0,nbins)
    if compare:
        h2 = h1.Clone("impactsOnMw_chart_alt")
    h1.GetYaxis().SetTitle("impacts on m_{{{boson}}} {units}".format(boson=boson, units="[MeV]" if args.scaleToMeV else ""))
    h1.GetYaxis().SetTitleOffset(1.05)
    h1.GetYaxis().SetTitleSize(0.045)
    h1.GetYaxis().SetLabelSize(0.04)
    h1.GetYaxis().SetNdivisions(505)
    #h1.GetYaxis().SetTitle("")
    for ik,k in enumerate(sortedGroups):
        bincontent = nuisGroup_nameVal[k] if not args.scaleToMeV else nuisGroup_nameVal[k] * args.prefitUncertainty
        label = getBetterLabel(k, args.isWlike)
        if compare:
            #print(k)
            bincontentAlt = nuisGroup_nameVal_alt[k] if k in nuisGroup_nameVal_alt.keys() else 0.0
            if args.scaleToMeV: bincontentAlt *= args.prefitUncertainty
            logger.info("%s: %2.3f / %2.3f" % (label, bincontent, bincontentAlt))
        else:
            logger.info("%s: %2.3f" % (label, bincontent))
        h1.GetXaxis().SetBinLabel(ik+1, label)
        h1.SetBinContent(ik+1,bincontent)
        if compare:
            h2.SetBinContent(ik+1,bincontentAlt)

    if args.justPrint:
        quit()
            
    if args.showTotal:
        h1.GetXaxis().SetBinLabel(nbins,"total")
        h1.SetBinContent(nbins,totalUncertainty_mW)
        if compare:
            h2.SetBinContent(nbins,totalUncertainty_mW_alt)
            
    h1.GetXaxis().SetTitleOffset(1.2)
    h1.GetXaxis().SetTitleSize(0.05)
    h1.GetXaxis().SetLabelSize(0.05)

    h1.SetBarWidth(0.8);
    h1.SetBarOffset(0.1);
    h1.SetFillColor(ROOT.TColor.GetColor("#5790fc"))   # used to be ROOT.kGreen-5
    h1.SetLineColor(ROOT.kBlack)
    #h1.SetFillStyle(3001)
    #h1.SetFillColorAlpha(ROOT.TColor.GetColor("#5790fc"), 0.5)
    if compare:
        h2.SetBarWidth(0.4);
        h2.SetBarOffset(0.2);
        h2.SetFillColor(ROOT.TColor.GetColor("#bd1f01")) # "#f89c20" is orange, used to be ROOT.kPink-6
        h2.SetLineColor(ROOT.kBlack)
        #h2.SetFillStyle(3001)
        h2.SetFillColorAlpha(ROOT.TColor.GetColor("#f89c20"), 0.9)
        h1.GetYaxis().SetRangeUser(0.0, 1.1* max(totalUncertainty_mW_alt, totalUncertainty_mW))
        
    cw,ch = args.canvasSize.split(',')
    c1 = ROOT.TCanvas("c1", "", int(cw), int(ch))
    #c1.SetFillColor(42)
    c1.SetGridx()
    c1.SetGridy()
    #ROOT.gPad.SetFrameFillColor(33)
    
    clm = 0.4
    crm = 0.16 if args.printAltVal else 0.12
    cbm = 0.1
    ctm = 0.1
    if args.margin:
        clm,crm,ctm,cbm = (float(x) for x in args.margin.split(','))
    c1.SetLeftMargin(clm)
    c1.SetRightMargin(crm)
    c1.SetBottomMargin(cbm)
    c1.SetTopMargin(ctm)
    c1.SetTickx(1)
    c1.SetTicky(1)
    h1.Draw("hbar1")
    if compare:
        h2.Draw("hbar1 SAME")
        leg = ROOT.TLegend(0.1,0.9,0.92,0.95)
        leg.SetNColumns(2)
        leg.SetFillColor(0)
        leg.SetFillColorAlpha(0,0.6)
        leg.SetShadowColor(0)
        leg.SetLineColor(0)
        leg.SetBorderSize(0)
        leg.SetTextFont(42)
        leg.SetTextSize(0.035)
        leg.AddEntry(h1, args.legendEntries[0], "LF")
        leg.AddEntry(h2, args.legendEntries[1], "LF")
        leg.Draw("SAME")
        
    #hval = copy.deepcopy(h1.Clone("hval"))
    hval = h1.Clone("hval")
    hval.Reset("ICESM")
    hval.GetXaxis().SetTitleOffset(1.2)
    hval.GetXaxis().SetTitleSize(0.05)
    hval.GetXaxis().SetLabelSize(0.05)
    lat = ROOT.TLatex()
    #lat.SetNDC();
    lat.SetTextFont(42)        
    lat.SetTextSize(0.035)
    latAlt = ROOT.TLatex()
    latAlt.SetTextFont(42)        
    latAlt.SetTextSize(0.025)
    latAlt.SetTextColor(ROOT.kPink-6)
    c1.Update()
    xtex = 1.05 * (ROOT.gPad.GetUxmax() - ROOT.gPad.GetUxmin())
    xtexAlt = 1.22 * (ROOT.gPad.GetUxmax() - ROOT.gPad.GetUxmin())
    step = (ROOT.gPad.GetUymax() - ROOT.gPad.GetUymin()) / h1.GetNbinsX()
    ytex = ROOT.gPad.GetUymin() + 0.25 * step
    ytexAlt = ytex # ROOT.gPad.GetUymin() + 0.07 * step
    for i in range(1, 1 + h1.GetNbinsX()):
        hval.GetXaxis().SetBinLabel(i, str(round(h1.GetBinContent(i), 1 if args.scaleToMeV else 3)))
        hval.SetBinContent(i, 0.0)
        lat.DrawLatex(xtex, ytex + step * (i-1), hval.GetXaxis().GetBinLabel(i))
        if args.printAltVal:
            altVal = str(round(h2.GetBinContent(i), 1 if args.scaleToMeV else 3))
            latAlt.DrawLatex(xtexAlt, ytexAlt + step * (i-1), altVal)
    hval.Draw("AXIS X+ SAME")
        
    postfix = args.postfix
    if len(postfix) and  not postfix.startswith("_"):
        postfix = "_" + postfix
    smallBoson = "z" if args.isWlike else "w"
    outdir_original = args.outdir
    outdir = createPlotDirAndCopyPhp(outdir_original, eoscp=args.eoscp)
    for ext in ["pdf", "png"]:
        c1.SaveAs(f"{outdir}/impactsOnM{smallBoson}{postfix}.{ext}")
    copyOutputToEos(outdir_original, eoscp=args.eoscp)
