#!/usr/bin/env python3

# script to make ratio of two TH2 histograms
# the two objects can have different bin ranges, but in order for the ratio to make sense, 
# the binning of the smaller TH2 should be a subset of the other histogram's binning
# In any case, the ratio is made looping on the first histogram passed (which is cloned to get a new one with the same binning) and getting the bin center
# for each bin, then that value is used to look for the bin of the other histogram
# therefore, in case the two TH2 have different binning, make sure that the first one has the finer granularity

################################
# Examples
################################

# python w-helicity-13TeV/makeRatioTH2.py ~/www/wmass/13TeV/scaleFactors/electron/fullID_extPt/smoothEfficiency_electrons_fullID.root scaleFactor ~/www/wmass/13TeV/scaleFactors/electron/fullID_noErfPlusLine/smoothEfficiency_electrons_fullID.root scaleFactor -o ~/www/wmass/13TeV/scaleFactors/ratio/electron/fullID_extPt__over__fullID/ -f ratio.root -n ratio2D -t "full ID scale factor ratio" -z ratio --ratioRange 0.9 1.1

# python w-helicity-13TeV/makeRatioTH2.py ../../data/fakerate/fakeRateSmoothed_el_mT40_35p9fb_signedEta_pt65_fullWMC_newTrigSF_fitpol2.root fr_pt_eta_ewk ~/www/wmass/13TeV/fake-rate/electron/FR_graphs/fakeRate_eta_pt_granular_mT40_35p9fb_signedEta_pt65_subtrAllMC_newTrigSF_fitpol2/fakeRateSmoothed_el_mT40_35p9fb_signedEta_pt65_subtrAlllMC_newTrigSF_fitpol2.root fr_pt_eta_ewk -o ~/www/wmass/13TeV/fake-rate/electron/FR_graphs/fakeRate_eta_pt_granular_mT40_35p9fb_signedEta_pt65_subtrAllMC_newTrigSF_fitpol2/ -f ratio_PR_WZ_allMC.root -n ratioPR_WZ_allMC -t "PR W,Z MC / all MC" -z "PR ratio" -r 0.98 1.02

# python w-helicity-13TeV/makeRatioTH2.py ../../data/fakerate/fakeRateSmoothed_el_mT40_35p9fb_signedEta_pt65_subtrAlllMC_newTrigSF_fitpol2.root fr_pt_eta_ewk ~/www/wmass/13TeV/fake-rate/electron/FR_graphs/fakeRate_eta_pt_granular_mT40_35p9fb_signedEta_pt65_subtrAllMC_newTrigSF_fitpol2_testTrigSF/fakeRateSmoothed_el_mT40_35p9fb_signedEta_pt65_subtrAlllMC_newTrigSF_fitpol2_testTrigSF.root fr_pt_eta_ewk -o ~/www/wmass/13TeV/fake-rate/electron/FR_graphs/fakeRate_eta_pt_granular_mT40_35p9fb_signedEta_pt65_subtrAllMC_newTrigSF_fitpol2/ -f ratioPR_nomi_test.root -n FILE -t "PR nomi / test" -z "PR ratio" -r 0.99 1.01

# python w-helicity-13TeV/makeRatioTH2.py /afs/cern.ch/user/m/mciprian/www/wmass/13TeV/scaleFactors/electron/trigger_extPt_30_45/smoothEfficiency_electrons_trigger.root scaleFactor /afs/cern.ch/user/m/mciprian/www/wmass/13TeV/scaleFactors/electron/trigger_extPt/smoothEfficiency_electrons_trigger.root scaleFactor -o /afs/cern.ch/user/m/mciprian/www/wmass/13TeV/scaleFactors/ratio/electron/trigger_ptUpTo45__Over__ptUpTo55/ -f trgSF_pt45_pt55.root -n FILE -t "p_{T} < 45 / p_{T} < 55" -z "trigger scale factor ratio" -r 0.95 1.05

# FR and PR ratio

# python w-helicity-13TeV/makeRatioTH2.py /afs/cern.ch/user/m/mciprian/www/wmass/13TeV/fake-rate/electron/FR_graphs_tests/fr_16_09_2018_eta_pt_granular_mT40_35p9fb_signedEta_subtrAllMC_allNewSF_fitPol1_minPtData32/fakeRateSmoothed_el_fr_16_09_2018_eta_pt_granular_mT40_35p9fb_signedEta_subtrAllMC_allNewSF_fitPol1_minPtData32.root fr_pt_eta_data /afs/cern.ch/user/m/mciprian/www/wmass/13TeV/fake-rate/electron/FR_graphs_tests/fr_18_09_2018_eta_pt_granular_mT40_35p9fb_signedEta_subtrAllMC_allNewSF_fitPol1_minPtData32_jetPt45/fakeRateSmoothed_el_fr_18_09_2018_eta_pt_granular_mT40_35p9fb_signedEta_subtrAllMC_allNewSF_fitPol1_minPtData32_jetPt45.root fr_pt_eta_data -o /afs/cern.ch/user/m/mciprian/www/wmass/13TeV/fake-rate/electron/ratio_FR_PR/subtrAllMC_allNewSF_fitPol1_minPtData32__jetPt30_Over_jetPt45/ -f ratio_FR -n FILE -t "FR jet p_{T} > 30 / jet p_{T} > 45" -z "Fake rate ratio" -r 0.9 1.1

# python w-helicity-13TeV/makeRatioTH2.py /afs/cern.ch/user/m/mciprian/www/wmass/13TeV/fake-rate/electron/FR_graphs_tests/fr_16_09_2018_eta_pt_granular_mT40_35p9fb_signedEta_subtrAllMC_allNewSF_fitPol1_minPtData32/fakeRateSmoothed_el_fr_16_09_2018_eta_pt_granular_mT40_35p9fb_signedEta_subtrAllMC_allNewSF_fitPol1_minPtData32.root fr_pt_eta_ewk /afs/cern.ch/user/m/mciprian/www/wmass/13TeV/fake-rate/electron/FR_graphs_tests/fr_18_09_2018_eta_pt_granular_mT40_35p9fb_signedEta_subtrAllMC_allNewSF_fitPol1_minPtData32_jetPt45/fakeRateSmoothed_el_fr_18_09_2018_eta_pt_granular_mT40_35p9fb_signedEta_subtrAllMC_allNewSF_fitPol1_minPtData32_jetPt45.root fr_pt_eta_ewk -o /afs/cern.ch/user/m/mciprian/www/wmass/13TeV/fake-rate/electron/ratio_FR_PR/subtrAllMC_allNewSF_fitPol1_minPtData32__jetPt30_Over_jetPt45/ -f ratio_PR -n FILE -t "PR jet p_{T} > 30 / jet p_{T} > 45" -z "Prompt rate ratio" -r 0.98 1.04

# muon FR (need to build the FR versu pt and eta
# python w-helicity-13TeV/makeRatioTH2.py ../../data/fakerate/frAndPr_fit_mu_2018-09-13_finerETA.root fakerates_smoothed_data_interpolated ../../data/fakerate/frAndPr_fit_mu_2018-09-19_jetPt45_finerETA.root fakerates_smoothed_data_interpolated_awayJetPt45 -o /afs/cern.ch/user/m/mciprian/www/wmass/13TeV/fake-rate/muon/ratio_FR_PR/fromMarc_jetPt30_Over_jetPt45/ -f ratio_FR -n FILE -t "FR jet p_{T} > 30 / jet p_{T} > 45" -z "Fake rate ratio" -r 0.9 1.2   --buildFakeRate -x "muon p_{T} [GeV]" -y "muon #eta" --h1Dbinning "75,0.9,1.2"

# python w-helicity-13TeV/makeRatioTH2.py plots/distribution/muonPlots/SKIMS_muons_latest/ptVsEta_W_metJecUp_plus/test_plots.root ptl1__etal1_W plots/distribution/muonPlots/SKIMS_muons_latest/ptVsEta_W_metNominal_plus/test_plots.root ptl1__etal1_W -o plots/distribution/muonPlots/SKIMS_muons_latest/ratio_metJecUpMT/ -f ratio_MetJecUp_nominal_W -n FILE -z "yields ratio for W^{+}" --ratioRange 0.97 1.005 -t "E_{T}^{miss}(JEC Up) / nominal"

################################
################################


import os, array, math
import argparse
from copy import *

## safe batch mode
import sys
args = sys.argv[:]
sys.argv = ['-b']
import ROOT
sys.argv = args
ROOT.gROOT.SetBatch(True)
ROOT.PyConfig.IgnoreCommandLineOptions = True

#sys.path.append(os.getcwd() + "/plotUtils/")
#from utility import *
from scripts.analysisTools.plotUtils.utility import *

if __name__ == "__main__":
            
    parser = argparse.ArgumentParser()
    parser.add_argument("file1", type=str, nargs=1, help="First file")
    parser.add_argument("hist1", type=str, nargs=1, help="Histogram name from first file")
    parser.add_argument("file2", type=str, nargs=1, help="Second file (can use 'SAME' if equal to first file)")
    parser.add_argument("hist2", type=str, nargs=1, help="Histogram name from second file")
    parser.add_argument('-o','--outdir',  default='', type=str, help='output directory to save things')
    parser.add_argument('-f','--outfilename', default='', type=str, help='Name of output file to save results')
    parser.add_argument('-n','--outhistname', default='', type=str, help='Name of output histogram saved in output file. If FILE is used, take same name as the output file, removing the extension')
    parser.add_argument('-x','--xAxisTitle',  default='', type=str, help='X axis title. If not given, use the one from hist1')
    parser.add_argument('-y','--yAxisTitle',  default='', type=str, help='Y axis title. If not given, use the one from hist1')
    parser.add_argument('-z','--zAxisTitle',  default='', type=str, help='Z axis title. If not given, use the one from hist1')
    parser.add_argument('-t','--histTitle',   default='', type=str, help='Title to assign to output histogram. It is used as a label for the canvas')
    parser.add_argument('-r','--ratioRange',  default=(0, -1),type=float, nargs=2, help="Min and max for the ratio in the plot")
    parser.add_argument(     '--h1Dbinning',  default='50,0.9,1.1', type=str, help='Comma separated list of 3 numbers: nbins,min,max')
    parser.add_argument('-v','--valBadRatio',  default='0', type=float, help='Value to be used in case of bad ratio (division by 0). The 1D histogram is not filled in case of bad ratio')
    parser.add_argument(     '--buildFakeRate', action="store_true", help="The input histograms have the parameters of the linear fits to fake-rate or prompt-rate versus eta: build the histogram with FR (PR) vs pt and eta (obsolete, no longer using pol1 to interpolate)")
    parser.add_argument(     '--skip1DPlot', action="store_true", help="Do not plot 1D distribution")
    parser.add_argument(     '--xRange',  default=(0,-1), type=float, nargs=2, help='Select range for X axis to plot. Also, bins outside this range are not considered in the 1D histogram. If min > max, the option is neglected')
    parser.add_argument(     '--yRange', default=(0,-1), type=float, nargs=2, help='Select range for Y axis to plot. Also, bins outside this range are not considered in the 1D histogram. If min > max, the option is neglected')
    parser.add_argument('-e', '--divide-error', dest="divideError", action="store_true", help="Make ratio of uncertainties (the output histogram will have no error assigned to it)")
    parser.add_argument('-E',  '--divide-relative-error', dest="divideRelativeError", action="store_true", help="Make ratio of relative uncertainties (the output histogram will have no error assigned to it)")
    parser.add_argument(     '--palette'  , default=55, type=int, help='Set palette: use a negative number to select a built-in one, otherwise the default is 55 (kRainbow)')
    parser.add_argument('-a', '--make-asymmetry', dest="makeAsymmetry", action="store_true", help="Make ratio of difference over the sum. For this to make sense, the binning of the two inputs must be consistent")
    parser.add_argument('-p', '--make-pulls', dest="makePulls", action="store_true", help="Make pulls of input histograms, i.e. (h1-h2)/error, where error is taken as the quadrature sum of the errors of the input")
    parser.add_argument(       '--pull-error-ScaleFactor', dest='pullErrorScaleFactor', default='1.', type=float, help='Inflate the error by this factor when making the pulls (because it is assumed the inputs are uncorrelated, so the error might need a correction)')
    parser.add_argument('-u', '--unroll', action="store_true",  help="Make plot of unrolled 1D histogram from the 2D ratio (along x)")
    parser.add_argument(      '--unrolly', action="store_true",  help="Unroll along y instead of along x")
    parser.add_argument(      '--roll1Dto2D', action="store_true",  help="Input histograms are 1D distributions to be unrolled into 2D. Need binning from option --binning-file-to-roll")
    parser.add_argument(      '--binning-file-to-roll', dest="binFileToRoll", default="", help="File with binning to roll 1D into 2D (the reco binning is used)")
    parser.add_argument(      '--drawOption',  default='colz0', type=str, help='Draw option for TH2')
    parser.add_argument(      '--set-ratio-unc', dest="setRatioUnc", default="num", choices=["num", "den", "both"], help="Set how to compute uncertainty for the ratio (num or den simply use the one of those component, neglecting the other, both propagates both) ")
    args = parser.parse_args()
    
    f1 = args.file1[0]
    h1 = args.hist1[0]
    f2 = f1 if args.file2[0]  == "SAME" else args.file2[0]
    h2 = args.hist2[0]

    ROOT.TH1.SetDefaultSumw2()

    print("")

    if args.outdir:
        outname = args.outdir
        addStringToEnd(outname,"/",notAddIfEndswithMatch=True)
        createPlotDirAndCopyPhp(outname)
    else:
        print("Error: you should specify an output folder using option -o <name>. Exit")
        quit()
    if not args.outfilename and args.outhistname == "FILE":
        print("Error: to use -n FILE you should specify an output file name using option -f <name>. Exit")
        quit()
    if not args.outhistname:
        print("Error: you should specify an output histogram name using option -n <name>. ")
        print("If FILE is used, take same name as the output file, removing the extension")
        print("Exit")
        quit()

    if args.makeAsymmetry:
        if args.divideRelativeError or args.divideError:
            print("Error: option -a incompatible with options -e and -E. Exit")
            quit()
    if args.buildFakeRate: args.makePulls = False
    if args.makePulls:
        if args.divideRelativeError or args.divideError:
            print("Error: option -p incompatible with options -e and -E. Exit")
            quit()
    if args.makeAsymmetry and args.makePulls:
            print("Error: option -a incompatible with options -p. Exit")
            quit()


    if args.outhistname == "FILE":
        args.outhistname = args.outfilename.split('.')[0]

    hratio = 0

    # file 1
    tf = safeOpenFile(f1)
    hist1 = safeGetObject(tf, h1)
    tf.Close()

    # file2
    tf = safeOpenFile(f2)
    hist2 = safeGetObject(tf, h2)
    tf.Close()

    hinput1 = hist1
    hinput2 = hist2

    if args.buildFakeRate:
        # in this case the input TH2 have eta on x axis and offset/slope on the other one
        # this is needed only for muons, for electrons I save directly the smoothed FR (PR)
        neta = hist1.GetNbinsX()
        etabins = [hist1.GetXaxis().GetBinLowEdge(i) for i in range(1,2+neta)]
        #etamin = hist1.GetXaxis().GetBinLowEdge(1)
        #etamax = hist1.GetXaxis().GetBinLowEdge(1+neta)
        #hFR1 = ROOT.TH2D(hist1.GetName()+"_FRorPR","",195,26,65,neta,etamin,etamax)
        #hFR2 = ROOT.TH2D(hist2.GetName()+"_FRorPR","",195,26,65,neta,etamin,etamax)
        hFR1 = ROOT.TH2D(hist1.GetName()+"_FRorPR","",195,26,65,neta,array('d',etabins))
        hFR2 = ROOT.TH2D(hist2.GetName()+"_FRorPR","",195,26,65,neta,array('d',etabins))
        for ix in range (1,1+hFR1.GetNbinsX()):
            for iy in range (1,1+hFR1.GetNbinsY()):
                fr1 = hist1.GetBinContent(iy,1) + hist1.GetBinContent(iy,2) * hFR1.GetXaxis().GetBinCenter(ix)
                hFR1.SetBinContent(ix,iy,fr1)
                fr2 = hist2.GetBinContent(iy,1) + hist2.GetBinContent(iy,2) * hFR2.GetXaxis().GetBinCenter(ix)
                hFR2.SetBinContent(ix,iy,fr2)
        hinput1 = hFR1
        hinput2 = hFR2

    if args.roll1Dto2D:
      # TO DO   
        if args.binFileToRoll:
            etaPtBinningVec = getDiffXsecBinning(args.binFileToRoll, "reco")
            recoBins = templateBinning(etaPtBinningVec[0],etaPtBinningVec[1])
            #following array is used to call function dressed2D()         
            binning = [recoBins.Neta, recoBins.etaBins, recoBins.Npt, recoBins.ptBins]
            hinput1 = dressed2D(hist1,binning,hist1.GetName()+"_rolled2D",hist1.GetName()+"_rolled2D")
            hinput2 = dressed2D(hist2,binning,hist2.GetName()+"_rolled2D",hist2.GetName()+"_rolled2D")
        else:
            print("Error: you need to pass the binning to roll 1D histograms into 2D")
            quit()

    xMin = args.xRange[0]
    xMax = args.xRange[1]
    yMin = args.yRange[0]
    yMax = args.yRange[1]

    hratio = hinput1.Clone(args.outhistname)
    hratio.SetTitle(args.histTitle)                
                
    hsum = None
    if args.makeAsymmetry:
        hsum = hinput1.Clone("diff")
        hsum.Add(hinput2)
        hratio.Add(hinput2, -1.)
        #hratio.Divide(hsum) # do this below
        

    nbins,minx,maxx = args.h1Dbinning.split(',')
    hratioDistr = ROOT.TH1D(args.outhistname+"_1D","Distribution of ratio values",int(nbins),float(minx),float(maxx))
    #print("minx, maxx = %s, %s" % (str(minx),str(maxx)))
    nout = 0

    for ix in range(1,1+hratio.GetNbinsX()):
        for iy in range(1,1+hratio.GetNbinsY()):
            xval = hratio.GetXaxis().GetBinCenter(ix) 
            yval = hratio.GetYaxis().GetBinCenter(iy) 
            hist2xbin = hinput2.GetXaxis().FindFixBin(xval)
            hist2ybin = hinput2.GetYaxis().FindFixBin(yval)
            if xMin < xMax:
                if xval < xMin or xval > xMax: continue
            if yMin < yMax:
                if yval < yMin or yval > yMax: continue
            denval = 0
            if args.makeAsymmetry:
                denval = hsum.GetBinContent(hist2xbin, hist2ybin)
            else:
                denval = hinput2.GetBinError(hist2xbin, hist2ybin) if args.divideError else hinput2.GetBinContent(hist2xbin, hist2ybin)
            if args.divideRelativeError:
                if hinput2.GetBinContent(hist2xbin, hist2ybin) != 0:
                    denval = hinput2.GetBinError(hist2xbin, hist2ybin) / hinput2.GetBinContent(hist2xbin, hist2ybin)
                else:
                    denval = 0
            if denval != 0:                
                numval = hratio.GetBinError(ix,iy) if args.divideError else hratio.GetBinContent(ix,iy)
                if args.divideRelativeError:
                    if hratio.GetBinContent(ix,iy) != 0:
                        numval = hratio.GetBinError(ix,iy) / hratio.GetBinContent(ix,iy)
                    else:
                        numval = 0
                ratio = numval / denval
                hratioDistr.Fill(ratio)
                hratio.SetBinContent(ix,iy,ratio)
                if args.divideRelativeError or args.divideError:
                    hratio.SetBinError(ix,iy, 0.0)
                else:
                    if args.setRatioUnc == "den":
                        hratio.SetBinError(ix, iy, hinput2.GetBinError(hist2xbin, hist2ybin))
                    elif args.setRatioUnc == "zero":
                        hratio.SetBinError(ix, iy, 0.0)
                    elif args.setRatioUnc == "both":
                        print("Error: '--set-ratio-unc' both not implemented yet. Abort")
                        quit()
                    hratio.SetBinError(ix,iy, hratio.GetBinError(ix,iy)/denval)
                #print(f"{ix} {iy} {ratio}")
                if ratio < float(minx) or ratio > float(maxx): nout += 1
                #profX.Fill()
            else: 
                print("Warning: found division by 0 in one bin: setting ratio to " + str(args.valBadRatio))
                hratio.SetBinContent(ix,iy,args.valBadRatio)
                hratio.SetBinError(ix,iy, 0.0)

    print("nout = " + str(nout) + f" outside [{minx}, {maxx}]")

    if args.xAxisTitle: hratio.GetXaxis().SetTitle(args.xAxisTitle)    
    if args.yAxisTitle: hratio.GetYaxis().SetTitle(args.yAxisTitle)    
    if args.zAxisTitle: hratio.GetZaxis().SetTitle(args.zAxisTitle)    
    xAxisTitle = hratio.GetXaxis().GetTitle()
    yAxisTitle = hratio.GetYaxis().GetTitle()
    zAxisTitle = hratio.GetZaxis().GetTitle()

    if xMax > xMin and not "::" in xAxisTitle: 
        xAxisTitle = xAxisTitle + "::" + str(args.xRange[0]) + "," + str(args.xRange[1])
    if yMax > yMin and not "::" in yAxisTitle: 
        yAxisTitle = yAxisTitle + "::" + str(args.yRange[0]) + "," + str(args.yRange[1])

    # print("xAxisTitle = " + xAxisTitle
    # print("yAxisTitle = " + yAxisTitle
    # print("zAxisTitle = " + zAxisTitle

    adjustSettings_CMS_lumi()

    canvas2D = ROOT.TCanvas("canvas2D","",700,700)
    
    # the axis name can be used to set the range if it is in the format "name::min,maz"
    # if this is not already the case, use the selected range from the input option
    if not "::" in zAxisTitle:
        if args.ratioRange[0] > args.ratioRange[1]:
            args.ratioRange = (hratio.GetBinContent(hratio.GetMinimumBin()), hratio.GetBinContent(hratio.GetMaximumBin()))
        zAxisTitle = zAxisTitle + "::" + str(args.ratioRange[0]) + "," + str(args.ratioRange[1])
    drawCorrelationPlot(hratio,xAxisTitle,yAxisTitle,zAxisTitle,
                        args.outhistname,"ForceTitle",outname,0,0,False,False,False,1,
                        palette=args.palette,passCanvas=canvas2D,drawOption=args.drawOption)
    
    canvas = ROOT.TCanvas("canvas","",800,700)
    if not args.skip1DPlot:
        drawTH1(hratioDistr, 
                hratio.GetZaxis().GetTitle() if args.zAxisTitle else "ratio",
                "number of events",
                f"ratioDistribution_{args.outhistname}",
                outname,
                passCanvas=canvas
        )

    # unroll 2D ratio into a 1D
    if args.unroll or args.unrolly:
        canvas_unroll = ROOT.TCanvas("canvas_unroll","",3000,800)
        bottomMargin = 0.12
        canvas_unroll.SetTickx(1)
        canvas_unroll.SetTicky(1)
        canvas_unroll.cd()
        canvas_unroll.SetBottomMargin(bottomMargin)                                            

        unrollNameID = "unrolled"
        xAxisTitle_unroll = f"Unrolled bin: '{yAxisTitle}' vs '{xAxisTitle}'"
        vertLineDivisions="{a},{b}".format(a=hratio.GetNbinsY(),b=hratio.GetNbinsX())
        nBinsUnrollVar = hratio.GetNbinsY()
        unrollAxis = hratio.GetYaxis()
        if args.unrolly:
            unrollNameID = "unrolledY"
            xAxisTitle_unroll = f"Unrolled bin: '{xAxisTitle}' vs '{yAxisTitle}'"
            vertLineDivisions="{a},{b}".format(a=hratio.GetNbinsX(),b=hratio.GetNbinsY())
            nBinsUnrollVar = hratio.GetNbinsX()
            unrollAxis = hratio.GetXaxis()

        unrollBinRanges = []
        if nBinsUnrollVar > 15:
            for ibin in range(nBinsUnrollVar):
                unrollBinRanges.append("") # keep dummy otherwise there's too much text most of the time
        else:
            for ibin in range(nBinsUnrollVar):
                unrollBinRanges.append("#splitline{{{v} in}}{{[{vmin},{vmax}]}}".format(v="x" if args.unrolly else "y",
                                                                                        vmin=int(unrollAxis.GetBinLowEdge(ibin+1)),
                                                                                        vmax=int(unrollAxis.GetBinLowEdge(ibin+2))))
         
        ratio_unrolled = unroll2Dto1D(hratio, newname=f"{unrollNameID}_{hratio.GetName()}", cropNegativeBins=False, invertUnroll=args.unrolly)
        unitLine = copy.deepcopy(ratio_unrolled.Clone("tmp_horizontalLineAt1"))
        unitLine.Reset("ICESM")
        ratioUnc = copy.deepcopy(ratio_unrolled.Clone("tmp_ratioUnc"))
        ratioUnc.Reset("ICESM")
        for ib in range(2+unitLine.GetNbinsX()):
            unitLine.SetBinContent(ib, 1.0)
            unitLine.SetBinError(ib, 0.0)
            # for plotting purpose put the uncertainty of the unrolled on the unit line, and reset it for the unrolled
            ratioUnc.SetBinContent(ib, 1.0)
            ratioUnc.SetBinError(ib, 0.0 if (args.divideRelativeError or args.divideError) else ratio_unrolled.GetBinError(ib))
            ratio_unrolled.SetBinError(ib, 0.0)

        zAxisTitle_unroll = zAxisTitle if "::" not in zAxisTitle else str(zAxisTitle.split("::")[0])
        whatUncertainty = "numerator" if args.setRatioUnc == "num" else "denominator" if args.setRatioUnc == "den" else "total"
        if args.divideError or args.divideRelativeError:
            drawNTH1([ratio_unrolled, unitLine], [f"Ratio {args.histTitle}", "Unity"],
                     xAxisTitle_unroll, zAxisTitle_unroll,
                     ratio_unrolled.GetName(), outname,
                     leftMargin=0.06, rightMargin=0.01,
                     legendCoords="0.06,0.99,0.91,0.99;3", lowerPanelHeight=0.0, skipLumi=True, passCanvas=canvas_unroll,
                     drawVertLines=vertLineDivisions,
                     textForLines=unrollBinRanges, transparentLegend=False,
                     onlyLineColor=True, useLineFirstHistogram=True, drawErrorAll=True, lineWidth=1,
                     colorVec=[ROOT.kRed])
        else:
            unrollHists = [ratio_unrolled, ratioUnc, unitLine]
            legUnrollHists = [f"Ratio {args.histTitle}", f"Uncertainty ({whatUncertainty})", "Unity"]
            colorUnrollHists = [ROOT.kBlack, ROOT.kRed] # only from second histogram onwards
            drawNTH1(unrollHists, legUnrollHists,
                     xAxisTitle_unroll, zAxisTitle_unroll,
                     ratio_unrolled.GetName(), outname,
                     leftMargin=0.06, rightMargin=0.01,
                     legendCoords="0.06,0.99,0.91,0.99;3", lowerPanelHeight=0.0, skipLumi=True, passCanvas=canvas_unroll,
                     drawVertLines=vertLineDivisions,
                     textForLines=unrollBinRanges, transparentLegend=False,
                     onlyLineColor=False, useLineFirstHistogram=True, drawErrorAll=True, lineWidth=1,
                     fillStyleSecondHistogram=1001, fillColorSecondHistogram=ROOT.kGray,
                     colorVec=colorUnrollHists)

    
    # making distribution of pulls
    if args.makePulls:
        hpull = ROOT.TH1D("pulls","Distribution of pulls",100,-5,5)
        hpull2D = copy.deepcopy(hinput1.Clone("hpull2D"))
        hpull2D.Reset("ICESM")
        plotTitleLatex = ""
        if xMin < xMax:
            plotTitleLatex += f"{round(xMin,1)} < x < {round(xMax,1)}   "
        if yMin < yMax:
            plotTitleLatex += f"{round(yMin,1)} < y < {round(yMax,1)}   "
        for ix in range(1,1+hinput1.GetNbinsX()):
            for iy in range(1,1+hinput1.GetNbinsY()):
                xval = hinput1.GetXaxis().GetBinCenter(ix)
                if xMin < xMax:
                    if xval < xMin or xval > xMax: continue
                yval = hinput1.GetYaxis().GetBinCenter(iy)
                if yMin < yMax:
                    if yval < yMin or yval > yMax: continue
                err = math.sqrt(pow(hinput1.GetBinError(ix,iy),2) + pow(hinput2.GetBinError(ix,iy),2))
                err *= args.pullErrorScaleFactor
                pull = hinput1.GetBinContent(ix,iy) - hinput2.GetBinContent(ix,iy)
                pull = pull/err
                hpull2D.SetBinContent(ix, iy, pull)
                hpull.Fill(pull)

        print(f"{args.outhistname}:")
        print(f"Histogram mean = {hpull.GetMean()}")
        print(f"Histogram RMS  = {hpull.GetStdDev()}")
        drawTH1(hpull, 
                "pulls",
                "number of events",
                f"pullDistribution_{args.outhistname}",
                outname,
                passCanvas=canvas,
                fitString="gaus;LEMSQ+;;-5;5",
                plotTitleLatex=plotTitleLatex
                )
        drawCorrelationPlot(hpull2D,xAxisTitle,yAxisTitle,"Pulls::-5,5",
                            f"pullDistribution2D_{args.outhistname}","ForceTitle",outname,0,0,
                            False,False,False,1,palette=args.palette,passCanvas=canvas2D,drawOption=args.drawOption)
 
    ###########################
    # Now save things
    ###########################
    if args.outfilename:
        if not args.outfilename.endswith(".root"):
            args.outfilename = args.outfilename + ".root"
        tf = ROOT.TFile.Open(outname+args.outfilename,'recreate')
        hratio.Write(args.outhistname)
        tf.Close()
        print("")
        print("Created file %s" % (outname+args.outfilename))
        print("")

                               
         
