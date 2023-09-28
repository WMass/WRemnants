#!/usr/bin/env python

import re, sys, os, os.path, subprocess, json, ROOT, copy, math
import numpy as np
from utilities import common, logging
from functools import partial

from array import array
import shutil
#sys.path.append(os.getcwd() + "/plotUtils/")
from scripts.analysisTools.plotUtils.CMS_lumi import *

logger = logging.child_logger(__name__)

#########################################################################
# trying to use same colors as mathplotlib in wremnants
colors_plots_ = {"Wmunu"      : ROOT.TColor.GetColor("#8B0000"), #ROOT.kRed+2,
                 "Zmumu"      : ROOT.TColor.GetColor("#87CEFA"), #lightskyblue, #ADD8E6 is lightblue #ROOT.kAzure+2,
                 "DYlowMass"  : ROOT.TColor.GetColor("#00BFFF"), #deepskyblue,
                 "Wtau"       : ROOT.TColor.GetColor("#FFA500"), #ROOT.kCyan+1, #backward compatibility
                 "Wtaunu"     : ROOT.TColor.GetColor("#FFA500"), # orange, use #FF8C00 for darkOrange #ROOT.kCyan+1,
                 "Ztautau"    : ROOT.TColor.GetColor("#00008B"), #ROOT.kSpring+9,
                 "Top"        : ROOT.TColor.GetColor("#008000"), #ROOT.kGreen+2,
                 "Diboson"    : ROOT.TColor.GetColor("#FFC0CB"), #ROOT.kViolet,
                 "Fake"       : ROOT.TColor.GetColor("#D3D3D3"), # dimgray is "#696969" #ROOT.kGray,
                 "QCD"        : ROOT.TColor.GetColor("#D3D3D3"), # light grey #ROOT.kGray,
                 "Other"      : ROOT.TColor.GetColor("#808080"), # grey #ROOT.kGray}
}

legEntries_plots_ = {"Wmunu"      : "W#rightarrow#mu#nu",
                     "Zmumu"      : "Z#rightarrow#mu#mu",
                     "DYlowMass"  : "Z#rightarrow#mu#mu 10<m<50 GeV",
                     "Wtau"       : "W#rightarrow#tau#nu", #backward compatibility
                     "Wtaunu"     : "W#rightarrow#tau#nu",
                     "Ztautau"    : "Z#rightarrow#tau#tau",
                     "Top"        : "t quark",
                     "Diboson"    : "Diboson",
                     "Fake"       : "Nonprompt", # or "Multijet"
                     "QCD"        : "QCD MC",
                     "Other"      : "Other"}   

#########################################################################

def printLine(marker='-', repeat=30):
    print(marker*repeat)

#########################################################################

def safeSystem(cmd, dryRun=False, quitOnFail=True):
    logger.info(cmd)
    if not dryRun:
        res = os.system(cmd)
        if res:
            logger.error('-'*30)
            logger.error("safeSystem(): error occurred when executing the following command. Aborting")
            logger.error(cmd)
            logger.error('-'*30)
            if quitOnFail:
                quit()
        return res
    else:
        return 0

def checkHistInFile(h, hname, fname, message=""):
    if not h:
        logger.error("Error {msg}: I couldn't find histogram {h} in file {f}".format(msg=message,h=hname,f=fname))
        quit()

def safeGetObject(fileObject, objectName, quitOnFail=True, silent=False, detach=True):
    obj = fileObject.Get(objectName)
    if obj is None:
        if not silent:
            logger.error(f"Could not get {objectName} from file {fileObject.GetName()}")
        if quitOnFail:
            quit()
        return None
    else:
        if detach:
            obj.SetDirectory(0)
        return obj
        
def safeOpenFile(fileName, quitOnFail=True, silent=False, mode="READ"):
    fileObject = ROOT.TFile.Open(fileName, mode)
    if not fileObject or fileObject.IsZombie():
        if not silent:
            logger.error(f"Could not open file {fileName}")
        if quitOnFail:
            quit()
        else:
            return None
    elif not fileObject.IsOpen():
        if not silent:
            logger.error(f"File {fileName} was not opened")
        if quitOnFail:
            quit()
        else:
            return None
    else:
        return fileObject
            
def checkNullObj(obj, objName="object", quitOnFail=True):

    if obj == None:
        logger.error(f"{objName} was None.")
        if quitOnFail:
            quit()
        else:
            return True
    else:
        return False
            
#########################################################################

def addStringToEnd(name, matchToAdd, notAddIfEndswithMatch=False):
    if notAddIfEndswithMatch and name.endswith(matchToAdd):
        return name
    elif not name.endswith(matchToAdd):
        return name + matchToAdd

#########################################################################

def getZaxisReasonableExtremesTH2(h, nSigma=3, minZtoUse=None, maxZtoUse=None):

    htmp = ROOT.TH1D("htmp","",1000,h.GetMinimum(),h.GetMaximum())
    nbins = h.GetNbinsX() * h.GetNbinsY()    
    for ibin in range (1,nbins+1):
        val = h.GetBinContent(ibin)
        canFill = True
        if minZtoUse != None:
            if val < minZtoUse: canFill = False
        if maxZtoUse != None:
            if val > maxZtoUse: canFill = False
        if canFill: htmp.Fill(val)

    mean = htmp.GetMean()
    stddev = htmp.GetStdDev()
    retmin = max(h.GetMinimum(),mean - nSigma*stddev)
    retmax = min(h.GetMaximum(),mean + nSigma*stddev)
    return retmin,retmax


#########################################################################

def getMinMaxHisto(h, excludeEmpty=True, sumError=True, 
                   excludeUnderflow=True, excludeOverflow=True,
                   excludeMin=None, excludeMax=None,
                   ybinLow=None, ybinHigh=None): # only for TH2
    
    # Warning, fix this function, GetBinContent with TH2 is not that simple, there are the underflow and overflow in each row and column
    # must check whether bin is underflow or overflow
    # therefore, the global bin is obtained as the number of bins +2, multiplied for each axis

    # excludeEmpty = True exclude bins with content 0.0. Useful when a histogram is filled with values in, for example, [1,2] but hassome empty bins
    # excludeMin/Max are used to select a range in which to look for maximum and minimum, useful to reject outliers, crazy or empty bins and so on
    # for histograms with non-negative values, excludeEmpty=True is equal to excludeMin==0.0

    # sumError is used to add or subtract error when looking for min/max (to have full error band in range)
    # when using excludeMin/Max, the errors are still ignored when evaluating the range

    # the better combination of options depends on dimension: for a TH1 is useful to visualize the error band in the plot range, while for a TH2 
    # only the bin content is interesting in the plot (the error is not reported with TH2::Draw, unless plotting it in a 3D space

    # one might exploit excludeMin/Max to select a rage depending on the distribution on the histogram bin content
    # for example, one can pass excludeMin=h.GetMean()-2*h.GetStdDev() and excludeMax=h.GetMean()+2*h.GetStdDev() so to 
    # select a range of 2 sigma around the mean

    dim = h.GetDimension()
    nbins = 0
    if   dim == 1: nbins = h.GetNbinsX() + 2
    elif dim == 2: nbins = (h.GetNbinsX() + 2) * (h.GetNbinsY() + 2)
    elif dim == 3: nbins = (h.GetNbinsX() + 2) * (h.GetNbinsY() + 2) * (h.GetNbinsZ() + 2)
    else:
        logger.error("In getMaxHisto(): dim = %d is not supported. Exit" % dim)
        quit()

    nBinMin = 0
    nBinMax = (nbins+1)
    if dim == 2:
        nXbins = h.GetNbinsX() + 2
        nBinMin = 0 if ybinLow == None else (nXbins * ybinLow) 
        nBinMax = (nbins + 1) if ybinHigh == None else (1 + nXbins * (ybinHigh+1)) 
        
    maxval = -sys.float_info.max
    minval = sys.float_info.max
    firstValidBin = -1
    for ibin in range (1,nbins+1):
        if ibin <= nBinMin or ibin >= nBinMax:
            continue
        if excludeUnderflow and h.IsBinUnderflow(ibin): continue
        if excludeOverflow and h.IsBinOverflow(ibin): continue
        tmpmax = h.GetBinContent(ibin)
        tmpmin = h.GetBinContent(ibin)
        if excludeEmpty and tmpmin == 0.0: continue
        if excludeMin != None and tmpmin <= excludeMin: continue
        if excludeMax != None and tmpmax >= excludeMax: continue
        if firstValidBin < 0: 
            #logger.debug("ibin %d:   tmpmin,tmpmax = %.2f, %.2f" % (ibin,tmpmin,tmpmax))
            firstValidBin = ibin
        if sumError:
            tmpmin -= h.GetBinError(ibin)
            tmpmax += h.GetBinError(ibin)
        if firstValidBin > 0 and ibin == firstValidBin:
            #the first time we pick a non empty bin, we set min and max to the histogram content in that bin
            minval = tmpmin
            maxval = tmpmax
            #logger.debug("#### ibin %d:   min,max = %.2f, %.2f" % (ibin,minval,maxval))
        else:
            minval = min(minval,tmpmin)
            maxval = max(maxval,tmpmax)
        #logger.debug("ibin %d:   min,max = %.2f, %.2f" % (ibin,minval,maxval))
    
    return minval,maxval

#########################################################################

def getMinMaxMultiHisto(hlist, excludeEmpty=True, sumError=True, 
                        excludeUnderflow=True, excludeOverflow=True,
                        excludeMin=None, excludeMax=None):

    minlist = sys.float_info.max
    maxlist = sys.float_info.min
    for h in hlist:
        if h.InheritsFrom("TH1"):
            minv, maxv = getMinMaxHisto(h, excludeEmpty, sumError, excludeUnderflow, excludeOverflow, excludeMin, excludeMax)
            minlist = min(minv, minlist)
            maxlist = max(maxv, maxlist)
        elif h.InheritsFrom("TGraph"):
            yvals = h.GetY()
            yerrhigh = h.GetEYhigh()
            yerrlow = h.GetEYlow()
            for i in range(len(yvals)):
                maxlist = max(yvals[i] + yerrhigh[i], maxlist)
                minlist = min(yvals[i] - yerrlow[i],  minlist)
    return minlist, maxlist
        
#########################################################################

def getMinimumTH(h, excludeMin=None):
    # get minimum excluding some values. For example, if an histogram has an empty bin, one might want to get the minimum such that it is > 0
    # underflow are not considered
    
    dim = h.GetDimension()
    retmin = sys.float_info.max

    if dim == 1:
        for ix in range(1,h.GetNbinsX()+1):
            if retmin > h.GetBinContent(ix):
                if excludeMin != None:
                    if h.GetBinContent(ix) > excludeMin: retmin = h.GetBinContent(ix)
                else:
                    retmin = h.GetBinContent(ix)

    elif dim == 2:
        for ix in range(1,h.GetNbinsX()+1):
            for iy in range(1,h.GetNbinsY()+1):
                if retmin > h.GetBinContent(ix,iy):
                    if excludeMin != None:
                        if h.GetBinContent(ix,iy) > excludeMin: retmin = h.GetBinContent(ix,iy)
                    else:
                        retmin = h.GetBinContent(ix,iy)

    elif dim == 3:
        for ix in range(1,h.GetNbinsX()+1):
            for iy in range(1,h.GetNbinsY()+1):
                for iz in range(1,h.GetNbinsZ()+1):
                    if retmin > h.GetBinContent(ix,iy,iz):
                        if excludeMin != None:
                            if h.GetBinContent(ix,iy,iz) > excludeMin: retmin = h.GetBinContent(ix,iy,iz)
                        else:
                            retmin = h.GetBinContent(ix,iy,iz)
                            

    else:
        raise RuntimeError("Error in getMinimumTH(): unsupported histogram's dimension (%d)" % dim)

    return retmin

#########################################################################

def getMaximumTH(h, excludeMax=None):
    # get maximum excluding some values. For example, if an histogram has a crazy bin, one might want to get the maximum value that is lower than that
    # overflow are not considered
    
    dim = h.GetDimension()
    retmax = sys.float_info.min

    if dim == 1:
        for ix in range(1,h.GetNbinsX()+1):
            if retmax < h.GetBinContent(ix):
                if excludeMax != None:
                    if h.GetBinContent(ix) < excludeMax: retmax = h.GetBinContent(ix)
                else:
                    retmax = h.GetBinContent(ix)

    elif dim == 2:
        for ix in range(1,h.GetNbinsX()+1):
            for iy in range(1,h.GetNbinsY()+1):
                if retmax < h.GetBinContent(ix,iy):
                    if excludeMax != None:
                        if h.GetBinContent(ix,iy) < excludeMax: retmax = h.GetBinContent(ix,iy)                        
                    else:
                        retmax = h.GetBinContent(ix,iy)

    elif dim == 3:
        for ix in range(1,h.GetNbinsX()+1):
            for iy in range(1,h.GetNbinsY()+1):
                for iz in range(1,h.GetNbinsZ()+1):
                    if retmax < h.GetBinContent(ix,iy,iz):
                        if excludeMax != None:
                            if h.GetBinContent(ix,iy,iz) < excludeMax: retmax = h.GetBinContent(ix,iy,iz)    
                        else:
                            retmax = h.GetBinContent(ix,iy,iz)

    else:
        raise RuntimeError("Error in getMaximumTH(): unsupported histogram's dimension (%d)" % dim)

    return retmax

#########################################################################    

def fillTH2fromTH2part(h2out, h2in,
                       xbinLow=1, ybinLow=1,
                       xbinHigh=None, ybinHigh=None,
                       xoffset=0, yoffset=0,
                       fillWithValuePlusError=False, scaleError=1.0,  # fill with binContent + scaleError*binError (negative scaleError to subtract)
                       fillWithError=False, useRelativeError=False, ratioValForZeroAtDen=0.0):

    if xbinHigh == None:
        xbinHigh = h2out.GetNbinsX()
    if ybinHigh == None:
        ybinHigh = h2out.GetNbinsY()

    for ix in range(xbinLow, 1 + xbinHigh):
        for iy in range(ybinLow, 1 + ybinHigh):
            if fillWithValuePlusError:
                content = h2in.GetBinContent(ix + xoffset, iy + yoffset) + scaleError * h2in.GetBinError(ix + xoffset, iy + yoffset)
                error   = abs(scaleError) * h2in.GetBinError(  ix + xoffset, iy + yoffset)
            elif fillWithError:
                content = scaleError * h2in.GetBinError(ix + xoffset, iy + yoffset)
                error   = 0.0
                if useRelativeError:
                    if h2in.GetBinContent(ix + xoffset, iy + yoffset)  != 0.0:
                        content = content / h2in.GetBinContent(ix + xoffset, iy + yoffset)
                    else:
                        content = ratioValForZeroAtDen
            else:
                content = h2in.GetBinContent(ix + xoffset, iy + yoffset)
                error   = abs(scaleError) * h2in.GetBinError(  ix + xoffset, iy + yoffset)

            h2out.SetBinContent(ix, iy, content)
            h2out.SetBinError(  ix, iy, error)

#########################################################################

def fillTH3fromTH3part(h3out, h3in,
                       xbinLow=1, ybinLow=1, zbinLow=1,
                       xbinHigh=None, ybinHigh=None, zbinHigh=None,
                       xoffset=0, yoffset=0, zoffset=0):

    if xbinHigh == None:
        xbinHigh = h3out.GetNbinsX()
    if ybinHigh == None:
        ybinHigh = h3out.GetNbinsY()
    if zbinHigh == None:
        zbinHigh = h3out.GetNbinsZ()

    for ix in range(xbinLow, 1 + xbinHigh):
        for iy in range(ybinLow, 1 + ybinHigh):
            for iz in range(zbinLow, 1 + zbinHigh):
                content = h3in.GetBinContent(ix + xoffset, iy + yoffset, iz + zoffset)
                error   = h3in.GetBinError(  ix + xoffset, iy + yoffset, iz + zoffset)
                h3out.SetBinContent(ix, iy, iz, content)
                h3out.SetBinError(  ix, iy, iz, error)


#########################################################################

# can't this use TH3.Projection?
def fillTH2fromTH3zrange(h2, h3, zbinLow=1, zbinHigh=1):
    for ix in range(1, 1 + h2.GetNbinsX()):
        for iy in range(1, 1 + h2.GetNbinsY()):
            error = ROOT.Double(0)
            h2.SetBinContent(ix, iy, h3.IntegralAndError(ix, ix, iy, iy, zbinLow, zbinHigh, error))
            h2.SetBinError(ix,iy,error);

#########################################################################

def fillTH2fromTH3zbin(h2, h3, zbin=1):
    #fillTH2fromTH3zrange(h2, h3, zbinLow=zbin, zbinHigh=zbin)

    for ix in range(1, 1 + h2.GetNbinsX()):
        for iy in range(1, 1 + h2.GetNbinsY()):
            h2.SetBinContent(ix, iy, h3.GetBinContent(ix, iy, zbin))
            h2.SetBinError(ix, iy, h3.GetBinError(ix, iy, zbin));

#########################################################################

def prepareLegend(x1, y1, x2, y2, textSize=0.035, nColumns=3):
    leg = ROOT.TLegend(x1,y1,x2,y2)
    leg.SetNColumns(nColumns)
    leg.SetFillColor(0)
    leg.SetFillColorAlpha(0,0.6)
    leg.SetShadowColor(0)
    leg.SetLineColor(0)
    leg.SetBorderSize(0)
    leg.SetTextFont(42)
    leg.SetTextSize(textSize)
    return leg

#########################################################################

def getTH2fromTH3(hist3D, name, binStart, binEnd=None):
    if binEnd == None:
        binEnd = binStart
    hist3D.GetZaxis().SetRange(binStart,binEnd)
    #print(f"getTH2fromTH3(): {name}   projecting bins from {binStart} to {binEnd}")
    # Order yx matters to have consistent axes!
    hist2D = hist3D.Project3D("yxe") # yxe is to make TH2 with y axis versus x axis 
    hist2D.SetName(name)
    return hist2D

    
#########################################################################

def fillTH3binFromTH2(h3, h2, zbin, scaleFactor=None):
    for ix in range(1, 1 + h2.GetNbinsX()):
        for iy in range(1, 1 + h2.GetNbinsY()):
            val   = h2.GetBinContent(ix, iy)
            error = h2.GetBinError(ix, iy)
            if scaleFactor != None:
                val   *= scaleFactor
                error *= scaleFactor
            h3.SetBinContent(ix, iy, zbin, val)
            h3.SetBinError(ix, iy, zbin, error);

def fillTHNplus1fromTHn(thnp1, thn, nbinLow=-1, nbinHigh=-1):
    # we assume the only difference is an additional dimension, which is the( N+1)-th (i.e. not in the middle, and all other axies are exactly the same in terms of range and binning)
    # nbinLow/High here represent the range of bin index for the n+1 dimension to be filled (as for the TH1 convention, 0 is underflow, 1 is first bin, etc...)
    # if nbinLow = nbinHigh and both are larger than -1, only that bin is filled
    # if both are negative the full range including under/overflow is filled
    # if only one of them is negative all the bins up to nbinHigh or from nbinLow are filled
    ndim = thn.GetNdimensions()
    #ndimp1 = ndim + 1
    nbinsThnp1 = thnp1.GetAxis(ndim).GetNbins() # would have used ndimp1 - 1
    if (nbinLow < 0 and nbinHigh < 0) or nbinLow > nbinHigh:
        nBinStart = 0
        nBinEnd = nbinsThnp1 + 1
    elif nbinLow < 0:
        nBinStart = 0
        nBinEnd = nbinHigh
    elif nbinHigh < 0:
        nBinStart = nbinLow
        nBinEnd   = nbinsThnp1+1
    else:
        nBinStart = max(0, nbinLow)
        nBinEnd   = min(nbinsThnp1 + 1, nbinHigh)

    nbinsTHn = thn.GetNbins() # for TH3 or lower one should use GetNcells() to emulate this, but one can get a proper THn from a TH3 to use consistent methods
    myArray = array("i", [0 for i in range(ndim)])
    for globalBin in range(nbinsTHn):
        # get content and also the array with single bin ids corresponding to iglobalBin
        binContent = thn.GetBinContent(globalBin, myArray)
        binError = thn.GetBinError(globalBin)
        for iNewDim in range(nBinStart, nBinEnd+1):
            newArray = array("i", [x for x in myArray] + [iNewDim])
            #array = numpy.append(array, value)            
            globalBinTHnP1 = thnp1.GetBin(newArray)
            thnp1.SetBinContent(globalBinTHnP1, binContent)
            thnp1.SetBinError(globalBinTHnP1, binError)

            
def multiplyByHistoWith1ptBin(h, h1bin):
    # multiply 2D histograms when one has only 1 pt bin
    # neglect uncertainty on histogram with 1 bin
    # it is assumed that the number of eta bins is the same
    for ix in range(1, 1 + h.GetNbinsX()):
        for iy in range(1, 1 + h.GetNbinsY()):
            h.SetBinContent(ix, iy, h.GetBinContent(ix, iy) * h1bin.GetBinContent(ix, 1))
            h.SetBinError(  ix, iy, h.GetBinError(  ix, iy) * h1bin.GetBinContent(ix, 1))


def multiplyByHistoWithLessPtBins(h, hless, neglectUncSecond=False):
    # multiply 2D histograms when one has less pt bins
    # neglect uncertainty on histogram with less bins
    # it is assumed that the number of eta bins is the same
    for ix in range(1, 1 + h.GetNbinsX()):
        for iy in range(1, 1 + h.GetNbinsY()):
            ybin = hless.GetYaxis().FindFixBin(h.GetYaxis().GetBinCenter(iy))
            hContent = h.GetBinContent(ix, iy)
            hlessContent = hless.GetBinContent(ix, ybin)
            hUnc = h.GetBinError(ix, iy)
            hlessUnc = hless.GetBinError(ix, ybin)
            if neglectUncSecond:
                unc = hUnc * hlessContent
            else:
                # uncertainty on product assuming uncorrelated pieces
                unc = math.sqrt(hContent*hContent*hlessUnc*hlessUnc + hlessContent*hlessContent*hUnc*hUnc) 
            h.SetBinContent(ix, iy, hContent * hlessContent)
            h.SetBinError(  ix, iy, unc)

# TODO: make this C++ function in wremnants/include/histHelpers.h
def scaleTH2byOtherTH2(h, hother, scaleUncertainty=True, divide=False):
    # multiply 2D histograms when one has less bins
    # it is used to apply a correction stored in a TH2
    # one can decide not to scale also the uncertainty (scaleUncertainty=False)
    maxXbinOther = hother.GetNbinsX()
    maxYbinOther = hother.GetNbinsY()
    for ix in range(1, 1 + h.GetNbinsX()):
        xbin = hother.GetXaxis().FindFixBin(h.GetXaxis().GetBinCenter(ix))
        xbin = sorted((1, xbin, maxXbinOther))[1] # creative way to clamp, faster than max(1, min(val, maxVal)) syntax
        for iy in range(1, 1 + h.GetNbinsY()):
            ybin = hother.GetYaxis().FindFixBin(h.GetYaxis().GetBinCenter(iy))
            ybin = sorted((1, ybin, maxYbinOther))[1]
            hContent = h.GetBinContent(ix, iy)
            hUnc = h.GetBinError(ix, iy)
            hotherContent = hother.GetBinContent(xbin, ybin)
            #hotherContent = 2.0 ## for debug
            #print(f"ix-iy-xbin-ybin-corr = {ix} - {iy} - {xbin} - {ybin} - {hotherContent}")
            if divide:
                hotherContent = 1.0 / hotherContent
            h.SetBinContent(ix, iy, hContent * hotherContent)
            if scaleUncertainty:
                h.SetBinError(  ix, iy, hUnc * hotherContent)

            
def getTH2morePtBins(h2, newname, nPt):
    xedges = [round(h2.GetXaxis().GetBinLowEdge(i), 2) for i in range(1, 2 + h2.GetNbinsX())]
    xarr = array('d', xedges)
    h2new = ROOT.TH2D(newname, "",
                      len(xedges) - 1, xarr,
                      nPt, h2.GetYaxis().GetBinLowEdge(1), h2.GetYaxis().GetBinLowEdge(1+h2.GetNbinsY()))
    for ix in range(1, 1 + h2new.GetNbinsX()):
        for iy in range(1, 1 + h2new.GetNbinsY()):
            ieta = h2.GetXaxis().FindFixBin(h2new.GetXaxis().GetBinCenter(ix))
            ipt  = h2.GetYaxis().FindFixBin(h2new.GetYaxis().GetBinCenter(iy))
            h2new.SetBinContent(ix, iy, h2.GetBinContent(ieta, ipt))
            h2new.SetBinError(  ix, iy, h2.GetBinError(  ieta, ipt))
    return h2new
            
#########################################################################

def createPlotDirAndCopyPhp(outdir):
    if outdir and not os.path.exists(outdir):
        os.makedirs(outdir)
    htmlpath = f"{os.environ['WREM_BASE']}/scripts/analysisTools/templates/index.php"
    shutil.copy(htmlpath, outdir)


#########################################################################

def getAxisRangeFromUser(axisNameTmp="", 
                         separator="::", 
                         rangeSeparator=","
                         ):
  
    setXAxisRangeFromUser = False;
    fields = axisNameTmp.split(separator)
    axisName = fields[0]
    
    if len(fields) > 1:
        setXAxisRangeFromUser = True;
        xmin = float(fields[1].split(rangeSeparator)[0])
        xmax = float(fields[1].split(rangeSeparator)[1])
    else:
        xmin = 0
        xmax = 0
        
    return axisName,setXAxisRangeFromUser,xmin,xmax


#########################################################################

def adjustSettings_CMS_lumi():

    ## dummy function to be called before using any other fucntion calling CMS_lumi
    ## for some reason, the settings of the very first plot are screwed up.
    ## To fix this issue, it is enough to call it to a dummy plot
    dummy = ROOT.TH1D("dummy","",10,0,10)
    for i in range(1,1+dummy.GetNbinsX()):
        dummy.SetBinContent(i,i)
    dummy.GetXaxis().SetTitle("x axis")
    dummy.GetYaxis().SetTitle("y axis")
    cdummy = ROOT.TCanvas("cdummy","",600,600)
    dummy.Draw("HE")
    CMS_lumi(cdummy,"",True,False)
    setTDRStyle()        
    ## no need to save the canvas    


#########################################################################

def drawTH1(htmp,
            labelXtmp="xaxis",
            labelYtmp="Events",
            canvasName="default",
            outdir= "./",
            canvasSize="700,625",
            passCanvas=None,
            moreTextLatex="",
            skipTdrStyle=False,
            drawStatBox=True,
            statBoxSpec=None, # to specify what to print (None uses a default which depends on having or not fitString
            fitString="", # can be "gaus;LEMSQ+;;-5;5"
            plotTitleLatex="",
            setLogY=False
):



    addStringToEnd(outdir,"/",notAddIfEndswithMatch=True)
    createPlotDirAndCopyPhp(outdir)

    labelX,setXAxisRangeFromUser,xmin,xmax = getAxisRangeFromUser(labelXtmp)
    labelY,setYAxisRangeFromUser,ymin,ymax = getAxisRangeFromUser(labelYtmp)

    h = htmp.Clone("htmp")

    cw,ch = canvasSize.split(',')
    canvas = passCanvas if passCanvas != None else ROOT.TCanvas("canvas","",int(cw),int(ch))
    canvas.SetTickx(1)
    canvas.SetTicky(1)
    canvas.cd()
    canvas.SetBottomMargin(0.14)
    leftMargin = 0.12
    canvas.SetLeftMargin(leftMargin)
    canvas.SetRightMargin(0.04)
    canvas.cd()

    h.SetLineColor(ROOT.kBlack)
    h.SetLineWidth(2)
    h.GetXaxis().SetTitle(labelX)
    h.GetXaxis().SetTitleOffset(1.2)
    h.GetXaxis().SetTitleSize(0.05)
    h.GetXaxis().SetLabelSize(0.04)
    h.GetYaxis().SetTitle(labelY)
    h.GetYaxis().SetTitleOffset(1.15)
    h.GetYaxis().SetTitleSize(0.05)
    h.GetYaxis().SetLabelSize(0.04)
    if (setXAxisRangeFromUser): h.GetXaxis().SetRangeUser(xmin,xmax)
    if (setYAxisRangeFromUser): h.GetYaxis().SetRangeUser(ymin,ymax)
    # force drawing stat box
    h.SetStats(1 if drawStatBox else 0)
    if "TH1" in h.ClassName():
        h.Draw("HIST")
    else:
        h.SetLineColor(ROOT.kBlack)
        h.SetMarkerStyle(20)
        h.SetMarkerColor(ROOT.kBlack)
        h.Draw("HE")
    if len(fitString):
        fitFunc,fitOpt,drawOpt,fitMin,fitMax = fitString.split(";")
        logger.info(f"Fitting with {fitFunc}")
        h.Fit(fitFunc,fitOpt,drawOpt,float(fitMin),float(fitMax))
        f1 = h.GetFunction(fitFunc)
        f1.SetLineWidth(2)
        #f1.SetLineStyle(9) # ROOT.kDashed == thin dashes, almost dotted
        f1.SetLineColor(ROOT.kRed+2)
        f1.Draw("L SAME")
        
    canvas.RedrawAxis("sameaxis")
    if not skipTdrStyle: 
        setTDRStyle()
    # force drawing stat box
    if drawStatBox:
        if len(fitString):
            ROOT.gStyle.SetOptFit(111)
            ROOT.gStyle.SetOptStat(statBoxSpec if statBoxSpec else 110010)
        else:
            ROOT.gStyle.SetOptStat(statBoxSpec if statBoxSpec else 111110)

    if len(plotTitleLatex):
        lat = ROOT.TLatex()
        lat.SetNDC();
        lat.SetTextFont(42)        
        lat.SetTextSize(0.04)
        lat.DrawLatex(leftMargin, 0.95, plotTitleLatex)

            
    if len(moreTextLatex):
        realtext = moreTextLatex.split("::")[0]
        x1,y1,ypass,textsize = 0.75,0.8,0.08,0.035
        if "::" in moreTextLatex:
            x1,y1,ypass,textsize = (float(x) for x in (moreTextLatex.split("::")[1]).split(","))            
        lat = ROOT.TLatex()
        lat.SetNDC();
        lat.SetTextFont(42)        
        lat.SetTextSize(textsize)
        for itx,tx in enumerate(realtext.split(";")):
            lat.DrawLatex(x1,y1-itx*ypass,tx)

    if setLogY:
        canvas.SetLogy()
    for ext in ["png","pdf"]:
        canvas.SaveAs(f"{outdir}{canvasName}.{ext}")
    if setLogY:
        canvas.SetLogy(0)



#########################################################################




# function to draw 2D histograms, can also plot profile along X on top
def drawCorrelationPlot(h2D_tmp,
                        labelXtmp="xaxis", labelYtmp="yaxis", labelZtmp="zaxis",
                        canvasName="default", plotLabel="", outdir="./",
                        rebinFactorX=0,
                        rebinFactorY=0,
                        smoothPlot=False,
                        drawProfileX=False,
                        scaleToUnitArea=False,
                        draw_both0_noLog1_onlyLog2=1,
                        leftMargin=0.16,
                        rightMargin=0.20,
                        nContours=51,
                        palette=55,
                        invertePalette=False,
                        canvasSize="700,625",
                        passCanvas=None,
                        bottomMargin=0.1,
                        plotError=False,
                        plotRelativeError=False,
                        lumi=None,
                        drawOption = "colz",
                        skipLumi=False,
                        zTitleOffSet=-1
                        ):


    ROOT.TH1.SetDefaultSumw2()
    adjustSettings_CMS_lumi()

    if (rebinFactorX): 
        if isinstance(rebinFactorX, int): h2D_tmp.RebinX(rebinFactorX)
        else:                             h2D_tmp.RebinX(len(rebinFactorX)-1,"",array('d',rebinFactorX)) # case in which rebinFactorX is a list of bin edges

    if (rebinFactorY): 
        if isinstance(rebinFactorY, int): h2D_tmp.RebinY(rebinFactorY)
        else:                             h2D_tmp.RebinY(len(rebinFactorY)-1,"",array('d',rebinFactorY)) # case in which rebinFactorX is a list of bin edges

    if plotError or plotRelativeError:
        herr = copy.deepcopy(h2D_tmp.Clone(h2D_tmp.GetName()+"_err"))
        herr.Reset("ICESM")
        for i in range(1,herr.GetNbinsX()+1):
            for j in range(1,herr.GetNbinsY()+1):
                errval = h2D_tmp.GetBinError(i,j)
                if plotRelativeError:
                    if h2D_tmp.GetBinContent(i,j) != 0.0:
                        errval = errval/h2D_tmp.GetBinContent(i,j)
                    else:
                        errval = 1.0 if errval == 0 else 0.0
                herr.SetBinContent(i,j,errval)
        h2D = herr
    else:
        h2D = h2D_tmp

    # dark blue to red
    ROOT.TColor.CreateGradientColorTable(4,
                                         array ("d", [0.00, 0.45, 0.55, 1.00]),
                                         array ("d", [0.00, 1.00, 1.00, 1.00]),
                                         array ("d", [0.00, 1.00, 1.00, 0.00]),
                                         array ("d", [1.00, 1.00, 1.00, 0.00]),
                                         255,  1.0)
                                         # array ("d", [0.00, 0.50, 1.00]),
                                         # array ("d", [0.00, 1.00, 1.00]),
                                         # array ("d", [0.00, 1.00, 0.00]),
                                         # array ("d", [1.00, 1.00, 0.00]),
                                         # 255,  0.95)

    if palette > 0:
        ROOT.gStyle.SetPalette(palette)  # 55:raibow palette ; 57: kBird (blue to yellow, default) ; 107 kVisibleSpectrum ; 77 kDarkRainBow 
    ROOT.gStyle.SetNumberContours(nContours) # default is 20 
    if invertePalette:
        ROOT.TColor.InvertPalette()

    labelX,setXAxisRangeFromUser,xmin,xmax = getAxisRangeFromUser(labelXtmp)
    labelY,setYAxisRangeFromUser,ymin,ymax = getAxisRangeFromUser(labelYtmp)
    labelZ,setZAxisRangeFromUser,zmin,zmax = getAxisRangeFromUser(labelZtmp)
        
    cw,ch = canvasSize.split(',')
    #canvas = ROOT.TCanvas("canvas",h2D.GetTitle() if plotLabel == "ForceTitle" else "",700,625)    
    canvas = passCanvas if passCanvas != None else ROOT.TCanvas("canvas","",int(cw),int(ch))
    canvas.SetTickx(1)
    canvas.SetTicky(1)
    canvas.SetLeftMargin(leftMargin)
    canvas.SetRightMargin(rightMargin)
    canvas.SetBottomMargin(bottomMargin)
    canvas.cd()

    addStringToEnd(outdir,"/",notAddIfEndswithMatch=True)
    createPlotDirAndCopyPhp(outdir)
    # normalize to 1
    if (scaleToUnitArea): h2D.Scale(1./h2D.Integral())

    h2DGraph = 0

    h2DPlot = 0
    if (not smoothPlot): h2DPlot = h2D
    else:
        h2DGraph = ROOT.TGraph2D()
        h2DGraph.SetNpx(300)
        h2DGraph.SetNpy(300)
        nPoint = 0
        for iBinX in range (1,1+h2D.GetNbinsX()):
            for iBinY in range(1,1+h2D.GetNbinsY()):
                h2DGraph.SetPoint(nPoint,h2D.GetXaxis().GetBinCenter(iBinX),h2D.GetYaxis().GetBinCenter(iBinY),h2D.GetBinContent(iBinX,iBinY))
                nPoint += 1
            

        h2DPlot = h2DGraph.GetHistogram()

    if plotLabel == "ForceTitle":
        h2DPlot.SetTitle(h2D_tmp.GetTitle())
  
    h2DPlot.GetXaxis().SetTitle(labelX)
    h2DPlot.GetYaxis().SetTitle(labelY)
    h2DPlot.GetXaxis().SetTitleSize(0.05)
    h2DPlot.GetXaxis().SetLabelSize(0.04)
    h2DPlot.GetXaxis().SetTitleOffset(0.95) # 1.1 goes outside sometimes, maybe depends on root version or canvas width
    h2DPlot.GetYaxis().SetTitleSize(0.05)
    h2DPlot.GetYaxis().SetLabelSize(0.04)
    h2DPlot.GetYaxis().SetTitleOffset(1.1)
    h2DPlot.GetZaxis().SetTitleSize(0.05)
    h2DPlot.GetZaxis().SetLabelSize(0.04)
    h2DPlot.GetZaxis().SetTitleOffset(1.2)

    h2DPlot.GetZaxis().SetTitle(labelZ) 
    h2DPlot.Draw(drawOption)
    
    if (setXAxisRangeFromUser): h2DPlot.GetXaxis().SetRangeUser(xmin,xmax)
    if (setYAxisRangeFromUser): h2DPlot.GetYaxis().SetRangeUser(ymin,ymax)
    if (setZAxisRangeFromUser): h2DPlot.GetZaxis().SetRangeUser(zmin,zmax)

    if (zTitleOffSet > 0):
        h2DPlot.GetZaxis().SetTitleOffset(zTitleOffSet)
    else:
        h2DPlot.GetZaxis().SetTitleOffset(h2DPlot.GetZaxis().GetTitleOffset()+0.4)

    h2DProfile = 0
    if drawProfileX:
        h2DProfile = h2D.ProfileX("%s_pfx" %h2D.GetName())
        h2DProfile.SetMarkerColor(ROOT.kBlack)
        h2DProfile.SetMarkerStyle(20)
        h2DProfile.SetMarkerSize(1)
        h2DProfile.Draw("EPsame")
        
    # not yet implemented
    setTDRStyle()
    if not skipLumi and not plotLabel == "ForceTitle": 
        if lumi != None: CMS_lumi(canvas,lumi,True,False)
        else:            CMS_lumi(canvas,"",True,False)

    if plotLabel == "ForceTitle":
        ROOT.gStyle.SetOptTitle(1)        

    #h2DPlot.GetZaxis().SetMaxDigits(1)  #for N>99, should use scientific notation, I'd like to make it work only with negative exponential but haven't succeeded yet
    # canvas.Modified()
    # canvas.Update()

    leg = ROOT.TLegend(0.25,0.83,0.75,0.93)
    leg.SetBorderSize(0)
    leg.SetTextFont(62)
    nLegEntries = 0
    if plotLabel not in ["", "ForceTitle"]:
        leg.AddEntry(0,plotLabel,"")
        nLegEntries += 1
    if drawProfileX:
        leg.AddEntry(h2DProfile, "Correlation = %.2f" % h2DPlot.GetCorrelationFactor(),"")
        nLegEntries += 1
    if nLegEntries == 0:
        leg.SetFillStyle(0)
        leg.SetFillColor(0)
    leg.Draw("same")

    if (draw_both0_noLog1_onlyLog2 == 0 or draw_both0_noLog1_onlyLog2 == 1):
        for ext in ['png', 'pdf']:
            canvas.SaveAs('{od}/{cn}.{ext}'.format(od=outdir, cn=canvasName, ext=ext))
        
    if (draw_both0_noLog1_onlyLog2 == 0 or draw_both0_noLog1_onlyLog2 == 2):
        canvas.SetLogz()
        for ext in ['png', 'pdf']:
            canvas.SaveAs('{od}/{cn}_logZ.{ext}'.format(od=outdir, cn=canvasName, ext=ext))
        canvas.SetLogz(0)


##########################################################


def drawSingleTH1(h1,
                  labelXtmp="xaxis", labelYtmp="yaxis",
                  canvasName="default", outdir="./",
                  rebinFactorX=0,
                  draw_both0_noLog1_onlyLog2=1,
                  topMargin=0.1,
                  leftMargin=0.15,
                  rightMargin=0.04,
                  bottomMargin=0.15,
                  labelRatioTmp="Rel.Unc.::0.5,1.5",
                  drawStatBox=False,
                  legendCoords="0.15,0.35,0.8,0.9",  # x1,x2,y1,y2
                  canvasSize="600,700",  # use X,Y to pass X and Y size     
                  lowerPanelHeight = 0.3,  # number from 0 to 1, 0.3 means 30% of space taken by lower panel. 0 means do not draw lower panel with relative error
                  drawLineTopPanel=None,
                  drawLineLowerPanel="luminosity uncertainty::0.025", # if not empty, draw band at 1+ number after ::, and add legend with title
                  passCanvas=None,
                  lumi=None,
                  skipLumi=False,
                  drawVertLines="", # "12,36": format --> N of sections (e.g: 12 pt bins), and N of bins in each section (e.g. 36 eta bins), assuming uniform bin width
                  textForLines=[],                       
                  moreText="",
                  moreTextLatex="",
                  ytextOffsetFromTop=0.15,
                  textSize=0.04,
                  textAngle=0
                  ):

    # moreText is used to pass some text to write somewhere (TPaveText is used)
    # e.g.  "stuff::x1,y1,x2,y2"  where xi and yi are the coordinates for the text
    # one can add more lines using the ";" key. FOr example, "stuff1;stuff2::x1,y1,x2,y2"
    # the coordinates should be defined taking into account how many lines will be drawn
    # if the coordinates are not passed (no "::"), then default ones are used, but this might not be satisfactory

    # moreTextLatex is similar, but used TLatex, and the four coordinates are x1,y1,ypass,textsize
    # where x1 and y1 are the coordinates the first line, and ypass is how much below y1 the second line is (and so on for following lines)

    if (rebinFactorX): 
        if isinstance(rebinFactorX, int): h1.Rebin(rebinFactorX)
        # case in which rebinFactorX is a list of bin edges
        else:                             h1.Rebin(len(rebinFactorX)-1,"",array('d',rebinFactorX)) 

    xAxisName,setXAxisRangeFromUser,xmin,xmax = getAxisRangeFromUser(labelXtmp)
    yAxisName,setYAxisRangeFromUser,ymin,ymax = getAxisRangeFromUser(labelYtmp)
    yRatioAxisName,setRatioYAxisRangeFromUser,yminRatio,ymaxRatio = getAxisRangeFromUser(labelRatioTmp)

    yAxisTitleOffset = 1.45 if leftMargin > 0.1 else 0.5

    addStringToEnd(outdir,"/",notAddIfEndswithMatch=True)
    createPlotDirAndCopyPhp(outdir)

    cw,ch = canvasSize.split(',')
    #canvas = ROOT.TCanvas("canvas",h2D.GetTitle() if plotLabel == "ForceTitle" else "",700,625)
    canvas = passCanvas if passCanvas != None else ROOT.TCanvas("canvas","",int(cw),int(ch))
    canvas.SetTickx(1)
    canvas.SetTicky(1)
    canvas.cd()
    canvas.SetLeftMargin(leftMargin)
    canvas.SetRightMargin(rightMargin)
    canvas.SetTopMargin(topMargin)
    canvas.cd()

    pad2 = 0
    if lowerPanelHeight: 
        canvas.SetBottomMargin(lowerPanelHeight)
        pad2 = ROOT.TPad("pad2","pad2",0,0.,1,0.9)
        pad2.SetTopMargin(1-lowerPanelHeight)
        pad2.SetRightMargin(rightMargin)
        pad2.SetLeftMargin(leftMargin)
        pad2.SetFillColor(0)
        pad2.SetGridy(1)
        pad2.SetFillStyle(0)
    else:
        canvas.SetBottomMargin(bottomMargin)


    frame = h1.Clone("frame")
    frame.GetXaxis().SetLabelSize(0.04)
    frame.SetStats(0)

    h1.SetLineColor(ROOT.kBlack)
    h1.SetMarkerColor(ROOT.kBlack)
    h1.SetMarkerStyle(20)
    h1.SetMarkerSize(0)

    #ymax = max(ymax, max(h1.GetBinContent(i)+h1.GetBinError(i) for i in range(1,h1.GetNbinsX()+1)))
    # if min and max were not set, set them based on histogram content
    if ymin == ymax == 0.0:
        ymin,ymax = getMinMaxHisto(h1,excludeEmpty=True,sumError=True)            
        yRangeOffset = 0.05 * (ymax - ymin)
        ymin -= yRangeOffset
        ymax += yRangeOffset

    if lowerPanelHeight:
        h1.GetXaxis().SetLabelSize(0)
        h1.GetXaxis().SetTitle("")  
    else:
        h1.GetXaxis().SetTitle(xAxisName)
        h1.GetXaxis().SetTitleOffset(1.2)
        h1.GetXaxis().SetTitleSize(0.05)
        h1.GetXaxis().SetLabelSize(0.04)
    h1.GetYaxis().SetTitle(yAxisName)
    h1.GetYaxis().SetTitleOffset(yAxisTitleOffset) 
    h1.GetYaxis().SetTitleSize(0.05)
    h1.GetYaxis().SetLabelSize(0.04)
    h1.GetYaxis().SetRangeUser(ymin, ymax)
    h1.GetYaxis().SetTickSize(0.01)
    if setXAxisRangeFromUser: h1.GetXaxis().SetRangeUser(xmin,xmax)
    h1.Draw("HIST")
    h1err = h1.Clone("h1err")
    h1err.SetFillColor(ROOT.kGray)
    h1err.SetFillStyle(1001)  # 3001 is better than 3002 for pdf, while 3002 is perfect for png
    #h1err.SetFillStyle(3002)
    #h1err.SetFillStyle(3005)
    h1err.Draw("E2same")
    h1.Draw("HIST same")

    if legendCoords != None and len(legendCoords) > 0:
        nColumnsLeg = 1
        if ";" in legendCoords: 
            nColumnsLeg = int(legendCoords.split(";")[1])
        legcoords = [float(x) for x in (legendCoords.split(";")[0]).split(',')]
        lx1,lx2,ly1,ly2 = legcoords[0],legcoords[1],legcoords[2],legcoords[3]
        leg = ROOT.TLegend(lx1,ly1,lx2,ly2)
        #leg.SetFillColor(0)
        #leg.SetFillStyle(0)
        #leg.SetBorderSize(0)
        leg.SetNColumns(nColumnsLeg)
        leg.AddEntry(h1,"Value","L")
        leg.AddEntry(h1err,"Uncertainty","F")
        leg.Draw("same")
    canvas.RedrawAxis("sameaxis")

    if drawStatBox:
        ROOT.gPad.Update()
        ROOT.gStyle.SetOptStat(1110)
        ROOT.gStyle.SetOptFit(1102)
    else:
        h1.SetStats(0)

    vertline = ROOT.TLine(36,0,36,canvas.GetUymax())
    vertline.SetLineColor(ROOT.kBlack)
    vertline.SetLineStyle(3)
    bintext = ROOT.TLatex()
    #bintext.SetNDC()
    bintext.SetTextFont(42)
    bintext.SetTextSize(textSize)
    bintext.SetTextAngle(textAngle)        
    #if len(textForLines): bintext.SetTextAngle(45 if "#eta" in textForLines[0] else 30)

    if len(drawVertLines):
        nptBins = int(drawVertLines.split(',')[0])
        etarange = float(drawVertLines.split(',')[1])        
        offsetXaxisHist = h1.GetXaxis().GetBinLowEdge(0)
        sliceLabelOffset = 6. if "#eta" in textForLines[0] else 6.
        for i in range(1,nptBins): # do not need line at canvas borders
            #vertline.DrawLine(offsetXaxisHist+etarange*i,0,offsetXaxisHist+etarange*i,canvas.GetUymax())
            vertline.DrawLine(etarange*i-offsetXaxisHist,ymin,etarange*i-offsetXaxisHist,ymax)
        if len(textForLines):
            for i in range(0,len(textForLines)): # we need nptBins texts
                #texoffset = 0.1 * (4 - (i%4))
                #ytext = (1. + texoffset)*ymax/2.  
                ytext = ymax - ytextOffsetFromTop*(ymax - ymin)  
                bintext.DrawLatex(etarange*i + etarange/sliceLabelOffset, ytext, textForLines[i])

    # redraw legend, or vertical lines appear on top of it
    if legendCoords != None and len(legendCoords) > 0:
        leg.Draw("same")
    
    if len(moreText):
        realtext = moreText.split("::")[0]
        x1,y1,x2,y2 = 0.7,0.8,0.9,0.9
        if "::" in moreText:
            x1,y1,x2,y2 = (float(x) for x in (moreText.split("::")[1]).split(","))
        pavetext = ROOT.TPaveText(x1,y1,x2,y2,"NB NDC")
        for tx in realtext.split(";"):
            pavetext.AddText(tx)
        pavetext.SetFillColor(0)
        pavetext.SetFillStyle(0)
        pavetext.SetBorderSize(0)
        pavetext.SetLineColor(0)
        pavetext.Draw("same")

    if len(moreTextLatex):
        realtext = moreTextLatex.split("::")[0]
        x1,y1,ypass,textsize = 0.75,0.8,0.08,0.035
        if "::" in moreTextLatex:
            x1,y1,ypass,textsize = (float(x) for x in (moreTextLatex.split("::")[1]).split(","))            
        lat = ROOT.TLatex()
        lat.SetNDC();
        lat.SetTextFont(42)        
        lat.SetTextSize(textsize)
        for itx,tx in enumerate(realtext.split(";")):
            lat.DrawLatex(x1,y1-itx*ypass,tx)


  # TPaveText *pvtxt = NULL;
  # if (yAxisName == "a.u.") {
  #   pvtxt = new TPaveText(0.6,0.6,0.95,0.7, "BR NDC")
  #   pvtxt.SetFillColor(0)
  #   pvtxt.SetFillStyle(0)
  #   pvtxt.SetBorderSize(0)
  #   pvtxt.AddText(Form("norm num/den = %.2f +/- %.2f",IntegralRatio,ratioError))
  #   pvtxt.Draw()
  # }

    setTDRStyle()
    if not skipLumi:
        if leftMargin > 0.1:
            if lumi != None: CMS_lumi(canvas,lumi,True,False)
            else:            CMS_lumi(canvas,"",True,False)
        else:
            latCMS = ROOT.TLatex()
            latCMS.SetNDC();
            latCMS.SetTextFont(42)
            latCMS.SetTextSize(0.045)
            latCMS.DrawLatex(leftMargin, 0.95, '#bf{CMS} #it{Preliminary}')
            if lumi != None: latCMS.DrawLatex(0.91, 0.95, '%s fb^{-1} (13 TeV)' % lumi)
            else:            latCMS.DrawLatex(0.96, 0.95, '(13 TeV)')

    if drawLineTopPanel != None:
        topline = ROOT.TF1("horiz_line",f"{drawLineTopPanel}",h1.GetXaxis().GetBinLowEdge(1),h1.GetXaxis().GetBinLowEdge(h1.GetNbinsX()+1))
        topline.SetLineColor(ROOT.kRed)
        topline.SetLineWidth(1)
        topline.Draw("Lsame")
        
    if lowerPanelHeight:
        pad2.Draw()
        pad2.cd()

        frame.Reset("ICES")
        if setRatioYAxisRangeFromUser: frame.GetYaxis().SetRangeUser(yminRatio,ymaxRatio)
        #else:                          
        #frame.GetYaxis().SetRangeUser(0.5,1.5)
        frame.GetYaxis().SetNdivisions(5)
        frame.GetYaxis().SetTitle(yRatioAxisName)
        frame.GetYaxis().SetTitleOffset(yAxisTitleOffset)
        frame.GetYaxis().SetTitleSize(0.05)
        frame.GetYaxis().SetLabelSize(0.04)
        frame.GetYaxis().CenterTitle()
        frame.GetXaxis().SetTitle(xAxisName)
        if setXAxisRangeFromUser: frame.GetXaxis().SetRangeUser(xmin,xmax)
        frame.GetXaxis().SetTitleOffset(1.2)
        frame.GetXaxis().SetTitleSize(0.05)

        ratio = h1.Clone("ratio")
        den_noerr = h1.Clone("den_noerr")
        for iBin in range (1,den_noerr.GetNbinsX()+1):
            den_noerr.SetBinError(iBin,0.)

        ratio.Divide(den_noerr)
        ratio.SetFillColor(ROOT.kGray+1)
        #den_noerr.SetFillColor(ROOT.kGray)
        frame.Draw()
        ratio.SetMarkerSize(0)
        ratio.SetMarkerStyle(0) # important to remove dots at y = 1
        ratio.Draw("E2same")

        line = ROOT.TF1("horiz_line","1",ratio.GetXaxis().GetBinLowEdge(1),ratio.GetXaxis().GetBinLowEdge(ratio.GetNbinsX()+1))
        line.SetLineColor(ROOT.kRed)
        line.SetLineWidth(1)
        line.Draw("Lsame")
        
        if drawLineLowerPanel:
            legEntry,yline = drawLineLowerPanel.split('::')
            line2 = ROOT.TF1("horiz_line_2",str(1+float(yline)),ratio.GetXaxis().GetBinLowEdge(1),ratio.GetXaxis().GetBinLowEdge(ratio.GetNbinsX()+1))
            line3 = ROOT.TF1("horiz_line_3",str(1-float(yline)),ratio.GetXaxis().GetBinLowEdge(1),ratio.GetXaxis().GetBinLowEdge(ratio.GetNbinsX()+1))
            line2.SetLineColor(ROOT.kBlue)
            line2.SetLineWidth(1)
            line2.Draw("Lsame")
            line3.SetLineColor(ROOT.kBlue)
            line3.SetLineWidth(1)
            line3.Draw("Lsame")
            x1leg2 = 0.2 if leftMargin > 0.1 else 0.07
            x2leg2 = 0.5 if leftMargin > 0.1 else 0.27
            y1leg2 = 0.25 if leftMargin > 0.1 else 0.3
            y2leg2 = 0.35 if leftMargin > 0.1 else 0.35
            leg2 = ROOT.TLegend(x1leg2, y1leg2, x2leg2, y2leg2)
            leg2.SetFillColor(0)
            leg2.SetFillStyle(0)
            leg2.SetBorderSize(0)
            leg2.AddEntry(line2,legEntry,"L")
            leg2.Draw("same")
        
        pad2.RedrawAxis("sameaxis")


    if draw_both0_noLog1_onlyLog2 != 2:
        canvas.SaveAs(outdir + canvasName + ".png")
        canvas.SaveAs(outdir + canvasName + ".pdf")

    if draw_both0_noLog1_onlyLog2 != 1:        
        if yAxisName == "a.u.": 
            h1.GetYaxis().SetRangeUser(max(0.0001,h1.GetMinimum()*0.8),h1.GetMaximum()*100)
        else:
            h1.GetYaxis().SetRangeUser(max(0.001,h1.GetMinimum()*0.8),h1.GetMaximum()*100)
        canvas.SetLogy()
        canvas.SaveAs(outdir + canvasName + "_logY.png")
        canvas.SaveAs(outdir + canvasName + "_logY.pdf")
        canvas.SetLogy(0)


################################################################

def pol1_root_(xvals, parms, xLowVal = 0.0, xFitRange = 1.0):
    xscaled = (xvals[0] - xLowVal) / xFitRange
    return parms[0] + parms[1]*xscaled

def pol2_root_(xvals, parms, xLowVal = 0.0, xFitRange = 1.0):
    xscaled = (xvals[0] - xLowVal) / xFitRange
    return parms[0] + parms[1]*xscaled + parms[2]*xscaled**2

def pol3_root_(xvals, parms, xLowVal = 0.0, xFitRange = 1.0):
    xscaled = (xvals[0] - xLowVal) / xFitRange
    return parms[0] + parms[1]*xscaled + parms[2]*xscaled**2 + parms[3]*xscaled**3

def polN_root_(xvals, parms, xLowVal = 0.0, xFitRange = 1.0, degree = 3):
    xscaled = (xvals[0] - xLowVal) / xFitRange
    ret = parms[0]
    for d in range(1, 1+degree):
        ret += parms[d]*xscaled**d
    return ret

def drawSingleTH1withFit(h1,
                         labelXtmp="xaxis", labelYtmp="yaxis",
                         canvasName="default", outdir="./",
                         rebinFactorX=0,
                         draw_both0_noLog1_onlyLog2=1,
                         leftMargin=0.15,
                         rightMargin=0.04,
                         labelRatioTmp="Rel.Unc.::0.5,1.5",
                         drawStatBox=False,
                         legendCoords="0.15,0.35,0.8,0.9",  # x1,x2,y1,y2
                         canvasSize="600,700",  # use X,Y to pass X and Y size     
                         lowerPanelHeight = 0.3,  # number from 0 to 1, 0.3 means 30% of space taken by lower panel. 0 means do not draw lower panel with relative error
                         drawLineLowerPanel="luminosity uncertainty::0.025", # if not empty, draw band at 1+ number after ::, and add legend with title
                         passCanvas=None,
                         lumi=None,
                         moreText="",
                         moreTextLatex="",
                         fitRange="0,40", # xmin and xmax
                         fitOptions="WLMFS+",
                         evalAt=None
):

    # moreText is used to pass some text to write somewhere (TPaveText is used)
    # e.g.  "stuff::x1,y1,x2,y2"  where xi and yi are the coordinates for the text
    # one can add more lines using the ";" key. FOr example, "stuff1;stuff2::x1,y1,x2,y2"
    # the coordinates should be defined taking into account how many lines will be drawn
    # if the coordinates are not passed (no "::"), then default ones are used, but this might not be satisfactory

    # moreTextLatex is similar, but used TLatex, and the four coordinates are x1,y1,ypass,textsize
    # where x1 and y1 are the coordinates the first line, and ypass is how much below y1 the second line is (and so on for following lines)

    if (rebinFactorX): 
        if isinstance(rebinFactorX, int): h1.Rebin(rebinFactorX)
        # case in which rebinFactorX is a list of bin edges
        else:                             h1.Rebin(len(rebinFactorX)-1,"",array('d',rebinFactorX)) 

    xAxisName,setXAxisRangeFromUser,xmin,xmax = getAxisRangeFromUser(labelXtmp)
    yAxisName,setYAxisRangeFromUser,ymin,ymax = getAxisRangeFromUser(labelYtmp)
    yRatioAxisName,setRatioYAxisRangeFromUser,yminRatio,ymaxRatio = getAxisRangeFromUser(labelRatioTmp)

    yAxisTitleOffset = 1.45 if leftMargin > 0.1 else 0.6

    addStringToEnd(outdir,"/",notAddIfEndswithMatch=True)
    createPlotDirAndCopyPhp(outdir)

    cw,ch = canvasSize.split(',')
    #canvas = ROOT.TCanvas("canvas",h2D.GetTitle() if plotLabel == "ForceTitle" else "",700,625)
    canvas = passCanvas if passCanvas != None else ROOT.TCanvas("canvas","",int(cw),int(ch))
    canvas.SetTickx(1)
    canvas.SetTicky(1)
    canvas.cd()
    canvas.SetLeftMargin(leftMargin)
    canvas.SetRightMargin(rightMargin)
    canvas.cd()

    pad2 = 0
    if lowerPanelHeight: 
        canvas.SetBottomMargin(lowerPanelHeight)
        pad2 = ROOT.TPad("pad2","pad2",0,0.,1,0.9)
        pad2.SetTopMargin(1-lowerPanelHeight)
        pad2.SetRightMargin(rightMargin)
        pad2.SetLeftMargin(leftMargin)
        pad2.SetFillColor(0)
        pad2.SetGridy(1)
        pad2.SetFillStyle(0)


    frame = h1.Clone("frame")
    frame.GetXaxis().SetLabelSize(0.04)
    frame.SetStats(0)

    h1.SetLineColor(ROOT.kBlack)
    h1.SetMarkerColor(ROOT.kBlack)
    h1.SetMarkerStyle(20)
    h1.SetMarkerSize(0)

    if ymin == ymax == 0.0:
        ymin,ymax = getMinMaxHisto(h1,excludeEmpty=True,sumError=True)            
        ymin *= 0.9
        ymax *= (1.1 if leftMargin > 0.1 else 2.0)
        if ymin < 0: ymin = 0

    if lowerPanelHeight:
        h1.GetXaxis().SetLabelSize(0)
        h1.GetXaxis().SetTitle("")  
    else:
        h1.GetXaxis().SetTitle(xAxisName)
        h1.GetXaxis().SetTitleOffset(1.2)
        h1.GetXaxis().SetTitleSize(0.05)
        h1.GetXaxis().SetLabelSize(0.04)
    h1.GetYaxis().SetTitle(yAxisName)
    h1.GetYaxis().SetTitleOffset(yAxisTitleOffset) 
    h1.GetYaxis().SetTitleSize(0.05)
    h1.GetYaxis().SetLabelSize(0.04)
    h1.GetYaxis().SetRangeUser(ymin, ymax)
    h1.GetYaxis().SetTickSize(0.01)
    if setXAxisRangeFromUser: h1.GetXaxis().SetRangeUser(xmin,xmax)
    h1.Draw("HIST")
    h1err = h1.Clone("h1err")
    h1err.SetFillColor(ROOT.kRed+2)
    h1err.SetFillStyle(3001)  # 3001 is better than 3002 for pdf, while 3002 is perfect for png
    #h1err.SetFillStyle(3002)
    #h1err.SetFillStyle(3005)
    h1err.Draw("E2same")
    #h1.Draw("HIST same")

    xMinFit, xMaxFit = map(float, fitRange.split(','))
    pol1_scaled = partial(pol1_root_, xLowVal=xMinFit, xFitRange=xMaxFit)
    fpol1 = ROOT.TF1("fpol1",pol1_scaled, h1.GetXaxis().GetBinLowEdge(1), 30, 2)
    fpol1.SetParLimits(1, -50.0, 0.0)
    pol2_scaled = partial(pol2_root_, xLowVal=xMinFit, xFitRange=xMaxFit)
    f1 = ROOT.TF1("f1",pol2_scaled, h1.GetXaxis().GetBinLowEdge(1), xMaxFit, 3)
    f1.SetParLimits(2, -10.0, 0.0)
    ## TODO: set coefficient of x^1 as 0 to have the maximum at 0?
    realFitOptions = fitOptions
    if "B" not in fitOptions:
        realFitOptions = "B" + fitOptions
    fitres = h1.Fit("f1", realFitOptions, "", xMinFit, xMaxFit)
    h1.Fit("fpol1", realFitOptions, "", xMinFit, 30)
    f1.SetLineWidth(3)
    #f1.SetLineStyle(9) # ROOT.kDashed == thin dashes, almost dotted
    f1.SetLineColor(ROOT.kBlue+1)
    f1.Draw("L SAME")
    fpol1.SetLineWidth(3)
    fpol1.SetLineColor(ROOT.kGreen+2)
    fpol1.Draw("L SAME")
    postfitpars = [f1.GetParameter(0), f1.GetParameter(1), f1.GetParameter(2)]
    f2 = ROOT.TF1("f2",pol2_scaled, xMaxFit, h1.GetXaxis().GetBinLowEdge(1+h1.GetNbinsX()), 3)
    f2.SetParameter(0, postfitpars[0])
    f2.SetParameter(1, postfitpars[1])
    f2.SetParameter(2, postfitpars[2])
    f2.SetLineColor(ROOT.kRed+2)
    f2.SetLineWidth(3)
    f2.SetLineStyle(9)
    f2.Draw("L SAME")
    f2pol1 = ROOT.TF1("f2pol1",pol1_scaled, 30, h1.GetXaxis().GetBinLowEdge(1+h1.GetNbinsX()), 2)
    f2pol1.SetParameter(0, fpol1.GetParameter(0))
    f2pol1.SetParameter(1, fpol1.GetParameter(1))
    f2pol1.SetLineColor(ROOT.kRed+2)
    f2pol1.SetLineWidth(3)
    f2pol1.SetLineStyle(9)
    f2pol1.Draw("L SAME")
    
    nColumnsLeg = 1
    if ";" in legendCoords: 
        nColumnsLeg = int(legendCoords.split(";")[1])
    legcoords = [float(x) for x in (legendCoords.split(";")[0]).split(',')]
    lx1,lx2,ly1,ly2 = legcoords[0],legcoords[1],legcoords[2],legcoords[3]
    leg = ROOT.TLegend(lx1,ly1,lx2,ly2)
    #leg.SetFillColor(0)
    #leg.SetFillStyle(0)
    #leg.SetBorderSize(0)
    leg.SetNColumns(nColumnsLeg)
    #leg.AddEntry(h1,"Value","L")
    #leg.AddEntry(h1err,"Uncertainty","F")
    leg.AddEntry(h1err,"Measurement","LF")
    leg.AddEntry(f1,f"Fit pol2 in [{int(xMinFit)}, {int(xMaxFit)}]","L")
    leg.AddEntry(fpol1,f"Fit pol1 in [{int(xMinFit)}, 30]","L")
    leg.AddEntry(f2,f"Extrapolation","L")
    leg.Draw("same")
    canvas.RedrawAxis("sameaxis")

    if drawStatBox:
        ROOT.gPad.Update()
        ROOT.gStyle.SetOptStat(1110)
        ROOT.gStyle.SetOptFit(1102)
    else:
        h1.SetStats(0)

    # redraw legend, or vertical lines appear on top of it
    leg.Draw("same")

    if len(moreText):
        realtext = moreText.split("::")[0]
        x1,y1,x2,y2 = 0.7,0.8,0.9,0.9
        if "::" in moreText:
            x1,y1,x2,y2 = (float(x) for x in (moreText.split("::")[1]).split(","))
        pavetext = ROOT.TPaveText(x1,y1,x2,y2,"NB NDC")
        for tx in realtext.split(";"):
            pavetext.AddText(tx)
        pavetext.SetFillColor(0)
        pavetext.SetFillStyle(0)
        pavetext.SetBorderSize(0)
        pavetext.SetLineColor(0)
        pavetext.Draw("same")

    if len(moreTextLatex):
        realtext = moreTextLatex.split("::")[0]
        x1,y1,ypass,textsize = 0.75,0.8,0.08,0.035
        if "::" in moreTextLatex:
            x1,y1,ypass,textsize = (float(x) for x in (moreTextLatex.split("::")[1]).split(","))            
        lat = ROOT.TLatex()
        lat.SetNDC();
        lat.SetTextFont(42)        
        lat.SetTextSize(textsize)
        for itx,tx in enumerate(realtext.split(";")):
            lat.DrawLatex(x1,y1-itx*ypass,tx)

    setTDRStyle()
    if leftMargin > 0.1:
        if lumi != None: CMS_lumi(canvas,lumi,True,False)
        else:            CMS_lumi(canvas,"",True,False)
    else:
        latCMS = ROOT.TLatex()
        latCMS.SetNDC();
        latCMS.SetTextFont(42)
        latCMS.SetTextSize(0.045)
        latCMS.DrawLatex(0.1, 0.95, '#bf{CMS} #it{Preliminary}')
        if lumi != None: latCMS.DrawLatex(0.85, 0.95, '%s fb^{-1} (13 TeV)' % lumi)
        else:            latCMS.DrawLatex(0.90, 0.95, '(13 TeV)' % lumi)

    if lowerPanelHeight:
        pad2.Draw()
        pad2.cd()

        frame.Reset("ICES")
        if setRatioYAxisRangeFromUser: frame.GetYaxis().SetRangeUser(yminRatio,ymaxRatio)
        #else:                          
        #frame.GetYaxis().SetRangeUser(0.5,1.5)
        frame.GetYaxis().SetNdivisions(5)
        frame.GetYaxis().SetTitle(yRatioAxisName)
        frame.GetYaxis().SetTitleOffset(yAxisTitleOffset)
        frame.GetYaxis().SetTitleSize(0.05)
        frame.GetYaxis().SetLabelSize(0.04)
        frame.GetYaxis().CenterTitle()
        frame.GetXaxis().SetTitle(xAxisName)
        if setXAxisRangeFromUser: frame.GetXaxis().SetRangeUser(xmin,xmax)
        frame.GetXaxis().SetTitleOffset(1.2)
        frame.GetXaxis().SetTitleSize(0.05)

        ratio = h1.Clone("ratio")
        den_noerr = h1.Clone("den_noerr")
        for iBin in range (1,den_noerr.GetNbinsX()+1):
            den_noerr.SetBinError(iBin,0.)

        ratio.Divide(den_noerr)
        ratio.SetFillColor(ROOT.kGray+1)
        #den_noerr.SetFillColor(ROOT.kGray)
        frame.Draw()
        ratio.SetMarkerSize(0)
        ratio.SetMarkerStyle(0) # important to remove dots at y = 1
        ratio.Draw("E2same")

        line = ROOT.TF1("horiz_line","1",ratio.GetXaxis().GetBinLowEdge(1),ratio.GetXaxis().GetBinLowEdge(ratio.GetNbinsX()+1))
        line.SetLineColor(ROOT.kRed)
        line.SetLineWidth(1)
        line.Draw("Lsame")

        if drawLineLowerPanel:
            legEntry,yline = drawLineLowerPanel.split('::')
            line2 = ROOT.TF1("horiz_line_2",str(1+float(yline)),ratio.GetXaxis().GetBinLowEdge(1),ratio.GetXaxis().GetBinLowEdge(ratio.GetNbinsX()+1))
            line3 = ROOT.TF1("horiz_line_3",str(1-float(yline)),ratio.GetXaxis().GetBinLowEdge(1),ratio.GetXaxis().GetBinLowEdge(ratio.GetNbinsX()+1))
            line2.SetLineColor(ROOT.kBlue)
            line2.SetLineWidth(1)
            line2.Draw("Lsame")
            line3.SetLineColor(ROOT.kBlue)
            line3.SetLineWidth(1)
            line3.Draw("Lsame")
            x1leg2 = 0.2 if leftMargin > 0.1 else 0.07
            x2leg2 = 0.5 if leftMargin > 0.1 else 0.27
            y1leg2 = 0.25 if leftMargin > 0.1 else 0.3
            y2leg2 = 0.35 if leftMargin > 0.1 else 0.35
            leg2 = ROOT.TLegend(x1leg2, y1leg2, x2leg2, y2leg2)
            leg2.SetFillColor(0)
            leg2.SetFillStyle(0)
            leg2.SetBorderSize(0)
            leg2.AddEntry(line2,legEntry,"L")
            leg2.Draw("same")

        
        pad2.RedrawAxis("sameaxis")


    if draw_both0_noLog1_onlyLog2 != 2:
        canvas.SaveAs(outdir + canvasName + ".png")
        canvas.SaveAs(outdir + canvasName + ".pdf")

    if draw_both0_noLog1_onlyLog2 != 1:        
        if yAxisName == "a.u.": 
            h1.GetYaxis().SetRangeUser(max(0.0001,h1.GetMinimum()*0.8),h1.GetMaximum()*100)
        else:
            h1.GetYaxis().SetRangeUser(max(0.001,h1.GetMinimum()*0.8),h1.GetMaximum()*100)
        canvas.SetLogy()
        canvas.SaveAs(outdir + canvasName + "_logY.png")
        canvas.SaveAs(outdir + canvasName + "_logY.pdf")
        canvas.SetLogy(0)

    if evalAt:
        return f1.Eval(evalAt), fpol1.Eval(evalAt)
    else:
        return f1, fpol1
        
################################################################

def drawNTH1(hists=[],
             legEntries=[],
             labelXtmp="xaxis", labelYtmp="yaxis",
             canvasName="default",
             outdir="./",
             rebinFactorX=0,
             draw_both0_noLog1_onlyLog2=1,
             topMargin=0.1,
             leftMargin=0.15,
             rightMargin=0.04,
             bottomMargin=0.15,
             labelRatioTmp="Rel.Unc.::0.5,1.5",
             drawStatBox=False,
             legendCoords="0.15,0.35,0.8,0.9",  # x1,x2,y1,y2
             canvasSize="600,700",  # use X,Y to pass X and Y size     
             lowerPanelHeight = 0.3,  # number from 0 to 1, 0.3 means 30% of space taken by lower panel. 0 means do not draw lower panel with relative error
             drawLineTopPanel=None,
             drawLineLowerPanel="", # if not empty, draw band at 1+ number after ::, and add legend with title
             passCanvas=None,
             lumi=None,
             drawLumiLatex=False,
             skipLumi=False,
             drawVertLines="", # coordinates of x where to print line
             textForLines=[],                       
             moreText="",
             moreTextLatex="",
             transparentLegend=True,
             onlyLineColor=False,
             drawErrorAll=False, # default draws error only on first histogram
             noErrorRatioDen=False, # remove gray error band from denominator
             yAxisExtendConstant=1.2,
             markerStyleFirstHistogram=20,
             useLineFirstHistogram=False,
             fillStyleSecondHistogram=3004,
             fillColorSecondHistogram=None,
             colorVec=None,
             setRatioRangeFromHisto=False, # currently only for 2 histograms in hists
             setOnlyLineRatio=False,
             lineWidth=2,
             ytextOffsetFromTop=0.15,  # in % of maxy-miny of the top panel
             useMultiHistRatioOption=False):

    # moreText is used to pass some text to write somewhere (TPaveText is used)
    # e.g.  "stuff::x1,y1,x2,y2"  where xi and yi are the coordinates for the text
    # one can add more lines using the ";" key. For example, "stuff1;stuff2::x1,y1,x2,y2"
    # the coordinates should be defined taking into account how many lines will be drawn
    # if the coordinates are not passed (no "::"), then default ones are used, but this might not be satisfactory

    # moreTextLatex is similar, but used TLatex, and the four coordinates are x1,y1,ypass,textsize
    # where x1 and y1 are the coordinates the first line, and ypass is how much below y1 the second line is (and so on for following lines)

    if len(hists) != len(legEntries):
        logger.warning("In drawNTH1: #(hists) != #(legEntries). Abort")
        quit()

    if (rebinFactorX): 
        if isinstance(rebinFactorX, int): h1.Rebin(rebinFactorX)
        # case in which rebinFactorX is a list of bin edges
        else:                             h1.Rebin(len(rebinFactorX)-1,"",array('d',rebinFactorX)) 

    xAxisName,setXAxisRangeFromUser,xmin,xmax = getAxisRangeFromUser(labelXtmp)
    yAxisName,setYAxisRangeFromUser,ymin,ymax = getAxisRangeFromUser(labelYtmp)
    yRatioAxisName,setRatioYAxisRangeFromUser,yminRatio,ymaxRatio = getAxisRangeFromUser(labelRatioTmp)
    #print(yminRatio,ymaxRatio)
    
    yAxisTitleOffset = 1.45 if leftMargin > 0.1 else 0.6

    adjustSettings_CMS_lumi()
    addStringToEnd(outdir,"/",notAddIfEndswithMatch=True)
    createPlotDirAndCopyPhp(outdir)
    

    cw,ch = canvasSize.split(',')
    #canvas = ROOT.TCanvas("canvas",h2D.GetTitle() if plotLabel == "ForceTitle" else "",700,625)
    canvas = passCanvas if passCanvas != None else ROOT.TCanvas("canvas","",int(cw),int(ch))
    canvas.SetTickx(1)
    canvas.SetTicky(1)
    canvas.cd()
    canvas.SetTopMargin(topMargin)
    canvas.SetLeftMargin(leftMargin)
    canvas.SetRightMargin(rightMargin)
    canvas.cd()

    pad2 = 0
    if lowerPanelHeight: 
        canvas.SetBottomMargin(lowerPanelHeight)
        pad2 = ROOT.TPad("pad2","pad2",0,0.,1,0.9)
        pad2.SetTopMargin(1-lowerPanelHeight)
        pad2.SetRightMargin(rightMargin)
        pad2.SetLeftMargin(leftMargin)
        pad2.SetFillColor(0)
        pad2.SetGridy(1)
        pad2.SetFillStyle(0)
    else:
        canvas.SetBottomMargin(bottomMargin)

    h1 = hists[0]
    hnums = [hists[i] for i in range(1,len(hists))]
    frame = h1.Clone("frame")
    frame.GetXaxis().SetLabelSize(0.04)
    frame.SetStats(0)

    h1.SetLineColor(ROOT.kBlack)
    if useLineFirstHistogram:
        h1.SetMarkerSize(0)
        h1.SetLineWidth(lineWidth) 
    else:
        h1.SetMarkerColor(ROOT.kBlack)
        h1.SetMarkerStyle(markerStyleFirstHistogram)

    if colorVec != None:
        colors = colorVec
    else:
        colors = [ROOT.kRed+2, ROOT.kBlue, ROOT.kGreen+2, ROOT.kOrange+7,
                  ROOT.kAzure+2, ROOT.kMagenta,
                  ROOT.kViolet, ROOT.kCyan+1, ROOT.kPink+2, ROOT.kSpring-8]
    for ic,h in enumerate(hnums):
        # h.SetLineColor(colors[ic])
        # h.SetFillColor(colors[ic])
        # if ic==0: h.SetFillStyle(3004)   
        # if ic==2: h.SetFillStyle(3002)   
        # h.SetFillColor(colors[ic])
        # h.SetMarkerSize(0)
        h.SetLineColor(colors[ic])
        h.SetMarkerSize(0)
        if not onlyLineColor:
            h.SetFillColor(colors[ic])
            if ic==0: 
                h.SetFillStyle(fillStyleSecondHistogram)
                if fillColorSecondHistogram:
                    h.SetLineWidth(lineWidth)
                    h.SetFillColor(fillColorSecondHistogram)
            if ic==1: 
                h.SetFillColor(0) 
                h.SetLineWidth(lineWidth) 
            if ic==2: 
                h.SetFillStyle(3002)           
            if ic==3:
                h.SetFillColor(0)
                h1.SetMarkerColor(ROOT.kGray+3)
                h1.SetMarkerStyle(25)
                #h1.SetMarkerSize(2)
        else:
            h.SetLineWidth(lineWidth)
    #ymax = max(ymax, max(h1.GetBinContent(i)+h1.GetBinError(i) for i in range(1,h1.GetNbinsX()+1)))
    # if min and max were not set, set them based on histogram content
    if ymin == ymax == 0.0:
        # ymin,ymax = getMinMaxHisto(h1,excludeEmpty=True,sumError=True)            
        # ymin *= 0.9
        # ymax *= (1.1 if leftMargin > 0.1 else 2.0)
        # if ymin < 0: ymin = 0
        #print "drawSingleTH1() >>> Histo: %s     minY,maxY = %.2f, %.2f" % (h1.GetName(),ymin,ymax)
        ymin = 9999.9
        ymax = -9999.9
        for h in hists:
            if h.GetBinContent(h.GetMaximumBin()) > ymax: ymax = h.GetBinContent(h.GetMaximumBin())
            if h.GetBinContent(h.GetMinimumBin()) < ymin: ymin = h.GetBinContent(h.GetMinimumBin())
        if ymin < 0: ymin = 0
        ymax = (ymax - ymin) * yAxisExtendConstant + ymin
        
    if lowerPanelHeight:
        h1.GetXaxis().SetLabelSize(0)
        h1.GetXaxis().SetTitle("")  
    else:
        h1.GetXaxis().SetTitle(xAxisName)
        h1.GetXaxis().SetTitleOffset(1.2)
        h1.GetXaxis().SetTitleSize(0.05)
        h1.GetXaxis().SetLabelSize(0.04)
    h1.GetYaxis().SetTitle(yAxisName)
    h1.GetYaxis().SetTitleOffset(yAxisTitleOffset) 
    h1.GetYaxis().SetTitleSize(0.05)
    h1.GetYaxis().SetLabelSize(0.04)
    h1.GetYaxis().SetRangeUser(ymin, ymax)
    h1.GetYaxis().SetTickSize(0.01)
    if setXAxisRangeFromUser: h1.GetXaxis().SetRangeUser(xmin,xmax)
    h1.Draw("HE" if useLineFirstHistogram else "PE")
    for ih,h in enumerate(hnums):
        if ih == 0 and fillColorSecondHistogram != None:
            h.Draw("E2 SAME" if drawErrorAll else "HIST SAME")
        else:
            h.Draw("HE SAME" if drawErrorAll else "HIST SAME")
    h1.Draw("HE SAME" if useLineFirstHistogram else "PE SAME")

    nColumnsLeg = 1
    legHeader = ""
    if ";" in legendCoords: 
        tokens = legendCoords.split(";")
        nColumnsLeg = int(tokens[1])
        if len(tokens) > 2:
            legHeader = tokens[2]
    legcoords = [float(x) for x in (legendCoords.split(";")[0]).split(',')]
    lx1,lx2,ly1,ly2 = legcoords[0],legcoords[1],legcoords[2],legcoords[3]
    leg = ROOT.TLegend(lx1,ly1,lx2,ly2)
    if transparentLegend:
        leg.SetFillColor(0)
        leg.SetFillStyle(0)
        leg.SetFillColorAlpha(0,0.6)
        leg.SetShadowColor(0)
        leg.SetBorderSize(0)
    leg.SetNColumns(nColumnsLeg)
    if legHeader:
        leg.SetHeader(legHeader)
    firstHistogramStyle = "L" if useLineFirstHistogram else "PE"
    for il,le in enumerate(legEntries):
        leg.AddEntry(hists[il],le,firstHistogramStyle if il == 0 else "L" if onlyLineColor else "FL")
    leg.Draw("same")
    canvas.RedrawAxis("sameaxis")

    if drawStatBox:
        ROOT.gPad.Update()
        ROOT.gStyle.SetOptStat(1110)
        ROOT.gStyle.SetOptFit(1102)
    else:
        for htmp in hists:
            htmp.SetStats(0)

    vertline = ROOT.TLine(36,0,36,canvas.GetUymax())
    vertline.SetLineColor(ROOT.kBlack)
    vertline.SetLineStyle(2)
    bintext = ROOT.TLatex()
    #bintext.SetNDC()
    bintext.SetTextSize(0.04)  # 0.03
    bintext.SetTextFont(42)
    #if len(textForLines): bintext.SetTextAngle(45 if "#eta" in textForLines[0] else 30)

    if len(drawVertLines):
        nptBins = int(drawVertLines.split(',')[0])
        etarange = float(drawVertLines.split(',')[1])
        offsetXaxisHist = h1.GetXaxis().GetBinLowEdge(0)
        sliceLabelOffset = 6. if "#eta" in textForLines[0] else 6.
        for i in range(1,nptBins): # do not need line at canvas borders 
            vertline.DrawLine(etarange*i-offsetXaxisHist,ymin,etarange*i-offsetXaxisHist,ymax)
        if len(textForLines):
            for i in range(0,len(textForLines)):
                ytext = ymax - ytextOffsetFromTop*(ymax - ymin)
                bintext.DrawLatex(etarange*i + etarange/sliceLabelOffset, ytext, textForLines[i])
    
    # if len(drawVertLines):
    #     nLines = len(drawVertLines)
    #     sliceLabelOffset = 10.
    #     for i in range(nLines):
    #         vertline.DrawLine(float(drawVertLines[i]), 0.0, float(drawVertLines[i]), ymax)
    #     if len(textForLines):
    #         for i in range(len(textForLines)): # we need nLines
    #             ytext = (1.1)*ymax/2.
    #             if i == 0:
    #                 bintext.DrawLatex(h1.GetXaxis().GetBinLowEdge(0) + sliceLabelOffset, ytext, textForLines[i])
    #             else:                    
    #                 bintext.DrawLatex(drawVertLines[i-1] + sliceLabelOffset, ytext, textForLines[i])

    # redraw legend, or vertical lines appear on top of it
    leg.Draw("same")

    if len(moreText):
        realtext = moreText.split("::")[0]
        x1,y1,x2,y2 = 0.7,0.8,0.9,0.9
        if "::" in moreText:
            x1,y1,x2,y2 = (float(x) for x in (moreText.split("::")[1]).split(","))
        pavetext = ROOT.TPaveText(x1,y1,x2,y2,"NB NDC")
        for tx in realtext.split(";"):
            pavetext.AddText(tx)
        pavetext.SetFillColor(0)
        pavetext.SetFillStyle(0)
        pavetext.SetBorderSize(0)
        pavetext.SetLineColor(0)
        pavetext.Draw("same")

    if len(moreTextLatex):
        realtext = moreTextLatex.split("::")[0]
        x1,y1,ypass,textsize = 0.75,0.8,0.08,0.035
        if "::" in moreTextLatex:
            x1,y1,ypass,textsize = (float(x) for x in (moreTextLatex.split("::")[1]).split(","))            
        lat = ROOT.TLatex()
        lat.SetNDC();
        lat.SetTextFont(42)        
        lat.SetTextSize(textsize)
        for itx,tx in enumerate(realtext.split(";")):
            lat.DrawLatex(x1,y1-itx*ypass,tx)

    setTDRStyle()
    if not skipLumi:
        if not drawLumiLatex:
            if lumi != None: CMS_lumi(canvas,lumi,True,False)
            else:            CMS_lumi(canvas,"",True,False)
        else:
            latCMS = ROOT.TLatex()
            latCMS.SetNDC();
            latCMS.SetTextFont(42)
            latCMS.SetTextSize(0.05)
            latCMS.DrawLatex(0.1, 0.95, '#bf{CMS} #it{Preliminary}')
            if lumi != None: latCMS.DrawLatex(0.85, 0.95, '%s fb^{-1} (13 TeV)' % lumi)
            else:            latCMS.DrawLatex(0.90, 0.95, '(13 TeV)')

    if drawLineTopPanel != None:
        topline = ROOT.TF1("horiz_line",f"{drawLineTopPanel}",h1.GetXaxis().GetBinLowEdge(1),h1.GetXaxis().GetBinLowEdge(h1.GetNbinsX()+1))
        topline.SetLineColor(ROOT.kBlack)
        topline.SetLineWidth(1)
        topline.SetLineStyle(2)
        topline.Draw("Lsame")

    if lowerPanelHeight:
        pad2.Draw()
        pad2.cd()

        frame.Reset("ICES")
        if setRatioYAxisRangeFromUser: frame.GetYaxis().SetRangeUser(yminRatio,ymaxRatio)
        #else:                          
        #frame.GetYaxis().SetRangeUser(0.5,1.5)
        frame.GetYaxis().SetNdivisions(5)
        frame.GetYaxis().SetTitle(yRatioAxisName)
        frame.GetYaxis().SetTitleOffset(yAxisTitleOffset)
        frame.GetYaxis().SetTitleSize(0.05)
        frame.GetYaxis().SetLabelSize(0.04)
        frame.GetYaxis().CenterTitle()
        frame.GetXaxis().SetTitle(xAxisName)
        if setXAxisRangeFromUser: frame.GetXaxis().SetRangeUser(xmin,xmax)
        frame.GetXaxis().SetTitleOffset(1.2)
        frame.GetXaxis().SetTitleSize(0.05)

        if len(hists) == 2 and not useMultiHistRatioOption:
            ratio = h1.Clone("ratio")
            den = hnums[0].Clone("den")
            den_noerr = hnums[0].Clone("den_noerr")
            for iBin in range (1,den_noerr.GetNbinsX()+1):
                den_noerr.SetBinError(iBin,0.)
            den.Divide(den_noerr)
            ratio.Divide(den_noerr)
            if setRatioRangeFromHisto:
                newymin = ratio.GetBinContent(ratio.GetMinimumBin())
                newymax = ratio.GetBinContent(ratio.GetMaximumBin())
                if newymin == newymax:
                    newymin *= 0.99
                    newymax *= 1.01
                else:
                    newdiff = newymax - newymin
                    #print(f"newdiff = {newdiff}")
                    newymin = max(0, newymin - 0.1 * newdiff)
                    newymax = newymax + 0.1 * newdiff
                if not setRatioYAxisRangeFromUser:
                    frame.GetYaxis().SetRangeUser(newymin, newymax)
            #den_noerr.SetFillColor(ROOT.kGray)
            frame.Draw()
            frame.SetMarkerSize(0)
            frame.SetMarkerStyle(0) # important to remove dots at y = 1
            if setOnlyLineRatio:
                ratio.SetLineColor(ROOT.kBlack)
                ratio.SetMarkerSize(0)
                ratio.Draw("LSAME")
            else:
                den.SetFillColor(ROOT.kGray+1)
                den.SetFillStyle(1001)
                den.Draw("E2same")
                ratio.Draw("EPSAME")
        else:
            ratio = h1.Clone("ratio")
            den_noerr = h1.Clone("den_noerr")
            for iBin in range (1,den_noerr.GetNbinsX()+1):
                den_noerr.SetBinError(iBin,0.)
            ratio.Divide(den_noerr)
            #den_noerr.SetFillColor(ROOT.kGray)
            frame.Draw()
            ratio.SetMarkerSize(0)
            ratio.SetMarkerStyle(0) # important to remove dots at y = 1
            if noErrorRatioDen:
                ratio.Draw("HIST same")
            else:
                ratio.SetFillColor(ROOT.kGray+1)
                ratio.SetFillStyle(1001)
                ratio.Draw("E2same")

            ratios = []
            newymin=0
            newymax=0
            for i,h in enumerate(hnums):
                ratios.append(h.Clone("ratio_"+str(i+1)))
                ratios[-1].Divide(den_noerr)
                #ratios[-1].SetLineColor(h.GetLineColor())
                #ratios[-1].SetMarkerSize(0)
                #ratios[-1].SetMarkerStyle(0)
                #ratios[-1].SetFillColor(0)
                if h.GetFillColor():
                    ratios[-1].Draw("E2 SAME")
                else:
                    ratios[-1].Draw("HE SAME" if drawErrorAll else "HIST SAME")
            
            newymin, newymax = getMinMaxMultiHisto(ratios, excludeEmpty=True, sumError=False, 
                                                   excludeUnderflow=True, excludeOverflow=True)
            if newymin == newymax:
                newymin *= 0.99
                newymax *= 1.01
            newdiff = newymax - newymin
            #print(f"newdiff = {newdiff}")
            newymin = max(0, newymin - 0.1 * newdiff)
            newymax = newymax + 0.1 * newdiff
            #print(newymin, newymax)
            if not setRatioYAxisRangeFromUser:
                logger.debug(f"drawNTH1(): setting y axis in ratio panel to this range: {newymin}, {newymax}")
                frame.GetYaxis().SetRangeUser(newymin, newymax)
            pad2.RedrawAxis("sameaxis")

        line = ROOT.TF1("horiz_line","1",ratio.GetXaxis().GetBinLowEdge(1),ratio.GetXaxis().GetBinLowEdge(ratio.GetNbinsX()+1))
        line.SetLineColor(ROOT.kBlack)
        line.SetLineWidth(1)
        line.Draw("Lsame")

        if drawLineLowerPanel:
            legEntry,yline = drawLineLowerPanel.split('::')
            line2 = ROOT.TF1("horiz_line_2",str(1+float(yline)),ratio.GetXaxis().GetBinLowEdge(1),ratio.GetXaxis().GetBinLowEdge(ratio.GetNbinsX()+1))
            line3 = ROOT.TF1("horiz_line_3",str(1-float(yline)),ratio.GetXaxis().GetBinLowEdge(1),ratio.GetXaxis().GetBinLowEdge(ratio.GetNbinsX()+1))
            line2.SetLineColor(ROOT.kBlue)
            line2.SetLineWidth(1)
            line2.Draw("Lsame")
            line3.SetLineColor(ROOT.kBlue)
            line3.SetLineWidth(1)
            line3.Draw("Lsame")
            x1leg2 = 0.2 if leftMargin > 0.1 else 0.07
            x2leg2 = 0.5 if leftMargin > 0.1 else 0.27
            y1leg2 = 0.25 if leftMargin > 0.1 else 0.3
            y2leg2 = 0.35 if leftMargin > 0.1 else 0.35
            leg2 = ROOT.TLegend(x1leg2, y1leg2, x2leg2, y2leg2)
            leg2.SetFillColor(0)
            leg2.SetFillStyle(0)
            leg2.SetBorderSize(0)
            leg2.AddEntry(line2,legEntry,"L")
            leg2.Draw("same")

        
        pad2.RedrawAxis("sameaxis")


    if draw_both0_noLog1_onlyLog2 != 2:
        canvas.SaveAs(outdir + canvasName + ".png")
        canvas.SaveAs(outdir + canvasName + ".pdf")

    if draw_both0_noLog1_onlyLog2 != 1:        
        if yAxisName == "a.u.": 
            h1.GetYaxis().SetRangeUser(max(0.0001,h1.GetMinimum()*0.8),h1.GetMaximum()*100)
        else:
            h1.GetYaxis().SetRangeUser(max(0.001,h1.GetMinimum()*0.8),h1.GetMaximum()*100)
        canvas.SetLogy()
        canvas.SaveAs(outdir + canvasName + "_logY.png")
        canvas.SaveAs(outdir + canvasName + "_logY.pdf")
        canvas.SetLogy(0)


################################################################


def drawDataAndMC(h1, h2,
                  labelXtmp="xaxis", labelYtmp="yaxis",
                  canvasName="default", outdir="./",
                  draw_both0_noLog1_onlyLog2=0,                  
                  leftMargin=0.15,
                  rightMargin=0.04,
                  rebinFactorX=0,
                  labelRatioTmp="Data/pred.::0.5,1.5",
                  drawStatBox=False,
                  #legendCoords="0.15,0.35,0.8,0.9",  # x1,x2,y1,y2
                  legendCoords="0.15,0.65,0.85,0.9;3",  # for unrolled use 1 line # x1,x2,y1,y2
                  canvasSize="600,700",  # use X,Y to pass X and Y size     
                  lowerPanelHeight=0.3,  # number from 0 to 1, 0.3 means 30% of space taken by lower panel. 0 means do not draw lower panel with relative error
                  #drawLineLowerPanel="lumi. uncertainty::0.025" # if not empty, draw band at 1+ number after ::, and add legend with title
                  #drawLineLowerPanel="", # if not empty, draw band at 1+ number after ::, and add legend with title
                  passCanvas=None,
                  lumi=None,
                  drawVertLines="", # "12,36": format --> N of sections (e.g: 12 pt bins), and N of bins in each section (e.g. 36 eta bins), assuming uniform bin width
                  textForLines=[],                       
                  moreText="",
                  moreTextLatex="",
                  invertRatio = False,  # make expected over observed if True
                  histMCpartialUnc = None,
                  histMCpartialUncLegEntry = "",
                  useDifferenceInLowerPanel = False,
                  noLegendLowerPanel = False,
                  legendEntries = [],
                  drawLumiLatex=False
                  ):

    # moreText is used to pass some text to write somewhere (TPaveText is used)
    # e.g.  "stuff::x1,y1,x2,y2"  where xi and yi are the coordinates for the text
    # one can add more lines using the ";" key. FOr example, "stuff1;stuff2::x1,y1,x2,y2"
    # the coordinates should be defined taking into account how many lines will be drawn
    # if the coordinates are not passed (no "::"), then default ones are used, but this might not be satisfactory

    # moreTextLatex is similar, but used TLatex, and the four coordinates are x1,y1,ypass,textsize
    # where x1 and y1 are the coordinates the first line, and ypass is how much below y1 the second line is (and so on for following lines)

    # h1 is data, h2 in MC

    if (rebinFactorX): 
        if isinstance(rebinFactorX, int): 
            h1.Rebin(rebinFactorX)
            h2.Rebin(rebinFactorX)
        # case in which rebinFactorX is a list of bin edges
        else:   
            h1.Rebin(len(rebinFactorX)-1,"",array('d',rebinFactorX)) 
            h2.Rebin(len(rebinFactorX)-1,"",array('d',rebinFactorX)) 

    xAxisName,setXAxisRangeFromUser,xmin,xmax = getAxisRangeFromUser(labelXtmp)
    yAxisName,setYAxisRangeFromUser,ymin,ymax = getAxisRangeFromUser(labelYtmp)
    yRatioAxisName,setRatioYAxisRangeFromUser,yminRatio,ymaxRatio = getAxisRangeFromUser(labelRatioTmp)

    yAxisTitleOffset = 1.45 if leftMargin > 0.1 else 0.6

    addStringToEnd(outdir,"/",notAddIfEndswithMatch=True)
    createPlotDirAndCopyPhp(outdir)

    cw,ch = canvasSize.split(',')
    #canvas = ROOT.TCanvas("canvas",h2D.GetTitle() if plotLabel == "ForceTitle" else "",700,625)
    canvas = passCanvas if passCanvas != None else ROOT.TCanvas("canvas","",int(cw),int(ch))
    canvas.SetTickx(1)
    canvas.SetTicky(1)
    canvas.cd()
    canvas.SetLeftMargin(leftMargin)
    canvas.SetRightMargin(rightMargin)
    canvas.cd()

    pad2 = 0
    if lowerPanelHeight: 
        canvas.SetBottomMargin(lowerPanelHeight)
        pad2 = ROOT.TPad("pad2","pad2",0,0.,1,0.9)
        pad2.SetTopMargin(1-lowerPanelHeight)
        pad2.SetRightMargin(rightMargin)
        pad2.SetLeftMargin(leftMargin)
        pad2.SetFillColor(0)
        pad2.SetGridy(1)
        pad2.SetFillStyle(0)


    frame = h1.Clone("frame")
    frame.GetXaxis().SetLabelSize(0.04)
    frame.SetStats(0)

    h1.SetLineColor(ROOT.kBlack)
    h1.SetMarkerColor(ROOT.kBlack)
    h1.SetMarkerStyle(20)
    h1.SetMarkerSize(1)

    #ymax = max(ymax, max(h1.GetBinContent(i)+h1.GetBinError(i) for i in range(1,h1.GetNbinsX()+1)))
    # if min and max were not set, set them based on histogram content
    if ymin == ymax == 0.0:
        ymin,ymax = getMinMaxHisto(h1,excludeEmpty=True,sumError=True)            
        ymin *= 0.9
        ymax *= (1.1 if leftMargin > 0.1 else 2.0)
        if ymin < 0: ymin = 0
        #print "drawSingleTH1() >>> Histo: %s     minY,maxY = %.2f, %.2f" % (h1.GetName(),ymin,ymax)

    # print "#### WARNING ####"
    # print "Hardcoding ymin = 0 in function drawDataAndMC(): change it if it is not what you need"
    # print "#################"
    # ymin = 0 # hardcoded

    if lowerPanelHeight:
        h1.GetXaxis().SetLabelSize(0)
        h1.GetXaxis().SetTitle("")  
    else:
        h1.GetXaxis().SetTitle(xAxisName)
        h1.GetXaxis().SetTitleOffset(1.2)
        h1.GetXaxis().SetTitleSize(0.05)
        h1.GetXaxis().SetLabelSize(0.04)
    h1.GetYaxis().SetTitle(yAxisName)
    h1.GetYaxis().SetTitleOffset(yAxisTitleOffset)
    h1.GetYaxis().SetTitleSize(0.05)
    h1.GetYaxis().SetLabelSize(0.04)
    h1.GetYaxis().SetRangeUser(ymin, ymax)    
    h1.GetYaxis().SetTickSize(0.01)
    if setXAxisRangeFromUser: h1.GetXaxis().SetRangeUser(xmin,xmax)
    h1.Draw("EP")
    #h1err = h1.Clone("h1err")
    #h1err.SetFillColor(ROOT.kRed+2)
    h2.SetFillStyle(1001)  # 3001 is better than 3002 for pdf, while 3002 is perfect for png
    h2.SetLineColor(ROOT.kRed+2)  # kGreen+2
    h2.SetFillColor(ROOT.kRed+1)    # kGreen
    h2.SetLineWidth(1)
    h2.Draw("E2 SAME")
    h2line = h2.Clone("h2line")
    h2line.SetFillColor(0)
    h3 = None
    if histMCpartialUnc != None:
        h3 = histMCpartialUnc.Clone("histMCpartialUnc")
        h3.SetFillColor(ROOT.kGreen)
        h3.SetFillStyle(1001)  # 3001, 3144 , 3244, 3003
        #h3.SetFillStyle(3244)  # 3144 , 3244, 3003
        h3.Draw("E2 SAME")
        #for i in range(1,1+h3.GetNbinsX()):
        #    print "PDF band: bin %d  val +/- error = %.3f +/- %.3f" % (i, h3.GetBinContent(i),h3.GetBinError(i))
    h2line.Draw("HIST SAME")
    h1.Draw("EP SAME")

    nColumnsLeg = 1
    if ";" in legendCoords: 
        nColumnsLeg = int(legendCoords.split(";")[1])
        #if histMCpartialUnc != None: nColumnsLeg = 3
    legcoords = [float(x) for x in (legendCoords.split(";")[0]).split(',')]
    lx1,lx2,ly1,ly2 = legcoords[0],legcoords[1],legcoords[2],legcoords[3]
    if histMCpartialUnc != None:
        ly2 = ly2 + 0.5 * (ly2 - ly1) # add one more row
    leg = ROOT.TLegend(lx1,ly1,lx2,ly2)
    #leg.SetFillColor(0)
    #leg.SetFillStyle(0)
    #leg.SetBorderSize(0)
    leg.SetNColumns(nColumnsLeg)
    if len(legendEntries):
        leg.AddEntry(h1,str(legendEntries[0]),"LPE")
        leg.AddEntry(h2,str(legendEntries[1]),"LF")        
    else:
        leg.AddEntry(h1,"measured","LPE")
        leg.AddEntry(h2,"aMC@NLO","LF")
    if histMCpartialUnc != None:
        leg.AddEntry(h3,histMCpartialUncLegEntry,"LF")
    #leg.AddEntry(h1err,"Uncertainty","LF")
    leg.Draw("same")
    canvas.RedrawAxis("sameaxis")

    if drawStatBox:
        ROOT.gPad.Update()
        ROOT.gStyle.SetOptStat(1110)
        ROOT.gStyle.SetOptFit(1102)
    else:
        h1.SetStats(0)

    vertline = ROOT.TLine(36,0,36,canvas.GetUymax())
    vertline.SetLineColor(ROOT.kBlack)
    vertline.SetLineStyle(3) # 2 for denser hatches
    bintext = ROOT.TLatex()
    #bintext.SetNDC()
    bintext.SetTextSize(0.025)  # 0.03
    bintext.SetTextFont(42)
    if len(textForLines):
        bintext.SetTextAngle(45 if "#eta" in textForLines[0] else 30)

    if len(drawVertLines):
        #print "drawVertLines = " + drawVertLines
        nptBins = int(drawVertLines.split(',')[0])
        etarange = float(drawVertLines.split(',')[1])        
        offsetXaxisHist = h1.GetXaxis().GetBinLowEdge(0)
        sliceLabelOffset = 6. if "#eta" in textForLines[0] else 6.
        for i in range(1,nptBins): # do not need line at canvas borders
            #vertline.DrawLine(offsetXaxisHist+etarange*i,0,offsetXaxisHist+etarange*i,canvas.GetUymax())
            vertline.DrawLine(etarange*i-offsetXaxisHist,0,etarange*i-offsetXaxisHist,ymax)
        if len(textForLines):
            for i in range(0,len(textForLines)): # we need nptBins texts
                #texoffset = 0.1 * (4 - (i%4))
                #ytext = (1. + texoffset)*ymax/2.  
                ytext = 0.6*ymax                  
                bintext.DrawLatex(etarange*i + etarange/sliceLabelOffset, ytext, textForLines[i])

    # redraw legend, or vertical lines appear on top of it
    leg.Draw("same")

    if len(moreText):
        realtext = moreText.split("::")[0]
        x1,y1,x2,y2 = 0.7,0.8,0.9,0.9
        if "::" in moreText:
            x1,y1,x2,y2 = (float(x) for x in (moreText.split("::")[1]).split(","))
        pavetext = ROOT.TPaveText(x1,y1,x2,y2,"NB NDC")
        for tx in realtext.split(";"):
            pavetext.AddText(tx)
        pavetext.SetFillColor(0)
        pavetext.SetFillStyle(0)
        pavetext.SetBorderSize(0)
        pavetext.SetLineColor(0)
        pavetext.Draw("same")

    if len(moreTextLatex):
        realtext = moreTextLatex.split("::")[0]
        x1,y1,ypass,textsize = 0.75,0.8,0.08,0.035
        if "::" in moreTextLatex:
            x1,y1,ypass,textsize = (float(x) for x in (moreTextLatex.split("::")[1]).split(","))            
        lat = ROOT.TLatex()
        lat.SetNDC();
        lat.SetTextFont(42)        
        lat.SetTextSize(textsize)
        for itx,tx in enumerate(realtext.split(";")):
            lat.DrawLatex(x1,y1-itx*ypass,tx)

  # TPaveText *pvtxt = NULL;
  # if (yAxisName == "a.u.") {
  #   pvtxt = new TPaveText(0.6,0.6,0.95,0.7, "BR NDC")
  #   pvtxt.SetFillColor(0)
  #   pvtxt.SetFillStyle(0)
  #   pvtxt.SetBorderSize(0)
  #   pvtxt.AddText(Form("norm num/den = %.2f +/- %.2f",IntegralRatio,ratioError))
  #   pvtxt.Draw()
  # }

    setTDRStyle()
    if drawLumiLatex:
        latCMS = ROOT.TLatex()
        latCMS.SetNDC();
        latCMS.SetTextFont(42)
        latCMS.SetTextSize(0.045)
        latCMS.DrawLatex(canvas.GetLeftMargin(), 0.95, '#bf{CMS} #it{Preliminary}')
        if lumi != None: latCMS.DrawLatex(0.7, 0.95, '%s fb^{-1} (13 TeV)' % lumi)
        else:            latCMS.DrawLatex(0.8, 0.95, '(13 TeV)' % lumi)
    else:
        if lumi != None: CMS_lumi(canvas,lumi,True,False)
        else:            CMS_lumi(canvas,"",True,False)

    if lowerPanelHeight:
        pad2.Draw()
        pad2.cd()

        frame.Reset("ICES")
        if setRatioYAxisRangeFromUser: frame.GetYaxis().SetRangeUser(yminRatio,ymaxRatio)
        #else:                          
        #frame.GetYaxis().SetRangeUser(0.5,1.5)
        frame.GetYaxis().SetNdivisions(5)
        frame.GetYaxis().SetTitle(yRatioAxisName)
        frame.GetYaxis().SetTitleOffset(yAxisTitleOffset)
        frame.GetYaxis().SetTitleSize(0.05)
        frame.GetYaxis().SetLabelSize(0.04)
        frame.GetYaxis().CenterTitle()
        frame.GetXaxis().SetTitle(xAxisName)
        if setXAxisRangeFromUser: frame.GetXaxis().SetRangeUser(xmin,xmax)
        frame.GetXaxis().SetTitleOffset(1.2)
        frame.GetXaxis().SetTitleSize(0.05)

        #ratio = copy.deepcopy(h1.Clone("ratio"))
        #den_noerr = copy.deepcopy(h2.Clone("den_noerr"))
        ratio = None
        den_noerr = None
        den = None
        if invertRatio:
            ratio = h2.Clone("ratio")
            den_noerr = h1.Clone("den_noerr")
            den = h1.Clone("den")
        else:
            ratio = h1.Clone("ratio")
            den_noerr = h2.Clone("den_noerr")
            den = h2.Clone("den")

        for iBin in range (1,den_noerr.GetNbinsX()+1):
            den_noerr.SetBinError(iBin,0.)

        if useDifferenceInLowerPanel:
            ratio.Add(den_noerr,-1.0)
            den.Add(den_noerr,-1.0)
        else:
            ratio.Divide(den_noerr)
            den.Divide(den_noerr)

        if invertRatio:
            if histMCpartialUnc == None:
                ratio.SetFillColor(ROOT.kGray+1)
                ratio.SetFillStyle(1001)  # make it solid again
                ratio.SetLineColor(ROOT.kGray+3)
            else:
                ratio.SetFillColor(ROOT.kRed+1) # kGreen+1
                ratio.SetFillStyle(1001)  # 1001 to make it solid again
                ratio.SetLineColor(ROOT.kRed+2) # kGreen+2                       
                ratio.SetLineWidth(1) # make it smaller when it is drawn on top of something
        else:
            if histMCpartialUnc == None:
                den.SetFillColor(ROOT.kGray+1)
                den.SetFillStyle(1001)  # make it solid again
                den.SetLineColor(ROOT.kRed)        
        if histMCpartialUnc != None:
            h3ratio = h3.Clone("h3ratio")
            if useDifferenceInLowerPanel:
                h3ratio.Add(den_noerr,-1.0)
            else:
                h3ratio.Divide(den_noerr)
            #h3ratio.SetFillStyle(3144) # 1001 for solid, 3144 instead of 3244, to have more dense hatches
            h3ratio.SetFillStyle(1001) # 
            h3ratio.SetFillColor(ROOT.kGreen+1)  # kRed-4
            h3ratio.SetLineColor(ROOT.kGreen+2)  # kRed-4

        frame.Draw()        
        if invertRatio:
            den.SetMarkerSize(0.85)
            den.SetMarkerStyle(20) 
            #den.Draw("EPsame")    # draw after red line, not now
            ratio.Draw("E2same")
        else:    
            ratio.SetMarkerSize(0.85)
            ratio.SetMarkerStyle(20) 
            den.Draw("E2same")
            #ratio.Draw("EPsame") # draw after red line, not now

        if histMCpartialUnc != None:
            h3ratio.Draw("E2 same")
        
        line = ROOT.TF1("horiz_line","0" if useDifferenceInLowerPanel else "1",
                        ratio.GetXaxis().GetBinLowEdge(1),ratio.GetXaxis().GetBinLowEdge(ratio.GetNbinsX()+1))
        line.SetLineWidth(1)
        line.Draw("Lsame")
        if invertRatio:
            ratioline = ratio.Clone("ratioline")
            ratioline.SetFillColor(0)
            if histMCpartialUnc != None and len(drawVertLines):
                # unrolled, here the red line would be too large and given the fluctuations is could be seen as an additional band
                ratioline.SetLineColor(histMCpartialUnc.GetLineColor())
            ratioline.SetFillStyle(0)
            if histMCpartialUnc != None: ratioline.Draw("HIST same") # to draw the line inside the band for the expected
            den.Draw("EPsame")
        else: 
            ratio.Draw("EPsame")

        x1leg2 = 0.15 if leftMargin > 0.1 else 0.07
        x2leg2 = 0.55 if leftMargin > 0.1 else 0.27
        y1leg2 = 0.28 if leftMargin > 0.1 else 0.3
        y2leg2 = 0.33 if leftMargin > 0.1 else 0.35
        leg2 = ROOT.TLegend(x1leg2, y1leg2, x2leg2, y2leg2)
        #leg2 = ROOT.TLegend(0.07,0.30,0.27,0.35)
        leg2.SetFillColor(0)
        leg2.SetFillStyle(0)
        leg2.SetBorderSize(0)
        if invertRatio:
            leg2.AddEntry(ratio,"total theory uncertainty","LF")
        else:
            leg2.AddEntry(den,"total theory uncertainty","LF")
        if not noLegendLowerPanel: leg2.Draw("same")
        if histMCpartialUnc != None:
            leg2bis = ROOT.TLegend(x1leg2 + 0.45, y1leg2, x2leg2 + 0.45, y2leg2)
            leg2bis.SetFillColor(0)
            leg2bis.SetFillStyle(0)
            leg2bis.SetBorderSize(0)
            leg2bis.AddEntry(h3ratio,histMCpartialUncLegEntry,"LF")
            if not noLegendLowerPanel: leg2bis.Draw("same")

        pad2.RedrawAxis("sameaxis")


    if draw_both0_noLog1_onlyLog2 != 2:
        canvas.SaveAs(outdir + canvasName + ".png")
        canvas.SaveAs(outdir + canvasName + ".pdf")

    if draw_both0_noLog1_onlyLog2 != 1:        
        if yAxisName == "a.u.": 
            h1.GetYaxis().SetRangeUser(max(0.0001,h1.GetMinimum()*0.8),h1.GetMaximum()*100)
        else:
            h1.GetYaxis().SetRangeUser(max(0.001,h1.GetMinimum()*0.8),h1.GetMaximum()*100)
        canvas.SetLogy()
        canvas.SaveAs(outdir + canvasName + "_logY.png")
        canvas.SaveAs(outdir + canvasName + "_logY.pdf")
        canvas.SetLogy(0)
        
          
#########################################################################

def drawTH1dataMCstack(h1, thestack, 
                       labelXtmp="xaxis", labelYtmp="yaxis",
                       canvasName="default", 
                       outdir="./",
                       legend=None,
                       ratioPadYaxisNameTmp="data/MC::0.5,1.5", 
                       draw_both0_noLog1_onlyLog2=1,
                       #minFractionToBeInLegend=0.001,
                       fillStyle=3001,
                       leftMargin=0.16,
                       rightMargin=0.05,
                       nContours=50,
                       palette=55,
                       canvasSize="700,625",
                       passCanvas=None,
                       normalizeMCToData=False,
                       hErrStack=None,   # might need to define an error on the stack in a special way
                       lumi=None,
                       yRangeScaleFactor=1.5, # if range of y axis is not explicitely passed, use (max-min) times this value
                       wideCanvas=False,
                       drawVertLines="", # "12,36": format --> N of sections (e.g: 12 pt bins), and N of bins in each section (e.g. 36 eta bins), assuming uniform bin width
                       textForLines=[], 
                       etaptbinning=[],
                       drawLumiLatex=False,
                       skipLumi=False,
                       xcmsText=-1, # customize x when using drawLumiLatex
                       noLegendRatio=False,
                       textSize=0.035,
                       textAngle=0.10,
                       textYheightOffset=0.55, # text printed at y = maxY of cancas times this constant
                       noRatioPanel=False
):

    # if normalizing stack to same area as data, we need to modify the stack
    # however, the stack might be used outside the function. In order to avoid any changes in the stack, it is copied here just for the plot

    ROOT.TH1.SetDefaultSumw2()
    adjustSettings_CMS_lumi()
    
    ROOT.gStyle.SetPalette(palette)  # 55:raibow palette ; 57: kBird (blue to yellow, default) ; 107 kVisibleSpectrum ; 77 kDarkRainBow 
    ROOT.gStyle.SetNumberContours(nContours) # default is 20 

    labelX,setXAxisRangeFromUser,xmin,xmax = getAxisRangeFromUser(labelXtmp)
    labelY,setYAxisRangeFromUser,ymin,ymax = getAxisRangeFromUser(labelYtmp)
    labelRatioY,setRatioYAxisRangeFromUser,yminRatio,ymaxRatio = getAxisRangeFromUser(ratioPadYaxisNameTmp)
    
    cw,ch = canvasSize.split(',')
    #canvas = ROOT.TCanvas("canvas",h2D.GetTitle() if plotLabel == "ForceTitle" else "",700,625)    
    canvas = passCanvas if passCanvas != None else ROOT.TCanvas("canvas","",int(cw),int(ch))
    canvas.SetTickx(1)
    canvas.SetTicky(1)
    canvas.SetLeftMargin(leftMargin)
    canvas.SetRightMargin(rightMargin)
    canvas.SetBottomMargin(0.12 if noRatioPanel else 0.3)
    canvas.cd()

    addStringToEnd(outdir,"/",notAddIfEndswithMatch=True)
    createPlotDirAndCopyPhp(outdir)

    dataNorm = h1.Integral()
    stackNorm = 0.0

    #dummystack = thestack
    dummystack = ROOT.THStack("dummy_{sn}".format(sn=thestack.GetName()),"")
    for hist in thestack.GetHists():        
        stackNorm += hist.Integral()
    for hist in thestack.GetHists():        
        hnew = copy.deepcopy(hist.Clone("dummy_{hn}".format(hn=hist.GetName())))
        if normalizeMCToData:
            hnew.Scale(dataNorm/stackNorm)
        dummystack.Add(hnew)    
        
    stackCopy = dummystack.GetStack().Last() # used to make ratioplot without affecting the plot and setting maximum
    # the error of the last should be the sum in quadrature of the errors of single components, as the Last is the sum of them
    # however, better to recreate it
    stackErr = stackCopy
    if hErrStack != None:
        stackErr = copy.deepcopy(hErrStack.Clone("stackErr"))

    # logger.info("drawTH1dataMCstack():  integral(data):  " + str(h1.Integral()))
    # logger.info("drawTH1dataMCstack():  integral(stack): " + str(stackCopy.Integral()))
    # logger.info("drawTH1dataMCstack():  integral(herr):  " + str(stackErr.Integral()))

    h1.SetStats(0)
    titleBackup = h1.GetTitle()
    h1.SetTitle("")

    pad2 = ROOT.TPad("pad2","pad2",0,0.,1,0.9)
    pad2.SetTopMargin(0.7)
    pad2.SetRightMargin(rightMargin)
    pad2.SetLeftMargin(leftMargin)
    pad2.SetFillColor(0)
    pad2.SetGridy(1)
    pad2.SetFillStyle(0)
    
    frame = h1.Clone("frame")
    frame.GetXaxis().SetLabelSize(0.04)
    frame.SetStats(0)

    h1.SetLineColor(ROOT.kBlack)
    h1.SetMarkerColor(ROOT.kBlack)
    h1.SetMarkerStyle(20)
    h1.SetMarkerSize(0.5 if (wideCanvas or leftMargin < 0.1) else 1.0)

    if not noRatioPanel:
        h1.GetXaxis().SetLabelSize(0)
        h1.GetXaxis().SetTitle("")
    else:
        h1.GetXaxis().SetTitle(labelX)
        h1.GetXaxis().SetTitleOffset(1.1)
        h1.GetXaxis().SetTitleSize(0.05)
    h1.GetYaxis().SetTitle(labelY)
    h1.GetYaxis().SetTitleOffset(0.5 if wideCanvas else 1.5)
    h1.GetYaxis().SetTitleSize(0.05)
    h1.GetYaxis().SetLabelSize(0.04)
    h1.GetYaxis().SetTickSize(0.01)
    ymaxBackup = 0
    if setYAxisRangeFromUser: 
        ymaxBackup = ymax
        h1.GetYaxis().SetRangeUser(ymin,ymax)
    else:
        ymaxBackup = max(h1.GetBinContent(h1.GetMaximumBin()),stackCopy.GetBinContent(stackCopy.GetMaximumBin())) * yRangeScaleFactor
        h1.GetYaxis().SetRangeUser(0.0, ymaxBackup)
    if setXAxisRangeFromUser: h1.GetXaxis().SetRangeUser(xmin,xmax)
    h1.Draw("EP")
    dummystack.Draw("HIST SAME")
    h1.Draw("EP SAME")

    vertline = ROOT.TLine(36,0,36,canvas.GetUymax())
    vertline.SetLineColor(ROOT.kBlack)
    vertline.SetLineStyle(3) # 2 larger hatches
    bintext = ROOT.TLatex()
    #bintext.SetNDC()
    bintext.SetTextSize(textSize)
    bintext.SetTextFont(42)
    bintext.SetTextAngle(textAngle)

    if len(drawVertLines):
        nptBins = int(drawVertLines.split(',')[0])
        etarange = float(drawVertLines.split(',')[1])        
        for i in range(1,nptBins): # do not need line at canvas borders
            #vertline.DrawLine(etarange*i,0,etarange*i,canvas.GetUymax())
            vertline.DrawLine(etarange*i,0,etarange*i,ymaxBackup)
        if len(textForLines):
            offsetText = etarange / (6. if len(textForLines) > 15 else 4.)
            for i in range(0,len(textForLines)): # we need nptBins texts
                bintext.DrawLatex(etarange*i + offsetText, textYheightOffset*ymaxBackup, textForLines[i])

    # legend.SetFillColor(0)
    # legend.SetFillStyle(0)
    # legend.SetBorderSize(0)
    legend.Draw("same")
    canvas.RedrawAxis("sameaxis")

    reduceSize = False
    offset = 0
    # check whether the Y axis will have exponential notatio
    if h1.GetBinContent(h1.GetMaximumBin()) > 1000000:
        reduceSize = True
        offset = 0.1
    if wideCanvas: 
        offset = 0.1
        latCMS = ROOT.TLatex()
        latCMS.SetNDC();
        latCMS.SetTextFont(42)
        latCMS.SetTextSize(0.045)
        latCMS.DrawLatex(leftMargin, 0.95, '#bf{CMS} #it{Preliminary}')
        if lumi != None: latCMS.DrawLatex(0.91, 0.95, '%s fb^{-1} (13 TeV)' % lumi)
        else:            latCMS.DrawLatex(0.96, 0.95, '(13 TeV)')
    else:    
        if not skipLumi:
            if not drawLumiLatex:
                if lumi != None: CMS_lumi(canvas,lumi,True,False)
                else:            CMS_lumi(canvas,"",True,False)
            else:
                latCMS = ROOT.TLatex()
                latCMS.SetNDC();
                latCMS.SetTextFont(42)
                latCMS.SetTextSize(0.035 if lumi != None else 0.04)
                latCMS.DrawLatex(leftMargin if xcmsText < 0 else xcmsText, 0.95, '#bf{CMS} #it{Preliminary}')
                if lumi != None: latCMS.DrawLatex(0.7, 0.95, '%s fb^{-1} (13 TeV)' % lumi)
                else:            latCMS.DrawLatex(0.8, 0.95, '(13 TeV)')

    setTDRStyle()

    if not noRatioPanel:
        pad2.Draw();
        pad2.cd();

        frame.Reset("ICES")
        if setRatioYAxisRangeFromUser: frame.GetYaxis().SetRangeUser(yminRatio,ymaxRatio)
        #else:                          
        #frame.GetYaxis().SetRangeUser(0.5,1.5)
        frame.GetYaxis().SetNdivisions(5)
        frame.GetYaxis().SetTitle(labelRatioY)
        frame.GetYaxis().SetTitleOffset(0.5 if wideCanvas else 1.5)
        frame.GetYaxis().SetTitleSize(0.05)
        frame.GetYaxis().SetLabelSize(0.04)
        frame.GetYaxis().CenterTitle()
        frame.GetXaxis().SetTitle(labelX)
        if setXAxisRangeFromUser: frame.GetXaxis().SetRangeUser(xmin,xmax)
        frame.GetXaxis().SetTitleOffset(1.2)
        frame.GetXaxis().SetTitleSize(0.05)

        #ratio = copy.deepcopy(h1.Clone("ratio"))
        #den_noerr = copy.deepcopy(stackErr.Clone("den_noerr"))
        ratio = h1.Clone("ratio")
        den_noerr = stackErr.Clone("den_noerr")
        den = stackErr.Clone("den")
        for iBin in range (1,den_noerr.GetNbinsX()+1):
            den_noerr.SetBinError(iBin,0.)

        ratio.Divide(den_noerr)
        den.Divide(den_noerr)
        den.SetFillColor(ROOT.kCyan)
        den.SetFillStyle(1001)  # make it solid again
        den.SetMarkerSize(0)
        #den.SetLineColor(ROOT.kRed)
        frame.Draw()        
        ratio.SetMarkerSize(0.6 if (wideCanvas or leftMargin < 0.1) else 0.85)
        ratio.SetMarkerStyle(20) 
        den.Draw("E2same")
        ratio.Draw("EPsame")

        # if not "unrolled_" in canvasName:
        #     for i in range(1,1+ratio.GetNbinsX()):
        #         print "Error data bin {bin}: {val}".format(bin=i,val=ratio.GetBinError(i))

        line = ROOT.TF1("horiz_line","1",ratio.GetXaxis().GetBinLowEdge(1),ratio.GetXaxis().GetBinLowEdge(ratio.GetNbinsX()+1))
        line.SetLineColor(ROOT.kRed)
        line.SetLineWidth(1)  # 1, not 2, which is too wide for canvas with large width
        line.Draw("Lsame")

        leg2 = ROOT.TLegend(0.2,0.25,0.4,0.30)
        leg2.SetFillColor(0)
        leg2.SetFillStyle(0)
        leg2.SetBorderSize(0)
        leg2.AddEntry(den,"tot. unc. exp.","LF")
        if not noLegendRatio:
            leg2.Draw("same")

        pad2.RedrawAxis("sameaxis")


    if draw_both0_noLog1_onlyLog2 != 2:
        canvas.SaveAs(outdir + canvasName + ".png")
        canvas.SaveAs(outdir + canvasName + ".pdf")

    if draw_both0_noLog1_onlyLog2 != 1:        
        if labelY == "a.u.": 
            h1.GetYaxis().SetRangeUser(max(0.0001,h1.GetMinimum()*0.8),h1.GetBinContent(h1.GetMaximumBin())*100)
        else:
            h1.GetYaxis().SetRangeUser(max(0.1,h1.GetMinimum()*0.8),h1.GetBinContent(h1.GetMaximumBin())*1000)
            canvas.SetLogy()
            canvas.SaveAs(outdir + canvasName + "_logY.png")
            canvas.SaveAs(outdir + canvasName + "_logY.pdf")
            canvas.SetLogy(0)
            

    h1.SetTitle(titleBackup)
  
    if "unrolled" in canvasName:

        _canvas_pull = ROOT.TCanvas("_canvas_pull","",800,800)
        _canvas_pull.SetTickx(1)
        _canvas_pull.SetTicky(1)
        _canvas_pull.SetGridx(1)
        _canvas_pull.SetGridy(1)
        _canvas_pull.SetTopMargin(0.1)
        _canvas_pull.SetBottomMargin(0.12)
        _canvas_pull.SetLeftMargin(0.12)
        _canvas_pull.SetRightMargin(0.04)
        # make pulls
        pulltitle = "unrolled {ch} {pf}".format(ch="plus" if "plus" in canvasName else "minus", pf="postfit" if "postfit" in canvasName else "prefit")
        hpull = ROOT.TH1D("hpull_"+canvasName,pulltitle,51,-5,5)    
        hpull.SetStats(1)
        _canvas_pull.cd()
        for i in range (1,ratio.GetNbinsX()+1):
            errTotDen = ratio.GetBinError(i)*ratio.GetBinError(i) + den.GetBinError(i)*den.GetBinError(i)            
            if errTotDen > 0.0:
                hpull.Fill((ratio.GetBinContent(i)-1)/math.sqrt(errTotDen))
        hpull.Draw("HIST")
        hpull.GetXaxis().SetTitle("pull")
        hpull.GetYaxis().SetTitle("Events")
        hpull.SetLineColor(ROOT.kBlack)
        hpull.SetLineWidth(2)
        ROOT.gStyle.SetOptTitle(1)                
        ROOT.gStyle.SetOptStat(111110)
        ROOT.gStyle.SetOptFit(1102)
        _canvas_pull.RedrawAxis("sameaxis")
        _canvas_pull.SaveAs(outdir + "pull_" + canvasName + ".png")    
        _canvas_pull.SaveAs(outdir + "pull_" + canvasName + ".pdf")

        if len(etaptbinning):
            etaThreshold = 1.2
            _canvas_pull.SetGridx(0)
            _canvas_pull.SetGridy(0)            
            _canvas_pull.SetRightMargin(0.16)            
            h2pull = ROOT.TH2D("h2pull_"+canvasName, pulltitle.replace("unrolled","rolled") ,
                               etaptbinning[0], array('d', etaptbinning[1]), etaptbinning[2], array('d', etaptbinning[3]))
            hpull.Reset("ICESM")  # will use again for pulls in EE only
            hpullEEp = hpull.Clone("hpullEEp")
            hpullEEm = hpull.Clone("hpullEEm")
            for i in range (1,ratio.GetNbinsX()+1):
                etabin = int((i-1)%etaptbinning[0] + 1)
                ptbin = int((i-1)/etaptbinning[0] + 1)
                errTotDen = ratio.GetBinError(i)*ratio.GetBinError(i) + den.GetBinError(i)*den.GetBinError(i)            
                if errTotDen > 0.0:
                    pullVal = (ratio.GetBinContent(i)-1)/math.sqrt(errTotDen)
                    h2pull.SetBinContent(etabin,ptbin, pullVal)
                    if abs(etaptbinning[1][etabin]) >= etaThreshold: 
                        hpull.Fill(pullVal)
                        if etaptbinning[1][etabin] > 0: hpullEEp.Fill(pullVal)
                        else:                           hpullEEm.Fill(pullVal)
            h2pull.GetXaxis().SetTitle("%s #eta" % "muon" if "muon" in labelX else "electron")
            h2pull.GetYaxis().SetTitle("%s p_{T}" % "muon" if "muon" in labelX else "electron")
            h2pull.GetZaxis().SetTitle("pull")
            h2pull.SetStats(0)
            h2pull.GetZaxis().SetRangeUser(-3,3)
            h2pull.Draw("COLZ")
            _canvas_pull.RedrawAxis("sameaxis")
            _canvas_pull.SaveAs(outdir + "pull2D_" + canvasName + ".png")
            _canvas_pull.SaveAs(outdir + "pull2D_" + canvasName + ".pdf")

            # add pulls for EE only
            _canvas_pull.SetTickx(1)
            _canvas_pull.SetTicky(1)
            _canvas_pull.SetGridx(1)
            _canvas_pull.SetGridy(1)
            _canvas_pull.SetTopMargin(0.1)
            _canvas_pull.SetBottomMargin(0.12)
            _canvas_pull.SetLeftMargin(0.12)
            _canvas_pull.SetRightMargin(0.04)
            hpull.Draw("HIST")
            hpull.GetXaxis().SetTitle("pull (only |#eta| >= %.1f)" % etaThreshold)
            hpull.GetYaxis().SetTitle("Events")
            hpullEEp.SetLineWidth(2)
            hpullEEp.SetLineColor(ROOT.kOrange+2)
            hpullEEp.SetFillColor(ROOT.kOrange+1)
            hpullEEp.SetFillStyle(3001)
            hpullEEm.SetLineWidth(2)
            hpullEEm.SetLineColor(ROOT.kBlue+2)
            hpullEEm.SetFillColor(ROOT.kAzure+1)
            hpullEEm.SetFillStyle(3244)    
            hpullEEp.Draw("HIST SAME")
            hpullEEm.Draw("HIST SAME")
            hpull.Draw("HIST SAME")
            legEE = ROOT.TLegend(0.15,0.5,0.45,0.8)
            legEE.SetFillStyle(0)
            legEE.SetFillColor(0)
            legEE.SetBorderSize(0)
            legEE.AddEntry(hpull,    "|#eta| > %.1f" % etaThreshold, "L")
            legEE.AddEntry(hpullEEp, "#eta > %.1f"   % etaThreshold,  "LF")
            legEE.AddEntry(hpullEEm, "#eta < -%.1f"  % etaThreshold, "LF")
            legEE.Draw("same")
            _canvas_pull.RedrawAxis("sameaxis")
            _canvas_pull.SaveAs(outdir + "pull_onlyEndcap_" + canvasName + ".png")    
            _canvas_pull.SaveAs(outdir + "pull_onlyEndcap_" + canvasName + ".pdf")
                      
################################################################

def drawCheckTheoryBand(h1, h2, h3,
                        labelXtmp="xaxis", labelYtmp="yaxis",
                        canvasName="default", outdir="./",
                        #rebinFactorX=0,
                        draw_both0_noLog1_onlyLog2=0,                  
                        leftMargin=0.15,
                        rightMargin=0.04,
                        labelRatioTmp="rel. unc..::0.95,1.05",
                        drawStatBox=False,
                        #legendCoords="0.15,0.35,0.8,0.9",  # x1,x2,y1,y2
                        legendCoords="0.15,0.85,0.85,0.9;3",  # for unrolled use 1 line # x1,x2,y1,y2
                        canvasSize="1000,700",  # use X,Y to pass X and Y size     
                        lowerPanelHeight = 0.0,  # number from 0 to 1, 0.3 means 30% of space taken by lower panel. 0 means do not draw lower panel with relative error
                        passCanvas=None,
                        lumi=None,
                        drawVertLines="", # "12,36": format --> N of sections (e.g: 12 pt bins), and N of bins in each section (e.g. 36 eta bins), assuming uniform bin width
                        textForLines=[],                       
                        moreText="",
                        moreTextLatex="",
                        invertRatio = False,  # make expected over observed if True
                        useDifferenceInLowerPanel = False,
                        noLegendLowerPanel = False,
                        legendEntries = []
                    ):

    # moreText is used to pass some text to write somewhere (TPaveText is used)
    # e.g.  "stuff::x1,y1,x2,y2"  where xi and yi are the coordinates for the text
    # one can add more lines using the ";" key. FOr example, "stuff1;stuff2::x1,y1,x2,y2"
    # the coordinates should be defined taking into account how many lines will be drawn
    # if the coordinates are not passed (no "::"), then default ones are used, but this might not be satisfactory

    # moreTextLatex is similar, but used TLatex, and the four coordinates are x1,y1,ypass,textsize
    # where x1 and y1 are the coordinates the first line, and ypass is how much below y1 the second line is (and so on for following lines)

    # h1 is data, h2 in MC

    #if (rebinFactorX): 
    #    if isinstance(rebinFactorX, int): h1.Rebin(rebinFactorX)
    #    # case in which rebinFactorX is a list of bin edges
    #    else:                             h1.Rebin(len(rebinFactorX)-1,"",array('d',rebinFactorX)) 

    xAxisName,setXAxisRangeFromUser,xmin,xmax = getAxisRangeFromUser(labelXtmp)
    yAxisName,setYAxisRangeFromUser,ymin,ymax = getAxisRangeFromUser(labelYtmp)
    yRatioAxisName,setRatioYAxisRangeFromUser,yminRatio,ymaxRatio = getAxisRangeFromUser(labelRatioTmp)

    yAxisTitleOffset = 1.45 if leftMargin > 0.1 else 0.6

    addStringToEnd(outdir,"/",notAddIfEndswithMatch=True)
    createPlotDirAndCopyPhp(outdir)

    cw,ch = canvasSize.split(',')
    #canvas = ROOT.TCanvas("canvas",h2D.GetTitle() if plotLabel == "ForceTitle" else "",700,625)
    canvas = passCanvas if passCanvas != None else ROOT.TCanvas("canvas","",int(cw),int(ch))
    canvas.SetTickx(1)
    canvas.SetTicky(1)
    canvas.cd()
    canvas.SetLeftMargin(leftMargin)
    canvas.SetRightMargin(rightMargin)
    canvas.cd()

    pad2 = 0
    if lowerPanelHeight: 
        canvas.SetBottomMargin(lowerPanelHeight)
        pad2 = ROOT.TPad("pad2","pad2",0,0.,1,0.9)
        pad2.SetTopMargin(1-lowerPanelHeight)
        pad2.SetRightMargin(rightMargin)
        pad2.SetLeftMargin(leftMargin)
        pad2.SetFillColor(0)
        pad2.SetGridy(1)
        pad2.SetFillStyle(0)


    frame = h1.Clone("frame")
    frame.GetXaxis().SetLabelSize(0.04)
    frame.SetStats(0)

    #ymax = max(ymax, max(h1.GetBinContent(i)+h1.GetBinError(i) for i in range(1,h1.GetNbinsX()+1)))
    # if min and max were not set, set them based on histogram content
    if ymin == ymax == 0.0:
        ymin,ymax = getMinMaxHisto(h1,excludeEmpty=True,sumError=True)            
        ymin *= 0.9
        ymax *= (1.1 if leftMargin > 0.1 else 2.0)
        if ymin < 0: ymin = 0
        #print "drawSingleTH1() >>> Histo: %s     minY,maxY = %.2f, %.2f" % (h1.GetName(),ymin,ymax)

    # print "#### WARNING ####"
    # print "Hardcoding ymin = 0 in function drawDataAndMC(): change it if it is not what you need"
    # print "#################"
    # ymin = 0 # hardcoded

    if lowerPanelHeight:
        h1.GetXaxis().SetLabelSize(0)
        h1.GetXaxis().SetTitle("")  
    else:
        h1.GetXaxis().SetTitle(xAxisName)
        h1.GetXaxis().SetTitleOffset(1.2)
        h1.GetXaxis().SetTitleSize(0.05)
        h1.GetXaxis().SetLabelSize(0.04)
    h1.GetYaxis().SetTitle(yAxisName)
    h1.GetYaxis().SetTitleOffset(yAxisTitleOffset)
    h1.GetYaxis().SetTitleSize(0.05)
    h1.GetYaxis().SetLabelSize(0.04)
    h1.GetYaxis().SetRangeUser(ymin, ymax)    
    h1.GetYaxis().SetTickSize(0.01)
    if setXAxisRangeFromUser: h1.GetXaxis().SetRangeUser(xmin,xmax)


    ratio = None
    den_noerr = None
    den = None
    ratio1 = h1.Clone("ratio1")
    ratio2 = h2.Clone("ratio2")
    ratio3 = h3.Clone("ratio3")
    den_noerr = h1.Clone("den_noerr")

    for iBin in range (1,den_noerr.GetNbinsX()+1):
        den_noerr.SetBinError(iBin,0.)

    h2IsGraph = False
    h3IsGraph = False
    ratio1.Divide(den_noerr)

    if ratio2.InheritsFrom("TH1"):
        ratio2.Divide(den_noerr)
    else:
        h2IsGraph = True
        for i in range(den_noerr.GetNbinsX()):
            ratio2.SetPoint(i, den_noerr.GetBinCenter(i+1), ratio2.Eval(den_noerr.GetBinCenter(i+1)) / h1.GetBinContent(i+1) )
            ratio2.SetPointEYhigh(i, ratio2.GetErrorYhigh(i) / h1.GetBinContent(i+1))
            ratio2.SetPointEYlow( i, ratio2.GetErrorYlow(i)  / h1.GetBinContent(i+1))

    if ratio3.InheritsFrom("TH1"):
        ratio3.Divide(den_noerr)
    else:
        h3IsGraph = True
        for i in range(den_noerr.GetNbinsX()):
            ratio3.SetPoint(i, den_noerr.GetBinCenter(i+1), ratio3.Eval(den_noerr.GetBinCenter(i+1)) / h1.GetBinContent(i+1) )
            ratio3.SetPointEYhigh(i, ratio3.GetErrorYhigh(i) / h1.GetBinContent(i+1))
            ratio3.SetPointEYlow( i, ratio3.GetErrorYlow(i)  / h1.GetBinContent(i+1))


    ratio1.SetFillColor(ROOT.kGreen) # kGreen+1
    ratio1.SetFillStyle(3001)  # 1001 to make it solid again
    ratio1.SetLineColor(ROOT.kGreen+1) # kGreen+2                       
    ratio1.SetLineWidth(1) # make it smaller when it is drawn on top of something
    ratio2.SetFillColor(ROOT.kRed+1) # kGreen+1
    ratio2.SetFillStyle(3244)  # 1001 to make it solid again
    ratio2.SetLineColor(ROOT.kRed+2) # kGreen+2                       
    ratio2.SetLineWidth(1) # make it smaller when it is drawn on top of something
    ratio3.SetLineColor(ROOT.kBlack)
    ratio3.SetMarkerColor(ROOT.kBlack)
    ratio3.SetMarkerStyle(20)
    ratio3.SetMarkerSize(1)
    
    ratio1.Draw("E2")
    ratio2.Draw("F2 SAME" if h2IsGraph else "E2 SAME")
    ratio3.Draw("EP SAME")
    line = ROOT.TF1("horiz_line","1",
                    ratio1.GetXaxis().GetBinLowEdge(1),ratio1.GetXaxis().GetBinLowEdge(ratio1.GetNbinsX()+1))
    line.SetLineColor(ROOT.kRed+3)
    line.SetLineWidth(1)
    line.Draw("Lsame")

    nColumnsLeg = 1
    if ";" in legendCoords: 
        nColumnsLeg = int(legendCoords.split(";")[1])
    legcoords = [float(x) for x in (legendCoords.split(";")[0]).split(',')]
    lx1,lx2,ly1,ly2 = legcoords[0],legcoords[1],legcoords[2],legcoords[3]
    leg = ROOT.TLegend(lx1,ly1,lx2,ly2)
    #leg.SetFillColor(0)
    #leg.SetFillStyle(0)
    #leg.SetBorderSize(0)
    leg.SetNColumns(3)
    leg.AddEntry(ratio1,"PDFs","LF")
    leg.AddEntry(ratio2,"#alpha_{S}","LF")        
    leg.AddEntry(ratio3,"QCD scales","LPE")        
    leg.Draw("same")
    canvas.RedrawAxis("sameaxis")

    if drawStatBox:
        ROOT.gPad.Update()
        ROOT.gStyle.SetOptStat(1110)
        ROOT.gStyle.SetOptFit(1102)
    else:
        h1.SetStats(0)

    vertline = ROOT.TLine(36,0,36,canvas.GetUymax())
    vertline.SetLineColor(ROOT.kBlack)
    vertline.SetLineStyle(3) # 2 for denser hatches
    bintext = ROOT.TLatex()
    #bintext.SetNDC()
    bintext.SetTextSize(0.025)  # 0.03
    bintext.SetTextFont(42)
    if len(textForLines):
        bintext.SetTextAngle(45 if "#eta" in textForLines[0] else 30)

    if len(drawVertLines):
        #print "drawVertLines = " + drawVertLines
        nptBins = int(drawVertLines.split(',')[0])
        etarange = float(drawVertLines.split(',')[1])        
        offsetXaxisHist = h1.GetXaxis().GetBinLowEdge(0)
        sliceLabelOffset = 6. if "#eta" in textForLines[0] else 6.
        for i in range(1,nptBins): # do not need line at canvas borders
            #vertline.DrawLine(offsetXaxisHist+etarange*i,0,offsetXaxisHist+etarange*i,canvas.GetUymax())
            vertline.DrawLine(etarange*i-offsetXaxisHist,0,etarange*i-offsetXaxisHist,ymax)
        if len(textForLines):
            for i in range(0,len(textForLines)): # we need nptBins texts
                #texoffset = 0.1 * (4 - (i%4))
                #ytext = (1. + texoffset)*ymax/2.  
                ytext = 0.6*ymax                  
                bintext.DrawLatex(etarange*i + etarange/sliceLabelOffset, ytext, textForLines[i])

    # redraw legend, or vertical lines appear on top of it
    leg.Draw("same")

    if len(moreText):
        realtext = moreText.split("::")[0]
        x1,y1,x2,y2 = 0.7,0.8,0.9,0.9
        if "::" in moreText:
            x1,y1,x2,y2 = (float(x) for x in (moreText.split("::")[1]).split(","))
        pavetext = ROOT.TPaveText(x1,y1,x2,y2,"NB NDC")
        for tx in realtext.split(";"):
            pavetext.AddText(tx)
        pavetext.SetFillColor(0)
        pavetext.SetFillStyle(0)
        pavetext.SetBorderSize(0)
        pavetext.SetLineColor(0)
        pavetext.Draw("same")

    if len(moreTextLatex):
        realtext = moreTextLatex.split("::")[0]
        x1,y1,ypass,textsize = 0.75,0.8,0.08,0.035
        if "::" in moreTextLatex:
            x1,y1,ypass,textsize = (float(x) for x in (moreTextLatex.split("::")[1]).split(","))            
        lat = ROOT.TLatex()
        lat.SetNDC();
        lat.SetTextFont(42)        
        lat.SetTextSize(textsize)
        for itx,tx in enumerate(realtext.split(";")):
            lat.DrawLatex(x1,y1-itx*ypass,tx)

  # TPaveText *pvtxt = NULL;
  # if (yAxisName == "a.u.") {
  #   pvtxt = new TPaveText(0.6,0.6,0.95,0.7, "BR NDC")
  #   pvtxt.SetFillColor(0)
  #   pvtxt.SetFillStyle(0)
  #   pvtxt.SetBorderSize(0)
  #   pvtxt.AddText(Form("norm num/den = %.2f +/- %.2f",IntegralRatio,ratioError))
  #   pvtxt.Draw()
  # }

    setTDRStyle()
    if leftMargin > 0.1:
        if lumi != None: CMS_lumi(canvas,lumi,True,False)
        else:            CMS_lumi(canvas,"",True,False)
    else:
        latCMS = ROOT.TLatex()
        latCMS.SetNDC();
        latCMS.SetTextFont(42)
        latCMS.SetTextSize(0.045)
        latCMS.DrawLatex(0.1, 0.95, '#bf{CMS} #it{Preliminary}')
        if lumi != None: latCMS.DrawLatex(0.85, 0.95, '%s fb^{-1} (13 TeV)' % lumi)
        else:            latCMS.DrawLatex(0.90, 0.95, '(13 TeV)' % lumi)

    if lowerPanelHeight:
        pad2.Draw()
        pad2.cd()

        frame.Reset("ICES")
        if setRatioYAxisRangeFromUser: frame.GetYaxis().SetRangeUser(yminRatio,ymaxRatio)
        #else:                          
        #frame.GetYaxis().SetRangeUser(0.5,1.5)
        frame.GetYaxis().SetNdivisions(5)
        frame.GetYaxis().SetTitle(yRatioAxisName)
        frame.GetYaxis().SetTitleOffset(yAxisTitleOffset)
        frame.GetYaxis().SetTitleSize(0.05)
        frame.GetYaxis().SetLabelSize(0.04)
        frame.GetYaxis().CenterTitle()
        frame.GetXaxis().SetTitle(xAxisName)
        if setXAxisRangeFromUser: frame.GetXaxis().SetRangeUser(xmin,xmax)
        frame.GetXaxis().SetTitleOffset(1.2)
        frame.GetXaxis().SetTitleSize(0.05)

        #ratio = copy.deepcopy(h1.Clone("ratio"))
        #den_noerr = copy.deepcopy(h2.Clone("den_noerr"))
        ratio = None
        den_noerr = None
        den = None
        if invertRatio:
            ratio = h2.Clone("ratio")
            den_noerr = h1.Clone("den_noerr")
            den = h1.Clone("den")
        else:
            ratio = h1.Clone("ratio")
            den_noerr = h2.Clone("den_noerr")
            den = h2.Clone("den")

        for iBin in range (1,den_noerr.GetNbinsX()+1):
            den_noerr.SetBinError(iBin,0.)

        if useDifferenceInLowerPanel:
            ratio.Add(den_noerr,-1.0)
            den.Add(den_noerr,-1.0)
        else:
            ratio.Divide(den_noerr)
            den.Divide(den_noerr)

        if invertRatio:
            if histMCpartialUnc == None:
                ratio.SetFillColor(ROOT.kGray+1)
                ratio.SetFillStyle(1001)  # make it solid again
                ratio.SetLineColor(ROOT.kGray+3)
            else:
                ratio.SetFillColor(ROOT.kRed+1) # kGreen+1
                ratio.SetFillStyle(1001)  # 1001 to make it solid again
                ratio.SetLineColor(ROOT.kRed+2) # kGreen+2                       
                ratio.SetLineWidth(1) # make it smaller when it is drawn on top of something
        else:
            if histMCpartialUnc == None:
                den.SetFillColor(ROOT.kGray+1)
                den.SetFillStyle(1001)  # make it solid again
                den.SetLineColor(ROOT.kRed)        
        if histMCpartialUnc != None:
            h3ratio = h3.Clone("h3ratio")
            if useDifferenceInLowerPanel:
                h3ratio.Add(den_noerr,-1.0)
            else:
                h3ratio.Divide(den_noerr)
            #h3ratio.SetFillStyle(3144) # 1001 for solid, 3144 instead of 3244, to have more dense hatches
            h3ratio.SetFillStyle(1001) # 
            h3ratio.SetFillColor(ROOT.kGreen+1)  # kRed-4
            h3ratio.SetLineColor(ROOT.kGreen+2)  # kRed-4

        frame.Draw()        
        if invertRatio:
            den.SetMarkerSize(0.85)
            den.SetMarkerStyle(20) 
            #den.Draw("EPsame")    # draw after red line, not now
            ratio.Draw("E2same")
        else:    
            ratio.SetMarkerSize(0.85)
            ratio.SetMarkerStyle(20) 
            den.Draw("E2same")
            #ratio.Draw("EPsame") # draw after red line, not now

        if histMCpartialUnc != None:
            h3ratio.Draw("E2 same")
        
        line = ROOT.TF1("horiz_line","0" if useDifferenceInLowerPanel else "1",
                        ratio.GetXaxis().GetBinLowEdge(1),ratio.GetXaxis().GetBinLowEdge(ratio.GetNbinsX()+1))
        line.SetLineWidth(1)
        line.Draw("Lsame")
        if invertRatio:
            ratioline = ratio.Clone("ratioline")
            ratioline.SetFillColor(0)
            if histMCpartialUnc != None and len(drawVertLines):
                # unrolled, here the red line would be too large and given the fluctuations is could be seen as an additional band
                ratioline.SetLineColor(histMCpartialUnc.GetLineColor())
            ratioline.SetFillStyle(0)
            if histMCpartialUnc != None: ratioline.Draw("HIST same") # to draw the line inside the band for the expected
            den.Draw("EPsame")
        else: 
            ratio.Draw("EPsame")

        x1leg2 = 0.15 if leftMargin > 0.1 else 0.07
        x2leg2 = 0.55 if leftMargin > 0.1 else 0.27
        y1leg2 = 0.28 if leftMargin > 0.1 else 0.3
        y2leg2 = 0.33 if leftMargin > 0.1 else 0.35
        leg2 = ROOT.TLegend(x1leg2, y1leg2, x2leg2, y2leg2)
        #leg2 = ROOT.TLegend(0.07,0.30,0.27,0.35)
        leg2.SetFillColor(0)
        leg2.SetFillStyle(0)
        leg2.SetBorderSize(0)
        if invertRatio:
            leg2.AddEntry(ratio,"total theory uncertainty","LF")
        else:
            leg2.AddEntry(den,"total theory uncertainty","LF")
        if not noLegendLowerPanel: leg2.Draw("same")
        if histMCpartialUnc != None:
            leg2bis = ROOT.TLegend(x1leg2 + 0.45, y1leg2, x2leg2 + 0.45, y2leg2)
            leg2bis.SetFillColor(0)
            leg2bis.SetFillStyle(0)
            leg2bis.SetBorderSize(0)
            leg2bis.AddEntry(h3ratio,histMCpartialUncLegEntry,"LF")
            if not noLegendLowerPanel: leg2bis.Draw("same")

        pad2.RedrawAxis("sameaxis")


    if draw_both0_noLog1_onlyLog2 != 2:
        canvas.SaveAs(outdir + canvasName + ".png")
        canvas.SaveAs(outdir + canvasName + ".pdf")

    if draw_both0_noLog1_onlyLog2 != 1:        
        if yAxisName == "a.u.": 
            h1.GetYaxis().SetRangeUser(max(0.0001,h1.GetMinimum()*0.8),h1.GetMaximum()*100)
        else:
            h1.GetYaxis().SetRangeUser(max(0.001,h1.GetMinimum()*0.8),h1.GetMaximum()*100)
        canvas.SetLogy()
        canvas.SaveAs(outdir + canvasName + "_logY.png")
        canvas.SaveAs(outdir + canvasName + "_logY.pdf")
        canvas.SetLogy(0)

#########################################################################


def drawXsecAndTheoryband(h1, h2,  # h1 is data, h2 is total uncertainty band
                          labelXtmp="xaxis", labelYtmp="yaxis",
                          canvasName="default", outdir="./",
                          #rebinFactorX=0,
                          draw_both0_noLog1_onlyLog2=0,                  
                          leftMargin=0.15,
                          rightMargin=0.04,
                          labelRatioTmp="Data/pred.::0.5,1.5",
                          drawStatBox=False,
                          #legendCoords="0.15,0.35,0.8,0.9",  # x1,x2,y1,y2
                          legendCoords="0.15,0.65,0.85,0.9;3",  # for unrolled use 1 line # x1,x2,y1,y2
                          canvasSize="600,700",  # use X,Y to pass X and Y size     
                          lowerPanelHeight = 0.3,  # number from 0 to 1, 0.3 means 30% of space taken by lower panel. 0 means do not draw lower panel with relative error
                          passCanvas=None,
                          lumi=None,
                          drawVertLines="", # "12,36": format --> N of sections (e.g: 12 pt bins), and N of bins in each section (e.g. 36 eta bins), assuming uniform bin width
                          textForLines=[],                       
                          moreText="",
                          moreTextLatex="",
                          invertRatio = False,  # make expected over observed if True
                          histMCpartialUnc = None,
                          histMCpartialUncLegEntry = "",
                          useDifferenceInLowerPanel = False,
                          noLegendLowerPanel = False,
                          legendEntries = []
                      ):

    # moreText is used to pass some text to write somewhere (TPaveText is used)
    # e.g.  "stuff::x1,y1,x2,y2"  where xi and yi are the coordinates for the text
    # one can add more lines using the ";" key. FOr example, "stuff1;stuff2::x1,y1,x2,y2"
    # the coordinates should be defined taking into account how many lines will be drawn
    # if the coordinates are not passed (no "::"), then default ones are used, but this might not be satisfactory

    # moreTextLatex is similar, but used TLatex, and the four coordinates are x1,y1,ypass,textsize
    # where x1 and y1 are the coordinates the first line, and ypass is how much below y1 the second line is (and so on for following lines)

    # h1 is data, h2 in MC

    #if (rebinFactorX): 
    #    if isinstance(rebinFactorX, int): h1.Rebin(rebinFactorX)
    #    # case in which rebinFactorX is a list of bin edges
    #    else:                             h1.Rebin(len(rebinFactorX)-1,"",array('d',rebinFactorX)) 

    # colorBandpart = {"line" : ROOT.kRed,
    #                  "fill" : ROOT.kRed-9}
    # colorBandTot = {"line" : ROOT.kGreen,
    #                 "fill" : ROOT.kGreen-9}
    # colorBandPart = {"line" : ROOT.kCyan+2,
    #                  "fill" : ROOT.kCyan}
    # colorBandTot = {"line" : ROOT.kOrange+2,
    #                 "fill" : ROOT.kOrange}
    colorBandPart = {"line" : ROOT.kCyan+2,
                     "fill" : ROOT.kCyan-7}
    colorBandTot = {"line" : ROOT.kOrange+7,
                    "fill" : ROOT.kOrange-3}

    xAxisName,setXAxisRangeFromUser,xmin,xmax = getAxisRangeFromUser(labelXtmp)
    yAxisName,setYAxisRangeFromUser,ymin,ymax = getAxisRangeFromUser(labelYtmp)
    yRatioAxisName,setRatioYAxisRangeFromUser,yminRatio,ymaxRatio = getAxisRangeFromUser(labelRatioTmp)

    yAxisTitleOffset = 1.45 if leftMargin > 0.1 else 0.6

    addStringToEnd(outdir,"/",notAddIfEndswithMatch=True)
    createPlotDirAndCopyPhp(outdir)

    cw,ch = canvasSize.split(',')
    #canvas = ROOT.TCanvas("canvas",h2D.GetTitle() if plotLabel == "ForceTitle" else "",700,625)
    canvas = passCanvas if passCanvas != None else ROOT.TCanvas("canvas","",int(cw),int(ch))
    canvas.SetTickx(1)
    canvas.SetTicky(1)
    canvas.cd()
    canvas.SetLeftMargin(leftMargin)
    canvas.SetRightMargin(rightMargin)
    canvas.cd()

    pad2 = 0
    if lowerPanelHeight: 
        canvas.SetBottomMargin(lowerPanelHeight)
        pad2 = ROOT.TPad("pad2","pad2",0,0.,1,0.9)
        pad2.SetTopMargin(1-lowerPanelHeight)
        pad2.SetRightMargin(rightMargin)
        pad2.SetLeftMargin(leftMargin)
        pad2.SetFillColor(0)
        pad2.SetGridy(1)
        pad2.SetFillStyle(0)


    frame = h1.Clone("frame")
    frame.GetXaxis().SetLabelSize(0.04)
    frame.SetStats(0)

    h1.SetLineColor(ROOT.kBlack)
    h1.SetMarkerColor(ROOT.kBlack)
    h1.SetMarkerStyle(20)
    h1.SetMarkerSize(1)

    #ymax = max(ymax, max(h1.GetBinContent(i)+h1.GetBinError(i) for i in range(1,h1.GetNbinsX()+1)))
    # if min and max were not set, set them based on histogram content
    if ymin == ymax == 0.0:
        ymin,ymax = getMinMaxHisto(h1,excludeEmpty=True,sumError=True)            
        ymin *= 0.9
        ymax *= (1.1 if leftMargin > 0.1 else 2.0)
        if ymin < 0: ymin = 0
        #print "drawSingleTH1() >>> Histo: %s     minY,maxY = %.2f, %.2f" % (h1.GetName(),ymin,ymax)

    # print "#### WARNING ####"
    # print "Hardcoding ymin = 0 in function drawDataAndMC(): change it if it is not what you need"
    # print "#################"
    # ymin = 0 # hardcoded

    if lowerPanelHeight:
        h1.GetXaxis().SetLabelSize(0)
        h1.GetXaxis().SetTitle("")  
    else:
        h1.GetXaxis().SetTitle(xAxisName)
        h1.GetXaxis().SetTitleOffset(1.2)
        h1.GetXaxis().SetTitleSize(0.05)
        h1.GetXaxis().SetLabelSize(0.04)
    h1.GetYaxis().SetTitle(yAxisName)
    h1.GetYaxis().SetTitleOffset(yAxisTitleOffset)
    h1.GetYaxis().SetTitleSize(0.05)
    h1.GetYaxis().SetLabelSize(0.04)
    h1.GetYaxis().SetRangeUser(ymin, ymax)    
    h1.GetYaxis().SetTickSize(0.01)
    if setXAxisRangeFromUser: h1.GetXaxis().SetRangeUser(xmin,xmax)
    h1.Draw("EP")
    #h1err = h1.Clone("h1err")
    #h1err.SetFillColor(ROOT.kRed+2)
    h2.SetFillStyle(1001)  # 3001 is better than 3002 for pdf, while 3002 is perfect for png
    h2.SetLineColor(colorBandTot["line"])  # kGreen+2
    h2.SetFillColor(colorBandTot["fill"])    # kGreen
    h2.SetLineWidth(1)
    h2.Draw("2 SAME")
    h3 = None
    if histMCpartialUnc != None:
        h3 = histMCpartialUnc.Clone("histMCpartialUnc")
        h3.SetFillColor(colorBandPart["fill"])
        h3.SetLineColor(colorBandPart["line"])
        h3.SetFillStyle(1001)  # 1001, 3001, 3144 , 3244, 3003
        #h3.SetFillStyle(3244)  # 3144 , 3244, 3003
        h3.Draw("2 SAME")
        #for i in range(1,1+h3.GetNbinsX()):
        #    print "PDF band: bin %d  val +/- error = %.3f +/- %.3f" % (i, h3.GetBinContent(i),h3.GetBinError(i))
    h2line = None
    h2line = h1.Clone("h2line")
    for i in range(1,h2line.GetNbinsX()+1):
        xval = h2line.GetBinCenter(i)
        yval = h2.Eval(xval)
        h2line.SetBinContent(i,yval)
    h2line.SetFillColor(0)
    h2line.SetLineColor(h2.GetLineColor())
    h2line.Draw("HIST SAME")
    h1.Draw("EP SAME")

    nColumnsLeg = 1
    if ";" in legendCoords: 
        nColumnsLeg = int(legendCoords.split(";")[1])
        #if histMCpartialUnc != None: nColumnsLeg = 3
    legcoords = [float(x) for x in (legendCoords.split(";")[0]).split(',')]
    lx1,lx2,ly1,ly2 = legcoords[0],legcoords[1],legcoords[2],legcoords[3]
    if histMCpartialUnc != None:
        ly2 = ly2 + 0.5 * (ly2 - ly1) # add one more row
    leg = ROOT.TLegend(lx1,ly1,lx2,ly2)
    #leg.SetFillColor(0)
    #leg.SetFillStyle(0)
    #leg.SetBorderSize(0)
    leg.SetNColumns(nColumnsLeg)
    if len(legendEntries):
        leg.AddEntry(h1,str(legendEntries[0]),"LPE")
        leg.AddEntry(h2,str(legendEntries[1]),"LF")        
    else:
        leg.AddEntry(h1,"measured","LPE")
        leg.AddEntry(h2,"aMC@NLO","LF")
    if histMCpartialUnc != None:
        leg.AddEntry(h3,histMCpartialUncLegEntry,"LF")
    #leg.AddEntry(h1err,"Uncertainty","LF")
    leg.Draw("same")
    canvas.RedrawAxis("sameaxis")

    if drawStatBox:
        ROOT.gPad.Update()
        ROOT.gStyle.SetOptStat(1110)
        ROOT.gStyle.SetOptFit(1102)
    else:
        h1.SetStats(0)

    vertline = ROOT.TLine(36,0,36,canvas.GetUymax())
    vertline.SetLineColor(ROOT.kBlack)
    vertline.SetLineStyle(3) # 2 for denser hatches
    bintext = ROOT.TLatex()
    #bintext.SetNDC()
    bintext.SetTextSize(0.025)  # 0.03
    bintext.SetTextFont(42)
    if len(textForLines):
        bintext.SetTextAngle(45 if "#eta" in textForLines[0] else 30)

    if len(drawVertLines):
        #print "drawVertLines = " + drawVertLines
        nptBins = int(drawVertLines.split(',')[0])
        etarange = float(drawVertLines.split(',')[1])        
        offsetXaxisHist = h1.GetXaxis().GetBinLowEdge(0)
        sliceLabelOffset = 6. if "#eta" in textForLines[0] else 6.
        for i in range(1,nptBins): # do not need line at canvas borders
            #vertline.DrawLine(offsetXaxisHist+etarange*i,0,offsetXaxisHist+etarange*i,canvas.GetUymax())
            vertline.DrawLine(etarange*i-offsetXaxisHist,0,etarange*i-offsetXaxisHist,ymax)
        if len(textForLines):
            for i in range(0,len(textForLines)): # we need nptBins texts
                #texoffset = 0.1 * (4 - (i%4))
                #ytext = (1. + texoffset)*ymax/2.  
                ytext = 0.6*ymax                  
                bintext.DrawLatex(etarange*i + etarange/sliceLabelOffset, ytext, textForLines[i])

    # redraw legend, or vertical lines appear on top of it
    leg.Draw("same")

    if len(moreText):
        realtext = moreText.split("::")[0]
        x1,y1,x2,y2 = 0.7,0.8,0.9,0.9
        if "::" in moreText:
            x1,y1,x2,y2 = (float(x) for x in (moreText.split("::")[1]).split(","))
        pavetext = ROOT.TPaveText(x1,y1,x2,y2,"NB NDC")
        for tx in realtext.split(";"):
            pavetext.AddText(tx)
        pavetext.SetFillColor(0)
        pavetext.SetFillStyle(0)
        pavetext.SetBorderSize(0)
        pavetext.SetLineColor(0)
        pavetext.Draw("same")

    if len(moreTextLatex):
        realtext = moreTextLatex.split("::")[0]
        x1,y1,ypass,textsize = 0.75,0.8,0.08,0.035
        if "::" in moreTextLatex:
            x1,y1,ypass,textsize = (float(x) for x in (moreTextLatex.split("::")[1]).split(","))            
        lat = ROOT.TLatex()
        lat.SetNDC();
        lat.SetTextFont(42)        
        lat.SetTextSize(textsize)
        for itx,tx in enumerate(realtext.split(";")):
            lat.DrawLatex(x1,y1-itx*ypass,tx)

  # TPaveText *pvtxt = NULL;
  # if (yAxisName == "a.u.") {
  #   pvtxt = new TPaveText(0.6,0.6,0.95,0.7, "BR NDC")
  #   pvtxt.SetFillColor(0)
  #   pvtxt.SetFillStyle(0)
  #   pvtxt.SetBorderSize(0)
  #   pvtxt.AddText(Form("norm num/den = %.2f +/- %.2f",IntegralRatio,ratioError))
  #   pvtxt.Draw()
  # }

    setTDRStyle()
    if leftMargin > 0.1:
        if lumi != None: CMS_lumi(canvas,lumi,True,False)
        else:            CMS_lumi(canvas,"",True,False)
    else:
        latCMS = ROOT.TLatex()
        latCMS.SetNDC();
        latCMS.SetTextFont(42)
        latCMS.SetTextSize(0.045)
        latCMS.DrawLatex(0.1, 0.95, '#bf{CMS} #it{Preliminary}')
        if lumi != None: latCMS.DrawLatex(0.85, 0.95, '%s fb^{-1} (13 TeV)' % lumi)
        else:            latCMS.DrawLatex(0.90, 0.95, '(13 TeV)' % lumi)

    if lowerPanelHeight:
        pad2.Draw()
        pad2.cd()

        frame.Reset("ICES")
        if setRatioYAxisRangeFromUser: frame.GetYaxis().SetRangeUser(yminRatio,ymaxRatio)
        #else:                          
        #frame.GetYaxis().SetRangeUser(0.5,1.5)
        frame.GetYaxis().SetNdivisions(5)
        frame.GetYaxis().SetTitle(yRatioAxisName)
        frame.GetYaxis().SetTitleOffset(yAxisTitleOffset)
        frame.GetYaxis().SetTitleSize(0.05)
        frame.GetYaxis().SetLabelSize(0.04)
        frame.GetYaxis().CenterTitle()
        frame.GetXaxis().SetTitle(xAxisName)
        if setXAxisRangeFromUser: frame.GetXaxis().SetRangeUser(xmin,xmax)
        frame.GetXaxis().SetTitleOffset(1.2)
        frame.GetXaxis().SetTitleSize(0.05)

        #ratio = copy.deepcopy(h1.Clone("ratio"))
        #den_noerr = copy.deepcopy(h2.Clone("den_noerr"))
        ratio = None
        den_noerr = None
        den = None
        den_noerr_TH1 = None
        if invertRatio:
            ratio = h2.Clone("ratio")
            den_noerr = h1.Clone("den_noerr")
            den = h1.Clone("den")
        else:
            ratio = h1.Clone("ratio")
            den_noerr = h2.Clone("den_noerr")
            den = h2.Clone("den")

        if den_noerr.InheritsFrom("TH1"):
            for iBin in range (1,den_noerr.GetNbinsX()+1):
                den_noerr.SetBinError(iBin,0.)
            if useDifferenceInLowerPanel:
                ratio.Add(den_noerr,-1.0)
                den.Add(den_noerr,-1.0)
            else:
                ratio.Divide(den_noerr)
                den.Divide(den_noerr)
        else:
            den_noerr_TH1 = h1.Clone("den_noerr_TH1")
            den_noerr_TH1.Reset("ICESM")                        
            for i in range(den_noerr_TH1.GetNbinsX()):
                yval = den_noerr.Eval( den_noerr_TH1.GetBinCenter(i+1) )
                den_noerr_TH1.SetBinContent(i+1, yval)
                den_noerr_TH1.SetBinError(i+1, 0)
                den_noerr.SetPointEYhigh(i, 0)
                den_noerr.SetPointEYlow( i, 0)

            if useDifferenceInLowerPanel:
                ratio.Add(den_noerr_TH1,-1.0)
                for i in range(den_noerr_TH1.GetNbinsX()):
                    xval = den_noerr_TH1.GetBinCenter(i+1)
                    yval = den_noerr_TH1.GetBinContent(i+1)
                    den.SetPoint(i, xval, 0.0)
            else:
                ratio.Divide(den_noerr_TH1)
                for i in range(den_noerr_TH1.GetNbinsX()):
                    xval = den_noerr_TH1.GetBinCenter(i+1)
                    yval = den_noerr_TH1.GetBinContent(i+1)
                    den.SetPoint(i, xval, 1.0)
                    den.SetPointEYhigh(i, den.GetErrorYhigh(i)/yval)
                    den.SetPointEYlow(i, den.GetErrorYlow(i)/yval)

        if invertRatio:
            if histMCpartialUnc == None:
                ratio.SetFillColor(ROOT.kGray+1)
                ratio.SetFillStyle(1001)  # make it solid again
                ratio.SetLineColor(ROOT.kGray+3)
            else:
                ratio.SetFillColor(colorBandPart["fill"]) # kGreen+1
                ratio.SetFillStyle(1001)  # 1001 to make it solid again
                #ratio.SetLineColor(colorBandTot["line"]) # kGreen+2                       
                ratio.SetLineColor(ROOT.kRed) # kGreen+2                       
                ratio.SetLineWidth(1) # make it smaller when it is drawn on top of something
        else:
            if histMCpartialUnc == None:
                den.SetFillColor(ROOT.kGray+1)
                den.SetFillStyle(1001)  # make it solid again
                den.SetLineColor(ROOT.kRed)        

        if histMCpartialUnc != None:
            h3ratio = h3.Clone("h3ratio")
            h3ratio.SetFillStyle(1001) # 
            h3ratio.SetFillColor(colorBandPart["fill"])  # kRed-4
            h3ratio.SetLineColor(colorBandPart["line"])  # kRed-4
            if h3ratio.InheritsFrom("TH1"):
                if useDifferenceInLowerPanel:
                    h3ratio.Add(den_noerr,-1.0)
                else:
                    h3ratio.Divide(den_noerr)
                #h3ratio.SetFillStyle(3144) # 1001 for solid, 3144 instead of 3244, to have more dense hatches
            else:
                if useDifferenceInLowerPanel:
                    for i in range(den_noerr_TH1.GetNbinsX()):
                        xval = den_noerr_TH1.GetBinCenter(i+1)
                        yval = den_noerr_TH1.GetBinContent(i+1)
                        h3ratio.SetPoint(i, xval, 0.0)
                else:
                    for i in range(den_noerr_TH1.GetNbinsX()):
                        xval = den_noerr_TH1.GetBinCenter(i+1)
                        yval = den_noerr_TH1.GetBinContent(i+1)
                        h3ratio.SetPoint(i, xval, 1.0)
                        h3ratio.SetPointEYhigh(i, h3ratio.GetErrorYhigh(i)/yval)
                        h3ratio.SetPointEYlow(i,  h3ratio.GetErrorYlow(i)/yval)



        frame.Draw()        
        if invertRatio:
            den.SetMarkerSize(0.85)
            den.SetMarkerStyle(20) 
            #den.Draw("EPsame")    # draw after red line, not now            
            ratio.Draw("F2same")
        else:    
            ratio.SetMarkerSize(0.85)
            ratio.SetMarkerStyle(20) 
            den.Draw("F2same")
            #ratio.Draw("EPsame") # draw after red line, not now

        if histMCpartialUnc != None:
            h3ratio.Draw("F2 same")
        
        line = ROOT.TF1("horiz_line","0" if useDifferenceInLowerPanel else "1",
                        ratio.GetXaxis().GetBinLowEdge(1),ratio.GetXaxis().GetBinLowEdge(ratio.GetNbinsX()+1))
        line.SetLineWidth(1)
        line.Draw("Lsame")
        # if invertRatio:
        #     ratioline = ratio.Clone("ratioline")
        #     ratioline.SetFillColor(0)
        #     if histMCpartialUnc != None and len(drawVertLines):
        #         # unrolled, here the red line would be too large and given the fluctuations is could be seen as an additional band
        #         ratioline.SetLineColor(histMCpartialUnc.GetLineColor())
        #     ratioline.SetFillStyle(0)
        #     if histMCpartialUnc != None: 
        #         if ratioline.InheritsFrom("TH1"):
        #             ratioline.Draw("HIST same") # to draw the line inside the band for the expected
        #         else:
        #             ratioline.Draw("L same")
        #     den.Draw("EPsame")
        #else: 
        #    ratio.Draw("EPsame")
        if invertRatio:
            den.Draw("EPsame") 
        else:
            ratio.Draw("EPsame")

        x1leg2 = 0.15 if leftMargin > 0.1 else 0.07
        x2leg2 = 0.55 if leftMargin > 0.1 else 0.27
        y1leg2 = 0.28 if leftMargin > 0.1 else 0.3
        y2leg2 = 0.33 if leftMargin > 0.1 else 0.35
        leg2 = ROOT.TLegend(x1leg2, y1leg2, x2leg2, y2leg2)
        #leg2 = ROOT.TLegend(0.07,0.30,0.27,0.35)
        leg2.SetFillColor(0)
        leg2.SetFillStyle(0)
        leg2.SetBorderSize(0)
        if invertRatio:
            leg2.AddEntry(ratio,"total theory uncertainty","LF")
        else:
            leg2.AddEntry(den,"total theory uncertainty","LF")
        if not noLegendLowerPanel: leg2.Draw("same")
        if histMCpartialUnc != None:
            leg2bis = ROOT.TLegend(x1leg2 + 0.45, y1leg2, x2leg2 + 0.45, y2leg2)
            leg2bis.SetFillColor(0)
            leg2bis.SetFillStyle(0)
            leg2bis.SetBorderSize(0)
            leg2bis.AddEntry(h3ratio,histMCpartialUncLegEntry,"LF")
            if not noLegendLowerPanel: leg2bis.Draw("same")

        pad2.RedrawAxis("sameaxis")
        ROOT.gPad.RedrawAxis()

    if draw_both0_noLog1_onlyLog2 != 2:
        canvas.SaveAs(outdir + canvasName + ".png")
        canvas.SaveAs(outdir + canvasName + ".pdf")

    if draw_both0_noLog1_onlyLog2 != 1:        
        if yAxisName == "a.u.": 
            h1.GetYaxis().SetRangeUser(max(0.0001,h1.GetMinimum()*0.8),h1.GetMaximum()*100)
        else:
            h1.GetYaxis().SetRangeUser(max(0.001,h1.GetMinimum()*0.8),h1.GetMaximum()*100)
        canvas.SetLogy()
        canvas.SaveAs(outdir + canvasName + "_logY.png")
        canvas.SaveAs(outdir + canvasName + "_logY.pdf")
        canvas.SetLogy(0)
        

#########################################################################

## some old functions from 2D xsec analysis, which are imported by other scripts, let's put them here

def getArrayParsingString(inputString, verbose=False, makeFloat=False):
    # convert string [a,b,c,...] to list of a b c ...
    tmp = inputString.replace('[','').replace(']','')
    tmp = tmp.split(',')
    if verbose:
        logger.info("Input: %s" % inputString)
        logger.info("Output: %s" % tmp)
    if makeFloat:
        ret = [float(x) for x in tmp]
    else:
        ret = tmp
    return ret

# new function
def getGlobalBin(ix, iy, nbinsX, binFrom0=True):
    # ix goes from 0 to nbinsX-1, like the value returned by "for ix in range(nbinsX)"
    # same is expected for iy
    # If this is the case, global bin starts from 0
    #However, if binFrom0=False, it is expected that the bins start from 1 (like those of a TH1) and the globalbin that is returned will start from 1 as well
    if binFrom0:
        return (ix + iy * nbinsX)
    else:
        return (ix-1 + (iy-1) * nbinsX) + 1  # trasform ix,iy in [1,N] to ix',iy' in [0,N-1], get global bin and sum 1 so that it starts from 1

def getXYBinsFromGlobalBin(globalbin, nbinsX, binFrom0=True):
    # global bin goes from 0 to nbinX*nbinsY-1 
    # returned x(y) is a number from 0 to nbinsX(Y) -1
    # however, if that is not the convention, then binFrom0 must be set to False: this manages the case where the global bin starts from 1 and the returned ix and iy will start from 1 as well
    tmp = globalbin if binFrom0 else (globalbin-1)
    iy = int(tmp/nbinsX)
    ix = tmp % nbinsX
    if not binFrom0:
        ix = ix + 1
        iy = iy + 1
    return ix,iy

def getArrayBinNumberFromValue(binEdgesArray,val):
    # assumes values in binEdgesArray are ordered in increasing order
    # we follow ROOT convention: when evaluating bin=ibin, upper edge belongs to ibin+1, lower edge belongs to ibin
    # return -2 for overflow, -1 for underflow, a number in [0,len(binEdgesArray)-1] otherwise
    ret = -2
    if val < binEdgesArray[0]: return -1
    for bin in range(len(binEdgesArray)-1):
        if val < binEdgesArray[bin+1]:
            ret = bin
            break
    return ret


class templateBinning:
    def __init__(self,etaBins=[],ptBins=[]):
        self.etaBins = etaBins
        self.ptBins = ptBins
        self.Neta = len(etaBins)-1
        self.Npt  = len(ptBins)-1
        self.NTotBins = self.Neta * self.Npt

    def printBin(self):
        print("###########################")
        print("Binning: eta-pt on x-y axis")
        print("eta bins: %s" % str(self.Neta))
        print("pt  bins: %s" % str(self.Npt))
        print("")

    def printBinAll(self):
        print("###########################")
        print("Binning: eta-pt on x-y axis (%d bins)" % self.NTotBins)
        print("eta bins: %s" % str(self.Neta))
        print("%s" % str(self.etaBins))
        print("-"*20)
        print("pt  bins: %s" % str(self.Npt))
        print("%s" % str(self.ptBins))
        print("-"*20)
        print("")

def getEtaPtBinning(inputBins, whichBins="reco"):
    
    # whichBins can be reco or gen
    # actually, gen was needed only for 2D xsec, might not be used anymore
    if whichBins not in ["reco", "gen"]:
        logger.error("In function getEtaPtBinning(): whichBins must be 'reco' or 'gen'. Exit")
        exit()

    # case in which we are passing a file containing the binning and not directly the binning itself
    if inputBins.startswith("file=") or re.match(".*binningPtEta.*.txt",inputBins):
        etaPtbinningFile = inputBins.replace("file=","")
        with open(etaPtbinningFile) as f:
            content = f.readlines()
        for x in content:
            if str(x).startswith(whichBins):
                tmpbinning = (x.split(whichBins+":")[1]).strip()
            else:
                continue
        etabinning = tmpbinning.split('*')[0]    # this is like [a,b,c,...], and is of type string. We nedd to get an array  
        ptbinning  = tmpbinning.split('*')[1]
    else:
        etabinning = inputBins.split('*')[0]    # this is like [a,b,c,...], and is of type string. We nedd to get an array  
        ptbinning  = inputBins.split('*')[1]
    etabinning = getArrayParsingString(etabinning,makeFloat=True)
    ptbinning  = getArrayParsingString(ptbinning,makeFloat=True)
    #binning = [len(etabinning)-1, etabinning, len(ptbinning)-1, ptbinning] 
    binning = [etabinning, ptbinning] 
    return binning

def roll1Dto2D(h1d, histo, invertXY=False):
    # need to know whether the unrolled histograms have consecutive pt shapes (usually y axis of 2D plots), in which case invertXY must be True, or eta shapes (usually x axis of 2D plots)
    # we used to have eta shapes in unrolled TH1D passed to combinetf, but now we feed combinetf directly with 2D histograms, and it internally unrolls them in the other axis
    nBinsXaxis2D = histo.GetNbinsY() if invertXY else histo.GetNbinsX()
    for i in range(1,h1d.GetNbinsX()+1):
        # histogram bin is numbered starting from 1, so add 1
        xbin = (i - 1) % nBinsXaxis2D + 1  
        ybin = (i - 1) / nBinsXaxis2D + 1
        xbin = int(xbin)
        ybin = int(ybin)
        if invertXY:
            # our 2D plots will always have eta on x axis, so we must also swap xbin and ybin defined above
            xbin,ybin = ybin,xbin
        histo.SetBinContent(xbin, ybin, h1d.GetBinContent(i))
        histo.SetBinError(xbin, ybin, h1d.GetBinError(i)) 

    return histo

def unroll2Dto1D(h, newname='', cropNegativeBins=True, silent=False, invertUnroll=False):
    nbins = h.GetNbinsX() * h.GetNbinsY()
    goodname = h.GetName()
    #h.SetName(goodname+"_2d")
    newh = ROOT.TH1D(f"{goodname}_unrolled" if not newname else newname, h.GetTitle(), nbins, 0.5, nbins+0.5)
    newh.Sumw2()
    if 'TH2' not in h.ClassName():
        raise RuntimeError("Calling rebin2Dto1D on something that is not TH2")
    if invertUnroll:
        for i in range(h.GetNbinsX()):
            for j in range(h.GetNbinsY()):
                ibin = 1 + j + i * h.GetNbinsY()
                newh.SetBinContent(ibin, h.GetBinContent(i+1, j+1))
                newh.SetBinError(ibin, h.GetBinError(i+1, j+1))
    else:
        for i in range(h.GetNbinsX()):
            for j in range(h.GetNbinsY()):
                ibin = 1 + i + j * h.GetNbinsX()
                newh.SetBinContent(ibin, h.GetBinContent(i+1, j+1))
                newh.SetBinError(ibin, h.GetBinError(i+1, j+1))

    if cropNegativeBins:
        for ibin in range(1, nbins+1):
            if newh.GetBinContent(ibin)<0:
                if not silent:
                    logger.warning('unroll2Dto1D(): cropping to zero bin %d in %s (was %f)'%(ibin, newh.GetName(), newh.GetBinContent(ibin)))
                newh.SetBinContent(ibin, 0)
    return newh


def dressed2D(h1d, binning, name, title='', invertXY=False):
    if len(binning) == 4:
        n1 = binning[0]; bins1 = array('d', binning[1])
        n2 = binning[2]; bins2 = array('d', binning[3])
        h2_1 = ROOT.TH2D(name, title, n1, bins1, n2, bins2 )
    else:
        n1 = binning[0]; min1 = binning[1]; max1 = binning[2]
        n2 = binning[3]; min2 = binning[4]; max2 = binning[5]
        h2_1 = ROOT.TH2D(name, title, n1, min1, max1, n2, min2, max2)
    h2_backrolled_1 = roll1Dto2D(h1d, h2_1, invertXY)
    return h2_backrolled_1

#==============================

def drawGraphCMS(grList, 
                 xAxisNameTmp = "xAxis", 
		 yAxisNameTmp = "yAxis", 
		 canvasName = "default",
		 outputDIR = "./",
		 leg_roc = None, # text for legend
                 legendCoords = "0.5,0.15,0.9,0.35;2", # number after ; sets the number of columns
                 lumi = None,
                 vecMCcolors = [ROOT.kBlack, ROOT.kRed, ROOT.kGreen+2, ROOT.kBlue, ROOT.kOrange+1, ROOT.kCyan+2, ROOT.kGray+2],
                 vecLineStyle = None,
                 vecMarkerStyle = None,
		 etabinText = "",
                 canvasSize="800,800",
                 passCanvas=None,
                 graphDrawStyle="p",
                 legEntryStyle="LF",
                 useOriginalGraphStyle=False, # if True use style from original graph
                 skipLumi=False,
                 solidLegend=False
             ):
    adjustSettings_CMS_lumi()
    xAxisName = ""
    xmin = 0
    xmax = 0
    xAxisName,setXAxisRangeFromUser,xmin,xmax = getAxisRangeFromUser(xAxisNameTmp)
    #
    yAxisName = ""
    ymin = 0
    ymax = 0
    yAxisName,setYAxisRangeFromUser,ymin,ymax = getAxisRangeFromUser(yAxisNameTmp)

    nGraphs = len(grList)

    cw,ch = canvasSize.split(',')
    canvas = passCanvas if passCanvas != None else ROOT.TCanvas("canvas","",int(cw),int(ch))
    canvas.SetTickx(1)
    canvas.SetTicky(1)
    canvas.cd()
    canvas.SetFillColor(0)
    canvas.SetGrid()
    canvas.SetLeftMargin(0.14)
    canvas.SetRightMargin(0.06)
    canvas.cd()

    nColumnsLeg = 1
    if ";" in legendCoords: 
        nColumnsLeg = int(legendCoords.split(";")[1])
    legcoords = [float(x) for x in (legendCoords.split(";")[0]).split(',')]
    lx1,ly1,lx2,ly2 = legcoords[0],legcoords[1],legcoords[2],legcoords[3]
    leg = ROOT.TLegend(lx1,ly1,lx2,ly2)
    if solidLegend:
        leg.SetFillColor(ROOT.kWhite)
    else:
        leg.SetFillColor(0)
        leg.SetFillStyle(0)
        leg.SetBorderSize(0)
    leg.SetNColumns(nColumnsLeg)

    for ig in range(0,nGraphs):
        if not useOriginalGraphStyle:
            grList[ig].SetMarkerColor(vecMCcolors[ig])
            grList[ig].SetLineColor(vecMCcolors[ig])
            grList[ig].SetLineWidth(2)
            grList[ig].SetFillColor(vecMCcolors[ig])
            if vecLineStyle:
                grList[ig].SetLineStyle(vecLineStyle[ig] if ig < len(vecLineStyle) else vecLineStyle[-1])
            if vecMarkerStyle:
                grList[ig].SetMarkerStyle(vecMarkerStyle[ig] if ig < len(vecMarkerStyle) else vecMarkerStyle[-1])
            else:
                grList[ig].SetMarkerStyle(20)
        if ig == 0: 
            grList[ig].Draw("a"+graphDrawStyle)
        else: 
            grList[ig].Draw(graphDrawStyle+" same")
        leg.AddEntry(grList[ig],leg_roc[ig],legEntryStyle)        

    canvas.RedrawAxis("sameaxis")

    grList[0].GetXaxis().SetTitleSize(0.05)
    grList[0].GetXaxis().SetLabelSize(0.04)
    grList[0].GetYaxis().SetTitleOffset(1.3)
    grList[0].GetYaxis().SetTitleSize(0.05)
    grList[0].GetYaxis().SetLabelSize(0.04)
    grList[0].GetXaxis().SetTitle(xAxisName)
    grList[0].GetYaxis().SetTitle(yAxisName)
    if setXAxisRangeFromUser:
        grList[0].GetXaxis().SetRangeUser(xmin,xmax)
    if setYAxisRangeFromUser:
        grList[0].GetYaxis().SetRangeUser(ymin,ymax)

    setTDRStyle() # check if it doesn't screw things up
    if not skipLumi: 
        if lumi != None: 
            CMS_lumi(canvas,lumi,True,False)
        else:   
            CMS_lumi(canvas,"",True,False)
        
    etabin = ROOT.TLatex()
    etabin.SetNDC() # not sure it is needed
    etabin.SetTextSize(0.05)
    etabin.SetTextFont(42)
    etabin.SetTextColor(ROOT.kBlack)
    if etabinText:
        etabin.DrawLatex(0.15,0.15,etabinText)

    canvas.RedrawAxis("sameaxis")
    leg.Draw("same")

    for ext in [".png",".pdf"]:
        canvas.SaveAs(outputDIR+canvasName+ext)
