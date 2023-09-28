#!/usr/bin/env python3

# python w-mass-13TeV/subMatrix.py cards/wmass_fixMassWeights_splitW/fit/hessian/fitresults_123456789_Asimov_clipSyst1p3_bbb1_cxs1.root -o plots/testNanoAOD/WmassPlots_jetEta2p4_fixMassWeight_splitW/afterFitPlots/subMatrix/ --param "muR$|.*muF$|muR.*Plus|muF.*Plus" --uniqueString "qcdScalesPlus" --title "correlation: QCD scales" --which-matrix "correlation" --skipLatexOnTop

# to make the matrix intelligible in HEPdata, add --use-hepdata-labels (and possibly adapt the name conversions)
# use --show-all-nuisances to show all nuisances

# python w-mass-13TeV/subMatrix.py /scratch/mciprian/CombineStudies/WMass/10Sept2022_qcdBkgVar/qcdScale_byHelicityPt/nominal/fit/hessian/fitresults_123456789_Asimov_bbb1_cxs0.root --outdir plots/fromMyWremnants/Wmass_fit/10Sept2022_qcdBkgVar/qcdScale_byHelicityPt/subMatrix/ --type hessian --postfix Asimov -p '.*pdf\d+' --uniqueString "PDFs" --title "correlation: PDFs" --skipLatexOnTop


import os, re, operator, math
import datetime
import argparse
from array import array

import utilitiesCMG
utilities = utilitiesCMG.util()

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

def niceName(name):

    if "helicity" in name:
        ret = name.split("_helicity")[0]
        if "qGen" in ret:
            Wcharge = "W^{+}" if "qGen1" in ret else "W^{ -}"
            ret = Wcharge + " " + ret.split("qGen")[1].split("_",1)[1]
        ret = ret.replace("gen","")
        return ret
    
    elif "CMS_prefire" in name:
        if "prefire_stat_m" in name:
            num = re.findall(r'\d+', name) # get number
            if int(num[0]) == 11:
                return "muonPrefire_stat hotspot"
            else:
                return f"muonPrefire_stat |#eta| bin {num[0]}"
        elif "prefire_syst_m" in name:
            return "muonPrefire_syst"
        else:
            return "ECAL prefire"
        # to be revisited
    elif re.match( "Fakes(Eta|EtaCharge|PtNorm|PtSlope)Uncorrelated.*",name):
        num = re.findall(r'\d+', name) # get number
        pfx = name.split(num[0])[1]    # split on number and read what's on the right
        leptonCharge = ""
        if len(pfx):
            leptonCharge = "{lep}{chs}".format(lep="#mu" if "mu" in pfx else "e", chs = "+" if "plus" in pfx else "-" if "minus" in pfx else "")
        tmpvar = ""
        if "FakesEtaCharge" in name: tmpvar = "#eta-ch"
        elif "FakesEta" in name: tmpvar = "#eta"
        elif "FakesPtNorm" in name: tmpvar = "p_{T}-norm"
        elif "FakesPtSlope" in name: tmpvar = "p_{T}-shape"    
        return "QCD bkg {var}-{n} {lepCh}".format(var=tmpvar, n=num[0], lepCh=leptonCharge)

    elif re.match(".*effStatTnP\d+.*",name):
        num = re.findall(r'\d+', name) # get number
        pfx = name.split("effStatTnP"+str(num[0]))[1]
        leptonCharge = ""
        if len(pfx):
            leptonCharge = "{chs}".format(chs = "+" if "Plus" in pfx else "-" if "Minus" in pfx else "")
        return "Eff.stat. {n1} {lepCh}".format(n1=num[0],lepCh=leptonCharge)

    elif re.match(".*QCDscale.*",name):
        # expect something like QCDscalePtChargeHelicity_PtVBin1genQ0AngCoeff0muF or less 
        # TODO: distinguish W and Z        
        boson = "" # "W"
        ptnum = re.findall(r'PtVBin\d+', name)
        chargenum = re.findall(r'genQ\d+', name)
        coeffnum = re.findall(r'AngCoeff\d+', name)
        coeffText = ""
        ptText = ""
        chargeText = ""
        if len(ptnum):
            ptText = f"p_{{T}} bin {ptnum[0].split('PtVBin')[1]}"
        if len(chargenum):
            chg = chargenum.split("genQ")[1]
            chargeText = "-" if "genQ0" in chargenum else "+" if "genQ1" in chargenum else "" # in case Z has a different convention
        if len(coeffnum):
            ncoeff = int(coeffnum[0].split("AngCoeff")[1]) - 1
            coeffText = f"A_{{{ncoeff}}}"
        scale = "#mu_{R}#mu_{F}" if "muRmuF" in name else "#mu_{R}" if "muR" in name else "#mu_{F}"
        return f"{boson}{chargeText} {coeffText} {ptText} {scale}"
    elif "CMS_" in name:
        return name
    elif name == "Z_nonClosure_parametrized_A_":
        return "muonScale_ZnonClosure"
    else:  
        return name

def niceNameHEPDATA(name):

    # try a simple but understandable form
    # use plain latex and not tlatex for math symbols: i.e. $\mu$ instead of #mu

    if re.match( "Fakes(Eta|EtaCharge|PtNorm|PtSlope)Uncorrelated.*",name):

        num = re.findall(r'\d+', name) # get number
        pfx = name.split(num[0])[1]    # split on number and read what's on the right
        leptonCharge = ""
        if len(pfx):
            if "Plus" in pfx or "Minus" in pfx:
                leptonCharge = "{lep}^{chs}".format(lep="\mu", # if "mu" in pfx else "e",
                                                    chs = "+" if "Plus" in pfx else "-")
            else:
                leptonCharge = "{lep}".format(lep="\mu")# if "mu" in pfx else "e")
        tmpvar = ""
        FakesBins = []
        pt_or_eta = ""
        # eta binning for fakes systematics (PtNorm uses another one chosen below)
        FakesBins = [-2.4,-2.1,-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5,2.1,2.4]

        if "FakesEtaCharge" in name: 
            tmpvar = "$\eta$-norm-chargeUncorr"
            pt_or_eta = "\eta"
        elif "FakesEta" in name: 
            tmpvar = "$\eta$-norm"
            pt_or_eta = "\eta"
        elif "FakesPtSlope" in name: 
            tmpvar = "$p_{T}$-shape"    
            pt_or_eta = "\eta"
        elif "FakesPtNorm" in name: 
            tmpvar = "$p_{T}$-norm"
            pt_or_eta = "p_{T}"
            # for pt norm has to bin on pt
            FakesBins = [26, 33, 36, 40.5, 45, 50, 56] if "mu" in pfx else [30, 36, 40.5, 45, 50, 56]
            
        #print "{n} FakeBins[{m}]".format(n=name,m=num)
        n = int(num[0])
        vlow = FakesBins[n-1]
        vhigh = FakesBins[n]
        return "QCD bkg {var}, ${l}<{v}<{h}$, ${lepCh}$".format(var=tmpvar, l=vlow, v=pt_or_eta, h=vhigh, lepCh=leptonCharge)

    elif re.match(".*effStatTnP\d+.*",name):

        ## utility arrays for the effstat binning
        _etaBinsEffStatMu = [round(-2.4 + 0.1* x,1) for x in range(0,49)]

        num = re.findall(r'\d+', name) # get number
        n1 = num[0]
        pfx = name.split("effStatTnP"+str(n1))[1]    # split on second number and read what's on the right
        leptonCharge = ""
        if len(pfx):
            leptonCharge = "{lep}^{chs}".format(lep="\mu", # if "mu" in pfx else "e",
                                                chs = "+" if "Plus" in pfx else "-" if "Minus" in pfx else "")
        etaBinsEffStat = _etaBinsEffStatMu 
        etalow  = str(etaBinsEffStat[n1-1])
        etahigh = str(etaBinsEffStat[n1])
        return "Eff.stat. ${l}<\eta<{h}$, ${lepCh}$".format(l=etalow,h=etahigh,lepCh=leptonCharge)

    elif "muonL1Prefire" in name:
        num = re.findall(r'\d+', name) # get number
        #print "{n} L1PrefireEleEffSys[{m}]".format(n=name,m=num)
        n = int(num[1]) # use second number, the first is '1' in 'L1Prefire...'
        etabinsPrefire = [-2.4, -2.25, -2.1, -1.67, -1.24, -1.035, -0.83, -0.4, 0, 0.4, 0.83, 1.035, 1.24, 1.67, 2.1, 2.25, 2.4]
        low = etabinsPrefire[n]
        high = etabinsPrefire[n+1]
        return "L1-trigger muon eff.syst., ${l}<\eta<{h}$".format(l=low,h=high)

    elif re.match( "smooth(el|mu)scale.*",name):
        num = re.findall(r'\d+', name) # get number 
        n = 0
        n2 = 0
        n3 = 0
        lep = "\mu" if "smoothmu" in name else "e"
        if "scaleStat" in name: 
            n = int(num[0])
            return "$p_{{T}}^{{{lep}}}$ scale stat.{n}".format(lep=lep,n=n)
        else:
            n = int(num[0])
            n2 = int(num[1])
            etabinsPtSyst = [0.0, 2.1, 2.4] if "smoothmu" in name else [0.0, 1.0, 1.5, 2.1, 2.4]
            # match the 'P' to select positive eta side
            if re.match(".*etaside\d+P(plus|minus)*",name): 
                low = str(etabinsPtSyst[n2])
                high = str(etabinsPtSyst[n2+1])
            else:
                low = "-"+str(etabinsPtSyst[n2+1])
                high = "-"+str(etabinsPtSyst[n2])
            return "$p_{{T}}^{{{lep}}}$ scale syst.{n}, ${l}<\eta<{h}${ch}".format(lep=lep,n=n,l=low,h=high,ch=", charge +" if "plus" in name else ", charge -" if "minus" in name else "")

    elif "TnPEffSyst" in name or "TestEffSyst" in name:
        num = re.findall(r'\d+', name) # get number
        #print "{n} EffSyst[{m}]".format(n=name,m=num)
        n = int(num[0])
        etabinsEffSyst = [0,1.0,1.5,2.4] if "mu" in name else [0,1.0,1.5,1.9,2.1,2.4]
        low = etabinsEffSyst[n]
        high = etabinsEffSyst[n+1]
        return "eff.syst., ${l}<|\eta|<{h}$, ${lep}$".format(lep="\mu" if "mu" in name else "e",l=low,h=high)

    elif "fsr" in name:
        return "QED final state radiation, ${lep}$".format(lep="\mu" if any(x in name for x in ["Mu", "mu"]) else "e")

    elif "massShift" in name:
        num = re.findall(r'\d+', name) # get number
        n = int(num[0])
        return "$m_W$ (%d MeV shift)" % d

    elif any(x in name for x in ["muR", "muF", "muRmuF"]):
        scale = "$\mu_{R}\mu_{F}$" if "muRmuF" in name else "$\mu_{R}$" if "muR" in name else "$\mu_{F}$"
        charge = "+" if "Plus" in name else "-" if "Minus" in name else ""
        if any(x == name for x in ["muR", "muF", "muRmuF"]):
            return "{s} (Drell-Yan bkg)".format(s=scale)
        else:
            num = re.findall(r'\d+', name) # get number
            n = int(num[0]) # goes from 1 to 10
            ptWbins = [0.0, 2.9, 4.7, 6.7, 9.0, 11.8, 15.0, 20.1, 27.2, 40.2, 13000.0]
            low = ptWbins[n-1]
            high = ptWbins[n]
            wch = "$W^{{{c}}}$".format(c=charge)
            txt = "(W signal and $W\\rightarrow\\tau\\nu$ bkg)"
            if n == 10:
                return "{s} {w}, $p_{{T}}^{{W}}>{l}$ {t}".format(w=wch,s=scale,l=low,t=txt)
            else:
                return "{s} {w}, ${l}<p_{{T}}^{{W}}<{h}$ {t}".format(w=wch,s=scale,l=low,h=high,t=txt)

    elif "alpha" in name:
        return "$\\alpha_{S}$"

    elif "pdf" in name:
        num = re.findall(r'\d+', name) # get number                                               
        n = int(num[0]) # goes from 1 to 10            return "$\alpha_{S}$"
        return "Hessian {i}".format(i=n)

    elif name.startswith("CMS_"):
        if "lumi" in name:
            return "luminosity"
        elif "Tau" in name:
            return "$W\\rightarrow\\tau\\nu$ bkg norm."
        elif "VV" in name:
            return "Diboson bkg norm."
        elif "Top" in name:
            return "t quark bkg norm."
        elif "flips" in name:
            return "Charge flips bkg norm."
        elif "bkg_lepeff" in name:
            return "eff.syst. bkg, ${l}$".format(l="e" if "We" in name else "\mu")
        elif "lepVeto" in name:
            return "second lepton veto, ${l}$".format(l="e" if "We" in name else "\mu")
        else:
            return name

    else:  
        return name
        

if __name__ == "__main__":

    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetPaintTextFormat('.3f')

    #date = datetime.date.today().isoformat()
    # these are no longer needed anymore for this fit, except for channelnone
    filter_matrixType_poiPostfix = {"channelpmaskedexp"     : "pmaskedexp",
                                    "channelpmaskedexpnorm" : "pmaskedexpnorm",
                                    "channelsumpois"        : "sumxsec",
                                    "channelsumpoisnorm"    : "sumxsecnorm",
                                    "channelchargepois"     : "chargeasym",
                                    "channelchargemetapois" : "chargemetaasym",
                                    "channelratiometapois"  : "ratiometaratio",
                                    "channelpolpois"        : "a4",  #  check if it is correct
                                    "channelnone"           : "pmaskedexp", # dummy, there are no POIs in this case
                                    "channelnois"           : "pmaskedexp", # dummy, there are no POIs in this case
                                    "channelmu"             : "mu",
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('fitresult', type=str, nargs=1, help="fitresult.root file from combinetf")
    parser.add_argument('-o','--outdir', default='', type=str, help='output directory to save the matrix')
    parser.add_argument('-p','--params', default='', type=str, help='parameters for which you want to show the correlation matrix. comma separated list of regexps')
    parser.add_argument('-t','--type'  , default='hessian', choices=['toys', 'scans', 'hessian'], type=str, help='which type of input file: toys(default),scans, or hessian')
    parser.add_argument(     '--postfix', default='', type=str, help='Postfix for the correlation matrix')
    parser.add_argument(     '--uniqueString',  default=None, required=True, type=str, help='Keyword for canvas name to uniquely identify the output plot')
    parser.add_argument(     '--nContours', default=51, type=int, help='Number of contours in palette. Default is 51 (keep it odd: no correlation is white with our palette)')
    parser.add_argument(     '--palette'  , default=0, type=int, help='Set palette: default is a built-in one, 55 is kRainbow')
    parser.add_argument(     '--vertical-labels-X', dest='verticalLabelsX',    default=False, action='store_true', help='Set labels on X axis vertically (sometimes they overlap if rotated)')
    parser.add_argument(     '--title'  , default='', type=str, help='Title for matrix ')
    parser.add_argument(     '--show-more-correlated' , dest='showMoreCorrelated',    default=0, type=int, help='Show the N nuisances more correlated (in absolute value) with the parameters given with --params. If 0, do not do this part')
    parser.add_argument('-m','--matrix-type', dest='matrixType',    default='channelnone', choices=list(filter_matrixType_poiPostfix.keys()), type=str, help='Select which matrix to read from file')
    parser.add_argument(     '--margin',  default='0.2,0.16,0.07,0.2', type=str, help='Pass canvas margin as "left,right,top,bottom"')
    parser.add_argument(     '--textMargin',  default='0.2', type=str, help='Canvas bottom and left margins (same value, to make text fit)')
    parser.add_argument(     '--canvasSize', default='', type=str, help='Pass canvas dimensions as "width,height" ')
    parser.add_argument(     '--show-all-nuisances', dest='showAllNuisances',  action='store_true', help='Show all nuisances in the matrix (e.g. to prepare HEPdata entries): this implies that option --params is only used to filter POIs (for fixed POI fit with no real POI it is suggested using --params "NOTEXISTING", otherwise leaving --params empty makes all nuisances be treated as POI for the sake of building the matrix)')
    parser.add_argument('--which-matrix',  dest='whichMatrix', choices=["both","covariance","correlation"], default='correlation', type=str, help='Which matrix: covariance|correlation|both')
    parser.add_argument(     '--skipLatexOnTop', action='store_true', help='Do not write "CMS blabla" on top (mainly useful when a title is needed)')
    parser.add_argument(     '--use-hepdata-labels', dest='useHepdataLabels',    default=False, action='store_true', help='Write axis labels using latex for hepdata, with some name polishing')
    args = parser.parse_args()


    ## light blue to orange
    # ROOT.TColor.CreateGradientColorTable(3,
    #                                   array ("d", [0.00, 0.50, 1.00]),
    #                                   ##array ("d", [1.00, 1.00, 0.00]),
    #                                   ##array ("d", [0.70, 1.00, 0.34]),
    #                                   ##array ("d", [0.00, 1.00, 0.82]),
    #                                   array ("d", [0.00, 1.00, 1.00]),
    #                                   array ("d", [0.34, 1.00, 0.65]),
    #                                   array ("d", [0.82, 1.00, 0.00]),
    #                                   255,  0.95)

    # dark blue to red
    ROOT.TColor.CreateGradientColorTable(3,
                                         array ("d", [0.00, 0.50, 1.00]),
                                         array ("d", [0.00, 1.00, 1.00]),
                                         array ("d", [0.00, 1.00, 0.00]),
                                         array ("d", [1.00, 1.00, 0.00]),
                                         255,  0.95)


    if args.outdir:
        createPlotDirAndCopyPhp(args.outdir)
    else:
        print("You must pass an output folder with option -o")
        quit()

        
    pois_regexps = list(args.params.split(','))
    print(f"Filtering parameters with the following regex: {pois_regexps}")

    params = []; indices = []

    ## directly store the mean and RMS into a dictionary
    fitvals = {}; fiterrs = {}

    cov = {}; corr = {}

    nNuisances = 0
    nPois = 0

    ### GET LIST OF PARAMETERS THAT MATCH THE SPECIFIED OPTION IN THE TOYFILE
    if args.type == 'toys':
        toyfile = ROOT.TFile(args.fitresult[0], 'read')
        _tree = toyfile.Get('fitresults')
        lol = _tree.GetListOfLeaves()

        for l in lol:
            ## skip a bunch of those we don't want
            if '_err'   in l.GetName(): continue
            if '_minos' in l.GetName(): continue
            if '_gen'   in l.GetName(): continue
            if '_In'    in l.GetName(): continue
            for poi in pois_regexps:
                if re.match(poi, l.GetName()):
                    ## draw the parameter into a histogram
                    _tree.Draw(l.GetName()+'>>h_'+l.GetName())
                    ## find that histogram and clone it
                    h = ROOT.gROOT.FindObject('h_'+l.GetName()).Clone()
                    ## store mean and rms into the dictionaries from before
                    fitvals[l.GetName()] = h.GetMean()
                    fiterrs[l.GetName()] = h.GetRMS()
                    ## also keep a list of the parameter names, for sorting
                    params.append(l.GetName())

    elif args.type == 'hessian':
        hessfile = ROOT.TFile(args.fitresult[0],'read')
        suffix = args.matrixType
        corrmatrix = hessfile.Get('correlation_matrix_'+suffix)
        covmatrix  = hessfile.Get('covariance_matrix_'+suffix)
        for ib in range(1,corrmatrix.GetNbinsX()+1):
            #if nNuisances > 5 and nPois > 10: break # only for tests
            for poi in pois_regexps:
                if re.match(poi, corrmatrix.GetXaxis().GetBinLabel(ib)):
                    ## store mean and rms into the dictionaries from before
                    ## also keep a list of the parameter names, for sorting
                    params .append(corrmatrix.GetXaxis().GetBinLabel(ib))
                    indices.append(ib)
                    nPois += 1
                elif args.showAllNuisances:
                    params .append(corrmatrix.GetXaxis().GetBinLabel(ib))
                    indices.append(ib)
                    nNuisances += 1

    if args.showAllNuisances:
        print(f"nPois = {nPois}")
        print(f"nNuisances = {nNuisances}")

    poiPostfix = filter_matrixType_poiPostfix[args.matrixType]

    print("Preparing list of parameters to build the matrix ...")

    for ip1, p1 in enumerate(params):
        for ip2, p2 in enumerate(params):
            if args.type == 'toys':
                var = '({x}-{x0})*({y}-{y0})'.format(x=p1,x0=fitvals[p1],y=p2,y0=fitvals[p2])
                _tree.Draw('{var}>>h_{x}_{y}'.format(var=var,x=p1,y=p2))
                h = ROOT.gROOT.FindObject('h_{x}_{y}'.format(x=p1,y=p2)).Clone()
                cov [(p1,p2)] = h.GetMean()
                corr[(p1,p2)] = cov[(p1,p2)]/(fiterrs[p1]*fiterrs[p2])
                # this part is obsolete and might not have all the proper scaling factors used for hessian below
            elif args.type == 'hessian':
                cov [(p1,p2)] = covmatrix.GetBinContent(indices[ip1],indices[ip2])
                corr[(p1,p2)] = corrmatrix.GetBinContent(indices[ip1],indices[ip2])
        

    print(f"===> Build covariance matrix from this set of params: {params}")

    p_tmp = set(params)
    params = list(p_tmp)

    ## sort the floatParams. alphabetically
    params = sorted(params)
    params = sorted(params, key = lambda x: 0 if "qGen1" in x else 1 if "qGen0" in x else 2)
    params = sorted(params, key= lambda x: 1 if x.endswith(poiPostfix) else 0)

    # sort if not using all params (otherwise the order should be already ok, as it is taken from the original matrix)
    if not args.showAllNuisances:

        params = sorted(params, key= lambda x: utilities.getNFromString(x, chooseIndex=0) if 'pdf' in x else 0)
        #params = sorted(params, key= lambda x: int(re.sub('\D','',x)) if ('muRmuF' in x and x != "muRmuF")  else 0)
        #params = sorted(params, key= lambda x: int(re.sub('\D','',x)) if ('muR' in x and x != "muR" and 'muRmuF' not in x) else 0)
        #params = sorted(params, key= lambda x: int(re.sub('\D','',x)) if ('muF' in x and x != "muF" and 'muRmuF' not in x) else 0)
        #params = sorted(params, key= lambda x: utilities.getNFromString(x) if 'effStat' in x else 0)            
        #params = sorted(params, key= lambda x: lepInFakeSystForSort(x) if 'Fakes' in x else 0)   
        # sort by charge if needed     

    print('='*30)
    print(f"sorted params = {params}")
    print('='*30)
    print("Now going to create the matrix")
    print('='*30)
    ch = 1200
    cw = 1200
    if args.canvasSize:
        cw = int(args.canvasSize.split(',')[0])
        ch = int(args.canvasSize.split(',')[1])
    c = ROOT.TCanvas("c","",cw,ch,)
    c.SetGridx()
    c.SetGridy()
    if args.nContours: ROOT.gStyle.SetNumberContours(args.nContours)
    if args.palette:   ROOT.gStyle.SetPalette(args.palette)

    clm = 0.15
    crm = 0.16
    cbm = 0.15
    ctm = 0.07
    if args.margin:
        clm,crm,ctm,cbm = (float(x) for x in args.margin.split(','))
    if args.textMargin:
        clm = float(args.textMargin)
        cbm = float(args.textMargin)
    c.SetLeftMargin(clm)
    c.SetRightMargin(crm)
    c.SetBottomMargin(cbm)
    c.SetTopMargin(ctm)

    ## make the new, smaller TH2D correlation matrix
    nbins = len(params)
    th2_sub = ROOT.TH2D('sub_corr_matrix', 'correlation matrix', nbins, 0., nbins, nbins, 0., nbins)
    th2_cov = ROOT.TH2D('sub_cov_matrix',  'covariance matrix', nbins, 0., nbins, nbins, 0., nbins)

    if 'pdf' in args.params:
        #th2_sub.SetTitle('correlations of PDF nuisance parameters')
        th2_sub.GetXaxis().SetLabelSize(0.025)
        th2_sub.GetYaxis().SetLabelSize(0.025)
        #th2_cov.SetTitle('covariance of PDF nuisance parameters')
        th2_cov.GetXaxis().SetLabelSize(0.025)
        th2_cov.GetYaxis().SetLabelSize(0.025)

    th2_sub.GetXaxis().SetTickLength(0.)
    th2_sub.GetYaxis().SetTickLength(0.)
    th2_cov.GetXaxis().SetTickLength(0.)
    th2_cov.GetYaxis().SetTickLength(0.)
    
    ## pretty nested loop. enumerate the tuples
    nParams = len(params)
    # set axis labels
    print("Setting Labels")
    for i,x in enumerate(params):
        sys.stdout.write('Row {num}/{tot}   \r'.format(num=i,tot=nParams))
        sys.stdout.flush()
        if args.useHepdataLabels:
            new_x = niceNameHEPDATA(x)
        else:
            new_x = niceName(x)
        th2_sub.GetXaxis().SetBinLabel(i+1, new_x)
        th2_sub.GetYaxis().SetBinLabel(i+1, new_x)
        th2_cov.GetXaxis().SetBinLabel(i+1, new_x)
        th2_cov.GetYaxis().SetBinLabel(i+1, new_x)
         
    print("Setting Values")
    for i,x in enumerate(params):
        for j,y in enumerate(params):
            if j>i: break
            sys.stdout.write('Row {num}/{tot}   Column {col}/{tot}   \r'.format(num=i,tot=nParams, col=j))
            sys.stdout.flush()
            ## note that the matrix is symmetric
            if not args.whichMatrix == "covariance":
                th2_sub.SetBinContent(i+1, j+1, corr[(x,y)])
                th2_sub.SetBinContent(j+1, i+1, corr[(x,y)])
            if not args.whichMatrix == "correlation":
                th2_cov.SetBinContent(i+1, j+1, cov [(x,y)])
                th2_cov.SetBinContent(j+1, i+1, cov [(x,y)])

    th2_sub.GetZaxis().SetRangeUser(-1, 1)
    
    covMax = max(abs(th2_cov.GetMaximum()), abs(th2_cov.GetMinimum()))
    th2_cov.GetZaxis().SetRangeUser(-1.*covMax, covMax)

    print('='*30)
    print("Now finally drawing the matrix")
    print('='*30)
    matricesToPlot = []
    if args.whichMatrix == "both": 
        matricesToPlot = [th2_sub, th2_cov]
    elif args.whichMatrix == "covariance": 
        matricesToPlot = [th2_cov]
    else:
        matricesToPlot = [th2_sub]

    for im,tmp_mat in enumerate(matricesToPlot):

        if args.whichMatrix == "both":
            corcov = 'Correlation' if not im else 'Covariance'
        else:
            corcov = 'Covariance' if args.whichMatrix == "covariance" else "Correlation"

        tmp_mat.SetTitle("")
        if args.title: 
            tmp_mat.SetTitle(args.title)
            tmp_mat.GetZaxis().SetTitle(corcov)
            tmp_mat.GetZaxis().SetTitleSize(0.04)
            tmp_mat.GetZaxis().SetTitleOffset(1.2)
                       
            args.skipLatexOnTop = True
            
        if args.outdir:
            ROOT.gStyle.SetPaintTextFormat('1.2f')
            if len(params)<30: tmp_mat.Draw('colz text45')
            else: tmp_mat.Draw('colz')

            lat = ROOT.TLatex()
            lat.SetNDC(); lat.SetTextFont(42)
            if not args.skipLatexOnTop:
                offsetLatex = c.GetLeftMargin()-0.15
                rightTextOffset = 0.58
                lat.DrawLatex(0.15+offsetLatex, 0.95, '#bf{CMS}') #it{Preliminary}')
                lat.DrawLatex(rightTextOffset+offsetLatex, 0.95, '16.8 fb^{-1} (13 TeV)')

            if args.verticalLabelsX: tmp_mat.LabelsOption("v","X")
            if nbins >= 20: tmp_mat.LabelsOption("v","X")

            paramsName = args.uniqueString

            suff = '' if not args.postfix else '_'+args.postfix
            outfname = args.outdir+'/small{corcov}_{pn}{suff}'.format(suff=suff,pn=paramsName,corcov=corcov)
            for i in ['pdf', 'png']:
                c.SaveAs('{ofn}.{i}'.format(ofn=outfname,i=i))
            # save matrix in root file
            matRootFile = ROOT.TFile.Open("{ofn}.root".format(ofn=outfname),"recreate")
            matRootFile.cd()
            tmp_mat.Write()
            matRootFile.Close("matrix{corcov}".format(corcov=corcov))


    if args.showMoreCorrelated:
        print("Option --show-more-correlated is not yet implemented")
        pass
