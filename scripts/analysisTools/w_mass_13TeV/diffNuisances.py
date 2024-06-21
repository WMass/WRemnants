#!/usr/bin/env python3

## example with loop with bash to do different plots at once
# for ic in {0..8}; do echo "python w-mass-13TeV/diffNuisances.py --infile /scratch/mciprian/CombineStudies/WMass/qcdScale_byHelicityPt_passSystToFakes_deepMET/testToys_debugToy_noQCDscaleFakes/testChargeComb/fit/hessian/fitresults_123456789_Asimov_bbb1_cxs0.root --outdir plots/fromMyWremnants/Wmass_fit/deepMET_passSystToFakes/qcdScale_byHelicityPt/debugToy_noQCDscaleFakes/diffNuisances/  -a --format html --type hessian  --suffix Asimov --pois '.*QCDscale_Coeff${ic}.*genVplus' --uniqueString 'QCDscales_Coeff${ic}_chargePlus' --y-setting -1.0 -0.5 0 0.5 1.0 " | bash; done

# python w-mass-13TeV/diffNuisances.py --infile /scratch/mciprian/CombineStudies/WMass/10Sept2022_qcdBkgVar/qcdScale_byHelicityPt/nominal/fit/hessian/fitresults_123456789_Asimov_bbb1_cxs0.root --outdir plots/fromMyWremnants/Wmass_fit/10Sept2022_qcdBkgVar/qcdScale_byHelicityPt/diffNuisances/  -a --format html --type hessian  --suffix Asimov --pois '.*prefire' --uniqueString 'prefire' --y-setting -1.0 -0.5 0 0.5 1.0 --bm 0.35

# python w-mass-13TeV/diffNuisances.py --infile /scratch/mciprian/CombineStudies/WMass/10Sept2022_qcdBkgVar/qcdScale_byHelicityPt/nominal/fit/hessian/fitresults_123456789_Asimov_bbb1_cxs0.root --outdir plots/fromMyWremnants/Wmass_fit/10Sept2022_qcdBkgVar/qcdScale_byHelicityPt/diffNuisances/  -a --format html --type hessian  --suffix Asimov --pois '.*QCDscale_Coeff0.*genVplus' --uniqueString 'QCDscales_Coeff0_chargePlus' --y-setting -1.0 -0.5 0 0.5 1.0; python w-mass-13TeV/diffNuisances.py --infile /scratch/mciprian/CombineStudies/WMass/10Sept2022_qcdBkgVar/qcdScale_byHelicityPt/nominal/fit/hessian/fitresults_123456789_Asimov_bbb1_cxs0.root --outdir plots/fromMyWremnants/Wmass_fit/10Sept2022_qcdBkgVar/qcdScale_byHelicityPt/diffNuisances/  -a --format html --type hessian  --suffix Asimov --pois '.*QCDscale_Coeff1.*genVplus' --uniqueString 'QCDscales_A0_chargePlus' --y-setting -1.0 -0.5 0 0.5 1.0; python w-mass-13TeV/diffNuisances.py --infile /scratch/mciprian/CombineStudies/WMass/10Sept2022_qcdBkgVar/qcdScale_byHelicityPt/nominal/fit/hessian/fitresults_123456789_Asimov_bbb1_cxs0.root --outdir plots/fromMyWremnants/Wmass_fit/10Sept2022_qcdBkgVar/qcdScale_byHelicityPt/diffNuisances/  -a --format html --type hessian  --suffix Asimov --pois '.*QCDscale_Coeff5.*genVplus' --uniqueString 'QCDscales_A4_chargePlus' --y-setting -1.0 -0.5 0 0.5 1.0

import re, os, math
import datetime
from optparse import OptionParser

from subMatrix import niceName, niceNameHEPDATA

import argparse

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

def sortEffSyst(name):
    sortEffSystDict = {"idip" : 0,
                       "iso"  : 1,
                       "reco" : 2,
                       "tracking" : 3,
                       "trigger"  : 4}
    for k,v in sortEffSystDict.items():
        if k in name:
            return v
    return -1

def sortByCharge(name):
    return 0 if "minus" in name else 1 if "plus" in name else 2

def sortParameters(params):
    
    params = sorted(params)
    # the sorting below assumes that one would use systematics from a specific group, not a totally random list, otherwise the order might end up a bit scrambled
    if any(re.match('pdf.*',x) for x in params):
        # generally there will be alphaS along with pdfs
        # put alpha at last position
        params = sorted(params, key= lambda x: 0 if 'AlphaS' in x else utilities.getNFromString(x, chooseIndex=0), reverse=False)
    elif any(re.match('.*QCDscale.*',x) for x in params):
        params = sorted(params, key= lambda x: utilities.getNFromString(x,useAll=True) if 'QCDscale' in x else 0)
        params = sorted(params, key= lambda x: 1 if re.match(".*muR$",x) else 2 if re.match(".*muRmuF$",x) else 3 if re.match(".*muF$",x) else 0)
    elif any(re.match('.*effStat.*',x) for x in params):
        params = sorted(params, key = lambda x: utilities.getNFromString(x,useAll=True), reverse=False)
    elif any(re.match('.*effSyst.*',x) for x in params):
        params = sorted(params, key = lambda x: utilities.getNFromString(x,useAll=True), reverse=False)
        params = sorted(params, key = lambda x: sortEffSyst(x))
    elif any(re.match('.*prefire.*',x) for x in params):
        params = sorted(params, key = lambda x: utilities.getNFromString(x,chooseIndex=0))
    elif any(re.match('.*CMS_scale_m.*',x) for x in params):
        params = sorted(params, key = lambda x: utilities.getNFromString(x,chooseIndex=0))
    elif any(re.match('.*Z_nonClosure.*',x) for x in params):
        params = sorted(params, key = lambda x: utilities.getNFromString(x,chooseIndex=0))
        params = sorted(params, key = lambda x: 0 if "_A_" in x else 1)
    elif any(re.match('.*polVar.*',x) for x in params):
        params = sorted(params, key = lambda x: (sortByCharge(x), 8, utilities.getNFromString(x,chooseIndex=0)) if "_UL_" in x else (sortByCharge(x), utilities.getNFromString(x,chooseIndex=0), utilities.getNFromString(x,chooseIndex=1)))
    elif any(re.match('.*ZmuonVeto.*',x) for x in params):
        params = sorted(params, key= lambda x: utilities.getNFromString(x), reverse=False)
    elif any(re.match('.*FakeRate.*',x) for x in params):
        params = sorted(params, key = lambda x: (utilities.getNFromString(x,chooseIndex=1), utilities.getNFromString(x,chooseIndex=0), utilities.getNFromString(x,chooseIndex=2)))
    return params

if __name__ == "__main__":

    parser = common_plot_parser()
    parser.add_argument('infile', nargs=1, type=str, help='file with the fitresult')
    parser.add_argument(      '--expected-infile'        , dest='expInfile'     , default=''        , type=str, help='file with the fitresult for expected, to plot together with observed (still to be implemented)')
    parser.add_argument('-o','--outdir', dest='outdir', default=None, type=str, help='If given, plot the pulls of the nuisances in this output directory')
    parser.add_argument("--vtol", "--val-tolerance", dest="vtol", default=0.30, type=float, help="Report nuisances whose value changes by more than this amount of sigmas")
    parser.add_argument("--vtol2", "--val-tolerance2", dest="vtol2", default=2.0, type=float, help="Report severely nuisances whose value changes by more than this amount of sigmas")
    parser.add_argument("-A", "--abs",      dest="absolute_values",    default=False,  action="store_true", help="Report also absolute values of nuisance values, not only the ones normalized to the input values")
    parser.add_argument("--no-all",      dest="show_all_parameters", action="store_false", help="If true print all nuisances, even the ones which are unchanged w.r.t. pre-fit values.")
    parser.add_argument("-p", "--pois",      default=None,   type=str,  help="Name of the nuisances to be plotted (comma separated list of regexps)")
    parser.add_argument("-x", "--excludeRegexp", default=None,  type=str,  help="Exclude names matching this regular expression, after filtering a list with --pois")
    parser.add_argument("-f", "--format", default="html", choices=['text', 'latex', 'html'], type=str,  help="Output format (in addition to pdf,png)")
    parser.add_argument(     '--postfix', default='', type=str, help='postfix for the correlation matrix')
    parser.add_argument(     '--uniqueString', default=None, required= True, type=str, help='Unique name to identify the output file names, which is nuisances_XXX_SUFF.EXT, with XXX passed by this option')
    parser.add_argument('-t', '--type',  default='hessian', choices=["hessian", "scans", "toys"], type=str, help='run the plot from which postfit? toys/scans/hessian')
    parser.add_argument('--bm', '--bottom-margin' , dest='setBottomMargin'     , default=0.3        , type=float, help='Bottom margin for the canvas')
    parser.add_argument(     '--canvasSize',  default='', type=str, help='Pass canvas dimensions as "width,height". Default is 800,600, but it is automatically adjusted for large number of parameters')
    parser.add_argument(      "--y-title",      dest="ytitle",  default="Fit #theta - #theta_{0}",   type=str,  help="Title for Y axis")
    parser.add_argument(      "--y-setting",    dest="ysetting",  nargs=5, default=[-5.0,-3,0,3,5.0], type=float,  help="Settings to customize y axis: pass list of ymin,yhalfd,ycen,yhalfu,ymax, where horizontal lines are drawn")
    parser.add_argument(      "--y-offset",    dest="yoffset",  nargs=2, default=[0.5,0.5], type=float,  help="Offset (positive value) for each side of the y axis, wrt to extremes of --y-setting")
    parser.add_argument(      "--postfitLegendLabelObs", type=str, default="Postfit observed",  help="Label for observed fit, usually when using --expected-infile (which can actually be another type of fit)")
    parser.add_argument(      "--postfitLegendLabelExp", type=str, default="Postfit expected",  help="Label for expected fit when using --expected-infile (which can actually be another type of fit)")
    parser.add_argument(     '--defaultYmax' , dest='defaultYmax', default=4.0, type=float, help='Set the default max value for y (but it is still overridden if some parameters are above)')
    #parser.add_argument(      "--y-setting",    dest="ysetting",  type=lambda s: [float(item) for item in s.split(',')], default="-5.0,-3,0,3,5.0",  help="Settings to customize y axis: comma-separated list of ymin,yhalfd,ycen,yhalfu,ymax, where horizontal lines are drawn")
    parser.add_argument('-R', "--rank-nuisances-by", dest="rankNuisancesBy",  default=None, choices=["", "pull", "sigma"],  type=str,  help="Rank nuisances based on either sigma or pull. It is devised to work with --pois '.*', but of course you can further filter nuisances and/or pois")
    parser.add_argument('-N','--show-N' , dest='showN',    default=0, type=int, help='To be used with -R: it shows only the N nuisances ranked. If not positive, no limit is used')    
    parser.add_argument(     '--lower-limit-pull' , dest='lowerLimitPull', default=-1.0, type=float, help='To be used with -R. Take only nuisances with pull above this value (in absolute value). If negative, use no limit')    
    parser.add_argument(     '--upper-limit-sigma' , dest='upperLimitSigma', default=-1.0, type=float, help='To be used with -R. Take only nuisances with postfit sigma below this value . If negative, use no limit')    
    parser.add_argument(     '--use-hepdata-labels', dest='useHepdataLabels',    default=False, action='store_true', help='Write axis labels using latex for hepdata, with some name polishing')
    # following options are needed to prepare list of nuisances and POIs consistently with covariance matrices for hepdata
    parser.add_argument(     '--prepare-as-covariance-matrix', dest='prepareAsCovarianceMatrix',    default=False, action='store_true', help='Sort as in subMatrix.py to get better correspondance between matrix and list of POIs and nuisance parameters, when preparing material for hepdata')
    args = parser.parse_args()

    infile = args.infile[0]
    infile_exp = args.expInfile
    plotObsWithExp = False
    if len(infile_exp):
        plotObsWithExp = True

    if not args.outdir:
        print("You must pass an output folder with option -o")
        quit()

    outdir_original = args.outdir
    outdir = createPlotDirAndCopyPhp(outdir_original, eoscp=args.eoscp)
    #valuesPrefit = dict((k,v) for k,v in valuesAndErrorsAll.items() if k.endswith('_gen'))
    pois_regexps = list(args.pois.split(','))

    if args.type == 'toys':
        valuesAndErrorsAll = utilities.getFromToys(infile,keepGen=True,params=pois_regexps)
        if plotObsWithExp:
            valuesAndErrorsAll_exp = utilities.getFromHessian(infile_exp,keepGen=True, params=pois_regexps)
    elif args.type == 'hessian':
        valuesAndErrorsAll = utilities.getFromHessian(infile,keepGen=True, params=pois_regexps)
        if plotObsWithExp:
            valuesAndErrorsAll_exp = utilities.getFromHessian(infile_exp,keepGen=True, params=pois_regexps)
    else:
        print("Sorry, scans method for option -t still has to be implemented")
        quit()
        
    print(f"Looking for regexp match {pois_regexps}")    
    valuesAndErrors = {}
    valuesAndErrors_exp = {}
    valuesPrefit = {}

    for ppatt in pois_regexps:
        regexp_ppatt = re.compile(ppatt)
        for (k,v) in valuesAndErrorsAll.items():
            if regexp_ppatt.match(k):
                if k.endswith('_gen'):
                    valuesPrefit   [k]  = v #dict((k,v) for k,v in valuesPrefit.items() if re.match(ppatt.replace('$','_gen'),k))
                else:
                    valuesAndErrors[k]  = v #dict((k,v) for k,v in valuesAndErrors.items() if re.match(ppatt,k) and not k.endswith('_gen'))
                    if plotObsWithExp:
                        valuesAndErrors_exp[k] = valuesAndErrorsAll_exp[k]

    params = valuesAndErrors.keys()

    if args.excludeRegexp != None:
        excl = re.compile(args.excludeRegexp)
        params = list(filter(lambda x: not excl.match(x), params))
    if len(params)==0:
        print("No parameters selected. Exiting.")
        quit()
            
    # if you are going to rank nuisances, just skip the sorting by names, to save some time
    if args.rankNuisancesBy:
        # this sorting will be managed later
        pass        
    else:
        params = sortParameters(params)
        print('='*30)
        print(f"Sorted params = {params}")


    nuis_p_i=0
    title = "#theta"

    hist_fit_s    = ROOT.TH1F("fit_s"   , '',len(params),0,len(params))
    hist_expfit_s = ROOT.TH1F("expfit_s", '',len(params),0,len(params))

    isFlagged = {}

    # maps from nuisance parameter name to the row to be printed in the table
    table = {}

    numberRankedNuisances = 0
    # loop over all fitted parameters
    for name in params:

        # keeps information to be printed about the nuisance parameter
        row = []
        flag = False

        ## this try catch catches the cases for which no gen (prefit) value exists. needed for the .* one
        try: 
            mean_p = valuesPrefit[name+'_gen'][0]
        except:
            #print(f"Exception caught: name = {name}")
            continue
        val_f,err_f = (valuesAndErrors[name][0],abs(valuesAndErrors[name][0]-valuesAndErrors[name][1]))
        if plotObsWithExp:
            expval_f,experr_f = (valuesAndErrors_exp[name][0],abs(valuesAndErrors_exp[name][0]-valuesAndErrors_exp[name][1]))
        else:
            expval_f,experr_f = val_f,err_f
        
        if args.absolute_values:
            valShift = val_f
            sigShift = err_f
        else:
            valShift = val_f - mean_p
            sigShift = err_f

        if args.rankNuisancesBy:
            if args.lowerLimitPull > 0.0:
                if abs(valShift) <= args.lowerLimitPull: continue
            if args.upperLimitSigma > 0.0:
                if sigShift >= args.upperLimitSigma: continue
            numberRankedNuisances += 1

        if outdir: 
            nuis_p_i+=1
            hist_fit_s.SetBinContent(nuis_p_i, val_f)
            hist_fit_s.SetBinError(nuis_p_i,err_f)
            hist_expfit_s.SetBinContent(nuis_p_i, expval_f)
            hist_expfit_s.SetBinError(nuis_p_i, experr_f)
            thisname = niceNameHEPDATA(name) if args.useHepdataLabels else niceName(name)
            hist_fit_s.GetXaxis().SetBinLabel(nuis_p_i, thisname)
            hist_expfit_s.GetXaxis().SetBinLabel(nuis_p_i, thisname)

        row += [" %+4.4f, %4.4f" % (valShift, sigShift)]

        if abs(val_f - mean_p) > args.vtol2*sigShift:

            # severely report this nuisance:
            # 
            # the best fit moved by more than 2.0 sigma or the uncertainty (sigma)
            # changed by more than 50% (default thresholds) w.r.t the prefit values
            isFlagged[name] = 2
            flag = True

        elif abs(val_f - mean_p) > args.vtol*sigShift:
            # report this nuisance:
            # 
            # the best fit moved by more than 0.3 sigma or the uncertainty (sigma)
            # changed by more than 10% (default thresholds) w.r.t the prefit values
            if args.show_all_parameters: isFlagged[name] = 1
            flag = True

        elif args.show_all_parameters:
            flag = True

        if flag or args.show_all_parameters: table[name] = row
    
    #end of loop over all fitted parameters

    #----------
    # print the results
    #----------
    if args.rankNuisancesBy:
        addPostfix = "rankBy{what}".format(what=args.rankNuisancesBy)
        if args.postfix:
            args.postfix += "_{add}".format(add=addPostfix)
        else:
            args.postfix = addPostfix
            
    outnameNoExt = "{od}/nuisances_{ps}_{suff}".format(od=outdir, ps=args.uniqueString, suff=args.postfix)

    for ext in args.format.split(','):
        txtfilename = "{noext}.{ext}".format(noext=outnameNoExt, ext=ext)
        txtfile = open(txtfilename,'w')
     
        fmtstring = "%-40s     %15s"
        highlight = "*%s*"
        morelight = "!%s!"
        pmsub, sigsub = None, None
        if ext == 'text':
            txtfile.write(fmtstring % ('name', 's+b fit\n'))
        elif ext == 'latex':
            pmsub  = (r"(\S+) \+/- (\S+)", r"$\1 \\pm \2$")
            sigsub = ("sig", r"$\\sigma$")
            highlight = "\\textbf{%s}"
            morelight = "{{\\color{red}\\textbf{%s}}}"
            if args.absolute_values:
                fmtstring = "%-40s &  %15s & %30s \\\\"
                txtfile.write( "\\begin{tabular}{|l|r|r|r|} \\hline \n")
                txtfile.write( (fmtstring % ('name', 'pre fit', '$s+b$ fit')), " \\hline \n")
            else:
                fmtstring = "%-40s & %15s \\\\"
                txtfile.write( "\\begin{tabular}{|l|r|} \\hline \n")
                what = r"\Delta x/\sigma_{\text{in}}$, $\sigma_{\text{out}}/\sigma_{\text{in}}$"
                txtfile.write( fmtstring % ('', '$s+b$ fit\n'))
                txtfile.write((fmtstring % ('name', what)))
                txtfile.write(" \\hline \n")
        elif ext == 'html':
            pmsub  = (r"(\S+) \+/- (\S+)", r"\1 &plusmn; \2")
            sigsub = ("sig", r"&sigma;")
            highlight = "<b>%s</b>"
            morelight = "<strong>%s</strong>"
            txtfile.write("""
        <html><head><title>Comparison of nuisances</title>
        <style type="text/css">
            td, th { border-bottom: 1px solid black; padding: 1px 1em; }
            td { font-family: 'Consolas', 'Courier New', courier, monospace; }
            strong { color: red; font-weight: bolder; }
        </style>
        </head><body style="font-family: 'Verdana', sans-serif; font-size: 10pt;"><h1>Comparison of nuisances</h1>
        <table>
        """)
     
     
            if args.absolute_values:
                what = "x, &sigma;<sub>fit</sub>";
            else:
                what = "&Delta;x, &sigma;<sub>fit</sub>";
            txtfile.write("<tr><th>nuisance</th><th>signal fit<br/>%s</th>\n" % (what))
            fmtstring = "<tr><td><tt>%-40s</tt> </td><td> %-15s </td></tr>\n"
     
        names = table.keys()

        if args.rankNuisancesBy:    
            # rank by pull
            names = sorted(names, key = lambda x: abs(float((table[x][0].split(',')[0]).strip())), reverse=True)
            if args.rankNuisancesBy == "sigma":
                # sort by sigma, using the more constrained nuisances on top (i.e. lower sigma)
                names = sorted(names, key = lambda x: float((table[x][0].split(',')[1]).strip()))
        else:
            names = sortParameters(names)
            
        highlighters = { 1:highlight, 2:morelight };
        nameCounter = 0
        for n in names:
            if args.showN > 0 and nameCounter > args.showN: 
                break
            nameCounter += 1
            v = table[n]
            if ext == "latex": n = n.replace(r"_", r"\_")
            if pmsub  != None: v = [ re.sub(pmsub[0],  pmsub[1],  i) for i in v ]
            if sigsub != None: v = [ re.sub(sigsub[0], sigsub[1], i) for i in v ]
            if (n) in isFlagged: v[-1] = highlighters[isFlagged[n]] % v[-1]
            txtfile.write(fmtstring % (n, v[0]))
            txtfile.write('\n')
     
        if ext == "latex":
            txtfile.write(" \\hline\n\end{tabular}\n")
        elif ext == "html":
            txtfile.write("</table></body></html>\n")
        txtfile.close()
        print(f"Info: {ext} file {txtfilename} has been created")

    if outdir:
        line = ROOT.TLine()
        lat  = ROOT.TLatex(); lat.SetNDC(); lat.SetTextFont(42); lat.SetTextSize(0.04)
        ROOT.gStyle.SetOptStat(0)
        fout = ROOT.TFile.Open("{fn}.root".format(fn=outnameNoExt),"RECREATE")

        # need to shrink histogram, as some bins might have been removed when ranking. 
        # Also, use same sorting as table
        hist_fit_s_ranked = None
        hist_expfit_s_ranked = None
        nbins = len(params) # default
        if args.rankNuisancesBy:
            nbins = min(args.showN, len(names)) if args.showN > 0 else len(names)
            hist_fit_s_ranked    = ROOT.TH1D("fit_s_ranked"   ,'',nbins,0,nbins)
            hist_expfit_s_ranked    = ROOT.TH1D("expfit_s_ranked"   ,'',nbins,0,nbins)
            index_name = 1
            for n in names:
                if args.showN > 0 and index_name > args.showN:
                    break 
                else:
                    thisname = niceNameHEPDATA(n) if args.useHepdataLabels else niceName(n)
                    binNumber = hist_fit_s.GetXaxis().FindFixBin(thisname)
                    if binNumber == -1:
                        print(f"Error when filling hist_fit_s_ranked. Could not find label {n} from hist_fit_s")
                        quit()
                    val = hist_fit_s.GetBinContent(binNumber)
                    err = hist_fit_s.GetBinError(binNumber)
                    hist_fit_s_ranked.SetBinContent(index_name,val)
                    hist_fit_s_ranked.SetBinError(index_name,err)
                    hist_fit_s_ranked.GetXaxis().SetBinLabel(index_name,n)
                    hist_fit_s = hist_fit_s_ranked

                    expval = hist_expfit_s.GetBinContent(binNumber)
                    experr = hist_expfit_s.GetBinError(binNumber)
                    hist_expfit_s_ranked.SetBinContent(index_name,expval)
                    hist_expfit_s_ranked.SetBinError(index_name,experr)
                    hist_expfit_s_ranked.GetXaxis().SetBinLabel(index_name,n)
                    hist_expfit_s = hist_expfit_s_ranked
                    index_name += 1

        # customize canvas width a bit
        cw = 800
        ch = 600
        if nbins >= 100:
            cw = 2000            
        elif nbins >= 50:
            cw = 1200            
        if args.canvasSize:
            cw = int(args.canvasSize.split(',')[0])
            ch = int(args.canvasSize.split(',')[1])
        canvas_nuis = ROOT.TCanvas("nuisances", "nuisances", cw, ch)

        clm = 0.1 if nbins < 100 else 0.05
        crm = 0.05 if nbins < 100 else 0.02
        cbm = args.setBottomMargin
        ctm = 0.1

        canvas_nuis.SetTickx(1)
        canvas_nuis.SetTicky(1)
        canvas_nuis.SetLeftMargin(clm)
        canvas_nuis.SetRightMargin(crm)
        canvas_nuis.SetBottomMargin(cbm)
        canvas_nuis.SetTopMargin(ctm)

        xLabelSize = 0.045 if nbins < 20 else 0.035
        if nbins > 180:
            xLabelSize = 0.0
            cbm = 0.1
            hist_fit_s.GetYaxis().SetTitleOffset(0.40)
            hist_fit_s.SetTitle(args.uniqueString)
        elif nbins > 120:
            xLabelSize = 0.025
            
        if plotObsWithExp:            

            # to plot observed (could be pseudodata) and expected, but also to compare two different fits (need to modify central value)
            x = array('d',[float(i+1) for i in range(nbins)])

            zero   = array('d',[0.0 for i in range(nbins)])
            one    = array('d',[1.0 for i in range(nbins)])
            exone  = array('d', [0.4 for i in range(nbins)])

            # yexp  = array('d',[hist_expfit_s.GetBinContent(i) for i in range(1, 1+hist_expfit_s.GetNbinsX()) ])
            eyexp  = array('d',[hist_expfit_s.GetBinError(i) for i in range(1, 1+hist_expfit_s.GetNbinsX()) ])
            exexp  = array('d', [0.25 for i in range(nbins)])

            yobs  = array('d',[hist_fit_s.GetBinContent(i) for i in range(1, 1+hist_fit_s.GetNbinsX()) ])
            eyobs  = array('d',[hist_fit_s.GetBinError(i) for i in range(1, 1+hist_fit_s.GetNbinsX()) ])

            maxPull = max(list([abs(y) for y in yobs]))
            maxz = max(args.defaultYmax, math.ceil(maxPull))

            lat = ROOT.TLatex(); lat.SetNDC()
            lat.SetTextFont(42)
            lat.SetTextSize(0.05)
            
            #leg = ROOT.TLegend(clm, 0.02, 1.0 - crm, 0.08)
            leg = ROOT.TLegend(clm, 0.92, 1.0 - crm, 0.98)
            leg.SetNColumns(3)
            leg.SetColumnSeparation(0.1)
            #leg.SetFillStyle(0)
            leg.SetFillColor(0)
            leg.SetBorderSize(0)

            ROOT.gStyle.SetOptStat(0)
            dummyh = ROOT.TH1D('dummyh','', nbins, x[0]-0.5, x[-1]+0.5)
            dummyh.GetYaxis().SetRangeUser(-maxz,maxz)
            dummyh.GetXaxis().SetRangeUser(-1.5, nbins+1.5)
            #dummyh.GetXaxis().LabelsOption('v') # gives warning: TAxis::Sort:0: RuntimeWarning: Cannot sort. No labels
            dummyh.GetYaxis().SetTitle('Fit #theta - #theta^{0}')
            dummyh.GetYaxis().CenterTitle()
            dummyh.GetXaxis().SetLabelSize(xLabelSize)
            dummyh.GetXaxis().SetTitleFont(42)
            dummyh.GetXaxis().SetTitleSize(0.05)
            dummyh.GetXaxis().SetLabelFont(42)
            dummyh.GetYaxis().SetTitleFont(42)
            dummyh.GetYaxis().SetTitleSize(0.06)
            dummyh.GetYaxis().SetTitleOffset(0.6)
            dummyh.GetYaxis().SetLabelFont(42)
            for i in range(nbins):
                dummyh.GetXaxis().SetBinLabel(i+1, hist_fit_s.GetXaxis().GetBinLabel(i+1))
            dummyh.GetXaxis().LabelsOption("v")
            dummyh.Draw()

            # prefit (trivial, all 1)
            gr_prefit = ROOT.TGraphErrors(nbins,x,zero,exone,one)
            gr_prefit.SetTitle('')
            gr_prefit.SetFillColor(ROOT.kAzure+6)
            gr_prefit.SetMarkerColor(ROOT.kAzure+6)
            gr_prefit.Draw('P2')
            leg.AddEntry(gr_prefit,'Prefit','f')

            gr_prefit.GetXaxis().SetNdivisions(nbins+1, ROOT.kFALSE)
            gr_prefit.GetYaxis().SetNdivisions(2*int(maxz))

            # expected 
            gr_expected = ROOT.TGraphErrors(nbins,x,zero,exexp,eyexp)
            gr_expected.GetYaxis().SetRangeUser(-maxz,maxz)
            gr_expected.GetXaxis().SetRangeUser(-0.5,nbins+0.5)
            gr_expected.SetFillColor(ROOT.kOrange+7)
            gr_expected.SetMarkerColor(ROOT.kOrange+7)
            gr_expected.Draw('P2')
            leg.AddEntry(gr_expected,args.postfitLegendLabelExp,'f')

            # observed
            gr_observed = ROOT.TGraphErrors(nbins,x,yobs,zero,eyobs)
            gr_observed.GetYaxis().SetRangeUser(-maxz,maxz)
            gr_observed.GetXaxis().SetRangeUser(-0.5,nbins+0.5)
            gr_observed.SetMarkerStyle(ROOT.kFullCircle)
            gr_observed.SetMarkerSize(1)
            gr_observed.Draw('P EZ')
            leg.AddEntry(gr_observed,args.postfitLegendLabelObs)

            #lat.DrawLatex(clm, 0.95, '#bf{CMS} #it{Preliminary}')
            #lat.DrawLatex(0.80, 0.95, '16.8 fb^{-1} (13 TeV)')

            leg.Draw('same')

            line = ROOT.TLine()
            for z in range(-int(maxz),int(maxz)):
                line.DrawLine(x[0]-1.5, z, nbins+1.5, z)
                line.SetLineStyle(3)
                line.SetLineColor(ROOT.kBlack)

            for ext in ['png', 'pdf']:
                canvas_nuis.SaveAs("{noext}_compareObsAndExp.{ext}".format(noext=outnameNoExt, ext=ext))
                
        else:


            # to plot from a single file
            ymin,yhalfd,ycen,yhalfu,ymax = args.ysetting

            # deal with extreme cases with POIs
            maxPull = max(list([(hist_fit_s.GetBinContent(i)+hist_fit_s.GetBinError(i)) for i in range(1, 1+hist_fit_s.GetNbinsX())]))
            maxz = max(args.defaultYmax, math.ceil(maxPull))
            minz = min(-maxz, ymin-args.yoffset[0])
            maxz = max(maxz, ymax+args.yoffset[1])
            #hist_fit_s.GetYaxis().SetRangeUser(ymin-args.yoffset[0],ymax+args.yoffset[1])
            hist_fit_s.GetYaxis().SetRangeUser(-maxz, maxz)
            hist_fit_s.SetLineColor  (39)
            hist_fit_s.SetMarkerColor(ROOT.kGray+3)
            hist_fit_s.SetMarkerStyle(20)
            hist_fit_s.SetMarkerSize(1.0)
            hist_fit_s.SetLineWidth(2)
            hist_fit_s.Draw("PE1")
            hist_fit_s.GetYaxis().SetTitle(args.ytitle)
            hist_fit_s.GetYaxis().SetTitleSize(0.05)
            hist_fit_s.GetYaxis().SetTitleOffset(0.90)

            if nbins > 180:
                hist_fit_s.GetYaxis().SetTitleOffset(0.40)
                hist_fit_s.SetTitle(args.uniqueString)
            hist_fit_s.GetXaxis().SetLabelSize(xLabelSize)
            hist_fit_s.GetXaxis().LabelsOption("v")

            #lat.DrawLatex(0.10, 0.92, '#bf{CMS} #it{Preliminary}')
            #lat.DrawLatex(0.71 +(0.1-crm), 0.92, '16.8 fb^{-1} (13 TeV)')
            line.DrawLine(0., ycen, nbins, ycen)
            line.DrawLine(0., ymax, nbins, ymax)
            line.DrawLine(0., ymin, nbins, ymin)
            line.SetLineStyle(2);
            line.SetLineColor(ROOT.kRed)
            line.DrawLine(0., yhalfu, nbins, yhalfu)
            line.DrawLine(0., yhalfd, nbins, yhalfd)
            line.SetLineStyle(3)
            line.DrawLine(0., 1., nbins, 1.)
            line.DrawLine(0.,-1., nbins,-1.)
            hist_fit_s.Draw("PE1 same") ## draw again over the lines

            canvas_nuis.SetGridx()
            canvas_nuis.RedrawAxis()
            canvas_nuis.RedrawAxis('g')
            # leg=ROOT.TLegend(0.6,0.7,0.89,0.89)
            # leg.SetFillColor(0)
            # leg.SetTextFont(42)
            # leg.AddEntry(hist_fit_s,"S+B fit"   ,"EPL")
            # leg.Draw()
            #fout.WriteTObject(canvas_nuis)
            hist_fit_s.Write()
            fout.Close()

            for ext in ['png', 'pdf']:
                canvas_nuis.SaveAs("{noext}.{ext}".format(noext=outnameNoExt, ext=ext))

    copyOutputToEos(outdir_original, eoscp=args.eoscp)

