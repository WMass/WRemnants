
import sys,array,math,os
import numpy as np
import ctypes
import json
import copy

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptTitle(0)

sys.path.insert(0, "scripts/lowPU/")
sys.path.insert(0, "scripts/lowPU/recoil/")
import functions
import plotter
import recoilLibs_scipy as rls
from wremnants.datasets.datagroups import Datagroups

import lz4.frame
import narf



def ewk_perp_RawPFMET():

    procs = ["EWK"]
    tag = "ewk_perp"
    comp = "perp"
    baseDir = "%s/%s" % (outDir, tag)
    procLabel = "EWK #rightarrow #mu^{+}#mu^{#minus}"
    metLabel = "%s, U_{#perp}" % (met)
    functions.prepareDir(baseDir, False)
    recoilMin, recoilMax = -100, 100
    procNames = groups.getProcNames(to_expand=procs)
    bhist = readProc(groups, "recoil_corr_xy_perp_qTbinned", procs)
    rebin=2
    
    if False:
        fitCfg = {}
        fitCfg['func_name'] = "linear"
        fitCfg['parms_init'] = [0, 0]
        fitCfg['parms_cte'] = [False, False]
        rls.parameterizeMean(bhist, comp, baseDir, fitCfg, binning_qT, procLabel, metLabel, procNames=procNames, yMin=-2, yMax=2)
        return

    fitCfg = {} 
    fitCfg['func_name'] = "ewk_perp"
    fitCfg['func_parms_vals'] = [10, 20, 70]
    fitCfg['func_parms_cfg'] = [0, 0, 0]
    outDir_fits_v0 = "%s/fits_v0" % baseDir
    if False:
        rls.doFitMultiGauss_fit(bhist, comp, fitCfg, procLabel, metLabel, outDir_fits_v0, binning_qT, yMin=1e-5, yMax=1e3, recoilLow=recoilMin, recoilHigh=recoilMax, rebin=rebin)
        return
        
    outDir_param_v0 = "%s/params_v0" % baseDir
    if False:
        jsOut = {}
        outDir_param = outDir_param_v0
        functions.prepareDir(outDir_param, True)
        with open("%s/results.json" % outDir_fits_v0) as f: jsIn = json.load(f)
        
        fitF, params, cParams = "power", [9.90024e-02, 9.34622e-01, 9.34622e-01], [False, False, False]
        rls.parameterizeGauss(jsIn, jsOut, comp, "p0", fitF, params, outDir_param, binning_qT, procLabel, metLabel, cParams=cParams, fitMin=0, fitMax=100, yMax=100, xMax=100, yTitle = "#sigma_{1} (GeV)", fitOpts="NS W")

        fitF, params, cParams = "power", [4.66409e-01, 3.46158e-01, 9.34622e-01], [False, False, False]
        rls.parameterizeGauss(jsIn, jsOut, comp, "p1", fitF, params, outDir_param, binning_qT, procLabel, metLabel, cParams=cParams, fitMin=0, fitMax=100, yMax=100, xMax=100, yTitle = "#sigma_{2} (GeV)", fitOpts="NS W")
        
        fitF, params, cParams = "linear", [4.66409e-01, 1], [False, False]
        rls.parameterizeGauss(jsIn, jsOut, comp, "p2", fitF, params, outDir_param, binning_qT, procLabel, metLabel, cParams=cParams, fitMin=0, fitMax=100, yMax=200, xMax=100, yTitle = "#sigma_{3} (GeV)", fitOpts="NS W")

        #rls.addParam(jsOut, "p3", "[0]", [0.75])
        #rls.addParam(jsOut, "p4", "[0]", [0.20])
     
        jsOut['nParams'] = len(jsOut)
        functions.writeJSON("%s/results.json" % outDir_param_v0, jsOut)
        return
        

    if True:
        outDir_refit = "%s/params_refit" % baseDir
        outDir_refit_fits = "%s/fits_refit" % baseDir
        functions.prepareDir(outDir_refit, False)
        functions.prepareDir(outDir_refit_fits, False)
    
        
        jsIn = functions.loadJSON("%s/results.json" % outDir_param_v0)
        fitCfg = jsIn
        fitCfg['func_name'] = "ewk_perp_cond"
        jsOut = rls.combinedFit_scipy(bhist, comp, fitCfg, binning_qT, chisq_refit=False, recoilLow=recoilMin, recoilHigh=recoilMax, rebin=rebin)
        functions.writeJSON("%s/results_refit.json" % baseDir, jsOut)

        jsIn = functions.loadJSON("%s/results_refit.json" % baseDir)
        rls.doFitMultiGauss_plot(bhist, comp, jsIn, procLabel, metLabel, outDir_refit_fits, binning_qT, yMin=1e-5, yMax=1e3, recoilLow=recoilMin, recoilHigh=recoilMax, rebin=rebin)
        
        rls.plotParameter("p0", jsIn, outDir_refit, binning_qT, procLabel, metLabel, yMin=0, yMax=50, yTitle = "#sigma_{1} (GeV)")
        rls.plotParameter("p1", jsIn, outDir_refit, binning_qT, procLabel, metLabel, yMin=0, yMax=50, yTitle = "#sigma_{2} (GeV)")
        rls.plotParameter("p2", jsIn, outDir_refit, binning_qT, procLabel, metLabel, yMin=0, yMax=100, yTitle = "#sigma_{3} (GeV)")


def ewk_para_qT_RawPFMET():

    procs = ["EWK"]
    tag = "ewk_para_qT"
    comp = "para"
    baseDir = "%s/%s" % (outDir, tag)
    procLabel = "EWK #rightarrow #mu^{+}#mu^{#minus}"
    metLabel = "%s, U_{#parallel}" % (met)
    functions.prepareDir(baseDir, False)
    procNames = groups.getProcNames(to_expand=procs)
    bhist = readProc(groups, "recoil_corr_xy_para_qT_qTbinned", procs)
    
    recoilMin, recoilMax = -100, 100
    rebin=2
    if False:
        
        fitCfg = {}
        fitCfg['func_name'] = "pol6"
        fitCfg['parms_init'] = [2.19057e+00, -2.37865e-01, 4.02425e-02, -1.34963e-03, 2.19717e-05, -1.73520e-07, 5.31624e-10]
        fitCfg['parms_cte'] = [False, False, False, False, False, False, False]
        rls.parameterizeMean(bhist, comp, baseDir, fitCfg, binning_qT, procLabel, metLabel, procNames, yMin=0, yMax=50)
        return
    
    fitCfg = {} 
    fitCfg['func_name'] = "ewk_para_qT"
    fitCfg['func_parms_vals'] = [10, 25, 50, 0, 0, 0]
    fitCfg['func_parms_cfg'] = [0, 0, 0, 0, 0, 0]
    outDir_fits_v0 = "%s/fits_v0" % baseDir
    if False:
        rls.doFitMultiGauss_fit(bhist, comp, fitCfg, procLabel, metLabel, outDir_fits_v0, binning_qT, yMin=1e-5, yMax=1e3, rebin=rebin, recoilLow=recoilMin, recoilHigh=recoilMax)
        return
        
    outDir_param_v0 = "%s/params_v0" % baseDir
    if False:
        jsOut = {}
        outDir_param = outDir_param_v0
        functions.prepareDir(outDir_param, True)
        with open("%s/results.json" % outDir_fits_v0) as f: jsIn = json.load(f)
        
        fitF, params, cParams = "power", [1, 1, 0], [False, False, False]
        rls.parameterizeGauss(jsIn, jsOut, comp, "p0", fitF, params, outDir_param, binning_qT, procLabel, metLabel, cParams=cParams, fitMin=0, fitMax=100, yMax=100, xMax=100, yTitle = "#sigma_{1} (GeV)", fitOpts="NS W")

        fitF, params, cParams = "power", [1, 1, 0], [False, True, False]
        rls.parameterizeGauss(jsIn, jsOut, comp, "p1", fitF, params, outDir_param, binning_qT, procLabel, metLabel, cParams=cParams, fitMin=0, fitMax=100, yMax=100, xMax=100, yTitle = "#sigma_{2} (GeV)", fitOpts="NS W")
        
        fitF, params, cParams = "power", [1, 1, 0], [False, False, False]
        rls.parameterizeGauss(jsIn, jsOut, comp, "p2", fitF, params, outDir_param, binning_qT, procLabel, metLabel, cParams=cParams, fitMin=0, fitMax=100, yMax=100, xMax=100, yTitle = "#sigma_{3} (GeV)", fitOpts="NS W")


        fitF, params, cParams = "linear", [0, 0], [False, False, False]
        rls.parameterizeGauss(jsIn, jsOut, comp, "p3", fitF, params, outDir_param, binning_qT, procLabel, metLabel, cParams=cParams, fitMin=0, fitMax=100, yMax=100, xMax=100, yMin=-50, yTitle = "#mu_{1} (GeV)", fitOpts="NS W")
        
        fitF, params, cParams = "linear", [0, 0], [False, False, False]
        rls.parameterizeGauss(jsIn, jsOut, comp, "p4", fitF, params, outDir_param, binning_qT, procLabel, metLabel, cParams=cParams, fitMin=0, fitMax=100, yMax=100, xMax=100, yMin=-50, yTitle = "#mu_{2} (GeV)", fitOpts="NS W", cutOffMin=0)

        fitF, params, cParams = "linear", [0, 0], [False, False, False]
        rls.parameterizeGauss(jsIn, jsOut, comp, "p5", fitF, params, outDir_param, binning_qT, procLabel, metLabel, cParams=cParams, fitMin=0, fitMax=100, yMax=100, xMax=100, yMin=-50, yTitle = "#mu_{3} (GeV)", fitOpts="NS W", cutOffMin=0)


        rls.addParam(jsOut, "p6", "[0]", [0.33])
        rls.addParam(jsOut, "p7", "[0]", [0.33])

     
        jsOut['nParams'] = len(jsOut)
        functions.writeJSON("%s/results.json" % outDir_param_v0, jsOut)
        return
        

    if True:
        outDir_refit = "%s/params_refit" % baseDir
        outDir_refit_fits = "%s/fits_refit" % baseDir
        functions.prepareDir(outDir_refit, False)
        functions.prepareDir(outDir_refit_fits, False)
    
        with open("%s/results.json" % outDir_param_v0) as f: jsIn = json.load(f)
        fitCfg = jsIn
        fitCfg['func_name'] = "ewk_para_qT_cond"
        jsOut = rls.combinedFit_scipy(bhist, comp, fitCfg, binning_qT, chisq_refit=False, rebin=rebin, recoilLow=recoilMin, recoilHigh=recoilMax)
        with open("%s/results_refit.json" % baseDir, "w") as outfile: json.dump(jsOut, outfile, indent=4)
        
      
        
        with open("%s/results_refit.json" % baseDir) as f: jsIn = json.load(f)
        rls.doFitMultiGauss_plot(bhist, comp, jsIn, procLabel, metLabel, outDir_refit_fits, binning_qT, yMin=1e-5, yMax=1e3, recoilLow=recoilMin, recoilHigh=recoilMax, rebin=rebin)
        
        rls.plotParameter("p0", jsIn, outDir_refit, binning_qT, procLabel, metLabel, yMin=0, yMax=50, yTitle = "#sigma_{1} (GeV)")
        rls.plotParameter("p1", jsIn, outDir_refit, binning_qT, procLabel, metLabel, yMin=0, yMax=50, yTitle = "#sigma_{2} (GeV)")
        rls.plotParameter("p2", jsIn, outDir_refit, binning_qT, procLabel, metLabel, yMin=0, yMax=50, yTitle = "#sigma_{3} (GeV)")
        rls.plotParameter("p3", jsIn, outDir_refit, binning_qT, procLabel, metLabel, yMin=-5, yMax=5, yTitle = "#mu_{1} (GeV)")
        rls.plotParameter("p4", jsIn, outDir_refit, binning_qT, procLabel, metLabel, yMin=-5, yMax=5, yTitle = "#mu_{2} (GeV)")
        rls.plotParameter("p5", jsIn, outDir_refit, binning_qT, procLabel, metLabel, yMin=-5, yMax=5, yTitle = "#mu_{3} (GeV)")
        rls.plotParameter("p6", jsIn, outDir_refit, binning_qT, procLabel, metLabel, yMin=0, yMax=1, yTitle = "n_{1} (GeV)")
        rls.plotParameter("p7", jsIn, outDir_refit, binning_qT, procLabel, metLabel, yMin=0, yMax=1, yTitle = "n_{2} (GeV)")  
          
   
def ttbar_perp_RawPFMET():

    procs = ["Top"]
    tag = "ttbar_perp"
    comp = "perp"
    baseDir = "%s/%s" % (outDir, tag)
    procLabel = "TTbar #rightarrow #mu^{+}#mu^{#minus}"
    metLabel = "%s, U_{#perp}" % (met)
    functions.prepareDir(baseDir, False)
    procNames = groups.getProcNames(to_expand=procs)
    recoilMin, recoilMax = -100, 100
    rebin = 2
    bhist = readProc(groups, "recoil_corr_xy_perp_qTbinned", procs)
    
    if False:
        fitCfg = {}
        fitCfg['func_name'] = "linear"
        fitCfg['parms_init'] = [0, 0]
        fitCfg['parms_cte'] = [False, False]
        rls.parameterizeMean(bhist, comp, baseDir, fitCfg, binning_qT, procLabel, metLabel, procNames=procNames, yMin=-2, yMax=2)
        return

    fitCfg = {} 
    fitCfg['func_name'] = "ttbar_perp"
    fitCfg['func_parms_vals'] = [20, 50, 100]
    fitCfg['func_parms_cfg'] = [0, 0, 0]
    
    outDir_fits_v0 = "%s/fits_v0" % baseDir
    if False:
        rls.doFitMultiGauss_fit(bhist, comp, fitCfg, procLabel, metLabel, outDir_fits_v0, binning_qT, yMin=1e-5, yMax=1e3, rebin=rebin, recoilLow=recoilMin, recoilHigh=recoilMax)
        return
        
    outDir_param_v0 = "%s/param_v0" % baseDir
    if False:
        jsOut = {}
        outDir_param = outDir_param_v0
        functions.prepareDir(outDir_param, True)
        with open("%s/results.json" % outDir_fits_v0) as f: jsIn = json.load(f)
        
        fitF, params, cParams = "quadratic", [9.90024e-02, 9.34622e-01, 9.34622e-01], [False, False, False]
        rls.parameterizeGauss(jsIn, jsOut, comp, "p0", fitF, params, outDir_param, binning_qT, procLabel, metLabel, cParams=cParams, fitMin=0, fitMax=100, yMax=100, xMax=100, yTitle = "#sigma_{1} (GeV)", fitOpts="NS w")

        fitF, params, cParams = "quadratic", [4.66409e-01, 3.46158e-01, 9.34622e-01], [False, False, False]
        rls.parameterizeGauss(jsIn, jsOut, comp, "p1", fitF, params, outDir_param, binning_qT, procLabel, metLabel, cParams=cParams, fitMin=0, fitMax=100, yMax=100, xMax=100, yTitle = "#sigma_{2} (GeV)", fitOpts="NS w")
        
        fitF, params, cParams = "quadratic", [4.66409e-01, 3.46158e-01, 9.34622e-01], [False, False, False]
        rls.parameterizeGauss(jsIn, jsOut, comp, "p2", fitF, params, outDir_param, binning_qT, procLabel, metLabel, cParams=cParams, fitMin=0, fitMax=100, yMax=200, xMax=100, yTitle = "#sigma_{3} (GeV)", fitOpts="NS w")

        rls.addParam(jsOut, "p3", "[0]", [0.5])
        rls.addParam(jsOut, "p4", "[0]", [0.4])
     
        jsOut['nParams'] = len(jsOut)
        functions.writeJSON("%s/results.json" % outDir_param_v0, jsOut)
        return
        
    if True:
        outDir_refit = "%s/params_refit" % baseDir
        outDir_refit_fits = "%s/fits_refit" % baseDir
        functions.prepareDir(outDir_refit, False)
        functions.prepareDir(outDir_refit_fits, False)
    
        jsIn = functions.loadJSON("%s/results.json" % outDir_param_v0)
        fitCfg = jsIn
        fitCfg['func_name'] = "ttbar_perp_cond"
        jsOut = rls.combinedFit_scipy(bhist, comp, fitCfg, binning_qT, chisq_refit=False, rebin=rebin, recoilLow=recoilMin, recoilHigh=recoilMax)
        functions.writeJSON("%s/results_refit.json" % baseDir, jsOut)

        jsIn = functions.loadJSON("%s/results_refit.json" % baseDir)
        rls.doFitMultiGauss_plot(bhist, comp, jsIn, procLabel, metLabel, outDir_refit_fits, binning_qT, yMin=1e-5, yMax=1e3, rebin=rebin, recoilLow=recoilMin, recoilHigh=recoilMax)
        
        rls.plotParameter("p0", jsIn, outDir_refit, binning_qT, procLabel, metLabel, yMin=0, yMax=100, yTitle = "#sigma_{1} (GeV)")
        rls.plotParameter("p1", jsIn, outDir_refit, binning_qT, procLabel, metLabel, yMin=0, yMax=100, yTitle = "#sigma_{2} (GeV)")
        rls.plotParameter("p2", jsIn, outDir_refit, binning_qT, procLabel, metLabel, yMin=0, yMax=200, yTitle = "#sigma_{3} (GeV)")
        rls.plotParameter("p3", jsIn, outDir_refit, binning_qT, procLabel, metLabel, yMin=0, yMax=1, yTitle = "n_{1} (GeV)")
        rls.plotParameter("p4", jsIn, outDir_refit, binning_qT, procLabel, metLabel, yMin=0, yMax=1, yTitle = "n_{2} (GeV)")

    
def ttbar_para_qT_RawPFMET():

    procs = ["Top"]
    tag = "ttbar_para_qT"
    comp = "para"
    baseDir = "%s/%s" % (outDir, tag)
    procLabel = "TTbar #rightarrow #mu^{+}#mu^{#minus}"
    metLabel = "%s, U_{#parallel}" % (met)
    functions.prepareDir(baseDir, False)
    outDir_fits_v0 = "%s/fits_v0" % baseDir
    rebin = 2 
    recoilMin, recoilMax = -100, 100
    bhist = readProc(groups, "recoil_corr_xy_para_qT_qTbinned", procs)
    procNames = groups.getProcNames(to_expand=procs)

    if False:
        fitCfg = {}
        fitCfg['func_name'] = "pol6"
        fitCfg['parms_init'] = [2.99723e+00, -1.47337e-01, 1.61576e-02, -1.42718e-04, -9.91942e-07, 1.47306e-08, -3.50397e-11]
        fitCfg['parms_cte'] = [False, False, False, False, False, False, False]
        rls.parameterizeMean(bhist, comp, baseDir, fitCfg, binning_qT, procLabel, metLabel, procNames, yMin=-5, yMax=50, recoilLow=recoilMin, recoilHigh=recoilMax)
        return

    
    fitCfg = {} 
    fitCfg['func_name'] = "ttbar_para_qT"
    fitCfg['func_parms_vals'] = [20, 50, 100, 0, 0, 0]
    fitCfg['func_parms_cfg'] = [0, 0, 0, 0, 0, 0]
    
    if False:
        rls.doFitMultiGauss_fit(bhist, comp, fitCfg, procLabel, metLabel, outDir_fits_v0, binning_qT, yMin=1e-5, yMax=1e3, recoilLow=recoilMin, recoilHigh=recoilMax, rebin=rebin)
        return
        
    outDir_param_v0 = "%s/params_v0" % baseDir
    if False:
        jsOut = {}

        outDir_param = outDir_param_v0
        functions.prepareDir(outDir_param, True)
        with open("%s/results.json" % outDir_fits_v0) as f: jsIn = json.load(f)
        
        fitF, params, cParams = "quadratic", [9.90024e-02, 9.34622e-01, 9.34622e-01], [False, False, False]
        rls.parameterizeGauss(jsIn, jsOut, comp, "p0", fitF, params, outDir_param, binning_qT, procLabel, metLabel, cParams=cParams, fitMin=10, fitMax=100, yMax=100, xMax=100, yTitle = "#sigma_{1} (GeV)", fitOpts="NS W")

        fitF, params, cParams = "quadratic", [4.66409e-01, 3.46158e-01, 9.34622e-01], [False, False, False]
        rls.parameterizeGauss(jsIn, jsOut, comp, "p1", fitF, params, outDir_param, binning_qT, procLabel, metLabel, cParams=cParams, fitMin=10, fitMax=100, yMax=100, xMax=100, yTitle = "#sigma_{2} (GeV)", fitOpts="NS W")
        
        fitF, params, cParams = "quadratic", [4.66409e-01, 3.46158e-01, 0], [False, False, False]
        rls.parameterizeGauss(jsIn, jsOut, comp, "p2", fitF, params, outDir_param, binning_qT, procLabel, metLabel, cParams=cParams, fitMin=10, fitMax=100, yMax=200, xMax=100, yTitle = "#sigma_{3} (GeV)", fitOpts="NS W")

        
        fitF, params, cParams = "quadratic", [4.66409e-01, 3.46158e-01, 0], [False, False, False]
        rls.parameterizeGauss(jsIn, jsOut, comp, "p3", fitF, params, outDir_param, binning_qT, procLabel, metLabel, cParams=cParams, fitMin=10, fitMax=100, yMax=100, xMax=100, yMin=-100, yTitle = "#mu_{1} (GeV)", fitOpts="NS W")
        
        fitF, params, cParams = "quadratic", [4.66409e-01, 3.46158e-01, 0], [False, False, False]
        rls.parameterizeGauss(jsIn, jsOut, comp, "p4", fitF, params, outDir_param, binning_qT, procLabel, metLabel, cParams=cParams, fitMin=10, fitMax=100, yMax=100, xMax=100, yMin=-100, yTitle = "#mu_{2} (GeV)", fitOpts="NS W")
        
        fitF, params, cParams = "quadratic", [4.66409e-01, 3.46158e-01, 0], [False, False, False]
        rls.parameterizeGauss(jsIn, jsOut, comp, "p5", fitF, params, outDir_param, binning_qT, procLabel, metLabel, cParams=cParams, fitMin=10, fitMax=100, yMax=100, xMax=100, yMin=-100, yTitle = "#mu_{3} (GeV)", fitOpts="NS W")
        

        rls.addParam(jsOut, "p6", "[0]", [0.65])
        rls.addParam(jsOut, "p7", "[0]", [0.25])
     
        jsOut['nParams'] = len(jsOut)
        functions.writeJSON("%s/results.json" % outDir_param_v0, jsOut)
        return
        

    if True:
        outDir_refit = "%s/params_refit" % baseDir
        outDir_refit_fits = "%s/fits_refit" % baseDir
        functions.prepareDir(outDir_refit, False)
        functions.prepareDir(outDir_refit_fits, False)
    
        jsIn = functions.loadJSON("%s/results.json" % outDir_param_v0)
        fitCfg = jsIn
        fitCfg['func_name'] = "ttbar_para_qT_cond"
        jsOut = rls.combinedFit_scipy(bhist, comp, fitCfg, binning_qT, chisq_refit=False, recoilLow=recoilMin, recoilHigh=recoilMax, rebin=rebin)
        functions.writeJSON("%s/results_refit.json" % baseDir, jsOut)
      
        jsIn = functions.loadJSON("%s/results_refit.json" % baseDir)
        rls.doFitMultiGauss_plot(bhist, comp, jsIn, procLabel, metLabel, outDir_refit_fits, binning_qT, yMin=1e-5, yMax=1e3, recoilLow=recoilMin, recoilHigh=recoilMax, rebin=rebin)
        
        rls.plotParameter("p0", jsIn, outDir_refit, binning_qT, procLabel, metLabel, yMin=0, yMax=50, yTitle = "#sigma_{1} (GeV)")
        rls.plotParameter("p1", jsIn, outDir_refit, binning_qT, procLabel, metLabel, yMin=0, yMax=50, yTitle = "#sigma_{2} (GeV)")
        rls.plotParameter("p2", jsIn, outDir_refit, binning_qT, procLabel, metLabel, yMin=0, yMax=50, yTitle = "#sigma_{3} (GeV)")
        rls.plotParameter("p3", jsIn, outDir_refit, binning_qT, procLabel, metLabel, yMin=-5, yMax=5, yTitle = "#mu_{1} (GeV)")
        rls.plotParameter("p4", jsIn, outDir_refit, binning_qT, procLabel, metLabel, yMin=-5, yMax=5, yTitle = "#mu_{2} (GeV)")
        rls.plotParameter("p5", jsIn, outDir_refit, binning_qT, procLabel, metLabel, yMin=-5, yMax=5, yTitle = "#mu_{3} (GeV)")
        rls.plotParameter("p6", jsIn, outDir_refit, binning_qT, procLabel, metLabel, yMin=0, yMax=1, yTitle = "n_{1} (GeV)")
        rls.plotParameter("p7", jsIn, outDir_refit, binning_qT, procLabel, metLabel, yMin=0, yMax=1, yTitle = "n_{2} (GeV)")

 
def zmumu_perp_RawPFMET():

    procs = ["Zmumu"]
    tag = "zmumu_perp"
    comp = "perp"
    baseDir = "%s/%s" % (outDir, tag)
    procLabel = "DY #rightarrow #mu^{+}#mu^{#minus}"
    metLabel = "%s, U_{#perp}" % (met)
    functions.prepareDir(baseDir, False)
    recoilMin, recoilMax = -100, 100
    bhist = readProc(groups, "recoil_corr_xy_perp_qTbinned", procs)
    procNames = groups.getProcNames(to_expand=procs)
    if False:
        fitCfg = {}
        fitCfg['func_name'] = "linear"
        fitCfg['parms_init'] = [0, 0]
        fitCfg['parms_cte'] = [False, False]
        rls.parameterizeMean(bhist, comp, baseDir, fitCfg, binning_qT, procLabel, metLabel, procNames, yMin=-2, yMax=2)
        return


    fitCfg = {} 
    fitCfg['func_name'] = "dy_perp"
    fitCfg['func_parms_vals'] = [15, 5, 8, 10, 0, 0]
    fitCfg['func_parms_cfg'] = [0, 0, 0, 0, 0, 0] # 0=float, 1=propagate, 2=TF1
    outDir_fits_v0 = "%s/fits_v0" % baseDir
    if False:
        rls.doFitMultiGauss_fit(bhist, comp, fitCfg, procLabel, metLabel, outDir_fits_v0, binning_qT, yMin=1e-3, yMax=1e6, recoilLow=recoilMin, recoilHigh=recoilMax)
        return
        
        
    outDir_param_v0 = "%s/params_v0" % baseDir
    if False:
        jsOut = {}
        outDir_param = outDir_param_v0
        functions.prepareDir(outDir_param, True)
        jsIn = functions.loadJSON("%s/results.json" % outDir_fits_v0)
        
        fitF, params, cParams = "power", [9.90024e-02, 9.34622e-01, 1.26746e+01], [False, False, False]
        rls.parameterizeGauss(jsIn, jsOut, comp, "p0", fitF, params, outDir_param, binning_qT, procLabel, metLabel, cParams=cParams, fitMin=0, fitMax=100, yMax=50, xMax=100, yTitle = "#sigma_{1} (GeV)")

        fitF, params, cParams = "power", [4.66409e-01, 3.46158e-01, 3.44635e+00], [False, False, False]
        rls.parameterizeGauss(jsIn, jsOut, comp, "p1", fitF, params, outDir_param, binning_qT, procLabel, metLabel, cParams=cParams, fitMin=0, fitMax=100, yMax=50, xMax=100, yTitle = "#sigma_{2} (GeV)")
        
        fitF, params, cParams = "power", [1, 1, 7.5], [False, False, True]
        rls.parameterizeGauss(jsIn, jsOut, comp, "p2", fitF, params, outDir_param, binning_qT, procLabel, metLabel, cParams=cParams, fitMin=0, fitMax=100, yMax=50, xMax=100, yTitle = "#sigma_{3} (GeV)", cutOffMin=7.5)
        
        fitF, params, cParams = "power", [3.01503e-01, 5.26286e-01, 8.13076e+00], [False, False, False]
        rls.parameterizeGauss(jsIn, jsOut, comp, "p3", fitF, params, outDir_param, binning_qT, procLabel, metLabel, cParams=cParams, fitMin=0, fitMax=100, yMax=50, xMax=100, yTitle = "#sigma_{4} (GeV)", cutOffMax=13)
        
        
        fitF, params, cParams = "linear", [0, 0], [False, False]
        rls.parameterizeGauss(jsIn, jsOut, comp, "p4", fitF, params, outDir_param, binning_qT, procLabel, metLabel, cParams=cParams, fitMin=0, fitMax=100, yMax=50, xMax=100, yMin=-5, cutOffMax=3, cutOffMin=-3, yTitle = "#mu_{1} (GeV)")
        
        fitF, params, cParams = "linear", [0, 0], [False, False]
        rls.parameterizeGauss(jsIn, jsOut, comp, "p5", fitF, params, outDir_param, binning_qT, procLabel, metLabel, cParams=cParams, fitMin=0, fitMax=100, yMax=50, xMax=100, yMin=-5, cutOffMax=3, cutOffMin=-3, yTitle = "#mu_{2} (GeV)")
        
        #fitF, params, cParams = "linear", [0, 0], [False, False]
        #rls.parameterizeGauss(jsIn, jsOut, comp, "p6", fitF, params, outDir_param, binning_qT, procLabel, metLabel, cParams=cParams, fitMin=0, fitMax=100, yMax=50, xMax=100, yMin=-5, cutOffMax=3, cutOffMin=-3, yTitle = "#mu_{3} (GeV)", doFit=True)

        
        # add norm parameters
        rls.addParam(jsOut, "p6", "[0]", [0.1])
        rls.addParam(jsOut, "p7", "[0]", [0.20])
        rls.addParam(jsOut, "p8", "[0]", [0.25])
        

        jsOut['nParams'] = len(jsOut)
        functions.writeJSON("%s/results.json" % outDir_param_v0, jsOut)
        return


    outDir_param_v0 = "%s/params_v0" % baseDir
    if False:
        jsOut = {}
        outDir_param = outDir_param_v0
        functions.prepareDir(outDir_param, True)
        jsIn = functions.loadJSON("%s/results.json" % outDir_fits_v0)
        
        fitF, params, cParams = "power", [9.90024e-02, 9.34622e-01, 1.26746e+01], [False, False, False]
        rls.parameterizeGauss(jsIn, jsOut, comp, "p0", fitF, params, outDir_param, binning_qT, procLabel, metLabel, cParams=cParams, fitMin=0, fitMax=100, yMax=50, xMax=100, yTitle = "#sigma_{1} (GeV)")

        fitF, params, cParams = "power", [4.66409e-01, 3.46158e-01, 3.44635e+00], [False, False, False]
        rls.parameterizeGauss(jsIn, jsOut, comp, "p1", fitF, params, outDir_param, binning_qT, procLabel, metLabel, cParams=cParams, fitMin=0, fitMax=100, yMax=50, xMax=100, yTitle = "#sigma_{2} (GeV)")
        
        fitF, params, cParams = "power", [1, 1, 6.5], [False, False, False]
        rls.parameterizeGauss(jsIn, jsOut, comp, "p2", fitF, params, outDir_param, binning_qT, procLabel, metLabel, cParams=cParams, fitMin=0, fitMax=100, yMax=50, xMax=100, yTitle = "#sigma_{3} (GeV)", cutOffMin=6)
        
        fitF, params, cParams = "power", [3.01503e-01, 5.26286e-01, 6.5], [False, False, False]
        rls.parameterizeGauss(jsIn, jsOut, comp, "p3", fitF, params, outDir_param, binning_qT, procLabel, metLabel, cParams=cParams, fitMin=0, fitMax=100, yMax=50, xMax=100, yTitle = "#sigma_{4} (GeV)")
        
        
        fitF, params, cParams = "linear", [0, 0], [False, False]
        rls.parameterizeGauss(jsIn, jsOut, comp, "p4", fitF, params, outDir_param, binning_qT, procLabel, metLabel, cParams=cParams, fitMin=0, fitMax=100, yMax=50, xMax=100, yMin=-5, cutOffMax=3, cutOffMin=-3, yTitle = "#mu_{1} (GeV)")
        
        fitF, params, cParams = "linear", [0, 0], [False, False]
        rls.parameterizeGauss(jsIn, jsOut, comp, "p5", fitF, params, outDir_param, binning_qT, procLabel, metLabel, cParams=cParams, fitMin=0, fitMax=100, yMax=50, xMax=100, yMin=-5, cutOffMax=3, cutOffMin=-3, yTitle = "#mu_{2} (GeV)")
        
        #fitF, params, cParams = "linear", [0, 0], [False, False]
        #rls.parameterizeGauss(jsIn, jsOut, comp, "p6", fitF, params, outDir_param, binning_qT, procLabel, metLabel, cParams=cParams, fitMin=0, fitMax=100, yMax=50, xMax=100, yMin=-5, cutOffMax=3, cutOffMin=-3, yTitle = "#mu_{3} (GeV)", doFit=True)

        
        # add norm parameters
        rls.addParam(jsOut, "p6", "[0]", [0.2])
        rls.addParam(jsOut, "p7", "[0]", [0.35])
        rls.addParam(jsOut, "p8", "[0]", [0.25])
        

        jsOut['nParams'] = len(jsOut)
        functions.writeJSON("%s/results.json" % outDir_param_v0, jsOut)
        return



    if True:
        outDir_refit = "%s/params_refit" % baseDir
        outDir_refit_fits = "%s/fits_refit" % baseDir
        functions.prepareDir(outDir_refit, False)
        functions.prepareDir(outDir_refit_fits, False)
    
        jsIn = functions.loadJSON("%s/results.json" % outDir_param_v0)
        fitCfg = jsIn
        fitCfg['func_name'] = "dy_perp_cond"
        jsOut = rls.combinedFit_scipy(bhist, comp, fitCfg, binning_qT, chisq_refit=False, outDir=baseDir, recoilLow=recoilMin, recoilHigh=recoilMax)
        functions.writeJSON("%s/results_refit.json" % baseDir, jsOut)
      
        
        jsIn = functions.loadJSON("%s/results_refit.json" % baseDir)
        rls.doFitMultiGauss_plot(bhist, comp, jsIn, procLabel, metLabel, outDir_refit_fits, binning_qT, yMin=1e-3, yMax=1e6, recoilLow=recoilMin, recoilHigh=recoilMax)
        
        rls.plotParameter("p0", jsIn, outDir_refit, binning_qT, procLabel, metLabel, yMin=0, yMax=50, yTitle = "#sigma_{1} (GeV)")
        rls.plotParameter("p1", jsIn, outDir_refit, binning_qT, procLabel, metLabel, yMin=0, yMax=50, yTitle = "#sigma_{2} (GeV)")
        rls.plotParameter("p2", jsIn, outDir_refit, binning_qT, procLabel, metLabel, yMin=0, yMax=50, yTitle = "#sigma_{3} (GeV)")
        rls.plotParameter("p3", jsIn, outDir_refit, binning_qT, procLabel, metLabel, yMin=0, yMax=50, yTitle = "#sigma_{4} (GeV)")
        rls.plotParameter("p4", jsIn, outDir_refit, binning_qT, procLabel, metLabel, yMin=-5, yMax=5, yTitle = "#mu_{1} (GeV)")
        rls.plotParameter("p5", jsIn, outDir_refit, binning_qT, procLabel, metLabel, yMin=-5, yMax=5, yTitle = "#mu_{2} (GeV)")
        #rls.plotParameter("p6", jsIn, outDir_refit, binning_qT, procLabel, metLabel, yMin=-5, yMax=5, yTitle = "#mu_{3} (GeV)")
        rls.plotParameter("p6", jsIn, outDir_refit, binning_qT, procLabel, metLabel, yMin=0, yMax=1, yTitle = "n_{1} (GeV)")
        rls.plotParameter("p7", jsIn, outDir_refit, binning_qT, procLabel, metLabel, yMin=0, yMax=1, yTitle = "n_{2} (GeV)")
        rls.plotParameter("p8", jsIn, outDir_refit, binning_qT, procLabel, metLabel, yMin=0, yMax=1, yTitle = "n_{3} (GeV)")
   

    #exportCfg = {}
    #exportCfg["mean"] = ["p4", "p5", "p4", "p5"]
    #exportCfg["sigma"] = ["p0", "p1", "p2", "p3"]
    #exportCfg["norm"] = ["p6", "p7", "p8"]
    #rls.export(exportCfg, "%s/recoil_zmumu_perp.json" % outCfgDir, "%s/results_refit.json" % baseDir)
   

def zmumu_para_qT_RawPFMET():

    procs = ["Zmumu"]
    tag = "zmumu_para_qT"
    comp = "para"
    baseDir = "%s/%s" % (outDir, tag)
    procLabel = "DY #rightarrow #mu^{+}#mu^{#minus}"
    metLabel = "%s, U_{#parallel}" % (met)
    functions.prepareDir(baseDir, False)
    procNames = groups.getProcNames(to_expand=procs)
    
    bhist = readProc(groups, "recoil_corr_xy_para_qT_qTbinned", procs)
    recoilMin, recoilMax = -100, 100
    
    if False:
        fitCfg = {}
        fitCfg['func_name'] = "pw_poly4_power"
        fitCfg['parms_init'] = [1.89766e+01, -1.23112e-01, 5.16421e-01, -3.18375e-02, 9.51270e-04, -8.04191e-06, 9.51172e-01]
        fitCfg['parms_cte'] = [False, False, False, False, False, False, False]
        
        fitCfg['func_name'] = "pw_poly6_poly1"
        fitCfg['parms_init'] = [40, -1.64348e-01, 5.56260e-01, -4.26692e-02, 2.10091e-03, -5.89702e-05, 8.59665e-07, -5.01217e-09]
        fitCfg['parms_cte'] = [True, False, False, False, False, False, False, False]
        
        rls.parameterizeMean(bhist, comp, baseDir, fitCfg, binning_qT, procLabel, metLabel, procNames, yMin=-2, yMax=16)
        return

    

    fitCfg = {} 
    fitCfg['func_name'] = "dy_para_qT"
    fitCfg['func_parms_vals'] = [15, 5, 8, 10, 0, 0]
    fitCfg['func_parms_cfg'] = [0, 0, 0, 0, 1, 1] # 0=float, 1=propagate, 2=TF1
    
    outDir_fits_v0 = "%s/fits_v0" % baseDir
    if False:
        rls.doFitMultiGauss_fit(bhist, comp, fitCfg, procLabel, metLabel, outDir_fits_v0, binning_qT, yMin=1e-3, yMax=1e6, recoilLow=recoilMin, recoilHigh=recoilMax)
        return
        
        
    outDir_param_v0 = "%s/params_v0" % baseDir
    if False:
        jsOut = {}
        outDir_param = outDir_param_v0
        functions.prepareDir(outDir_param, True)
        with open("%s/results.json" % outDir_fits_v0) as f: jsIn = json.load(f)
        
        #fitF, params, cParams = "linear", [9.90024e-02, 9.34622e-01], [False, False, False]
        fitF, params, cParams = "power", [4.54931e+00, 2.05145e-01, -4.25497e+00], [False, False, False]
        rls.parameterizeGauss(jsIn, jsOut, comp, "p0", fitF, params, outDir_param, binning_qT, procLabel, metLabel, cParams=cParams, fitMin=0, fitMax=90, yMax=50, xMax=100, yTitle = "#sigma_{1} (GeV)", cutOffMin=10, cutOffMax=50)

        fitF, params, cParams = "power", [4.66409e-01, 3.46158e-01, 3.44635e+00], [False, False, False]
        rls.parameterizeGauss(jsIn, jsOut, comp, "p1", fitF, params, outDir_param, binning_qT, procLabel, metLabel, cParams=cParams, fitMin=0, fitMax=100, yMax=50, xMax=100, yTitle = "#sigma_{2} (GeV)")
        
        fitF, params, cParams = "power", [5.18556e-01, 3.95983e-01, 5.62069e+00], [False, False, False]
        rls.parameterizeGauss(jsIn, jsOut, comp, "p2", fitF, params, outDir_param, binning_qT, procLabel, metLabel, cParams=cParams, fitMin=0, fitMax=100, yMax=50, xMax=100, yTitle = "#sigma_{3} (GeV)")
        
        fitF, params, cParams = "power", [3.01503e-01, 5.26286e-01, 8.13076e+00], [False, False, False]
        rls.parameterizeGauss(jsIn, jsOut, comp, "p3", fitF, params, outDir_param, binning_qT, procLabel, metLabel, cParams=cParams, fitMin=0, fitMax=30, yMax=50, xMax=100, yTitle = "#sigma_{4} (GeV)")
        
        
        #fitF, params, cParams = "linear", [9.90024e-02, 9.34622e-01], [False, False, False]
        fitF, params, cParams = "power", [4.54931e+00, 2.05145e-01, -4.25497e+00], [False, False, False]
        rls.parameterizeGauss(jsIn, jsOut, comp, "p4", fitF, params, outDir_param, binning_qT, procLabel, metLabel, cParams=cParams, fitMin=0, fitMax=90, yMax=50, xMax=100, yMin=-20, yTitle = "#mu_{1} (GeV)")
        
        #fitF, params, cParams = "pw_poly4_poly1", [15, -4.26860e-01, 8.23660e-01, -8.92751e-02, 5.60430e-03, -1.38168e-04], [True, False, False, False, False, False]
        fitF, params, cParams = "power", [4.54931e+00, 2.05145e-01, -4.25497e+00], [False, False, False]
        rls.parameterizeGauss(jsIn, jsOut, comp, "p5", fitF, params, outDir_param, binning_qT, procLabel, metLabel, cParams=cParams, fitMin=0, fitMax=30, yMax=50, xMax=100, yMin=-20, yTitle = "#mu_{2} (GeV)")
        
        #fitF, params, cParams = "pw_poly4_poly1", [15, -4.26860e-01, 8.23660e-01, -8.92751e-02, 5.60430e-03, -1.38168e-04], [True, False, False, False, False, False]
        #fitF, params, cParams = "power", [4.54931e+00, 2.05145e-01, -4.25497e+00], [False, False, False]
        #rls.parameterizeGauss(jsIn, jsOut, comp, "p6", fitF, params, outDir_param, binning_qT, procLabel, metLabel, cParams=cParams, fitMin=0, fitMax=30, yMax=50, xMax=100, yMin=-20, yTitle = "#mu_{3} (GeV)")
        
        #fitF, params, cParams = "pw_poly4_poly1", [15, -4.26860e-01, 8.23660e-01, -8.92751e-02, 5.60430e-03, -1.38168e-04], [True, False, False, False, False, False]
        #fitF, params, cParams = "power", [4.54931e+00, 2.05145e-01, -4.25497e+00], [False, False, False]
        #rls.parameterizeGauss(jsIn, jsOut, comp, "p7", fitF, params, outDir_param, binning_qT, procLabel, metLabel, cParams=cParams, fitMin=0, fitMax=30, yMax=50, xMax=100, yMin=-20, yTitle = "#mu_{4} (GeV)")
   
        # add norm parameters
        rls.addParam(jsOut, "p6", "[0]", [0.01])
        rls.addParam(jsOut, "p7", "[0]", [0.2])
        rls.addParam(jsOut, "p8", "[0]", [0.6])
        

        jsOut['nParams'] = len(jsOut)
        functions.writeJSON("%s/results.json" % outDir_param_v0, jsOut)
        return



    if False:
        outDir_refit = "%s/params_refit" % baseDir
        outDir_refit_fits = "%s/fits_refit" % baseDir
        functions.prepareDir(outDir_refit, False)
        functions.prepareDir(outDir_refit_fits, False)
    
        jsIn = functions.loadJSON("%s/results.json" % outDir_param_v0)
        fitCfg = jsIn
        fitCfg['func_name'] = "dy_para_qT_cond"
        jsOut = rls.combinedFit_scipy(bhist, comp, fitCfg, binning_qT, chisq_refit=False, outDir=baseDir, recoilLow=recoilMin, recoilHigh=recoilMax)
        functions.writeJSON("%s/results_refit.json" % baseDir, jsOut)
        
        jsIn = functions.loadJSON("%s/results_refit.json" % baseDir)
        rls.doFitMultiGauss_plot(bhist, comp, jsIn, procLabel, metLabel, outDir_refit_fits, binning_qT, yMin=1e-3, yMax=1e6, recoilLow=recoilMin, recoilHigh=recoilMax)
        
        rls.plotParameter("p0", jsIn, outDir_refit, binning_qT, procLabel, metLabel, yMin=0, yMax=50, yTitle = "#sigma_{1} (GeV)")
        rls.plotParameter("p1", jsIn, outDir_refit, binning_qT, procLabel, metLabel, yMin=0, yMax=50, yTitle = "#sigma_{2} (GeV)")
        rls.plotParameter("p2", jsIn, outDir_refit, binning_qT, procLabel, metLabel, yMin=0, yMax=50, yTitle = "#sigma_{3} (GeV)")
        rls.plotParameter("p3", jsIn, outDir_refit, binning_qT, procLabel, metLabel, yMin=0, yMax=50, yTitle = "#sigma_{4} (GeV)")
        rls.plotParameter("p4", jsIn, outDir_refit, binning_qT, procLabel, metLabel, yMin=-20, yMax=50, yTitle = "#mu_{1} (GeV)")
        rls.plotParameter("p5", jsIn, outDir_refit, binning_qT, procLabel, metLabel, yMin=-20, yMax=50, yTitle = "#mu_{2} (GeV)")
        #rls.plotParameter("p6", jsIn, outDir_refit, binning_qT, procLabel, metLabel, yMin=-5, yMax=5, yTitle = "#mu_{3} (GeV)")
        #rls.plotParameter("p7", jsIn, outDir_refit, binning_qT, procLabel, metLabel, yMin=-5, yMax=5, yTitle = "#mu_{4} (GeV)")
        rls.plotParameter("p6", jsIn, outDir_refit, binning_qT, procLabel, metLabel, yMin=0, yMax=1, yTitle = "n_{2} (GeV)")
        rls.plotParameter("p7", jsIn, outDir_refit, binning_qT, procLabel, metLabel, yMin=0, yMax=1, yTitle = "n_{3} (GeV)")
        rls.plotParameter("p8", jsIn, outDir_refit, binning_qT, procLabel, metLabel, yMin=0, yMax=1, yTitle = "n_{3} (GeV)")
        
        
 
    exportCfg = {}
    exportCfg["mean"] = ["p4", "p4", "p5", "p5"]
    exportCfg["sigma"] = ["p0", "p1", "p2", "p3"]
    exportCfg["norm"] = ["p6", "p7", "p8"]
    rls.export(exportCfg, "%s/recoil_zmumu_para.json" % outCfgDir, "%s/results_refit.json" % baseDir) # , "%s/parametric_mean.json" % baseDir
   
  
def singlemuon_perp_RawPFMET():
    
    procs = ["SingleMuon"]
    tag = "singlemuon_perp"
    comp = "perp"
    baseDir = "%s/%s" % (outDir, tag)
    procLabel = "Data #rightarrow #mu^{+}#mu^{#minus}"
    metLabel = "%s, U_{#perp}" % (met)
    functions.prepareDir(baseDir, False)
    procNames = groups.getProcNames(to_expand=procs)
    
    recoilMin, recoilMax = -100, 100
    bhist = readProc(groups, "recoil_corr_xy_perp_qTbinned", procs)
    if False:
        fitCfg = {}
        fitCfg['func_name'] = "linear"
        fitCfg['parms_init'] = [0, 0]
        fitCfg['parms_cte'] = [False, False]
        rls.parameterizeMean(bhist, comp, baseDir, fitCfg, binning_qT, procLabel, metLabel, procNames=procNames, yMin=-2, yMax=2)
        return

    fitCfg = {} 
    fitCfg['func_name'] = "data_perp"
    fitCfg['func_parms_vals'] = [40, 5, 10, 15, 0, 0, 0]
    fitCfg['func_parms_cfg'] = [0, 0, 0, 0, 0, 0, 0] # 0=float, 1=propagate, 2=TF1
    
    bkgCfg = {}
    bkgCfg['procs'] = ["ttbar", "ewk"]
    bkgCfg['parms'] = [functions.loadJSON("%s/ttbar_perp/results_refit.json"%outDir), functions.loadJSON("%s/ewk_perp/results_refit.json"%outDir)]
    bkgCfg['yields'] = [functions.loadJSON("%s/ttbar_perp/yields.json"%outDir), functions.loadJSON("%s/ewk_perp/yields.json"%outDir)]
    bkgCfg['norms'] = [1.0, 1.0]
    bkgCfg['data_yields'] = functions.loadJSON("%s/yields.json"%baseDir)
    
    outDir_fits_v0 = "%s/fits_v0" % baseDir
    if False:
        rls.doFitMultiGauss_fit(bhist, comp, fitCfg, procLabel, metLabel, outDir_fits_v0, binning_qT, bkgCfg=bkgCfg, yMin=1e-3, yMax=1e6, recoilLow=recoilMin, recoilHigh=recoilMax)
        return
        
    outDir_param_v0 = "%s/params_v0" % baseDir
    if False:
        jsOut = {}
        outDir_param = outDir_param_v0
        functions.prepareDir(outDir_param, True)
        with open("%s/results.json" % outDir_fits_v0) as f: jsIn = json.load(f)
        
        fitF, params, cParams = "power", [9.90024e-02, 9.34622e-01, 1.26746e+01], [False, False, False]
        rls.parameterizeGauss(jsIn, jsOut, comp, "p0", fitF, params, outDir_param, binning_qT, procLabel, metLabel, cParams=cParams, fitMin=0, fitMax=100, yMax=50, xMax=100, yTitle = "#sigma_{1} (GeV)", cutOffMin=5)

        fitF, params, cParams = "power", [4.66409e-01, 3.46158e-01, 3.44635e+00], [False, False, False]
        rls.parameterizeGauss(jsIn, jsOut, comp, "p1", fitF, params, outDir_param, binning_qT, procLabel, metLabel, cParams=cParams, fitMin=0, fitMax=100, yMax=50, xMax=100, yTitle = "#sigma_{2} (GeV)")
        
        fitF, params, cParams = "power", [1, 1, 1], [False, False, False]
        rls.parameterizeGauss(jsIn, jsOut, comp, "p2", fitF, params, outDir_param, binning_qT, procLabel, metLabel, cParams=cParams, fitMin=0, fitMax=100, yMax=50, xMax=100, yTitle = "#sigma_{3} (GeV)", cutOffMin=6)
        
        fitF, params, cParams = "power", [3.01503e-01, 5.26286e-01, 8.13076e+00], [False, False, False]
        rls.parameterizeGauss(jsIn, jsOut, comp, "p3", fitF, params, outDir_param, binning_qT, procLabel, metLabel, cParams=cParams, fitMin=0, fitMax=100, yMax=50, xMax=100, yTitle = "#sigma_{4} (GeV)")
        
        
        fitF, params, cParams = "linear", [0, 0], [False, False]
        rls.parameterizeGauss(jsIn, jsOut, comp, "p4", fitF, params, outDir_param, binning_qT, procLabel, metLabel, cParams=cParams, fitMin=0, fitMax=100, yMax=50, xMax=100, yMin=-5, yTitle = "#mu_{1} (GeV)", fitOpts="NS W")
        
        fitF, params, cParams = "linear", [0, 0], [False, False]
        rls.parameterizeGauss(jsIn, jsOut, comp, "p5", fitF, params, outDir_param, binning_qT, procLabel, metLabel, cParams=cParams, fitMin=0, fitMax=100, yMax=50, xMax=100, yMin=-5, yTitle = "#mu_{2} (GeV)", fitOpts="NS W")
        
        fitF, params, cParams = "linear", [0, 0], [False, False]
        rls.parameterizeGauss(jsIn, jsOut, comp, "p6", fitF, params, outDir_param, binning_qT, procLabel, metLabel, cParams=cParams, fitMin=0, fitMax=100, yMax=50, xMax=100, yMin=-5, yTitle = "#mu_{3} (GeV)", fitOpts="NS W")

        
        # add norm parameters
        rls.addParam(jsOut, "p7", "[0]", [0.1])
        rls.addParam(jsOut, "p8", "[0]", [0.20])
        rls.addParam(jsOut, "p9", "[0]", [0.25])
        
        
        jsOut['nParams'] = len(jsOut)
        functions.writeJSON("%s/results.json" % outDir_param_v0, jsOut)
        return


    if False:
        outDir_refit = "%s/params_refit" % baseDir
        outDir_refit_fits = "%s/fits_refit" % baseDir
        functions.prepareDir(outDir_refit, True)
        functions.prepareDir(outDir_refit_fits, True)
        
        #'''
        #jsIn = functions.loadJSON("%s/results.json" % outDir_param_v0)
        jsIn = functions.loadJSON("%s/zmumu_perp/results_refit.json" % outDir) # take the MC as starting values
        #jsIn['p7']['p0'] = 0.001 # perturb the value
        # single mean for all gauss
        jsIn['p4']['p0'] = 0.0
        jsIn['p4']['p1'] = 0.0
        jsIn['p5'] = copy.deepcopy(jsIn['p6'])
        jsIn['p6'] = copy.deepcopy(jsIn['p7'])
        jsIn['p7'] = copy.deepcopy(jsIn['p8'])
        
        jsIn['nParams'] = jsIn['nParams']-1
        del jsIn['p8']
        
        # remove component of p1
        #del jsIn['p1']['p2']
        #jsIn['p1']['nParams'] -= 1
        jsIn['p6'] = copy.deepcopy(jsIn['p7'])
        del jsIn['p7']
        jsIn['nParams'] = jsIn['nParams']-1
        
        fitCfg = jsIn
        fitCfg['func_name'] = "data_perp_cond"
        jsOut = rls.combinedFit_scipy(bhist, comp, fitCfg, binning_qT, bkgCfg=bkgCfg, chisq_refit=False, outDir=baseDir, recoilLow=recoilMin, recoilHigh=recoilMax)
        functions.writeJSON("%s/results_refit.json" % baseDir, jsOut)
        #'''
        
        jsIn = functions.loadJSON("%s/results_refit.json" % baseDir)
        rls.doFitMultiGauss_plot(bhist, comp, jsIn, procLabel, metLabel, outDir_refit_fits, binning_qT, bkgCfg=bkgCfg, yMin=1e-3, yMax=1e6, recoilLow=recoilMin, recoilHigh=recoilMax, plotSignal=True)
        
        rls.plotParameter("p0", jsIn, outDir_refit, binning_qT, procLabel, metLabel, yMin=0, yMax=50, yTitle = "#sigma_{1} (GeV)")
        rls.plotParameter("p1", jsIn, outDir_refit, binning_qT, procLabel, metLabel, yMin=0, yMax=50, yTitle = "#sigma_{2} (GeV)")
        rls.plotParameter("p2", jsIn, outDir_refit, binning_qT, procLabel, metLabel, yMin=0, yMax=50, yTitle = "#sigma_{3} (GeV)")
        rls.plotParameter("p3", jsIn, outDir_refit, binning_qT, procLabel, metLabel, yMin=0, yMax=50, yTitle = "#sigma_{4} (GeV)")
        rls.plotParameter("p4", jsIn, outDir_refit, binning_qT, procLabel, metLabel, yMin=-5, yMax=5, yTitle = "#mu_{1} (GeV)")
        #rls.plotParameter("p5", jsIn, outDir_refit, binning_qT, procLabel, metLabel, yMin=-5, yMax=5, yTitle = "#mu_{2} (GeV)")
        #rls.plotParameter("p6", jsIn, outDir_refit, binning_qT, procLabel, metLabel, yMin=-5, yMax=5, yTitle = "#mu_{3} (GeV)")
        rls.plotParameter("p5", jsIn, outDir_refit, binning_qT, procLabel, metLabel, yMin=0, yMax=1, yTitle = "n_{1} (GeV)")
        rls.plotParameter("p6", jsIn, outDir_refit, binning_qT, procLabel, metLabel, yMin=0, yMax=1, yTitle = "n_{2} (GeV)")
        #rls.plotParameter("p7", jsIn, outDir_refit, binning_qT, procLabel, metLabel, yMin=0, yMax=1, yTitle = "n_{3} (GeV)")
   
         
    exportCfg = {}
    exportCfg["mean"] = ["p4", "p4", "p4", "p4"]
    exportCfg["sigma"] = ["p0", "p1", "p2", "p3"]
    exportCfg["norm"] = ["p5", 0.21, "p6"]
    rls.export(exportCfg, "%s/recoil_data_perp.json" % outCfgDir, "%s/results_refit.json" % baseDir)
         
    
def singlemuon_para_qT_RawPFMET():

    procs = ["SingleMuon"]
    tag = "singlemuon_para_qT"
    comp = "para"
    baseDir = "%s/%s" % (outDir, tag)
    procLabel = "Data #rightarrow #mu^{+}#mu^{#minus}"
    metLabel = "%s, U_{#perp}" % (met)
    functions.prepareDir(baseDir, False)
    rebin = 1

    outDir_fits_v0 = "%s/fits_v0" % baseDir
    recoilMin, recoilMax = -100, 100
    
    bhist = readProc(groups, "recoil_corr_xy_para_qT_qTbinned", procs)
    procNames = groups.getProcNames(to_expand=procs)
    if False:
        fitCfg = {}
        fitCfg['func_name'] = "pw_poly4_power"
        fitCfg['parms_init'] = [2.37500e+01, 1.41561e-01, 4.30537e-01, -1.62911e-02, 4.85259e-04, -6.32757e-06, 9.75226e-01]
        fitCfg['parms_cte'] = [False, False, False, False, False, False, False]
        rls.parameterizeMean(bhist, comp, baseDir, fitCfg, binning_qT, procLabel, metLabel, procNames, yMin=0, yMax=20)
        return
        
    fitCfg = {} 
    fitCfg['func_name'] = "data_para_qT"
    fitCfg['func_parms_vals'] = [15, 5, 8, 0, 0, 0]
    fitCfg['func_parms_cfg'] = [0, 0, 0, 0, 0, 0] # 0=float, 1=propagate, 2=TF1

    bkgCfg = {}
    bkgCfg['procs'] = ["ttbar", "ewk"]
    bkgCfg['parms'] = [functions.loadJSON("%s/ttbar_para_qT/results_refit.json"%outDir), functions.loadJSON("%s/ewk_para_qT/results_refit.json"%outDir)]
    bkgCfg['yields'] = [functions.loadJSON("%s/ttbar_para_qT/yields.json"%outDir), functions.loadJSON("%s/ewk_para_qT/yields.json"%outDir)]
    bkgCfg['norms'] = [1.0, 1.0]
    bkgCfg['data_yields'] = functions.loadJSON("%s/singlemuon_para_qT/yields.json"%outDir) # needed for the background fractions

    if False:
        rls.doFitMultiGauss_fit(bhist, comp, fitCfg, procLabel, metLabel, outDir_fits_v0, binning_qT, bkgCfg=bkgCfg, yMin=1e-3, yMax=1e6, recoilLow=recoilMin, recoilHigh=recoilMax)
        return
        
    outDir_param_v0 = "%s/params_v0" % baseDir
    if False:
        jsOut = {}
        outDir_param = outDir_param_v0
        functions.prepareDir(outDir_param, True)
        with open("%s/results.json" % outDir_fits_v0) as f: jsIn = json.load(f)
        
        fitF, params, cParams = "power", [1, 1, 1], [False, False, False]
        rls.parameterizeGauss(jsIn, jsOut, comp, "p0", fitF, params, outDir_param, binning_qT, procLabel, metLabel, cParams=cParams, fitMin=0, fitMax=39, yMax=50, xMax=100, cutOffMin=12, cutOffMax=18, yTitle = "#sigma_{1} (GeV)")

        fitF, params, cParams = "power", [1, 1, 1], [False, False, False]
        rls.parameterizeGauss(jsIn, jsOut, comp, "p1", fitF, params, outDir_param, binning_qT, procLabel, metLabel, cParams=cParams, fitMin=0, fitMax=50, yMax=50, xMax=100, yTitle = "#sigma_{2} (GeV)")
        
        fitF, params, cParams = "power", [1, 1, 1], [False, False, False]
        rls.parameterizeGauss(jsIn, jsOut, comp, "p2", fitF, params, outDir_param, binning_qT, procLabel, metLabel, cParams=cParams, fitMin=0, fitMax=50, yMax=50, xMax=100, cutOffMax=12, yTitle = "#sigma_{3} (GeV)")
        
        fitF, params, cParams = "linear", [0, 0], [False, False]
        rls.parameterizeGauss(jsIn, jsOut, comp, "p3", fitF, params, outDir_param, binning_qT, procLabel, metLabel, cParams=cParams, fitMin=0, fitMax=50, yMax=50, xMax=100, yMin=-20, yTitle = "#mu_{1} (GeV)", fitOpts="NS W")
        
        fitF, params, cParams = "linear", [0, 0], [False, False]
        rls.parameterizeGauss(jsIn, jsOut, comp, "p4", fitF, params, outDir_param, binning_qT, procLabel, metLabel, cParams=cParams, fitMin=0, fitMax=50, yMax=50, xMax=100, yMin=-20, yTitle = "#mu_{2} (GeV)", fitOpts="NS W")
        
        fitF, params, cParams = "linear", [0, 0], [False, False]
        rls.parameterizeGauss(jsIn, jsOut, comp, "p5", fitF, params, outDir_param, binning_qT, procLabel, metLabel, cParams=cParams, fitMin=0, fitMax=50, yMax=50, xMax=100, yMin=-20, yTitle = "#mu_{3} (GeV)", fitOpts="NS W")
        
        
        # add norm parameters
        rls.addParam(jsOut, "p6", "[0]", [0.1])
        rls.addParam(jsOut, "p7", "[0]", [0.2])


        jsOut['nParams'] = len(jsOut)
        functions.writeJSON("%s/results.json" % outDir_param_v0, jsOut)
        return


    if False:
        outDir_refit = "%s/params_refit" % baseDir
        outDir_refit_fits = "%s/fits_refit" % baseDir
        functions.prepareDir(outDir_refit, False)
        functions.prepareDir(outDir_refit_fits, False)
    
        #with open("%s/results.json" % outDir_param_v0) as f: jsIn = json.load(f)
        with open("%s/zmumu_para_qT/results_refit.json" % outDir) as f: jsIn = json.load(f) # take the MC as starting values
        fitCfg = jsIn
        fitCfg['func_name'] = "data_para_qT_cond"
        jsOut = rls.combinedFit_scipy(bhist, comp, fitCfg, binning_qT, bkgCfg=bkgCfg, chisq_refit=False, outDir=baseDir, recoilLow=recoilMin, recoilHigh=recoilMax)
        with open("%s/results_refit.json" % baseDir, "w") as outfile: json.dump(jsOut, outfile, indent=4)
         
        with open("%s/results_refit.json" % baseDir) as f: jsIn = json.load(f)
        rls.doFitMultiGauss_plot(bhist, comp, jsIn, procLabel, metLabel, outDir_refit_fits, binning_qT, bkgCfg=bkgCfg, yMin=1e-3, yMax=1e6, recoilLow=recoilMin, recoilHigh=recoilMax, plotSignal=True)
        
        rls.plotParameter("p0", jsIn, outDir_refit, binning_qT, procLabel, metLabel, yMin=0, yMax=50, yTitle = "#sigma_{1} (GeV)")
        rls.plotParameter("p1", jsIn, outDir_refit, binning_qT, procLabel, metLabel, yMin=0, yMax=50, yTitle = "#sigma_{2} (GeV)")
        rls.plotParameter("p2", jsIn, outDir_refit, binning_qT, procLabel, metLabel, yMin=0, yMax=50, yTitle = "#sigma_{3} (GeV)")
        rls.plotParameter("p3", jsIn, outDir_refit, binning_qT, procLabel, metLabel, yMin=0, yMax=50, yTitle = "#sigma_{4} (GeV)")
        rls.plotParameter("p4", jsIn, outDir_refit, binning_qT, procLabel, metLabel, yMin=-20, yMax=50, yTitle = "#mu_{1} (GeV)")
        rls.plotParameter("p5", jsIn, outDir_refit, binning_qT, procLabel, metLabel, yMin=-20, yMax=50, yTitle = "#mu_{2} (GeV)")
        #rls.plotParameter("p6", jsIn, outDir_refit, binning_qT, procLabel, metLabel, yMin=-5, yMax=5, yTitle = "#mu_{3} (GeV)")
        #rls.plotParameter("p7", jsIn, outDir_refit, binning_qT, procLabel, metLabel, yMin=-5, yMax=5, yTitle = "#mu_{4} (GeV)")
        rls.plotParameter("p6", jsIn, outDir_refit, binning_qT, procLabel, metLabel, yMin=0, yMax=1, yTitle = "n_{2} (GeV)")
        rls.plotParameter("p7", jsIn, outDir_refit, binning_qT, procLabel, metLabel, yMin=0, yMax=1, yTitle = "n_{3} (GeV)")
        rls.plotParameter("p8", jsIn, outDir_refit, binning_qT, procLabel, metLabel, yMin=0, yMax=1, yTitle = "n_{3} (GeV)")
   
         
    exportCfg = {}
    exportCfg["mean"] = ["p4", "p4", "p5", "p5"]
    exportCfg["sigma"] = ["p0", "p1", "p2", "p3"]
    exportCfg["norm"] = ["p6", "p7", "p8"]
    rls.export(exportCfg, "%s/recoil_data_para.json" % outCfgDir, "%s/results_refit.json" % baseDir)
   
 

 
 
def do_zz_perp_DeepMETReso():

    proc = "ZZ"
    tag = "zz_perp"
    comp = "perp"
    baseDir = "%s/%s" % (outDir, tag)
    name = "ZZ #rightarrow #mu^{+}#mu^{#minus}"
    label = "%s, %s" % (name, met)
    functions.prepareDir(baseDir, False)
    rebin = 2

    ###### STEP 1: floating fits
    iteration = 0
    outDir_fits = "%s/fits_v0" % baseDir
    fitCfg = {}
    fitCfg['mean'], fitCfg['mean_cfg'] = [0, 0, 0], [0, 0, 0]
    fitCfg['sigma'], fitCfg['sigma_cfg'] = [10, 30, 50], [0, 0, 0]
    fitCfg['norm'], fitCfg['norm_cfg'] = [0.50, 0.40], [0, 0]
    nGauss = len(fitCfg['mean'])
    
    if True: 
        recoilLibs.doFitMultiGauss(rFile, proc, comp, fitCfg, label, outDir_fits, recoil_qTbins, qTmax=500, rebin=rebin, singleMean=True, xMin=-200, xMax=200)
        return
        
        
    ###### STEP 2: parameterize
    outDir_param = "%s/param_v0" % baseDir
    if True:
        jsOut = {}
        jsOut['nGauss'] = nGauss
        functions.prepareDir(outDir_param, False)
        with open("%s/results.json" % outDir_fits) as f: jsIn = json.load(f)
        
        # sigma
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [8.69746e-01, 5.59970e-01, 8], [False, False, False]
        doFit(jsIn, jsOut, "sigma1", fitF, params, outDir_param, label, cParams=cParams, fitMin=0, fitMax=100, yMin=0, yMax=100, yTitle = "#sigma_{1} (GeV)")
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [8.69746e-01, 5.59970e-01, 8], [False, False, False]
        doFit(jsIn, jsOut, "sigma2", fitF, params, outDir_param, label,cParams=cParams,  fitMin=0, fitMax=50, yMin=0, yMax=100, yTitle = "#sigma_{2} (GeV)")
       
        # mean
        fitF, params, cParams = "[0]", [fitCfg['mean'][0]], [True]
        doFit(jsIn, jsOut, "mean1", fitF, params, outDir_param, label, cParams=cParams, fitMin=0, fitMax=50, yMin=-5, yMax=5, yTitle = "#mu_{1} (GeV)", doFit=False)
        fitF, params, cParams = "[0]", [fitCfg['mean'][1]], [True]
        doFit(jsIn, jsOut, "mean2", fitF, params, outDir_param, label, cParams=cParams, fitMin=0, fitMax=50, yMin=-5, yMax=5, yTitle = "#mu_{2} (GeV)", doFit=False)

        # norm
        fitF, params, cParams = "[0]", [fitCfg['norm'][0]], [True]
        doFit(jsIn, jsOut, "norm1", fitF, params, outDir_param, label, cParams=cParams, fitMin=0, fitMax=50, yMin=0, yMax=1, yTitle = "n_{1} (GeV)", doFit=False)
        
        with open("%s/results.json" % outDir_param, "w") as outfile: json.dump(jsOut, outfile, indent=4)
    

    ###### STEP 3: refit
    if True:
        with open("%s/results.json" % outDir_param) as f: jsIn = json.load(f)
        jsOut = combinedFit(bkg, comp, jsIn, rebin=rebin, qTmax=50)
        with open("%s/results_refit.json" % outDir_param, "w") as outfile: json.dump(jsOut, outfile, indent=4)


    ###### STEP 4: validate
    if True:
        fitCfg['mean_cfg'] = [1, 1]
        fitCfg['sigma_cfg'] = [2, 2]
        fitCfg['norm_cfg'] = [1]
        outDir_refit = "%s/refit_v0" % baseDir
        with open("%s/results_refit.json" % outDir_param) as f: jsIn = json.load(f)
        doFitMultiGauss(bkg, comp, fitCfg, label, outDir_refit, funcJs=jsIn, qTmax=200, rebin=rebin, xMin=-200, xMax=200)

def do_zz_para_DeepMETReso():

    proc = "ZZ"
    tag = "zz_para"
    comp = "para"
    baseDir = "%s/%s" % (outDir, tag)
    name = "ZZ #rightarrow #mu^{+}#mu^{#minus}"
    label = "%s, %s" % (name, met)
    functions.prepareDir(baseDir, False)
    rebin = 2

    ###### STEP 1: floating fits
    iteration = 0
    outDir_fits = "%s/fits_v0" % baseDir
    fitCfg = {}
    fitCfg['mean'], fitCfg['mean_cfg'] = [0, 0, 0], [0, 0, 0]
    fitCfg['sigma'], fitCfg['sigma_cfg'] = [10, 30, 50], [0, 0, 0]
    fitCfg['norm'], fitCfg['norm_cfg'] = [0.50, 0.40], [0, 0]
    nGauss = len(fitCfg['mean'])
    
    if True: 
        recoilLibs.doFitMultiGauss(rFile, proc, comp, fitCfg, label, outDir_fits, recoil_qTbins, qTmax=500, rebin=rebin, singleMean=False, xMin=-200, xMax=200)
        return
    
    ###### STEP 2: parameterize
    outDir_param = "%s/param_v0" % baseDir
    if False:
        jsOut = {}
        jsOut['nGauss'] = nGauss
        functions.prepareDir(outDir_param, False)
        with open("%s/results.json" % outDir_fits) as f: jsIn = json.load(f)
        
        # sigma
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [8.69746e-01, 5.59970e-01, 8], [False, False, False]
        doFit(jsIn, jsOut, "sigma1", fitF, params, outDir_param, label, cParams=cParams, fitMin=0, fitMax=100, yMin=0, yMax=100, yTitle = "#sigma_{1} (GeV)")
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [8.69746e-01, 5.59970e-01, 8], [False, False, False]
        doFit(jsIn, jsOut, "sigma2", fitF, params, outDir_param, label,cParams=cParams,  fitMin=0, fitMax=50, yMin=0, yMax=100, yTitle = "#sigma_{2} (GeV)")
       
        # mean
        fitF, params, cParams = "[1]*x + [0]", [0, 0], [False, False]
        doFit(jsIn, jsOut, "mean1", fitF, params, outDir_param, label, cParams=cParams, fitMin=0, fitMax=80, yMin=-5, yMax=50, yTitle = "#mu_{1} (GeV)", doFit=True)
        doFit(jsIn, jsOut, "mean2", fitF, params, outDir_param, label, cParams=cParams, fitMin=0, fitMax=80, yMin=-5, yMax=50, yTitle = "#mu_{2} (GeV)", doFit=True)

        # norm
        fitF, params, cParams = "[0]", [fitCfg['norm'][0]], [True]
        doFit(jsIn, jsOut, "norm1", fitF, params, outDir_param, label, cParams=cParams, fitMin=0, fitMax=50, yMin=0, yMax=1, yTitle = "n_{1} (GeV)", doFit=False)
        
        with open("%s/results.json" % outDir_param, "w") as outfile: json.dump(jsOut, outfile, indent=4)
    

    ###### STEP 3: refit
    if False:
        with open("%s/results.json" % outDir_param) as f: jsIn = json.load(f)
        jsOut = combinedFit(bkg, comp, jsIn, rebin=rebin, qTmax=50, singleMean=True)
        with open("%s/results_refit.json" % outDir_param, "w") as outfile: json.dump(jsOut, outfile, indent=4)


    ###### STEP 4: validate
    if True:
        fitCfg['mean_cfg'] = [1, 1]
        fitCfg['sigma_cfg'] = [2, 2]
        fitCfg['norm_cfg'] = [1]
        outDir_refit = "%s/refit_v0" % baseDir
        with open("%s/results_refit.json" % outDir_param) as f: jsIn = json.load(f)
        doFitMultiGauss(bkg, comp, fitCfg, label, outDir_refit, funcJs=jsIn, qTmax=200, rebin=rebin, xMin=-200, xMax=200, singleMean=True)
 
def do_ewk_perp_DeepMETReso():

    proc = "EWK"
    tag = "ewk_perp"
    comp = "perp"
    baseDir = "%s/%s" % (outDir, tag)
    name = "EWK #rightarrow #mu^{+}#mu^{#minus}"
    label = "%s, %s" % (name, met)
    functions.prepareDir(baseDir, False)
    rebin = 2

    ###### STEP 1: floating fits
    iteration = 0
    outDir_fits = "%s/fits_v0" % baseDir
    fitCfg = {}
    fitCfg['mean'], fitCfg['mean_cfg'] = [0, 0], [1, 1]
    fitCfg['sigma'], fitCfg['sigma_cfg'] = [20, 50], [0, 0]
    fitCfg['norm'], fitCfg['norm_cfg'] = [0.9], [0]
    nGauss = len(fitCfg['mean'])
    
    if True: 
        recoilLibs.doFitMultiGauss(rFile, proc, comp, fitCfg, label, outDir_fits, recoil_qTbins, qTmax=500, rebin=rebin, singleMean=True, xMin=-200, xMax=200)
        return
        
        
    ###### STEP 2: parameterize
    outDir_param = "%s/param_v0" % baseDir
    if True:
        jsOut = {}
        jsOut['nGauss'] = nGauss
        functions.prepareDir(outDir_param, False)
        with open("%s/results.json" % outDir_fits) as f: jsIn = json.load(f)
        
        # sigma
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [8.69746e-01, 5.59970e-01, 8], [False, False, False]
        doFit(jsIn, jsOut, "sigma1", fitF, params, outDir_param, label, cParams=cParams, fitMin=0, fitMax=100, yMin=0, yMax=100, yTitle = "#sigma_{1} (GeV)")
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [8.69746e-01, 5.59970e-01, 8], [False, False, False]
        doFit(jsIn, jsOut, "sigma2", fitF, params, outDir_param, label,cParams=cParams,  fitMin=0, fitMax=50, yMin=0, yMax=100, yTitle = "#sigma_{2} (GeV)")
       
        # mean
        fitF, params, cParams = "[0]", [fitCfg['mean'][0]], [True]
        doFit(jsIn, jsOut, "mean1", fitF, params, outDir_param, label, cParams=cParams, fitMin=0, fitMax=50, yMin=-5, yMax=5, yTitle = "#mu_{1} (GeV)", doFit=False)
        fitF, params, cParams = "[0]", [fitCfg['mean'][1]], [True]
        doFit(jsIn, jsOut, "mean2", fitF, params, outDir_param, label, cParams=cParams, fitMin=0, fitMax=50, yMin=-5, yMax=5, yTitle = "#mu_{2} (GeV)", doFit=False)

        # norm
        fitF, params, cParams = "[0]", [fitCfg['norm'][0]], [True]
        doFit(jsIn, jsOut, "norm1", fitF, params, outDir_param, label, cParams=cParams, fitMin=0, fitMax=50, yMin=0, yMax=1, yTitle = "n_{1} (GeV)", doFit=False)
        
        with open("%s/results.json" % outDir_param, "w") as outfile: json.dump(jsOut, outfile, indent=4)
    

    ###### STEP 3: refit
    if True:
        with open("%s/results.json" % outDir_param) as f: jsIn = json.load(f)
        jsOut = combinedFit(bkg, comp, jsIn, rebin=rebin, qTmax=50)
        with open("%s/results_refit.json" % outDir_param, "w") as outfile: json.dump(jsOut, outfile, indent=4)


    ###### STEP 4: validate
    if True:
        fitCfg['mean_cfg'] = [1, 1]
        fitCfg['sigma_cfg'] = [2, 2]
        fitCfg['norm_cfg'] = [1]
        outDir_refit = "%s/refit_v0" % baseDir
        with open("%s/results_refit.json" % outDir_param) as f: jsIn = json.load(f)
        doFitMultiGauss(bkg, comp, fitCfg, label, outDir_refit, funcJs=jsIn, qTmax=200, rebin=rebin, xMin=-200, xMax=200)

def do_ewk_para_DeepMETReso():

    proc = "EWK"
    tag = "ewk_para"
    comp = "para"
    baseDir = "%s/%s" % (outDir, tag)
    name = "EWK #rightarrow #mu^{+}#mu^{#minus}"
    label = "%s, %s" % (name, met)
    functions.prepareDir(baseDir, False)
    rebin = 2

    ###### STEP 1: floating fits
    iteration = 0
    outDir_fits = "%s/fits_v0" % baseDir
    fitCfg = {}
    fitCfg['mean'], fitCfg['mean_cfg'] = [0, 0], [0, 0]
    fitCfg['sigma'], fitCfg['sigma_cfg'] = [10, 50], [0, 0]
    fitCfg['norm'], fitCfg['norm_cfg'] = [0.95], [0]
    nGauss = len(fitCfg['mean'])
    
    if True: 
        recoilLibs.doFitMultiGauss(rFile, proc, comp, fitCfg, label, outDir_fits, recoil_qTbins, qTmax=500, rebin=rebin, singleMean=False, xMin=-200, xMax=200)
        return
    
    ###### STEP 2: parameterize
    outDir_param = "%s/param_v0" % baseDir
    if False:
        jsOut = {}
        jsOut['nGauss'] = nGauss
        functions.prepareDir(outDir_param, False)
        with open("%s/results.json" % outDir_fits) as f: jsIn = json.load(f)
        
        # sigma
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [8.69746e-01, 5.59970e-01, 8], [False, False, False]
        doFit(jsIn, jsOut, "sigma1", fitF, params, outDir_param, label, cParams=cParams, fitMin=0, fitMax=100, yMin=0, yMax=100, yTitle = "#sigma_{1} (GeV)")
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [8.69746e-01, 5.59970e-01, 8], [False, False, False]
        doFit(jsIn, jsOut, "sigma2", fitF, params, outDir_param, label,cParams=cParams,  fitMin=0, fitMax=50, yMin=0, yMax=100, yTitle = "#sigma_{2} (GeV)")
       
        # mean
        fitF, params, cParams = "[1]*x + [0]", [0, 0], [False, False]
        doFit(jsIn, jsOut, "mean1", fitF, params, outDir_param, label, cParams=cParams, fitMin=0, fitMax=80, yMin=-5, yMax=50, yTitle = "#mu_{1} (GeV)", doFit=True)
        doFit(jsIn, jsOut, "mean2", fitF, params, outDir_param, label, cParams=cParams, fitMin=0, fitMax=80, yMin=-5, yMax=50, yTitle = "#mu_{2} (GeV)", doFit=True)

        # norm
        fitF, params, cParams = "[0]", [fitCfg['norm'][0]], [True]
        doFit(jsIn, jsOut, "norm1", fitF, params, outDir_param, label, cParams=cParams, fitMin=0, fitMax=50, yMin=0, yMax=1, yTitle = "n_{1} (GeV)", doFit=False)
        
        with open("%s/results.json" % outDir_param, "w") as outfile: json.dump(jsOut, outfile, indent=4)
    

    ###### STEP 3: refit
    if False:
        with open("%s/results.json" % outDir_param) as f: jsIn = json.load(f)
        jsOut = combinedFit(bkg, comp, jsIn, rebin=rebin, qTmax=50, singleMean=True)
        with open("%s/results_refit.json" % outDir_param, "w") as outfile: json.dump(jsOut, outfile, indent=4)


    ###### STEP 4: validate
    if True:
        fitCfg['mean_cfg'] = [1, 1]
        fitCfg['sigma_cfg'] = [2, 2]
        fitCfg['norm_cfg'] = [1]
        outDir_refit = "%s/refit_v0" % baseDir
        with open("%s/results_refit.json" % outDir_param) as f: jsIn = json.load(f)
        doFitMultiGauss(bkg, comp, fitCfg, label, outDir_refit, funcJs=jsIn, qTmax=200, rebin=rebin, xMin=-200, xMax=200, singleMean=True)

def do_ttbar_perp_DeepMETReso():

    proc = "TTbar"
    tag = "ttbar_perp"
    comp = "perp"
    baseDir = "%s/%s" % (outDir, tag)
    name = "TTbar #rightarrow #mu^{+}#mu^{#minus}"
    label = "%s, %s" % (name, met)
    functions.prepareDir(baseDir, False)
    rebin = 2

    ###### STEP 1: floating fits
    iteration = 0
    outDir_fits = "%s/fits_v0" % baseDir
    fitCfg = {}
    fitCfg['mean'], fitCfg['mean_cfg'] = [0, 0], [0, 0]
    fitCfg['sigma'], fitCfg['sigma_cfg'] = [40, 70], [0, 0]
    fitCfg['norm'], fitCfg['norm_cfg'] = [0.5], [0]
    nGauss = len(fitCfg['mean'])
    
    if True: 
        recoilLibs.doFitMultiGauss(rFile, proc, comp, fitCfg, label, outDir_fits, recoil_qTbins, qTmax=500, rebin=rebin, singleMean=True, xMin=-200, xMax=200)
        return
    
    ###### STEP 2: parameterize
    outDir_param = "%s/param_v0" % baseDir
    if True:
        jsOut = {}
        jsOut['nGauss'] = nGauss
        functions.prepareDir(outDir_param, False)
        with open("%s/results.json" % outDir_fits) as f: jsIn = json.load(f)
        
        # sigma
        fitF, params, cParams = "[0]*x*x + [1]*x + [2]", [0, 0, 0], [False, False, False]
        doFit(jsIn, jsOut, "sigma1", fitF, params, outDir_param, label, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=100, yTitle = "#sigma_{1} (GeV)")
        fitF, params, cParams = "[0]*x*x + [1]*x + [2]", [0, 0, 0], [False, False, False]
        doFit(jsIn, jsOut, "sigma2", fitF, params, outDir_param, label,cParams=cParams,  fitMin=0, fitMax=150, yMin=0, yMax=100, yTitle = "#sigma_{2} (GeV)")
       
        # mean
        fitF, params, cParams = "[0]", [fitCfg['mean'][0]], [True]
        doFit(jsIn, jsOut, "mean1", fitF, params, outDir_param, label, cParams=cParams, fitMin=0, fitMax=50, yMin=-5, yMax=5, yTitle = "#mu_{1} (GeV)", doFit=False)
        fitF, params, cParams = "[0]", [fitCfg['mean'][1]], [True]
        doFit(jsIn, jsOut, "mean2", fitF, params, outDir_param, label, cParams=cParams, fitMin=0, fitMax=50, yMin=-5, yMax=5, yTitle = "#mu_{2} (GeV)", doFit=False)

        # norm
        fitF, params, cParams = "[0]", [fitCfg['norm'][0]], [True]
        doFit(jsIn, jsOut, "norm1", fitF, params, outDir_param, label, cParams=cParams, fitMin=0, fitMax=50, yMin=0, yMax=1, yTitle = "n_{1} (GeV)", doFit=False)
        
        with open("%s/results.json" % outDir_param, "w") as outfile: json.dump(jsOut, outfile, indent=4)
    

    ###### STEP 3: refit
    if False:
        with open("%s/results.json" % outDir_param) as f: jsIn = json.load(f)
        jsOut = combinedFit(bkg, comp, jsIn, rebin=rebin, qTmax=150)
        with open("%s/results_refit.json" % outDir_param, "w") as outfile: json.dump(jsOut, outfile, indent=4)


    ###### STEP 4: validate
    if False:
        fitCfg['mean_cfg'] = [1, 1]
        fitCfg['sigma_cfg'] = [2, 2]
        fitCfg['norm_cfg'] = [1]
        outDir_refit = "%s/refit_v0" % baseDir
        with open("%s/results_refit.json" % outDir_param) as f: jsIn = json.load(f)
        doFitMultiGauss(bkg, comp, fitCfg, label, outDir_refit, funcJs=jsIn, qTmax=200, rebin=rebin, xMin=-200, xMax=200)

def do_ttbar_para_DeepMETReso():

    proc = "TTbar"
    tag = "ttbar_para"
    comp = "para"
    baseDir = "%s/%s" % (outDir, tag)
    name = "TTbar #rightarrow #mu^{+}#mu^{#minus}"
    label = "%s, %s" % (name, met)
    functions.prepareDir(baseDir, False)
    rebin = 2

    ###### STEP 1: floating fits
    iteration = 0
    outDir_fits = "%s/fits_v0" % baseDir
    fitCfg = {}
    fitCfg['mean'], fitCfg['mean_cfg'] = [0, 0], [0, 0]
    fitCfg['sigma'], fitCfg['sigma_cfg'] = [40, 70], [0, 0]
    fitCfg['norm'], fitCfg['norm_cfg'] = [0.5], [0]
    nGauss = len(fitCfg['mean'])
    
    if True: 
        recoilLibs.doFitMultiGauss(rFile, proc, comp, fitCfg, label, outDir_fits, recoil_qTbins, qTmax=500, rebin=rebin, singleMean=False, xMin=-200, xMax=200)
        return

    
    ###### STEP 2: parameterize
    outDir_param = "%s/param_v0" % baseDir
    if True:
        jsOut = {}
        jsOut['nGauss'] = nGauss
        functions.prepareDir(outDir_param, False)
        with open("%s/results.json" % outDir_fits) as f: jsIn = json.load(f)
        
        # sigma
        fitF, params, cParams = "[0]*x*x*x + [1]*x*x + [2]*x + [3]", [0, 0, 0, 0], [False, False, False, False]
        doFit(jsIn, jsOut, comp, "sigma1", fitF, params, outDir_param, label, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=100, yTitle = "#sigma_{1} (GeV)")
        fitF, params, cParams = "[0]*x*x*x + [1]*x*x + [2]*x + [3]", [0, 0, 0, 0], [False, False, False, False]
        doFit(jsIn, jsOut, comp, "sigma2", fitF, params, outDir_param, label,cParams=cParams,  fitMin=0, fitMax=150, yMin=0, yMax=100, yTitle = "#sigma_{2} (GeV)")
                
        # mean
        fitF, params, cParams = "[0]*x*x*x + [1]*x*x + [2]*x + [3]", [0, 0, 0, 0], [False, False, False, False]
        doFit(jsIn, jsOut, comp, "mean1", fitF, params, outDir_param, label, cParams=cParams, fitMin=0, fitMax=150, yMin=-20, yMax=50, yTitle = "#mu_{1} (GeV)", doFit=True)
        fitF, params, cParams = "[0]*x*x*x + [1]*x*x + [2]*x + [3]", [0, 0, 0, 0], [False, False, False, False]
        doFit(jsIn, jsOut, comp, "mean2", fitF, params, outDir_param, label, cParams=cParams, fitMin=0, fitMax=150, yMin=-20, yMax=50, yTitle = "#mu_{2} (GeV)", doFit=True)

        # norm
        fitF, params, cParams = "[0]", [fitCfg['norm'][0]], [True]
        doFit(jsIn, jsOut, comp, "norm1", fitF, params, outDir_param, label, cParams=cParams, fitMin=0, fitMax=50, yMin=0, yMax=1, yTitle = "n_{1} (GeV)", doFit=False)
        
        with open("%s/results.json" % outDir_param, "w") as outfile: json.dump(jsOut, outfile, indent=4)
    

    ###### STEP 3: refit
    if False:
        with open("%s/results.json" % outDir_param) as f: jsIn = json.load(f)
        jsOut = combinedFit(bkg, comp, jsIn, rebin=rebin, qTmax=150, singleMean=True)
        with open("%s/results_refit.json" % outDir_param, "w") as outfile: json.dump(jsOut, outfile, indent=4)


    ###### STEP 4: validate
    if False:
        fitCfg['mean_cfg'] = [1, 1]
        fitCfg['sigma_cfg'] = [2, 2]
        fitCfg['norm_cfg'] = [1]
        outDir_refit = "%s/refit_v0" % baseDir
        with open("%s/results_refit.json" % outDir_param) as f: jsIn = json.load(f)
        doFitMultiGauss(bkg, comp, fitCfg, label, outDir_refit, funcJs=jsIn, qTmax=200, rebin=rebin, xMin=-200, xMax=200, singleMean=True)

 
def do_dymumu_perp_DeepMETReso():

    proc = "Zmumu"
    tag = "dymumu_perp"
    comp = "perp"
    baseDir = "%s/%s" % (outDir, tag)
    name = "DY #rightarrow #mu^{+}#mu^{#minus}"
    label = "%s, %s" % (name, met)
    functions.prepareDir(baseDir, False)
    rebin = 1

    ###### STEP 1: floating fits
    iteration = 0
    outDir_fits = "%s/fits_v0" % baseDir
    fitCfg = {}

    fitCfg['mean'], fitCfg['mean_cfg'] = [0, 0, 0, 0], [1, 1, 1, 1] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
    fitCfg['sigma'], fitCfg['sigma_cfg'] = [25, 5, 9, 14], [0, 0, 0, 0]
    fitCfg['norm'], fitCfg['norm_cfg'] = [0.007, 0.30, 0.55], [1, 0, 0]
    
    nGauss = len(fitCfg['mean'])
    
    if True: 
        recoilLibs.doFitMultiGauss(rFile, proc, comp, fitCfg, label, outDir_fits, recoil_qTbins, qTmax=500, rebin=rebin, singleMean=True, xMin=-150, xMax=150)
        return
        
        
  

    ###### STEP 2: parameterize
    outDir_param = "%s/param_v0" % baseDir
    if False:
        jsOut = {}
        jsOut['nGauss'] = nGauss
        functions.prepareDir(outDir_param, False)
        with open("%s/results.json" % outDir_fits) as f: jsIn = json.load(f)
        
        # sigma
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [1.98628e-01, 5.72942e-01, 3.83471e+00], [False, False, False]
        doFit(jsIn, jsOut, "sigma1", fitF, params, outDir_param, label, cParams=cParams, fitMin=0, fitMax=50, yMin=0, yMax=20, yTitle = "#sigma_{1} (GeV)")
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [1.45588e-01, 6.81281e-01, 7.02926e+00], [False, False, False]
        doFit(jsIn, jsOut, "sigma2", fitF, params, outDir_param, label,cParams=cParams,  fitMin=0, fitMax=50, yMin=0, yMax=20, yTitle = "#sigma_{2} (GeV)")
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [4.61160e-01, 4.48764e-01, 9.41019e+00], [False, False, False]
        doFit(jsIn, jsOut, "sigma3", fitF, params, outDir_param, label, cParams=cParams, fitMin=0, fitMax=50, yMin=0, yMax=20, yTitle = "#sigma_{3} (GeV)")
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [1.24638e-01, 1, 1.36732e+01], [False, False, False]
        doFit(jsIn, jsOut, "sigma4", fitF, params, outDir_param, label, cParams=cParams, fitMin=0, fitMax=50, yMin=0, yMax=20, yTitle = "#sigma_{4} (GeV)")
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [1.24638e-01, 1, 1.36732e+01], [False, False, False]
        doFit(jsIn, jsOut, "sigma5", fitF, params, outDir_param, label, cParams=cParams, fitMin=0, fitMax=50, yMin=10, yMax=50, yTitle = "#sigma_{5} (GeV)")

        # mean
        fitF, params, cParams = "[0]", [fitCfg['mean'][0]], [True]
        doFit(jsIn, jsOut, "mean1", fitF, params, outDir_param, label, cParams=cParams, fitMin=0, fitMax=50, yMin=-5, yMax=5, yTitle = "#mu_{1} (GeV)", doFit=False)
        fitF, params, cParams = "[0]", [fitCfg['mean'][1]], [True]
        doFit(jsIn, jsOut, "mean2", fitF, params, outDir_param, label, cParams=cParams, fitMin=0, fitMax=50, yMin=-5, yMax=5, yTitle = "#mu_{2} (GeV)", doFit=False)
        fitF, params, cParams = "[0]", [fitCfg['mean'][2]], [True]
        doFit(jsIn, jsOut, "mean3", fitF, params, outDir_param, label, cParams=cParams, fitMin=0, fitMax=50, yMin=-5, yMax=5, yTitle = "#mu_{3} (GeV)", doFit=False)
        fitF, params, cParams = "[0]", [fitCfg['mean'][3]], [True]
        doFit(jsIn, jsOut, "mean4", fitF, params, outDir_param, label, cParams=cParams, fitMin=0, fitMax=50, yMin=-5, yMax=5, yTitle = "#mu_{4} (GeV)", doFit=False)
        fitF, params, cParams = "[0]", [fitCfg['mean'][4]], [True]
        doFit(jsIn, jsOut, "mean5", fitF, params, outDir_param, label, cParams=cParams, fitMin=0, fitMax=50, yMin=-5, yMax=5, yTitle = "#mu_{5} (GeV)", doFit=False)

        # norm
        fitF, params, cParams = "[0]", [fitCfg['norm'][0]], [True]
        doFit(jsIn, jsOut, "norm1", fitF, params, outDir_param, label, cParams=cParams, fitMin=0, fitMax=50, yMin=0, yMax=1, yTitle = "n_{1} (GeV)", doFit=False)
        fitF, params, cParams = "[0]", [fitCfg['norm'][1]], [True]
        doFit(jsIn, jsOut, "norm2", fitF, params, outDir_param, label, cParams=cParams, fitMin=0, fitMax=50, yMin=0, yMax=1, yTitle = "n_{2} (GeV)", doFit=False)
        fitF, params, cParams = "[0]", [fitCfg['norm'][2]], [True]
        doFit(jsIn, jsOut, "norm3", fitF, params, outDir_param, label, cParams=cParams, fitMin=0, fitMax=50, yMin=0, yMax=1, yTitle = "n_{3} (GeV)", doFit=False)
        fitF, params, cParams = "[0]", [fitCfg['norm'][3]], [True]
        doFit(jsIn, jsOut, "norm4", fitF, params, outDir_param, label, cParams=cParams, fitMin=0, fitMax=50, yMin=0, yMax=1, yTitle = "n_{4} (GeV)", doFit=False)

        with open("%s/results.json" % outDir_param, "w") as outfile: json.dump(jsOut, outfile, indent=4)
    

    ###### STEP 3: refit
    if True:
        with open("%s/results.json" % outDir_param) as f: jsIn = json.load(f)
        jsOut = combinedFit(proc, comp, jsIn, rebin=rebin)
        with open("%s/results_refit.json" % outDir_param, "w") as outfile: json.dump(jsOut, outfile, indent=4)


    ###### STEP 4: validate
    if True:
        fitCfg['mean_cfg'] = [1, 1, 1, 1] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
        fitCfg['sigma_cfg'] = [2, 2, 2, 2]
        fitCfg['norm_cfg'] = [1, 1, 1]
        
        fitCfg['mean_cfg'] = [2, 2, 2, 2, 2] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
        fitCfg['sigma_cfg'] = [2, 2, 2, 2, 2]
        fitCfg['norm_cfg'] = [1, 1, 1, 1]
        outDir_refit = "%s/refit_v0" % baseDir
        with open("%s/results_refit.json" % outDir_param) as f: jsIn = json.load(f)
        doFitMultiGauss(proc, comp, fitCfg, label, outDir_refit, funcJs=jsIn, qTmax=200, rebin=rebin)


def do_dymumu_para_DeepMETReso():

    proc = "Zmumu"
    tag = "dymumu_para"
    comp = "para"
    baseDir = "%s/%s" % (outDir, tag)
    name = "DY #rightarrow #mu^{+}#mu^{#minus}"
    label = "%s, %s" % (name, met)
    functions.prepareDir(baseDir, False)
    rebin = 1

    ###### STEP 1: floating fits
    outDir_fits = "%s/fits_v0" % baseDir
    fitCfg = {}

    ## 1 GeV; floating means
    fitCfg['mean'], fitCfg['mean_cfg'] = [0, 0, 0, 0], [0, 0, 0, 0] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
    fitCfg['sigma'], fitCfg['sigma_cfg'] = [30, 5, 9, 14], [0, 0, 0, 0]
    fitCfg['norm'], fitCfg['norm_cfg'] = [0.005, 0.30, 0.55], [1, 0, 0]
    
    # 0.5 GeV
    fitCfg['mean'], fitCfg['mean_cfg'] = [0, 0, 0, 0], [0, 0, 0, 0] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
    fitCfg['sigma'], fitCfg['sigma_cfg'] = [30, 5, 8, 10], [0, 0, 0, 0]
    fitCfg['norm'], fitCfg['norm_cfg'] = [0.005, 0.25, 0.25], [0, 0, 0]
    
    nGauss = len(fitCfg['mean'])
    
    if True: 
        recoilLibs.doFitMultiGauss(rFile, proc, comp, fitCfg, label, outDir_fits, recoil_qTbins, qTmax=500, rebin=rebin, singleMean=True, xMin=-150, xMax=150)
        return

    ###### STEP 2: parameterize
    outDir_param = "%s/param_v0" % baseDir
    if False:
        jsOut = {}
        jsOut['nGauss'] = nGauss
        functions.prepareDir(outDir_param, False)
        with open("%s/results.json" % outDir_fits) as f: jsIn = json.load(f)
        
        # sigma
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [1.98628e-01, 5.72942e-01, 3.83471e+00], [False, False, False]
        doFit(jsIn, jsOut, "sigma1", fitF, params, outDir_param, label, cParams=cParams, fitMin=0, fitMax=50, yMin=0, yMax=20, yTitle = "#sigma_{1} (GeV)")
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [1.45588e-01, 6.81281e-01, 7.02926e+00], [False, False, False]
        doFit(jsIn, jsOut, "sigma2", fitF, params, outDir_param, label,cParams=cParams,  fitMin=0, fitMax=50, yMin=0, yMax=20, yTitle = "#sigma_{2} (GeV)")
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [4.61160e-01, 4.48764e-01, 9.41019e+00], [False, False, False]
        doFit(jsIn, jsOut, "sigma3", fitF, params, outDir_param, label, cParams=cParams, fitMin=0, fitMax=50, yMin=0, yMax=20, yTitle = "#sigma_{3} (GeV)")
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [1.24638e-01, 1, 1.36732e+01], [False, False, False]
        doFit(jsIn, jsOut, "sigma4", fitF, params, outDir_param, label, cParams=cParams, fitMin=0, fitMax=50, yMin=0, yMax=20, yTitle = "#sigma_{4} (GeV)")
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [1.24638e-01, 1, 1.36732e+01], [False, False, False]
        doFit(jsIn, jsOut, "sigma5", fitF, params, outDir_param, label, cParams=cParams, fitMin=0, fitMax=50, yMin=10, yMax=50, yTitle = "#sigma_{5} (GeV)")

        # mean
        fitF, params, cParams = "(x<20)*1*([0] + [1]*x + [2]*x*x + [3]*x*x*x) +0+ (x>20)*1*([4] + [5]*x)", [0, 0, 0, 0, 0, 0], [False, False, False, False, False, False]
        doFit(jsIn, jsOut, "mean1", fitF, params, outDir_param, label, cParams=cParams, fitMin=0, fitMax=100, yMin=0, yMax=20, yTitle = "#mu_{1} (GeV)", doFit=True)
        doFit(jsIn, jsOut, "mean2", fitF, params, outDir_param, label, cParams=cParams, fitMin=0, fitMax=50, yMin=0, yMax=20, yTitle = "#mu_{2} (GeV)", doFit=True)
        doFit(jsIn, jsOut, "mean3", fitF, params, outDir_param, label, cParams=cParams, fitMin=0, fitMax=50, yMin=0, yMax=20, yTitle = "#mu_{3} (GeV)", doFit=True)
        doFit(jsIn, jsOut, "mean4", fitF, params, outDir_param, label, cParams=cParams, fitMin=0, fitMax=50, yMin=0, yMax=20, yTitle = "#mu_{4} (GeV)", doFit=True)
        doFit(jsIn, jsOut, "mean5", fitF, params, outDir_param, label, cParams=cParams, fitMin=0, fitMax=50, yMin=0, yMax=20, yTitle = "#mu_{5} (GeV)", doFit=True)

        # norm
        fitF, params, cParams = "[0]", [fitCfg['norm'][0]], [True]
        doFit(jsIn, jsOut, "norm1", fitF, params, outDir_param, label, cParams=cParams, fitMin=0, fitMax=50, yMin=0, yMax=1, yTitle = "n_{1} (GeV)", doFit=False)
        fitF, params, cParams = "[0]", [fitCfg['norm'][1]], [True]
        doFit(jsIn, jsOut, "norm2", fitF, params, outDir_param, label, cParams=cParams, fitMin=0, fitMax=50, yMin=0, yMax=1, yTitle = "n_{2} (GeV)", doFit=False)
        fitF, params, cParams = "[0]", [fitCfg['norm'][2]], [True]
        doFit(jsIn, jsOut, "norm3", fitF, params, outDir_param, label, cParams=cParams, fitMin=0, fitMax=50, yMin=0, yMax=1, yTitle = "n_{3} (GeV)", doFit=False)
        fitF, params, cParams = "[0]", [fitCfg['norm'][3]], [True]
        doFit(jsIn, jsOut, "norm4", fitF, params, outDir_param, label, cParams=cParams, fitMin=0, fitMax=50, yMin=0, yMax=1, yTitle = "n_{4} (GeV)", doFit=False)
        
        with open("%s/results.json" % outDir_param, "w") as outfile: json.dump(jsOut, outfile, indent=4)
    

    ###### STEP 3: refit
    if True:
        with open("%s/results.json" % outDir_param) as f: jsIn = json.load(f)
        jsOut = combinedFit(proc, comp, jsIn, rebin=rebin, singleMean=True)
        with open("%s/results_refit.json" % outDir_param, "w") as outfile: json.dump(jsOut, outfile, indent=4)


    ###### STEP 4: validate
    if True:
        fitCfg['mean_cfg'] = [2, 2, 2, 2, 2] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
        fitCfg['sigma_cfg'] = [2, 2, 2, 2, 2]
        fitCfg['norm_cfg'] = [2, 2, 2, 2]
        outDir_refit = "%s/refit_v0" % baseDir
        with open("%s/results_refit.json" % outDir_param) as f: jsIn = json.load(f)
        #doFitMultiGauss(proc, comp, fitCfg, label, outDir_refit, funcJs=jsIn, qTmax=200, rebin=rebin)

        outDir_refit = "%s/param_v0_refit" % baseDir
        functions.prepareDir(outDir_refit)
        doPlot(comp, "sigma1", jsIn, outDir_refit, label, yMin=0, yMax=20, yTitle = "#sigma_{1} (GeV)")
        doPlot(comp, "sigma2", jsIn, outDir_refit, label, yMin=0, yMax=20, yTitle = "#sigma_{2} (GeV)")
        doPlot(comp, "sigma3", jsIn, outDir_refit, label, yMin=0, yMax=20, yTitle = "#sigma_{3} (GeV)")
        doPlot(comp, "sigma4", jsIn, outDir_refit, label, yMin=0, yMax=20, yTitle = "#sigma_{4} (GeV)")
        doPlot(comp, "sigma5", jsIn, outDir_refit, label, yMin=0, yMax=20, yTitle = "#sigma_{5} (GeV)")
        doPlot(comp, "mean1", jsIn, outDir_refit, label, yMin=0, yMax=20, yTitle = "#mu_{1} (GeV)")
        doPlot(comp, "mean2", jsIn, outDir_refit, label, yMin=0, yMax=20, yTitle = "#mu_{2} (GeV)")
        doPlot(comp, "mean3", jsIn, outDir_refit, label, yMin=0, yMax=20, yTitle = "#mu_{3} (GeV)")

 
def do_singlemuon_perp_DeepMETReso():

    proc = "Data"
    tag = "singlemuon_perp"
    comp = "perp"
    baseDir = "%s/%s" % (outDir, tag)
    name = "SingleMuon #rightarrow #mu^{+}#mu^{#minus}"
    label = "%s, %s" % (name, met)
    functions.prepareDir(baseDir, False)
    rebin = 1

    ###### STEP 1: floating fits
    iteration = 0
    outDir_fits_v0 = "%s/fits_v0" % baseDir
    fitCfg = {}
    fitCfg['mean'], fitCfg['mean_cfg'] = [0, 0, 0,0], [0, 0, 0,0] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
    fitCfg['sigma'], fitCfg['sigma_cfg'] = [35, 5, 8, 20], [0, 0, 0,0]
    fitCfg['norm'], fitCfg['norm_cfg'] = [0.005, 0.1, 0.6], [1, 0,0]
    
    fitCfg['mean'], fitCfg['mean_cfg'] = [0, 0, 0, 0], [1, 1, 1, 1] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
    fitCfg['sigma'], fitCfg['sigma_cfg'] = [25, 5, 9, 14], [0, 0, 0, 0]
    fitCfg['norm'], fitCfg['norm_cfg'] = [0.007, 0.30, 0.55], [1, 0, 0]

    bkgCfg = {}
    bkgCfg['TTbar'] = { "filename": "/eos/user/j/jaeyserm/www/wmass/highPU/recoilCorrection/recoil_mumu_DeepMETReso/ttbar_perp/fits_v0/results.json", "norm": 1.0, "float": False, "subtract": False }
    bkgCfg['EWK'] = { "filename": "/eos/user/j/jaeyserm/www/wmass/highPU/recoilCorrection/recoil_mumu_DeepMETReso/ewk_perp/fits_v0/results.json", "norm": 1.0, "float": False, "subtract": False }
    bkgCfg['ZZ'] = { "filename": "/eos/user/j/jaeyserm/www/wmass/highPU/recoilCorrection/recoil_mumu_DeepMETReso/zz_perp/fits_v0/results.json", "norm": 1.0, "float": False, "subtract": False }
    nGauss = len(fitCfg['mean'])
    if True: 
        recoilLibs.doFitMultiGauss(rFile, proc, comp, fitCfg, label, outDir_fits_v0, recoil_qTbins, qTmax=500, rebin=rebin, bkgCfg=bkgCfg, singleMean=True, xMin=-150, xMax=150)
        return
    
    ###### STEP 2: parameterize sigma1
    outDir_param_v0 = "%s/param_v0" % baseDir
    if False:
        jsOut = {}
        jsOut['nGauss'] = nGauss
        functions.prepareDir(outDir_param_v0, True)
        with open("%s/results.json" % outDir_fits_v0) as f: jsIn = json.load(f)
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [7.26314e-01, 2.49241e-01, 4.41560e+00], [False, False, False]
        ###doFit(jsIn, jsOut, comp, "sigma1", fitF, params, outDir_param_v0, label, cParams=cParams, fitMin=0, fitMax=60, yMin=0, yMax=20, yTitle = "#sigma_{1} (GeV)")
        doFit(jsIn, jsOut, comp, "sigma1", fitF, params, outDir_param_v0, label, cParams=cParams, fitMin=0, fitMax=60, yMin=0, yMax=20, yTitle = "#sigma_{1} (GeV)")
 
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [2.44857e-01, 5.19971e-01, 4.91301e+00], [False, False, False]
        doFit(jsIn, jsOut, comp, "sigma2", fitF, params, outDir_param_v0, label, cParams=cParams, fitMin=0, fitMax=120, yMin=0, yMax=20, yTitle = "#sigma_{2} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [1, 0, 10], [False, False, False]
        doFit(jsIn, jsOut, comp, "sigma3", fitF, params, outDir_param_v0, label, cParams=cParams, fitMin=0, fitMax=120, yMin=0, yMax=30, yTitle = "#sigma_{3} (GeV)")
        
        with open("%s/results.json" % outDir_param_v0, "w") as outfile: json.dump(jsOut, outfile, indent=4)
        sys.exit()
        
        
    ###### STEP 3: fit with parameterized sigma1
    outDir_fits_v1 = "%s/fits_v1" % baseDir
    if False:
        fitCfg['mean_cfg'] = [1, 1, 1] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
        fitCfg['sigma_cfg'] = [2, 0, 0]
        fitCfg['norm_cfg'] = [1, 1]
        with open("%s/results.json" % outDir_param_v0) as f: jsIn = json.load(f)
        doFitMultiGauss(proc, comp, fitCfg, label, outDir_fits_v1, funcJs=jsIn, qTmax=200, rebin=rebin, bkgCfg=bkgCfg)    
        sys.exit()
        
        
    ###### STEP 4: parameterize other parameters
    outDir_param_v1 = "%s/param_v1" % baseDir
    if False:
        jsOut = {}
        jsOut['nGauss'] = nGauss
        functions.prepareDir(outDir_param_v1, True)
        with open("%s/results.json" % outDir_fits_v1) as f: jsIn = json.load(f)
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [9.62059e+00, 2.88035e-02, -4.51838e+00], [False, False, False]
        doFit(jsIn, jsOut, comp, "sigma1", fitF, params, outDir_param_v1, label, cParams=cParams, fitMin=0, fitMax=60, yMin=0, yMax=20, yTitle = "#sigma_{1} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [2.44857e-01, 5.19971e-01, 4.91301e+00], [False, False, False]
        doFit(jsIn, jsOut, comp, "sigma2", fitF, params, outDir_param_v1, label, cParams=cParams, fitMin=0, fitMax=100, yMin=0, yMax=20, yTitle = "#sigma_{2} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [1.33181e-02, 1.33470e+00, 1.35768e+01], [False, True, False]
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [1.33181e-02, 1., 1.35768e+01], [False, True, False]
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [8.121e-02, 1., 1.306e+01], [False, True, False]
        doFit(jsIn, jsOut, comp, "sigma3", fitF, params, outDir_param_v1, label, cParams=cParams, fitMin=0, fitMax=120, yMin=0, yMax=30, yTitle = "#sigma_{3} (GeV)")
        #sys.exit()
        # mean
        fitF, params, cParams = "[0]", [fitCfg['mean'][0]], [True]
        doFit(jsIn, jsOut, comp, "mean1", fitF, params, outDir_param_v1, label, cParams=cParams, fitMin=0, fitMax=50, yMin=-5, yMax=5, yTitle = "#mu_{1} (GeV)", doFit=False)
        fitF, params, cParams = "[0]", [fitCfg['mean'][1]], [True]
        doFit(jsIn, jsOut, comp, "mean2", fitF, params, outDir_param_v1, label, cParams=cParams, fitMin=0, fitMax=50, yMin=-5, yMax=5, yTitle = "#mu_{2} (GeV)", doFit=False)
        fitF, params, cParams = "[0]", [fitCfg['mean'][2]], [True]
        doFit(jsIn, jsOut, comp, "mean3", fitF, params, outDir_param_v1, label, cParams=cParams, fitMin=0, fitMax=50, yMin=-5, yMax=5, yTitle = "#mu_{3} (GeV)", doFit=False)


        # norm
        fitF, params, cParams = "[0]", [fitCfg['norm'][0]], [True]
        doFit(jsIn, jsOut, comp, "norm1", fitF, params, outDir_param_v1, label, cParams=cParams, fitMin=0, fitMax=50, yMin=0, yMax=1, yTitle = "n_{1} (GeV)", doFit=False)
        fitF, params, cParams = "[0]", [fitCfg['norm'][1]], [True]
        doFit(jsIn, jsOut, comp, "norm2", fitF, params, outDir_param_v1, label, cParams=cParams, fitMin=0, fitMax=50, yMin=0, yMax=1, yTitle = "n_{2} (GeV)", doFit=False)

        with open("%s/results.json" % outDir_param_v1, "w") as outfile: json.dump(jsOut, outfile, indent=4)    
        sys.exit()
        
    ###### STEP 5: refit
    if True:
        with open("%s/results.json" % outDir_param_v1) as f: jsIn = json.load(f)
        jsOut = combinedFit(proc, comp, jsIn, rebin=rebin, bkgCfg=bkgCfg, qTmax=50)
        with open("%s/results_refit.json" % outDir_param_v1, "w") as outfile: json.dump(jsOut, outfile, indent=4)
        #sys.exit()
        
    ###### STEP 6: final validation
    if True:
        fitCfg['mean_cfg'] = [1, 1, 1] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
        fitCfg['sigma_cfg'] = [2, 2, 2]
        fitCfg['norm_cfg'] = [1, 1]

        outDir_refit = "%s/refit_v1" % baseDir
        with open("%s/results_refit.json" % outDir_param_v1) as f: jsIn = json.load(f)
        doFitMultiGauss(proc, comp, fitCfg, label, outDir_refit, funcJs=jsIn, qTmax=200, rebin=rebin, bkgCfg=bkgCfg)
 
        outDir_refit = "%s/param_v1_refit" % baseDir
        functions.prepareDir(outDir_refit)
        doPlot(comp, "sigma1", jsIn, outDir_refit, label, yMin=0, yMax=20, yTitle = "#sigma_{1} (GeV)")
        doPlot(comp, "sigma2", jsIn, outDir_refit, label, yMin=0, yMax=30, yTitle = "#sigma_{2} (GeV)")
        doPlot(comp, "sigma3", jsIn, outDir_refit, label, yMin=0, yMax=30, yTitle = "#sigma_{3} (GeV)")
 
def do_singlemuon_para_DeepMETReso():

    proc = "Data"
    tag = "singlemuon_para"
    comp = "para"
    baseDir = "%s/%s" % (outDir, tag)
    name = "SingleMuon #rightarrow #mu^{+}#mu^{#minus}"
    label = "%s, %s" % (name, met)
    functions.prepareDir(baseDir, False)
    rebin = 1
    #rebin = [-300, -100, -80, -70, -60, -50, -46, -42, -38, -34] + list(range(-30, 30, 1)) + [30, 34, 38, 42, 46, 50, 60, 70, 80, 100, 300]


    ###### STEP 1: floating fits
    iteration = 0
    outDir_fits_v0 = "%s/fits_v0" % baseDir
    fitCfg = {}
    fitCfg['mean'], fitCfg['mean_cfg'] = [0, 0, 0,0], [1, 0, 0, 0] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
    fitCfg['sigma'], fitCfg['sigma_cfg'] = [35, 5, 8, 20], [0, 0, 0, 0]
    fitCfg['norm'], fitCfg['norm_cfg'] = [0.005, 0.1, 0.6], [1, 0, 0]

    fitCfg['mean'], fitCfg['mean_cfg'] = [0, 0, 0, 0], [0, 0, 0, 0] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
    fitCfg['sigma'], fitCfg['sigma_cfg'] = [30, 5, 9, 14], [0, 0, 0, 0]
    fitCfg['norm'], fitCfg['norm_cfg'] = [0.005, 0.30, 0.55], [1, 0, 0]

    bkgCfg = {}
    bkgCfg['TTbar'] = { "filename": "/eos/user/j/jaeyserm/www/wmass/highPU/recoilCorrection/recoil_mumu_DeepMETReso/ttbar_para/fits_v0/results.json", "norm": 1.0, "float": False, "subtract": False }
    bkgCfg['EWK'] = { "filename": "/eos/user/j/jaeyserm/www/wmass/highPU/recoilCorrection/recoil_mumu_DeepMETReso/ewk_para/fits_v0/results.json", "norm": 1.0, "float": False, "subtract": False }
    bkgCfg['ZZ'] = { "filename": "/eos/user/j/jaeyserm/www/wmass/highPU/recoilCorrection/recoil_mumu_DeepMETReso/zz_para/fits_v0/results.json", "norm": 1.0, "float": False, "subtract": False }
    nGauss = len(fitCfg['mean'])
    if True: 
        recoilLibs.doFitMultiGauss(rFile, proc, comp, fitCfg, label, outDir_fits_v0, recoil_qTbins, qTmax=500, rebin=rebin, bkgCfg=bkgCfg, singleMean=False, xMin=-150, xMax=150)
        return

    
    ###### STEP 2: parameterize mean
    outDir_param_v0 = "%s/param_v0" % baseDir
    if False:
        jsOut = {}
        jsOut['nGauss'] = nGauss
        functions.prepareDir(outDir_param_v0, True)
        with open("%s/results.json" % outDir_fits_v0) as f: jsIn = json.load(f)

        # mean
        #fitF, params, cParams = "(x<30)*1*([0] + [1]*x + [2]*x*x + [3]*x*x*x) +0+ (x>30)*1*(    ([0] + [1]*30 + [2]*30*30 + [3]*30*30*30 - [4]*30) + [4]*x)", [1.54013e-01, 4.20263e-01, -1.38828e-02, 2.80192e-04, 9.44310e-02], [False, False, False, False, False, False]
        #fitF, params, cParams = "(x<[4])*1*([0] + [1]*x + [2]*x*x + [3]*x*x*x) +0+ (x>[4])*1*(    ([0] + [1]*[4] + [2]*[4]*[4] + [3]*[4]*[4]*[4] - ([1] + 2*[2]*[4] + 3*[3]*[4]*[4])*[4]) + ([1] + 2*[2]*[4] + 3*[3]*[4]*[4])*x)", [1.54013e-01, 4.20263e-01, -1.38828e-02, 2.80192e-04, 30], [False, False, False, False, False, False]
        #fitF, params, cParams = "(x<36)*1*([0] + [1]*x + [2]*x*x + [3]*x*x*x) +0+ (x>36)*1*(    ([0] + [1]*36 + [2]*36*36 + [3]*36*36*36 - ([1] + 2*[2]*36 + 3*[3]*36*36)*36) + ([1] + 2*[2]*36 + 3*[3]*36*36)*x  )", [0, 4.20263e-01, -1.38828e-02, 2.80192e-04], [True, False, False, False]
        #fitF, params, cParams = "(x<30)*1*([0] + [1]*x + [2]*x*x + [3]*x*x*x) +0+ (x>30)*1*(    ([0] + [1]*30 + [2]*30*30 + [3]*30*30*30 - ([1] + 2*[2]*30 + 3*[3]*30*30 - 2*[4]*30)*30 - [4]*30*30) + ([1] + 2*[2]*30 + 3*[3]*30*30 - 2*[4]*30)*x + [4]*x*x  )", [1.58212e-01, 4.03838e-01, -1.03807e-02, 1.21562e-04, -2.26096e-04], [False, False, False, False, False]
        #fitF, params, cParams = "(x<30)*1*([0] + [1]*x + [2]*x*x + [3]*x*x*x) +0+ (x>30)*1*(    ([0] + [1]*30 + [2]*30*30 + [3]*30*30*30 - ([1] + 2*[2]*30 + 3*[3]*30*30)*30) + ([1] + 2*[2]*30 + 3*[3]*30*30)*x )", [1.58212e-01, 4.03838e-01, -1.03807e-02, 1.21562e-04], [False, False, False, False]

        fitF, params, cParams = "(x<30)*1*([0] + [1]*x + [2]*x*x + [3]*x*x*x) +0+ (x>30)*1*(  ([0] + [1]*30 + [2]*30*30 + [3]*30*30*30 - [4]*30) + [4]*x   )", [2.45501e-01, 3.67274e-01, -7.26948e-03, 5.62285e-05, 8.95248e-02], [False, False, False, False, False]
        doFit(jsIn, jsOut, comp, "mean1", fitF, params, outDir_param_v0, label, cParams=cParams, fitMin=0, fitMax=140, yMin=0, yMax=50, yTitle = "#mu_{1} (GeV)", doFit=True)
        doFit(jsIn, jsOut, comp, "mean2", fitF, params, outDir_param_v0, label, cParams=cParams, fitMin=0, fitMax=140, yMin=0, yMax=50, yTitle = "#mu_{2} (GeV)", doFit=True)

        with open("%s/results.json" % outDir_param_v0, "w") as outfile: json.dump(jsOut, outfile, indent=4)
        sys.exit()
        
    ###### STEP 3: refit with fixed means
    outDir_fits_v1 = "%s/fits_v1" % baseDir
    if False:
        fitCfg['mean_cfg'] = [2, 2] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
        fitCfg['sigma_cfg'] = [0, 0]
        fitCfg['norm_cfg'] = [0]

        with open("%s/results.json" % outDir_param_v0) as f: jsIn = json.load(f)
        doFitMultiGauss(proc, comp, fitCfg, label, outDir_fits_v1, funcJs=jsIn, qTmax=150, rebin=rebin, bkgCfg=bkgCfg, singleMean=True, orderSigmas=True)
        sys.exit()

    ###### STEP 4: parameterize norm1
    outDir_param_v1 = "%s/param_v1" % baseDir
    if False:
        jsOut = {}
        jsOut['nGauss'] = nGauss
        functions.prepareDir(outDir_param_v1, True)
        with open("%s/results.json" % outDir_fits_v1) as f: jsIn = json.load(f)
        
        # means
        fitF, params, cParams = "(x<30)*1*([0] + [1]*x + [2]*x*x + [3]*x*x*x) +0+ (x>30)*1*(  ([0] + [1]*30 + [2]*30*30 + [3]*30*30*30 - [4]*30) + [4]*x   )", [2.45501e-01, 3.67274e-01, -7.26948e-03, 5.62285e-05, 8.95248e-02], [False, False, False, False, False]
        doFit(jsIn, jsOut, comp, "mean1", fitF, params, outDir_param_v1, label, cParams=cParams, fitMin=0, fitMax=140, yMin=0, yMax=50, yTitle = "#mu_{1} (GeV)", doFit=True)
        doFit(jsIn, jsOut, comp, "mean2", fitF, params, outDir_param_v1, label, cParams=cParams, fitMin=0, fitMax=140, yMin=0, yMax=50, yTitle = "#mu_{2} (GeV)", doFit=True)
       
        # sigma1
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [2.44857e-01, 5.19971e-01, 4.91301e+00], [False, False, False]
        doFit(jsIn, jsOut, comp, "sigma1", fitF, params, outDir_param_v1, label, cParams=cParams, fitMin=0, fitMax=80, yMin=0, yMax=30, yTitle = "#sigma_{1} (GeV)", doFit=True)
        
        # sigma2
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [2.44857e-01, 5.19971e-01, 11], [False, False, False]
        doFit(jsIn, jsOut, comp, "sigma2", fitF, params, outDir_param_v1, label, cParams=cParams, fitMin=0, fitMax=80, yMin=0, yMax=30, yTitle = "#sigma_{2} (GeV)", doFit=True)
        
        # norm
        fitF, params, cParams = "[0]*TMath::Exp(-[1]*x)", [0.72, 0.015], [True, True] # ad-hoc exp
        doFit(jsIn, jsOut, comp, "norm1", fitF, params, outDir_param_v1, label, cParams=cParams, fitMin=0, fitMax=100, yMin=0, yMax=1, yTitle = "n_{1} (GeV)", doFit=False)

        with open("%s/results.json" % outDir_param_v1, "w") as outfile: json.dump(jsOut, outfile, indent=4)
        sys.exit()
    
    ###### STEP 5: refit with fixed means and fixed norm1
    outDir_fits_v2 = "%s/fits_v2" % baseDir
    if False:
        fitCfg['mean_cfg'] = [2, 2] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
        fitCfg['sigma_cfg'] = [0, 0]
        fitCfg['norm_cfg'] = [2]

        with open("%s/results.json" % outDir_param_v1) as f: jsIn = json.load(f)
        doFitMultiGauss(proc, comp, fitCfg, label, outDir_fits_v2, funcJs=jsIn, qTmax=150, rebin=rebin, bkgCfg=bkgCfg, singleMean=True, orderSigmas=False)
        sys.exit()
    
    ###### STEP 5: parameterize sigma2
    outDir_param_v2 = "%s/param_v2" % baseDir
    if False:
        jsOut = {}
        jsOut['nGauss'] = nGauss
        functions.prepareDir(outDir_param_v2, True)
        with open("%s/results.json" % outDir_fits_v2) as f: jsIn = json.load(f)

        # means
        fitF, params, cParams = "(x<30)*1*([0] + [1]*x + [2]*x*x + [3]*x*x*x) +0+ (x>30)*1*(    ([0] + [1]*30 + [2]*30*30 + [3]*30*30*30 - ([1] + 2*[2]*30 + 3*[3]*30*30)*30) + ([1] + 2*[2]*30 + 3*[3]*30*30)*x )", [1.58212e-01, 4.03838e-01, -1.03807e-02, 1.21562e-04], [False, False, False, False]
        doFit(jsIn, jsOut, comp, "mean1", fitF, params, outDir_param_v2, label, cParams=cParams, fitMin=0, fitMax=120, yMin=0, yMax=50, yTitle = "#mu_{1} (GeV)", doFit=True)
        doFit(jsIn, jsOut, comp, "mean2", fitF, params, outDir_param_v2, label, cParams=cParams, fitMin=0, fitMax=120, yMin=0, yMax=50, yTitle = "#mu_{2} (GeV)", doFit=True)
       
        # sigma1
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [3.95805e-01, 4.45398e-01, 5.94177e+00], [False, False, False]
        doFit(jsIn, jsOut, comp, "sigma1", fitF, params, outDir_param_v2, label, cParams=cParams, fitMin=0, fitMax=80, yMin=0, yMax=30, yTitle = "#sigma_{1} (GeV)")
        
        # sigma2
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [3.06666e-02, 1.07437e+00, 1.11954e+01], [False, False, False]
        doFit(jsIn, jsOut, comp, "sigma2", fitF, params, outDir_param_v2, label, cParams=cParams, fitMin=0, fitMax=140, yMin=0, yMax=30, yTitle = "#sigma_{2} (GeV)")
        
        # norm
        fitF, params, cParams = "[0]*TMath::Exp(-[1]*x)", [0.72, 0.015], [True, True] # ad-hoc exp
        doFit(jsIn, jsOut, comp, "norm1", fitF, params, outDir_param_v2, label, cParams=cParams, fitMin=0, fitMax=100, yMin=0, yMax=1, yTitle = "n_{1} (GeV)", doFit=True)

        with open("%s/results.json" % outDir_param_v2, "w") as outfile: json.dump(jsOut, outfile, indent=4)
        sys.exit()



    ###### STEP 5: refit with fixed means, sigma2 and norm1
    outDir_fits_v3 = "%s/fits_v3" % baseDir
    if False:
        fitCfg['mean_cfg'] = [2, 2] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
        fitCfg['sigma_cfg'] = [0, 2]
        fitCfg['norm_cfg'] = [2]

        with open("%s/results.json" % outDir_param_v2) as f: jsIn = json.load(f)
        doFitMultiGauss(proc, comp, fitCfg, label, outDir_fits_v3, funcJs=jsIn, qTmax=150, rebin=rebin, bkgCfg=bkgCfg, singleMean=True)
        sys.exit()
        
    ###### STEP 5: parameterize sigma2 and sigma3
    outDir_param_v3 = "%s/param_v3" % baseDir
    if True:
        jsOut = {}
        jsOut['nGauss'] = nGauss
        functions.prepareDir(outDir_param_v3, True)
        with open("%s/results.json" % outDir_fits_v3) as f: jsIn = json.load(f)

        # means
        fitF, params, cParams = "(x<30)*1*([0] + [1]*x + [2]*x*x + [3]*x*x*x) +0+ (x>30)*1*(    ([0] + [1]*30 + [2]*30*30 + [3]*30*30*30 - ([1] + 2*[2]*30 + 3*[3]*30*30)*30) + ([1] + 2*[2]*30 + 3*[3]*30*30)*x )", [3.08663e-01, 3.48668e-01, -6.06558e-03, 3.62872e-05], [False, False, True, True]
        doFit(jsIn, jsOut, comp, "mean1", fitF, params, outDir_param_v3, label, cParams=cParams, fitMin=0, fitMax=120, yMin=0, yMax=50, yTitle = "#mu_{1} (GeV)", doFit=True)
        doFit(jsIn, jsOut, comp, "mean2", fitF, params, outDir_param_v3, label, cParams=cParams, fitMin=0, fitMax=120, yMin=0, yMax=50, yTitle = "#mu_{2} (GeV)", doFit=True)
       
        # sigma1
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [3.15325e-01, 5.07352e-01, 6.04702e+00], [False, False, False]
        doFit(jsIn, jsOut, comp, "sigma1", fitF, params, outDir_param_v3, label, cParams=cParams, fitMin=0, fitMax=70, yMin=0, yMax=30, yTitle = "#sigma_{1} (GeV)", doFit=True)
        
        # sigma2
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [3.04763e-02, 1.07898e+00, 1.11939e+01], [False, False, False]
        doFit(jsIn, jsOut, comp, "sigma2", fitF, params, outDir_param_v3, label, cParams=cParams, fitMin=0, fitMax=140, yMin=0, yMax=30, yTitle = "#sigma_{2} (GeV)")
        
        # norm
        fitF, params, cParams = "[0]*TMath::Exp(-[1]*x)", [0.72, 0.015], [True, True] # ad-hoc exp
        doFit(jsIn, jsOut, comp, "norm1", fitF, params, outDir_param_v3, label, cParams=cParams, fitMin=0, fitMax=120, yMin=0, yMax=1, yTitle = "n_{1} (GeV)", doFit=False)

        with open("%s/results.json" % outDir_param_v3, "w") as outfile: json.dump(jsOut, outfile, indent=4)
        #sys.exit()
    
    ###### STEP 5: refit
    if True:
        with open("%s/results.json" % outDir_param_v3) as f: jsIn = json.load(f)
        jsOut = combinedFit(proc, comp, jsIn, rebin=rebin, bkgCfg=bkgCfg, singleMean=True, qTmax=70) # 95 was good
        with open("%s/results_refit.json" % outDir_param_v3, "w") as outfile: json.dump(jsOut, outfile, indent=4)
        #sys.exit()
   
    ###### STEP 6: final validation
    if True:
        fitCfg['mean_cfg'] = [2, 2] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
        fitCfg['sigma_cfg'] = [2, 2]
        fitCfg['norm_cfg'] = [2]

        outDir_refit = "%s/refit_v3" % baseDir
        with open("%s/results_refit.json" % outDir_param_v3) as f: jsIn = json.load(f)
        doFitMultiGauss(proc, comp, fitCfg, label, outDir_refit, funcJs=jsIn, qTmax=150, rebin=rebin, bkgCfg=bkgCfg, singleMean=True)
 
        outDir_refit = "%s/param_v3_refit" % baseDir
        functions.prepareDir(outDir_refit)
        doPlot(comp, "norm1", jsIn, outDir_refit, label, yMin=0, yMax=1, yTitle = "n_{1} (GeV)")
        doPlot(comp, "sigma1", jsIn, outDir_refit, label, yMin=0, yMax=30, yTitle = "#sigma_{1} (GeV)")
        doPlot(comp, "sigma2", jsIn, outDir_refit, label, yMin=0, yMax=30, yTitle = "#sigma_{2} (GeV)")
        doPlot(comp, "mean1", jsIn, outDir_refit, label, yMin=0, yMax=20, yTitle = "#mu_{1} (GeV)")
        doPlot(comp, "mean2", jsIn, outDir_refit, label, yMin=0, yMax=20, yTitle = "#mu_{2} (GeV)")

def readProc(groups, hName, procs):

    label = "%s_%s" % (hName, procs[0]) 
    groups.setHists(hName, "", label=label, procsToRead=procs)
    bhist = groups.groups[procs[0]][label]
    return bhist 

def prepareFile(fInName, fOutName):

    def readProc(groups_mumu, hName, procName):

        if isinstance(procName, str):
            label = "%s_%s" % (hName, procName) 
            groups_mumu.setHists(hName, "", label=label, procsToRead=[procName], selectSignal=False)
            bhist = groups_mumu.groups[procName][label]
        else:
            label = "%s_%s" % (hName, procName[0]) 
            groups_mumu.setHists(hName, "", label=label, procsToRead=procName, selectSignal=False)
            bhist = groups_mumu.groups[procName[0]][label]
        return bhist 


    datagroups = Datagroups(fInName)
    procs = ["ZZ", "EWK_noZZ", "Top"]
    if flavor == "mumu": procs += ["SingleMuon", "Zmumu"]
    if flavor == "ee": procs += ["SingleElectron", "DYee"]
    
    fOut = ROOT.TFile(fOutName, "RECREATE")

    for proc in procs:
        print(proc)
        bhist_para_qT = readProc(datagroups, "recoil_corr_xy_para_qT_qTbinned", proc) # recoil_corr_xy_para_qT_qTbinned recoil_uncorr_para_qT_qTbinned
        bhist_perp = readProc(datagroups, "recoil_corr_xy_perp_qTbinned", proc) # recoil_corr_xy_perp_qTbinned recoil_uncorr_perp_qTbinned
        rhist_para_qT = narf.hist_to_root(bhist_para_qT)
        rhist_perp = narf.hist_to_root(bhist_perp)
        
        rhist_para_qT.SetName("%s_para" % (proc))
        rhist_perp.SetName("%s_perp" % (proc))
        
        #print(rhist_perp.Integral(), rhist_perp.GetNbinsY())
        
        #rhist_perp = rhist_perp.RebinY(2)
        #rhist_perp.Scale(1, "width")
        #print(rhist_perp.Integral(), rhist_perp.GetNbinsY())
        #quit()
        rhist_para_qT.Write()
        rhist_perp.Write()
        
        '''
        for iBin in range(1, rhist_para_qT.GetNbinsX()+2):

            hist_para = rhist_para_qT.ProjectionY("%s_para_bin%d" % (proc, iBin), iBin, iBin)
            hist_para.Write()
            print(iBin, hist_para.Integral())
            
            hist_perp = rhist_perp.ProjectionY("%s_perp_bin%d" % (proc, iBin), iBin, iBin)
            hist_perp.Write()
        '''
        
        
    '''
    # GEN level qT 
    if flavor == "mumu": proc = "DYmumu"
    if flavor == "ee": proc = "DYee"
    if flavor == "mu": proc = ["WplusJetsToMuNu", "WminusJetsToMuNu"] # merge plus and minus
    if flavor == "e": proc = ["WplusJetsToENu", "WminusJetsToENu"] # merge plus and minus
    bhist_para_qT = readProc(datagroups, "recoil_corr_xy_para_qT_qTbinned_gen", proc)
    bhist_perp = readProc(datagroups, "recoil_corr_xy_perp_qTbinned_gen", proc)
    rhist_para_qT = narf.hist_to_root(bhist_para_qT)
    rhist_perp = narf.hist_to_root(bhist_perp)
    for iBin in range(1, rhist_para_qT.GetNbinsX()+1):
            
        hist_para = rhist_para_qT.ProjectionY("%s_gen_para_bin%d" % (sig, iBin), iBin, iBin)
        print(hist_para)
        hist_para.Write()
            
        hist_perp = rhist_perp.ProjectionY("%s_gen_perp_bin%d" % (sig, iBin), iBin, iBin)
        hist_perp.Write()
    '''    
    fOut.ls()
    fOut.Close()
 

def parameterize_mean(procs, comp, out):

    proc = "Zmumu"
    tag = "zmumu_para"
    comp = "para"
    baseDir = "%s/%s" % (outDir, tag)
    name = "DY #rightarrow #mu^{+}#mu^{#minus}"
    label = "%s, %s" % (name, met)
    functions.prepareDir(baseDir, False)
    rebin = 1
    
    
    
    bhist = readProc(groups, "recoil_corr_xy_para_qT_corr_qTbinned", [proc])
    outDir_mean = "%s/parameterize_mean" % baseDir
    functions.prepareDir(outDir_mean, False)
    
    fitCfg = {}
    fitCfg['func_name'] = "pw_power_lin"
    fitCfg['parms_init'] = [5, 1, 1, 1]
    fitCfg['parms_cte'] = [False, False, False, False]
    
    fitCfg = {}
    fitCfg['func_name'] = "pw_poly4_poly1"
    fitCfg['parms_init'] = [2.34060e+01, -1.42488e-01, 5.35185e-01, -3.64996e-02, 1.36259e-03, -1.97291e-05]
    fitCfg['parms_cte'] = [False, False, False, False, False, False]
    
    
    fitCfg = {}
    fitCfg['func_name'] = "pw_poly4_power"
    fitCfg['parms_init'] = [1.89747e+01, -1.23126e-01, 5.16425e-01, -3.18374e-02, 9.51272e-04, -8.04189e-06, 9.51170e-01]
    fitCfg['parms_cte'] = [False, False, False, False, False, False, False]
    
    #fitCfg = {}
    #fitCfg['func_name'] = "pw_poly5_power"
    #fitCfg['parms_init'] = [15, 0, 5.23415e-01, -3.22912e-02, 8.41808e-04, 8.40664e-07, 0, 8.00323e-01]
    #fitCfg['parms_cte'] = [False, False, False, False, False, False, False, False]
    
    
    rls.parameterize_mean(bhist, outDir_mean, fitCfg, binning_qT, label, yMin=-2, yMax=12)


    
def testYields():

    import hist
    s = hist.tag.Slicer()

    def readProc(groups_mumu, hName, procName):

        label = "%s_%s" % (hName, procName[0]) 
        groups_mumu.setHists(hName, "", label=label, procsToRead=[procName], selectSignal=False)
        bhist = groups_mumu.groups[procName][label]
        return bhist 


    bhist = readProc(groups, "recoil_corr_xy_perp_qTbinned", "Zmumu")
    h =  bhist[{"qTbinned": s[complex(0,0):complex(0,0.5)]}]
    h =  h[{"qTbinned": s[0:hist.overflow:hist.sum]}] # remove the overflow in the sum
    h_root = narf.hist_to_root(h)
    print(h_root.Integral())
    
    #    rhist_para_qT = narf.hist_to_root(bhist_para_qT)
    #    rhist_perp = narf.hist_to_root(bhist_perp)
        
    h_root_2d = narf.hist_to_root(bhist)
    
    
    h_root_ = h_root_2d.ProjectionY("dd", 1, 1)
    print(h_root_.Integral())
    print(h_root_2d.Integral())
    print(bhist)
    
    
    bhist = readProc(groups, "recoil_corr_xy_perp", "Zmumu")
    f = narf.hist_to_root(bhist)
    print(f.Integral())
    
    
if __name__ == "__main__":

    met = "RawPFMET" # DeepMETReso RawPFMET
    flavor = "mumu" # mu, e, mumu, ee
    
    print("Start")
    met = "RawPFMET" # RawPFMET DeepMETReso
    flavor = "mumu" # mu, e, mumu, ee
    

    outDir = f"/eos/user/j/jaeyserm/www/wmass/lowPU/Z{flavor}_{met}/plots/"
    outCfgDir = f"wremnants/data/recoil/lowPU/{flavor}_{met}/"
    #functions.prepareDir(outDir, remove=True)

    groups = make_datagroups_lowPU("lowPU_%s_RawPFMET_%s_nnpdf31.pkl.lz4" % (flavor, met))


    outDir = "/eos/user/j/jaeyserm/www/wmass/lowPU/recoilCorrection/recoil_%s_%s/" % (flavor, met)
    functions.prepareDir(outDir, False)
    
    binning_qT = list(functions.drange(0, 100, 0.5)) + [100]


    if met == "RawPFMET":

        #zmumu_para_qT_RawPFMET()
        zmumu_perp_RawPFMET()
        
        #ttbar_para_qT_RawPFMET()
        #ttbar_perp_RawPFMET()
        
        #ewk_para_qT_RawPFMET()
        #ewk_perp_RawPFMET()
        
        #zz_para_qT_RawPFMET()
        #zz_perp_RawPFMET()

        #singlemuon_para_qT_RawPFMET()
        #singlemuon_perp_RawPFMET()
        pass

    if met == "DeepMETReso":
        
        do_dymumu_para_DeepMETReso()
        #do_dymumu_perp_DeepMETReso()
        
        #do_ttbar_para_DeepMETReso()
        #do_ttbar_perp_DeepMETReso()
        
        #do_ewk_para_DeepMETReso()
        #do_ewk_perp_DeepMETReso()
        
        #do_zz_para_DeepMETReso()
        #do_zz_perp_DeepMETReso()
        
        #do_singlemuon_para_DeepMETReso()
        #do_singlemuon_perp_DeepMETReso()