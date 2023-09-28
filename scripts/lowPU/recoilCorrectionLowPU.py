
import sys,array,math,os
import numpy as np
import ctypes
import json

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptTitle(0)

import functions
import plotter
import recoilLibs

from wremnants.datasets.datagroups import Datagroups

import lz4.frame
import narf


def do_zz_perp_RawPFMET():

    proc = "ZZ"
    tag = "zz_perp"
    comp = "perp"
    baseDir = "%s/%s" % (outDir, tag)
    name = "ZZ #rightarrow #mu^{+}#mu^{#minus}"
    label = "%s, %s" % (name, met)
    functions.prepareDir(baseDir, False)
    rebin = 1

    outDir_fits = "%s/fits_v0" % baseDir
    fitCfg = {}
    fitCfg['mean'], fitCfg['mean_cfg'] = [0, 0, 0], [0, -1, -1]
    fitCfg['sigma'], fitCfg['sigma_cfg'] = [10, 25, 40], [0, 0, 0]
    fitCfg['norm'], fitCfg['norm_cfg'] = [0.3, 0.3], [0, 0]
    
    recoilLibs.doFitMultiGauss(rFile, proc, comp, fitCfg, label, outDir_fits, recoil_qTbins, qTmax=qTmax, rebin=rebin, xMin=-200, xMax=200, ratio=True)
  
def do_zz_para_RawPFMET():

    proc = "ZZ"
    tag = "zz_para"
    comp = "para"
    baseDir = "%s/%s" % (outDir, tag)
    name = "ZZ #rightarrow #mu^{+}#mu^{#minus}"
    label = "%s, %s" % (name, met)
    functions.prepareDir(baseDir, False)
    rebin = 1

    outDir_fits = "%s/fits_v0" % baseDir
    fitCfg = {}
    fitCfg['mean'], fitCfg['mean_cfg'] = [0, 0, 0], [0, 0, 0]
    fitCfg['sigma'], fitCfg['sigma_cfg'] = [10, 25, 40], [0, 0, 0]
    fitCfg['norm'], fitCfg['norm_cfg'] = [0.3, 0.3], [0, 0]
    
    recoilLibs.doFitMultiGauss(rFile, proc, comp, fitCfg, label, outDir_fits, recoil_qTbins, qTmax=qTmax, rebin=rebin, xMin=-200, xMax=200, ratio=True)

def do_ewk_perp_RawPFMET():

    proc = "EWK_noZZ"
    tag = "ewk_perp"
    comp = "perp"
    baseDir = "%s/%s" % (outDir, tag)
    name = "EWK #rightarrow #mu^{+}#mu^{#minus}"
    label = "%s, %s" % (name, met)
    functions.prepareDir(baseDir, False)
    rebin = 1

    outDir_fits = "%s/fits_v0" % baseDir
    fitCfg = {}
    fitCfg['mean'], fitCfg['mean_cfg'] = [0, 0, 0], [0, -1, -1]
    fitCfg['sigma'], fitCfg['sigma_cfg'] = [10, 30, 50], [0, 0, 0]
    fitCfg['norm'], fitCfg['norm_cfg'] = [0.5, 0.4], [0, 0]
    nGauss = len(fitCfg['mean'])
    
    recoilLibs.doFitMultiGauss(rFile, proc, comp, fitCfg, label, outDir_fits, recoil_qTbins, qTmax=qTmax, rebin=rebin, xMin=-200, xMax=200, ratio=True)

def do_ewk_para_RawPFMET():

    proc = "EWK_noZZ"
    tag = "ewk_para"
    comp = "para"
    baseDir = "%s/%s" % (outDir, tag)
    name = "EWK #rightarrow #mu^{+}#mu^{#minus}"
    label = "%s, %s" % (name, met)
    functions.prepareDir(baseDir, False)
    rebin = 1

    outDir_fits = "%s/fits_v0" % baseDir
    fitCfg = {}
    fitCfg['mean'], fitCfg['mean_cfg'] = [0, 0, 0], [0, 0, 0]
    fitCfg['sigma'], fitCfg['sigma_cfg'] = [10, 30, 50], [0, 0, 0]
    fitCfg['norm'], fitCfg['norm_cfg'] = [0.5, 0.4], [0, 0]

    recoilLibs.doFitMultiGauss(rFile, proc, comp, fitCfg, label, outDir_fits, recoil_qTbins, qTmax=qTmax, rebin=rebin, xMin=-200, xMax=200, ratio=True)
    
def do_ttbar_perp_RawPFMET():

    proc = "TTbar"
    tag = "ttbar_perp"
    comp = "perp"
    baseDir = "%s/%s" % (outDir, tag)
    name = "TTbar #rightarrow #mu^{+}#mu^{#minus}"
    label = "%s, %s" % (name, met)
    functions.prepareDir(baseDir, False)
    rebin = 1

    outDir_fits = "%s/fits_v0" % baseDir
    fitCfg = {}
    fitCfg['mean'], fitCfg['mean_cfg'] = [0, 0, 0], [0, -1, -1]
    fitCfg['sigma'], fitCfg['sigma_cfg'] = [25, 35, 50], [0, 0, 0]
    fitCfg['norm'], fitCfg['norm_cfg'] = [0.3, 0.3], [0, 0]
    
    recoilLibs.doFitMultiGauss(rFile, proc, comp, fitCfg, label, outDir_fits, recoil_qTbins, qTmax=qTmax, rebin=rebin, xMin=-200, xMax=200, ratio=True)

def do_ttbar_para_RawPFMET():

    proc = "TTbar"
    tag = "ttbar_para"
    comp = "para"
    baseDir = "%s/%s" % (outDir, tag)
    name = "TTbar #rightarrow #mu^{+}#mu^{#minus}"
    label = "%s, %s" % (name, met)
    functions.prepareDir(baseDir, False)
    rebin = 1

    outDir_fits = "%s/fits_v0" % baseDir
    fitCfg = {}
    
    fitCfg['mean'], fitCfg['mean_cfg'] = [0, 0, 0], [0, 0, 0]
    fitCfg['sigma'], fitCfg['sigma_cfg'] = [25, 35, 50], [0, 0, 0]
    fitCfg['norm'], fitCfg['norm_cfg'] = [0.3, 0.3], [0, 0]

    recoilLibs.doFitMultiGauss(rFile, proc, comp, fitCfg, label, outDir_fits, recoil_qTbins, qTmax=qTmax, rebin=rebin, xMin=-200, xMax=200, ratio=True)

def do_dymumu_perp_RawPFMET_zlib():

    proc = "Zmumu"
    tag = "zmumu_perp"
    comp = "perp"
    baseDir = "%s/%s_zlib" % (outDir, tag)
    name = "DY #rightarrow #mu^{+}#mu^{#minus}"
    label = "%s, %s" % (name, met)
    functions.prepareDir(baseDir, False)
    rebin = 1

    outDir_fits_v0 = "%s/fits_v0" % baseDir
    fitCfg = {}   
    
    fitCfg['mean'], fitCfg['mean_cfg'] = [0, 0, 0, 0], [0, 0, 0, 0] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
    fitCfg['sigma'], fitCfg['sigma_cfg'] = [15, 5, 8, 10], [0, 0, 0, 0]
    fitCfg['norm'], fitCfg['norm_cfg'] = [0.05, 0.20, 0.25], [0, 0, 0]
    

    recoil_qTbins = list(functions.drange(0, 50, 0.5)) +  list(range(50, 100, 1)) + list(range(100, 150, 2)) + [150]
    
    nGauss = len(fitCfg['mean'])
    
    if False: 
        recoilLibs.doFitMultiGauss(rFile, proc, comp, fitCfg, label, outDir_fits_v0, recoil_qTbins, qTmax=150, propagate=False, rebin=rebin, singleMean=True, xMin=-100, xMax=100)
        return
        
    outDir_param_v0 = "%s/param_v0" % baseDir
    if False:
        jsOut = {}
        jsOut['nGauss'] = nGauss
        outDir_param = outDir_param_v0
        functions.prepareDir(outDir_param, True)
        with open("%s/results.json" % outDir_fits_v0) as f: jsIn = json.load(f)
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [5.20382e+00, 1.78386e-01, 7.51449e+00], [False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma1", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{1} (GeV)")
 
        t = 25
        fLeft, dfLeft  = "[0] + [1]*x + [2]*x*x + [3]*x*x*x + [4]*x*x*x*x", "[1] + 2*[2]*x + 3*[3]*x*x + 4*[4]*x*x*x"
        fLeftEval, dfLeftEval = fLeft.replace("x", str(t)), dfLeft.replace("x", str(t))
        fitF = "(x<{t})*1*({fLeft}) +0+ (x>{t})*1*(  ({dfLeftEval})*x + ({fLeftEval}-({dfLeftEval})*{t}) )".format(t=t, fLeft=fLeft, fLeftEval=fLeftEval, dfLeftEval=dfLeftEval)
        params, cParams = [4.00314e+00, 6.34998e-02, -3.94494e-04, 2.83885e-05, -1.51997e-06], [False, False, False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma2", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=1, fitMax=150, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{2} (GeV)")
        
        fitF, params, cParams = "[0] + [1]*x", [7.45735e+00, 6.52883e-03], [False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma3", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{2} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [1.15048e-02, 9.60879e-01, 2.26090e+01], [False, False, False]
        t = 25
        fLeft, dfLeft  = "[0] + [1]*x + [2]*x*x + [3]*x*x*x + [4]*x*x*x*x", "[1] + 2*[2]*x + 3*[3]*x*x + 4*[4]*x*x*x"
        fLeftEval, dfLeftEval = fLeft.replace("x", str(t)), dfLeft.replace("x", str(t))
        fitF = "(x<{t})*1*({fLeft}) +0+ (x>{t})*1*(  ({dfLeftEval})*x + ({fLeftEval}-({dfLeftEval})*{t}) )".format(t=t, fLeft=fLeft, fLeftEval=fLeftEval, dfLeftEval=dfLeftEval)
        params, cParams = [9.99069e+00, -7.11533e-02, 1.92415e-02, -1.05192e-03, 1.72908e-05], [False, False, False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma4", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{4} (GeV)")
        

        t = 25
        fLeft, dfLeft  = "[0] + [1]*x + [2]*x*x + [3]*x*x*x + [4]*x*x*x*x", "[1] + 2*[2]*x + 3*[3]*x*x + 4*[4]*x*x*x"
        fLeftEval, dfLeftEval = fLeft.replace("x", str(t)), dfLeft.replace("x", str(t))
        fitF = "(x<{t})*1*({fLeft}) +0+ (x>{t})*1*(  ({dfLeftEval})*x + ({fLeftEval}-({dfLeftEval})*{t}) )".format(t=t, fLeft=fLeft, fLeftEval=fLeftEval, dfLeftEval=dfLeftEval)
        params, cParams = [4.95396e-02, -2.88434e-03, 5.76138e-05, 2.54185e-06, -6.16914e-08], [True, True, True, True, True]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm1", fitF, params, outDir_param, recoil_qTbins, label, cutOffMin=0.02, cParams=cParams, fitMin=2, fitMax=60  , yMin=0, yMax=1, xMax=150, yTitle = "n_{3} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Exp(-[1]*x)", [2.33265e-01, 4.39271e-03], [True, True]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm2", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=1, xMax=150, yTitle = "n_{3} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Exp(-[1]*x)", [5.15149e-01, 1.71424e-02], [True, True]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm3", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=30, yMin=0, yMax=1, xMax=150, yTitle = "n_{3} (GeV)")
        
        with open("%s/results.json" % outDir_param_v0, "w") as outfile: json.dump(jsOut, outfile, indent=4)
        return

    outDir_fits_v1 = "%s/fits_v1" % baseDir
    if False:
        fitCfg['mean_cfg'] = [1, 1, 1, 1] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
        fitCfg['sigma_cfg'] = [0, 0, 0, 0]
        fitCfg['norm_cfg'] = [0, 2, 0]
        fitCfg['sigma'] = [12.6, 3.8, 7.9, 13]
        with open("%s/results.json" % outDir_param_v0) as f: jsIn = json.load(f)
        recoilLibs.doFitMultiGauss(rFile, proc, comp, fitCfg, label, outDir_fits_v1, recoil_qTbins, funcJs=jsIn, qTmax=150, rebin=rebin, singleMean=True, xMin=-100, xMax=100)
        return

    outDir_param_v1 = "%s/param_v1" % baseDir
    if False:
        jsOut = {}
        jsOut['nGauss'] = nGauss
        outDir_param = outDir_param_v1
        functions.prepareDir(outDir_param, True)
        with open("%s/results.json" % outDir_fits_v1) as f: jsIn = json.load(f)
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [5.20382e+00, 1.78386e-01, 7.51449e+00], [False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma1", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{1} (GeV)")
 
        t = 25
        fLeft, dfLeft  = "[0] + [1]*x + [2]*x*x + [3]*x*x*x + [4]*x*x*x*x", "[1] + 2*[2]*x + 3*[3]*x*x + 4*[4]*x*x*x"
        fLeftEval, dfLeftEval = fLeft.replace("x", str(t)), dfLeft.replace("x", str(t))
        fitF = "(x<{t})*1*({fLeft}) +0+ (x>{t})*1*(  ({dfLeftEval})*x + ({fLeftEval}-({dfLeftEval})*{t}) )".format(t=t, fLeft=fLeft, fLeftEval=fLeftEval, dfLeftEval=dfLeftEval)
        params, cParams = [4.00314e+00, 6.34998e-02, -3.94494e-04, 2.83885e-05, -1.51997e-06], [False, False, False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma2", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=1, fitMax=150, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{2} (GeV)")
        
        fitF, params, cParams = "[0] + [1]*x", [7.45735e+00, 6.52883e-03], [False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma3", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{2} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [1.15048e-02, 9.60879e-01, 2.26090e+01], [False, False, False]
        t = 25
        fLeft, dfLeft  = "[0] + [1]*x + [2]*x*x + [3]*x*x*x + [4]*x*x*x*x", "[1] + 2*[2]*x + 3*[3]*x*x + 4*[4]*x*x*x"
        fLeftEval, dfLeftEval = fLeft.replace("x", str(t)), dfLeft.replace("x", str(t))
        fitF = "(x<{t})*1*({fLeft}) +0+ (x>{t})*1*(  ({dfLeftEval})*x + ({fLeftEval}-({dfLeftEval})*{t}) )".format(t=t, fLeft=fLeft, fLeftEval=fLeftEval, dfLeftEval=dfLeftEval)
        params, cParams = [9.99069e+00, -7.11533e-02, 1.92415e-02, -1.05192e-03, 1.72908e-05], [False, False, False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma4", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{4} (GeV)")
        

        t = 25
        fLeft, dfLeft  = "[0] + [1]*x + [2]*x*x + [3]*x*x*x + [4]*x*x*x*x", "[1] + 2*[2]*x + 3*[3]*x*x + 4*[4]*x*x*x"
        fLeftEval, dfLeftEval = fLeft.replace("x", str(t)), dfLeft.replace("x", str(t))
        fitF = "(x<{t})*1*({fLeft}) +0+ (x>{t})*1*(  ({dfLeftEval})*x + ({fLeftEval}-({dfLeftEval})*{t}) )".format(t=t, fLeft=fLeft, fLeftEval=fLeftEval, dfLeftEval=dfLeftEval)
        params, cParams = [4.95396e-02, -2.88434e-03, 5.76138e-05, 2.54185e-06, -6.16914e-08], [True, True, True, True, True]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm1", fitF, params, outDir_param, recoil_qTbins, label, cutOffMin=0.02, cParams=cParams, fitMin=2, fitMax=60  , yMin=0, yMax=1, xMax=150, yTitle = "n_{3} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Exp(-[1]*x)", [2.33265e-01, 4.39271e-03], [True, True]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm2", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=1, xMax=150, yTitle = "n_{3} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Exp(-[1]*x)", [5.15149e-01, 1.71424e-02], [True, True]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm3", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=30, yMin=0, yMax=1, xMax=150, yTitle = "n_{3} (GeV)")
        
        with open("%s/results.json" % outDir_param_v1, "w") as outfile: json.dump(jsOut, outfile, indent=4)    
        return

    outDir_fits_v2 = "%s/fits_v2" % baseDir
    if False:
        fitCfg['mean_cfg'] = [1, 1, 1, 1] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
        fitCfg['sigma_cfg'] = [0, 0, 0, 0]
        fitCfg['norm_cfg'] = [0, 2, 2]
        fitCfg['sigma'] = [13, 4, 7, 10]
        with open("%s/results.json" % outDir_param_v1) as f: jsIn = json.load(f)
        recoilLibs.doFitMultiGauss(rFile, proc, comp, fitCfg, label, outDir_fits_v2, recoil_qTbins, funcJs=jsIn, qTmax=150, rebin=rebin, singleMean=True, xMin=-100, xMax=100)
        return
        
    outDir_param_v2 = "%s/param_v2" % baseDir
    if False:
        jsOut = {}
        jsOut['nGauss'] = nGauss
        outDir_param = outDir_param_v2
        functions.prepareDir(outDir_param, True)
        with open("%s/results.json" % outDir_fits_v2) as f: jsIn = json.load(f)
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [5.20382e+00, 1.78386e-01, 7.51449e+00], [False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma1", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{1} (GeV)")
 
        t = 25
        fLeft, dfLeft  = "[0] + [1]*x + [2]*x*x + [3]*x*x*x + [4]*x*x*x*x", "[1] + 2*[2]*x + 3*[3]*x*x + 4*[4]*x*x*x"
        fLeftEval, dfLeftEval = fLeft.replace("x", str(t)), dfLeft.replace("x", str(t))
        fitF = "(x<{t})*1*({fLeft}) +0+ (x>{t})*1*(  ({dfLeftEval})*x + ({fLeftEval}-({dfLeftEval})*{t}) )".format(t=t, fLeft=fLeft, fLeftEval=fLeftEval, dfLeftEval=dfLeftEval)
        params, cParams = [4.00314e+00, 6.34998e-02, -3.94494e-04, 2.83885e-05, -1.51997e-06], [False, False, False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma2", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=1, fitMax=150, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{2} (GeV)")
        
        fitF, params, cParams = "[0] + [1]*x", [7.45735e+00, 6.52883e-03], [False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma3", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{2} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [1.15048e-02, 9.60879e-01, 2.26090e+01], [False, False, False]
        t = 25
        fLeft, dfLeft  = "[0] + [1]*x + [2]*x*x + [3]*x*x*x + [4]*x*x*x*x", "[1] + 2*[2]*x + 3*[3]*x*x + 4*[4]*x*x*x"
        fLeftEval, dfLeftEval = fLeft.replace("x", str(t)), dfLeft.replace("x", str(t))
        fitF = "(x<{t})*1*({fLeft}) +0+ (x>{t})*1*(  ({dfLeftEval})*x + ({fLeftEval}-({dfLeftEval})*{t}) )".format(t=t, fLeft=fLeft, fLeftEval=fLeftEval, dfLeftEval=dfLeftEval)
        params, cParams = [9.99069e+00, -7.11533e-02, 1.92415e-02, -1.05192e-03, 1.72908e-05], [False, False, False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma4", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{4} (GeV)")
        

        t = 25
        fLeft, dfLeft  = "[0] + [1]*x + [2]*x*x + [3]*x*x*x + [4]*x*x*x*x", "[1] + 2*[2]*x + 3*[3]*x*x + 4*[4]*x*x*x"
        fLeftEval, dfLeftEval = fLeft.replace("x", str(t)), dfLeft.replace("x", str(t))
        fitF = "(x<{t})*1*({fLeft}) +0+ (x>{t})*1*(  ({dfLeftEval})*x + ({fLeftEval}-({dfLeftEval})*{t}) )".format(t=t, fLeft=fLeft, fLeftEval=fLeftEval, dfLeftEval=dfLeftEval)
        params, cParams = [4.95396e-02, -2.88434e-03, 5.76138e-05, 2.54185e-06, -6.16914e-08], [True, True, True, True, True]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm1", fitF, params, outDir_param, recoil_qTbins, label, cutOffMin=0.02, cParams=cParams, fitMin=2, fitMax=60  , yMin=0, yMax=1, xMax=150, yTitle = "n_{3} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Exp(-[1]*x)", [2.33265e-01, 4.39271e-03], [True, True]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm2", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=1, xMax=150, yTitle = "n_{3} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Exp(-[1]*x)", [5.15149e-01, 1.71424e-02], [True, True]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm3", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=30, yMin=0, yMax=1, xMax=150, yTitle = "n_{3} (GeV)")
        
        with open("%s/results.json" % outDir_param_v2, "w") as outfile: json.dump(jsOut, outfile, indent=4)    
        return
        
    outDir_fits_v3 = "%s/fits_v3" % baseDir
    if False:
        fitCfg['mean_cfg'] = [1, 1, 1, 1] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
        fitCfg['sigma_cfg'] = [0, 0, 0, 0]
        fitCfg['norm_cfg'] = [2, 2, 2]
        fitCfg['sigma'] = [13, 4, 7, 10]
        with open("%s/results.json" % outDir_param_v2) as f: jsIn = json.load(f)
        recoilLibs.doFitMultiGauss(rFile, proc, comp, fitCfg, label, outDir_fits_v3, recoil_qTbins, funcJs=jsIn, qTmax=150, rebin=rebin, singleMean=True, xMin=-100, xMax=100)
        return
        
    outDir_param_v3 = "%s/param_v3" % baseDir
    if False:
        jsOut = {}
        jsOut['nGauss'] = nGauss
        outDir_param = outDir_param_v3
        functions.prepareDir(outDir_param, True)
        with open("%s/results.json" % outDir_fits_v3) as f: jsIn = json.load(f)
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [5.20382e+00, 1.78386e-01, 7.51449e+00], [False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma1", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{1} (GeV)")
 
        t = 25
        fLeft, dfLeft  = "[0] + [1]*x + [2]*x*x + [3]*x*x*x + [4]*x*x*x*x", "[1] + 2*[2]*x + 3*[3]*x*x + 4*[4]*x*x*x"
        fLeftEval, dfLeftEval = fLeft.replace("x", str(t)), dfLeft.replace("x", str(t))
        fitF = "(x<{t})*1*({fLeft}) +0+ (x>{t})*1*(  ({dfLeftEval})*x + ({fLeftEval}-({dfLeftEval})*{t}) )".format(t=t, fLeft=fLeft, fLeftEval=fLeftEval, dfLeftEval=dfLeftEval)
        params, cParams = [4.00314e+00, 6.34998e-02, -3.94494e-04, 2.83885e-05, -1.51997e-06], [False, False, False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma2", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=1, fitMax=150, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{2} (GeV)")
        
        fitF, params, cParams = "[0] + [1]*x", [7.45735e+00, 6.52883e-03], [False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma3", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{2} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [1.15048e-02, 9.60879e-01, 2.26090e+01], [False, False, False]
        t = 25
        fLeft, dfLeft  = "[0] + [1]*x + [2]*x*x + [3]*x*x*x + [4]*x*x*x*x", "[1] + 2*[2]*x + 3*[3]*x*x + 4*[4]*x*x*x"
        fLeftEval, dfLeftEval = fLeft.replace("x", str(t)), dfLeft.replace("x", str(t))
        fitF = "(x<{t})*1*({fLeft}) +0+ (x>{t})*1*(  ({dfLeftEval})*x + ({fLeftEval}-({dfLeftEval})*{t}) )".format(t=t, fLeft=fLeft, fLeftEval=fLeftEval, dfLeftEval=dfLeftEval)
        params, cParams = [9.99069e+00, -7.11533e-02, 1.92415e-02, -1.05192e-03, 1.72908e-05], [False, False, False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma4", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{4} (GeV)")
        

        t = 25
        fLeft, dfLeft  = "[0] + [1]*x + [2]*x*x + [3]*x*x*x + [4]*x*x*x*x", "[1] + 2*[2]*x + 3*[3]*x*x + 4*[4]*x*x*x"
        fLeftEval, dfLeftEval = fLeft.replace("x", str(t)), dfLeft.replace("x", str(t))
        fitF = "(x<{t})*1*({fLeft}) +0+ (x>{t})*1*(  ({dfLeftEval})*x + ({fLeftEval}-({dfLeftEval})*{t}) )".format(t=t, fLeft=fLeft, fLeftEval=fLeftEval, dfLeftEval=dfLeftEval)
        params, cParams = [4.95396e-02, -2.88434e-03, 5.76138e-05, 2.54185e-06, -6.16914e-08], [True, True, True, True, True]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm1", fitF, params, outDir_param, recoil_qTbins, label, cutOffMin=0.02, cParams=cParams, fitMin=2, fitMax=60  , yMin=0, yMax=1, xMax=150, yTitle = "n_{3} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Exp(-[1]*x)", [2.33265e-01, 4.39271e-03], [True, True]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm2", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=1, xMax=150, yTitle = "n_{3} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Exp(-[1]*x)", [5.15149e-01, 1.71424e-02], [True, True]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm3", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=30, yMin=0, yMax=1, xMax=150, yTitle = "n_{3} (GeV)")
        
        with open("%s/results.json" % outDir_param_v3, "w") as outfile: json.dump(jsOut, outfile, indent=4)    
        return        

    outDir_fits_v4 = "%s/fits_v4" % baseDir
    if False:
        fitCfg['mean_cfg'] = [1, 1, 1, 1] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
        fitCfg['sigma_cfg'] = [2, 0, 0, 0]
        fitCfg['norm_cfg'] = [2, 2, 2]
        with open("%s/results.json" % outDir_param_v3) as f: jsIn = json.load(f)
        recoilLibs.doFitMultiGauss(rFile, proc, comp, fitCfg, label, outDir_fits_v4, recoil_qTbins, funcJs=jsIn, qTmax=150, rebin=rebin, singleMean=True, xMin=-100, xMax=100)
        return        
        
    outDir_param_v4 = "%s/param_v4" % baseDir
    if False:
        jsOut = {}
        jsOut['nGauss'] = nGauss
        outDir_param = outDir_param_v4
        functions.prepareDir(outDir_param, True)
        with open("%s/results.json" % outDir_fits_v4) as f: jsIn = json.load(f)
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [5.20382e+00, 1.78386e-01, 7.51449e+00], [False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma1", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{1} (GeV)")
 
        t = 25
        fLeft, dfLeft  = "[0] + [1]*x + [2]*x*x + [3]*x*x*x + [4]*x*x*x*x", "[1] + 2*[2]*x + 3*[3]*x*x + 4*[4]*x*x*x"
        fLeftEval, dfLeftEval = fLeft.replace("x", str(t)), dfLeft.replace("x", str(t))
        fitF = "(x<{t})*1*({fLeft}) +0+ (x>{t})*1*(  ({dfLeftEval})*x + ({fLeftEval}-({dfLeftEval})*{t}) )".format(t=t, fLeft=fLeft, fLeftEval=fLeftEval, dfLeftEval=dfLeftEval)
        params, cParams = [4.00314e+00, 6.34998e-02, -3.94494e-04, 2.83885e-05, -1.51997e-06], [False, False, False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma2", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=1, fitMax=150, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{2} (GeV)")
        
        fitF, params, cParams = "[0] + [1]*x", [7.45735e+00, 6.52883e-03], [False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma3", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{2} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [1.15048e-02, 9.60879e-01, 2.26090e+01], [False, False, False]
        t = 25
        fLeft, dfLeft  = "[0] + [1]*x + [2]*x*x + [3]*x*x*x + [4]*x*x*x*x", "[1] + 2*[2]*x + 3*[3]*x*x + 4*[4]*x*x*x"
        fLeftEval, dfLeftEval = fLeft.replace("x", str(t)), dfLeft.replace("x", str(t))
        fitF = "(x<{t})*1*({fLeft}) +0+ (x>{t})*1*(  ({dfLeftEval})*x + ({fLeftEval}-({dfLeftEval})*{t}) )".format(t=t, fLeft=fLeft, fLeftEval=fLeftEval, dfLeftEval=dfLeftEval)
        params, cParams = [9.99069e+00, -7.11533e-02, 1.92415e-02, -1.05192e-03, 1.72908e-05], [False, False, False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma4", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{4} (GeV)")
        

        t = 25
        fLeft, dfLeft  = "[0] + [1]*x + [2]*x*x + [3]*x*x*x + [4]*x*x*x*x", "[1] + 2*[2]*x + 3*[3]*x*x + 4*[4]*x*x*x"
        fLeftEval, dfLeftEval = fLeft.replace("x", str(t)), dfLeft.replace("x", str(t))
        fitF = "(x<{t})*1*({fLeft}) +0+ (x>{t})*1*(  ({dfLeftEval})*x + ({fLeftEval}-({dfLeftEval})*{t}) )".format(t=t, fLeft=fLeft, fLeftEval=fLeftEval, dfLeftEval=dfLeftEval)
        params, cParams = [4.95396e-02, -2.88434e-03, 5.76138e-05, 2.54185e-06, -6.16914e-08], [True, True, True, True, True]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm1", fitF, params, outDir_param, recoil_qTbins, label, cutOffMin=0.02, cParams=cParams, fitMin=2, fitMax=60  , yMin=0, yMax=1, xMax=150, yTitle = "n_{3} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Exp(-[1]*x)", [2.33265e-01, 4.39271e-03], [True, True]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm2", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=1, xMax=150, yTitle = "n_{3} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Exp(-[1]*x)", [5.15149e-01, 1.71424e-02], [True, True]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm3", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=30, yMin=0, yMax=1, xMax=150, yTitle = "n_{3} (GeV)")
        
        with open("%s/results.json" % outDir_param_v4, "w") as outfile: json.dump(jsOut, outfile, indent=4)    
        return

    outDir_fits_v5 = "%s/fits_v5" % baseDir
    if False:
        fitCfg['mean_cfg'] = [1, 1, 1, 1] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
        fitCfg['sigma_cfg'] = [2, 2, 0, 0]
        fitCfg['norm_cfg'] = [2, 2, 2]
        with open("%s/results.json" % outDir_param_v4) as f: jsIn = json.load(f)
        recoilLibs.doFitMultiGauss(rFile, proc, comp, fitCfg, label, outDir_fits_v5, recoil_qTbins, funcJs=jsIn, qTmax=150, rebin=rebin, singleMean=True, xMin=-100, xMax=100)
        return   
        
    outDir_param_v5 = "%s/param_v5" % baseDir
    if False:
        jsOut = {}
        jsOut['nGauss'] = nGauss
        outDir_param = outDir_param_v5
        functions.prepareDir(outDir_param, True)
        with open("%s/results.json" % outDir_fits_v5) as f: jsIn = json.load(f)

        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [5.20382e+00, 1.78386e-01, 7.51449e+00], [False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma1", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{1} (GeV)")
 
        t = 25
        fLeft, dfLeft  = "[0] + [1]*x + [2]*x*x + [3]*x*x*x + [4]*x*x*x*x", "[1] + 2*[2]*x + 3*[3]*x*x + 4*[4]*x*x*x"
        fLeftEval, dfLeftEval = fLeft.replace("x", str(t)), dfLeft.replace("x", str(t))
        fitF = "(x<{t})*1*({fLeft}) +0+ (x>{t})*1*(  ({dfLeftEval})*x + ({fLeftEval}-({dfLeftEval})*{t}) )".format(t=t, fLeft=fLeft, fLeftEval=fLeftEval, dfLeftEval=dfLeftEval)
        params, cParams = [4.00314e+00, 6.34998e-02, -3.94494e-04, 2.83885e-05, -1.51997e-06], [False, False, False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma2", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=1, fitMax=150, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{2} (GeV)")
        
        fitF, params, cParams = "[0] + [1]*x", [7.45735e+00, 6.52883e-03], [False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma3", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{2} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [1.15048e-02, 9.60879e-01, 2.26090e+01], [False, False, False]
        t = 25
        fLeft, dfLeft  = "[0] + [1]*x + [2]*x*x + [3]*x*x*x + [4]*x*x*x*x", "[1] + 2*[2]*x + 3*[3]*x*x + 4*[4]*x*x*x"
        fLeftEval, dfLeftEval = fLeft.replace("x", str(t)), dfLeft.replace("x", str(t))
        fitF = "(x<{t})*1*({fLeft}) +0+ (x>{t})*1*(  ({dfLeftEval})*x + ({fLeftEval}-({dfLeftEval})*{t}) )".format(t=t, fLeft=fLeft, fLeftEval=fLeftEval, dfLeftEval=dfLeftEval)
        params, cParams = [9.99069e+00, -7.11533e-02, 1.92415e-02, -1.05192e-03, 1.72908e-05], [False, False, False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma4", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{4} (GeV)")
        

        t = 25
        fLeft, dfLeft  = "[0] + [1]*x + [2]*x*x + [3]*x*x*x + [4]*x*x*x*x", "[1] + 2*[2]*x + 3*[3]*x*x + 4*[4]*x*x*x"
        fLeftEval, dfLeftEval = fLeft.replace("x", str(t)), dfLeft.replace("x", str(t))
        fitF = "(x<{t})*1*({fLeft}) +0+ (x>{t})*1*(  ({dfLeftEval})*x + ({fLeftEval}-({dfLeftEval})*{t}) )".format(t=t, fLeft=fLeft, fLeftEval=fLeftEval, dfLeftEval=dfLeftEval)
        params, cParams = [4.95396e-02, -2.88434e-03, 5.76138e-05, 2.54185e-06, -6.16914e-08], [True, True, True, True, True]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm1", fitF, params, outDir_param, recoil_qTbins, label, cutOffMin=0.02, cParams=cParams, fitMin=2, fitMax=60  , yMin=0, yMax=1, xMax=150, yTitle = "n_{3} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Exp(-[1]*x)", [2.33265e-01, 4.39271e-03], [True, True]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm2", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=1, xMax=150, yTitle = "n_{3} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Exp(-[1]*x)", [5.15149e-01, 1.71424e-02], [True, True]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm3", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=30, yMin=0, yMax=1, xMax=150, yTitle = "n_{3} (GeV)")
        
        with open("%s/results.json" % outDir_param_v5, "w") as outfile: json.dump(jsOut, outfile, indent=4)    
        return   

    outDir_fits_v6 = "%s/fits_v6" % baseDir
    if False:
        fitCfg['mean_cfg'] = [1, 1, 1, 1] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
        fitCfg['sigma_cfg'] = [2, 2, 2, 0]
        fitCfg['norm_cfg'] = [2, 2, 2]
        with open("%s/results.json" % outDir_param_v5) as f: jsIn = json.load(f)
        recoilLibs.doFitMultiGauss(rFile, proc, comp, fitCfg, label, outDir_fits_v6, recoil_qTbins, funcJs=jsIn, qTmax=150, rebin=rebin, singleMean=True, xMin=-100, xMax=100)
        return        
 
    outDir_param_v6 = "%s/param_v6" % baseDir
    if True:
        jsOut = {}
        jsOut['nGauss'] = nGauss
        outDir_param = outDir_param_v6
        functions.prepareDir(outDir_param, True)
        with open("%s/results.json" % outDir_fits_v6) as f: jsIn = json.load(f)
        
        fitF, params, cParams = "[0]", [0.0], [True]
        recoilLibs.doFit(jsIn, jsOut, comp, "mean1", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=40, yMin=0, yMax=1, xMax=150, yTitle = "#mu_{1} (GeV)", doFit=False)
        recoilLibs.doFit(jsIn, jsOut, comp, "mean2", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=40, yMin=0, yMax=1, xMax=150, yTitle = "#mu_{2} (GeV)", doFit=False)
        recoilLibs.doFit(jsIn, jsOut, comp, "mean3", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=40, yMin=0, yMax=1, xMax=150, yTitle = "#mu_{3} (GeV)", doFit=False)
        recoilLibs.doFit(jsIn, jsOut, comp, "mean4", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=40, yMin=0, yMax=1, xMax=150, yTitle = "#mu_{4} (GeV)", doFit=False)
        
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [5.20382e+00, 1.78386e-01, 7.51449e+00], [False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma1", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{1} (GeV)")
 
        t = 25
        fLeft, dfLeft  = "[0] + [1]*x + [2]*x*x + [3]*x*x*x + [4]*x*x*x*x", "[1] + 2*[2]*x + 3*[3]*x*x + 4*[4]*x*x*x"
        fLeftEval, dfLeftEval = fLeft.replace("x", str(t)), dfLeft.replace("x", str(t))
        fitF = "(x<{t})*1*({fLeft}) +0+ (x>{t})*1*(  ({dfLeftEval})*x + ({fLeftEval}-({dfLeftEval})*{t}) )".format(t=t, fLeft=fLeft, fLeftEval=fLeftEval, dfLeftEval=dfLeftEval)
        params, cParams = [4.00314e+00, 6.34998e-02, -3.94494e-04, 2.83885e-05, -1.51997e-06], [False, False, False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma2", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=1, fitMax=150, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{2} (GeV)")
        
        fitF, params, cParams = "[0] + [1]*x", [7.45735e+00, 6.52883e-03], [False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma3", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{2} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [1.15048e-02, 9.60879e-01, 2.26090e+01], [False, False, False]
        t = 25
        fLeft, dfLeft  = "[0] + [1]*x + [2]*x*x + [3]*x*x*x + [4]*x*x*x*x", "[1] + 2*[2]*x + 3*[3]*x*x + 4*[4]*x*x*x"
        fLeftEval, dfLeftEval = fLeft.replace("x", str(t)), dfLeft.replace("x", str(t))
        fitF = "(x<{t})*1*({fLeft}) +0+ (x>{t})*1*(  ({dfLeftEval})*x + ({fLeftEval}-({dfLeftEval})*{t}) )".format(t=t, fLeft=fLeft, fLeftEval=fLeftEval, dfLeftEval=dfLeftEval)
        params, cParams = [9.99069e+00, -7.11533e-02, 1.92415e-02, -1.05192e-03, 1.72908e-05], [False, False, False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma4", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{4} (GeV)")
        

        t = 25
        fLeft, dfLeft  = "[0] + [1]*x + [2]*x*x + [3]*x*x*x + [4]*x*x*x*x", "[1] + 2*[2]*x + 3*[3]*x*x + 4*[4]*x*x*x"
        fLeftEval, dfLeftEval = fLeft.replace("x", str(t)), dfLeft.replace("x", str(t))
        fitF = "(x<{t})*1*({fLeft}) +0+ (x>{t})*1*(  ({dfLeftEval})*x + ({fLeftEval}-({dfLeftEval})*{t}) )".format(t=t, fLeft=fLeft, fLeftEval=fLeftEval, dfLeftEval=dfLeftEval)
        params, cParams = [4.95396e-02, -2.88434e-03, 5.76138e-05, 2.54185e-06, -6.16914e-08], [True, True, True, True, True]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm1", fitF, params, outDir_param, recoil_qTbins, label, cutOffMin=0.02, cParams=cParams, fitMin=2, fitMax=60  , yMin=0, yMax=1, xMax=150, yTitle = "n_{3} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Exp(-[1]*x)", [2.33265e-01, 4.39271e-03], [True, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm2", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=1, xMax=150, yTitle = "n_{3} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Exp(-[1]*x)", [5.15149e-01, 1.71424e-02], [True, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm3", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=30, yMin=0, yMax=1, xMax=150, yTitle = "n_{3} (GeV)")
        
        with open("%s/results.json" % outDir_param, "w") as outfile: json.dump(jsOut, outfile, indent=4)    
        #return    


    ########################  
    # Nominal refit
    ########################
    if True:
        outDir_refit = "%s/params_refit" % baseDir
        functions.prepareDir(outDir_refit)
        
        # need to get the original fit config, to not fit twice some entangled params (treat them as RooRealVar alias)
        fitCfg['mean_cfg'] = [1, 1, 1, 1] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
        fitCfg['sigma_cfg'] = [2, 2, 2, 2]
        fitCfg['norm_cfg'] = [2, 2, 2]
        with open("%s/results.json" % outDir_param_v6) as f: jsIn = json.load(f)
        jsOut = recoilLibs.combinedFit(rFile, proc, comp, fitCfg, jsIn, recoil_qTbins, rebin=rebin, qTmax=120) # 120 cov matrix OK. but larger errors (though dominated by DATA)? was 100 before
        with open("%s/results_refit.json" % outDir_refit, "w") as outfile: json.dump(jsOut, outfile, indent=4)
        #return
        

        fitCfg['mean_cfg'] = [2, 2, 2, 2] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
        fitCfg['sigma_cfg'] = [2, 2, 2, 2]
        fitCfg['norm_cfg'] = [2, 2, 2]
        
        with open("%s/results_refit.json" % outDir_refit) as f: jsIn = json.load(f)
        recoilLibs.doPlot(comp, "norm1", jsIn, outDir_refit, recoil_qTbins, label, yMin=0, yMax=1, xMax=150, yTitle = "n_{1} (GeV)")
        recoilLibs.doPlot(comp, "norm2", jsIn, outDir_refit, recoil_qTbins, label, yMin=0, yMax=1, xMax=150, yTitle = "n_{2} (GeV)")
        recoilLibs.doPlot(comp, "norm3", jsIn, outDir_refit, recoil_qTbins, label, yMin=0, yMax=1, xMax=150, yTitle = "n_{3} (GeV)")
        recoilLibs.doPlot(comp, "sigma1", jsIn, outDir_refit, recoil_qTbins, label, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{1} (GeV)")
        recoilLibs.doPlot(comp, "sigma2", jsIn, outDir_refit, recoil_qTbins, label, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{2} (GeV)")
        recoilLibs.doPlot(comp, "sigma3", jsIn, outDir_refit, recoil_qTbins, label, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{3} (GeV)")
        recoilLibs.doPlot(comp, "sigma4", jsIn, outDir_refit, recoil_qTbins, label, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{4} (GeV)")

        outDir_refit_fits = "%s/fits_refit" % baseDir
        recoilLibs.doFitMultiGauss(rFile, proc, comp, fitCfg, label, outDir_refit_fits, recoil_qTbins, funcJs=jsIn, qTmax=qTmax, rebin=rebin, xMin=-100, xMax=100)
        ROOT.gROOT.Reset()
      

def do_dymumu_perp_RawPFMET():

    proc = "DYmumu"
    tag = "dymumu_perp"
    comp = "perp"
    baseDir = "%s/%s" % (outDir, tag)
    name = "DY #rightarrow #mu^{+}#mu^{#minus}"
    label = "%s, %s" % (name, met)
    functions.prepareDir(baseDir, False)
    rebin = 1

    outDir_fits_v0 = "%s/fits_v0" % baseDir
    fitCfg = {}   
    
    fitCfg['mean'], fitCfg['mean_cfg'] = [0, 0, 0, 0], [0, 0, 0, 0] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
    fitCfg['sigma'], fitCfg['sigma_cfg'] = [15, 5, 8, 10], [0, 0, 0, 0]
    fitCfg['norm'], fitCfg['norm_cfg'] = [0.05, 0.20, 0.25], [0, 0, 0]
    

    recoil_qTbins = list(functions.drange(0, 50, 0.5)) +  list(range(50, 100, 1)) + list(range(100, 150, 2)) + [150]
    
    nGauss = len(fitCfg['mean'])
    
    if False: 
        recoilLibs.doFitMultiGauss(rFile, proc, comp, fitCfg, label, outDir_fits_v0, recoil_qTbins, qTmax=150, propagate=False, rebin=rebin, singleMean=True, xMin=-100, xMax=100)
        return
        
    outDir_param_v0 = "%s/param_v0" % baseDir
    if False:
        jsOut = {}
        jsOut['nGauss'] = nGauss
        outDir_param = outDir_param_v0
        functions.prepareDir(outDir_param, True)
        with open("%s/results.json" % outDir_fits_v0) as f: jsIn = json.load(f)
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [5.20382e+00, 1.78386e-01, 7.51449e+00], [False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma1", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{1} (GeV)")
 
        t = 25
        fLeft, dfLeft  = "[0] + [1]*x + [2]*x*x + [3]*x*x*x + [4]*x*x*x*x", "[1] + 2*[2]*x + 3*[3]*x*x + 4*[4]*x*x*x"
        fLeftEval, dfLeftEval = fLeft.replace("x", str(t)), dfLeft.replace("x", str(t))
        fitF = "(x<{t})*1*({fLeft}) +0+ (x>{t})*1*(  ({dfLeftEval})*x + ({fLeftEval}-({dfLeftEval})*{t}) )".format(t=t, fLeft=fLeft, fLeftEval=fLeftEval, dfLeftEval=dfLeftEval)
        params, cParams = [4.00314e+00, 6.34998e-02, -3.94494e-04, 2.83885e-05, -1.51997e-06], [False, False, False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma2", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=1, fitMax=150, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{2} (GeV)")
        
        fitF, params, cParams = "[0] + [1]*x", [7.45735e+00, 6.52883e-03], [False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma3", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{2} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [1.15048e-02, 9.60879e-01, 2.26090e+01], [False, False, False]
        t = 25
        fLeft, dfLeft  = "[0] + [1]*x + [2]*x*x + [3]*x*x*x + [4]*x*x*x*x", "[1] + 2*[2]*x + 3*[3]*x*x + 4*[4]*x*x*x"
        fLeftEval, dfLeftEval = fLeft.replace("x", str(t)), dfLeft.replace("x", str(t))
        fitF = "(x<{t})*1*({fLeft}) +0+ (x>{t})*1*(  ({dfLeftEval})*x + ({fLeftEval}-({dfLeftEval})*{t}) )".format(t=t, fLeft=fLeft, fLeftEval=fLeftEval, dfLeftEval=dfLeftEval)
        params, cParams = [9.99069e+00, -7.11533e-02, 1.92415e-02, -1.05192e-03, 1.72908e-05], [False, False, False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma4", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{4} (GeV)")
        

        t = 25
        fLeft, dfLeft  = "[0] + [1]*x + [2]*x*x + [3]*x*x*x + [4]*x*x*x*x", "[1] + 2*[2]*x + 3*[3]*x*x + 4*[4]*x*x*x"
        fLeftEval, dfLeftEval = fLeft.replace("x", str(t)), dfLeft.replace("x", str(t))
        fitF = "(x<{t})*1*({fLeft}) +0+ (x>{t})*1*(  ({dfLeftEval})*x + ({fLeftEval}-({dfLeftEval})*{t}) )".format(t=t, fLeft=fLeft, fLeftEval=fLeftEval, dfLeftEval=dfLeftEval)
        params, cParams = [4.95396e-02, -2.88434e-03, 5.76138e-05, 2.54185e-06, -6.16914e-08], [True, True, True, True, True]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm1", fitF, params, outDir_param, recoil_qTbins, label, cutOffMin=0.02, cParams=cParams, fitMin=2, fitMax=60  , yMin=0, yMax=1, xMax=150, yTitle = "n_{3} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Exp(-[1]*x)", [2.33265e-01, 4.39271e-03], [True, True]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm2", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=1, xMax=150, yTitle = "n_{3} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Exp(-[1]*x)", [5.15149e-01, 1.71424e-02], [True, True]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm3", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=30, yMin=0, yMax=1, xMax=150, yTitle = "n_{3} (GeV)")
        
        with open("%s/results.json" % outDir_param_v0, "w") as outfile: json.dump(jsOut, outfile, indent=4)
        return

    outDir_fits_v1 = "%s/fits_v1" % baseDir
    if False:
        fitCfg['mean_cfg'] = [1, 1, 1, 1] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
        fitCfg['sigma_cfg'] = [0, 0, 0, 0]
        fitCfg['norm_cfg'] = [0, 2, 0]
        fitCfg['sigma'] = [12.6, 3.8, 7.9, 13]
        with open("%s/results.json" % outDir_param_v0) as f: jsIn = json.load(f)
        recoilLibs.doFitMultiGauss(rFile, proc, comp, fitCfg, label, outDir_fits_v1, recoil_qTbins, funcJs=jsIn, qTmax=150, rebin=rebin, singleMean=True, xMin=-100, xMax=100)
        return

    outDir_param_v1 = "%s/param_v1" % baseDir
    if False:
        jsOut = {}
        jsOut['nGauss'] = nGauss
        outDir_param = outDir_param_v1
        functions.prepareDir(outDir_param, True)
        with open("%s/results.json" % outDir_fits_v1) as f: jsIn = json.load(f)
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [5.20382e+00, 1.78386e-01, 7.51449e+00], [False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma1", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{1} (GeV)")
 
        t = 25
        fLeft, dfLeft  = "[0] + [1]*x + [2]*x*x + [3]*x*x*x + [4]*x*x*x*x", "[1] + 2*[2]*x + 3*[3]*x*x + 4*[4]*x*x*x"
        fLeftEval, dfLeftEval = fLeft.replace("x", str(t)), dfLeft.replace("x", str(t))
        fitF = "(x<{t})*1*({fLeft}) +0+ (x>{t})*1*(  ({dfLeftEval})*x + ({fLeftEval}-({dfLeftEval})*{t}) )".format(t=t, fLeft=fLeft, fLeftEval=fLeftEval, dfLeftEval=dfLeftEval)
        params, cParams = [4.00314e+00, 6.34998e-02, -3.94494e-04, 2.83885e-05, -1.51997e-06], [False, False, False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma2", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=1, fitMax=150, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{2} (GeV)")
        
        fitF, params, cParams = "[0] + [1]*x", [7.45735e+00, 6.52883e-03], [False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma3", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{2} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [1.15048e-02, 9.60879e-01, 2.26090e+01], [False, False, False]
        t = 25
        fLeft, dfLeft  = "[0] + [1]*x + [2]*x*x + [3]*x*x*x + [4]*x*x*x*x", "[1] + 2*[2]*x + 3*[3]*x*x + 4*[4]*x*x*x"
        fLeftEval, dfLeftEval = fLeft.replace("x", str(t)), dfLeft.replace("x", str(t))
        fitF = "(x<{t})*1*({fLeft}) +0+ (x>{t})*1*(  ({dfLeftEval})*x + ({fLeftEval}-({dfLeftEval})*{t}) )".format(t=t, fLeft=fLeft, fLeftEval=fLeftEval, dfLeftEval=dfLeftEval)
        params, cParams = [9.99069e+00, -7.11533e-02, 1.92415e-02, -1.05192e-03, 1.72908e-05], [False, False, False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma4", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{4} (GeV)")
        

        t = 25
        fLeft, dfLeft  = "[0] + [1]*x + [2]*x*x + [3]*x*x*x + [4]*x*x*x*x", "[1] + 2*[2]*x + 3*[3]*x*x + 4*[4]*x*x*x"
        fLeftEval, dfLeftEval = fLeft.replace("x", str(t)), dfLeft.replace("x", str(t))
        fitF = "(x<{t})*1*({fLeft}) +0+ (x>{t})*1*(  ({dfLeftEval})*x + ({fLeftEval}-({dfLeftEval})*{t}) )".format(t=t, fLeft=fLeft, fLeftEval=fLeftEval, dfLeftEval=dfLeftEval)
        params, cParams = [4.95396e-02, -2.88434e-03, 5.76138e-05, 2.54185e-06, -6.16914e-08], [True, True, True, True, True]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm1", fitF, params, outDir_param, recoil_qTbins, label, cutOffMin=0.02, cParams=cParams, fitMin=2, fitMax=60  , yMin=0, yMax=1, xMax=150, yTitle = "n_{3} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Exp(-[1]*x)", [2.33265e-01, 4.39271e-03], [True, True]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm2", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=1, xMax=150, yTitle = "n_{3} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Exp(-[1]*x)", [5.15149e-01, 1.71424e-02], [True, True]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm3", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=30, yMin=0, yMax=1, xMax=150, yTitle = "n_{3} (GeV)")
        
        with open("%s/results.json" % outDir_param_v1, "w") as outfile: json.dump(jsOut, outfile, indent=4)    
        return

    outDir_fits_v2 = "%s/fits_v2" % baseDir
    if False:
        fitCfg['mean_cfg'] = [1, 1, 1, 1] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
        fitCfg['sigma_cfg'] = [0, 0, 0, 0]
        fitCfg['norm_cfg'] = [0, 2, 2]
        fitCfg['sigma'] = [13, 4, 7, 10]
        with open("%s/results.json" % outDir_param_v1) as f: jsIn = json.load(f)
        recoilLibs.doFitMultiGauss(rFile, proc, comp, fitCfg, label, outDir_fits_v2, recoil_qTbins, funcJs=jsIn, qTmax=150, rebin=rebin, singleMean=True, xMin=-100, xMax=100)
        return
        
    outDir_param_v2 = "%s/param_v2" % baseDir
    if False:
        jsOut = {}
        jsOut['nGauss'] = nGauss
        outDir_param = outDir_param_v2
        functions.prepareDir(outDir_param, True)
        with open("%s/results.json" % outDir_fits_v2) as f: jsIn = json.load(f)
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [5.20382e+00, 1.78386e-01, 7.51449e+00], [False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma1", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{1} (GeV)")
 
        t = 25
        fLeft, dfLeft  = "[0] + [1]*x + [2]*x*x + [3]*x*x*x + [4]*x*x*x*x", "[1] + 2*[2]*x + 3*[3]*x*x + 4*[4]*x*x*x"
        fLeftEval, dfLeftEval = fLeft.replace("x", str(t)), dfLeft.replace("x", str(t))
        fitF = "(x<{t})*1*({fLeft}) +0+ (x>{t})*1*(  ({dfLeftEval})*x + ({fLeftEval}-({dfLeftEval})*{t}) )".format(t=t, fLeft=fLeft, fLeftEval=fLeftEval, dfLeftEval=dfLeftEval)
        params, cParams = [4.00314e+00, 6.34998e-02, -3.94494e-04, 2.83885e-05, -1.51997e-06], [False, False, False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma2", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=1, fitMax=150, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{2} (GeV)")
        
        fitF, params, cParams = "[0] + [1]*x", [7.45735e+00, 6.52883e-03], [False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma3", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{2} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [1.15048e-02, 9.60879e-01, 2.26090e+01], [False, False, False]
        t = 25
        fLeft, dfLeft  = "[0] + [1]*x + [2]*x*x + [3]*x*x*x + [4]*x*x*x*x", "[1] + 2*[2]*x + 3*[3]*x*x + 4*[4]*x*x*x"
        fLeftEval, dfLeftEval = fLeft.replace("x", str(t)), dfLeft.replace("x", str(t))
        fitF = "(x<{t})*1*({fLeft}) +0+ (x>{t})*1*(  ({dfLeftEval})*x + ({fLeftEval}-({dfLeftEval})*{t}) )".format(t=t, fLeft=fLeft, fLeftEval=fLeftEval, dfLeftEval=dfLeftEval)
        params, cParams = [9.99069e+00, -7.11533e-02, 1.92415e-02, -1.05192e-03, 1.72908e-05], [False, False, False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma4", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{4} (GeV)")
        

        t = 25
        fLeft, dfLeft  = "[0] + [1]*x + [2]*x*x + [3]*x*x*x + [4]*x*x*x*x", "[1] + 2*[2]*x + 3*[3]*x*x + 4*[4]*x*x*x"
        fLeftEval, dfLeftEval = fLeft.replace("x", str(t)), dfLeft.replace("x", str(t))
        fitF = "(x<{t})*1*({fLeft}) +0+ (x>{t})*1*(  ({dfLeftEval})*x + ({fLeftEval}-({dfLeftEval})*{t}) )".format(t=t, fLeft=fLeft, fLeftEval=fLeftEval, dfLeftEval=dfLeftEval)
        params, cParams = [4.95396e-02, -2.88434e-03, 5.76138e-05, 2.54185e-06, -6.16914e-08], [True, True, True, True, True]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm1", fitF, params, outDir_param, recoil_qTbins, label, cutOffMin=0.02, cParams=cParams, fitMin=2, fitMax=60  , yMin=0, yMax=1, xMax=150, yTitle = "n_{3} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Exp(-[1]*x)", [2.33265e-01, 4.39271e-03], [True, True]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm2", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=1, xMax=150, yTitle = "n_{3} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Exp(-[1]*x)", [5.15149e-01, 1.71424e-02], [True, True]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm3", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=30, yMin=0, yMax=1, xMax=150, yTitle = "n_{3} (GeV)")
        
        with open("%s/results.json" % outDir_param_v2, "w") as outfile: json.dump(jsOut, outfile, indent=4)    
        return
        
    outDir_fits_v3 = "%s/fits_v3" % baseDir
    if False:
        fitCfg['mean_cfg'] = [1, 1, 1, 1] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
        fitCfg['sigma_cfg'] = [0, 0, 0, 0]
        fitCfg['norm_cfg'] = [2, 2, 2]
        fitCfg['sigma'] = [13, 4, 7, 10]
        with open("%s/results.json" % outDir_param_v2) as f: jsIn = json.load(f)
        recoilLibs.doFitMultiGauss(rFile, proc, comp, fitCfg, label, outDir_fits_v3, recoil_qTbins, funcJs=jsIn, qTmax=150, rebin=rebin, singleMean=True, xMin=-100, xMax=100)
        return
        
    outDir_param_v3 = "%s/param_v3" % baseDir
    if False:
        jsOut = {}
        jsOut['nGauss'] = nGauss
        outDir_param = outDir_param_v3
        functions.prepareDir(outDir_param, True)
        with open("%s/results.json" % outDir_fits_v3) as f: jsIn = json.load(f)
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [5.20382e+00, 1.78386e-01, 7.51449e+00], [False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma1", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{1} (GeV)")
 
        t = 25
        fLeft, dfLeft  = "[0] + [1]*x + [2]*x*x + [3]*x*x*x + [4]*x*x*x*x", "[1] + 2*[2]*x + 3*[3]*x*x + 4*[4]*x*x*x"
        fLeftEval, dfLeftEval = fLeft.replace("x", str(t)), dfLeft.replace("x", str(t))
        fitF = "(x<{t})*1*({fLeft}) +0+ (x>{t})*1*(  ({dfLeftEval})*x + ({fLeftEval}-({dfLeftEval})*{t}) )".format(t=t, fLeft=fLeft, fLeftEval=fLeftEval, dfLeftEval=dfLeftEval)
        params, cParams = [4.00314e+00, 6.34998e-02, -3.94494e-04, 2.83885e-05, -1.51997e-06], [False, False, False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma2", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=1, fitMax=150, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{2} (GeV)")
        
        fitF, params, cParams = "[0] + [1]*x", [7.45735e+00, 6.52883e-03], [False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma3", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{2} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [1.15048e-02, 9.60879e-01, 2.26090e+01], [False, False, False]
        t = 25
        fLeft, dfLeft  = "[0] + [1]*x + [2]*x*x + [3]*x*x*x + [4]*x*x*x*x", "[1] + 2*[2]*x + 3*[3]*x*x + 4*[4]*x*x*x"
        fLeftEval, dfLeftEval = fLeft.replace("x", str(t)), dfLeft.replace("x", str(t))
        fitF = "(x<{t})*1*({fLeft}) +0+ (x>{t})*1*(  ({dfLeftEval})*x + ({fLeftEval}-({dfLeftEval})*{t}) )".format(t=t, fLeft=fLeft, fLeftEval=fLeftEval, dfLeftEval=dfLeftEval)
        params, cParams = [9.99069e+00, -7.11533e-02, 1.92415e-02, -1.05192e-03, 1.72908e-05], [False, False, False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma4", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{4} (GeV)")
        

        t = 25
        fLeft, dfLeft  = "[0] + [1]*x + [2]*x*x + [3]*x*x*x + [4]*x*x*x*x", "[1] + 2*[2]*x + 3*[3]*x*x + 4*[4]*x*x*x"
        fLeftEval, dfLeftEval = fLeft.replace("x", str(t)), dfLeft.replace("x", str(t))
        fitF = "(x<{t})*1*({fLeft}) +0+ (x>{t})*1*(  ({dfLeftEval})*x + ({fLeftEval}-({dfLeftEval})*{t}) )".format(t=t, fLeft=fLeft, fLeftEval=fLeftEval, dfLeftEval=dfLeftEval)
        params, cParams = [4.95396e-02, -2.88434e-03, 5.76138e-05, 2.54185e-06, -6.16914e-08], [True, True, True, True, True]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm1", fitF, params, outDir_param, recoil_qTbins, label, cutOffMin=0.02, cParams=cParams, fitMin=2, fitMax=60  , yMin=0, yMax=1, xMax=150, yTitle = "n_{3} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Exp(-[1]*x)", [2.33265e-01, 4.39271e-03], [True, True]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm2", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=1, xMax=150, yTitle = "n_{3} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Exp(-[1]*x)", [5.15149e-01, 1.71424e-02], [True, True]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm3", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=30, yMin=0, yMax=1, xMax=150, yTitle = "n_{3} (GeV)")
        
        with open("%s/results.json" % outDir_param_v3, "w") as outfile: json.dump(jsOut, outfile, indent=4)    
        return        

    outDir_fits_v4 = "%s/fits_v4" % baseDir
    if False:
        fitCfg['mean_cfg'] = [1, 1, 1, 1] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
        fitCfg['sigma_cfg'] = [2, 0, 0, 0]
        fitCfg['norm_cfg'] = [2, 2, 2]
        with open("%s/results.json" % outDir_param_v3) as f: jsIn = json.load(f)
        recoilLibs.doFitMultiGauss(rFile, proc, comp, fitCfg, label, outDir_fits_v4, recoil_qTbins, funcJs=jsIn, qTmax=150, rebin=rebin, singleMean=True, xMin=-100, xMax=100)
        return        
        
    outDir_param_v4 = "%s/param_v4" % baseDir
    if False:
        jsOut = {}
        jsOut['nGauss'] = nGauss
        outDir_param = outDir_param_v4
        functions.prepareDir(outDir_param, True)
        with open("%s/results.json" % outDir_fits_v4) as f: jsIn = json.load(f)
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [5.20382e+00, 1.78386e-01, 7.51449e+00], [False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma1", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{1} (GeV)")
 
        t = 25
        fLeft, dfLeft  = "[0] + [1]*x + [2]*x*x + [3]*x*x*x + [4]*x*x*x*x", "[1] + 2*[2]*x + 3*[3]*x*x + 4*[4]*x*x*x"
        fLeftEval, dfLeftEval = fLeft.replace("x", str(t)), dfLeft.replace("x", str(t))
        fitF = "(x<{t})*1*({fLeft}) +0+ (x>{t})*1*(  ({dfLeftEval})*x + ({fLeftEval}-({dfLeftEval})*{t}) )".format(t=t, fLeft=fLeft, fLeftEval=fLeftEval, dfLeftEval=dfLeftEval)
        params, cParams = [4.00314e+00, 6.34998e-02, -3.94494e-04, 2.83885e-05, -1.51997e-06], [False, False, False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma2", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=1, fitMax=150, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{2} (GeV)")
        
        fitF, params, cParams = "[0] + [1]*x", [7.45735e+00, 6.52883e-03], [False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma3", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{2} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [1.15048e-02, 9.60879e-01, 2.26090e+01], [False, False, False]
        t = 25
        fLeft, dfLeft  = "[0] + [1]*x + [2]*x*x + [3]*x*x*x + [4]*x*x*x*x", "[1] + 2*[2]*x + 3*[3]*x*x + 4*[4]*x*x*x"
        fLeftEval, dfLeftEval = fLeft.replace("x", str(t)), dfLeft.replace("x", str(t))
        fitF = "(x<{t})*1*({fLeft}) +0+ (x>{t})*1*(  ({dfLeftEval})*x + ({fLeftEval}-({dfLeftEval})*{t}) )".format(t=t, fLeft=fLeft, fLeftEval=fLeftEval, dfLeftEval=dfLeftEval)
        params, cParams = [9.99069e+00, -7.11533e-02, 1.92415e-02, -1.05192e-03, 1.72908e-05], [False, False, False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma4", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{4} (GeV)")
        

        t = 25
        fLeft, dfLeft  = "[0] + [1]*x + [2]*x*x + [3]*x*x*x + [4]*x*x*x*x", "[1] + 2*[2]*x + 3*[3]*x*x + 4*[4]*x*x*x"
        fLeftEval, dfLeftEval = fLeft.replace("x", str(t)), dfLeft.replace("x", str(t))
        fitF = "(x<{t})*1*({fLeft}) +0+ (x>{t})*1*(  ({dfLeftEval})*x + ({fLeftEval}-({dfLeftEval})*{t}) )".format(t=t, fLeft=fLeft, fLeftEval=fLeftEval, dfLeftEval=dfLeftEval)
        params, cParams = [4.95396e-02, -2.88434e-03, 5.76138e-05, 2.54185e-06, -6.16914e-08], [True, True, True, True, True]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm1", fitF, params, outDir_param, recoil_qTbins, label, cutOffMin=0.02, cParams=cParams, fitMin=2, fitMax=60  , yMin=0, yMax=1, xMax=150, yTitle = "n_{3} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Exp(-[1]*x)", [2.33265e-01, 4.39271e-03], [True, True]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm2", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=1, xMax=150, yTitle = "n_{3} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Exp(-[1]*x)", [5.15149e-01, 1.71424e-02], [True, True]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm3", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=30, yMin=0, yMax=1, xMax=150, yTitle = "n_{3} (GeV)")
        
        with open("%s/results.json" % outDir_param_v4, "w") as outfile: json.dump(jsOut, outfile, indent=4)    
        return

    outDir_fits_v5 = "%s/fits_v5" % baseDir
    if False:
        fitCfg['mean_cfg'] = [1, 1, 1, 1] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
        fitCfg['sigma_cfg'] = [2, 2, 0, 0]
        fitCfg['norm_cfg'] = [2, 2, 2]
        with open("%s/results.json" % outDir_param_v4) as f: jsIn = json.load(f)
        recoilLibs.doFitMultiGauss(rFile, proc, comp, fitCfg, label, outDir_fits_v5, recoil_qTbins, funcJs=jsIn, qTmax=150, rebin=rebin, singleMean=True, xMin=-100, xMax=100)
        return   
        
    outDir_param_v5 = "%s/param_v5" % baseDir
    if False:
        jsOut = {}
        jsOut['nGauss'] = nGauss
        outDir_param = outDir_param_v5
        functions.prepareDir(outDir_param, True)
        with open("%s/results.json" % outDir_fits_v5) as f: jsIn = json.load(f)

        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [5.20382e+00, 1.78386e-01, 7.51449e+00], [False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma1", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{1} (GeV)")
 
        t = 25
        fLeft, dfLeft  = "[0] + [1]*x + [2]*x*x + [3]*x*x*x + [4]*x*x*x*x", "[1] + 2*[2]*x + 3*[3]*x*x + 4*[4]*x*x*x"
        fLeftEval, dfLeftEval = fLeft.replace("x", str(t)), dfLeft.replace("x", str(t))
        fitF = "(x<{t})*1*({fLeft}) +0+ (x>{t})*1*(  ({dfLeftEval})*x + ({fLeftEval}-({dfLeftEval})*{t}) )".format(t=t, fLeft=fLeft, fLeftEval=fLeftEval, dfLeftEval=dfLeftEval)
        params, cParams = [4.00314e+00, 6.34998e-02, -3.94494e-04, 2.83885e-05, -1.51997e-06], [False, False, False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma2", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=1, fitMax=150, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{2} (GeV)")
        
        fitF, params, cParams = "[0] + [1]*x", [7.45735e+00, 6.52883e-03], [False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma3", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{2} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [1.15048e-02, 9.60879e-01, 2.26090e+01], [False, False, False]
        t = 25
        fLeft, dfLeft  = "[0] + [1]*x + [2]*x*x + [3]*x*x*x + [4]*x*x*x*x", "[1] + 2*[2]*x + 3*[3]*x*x + 4*[4]*x*x*x"
        fLeftEval, dfLeftEval = fLeft.replace("x", str(t)), dfLeft.replace("x", str(t))
        fitF = "(x<{t})*1*({fLeft}) +0+ (x>{t})*1*(  ({dfLeftEval})*x + ({fLeftEval}-({dfLeftEval})*{t}) )".format(t=t, fLeft=fLeft, fLeftEval=fLeftEval, dfLeftEval=dfLeftEval)
        params, cParams = [9.99069e+00, -7.11533e-02, 1.92415e-02, -1.05192e-03, 1.72908e-05], [False, False, False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma4", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{4} (GeV)")
        

        t = 25
        fLeft, dfLeft  = "[0] + [1]*x + [2]*x*x + [3]*x*x*x + [4]*x*x*x*x", "[1] + 2*[2]*x + 3*[3]*x*x + 4*[4]*x*x*x"
        fLeftEval, dfLeftEval = fLeft.replace("x", str(t)), dfLeft.replace("x", str(t))
        fitF = "(x<{t})*1*({fLeft}) +0+ (x>{t})*1*(  ({dfLeftEval})*x + ({fLeftEval}-({dfLeftEval})*{t}) )".format(t=t, fLeft=fLeft, fLeftEval=fLeftEval, dfLeftEval=dfLeftEval)
        params, cParams = [4.95396e-02, -2.88434e-03, 5.76138e-05, 2.54185e-06, -6.16914e-08], [True, True, True, True, True]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm1", fitF, params, outDir_param, recoil_qTbins, label, cutOffMin=0.02, cParams=cParams, fitMin=2, fitMax=60  , yMin=0, yMax=1, xMax=150, yTitle = "n_{3} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Exp(-[1]*x)", [2.33265e-01, 4.39271e-03], [True, True]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm2", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=1, xMax=150, yTitle = "n_{3} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Exp(-[1]*x)", [5.15149e-01, 1.71424e-02], [True, True]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm3", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=30, yMin=0, yMax=1, xMax=150, yTitle = "n_{3} (GeV)")
        
        with open("%s/results.json" % outDir_param_v5, "w") as outfile: json.dump(jsOut, outfile, indent=4)    
        return   

    outDir_fits_v6 = "%s/fits_v6" % baseDir
    if False:
        fitCfg['mean_cfg'] = [1, 1, 1, 1] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
        fitCfg['sigma_cfg'] = [2, 2, 2, 0]
        fitCfg['norm_cfg'] = [2, 2, 2]
        with open("%s/results.json" % outDir_param_v5) as f: jsIn = json.load(f)
        recoilLibs.doFitMultiGauss(rFile, proc, comp, fitCfg, label, outDir_fits_v6, recoil_qTbins, funcJs=jsIn, qTmax=150, rebin=rebin, singleMean=True, xMin=-100, xMax=100)
        return        
 
    outDir_param_v6 = "%s/param_v6" % baseDir
    if True:
        jsOut = {}
        jsOut['nGauss'] = nGauss
        outDir_param = outDir_param_v6
        functions.prepareDir(outDir_param, True)
        with open("%s/results.json" % outDir_fits_v6) as f: jsIn = json.load(f)
        
        fitF, params, cParams = "[0]", [0.0], [True]
        recoilLibs.doFit(jsIn, jsOut, comp, "mean1", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=40, yMin=0, yMax=1, xMax=150, yTitle = "#mu_{1} (GeV)", doFit=False)
        recoilLibs.doFit(jsIn, jsOut, comp, "mean2", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=40, yMin=0, yMax=1, xMax=150, yTitle = "#mu_{2} (GeV)", doFit=False)
        recoilLibs.doFit(jsIn, jsOut, comp, "mean3", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=40, yMin=0, yMax=1, xMax=150, yTitle = "#mu_{3} (GeV)", doFit=False)
        recoilLibs.doFit(jsIn, jsOut, comp, "mean4", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=40, yMin=0, yMax=1, xMax=150, yTitle = "#mu_{4} (GeV)", doFit=False)
        
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [5.20382e+00, 1.78386e-01, 7.51449e+00], [False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma1", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{1} (GeV)")
 
        t = 25
        fLeft, dfLeft  = "[0] + [1]*x + [2]*x*x + [3]*x*x*x + [4]*x*x*x*x", "[1] + 2*[2]*x + 3*[3]*x*x + 4*[4]*x*x*x"
        fLeftEval, dfLeftEval = fLeft.replace("x", str(t)), dfLeft.replace("x", str(t))
        fitF = "(x<{t})*1*({fLeft}) +0+ (x>{t})*1*(  ({dfLeftEval})*x + ({fLeftEval}-({dfLeftEval})*{t}) )".format(t=t, fLeft=fLeft, fLeftEval=fLeftEval, dfLeftEval=dfLeftEval)
        params, cParams = [4.00314e+00, 6.34998e-02, -3.94494e-04, 2.83885e-05, -1.51997e-06], [False, False, False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma2", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=1, fitMax=150, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{2} (GeV)")
        
        fitF, params, cParams = "[0] + [1]*x", [7.45735e+00, 6.52883e-03], [False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma3", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{2} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [1.15048e-02, 9.60879e-01, 2.26090e+01], [False, False, False]
        t = 25
        fLeft, dfLeft  = "[0] + [1]*x + [2]*x*x + [3]*x*x*x + [4]*x*x*x*x", "[1] + 2*[2]*x + 3*[3]*x*x + 4*[4]*x*x*x"
        fLeftEval, dfLeftEval = fLeft.replace("x", str(t)), dfLeft.replace("x", str(t))
        fitF = "(x<{t})*1*({fLeft}) +0+ (x>{t})*1*(  ({dfLeftEval})*x + ({fLeftEval}-({dfLeftEval})*{t}) )".format(t=t, fLeft=fLeft, fLeftEval=fLeftEval, dfLeftEval=dfLeftEval)
        params, cParams = [9.99069e+00, -7.11533e-02, 1.92415e-02, -1.05192e-03, 1.72908e-05], [False, False, False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma4", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{4} (GeV)")
        

        t = 25
        fLeft, dfLeft  = "[0] + [1]*x + [2]*x*x + [3]*x*x*x + [4]*x*x*x*x", "[1] + 2*[2]*x + 3*[3]*x*x + 4*[4]*x*x*x"
        fLeftEval, dfLeftEval = fLeft.replace("x", str(t)), dfLeft.replace("x", str(t))
        fitF = "(x<{t})*1*({fLeft}) +0+ (x>{t})*1*(  ({dfLeftEval})*x + ({fLeftEval}-({dfLeftEval})*{t}) )".format(t=t, fLeft=fLeft, fLeftEval=fLeftEval, dfLeftEval=dfLeftEval)
        params, cParams = [4.95396e-02, -2.88434e-03, 5.76138e-05, 2.54185e-06, -6.16914e-08], [True, True, True, True, True]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm1", fitF, params, outDir_param, recoil_qTbins, label, cutOffMin=0.02, cParams=cParams, fitMin=2, fitMax=60  , yMin=0, yMax=1, xMax=150, yTitle = "n_{3} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Exp(-[1]*x)", [2.33265e-01, 4.39271e-03], [True, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm2", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=1, xMax=150, yTitle = "n_{3} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Exp(-[1]*x)", [5.15149e-01, 1.71424e-02], [True, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm3", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=30, yMin=0, yMax=1, xMax=150, yTitle = "n_{3} (GeV)")
        
        with open("%s/results.json" % outDir_param, "w") as outfile: json.dump(jsOut, outfile, indent=4)    
        #return    


    ########################  
    # Nominal refit
    ########################
    if True:
        outDir_refit = "%s/params_refit" % baseDir
        functions.prepareDir(outDir_refit)
        
        # need to get the original fit config, to not fit twice some entangled params (treat them as RooRealVar alias)
        fitCfg['mean_cfg'] = [1, 1, 1, 1] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
        fitCfg['sigma_cfg'] = [2, 2, 2, 2]
        fitCfg['norm_cfg'] = [2, 2, 2]
        with open("%s/results.json" % outDir_param_v6) as f: jsIn = json.load(f)
        jsOut = recoilLibs.combinedFit(rFile, proc, comp, fitCfg, jsIn, recoil_qTbins, rebin=rebin, qTmax=120) # 120 cov matrix OK. but larger errors (though dominated by DATA)? was 100 before
        with open("%s/results_refit.json" % outDir_refit, "w") as outfile: json.dump(jsOut, outfile, indent=4)
        #return
        

        fitCfg['mean_cfg'] = [2, 2, 2, 2] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
        fitCfg['sigma_cfg'] = [2, 2, 2, 2]
        fitCfg['norm_cfg'] = [2, 2, 2]
        
        with open("%s/results_refit.json" % outDir_refit) as f: jsIn = json.load(f)
        recoilLibs.doPlot(comp, "norm1", jsIn, outDir_refit, recoil_qTbins, label, yMin=0, yMax=1, xMax=150, yTitle = "n_{1} (GeV)")
        recoilLibs.doPlot(comp, "norm2", jsIn, outDir_refit, recoil_qTbins, label, yMin=0, yMax=1, xMax=150, yTitle = "n_{2} (GeV)")
        recoilLibs.doPlot(comp, "norm3", jsIn, outDir_refit, recoil_qTbins, label, yMin=0, yMax=1, xMax=150, yTitle = "n_{3} (GeV)")
        recoilLibs.doPlot(comp, "sigma1", jsIn, outDir_refit, recoil_qTbins, label, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{1} (GeV)")
        recoilLibs.doPlot(comp, "sigma2", jsIn, outDir_refit, recoil_qTbins, label, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{2} (GeV)")
        recoilLibs.doPlot(comp, "sigma3", jsIn, outDir_refit, recoil_qTbins, label, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{3} (GeV)")
        recoilLibs.doPlot(comp, "sigma4", jsIn, outDir_refit, recoil_qTbins, label, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{4} (GeV)")

        outDir_refit_fits = "%s/fits_refit" % baseDir
        recoilLibs.doFitMultiGauss(rFile, proc, comp, fitCfg, label, outDir_refit_fits, recoil_qTbins, funcJs=jsIn, qTmax=qTmax, rebin=rebin, xMin=-100, xMax=100)
        ROOT.gROOT.Reset()
      
def do_dymumu_para_RawPFMET_OK():

    proc = "DYmumu"
    tag = "dymumu_para"
    comp = "para"
    baseDir = "%s/%s" % (outDir, tag)
    name = "DY #rightarrow #mu^{+}#mu^{#minus}"
    label = "%s, %s" % (name, met)
    functions.prepareDir(baseDir, False)
    rebin = 1


    outDir_fits_v0 = "%s/fits_v0" % baseDir
    fitCfg = {}   
    
    fitCfg['mean'], fitCfg['mean_cfg'] = [0, 0, 0, 0, 0], [0, 0, 0, 0, 0] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
    fitCfg['sigma'], fitCfg['sigma_cfg'] = [20, 5, 8, 10, 15], [0, 0, 0, 0, 0]
    fitCfg['norm'], fitCfg['norm_cfg'] = [0.05, 0.20, 0.25, 0.25], [0, 0, 0, 0]
    
    fitCfg['mean'], fitCfg['mean_cfg'] = [0, 0, 0, 0, 0], [0, 0, 0, 0, 0] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
    fitCfg['sigma'], fitCfg['sigma_cfg'] = [4.5, 7.5, 10, 15, 30], [0, 0, 0, 0, 0]
    fitCfg['norm'], fitCfg['norm_cfg'] = [0.2, 0.35, 0.15, 0.15], [0, 0, 0, 0]
    
    fitCfg['mean'], fitCfg['mean_cfg'] = [0, 0, 0, 0], [0, -1, 0, -3] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
    fitCfg['sigma'], fitCfg['sigma_cfg'] = [4, 7, 10, 15], [0, 0, 0, 0]
    fitCfg['norm'], fitCfg['norm_cfg'] = [0.25, 0.25, 0.35], [0, 1, 1]

    recoil_qTbins = list(functions.drange(0, 50, 0.5)) +  list(range(50, 100, 1)) + list(range(100, 150, 2)) + [150]
    
    nGauss = len(fitCfg['mean'])
    
    if False: 
        recoilLibs.doFitMultiGauss(rFile, proc, comp, fitCfg, label, outDir_fits_v0, recoil_qTbins, qTmax=qTmax, propagate=True, rebin=rebin, xMin=-100, xMax=100)
        return
        
    outDir_param_v0 = "%s/param_v0" % baseDir
    if False:
        jsOut = {}
        jsOut['nGauss'] = nGauss
        outDir_param = outDir_param_v0
        functions.prepareDir(outDir_param, True)
        with open("%s/results.json" % outDir_fits_v0) as f: jsIn = json.load(f)
        
        t = 18
        fLeft, dfLeft  = "[0] + [1]*x + [2]*x*x + [3]*x*x*x + [4]*x*x*x*x", "[1] + 2*[2]*x + 3*[3]*x*x + 4*[4]*x*x*x"
        fLeftEval, dfLeftEval = fLeft.replace("x", str(t)), dfLeft.replace("x", str(t))
        fitF = "(x<{t})*1*({fLeft}) +0+ (x>{t})*1*(  ({dfLeftEval})*x + ({fLeftEval}-({dfLeftEval})*{t}) )".format(t=t, fLeft=fLeft, fLeftEval=fLeftEval, dfLeftEval=dfLeftEval)
        params, cParams = [0, 0, 0, 0, 0], [False, False, False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "mean1", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=100, yMin=-5, yMax=50, xMax=150, yTitle = "#mu_{1} (GeV)")
        recoilLibs.doFit(jsIn, jsOut, comp, "mean2", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=100, yMin=-5, yMax=50, xMax=150, yTitle = "#mu_{2} (GeV)")
        
        t = 10
        fLeft, dfLeft  = "[0] + [1]*x + [2]*x*x + [3]*x*x*x + [4]*x*x*x*x", "[1] + 2*[2]*x + 3*[3]*x*x + 4*[4]*x*x*x"
        fLeftEval, dfLeftEval = fLeft.replace("x", str(t)), dfLeft.replace("x", str(t))
        fitF = "(x<{t})*1*({fLeft}) +0+ (x>{t})*1*(  ({dfLeftEval})*x + ({fLeftEval}-({dfLeftEval})*{t}) )".format(t=t, fLeft=fLeft, fLeftEval=fLeftEval, dfLeftEval=dfLeftEval)
        params, cParams = [4.28651e-01, 3.16656e-01, -8.29820e-03, 9.19512e-05, 0], [False, False, False, False, True]
        recoilLibs.doFit(jsIn, jsOut, comp, "mean3", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=150, yMin=-5, yMax=50, xMax=150, yTitle = "#mu_{3} (GeV)")
        recoilLibs.doFit(jsIn, jsOut, comp, "mean4", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=150, yMin=-5, yMax=50, xMax=150, yTitle = "#mu_{4} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [7.29092e-02, 1.02290e+00, 3.89730e+00], [False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma1", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=30, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{1} (GeV)")
 
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [1.52752e+01, 1.24731e-02, -6.94362e+00], [False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma2", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=100, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{2} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [1.06057e-02, 1.10462e+00, 1.33946e+01], [False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma3", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=100, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{2} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [1.15048e-02, 9.60879e-01, 2.26090e+01], [False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma4", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=100, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{4} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Exp(-[1]*x)", [2.78987e-01, 8.26000e-03], [False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm1", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=25, yMin=0, yMax=1, xMax=150, yTitle = "n_{1} (GeV)")
        
        fitF, params, cParams = "[0]", [0.25], [True]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm2", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=50, yMin=0, yMax=1, xMax=150, yTitle = "n_{2} (GeV)")
        
        fitF, params, cParams = "[0]", [0.35], [True]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm3", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=50, yMin=0, yMax=1, xMax=150, yTitle = "n_{3} (GeV)")
        

        with open("%s/results.json" % outDir_param, "w") as outfile: json.dump(jsOut, outfile, indent=4)
        return
        
    outDir_fits_v1 = "%s/fits_v1" % baseDir
    if False:
        fitCfg['mean_cfg'] = [0, -1, 0, -3] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
        fitCfg['sigma_cfg'] = [0, 0, 0, 0]
        fitCfg['norm_cfg'] = [2, 2, 2]
        with open("%s/results.json" % outDir_param_v0) as f: jsIn = json.load(f)
        recoilLibs.doFitMultiGauss(rFile, proc, comp, fitCfg, label, outDir_fits_v1, recoil_qTbins, funcJs=jsIn, qTmax=qTmax, propagate=True, rebin=rebin, xMin=-100, xMax=100)
        return
        
    outDir_param_v1 = "%s/param_v1" % baseDir
    if False:
        jsOut = {}
        jsOut['nGauss'] = nGauss
        outDir_param = outDir_param_v1
        functions.prepareDir(outDir_param, True)
        with open("%s/results.json" % outDir_fits_v1) as f: jsIn = json.load(f)
        
        t = 20
        fLeft, dfLeft  = "[0] + [1]*x + [2]*x*x + [3]*x*x*x + [4]*x*x*x*x", "[1] + 2*[2]*x + 3*[3]*x*x + 4*[4]*x*x*x"
        fLeftEval, dfLeftEval = fLeft.replace("x", str(t)), dfLeft.replace("x", str(t))
        fitF = "(x<{t})*1*({fLeft}) +0+ (x>{t})*1*(  ({dfLeftEval})*x + ({fLeftEval}-({dfLeftEval})*{t}) )".format(t=t, fLeft=fLeft, fLeftEval=fLeftEval, dfLeftEval=dfLeftEval)
        params, cParams = [-1.49478e-01, 6.37319e-01, -5.64081e-02, 2.87988e-03, -5.58069e-05], [False, False, False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "mean1", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=150, yMin=-5, yMax=50, xMax=150, yTitle = "#mu_{1} (GeV)")
        recoilLibs.doFit(jsIn, jsOut, comp, "mean2", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=150, yMin=-5, yMax=50, xMax=150, yTitle = "#mu_{2} (GeV)")
        
        t = 15
        fLeft, dfLeft  = "[0] + [1]*x + [2]*x*x + [3]*x*x*x + [4]*x*x*x*x", "[1] + 2*[2]*x + 3*[3]*x*x + 4*[4]*x*x*x"
        fLeftEval, dfLeftEval = fLeft.replace("x", str(t)), dfLeft.replace("x", str(t))
        fitF = "(x<{t})*1*({fLeft}) +0+ (x>{t})*1*(  ({dfLeftEval})*x + ({fLeftEval}-({dfLeftEval})*{t}) )".format(t=t, fLeft=fLeft, fLeftEval=fLeftEval, dfLeftEval=dfLeftEval)
        params, cParams = [3.15241e-02, 3.42571e-01, -1.41860e-03, -1.21510e-03, 4.31265e-05], [False, False, False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "mean3", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=150, yMin=-5, yMax=50, xMax=150, yTitle = "#mu_{3} (GeV)")
        recoilLibs.doFit(jsIn, jsOut, comp, "mean4", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=150, yMin=-5, yMax=50, xMax=150, yTitle = "#mu_{4} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [7.29092e-02, 1.02290e+00, 3.89730e+00], [False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma1", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=30, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{1} (GeV)")
 
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [1.52752e+01, 1.24731e-02, -6.94362e+00], [False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma2", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=100, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{2} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [1.06057e-02, 1.10462e+00, 1.33946e+01], [False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma3", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=100, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{2} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [1.15048e-02, 9.60879e-01, 2.26090e+01], [False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma4", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=100, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{4} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Exp(-[1]*x)", [2.78987e-01, 8.26000e-03], [False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm1", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=25, yMin=0, yMax=1, xMax=150, yTitle = "n_{1} (GeV)")
        
        fitF, params, cParams = "[0]", [0.25], [True]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm2", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=50, yMin=0, yMax=1, xMax=150, yTitle = "n_{2} (GeV)")
        
        fitF, params, cParams = "[0]", [0.35], [True]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm3", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=50, yMin=0, yMax=1, xMax=150, yTitle = "n_{3} (GeV)")
        
        with open("%s/results.json" % outDir_param, "w") as outfile: json.dump(jsOut, outfile, indent=4)    
        return
        
    outDir_fits_v2 = "%s/fits_v2" % baseDir
    if False:
        fitCfg['mean_cfg'] = [2, 2, 2, 2] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
        fitCfg['sigma_cfg'] = [0, 0, 0, 0]
        fitCfg['norm_cfg'] = [2, 2, 2]
        with open("%s/results.json" % outDir_param_v1) as f: jsIn = json.load(f)
        recoilLibs.doFitMultiGauss(rFile, proc, comp, fitCfg, label, outDir_fits_v2, recoil_qTbins, funcJs=jsIn, propagate=True, qTmax=qTmax, rebin=rebin, xMin=-100, xMax=100)
        return
        
    outDir_param_v2 = "%s/param_v2" % baseDir
    if False:
        jsOut = {}
        jsOut['nGauss'] = nGauss
        outDir_param = outDir_param_v2
        functions.prepareDir(outDir_param_v2, True)
        with open("%s/results.json" % outDir_fits_v2) as f: jsIn = json.load(f)
        
        t = 20
        fLeft, dfLeft  = "[0] + [1]*x + [2]*x*x + [3]*x*x*x + [4]*x*x*x*x", "[1] + 2*[2]*x + 3*[3]*x*x + 4*[4]*x*x*x"
        fLeftEval, dfLeftEval = fLeft.replace("x", str(t)), dfLeft.replace("x", str(t))
        fitF = "(x<{t})*1*({fLeft}) +0+ (x>{t})*1*(  ({dfLeftEval})*x + ({fLeftEval}-({dfLeftEval})*{t}) )".format(t=t, fLeft=fLeft, fLeftEval=fLeftEval, dfLeftEval=dfLeftEval)
        params, cParams = [-1.49478e-01, 6.37319e-01, -5.64081e-02, 2.87988e-03, -5.58069e-05], [False, False, False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "mean1", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=150, yMin=-5, yMax=50, xMax=150, yTitle = "#mu_{1} (GeV)")
        recoilLibs.doFit(jsIn, jsOut, comp, "mean2", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=150, yMin=-5, yMax=50, xMax=150, yTitle = "#mu_{2} (GeV)")
        
        t = 15
        fLeft, dfLeft  = "[0] + [1]*x + [2]*x*x + [3]*x*x*x + [4]*x*x*x*x", "[1] + 2*[2]*x + 3*[3]*x*x + 4*[4]*x*x*x"
        fLeftEval, dfLeftEval = fLeft.replace("x", str(t)), dfLeft.replace("x", str(t))
        fitF = "(x<{t})*1*({fLeft}) +0+ (x>{t})*1*(  ({dfLeftEval})*x + ({fLeftEval}-({dfLeftEval})*{t}) )".format(t=t, fLeft=fLeft, fLeftEval=fLeftEval, dfLeftEval=dfLeftEval)
        params, cParams = [3.15241e-02, 3.42571e-01, -1.41860e-03, -1.21510e-03, 4.31265e-05], [False, False, False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "mean3", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=150, yMin=-5, yMax=50, xMax=150, yTitle = "#mu_{3} (GeV)")
        recoilLibs.doFit(jsIn, jsOut, comp, "mean4", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=150, yMin=-5, yMax=50, xMax=150, yTitle = "#mu_{4} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [1.48366e-01, 8.35984e-01, 4.02649e+00], [False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma1", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=80, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{1} (GeV)")
 
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [1.52752e+01, 1.24731e-02, -6.94362e+00], [False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma2", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=100, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{2} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [1.06057e-02, 1.10462e+00, 1.33946e+01], [False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma3", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=100, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{2} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [1.15048e-02, 9.60879e-01, 2.26090e+01], [False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma4", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=100, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{4} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Exp(-[1]*x)", [2.78987e-01, 8.26000e-03], [False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm1", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=25, yMin=0, yMax=1, xMax=150, yTitle = "n_{1} (GeV)")
        
        fitF, params, cParams = "[0]", [0.25], [True]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm2", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=50, yMin=0, yMax=1, xMax=150, yTitle = "n_{2} (GeV)")
        
        fitF, params, cParams = "[0]", [0.35], [True]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm3", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=50, yMin=0, yMax=1, xMax=150, yTitle = "n_{3} (GeV)")
        
        with open("%s/results.json" % outDir_param, "w") as outfile: json.dump(jsOut, outfile, indent=4)    
        return
        
    outDir_fits_v3 = "%s/fits_v3" % baseDir
    if False:
        fitCfg['mean_cfg'] = [2, 2, 2, 2] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
        fitCfg['sigma_cfg'] = [2, 0, 0, 0]
        fitCfg['norm_cfg'] = [2, 2, 2]
        with open("%s/results.json" % outDir_param_v2) as f: jsIn = json.load(f)
        recoilLibs.doFitMultiGauss(rFile, proc, comp, fitCfg, label, outDir_fits_v3, recoil_qTbins, funcJs=jsIn, qTmax=qTmax, rebin=rebin, xMin=-100, xMax=100)
        return
    
    outDir_param_v3 = "%s/param_v3" % baseDir
    if False:
        jsOut = {}
        jsOut['nGauss'] = nGauss
        outDir_param = outDir_param_v3
        functions.prepareDir(outDir_param, True)
        with open("%s/results.json" % outDir_fits_v3) as f: jsIn = json.load(f)
        
        t = 20
        fLeft, dfLeft  = "[0] + [1]*x + [2]*x*x + [3]*x*x*x + [4]*x*x*x*x", "[1] + 2*[2]*x + 3*[3]*x*x + 4*[4]*x*x*x"
        fLeftEval, dfLeftEval = fLeft.replace("x", str(t)), dfLeft.replace("x", str(t))
        fitF = "(x<{t})*1*({fLeft}) +0+ (x>{t})*1*(  ({dfLeftEval})*x + ({fLeftEval}-({dfLeftEval})*{t}) )".format(t=t, fLeft=fLeft, fLeftEval=fLeftEval, dfLeftEval=dfLeftEval)
        params, cParams = [-1.49478e-01, 6.37319e-01, -5.64081e-02, 2.87988e-03, -5.58069e-05], [False, False, False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "mean1", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=150, yMin=-5, yMax=50, xMax=150, yTitle = "#mu_{1} (GeV)")
        recoilLibs.doFit(jsIn, jsOut, comp, "mean2", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=150, yMin=-5, yMax=50, xMax=150, yTitle = "#mu_{2} (GeV)")
        
        t = 15
        fLeft, dfLeft  = "[0] + [1]*x + [2]*x*x + [3]*x*x*x + [4]*x*x*x*x", "[1] + 2*[2]*x + 3*[3]*x*x + 4*[4]*x*x*x"
        fLeftEval, dfLeftEval = fLeft.replace("x", str(t)), dfLeft.replace("x", str(t))
        fitF = "(x<{t})*1*({fLeft}) +0+ (x>{t})*1*(  ({dfLeftEval})*x + ({fLeftEval}-({dfLeftEval})*{t}) )".format(t=t, fLeft=fLeft, fLeftEval=fLeftEval, dfLeftEval=dfLeftEval)
        params, cParams = [3.15241e-02, 3.42571e-01, -1.41860e-03, -1.21510e-03, 4.31265e-05], [False, False, False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "mean3", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=150, yMin=-5, yMax=50, xMax=150, yTitle = "#mu_{3} (GeV)")
        recoilLibs.doFit(jsIn, jsOut, comp, "mean4", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=150, yMin=-5, yMax=50, xMax=150, yTitle = "#mu_{4} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [1.48366e-01, 8.35984e-01, 4.02649e+00], [False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma1", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=80, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{1} (GeV)")
 
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [1.58964e-01, 8.12765e-01, 6.90134e+00], [False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma2", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=5, fitMax=100, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{2} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [1.06057e-02, 1.10462e+00, 1.33946e+01], [False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma3", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=100, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{2} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [1.15048e-02, 9.60879e-01, 2.26090e+01], [False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma4", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=100, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{4} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Exp(-[1]*x)", [2.78987e-01, 8.26000e-03], [False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm1", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=25, yMin=0, yMax=1, xMax=150, yTitle = "n_{1} (GeV)")
        
        fitF, params, cParams = "[0]", [0.25], [True]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm2", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=50, yMin=0, yMax=1, xMax=150, yTitle = "n_{2} (GeV)")
        
        fitF, params, cParams = "[0]", [0.35], [True]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm3", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=50, yMin=0, yMax=1, xMax=150, yTitle = "n_{3} (GeV)")
        
        with open("%s/results.json" % outDir_param, "w") as outfile: json.dump(jsOut, outfile, indent=4)    
        return        

    outDir_fits_v4 = "%s/fits_v4" % baseDir
    if False:
        fitCfg['mean_cfg'] = [2, 2, 2, 2] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
        fitCfg['sigma_cfg'] = [2, 2, 0, 0]
        fitCfg['norm_cfg'] = [2, 2, 2]
        with open("%s/results.json" % outDir_param_v3) as f: jsIn = json.load(f)
        recoilLibs.doFitMultiGauss(rFile, proc, comp, fitCfg, label, outDir_fits_v4, recoil_qTbins, funcJs=jsIn, qTmax=qTmax, rebin=rebin, xMin=-100, xMax=100)
        return        
        
    outDir_param_v4 = "%s/param_v4" % baseDir
    if False:
        jsOut = {}
        jsOut['nGauss'] = nGauss
        outDir_param = outDir_param_v4
        functions.prepareDir(outDir_param, True)
        with open("%s/results.json" % outDir_fits_v4) as f: jsIn = json.load(f)
        
        t = 20
        fLeft, dfLeft  = "[0] + [1]*x + [2]*x*x + [3]*x*x*x + [4]*x*x*x*x", "[1] + 2*[2]*x + 3*[3]*x*x + 4*[4]*x*x*x"
        fLeftEval, dfLeftEval = fLeft.replace("x", str(t)), dfLeft.replace("x", str(t))
        fitF = "(x<{t})*1*({fLeft}) +0+ (x>{t})*1*(  ({dfLeftEval})*x + ({fLeftEval}-({dfLeftEval})*{t}) )".format(t=t, fLeft=fLeft, fLeftEval=fLeftEval, dfLeftEval=dfLeftEval)
        params, cParams = [-1.49478e-01, 6.37319e-01, -5.64081e-02, 2.87988e-03, -5.58069e-05], [False, False, False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "mean1", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=150, yMin=-5, yMax=50, xMax=150, yTitle = "#mu_{1} (GeV)")
        recoilLibs.doFit(jsIn, jsOut, comp, "mean2", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=150, yMin=-5, yMax=50, xMax=150, yTitle = "#mu_{2} (GeV)")
        
        t = 15
        fLeft, dfLeft  = "[0] + [1]*x + [2]*x*x + [3]*x*x*x + [4]*x*x*x*x", "[1] + 2*[2]*x + 3*[3]*x*x + 4*[4]*x*x*x"
        fLeftEval, dfLeftEval = fLeft.replace("x", str(t)), dfLeft.replace("x", str(t))
        fitF = "(x<{t})*1*({fLeft}) +0+ (x>{t})*1*(  ({dfLeftEval})*x + ({fLeftEval}-({dfLeftEval})*{t}) )".format(t=t, fLeft=fLeft, fLeftEval=fLeftEval, dfLeftEval=dfLeftEval)
        params, cParams = [3.15241e-02, 3.42571e-01, -1.41860e-03, -1.21510e-03, 4.31265e-05], [False, False, False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "mean3", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=150, yMin=-5, yMax=50, xMax=150, yTitle = "#mu_{3} (GeV)")
        recoilLibs.doFit(jsIn, jsOut, comp, "mean4", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=150, yMin=-5, yMax=50, xMax=150, yTitle = "#mu_{4} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [1.48366e-01, 8.35984e-01, 4.02649e+00], [False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma1", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=80, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{1} (GeV)")
 
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [1.58964e-01, 8.12765e-01, 6.90134e+00], [False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma2", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=5, fitMax=100, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{2} (GeV)")
        
        fitF, params, cParams = "[0] + [1]*x", [8.53039e+00, 3.00094e-02], [False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma3", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, cutOffMax=12, fitMin=0, fitMax=100, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{2} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [1.15048e-02, 9.60879e-01, 2.26090e+01], [False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma4", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=100, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{4} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Exp(-[1]*x)", [2.78987e-01, 8.26000e-03], [False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm1", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=25, yMin=0, yMax=1, xMax=150, yTitle = "n_{1} (GeV)")
        
        fitF, params, cParams = "[0]", [0.25], [True]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm2", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=50, yMin=0, yMax=1, xMax=150, yTitle = "n_{2} (GeV)")
        
        fitF, params, cParams = "[0]", [0.35], [True]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm3", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=50, yMin=0, yMax=1, xMax=150, yTitle = "n_{3} (GeV)")
        
        with open("%s/results.json" % outDir_param, "w") as outfile: json.dump(jsOut, outfile, indent=4)    
        return

    outDir_fits_v5 = "%s/fits_v5" % baseDir
    if False:
        fitCfg['mean_cfg'] = [2, 2, 2, 2] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
        fitCfg['sigma_cfg'] = [2, 2, 2, 0]
        fitCfg['norm_cfg'] = [2, 2, 2]
        with open("%s/results.json" % outDir_param_v4) as f: jsIn = json.load(f)
        recoilLibs.doFitMultiGauss(rFile, proc, comp, fitCfg, label, outDir_fits_v5, recoil_qTbins, funcJs=jsIn, qTmax=qTmax, rebin=rebin, xMin=-100, xMax=100)
        return   
        
    outDir_param_v5 = "%s/param_v5" % baseDir
    if True:
        jsOut = {}
        jsOut['nGauss'] = nGauss
        outDir_param = outDir_param_v5
        functions.prepareDir(outDir_param, True)
        with open("%s/results.json" % outDir_fits_v5) as f: jsIn = json.load(f)
        
        t = 20
        fLeft, dfLeft  = "[0] + [1]*x + [2]*x*x + [3]*x*x*x + [4]*x*x*x*x", "[1] + 2*[2]*x + 3*[3]*x*x + 4*[4]*x*x*x"
        fLeftEval, dfLeftEval = fLeft.replace("x", str(t)), dfLeft.replace("x", str(t))
        fitF = "(x<{t})*1*({fLeft}) +0+ (x>{t})*1*(  ({dfLeftEval})*x + ({fLeftEval}-({dfLeftEval})*{t}) )".format(t=t, fLeft=fLeft, fLeftEval=fLeftEval, dfLeftEval=dfLeftEval)
        params, cParams = [-1.49478e-01, 6.37319e-01, -5.64081e-02, 2.87988e-03, -5.58069e-05], [False, False, False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "mean1", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=150, yMin=-5, yMax=50, xMax=150, yTitle = "#mu_{1} (GeV)")
        recoilLibs.doFit(jsIn, jsOut, comp, "mean2", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=150, yMin=-5, yMax=50, xMax=150, yTitle = "#mu_{2} (GeV)")
        
        t = 15
        fLeft, dfLeft  = "[0] + [1]*x + [2]*x*x + [3]*x*x*x + [4]*x*x*x*x", "[1] + 2*[2]*x + 3*[3]*x*x + 4*[4]*x*x*x"
        fLeftEval, dfLeftEval = fLeft.replace("x", str(t)), dfLeft.replace("x", str(t))
        fitF = "(x<{t})*1*({fLeft}) +0+ (x>{t})*1*(  ({dfLeftEval})*x + ({fLeftEval}-({dfLeftEval})*{t}) )".format(t=t, fLeft=fLeft, fLeftEval=fLeftEval, dfLeftEval=dfLeftEval)
        params, cParams = [3.15241e-02, 3.42571e-01, -1.41860e-03, -1.21510e-03, 4.31265e-05], [False, False, False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "mean3", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=150, yMin=-5, yMax=50, xMax=150, yTitle = "#mu_{3} (GeV)")
        recoilLibs.doFit(jsIn, jsOut, comp, "mean4", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=150, yMin=-5, yMax=50, xMax=150, yTitle = "#mu_{4} (GeV)")
        
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [1.48366e-01, 8.35984e-01, 4.02649e+00], [False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma1", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=80, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{1} (GeV)")
 
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [1.58964e-01, 8.12765e-01, 6.90134e+00], [False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma2", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=5, fitMax=100, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{2} (GeV)")
        
        fitF, params, cParams = "[0] + [1]*x", [8.53039e+00, 3.00094e-02], [False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma3", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, cutOffMax=12, fitMin=0, fitMax=100, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{2} (GeV)")
        
        t = 20
        fLeft, dfLeft  = "[0] + [1]*x + [2]*x*x + [3]*x*x*x + [4]*x*x*x*x", "[1] + 2*[2]*x + 3*[3]*x*x + 4*[4]*x*x*x"
        fLeftEval, dfLeftEval = fLeft.replace("x", str(t)), dfLeft.replace("x", str(t))
        fitF = "(x<{t})*1*({fLeft}) +0+ (x>{t})*1*(  ({dfLeftEval})*x + ({fLeftEval}-({dfLeftEval})*{t}) )".format(t=t, fLeft=fLeft, fLeftEval=fLeftEval, dfLeftEval=dfLeftEval)
        params, cParams = [1.21851e+01, -1.13968e-01, 2.68152e-02, -1.47007e-03, 2.74312e-05], [False, False, False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma4", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{4} (GeV)")
        
        
        
        fitF, params, cParams = "[0]*TMath::Exp(-[1]*x)", [2.78987e-01, 8.26000e-03], [False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm1", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=1, xMax=150, yTitle = "n_{1} (GeV)")
        
        fitF, params, cParams = "[0]", [0.25], [False]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm2", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=1, xMax=150, yTitle = "n_{2} (GeV)", doFit=False)
        
        fitF, params, cParams = "[0]", [0.35], [False]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm3", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=1, xMax=150, yTitle = "n_{3} (GeV)", doFit=False)
        
        with open("%s/results.json" % outDir_param, "w") as outfile: json.dump(jsOut, outfile, indent=4)    
        #return   
   
    ########################  
    # Nominal refit
    ########################
    if True:
        outDir_refit = "%s/params_refit" % baseDir
        functions.prepareDir(outDir_refit)
        
        # need to get the original fit config, to not fit twice some entangled params (treat them as RooRealVar alias)
        fitCfg['mean_cfg'] = [0, -1, 0, -3] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
        fitCfg['sigma_cfg'] = [2, 2, 2, 2]
        fitCfg['norm_cfg'] = [2, 2, 2]
        with open("%s/results.json" % outDir_param_v5) as f: jsIn = json.load(f)
        jsOut = recoilLibs.combinedFit(rFile, proc, comp, fitCfg, jsIn, recoil_qTbins, rebin=rebin, qTmax=100)
        with open("%s/results_refit.json" % outDir_refit, "w") as outfile: json.dump(jsOut, outfile, indent=4)
        #return
        

        fitCfg['mean_cfg'] = [2, 2, 2, 2] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
        fitCfg['sigma_cfg'] = [2, 2, 2, 2]
        fitCfg['norm_cfg'] = [2, 2, 2]
        
        with open("%s/results_refit.json" % outDir_refit) as f: jsIn = json.load(f)
        recoilLibs.doPlot(comp, "norm1", jsIn, outDir_refit, recoil_qTbins, label, yMin=0, yMax=1, xMax=150, yTitle = "n_{1} (GeV)")
        recoilLibs.doPlot(comp, "norm2", jsIn, outDir_refit, recoil_qTbins, label, yMin=0, yMax=1, xMax=150, yTitle = "n_{2} (GeV)")
        recoilLibs.doPlot(comp, "norm3", jsIn, outDir_refit, recoil_qTbins, label, yMin=0, yMax=1, xMax=150, yTitle = "n_{3} (GeV)")
        recoilLibs.doPlot(comp, "sigma1", jsIn, outDir_refit, recoil_qTbins, label, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{1} (GeV)")
        recoilLibs.doPlot(comp, "sigma2", jsIn, outDir_refit, recoil_qTbins, label, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{2} (GeV)")
        recoilLibs.doPlot(comp, "sigma3", jsIn, outDir_refit, recoil_qTbins, label, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{3} (GeV)")
        recoilLibs.doPlot(comp, "sigma4", jsIn, outDir_refit, recoil_qTbins, label, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{4} (GeV)")
        recoilLibs.doPlot(comp, "mean1", jsIn, outDir_refit, recoil_qTbins, label, yMin=-5, yMax=50, xMax=150, yTitle = "#mu_{1} (GeV)")
        recoilLibs.doPlot(comp, "mean2", jsIn, outDir_refit, recoil_qTbins, label, yMin=-5, yMax=50, xMax=150, yTitle = "#mu_{2} (GeV)")
        recoilLibs.doPlot(comp, "mean3", jsIn, outDir_refit, recoil_qTbins, label, yMin=-5, yMax=50, xMax=150, yTitle = "#mu_{3} (GeV)")
        recoilLibs.doPlot(comp, "mean4", jsIn, outDir_refit, recoil_qTbins, label, yMin=-5, yMax=50, xMax=150, yTitle = "#mu_{4} (GeV)")

        outDir_refit_fits = "%s/fits_refit" % baseDir
        recoilLibs.doFitMultiGauss(rFile, proc, comp, fitCfg, label, outDir_refit_fits, recoil_qTbins, funcJs=jsIn, qTmax=qTmax, rebin=rebin, xMin=-100, xMax=100)
        ROOT.gROOT.Reset()

def do_dymumu_para_RawPFMET():

    proc = "DYmumu"
    tag = "dymumu_para"
    comp = "para"
    baseDir = "%s/%s" % (outDir, tag)
    name = "DY #rightarrow #mu^{+}#mu^{#minus}"
    label = "%s, %s" % (name, met)
    functions.prepareDir(baseDir, False)
    rebin = 1


    outDir_fits_v0 = "%s/fits_v0" % baseDir
    fitCfg = {}   
    
    fitCfg['mean'], fitCfg['mean_cfg'] = [0, 0, 0, 0, 0], [0, 0, 0, 0, 0] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
    fitCfg['sigma'], fitCfg['sigma_cfg'] = [20, 5, 8, 10, 15], [0, 0, 0, 0, 0]
    fitCfg['norm'], fitCfg['norm_cfg'] = [0.05, 0.20, 0.25, 0.25], [0, 0, 0, 0]
    
    fitCfg['mean'], fitCfg['mean_cfg'] = [0, 0, 0, 0, 0], [0, 0, 0, 0, 0] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
    fitCfg['sigma'], fitCfg['sigma_cfg'] = [4.5, 7.5, 10, 15, 30], [0, 0, 0, 0, 0]
    fitCfg['norm'], fitCfg['norm_cfg'] = [0.2, 0.35, 0.15, 0.15], [0, 0, 0, 0]
    
    fitCfg['mean'], fitCfg['mean_cfg'] = [0, 0, 0, 0], [0, -1, 0, -3] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
    fitCfg['sigma'], fitCfg['sigma_cfg'] = [4, 7, 10, 15], [0, 0, 0, 0]
    fitCfg['norm'], fitCfg['norm_cfg'] = [0.25, 0.25, 0.35], [1, 0, 0]

    recoil_qTbins = list(functions.drange(0, 50, 0.5)) +  list(range(50, 100, 1)) + list(range(100, 150, 2)) + [150]
    
    nGauss = len(fitCfg['mean'])
    
    if False: 
        recoilLibs.doFitMultiGauss(rFile, proc, comp, fitCfg, label, outDir_fits_v0, recoil_qTbins, qTmax=qTmax, propagate=True, rebin=rebin, xMin=-100, xMax=100)
        return
        
    outDir_param_v0 = "%s/param_v0" % baseDir
    if False:
        jsOut = {}
        jsOut['nGauss'] = nGauss
        outDir_param = outDir_param_v0
        functions.prepareDir(outDir_param, True)
        with open("%s/results.json" % outDir_fits_v0) as f: jsIn = json.load(f)
        
        t = 15
        fLeft, dfLeft  = "[0] + [1]*x + [2]*x*x + [3]*x*x*x + [4]*x*x*x*x", "[1] + 2*[2]*x + 3*[3]*x*x + 4*[4]*x*x*x"
        fLeftEval, dfLeftEval = fLeft.replace("x", str(t)), dfLeft.replace("x", str(t))
        fitF = "(x<{t})*1*({fLeft}) +0+ (x>{t})*1*(  ({dfLeftEval})*x + ({fLeftEval}-({dfLeftEval})*{t}) )".format(t=t, fLeft=fLeft, fLeftEval=fLeftEval, dfLeftEval=dfLeftEval)
        params, cParams = [-3.12952e-01, 7.52263e-01, -8.53549e-02, 5.66169e-03, -1.44577e-04], [False, False, False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "mean1", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=100, yMin=-5, yMax=50, xMax=150, yTitle = "#mu_{1} (GeV)")
        recoilLibs.doFit(jsIn, jsOut, comp, "mean2", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=100, yMin=-5, yMax=50, xMax=150, yTitle = "#mu_{2} (GeV)")
        
        t = 15
        fLeft, dfLeft  = "[0] + [1]*x + [2]*x*x + [3]*x*x*x + [4]*x*x*x*x", "[1] + 2*[2]*x + 3*[3]*x*x + 4*[4]*x*x*x"
        fLeftEval, dfLeftEval = fLeft.replace("x", str(t)), dfLeft.replace("x", str(t))
        fitF = "(x<{t})*1*({fLeft}) +0+ (x>{t})*1*(  ({dfLeftEval})*x + ({fLeftEval}-({dfLeftEval})*{t}) )".format(t=t, fLeft=fLeft, fLeftEval=fLeftEval, dfLeftEval=dfLeftEval)
        params, cParams = [6.00374e-02, 3.61545e-01, -1.21548e-02, 9.07611e-05, 0], [False, False, False, False, True]
        recoilLibs.doFit(jsIn, jsOut, comp, "mean3", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=100, yMin=-5, yMax=50, xMax=150, yTitle = "#mu_{3} (GeV)")
        recoilLibs.doFit(jsIn, jsOut, comp, "mean4", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=100, yMin=-5, yMax=50, xMax=150, yTitle = "#mu_{4} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [7.29092e-02, 1.02290e+00, 3.89730e+00], [False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma1", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=30, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{1} (GeV)")
 
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [1.52752e+01, 1.24731e-02, -6.94362e+00], [False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma2", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=100, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{2} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [1.06057e-02, 1.10462e+00, 1.33946e+01], [False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma3", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=100, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{2} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [1.15048e-02, 9.60879e-01, 2.26090e+01], [False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma4", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=100, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{4} (GeV)")
        
        fitF, params, cParams = "[0]", [0.25], [True]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm1", fitF, params, outDir_param, recoil_qTbins, label, doFit=False, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=1, xMax=150, yTitle = "n_{1} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Exp(-[1]*x)", [2.99932e-01, 2.26921e-03], [False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm2", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=25, yMin=0, yMax=1, xMax=150, yTitle = "n_{2} (GeV)")
        
        fitF, params, cParams = "[0]", [0.35], [False]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm3", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=50, yMin=0, yMax=1, xMax=150, yTitle = "n_{3} (GeV)")
        

        with open("%s/results.json" % outDir_param, "w") as outfile: json.dump(jsOut, outfile, indent=4)
        return
        
    outDir_fits_v1 = "%s/fits_v1" % baseDir
    if False:
        fitCfg['mean_cfg'] = [2, 2, 2, 2] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
        fitCfg['sigma_cfg'] = [0, 0, 0, 0]
        fitCfg['norm_cfg'] = [2, 0, 0]
        with open("%s/results.json" % outDir_param_v0) as f: jsIn = json.load(f)
        recoilLibs.doFitMultiGauss(rFile, proc, comp, fitCfg, label, outDir_fits_v1, recoil_qTbins, funcJs=jsIn, qTmax=qTmax, propagate=True, rebin=rebin, xMin=-100, xMax=100)
        return
        
    outDir_param_v1 = "%s/param_v1" % baseDir
    if False:
        jsOut = {}
        jsOut['nGauss'] = nGauss
        outDir_param = outDir_param_v1
        functions.prepareDir(outDir_param, True)
        with open("%s/results.json" % outDir_fits_v1) as f: jsIn = json.load(f)
        
        t = 15
        fLeft, dfLeft  = "[0] + [1]*x + [2]*x*x + [3]*x*x*x + [4]*x*x*x*x", "[1] + 2*[2]*x + 3*[3]*x*x + 4*[4]*x*x*x"
        fLeftEval, dfLeftEval = fLeft.replace("x", str(t)), dfLeft.replace("x", str(t))
        fitF = "(x<{t})*1*({fLeft}) +0+ (x>{t})*1*(  ({dfLeftEval})*x + ({fLeftEval}-({dfLeftEval})*{t}) )".format(t=t, fLeft=fLeft, fLeftEval=fLeftEval, dfLeftEval=dfLeftEval)
        params, cParams = [-3.12952e-01, 7.52263e-01, -8.53549e-02, 5.66169e-03, -1.44577e-04], [False, False, False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "mean1", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=100, yMin=-5, yMax=50, xMax=150, yTitle = "#mu_{1} (GeV)")
        recoilLibs.doFit(jsIn, jsOut, comp, "mean2", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=100, yMin=-5, yMax=50, xMax=150, yTitle = "#mu_{2} (GeV)")
        
        t = 15
        fLeft, dfLeft  = "[0] + [1]*x + [2]*x*x + [3]*x*x*x + [4]*x*x*x*x", "[1] + 2*[2]*x + 3*[3]*x*x + 4*[4]*x*x*x"
        fLeftEval, dfLeftEval = fLeft.replace("x", str(t)), dfLeft.replace("x", str(t))
        fitF = "(x<{t})*1*({fLeft}) +0+ (x>{t})*1*(  ({dfLeftEval})*x + ({fLeftEval}-({dfLeftEval})*{t}) )".format(t=t, fLeft=fLeft, fLeftEval=fLeftEval, dfLeftEval=dfLeftEval)
        params, cParams = [6.00374e-02, 3.61545e-01, -1.21548e-02, 9.07611e-05, 0], [False, False, False, False, True]
        recoilLibs.doFit(jsIn, jsOut, comp, "mean3", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=100, yMin=-5, yMax=50, xMax=150, yTitle = "#mu_{3} (GeV)")
        recoilLibs.doFit(jsIn, jsOut, comp, "mean4", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=100, yMin=-5, yMax=50, xMax=150, yTitle = "#mu_{4} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [7.29092e-02, 1.02290e+00, 3.89730e+00], [False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma1", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=30, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{1} (GeV)")
 
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [1.52752e+01, 1.24731e-02, -6.94362e+00], [False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma2", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=100, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{2} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [1.06057e-02, 1.10462e+00, 1.33946e+01], [False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma3", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=100, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{2} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [1.15048e-02, 9.60879e-01, 2.26090e+01], [False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma4", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=100, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{4} (GeV)")
        
        fitF, params, cParams = "[0]", [0.25], [True]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm1", fitF, params, outDir_param, recoil_qTbins, label, doFit=False, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=1, xMax=150, yTitle = "n_{1} (GeV)")
        
        fitF, params, cParams = "[0] + [1]*x", [3.03159e-01, -6.80847e-04], [False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm2", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=100, yMin=0, yMax=1, xMax=150, yTitle = "n_{2} (GeV)")
        
        fitF, params, cParams = "[0] + [1]*x", [4.42663e-01, 2.81596e-04], [False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm3", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=100, yMin=0, yMax=1, xMax=150, yTitle = "n_{3} (GeV)")
        
        with open("%s/results.json" % outDir_param, "w") as outfile: json.dump(jsOut, outfile, indent=4)    
        return
        
    outDir_fits_v2 = "%s/fits_v2" % baseDir
    if False:
        fitCfg['mean_cfg'] = [2, 2, 2, 2] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
        fitCfg['sigma_cfg'] = [0, 0, 0, 0]
        fitCfg['norm_cfg'] = [2, 2, 2]
        with open("%s/results.json" % outDir_param_v1) as f: jsIn = json.load(f)
        recoilLibs.doFitMultiGauss(rFile, proc, comp, fitCfg, label, outDir_fits_v2, recoil_qTbins, funcJs=jsIn, propagate=True, qTmax=qTmax, rebin=rebin, xMin=-100, xMax=100)
        return
        
    outDir_param_v2 = "%s/param_v2" % baseDir
    if False:
        jsOut = {}
        jsOut['nGauss'] = nGauss
        outDir_param = outDir_param_v2
        functions.prepareDir(outDir_param_v2, True)
        with open("%s/results.json" % outDir_fits_v2) as f: jsIn = json.load(f)
        
        t = 15
        fLeft, dfLeft  = "[0] + [1]*x + [2]*x*x + [3]*x*x*x + [4]*x*x*x*x", "[1] + 2*[2]*x + 3*[3]*x*x + 4*[4]*x*x*x"
        fLeftEval, dfLeftEval = fLeft.replace("x", str(t)), dfLeft.replace("x", str(t))
        fitF = "(x<{t})*1*({fLeft}) +0+ (x>{t})*1*(  ({dfLeftEval})*x + ({fLeftEval}-({dfLeftEval})*{t}) )".format(t=t, fLeft=fLeft, fLeftEval=fLeftEval, dfLeftEval=dfLeftEval)
        params, cParams = [-3.12952e-01, 7.52263e-01, -8.53549e-02, 5.66169e-03, -1.44577e-04], [False, False, False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "mean1", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=100, yMin=-5, yMax=50, xMax=150, yTitle = "#mu_{1} (GeV)")
        recoilLibs.doFit(jsIn, jsOut, comp, "mean2", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=100, yMin=-5, yMax=50, xMax=150, yTitle = "#mu_{2} (GeV)")
        
        t = 15
        fLeft, dfLeft  = "[0] + [1]*x + [2]*x*x + [3]*x*x*x + [4]*x*x*x*x", "[1] + 2*[2]*x + 3*[3]*x*x + 4*[4]*x*x*x"
        fLeftEval, dfLeftEval = fLeft.replace("x", str(t)), dfLeft.replace("x", str(t))
        fitF = "(x<{t})*1*({fLeft}) +0+ (x>{t})*1*(  ({dfLeftEval})*x + ({fLeftEval}-({dfLeftEval})*{t}) )".format(t=t, fLeft=fLeft, fLeftEval=fLeftEval, dfLeftEval=dfLeftEval)
        params, cParams = [6.00374e-02, 3.61545e-01, -1.21548e-02, 9.07611e-05, 0], [False, False, False, False, True]
        recoilLibs.doFit(jsIn, jsOut, comp, "mean3", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=100, yMin=-5, yMax=50, xMax=150, yTitle = "#mu_{3} (GeV)")
        recoilLibs.doFit(jsIn, jsOut, comp, "mean4", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=100, yMin=-5, yMax=50, xMax=150, yTitle = "#mu_{4} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [1.34315e-01, 9.02953e-01, 4.04100e+00], [False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma1", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=30, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{1} (GeV)")
 
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [1.52752e+01, 1.24731e-02, -6.94362e+00], [False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma2", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=100, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{2} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [1.06057e-02, 1.10462e+00, 1.33946e+01], [False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma3", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=100, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{2} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [1.15048e-02, 9.60879e-01, 2.26090e+01], [False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma4", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=100, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{4} (GeV)")
        
        fitF, params, cParams = "[0]", [0.25], [True]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm1", fitF, params, outDir_param, recoil_qTbins, label, doFit=False, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=1, xMax=150, yTitle = "n_{1} (GeV)")
        
        fitF, params, cParams = "[0] + [1]*x", [3.03159e-01, -6.80847e-04], [False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm2", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=100, yMin=0, yMax=1, xMax=150, yTitle = "n_{2} (GeV)")
        
        fitF, params, cParams = "[0] + [1]*x", [4.42663e-01, 2.81596e-04], [False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm3", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=100, yMin=0, yMax=1, xMax=150, yTitle = "n_{3} (GeV)")
        
        with open("%s/results.json" % outDir_param, "w") as outfile: json.dump(jsOut, outfile, indent=4)    
        return
        
    outDir_fits_v3 = "%s/fits_v3" % baseDir
    if False:
        fitCfg['mean_cfg'] = [2, 2, 2, 2] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
        fitCfg['sigma_cfg'] = [2, 0, 0, 0]
        fitCfg['norm_cfg'] = [2, 2, 2]
        with open("%s/results.json" % outDir_param_v2) as f: jsIn = json.load(f)
        recoilLibs.doFitMultiGauss(rFile, proc, comp, fitCfg, label, outDir_fits_v3, recoil_qTbins, funcJs=jsIn, qTmax=qTmax, rebin=rebin, xMin=-100, xMax=100)
        return
    
    outDir_param_v3 = "%s/param_v3" % baseDir
    if False:
        jsOut = {}
        jsOut['nGauss'] = nGauss
        outDir_param = outDir_param_v3
        functions.prepareDir(outDir_param, True)
        with open("%s/results.json" % outDir_fits_v3) as f: jsIn = json.load(f)
        
        t = 15
        fLeft, dfLeft  = "[0] + [1]*x + [2]*x*x + [3]*x*x*x + [4]*x*x*x*x", "[1] + 2*[2]*x + 3*[3]*x*x + 4*[4]*x*x*x"
        fLeftEval, dfLeftEval = fLeft.replace("x", str(t)), dfLeft.replace("x", str(t))
        fitF = "(x<{t})*1*({fLeft}) +0+ (x>{t})*1*(  ({dfLeftEval})*x + ({fLeftEval}-({dfLeftEval})*{t}) )".format(t=t, fLeft=fLeft, fLeftEval=fLeftEval, dfLeftEval=dfLeftEval)
        params, cParams = [-3.12952e-01, 7.52263e-01, -8.53549e-02, 5.66169e-03, -1.44577e-04], [False, False, False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "mean1", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=100, yMin=-5, yMax=50, xMax=150, yTitle = "#mu_{1} (GeV)")
        recoilLibs.doFit(jsIn, jsOut, comp, "mean2", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=100, yMin=-5, yMax=50, xMax=150, yTitle = "#mu_{2} (GeV)")
        
        t = 15
        fLeft, dfLeft  = "[0] + [1]*x + [2]*x*x + [3]*x*x*x + [4]*x*x*x*x", "[1] + 2*[2]*x + 3*[3]*x*x + 4*[4]*x*x*x"
        fLeftEval, dfLeftEval = fLeft.replace("x", str(t)), dfLeft.replace("x", str(t))
        fitF = "(x<{t})*1*({fLeft}) +0+ (x>{t})*1*(  ({dfLeftEval})*x + ({fLeftEval}-({dfLeftEval})*{t}) )".format(t=t, fLeft=fLeft, fLeftEval=fLeftEval, dfLeftEval=dfLeftEval)
        params, cParams = [6.00374e-02, 3.61545e-01, -1.21548e-02, 9.07611e-05, 0], [False, False, False, False, True]
        recoilLibs.doFit(jsIn, jsOut, comp, "mean3", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=100, yMin=-5, yMax=50, xMax=150, yTitle = "#mu_{3} (GeV)")
        recoilLibs.doFit(jsIn, jsOut, comp, "mean4", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=100, yMin=-5, yMax=50, xMax=150, yTitle = "#mu_{4} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [1.34315e-01, 9.02953e-01, 4.04100e+00], [False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma1", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=30, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{1} (GeV)")
 
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [3.25330e-01, 5.61078e-01, 6.13897e+00], [False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma2", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=2, fitMax=100, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{2} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [1.06057e-02, 1.10462e+00, 1.33946e+01], [False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma3", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=100, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{2} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [1.15048e-02, 9.60879e-01, 2.26090e+01], [False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma4", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=100, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{4} (GeV)")
        
        fitF, params, cParams = "[0]", [0.25], [True]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm1", fitF, params, outDir_param, recoil_qTbins, label, doFit=False, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=1, xMax=150, yTitle = "n_{1} (GeV)")
        
        fitF, params, cParams = "[0] + [1]*x", [3.03159e-01, -6.80847e-04], [False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm2", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=100, yMin=0, yMax=1, xMax=150, yTitle = "n_{2} (GeV)")
        
        fitF, params, cParams = "[0] + [1]*x", [4.42663e-01, 2.81596e-04], [False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm3", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=100, yMin=0, yMax=1, xMax=150, yTitle = "n_{3} (GeV)")
        
        with open("%s/results.json" % outDir_param, "w") as outfile: json.dump(jsOut, outfile, indent=4)    
        return        

    outDir_fits_v4 = "%s/fits_v4" % baseDir
    if False:
        fitCfg['mean_cfg'] = [2, 2, 2, 2] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
        fitCfg['sigma_cfg'] = [2, 2, 0, 0]
        fitCfg['norm_cfg'] = [2, 2, 2]
        with open("%s/results.json" % outDir_param_v3) as f: jsIn = json.load(f)
        recoilLibs.doFitMultiGauss(rFile, proc, comp, fitCfg, label, outDir_fits_v4, recoil_qTbins, funcJs=jsIn, qTmax=qTmax, rebin=rebin, xMin=-100, xMax=100)
        return        
        
    outDir_param_v4 = "%s/param_v4" % baseDir
    if False:
        jsOut = {}
        jsOut['nGauss'] = nGauss
        outDir_param = outDir_param_v4
        functions.prepareDir(outDir_param, True)
        with open("%s/results.json" % outDir_fits_v4) as f: jsIn = json.load(f)
        
        t = 15
        fLeft, dfLeft  = "[0] + [1]*x + [2]*x*x + [3]*x*x*x + [4]*x*x*x*x", "[1] + 2*[2]*x + 3*[3]*x*x + 4*[4]*x*x*x"
        fLeftEval, dfLeftEval = fLeft.replace("x", str(t)), dfLeft.replace("x", str(t))
        fitF = "(x<{t})*1*({fLeft}) +0+ (x>{t})*1*(  ({dfLeftEval})*x + ({fLeftEval}-({dfLeftEval})*{t}) )".format(t=t, fLeft=fLeft, fLeftEval=fLeftEval, dfLeftEval=dfLeftEval)
        params, cParams = [-3.12952e-01, 7.52263e-01, -8.53549e-02, 5.66169e-03, -1.44577e-04], [False, False, False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "mean1", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=100, yMin=-5, yMax=50, xMax=150, yTitle = "#mu_{1} (GeV)")
        recoilLibs.doFit(jsIn, jsOut, comp, "mean2", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=100, yMin=-5, yMax=50, xMax=150, yTitle = "#mu_{2} (GeV)")
        
        t = 15
        fLeft, dfLeft  = "[0] + [1]*x + [2]*x*x + [3]*x*x*x + [4]*x*x*x*x", "[1] + 2*[2]*x + 3*[3]*x*x + 4*[4]*x*x*x"
        fLeftEval, dfLeftEval = fLeft.replace("x", str(t)), dfLeft.replace("x", str(t))
        fitF = "(x<{t})*1*({fLeft}) +0+ (x>{t})*1*(  ({dfLeftEval})*x + ({fLeftEval}-({dfLeftEval})*{t}) )".format(t=t, fLeft=fLeft, fLeftEval=fLeftEval, dfLeftEval=dfLeftEval)
        params, cParams = [6.00374e-02, 3.61545e-01, -1.21548e-02, 9.07611e-05, 0], [False, False, False, False, True]
        recoilLibs.doFit(jsIn, jsOut, comp, "mean3", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=100, yMin=-5, yMax=50, xMax=150, yTitle = "#mu_{3} (GeV)")
        recoilLibs.doFit(jsIn, jsOut, comp, "mean4", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=100, yMin=-5, yMax=50, xMax=150, yTitle = "#mu_{4} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [1.34315e-01, 9.02953e-01, 4.04100e+00], [False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma1", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=30, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{1} (GeV)")
 
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [3.25330e-01, 5.61078e-01, 6.13897e+00], [False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma2", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=2, fitMax=100, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{2} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [2.86771e-01, 6.59622e-01, 9.06788e+00], [False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma3", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=4, fitMax=150, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{2} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [1.15048e-02, 9.60879e-01, 2.26090e+01], [False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma4", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=100, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{4} (GeV)")
        
        fitF, params, cParams = "[0]", [0.25], [True]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm1", fitF, params, outDir_param, recoil_qTbins, label, doFit=False, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=1, xMax=150, yTitle = "n_{1} (GeV)")
        
        fitF, params, cParams = "[0] + [1]*x", [3.03159e-01, -6.80847e-04], [False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm2", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=100, yMin=0, yMax=1, xMax=150, yTitle = "n_{2} (GeV)")
        
        fitF, params, cParams = "[0] + [1]*x", [4.42663e-01, 2.81596e-04], [False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm3", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=100, yMin=0, yMax=1, xMax=150, yTitle = "n_{3} (GeV)")
        
        with open("%s/results.json" % outDir_param, "w") as outfile: json.dump(jsOut, outfile, indent=4)    
        return

    outDir_fits_v5 = "%s/fits_v5" % baseDir
    if False:
        fitCfg['mean_cfg'] = [2, 2, 2, 2] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
        fitCfg['sigma_cfg'] = [2, 2, 2, 0]
        fitCfg['norm_cfg'] = [2, 2, 2]
        with open("%s/results.json" % outDir_param_v4) as f: jsIn = json.load(f)
        recoilLibs.doFitMultiGauss(rFile, proc, comp, fitCfg, label, outDir_fits_v5, recoil_qTbins, funcJs=jsIn, qTmax=qTmax, rebin=rebin, xMin=-100, xMax=100)
        return   
        
    outDir_param_v5 = "%s/param_v5" % baseDir
    if True:
        jsOut = {}
        jsOut['nGauss'] = nGauss
        outDir_param = outDir_param_v5
        functions.prepareDir(outDir_param, True)
        with open("%s/results.json" % outDir_fits_v5) as f: jsIn = json.load(f)
        
        t = 15
        fLeft, dfLeft  = "[0] + [1]*x + [2]*x*x + [3]*x*x*x + [4]*x*x*x*x", "[1] + 2*[2]*x + 3*[3]*x*x + 4*[4]*x*x*x"
        fLeftEval, dfLeftEval = fLeft.replace("x", str(t)), dfLeft.replace("x", str(t))
        fitF = "(x<{t})*1*({fLeft}) +0+ (x>{t})*1*(  ({dfLeftEval})*x + ({fLeftEval}-({dfLeftEval})*{t}) )".format(t=t, fLeft=fLeft, fLeftEval=fLeftEval, dfLeftEval=dfLeftEval)
        params, cParams = [-3.12952e-01, 7.52263e-01, -8.53549e-02, 5.66169e-03, -1.44577e-04], [False, False, False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "mean1", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=100, yMin=-5, yMax=50, xMax=150, yTitle = "#mu_{1} (GeV)")
        recoilLibs.doFit(jsIn, jsOut, comp, "mean2", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=100, yMin=-5, yMax=50, xMax=150, yTitle = "#mu_{2} (GeV)")
        
        t = 15
        fLeft, dfLeft  = "[0] + [1]*x + [2]*x*x + [3]*x*x*x + [4]*x*x*x*x", "[1] + 2*[2]*x + 3*[3]*x*x + 4*[4]*x*x*x"
        fLeftEval, dfLeftEval = fLeft.replace("x", str(t)), dfLeft.replace("x", str(t))
        fitF = "(x<{t})*1*({fLeft}) +0+ (x>{t})*1*(  ({dfLeftEval})*x + ({fLeftEval}-({dfLeftEval})*{t}) )".format(t=t, fLeft=fLeft, fLeftEval=fLeftEval, dfLeftEval=dfLeftEval)
        params, cParams = [6.00374e-02, 3.61545e-01, -1.21548e-02, 9.07611e-05, 0], [False, False, False, False, True]
        recoilLibs.doFit(jsIn, jsOut, comp, "mean3", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=100, yMin=-5, yMax=50, xMax=150, yTitle = "#mu_{3} (GeV)")
        recoilLibs.doFit(jsIn, jsOut, comp, "mean4", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=100, yMin=-5, yMax=50, xMax=150, yTitle = "#mu_{4} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [1.34315e-01, 9.02953e-01, 4.04100e+00], [False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma1", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=30, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{1} (GeV)")
 
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [3.25330e-01, 5.61078e-01, 6.13897e+00], [False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma2", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=2, fitMax=100, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{2} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [2.86771e-01, 6.59622e-01, 9.06788e+00], [False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma3", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=4, fitMax=150, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{2} (GeV)")
        
        #fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [1.15048e-02, 9.60879e-01, 2.26090e+01], [False, False, False]
        t = 30
        fLeft, dfLeft  = "[0] + [1]*x + [2]*x*x + [3]*x*x*x + [4]*x*x*x*x", "[1] + 2*[2]*x + 3*[3]*x*x + 4*[4]*x*x*x"
        fLeftEval, dfLeftEval = fLeft.replace("x", str(t)), dfLeft.replace("x", str(t))
        fitF = "(x<{t})*1*({fLeft}) +0+ (x>{t})*1*(  ({dfLeftEval})*x + ({fLeftEval}-({dfLeftEval})*{t}) )".format(t=t, fLeft=fLeft, fLeftEval=fLeftEval, dfLeftEval=dfLeftEval)
        params, cParams = [2.02182e+01, -2.42581e-01, -7.30371e-03, 1.01107e-03, -1.79551e-05], [False, False, False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma4", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=100, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{4} (GeV)")
        
        fitF, params, cParams = "[0]", [0.25], [True]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm1", fitF, params, outDir_param, recoil_qTbins, label, doFit=False, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=1, xMax=150, yTitle = "n_{1} (GeV)")
        
        fitF, params, cParams = "[0] + [1]*x", [3.03159e-01, -6.80847e-04], [True, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm2", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=100, yMin=0, yMax=1, xMax=150, yTitle = "n_{2} (GeV)")
        
        fitF, params, cParams = "[0] + [1]*x", [4.42663e-01, 2.81596e-04], [True, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm3", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=100, yMin=0, yMax=1, xMax=150, yTitle = "n_{3} (GeV)")
        
        with open("%s/results.json" % outDir_param, "w") as outfile: json.dump(jsOut, outfile, indent=4)    
        #return   
   
    ########################  
    # Nominal refit
    ########################
    if True:
        outDir_refit = "%s/params_refit" % baseDir
        functions.prepareDir(outDir_refit)
        
        # need to get the original fit config, to not fit twice some entangled params (treat them as RooRealVar alias)
        fitCfg['mean_cfg'] = [0, -1, 0, -3] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
        fitCfg['sigma_cfg'] = [2, 2, 2, 2]
        fitCfg['norm_cfg'] = [2, 2, 2]
        with open("%s/results.json" % outDir_param_v5) as f: jsIn = json.load(f)
        jsOut = recoilLibs.combinedFit(rFile, proc, comp, fitCfg, jsIn, recoil_qTbins, rebin=rebin, qTmax=150)
        with open("%s/results_refit.json" % outDir_refit, "w") as outfile: json.dump(jsOut, outfile, indent=4)
        #return
        

        fitCfg['mean_cfg'] = [2, 2, 2, 2] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
        fitCfg['sigma_cfg'] = [2, 2, 2, 2]
        fitCfg['norm_cfg'] = [2, 2, 2]
        
        with open("%s/results_refit.json" % outDir_refit) as f: jsIn = json.load(f)
        recoilLibs.doPlot(comp, "norm1", jsIn, outDir_refit, recoil_qTbins, label, yMin=0, yMax=1, xMax=150, yTitle = "n_{1} (GeV)")
        recoilLibs.doPlot(comp, "norm2", jsIn, outDir_refit, recoil_qTbins, label, yMin=0, yMax=1, xMax=150, yTitle = "n_{2} (GeV)")
        recoilLibs.doPlot(comp, "norm3", jsIn, outDir_refit, recoil_qTbins, label, yMin=0, yMax=1, xMax=150, yTitle = "n_{3} (GeV)")
        recoilLibs.doPlot(comp, "sigma1", jsIn, outDir_refit, recoil_qTbins, label, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{1} (GeV)")
        recoilLibs.doPlot(comp, "sigma2", jsIn, outDir_refit, recoil_qTbins, label, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{2} (GeV)")
        recoilLibs.doPlot(comp, "sigma3", jsIn, outDir_refit, recoil_qTbins, label, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{3} (GeV)")
        recoilLibs.doPlot(comp, "sigma4", jsIn, outDir_refit, recoil_qTbins, label, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{4} (GeV)")
        recoilLibs.doPlot(comp, "mean1", jsIn, outDir_refit, recoil_qTbins, label, yMin=-5, yMax=50, xMax=150, yTitle = "#mu_{1} (GeV)")
        recoilLibs.doPlot(comp, "mean2", jsIn, outDir_refit, recoil_qTbins, label, yMin=-5, yMax=50, xMax=150, yTitle = "#mu_{2} (GeV)")
        recoilLibs.doPlot(comp, "mean3", jsIn, outDir_refit, recoil_qTbins, label, yMin=-5, yMax=50, xMax=150, yTitle = "#mu_{3} (GeV)")
        recoilLibs.doPlot(comp, "mean4", jsIn, outDir_refit, recoil_qTbins, label, yMin=-5, yMax=50, xMax=150, yTitle = "#mu_{4} (GeV)")

        outDir_refit_fits = "%s/fits_refit" % baseDir
        recoilLibs.doFitMultiGauss(rFile, proc, comp, fitCfg, label, outDir_refit_fits, recoil_qTbins, funcJs=jsIn, qTmax=qTmax, rebin=rebin, xMin=-100, xMax=100)
        ROOT.gROOT.Reset()
        
        
def do_singlemuon_perp_RawPFMET():

    proc = "SingleMuon"
    tag = "singlemuon_perp"
    comp = "perp"
    baseDir = "%s/%s" % (outDir, tag)
    name = "SingleMuon #rightarrow #mu^{+}#mu^{#minus}"
    label = "%s, %s" % (name, met)
    functions.prepareDir(baseDir, False)
    rebin = 1

    outDir_fits_v0 = "%s/fits_v0" % baseDir
    fitCfg = {}   
    fitCfg['mean'], fitCfg['mean_cfg'] = [0, 0, 0, 0], [1, 1, 1, 1] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
    fitCfg['sigma'], fitCfg['sigma_cfg'] = [15, 5, 8, 10], [0, 0, 0, 0]
    fitCfg['norm'], fitCfg['norm_cfg'] = [0.05, 0.35, 0.25], [1, 0, 0]
        
    nGauss = len(fitCfg['mean'])
    
    bkgCfg = {}
    bkgCfg['TTbar'] = { "filename": "/eos/user/j/jaeyserm/www/wmass/lowPU/recoilCorrection/recoil_mumu_RawPFMET/ttbar_perp/fits_v0/results.json", "norm": 1.00, "float": False, "subtract": False }
    bkgCfg['EWK'] = { "filename": "/eos/user/j/jaeyserm/www/wmass/lowPU/recoilCorrection/recoil_mumu_RawPFMET/ewk_perp/fits_v0/results.json", "norm": 1.00, "float": False, "subtract": False }
    bkgCfg['ZZ'] = { "filename": "/eos/user/j/jaeyserm/www/wmass/lowPU/recoilCorrection/recoil_mumu_RawPFMET/zz_perp/fits_v0/results.json", "norm": 1.00, "float": False, "subtract": False }
    

    if False: 
        recoilLibs.doFitMultiGauss(rFile, proc, comp, fitCfg, label, outDir_fits_v0, recoil_qTbins, bkgCfg=bkgCfg, qTmax=qTmax, propagate=False, rebin=rebin, xMin=-100, xMax=100)
        return
        
    outDir_param_v0 = "%s/param_v0" % baseDir
    if False:
        jsOut = {}
        jsOut['nGauss'] = nGauss
        outDir_param = outDir_param_v0
        functions.prepareDir(outDir_param, True)
        with open("%s/results.json" % outDir_fits_v0) as f: jsIn = json.load(f)
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [6.34447e+00, 9.71985e-02, 7.13032e+00], [True, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma1", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=70, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{1} (GeV)")
 
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [2.72251e+00, 1.52312e-01, 2.15765e+00], [True, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma2", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, cutOffMin=5, fitMin=0, fitMax=150, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{2} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Exp(-[1]*x)", [9.24465e+00, 2.21692e-02], [False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma3", fitF, params, outDir_param, recoil_qTbins, label, cutOffMin=2, cutOffMax=10, cParams=cParams, fitMin=0, fitMax=50, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{3} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [1.63014e-01, 4.04683e-01, 10], [False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma4", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{4} (GeV)")
        

        fitF, params, cParams = "[0]", [0.05], [False]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm1", fitF, params, outDir_param, recoil_qTbins, label, doFit=False, cutOffMin=0.03, cutOffMax=0.5, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=1, xMax=150, yTitle = "n_{3} (GeV)")

        fitF, params, cParams = "[0]*TMath::Exp(-[1]*x)", [3.08576e-01, 3.17645e-03], [False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm2", fitF, params, outDir_param, recoil_qTbins, label, excludeBins=[39], cutOffMin=0.15, cutOffMax=0.35, cParams=cParams, fitMin=0, fitMax=80, yMin=0, yMax=1, xMax=150, yTitle = "n_{3} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Exp(-[1]*x)", [4.93636e-01, 7.88253e-02], [False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm3", fitF, params, outDir_param, recoil_qTbins, label, doFit=False, cParams=cParams, fitMin=0, fitMax=30, yMin=0, yMax=1, xMax=150, yTitle = "n_{3} (GeV)")
        
        with open("%s/results.json" % outDir_param, "w") as outfile: json.dump(jsOut, outfile, indent=4)
        return
        
    outDir_fits_v1 = "%s/fits_v1" % baseDir
    if False:
        fitCfg['mean_cfg'] = [1, 1, 1, 1] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
        fitCfg['sigma_cfg'] = [0, 0, 0, 0]
        fitCfg['norm_cfg'] = [2, 2, 0]
        #fitCfg['sigma'] = [15, 5, 7, 10]
        with open("%s/results.json" % outDir_param_v0) as f: jsIn = json.load(f)
        recoilLibs.doFitMultiGauss(rFile, proc, comp, fitCfg, label, outDir_fits_v1, recoil_qTbins, bkgCfg=bkgCfg, funcJs=jsIn, qTmax=qTmax, rebin=rebin, xMin=-100, xMax=100)
        return
        
    outDir_param_v1 = "%s/param_v1" % baseDir
    if False:
        jsOut = {}
        jsOut['nGauss'] = nGauss
        outDir_param = outDir_param_v1
        functions.prepareDir(outDir_param, True)
        with open("%s/results.json" % outDir_fits_v1) as f: jsIn = json.load(f)
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [6.34447e+00, 9.71985e-02, 7.13032e+00], [True, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma1", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=70, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{1} (GeV)")
 
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [2.72251e+00, 1.52312e-01, 2.15765e+00], [True, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma2", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, cutOffMin=5, fitMin=0, fitMax=150, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{2} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Exp(-[1]*x)", [9.24465e+00, 2.21692e-02], [False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma3", fitF, params, outDir_param, recoil_qTbins, label, cutOffMin=2, cutOffMax=10, cParams=cParams, fitMin=0, fitMax=50, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{3} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [1.63014e-01, 4.04683e-01, 10], [False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma4", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{4} (GeV)")
        

        fitF, params, cParams = "[0]", [0.05], [False]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm1", fitF, params, outDir_param, recoil_qTbins, label, doFit=False, cutOffMin=0.03, cutOffMax=0.5, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=1, xMax=150, yTitle = "n_{3} (GeV)")

        fitF, params, cParams = "[0]*TMath::Exp(-[1]*x)", [3.08576e-01, 3.17645e-03], [False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm2", fitF, params, outDir_param, recoil_qTbins, label, excludeBins=[39], cutOffMin=0.15, cutOffMax=0.35, cParams=cParams, fitMin=0, fitMax=80, yMin=0, yMax=1, xMax=150, yTitle = "n_{3} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Exp(-[1]*x)", [4.93636e-01, 7.88253e-02], [False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm3", fitF, params, outDir_param, recoil_qTbins, label, doFit=False, cParams=cParams, fitMin=0, fitMax=30, yMin=0, yMax=1, xMax=150, yTitle = "n_{3} (GeV)")
        
        with open("%s/results.json" % outDir_param, "w") as outfile: json.dump(jsOut, outfile, indent=4)
        return
        
    outDir_fits_v2 = "%s/fits_v2" % baseDir
    if False:
        fitCfg['mean_cfg'] = [1, 1, 1, 1] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
        fitCfg['sigma_cfg'] = [2, 0, 0, 0]
        fitCfg['norm_cfg'] = [2, 2, 2]
        with open("%s/results.json" % outDir_param_v1) as f: jsIn = json.load(f)
        recoilLibs.doFitMultiGauss(rFile, proc, comp, fitCfg, label, outDir_fits_v2, recoil_qTbins, bkgCfg=bkgCfg, funcJs=jsIn, qTmax=qTmax, rebin=rebin, xMin=-100, xMax=100)
        return
        
    outDir_param_v2 = "%s/param_v2" % baseDir
    if False:
        jsOut = {}
        jsOut['nGauss'] = nGauss
        outDir_param = outDir_param_v2
        functions.prepareDir(outDir_param, True)
        with open("%s/results.json" % outDir_fits_v2) as f: jsIn = json.load(f)
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [6.34447e+00, 9.71985e-02, 7.13032e+00], [True, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma1", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=70, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{1} (GeV)")
 
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [2.72251e+00, 1.52312e-01, 2.15765e+00], [True, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma2", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, cutOffMin=5, fitMin=0, fitMax=150, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{2} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Exp(-[1]*x)", [9.24465e+00, 2.21692e-02], [False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma3", fitF, params, outDir_param, recoil_qTbins, label, cutOffMin=2, cutOffMax=10, cParams=cParams, fitMin=0, fitMax=50, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{3} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [1.63014e-01, 4.04683e-01, 10], [False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma4", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{4} (GeV)")
        

        fitF, params, cParams = "[0]", [0.05], [False]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm1", fitF, params, outDir_param, recoil_qTbins, label, doFit=False, cutOffMin=0.03, cutOffMax=0.5, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=1, xMax=150, yTitle = "n_{3} (GeV)")

        fitF, params, cParams = "[0]*TMath::Exp(-[1]*x)", [3.08576e-01, 3.17645e-03], [False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm2", fitF, params, outDir_param, recoil_qTbins, label, excludeBins=[39], cutOffMin=0.15, cutOffMax=0.35, cParams=cParams, fitMin=0, fitMax=80, yMin=0, yMax=1, xMax=150, yTitle = "n_{3} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Exp(-[1]*x)", [4.93636e-01, 7.88253e-02], [False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm3", fitF, params, outDir_param, recoil_qTbins, label, doFit=False, cParams=cParams, fitMin=0, fitMax=30, yMin=0, yMax=1, xMax=150, yTitle = "n_{3} (GeV)")
        
        
        with open("%s/results.json" % outDir_param, "w") as outfile: json.dump(jsOut, outfile, indent=4)
        return
        
    outDir_fits_v3 = "%s/fits_v3" % baseDir
    if False:
        fitCfg['mean_cfg'] = [1, 1, 1, 1] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
        fitCfg['sigma_cfg'] = [2, 0, 2, 0]
        fitCfg['norm_cfg'] = [2, 2, 2]
        with open("%s/results.json" % outDir_param_v2) as f: jsIn = json.load(f)
        recoilLibs.doFitMultiGauss(rFile, proc, comp, fitCfg, label, outDir_fits_v3, recoil_qTbins, bkgCfg=bkgCfg, funcJs=jsIn, qTmax=qTmax, rebin=rebin, xMin=-100, xMax=100)
        return
        
    outDir_param_v3 = "%s/param_v3" % baseDir
    if False:
        jsOut = {}
        jsOut['nGauss'] = nGauss
        outDir_param = outDir_param_v3
        functions.prepareDir(outDir_param_v3, True)
        with open("%s/results.json" % outDir_fits_v3) as f: jsIn = json.load(f)
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [6.34447e+00, 9.71985e-02, 7.13032e+00], [True, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma1", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=70, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{1} (GeV)")
 
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [2.72251e+00, 1.52312e-01, 2.15765e+00], [True, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma2", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, cutOffMin=5, fitMin=0, fitMax=150, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{2} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Exp(-[1]*x)", [9.24465e+00, 2.21692e-02], [False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma3", fitF, params, outDir_param, recoil_qTbins, label, cutOffMin=2, cutOffMax=10, cParams=cParams, fitMin=0, fitMax=50, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{3} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [1.63014e-01, 4.04683e-01, 10], [False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma4", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{4} (GeV)")
        

        fitF, params, cParams = "[0]", [0.05], [False]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm1", fitF, params, outDir_param, recoil_qTbins, label, doFit=False, cutOffMin=0.03, cutOffMax=0.5, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=1, xMax=150, yTitle = "n_{3} (GeV)")

        fitF, params, cParams = "[0]*TMath::Exp(-[1]*x)", [3.08576e-01, 3.17645e-03], [False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm2", fitF, params, outDir_param, recoil_qTbins, label, excludeBins=[39], cutOffMin=0.15, cutOffMax=0.35, cParams=cParams, fitMin=0, fitMax=80, yMin=0, yMax=1, xMax=150, yTitle = "n_{3} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Exp(-[1]*x)", [4.93636e-01, 7.88253e-02], [False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm3", fitF, params, outDir_param, recoil_qTbins, label, doFit=False, cParams=cParams, fitMin=0, fitMax=30, yMin=0, yMax=1, xMax=150, yTitle = "n_{3} (GeV)")
        
        with open("%s/results.json" % outDir_param, "w") as outfile: json.dump(jsOut, outfile, indent=4)    
        return        

    outDir_fits_v4 = "%s/fits_v4" % baseDir
    if False:
        fitCfg['mean_cfg'] = [1, 1, 1, 1] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
        fitCfg['sigma_cfg'] = [2, 2, 0, 2]
        fitCfg['norm_cfg'] = [2, 2, 2]
        fitCfg['sigma'] = [14, 6, 7, 10]
        with open("%s/results.json" % outDir_param_v3) as f: jsIn = json.load(f)
        recoilLibs.doFitMultiGauss(rFile, proc, comp, fitCfg, label, outDir_fits_v4, recoil_qTbins, bkgCfg=bkgCfg, funcJs=jsIn, qTmax=150, rebin=rebin, xMin=-100, xMax=100)
        return        
        
    outDir_param_v4 = "%s/param_v4" % baseDir
    if False:
        jsOut = {}
        jsOut['nGauss'] = nGauss
        outDir_param = outDir_param_v4
        functions.prepareDir(outDir_param, True)
        with open("%s/results.json" % outDir_fits_v4) as f: jsIn = json.load(f)
        
        fitF, params, cParams = "[0]", [0.0], [True]
        recoilLibs.doFit(jsIn, jsOut, comp, "mean1", fitF, params, outDir_param, recoil_qTbins, label, doFit=False, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=1, xMax=150, yTitle = "#mu_{1} (GeV)")
        recoilLibs.doFit(jsIn, jsOut, comp, "mean2", fitF, params, outDir_param, recoil_qTbins, label, doFit=False, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=1, xMax=150, yTitle = "#mu_{2} (GeV)")
        recoilLibs.doFit(jsIn, jsOut, comp, "mean3", fitF, params, outDir_param, recoil_qTbins, label, doFit=False, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=1, xMax=150, yTitle = "#mu_{3} (GeV)")
        recoilLibs.doFit(jsIn, jsOut, comp, "mean4", fitF, params, outDir_param, recoil_qTbins, label, doFit=False, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=1, xMax=150, yTitle = "#mu_{4} (GeV)")
   
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [6.34447e+00, 9.71985e-02, 7.13032e+00], [True, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma1", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=70, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{1} (GeV)")
 
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [2.72251e+00, 1.52312e-01, 2.15765e+00], [True, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma2", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, cutOffMin=5, fitMin=0, fitMax=150, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{2} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Exp(-[1]*x)", [9.24465e+00, 2.21692e-02], [False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma3", fitF, params, outDir_param, recoil_qTbins, label, cutOffMin=2, cutOffMax=10, cParams=cParams, fitMin=0, fitMax=50, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{3} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [1.63014e-01, 4.04683e-01, 10], [False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma4", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{4} (GeV)")
        

        fitF, params, cParams = "[0]", [0.05], [True]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm1", fitF, params, outDir_param, recoil_qTbins, label, doFit=False, cutOffMin=0.03, cutOffMax=0.5, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=1, xMax=150, yTitle = "n_{3} (GeV)")

        fitF, params, cParams = "[0]*TMath::Exp(-[1]*x)", [3.08576e-01, 3.17645e-03], [True, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm2", fitF, params, outDir_param, recoil_qTbins, label, excludeBins=[39], cutOffMin=0.15, cutOffMax=0.35, cParams=cParams, fitMin=0, fitMax=80, yMin=0, yMax=1, xMax=150, yTitle = "n_{3} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Exp(-[1]*x)", [4.93636e-01, 7.88253e-02], [True, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm3", fitF, params, outDir_param, recoil_qTbins, label, doFit=False, cParams=cParams, fitMin=0, fitMax=30, yMin=0, yMax=1, xMax=150, yTitle = "n_{3} (GeV)")
        
        with open("%s/results.json" % outDir_param, "w") as outfile: json.dump(jsOut, outfile, indent=4)    
        return


    ########################  
    # Nominal refit
    ########################
    if True:
        outDir_refit = "%s/params_refit" % baseDir
        functions.prepareDir(outDir_refit)
    
        # need to get the original fit config, to not fit twice some entangled params (treat them as RooRealVar alias)
        fitCfg['mean_cfg'] = [2, 2, 2, 2] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
        fitCfg['sigma_cfg'] = [2, 2, 2, 2]
        fitCfg['norm_cfg'] = [2, 2, 2]
        with open("%s/results.json" % outDir_param_v4) as f: jsIn = json.load(f)
        jsOut = recoilLibs.combinedFit(rFile, proc, comp, fitCfg, jsIn, recoil_qTbins, bkgCfg=bkgCfg, rebin=rebin, qTmax=150)
        with open("%s/results_refit.json" % outDir_refit, "w") as outfile: json.dump(jsOut, outfile, indent=4)
        #return
        
        fitCfg['mean_cfg'] = [2, 2, 2, 2] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
        fitCfg['sigma_cfg'] = [2, 2, 2, 2]
        fitCfg['norm_cfg'] = [2, 2, 2]
        
        with open("%s/results_refit.json" % outDir_refit) as f: jsIn = json.load(f)
        recoilLibs.doPlot(comp, "norm1", jsIn, outDir_refit, recoil_qTbins, label, yMin=0, yMax=1, xMax=150, yTitle = "n_{1} (GeV)")
        recoilLibs.doPlot(comp, "norm2", jsIn, outDir_refit, recoil_qTbins, label, yMin=0, yMax=1, xMax=150, yTitle = "n_{2} (GeV)")
        recoilLibs.doPlot(comp, "norm3", jsIn, outDir_refit, recoil_qTbins, label, yMin=0, yMax=1, xMax=150, yTitle = "n_{3} (GeV)")
        recoilLibs.doPlot(comp, "sigma1", jsIn, outDir_refit, recoil_qTbins, label, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{1} (GeV)")
        recoilLibs.doPlot(comp, "sigma2", jsIn, outDir_refit, recoil_qTbins, label, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{2} (GeV)")
        recoilLibs.doPlot(comp, "sigma3", jsIn, outDir_refit, recoil_qTbins, label, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{3} (GeV)")
        recoilLibs.doPlot(comp, "sigma4", jsIn, outDir_refit, recoil_qTbins, label, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{4} (GeV)")

        outDir_refit_fits = "%s/fits_refit" % baseDir
        recoilLibs.doFitMultiGauss(rFile, proc, comp, fitCfg, label, outDir_refit_fits, recoil_qTbins, bkgCfg=bkgCfg, funcJs=jsIn, qTmax=qTmax, rebin=rebin, xMin=-100, xMax=100)
      
      
    ########################  
    # Background uncertainty
    ########################
    if False:
        baseDir_bkg = "%s/bkgUnc/" % baseDir
        functions.prepareDir(baseDir_bkg, False)
        bkgs, bkgs_perts = ['TTbar', 'EWK', 'ZZ'], [1.05, 1.1, 1.1]
        for i, bkg in enumerate(bkgs):
            if bkg != "EWK": continue # need to cleanup... one by one
            for b in bkgCfg: 
                if b == bkg: bkgCfg[b]['norm'] = bkgs_perts[i]
                else: bkgCfg[b]['norm'] = 1.0
                
            outDir_refit = "%s/params_refit_%s" % (baseDir_bkg, bkg)
            functions.prepareDir(outDir_refit)

            # need to get the original fit config, to not fit twice some entangled params (treat them as RooRealVar alias)
            fitCfg['mean_cfg'] = [2, 2, 2, 2] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
            fitCfg['sigma_cfg'] = [2, 2, 2, 2]
            fitCfg['norm_cfg'] = [2, 2, 2]
            with open("%s/results.json" % outDir_param_v4) as f: jsIn = json.load(f)
            jsOut = recoilLibs.combinedFit(rFile, proc, comp, fitCfg, jsIn, recoil_qTbins, bkgCfg=bkgCfg, rebin=rebin, qTmax=150)
            with open("%s/results_refit.json" % outDir_refit, "w") as outfile: json.dump(jsOut, outfile, indent=4)
            #return

            with open("%s/results_refit.json" % outDir_refit) as f: jsIn = json.load(f)
            recoilLibs.doPlot(comp, "norm1", jsIn, outDir_refit, recoil_qTbins, label, yMin=0, yMax=1, xMax=150, yTitle = "n_{1} (GeV)")
            recoilLibs.doPlot(comp, "norm2", jsIn, outDir_refit, recoil_qTbins, label, yMin=0, yMax=1, xMax=150, yTitle = "n_{2} (GeV)")
            recoilLibs.doPlot(comp, "norm3", jsIn, outDir_refit, recoil_qTbins, label, yMin=0, yMax=1, xMax=150, yTitle = "n_{3} (GeV)")
            recoilLibs.doPlot(comp, "sigma1", jsIn, outDir_refit, recoil_qTbins, label, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{1} (GeV)")
            recoilLibs.doPlot(comp, "sigma2", jsIn, outDir_refit, recoil_qTbins, label, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{2} (GeV)")
            recoilLibs.doPlot(comp, "sigma3", jsIn, outDir_refit, recoil_qTbins, label, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{3} (GeV)")
            recoilLibs.doPlot(comp, "sigma4", jsIn, outDir_refit, recoil_qTbins, label, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{4} (GeV)")

            outDir_refit_fits = "%s/fits_refit_%s" % (baseDir_bkg, bkg)
            recoilLibs.doFitMultiGauss(rFile, proc, comp, fitCfg, label, outDir_refit_fits, recoil_qTbins, bkgCfg=bkgCfg, funcJs=jsIn, qTmax=qTmax, rebin=rebin, xMin=-100, xMax=100)

def do_singlemuon_para_RawPFMET():

    proc = "SingleMuon"
    tag = "singlemuon_para"
    comp = "para"
    baseDir = "%s/%s" % (outDir, tag)
    name = "SingleMuon #rightarrow #mu^{+}#mu^{#minus}"
    label = "%s, %s" % (name, met)
    functions.prepareDir(baseDir, False)
    rebin = 1

    outDir_fits_v0 = "%s/fits_v0" % baseDir
    fitCfg = {}
    
    fitCfg['mean'], fitCfg['mean_cfg'] = [0, 0, 0], [0, -1, -1] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
    fitCfg['sigma'], fitCfg['sigma_cfg'] = [5, 10,  18], [0, 0, 0]
    fitCfg['norm'], fitCfg['norm_cfg'] = [0.30, 0.40], [1, 1]
    
    nGauss = len(fitCfg['mean'])
    
    bkgCfg = {}
    bkgCfg['TTbar'] = { "filename": "/eos/user/j/jaeyserm/www/wmass/lowPU/recoilCorrection/recoil_mumu_RawPFMET/ttbar_para/fits_v0/results.json", "norm": 1.0, "float": False, "subtract": False }
    bkgCfg['EWK'] = { "filename": "/eos/user/j/jaeyserm/www/wmass/lowPU/recoilCorrection/recoil_mumu_RawPFMET/ewk_para/fits_v0/results.json", "norm": 1.0, "float": False, "subtract": False }
    bkgCfg['ZZ'] = { "filename": "/eos/user/j/jaeyserm/www/wmass/lowPU/recoilCorrection/recoil_mumu_RawPFMET/zz_para/fits_v0/results.json", "norm": 1.0, "float": False, "subtract": False }
    
    
    if False: 
        recoilLibs.doFitMultiGauss(rFile, proc, comp, fitCfg, label, outDir_fits_v0, recoil_qTbins, bkgCfg=bkgCfg, qTmax=qTmax, propagate=True, rebin=rebin, xMin=-100, xMax=100)
        return
        
    outDir_param_v0 = "%s/param_v0" % baseDir
    if False:
        jsOut = {}
        jsOut['nGauss'] = nGauss
        outDir_param = outDir_param_v0
        functions.prepareDir(outDir_param, True)
        with open("%s/results.json" % outDir_fits_v0) as f: jsIn = json.load(f)
        
        t = 25
        fLeft, dfLeft  = "[0] + [1]*x + [2]*x*x + [3]*x*x*x + [4]*x*x*x*x", "[1] + 2*[2]*x + 3*[3]*x*x + 4*[4]*x*x*x"
        fLeftEval, dfLeftEval = fLeft.replace("x", str(t)), dfLeft.replace("x", str(t))
        fitF = "(x<{t})*1*({fLeft}) +0+ (x>{t})*1*(  ({dfLeftEval})*x + ({fLeftEval}-({dfLeftEval})*{t}) )".format(t=t, fLeft=fLeft, fLeftEval=fLeftEval, dfLeftEval=dfLeftEval)
        params, cParams = [5.25854e-02, 4.82763e-01, -2.48218e-02, 1.00995e-03, -1.68082e-05], [False, False, False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "mean1", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=100, yMin=-5, yMax=50, xMax=150, yTitle = "#mu_{1} (GeV)")
        recoilLibs.doFit(jsIn, jsOut, comp, "mean2", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=100, yMin=-5, yMax=50, xMax=150, yTitle = "#mu_{2} (GeV)")
        recoilLibs.doFit(jsIn, jsOut, comp, "mean3", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=100, yMin=-5, yMax=50, xMax=150, yTitle = "#mu_{3} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [2.34808e-02, 1.34637e+00, 5.52070e+00], [False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma1", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=30, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{1} (GeV)")
 
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [5.34784e+00, 1.22821e-01, 1.65750e+00], [False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma2", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=100, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{2} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [9.07914e-02, 9.73452e-01, 1.11361e+01], [False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma3", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=100, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{2} (GeV)")
        
        fitF, params, cParams = "[0]", [0.3], [False]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm1", fitF, params, outDir_param, recoil_qTbins, label, doFit=False, cParams=cParams, fitMin=0, fitMax=20, yMin=0, yMax=1, xMax=150, yTitle = "n_{1} (GeV)")
        
        fitF, params, cParams = "[0]", [0.4], [False]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm2", fitF, params, outDir_param, recoil_qTbins, label, doFit=False, cParams=cParams, fitMin=0, fitMax=20, yMin=0, yMax=1, xMax=150, yTitle = "n_{2} (GeV)")


        with open("%s/results.json" % outDir_param, "w") as outfile: json.dump(jsOut, outfile, indent=4)
        return
        
    outDir_fits_v1 = "%s/fits_v1" % baseDir
    if False:
        fitCfg['mean_cfg'] = [2, 2, 2] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
        fitCfg['sigma_cfg'] = [0, 0, 0]
        fitCfg['norm_cfg'] = [2, 2]
        with open("%s/results.json" % outDir_param_v0) as f: jsIn = json.load(f)
        recoilLibs.doFitMultiGauss(rFile, proc, comp, fitCfg, label, outDir_fits_v1, recoil_qTbins, bkgCfg=bkgCfg, funcJs=jsIn, qTmax=qTmax, propagate=True, rebin=rebin, xMin=-100, xMax=100)
        return
        
    outDir_param_v1 = "%s/param_v1" % baseDir
    if False:
        jsOut = {}
        jsOut['nGauss'] = nGauss
        outDir_param = outDir_param_v1
        functions.prepareDir(outDir_param, True)
        with open("%s/results.json" % outDir_fits_v1) as f: jsIn = json.load(f)
        
        t = 25
        fLeft, dfLeft  = "[0] + [1]*x + [2]*x*x + [3]*x*x*x + [4]*x*x*x*x", "[1] + 2*[2]*x + 3*[3]*x*x + 4*[4]*x*x*x"
        fLeftEval, dfLeftEval = fLeft.replace("x", str(t)), dfLeft.replace("x", str(t))
        fitF = "(x<{t})*1*({fLeft}) +0+ (x>{t})*1*(  ({dfLeftEval})*x + ({fLeftEval}-({dfLeftEval})*{t}) )".format(t=t, fLeft=fLeft, fLeftEval=fLeftEval, dfLeftEval=dfLeftEval)
        params, cParams = [5.25854e-02, 4.82763e-01, -2.48218e-02, 1.00995e-03, -1.68082e-05], [False, False, False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "mean1", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=100, yMin=-5, yMax=50, xMax=150, yTitle = "#mu_{1} (GeV)")
        recoilLibs.doFit(jsIn, jsOut, comp, "mean2", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=100, yMin=-5, yMax=50, xMax=150, yTitle = "#mu_{2} (GeV)")
        recoilLibs.doFit(jsIn, jsOut, comp, "mean3", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=100, yMin=-5, yMax=50, xMax=150, yTitle = "#mu_{3} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [2.34808e-02, 1.34637e+00, 5.52070e+00], [False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma1", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=30, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{1} (GeV)")
 
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [5.34784e+00, 1.22821e-01, 1.65750e+00], [False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma2", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=100, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{2} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [9.07914e-02, 9.73452e-01, 1.11361e+01], [False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma3", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=100, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{2} (GeV)")
        
        fitF, params, cParams = "[0]", [0.3], [False]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm1", fitF, params, outDir_param, recoil_qTbins, label, doFit=False, cParams=cParams, fitMin=0, fitMax=20, yMin=0, yMax=1, xMax=150, yTitle = "n_{1} (GeV)")
        
        fitF, params, cParams = "[0]", [0.4], [False]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm2", fitF, params, outDir_param, recoil_qTbins, label, doFit=False, cParams=cParams, fitMin=0, fitMax=20, yMin=0, yMax=1, xMax=150, yTitle = "n_{2} (GeV)")


        with open("%s/results.json" % outDir_param, "w") as outfile: json.dump(jsOut, outfile, indent=4)    
        return
        
    outDir_fits_v2 = "%s/fits_v2" % baseDir
    if False:
        fitCfg['mean_cfg'] = [2, 2, 2] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
        fitCfg['sigma_cfg'] = [2, 0, 0]
        fitCfg['norm_cfg'] = [2, 2]
        with open("%s/results.json" % outDir_param_v1) as f: jsIn = json.load(f)
        recoilLibs.doFitMultiGauss(rFile, proc, comp, fitCfg, label, outDir_fits_v2, recoil_qTbins, bkgCfg=bkgCfg, funcJs=jsIn, propagate=True, qTmax=qTmax, rebin=rebin, xMin=-100, xMax=100)
        return
        
    outDir_param_v2 = "%s/param_v2" % baseDir
    if False:
        jsOut = {}
        jsOut['nGauss'] = nGauss
        outDir_param = outDir_param_v2
        functions.prepareDir(outDir_param_v2, True)
        with open("%s/results.json" % outDir_fits_v2) as f: jsIn = json.load(f)
        
        t = 25
        fLeft, dfLeft  = "[0] + [1]*x + [2]*x*x + [3]*x*x*x + [4]*x*x*x*x", "[1] + 2*[2]*x + 3*[3]*x*x + 4*[4]*x*x*x"
        fLeftEval, dfLeftEval = fLeft.replace("x", str(t)), dfLeft.replace("x", str(t))
        fitF = "(x<{t})*1*({fLeft}) +0+ (x>{t})*1*(  ({dfLeftEval})*x + ({fLeftEval}-({dfLeftEval})*{t}) )".format(t=t, fLeft=fLeft, fLeftEval=fLeftEval, dfLeftEval=dfLeftEval)
        params, cParams = [5.25854e-02, 4.82763e-01, -2.48218e-02, 1.00995e-03, -1.68082e-05], [False, False, False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "mean1", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=100, yMin=-5, yMax=50, xMax=150, yTitle = "#mu_{1} (GeV)")
        recoilLibs.doFit(jsIn, jsOut, comp, "mean2", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=100, yMin=-5, yMax=50, xMax=150, yTitle = "#mu_{2} (GeV)")
        recoilLibs.doFit(jsIn, jsOut, comp, "mean3", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=100, yMin=-5, yMax=50, xMax=150, yTitle = "#mu_{3} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [2.34808e-02, 1.34637e+00, 5.52070e+00], [False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma1", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=30, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{1} (GeV)")
 
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [5.34784e+00, 1.22821e-01, 1.65750e+00], [False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma2", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=100, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{2} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [9.07914e-02, 9.73452e-01, 1.11361e+01], [False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma3", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=100, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{2} (GeV)")
        
        fitF, params, cParams = "[0]", [0.3], [False]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm1", fitF, params, outDir_param, recoil_qTbins, label, doFit=False, cParams=cParams, fitMin=0, fitMax=20, yMin=0, yMax=1, xMax=150, yTitle = "n_{1} (GeV)")
        
        fitF, params, cParams = "[0]", [0.4], [False]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm2", fitF, params, outDir_param, recoil_qTbins, label, doFit=False, cParams=cParams, fitMin=0, fitMax=20, yMin=0, yMax=1, xMax=150, yTitle = "n_{2} (GeV)")


        with open("%s/results.json" % outDir_param, "w") as outfile: json.dump(jsOut, outfile, indent=4)    
        return
        
    outDir_fits_v3 = "%s/fits_v3" % baseDir
    if False:
        fitCfg['mean_cfg'] = [2, 2, 2] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
        fitCfg['sigma_cfg'] = [2, 2, 0]
        fitCfg['norm_cfg'] = [2, 2, 2]
        with open("%s/results.json" % outDir_param_v2) as f: jsIn = json.load(f)
        recoilLibs.doFitMultiGauss(rFile, proc, comp, fitCfg, label, outDir_fits_v3, recoil_qTbins, bkgCfg=bkgCfg, funcJs=jsIn, qTmax=qTmax, rebin=rebin, xMin=-100, xMax=100)
        return
    
    outDir_param_v3 = "%s/param_v3" % baseDir
    if True:
        jsOut = {}
        jsOut['nGauss'] = nGauss
        outDir_param = outDir_param_v3
        functions.prepareDir(outDir_param, True)
        with open("%s/results.json" % outDir_fits_v3) as f: jsIn = json.load(f)
        
        t = 25
        fLeft, dfLeft  = "[0] + [1]*x + [2]*x*x + [3]*x*x*x + [4]*x*x*x*x", "[1] + 2*[2]*x + 3*[3]*x*x + 4*[4]*x*x*x"
        fLeftEval, dfLeftEval = fLeft.replace("x", str(t)), dfLeft.replace("x", str(t))
        fitF = "(x<{t})*1*({fLeft}) +0+ (x>{t})*1*(  ({dfLeftEval})*x + ({fLeftEval}-({dfLeftEval})*{t}) )".format(t=t, fLeft=fLeft, fLeftEval=fLeftEval, dfLeftEval=dfLeftEval)
        params, cParams = [5.25854e-02, 4.82763e-01, -2.48218e-02, 1.00995e-03, -1.68082e-05], [False, False, False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "mean1", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=100, yMin=-5, yMax=50, xMax=150, yTitle = "#mu_{1} (GeV)")
        recoilLibs.doFit(jsIn, jsOut, comp, "mean2", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=100, yMin=-5, yMax=50, xMax=150, yTitle = "#mu_{2} (GeV)")
        recoilLibs.doFit(jsIn, jsOut, comp, "mean3", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=100, yMin=-5, yMax=50, xMax=150, yTitle = "#mu_{3} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [2.34808e-02, 1.34637e+00, 5.52070e+00], [False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma1", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=30, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{1} (GeV)")
 
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [5.34784e+00, 1.22821e-01, 1.65750e+00], [False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma2", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=100, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{2} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [9.07914e-02, 9.73452e-01, 1.11361e+01], [False, False, False]
        recoilLibs.doFit(jsIn, jsOut, comp, "sigma3", fitF, params, outDir_param, recoil_qTbins, label, cParams=cParams, fitMin=0, fitMax=100, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{2} (GeV)")
        
        fitF, params, cParams = "[0]", [0.3], [False]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm1", fitF, params, outDir_param, recoil_qTbins, label, doFit=False, cParams=cParams, fitMin=0, fitMax=20, yMin=0, yMax=1, xMax=150, yTitle = "n_{1} (GeV)")
        
        fitF, params, cParams = "[0]", [0.4], [False]
        recoilLibs.doFit(jsIn, jsOut, comp, "norm2", fitF, params, outDir_param, recoil_qTbins, label, doFit=False, cParams=cParams, fitMin=0, fitMax=20, yMin=0, yMax=1, xMax=150, yTitle = "n_{2} (GeV)")

        with open("%s/results.json" % outDir_param, "w") as outfile: json.dump(jsOut, outfile, indent=4)    
        #return        


    ########################  
    # Nominal refit
    ########################
    if True:
        outDir_refit = "%s/params_refit" % baseDir
        functions.prepareDir(outDir_refit)
    
        # need to get the original fit config, to not fit twice some entangled params (treat them as RooRealVar alias)
        fitCfg['mean_cfg'] = [0, -1, -1] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
        fitCfg['sigma_cfg'] = [2, 2, 2]
        fitCfg['norm_cfg'] = [2, 2]
        with open("%s/results.json" % outDir_param_v3) as f: jsIn = json.load(f)
        jsOut = recoilLibs.combinedFit(rFile, proc, comp, fitCfg, jsIn, recoil_qTbins, bkgCfg=bkgCfg, rebin=rebin, qTmax=150)
        with open("%s/results_refit.json" % outDir_refit, "w") as outfile: json.dump(jsOut, outfile, indent=4)
        #return
        
        fitCfg['mean_cfg'] = [2, 2, 2] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
        fitCfg['sigma_cfg'] = [2, 2, 2]
        fitCfg['norm_cfg'] = [2, 2]
        
        with open("%s/results_refit.json" % outDir_refit) as f: jsIn = json.load(f)
        recoilLibs.doPlot(comp, "norm1", jsIn, outDir_refit, recoil_qTbins, label, yMin=0, yMax=1, xMax=150, yTitle = "n_{1} (GeV)")
        recoilLibs.doPlot(comp, "norm2", jsIn, outDir_refit, recoil_qTbins, label, yMin=0, yMax=1, xMax=150, yTitle = "n_{2} (GeV)")
        recoilLibs.doPlot(comp, "sigma1", jsIn, outDir_refit, recoil_qTbins, label, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{1} (GeV)")
        recoilLibs.doPlot(comp, "sigma2", jsIn, outDir_refit, recoil_qTbins, label, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{2} (GeV)")
        recoilLibs.doPlot(comp, "sigma3", jsIn, outDir_refit, recoil_qTbins, label, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{3} (GeV)")
        recoilLibs.doPlot(comp, "mean1", jsIn, outDir_refit, recoil_qTbins, label, yMin=-5, yMax=50, xMax=150, yTitle = "#mu_{1} (GeV)")
        recoilLibs.doPlot(comp, "mean2", jsIn, outDir_refit, recoil_qTbins, label, yMin=-5, yMax=50, xMax=150, yTitle = "#mu_{2} (GeV)")
        recoilLibs.doPlot(comp, "mean3", jsIn, outDir_refit, recoil_qTbins, label, yMin=-5, yMax=50, xMax=150, yTitle = "#mu_{3} (GeV)")

        outDir_refit_fits = "%s/fits_refit" % baseDir
        recoilLibs.doFitMultiGauss(rFile, proc, comp, fitCfg, label, outDir_refit_fits, recoil_qTbins, bkgCfg=bkgCfg, funcJs=jsIn, qTmax=qTmax, rebin=rebin, xMin=-100, xMax=100)
      
      
    ########################  
    # Background uncertainty
    ########################
    if False:
        baseDir_bkg = "%s/bkgUnc/" % baseDir
        functions.prepareDir(baseDir_bkg, False)
        bkgs, bkgs_perts = ['TTbar', 'EWK', 'ZZ'], [1.05, 1.1, 1.1]
        for i, bkg in enumerate(bkgs):
            if bkg != "ZZ": continue # need to cleanup... one by one
            for b in bkgCfg: 
                if b == bkg: bkgCfg[b]['norm'] = bkgs_perts[i]
                else: bkgCfg[b]['norm'] = 1.0
                
            outDir_refit = "%s/params_refit_%s" % (baseDir_bkg, bkg)
            functions.prepareDir(outDir_refit)

            # need to get the original fit config, to not fit twice some entangled params (treat them as RooRealVar alias)
            fitCfg['mean_cfg'] = [0, -1, -1] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
            fitCfg['sigma_cfg'] = [2, 2, 2]
            fitCfg['norm_cfg'] = [2, 2]
            with open("%s/results.json" % outDir_param_v3) as f: jsIn = json.load(f)
            jsOut = recoilLibs.combinedFit(rFile, proc, comp, fitCfg, jsIn, recoil_qTbins, bkgCfg=bkgCfg, rebin=rebin, qTmax=150)
            with open("%s/results_refit.json" % outDir_refit, "w") as outfile: json.dump(jsOut, outfile, indent=4)
            #return

            fitCfg['mean_cfg'] = [2, 2, 2] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
            fitCfg['sigma_cfg'] = [2, 2, 2]
            fitCfg['norm_cfg'] = [2, 2]
            with open("%s/results_refit.json" % outDir_refit) as f: jsIn = json.load(f)
            recoilLibs.doPlot(comp, "norm1", jsIn, outDir_refit, recoil_qTbins, label, yMin=0, yMax=1, xMax=150, yTitle = "n_{1} (GeV)")
            recoilLibs.doPlot(comp, "norm2", jsIn, outDir_refit, recoil_qTbins, label, yMin=0, yMax=1, xMax=150, yTitle = "n_{2} (GeV)")
            recoilLibs.doPlot(comp, "sigma1", jsIn, outDir_refit, recoil_qTbins, label, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{1} (GeV)")
            recoilLibs.doPlot(comp, "sigma2", jsIn, outDir_refit, recoil_qTbins, label, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{2} (GeV)")
            recoilLibs.doPlot(comp, "sigma3", jsIn, outDir_refit, recoil_qTbins, label, yMin=0, yMax=50, xMax=150, yTitle = "#sigma_{3} (GeV)")
            recoilLibs.doPlot(comp, "mean1", jsIn, outDir_refit, recoil_qTbins, label, yMin=-5, yMax=50, xMax=150, yTitle = "#mu_{1} (GeV)")
            recoilLibs.doPlot(comp, "mean2", jsIn, outDir_refit, recoil_qTbins, label, yMin=-5, yMax=50, xMax=150, yTitle = "#mu_{2} (GeV)")

            outDir_refit_fits = "%s/fits_refit_%s" % (baseDir_bkg, bkg)
            recoilLibs.doFitMultiGauss(rFile, proc, comp, fitCfg, label, outDir_refit_fits, recoil_qTbins, bkgCfg=bkgCfg, funcJs=jsIn, qTmax=qTmax, rebin=rebin, xMin=-100, xMax=100)


 
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
    procs = ["ZZ", "EWK_noZZ", "TTbar"]
    if flavor == "mumu": procs += ["SingleMuon", "DYmumu"]
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
     
if __name__ == "__main__":

    met = "RawPFMET" # DeepMETReso RawPFMET
    flavor = "mumu" # mu, e, mumu, ee
    
    print("Start")
    met = "RawPFMET" # RawPFMET DeepMETReso
    flavor = "mumu" # mu, e, mumu, ee
    

    outDir = "/eos/user/j/jaeyserm/www/wmass/lowPU/Z%s_%s/plots/" % (flavor, met)
    #functions.prepareDir(outDir, remove=True)
    
    
    print("Open")
    groups = Datagroups("lowPU_%s_RawPFMET_%s_nnpdf31.pkl.lz4" % (flavor, met))
    
    quit()
    #rFile = "wremnants/data/lowPU/recoil/lowPU_%s_%s.root" % (flavor, met)
    #prepareFile("lowPU_%s_%s.pkl.lz4" % (flavor, met), rFile)
    #sys.exit()
    
    #recoil_qTbins = list(functions.drange(0, 50, 0.5)) + list(range(50, 100, 1)) + list(range(100, 150, 2)) + list(range(150, 200, 5)) + list(range(200, 300, 10)) + [300]
    #recoil_qTbins = list(range(0, 100, 1)) + list(range(100, 150, 2)) + list(range(150, 200, 5)) + list(range(200, 300, 10)) + [300]
    recoil_qTbins = list(range(0, 100, 1)) + list(range(100, 150, 2)) + [150]
    recoil_qTbins = list(range(0, 60, 1)) + list(range(60, 100, 2)) + list(range(100, 150, 5)) + [150]
    recoil_qTbins = list(range(0, 100, 1)) + list(range(100, 150, 2)) + [150]
    #recoil_qTbins = list(range(0, 20, 1)) + list(range(20, 40, 2)) + list(range(40, 80, 5)) + list(range(80, 150, 10)) + [150]
    outDir = "/eos/user/j/jaeyserm/www/wmass/lowPU/recoilCorrection/recoil_%s_%s/" % (flavor, met)
    functions.prepareDir(outDir, False)
    
    #ROOT.Math.MinimizerOptions.PrintDefault()
    #sys.exit()
    #ROOT.Math.MinimizerOptions.SetDefaultPrecision(1e-15)
    #ROOT.Math.MinimizerOptions.SetDefaultMinimizer("Minuit2")
    #ROOT.Math.MinimizerOptions.SetMinimizerAlgorithm("Fumili2") # Migrad Minimize Simplex Fumili2


    qTmax = 150
    
    '''
               Minimizer Type :          Minuit
      Minimizer Algorithm :          Migrad
                 Strategy :               1
                Tolerance :            0.01
           Max func calls :               0
           Max iterations :               0
           Func Precision :              -1
         Error definition :               1
              Print Level :               0

    min->SetMaxFunctionCalls(1e10);
min->SetMaxIterations(1e10);
min->SetTolerance(1e-14);

    '''

    if met == "RawPFMET":
    
        #do_dymumu_para_RawPFMET()
        do_dymumu_perp_RawPFMET_zlib()
        
        #do_ttbar_para_RawPFMET()
        #do_ttbar_perp_RawPFMET()
        
        #do_ewk_para_RawPFMET()
        #do_ewk_perp_RawPFMET()
        
        #do_zz_para_RawPFMET()
        #do_zz_perp_RawPFMET()
        
        #do_singlemuon_para_RawPFMET()
        #do_singlemuon_perp_RawPFMET()
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