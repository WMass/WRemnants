

import sys,array,math,os,copy,decimal,ctypes,json
import math
#sys.path.insert(0, "/home/j/jaeyserm/analysis/WRemnants/env/lib/python3.10/site-packages")

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptTitle(0)

import hist
import boost_histogram as bh
import numpy as np

import lz4.frame
import pickle
import narf
import time

import zfit
import zfit.z.numpy  # numpy-like backend
from zfit import z  # backend


zfit.run.set_n_cpu(1)
zfit.run.set_cpus_explicit(1, 1)

zfit.run.set_n_cpu(256)
zfit.run.set_cpus_explicit(256, 256)
print(zfit.run.n_cpu)

import functions
import plotter
import recoilFunctions as rf

import matplotlib as mpl
import matplotlib.pyplot as plt

import tensorflow as tf

from wremnants.datasets.datagroups import Datagroups

import mplhep as hep
hep.style.use(hep.style.ROOT)





__topRight__ = "199 pb^{#minus1} (13 TeV)"


__min_recoil__, __max_recoil__ = -100, 100
__min_mean__, __max_mean__ = -50, 500
__min_sigma__, __max_sigma__ = 0.1, 100

def plot_aux_ratio(ax, xMin, xMax, lumi, com=13):
    scale = max(1, np.divide(*ax[0].get_figure().get_size_inches())*0.3)
    _lumi = r"{lumi} ({com} TeV)".format(lumi="{0:.1f}".format(lumi) + r" $\mathrm{fb^{-1}}$", com=str(com))
    ax[1].plot([xMin, xMax], [1, 1], 'k-', lw=1)
    hep.cms.label(ax=ax[0], rlabel="", fontsize=20*scale,  label="Preliminary", data=True, com=13) # https://github.com/scikit-hep/mplhep/blob/master/src/mplhep/label.py
    ax[0].text(x=1, y=0.997, s=_lumi, transform=ax[0].transAxes, ha="right", va="bottom", fontsize=20*scale, fontweight="normal", fontname="TeX Gyre Heros") # lumi hack



def getParamIdx(name, params):
    
    for i in range(0, len(params)): 
       
        if name == params[i].GetName(): return i
            
    return -1
    


def diagonalize(fitRes, verbose=False):

    cov = fitRes.covarianceMatrix() # covariance matrix: diagonal elements are the variances of each of the parameters
    if verbose: cov.Print()
    params = fitRes.floatParsFinal()
    npars = params.getSize()
    sigma = 1
    
    nom = np.zeros(npars)
    for i in range(0, npars): nom[i] = params[i].getVal()

        
    cov_ = np.zeros((npars, npars))
    for i in range(0, npars):
        for j in range(0, npars):
            cov_[i][j] = cov[i][j]
         
        
    # eigenvalues and eigenvectors
    eig_vals, eig_vec = np.linalg.eig(cov_)
    eig_vec_inv = np.linalg.inv(eig_vec)
        
    #print("COVARIANCE")
    #print(cov_)
    #print("EIGENVECTORS")
    #print(eig_vec)
    #print("EIGENVECTORS INV")
    #print(eig_vec_inv)
    #print("EIGENVALUES")
    #print(eig_vals)
    #print("NOMINAL")
    #print(nom)
                
    ret = []
    for iVar in range(0, npars):
            
        #print("******** vary", iVar+1, "********")
            
        dnom = copy.deepcopy(eig_vals)
        for i, eig in enumerate(dnom): 
           
            if i == iVar: dnom[i] = sigma*math.sqrt(eig)
            else: dnom[i] = 0
               
        #print("PERTURBED NON-ROTATED")
        #print(dnom)
            
        # rotate perturbation back to nominal base
        dnom = np.dot(eig_vec, dnom)
        nom_pert = nom + dnom
        
        ret.append(nom_pert)
        
        #print("%f %f %f" % (nom_pert[0], nom_pert[1], nom_pert[2]))
        #print("PERTURBED ROTATED")
        #print(nom_pert)
     
    return ret

def mirrorx(h):

    hnew = h.Clone("%s_mirrorx" % h.GetName())
    for i in range(1, hnew.GetNbinsX()+1):
        idx = hnew.GetNbinsX()-i+1
        hnew.SetBinContent(i, h.GetBinContent(idx))
        hnew.SetBinError(i, h.GetBinError(idx))

    #sys.exit()
    return hnew


def chi2ss(y_pred, y, y_error, nParams):
    ret = 0
    nBins = 0
    for i in range(0, len(y)):
        if y[i] <= 0: continue
        ret += (((y_pred[i] - y[i]) / y_error[i]) ** 2)
        nBins += 1
    return ret/(nBins-nParams)
            
    
 
def doFitMultiGauss(bhist, comp, fitCfg, label, outDir_, recoil_qTbins, ratio=True, funcJs=None, excludeBins=[], qTstartBin = 1, qTmin=0, qTmax=300, doFit=True, rebin=1, xMin=-100, xMax=100, yMin=1e-6, yMax=1e4, propagate=True, bkgCfg={}, orderSigmas=False, yRatio = 1.15):
    
    cfgPlot = {

        'logy'              : True,
        'logx'              : False,
    
        'xmin'              : xMin,
        'xmax'              : xMax,
        'ymin'              : yMin,
        'ymax'              : yMax,
        
        'xtitle'            : "Recoil U_{#parallel} (GeV)" if comp == "para" else "Recoil U_{#perp}   (GeV)",
        'ytitle'            : "Events" ,
        
        'topRight'          : __topRight__, 
        'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",
        
        'ratiofraction'     : 0.25,
        'ytitleR'           : "Ratio" if ratio else "Pull",
        'yminR'             : (1-(yRatio-1)) if ratio else -2.5,
        'ymaxR'             : yRatio if ratio else 2.5,
    }
    
    recoil_bins = bhist.axes[1].edges
    bins = zfit.binned.VariableBinning(recoil_bins, name="recoil_para_qT")
    recoil = zfit.Space('recoil_para_qT', limits=(min(recoil_bins), max(recoil_bins)))
    recoil_binned = zfit.Space('recoil_para_qT', binning=bins)
    
    
    functions.prepareDir(outDir_, True) # remove and recreate dir
    nGauss = len(fitCfg['mean'])
    dof_param = (fitCfg['mean_cfg'] + fitCfg['sigma_cfg'] + fitCfg['norm_cfg']).count(3)
    dof = (fitCfg['mean_cfg'] + fitCfg['sigma_cfg'] + fitCfg['norm_cfg']).count(0) + dof_param # if cfg=0 or 3, add DOF



    # construct gauss
    mean_ptrs, sigma_ptrs, norm_ptrs, gauss_ptrs = [], [], [], []
    for iGauss in range(1, nGauss+1):
        if fitCfg['mean_cfg'][iGauss-1] >= 0:
            mean = zfit.Parameter("mean%d"%iGauss, fitCfg['mean'][iGauss-1], __min_mean__, __max_mean__, step_size=0.001)
            if fitCfg['mean_cfg'][iGauss-1] in [1, 2]: mean.floating = False
        else:
            dependent_tensor = mean_ptrs[abs(fitCfg['mean_cfg'][iGauss-1]-1)]
            mean = zfit.ComposedParameter("mean%d"%iGauss, dependent_tensor)
        sigma = zfit.Parameter("sigma%d"%iGauss, fitCfg['sigma'][iGauss-1], __min_sigma__, __max_sigma__, step_size=0.001)
        if fitCfg['sigma_cfg'][iGauss-1] in [1, 2]: sigma.floating = False
        gauss = zfit.pdf.Gauss(obs=recoil, mu=mean, sigma=sigma, name="gauss%d"%iGauss)
        gauss_binned = zfit.pdf.BinnedFromUnbinnedPDF(gauss, recoil_binned)
        mean_ptrs.append(mean)
        sigma_ptrs.append(sigma)
        gauss_ptrs.append(gauss_binned)
    
    # identical norms for each gauss
    for iGauss in range(1, nGauss):
        norm = zfit.Parameter("norm%d"%iGauss, fitCfg['norm'][iGauss-1], 0, 1, step_size=0.001)
        if fitCfg['norm_cfg'][iGauss-1] in [1, 2]: norm.floating = False
        norm_ptrs.append(norm)

    
    # make the TF1s 
    if dof_param > 0 and funcJs == None: sys.exit("Parametric fit expected but input json not defined!")
    funcs = {}
    if funcJs != None:
        for iGauss in range(1, nGauss+1):
            if "mean%d" % iGauss in funcJs:
                funcs["mean%d" % iGauss] = ROOT.TF1("mean%d" % iGauss, funcJs['mean%d' % iGauss]['func'], 0, 500)
                for iParam in range(0, funcJs['mean%d' % iGauss]['nParams']): funcs["mean%d" % iGauss].SetParameter(iParam, funcJs['mean%d' % iGauss]['p%d' % iParam])
            if "sigma%d" % iGauss in funcJs:
                funcs["sigma%d" % iGauss] = ROOT.TF1("sigma%d" % iGauss, funcJs['sigma%d' % iGauss]['func'], 0, 500)
                for iParam in range(0, funcJs['sigma%d' % iGauss]['nParams']): funcs["sigma%d" % iGauss].SetParameter(iParam, funcJs['sigma%d' % iGauss]['p%d' % iParam])

        for iGauss in range(1, nGauss):
            if "norm%d" % iGauss in funcJs:
                funcs["norm%d" % iGauss] = ROOT.TF1("norm%d" % iGauss, funcJs['norm%d' % iGauss]['func'], 0, 500)
                for iParam in range(0, funcJs['norm%d' % iGauss]['nParams']): funcs["norm%d" % iGauss].SetParameter(iParam, funcJs['norm%d' % iGauss]['p%d' % iParam])
       
       
    g_chi2 = ROOT.TGraphErrors()
    g_yields = ROOT.TGraphErrors()
    model = zfit.pdf.BinnedSumPDF(gauss_ptrs, fracs=norm_ptrs, obs=recoil_binned, name="gauss")
    ##model_unbinned = zfit.pdf.BinnedSumPDF(gauss_ptrs, fracs=norm_ptrs, obs=recoil_binned, name="gauss")
    '''
    obs = zfit.Space('x', limits=(-10, 10), binning=100)

    mu = zfit.Parameter("mu", 2.4, -1, 5)
    sigma = zfit.Parameter("sigma", 1.3,  0, 5)

    gauss = zfit.pdf.Gauss(obs=obs, mu=mu, sigma=sigma)

    gausbinned = zfit.pdf.BinnedFromUnbinnedPDF(gauss, obs)

    somerandomdata = np.array(gausbinned.sample(1000).values())

    sampler = gausbinned.create_sampler(1000)

    sampler.sample_holder.assign(somerandomdata)

    nll = zfit.loss.BinnedNLL(model=gausbinned, data=sampler)
    '''
    
    zfit.settings.set_seed(1)
    # tf.random.set_seed(1234)
    
    s = hist.tag.Slicer()
    h =  bhist[{"qTbinned": s[complex(0,0):complex(0,0.5)]}]
    h =  h[{"qTbinned": s[0:hist.overflow:hist.sum]}] # remove the overflow in the sum
    data_ = zfit.data.BinnedData.from_hist(h)
    ##data_unbinned = data_.to_unbinned()
    data_vals = data_.values()
    data_vars = data_.variances()

    
    minimizer = zfit.minimize.Minuit() # mode=0, gradient=True tol=1e-4, verbosity=5, gradient=True
    #minimizer = zfit.minimize.ScipyTrustConstrV1() # tol=1e-4, verbosity=5, gradient=True


    # sampler trick
    somerandomdata = np.array(model.sample(1000).values())
    sampler = model.create_sampler(1000)
    sampler.holder.variances = data_vars
    sampler.sample_holder.assign(somerandomdata)
    #print(sampler.holder.variances)
    
    #sys.exit()
    
    
    start = time.time()
    print("##### START BinnedNLL")
    nll = zfit.loss.BinnedNLL(model=model, data=sampler)
    #nll = zfit.loss.BinnedNLL(model=model, data=data_)
    end = time.time()
    print("##### END BinnedNLL, DURATION", (end-start))
    
    start = time.time()
    print("##### START minimize")
    result = minimizer.minimize(nll)
    end = time.time()
    print("##### END minimize, DURATION", (end-start))
    
    
            
            
    outDict = {}
    outDict['nGauss'] = nGauss
    recoil_qTbins_ = [0]
 
    if qTstartBin >= len(recoil_qTbins) or qTstartBin < 1: sys.exit("qTstartBin not in range (should be at least 1)")
    if recoil_qTbins[qTstartBin-1] < qTmin: sys.exit("qTstartBin is lower than the minimum range defined")
    if recoil_qTbins[qTstartBin] > qTmax: sys.exit("qTstartBin is higher than the maximum range defined")
    qTbin = qTstartBin
    direction = 1 if qTstartBin == 1 else -1
    #nll = None
    while True:
        
        if qTbin >= len(recoil_qTbins): break
        qT = 0.5*(recoil_qTbins[qTbin-1] + recoil_qTbins[qTbin])
        qTlow, qThigh = recoil_qTbins[qTbin-1], recoil_qTbins[qTbin]
        if qTlow >= qTmax: break
        
        print("##### DO BIN", qTbin, qT, qTlow, "->",  qThigh)

        s = hist.tag.Slicer()
        h =  bhist[{"qTbinned": s[complex(0,qTlow):complex(0,qThigh)]}]
        h =  h[{"qTbinned": s[0:hist.overflow:hist.sum]}] # remove the overflow in the sum
        
        #h_root = narf.hist_to_root(h_)
        
        #for i in range(0, h_root.GetNbinsX()+1):
        #    print(i, h_root.GetBinContent(i), h_root.GetBinError(i))
        #    h_root.SetBinError(i, h_root.GetBinError(i)*10.)
        #    h_root.SetBinError(i, 1e-6)
        #    h_root.SetBinContent(i, h_root.GetBinContent(i)*10.)
        #h = narf.root_to_hist(h_root, axis_names=["recoil_para_qT"])
        
        yield_ = h.sum().value
        yield_err = math.sqrt(h.sum().variance)
        data__ = zfit.data.BinnedData.from_hist(h)
        data_vals = data__.values()
        data_vars = data__.variances()
        
        #print(data_vals)
        #print(data_vars)
        #sys.exit()
        #data__.holder.variances = data__.holder.variances*10
        #print(data__.holder.variances)
        
        
        # set the variable according to the functional forms
        for iGauss in range(1, nGauss+1):
            if "mean%d" % iGauss in funcs:
                if not (fitCfg['mean_cfg'][iGauss-1] < 0 and iGauss != 1):
                    mean_ptrs[iGauss-1].set_value(funcs["mean%d" % iGauss].Eval(qT))
            elif not propagate:
                if not (fitCfg['mean_cfg'][iGauss-1] < 0 and iGauss != 1):
                    mean_ptrs[iGauss-1].set_value(fitCfg['mean'][iGauss-1])
            if "sigma%d" % iGauss in funcs: 
                sigma_ptrs[iGauss-1].set_value(funcs["sigma%d" % iGauss].Eval(qT))
            elif not propagate:    
                sigma_ptrs[iGauss-1].set_value(fitCfg['sigma'][iGauss-1])
            
        for iGauss in range(1, nGauss):
            if "norm%d" % iGauss in funcs:
                norm_ptrs[iGauss-1].set_value(funcs["norm%d" % iGauss].Eval(qT))
            elif not propagate:
                norm_ptrs[iGauss-1].set_value(fitCfg['norm'][iGauss-1])
        
        
        
        

        fitValid = True
        if dof > 0 and yield_ > 0:
            
            start = time.time()
            print("##### START BinnedNLL")
            sampler.sample_holder.assign(data_vals)
            sampler.holder.variances = data_vars
            
            #data_.holder.values = data_vals
            #data_.holder.variances = data_vars
            
            # create new loss per point
            #nll = zfit.loss.BinnedNLL(model=model, data=data__) #BinnedChi2 BinnedNLL # , options={"subtr_const" : False}
            #nll_ = nll.create_new()
            
            # use initial nll, but update the data
            #if not nll:
            #    nll = zfit.loss.BinnedNLL(model=model, data=data__)
            #else: 
            #    nll = nll.create_new(data=data__)
            end = time.time()
            print("##### END BinnedNLL, DURATION", (end-start))

            
            
            start = time.time()
            print("##### START MIMIMIZATION")
            result = minimizer.minimize(loss=nll)
            end = time.time()
            print("##### END MIMIMIZATION, DURATION", (end-start))
            params = result.params
            print(result)
            print(params)
            #fitValid = (fitRes.covQual() == 3 and fitRes.status() == 0)
            
            start = time.time()
            print("##### START HESSE")
            param_errors = result.hesse(name='hesse')
            end = time.time()
            print("##### END HESSE, DURATION", (end-start))
            
        else: fitValid = True
        
        
        if direction > 0 and qTbin == qTstartBin and not qTstartBin == 1:  # avoid plotting/processing twice the start bin
            qTbin += direction
            continue
        

        ## plot
        plotter.cfg = cfgPlot
        canvas, padT, padB = plotter.canvasRatio()
        dummyT, dummyB, dummyL = plotter.dummyRatio(line=1 if ratio else 0)
        
        canvas.cd()
        padT.Draw()
        padT.cd()
        padT.SetGrid()
        padT.SetTickx()
        padT.SetTicky()
        dummyT.Draw("HIST")
        
        hist_root = narf.hist_to_root(h)
        hist_root.Scale(1./yield_, "width")
        hist_root.SetLineColor(ROOT.kBlack)
        hist_root.SetMarkerStyle(20)
        hist_root.SetMarkerColor(ROOT.kBlack)
        hist_root.SetLineColor(ROOT.kBlack)
        hist_root.Draw("PE SAME")

        centers = bhist.axes[1].centers
        pdf_values = zfit.run(model.pdf(centers, norm_range=recoil_binned))
        g_pdf = ROOT.TGraphErrors()
        for i in range(0, len(centers)): g_pdf.SetPoint(i, centers[i], pdf_values[i])
        g_pdf.SetLineColor(ROOT.kBlue)
        g_pdf.SetLineWidth(3)
        g_pdf.Draw("L SAME")
        
        histRatio = hist_root.Clone("ratio")
        histRatio.Reset("ACE")
        histRatio.SetMarkerStyle(8)
        histRatio.SetMarkerSize(0.7)
        histRatio.SetMarkerColor(ROOT.kBlack)
        histRatio.SetLineColor(ROOT.kBlack)
        histRatio.SetLineWidth(1)
        chi2, histRatio = ratioHist_(hist_root, histRatio, g_pdf)
        

        latex = ROOT.TLatex()
        latex.SetNDC()
        latex.SetTextSize(0.040)
        latex.SetTextColor(1)
        latex.SetTextFont(42)
        latex.DrawLatex(0.20, 0.85, label)
        latex.DrawLatex(0.20, 0.80, "q_{T} = [%.1f, %.1f] GeV" % (qTlow, qThigh))
        latex.DrawLatex(0.20, 0.75, "Mean = %.3f #pm %.3f" % (hist_root.GetMean(), hist_root.GetMeanError()))
        latex.DrawLatex(0.20, 0.70, "RMS = %.3f #pm %.3f" % (hist_root.GetRMS(), hist_root.GetRMSError()))
        latex.DrawLatex(0.20, 0.65, "Yield = %.3f #pm %.3f" % (yield_, yield_err))
        latex.DrawLatex(0.20, 0.60, "#chi^{2} = %.3f" % chi2)
        #if dof > 0 and yield_ > 0:
        #    latex.DrawLatex(0.20, 0.55, "#color[%d]{Cov. matrix quality = %d}" % (2 if fitRes.covQual() != 3 else 1, fitRes.covQual()))
        #    latex.DrawLatex(0.20, 0.50, "#color[%d]{MINUIT status = %d}" % (2 if fitRes.status() != 0 else 1, fitRes.status()))
 
        latex.SetTextSize(0.035)
        
        # save output
        outDict[qTbin] = {}
        outDict[qTbin]['yield'] = yield_
        outDict[qTbin]['yield_err'] = yield_err
        
        for i, s in enumerate(mean_ptrs):
            val = zfit.run(s)
            err = param_errors[s]['error'] if s in param_errors else 0
            latex.DrawLatex(0.7, 0.87-i*0.04, "#mu_{%d} = %.3f #pm %.3f" % (i+1, val, err))
            #if fitCfg['mean_cfg'][i] >= 0 else mean_ptrs[abs(fitCfg['mean_cfg'][i])-1].value())) 
            outDict[qTbin]['mean%d' % (i+1)], outDict[qTbin]['mean%d_err' % (i+1)] = val, err 
        for i, s in enumerate(sigma_ptrs):
            val = zfit.run(s)
            err = param_errors[s]['error'] if s in param_errors else 0
            latex.DrawLatex(0.7, 0.87-i*0.04-0.04*len(mean_ptrs), "#sigma_{%d} = %.3f #pm %.3f" % (i+1, val, err))
            outDict[qTbin]['sigma%d' % (i+1)], outDict[qTbin]['sigma%d_err' % (i+1)] = val, err
        for i, s in enumerate(norm_ptrs):
            val = zfit.run(s)
            err = param_errors[s]['error'] if s in param_errors else 0
            latex.DrawLatex(0.7, 0.87-i*0.04-0.04*len(mean_ptrs)*2, "n_{%d} = %.3f #pm %.3f" % (i+1, val, err))
            outDict[qTbin]['norm%d' % (i+1)], outDict[qTbin]['norm%d_err' % (i+1)] = val, err
        

        plotter.auxRatio()
        canvas.cd()
        padB.Draw()
        padB.cd()
        padB.SetGrid()
        dummyB.Draw("HIST")
        histRatio.Draw("SAME E0")
        dummyL.Draw("SAME")
        canvas.Modify()
        canvas.Update()
        canvas.Draw()
        canvas.SaveAs("%s/%03d_recoil.png" % (outDir_, qTbin))
        #canvas.SaveAs("%s/%03d_recoil.pdf" % (outDir_, qTbin))
        dummyB.Delete()
        dummyT.Delete()
        padT.Delete()
        padB.Delete()
        
        
        
        outDict[qTbin]['chi2'] = chi2
        g_chi2.SetPoint(qTbin-1, qT, chi2)
        g_yields.SetPoint(qTbin-1, qT, yield_)
        g_yields.SetPointError(qTbin-1, 0, yield_err)
        

        recoil_qTbins_.append(recoil_qTbins[qTbin])
        if qTlow == qTmin:
            qTbin = qTstartBin # redo the fit at qTstartBin for initial conditions
            if qTstartBin != 1: qTbin -= 1 # subtract 1, as 1 is added below with direction
            direction = +1
        qTbin += direction # update bin 
        
    
    outDict['qTbins'] = recoil_qTbins_
    with open("%s/results.json" % outDir_, "w") as outfile: json.dump(outDict, outfile, indent=4)
    plotChi2(g_chi2, "%s/chi2" % outDir_, label, "U_{#perp}" if comp == "perp" else "U_{#parallel}", xMin=qTmin, xMax=qTmax)
    plotYields(g_yields, "%s/yields" % outDir_, label, "U_{#perp}" if comp == "perp" else "U_{#parallel}", xMin=qTmin, xMax=qTmax)

    with open("%s/results.json" % outDir_) as f: outDict = json.load(f)
    doGlobalPlot(bhist, comp, outDict, recoil_qTbins_, outDir_, label, funcJs=funcJs, ratio=ratio, rebin=rebin, bkgCfg=bkgCfg, qTmin=qTmin, qTmax=qTmax, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, yRatio=yRatio)


def doGlobalPlot(bhist, comp, jsIn, recoil_qTbins, outDir_, label, funcJs=None, ratio=True, qTmin=0, qTmax=50, rebin=1, bkgCfg={}, xMin=-100, xMax=100, yMin=1e-6, yMax=1e4, yRatio = 1.15):
    
    recoil_bins = bhist.axes[1].edges
    #bins = zfit.binned.VariableBinning(recoil_bins, name="recoil_para_qT")
    #recoil = zfit.Space('recoil_para_qT', limits=(min(recoil_bins), max(recoil_bins)))
    #recoil_binned = zfit.Space('recoil_para_qT', binning=bins)
    
    nGauss = jsIn['nGauss']
    
    totYield = 0
    totHist = None
    s = hist.tag.Slicer()
    for qTbin in range(1, len(recoil_qTbins)):

        totYield += jsIn[str(qTbin)]['yield']
        
        qTmin_, qTmax_ = recoil_qTbins[qTbin-1], recoil_qTbins[qTbin]
        h = bhist[{"qTbinned": s[complex(0,qTmin_):complex(0,qTmax_)]}]
        h = h[{"qTbinned": s[0:hist.overflow:hist.sum]}] # remove the overflow in the sum
        h = narf.hist_to_root(h)
        if totHist == None: totHist = h.Clone("totHist")
        else: totHist.Add(h)
    
    nGauss = jsIn['nGauss']
    recoil = ROOT.RooRealVar("recoil", "", 0, __min_recoil__, __max_recoil__)

    ptrs = {}
    garbage = [] # need to store the variables for memory issues
    means_arglist, sigmas_arglist, norms_arglist = [], [], []
    
    
    

    pdfs_arglist_tot, norms_arglist_tot = ROOT.RooArgList(), ROOT.RooArgList()
    pdfs_arglist_tot.setName("pdfs_arglist_tot")
    norms_arglist_tot.setName("norms_arglist_tot")
    for qTbin in range(1, len(recoil_qTbins)):
    
        qT = 0.5*(recoil_qTbins[qTbin-1] + recoil_qTbins[qTbin])
        if qT > qTmax or qT < qTmin: continue
        qTmin_, qTmax_ = recoil_qTbins[qTbin-1], recoil_qTbins[qTbin]
        yield_ = jsIn[str(qTbin)]['yield']
        

        pdfs, norms = ROOT.RooArgList(), ROOT.RooArgList()
        pdfs.setName("pdfs_bin%d" % qTbin)
        norms.setName("norms_bin%d" % qTbin)
        for iGauss in range(1, nGauss+1):
            mean_ = ROOT.RooRealVar("mean%d_bin%d" % (iGauss, qTbin), "", jsIn[str(qTbin)]['mean%d' % iGauss], __min_mean__, __max_mean__)
            sigma_ = ROOT.RooRealVar("sigma%d_bin%d" % (iGauss, qTbin), "", jsIn[str(qTbin)]['sigma%d' % iGauss], __min_sigma__, __max_sigma__)
            gauss = ROOT.RooGaussian("gauss%d_bin%d" % (iGauss, qTbin), "", recoil, mean_, sigma_)
            pdfs.addOwned(gauss)
            garbage.append(mean_)
            garbage.append(sigma_)
            garbage.append(gauss)

        for iGauss in range(1, nGauss):
            norm_ = ROOT.RooRealVar("norm%d_bin%d" % (iGauss, qTbin), "", jsIn[str(qTbin)]['norm%d' % iGauss], 0, 1)
            norms.addOwned(norm_)
            garbage.append(norm_)
        
        garbage.append(pdfs)
        garbage.append(norms)
        pdf_sig = ROOT.RooAddPdf("pdf_sig_bin%d" % (qTbin), '', pdfs, norms)
        ptrs[pdf_sig.GetName()] = pdf_sig
        
        # include backgrounds (if configured)
        pdfs_arglist_, norms_arglist_ = ROOT.RooArgList(), ROOT.RooArgList()
        pdfs_arglist_.setName("pdfs_arglist_bin%d" % (qTbin))
        norms_arglist_.setName("norms_arglist_bin%d" % (qTbin))
        for bkg in bkgCfg:
            with open(bkgCfg[bkg]['filename']) as f: jsIn_bkg = json.load(f)
            pdfs_bkgs_arglist, norms_bkgs_arglist = ROOT.RooArgList(), ROOT.RooArgList()
            pdfs_bkgs_arglist.setName("pdfs_bkgs_arglist_bin%d_%s" % (qTbin, bkg))
            norms_bkgs_arglist.setName("norms_bkgs_arglist_bin%d_%s" % (qTbin, bkg))
            for iGauss in range(1, jsIn_bkg['nGauss']+1):
                mean = ROOT.RooRealVar("mean%d_bin%d_%s" % (iGauss, qTbin, bkg), "", jsIn_bkg[str(qTbin)]["mean%d" % iGauss], __min_mean__, __max_mean__)
                sigma = ROOT.RooRealVar("sigma%d_bin%d_%s" % (iGauss, qTbin, bkg), "", jsIn_bkg[str(qTbin)]["sigma%d" % iGauss], __min_sigma__, __max_sigma__)
                mean.setConstant(ROOT.kTRUE)
                sigma.setConstant(ROOT.kTRUE)
                gauss = ROOT.RooGaussian("gauss%d_bin%d_%s" % (iGauss, qTbin, bkg), "", recoil, mean, sigma)
                garbage.append(mean)
                garbage.append(sigma)
                garbage.append(gauss)
                pdfs_bkgs_arglist.addOwned(gauss)
            for iGauss in range(1, jsIn_bkg['nGauss']):
                norm = ROOT.RooRealVar("norm%d_bin%d_%s" % (iGauss, qTbin, bkg), "", jsIn_bkg[str(qTbin)]["norm%d" % iGauss], 0, 1)
                norm.setConstant(ROOT.kTRUE)
                norms_bkgs_arglist.addOwned(norm)
                garbage.append(norm)
            pdf_bkg = ROOT.RooAddPdf("bin%d_%s" % (qTbin, bkg), '', pdfs_bkgs_arglist, norms_bkgs_arglist)
            pdfs_arglist_.addOwned(pdf_bkg)
            norm_bkg = ROOT.RooRealVar("relyield_bin%d_%s" % (qTbin, bkg), "", (jsIn_bkg[str(qTbin)]['yield']*bkgCfg[bkg]['norm'])/yield_, 0, 1)
            if not bkgCfg[bkg]['float']: norm_bkg.setConstant(ROOT.kTRUE)
            norms_arglist_.addOwned(norm_bkg)
            garbage.append(pdfs_bkgs_arglist)
            garbage.append(norms_bkgs_arglist)
        
        garbage.append(pdfs_arglist_)
        garbage.append(norms_arglist_)
        pdfs_arglist_.addClone(pdf_sig) # append signal as the last one
        pdf_tot_ = ROOT.RooAddPdf("pdf_tot_bin%d" % (qTbin), '', pdfs_arglist_, norms_arglist_)
        garbage.append(pdf_tot_)
        ptrs[pdf_tot_.GetName()] = pdf_tot_
        
        norm_bkg = ROOT.RooRealVar("totyield_bin%d" % (qTbin), "", jsIn[str(qTbin)]['yield']/totYield)
        norms_arglist_tot.addOwned(norm_bkg)
        pdfs_arglist_tot.addOwned(pdf_tot_)
        garbage.append(norm_bkg)
        
        

    pdf_tot = ROOT.RooAddPdf("pdf_tot", '', pdfs_arglist_tot, norms_arglist_tot)
    #totHist.Scale(1./totHist.Integral(), "width")
    
 
    
    centers = bhist.axes[1].centers
    g_pdf = ROOT.TGraphErrors()
    for i in range(1, totHist.GetNbinsX()+1):
        recoil.setVal(totHist.GetBinLowEdge(i))
        pdfLow = pdf_tot.getVal(recoil)
        recoil.setVal(totHist.GetBinLowEdge(i+1))
        pdfHigh = pdf_tot.getVal(recoil)
        #g_pdf.SetPoint(i-1, totHist.GetBinCenter(i), 0.5*(pdfLow+pdfHigh))
        
        recoil.setVal(totHist.GetBinCenter(i))
        g_pdf.SetPoint(i-1, totHist.GetBinCenter(i), pdf_tot.getVal(recoil))
    g_pdf.SetLineColor(ROOT.kBlue)
    g_pdf.SetLineWidth(3)
    
    totHist.SetLineColor(ROOT.kBlack)
    totHist.SetMarkerStyle(20)
    totHist.SetMarkerColor(ROOT.kBlack)
    totHist.SetLineColor(ROOT.kBlack)
    totHist.Scale(1./totHist.Integral(), "width")
    
    histRatio = totHist.Clone("ratio")
    histRatio.Reset("ACE")
    histRatio.SetMarkerStyle(8)
    histRatio.SetMarkerSize(0.7)
    histRatio.SetMarkerColor(ROOT.kBlack)
    histRatio.SetLineColor(ROOT.kBlack)
    histRatio.SetLineWidth(1)
    chi2, histRatio = ratioHist_(totHist, histRatio, g_pdf)
    
    
    
   
    #rdh_tot = ROOT.RooDataHist("rdh_tot", "", ROOT.RooArgList(recoil), ROOT.RooFit.Import(totHist))
    
    doUnc = funcJs != None and 'nStatVars' in funcJs # uncertainties from the refit
    pdf_tot_vars = []
    if doUnc:
    
        histUnc = totHist.Clone("histUnc")
        histUnc.SetFillColor(ROOT.kBlack)
        histUnc.SetMarkerSize(0)
        histUnc.SetLineWidth(0)
        histUnc.SetFillStyle(3004)    
        histUnc.Reset("ICE")
        
        histRatioUnc = totHist.Clone("histRatioUnc")
        histRatioUnc.SetFillColor(ROOT.kBlack)
        histRatioUnc.SetMarkerSize(0)
        histRatioUnc.SetLineWidth(0)
        histRatioUnc.SetFillStyle(3004)    
        histRatioUnc.Reset("ICE")

        nStatVars = funcJs['nStatVars']
        for nStat in range(0, nStatVars):
            nStatName = "stat%d" % nStat
            #if nStat != 0: continue
            # get the functions
            funcs_mean, funcs_sigma, funcs_norm = [], [], []
            for iGauss in range(1, nGauss+1):
                func_mean = ROOT.TF1("mean%d_%s" % (iGauss, nStatName), funcJs['mean%d' % iGauss]['func'], 0, 500)
                for iParam in range(0, funcJs['mean%d' % iGauss]['nParams']): func_mean.SetParameter(iParam, funcJs[nStatName]['mean%d' % iGauss]['p%d' % iParam])
                funcs_mean.append(func_mean)
                func_sigma = ROOT.TF1("sigma%d_%s" % (iGauss, nStatName), funcJs['sigma%d' % iGauss]['func'], 0, 500)
                for iParam in range(0, funcJs['sigma%d' % iGauss]['nParams']): func_sigma.SetParameter(iParam, funcJs[nStatName]['sigma%d' % iGauss]['p%d' % iParam])
                funcs_sigma.append(func_sigma)
            for iGauss in range(1, nGauss):
                func_norm = ROOT.TF1("norm%d_%s" % (iGauss, nStatName), funcJs['norm%d' % iGauss]['func'], 0, 500)
                for iParam in range(0, funcJs['norm%d' % iGauss]['nParams']): func_norm.SetParameter(iParam, funcJs[nStatName]['norm%d' % iGauss]['p%d' % iParam])
                funcs_norm.append(func_norm)
                    
            pdfs_arglist_tot_var, norms_arglist_tot_var = ROOT.RooArgList(), ROOT.RooArgList()
            pdfs_arglist_tot_var.setName("pdfs_arglist_tot_%s" % nStatName)
            norms_arglist_tot_var.setName("norms_arglist_tot_%s" % nStatName)
            for qTbin in range(1, len(recoil_qTbins)):
            
                qT = 0.5*(recoil_qTbins[qTbin-1] + recoil_qTbins[qTbin])
                if qT > qTmax or qT < qTmin: continue
                qTmin_, qTmax_ = recoil_qTbins[qTbin-1], recoil_qTbins[qTbin]
                
                pdfs, norms = ROOT.RooArgList(), ROOT.RooArgList()
                pdfs.setName("pdfs_bin%d_%s" % (qTbin, nStatName))
                norms.setName("norms_bin%d_%s" % (qTbin, nStatName))
                for iGauss in range(1, nGauss+1):
                    mean_ = ROOT.RooRealVar("mean%d_bin%d_%s" % (iGauss, qTbin, nStatName), "", funcs_mean[iGauss-1].Eval(qT), __min_mean__, __max_mean__)
                    sigma_ = ROOT.RooRealVar("sigma%d_bin%d_%s" % (iGauss, qTbin, nStatName), "", funcs_sigma[iGauss-1].Eval(qT), __min_sigma__, __max_sigma__)
                    gauss = ROOT.RooGaussian("gauss%d_bin%d_%s" % (iGauss, qTbin, nStatName), "", recoil, mean_, sigma_)
                    pdfs.addOwned(gauss)
                    garbage.append(mean_)
                    garbage.append(sigma_)
                    garbage.append(gauss)

                for iGauss in range(1, nGauss):
                    norm_ = ROOT.RooRealVar("norm%d_bin%d_%s" % (iGauss, qTbin, nStatName), "", funcs_norm[iGauss-1].Eval(qT), 0, 1)
                    norms.addOwned(norm_)
                    garbage.append(norm_)

                garbage.append(pdfs)
                garbage.append(norms)
                pdf_sig_var = ROOT.RooAddPdf("pdf_sig_bin%d_%s" % (qTbin, nStatName), '', pdfs, norms)
                ptrs[pdf_sig_var.GetName()] = pdf_sig_var
                
                # include backgrounds (if configured)
                pdfs_arglist_var_, norms_arglist_var_ = ROOT.RooArgList(), ROOT.RooArgList()
                pdfs_arglist_var_.setName("pdfs_arglist_bin%d_%s" % (qTbin, nStatName))
                norms_arglist_var_.setName("norms_arglist_bin%d_%s" % (qTbin, nStatName))
                for bkg in bkgCfg:
                    with open(bkgCfg[bkg]['filename']) as f: jsIn_bkg = json.load(f)
                    pdfs_bkgs_arglist, norms_bkgs_arglist = ROOT.RooArgList(), ROOT.RooArgList()
                    pdfs_bkgs_arglist.setName("pdfs_bkgs_arglist_bin%d_%s_%s" % (qTbin, bkg, nStatName))
                    norms_bkgs_arglist.setName("norms_bkgs_arglist_bin%d_%s_%s" % (qTbin, bkg, nStatName))
                    for iGauss in range(1, jsIn_bkg['nGauss']+1):
                        mean = ROOT.RooRealVar("mean%d_bin%d_%s_%s" % (iGauss, qTbin, bkg, nStatName), "", jsIn_bkg[str(qTbin)]["mean%d" % iGauss], __min_mean__, __max_mean__)
                        sigma = ROOT.RooRealVar("sigma%d_bin%d_%s_%s" % (iGauss, qTbin, bkg, nStatName), "", jsIn_bkg[str(qTbin)]["sigma%d" % iGauss], __min_sigma__, __max_sigma__)
                        mean.setConstant(ROOT.kTRUE)
                        sigma.setConstant(ROOT.kTRUE)
                        gauss = ROOT.RooGaussian("gauss%d_bin%d_%s_%s" % (iGauss, qTbin, bkg, nStatName), "", recoil, mean, sigma)
                        garbage.append(mean)
                        garbage.append(sigma)
                        garbage.append(gauss)
                        pdfs_bkgs_arglist.addOwned(gauss)
                    for iGauss in range(1, jsIn_bkg['nGauss']):
                        norm = ROOT.RooRealVar("norm%d_bin%d_%s_%s" % (iGauss, qTbin, bkg, nStatName), "", jsIn_bkg[str(qTbin)]["norm%d" % iGauss], 0, 1)
                        norm.setConstant(ROOT.kTRUE)
                        norms_bkgs_arglist.addOwned(norm)
                        garbage.append(norm)
                    pdf_bkg = ROOT.RooAddPdf("bin%d_%s_%s" % (qTbin, bkg, nStatName), '', pdfs_bkgs_arglist, norms_bkgs_arglist)
                    pdfs_arglist_.addOwned(pdf_bkg)
                    norm_bkg = ROOT.RooRealVar("relyield_bin%d_%s_%s" % (qTbin, bkg, nStatName), "", (jsIn_bkg[str(qTbin)]['yield']*bkgCfg[bkg]['norm'])/yield_, 0, 1)
                    if not bkgCfg[bkg]['float']: norm_bkg.setConstant(ROOT.kTRUE)
                    norms_arglist_.addOwned(norm_bkg)
                    garbage.append(pdfs_bkgs_arglist)
                    garbage.append(norms_bkgs_arglist)
                
                garbage.append(pdfs_arglist_var_)
                garbage.append(norms_arglist_var_)
                pdfs_arglist_var_.addClone(pdf_sig_var) # append signal as the last one
                pdf_tot_var_ = ROOT.RooAddPdf("pdf_tot_bin%d_%s" % (qTbin, nStatName), '', pdfs_arglist_var_, norms_arglist_var_)
                garbage.append(pdf_tot_var_)
                ptrs[pdf_tot_var_.GetName()] = pdf_tot_var_
                
                norm_bkg_var = ROOT.RooRealVar("totyield_bin%d_%s" % (qTbin, nStatName), "", jsIn[str(qTbin)]['yield']/totYield)
                norms_arglist_tot_var.addOwned(norm_bkg_var)
                pdfs_arglist_tot_var.addOwned(pdf_tot_var_)
                garbage.append(norm_bkg_var)
                
            pdf_tot_var = ROOT.RooAddPdf("pdf_tot_%s" % nStatName, '', pdfs_arglist_tot_var, norms_arglist_tot_var)
            pdf_tot_vars.append(pdf_tot_var)
            garbage.append(pdf_tot_var) 
            
            tmp = ROOT.RooArgSet(recoil)
            for i in range(1, histRatioUnc.GetNbinsX()+1):

                recoil.setVal(histRatioUnc.GetBinCenter(i))
                pdf_eval_unc = pdf_tot_var.getVal(tmp)
                pdf_eval = pdf_tot.getVal(tmp)
                if(pdf_eval > 0): 
                    err = abs(pdf_eval_unc - pdf_eval) 
                    #print(i, histRatioUnc.GetBinCenter(i), histRatioUnc.GetBinContent(i), pdf_eval_unc, pdf_eval, err)
                    histRatioUnc.SetBinError(i, histRatioUnc.GetBinError(i) + err*err)
                    histUnc.SetBinError(i, histUnc.GetBinError(i) + err*err)
                else: err = 0
    
        tmp = ROOT.RooArgSet(recoil)
        for i in range(1, histRatioUnc.GetNbinsX()+1): 
            recoil.setVal(histRatioUnc.GetBinCenter(i))
            pdf_eval = pdf_tot.getVal(tmp)
            if pdf_eval > 0: 
                histRatioUnc.SetBinError(i, histRatioUnc.GetBinError(i)**0.5/pdf_eval)
                histUnc.SetBinError(i, histUnc.GetBinError(i)**0.5)
            histRatioUnc.SetBinContent(i, 1)
            histUnc.SetBinContent(i, pdf_eval)
            #print(i, histRatioUnc.GetBinCenter(i), histRatioUnc.GetBinError(i), histUnc.GetBinError(i))
     

    cfgPlot = {

        'logy'              : True,
        'logx'              : False,
    
        'xmin'              : xMin,
        'xmax'              : xMax,
        'ymin'              : yMin,
        'ymax'              : yMax,
        
        'xtitle'            : "Recoil U_{#parallel} (GeV)" if comp == "para" else "Recoil U_{#perp}   (GeV)",
        'ytitle'            : "Events" ,
        
        'topRight'          : __topRight__,
        'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",
        
        'ratiofraction'     : 0.25,
        'ytitleR'           : "Ratio" if ratio else "Pull",
        'yminR'             : (1-(yRatio-1)) if ratio else -2.5,
        'ymaxR'             : yRatio if ratio else 2.5,
    }
    
    
    # plot
    plotter.cfg = cfgPlot
    canvas, padT, padB = plotter.canvasRatio()
    dummyT, dummyB, dummyL = plotter.dummyRatio(line=1 if ratio else 0)
        
    canvas.cd()
    padT.Draw()
    padT.cd()
    padT.SetTickx()
    padT.SetTicky()
    padT.SetGrid()
    dummyT.Draw("HIST")
    totHist.Draw("PE0 SAME")
    g_pdf.Draw("L SAME")    
    
    
    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.040)
    latex.SetTextColor(1)
    latex.SetTextFont(42)
    latex.DrawLatex(0.20, 0.85, label)
    latex.DrawLatex(0.20, 0.80, "#chi^{2} = %.3f" % chi2)
    plotter.auxRatio()

    canvas.cd()
    padB.Draw()
    padB.cd()
    padB.SetGrid()
    dummyB.Draw("HIST")
    histRatio.Draw("SAME E0")
    if doUnc: histRatioUnc.Draw("E2 SAME")
    dummyL.Draw("SAME")
    canvas.Modify()
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs("%s/global.png" % outDir_)
    canvas.SaveAs("%s/global.pdf" % outDir_)
    dummyB.Delete()
    dummyT.Delete()
    padT.Delete()
    padB.Delete()    

def doGlobalPlot__(bhist, comp, jsIn, recoil_qTbins, outDir_, label, funcJs=None, ratio=True, qTmin=0, qTmax=50, rebin=1, bkgCfg={}, xMin=-100, xMax=100, yMin=1e-6, yMax=1e4, yRatio = 1.15):

    
    recoil_bins = bhist.axes[1].edges
    bins = zfit.binned.VariableBinning(recoil_bins, name="recoil_para_qT")
    recoil = zfit.Space('recoil_para_qT', limits=(min(recoil_bins), max(recoil_bins)))
    recoil_binned = zfit.Space('recoil_para_qT', binning=bins)
    
    nGauss = jsIn['nGauss']
    
    totYield = 0
    totHist = None
    s = hist.tag.Slicer()
    for qTbin in range(1, len(recoil_qTbins)):

        totYield += jsIn[str(qTbin)]['yield']
        
        qTmin_, qTmax_ = recoil_qTbins[qTbin-1], recoil_qTbins[qTbin]
        h = bhist[{"qTbinned": s[complex(0,qTmin_):complex(0,qTmax_)]}]
        h = h[{"qTbinned": s[0:hist.overflow:hist.sum]}] # remove the overflow in the sum
        h = narf.hist_to_root(h)
        if totHist == None: totHist = h.Clone("totHist")
        else: totHist.Add(h)
        

    
    pdfs, norms, garbage = [], [], []
    for qTbin in range(1, len(recoil_qTbins)):
        yield_ = jsIn[str(qTbin)]['yield']
        
        # construct gauss
        norm_ptrs, gauss_ptrs = [], []
        for iGauss in range(1, nGauss+1):
            mean = zfit.Parameter("qTbin%d_mean%d"%(qTbin, iGauss), jsIn[str(qTbin)]['mean%d' % iGauss], __min_mean__, __max_mean__, step_size=0.001)
            sigma = zfit.Parameter("qTbin%d_sigma%d"%(qTbin, iGauss), jsIn[str(qTbin)]['sigma%d' % iGauss], __min_sigma__, __max_sigma__, step_size=0.001)
            gauss = zfit.pdf.Gauss(obs=recoil, mu=mean, sigma=sigma, name="gauss%d"%iGauss)
            gauss_binned = zfit.pdf.BinnedFromUnbinnedPDF(gauss, recoil_binned)
            #mean_ptrs.append(mean)
            #sigma_ptrs.append(sigma)
            gauss_ptrs.append(gauss_binned)
        
        # identical norms for each gauss
        for iGauss in range(1, nGauss):
            norm = zfit.Parameter("qTbin%d_norm%d"%(qTbin, iGauss), jsIn[str(qTbin)]['norm%d' % iGauss], 0, 1, step_size=0.001)
            norm_ptrs.append(norm)
        
        pdf_sig = zfit.pdf.BinnedSumPDF(gauss_ptrs, fracs=norm_ptrs, obs=recoil_binned, name="pdf_sig_qTbin%d"%qTbin)
        #ptrs["pdf_sig_qTbin%d"%qTbin] = pdf_sig
        

        norm_sig = zfit.Parameter("norm_sig_qTbin%d"%qTbin, yield_/totYield, 0, 1, step_size=0.001)
        pdfs.append(pdf_sig)
        norms.append(norm_sig)
      
    if len(pdfs) == 1: pdf_tot = pdfs[0]
    else: pdf_tot = zfit.pdf.BinnedSumPDF(pdfs, fracs=norms[:-1], obs=recoil_binned, name="pdf_tot")
    
    centers = bhist.axes[1].centers
    pdf_values = zfit.run(pdf_tot.pdf(centers, norm_range=recoil_binned))
    g_pdf = ROOT.TGraphErrors()
    for i in range(0, len(centers)): g_pdf.SetPoint(i, centers[i], pdf_values[i])
    g_pdf.SetLineColor(ROOT.kBlue)
    g_pdf.SetLineWidth(3)
    
        
        
    totHist.SetLineColor(ROOT.kBlack)
    totHist.SetMarkerStyle(20)
    totHist.SetMarkerColor(ROOT.kBlack)
    totHist.SetLineColor(ROOT.kBlack)
    totHist.Scale(1./totHist.Integral(), "width")
    
        
    histRatio = totHist.Clone("ratio")
    histRatio.Reset("ACE")
    histRatio.SetMarkerStyle(8)
    histRatio.SetMarkerSize(0.7)
    histRatio.SetMarkerColor(ROOT.kBlack)
    histRatio.SetLineColor(ROOT.kBlack)
    histRatio.SetLineWidth(1)
    chi2, histRatio = ratioHist_(totHist, histRatio, g_pdf)
    
    
    

    
    
    doUnc = funcJs != None and 'nStatVars' in funcJs # uncertainties from the refit
    pdf_tot_vars = []
    if False:
    
        histUnc = totHist.Clone("histUnc")
        histUnc.SetFillColor(ROOT.kBlack)
        histUnc.SetMarkerSize(0)
        histUnc.SetLineWidth(0)
        histUnc.SetFillStyle(3004)    
        histUnc.Reset("ICE")
        
        histRatioUnc = totHist.Clone("histRatioUnc")
        histRatioUnc.SetFillColor(ROOT.kBlack)
        histRatioUnc.SetMarkerSize(0)
        histRatioUnc.SetLineWidth(0)
        histRatioUnc.SetFillStyle(3004)    
        histRatioUnc.Reset("ICE")

        nStatVars = funcJs['nStatVars']
        for nStat in range(0, nStatVars):
            nStatName = "stat%d" % nStat
            #if nStat != 0: continue
            # get the functions
            funcs_mean, funcs_sigma, funcs_norm = [], [], []
            for iGauss in range(1, nGauss+1):
                func_mean = ROOT.TF1("mean%d_%s" % (iGauss, nStatName), funcJs['mean%d' % iGauss]['func'], 0, 500)
                for iParam in range(0, funcJs['mean%d' % iGauss]['nParams']): func_mean.SetParameter(iParam, funcJs[nStatName]['mean%d' % iGauss]['p%d' % iParam])
                funcs_mean.append(func_mean)
                func_sigma = ROOT.TF1("sigma%d_%s" % (iGauss, nStatName), funcJs['sigma%d' % iGauss]['func'], 0, 500)
                for iParam in range(0, funcJs['sigma%d' % iGauss]['nParams']): func_sigma.SetParameter(iParam, funcJs[nStatName]['sigma%d' % iGauss]['p%d' % iParam])
                funcs_sigma.append(func_sigma)
            for iGauss in range(1, nGauss):
                func_norm = ROOT.TF1("norm%d_%s" % (iGauss, nStatName), funcJs['norm%d' % iGauss]['func'], 0, 500)
                for iParam in range(0, funcJs['norm%d' % iGauss]['nParams']): func_norm.SetParameter(iParam, funcJs[nStatName]['norm%d' % iGauss]['p%d' % iParam])
                funcs_norm.append(func_norm)
                    
            pdfs_arglist_tot_var, norms_arglist_tot_var = ROOT.RooArgList(), ROOT.RooArgList()
            pdfs_arglist_tot_var.setName("pdfs_arglist_tot_%s" % nStatName)
            norms_arglist_tot_var.setName("norms_arglist_tot_%s" % nStatName)
            for qTbin in range(1, len(recoil_qTbins)):
            
                qT = 0.5*(recoil_qTbins[qTbin-1] + recoil_qTbins[qTbin])
                if qT > qTmax or qT < qTmin: continue
                qTmin_, qTmax_ = recoil_qTbins[qTbin-1], recoil_qTbins[qTbin]
                
                pdfs, norms = ROOT.RooArgList(), ROOT.RooArgList()
                pdfs.setName("pdfs_bin%d_%s" % (qTbin, nStatName))
                norms.setName("norms_bin%d_%s" % (qTbin, nStatName))
                for iGauss in range(1, nGauss+1):
                    mean_ = ROOT.RooRealVar("mean%d_bin%d_%s" % (iGauss, qTbin, nStatName), "", funcs_mean[iGauss-1].Eval(qT), __min_mean__, __max_mean__)
                    sigma_ = ROOT.RooRealVar("sigma%d_bin%d_%s" % (iGauss, qTbin, nStatName), "", funcs_sigma[iGauss-1].Eval(qT), __min_sigma__, __max_sigma__)
                    gauss = ROOT.RooGaussian("gauss%d_bin%d_%s" % (iGauss, qTbin, nStatName), "", recoil, mean_, sigma_)
                    pdfs.addOwned(gauss)
                    garbage.append(mean_)
                    garbage.append(sigma_)
                    garbage.append(gauss)

                for iGauss in range(1, nGauss):
                    norm_ = ROOT.RooRealVar("norm%d_bin%d_%s" % (iGauss, qTbin, nStatName), "", funcs_norm[iGauss-1].Eval(qT), 0, 1)
                    norms.addOwned(norm_)
                    garbage.append(norm_)

                garbage.append(pdfs)
                garbage.append(norms)
                pdf_sig_var = ROOT.RooAddPdf("pdf_sig_bin%d_%s" % (qTbin, nStatName), '', pdfs, norms)
                ptrs[pdf_sig_var.GetName()] = pdf_sig_var
                
                # include backgrounds (if configured)
                pdfs_arglist_var_, norms_arglist_var_ = ROOT.RooArgList(), ROOT.RooArgList()
                pdfs_arglist_var_.setName("pdfs_arglist_bin%d_%s" % (qTbin, nStatName))
                norms_arglist_var_.setName("norms_arglist_bin%d_%s" % (qTbin, nStatName))
                for bkg in bkgCfg:
                    with open(bkgCfg[bkg]['filename']) as f: jsIn_bkg = json.load(f)
                    pdfs_bkgs_arglist, norms_bkgs_arglist = ROOT.RooArgList(), ROOT.RooArgList()
                    pdfs_bkgs_arglist.setName("pdfs_bkgs_arglist_bin%d_%s_%s" % (qTbin, bkg, nStatName))
                    norms_bkgs_arglist.setName("norms_bkgs_arglist_bin%d_%s_%s" % (qTbin, bkg, nStatName))
                    for iGauss in range(1, jsIn_bkg['nGauss']+1):
                        mean = ROOT.RooRealVar("mean%d_bin%d_%s_%s" % (iGauss, qTbin, bkg, nStatName), "", jsIn_bkg[str(qTbin)]["mean%d" % iGauss], __min_mean__, __max_mean__)
                        sigma = ROOT.RooRealVar("sigma%d_bin%d_%s_%s" % (iGauss, qTbin, bkg, nStatName), "", jsIn_bkg[str(qTbin)]["sigma%d" % iGauss], __min_sigma__, __max_sigma__)
                        mean.setConstant(ROOT.kTRUE)
                        sigma.setConstant(ROOT.kTRUE)
                        gauss = ROOT.RooGaussian("gauss%d_bin%d_%s_%s" % (iGauss, qTbin, bkg, nStatName), "", recoil, mean, sigma)
                        garbage.append(mean)
                        garbage.append(sigma)
                        garbage.append(gauss)
                        pdfs_bkgs_arglist.addOwned(gauss)
                    for iGauss in range(1, jsIn_bkg['nGauss']):
                        norm = ROOT.RooRealVar("norm%d_bin%d_%s_%s" % (iGauss, qTbin, bkg, nStatName), "", jsIn_bkg[str(qTbin)]["norm%d" % iGauss], 0, 1)
                        norm.setConstant(ROOT.kTRUE)
                        norms_bkgs_arglist.addOwned(norm)
                        garbage.append(norm)
                    pdf_bkg = ROOT.RooAddPdf("bin%d_%s_%s" % (qTbin, bkg, nStatName), '', pdfs_bkgs_arglist, norms_bkgs_arglist)
                    pdfs_arglist_.addOwned(pdf_bkg)
                    norm_bkg = ROOT.RooRealVar("relyield_bin%d_%s_%s" % (qTbin, bkg, nStatName), "", (jsIn_bkg[str(qTbin)]['yield']*bkgCfg[bkg]['norm'])/yield_, 0, 1)
                    if not bkgCfg[bkg]['float']: norm_bkg.setConstant(ROOT.kTRUE)
                    norms_arglist_.addOwned(norm_bkg)
                    garbage.append(pdfs_bkgs_arglist)
                    garbage.append(norms_bkgs_arglist)
                
                garbage.append(pdfs_arglist_var_)
                garbage.append(norms_arglist_var_)
                pdfs_arglist_var_.addClone(pdf_sig_var) # append signal as the last one
                pdf_tot_var_ = ROOT.RooAddPdf("pdf_tot_bin%d_%s" % (qTbin, nStatName), '', pdfs_arglist_var_, norms_arglist_var_)
                garbage.append(pdf_tot_var_)
                ptrs[pdf_tot_var_.GetName()] = pdf_tot_var_
                
                norm_bkg_var = ROOT.RooRealVar("totyield_bin%d_%s" % (qTbin, nStatName), "", jsIn[str(qTbin)]['yield']/totYield)
                norms_arglist_tot_var.addOwned(norm_bkg_var)
                pdfs_arglist_tot_var.addOwned(pdf_tot_var_)
                garbage.append(norm_bkg_var)
                
            pdf_tot_var = ROOT.RooAddPdf("pdf_tot_%s" % nStatName, '', pdfs_arglist_tot_var, norms_arglist_tot_var)
            pdf_tot_vars.append(pdf_tot_var)
            garbage.append(pdf_tot_var) 
            
            tmp = ROOT.RooArgSet(recoil)
            for i in range(1, histRatioUnc.GetNbinsX()+1):

                recoil.setVal(histRatioUnc.GetBinCenter(i))
                pdf_eval_unc = pdf_tot_var.getVal(tmp)
                pdf_eval = pdf_tot.getVal(tmp)
                if(pdf_eval > 0): 
                    err = abs(pdf_eval_unc - pdf_eval) 
                    #print(i, histRatioUnc.GetBinCenter(i), histRatioUnc.GetBinContent(i), pdf_eval_unc, pdf_eval, err)
                    histRatioUnc.SetBinError(i, histRatioUnc.GetBinError(i) + err*err)
                    histUnc.SetBinError(i, histUnc.GetBinError(i) + err*err)
                else: err = 0
    
        tmp = ROOT.RooArgSet(recoil)
        for i in range(1, histRatioUnc.GetNbinsX()+1): 
            recoil.setVal(histRatioUnc.GetBinCenter(i))
            pdf_eval = pdf_tot.getVal(tmp)
            if pdf_eval > 0: 
                histRatioUnc.SetBinError(i, histRatioUnc.GetBinError(i)**0.5/pdf_eval)
                histUnc.SetBinError(i, histUnc.GetBinError(i)**0.5)
            histRatioUnc.SetBinContent(i, 1)
            histUnc.SetBinContent(i, pdf_eval)
            #print(i, histRatioUnc.GetBinCenter(i), histRatioUnc.GetBinError(i), histUnc.GetBinError(i))
     

    cfgPlot = {

        'logy'              : True,
        'logx'              : False,
    
        'xmin'              : xMin,
        'xmax'              : xMax,
        'ymin'              : yMin,
        'ymax'              : yMax,
        
        'xtitle'            : "Recoil U_{#parallel} (GeV)" if comp == "para" else "Recoil U_{#perp}   (GeV)",
        'ytitle'            : "Events" ,
        
        'topRight'          : __topRight__,
        'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",
        
        'ratiofraction'     : 0.25,
        'ytitleR'           : "Ratio" if ratio else "Pull",
        'yminR'             : (1-(yRatio-1)) if ratio else -2.5,
        'ymaxR'             : yRatio if ratio else 2.5,
    }
    
    
    # plot
    plotter.cfg = cfgPlot
    canvas, padT, padB = plotter.canvasRatio()
    dummyT, dummyB, dummyL = plotter.dummyRatio(line=1 if ratio else 0)
        
    canvas.cd()
    padT.Draw()
    padT.cd()
    padT.SetTickx()
    padT.SetTicky()
    padT.SetGrid()
    dummyT.Draw("HIST")
    totHist.Draw("PE SAME")
    g_pdf.Draw("L SAME")

    
    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.040)
    latex.SetTextColor(1)
    latex.SetTextFont(42)
    latex.DrawLatex(0.20, 0.85, label)
    latex.DrawLatex(0.20, 0.80, "#chi^{2} = %.3f" % chi2)

    plotter.auxRatio()
    canvas.cd()
    padB.Draw()
    padB.cd()
    padB.SetGrid()
    dummyB.Draw("HIST")
    histRatio.Draw("SAME E0")
    ###if doUnc: histRatioUnc.Draw("E2 SAME")
    dummyL.Draw("SAME")
    canvas.Modify()
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs("%s/global.png" % outDir_)
    canvas.SaveAs("%s/global.pdf" % outDir_)
    dummyB.Delete()
    dummyT.Delete()
    padT.Delete()
    padB.Delete()    


def plotChi2(g_chi2, fOut, label, recoilComp, xMin=0, xMax=100, yMin=0, yMax=3):

    cfg_chi2 = {

        'logy'              : False,
        'logx'              : False,
        
        'xmin'              : xMin,
        'xmax'              : xMax,
        'ymin'              : yMin,
        'ymax'              : yMax,
            
        'xtitle'            : "q_{T} (GeV)",
        'ytitle'            : "#chi^{2}",
            
        'topRight'          : __topRight__, 
        'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",

    } 
    
    g_chi2.SetLineColor(ROOT.kRed)
    g_chi2.SetMarkerStyle(8)
    g_chi2.SetMarkerSize(1)
    g_chi2.SetMarkerColor(ROOT.kRed)
    
    f_chi2 = ROOT.TF1("f_chi2", "[0]*x + [1]", 0, 50)
    f_chi2.SetParameters(0, 1)
    f_chi2.SetLineColor(ROOT.kBlue)
    f_chi2.SetLineWidth(2)
    g_chi2.Fit("f_chi2", "NSE", "", 0, 50)
    
    plotter.cfg = cfg_chi2
    canvas = plotter.canvas()
    dummy = plotter.dummy()   
    canvas.cd()
    dummy.Draw("HIST")
    g_chi2.Draw("SAME LP")
    f_chi2.Draw("SAME L")
    plotter.aux()
    
    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.040)
    latex.SetTextColor(1)
    latex.SetTextFont(42)
    latex.DrawLatex(0.20, 0.88, label)
    latex.DrawLatex(0.20, 0.82, recoilComp)
    
    canvas.Modify()
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs("%s.png" % fOut)
    canvas.SaveAs("%s.pdf" % fOut)
    canvas.Delete()


def ratioHist_(hNom, hRatio, pdf):

    chi2 = 0
    nBins = 0
    for i in range(1, hRatio.GetNbinsX()+1):
            
        xLow , xHigh = hRatio.GetBinLowEdge(i), hRatio.GetBinLowEdge(i+1)
        pdfLow, pdfHigh = pdf.Eval(xLow), pdf.Eval(xHigh)
        pdfVal = 0.5*(pdfLow + pdfHigh)
        pdfVal = pdf.Eval(hRatio.GetBinCenter(i))
        y, y_err = 0, 0
        if(pdfVal > 0): 
            y = hNom.GetBinContent(i)/pdfVal
            y_err = hNom.GetBinError(i)/pdfVal
        hRatio.SetBinContent(i, y)
        hRatio.SetBinError(i, y_err)
    
        if hNom.GetBinContent(i) > 0:
            chi2 += (((pdfVal - hNom.GetBinContent(i)) / hNom.GetBinError(i)) ** 2)
            nBins += 1
            
    chi2 /= nBins      
    return chi2, hRatio
    

def ratioHist(hNom, hRatio, pdf, recoil, norm):

    for i in range(1, hRatio.GetNbinsX()+1):
            
        recoil.setVal(hRatio.GetBinLowEdge(i))
        pdfLow = pdf.getVal(norm)
        recoil.setVal(hRatio.GetBinLowEdge(i+1))
        pdfHigh = pdf.getVal(norm)
        pdf_eval = 0.5*(pdfLow + pdfHigh)
        y, y_err = 0, 0
        if(pdf_eval > 0): 
            y = hNom.GetBinContent(i)/pdf_eval
            y_err = hNom.GetBinError(i)/pdf_eval
        hRatio.SetBinContent(i, y)
        hRatio.SetBinError(i, y_err)
        #print(i, hRatio.GetBinLowEdge(i), hRatio.GetBinCenter(i), hRatio.GetBinLowEdge(i+1), hRatio.GetBinContent(i), pdfLow, pdfHigh, pdf_eval)
    return hRatio

def plotYields(g_yields, fOut, label, recoilComp, xMin=0, xMax=100, yMin=0, yMax=3):

    cfg_chi2 = {

        'logy'              : False,
        'logx'              : False,
        
        'xmin'              : xMin,
        'xmax'              : xMax,
        'ymin'              : 0,
        'ymax'              : 1.3*max(g_yields.GetY()),
            
        'xtitle'            : "q_{T} (GeV)",
        'ytitle'            : "Event yield",
            
        'topRight'          : __topRight__, 
        'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",

    } 
    
    g_yields.SetLineColor(ROOT.kRed)
    g_yields.SetMarkerStyle(8)
    g_yields.SetMarkerSize(1)
    g_yields.SetMarkerColor(ROOT.kRed)
    
    
    plotter.cfg = cfg_chi2
    canvas = plotter.canvas()
    dummy = plotter.dummy()   
    canvas.cd()
    dummy.Draw("HIST")
    g_yields.Draw("SAME LP")
    plotter.aux()
    
    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.040)
    latex.SetTextColor(1)
    latex.SetTextFont(42)
    latex.DrawLatex(0.20, 0.88, label)
    latex.DrawLatex(0.20, 0.82, recoilComp)
    
    canvas.Modify()
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs("%s.png" % fOut)
    canvas.SaveAs("%s.pdf" % fOut)
    canvas.Delete()






def doPlot(comp, param, jsIn, outDir, recoil_qTbins, label, xMin=0, xMax=150, yMin=0, yMax=30, yTitle=""):


    f_base = ROOT.TF1("f_base", jsIn[param]['func'], 0, 500)
    for i in range(0, jsIn[param]['nParams']): f_base.SetParameter(i, jsIn[param]['p%d' % i])
    f_unc = ROOT.TF1("f_unc", jsIn[param]['func'], 0, 500)
    for i in range(0, jsIn[param]['nParams']): f_unc.SetParameter(i, jsIn[param]['p%d' % i])
    
    f_base.SetLineColor(ROOT.kBlack)

    g = ROOT.TGraphErrors()
    #g.SetLineColor(ROOT.kBlack)
    #g.SetMarkerStyle(20)
    #g.SetMarkerSize(0.9)
    #g.SetMarkerColor(ROOT.kBlack)
    #g.SetLineColor(ROOT.kBlack)
    
           
    iPoint = 0
    for qTbin in range(1, len(recoil_qTbins)):
        x = 0.5*(recoil_qTbins[qTbin-1] + recoil_qTbins[qTbin])
        x_err = 0.
        y = f_base.Eval(x)
        
        # loop over all stat uncertainties
        y_err = 0
        for iStat in range(0, jsIn['nStatVars']):
            for i in range(0, jsIn[param]['nParams']): 
                if param in jsIn['stat%d' % iStat]: f_unc.SetParameter(i, jsIn['stat%d' % iStat][param]['p%d' % i])
                else : f_unc.SetParameter(i, jsIn[param]['p%d' % i])
            up = f_unc.Eval(x)
            s = up - y
            y_err += s**2
        y_err = y_err**(0.5)    
        #print(x, y, y_err)
        
        g.SetPoint(iPoint, x, y)   
        g.SetPointError(iPoint, 0, y_err)
        #print(x, y)
        iPoint += 1


    cfg = {

        'logy'              : False,
        'logx'              : False,
        
        'xmin'              : xMin,
        'xmax'              : xMax,
        'ymin'              : yMin,
        'ymax'              : yMax,
            
        'xtitle'            : "q_{T} (GeV)",
        'ytitle'            : yTitle,
            
        'topRight'          : __topRight__, 
        'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",
            
        'ratiofraction'     : 0.25,
        'ytitleR'           : "Pull",
        'yminR'             : -2.5, # 0.88
        'ymaxR'             : 2.5, # 1.12
    }         

    plotter.cfg = cfg
    canvas, padT, padB = plotter.canvasRatio()
    dummyT, dummyB, dummyL = plotter.dummyRatio()
        
    ## TOP PAD ##
    canvas.cd()
    padT.Draw()
    padT.cd()
    padT.SetTickx()
    padT.SetTicky()
    dummyT.Draw("HIST")
  
    
    g.SetFillColor(ROOT.kGreen+1)
    g.Draw("3SAME")
    f_base.Draw("L SAME")
    plotter.auxRatio()
    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.045)
    latex.SetTextColor(1)
    latex.SetTextFont(42)
    latex.DrawLatex(0.18, 0.85, label)
    latex.DrawLatex(0.18, 0.80, "U_{#parallel}" if comp == "para" else "U_{#perp}")

    ## BOTTOM PAD ##
    canvas.cd()
    padB.Draw()
    padB.cd()
    dummyB.Draw("HIST")
        


    line = ROOT.TLine(cfg['xmin'], 0, cfg['xmax'], 0)
    line.SetLineColor(ROOT.kBlue+2)
    line.SetLineWidth(2)
    line.Draw("SAME")

    canvas.Modify()
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs("%s/%s_refit.png" % (outDir, param))
    canvas.SaveAs("%s/%s_refit.pdf" % (outDir, param))
    canvas.Delete()
    
 




def doFit(jsIn, jsOut, comp, param, fitFuncName, iParams, outDir, recoil_qTbins, label, cParams=[], excludeBins=[], fitMin=0, fitMax=200, xMin=0, xMax=150, yMin=0, yMax=30, yTitle="", doFit=True, cutOffMin=-99999, cutOffMax=99999, yRatio = 1.15):

    fitF = getattr(rf, fitFuncName + "_")()
    fit = ROOT.TF1("fit", fitF, 0, 500) if fitF != "" else None
    if cParams == []: cParams = [False]*len(iParams)
    for i, iPar in enumerate(iParams):
        if cParams[i]: fit.FixParameter(i, iPar)
        else: fit.SetParameter(i, iPar)


    g = ROOT.TGraphErrors()
    g.SetLineColor(ROOT.kBlack)
    g.SetMarkerStyle(20)
    g.SetMarkerSize(0.55)
    g.SetMarkerColor(ROOT.kBlack)
    g.SetLineColor(ROOT.kBlack)
           
    iPoint = 0
    for qTbin in range(1, len(recoil_qTbins)):
        if not str(qTbin) in jsIn:
            #print("WARNING: qTbin %s not found, skip it" % qTbin)
            continue
        if qTbin in excludeBins: continue
        x = 0.5*(recoil_qTbins[qTbin-1] + recoil_qTbins[qTbin])
        x_err = 0.5*(recoil_qTbins[qTbin] - recoil_qTbins[qTbin-1])
        if not param in jsIn[str(qTbin)]:
            print("WARNING: param %s not found for qTbin %d" % (param, qTbin))
            continue
        y = jsIn[str(qTbin)][param]
        y_err = jsIn[str(qTbin)][param + "_err"]
        
        if y < cutOffMin: continue
        if y > cutOffMax: continue
        g.SetPoint(iPoint, x, y)   
        g.SetPointError(iPoint, x_err, y_err)
        iPoint += 1

  
    if doFit and fit:
        result = g.Fit(fit.GetName(), "NSE", "", fitMin, fitMax) 
        fit.SetLineColor(ROOT.kRed)
        #fit.GetXaxis().SetRangeUser(0, 200)
        fit.SetLineWidth(2)

        
        cov = result.GetCorrelationMatrix()
        cov.Print()
        
        values = result.GetConfidenceIntervals(0.68, False)
        uncBand = ROOT.TGraphErrors()
        uncBand_ratio = ROOT.TGraphErrors()
        for i in range(len(values)):
            uncBand.SetPoint(i, g.GetX()[i], fit.Eval(g.GetX()[i]))
            uncBand.SetPointError(i, 0, values[i])
            uncBand_ratio.SetPoint(i, g.GetX()[i], 0)
            uncBand_ratio.SetPointError(i, 0, values[i]/fit.Eval(g.GetX()[i]))

        # ratio 
        g_ratio = ROOT.TGraphErrors()
        g_ratio.SetLineColor(ROOT.kBlack)
        g_ratio.SetMarkerStyle(20)
        g_ratio.SetMarkerSize(0.55)
        g_ratio.SetMarkerColor(ROOT.kBlack)
        g_ratio.SetLineColor(ROOT.kBlack)
            
        iPoint = 0
        for qTbin in range(1, len(recoil_qTbins)):
            if not str(qTbin) in jsIn:
                #print("WARNING: qTbin %s not found, skip it" % qTbin)
                continue
            x = 0.5*(recoil_qTbins[qTbin-1] + recoil_qTbins[qTbin])
            x_err = 0.5*(recoil_qTbins[qTbin] - recoil_qTbins[qTbin-1])
            if not param in jsIn[str(qTbin)]: continue
            y = jsIn[str(qTbin)][param]
            y_err = jsIn[str(qTbin)][param + "_err"]
            y_fit = fit.Eval(x)
            #print(x, y, y_fit, y/y_fit)
            if y_err > 0: 
                pull = (y-y_fit)/y_err
                ratio = y / y_fit
                ratio_err = y_err/abs(y)
            else:
                pull = 0
                ratio = 1.0
                ratio_err = 0.0
            g_ratio.SetPoint(iPoint, x, ratio)
            g_ratio.SetPointError(iPoint, 0, ratio_err)
            iPoint += 1
     
    cfg = {

        'logy'              : False,
        'logx'              : False,
        
        'xmin'              : xMin,
        'xmax'              : xMax,
        'ymin'              : yMin,
        'ymax'              : yMax,
            
        'xtitle'            : "q_{T} (GeV)",
        'ytitle'            : yTitle,
            
        'topRight'          : __topRight__, 
        'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",
            
        'ratiofraction'     : 0.25,
        'ytitleR'           : "Ratio",
        'yminR'             : (1-(yRatio-1)),
        'ymaxR'             : yRatio,
    }         

    plotter.cfg = cfg
    canvas, padT, padB = plotter.canvasRatio()
    dummyT, dummyB, dummyL = plotter.dummyRatio()
        
    ## TOP PAD ##
    canvas.cd()
    padT.Draw()
    padT.cd()
    padT.SetTickx()
    padT.SetTicky()
    dummyT.Draw("HIST")
  
    if doFit and fit:
        uncBand.SetFillColor(ROOT.kGreen+1)
        uncBand.Draw("3SAME")
    g.Draw("PE SAME")
    if fit: fit.Draw("L SAME")  

    plotter.auxRatio()
    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.045)
    latex.SetTextColor(1)
    latex.SetTextFont(42)
    latex.DrawLatex(0.18, 0.85, label)
    latex.DrawLatex(0.18, 0.80, "U_{#parallel}" if comp == "para" else "U_{#perp}")
    
    for i in range(0, len(iParams)):
        latex.DrawLatex(0.65, 0.85-i*0.05, "a_{%d} = %.3f #pm %.3f" % (i, fit.GetParameter(i), fit.GetParError(i)))
        
    ## BOTTOM PAD ##
    canvas.cd()
    padB.Draw()
    padB.cd()
    dummyB.Draw("HIST")
        
    if doFit and fit:
        #uncBand_ratio.SetFillColor(ROOT.kGreen+1)
        #uncBand_ratio.Draw("3SAME")
        g_ratio.Draw("PE SAME")

    line = ROOT.TLine(cfg['xmin'], 1, cfg['xmax'], 1)
    line.SetLineColor(ROOT.kBlue+2)
    line.SetLineWidth(2)
    line.Draw("SAME")

    canvas.Modify()
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs("%s/%s.png" % (outDir, param))
    canvas.SaveAs("%s/%s.pdf" % (outDir, param))
    canvas.Delete()
    
    jsOut[param] = {}
    jsOut[param]["func"] = fitF
    jsOut[param]["funcName"] = fitFuncName
    jsOut[param]["nParams"] = len(iParams)
    for i, iPar in enumerate(iParams):
        jsOut[param]["p%d" % i] = fit.GetParameter(i)
        jsOut[param]["p%d_err" % i] = fit.GetParError(i)
        jsOut[param]["p%d_isCte" % i] = cParams[i]
 
    
def doFit_new(jsIn, jsOut, comp, param, fitF, iParams, outDir, recoil_qTbins, label, cParams=[], excludeBins=[], fitMin=0, fitMax=200, xMin=0, xMax=150, yMin=0, yMax=30, yTitle="", doFit=True, cutOffMin=-99999, cutOffMax=99999):

    fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [7.40516e-02, 9.99276e-01, 1.27330e+01], [False, False, False]
    

    bins = zfit.binned.VariableBinning(recoil_qTbins, name="recoil_qTbins")
    obs = zfit.Space('recoil_qTbins', binning=bins)
    
    
    coeff1 = zfit.Parameter("coeff1", 0)
    coeff2 = zfit.Parameter("coeff2", 0)
    
    poly = zfit.models.polynomials.RecursivePolynomial(obs, [coeff1, coeff2])
    
    
    sys.exit()
    fit = ROOT.TF1("fit", fitF, 0, 500) if fitF != "" else None
    if cParams == []: cParams = [False]*len(iParams)
    for i, iPar in enumerate(iParams):
        if cParams[i]: fit.FixParameter(i, iPar)
        else: fit.SetParameter(i, iPar)


    g = ROOT.TGraphErrors()
    g.SetLineColor(ROOT.kBlack)
    g.SetMarkerStyle(20)
    g.SetMarkerSize(0.9)
    g.SetMarkerColor(ROOT.kBlack)
    g.SetLineColor(ROOT.kBlack)
           
    iPoint = 0
    for qTbin in range(1, len(recoil_qTbins)):
        if not str(qTbin) in jsIn:
            #print("WARNING: qTbin %s not found, skip it" % qTbin)
            continue
        if qTbin in excludeBins: continue
        x = 0.5*(recoil_qTbins[qTbin-1] + recoil_qTbins[qTbin])
        x_err = 0.5*(recoil_qTbins[qTbin] - recoil_qTbins[qTbin-1])
        if not param in jsIn[str(qTbin)]:
            print("WARNING: param %s not found for qTbin %d" % (param, qTbin))
            continue
        y = jsIn[str(qTbin)][param]
        y_err = jsIn[str(qTbin)][param + "_err"]
        
        if y < cutOffMin: continue
        if y > cutOffMax: continue
        g.SetPoint(iPoint, x, y)   
        g.SetPointError(iPoint, x_err, y_err)
        iPoint += 1

  
    if doFit and fit:
        result = g.Fit(fit.GetName(), "NSE", "", fitMin, fitMax) 
        fit.SetLineColor(ROOT.kRed)
        #fit.GetXaxis().SetRangeUser(0, 200)
        fit.SetLineWidth(2)

        
        cov = result.GetCorrelationMatrix()
        cov.Print()
        
        values = result.GetConfidenceIntervals(0.68, False)
        uncBand = ROOT.TGraphErrors()
        uncBand_ratio = ROOT.TGraphErrors()
        for i in range(len(values)):
            uncBand.SetPoint(i, g.GetX()[i], fit.Eval(g.GetX()[i]))
            uncBand.SetPointError(i, 0, values[i])
            uncBand_ratio.SetPoint(i, g.GetX()[i], 0)
            uncBand_ratio.SetPointError(i, 0, values[i]/fit.Eval(g.GetX()[i]))

        # ratio 
        g_ratio = ROOT.TGraphErrors()
        g_ratio.SetLineColor(ROOT.kBlack)
        g_ratio.SetMarkerStyle(20)
        g_ratio.SetMarkerSize(0.9)
        g_ratio.SetMarkerColor(ROOT.kBlack)
        g_ratio.SetLineColor(ROOT.kBlack)
            
        iPoint = 0
        for qTbin in range(1, len(recoil_qTbins)):
            if not str(qTbin) in jsIn:
                #print("WARNING: qTbin %s not found, skip it" % qTbin)
                continue
            x = 0.5*(recoil_qTbins[qTbin-1] + recoil_qTbins[qTbin])
            x_err = 0.5*(recoil_qTbins[qTbin] - recoil_qTbins[qTbin-1])
            if not param in jsIn[str(qTbin)]: continue
            y = jsIn[str(qTbin)][param]
            y_err = jsIn[str(qTbin)][param + "_err"]
            y_fit = fit.Eval(x)
            #print(x, y, y_fit, y/y_fit)
            if y_err > 0: pull = (y-y_fit)/y_err
            else: pull = 0
            g_ratio.SetPoint(iPoint, x, pull)
            #g_ratio.SetPointError(iPoint, x_err, y_err/abs(y_fit))
            iPoint += 1
     
    cfg = {

        'logy'              : False,
        'logx'              : False,
        
        'xmin'              : xMin,
        'xmax'              : xMax,
        'ymin'              : yMin,
        'ymax'              : yMax,
            
        'xtitle'            : "q_{T} (GeV)",
        'ytitle'            : yTitle,
            
        'topRight'          : __topRight__, 
        'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",
            
        'ratiofraction'     : 0.25,
        'ytitleR'           : "Pull",
        'yminR'             : -2.5, # 0.88
        'ymaxR'             : 2.5, # 1.12
    }         

    plotter.cfg = cfg
    canvas, padT, padB = plotter.canvasRatio()
    dummyT, dummyB, dummyL = plotter.dummyRatio()
        
    ## TOP PAD ##
    canvas.cd()
    padT.Draw()
    padT.cd()
    padT.SetTickx()
    padT.SetTicky()
    dummyT.Draw("HIST")
  
    if doFit and fit:
        uncBand.SetFillColor(ROOT.kGreen+1)
        uncBand.Draw("3SAME")
    g.Draw("PE SAME")
    if fit: fit.Draw("L SAME")  

    plotter.auxRatio()
    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.045)
    latex.SetTextColor(1)
    latex.SetTextFont(42)
    latex.DrawLatex(0.18, 0.85, label)
    latex.DrawLatex(0.18, 0.80, "U_{#parallel}" if comp == "para" else "U_{#perp}")
    
    for i in range(0, len(iParams)):
        latex.DrawLatex(0.65, 0.85-i*0.05, "a_{%d} = %.3f #pm %.3f" % (i, fit.GetParameter(i), fit.GetParError(i)))
        
    ## BOTTOM PAD ##
    canvas.cd()
    padB.Draw()
    padB.cd()
    dummyB.Draw("HIST")
        
    if doFit and fit:
        uncBand_ratio.SetFillColor(ROOT.kGreen+1)
        uncBand_ratio.Draw("3SAME")
        g_ratio.Draw("PE SAME")

    line = ROOT.TLine(cfg['xmin'], 0, cfg['xmax'], 0)
    line.SetLineColor(ROOT.kBlue+2)
    line.SetLineWidth(2)
    line.Draw("SAME")

    canvas.Modify()
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs("%s/%s.png" % (outDir, param))
    canvas.SaveAs("%s/%s.pdf" % (outDir, param))
    canvas.Delete()
    
    jsOut[param] = {}
    jsOut[param]["func"] = fitF
    jsOut[param]["nParams"] = len(iParams)
    for i, iPar in enumerate(iParams):
        jsOut[param]["p%d" % i] = fit.GetParameter(i)
        jsOut[param]["p%d_err" % i] = fit.GetParError(i)
        jsOut[param]["p%d_isCte" % i] = cParams[i]


  
 
def combinedFit(bhist, comp, fitCfg, jsIn, recoil_qTbins, qTmin=0, qTmax=50, rebin=1, singleMean=False, bkgCfg={}):
    
    recoil_bins = bhist.axes[1].edges
    bins = zfit.binned.VariableBinning(recoil_bins, name="recoil_para_qT")
    recoil = zfit.Space('recoil_para_qT', limits=(min(recoil_bins), max(recoil_bins)))
    recoil_binned = zfit.Space('recoil_para_qT', binning=bins)
    
    nGauss = jsIn['nGauss']
    
    '''
    # construct gauss
    mean_ptrs, sigma_ptrs, norm_ptrs, gauss_ptrs = [], [], [], []
    for iGauss in range(1, nGauss+1):
        if fitCfg['mean_cfg'][iGauss-1] >= 0:
            mean = zfit.Parameter("mean%d"%iGauss, fitCfg['mean'][iGauss-1], __min_mean__, __max_mean__, step_size=0.001)
            if fitCfg['mean_cfg'][iGauss-1] in [1, 2]: mean.floating = False
        else:
            dependent_tensor = mean_ptrs[abs(fitCfg['mean_cfg'][iGauss-1]-1)]
            mean = zfit.ComposedParameter("mean%d"%iGauss, dependent_tensor)
        sigma = zfit.Parameter("sigma%d"%iGauss, fitCfg['sigma'][iGauss-1], __min_sigma__, __max_sigma__, step_size=0.001)
        if fitCfg['sigma_cfg'][iGauss-1] in [1, 2]: sigma.floating = False
        gauss = zfit.pdf.Gauss(obs=recoil, mu=mean, sigma=sigma, name="gauss%d"%iGauss)
        gauss_binned = zfit.pdf.BinnedFromUnbinnedPDF(gauss, recoil_binned)
        mean_ptrs.append(mean)
        sigma_ptrs.append(sigma)
        gauss_ptrs.append(gauss_binned)
    
    # identical norms for each gauss
    for iGauss in range(1, nGauss):
        norm = zfit.Parameter("norm%d"%iGauss, fitCfg['norm'][iGauss-1], 0, 1, step_size=0.001)
        if fitCfg['norm_cfg'][iGauss-1] in [1, 2]: norm.floating = False
        norm_ptrs.append(norm)
    '''

    cats = ROOT.RooCategory("category", "") # for each qT bin, define category
    hists = ROOT.std.map("string, RooDataHist*")() # container holding all RooDataHists
    ptrs = {}
    fitFunctions = {}
    params_mean, params_sigma, params_norm, params_qT = {}, {}, {}, {}
    params = {}
    for iGauss in range(1, nGauss+1):

        params_ = []
        for iParam in range(0, jsIn['sigma%d' % iGauss]['nParams']):
            tmp = 'sigma%d_p%d' % (iGauss, iParam)
            tmp_ = 'p%d' % iParam
            param = zfit.Parameter(tmp, float(jsIn['sigma%d' % iGauss]['p%d' % iParam]), step_size=0.001)
            if jsIn['sigma%d' % iGauss]['p%d_isCte' % iParam]: param.floating = False
            params_.append(param)
        params_sigma["gauss%d" % iGauss] = params_
        params["sigma_gauss%d" % iGauss] = params_
    
        params_ = []
        for iParam in range(0, jsIn['mean%d' % iGauss]['nParams']):
            tmp = 'mean%d_p%d' % (iGauss, iParam)
            tmp_ = 'p%d' % iParam
            if fitCfg['mean_cfg'][iGauss-1] >= 0:
                param = zfit.Parameter(tmp, float(jsIn['mean%d' % iGauss]['p%d' % iParam]), step_size=0.001)
                if jsIn['mean%d' % iGauss]['p%d_isCte' % iParam]: param.floating = False
            else:
                dependent_tensor = params_[abs(fitCfg['mean_cfg'][iGauss-1]-1)]
                param = zfit.ComposedParameter(tmp, dependent_tensor)
            params_.append(param)
        params_mean["gauss%d" % iGauss] = params_
        params["mean_gauss%d" % iGauss] = params_

        fitF = jsIn['sigma%d' % iGauss]['func'].replace("Exp", "ABC").replace("x", "{0}").replace("ABC", "Exp") #.replace("Exp", "ABC").replace("x", "{0}").replace("[", "x[").replace("ABC", "Exp")
        fitFunctions['sigma%d' % iGauss] = fitF
        fitF = jsIn['mean%d' % iGauss]['func'].replace("Exp", "ABC").replace("x", "{0}").replace("ABC", "Exp") #.replace("Exp", "ABC").replace("x", "{0}").replace("[", "x[").replace("ABC", "Exp")
        fitFunctions['mean%d' % iGauss] = fitF
   

    for iGauss in range(1, nGauss):
        params_ = []
        for iParam in range(0, jsIn['norm%d' % iGauss]['nParams']):
            tmp = 'norm%d_p%d' % (iGauss, iParam)
            tmp_ = 'p%d' % iParam
            param = zfit.Parameter(tmp, float(jsIn['norm%d' % iGauss]['p%d' % iParam]), step_size=0.001)
            if jsIn['norm%d' % iGauss]['p%d_isCte' % iParam]: param.floating = False
            params_.append(param)
        params_norm["gauss%d" % iGauss] = params_
        params["norm_gauss%d" % iGauss] = params_
        
        fitF = jsIn['norm%d' % iGauss]['func'].replace("Exp", "ABC").replace("x", "{0}").replace("[", "x[").replace("ABC", "Exp")
        fitFunctions['norm%d' % iGauss] = fitF



    models, datas = [], []
    nll_simultaneous = None
    out = {}
    out['nGauss'] = nGauss
    nBins = 0
    for qTbin in range(1, len(recoil_qTbins)):
    
        if qTbin >= len(recoil_qTbins): break
        qT = 0.5*(recoil_qTbins[qTbin-1] + recoil_qTbins[qTbin])
        qTlow, qThigh = recoil_qTbins[qTbin-1], recoil_qTbins[qTbin]
        if qTlow >= qTmax: break
        print("##### DO BIN", qTbin, qT, qTlow, "->",  qThigh)
        nBins += 1
        
        qTp = zfit.Parameter("qT%s" % nBins, qT)
        qTp.floating = False
        params_qT["qT%s" % nBins] = qTp
        
        s = hist.tag.Slicer()
        h =  bhist[{"qTbinned": s[complex(0,qTlow):complex(0,qThigh)]}]
        h =  h[{"qTbinned": s[0:hist.overflow:hist.sum]}] # remove the overflow in the sum
        
        yield_ = h.sum().value
        yield_err = math.sqrt(h.sum().variance)
        data__ = zfit.data.BinnedData.from_hist(h)

        
        gauss_pdfs, gauss_norms = [], []      
        for iGauss in range(1, nGauss+1):
        
            # sigma
            func = getattr(rf, jsIn['sigma%d' % iGauss]['funcName'])
            params_ = [qTp] + params["sigma_gauss%d" % iGauss]
            p_sigma = zfit.ComposedParameter('sigma_gauss%d_qT%d' % (iGauss, qTbin), func, params=params_)

            # mean
            func = getattr(rf, jsIn['mean%d' % iGauss]['funcName'])
            params_ = [qTp] + params["mean_gauss%d" % iGauss]
            p_mean = zfit.ComposedParameter('mean_gauss%d_qT%d' % (iGauss, qTbin), func, params=params_)
            
            # construct gauss
            gauss = zfit.pdf.Gauss(obs=recoil, mu=p_mean, sigma=p_sigma, name="gauss%d"%iGauss)
            gauss_binned = zfit.pdf.BinnedFromUnbinnedPDF(gauss, recoil_binned)
            gauss_pdfs.append(gauss_binned)
            
            # norm
            if iGauss == nGauss: continue
            func = getattr(rf, jsIn['norm%d' % iGauss]['funcName'])
            params_ = [qTp] + params["norm_gauss%d" % iGauss]
            p_norm = zfit.ComposedParameter('norm_gauss%d_qT%d' % (iGauss, qTbin), func, params=params_)
            gauss_norms.append(p_norm)
       
        
        model = zfit.pdf.BinnedSumPDF(gauss_pdfs, fracs=gauss_norms, obs=recoil_binned, name="nll_gauss%d" % iGauss)
        models.append(model)
        datas.append(data__)
        
        continue
        
        
        print("Construct NLL")
        nll = zfit.loss.BinnedNLL(model=model, data=data__, options={"subtr_const" : False})
        nlls.append(nll)
        print("Done")
        
        if nll_simultaneous == None: nll_simultaneous = nll
        else: nll_simultaneous += nll

    
        #break
      
            
            
            

        
        '''
        # include backgrounds (if configured)
        pdfs_arglist_, norms_arglist_ = ROOT.RooArgList(), ROOT.RooArgList()
        pdfs_arglist_.setName("pdfs_arglist_bin%d" % (qTbin))
        norms_arglist_.setName("norms_arglist_bin%d" % (qTbin))
        for bkg in bkgCfg:
            with open(bkgCfg[bkg]['filename']) as f: jsIn_bkg = json.load(f)
            pdfs_bkgs_arglist, norms_bkgs_arglist = ROOT.RooArgList(), ROOT.RooArgList()
            pdfs_bkgs_arglist.setName("pdfs_bkgs_arglist_bin%d_%s" % (qTbin, bkg))
            norms_bkgs_arglist.setName("norms_bkgs_arglist_bin%d_%s" % (qTbin, bkg))
            for iGauss in range(1, jsIn_bkg['nGauss']+1):
                mean = ROOT.RooRealVar("mean%d_bin%d_%s" % (iGauss, qTbin, bkg), "", jsIn_bkg[str(qTbin)]["mean%d" % iGauss], __min_mean__, __max_mean__)
                sigma = ROOT.RooRealVar("sigma%d_bin%d_%s" % (iGauss, qTbin, bkg), "", jsIn_bkg[str(qTbin)]["sigma%d" % iGauss], __min_sigma__, __max_sigma__)
                mean.setConstant(ROOT.kTRUE)
                sigma.setConstant(ROOT.kTRUE)
                gauss = ROOT.RooGaussian("gauss%d_bin%d_%s" % (iGauss, qTbin, bkg), "", recoil, mean, sigma)
                garbage.append(mean)
                garbage.append(sigma)
                garbage.append(gauss)
                pdfs_bkgs_arglist.addOwned(gauss)
            for iGauss in range(1, jsIn_bkg['nGauss']):
                norm = ROOT.RooRealVar("norm%d_bin%d_%s" % (iGauss, qTbin, bkg), "", jsIn_bkg[str(qTbin)]["norm%d" % iGauss], 0, 1)
                norm.setConstant(ROOT.kTRUE)
                norms_bkgs_arglist.addOwned(norm)
                garbage.append(norm)
            pdf_bkg = ROOT.RooAddPdf("bin%d_%s" % (qTbin, bkg), '', pdfs_bkgs_arglist, norms_bkgs_arglist)
            pdfs_arglist_.addOwned(pdf_bkg)
            norm_bkg = ROOT.RooRealVar("relyield_bin%d_%s" % (qTbin, bkg), "", (jsIn_bkg[str(qTbin)]['yield']*bkgCfg[bkg]['norm'])/yield_, 0, 1)
            if not bkgCfg[bkg]['float']: norm_bkg.setConstant(ROOT.kTRUE)
            norms_arglist_.addOwned(norm_bkg)
            garbage.append(pdfs_bkgs_arglist)
            garbage.append(norms_bkgs_arglist)
        
        garbage.append(pdfs_arglist_)
        garbage.append(norms_arglist_)
        pdfs_arglist_.addClone(pdf_sig) # append signal as the last one
        pdf_tot_ = ROOT.RooAddPdf("pdf_tot_bin%d" % (qTbin), '', pdfs_arglist_, norms_arglist_)
        #garbage.append(pdf_tot_)
        ptrs[pdf_tot_.GetName()] = pdf_tot_
        pdf_tot_.Print()
        pdf_tot.addPdf(pdf_tot_, "cat_bin%s" % (qTbin))
        
        print("----------------------------------->", rdh.numEntries(), rdh.sumEntries())  
        '''
    
    # minimize
    
    start = time.time()
    print("##### START BinnedNLL")
    nll = zfit.loss.BinnedNLL(model=models, data=datas) # , options={"subtr_const" : False}
    end = time.time()
    print("##### END BinnedNLL, DURATION", (end-start))

    start = time.time()
    print("##### START MIMIMIZATION")
    minimizer = zfit.minimize.Minuit(verbosity=7)
    result = minimizer.minimize(nll)
    end = time.time()
    print("##### END MIMIMIZATION, DURATION", (end-start))
    params = result.params
    print(result)
    print(params)
            
    start = time.time()
    print("##### START HESSE")
    param_errors = result.hesse(name='hesse')
    end = time.time()
    print("##### END HESSE, DURATION", (end-start))
    
    
    sys.exit()
    
    #recoil.setBins(600) # seems to be necessary ??
    rdh_tot = ROOT.RooDataHist("rdh_tot", "", ROOT.RooArgList(recoil), cats, hists) # total RDH   , ROOT.RooFit.Minos(ROOT.kTRUE) , , ROOT.RooFit.Hesse(ROOT.kTRUE) 
    # ROOT.RooFit.SumW2Error(ROOT.kTRUE) --> errors calculated according to inverse Hessian method (see https://root-forum.cern.ch/t/using-roofit-with-weighted-data/20207/2)
    # ROOT.RooFit.Minos(ROOT.kTRUE) does not work with weighted values
    #fitRes = pdf_tot.fitTo(rdh_tot,  ROOT.RooFit.Save(ROOT.kTRUE), ROOT.RooFit.Minos(ROOT.kTRUE)) # ROOT.RooFit.SumW2Error(ROOT.kTRUE), # , ROOT.RooFit.Minos(ROOT.kTRUE)
    fitRes = pdf_tot.fitTo(rdh_tot,  ROOT.RooFit.Save(ROOT.kTRUE), ROOT.RooFit.SumW2Error(ROOT.kTRUE)) # ROOT.RooFit.AsymptoticError(ROOT.kTRUE) ROOT.RooFit.SumW2Error(ROOT.kTRUE)
    ## https://root.cern.ch/doc/master/classRooAbsPdf.html#a5f79f16f4a26a19c9e66fb5c080f59c5  ROOT.RooFit.AsymptoticError(ROOT.kTRUE) , ROOT.RooFit.Strategy(2)
    fitValid = (fitRes.covQual() == 3 and fitRes.status() == 0)
    print("****************************")
    print("FIT STATUS")
    print("Covariance Quality = %d" % fitRes.covQual())
    print("Fit status = %d" % fitRes.status())
    print("****************************")
    
    cov = fitRes.covarianceMatrix()
    cov.Print()
    
    #
    
    floatingParams = fitRes.floatParsFinal()
    jsOut = copy.deepcopy(jsIn)
    for iGauss in range(1, nGauss+1):
        for iParam in range(0, jsIn['sigma%d' % iGauss]['nParams']):
            tmp = 'sigma%d_p%d' % (iGauss, iParam)
            p = getParamIdx(tmp, floatingParams)
            print(p)
            if p == -1: val, err = ptrs[tmp].getVal(), 0
            else: val, err = floatingParams[p].getVal(), floatingParams[p].getError()
            jsOut['sigma%d' % iGauss]['p%d' % iParam] = val
            jsOut['sigma%d' % iGauss]['p%d_err' % iParam] = err
            print("%s \t %.3e" % (tmp, val))
          
        for iParam in range(0, jsIn['mean%d' % iGauss]['nParams']):
            tmp = 'mean%d_p%d' % (iGauss, iParam)
            p = getParamIdx(tmp, floatingParams)
            if p == -1: val, err = ptrs[tmp].getVal(), 0
            else: val, err = floatingParams[p].getVal(), floatingParams[p].getError()
            if fitCfg['mean_cfg'][iGauss-1] < 0:
                idx = abs(fitCfg['mean_cfg'][iGauss-1])
                tmp_1 = 'mean%d_p%d' % (idx, iParam)
                p = getParamIdx(tmp_1, floatingParams) # try first Gauss
                if p == -1: val, err = ptrs[tmp_1].getVal(), 0
                else: val, err = floatingParams[p].getVal(), floatingParams[p].getError()
            jsOut['mean%d' % iGauss]['p%d' % iParam] = val
            jsOut['mean%d' % iGauss]['p%d_err' % iParam] = err
            #print("%s \t %.3e" % (tmp, val))
    
    for iGauss in range(1, nGauss):
        for iParam in range(0, jsIn['norm%d' % iGauss]['nParams']):
            tmp = 'norm%d_p%d' % (iGauss, iParam)
            p = getParamIdx(tmp, floatingParams)
            if p == -1: val, err = ptrs[tmp].getVal(), 0
            else: val, err = floatingParams[p].getVal(), floatingParams[p].getError()
            jsOut['norm%d' % iGauss]['p%d' % iParam] = val
            jsOut['norm%d' % iGauss]['p%d_err' % iParam] = err
            #print("%s \t %.3e" % (tmp, val))
    
    fIn.Close()
    
    # diagonalize covariance matrix and store perturbations
    variations = diagonalize(fitRes, verbose=False)
    jsOut['nStatVars'] = len(variations)
    for iVar, var in enumerate(variations):
        
        #print(iVar, var)
        jsOut['stat%d' % iVar] = {}
        t = jsOut['stat%d' % iVar]
        
        for iGauss in range(1, nGauss+1):
            t['sigma%d' % iGauss] = {}
            for iParam in range(0, jsIn['sigma%d' % iGauss]['nParams']):
                tmp = 'sigma%d_p%d' % (iGauss, iParam)
                p = getParamIdx(tmp, floatingParams)
                if p == -1: val = ptrs[tmp].getVal()
                else: val = var[p]
                t['sigma%d' % iGauss]['p%d' % iParam] = val
            
            t['mean%d' % iGauss] = {}  
            for iParam in range(0, jsIn['mean%d' % iGauss]['nParams']):
                tmp = 'mean%d_p%d' % (iGauss, iParam)
                p = getParamIdx(tmp, floatingParams)
                if p == -1: val = ptrs[tmp].getVal()
                else: val = var[p]
                if fitCfg['mean_cfg'][iGauss-1] < 0:
                    idx = abs(fitCfg['mean_cfg'][iGauss-1])
                    tmp_1 = 'mean%d_p%d' % (idx, iParam)
                    p = getParamIdx(tmp_1, floatingParams) # try first Gauss
                    if p == -1: val = ptrs[tmp_1].getVal()
                    else: val = var[p]
                    #print(p, val)
                t['mean%d' % iGauss]['p%d' % iParam] = val

        for iGauss in range(1, nGauss):
            t['norm%d' % iGauss] = {}
            for iParam in range(0, jsIn['norm%d' % iGauss]['nParams']):
                tmp = 'norm%d_p%d' % (iGauss, iParam)
                p = getParamIdx(tmp, floatingParams)
                if p == -1: val = ptrs[tmp].getVal()
                else: val = var[p]
                t['norm%d' % iGauss]['p%d' % iParam] = val
 
    return jsOut
 

