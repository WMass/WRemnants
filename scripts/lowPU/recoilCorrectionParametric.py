
import sys,array,math,os,copy,decimal
import numpy as np
import ctypes
import json

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptTitle(0)

import functions
import plotter


from wremnants.datasets.datagroups import Datagroups

import lz4.frame
import pickle
import narf
import numpy as np

mirror = False


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



 
def doFitMultiGauss(proc, comp, fitCfg, label, outDir_, funcJs=None, qTmin=0, qTmax=300, doFit = True, rebin=1, xMin=-100, xMax=100, yMin=1e-6, yMax=1e4, singleMean=False, bkgCfg={}, orderSigmas=False):
    
    cfgPlot = {

        'logy'              : True,
        'logx'              : False,
    
        'xmin'              : xMin,
        'xmax'              : xMax,
        'ymin'              : yMin,
        'ymax'              : yMax,
        
        'xtitle'            : "Recoil U_{#parallel} (GeV)" if comp == "para" else "Recoil U_{#perp}   (GeV)",
        'ytitle'            : "Events" ,
        
        'topRight'          : "199 pb^{#minus1} (13 TeV)", 
        'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",
        
        'ratiofraction'     : 0.25,
        'ytitleR'           : "Pull",
        'yminR'             : -2.5,
        'ymaxR'             : 2.5,
    }
    
    #fIn = ROOT.TFile(fIn_)
    
    functions.prepareDir(outDir_, True) # remove and recreate dir
    recoil = ROOT.RooRealVar("recoil", "", 0, -300, 300) 
    nGauss = len(fitCfg['mean'])
    dof_param = (fitCfg['mean_cfg'] + fitCfg['sigma_cfg'] + fitCfg['norm_cfg']).count(3)
    dof = (fitCfg['mean_cfg'] + fitCfg['sigma_cfg'] + fitCfg['norm_cfg']).count(0) + dof_param # if cfg=0 or 3, add DOF

    # construct gauss
    pdfs, norms = ROOT.RooArgList(), ROOT.RooArgList()
    mean_ptrs, sigma_ptrs, norm_ptrs, gauss_ptrs = [], [], [], []
    for iGauss in range(1, nGauss+1):
        if singleMean and iGauss != 1:
            mean = ROOT.RooFormulaVar("mean%d" % iGauss, "@0", ROOT.RooArgList(mean_ptrs[0]))
        else:
            mean = ROOT.RooRealVar("mean%d" % iGauss, "", fitCfg['mean'][iGauss-1], -300, 300)
            if fitCfg['mean_cfg'][iGauss-1] in [1, 2]: mean.setConstant(ROOT.kTRUE)
        sigma = ROOT.RooRealVar("sigma%d" % iGauss, "", fitCfg['sigma'][iGauss-1], 2, 100)
        if fitCfg['sigma_cfg'][iGauss-1] in [1, 2]: sigma.setConstant(ROOT.kTRUE)     
        gauss = ROOT.RooGaussian("gauss%d" % iGauss, "", recoil, mean, sigma)
        pdfs.addOwned(gauss)
        mean_ptrs.append(mean)
        sigma_ptrs.append(sigma)
        gauss_ptrs.append(gauss)
    
    # identical norms for each gauss
    for iGauss in range(1, nGauss):
        norm = ROOT.RooRealVar("norm%d" % iGauss, "", fitCfg['norm'][iGauss-1], 0, 1)
        if fitCfg['norm_cfg'][iGauss-1] in [1, 2]: norm.setConstant(ROOT.kTRUE)
        norms.addOwned(norm)
        norm_ptrs.append(norm)
        
        
    # make the TF1s 
    if dof_param > 0 and funcJs == None: sys.exit("Parametric fit expected but input json not defined!")
    funcs = {}
    if funcJs != None:
        for iGauss in range(1, nGauss+1):
            if "mean%d" % iGauss in funcJs:
                funcs["mean%d" % iGauss] = ROOT.TF1("mean%d" % iGauss, funcJs['mean%d' % iGauss]['func'], 0, 300)
                for iParam in range(0, funcJs['mean%d' % iGauss]['nParams']): funcs["mean%d" % iGauss].SetParameter(iParam, funcJs['mean%d' % iGauss]['p%d' % iParam])
            if "sigma%d" % iGauss in funcJs:
                funcs["sigma%d" % iGauss] = ROOT.TF1("sigma%d" % iGauss, funcJs['sigma%d' % iGauss]['func'], 0, 300)
                for iParam in range(0, funcJs['sigma%d' % iGauss]['nParams']): funcs["sigma%d" % iGauss].SetParameter(iParam, funcJs['sigma%d' % iGauss]['p%d' % iParam])

        for iGauss in range(1, nGauss):
            if "norm%d" % iGauss in funcJs:
                funcs["norm%d" % iGauss] = ROOT.TF1("norm%d" % iGauss, funcJs['norm%d' % iGauss]['func'], 0, 300)
                for iParam in range(0, funcJs['norm%d' % iGauss]['nParams']): funcs["norm%d" % iGauss].SetParameter(iParam, funcJs['norm%d' % iGauss]['p%d' % iParam])
        
    pdf_sig = ROOT.RooAddPdf("pdf_sig", '', pdfs, norms)
    g_chi2 = ROOT.TGraphErrors()
    

    out = {}
    out['nGauss'] = nGauss
    recoil_qTbins_ = [0]
    for qTbin in range(1, len(recoil_qTbins)):
    
        qT = 0.5*(recoil_qTbins[qTbin-1] + recoil_qTbins[qTbin]) if qTbin < len(recoil_qTbins)-1 else (recoil_qTbins[qTbin-1] + 20)
        if qT > qTmax or qT < qTmin: continue
        recoil_qTbins_.append(recoil_qTbins[qTbin])
        qTmin_, qTmax_ = recoil_qTbins[qTbin-1], recoil_qTbins[qTbin]
        
        #hName = "%s_%s_bin%d" % (proc, comp, qTbin)
        #fIn = ROOT.TFile(fIn_)
        #hist = copy.deepcopy(fIn.Get(hName))
        #fIn.Close()
        #hist.Rebin(rebin)
        
        print(qTmin_, qTmax_)
        #qTbins_ = list(functions.drange(qTmin_, qTmax_, 0.5))
        qTbins_ = list(range(qTmin_, qTmax_, 1))
        hist = None
        fIn = ROOT.TFile(fIn_)
        for b in qTbins_:
            hName = "%s_%s_bin%d" % (proc, comp, qTbin)
            h = fIn.Get(hName)
            #h = functions.Rebin(h, rebin, binWidth=False)
            if hist == None: 
                hist = copy.deepcopy(h)
                hist.SetName("hist")
            else: hist.Add(h)
        
        
        yield_err = ctypes.c_double(1.)
        yield_ = hist.IntegralAndError(0, hist.GetNbinsX() + 1, yield_err)
        
        # set the variable according to the functional forms
        for iGauss in range(1, nGauss+1):
            if "mean%d" % iGauss in funcs:
                if not (singleMean and iGauss != 1):
                    mean_ptrs[iGauss-1].setVal(funcs["mean%d" % iGauss].Eval(qT))
            if "sigma%d" % iGauss in funcs: 
                sigma_ptrs[iGauss-1].setVal(funcs["sigma%d" % iGauss].Eval(qT))
            
        for iGauss in range(1, nGauss):
            if "norm%d" % iGauss in funcs: norm_ptrs[iGauss-1].setVal(funcs["norm%d" % iGauss].Eval(qT))
            
            
        # include backgrounds (if configured)
        pdfs_arglist, norms_arglist = ROOT.RooArgList(), ROOT.RooArgList()
        pdfs_arglist.setName("pdfs_arglist_bin%d" % (qTbin))
        norms_arglist.setName("norms_arglist_bin%d" % (qTbin))
        bkg_ptrs = []
        for bkg in bkgCfg:
            if bkgCfg[bkg]['subtract']:
                for b in qTbins_:
                    h = fIn.Get("%s_%s_bin%d" % (bkg, comp, qTbin))
                    #h = functions.Rebin(h, rebin, binWidth=False)
                    hist.Add(h, -1.*bkgCfg[bkg]['norm'])
            else:
                with open(bkgCfg[bkg]['filename']) as f: jsIn = json.load(f)
                pdfs_bkgs_arglist, norms_bkgs_arglist = ROOT.RooArgList(), ROOT.RooArgList()
                pdfs_bkgs_arglist.setName("pdfs_bkgs_arglist_bin%d_%s" % (qTbin, bkg))
                norms_bkgs_arglist.setName("norms_bkgs_arglist_bin%d_%s" % (qTbin, bkg))
                for iGauss in range(1, jsIn['nGauss']+1):
                    mean = ROOT.RooRealVar("mean%d_bin%d_%s" % (iGauss, qTbin, bkg), "", jsIn[str(qTbin)]["mean%d" % iGauss], -100, 100)
                    sigma = ROOT.RooRealVar("sigma%d_bin%d_%s" % (iGauss, qTbin, bkg), "", jsIn[str(qTbin)]["sigma%d" % iGauss], 0.1, 100)
                    mean.setConstant(ROOT.kTRUE)
                    sigma.setConstant(ROOT.kTRUE)
                    gauss = ROOT.RooGaussian("gauss%d_bin%d_%s" % (iGauss, qTbin, bkg), "", recoil, mean, sigma)
                    bkg_ptrs.append(mean)
                    bkg_ptrs.append(sigma)
                    bkg_ptrs.append(gauss)
                    pdfs_bkgs_arglist.addOwned(gauss)
                for iGauss in range(1, jsIn['nGauss']):
                    norm = ROOT.RooRealVar("norm%d_bin%d_%s" % (iGauss, qTbin, bkg), "", jsIn[str(qTbin)]["norm%d" % iGauss], 0, 1)
                    norm.setConstant(ROOT.kTRUE)
                    norms_bkgs_arglist.addOwned(norm)
                    bkg_ptrs.append(norm)
                pdf_bkg = ROOT.RooAddPdf("bin%d_%s" % (qTbin, bkg), '', pdfs_bkgs_arglist, norms_bkgs_arglist)
                pdf_bkg.Print()
                pdfs_arglist.addOwned(pdf_bkg)
                norm_bkg = ROOT.RooRealVar("relyield_bin%d_%s" % (qTbin, bkg), "", (jsIn[str(qTbin)]['yield']*bkgCfg[bkg]['norm'])/yield_, 0, 1)
                if not bkgCfg[bkg]['float']: norm_bkg.setConstant(ROOT.kTRUE)
                norms_arglist.addOwned(norm_bkg)
                norm_bkg.Print()
                bkg_ptrs.append(pdfs_bkgs_arglist)
                bkg_ptrs.append(norms_bkgs_arglist)
        pdfs_arglist.addClone(pdf_sig) # append signal as the last one
        pdf_tot = ROOT.RooAddPdf("pdf_tot", '', pdfs_arglist, norms_arglist)
        
        #hist.Scale(1./yield_, "width")
        hist = functions.Rebin(hist, rebin, binWidth=False)


        #print(hist)
        #sys.exit()
        #for iBin in range(0, hist.GetNbinsX()+1): 
        #    if hist.GetBinContent(iBin) <= 0: 
        #        hist.SetBinContent(iBin, 0)
        #        hist.SetBinError(iBin, 0)
        yield_ = hist.IntegralAndError(0, hist.GetNbinsX() + 1, yield_err)
        #hist = functions.Rebin(hist, rebin, binWidth=False)
        hist.Scale(1./yield_)
        rdh = ROOT.RooDataHist("rdh", "", ROOT.RooArgList(recoil), ROOT.RooFit.Import(hist, ROOT.kFALSE)) # 
        if orderSigmas:
            for iGauss in range(2, nGauss+1): sigma_ptrs[iGauss-1].setRange(sigma_ptrs[iGauss-2].getVal(), 100)
                
        
        if dof > 0:
            fitRes = pdf_tot.fitTo(rdh, ROOT.RooFit.SumW2Error(ROOT.kTRUE), ROOT.RooFit.Save(ROOT.kTRUE)) # ROOT.RooFit.Extended(ROOT.kTRUE), 
            fitValid = (fitRes.covQual() == 3 and fitRes.status() == 0)
        else: fitValid = True
        

        ## plot
        plotter.cfg = cfgPlot
        canvas, padT, padB = plotter.canvasRatio()
        dummyT, dummyB, dummyL = plotter.dummyRatio(line=0)
        
        canvas.cd()
        padT.Draw()
        padT.cd()
        dummyT.Draw("HIST")
        
        ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.ERROR)
        plt = recoil.frame()
        if dof > 0:
            if fitValid: pdf_tot.plotOn(plt, ROOT.RooFit.VisualizeError(fitRes, 1), ROOT.RooFit.Name("errorband"), ROOT.RooFit.FillColor(ROOT.kGreen+1))
            else: pdf_tot.plotOn(plt, ROOT.RooFit.VisualizeError(fitRes, 1), ROOT.RooFit.Name("errorband"), ROOT.RooFit.FillColor(ROOT.kOrange))
        rdh.plotOn(plt)
        colors = [ROOT.kBlue, ROOT.kGreen+1, ROOT.kOrange, ROOT.kCyan, ROOT.kViolet]
        for i, g in enumerate(gauss_ptrs):
            n = norm_ptrs[i].getVal() if i < len(gauss_ptrs)-1 else (1.-sum([x.getVal() for x in norm_ptrs]))
            g.plotOn(plt, ROOT.RooFit.LineColor(colors[i]), ROOT.RooFit.Normalization(n, ROOT.RooAbsReal.NumEvent), ROOT.RooFit.LineStyle(ROOT.kDashed))
        pdf_tot.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kRed)) # ROOT.RooFit.Normalization(1, ROOT.RooAbsReal.NumEvent)
        histpull = plt.pullHist()
        chi2 = plt.chiSquare()
        pdf_sig.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kRed), ROOT.RooFit.LineStyle(ROOT.kDashed)) # ROOT.RooFit.Normalization(1, ROOT.RooAbsReal.NumEvent)
 
        latex = ROOT.TLatex()
        latex.SetNDC()
        latex.SetTextSize(0.040)
        latex.SetTextColor(1)
        latex.SetTextFont(42)
        latex.DrawLatex(0.20, 0.85, label)
        latex.DrawLatex(0.20, 0.80, "q_{T} = [%.1f, %.1f] GeV" % (qTmin_, qTmax_))
        latex.DrawLatex(0.20, 0.75, "Mean = %.3f #pm %.3f" % (hist.GetMean(), hist.GetMeanError()))
        latex.DrawLatex(0.20, 0.70, "RMS = %.3f #pm %.3f" % (hist.GetRMS(), hist.GetRMSError()))
        latex.DrawLatex(0.20, 0.65, "Yield = %.3f #pm %.3f" % (yield_, yield_err.value))
        latex.DrawLatex(0.20, 0.60, "#chi^{2} = %.3f" % chi2)
        if dof > 0:
            latex.DrawLatex(0.20, 0.55, "#color[%d]{Cov. matrix quality = %d}" % (2 if fitRes.covQual() != 3 else 1, fitRes.covQual()))
            latex.DrawLatex(0.20, 0.50, "#color[%d]{MINUIT status = %d}" % (2 if fitRes.status() != 0 else 1, fitRes.status()))
 
        latex.SetTextSize(0.035)
        for i, s in enumerate(mean_ptrs): latex.DrawLatex(0.7, 0.87-i*0.04, "#mu_{%d} = %.3f #pm %.3f" % (i+1, s.getVal(), s.getError() if not singleMean else mean_ptrs[0].getError())) 
        for i, s in enumerate(sigma_ptrs): latex.DrawLatex(0.7, 0.87-i*0.04-0.04*len(mean_ptrs), "#sigma_{%d} = %.3f #pm %.3f" % (i+1, s.getVal(), s.getError()))
        for i, s in enumerate(norm_ptrs): latex.DrawLatex(0.7, 0.87-i*0.04-0.04*len(mean_ptrs)*2, "n_{%d} = %.3f #pm %.3f" % (i+1, s.getVal(), s.getError()))
        
        plt.Draw("SAME")
        plotter.auxRatio()

        canvas.cd()
        padB.Draw()
        padB.cd()
        dummyB.Draw("HIST")
        plt = recoil.frame()
        plt.addPlotable(histpull, "P")
        plt.Draw("SAME")
        dummyL.Draw("SAME")

        canvas.Modify()
        canvas.Update()
        canvas.Draw()
        canvas.SaveAs("%s/%03d_recoil.png" % (outDir_, qTbin))
        #canvas.SaveAs("%s/%03d_recoil.pdf" % (outDir_, qTbin))
        ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)   
        dummyB.Delete()
        dummyT.Delete()
        padT.Delete()
        padB.Delete()
        
        # save output
        out[qTbin] = {}
        out[qTbin]['yield'] = yield_
        out[qTbin]['yield_err'] = yield_err.value
        
        
        for i, s in enumerate(mean_ptrs):
            out[qTbin]['mean%d' % (i+1)], out[qTbin]['mean%d_err' % (i+1)] = s.getVal(), s.getError() if not singleMean else mean_ptrs[0].getError()
        
        for i, s in enumerate(sigma_ptrs):
            out[qTbin]['sigma%d' % (i+1)], out[qTbin]['sigma%d_err' % (i+1)] = s.getVal(), s.getError()
        
        for i, n in enumerate(norm_ptrs):
            out[qTbin]['norm%d' % (i+1)], out[qTbin]['norm%d_err' % (i+1)] = n.getVal(), n.getError()

        out[qTbin]['chi2'] = chi2
        g_chi2.SetPoint(qTbin-1, qT, chi2)

    out['qTbins'] = recoil_qTbins_
    with open("%s/results.json" % outDir_, "w") as outfile: json.dump(out, outfile, indent=4)
    plotChi2(g_chi2, "%s/chi2" % outDir_, label, "U_{#perp}" if comp == "perp" else "U_{#parallel}")



def doFit_data_perp():

    tag = "data_perp"
    cfg['xtitle'] = "U_{#perp}   (GeV)"
    cfg['xmin'], cfg['xmax'] = -100, 100
    cfg['ymax'] = 1e3
    cfg['ymin'] = 1e-6
    
    
    
    functions.prepareDir("%s/%s" % (outDir, tag))
    
    hOut = ROOT.TH3D(tag, "", 150, 0, 150, 20, 0, 20, 20, 0, 20) # qt, nominal, error

    if flavor == "mumu" and met == "DeepMETReso":
        mean1_, mean2_, mean3_, norm1_, norm2_, sigma1_, sigma2_, sigma3_ = 0, 0, 0,   0.6, 0.3,   6, 4, 10
        mean1_, mean2_, mean3_, norm1_, norm2_, sigma1_, sigma2_, sigma3_ = 0, 0, 0,   0.6, 0.3,   6, 4, 10
        
    if flavor == "mumu" and met == "RawPFMET":
        mean1_, mean2_, mean3_, norm1_, norm2_, sigma1_, sigma2_, sigma3_ = 0, 0, 0,   0.6, 0.37,   8, 5, 15
        
        nGauss = 5
        sigmas_ = [3, 5, 8, 11, 15]
        norms_ = [0.1, 0.1, 0.5, 0.15]

        nGauss = 4
        sigmas_ = [5, 8, 11, 15]
        norms_ = [0.22, 0.5, 0.25]
        
        nGauss = 3
        sigmas_ = [6, 10, 15]
        norms_ = [0.3, 0.6]
        
        
    # signal model
    recoil = ROOT.RooRealVar("recoil", "Recoil parallel (GeV)", 0, -500, 500) 
    mean = ROOT.RooRealVar("mean", "", 0)

    # construct gauss
    pdfs = ROOT.RooArgList()
    sigmas, ptrs = [], [] # store pointer to objects
    for iGauss in range(0, nGauss): 
        sigma = ROOT.RooRealVar("sigma%d" % iGauss, "", sigmas_[iGauss], 0.1, 100)
        gauss = ROOT.RooGaussian("gauss%d" % iGauss, "", recoil, mean, sigma)
        pdfs.addOwned(gauss)
        sigmas.append(sigma)
        ptrs.append(gauss)
    
    # identical norms for each gauss
    norms = ROOT.RooArgList()
    norm = ROOT.RooRealVar("norm", "", 1./nGauss)
    for iGauss in range(0, nGauss-1): #norms.addClone(norm)
        norm = ROOT.RooRealVar("norm%d" % iGauss, "", norms_[iGauss])
        norms.addOwned(norm)
        ptrs.append(norm)
    
    pdf_sig = ROOT.RooAddPdf("pdf_sig", '', pdfs, norms)
    
    

    
    # ewk
    yield_ewk = ROOT.RooRealVar("yield_ewk", "", 0)
    mean_ewk = ROOT.RooRealVar("mean_ewk", "", 0) # fixed
    norm_ewk = ROOT.RooRealVar("norm_ewk", "", 0.7)
    sigma1_ewk = ROOT.RooRealVar("sigma1_ewk", "", 10)
    sigma2_ewk = ROOT.RooRealVar("sigma2_ewk", "", 20)
    gauss1_ewk = ROOT.RooGaussian("gauss1_ewk", "", recoil, mean_ewk, sigma1_ewk)
    gauss2_ewk = ROOT.RooGaussian("gauss2_ewk", "", recoil, mean_ewk, sigma2_ewk)
    pdf_ewk = ROOT.RooAddPdf("pdf_ewk", '', ROOT.RooArgList(gauss1_ewk, gauss2_ewk), ROOT.RooArgList(norm_ewk))
    norm_ewk_func = ROOT.TF1("norm_ewk_func", "[0]*x + [1]", 0, 500)
    norm_ewk_func.SetParameters(-1.25835e-03, 8.64469e-01)
    sigma1_ewk_func = ROOT.TF1("sigma1_ewk_func", "[0]*TMath::Power(x+[1], [2])", 0, 500)
    sigma1_ewk_func.SetParameters(6.76272e+00, -4.94509e-01, 1.33996e-01)
    sigma2_ewk_func = ROOT.TF1("sigma2_ewk_func", "[0]*TMath::Power(x+[1], [2])", 0, 500)
    sigma2_ewk_func.SetParameters(1.45688e+01, -4.99997e-01, 2.80827e-01)

    
    # ttbar
    yield_ttbar = ROOT.RooRealVar("yield_ttbar", "", 0)
    mean_ttbar = ROOT.RooRealVar("mean_ttbar", "", 0)
    norm_ttbar = ROOT.RooRealVar("norm_ttbar", "", 0.7)
    sigma1_ttbar = ROOT.RooRealVar("sigma1_ttbar", "", 10)
    sigma2_ttbar = ROOT.RooRealVar("sigma2_ttbar", "", 20)
    gauss1_ttbar = ROOT.RooGaussian("gauss1_ttbar", "", recoil, mean_ttbar, sigma1_ttbar)
    gauss2_ttbar = ROOT.RooGaussian("gauss2_ttbar", "", recoil, mean_ttbar, sigma2_ttbar)
    pdf_ttbar = ROOT.RooAddPdf("pdf_ttbar", '', ROOT.RooArgList(gauss1_ttbar, gauss2_ttbar), ROOT.RooArgList(norm_ttbar))
    sigma1_ttbar_func = ROOT.TF1("sigma1_ttbar_func", "[0]*x + [1]", 0, 500)
    sigma1_ttbar_func.SetParameters(-4.42073e-02, 4.68472e+01)
    sigma2_ttbar_func = ROOT.TF1("sigma2_ttbar_func", "[0]*x*x + [1]*x + [2]", 0, 500)
    sigma2_ttbar_func.SetParameters(-1.08054e-03, 7.03511e-02, 7.01008e+01)
    
    # total data model
    pdf = ROOT.RooAddPdf("pdf", '', ROOT.RooArgList(pdf_ttbar, pdf_ewk, pdf_sig), ROOT.RooArgList(yield_ttbar, yield_ewk))
    
    

    
    doFit = False
    s1 = ROOT.TF1("s1", "[0]*TMath::Power(x+[1], [2])", 0, 200)
    #s1.SetParameters(4.99982e+00, 7.67005e-01, 5.53110e-02)
    s1.SetParameters(5.02232e+00, 1.39195e+00, 6.33123e-02)
    
    s2 = ROOT.TF1("s2", "[0]*TMath::Power(x+[1], [2])", 0, 200)
    #s2.SetParameters(3.68495e-01, 1.13750e+02, 6.69613e-01)
    s2.SetParameters(3.71235e-01, 1.07052e+02, 6.74092e-01)
    
    s3 = ROOT.TF1("s3", "[0]*TMath::Power(x+[1], [2])", 0, 200)
    #s3.SetParameters(5.64193e-01, 1.59898e+02, 6.22697e-01)
    s3.SetParameters(5.69628e-01, 1.19781e+02, 6.57171e-01)
    
    s4 = ROOT.TF1("s4", "[0]", 0, 200)
    s4.SetParameter(0, 0)
    #s4.SetParameter(0, 1.61676e+01)

    g_chi2 = ROOT.TGraphErrors()
    g_chi2.SetLineColor(ROOT.kRed)
    g_chi2.SetMarkerStyle(8)
    g_chi2.SetMarkerSize(1)
    g_chi2.SetMarkerColor(ROOT.kRed)
    
    #params = [mean1, sigma1, norm1, mean2, sigma2, norm2, mean3, sigma3, norm3]
    
    for qTbin in range(1, len(recoil_qTbins)):
        
        qT = 0.5*(recoil_qTbins[qTbin-1] + recoil_qTbins[qTbin]) if qTbin < len(recoil_qTbins)-1 else (recoil_qTbins[qTbin-1] + 20)
        qTmin, qTmax = recoil_qTbins[qTbin-1], recoil_qTbins[qTbin]
        hName = "%s_perp_bin%d" % ("SingleMuon", qTbin)
        
        if qT > 200: continue
        
        hist = fIn.Get(hName)
        hist.Rebin(1)
        
        hist_ttbar = fIn.Get("%s_perp_bin%d" % ("TTbar", qTbin))
        hist_ewk = fIn.Get("%s_perp_bin%d" % ("EWK", qTbin))
    
        norm = hist.Integral()
        hist.Scale(1./norm)
        rdh = ROOT.RooDataHist("rdh", "", ROOT.RooArgList(recoil), ROOT.RooFit.Import(hist))
        
        yield_ttbar.setVal(hist_ttbar.Integral() / norm)
        yield_ewk.setVal(hist_ewk.Integral() / norm)
        
        # set the background parameters
        norm_ewk.setVal(norm_ewk_func.Eval(qT))
        sigma1_ewk.setVal(sigma1_ewk_func.Eval(qT))
        sigma2_ewk.setVal(sigma2_ewk_func.Eval(qT))
        sigma1_ttbar.setVal(sigma1_ttbar_func.Eval(qT))
        sigma2_ttbar.setVal(sigma2_ttbar_func.Eval(qT))
        
        #sigmas[0].setVal(s1.Eval(qT))
        #sigmas[0].setConstant(ROOT.kTRUE)
        
        #sigmas[1].setVal(s2.Eval(qT))
        #sigmas[1].setConstant(ROOT.kTRUE)

        if doFit:
    
            fitRes = pdf.fitTo(rdh, ROOT.RooFit.SumW2Error(ROOT.kTRUE), ROOT.RooFit.Save(ROOT.kTRUE)) # ROOT.RooFit.Extended(ROOT.kTRUE), 
            fitValid = (fitRes.covQual() == 3 and fitRes.status() == 0)
            #mean2.setVal(mean1.getVal())
            #mean3.setVal(mean1.getVal())    
        else: 
        
            fitValid = True
            # set variables
            sigmas[0].setVal(s1.Eval(qT))
            sigmas[1].setVal(s2.Eval(qT))
            sigmas[2].setVal(s3.Eval(qT))
            #sigmas[3].setVal(s4.Eval(qT))
            sigmas[0].setConstant(ROOT.kTRUE)
            sigmas[1].setConstant(ROOT.kTRUE)
            sigmas[2].setConstant(ROOT.kTRUE)
            #sigmas[3].setConstant(ROOT.kTRUE)
            mean.setConstant(ROOT.kTRUE)
        
        ## plot
        plotter.cfg = cfg
        canvas, padT, padB = plotter.canvasRatio()
        dummyT, dummyB, dummyL = plotter.dummyRatio(line=0)
        
        canvas.cd()
        padT.Draw()
        padT.cd()
        dummyT.Draw("HIST")
        
        ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.ERROR)
        plt = recoil.frame()
        
        if doFit:
            if fitValid: pdf.plotOn(plt, ROOT.RooFit.VisualizeError(fitRes, 1), ROOT.RooFit.Name("errorband"), ROOT.RooFit.FillColor(ROOT.kGreen+1))
            else: pdf.plotOn(plt, ROOT.RooFit.VisualizeError(fitRes, 1), ROOT.RooFit.Name("errorband"), ROOT.RooFit.FillColor(ROOT.kOrange))
        rdh.plotOn(plt)
        #gauss1.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kBlue), ROOT.RooFit.Normalization(norm1.getVal(), ROOT.RooAbsReal.NumEvent), ROOT.RooFit.LineStyle(ROOT.kDashed))
        #gauss2.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kGreen), ROOT.RooFit.Normalization(norm2.getVal(), ROOT.RooAbsReal.NumEvent), ROOT.RooFit.LineStyle(ROOT.kDashed))
        #gauss3.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kCyan), ROOT.RooFit.Normalization(norm3.getVal(), ROOT.RooAbsReal.NumEvent), ROOT.RooFit.LineStyle(ROOT.kDashed))
        pdf.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kRed))
        histpull = plt.pullHist()
        chi2 = plt.chiSquare() 

        latex = ROOT.TLatex()
        latex.SetNDC()
        latex.SetTextSize(0.040)
        latex.SetTextColor(1)
        latex.SetTextFont(42)
        latex.DrawLatex(0.20, 0.85, "%s, DATA" % label)
        latex.DrawLatex(0.20, 0.80, "q_{T} = [%.1f, %.1f] GeV" % (qTmin, qTmax))
        latex.DrawLatex(0.20, 0.75, "Mean = %.3f #pm %.3f" % (hist.GetMean(), hist.GetMeanError()))
        latex.DrawLatex(0.20, 0.70, "RMS = %.3f #pm %.3f" % (hist.GetRMS(), hist.GetRMSError()))
        latex.DrawLatex(0.20, 0.65, "#chi^{2}/ndof = %.3f" % chi2)
        if doFit:
            latex.DrawLatex(0.20, 0.60, "#color[%d]{Cov. matrix quality = %d}" % (2 if fitRes.covQual() != 3 else 1, fitRes.covQual()))
            latex.DrawLatex(0.20, 0.55, "#color[%d]{MINUIT status = %d}" % (2 if fitRes.status() != 0 else 1, fitRes.status()))
        
        
        latex.DrawLatex(0.60, 0.85, "#mu_{1} = %.3f #pm %.3f" % (mean.getVal(), mean.getError()))
        for i, s in enumerate(sigmas):
             latex.DrawLatex(0.60, 0.80-i*0.05, "#sigma_{%d} = %.3f #pm %.3f" % (i+1, s.getVal(), s.getError()))
        '''
        
        latex.DrawLatex(0.60, 0.80, "#mu_{2} = %.3f #pm %.3f" % (mean2.getVal(), mean2.getError()))
        latex.DrawLatex(0.60, 0.75, "#mu_{3} = %.3f #pm %.3f" % (mean3.getVal(), mean3.getError()))
        latex.DrawLatex(0.60, 0.70, "#sigma_{1} = %.3f #pm %.3f" % (sigma1.getVal(), sigma1.getError()))
        latex.DrawLatex(0.60, 0.65, "#sigma_{2} = %.3f #pm %.3f" % (sigma2.getVal(), sigma2.getError()))
        latex.DrawLatex(0.60, 0.60, "#sigma_{3} = %.3f #pm %.3f" % (sigma3.getVal(), sigma3.getError()))
        latex.DrawLatex(0.60, 0.55, "N_{1} = %.3f #pm %.3f" % (norm1.getVal(), norm1.getError()))
        latex.DrawLatex(0.60, 0.50, "N_{2} = %.3f #pm %.3f" % (norm2.getVal(), norm2.getError()))
        '''
        plt.Draw("SAME")
        plotter.auxRatio()

        canvas.cd()
        padB.Draw()
        padB.cd()
        dummyB.Draw("HIST")
        plt = recoil.frame()
        plt.addPlotable(histpull, "P")
        plt.Draw("SAME")
        dummyL.Draw("SAME")

        canvas.Modify()
        canvas.Update()
        canvas.Draw()
        canvas.SaveAs("%s/%s/%03d_recoil.png" % (outDir, tag, qTbin))
        #canvas.SaveAs("%s/%03d_recoil_%s.pdf" % (outDir, qTbin, name))
        ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)   
        dummyB.Delete()
        dummyT.Delete()
        padT.Delete()
        padB.Delete()
        
        # save output
        hOut.SetBinContent(qTbin, 1, 1, mean.getVal())
        hOut.SetBinContent(qTbin, 1, 0, mean.getError())
        
        for i, s in enumerate(sigmas):
            hOut.SetBinContent(qTbin, i+2, 1, s.getVal())
            hOut.SetBinContent(qTbin, i+2, 0, s.getError())
        
        g_chi2.SetPoint(qTbin-1, qT, chi2)
        #sys.exit()
        continue
        #sys.exit()
        hOut.SetBinContent(qTbin, 1, 1, mean1.getVal())
        hOut.SetBinContent(qTbin, 1, 0, mean1.getError())
        hOut.SetBinContent(qTbin, 2, 1, sigma1.getVal())
        hOut.SetBinContent(qTbin, 2, 0, sigma1.getError())
        hOut.SetBinContent(qTbin, 3, 1, norm1.getVal())
        hOut.SetBinContent(qTbin, 3, 0, norm1.getError())
        
        hOut.SetBinContent(qTbin, 4, 1, mean2.getVal())
        hOut.SetBinContent(qTbin, 4, 0, mean2.getError())
        hOut.SetBinContent(qTbin, 5, 1, sigma2.getVal())
        hOut.SetBinContent(qTbin, 5, 0, sigma2.getError())
        hOut.SetBinContent(qTbin, 6, 1, norm2.getVal())
        hOut.SetBinContent(qTbin, 6, 0, norm2.getError())
        
        hOut.SetBinContent(qTbin, 7, 1, mean3.getVal())
        hOut.SetBinContent(qTbin, 7, 0, mean3.getError())
        hOut.SetBinContent(qTbin, 8, 1, sigma3.getVal())
        hOut.SetBinContent(qTbin, 8, 0, sigma3.getError())
        hOut.SetBinContent(qTbin, 9, 1, norm3.getVal())
        hOut.SetBinContent(qTbin, 9, 0, 0)
        
        hOut.SetBinContent(qTbin, 10, 1, norm)
        hOut.SetBinContent(qTbin, 0, 0, chi2)
        
        

        # diagonalize covariance matrix and store perturbations
        floatingParams = fitRes.floatParsFinal() # floating parameters
        variations = diagonalize(fitRes)
        # print(p.GetName(), p.isConstant(), type(p), isinstance(p, ROOT.RooFormulaVar))
        for iVar, var in enumerate(variations):
            for iPar, p in enumerate(params):
                if fitValid:
                    if getParamIdx(p.GetName(), floatingParams) > -1: val = var[getParamIdx(p.GetName(), floatingParams)]
                    else: val = p.getVal()
                    hOut.SetBinContent(qTbin, iPar+1, iVar+2, val)
                    print(iVar, iPar, p.GetName(), val)
                else: hOut.SetBinContent(qTbin, iPar+1, iVar+2, p.getVal())


    fOut = ROOT.TFile(fOut_.replace(".root", "_data_perp.root"), "RECREATE")
    hOut.Write()
    fOut.Close()
    
    
    # plot the chi2
    cfg_chi2 = {

        'logy'              : False,
        'logx'              : False,
        
        'xmin'              : 0,
        'xmax'              : 150,
        'ymin'              : 0,
        'ymax'              : 3,
            
        'xtitle'            : "q_{T} (GeV)",
        'ytitle'            : "#chi^{2}",
            
        'topRight'          : "199 pb^{#minus1} (13 TeV)", 
        'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",

    } 
    

    plotter.cfg = cfg_chi2
    canvas = plotter.canvas()
    dummy = plotter.dummy()   
    canvas.cd()
    dummy.Draw("HIST")
    g_chi2.Draw("SAME LP")
    plotter.aux()
    
    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.035)
    latex.SetTextColor(1)
    latex.SetTextFont(42)
    latex.DrawLatex(0.20, 0.90, "%s, MC" % label)
    latex.DrawLatex(0.20, 0.85, "U_{#perp}")
    
    canvas.Modify()
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs("%s/%s/chi2.png" % (outDir, tag))
    canvas.SaveAs("%s/%s/chi2.pdf" % (outDir, tag))
    canvas.Delete()


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
            
        'topRight'          : "199 pb^{#minus1} (13 TeV)", 
        'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",

    } 
    
    g_chi2.SetLineColor(ROOT.kRed)
    g_chi2.SetMarkerStyle(8)
    g_chi2.SetMarkerSize(1)
    g_chi2.SetMarkerColor(ROOT.kRed)
    
    
    plotter.cfg = cfg_chi2
    canvas = plotter.canvas()
    dummy = plotter.dummy()   
    canvas.cd()
    dummy.Draw("HIST")
    g_chi2.Draw("SAME LP")
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

def doFit_ttbar_perp(bkg, tag, outDir_, name, label, doFit = True):

    cfg['xtitle'] = "U_{#perp}   (GeV)"
    cfg['xmin'], cfg['xmax'] = -200, 200
    cfg['ymax'] = 1e3
    cfg['ymin'] = 1e-6
    
    outDirFits = outDir_
    if doFit: outDirFits = "%s/fits/" % (outDir_)
    else: outDirFits = "%s/validation/" % (outDir_)
    functions.prepareDir(outDirFits, True)
    

    # model
    recoil = ROOT.RooRealVar("recoil", "Recoil parallel (GeV)", 0, -500, 500) 
    mean = ROOT.RooRealVar("mean", "", 0) # fixed
    norm = ROOT.RooRealVar("norm", "", 0.7) # fixed
    sigma1 = ROOT.RooRealVar("sigma1", "", 10, 0.1, 100)
    sigma2 = ROOT.RooRealVar("sigma2", "", 20, 0.1, 100)
    gauss1 = ROOT.RooGaussian("gauss1", "", recoil, mean, sigma1)
    gauss2 = ROOT.RooGaussian("gauss2", "", recoil, mean, sigma2)
    pdf = ROOT.RooAddPdf("pdf", '', ROOT.RooArgList(gauss1, gauss2), ROOT.RooArgList(norm))
    
    s1 = ROOT.TF1("s1", "[0]*x + [1]", 0, 200)
    s1.SetParameters(-4.42073e-02, 4.68472e+01)
    

    g_chi2 = ROOT.TGraphErrors()
    g_chi2.SetLineColor(ROOT.kRed)
    g_chi2.SetMarkerStyle(8)
    g_chi2.SetMarkerSize(1)
    g_chi2.SetMarkerColor(ROOT.kRed)
    
    
    totNorm = 0
    out = {}
    for qTbin in range(1, len(recoil_qTbins)):
    
        out[qTbin] = {}
        
        qT = 0.5*(recoil_qTbins[qTbin-1] + recoil_qTbins[qTbin]) if qTbin < len(recoil_qTbins)-1 else (recoil_qTbins[qTbin-1] + 20)
        qTmin, qTmax = recoil_qTbins[qTbin-1], recoil_qTbins[qTbin]
        hName = "%s_perp_bin%d" % (bkg, qTbin)
        if qT > 100: continue

        hist = fIn.Get(hName)
        hist.Rebin(2)
        yield_err = ctypes.c_double(1.)
        yield_ = hist.IntegralAndError(0, hist.GetNbinsX() + 1, yield_err)
        hist.Scale(1./yield_)
        rdh = ROOT.RooDataHist("rdh", "", ROOT.RooArgList(recoil), ROOT.RooFit.Import(hist))

        if doFit:
            fitRes = pdf.fitTo(rdh, ROOT.RooFit.SumW2Error(ROOT.kTRUE), ROOT.RooFit.Save(ROOT.kTRUE))
            fitValid = (fitRes.covQual() == 3 and fitRes.status() == 0)

        else: 
        
            sigma1.setVal(s1.Eval(qT))
            sigma1.setConstant(ROOT.kTRUE)
            fitValid = True
            mean.setConstant(ROOT.kTRUE)
        
        ## plot
        plotter.cfg = cfg
        canvas, padT, padB = plotter.canvasRatio()
        dummyT, dummyB, dummyL = plotter.dummyRatio(line=0)
        
        canvas.cd()
        padT.Draw()
        padT.cd()
        dummyT.Draw("HIST")
        
        ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.ERROR)
        plt = recoil.frame()
        
        if doFit:
            if fitValid: pdf.plotOn(plt, ROOT.RooFit.VisualizeError(fitRes, 1), ROOT.RooFit.Name("errorband"), ROOT.RooFit.FillColor(ROOT.kGreen+1))
            else: pdf.plotOn(plt, ROOT.RooFit.VisualizeError(fitRes, 1), ROOT.RooFit.Name("errorband"), ROOT.RooFit.FillColor(ROOT.kOrange))
        rdh.plotOn(plt)
        pdf.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kRed))
        histpull = plt.pullHist()
        chi2 = plt.chiSquare() 

        latex = ROOT.TLatex()
        latex.SetNDC()
        latex.SetTextSize(0.040)
        latex.SetTextColor(1)
        latex.SetTextFont(42)
        latex.DrawLatex(0.20, 0.85, label)
        latex.DrawLatex(0.20, 0.80, "q_{T} = [%.1f, %.1f] GeV" % (qTmin, qTmax))
        latex.DrawLatex(0.20, 0.75, "Mean = %.3f #pm %.3f" % (hist.GetMean(), hist.GetMeanError()))
        latex.DrawLatex(0.20, 0.70, "RMS = %.3f #pm %.3f" % (hist.GetRMS(), hist.GetRMSError()))
        latex.DrawLatex(0.20, 0.65, "Yield = %.3f #pm %.3f" % (yield_, yield_err.value))
        latex.DrawLatex(0.20, 0.60, "#chi^{2}/ndof = %.3f" % chi2)
        if doFit:
            latex.DrawLatex(0.20, 0.55, "#color[%d]{Cov. matrix quality = %d}" % (2 if fitRes.covQual() != 3 else 1, fitRes.covQual()))
            latex.DrawLatex(0.20, 0.50, "#color[%d]{MINUIT status = %d}" % (2 if fitRes.status() != 0 else 1, fitRes.status()))
        
        latex.DrawLatex(0.60, 0.85, "#mu_{1} = %.3f #pm %.3f" % (mean.getVal(), mean.getError()))
        latex.DrawLatex(0.60, 0.80, "#sigma_{1} = %.3f #pm %.3f" % (sigma1.getVal(), sigma1.getError()))
        latex.DrawLatex(0.60, 0.75, "#sigma_{2} = %.3f #pm %.3f" % (sigma2.getVal(), sigma2.getError()))
        latex.DrawLatex(0.60, 0.70, "N = %.3f #pm %.3f" % (norm.getVal(), norm.getError()))
        
        plt.Draw("SAME")
        plotter.auxRatio()

        canvas.cd()
        padB.Draw()
        padB.cd()
        dummyB.Draw("HIST")
        plt = recoil.frame()
        plt.addPlotable(histpull, "P")
        plt.Draw("SAME")
        dummyL.Draw("SAME")

        canvas.Modify()
        canvas.Update()
        canvas.Draw()
        canvas.SaveAs("%s/%03d_recoil.png" % (outDirFits, qTbin))
        #canvas.SaveAs("%s/%03d_recoil.pdf" % (outDirFits, qTbin))
        ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)   
        dummyB.Delete()
        dummyT.Delete()
        padT.Delete()
        padB.Delete()
        
        # save output
        out[qTbin]['yield'] = yield_
        out[qTbin]['yield_err'] = yield_err.value
        
        out[qTbin]['mean'] = mean.getVal()
        out[qTbin]['mean_err'] = mean.getError()
        
        out[qTbin]['sigma1'] = sigma1.getVal()
        out[qTbin]['sigma1_err'] = sigma1.getError()
        
        out[qTbin]['sigma2'] = sigma2.getVal()
        out[qTbin]['sigma2_err'] = sigma2.getError()
        
        out[qTbin]['norm'] = norm.getVal()
        out[qTbin]['norm_err'] = norm.getError()
        
        out[qTbin]['chi2'] = chi2
        
        g_chi2.SetPoint(qTbin-1, qT, chi2)

    with open("%s/%s.json" % (outDir_, "fit" if doFit else "validation"), "w") as outfile: json.dump(out, outfile, indent=4)    
    plotChi2(g_chi2, "%s/chi2_%s" % (outDir_, "fit" if doFit else "validation"), label, "U_{#perp}")



def doFit_ewk_perp():

    bkg = "EWK"
    tag = "ewk_perp"
    cfg['xtitle'] = "U_{#perp}   (GeV)"
    cfg['xmin'], cfg['xmax'] = -200, 200
    cfg['ymax'] = 1e3
    cfg['ymin'] = 1e-6
    
    
    
    functions.prepareDir("%s/%s" % (outDir, tag))
    
    hOut = ROOT.TH3D(tag, "", 150, 0, 150, 20, 0, 20, 20, 0, 20) # qt, nominal, error

    if flavor == "mumu" and met == "DeepMETReso":
        mean1_, mean2_, mean3_, norm1_, norm2_, sigma1_, sigma2_, sigma3_ = 0, 0, 0,   0.6, 0.3,   6, 4, 10
        mean1_, mean2_, mean3_, norm1_, norm2_, sigma1_, sigma2_, sigma3_ = 0, 0, 0,   0.6, 0.3,   6, 4, 10
        
    if flavor == "mumu" and met == "RawPFMET":
        mean1_, mean2_, mean3_, norm1_, norm2_, sigma1_, sigma2_, sigma3_ = 0, 0, 0,   0.6, 0.37,   8, 5, 15

        
        
    # model
    recoil = ROOT.RooRealVar("recoil", "Recoil parallel (GeV)", 0, -500, 500) 
    mean = ROOT.RooRealVar("mean", "", 0) # fixed
    norm = ROOT.RooRealVar("norm", "", 0.7, 0, 1) # fixed

    # construct gauss
    sigma1 = ROOT.RooRealVar("sigma1", "", 10, 0.1, 100)
    sigma2 = ROOT.RooRealVar("sigma2", "", 20, 0.1, 100)
    gauss1 = ROOT.RooGaussian("gauss1", "", recoil, mean, sigma1)
    gauss2 = ROOT.RooGaussian("gauss2", "", recoil, mean, sigma2)
    pdf = ROOT.RooAddPdf("pdf", '', ROOT.RooArgList(gauss1, gauss2), ROOT.RooArgList(norm))
    
    s1 = ROOT.TF1("s1", "[0]*TMath::Power(x+[1], [2])", 0, 200)
    s1.SetParameters(6.76270e+00, -4.94508e-01, 1.33995e-01)
    
    n = ROOT.TF1("fit_%s_norm" % tag, "[0]*x + [1]", 0, 200)
    n.SetParameters(-1.25835e-03, 8.64469e-01)
    
    doFit = True

    g_chi2 = ROOT.TGraphErrors()
    g_chi2.SetLineColor(ROOT.kRed)
    g_chi2.SetMarkerStyle(8)
    g_chi2.SetMarkerSize(1)
    g_chi2.SetMarkerColor(ROOT.kRed)
    
    #params = [mean1, sigma1, norm1, mean2, sigma2, norm2, mean3, sigma3, norm3]
    
    totNorm = 0
    for qTbin in range(1, len(recoil_qTbins)):
        
        qT = 0.5*(recoil_qTbins[qTbin-1] + recoil_qTbins[qTbin]) if qTbin < len(recoil_qTbins)-1 else (recoil_qTbins[qTbin-1] + 20)
        qTmin, qTmax = recoil_qTbins[qTbin-1], recoil_qTbins[qTbin]
        hName = "%s_perp_bin%d" % (bkg, qTbin)
        if qT > 200: continue

        hist = fIn.Get(hName)
        hist.Rebin(4)
    
        norm_ = hist.Integral()
        norm_err_ = ctypes.c_double(1.)
        norm_ = hist.IntegralAndError(0, hist.GetNbinsX() + 1, norm_err_)
        hist.Scale(1./norm_)
        rdh = ROOT.RooDataHist("rdh", "", ROOT.RooArgList(recoil), ROOT.RooFit.Import(hist))
        
        sigma1.setVal(s1.Eval(qT))
        sigma1.setConstant(ROOT.kTRUE)
        
        norm.setVal(n.Eval(qT))
        norm.setConstant(ROOT.kTRUE)

        if doFit:
            fitRes = pdf.fitTo(rdh, ROOT.RooFit.SumW2Error(ROOT.kTRUE), ROOT.RooFit.Save(ROOT.kTRUE)) # ROOT.RooFit.Extended(ROOT.kTRUE), 
            fitValid = (fitRes.covQual() == 3 and fitRes.status() == 0)

        else: 
        
            fitValid = True
            mean.setConstant(ROOT.kTRUE)
        
        ## plot
        plotter.cfg = cfg
        canvas, padT, padB = plotter.canvasRatio()
        dummyT, dummyB, dummyL = plotter.dummyRatio(line=0)
        
        canvas.cd()
        padT.Draw()
        padT.cd()
        dummyT.Draw("HIST")
        
        ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.ERROR)
        plt = recoil.frame()
        
        if doFit:
            if fitValid: pdf.plotOn(plt, ROOT.RooFit.VisualizeError(fitRes, 1), ROOT.RooFit.Name("errorband"), ROOT.RooFit.FillColor(ROOT.kGreen+1))
            else: pdf.plotOn(plt, ROOT.RooFit.VisualizeError(fitRes, 1), ROOT.RooFit.Name("errorband"), ROOT.RooFit.FillColor(ROOT.kOrange))
        rdh.plotOn(plt)
        pdf.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kRed))
        histpull = plt.pullHist()
        chi2 = plt.chiSquare() 

        latex = ROOT.TLatex()
        latex.SetNDC()
        latex.SetTextSize(0.040)
        latex.SetTextColor(1)
        latex.SetTextFont(42)
        latex.DrawLatex(0.20, 0.85, "%s, MC" % label)
        latex.DrawLatex(0.20, 0.80, "q_{T} = [%.1f, %.1f] GeV" % (qTmin, qTmax))
        latex.DrawLatex(0.20, 0.75, "Mean = %.3f #pm %.3f" % (hist.GetMean(), hist.GetMeanError()))
        latex.DrawLatex(0.20, 0.70, "RMS = %.3f #pm %.3f" % (hist.GetRMS(), hist.GetRMSError()))
        latex.DrawLatex(0.20, 0.65, "#chi^{2}/ndof = %.3f" % chi2)
        if doFit:
            latex.DrawLatex(0.20, 0.60, "#color[%d]{Cov. matrix quality = %d}" % (2 if fitRes.covQual() != 3 else 1, fitRes.covQual()))
            latex.DrawLatex(0.20, 0.55, "#color[%d]{MINUIT status = %d}" % (2 if fitRes.status() != 0 else 1, fitRes.status()))
        latex.DrawLatex(0.20, 0.50, "Norm = %.3f #pm %.3f" % (norm_, norm_err_.value))
        
        latex.DrawLatex(0.60, 0.85, "#mu_{1} = %.3f #pm %.3f" % (mean.getVal(), mean.getError()))
        latex.DrawLatex(0.60, 0.80, "#sigma_{1} = %.3f #pm %.3f" % (sigma1.getVal(), sigma1.getError()))
        latex.DrawLatex(0.60, 0.75, "#sigma_{2} = %.3f #pm %.3f" % (sigma2.getVal(), sigma2.getError()))
        latex.DrawLatex(0.60, 0.70, "N = %.3f #pm %.3f" % (norm.getVal(), norm.getError()))
        
        plt.Draw("SAME")
        plotter.auxRatio()

        canvas.cd()
        padB.Draw()
        padB.cd()
        dummyB.Draw("HIST")
        plt = recoil.frame()
        plt.addPlotable(histpull, "P")
        plt.Draw("SAME")
        dummyL.Draw("SAME")

        canvas.Modify()
        canvas.Update()
        canvas.Draw()
        canvas.SaveAs("%s/%s/%03d_recoil.png" % (outDir, tag, qTbin))
        #canvas.SaveAs("%s/%03d_recoil_%s.pdf" % (outDir, qTbin, name))
        ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)   
        dummyB.Delete()
        dummyT.Delete()
        padT.Delete()
        padB.Delete()
        
        # save output
        hOut.SetBinContent(qTbin, 1, 1, mean.getVal())
        hOut.SetBinContent(qTbin, 1, 0, mean.getError())
       
        hOut.SetBinContent(qTbin, 2, 1, sigma1.getVal())
        hOut.SetBinContent(qTbin, 2, 0, sigma1.getError())
        
        hOut.SetBinContent(qTbin, 3, 1, sigma2.getVal())
        hOut.SetBinContent(qTbin, 3, 0, sigma2.getError())
        
        hOut.SetBinContent(qTbin, 4, 1, norm.getVal())
        hOut.SetBinContent(qTbin, 4, 0, norm.getError())
        
        hOut.SetBinContent(qTbin, 5, 1, norm_)
        hOut.SetBinContent(qTbin, 5, 0, norm_err_.value)
        
        
        g_chi2.SetPoint(qTbin-1, qT, chi2)
        

    fOut = ROOT.TFile(fOut_.replace(".root", "_ewk_perp.root"), "RECREATE")
    hOut.Write()
    fOut.Close()
    
    
    # plot the chi2
    cfg_chi2 = {

        'logy'              : False,
        'logx'              : False,
        
        'xmin'              : 0,
        'xmax'              : 100,
        'ymin'              : 0,
        'ymax'              : 3,
            
        'xtitle'            : "q_{T} (GeV)",
        'ytitle'            : "#chi^{2}",
            
        'topRight'          : "199 pb^{#minus1} (13 TeV)", 
        'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",

    } 
    

    plotter.cfg = cfg_chi2
    canvas = plotter.canvas()
    dummy = plotter.dummy()   
    canvas.cd()
    dummy.Draw("HIST")
    g_chi2.Draw("SAME LP")
    plotter.aux()
    
    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.035)
    latex.SetTextColor(1)
    latex.SetTextFont(42)
    latex.DrawLatex(0.20, 0.90, "%s, MC" % label)
    latex.DrawLatex(0.20, 0.85, "U_{#perp}")
    
    canvas.Modify()
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs("%s/%s/chi2.png" % (outDir, tag))
    canvas.SaveAs("%s/%s/chi2.pdf" % (outDir, tag))
    canvas.Delete()




def doFit_mc_gen_perp():

    tag = "mc_gen_perp"
    cfg['xtitle'] = "U_{#perp}   (GeV)"
    cfg['xmin'], cfg['xmax'] = -100, 100
    cfg['ymax'] = 1e3
    cfg['ymin'] = 1e-5
    
    functions.prepareDir("%s/%s" % (outDir, tag))
    
    hOut = ROOT.TH3D(tag, "", 150, 0, 150, 20, 0, 20, 20, 0, 20) # qt, nominal, error

    if flavor == "mumu" and met == "DeepMETReso":
        mean1_, mean2_, mean3_, norm1_, norm2_, sigma1_, sigma2_, sigma3_ = 0, 0, 0,   0.6, 0.3,   6, 4, 10
        mean1_, mean2_, mean3_, norm1_, norm2_, sigma1_, sigma2_, sigma3_ = 0, 0, 0,   0.6, 0.3,   6, 4, 10
        
    if flavor == "mumu" and met == "RawPFMET":
        mean1_, mean2_, mean3_, norm1_, norm2_, sigma1_, sigma2_, sigma3_ = 0, 0, 0,   0.6, 0.37,   8, 5, 15
    
    if flavor == "mu" and met == "RawPFMET":
        mean1_, mean2_, mean3_, norm1_, norm2_, sigma1_, sigma2_, sigma3_ = 0, 0, 0,   0.6, 0.37,   8, 5, 15
    
    # model
    recoil = ROOT.RooRealVar("recoil", "Recoil parallel (GeV)", 0, -500, 500)   
    mean1 = ROOT.RooRealVar("mean1", "", mean1_, -500, 500)
    mean2 = ROOT.RooRealVar("mean2", "", mean2_, -500, 500)
    mean3 = ROOT.RooRealVar("mean3", "", mean3_, -500, 500)
    sigma1 = ROOT.RooRealVar("sigma1", "", sigma1_, 0.1, 100)
    sigma2 = ROOT.RooRealVar("sigma2", "", sigma2_, 0.1, 100)
    sigma3 = ROOT.RooRealVar("sigma3", "", sigma3_, 0.1, 100)
    norm1 = ROOT.RooRealVar("norm1", "", norm1_, 0, 1)
    norm2 = ROOT.RooRealVar("norm2", "", norm2_, 0, 1)
    norm3 = ROOT.RooFormulaVar("norm3", "(1. - @0 - @1)", ROOT.RooArgList(norm1, norm2))
    gauss1 = ROOT.RooGaussian("gauss1", "gauss1", recoil, mean1, sigma1)
    gauss2 = ROOT.RooGaussian("gauss2", "gauss2", recoil, mean1, sigma2)
    gauss3 = ROOT.RooGaussian("gauss3", "gauss3", recoil, mean1, sigma3)
    pdf = ROOT.RooAddPdf("pdf", '', ROOT.RooArgList(gauss1, gauss2, gauss3), ROOT.RooArgList(norm1, norm2))
    
    norm1.setConstant(ROOT.kTRUE)
    norm2.setConstant(ROOT.kTRUE)
    #sigma2.setConstant(ROOT.kTRUE)
    #sigma3.setConstant(ROOT.kTRUE)
    #mean1.setConstant(ROOT.kTRUE)
    #sigma1.setConstant(ROOT.kTRUE)
    
    g_chi2 = ROOT.TGraphErrors()
    g_chi2.SetLineColor(ROOT.kRed)
    g_chi2.SetMarkerStyle(8)
    g_chi2.SetMarkerSize(1)
    g_chi2.SetMarkerColor(ROOT.kRed)
    
    params = [mean1, sigma1, norm1, mean2, sigma2, norm2, mean3, sigma3, norm3]
    
    for qTbin in range(1, len(recoil_qTbins)):
        
        qT = 0.5*(recoil_qTbins[qTbin-1] + recoil_qTbins[qTbin]) if qTbin < len(recoil_qTbins)-1 else (recoil_qTbins[qTbin-1] + 20)
        qTmin, qTmax = recoil_qTbins[qTbin-1], recoil_qTbins[qTbin]
        hName = "%s_gen_perp_bin%d" % (sig, qTbin)
        
        hist = fIn.Get(hName)
        hist.Rebin(1)
    
        norm = hist.Integral()
        hist.Scale(1./norm)
        rdh = ROOT.RooDataHist("rdh", "", ROOT.RooArgList(recoil), ROOT.RooFit.Import(hist))
        
        # set variables
        #mean1.setVal(func_mean1.Eval(qT))
        #mean1.setConstant(ROOT.kTRUE)
        
        mean1.setVal(0)
        mean1.setConstant(ROOT.kTRUE)
        mean2.setVal(0)
        mean2.setConstant(ROOT.kTRUE)
        mean3.setVal(0)
        mean3.setConstant(ROOT.kTRUE)

        fitRes = pdf.fitTo(rdh, ROOT.RooFit.SumW2Error(ROOT.kTRUE), ROOT.RooFit.Save(ROOT.kTRUE)) # ROOT.RooFit.Extended(ROOT.kTRUE), 
        fitValid = (fitRes.covQual() == 3 and fitRes.status() == 0)
        #mean2.setVal(mean1.getVal())
        #mean3.setVal(mean1.getVal())    

        ## plot
        plotter.cfg = cfg
        canvas, padT, padB = plotter.canvasRatio()
        dummyT, dummyB, dummyL = plotter.dummyRatio(line=0)
        
        canvas.cd()
        padT.Draw()
        padT.cd()
        dummyT.Draw("HIST")
        
        ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.ERROR)
        plt = recoil.frame()
        rdh.plotOn(plt)
        if fitValid: pdf.plotOn(plt, ROOT.RooFit.VisualizeError(fitRes, 1), ROOT.RooFit.Name("errorband"), ROOT.RooFit.FillColor(ROOT.kGreen+1))
        else: pdf.plotOn(plt, ROOT.RooFit.VisualizeError(fitRes, 1), ROOT.RooFit.Name("errorband"), ROOT.RooFit.FillColor(ROOT.kOrange))
        gauss1.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kBlue), ROOT.RooFit.Normalization(norm1.getVal(), ROOT.RooAbsReal.NumEvent), ROOT.RooFit.LineStyle(ROOT.kDashed))
        gauss2.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kGreen), ROOT.RooFit.Normalization(norm2.getVal(), ROOT.RooAbsReal.NumEvent), ROOT.RooFit.LineStyle(ROOT.kDashed))
        gauss3.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kCyan), ROOT.RooFit.Normalization(norm3.getVal(), ROOT.RooAbsReal.NumEvent), ROOT.RooFit.LineStyle(ROOT.kDashed))
        pdf.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kRed))
        histpull = plt.pullHist()
        chi2 = plt.chiSquare() 

        latex = ROOT.TLatex()
        latex.SetNDC()
        latex.SetTextSize(0.040)
        latex.SetTextColor(1)
        latex.SetTextFont(42)
        latex.DrawLatex(0.20, 0.85, "%s, MC" % label)
        latex.DrawLatex(0.20, 0.80, "q_{T} = [%.1f, %.1f] GeV" % (qTmin, qTmax))
        latex.DrawLatex(0.20, 0.75, "Mean = %.3f #pm %.3f" % (hist.GetMean(), hist.GetMeanError()))
        latex.DrawLatex(0.20, 0.70, "RMS = %.3f #pm %.3f" % (hist.GetRMS(), hist.GetRMSError()))
        latex.DrawLatex(0.20, 0.65, "#chi^{2}/ndof = %.3f" % chi2)
        latex.DrawLatex(0.20, 0.60, "#color[%d]{Cov. matrix quality = %d}" % (2 if fitRes.covQual() != 3 else 1, fitRes.covQual()))
        latex.DrawLatex(0.20, 0.55, "#color[%d]{MINUIT status = %d}" % (2 if fitRes.status() != 0 else 1, fitRes.status()))
        
            
        latex.DrawLatex(0.60, 0.85, "#mu_{1} = %.3f #pm %.3f" % (mean1.getVal(), mean1.getError()))
        latex.DrawLatex(0.60, 0.80, "#mu_{2} = %.3f #pm %.3f" % (mean2.getVal(), mean2.getError()))
        latex.DrawLatex(0.60, 0.75, "#mu_{3} = %.3f #pm %.3f" % (mean3.getVal(), mean3.getError()))
        latex.DrawLatex(0.60, 0.70, "#sigma_{1} = %.3f #pm %.3f" % (sigma1.getVal(), sigma1.getError()))
        latex.DrawLatex(0.60, 0.65, "#sigma_{2} = %.3f #pm %.3f" % (sigma2.getVal(), sigma2.getError()))
        latex.DrawLatex(0.60, 0.60, "#sigma_{3} = %.3f #pm %.3f" % (sigma3.getVal(), sigma3.getError()))
        latex.DrawLatex(0.60, 0.55, "N_{1} = %.3f #pm %.3f" % (norm1.getVal(), norm1.getError()))
        latex.DrawLatex(0.60, 0.50, "N_{2} = %.3f #pm %.3f" % (norm2.getVal(), norm2.getError()))
        
        plt.Draw("SAME")
        plotter.auxRatio()

        canvas.cd()
        padB.Draw()
        padB.cd()
        dummyB.Draw("HIST")
        plt = recoil.frame()
        plt.addPlotable(histpull, "P")
        plt.Draw("SAME")
        dummyL.Draw("SAME")

        canvas.Modify()
        canvas.Update()
        canvas.Draw()
        canvas.SaveAs("%s/%s/%03d_recoil.png" % (outDir, tag, qTbin))
        #canvas.SaveAs("%s/%03d_recoil_%s.pdf" % (outDir, qTbin, name))
        ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)   
        dummyB.Delete()
        dummyT.Delete()
        padT.Delete()
        padB.Delete()
        
        # save output
        hOut.SetBinContent(qTbin, 1, 1, mean1.getVal())
        hOut.SetBinContent(qTbin, 1, 0, mean1.getError())
        hOut.SetBinContent(qTbin, 2, 1, sigma1.getVal())
        hOut.SetBinContent(qTbin, 2, 0, sigma1.getError())
        hOut.SetBinContent(qTbin, 3, 1, norm1.getVal())
        hOut.SetBinContent(qTbin, 3, 0, norm1.getError())
        
        hOut.SetBinContent(qTbin, 4, 1, mean2.getVal())
        hOut.SetBinContent(qTbin, 4, 0, mean2.getError())
        hOut.SetBinContent(qTbin, 5, 1, sigma2.getVal())
        hOut.SetBinContent(qTbin, 5, 0, sigma2.getError())
        hOut.SetBinContent(qTbin, 6, 1, norm2.getVal())
        hOut.SetBinContent(qTbin, 6, 0, norm2.getError())
        
        hOut.SetBinContent(qTbin, 7, 1, mean3.getVal())
        hOut.SetBinContent(qTbin, 7, 0, mean3.getError())
        hOut.SetBinContent(qTbin, 8, 1, sigma3.getVal())
        hOut.SetBinContent(qTbin, 8, 0, sigma3.getError())
        hOut.SetBinContent(qTbin, 9, 1, norm3.getVal())
        hOut.SetBinContent(qTbin, 9, 0, 0)
        
        hOut.SetBinContent(qTbin, 10, 1, norm)
        hOut.SetBinContent(qTbin, 0, 0, chi2)
        
        g_chi2.SetPoint(qTbin-1, qT, chi2)

        # diagonalize covariance matrix and store perturbations
        floatingParams = fitRes.floatParsFinal() # floating parameters
        variations = diagonalize(fitRes)
        # print(p.GetName(), p.isConstant(), type(p), isinstance(p, ROOT.RooFormulaVar))
        for iVar, var in enumerate(variations):
            for iPar, p in enumerate(params):
                if fitValid:
                    if getParamIdx(p.GetName(), floatingParams) > -1: val = var[getParamIdx(p.GetName(), floatingParams)]
                    else: val = p.getVal()
                    hOut.SetBinContent(qTbin, iPar+1, iVar+2, val)
                    print(iVar, iPar, p.GetName(), val)
                else: hOut.SetBinContent(qTbin, iPar+1, iVar+2, p.getVal())


    fOut = ROOT.TFile(fOut_.replace(".root", "_mc_gen_perp.root"), "RECREATE")
    hOut.Write()
    fOut.Close()
    
    
    # plot the chi2
    cfg_chi2 = {

        'logy'              : False,
        'logx'              : False,
        
        'xmin'              : 0,
        'xmax'              : 100,
        'ymin'              : 0,
        'ymax'              : 3,
            
        'xtitle'            : "q_{T} (GeV)",
        'ytitle'            : "#chi^{2}",
            
        'topRight'          : "199 pb^{#minus1} (13 TeV)", 
        'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",

    } 
    

    plotter.cfg = cfg_chi2
    canvas = plotter.canvas()
    dummy = plotter.dummy()   
    canvas.cd()
    dummy.Draw("HIST")
    g_chi2.Draw("SAME LP")
    plotter.aux()
    
    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.035)
    latex.SetTextColor(1)
    latex.SetTextFont(42)
    latex.DrawLatex(0.20, 0.90, "%s, MC" % label)
    latex.DrawLatex(0.20, 0.85, "U_{#perp}")
    
    canvas.Modify()
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs("%s/%s/chi2.png" % (outDir, tag))
    canvas.SaveAs("%s/%s/chi2.pdf" % (outDir, tag))
    canvas.Delete()


def doFit_mc_para():

    tag = "mc_para"
    cfg['xtitle'] = "U_{#parallel} + q_{T} (GeV)"
    cfg['xmin'], cfg['xmax'] = -100, 100
    cfg['ymax'] = 1e3
    cfg['ymin'] = 1e-5
    
    functions.prepareDir("%s/%s" % (outDir, tag))
    
    hOut = ROOT.TH3D(tag, "", 150, 0, 150, 20, 0, 20, 20, 0, 20) # qt, nominal, error
    
    if flavor == "mumu" and met == "DeepMETReso":
        mean1_, mean2_, mean3_, norm1_, norm2_, sigma1_, sigma2_, sigma3_ = 0, 0, 0,   0.6, 0.3,   6, 4, 10
        mean1_, mean2_, mean3_, norm1_, norm2_, sigma1_, sigma2_, sigma3_ = 0, 0, 0,   0.6, 0.3,   6, 4, 10
        
    if flavor == "mumu" and met == "RawPFMET":
        mean1_, mean2_, mean3_, norm1_, norm2_, sigma1_, sigma2_, sigma3_ = 0.2, 0.2, 0.2,    0.5, 0.35,     8, 5, 15
    
    # model
    recoil = ROOT.RooRealVar("recoil", "Recoil parallel (GeV)", 0, -500, 500)   
    mean1 = ROOT.RooRealVar("mean1", "", mean1_, -500, 500)
    mean2 = ROOT.RooRealVar("mean2", "", mean2_, -500, 500)
    mean3 = ROOT.RooRealVar("mean3", "", mean3_, -500, 500)
    sigma1 = ROOT.RooRealVar("sigma1", "", sigma1_, 0.1, 100)
    sigma2 = ROOT.RooRealVar("sigma2", "", sigma2_, 0.1, 100)
    sigma3 = ROOT.RooRealVar("sigma3", "", sigma3_, 0.1, 100)
    norm1 = ROOT.RooRealVar("norm1", "", norm1_, 0, 1)
    norm2 = ROOT.RooRealVar("norm2", "", norm2_, 0, 1)
    norm3 = ROOT.RooFormulaVar("norm3", "(1. - @0 - @1)", ROOT.RooArgList(norm1, norm2))
    gauss1 = ROOT.RooGaussian("gauss1", "gauss1", recoil, mean1, sigma1)
    gauss2 = ROOT.RooGaussian("gauss2", "gauss2", recoil, mean1, sigma2)
    gauss3 = ROOT.RooGaussian("gauss3", "gauss3", recoil, mean1, sigma3)
    pdf = ROOT.RooAddPdf("pdf", '', ROOT.RooArgList(gauss1, gauss2, gauss3), ROOT.RooArgList(norm1, norm2))
    
    norm1.setConstant(ROOT.kTRUE)
    norm2.setConstant(ROOT.kTRUE)
    
    g_chi2 = ROOT.TGraphErrors()
    g_chi2.SetLineColor(ROOT.kRed)
    g_chi2.SetMarkerStyle(8)
    g_chi2.SetMarkerSize(1)
    g_chi2.SetMarkerColor(ROOT.kRed)
    
    params = [mean1, sigma1, norm1, mean2, sigma2, norm2, mean3, sigma3, norm3]
    
    for qTbin in range(1, len(recoil_qTbins)):

        qT = 0.5*(recoil_qTbins[qTbin-1] + recoil_qTbins[qTbin]) if qTbin < len(recoil_qTbins)-1 else (recoil_qTbins[qTbin-1] + 20)
        qTmin, qTmax = recoil_qTbins[qTbin-1], recoil_qTbins[qTbin]
        hName = "%s_para_bin%d" % (sig, qTbin)
        
        
        hist = fIn.Get(hName)
        hist.Rebin(1)
    
        norm = hist.Integral()
        hist.Scale(1./norm)
        rdh = ROOT.RooDataHist("rdh", "", ROOT.RooArgList(recoil), ROOT.RooFit.Import(hist))
       

        fitRes = pdf.fitTo(rdh, ROOT.RooFit.SumW2Error(ROOT.kTRUE), ROOT.RooFit.Save(ROOT.kTRUE)) # ROOT.RooFit.Extended(ROOT.kTRUE), 
        fitValid = (fitRes.covQual() == 3 and fitRes.status() == 0)
        
        mean2.setVal(mean1.getVal())
        mean3.setVal(mean1.getVal())

        ## plot
        plotter.cfg = cfg
        canvas, padT, padB = plotter.canvasRatio()
        dummyT, dummyB, dummyL = plotter.dummyRatio(line=0)
        
        canvas.cd()
        padT.Draw()
        padT.cd()
        dummyT.Draw("HIST")
        
        ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.ERROR)
        plt = recoil.frame()
        rdh.plotOn(plt)
        if fitValid: pdf.plotOn(plt, ROOT.RooFit.VisualizeError(fitRes, 1), ROOT.RooFit.Name("errorband"), ROOT.RooFit.FillColor(ROOT.kGreen+1))
        else: pdf.plotOn(plt, ROOT.RooFit.VisualizeError(fitRes, 1), ROOT.RooFit.Name("errorband"), ROOT.RooFit.FillColor(ROOT.kOrange))
        gauss1.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kBlue), ROOT.RooFit.Normalization(norm1.getVal(), ROOT.RooAbsReal.NumEvent), ROOT.RooFit.LineStyle(ROOT.kDashed))
        gauss2.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kGreen), ROOT.RooFit.Normalization(norm2.getVal(), ROOT.RooAbsReal.NumEvent), ROOT.RooFit.LineStyle(ROOT.kDashed))
        gauss3.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kCyan), ROOT.RooFit.Normalization(norm3.getVal(), ROOT.RooAbsReal.NumEvent), ROOT.RooFit.LineStyle(ROOT.kDashed))
        pdf.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kRed))
        histpull = plt.pullHist()
        chi2 = plt.chiSquare() 

        latex = ROOT.TLatex()
        latex.SetNDC()
        latex.SetTextSize(0.040)
        latex.SetTextColor(1)
        latex.SetTextFont(42)
        latex.DrawLatex(0.20, 0.85, "%s, MC" % label)
        latex.DrawLatex(0.20, 0.80, "q_{T} = [%.1f, %.1f] GeV" % (qTmin, qTmax))
        latex.DrawLatex(0.20, 0.75, "Mean = %.3f #pm %.3f" % (hist.GetMean(), hist.GetMeanError()))
        latex.DrawLatex(0.20, 0.70, "RMS = %.3f #pm %.3f" % (hist.GetRMS(), hist.GetRMSError()))
        latex.DrawLatex(0.20, 0.65, "#chi^{2}/ndof = %.3f" % chi2)
        latex.DrawLatex(0.20, 0.60, "#color[%d]{Cov. matrix quality = %d}" % (2 if fitRes.covQual() != 3 else 1, fitRes.covQual()))
        latex.DrawLatex(0.20, 0.55, "#color[%d]{MINUIT status = %d}" % (2 if fitRes.status() != 0 else 1, fitRes.status()))
        
            
        latex.DrawLatex(0.60, 0.85, "#mu_{1} = %.3f #pm %.3f" % (mean1.getVal(), mean1.getError()))
        latex.DrawLatex(0.60, 0.80, "#mu_{2} = %.3f #pm %.3f" % (mean2.getVal(), mean2.getError()))
        latex.DrawLatex(0.60, 0.75, "#mu_{3} = %.3f #pm %.3f" % (mean3.getVal(), mean3.getError()))
        latex.DrawLatex(0.60, 0.70, "#sigma_{1} = %.3f #pm %.3f" % (sigma1.getVal(), sigma1.getError()))
        latex.DrawLatex(0.60, 0.65, "#sigma_{2} = %.3f #pm %.3f" % (sigma2.getVal(), sigma2.getError()))
        latex.DrawLatex(0.60, 0.60, "#sigma_{3} = %.3f #pm %.3f" % (sigma3.getVal(), sigma3.getError()))
        latex.DrawLatex(0.60, 0.55, "N_{1} = %.3f #pm %.3f" % (norm1.getVal(), norm1.getError()))
        latex.DrawLatex(0.60, 0.50, "N_{2} = %.3f #pm %.3f" % (norm2.getVal(), norm2.getError()))
        
        plt.Draw("SAME")
        plotter.auxRatio()

        canvas.cd()
        padB.Draw()
        padB.cd()
        dummyB.Draw("HIST")
        plt = recoil.frame()
        plt.addPlotable(histpull, "P")
        plt.Draw("SAME")
        dummyL.Draw("SAME")

        canvas.Modify()
        canvas.Update()
        canvas.Draw()
        canvas.SaveAs("%s/%s/%03d_recoil.png" % (outDir, tag, qTbin))
        #canvas.SaveAs("%s/%03d_recoil_%s.pdf" % (outDir, qTbin, name))
        ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)   
        dummyB.Delete()
        dummyT.Delete()
        padT.Delete()
        padB.Delete()
        
        # save output
        hOut.SetBinContent(qTbin, 1, 1, mean1.getVal())
        hOut.SetBinContent(qTbin, 1, 0, mean1.getError())
        hOut.SetBinContent(qTbin, 2, 1, sigma1.getVal())
        hOut.SetBinContent(qTbin, 2, 0, sigma1.getError())
        hOut.SetBinContent(qTbin, 3, 1, norm1.getVal())
        hOut.SetBinContent(qTbin, 3, 0, norm1.getError())
        
        hOut.SetBinContent(qTbin, 4, 1, mean2.getVal())
        hOut.SetBinContent(qTbin, 4, 0, mean2.getError())
        hOut.SetBinContent(qTbin, 5, 1, sigma2.getVal())
        hOut.SetBinContent(qTbin, 5, 0, sigma2.getError())
        hOut.SetBinContent(qTbin, 6, 1, norm2.getVal())
        hOut.SetBinContent(qTbin, 6, 0, norm2.getError())
        
        hOut.SetBinContent(qTbin, 7, 1, mean3.getVal())
        hOut.SetBinContent(qTbin, 7, 0, mean3.getError())
        hOut.SetBinContent(qTbin, 8, 1, sigma3.getVal())
        hOut.SetBinContent(qTbin, 8, 0, sigma3.getError())
        hOut.SetBinContent(qTbin, 9, 1, norm3.getVal())
        hOut.SetBinContent(qTbin, 9, 0, 0)
        
        hOut.SetBinContent(qTbin, 10, 1, norm)
        hOut.SetBinContent(qTbin, 0, 0, chi2)
        
        g_chi2.SetPoint(qTbin-1, qT, chi2)

        # diagonalize covariance matrix and store perturbations
        floatingParams = fitRes.floatParsFinal() # floating parameters
        variations = diagonalize(fitRes)
        # print(p.GetName(), p.isConstant(), type(p), isinstance(p, ROOT.RooFormulaVar))
        for iVar, var in enumerate(variations):
            for iPar, p in enumerate(params):
                if fitValid:
                    if getParamIdx(p.GetName(), floatingParams) > -1: val = var[getParamIdx(p.GetName(), floatingParams)]
                    else: val = p.getVal()
                    hOut.SetBinContent(qTbin, iPar+1, iVar+2, val)
                    print(iVar, iPar, p.GetName(), val)
                else: hOut.SetBinContent(qTbin, iPar+1, iVar+2, p.getVal())


    fOut = ROOT.TFile(fOut_.replace(".root", "_mc_para.root"), "RECREATE")
    hOut.Write()
    fOut.Close()
    
    
    # plot the chi2
    cfg_chi2 = {

        'logy'              : False,
        'logx'              : False,
        
        'xmin'              : 0,
        'xmax'              : 100,
        'ymin'              : 0,
        'ymax'              : 3,
            
        'xtitle'            : "q_{T} (GeV)",
        'ytitle'            : "#chi^{2}",
            
        'topRight'          : "199 pb^{#minus1} (13 TeV)", 
        'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",

    } 
    

    plotter.cfg = cfg_chi2
    canvas = plotter.canvas()
    dummy = plotter.dummy()   
    canvas.cd()
    dummy.Draw("HIST")
    g_chi2.Draw("SAME LP")
    plotter.aux()
    
    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.035)
    latex.SetTextColor(1)
    latex.SetTextFont(42)
    latex.DrawLatex(0.20, 0.90, "%s, MC" % label)
    latex.DrawLatex(0.20, 0.85, "U_{#parallel} + q_{T} ")
    
    canvas.Modify()
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs("%s/%s/chi2.png" % (outDir, tag))
    canvas.SaveAs("%s/%s/chi2.pdf" % (outDir, tag))
    canvas.Delete()




def doFit_mc_gen_para():

    tag = "mc_gen_para"
    cfg['xtitle'] = "U_{#parallel} + q_{T} (GeV)"
    cfg['xmin'], cfg['xmax'] = -100, 100
    cfg['ymax'] = 1e3
    cfg['ymin'] = 1e-5
    
    functions.prepareDir("%s/%s" % (outDir, tag))
    
    hOut = ROOT.TH3D(tag, "", 150, 0, 150, 20, 0, 20, 20, 0, 20) # qt, nominal, error
    
    if flavor == "mumu" and met == "DeepMETReso":
        mean1_, mean2_, mean3_, norm1_, norm2_, sigma1_, sigma2_, sigma3_ = 0, 0, 0,   0.6, 0.3,   6, 4, 10
        mean1_, mean2_, mean3_, norm1_, norm2_, sigma1_, sigma2_, sigma3_ = 0, 0, 0,   0.6, 0.3,   6, 4, 10
        
    if flavor == "mumu" and met == "RawPFMET":
        mean1_, mean2_, mean3_, norm1_, norm2_, sigma1_, sigma2_, sigma3_ = 0.2, 0.2, 0.2,    0.5, 0.35,     8, 5, 15
        
    if flavor == "mu" and met == "RawPFMET":
        mean1_, mean2_, mean3_, norm1_, norm2_, sigma1_, sigma2_, sigma3_ = 0.2, 0.2, 0.2,    0.5, 0.35,     8, 5, 15
    
    # model
    recoil = ROOT.RooRealVar("recoil", "Recoil parallel (GeV)", 0, -500, 500)   
    mean1 = ROOT.RooRealVar("mean1", "", mean1_, -500, 500)
    mean2 = ROOT.RooRealVar("mean2", "", mean2_, -500, 500)
    mean3 = ROOT.RooRealVar("mean3", "", mean3_, -500, 500)
    sigma1 = ROOT.RooRealVar("sigma1", "", sigma1_, 0.1, 100)
    sigma2 = ROOT.RooRealVar("sigma2", "", sigma2_, 0.1, 100)
    sigma3 = ROOT.RooRealVar("sigma3", "", sigma3_, 0.1, 100)
    norm1 = ROOT.RooRealVar("norm1", "", norm1_, 0, 1)
    norm2 = ROOT.RooRealVar("norm2", "", norm2_, 0, 1)
    norm3 = ROOT.RooFormulaVar("norm3", "(1. - @0 - @1)", ROOT.RooArgList(norm1, norm2))
    gauss1 = ROOT.RooGaussian("gauss1", "gauss1", recoil, mean1, sigma1)
    gauss2 = ROOT.RooGaussian("gauss2", "gauss2", recoil, mean1, sigma2)
    gauss3 = ROOT.RooGaussian("gauss3", "gauss3", recoil, mean1, sigma3)
    pdf = ROOT.RooAddPdf("pdf", '', ROOT.RooArgList(gauss1, gauss2, gauss3), ROOT.RooArgList(norm1, norm2))
    
    norm1.setConstant(ROOT.kTRUE)
    norm2.setConstant(ROOT.kTRUE)
    
    g_chi2 = ROOT.TGraphErrors()
    g_chi2.SetLineColor(ROOT.kRed)
    g_chi2.SetMarkerStyle(8)
    g_chi2.SetMarkerSize(1)
    g_chi2.SetMarkerColor(ROOT.kRed)
    
    params = [mean1, sigma1, norm1, mean2, sigma2, norm2, mean3, sigma3, norm3]
    
    for qTbin in range(1, len(recoil_qTbins)):

        qT = 0.5*(recoil_qTbins[qTbin-1] + recoil_qTbins[qTbin]) if qTbin < len(recoil_qTbins)-1 else (recoil_qTbins[qTbin-1] + 20)
        qTmin, qTmax = recoil_qTbins[qTbin-1], recoil_qTbins[qTbin]
        hName = "%s_gen_para_bin%d" % (sig, qTbin)
        
        hist = fIn.Get(hName)
        hist.Rebin(1)
    
        norm = hist.Integral()
        hist.Scale(1./norm)
        rdh = ROOT.RooDataHist("rdh", "", ROOT.RooArgList(recoil), ROOT.RooFit.Import(hist))
       

        fitRes = pdf.fitTo(rdh, ROOT.RooFit.SumW2Error(ROOT.kTRUE), ROOT.RooFit.Save(ROOT.kTRUE)) # ROOT.RooFit.Extended(ROOT.kTRUE), 
        fitValid = (fitRes.covQual() == 3 and fitRes.status() == 0)
        
        mean2.setVal(mean1.getVal())
        mean3.setVal(mean1.getVal())

        ## plot
        plotter.cfg = cfg
        canvas, padT, padB = plotter.canvasRatio()
        dummyT, dummyB, dummyL = plotter.dummyRatio(line=0)
        
        canvas.cd()
        padT.Draw()
        padT.cd()
        dummyT.Draw("HIST")
        
        ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.ERROR)
        plt = recoil.frame()
        rdh.plotOn(plt)
        if fitValid: pdf.plotOn(plt, ROOT.RooFit.VisualizeError(fitRes, 1), ROOT.RooFit.Name("errorband"), ROOT.RooFit.FillColor(ROOT.kGreen+1))
        else: pdf.plotOn(plt, ROOT.RooFit.VisualizeError(fitRes, 1), ROOT.RooFit.Name("errorband"), ROOT.RooFit.FillColor(ROOT.kOrange))
        gauss1.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kBlue), ROOT.RooFit.Normalization(norm1.getVal(), ROOT.RooAbsReal.NumEvent), ROOT.RooFit.LineStyle(ROOT.kDashed))
        gauss2.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kGreen), ROOT.RooFit.Normalization(norm2.getVal(), ROOT.RooAbsReal.NumEvent), ROOT.RooFit.LineStyle(ROOT.kDashed))
        gauss3.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kCyan), ROOT.RooFit.Normalization(norm3.getVal(), ROOT.RooAbsReal.NumEvent), ROOT.RooFit.LineStyle(ROOT.kDashed))
        pdf.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kRed))
        histpull = plt.pullHist()
        chi2 = plt.chiSquare() 

        latex = ROOT.TLatex()
        latex.SetNDC()
        latex.SetTextSize(0.040)
        latex.SetTextColor(1)
        latex.SetTextFont(42)
        latex.DrawLatex(0.20, 0.85, "%s, MC" % label)
        latex.DrawLatex(0.20, 0.80, "q_{T} = [%.1f, %.1f] GeV" % (qTmin, qTmax))
        latex.DrawLatex(0.20, 0.75, "Mean = %.3f #pm %.3f" % (hist.GetMean(), hist.GetMeanError()))
        latex.DrawLatex(0.20, 0.70, "RMS = %.3f #pm %.3f" % (hist.GetRMS(), hist.GetRMSError()))
        latex.DrawLatex(0.20, 0.65, "#chi^{2}/ndof = %.3f" % chi2)
        latex.DrawLatex(0.20, 0.60, "#color[%d]{Cov. matrix quality = %d}" % (2 if fitRes.covQual() != 3 else 1, fitRes.covQual()))
        latex.DrawLatex(0.20, 0.55, "#color[%d]{MINUIT status = %d}" % (2 if fitRes.status() != 0 else 1, fitRes.status()))
        
            
        latex.DrawLatex(0.60, 0.85, "#mu_{1} = %.3f #pm %.3f" % (mean1.getVal(), mean1.getError()))
        latex.DrawLatex(0.60, 0.80, "#mu_{2} = %.3f #pm %.3f" % (mean2.getVal(), mean2.getError()))
        latex.DrawLatex(0.60, 0.75, "#mu_{3} = %.3f #pm %.3f" % (mean3.getVal(), mean3.getError()))
        latex.DrawLatex(0.60, 0.70, "#sigma_{1} = %.3f #pm %.3f" % (sigma1.getVal(), sigma1.getError()))
        latex.DrawLatex(0.60, 0.65, "#sigma_{2} = %.3f #pm %.3f" % (sigma2.getVal(), sigma2.getError()))
        latex.DrawLatex(0.60, 0.60, "#sigma_{3} = %.3f #pm %.3f" % (sigma3.getVal(), sigma3.getError()))
        latex.DrawLatex(0.60, 0.55, "N_{1} = %.3f #pm %.3f" % (norm1.getVal(), norm1.getError()))
        latex.DrawLatex(0.60, 0.50, "N_{2} = %.3f #pm %.3f" % (norm2.getVal(), norm2.getError()))
        
        plt.Draw("SAME")
        plotter.auxRatio()

        canvas.cd()
        padB.Draw()
        padB.cd()
        dummyB.Draw("HIST")
        plt = recoil.frame()
        plt.addPlotable(histpull, "P")
        plt.Draw("SAME")
        dummyL.Draw("SAME")

        canvas.Modify()
        canvas.Update()
        canvas.Draw()
        canvas.SaveAs("%s/%s/%03d_recoil.png" % (outDir, tag, qTbin))
        #canvas.SaveAs("%s/%03d_recoil_%s.pdf" % (outDir, qTbin, name))
        ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)   
        dummyB.Delete()
        dummyT.Delete()
        padT.Delete()
        padB.Delete()
        
        # save output
        hOut.SetBinContent(qTbin, 1, 1, mean1.getVal())
        hOut.SetBinContent(qTbin, 1, 0, mean1.getError())
        hOut.SetBinContent(qTbin, 2, 1, sigma1.getVal())
        hOut.SetBinContent(qTbin, 2, 0, sigma1.getError())
        hOut.SetBinContent(qTbin, 3, 1, norm1.getVal())
        hOut.SetBinContent(qTbin, 3, 0, norm1.getError())
        
        hOut.SetBinContent(qTbin, 4, 1, mean2.getVal())
        hOut.SetBinContent(qTbin, 4, 0, mean2.getError())
        hOut.SetBinContent(qTbin, 5, 1, sigma2.getVal())
        hOut.SetBinContent(qTbin, 5, 0, sigma2.getError())
        hOut.SetBinContent(qTbin, 6, 1, norm2.getVal())
        hOut.SetBinContent(qTbin, 6, 0, norm2.getError())
        
        hOut.SetBinContent(qTbin, 7, 1, mean3.getVal())
        hOut.SetBinContent(qTbin, 7, 0, mean3.getError())
        hOut.SetBinContent(qTbin, 8, 1, sigma3.getVal())
        hOut.SetBinContent(qTbin, 8, 0, sigma3.getError())
        hOut.SetBinContent(qTbin, 9, 1, norm3.getVal())
        hOut.SetBinContent(qTbin, 9, 0, 0)
        
        hOut.SetBinContent(qTbin, 10, 1, norm)
        hOut.SetBinContent(qTbin, 0, 0, chi2)
        
        g_chi2.SetPoint(qTbin-1, qT, chi2)

        # diagonalize covariance matrix and store perturbations
        floatingParams = fitRes.floatParsFinal() # floating parameters
        variations = diagonalize(fitRes)
        # print(p.GetName(), p.isConstant(), type(p), isinstance(p, ROOT.RooFormulaVar))
        for iVar, var in enumerate(variations):
            for iPar, p in enumerate(params):
                if fitValid:
                    if getParamIdx(p.GetName(), floatingParams) > -1: val = var[getParamIdx(p.GetName(), floatingParams)]
                    else: val = p.getVal()
                    hOut.SetBinContent(qTbin, iPar+1, iVar+2, val)
                    print(iVar, iPar, p.GetName(), val)
                else: hOut.SetBinContent(qTbin, iPar+1, iVar+2, p.getVal())


    fOut = ROOT.TFile(fOut_.replace(".root", "_mc_gen_para.root"), "RECREATE")
    hOut.Write()
    fOut.Close()
    
    
    # plot the chi2
    cfg_chi2 = {

        'logy'              : False,
        'logx'              : False,
        
        'xmin'              : 0,
        'xmax'              : 100,
        'ymin'              : 0,
        'ymax'              : 3,
            
        'xtitle'            : "q_{T} (GeV)",
        'ytitle'            : "#chi^{2}",
            
        'topRight'          : "199 pb^{#minus1} (13 TeV)", 
        'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",

    } 
    

    plotter.cfg = cfg_chi2
    canvas = plotter.canvas()
    dummy = plotter.dummy()   
    canvas.cd()
    dummy.Draw("HIST")
    g_chi2.Draw("SAME LP")
    plotter.aux()
    
    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.035)
    latex.SetTextColor(1)
    latex.SetTextFont(42)
    latex.DrawLatex(0.20, 0.90, "%s, MC" % label)
    latex.DrawLatex(0.20, 0.85, "U_{#parallel} + q_{T} ")
    
    canvas.Modify()
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs("%s/%s/chi2.png" % (outDir, tag))
    canvas.SaveAs("%s/%s/chi2.pdf" % (outDir, tag))
    canvas.Delete()


def doFit_bkg_perp(recoil, ws, bkg, hist_bkg, qTbin, qT, qTmin, qTmax):

    mean1_bkg_, norm1_bkg_, sigma1_bkg_, sigma2_bkg_ = 0, 0.6, 10, 15

    mean1_bkg = ROOT.RooRealVar("mean1_bkg_%s_bin%d" % (bkg, qTbin), "", mean1_bkg_, -500, 500)
    sigma1_bkg = ROOT.RooRealVar("sigma1_bkg_%s_bin%d" % (bkg, qTbin), "", sigma1_bkg_, 0.1, 100)
    sigma2_bkg = ROOT.RooRealVar("sigma2_bkg_%s_bin%d" % (bkg, qTbin), "", sigma2_bkg_, 0.1, 100)
    norm1_bkg = ROOT.RooRealVar("norm1_bkg_%s_bin%d" % (bkg, qTbin), "", norm1_bkg_, 0, 1)
    gauss1_bkg = ROOT.RooGaussian("gauss1_bkg_%s_bin%d" % (bkg, qTbin), "", recoil, mean1_bkg, sigma1_bkg)
    gauss2_bkg = ROOT.RooGaussian("gauss2_bkg_%s_bin%d" % (bkg, qTbin), "", recoil, mean1_bkg, sigma2_bkg)
    pdf_bkg = ROOT.RooAddPdf("pdf_bkg_%s_bin%d" % (bkg, qTbin), '', ROOT.RooArgList(gauss1_bkg, gauss2_bkg), ROOT.RooArgList(norm1_bkg))

    norm_bkg = hist_bkg.Integral()
    hist_bkg.Scale(1./norm_bkg)
        
    tag = "%s_perp" % bkg
    functions.prepareDir("%s//%s/" % (outDir, tag), remove=False)
    cfg['xmin'], cfg['xmax'] = -300, 300
    rdh_bkg = ROOT.RooDataHist("rdh_bkg_%s_bin%d" % (bkg, qTbin), "", ROOT.RooArgList(recoil), ROOT.RooFit.Import(hist_bkg))
    fitRes_bkg = pdf_bkg.fitTo(rdh_bkg, ROOT.RooFit.SumW2Error(ROOT.kTRUE), ROOT.RooFit.Save(ROOT.kTRUE))
    fitValid = (fitRes_bkg.covQual() == 3 and fitRes_bkg.status() == 0)
        
    plotter.cfg = cfg
    canvas, padT, padB = plotter.canvasRatio()
    dummyT, dummyB, dummyL = plotter.dummyRatio(line=0)
    canvas.cd()
    padT.Draw()
    padT.cd()
    dummyT.Draw("HIST")
        
    ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.ERROR)
    plt = recoil.frame()
    if fitValid: pdf_bkg.plotOn(plt, ROOT.RooFit.VisualizeError(fitRes_bkg, 1), ROOT.RooFit.Name("errorband"), ROOT.RooFit.FillColor(ROOT.kGreen+1))
    else: pdf_bkg.plotOn(plt, ROOT.RooFit.VisualizeError(fitRes_bkg, 1), ROOT.RooFit.Name("errorband"), ROOT.RooFit.FillColor(ROOT.kOrange))
    rdh_bkg.plotOn(plt)
    gauss1_bkg.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kBlue), ROOT.RooFit.Normalization(norm1_bkg.getVal(), ROOT.RooAbsReal.NumEvent), ROOT.RooFit.LineStyle(ROOT.kDashed))
    gauss2_bkg.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kGreen), ROOT.RooFit.Normalization(1.-norm1_bkg.getVal(), ROOT.RooAbsReal.NumEvent), ROOT.RooFit.LineStyle(ROOT.kDashed))
    pdf_bkg.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kRed))
    histpull = plt.pullHist()
    chi2 = plt.chiSquare() 
 
    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.040)
    latex.SetTextColor(1)
    latex.SetTextFont(42)
    latex.DrawLatex(0.20, 0.85, "%s, %s" % (label, bkg))
    latex.DrawLatex(0.20, 0.80, "q_{T} = [%.1f, %.1f] GeV" % (qTmin, qTmax))
    latex.DrawLatex(0.20, 0.75, "Mean = %.3f #pm %.3f" % (hist_bkg.GetMean(), hist_bkg.GetMeanError()))
    latex.DrawLatex(0.20, 0.70, "RMS = %.3f #pm %.3f" % (hist_bkg.GetRMS(), hist_bkg.GetRMSError()))
    latex.DrawLatex(0.20, 0.65, "#chi^{2}/ndof = %.3f" % chi2)
    latex.DrawLatex(0.20, 0.60, "Cov. matrix quality = %d" % fitRes_bkg.covQual())
    latex.DrawLatex(0.20, 0.55, "MINUIT status = %d" % fitRes_bkg.status())
            
    latex.DrawLatex(0.60, 0.85, "#mu_{1} = %.3f #pm %.3f" % (mean1_bkg.getVal(), mean1_bkg.getError()))
    latex.DrawLatex(0.60, 0.80, "#sigma_{1} = %.3f #pm %.3f" % (sigma1_bkg.getVal(), sigma1_bkg.getError()))
    latex.DrawLatex(0.60, 0.75, "#sigma_{2} = %.3f #pm %.3f" % (sigma2_bkg.getVal(), sigma2_bkg.getError()))
    latex.DrawLatex(0.60, 0.70, "N_{1} = %.3f #pm %.3f" % (norm1_bkg.getVal(), norm1_bkg.getError()))

    plt.Draw("SAME")
    plotter.auxRatio()
    canvas.cd()
    padB.Draw()
    padB.cd()
    dummyB.Draw("HIST")
    plt = recoil.frame()
    plt.addPlotable(histpull, "P")
    plt.Draw("SAME")
    dummyL.Draw("SAME")
    canvas.Modify()
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs("%s/%s/%03d_recoil.png" % (outDir, tag, qTbin))
    #canvas.SaveAs("%s/%03d_recoil_%s.pdf" % (outDir, qTbin, name))
    ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)   
    dummyB.Delete()
    dummyT.Delete()
    padT.Delete()
    padB.Delete()
    
    
    mean1_bkg.setConstant(ROOT.kTRUE)
    sigma1_bkg.setConstant(ROOT.kTRUE)
    sigma2_bkg.setConstant(ROOT.kTRUE)
    norm1_bkg.setConstant(ROOT.kTRUE)
    
    ws.Import(pdf_bkg)
 

def doFit_bkg_para(recoil, ws, bkg, hist_bkg, qTbin, qT, qTmin, qTmax):

    mean1_bkg_, norm1_bkg_, sigma1_bkg_, sigma2_bkg_ = 0, 0.6, 10, 15

    mean1_bkg = ROOT.RooRealVar("mean1_bkg_%s_bin%d" % (bkg, qTbin), "", mean1_bkg_, -500, 500)
    sigma1_bkg = ROOT.RooRealVar("sigma1_bkg_%s_bin%d" % (bkg, qTbin), "", sigma1_bkg_, 0.1, 100)
    sigma2_bkg = ROOT.RooRealVar("sigma2_bkg_%s_bin%d" % (bkg, qTbin), "", sigma2_bkg_, 0.1, 100)
    norm1_bkg = ROOT.RooRealVar("norm1_bkg_%s_bin%d" % (bkg, qTbin), "", norm1_bkg_, 0, 1)
    gauss1_bkg = ROOT.RooGaussian("gauss1_bkg_%s_bin%d" % (bkg, qTbin), "", recoil, mean1_bkg, sigma1_bkg)
    gauss2_bkg = ROOT.RooGaussian("gauss2_bkg_%s_bin%d" % (bkg, qTbin), "", recoil, mean1_bkg, sigma2_bkg)
    pdf_bkg = ROOT.RooAddPdf("pdf_bkg_%s_bin%d" % (bkg, qTbin), '', ROOT.RooArgList(gauss1_bkg, gauss2_bkg), ROOT.RooArgList(norm1_bkg))

    norm_bkg = hist_bkg.Integral()
    hist_bkg.Scale(1./norm_bkg)
        
    tag = "%s_para" % bkg
    functions.prepareDir("%s//%s/" % (outDir, tag), remove=False)
    cfg['xmin'], cfg['xmax'] = -300, 300
    rdh_bkg = ROOT.RooDataHist("rdh_bkg_%s_bin%d" % (bkg, qTbin), "", ROOT.RooArgList(recoil), ROOT.RooFit.Import(hist_bkg))
    fitRes_bkg = pdf_bkg.fitTo(rdh_bkg, ROOT.RooFit.SumW2Error(ROOT.kTRUE), ROOT.RooFit.Save(ROOT.kTRUE))
    fitValid = (fitRes_bkg.covQual() == 3 and fitRes_bkg.status() == 0)
        
    plotter.cfg = cfg
    canvas, padT, padB = plotter.canvasRatio()
    dummyT, dummyB, dummyL = plotter.dummyRatio(line=0)
    canvas.cd()
    padT.Draw()
    padT.cd()
    dummyT.Draw("HIST")
        
    ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.ERROR)
    plt = recoil.frame()
    if fitValid: pdf_bkg.plotOn(plt, ROOT.RooFit.VisualizeError(fitRes_bkg, 1), ROOT.RooFit.Name("errorband"), ROOT.RooFit.FillColor(ROOT.kGreen+1))
    else: pdf_bkg.plotOn(plt, ROOT.RooFit.VisualizeError(fitRes_bkg, 1), ROOT.RooFit.Name("errorband"), ROOT.RooFit.FillColor(ROOT.kOrange))
    rdh_bkg.plotOn(plt)
    gauss1_bkg.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kBlue), ROOT.RooFit.Normalization(norm1_bkg.getVal(), ROOT.RooAbsReal.NumEvent), ROOT.RooFit.LineStyle(ROOT.kDashed))
    gauss2_bkg.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kGreen), ROOT.RooFit.Normalization(1.-norm1_bkg.getVal(), ROOT.RooAbsReal.NumEvent), ROOT.RooFit.LineStyle(ROOT.kDashed))
    pdf_bkg.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kRed))
    histpull = plt.pullHist()
    chi2 = plt.chiSquare() 
 
    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.040)
    latex.SetTextColor(1)
    latex.SetTextFont(42)
    latex.DrawLatex(0.20, 0.85, "%s, %s" % (label, bkg))
    latex.DrawLatex(0.20, 0.80, "q_{T} = [%.1f, %.1f] GeV" % (qTmin, qTmax))
    latex.DrawLatex(0.20, 0.75, "Mean = %.3f #pm %.3f" % (hist_bkg.GetMean(), hist_bkg.GetMeanError()))
    latex.DrawLatex(0.20, 0.70, "RMS = %.3f #pm %.3f" % (hist_bkg.GetRMS(), hist_bkg.GetRMSError()))
    latex.DrawLatex(0.20, 0.65, "#chi^{2}/ndof = %.3f" % chi2)
    latex.DrawLatex(0.20, 0.60, "Cov. matrix quality = %d" % fitRes_bkg.covQual())
    latex.DrawLatex(0.20, 0.55, "MINUIT status = %d" % fitRes_bkg.status())
            
    latex.DrawLatex(0.60, 0.85, "#mu_{1} = %.3f #pm %.3f" % (mean1_bkg.getVal(), mean1_bkg.getError()))
    latex.DrawLatex(0.60, 0.80, "#sigma_{1} = %.3f #pm %.3f" % (sigma1_bkg.getVal(), sigma1_bkg.getError()))
    latex.DrawLatex(0.60, 0.75, "#sigma_{2} = %.3f #pm %.3f" % (sigma2_bkg.getVal(), sigma2_bkg.getError()))
    latex.DrawLatex(0.60, 0.70, "N_{1} = %.3f #pm %.3f" % (norm1_bkg.getVal(), norm1_bkg.getError()))

    plt.Draw("SAME")
    plotter.auxRatio()
    canvas.cd()
    padB.Draw()
    padB.cd()
    dummyB.Draw("HIST")
    plt = recoil.frame()
    plt.addPlotable(histpull, "P")
    plt.Draw("SAME")
    dummyL.Draw("SAME")
    canvas.Modify()
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs("%s/%s/%03d_recoil.png" % (outDir, tag, qTbin))
    #canvas.SaveAs("%s/%03d_recoil_%s.pdf" % (outDir, qTbin, name))
    ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)   
    dummyB.Delete()
    dummyT.Delete()
    padT.Delete()
    padB.Delete()
    
    
    mean1_bkg.setConstant(ROOT.kTRUE)
    sigma1_bkg.setConstant(ROOT.kTRUE)
    sigma2_bkg.setConstant(ROOT.kTRUE)
    norm1_bkg.setConstant(ROOT.kTRUE)
    
    ws.Import(pdf_bkg) 
    
    

def doFit_data_perp_OLD():

    tag = "data_perp"
    cfg['xtitle'] = "U_{#perp}   (GeV)"
    cfg['xmin'], cfg['xmax'] = -100, 100
    cfg['ymax'] = 1e3
    cfg['ymin'] = 1e-5
    
    functions.prepareDir("%s/%s" % (outDir, tag))
    hOut = ROOT.TH3D(tag, "", 150, 0, 150, 20, 0, 20, 20, 0, 20) # qt, nominal, error
    recoil = ROOT.RooRealVar("recoil", "Recoil parallel (GeV)", 0, -300, 300)
    
    if flavor == "mumu" and met == "DeepMETReso":
        mean1_, mean2_, mean3_, norm1_, norm2_, sigma1_, sigma2_, sigma3_ = 0, 0, 0,   0.6, 0.3,   6, 4, 10
        mean1_, mean2_, mean3_, norm1_, norm2_, sigma1_, sigma2_, sigma3_ = 0, 0, 0,   0.6, 0.3,   6, 4, 10
        
    if flavor == "mumu" and met == "RawPFMET":
        mean1_, mean2_, mean3_, norm1_, norm2_, sigma1_, sigma2_, sigma3_ = 0, 0, 0,    0.55, 0.35,     8, 5, 15
        #mean1_, mean2_, mean3_, norm1_, norm2_, sigma1_, sigma2_, sigma3_ = 0, 0, 0,    0.9, 0.1,     8, 5, 10
        #mean1_, mean2_, mean3_, norm1_, norm2_, sigma1_, sigma2_, sigma3_ = 0, 0, 0,    0.75, 0.25,     8, 5, 10
    

    # signal model
    mean1 = ROOT.RooRealVar("mean1", "", mean1_, -300, 300)
    mean2 = ROOT.RooRealVar("mean2", "", mean2_, -300, 300)
    mean3 = ROOT.RooRealVar("mean3", "", mean3_, -300, 300)
    sigma1 = ROOT.RooRealVar("sigma1", "", sigma1_, 0.1, 100)
    sigma2 = ROOT.RooRealVar("sigma2", "", sigma2_, 0.1, 100)
    sigma3 = ROOT.RooRealVar("sigma3", "", sigma3_, 0.1, 100)
    norm1 = ROOT.RooRealVar("norm1", "", norm1_, 0, 1)
    norm2 = ROOT.RooRealVar("norm2", "", norm2_, 0, 1)
    norm3 = ROOT.RooFormulaVar("norm3", "(1. - @0 - @1)", ROOT.RooArgList(norm1, norm2))
    gauss1 = ROOT.RooGaussian("gauss1", "gauss1", recoil, mean1, sigma1)
    gauss2 = ROOT.RooGaussian("gauss2", "gauss2", recoil, mean1, sigma2)
    gauss3 = ROOT.RooGaussian("gauss3", "gauss3", recoil, mean1, sigma3)
    pdf_sig = ROOT.RooAddPdf("pdf_sig", '', ROOT.RooArgList(gauss1, gauss2, gauss3), ROOT.RooArgList(norm1, norm2))
    
        
    #mean1.setVal(0)
    #mean1.setConstant(ROOT.kTRUE)
    #mean2.setVal(0)
    #mean2.setConstant(ROOT.kTRUE)
    #mean3.setVal(0)
    #mean3.setConstant(ROOT.kTRUE)

    norm1.setConstant(ROOT.kTRUE)
    norm2.setConstant(ROOT.kTRUE)
    mean2.setConstant(ROOT.kTRUE)
    mean3.setConstant(ROOT.kTRUE)
        
    
    norm_bkg_var = ROOT.RooRealVar("norm_bkg_var", "", 0, 0, 1) # background fraction w.r.t. data (let it float, constrained by Gauss)
    
    g_chi2 = ROOT.TGraphErrors()
    g_chi2.SetLineColor(ROOT.kRed)
    g_chi2.SetMarkerStyle(8)
    g_chi2.SetMarkerSize(1)
    g_chi2.SetMarkerColor(ROOT.kRed)
    
    params = [mean1, sigma1, norm1, mean2, sigma2, norm2, mean3, sigma3, norm3, norm_bkg_var]
    
    for qTbin in range(1, len(recoil_qTbins)):
    
        qT = 0.5*(recoil_qTbins[qTbin-1] + recoil_qTbins[qTbin]) if qTbin < len(recoil_qTbins)-1 else (recoil_qTbins[qTbin-1] + 20)
        qTmin, qTmax = recoil_qTbins[qTbin-1], recoil_qTbins[qTbin]
       
        hist_data = fIn.Get("SingleMuon_perp_bin%d" % qTbin)
        #hist_data.Rebin(2)
        norm_data = hist_data.Integral()
        hist_data.Scale(1./norm_data)
        
        mean1.setVal(0)
        mean1.setConstant(ROOT.kTRUE)
        mean2.setVal(0)
        mean2.setConstant(ROOT.kTRUE)
        mean3.setVal(0)
        mean3.setConstant(ROOT.kTRUE)
        
        
        sigma3.setConstant(ROOT.kFALSE)
        
        # compute the total background norm (necessary for PDF normalization)
        norm_bkg = 0
        for bkg in bkgs: norm_bkg += fIn.Get("%s_perp_bin%d" % (bkg, qTbin)).Integral()
 

        # do the background fits and construct PDF 
        w_bkg = ROOT.RooWorkspace("w_bin%d" % qTbin, "")
        pdfs_bkg = ROOT.RooArgList()
        norms_bkg = ROOT.RooArgList()
        for bkg in bkgs:
        
            hist_bkg = fIn.Get("%s_perp_bin%d" % (bkg, qTbin))
            norm = hist_bkg.Integral()
            hist_bkg.Rebin(2)
            doFit_bkg_perp(recoil, w_bkg, bkg, hist_bkg, qTbin, qT, qTmin, qTmax)
            
            pdf = w_bkg.pdf("pdf_bkg_%s_bin%d" % (bkg, qTbin))
            pdfs_bkg.add(pdf)
            
            norm_ = ROOT.RooRealVar("norm_bkg_%s_bin%d" % (bkg, qTbin), "", norm/norm_bkg) # fix internal norm
            norms_bkg.addOwned(norm_)

        # construct bkg PDF  
        pdfs_bkg.Print()
        norms_bkg.Print()
        pdf_bkg = ROOT.RooAddPdf("pdf_bkg_bin%d" % qTbin, '', pdfs_bkg, norms_bkg)
        pdf_bkg.Print()
        
        # construct bkg+sig PDF with constraint term on BKG normalization
        val = norm_bkg/norm_data
        #norm_bkg_var = ROOT.RooRealVar("norm_bkg_var_bin%d" % qTbin, "", val, 0, 1) # let it float
        norm_bkg_var.setVal(val)
        if bkg_constr <= 0: norm_bkg_var.setConstant(ROOT.kTRUE)
        norm_bkg_constr = ROOT.RooGaussian("norm_bkg_constr_bin%d" % qTbin, "", norm_bkg_var, ROOT.RooFit.RooConst(val), ROOT.RooFit.RooConst(val*bkg_constr))
        
        pdf_sig_bkg = ROOT.RooAddPdf("pdf_sig_bkg_bin%d" % qTbin, '', ROOT.RooArgList(pdf_bkg, pdf_sig), ROOT.RooArgList(norm_bkg_var)) # a*BKG + SIG
        pdf = ROOT.RooProdPdf("pdf_bin%d" % qTbin, "", ROOT.RooArgSet(pdf_sig_bkg, norm_bkg_constr)) # a*BKG + SIG with constraint term on a
        
        pdf_sig_bkg.Print()
        pdf.Print()
        
        # do the fit
        rdh_data = ROOT.RooDataHist("rdh_data_bin%d" % qTbin, "", ROOT.RooArgList(recoil), ROOT.RooFit.Import(hist_data))
        doFit = True
        nTrials = 0
        while doFit:
            
            fitRes_data = pdf.fitTo(rdh_data, ROOT.RooFit.SumW2Error(ROOT.kTRUE), ROOT.RooFit.Save(ROOT.kTRUE), ROOT.RooFit.Constrain(norm_bkg_constr))
            fitValid = (fitRes_data.covQual() == 3 and fitRes_data.status() == 0)
        
            nTrials += 1
            # do check on sigma3
            if sigma3.getError() > 0.9*sigma3.getVal() or not fitValid:
                
                print(sigma3.getError(), sigma3.getVal())
                sigma3.setVal(0.95*sigma3.getVal())
                #sigma3.setConstant(ROOT.kTRUE)
            else: doFit = False
            if nTrials > 20: doFit = False
 
        '''
        # constraints for each BKG term
        # do the background fits and construct PDF
        w_bkg = ROOT.RooWorkspace("w_bin%d" % qTbin, "")
        pdfs = ROOT.RooArgList()
        norms = ROOT.RooArgList()
        constraints = ROOT.RooArgSet()
        for bkg in bkgs:
        
            hist_bkg = fIn.Get("%s_perp_bin%d" % (bkg, qTbin))
            norm = hist_bkg.Integral()
            norm_bkg += norm
            hist_bkg.Rebin(2)
            doFit_bkg_perp(recoil, w_bkg, bkg, hist_bkg, qTbin, qT, qTmin, qTmax)
            
            pdf = w_bkg.pdf("pdf_bkg_%s_bin%d" % (bkg, qTbin))
            pdfs.add(pdf)
            
            norm_val = norm/norm_data
            norm_ = ROOT.RooRealVar("norm_bkg_%s_bin%d" % (bkg, qTbin), "", norm_val) # , 0, 1
            norms.addOwned(norm_)
            
            fconstraint = ROOT.RooGaussian("constraint_norm_bkg_%s_bin%d" % (bkg, qTbin), "", norm_, ROOT.RooFit.RooConst(norm_val), ROOT.RooFit.RooConst(norm_val*1.02))
            constraints.addOwned(fconstraint)
            
        pdfs.add(pdf_sig)
        pdfs.Print()
        norms.Print()
        rdh_data = ROOT.RooDataHist("rdh_data_bin%d" % qTbin, "", ROOT.RooArgList(recoil), ROOT.RooFit.Import(hist_data))
        pdf_tot = ROOT.RooAddPdf("pdf_bin%d" % qTbin, '', pdfs, norms) # total PDF = sum of all backgrounds + sig
        pdf_tot.Print()
        pdf = ROOT.RooProdPdf("constraint_pdf_bin%d" % qTbin, "", ROOT.RooArgSet(pdf_tot, constraints)) # PDF with constraint terms
        pdf.Print()
        #fitRes_data = pdf.fitTo(rdh_data, ROOT.RooFit.SumW2Error(ROOT.kTRUE), ROOT.RooFit.Save(ROOT.kTRUE))
        fitRes_data = pdf.fitTo(rdh_data, ROOT.RooFit.SumW2Error(ROOT.kTRUE), ROOT.RooFit.Save(ROOT.kTRUE), ROOT.RooFit.Constrain(constraints))
        '''

        #############
        # SIG+BKG FIT
        #############
        cfg['xmin'], cfg['xmax'] = -100, 100
        #tag = "data_perp_new"
        
        #rdh_data = ROOT.RooDataHist("rdh_data_bin%d" % qTbin, "", ROOT.RooArgList(recoil), ROOT.RooFit.Import(hist_data))
        #fitRes_data = pdf.fitTo(rdh_data, ROOT.RooFit.SumW2Error(ROOT.kTRUE), ROOT.RooFit.Save(ROOT.kTRUE))
        
        
        mean2.setVal(mean1.getVal())
        mean3.setVal(mean1.getVal())
        
        plotter.cfg = cfg
        canvas, padT, padB = plotter.canvasRatio()
        dummyT, dummyB, dummyL = plotter.dummyRatio(line=0)
        canvas.cd()
        padT.Draw()
        padT.cd()
        dummyT.Draw("HIST")
        
        ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.ERROR)
        plt = recoil.frame()
        
        if fitValid: pdf_sig.plotOn(plt, ROOT.RooFit.VisualizeError(fitRes_data, 1), ROOT.RooFit.Name("errorband"), ROOT.RooFit.FillColor(ROOT.kGreen+1))
        else: pdf_sig.plotOn(plt, ROOT.RooFit.VisualizeError(fitRes_data, 1), ROOT.RooFit.Name("errorband"), ROOT.RooFit.FillColor(ROOT.kOrange))
        rdh_data.plotOn(plt)
        gauss1.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kBlue), ROOT.RooFit.Normalization(norm1.getVal(), ROOT.RooAbsReal.NumEvent), ROOT.RooFit.LineStyle(ROOT.kDashed))
        gauss2.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kGreen), ROOT.RooFit.Normalization(norm2.getVal(), ROOT.RooAbsReal.NumEvent), ROOT.RooFit.LineStyle(ROOT.kDashed))
        gauss3.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kCyan), ROOT.RooFit.Normalization(norm3.getVal(), ROOT.RooAbsReal.NumEvent), ROOT.RooFit.LineStyle(ROOT.kDashed))
        pdf_sig.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kRed), ROOT.RooFit.LineStyle(ROOT.kDashed))
        pdf.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kRed))
        histpull = plt.pullHist()
        chi2 = plt.chiSquare() 
      
        latex = ROOT.TLatex()
        latex.SetNDC()
        latex.SetTextSize(0.040)
        latex.SetTextColor(1)
        latex.SetTextFont(42)
        latex.DrawLatex(0.20, 0.85, "%s, Data" % label)
        latex.DrawLatex(0.20, 0.80, "q_{T} = [%.1f, %.1f] GeV" % (qTmin, qTmax))
        latex.DrawLatex(0.20, 0.75, "Mean = %.3f #pm %.3f" % (hist_data.GetMean(), hist_data.GetMeanError()))
        latex.DrawLatex(0.20, 0.70, "RMS = %.3f #pm %.3f" % (hist_data.GetRMS(), hist_data.GetRMSError()))
        latex.DrawLatex(0.20, 0.65, "#chi^{2}/ndof = %.3f" % chi2)
        latex.DrawLatex(0.20, 0.60, "#color[%d]{Cov. matrix quality = %d}" % (2 if fitRes_data.covQual() != 3 else 1, fitRes_data.covQual()))
        latex.DrawLatex(0.20, 0.55, "#color[%d]{MINUIT status = %d}" % (2 if fitRes_data.status() != 0 else 1, fitRes_data.status()))
        latex.DrawLatex(0.20, 0.50, "Fit trials = %d" % nTrials)
        

        latex.DrawLatex(0.60, 0.85, "#mu_{1} = %.3f #pm %.3f" % (mean1.getVal(), mean1.getError()))
        latex.DrawLatex(0.60, 0.80, "#mu_{2} = %.3f #pm %.3f" % (mean2.getVal(), mean2.getError()))
        latex.DrawLatex(0.60, 0.75, "#mu_{3} = %.3f #pm %.3f" % (mean3.getVal(), mean3.getError()))
        latex.DrawLatex(0.60, 0.70, "#sigma_{1} = %.3f #pm %.3f" % (sigma1.getVal(), sigma1.getError()))
        latex.DrawLatex(0.60, 0.65, "#sigma_{2} = %.3f #pm %.3f" % (sigma2.getVal(), sigma2.getError()))
        latex.DrawLatex(0.60, 0.60, "#sigma_{3} = %.3f #pm %.3f" % (sigma3.getVal(), sigma3.getError()))
        latex.DrawLatex(0.60, 0.55, "N_{1} = %.3f #pm %.3f" % (norm1.getVal(), norm1.getError()))
        latex.DrawLatex(0.60, 0.50, "N_{2} = %.3f #pm %.3f" % (norm2.getVal(), norm2.getError()))
        

        plt.Draw("SAME")
        plotter.auxRatio()
           
        ## BOTTOM PAD ##
        canvas.cd()
        padB.Draw()
        padB.cd()
        dummyB.Draw("HIST")
        plt = recoil.frame()
        plt.addPlotable(histpull, "P")
        plt.Draw("SAME")
        dummyL.Draw("SAME")

        canvas.Modify()
        canvas.Update()
        canvas.Draw()
        canvas.SaveAs("%s/%s/%03d_recoil.png" % (outDir, tag, qTbin))
        #canvas.SaveAs("%s/%03d_recoil_%s.pdf" % (outDir, qTbin, name))

        ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)   

        dummyB.Delete()
        dummyT.Delete()
        padT.Delete()
        padB.Delete()
        
        
        hOut.SetBinContent(qTbin, 1, 1, mean1.getVal())
        hOut.SetBinContent(qTbin, 1, 0, mean1.getError())
        hOut.SetBinContent(qTbin, 2, 1, sigma1.getVal())
        hOut.SetBinContent(qTbin, 2, 0, sigma1.getError())
        hOut.SetBinContent(qTbin, 3, 1, norm1.getVal())
        hOut.SetBinContent(qTbin, 3, 0, norm1.getError())
        
        hOut.SetBinContent(qTbin, 4, 1, mean2.getVal())
        hOut.SetBinContent(qTbin, 4, 0, mean2.getError())
        hOut.SetBinContent(qTbin, 5, 1, sigma2.getVal())
        hOut.SetBinContent(qTbin, 5, 0, sigma2.getError())
        hOut.SetBinContent(qTbin, 6, 1, norm2.getVal())
        hOut.SetBinContent(qTbin, 6, 0, norm2.getError())
        
        hOut.SetBinContent(qTbin, 7, 1, mean3.getVal())
        hOut.SetBinContent(qTbin, 7, 0, mean3.getError())
        hOut.SetBinContent(qTbin, 8, 1, sigma3.getVal())
        #hOut.SetBinContent(qTbin, 8, 0, sigma3.getError())
        hOut.SetBinContent(qTbin, 9, 1, norm3.getVal())
        hOut.SetBinContent(qTbin, 9, 0, 0)
        
        hOut.SetBinContent(qTbin, 10, 1, norm_data-norm_bkg)
        hOut.SetBinContent(qTbin, 0, 0, chi2)
        
        g_chi2.SetPoint(qTbin-1, qT, chi2)

        # diagonalize covariance matrix and store perturbations
        floatingParams = fitRes_data.floatParsFinal() # floating parameters
        variations = diagonalize(fitRes_data)
        # print(p.GetName(), p.isConstant(), type(p), isinstance(p, ROOT.RooFormulaVar))
        for iVar, var in enumerate(variations):
            for iPar, p in enumerate(params):
                if fitValid:
                    if getParamIdx(p.GetName(), floatingParams) > -1: val = var[getParamIdx(p.GetName(), floatingParams)]
                    else: val = p.getVal()
                    hOut.SetBinContent(qTbin, iPar+1, iVar+2, val)
                    print(iVar, iPar, p.GetName(), val)
                else: hOut.SetBinContent(qTbin, iPar+1, iVar+2, p.getVal())

    fOut = ROOT.TFile(fOut_.replace(".root", "_data_perp.root"), "RECREATE")
    hOut.Write()
    fOut.Close()    



    # plot the chi2
    cfg_chi2 = {

        'logy'              : False,
        'logx'              : False,
        
        'xmin'              : 0,
        'xmax'              : 100,
        'ymin'              : 0,
        'ymax'              : 3,
            
        'xtitle'            : "q_{T} (GeV)",
        'ytitle'            : "#chi^{2}",
            
        'topRight'          : "199 pb^{#minus1} (13 TeV)", 
        'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",

    } 
    

    plotter.cfg = cfg_chi2
    canvas = plotter.canvas()
    dummy = plotter.dummy()   
    canvas.cd()
    dummy.Draw("HIST")
    g_chi2.Draw("SAME LP")
    plotter.aux()
    
    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.035)
    latex.SetTextColor(1)
    latex.SetTextFont(42)
    latex.DrawLatex(0.20, 0.90, "%s, MC" % label)
    latex.DrawLatex(0.20, 0.85, "U_{#perp} ")
    
    canvas.Modify()
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs("%s/%s/chi2.png" % (outDir, tag))
    canvas.SaveAs("%s/%s/chi2.pdf" % (outDir, tag))
    canvas.Delete()




def doFit_data_perp_2gauss():

    tag = "data_perp"
    cfg['xtitle'] = "U_{#perp}   (GeV)"
    cfg['xmin'], cfg['xmax'] = -100, 100
    cfg['ymax'] = 1e3
    cfg['ymin'] = 1e-5
    
    functions.prepareDir("%s/%s" % (outDir, tag))
    
    hOut = ROOT.TH3D(tag, "", 150, 0, 150, 20, 0, 20, 20, 0, 20) # qt, nominal, error
    
    #mean1_bkg_, norm1_bkg_, sigma1_bkg_, sigma2_bkg_ = 0, 0.6, 10, 15 # DeepMET
    #mean1_, mean2_, mean3_, norm1_, norm2_, sigma1_, sigma2_, sigma3_ = 0, 0, 0,   0.6, 0.3,   8, 5, 10 # DeepMET
    
    recoil = ROOT.RooRealVar("recoil", "Recoil parallel (GeV)", 0, -300, 300)
    
    if flavor == "mumu" and met == "DeepMETReso":
        mean1_, mean2_, mean3_, norm1_, norm2_, sigma1_, sigma2_, sigma3_ = 0, 0, 0,   0.6, 0.3,   6, 4, 10
        mean1_, mean2_, mean3_, norm1_, norm2_, sigma1_, sigma2_, sigma3_ = 0, 0, 0,   0.6, 0.3,   6, 4, 10
        
    if flavor == "mumu" and met == "RawPFMET":
        mean1_, mean2_, norm1_, sigma1_, sigma2_ = 0, 0,    0.5,     11, 6
    
    sigma = ROOT.RooRealVar("sigma", "", 10)

    # signal model
    mean1 = ROOT.RooRealVar("mean1", "", mean1_, -300, 300)
    mean2 = ROOT.RooRealVar("mean2", "", mean2_, -300, 300)
    sigma1 = ROOT.RooRealVar("sigma1", "", sigma1_, 0.1, 20)
    sigma2 = ROOT.RooRealVar("sigma2", "", sigma2_, 0.1, 20)
    norm1 = ROOT.RooRealVar("norm1", "", norm1_, 0, 1)
    norm2 = ROOT.RooFormulaVar("norm2", "(1. - @0)", ROOT.RooArgList(norm1))
    gauss1 = ROOT.RooGaussian("gauss1", "gauss1", recoil, mean1, sigma1)
    gauss2 = ROOT.RooGaussian("gauss2", "gauss2", recoil, mean1, sigma2)
    pdf_sig = ROOT.RooAddPdf("pdf_sig", '', ROOT.RooArgList(gauss1, gauss2), ROOT.RooArgList(norm1))
    

        
    #sigma1.setConstant(ROOT.kTRUE)
        
    #mean1.setVal(0)
    #mean1.setConstant(ROOT.kTRUE)
    #mean2.setVal(0)
    #mean2.setConstant(ROOT.kTRUE)


    norm1.setConstant(ROOT.kTRUE)
    mean2.setConstant(ROOT.kTRUE)
        
    
    norm_bkg_var = ROOT.RooRealVar("norm_bkg_var", "", 0, 0, 1) # background fraction w.r.t. data (let it float, constrained by Gauss)
    
    g_chi2 = ROOT.TGraphErrors()
    g_chi2.SetLineColor(ROOT.kRed)
    g_chi2.SetMarkerStyle(8)
    g_chi2.SetMarkerSize(1)
    g_chi2.SetMarkerColor(ROOT.kRed)
    
    params = [mean1, sigma1, norm1, mean2, sigma2, norm2, norm_bkg_var]
    
    for qTbin in range(1, len(recoil_qTbins)):
    
        qT = 0.5*(recoil_qTbins[qTbin-1] + recoil_qTbins[qTbin]) if qTbin < len(recoil_qTbins)-1 else (recoil_qTbins[qTbin-1] + 20)
        qTmin, qTmax = recoil_qTbins[qTbin-1], recoil_qTbins[qTbin]
        
        hist_data = fIn.Get("SingleMuon_perp_bin%d" % qTbin)
        #hist_data.Rebin(2)
        norm_data = hist_data.Integral()
        hist_data.Scale(1./norm_data)
        
        #if qT <= 50: norm1.setVal(0.4)
        #else: norm1.setVal(0.6)
        
        mean1.setVal(0)
        mean1.setConstant(ROOT.kTRUE)
        mean2.setVal(0)
        mean2.setConstant(ROOT.kTRUE)

        
        # compute the total background norm (necessary for PDF normalization)
        norm_bkg = 0
        for bkg in bkgs: norm_bkg += fIn.Get("%s_perp_bin%d" % (bkg, qTbin)).Integral()
 

        # do the background fits and construct PDF 
        w_bkg = ROOT.RooWorkspace("w_bin%d" % qTbin, "")
        pdfs_bkg = ROOT.RooArgList()
        norms_bkg = ROOT.RooArgList()
        for bkg in bkgs:
        
            hist_bkg = fIn.Get("%s_perp_bin%d" % (bkg, qTbin))
            norm = hist_bkg.Integral()
            hist_bkg.Rebin(2)
            doFit_bkg_perp(recoil, w_bkg, bkg, hist_bkg, qTbin, qT, qTmin, qTmax)
            
            pdf = w_bkg.pdf("pdf_bkg_%s_bin%d" % (bkg, qTbin))
            pdfs_bkg.add(pdf)
            
            norm_ = ROOT.RooRealVar("norm_bkg_%s_bin%d" % (bkg, qTbin), "", norm/norm_bkg) # fix internal norm
            norms_bkg.addOwned(norm_)

        # construct bkg PDF  
        pdfs_bkg.Print()
        norms_bkg.Print()
        pdf_bkg = ROOT.RooAddPdf("pdf_bkg_bin%d" % qTbin, '', pdfs_bkg, norms_bkg)
        pdf_bkg.Print()
        
        # construct bkg+sig PDF with constraint term on BKG normalization
        val = norm_bkg/norm_data
        #norm_bkg_var = ROOT.RooRealVar("norm_bkg_var_bin%d" % qTbin, "", val, 0, 1) # let it float
        norm_bkg_var.setVal(val)
        if bkg_constr <= 0: norm_bkg_var.setConstant(ROOT.kTRUE)
        norm_bkg_constr = ROOT.RooGaussian("norm_bkg_constr_bin%d" % qTbin, "", norm_bkg_var, ROOT.RooFit.RooConst(val), ROOT.RooFit.RooConst(val*bkg_constr))
        
        pdf_sig_bkg = ROOT.RooAddPdf("pdf_sig_bkg_bin%d" % qTbin, '', ROOT.RooArgList(pdf_bkg, pdf_sig), ROOT.RooArgList(norm_bkg_var)) # a*BKG + SIG
        pdf = ROOT.RooProdPdf("pdf_bin%d" % qTbin, "", ROOT.RooArgSet(pdf_sig_bkg, norm_bkg_constr)) # a*BKG + SIG with constraint term on a
        
        pdf_sig_bkg.Print()
        pdf.Print()
        
        # do the fit
        rdh_data = ROOT.RooDataHist("rdh_data_bin%d" % qTbin, "", ROOT.RooArgList(recoil), ROOT.RooFit.Import(hist_data))
        fitRes_data = pdf.fitTo(rdh_data, ROOT.RooFit.SumW2Error(ROOT.kTRUE), ROOT.RooFit.Save(ROOT.kTRUE), ROOT.RooFit.Constrain(norm_bkg_constr))


 
        '''
        # constraints for each BKG term
        # do the background fits and construct PDF
        w_bkg = ROOT.RooWorkspace("w_bin%d" % qTbin, "")
        pdfs = ROOT.RooArgList()
        norms = ROOT.RooArgList()
        constraints = ROOT.RooArgSet()
        for bkg in bkgs:
        
            hist_bkg = fIn.Get("%s_perp_bin%d" % (bkg, qTbin))
            norm = hist_bkg.Integral()
            norm_bkg += norm
            hist_bkg.Rebin(2)
            doFit_bkg_perp(recoil, w_bkg, bkg, hist_bkg, qTbin, qT, qTmin, qTmax)
            
            pdf = w_bkg.pdf("pdf_bkg_%s_bin%d" % (bkg, qTbin))
            pdfs.add(pdf)
            
            norm_val = norm/norm_data
            norm_ = ROOT.RooRealVar("norm_bkg_%s_bin%d" % (bkg, qTbin), "", norm_val) # , 0, 1
            norms.addOwned(norm_)
            
            fconstraint = ROOT.RooGaussian("constraint_norm_bkg_%s_bin%d" % (bkg, qTbin), "", norm_, ROOT.RooFit.RooConst(norm_val), ROOT.RooFit.RooConst(norm_val*1.02))
            constraints.addOwned(fconstraint)
            
        pdfs.add(pdf_sig)
        pdfs.Print()
        norms.Print()
        rdh_data = ROOT.RooDataHist("rdh_data_bin%d" % qTbin, "", ROOT.RooArgList(recoil), ROOT.RooFit.Import(hist_data))
        pdf_tot = ROOT.RooAddPdf("pdf_bin%d" % qTbin, '', pdfs, norms) # total PDF = sum of all backgrounds + sig
        pdf_tot.Print()
        pdf = ROOT.RooProdPdf("constraint_pdf_bin%d" % qTbin, "", ROOT.RooArgSet(pdf_tot, constraints)) # PDF with constraint terms
        pdf.Print()
        #fitRes_data = pdf.fitTo(rdh_data, ROOT.RooFit.SumW2Error(ROOT.kTRUE), ROOT.RooFit.Save(ROOT.kTRUE))
        fitRes_data = pdf.fitTo(rdh_data, ROOT.RooFit.SumW2Error(ROOT.kTRUE), ROOT.RooFit.Save(ROOT.kTRUE), ROOT.RooFit.Constrain(constraints))
        '''

        #############
        # SIG+BKG FIT
        #############
        cfg['xmin'], cfg['xmax'] = -100, 100
        #tag = "data_perp_new"
        
        #rdh_data = ROOT.RooDataHist("rdh_data_bin%d" % qTbin, "", ROOT.RooArgList(recoil), ROOT.RooFit.Import(hist_data))
        #fitRes_data = pdf.fitTo(rdh_data, ROOT.RooFit.SumW2Error(ROOT.kTRUE), ROOT.RooFit.Save(ROOT.kTRUE))
        fitValid = (fitRes_data.covQual() == 3 and fitRes_data.status() == 0)
        
        mean2.setVal(mean1.getVal())
        
        plotter.cfg = cfg
        canvas, padT, padB = plotter.canvasRatio()
        dummyT, dummyB, dummyL = plotter.dummyRatio(line=0)
        canvas.cd()
        padT.Draw()
        padT.cd()
        dummyT.Draw("HIST")
        
        ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.ERROR)
        plt = recoil.frame()
        
        if fitValid: pdf_sig.plotOn(plt, ROOT.RooFit.VisualizeError(fitRes_data, 1), ROOT.RooFit.Name("errorband"), ROOT.RooFit.FillColor(ROOT.kGreen+1))
        else: pdf_sig.plotOn(plt, ROOT.RooFit.VisualizeError(fitRes_data, 1), ROOT.RooFit.Name("errorband"), ROOT.RooFit.FillColor(ROOT.kOrange))
        rdh_data.plotOn(plt)
        gauss1.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kBlue), ROOT.RooFit.Normalization(norm1.getVal(), ROOT.RooAbsReal.NumEvent), ROOT.RooFit.LineStyle(ROOT.kDashed))
        gauss2.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kGreen), ROOT.RooFit.Normalization(norm2.getVal(), ROOT.RooAbsReal.NumEvent), ROOT.RooFit.LineStyle(ROOT.kDashed))
        pdf_sig.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kRed), ROOT.RooFit.LineStyle(ROOT.kDashed))
        pdf.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kRed))
        histpull = plt.pullHist()
        chi2 = plt.chiSquare() 
      
        latex = ROOT.TLatex()
        latex.SetNDC()
        latex.SetTextSize(0.040)
        latex.SetTextColor(1)
        latex.SetTextFont(42)
        latex.DrawLatex(0.20, 0.85, "%s, Data" % label)
        latex.DrawLatex(0.20, 0.80, "q_{T} = [%.1f, %.1f] GeV" % (qTmin, qTmax))
        latex.DrawLatex(0.20, 0.75, "Mean = %.3f #pm %.3f" % (hist_data.GetMean(), hist_data.GetMeanError()))
        latex.DrawLatex(0.20, 0.70, "RMS = %.3f #pm %.3f" % (hist_data.GetRMS(), hist_data.GetRMSError()))
        latex.DrawLatex(0.20, 0.65, "#chi^{2}/ndof = %.3f" % chi2)
        latex.DrawLatex(0.20, 0.60, "#color[%d]{Cov. matrix quality = %d}" % (2 if fitRes_data.covQual() != 3 else 1, fitRes_data.covQual()))
        latex.DrawLatex(0.20, 0.55, "#color[%d]{MINUIT status = %d}" % (2 if fitRes_data.status() != 0 else 1, fitRes_data.status()))

        latex.DrawLatex(0.60, 0.85, "#mu_{1} = %.3f #pm %.3f" % (mean1.getVal(), mean1.getError()))
        latex.DrawLatex(0.60, 0.80, "#mu_{2} = %.3f #pm %.3f" % (mean2.getVal(), mean2.getError()))
        latex.DrawLatex(0.60, 0.70, "#sigma_{1} = %.3f #pm %.3f" % (sigma1.getVal(), sigma1.getError()))
        latex.DrawLatex(0.60, 0.65, "#sigma_{2} = %.3f #pm %.3f" % (sigma2.getVal(), sigma2.getError()))
        latex.DrawLatex(0.60, 0.55, "N_{1} = %.3f #pm %.3f" % (norm1.getVal(), norm1.getError()))
        

        plt.Draw("SAME")
        plotter.auxRatio()
           
        ## BOTTOM PAD ##
        canvas.cd()
        padB.Draw()
        padB.cd()
        dummyB.Draw("HIST")
        plt = recoil.frame()
        plt.addPlotable(histpull, "P")
        plt.Draw("SAME")
        dummyL.Draw("SAME")

        canvas.Modify()
        canvas.Update()
        canvas.Draw()
        canvas.SaveAs("%s/%s/%03d_recoil.png" % (outDir, tag, qTbin))
        #canvas.SaveAs("%s/%03d_recoil_%s.pdf" % (outDir, qTbin, name))

        ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)   

        dummyB.Delete()
        dummyT.Delete()
        padT.Delete()
        padB.Delete()
        
        
        hOut.SetBinContent(qTbin, 1, 1, mean1.getVal())
        hOut.SetBinContent(qTbin, 1, 0, mean1.getError())
        hOut.SetBinContent(qTbin, 2, 1, sigma1.getVal())
        hOut.SetBinContent(qTbin, 2, 0, sigma1.getError())
        hOut.SetBinContent(qTbin, 3, 1, norm1.getVal())
        hOut.SetBinContent(qTbin, 3, 0, norm1.getError())
        
        hOut.SetBinContent(qTbin, 4, 1, mean2.getVal())
        hOut.SetBinContent(qTbin, 4, 0, mean2.getError())
        hOut.SetBinContent(qTbin, 5, 1, sigma2.getVal())
        hOut.SetBinContent(qTbin, 5, 0, sigma2.getError())
        hOut.SetBinContent(qTbin, 6, 1, norm2.getVal())
        hOut.SetBinContent(qTbin, 6, 0, 0)
        

        
        hOut.SetBinContent(qTbin, 10, 1, norm_data-norm_bkg)
        hOut.SetBinContent(qTbin, 0, 0, chi2)
        
        g_chi2.SetPoint(qTbin-1, qT, chi2)

        # diagonalize covariance matrix and store perturbations
        floatingParams = fitRes_data.floatParsFinal() # floating parameters
        variations = diagonalize(fitRes_data)
        # print(p.GetName(), p.isConstant(), type(p), isinstance(p, ROOT.RooFormulaVar))
        for iVar, var in enumerate(variations):
            for iPar, p in enumerate(params):
                if fitValid:
                    if getParamIdx(p.GetName(), floatingParams) > -1: val = var[getParamIdx(p.GetName(), floatingParams)]
                    else: val = p.getVal()
                    hOut.SetBinContent(qTbin, iPar+1, iVar+2, val)
                    print(iVar, iPar, p.GetName(), val)
                else: hOut.SetBinContent(qTbin, iPar+1, iVar+2, p.getVal())
 
    fOut = ROOT.TFile(fOut_.replace(".root", "_data_perp.root"), "RECREATE")
    hOut.Write()
    fOut.Close()    



    # plot the chi2
    cfg_chi2 = {

        'logy'              : False,
        'logx'              : False,
        
        'xmin'              : 0,
        'xmax'              : 100,
        'ymin'              : 0,
        'ymax'              : 3,
            
        'xtitle'            : "q_{T} (GeV)",
        'ytitle'            : "#chi^{2}",
            
        'topRight'          : "199 pb^{#minus1} (13 TeV)", 
        'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",

    } 
    

    plotter.cfg = cfg_chi2
    canvas = plotter.canvas()
    dummy = plotter.dummy()   
    canvas.cd()
    dummy.Draw("HIST")
    g_chi2.Draw("SAME LP")
    plotter.aux()
    
    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.035)
    latex.SetTextColor(1)
    latex.SetTextFont(42)
    latex.DrawLatex(0.20, 0.90, "%s, MC" % label)
    latex.DrawLatex(0.20, 0.85, "U_{#perp} ")
    
    canvas.Modify()
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs("%s/%s/chi2.png" % (outDir, tag))
    canvas.SaveAs("%s/%s/chi2.pdf" % (outDir, tag))
    canvas.Delete()


def doFit_data_para():

    tag = "data_para"
    cfg['xtitle'] = "U_{#parallel} + q_{T} (GeV)"
    cfg['xmin'], cfg['xmax'] = -100, 100
    cfg['ymax'] = 1e3
    cfg['ymin'] = 1e-5
    
    functions.prepareDir("%s/%s" % (outDir, tag))
    
    hOut = ROOT.TH3D(tag, "", 150, 0, 150, 20, 0, 20, 20, 0, 20) # qt, nominal, error
    
    if flavor == "mumu" and met == "DeepMETReso":
        mean1__, mean2__, mean3__, norm1__, norm2__, sigma1__, sigma2__, sigma3__ = 0, 0, 0,   0.6, 0.3,   8, 5, 10 # DeepMET
    
    if flavor == "mumu" and met == "RawPFMET":
        mean1_, mean2_, mean3_, norm1_, norm2_, sigma1_, sigma2_, sigma3_ = 0.0, 0.0, 0.0,    0.55, 0.35,     8.5, 5, 15
    

    recoil = ROOT.RooRealVar("recoil", "Recoil parallel (GeV)", 0, -300, 300)
    
    
    # signal model
    mean1 = ROOT.RooRealVar("mean1", "", mean1_, -300, 300)
    mean2 = ROOT.RooRealVar("mean2", "", mean2_, -300, 300)
    mean3 = ROOT.RooRealVar("mean3", "", mean3_, -300, 300)
    sigma1 = ROOT.RooRealVar("sigma1", "", sigma1_, 0.1, 100)
    sigma2 = ROOT.RooRealVar("sigma2", "", sigma2_, 0.1, 100)
    sigma3 = ROOT.RooRealVar("sigma3", "", sigma3_, 0.1, 100)
    norm1 = ROOT.RooRealVar("norm1", "", norm1_, 0, 1)
    norm2 = ROOT.RooRealVar("norm2", "", norm2_, 0, 1)
    norm3 = ROOT.RooFormulaVar("norm3", "(1. - @0 - @1)", ROOT.RooArgList(norm1, norm2))
    gauss1 = ROOT.RooGaussian("gauss1", "gauss1", recoil, mean1, sigma1)
    gauss2 = ROOT.RooGaussian("gauss2", "gauss2", recoil, mean1, sigma2)
    gauss3 = ROOT.RooGaussian("gauss3", "gauss3", recoil, mean1, sigma3)
    pdf_sig = ROOT.RooAddPdf("pdf_sig", '', ROOT.RooArgList(gauss1, gauss2, gauss3), ROOT.RooArgList(norm1, norm2))

    norm1.setConstant(ROOT.kTRUE)
    norm2.setConstant(ROOT.kTRUE)

    norm_bkg_var = ROOT.RooRealVar("norm_bkg_var", "", 0, 0, 1) # background fraction w.r.t. data (let it float, constrained by Gauss)
    
    g_chi2 = ROOT.TGraphErrors()
    g_chi2.SetLineColor(ROOT.kRed)
    g_chi2.SetMarkerStyle(8)
    g_chi2.SetMarkerSize(1)
    g_chi2.SetMarkerColor(ROOT.kRed)
    
    params = [mean1, sigma1, norm1, mean2, sigma2, norm2, mean3, sigma3, norm3, norm_bkg_var]
    
    for qTbin in range(1, len(recoil_qTbins)):
    
        qT = 0.5*(recoil_qTbins[qTbin-1] + recoil_qTbins[qTbin]) if qTbin < len(recoil_qTbins)-1 else (recoil_qTbins[qTbin-1] + 20)
        qTmin, qTmax = recoil_qTbins[qTbin-1], recoil_qTbins[qTbin]
        
        hist_data = fIn.Get("SingleMuon_para_bin%d" % qTbin)
        norm_data = hist_data.Integral()
        hist_data.Scale(1./norm_data)
        
        
        # compute the total background norm (necessary for PDF normalization)
        norm_bkg = 0
        for bkg in bkgs: norm_bkg += fIn.Get("%s_para_bin%d" % (bkg, qTbin)).Integral()
 

        # do the background fits and construct PDF 
        w_bkg = ROOT.RooWorkspace("w_bin%d" % qTbin, "")
        pdfs_bkg = ROOT.RooArgList()
        norms_bkg = ROOT.RooArgList()
        for bkg in bkgs:
        
            hist_bkg = fIn.Get("%s_perp_bin%d" % (bkg, qTbin))
            norm = hist_bkg.Integral()
            hist_bkg.Rebin(2)
            doFit_bkg_perp(recoil, w_bkg, bkg, hist_bkg, qTbin, qT, qTmin, qTmax)
            
            pdf = w_bkg.pdf("pdf_bkg_%s_bin%d" % (bkg, qTbin))
            pdfs_bkg.add(pdf)
            
            norm_ = ROOT.RooRealVar("norm_bkg_%s_bin%d" % (bkg, qTbin), "", norm/norm_bkg) # fix internal norm
            norms_bkg.addOwned(norm_)

        # construct bkg PDF  
        pdfs_bkg.Print()
        norms_bkg.Print()
        pdf_bkg = ROOT.RooAddPdf("pdf_bkg_bin%d" % qTbin, '', pdfs_bkg, norms_bkg)
        pdf_bkg.Print()
        
        # construct bkg+sig PDF with constraint term on BKG normalization
        val = norm_bkg/norm_data
        #norm_bkg_var = ROOT.RooRealVar("norm_bkg_var_bin%d" % qTbin, "", val, 0, 1) # let it float
        norm_bkg_var.setVal(val)
        if bkg_constr <= 0: norm_bkg_var.setConstant(ROOT.kTRUE)
        norm_bkg_constr = ROOT.RooGaussian("norm_bkg_constr_bin%d" % qTbin, "", norm_bkg_var, ROOT.RooFit.RooConst(val), ROOT.RooFit.RooConst(val*bkg_constr))
        
        pdf_sig_bkg = ROOT.RooAddPdf("pdf_sig_bkg_bin%d" % qTbin, '', ROOT.RooArgList(pdf_bkg, pdf_sig), ROOT.RooArgList(norm_bkg_var)) # a*BKG + SIG
        pdf = ROOT.RooProdPdf("pdf_bin%d" % qTbin, "", ROOT.RooArgSet(pdf_sig_bkg, norm_bkg_constr)) # a*BKG + SIG with constraint term on a
        
        pdf_sig_bkg.Print()
        pdf.Print()
 
 
        # do the fit
        rdh_data = ROOT.RooDataHist("rdh_data_bin%d" % qTbin, "", ROOT.RooArgList(recoil), ROOT.RooFit.Import(hist_data))
        doFit = True
        nTrials = 0
        while doFit:
            
            fitRes_data = pdf.fitTo(rdh_data, ROOT.RooFit.SumW2Error(ROOT.kTRUE), ROOT.RooFit.Save(ROOT.kTRUE), ROOT.RooFit.Constrain(norm_bkg_constr))
            fitValid = (fitRes_data.covQual() == 3 and fitRes_data.status() == 0)
        
            nTrials += 1
            # do check on sigma3
            if sigma3.getError() > 0.9*sigma3.getVal() or not fitValid:
                
                print(sigma3.getError(), sigma3.getVal())
                sigma3.setVal(0.95*sigma3.getVal())
                #sigma3.setConstant(ROOT.kTRUE)
            else: doFit = False
            if nTrials > 20: doFit = False
        
        

       
        mean2.setVal(mean1.getVal())
        mean3.setVal(mean1.getVal())

        #############
        # SIG+BKG FIT
        #############
        cfg['xmin'], cfg['xmax'] = -100, 100
        #tag = "data_para"
        
        fitValid = (fitRes_data.covQual() == 3 and fitRes_data.status() == 0)
     
        plotter.cfg = cfg
        canvas, padT, padB = plotter.canvasRatio()
        dummyT, dummyB, dummyL = plotter.dummyRatio(line=0)
        canvas.cd()
        padT.Draw()
        padT.cd()
        dummyT.Draw("HIST")
        
        ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.ERROR)
        plt = recoil.frame()
        
        if fitValid: pdf_sig.plotOn(plt, ROOT.RooFit.VisualizeError(fitRes_data, 1), ROOT.RooFit.Name("errorband"), ROOT.RooFit.FillColor(ROOT.kGreen+1))
        else: pdf_sig.plotOn(plt, ROOT.RooFit.VisualizeError(fitRes_data, 1), ROOT.RooFit.Name("errorband"), ROOT.RooFit.FillColor(ROOT.kOrange))
        rdh_data.plotOn(plt)
        gauss1.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kBlue), ROOT.RooFit.Normalization(norm1.getVal(), ROOT.RooAbsReal.NumEvent), ROOT.RooFit.LineStyle(ROOT.kDashed))
        gauss2.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kGreen), ROOT.RooFit.Normalization(norm2.getVal(), ROOT.RooAbsReal.NumEvent), ROOT.RooFit.LineStyle(ROOT.kDashed))
        gauss3.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kCyan), ROOT.RooFit.Normalization(norm3.getVal(), ROOT.RooAbsReal.NumEvent), ROOT.RooFit.LineStyle(ROOT.kDashed))
        pdf_sig.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kRed), ROOT.RooFit.LineStyle(ROOT.kDashed))
        pdf.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kRed))
        
        histpull = plt.pullHist()
        chi2 = plt.chiSquare() 
      
        latex = ROOT.TLatex()
        latex.SetNDC()
        latex.SetTextSize(0.040)
        latex.SetTextColor(1)
        latex.SetTextFont(42)
        latex.DrawLatex(0.20, 0.85, "%s, Data" % label)
        latex.DrawLatex(0.20, 0.80, "q_{T} = [%.1f, %.1f] GeV" % (qTmin, qTmax))
        latex.DrawLatex(0.20, 0.75, "Mean = %.3f #pm %.3f" % (hist_data.GetMean(), hist_data.GetMeanError()))
        latex.DrawLatex(0.20, 0.70, "RMS = %.3f #pm %.3f" % (hist_data.GetRMS(), hist_data.GetRMSError()))
        latex.DrawLatex(0.20, 0.65, "#chi^{2}/ndof = %.3f" % chi2)
        latex.DrawLatex(0.20, 0.60, "#color[%d]{Cov. matrix quality = %d}" % (2 if fitRes_data.covQual() != 3 else 1, fitRes_data.covQual()))
        latex.DrawLatex(0.20, 0.55, "#color[%d]{MINUIT status = %d}" % (2 if fitRes_data.status() != 0 else 1, fitRes_data.status()))
        latex.DrawLatex(0.20, 0.50, "Fit trials = %d" % nTrials)

        latex.DrawLatex(0.60, 0.85, "#mu_{1} = %.3f #pm %.3f" % (mean1.getVal(), mean1.getError()))
        latex.DrawLatex(0.60, 0.80, "#mu_{2} = %.3f #pm %.3f" % (mean2.getVal(), mean2.getError()))
        latex.DrawLatex(0.60, 0.75, "#mu_{3} = %.3f #pm %.3f" % (mean3.getVal(), mean3.getError()))
        latex.DrawLatex(0.60, 0.70, "#sigma_{1} = %.3f #pm %.3f" % (sigma1.getVal(), sigma1.getError()))
        latex.DrawLatex(0.60, 0.65, "#sigma_{2} = %.3f #pm %.3f" % (sigma2.getVal(), sigma2.getError()))
        latex.DrawLatex(0.60, 0.60, "#sigma_{3} = %.3f #pm %.3f" % (sigma3.getVal(), sigma3.getError()))
        latex.DrawLatex(0.60, 0.55, "N_{1} = %.3f #pm %.3f" % (norm1.getVal(), norm1.getError()))
        latex.DrawLatex(0.60, 0.50, "N_{2} = %.3f #pm %.3f" % (norm2.getVal(), norm2.getError()))
        

        plt.Draw("SAME")
        plotter.auxRatio()
           
        ## BOTTOM PAD ##
        canvas.cd()
        padB.Draw()
        padB.cd()
        dummyB.Draw("HIST")
        plt = recoil.frame()
        plt.addPlotable(histpull, "P")
        plt.Draw("SAME")
        dummyL.Draw("SAME")

        canvas.Modify()
        canvas.Update()
        canvas.Draw()
        canvas.SaveAs("%s/%s/%03d_recoil.png" % (outDir, tag, qTbin))
        #canvas.SaveAs("%s/%03d_recoil_%s.pdf" % (outDir, qTbin, name))

        ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)   

        dummyB.Delete()
        dummyT.Delete()
        padT.Delete()
        padB.Delete()
        
        


        hOut.SetBinContent(qTbin, 1, 1, mean1.getVal())
        hOut.SetBinContent(qTbin, 1, 0, mean1.getError())
        hOut.SetBinContent(qTbin, 2, 1, sigma1.getVal())
        hOut.SetBinContent(qTbin, 2, 0, sigma1.getError())
        hOut.SetBinContent(qTbin, 3, 1, norm1.getVal())
        hOut.SetBinContent(qTbin, 3, 0, norm1.getError())
        
        hOut.SetBinContent(qTbin, 4, 1, mean2.getVal())
        hOut.SetBinContent(qTbin, 4, 0, mean2.getError())
        hOut.SetBinContent(qTbin, 5, 1, sigma2.getVal())
        hOut.SetBinContent(qTbin, 5, 0, sigma2.getError())
        hOut.SetBinContent(qTbin, 6, 1, norm2.getVal())
        hOut.SetBinContent(qTbin, 6, 0, norm2.getError())
        
        hOut.SetBinContent(qTbin, 7, 1, mean3.getVal())
        hOut.SetBinContent(qTbin, 7, 0, mean3.getError())
        hOut.SetBinContent(qTbin, 8, 1, sigma3.getVal())
        hOut.SetBinContent(qTbin, 8, 0, sigma3.getError())
        hOut.SetBinContent(qTbin, 9, 1, norm3.getVal())
        hOut.SetBinContent(qTbin, 9, 0, 0)
        
        hOut.SetBinContent(qTbin, 10, 1, norm_data-norm_bkg)
        hOut.SetBinContent(qTbin, 0, 0, chi2)
        
        g_chi2.SetPoint(qTbin-1, qT, chi2)

        # diagonalize covariance matrix and store perturbations
        floatingParams = fitRes_data.floatParsFinal() # floating parameters
        variations = diagonalize(fitRes_data)
        # print(p.GetName(), p.isConstant(), type(p), isinstance(p, ROOT.RooFormulaVar))
        for iVar, var in enumerate(variations):
            for iPar, p in enumerate(params):
                if fitValid:
                    if getParamIdx(p.GetName(), floatingParams) > -1: val = var[getParamIdx(p.GetName(), floatingParams)]
                    else: val = p.getVal()
                    hOut.SetBinContent(qTbin, iPar+1, iVar+2, val)
                    print(iVar, iPar, p.GetName(), val)
                else: hOut.SetBinContent(qTbin, iPar+1, iVar+2, p.getVal())
        
        
    fOut = ROOT.TFile(fOut_.replace(".root", "_data_para.root"), "RECREATE")
    hOut.Write()
    fOut.Close()    
 
 
    # plot the chi2
    cfg_chi2 = {

        'logy'              : False,
        'logx'              : False,
        
        'xmin'              : 0,
        'xmax'              : 100,
        'ymin'              : 0,
        'ymax'              : 3,
            
        'xtitle'            : "q_{T} (GeV)",
        'ytitle'            : "#chi^{2}",
            
        'topRight'          : "199 pb^{#minus1} (13 TeV)", 
        'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",

    } 
    

    plotter.cfg = cfg_chi2
    canvas = plotter.canvas()
    dummy = plotter.dummy()   
    canvas.cd()
    dummy.Draw("HIST")
    g_chi2.Draw("SAME LP")
    plotter.aux()
    
    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.035)
    latex.SetTextColor(1)
    latex.SetTextFont(42)
    latex.DrawLatex(0.20, 0.90, "%s, MC" % label)
    latex.DrawLatex(0.20, 0.85, "U_{#parallel} + q_{T} ")
    
    canvas.Modify()
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs("%s/%s/chi2.png" % (outDir, tag))
    canvas.SaveAs("%s/%s/chi2.pdf" % (outDir, tag))
    canvas.Delete()



def doFit_data_para_2gauss():

    tag = "data_para"
    cfg['xtitle'] = "U_{#parallel} + q_{T} (GeV)"
    cfg['xmin'], cfg['xmax'] = -100, 100
    cfg['ymax'] = 1e3
    cfg['ymin'] = 1e-5
    
    functions.prepareDir("%s/%s" % (outDir, tag))
    
    hOut = ROOT.TH3D(tag, "", 150, 0, 150, 20, 0, 20, 20, 0, 20) # qt, nominal, error
    
    if flavor == "mumu" and met == "DeepMETReso":
        mean1__, mean2__, mean3__, norm1__, norm2__, sigma1__, sigma2__, sigma3__ = 0, 0, 0,   0.6, 0.3,   8, 5, 10 # DeepMET
    
    if flavor == "mumu" and met == "RawPFMET":
        mean1_, mean2_, norm1_, sigma1_, sigma2_ = 0.0, 0.0,    0.45,     10, 5
    

    recoil = ROOT.RooRealVar("recoil", "Recoil parallel (GeV)", 0, -300, 300)
    
    
    # signal model
    mean1 = ROOT.RooRealVar("mean1", "", mean1_, -300, 300)
    mean2 = ROOT.RooRealVar("mean2", "", mean2_, -300, 300)
    sigma1 = ROOT.RooRealVar("sigma1", "", sigma1_, 0.1, 100)
    sigma2 = ROOT.RooRealVar("sigma2", "", sigma2_, 0.1, 100)
    norm1 = ROOT.RooRealVar("norm1", "", norm1_, 0, 1)
    norm2 = ROOT.RooFormulaVar("norm2", "(1. - @0)", ROOT.RooArgList(norm1))
    gauss1 = ROOT.RooGaussian("gauss1", "gauss1", recoil, mean1, sigma1)
    gauss2 = ROOT.RooGaussian("gauss2", "gauss2", recoil, mean1, sigma2)
    pdf_sig = ROOT.RooAddPdf("pdf_sig", '', ROOT.RooArgList(gauss1, gauss2), ROOT.RooArgList(norm1))

    norm1.setConstant(ROOT.kTRUE)

    norm_bkg_var = ROOT.RooRealVar("norm_bkg_var", "", 0, 0, 1) # background fraction w.r.t. data (let it float, constrained by Gauss)
    
    g_chi2 = ROOT.TGraphErrors()
    g_chi2.SetLineColor(ROOT.kRed)
    g_chi2.SetMarkerStyle(8)
    g_chi2.SetMarkerSize(1)
    g_chi2.SetMarkerColor(ROOT.kRed)
    
    params = [mean1, sigma1, norm1, mean2, sigma2, norm2, norm_bkg_var]
    
    for qTbin in range(1, len(recoil_qTbins)):
    
        qT = 0.5*(recoil_qTbins[qTbin-1] + recoil_qTbins[qTbin]) if qTbin < len(recoil_qTbins)-1 else (recoil_qTbins[qTbin-1] + 20)
        qTmin, qTmax = recoil_qTbins[qTbin-1], recoil_qTbins[qTbin]
        
        hist_data = fIn.Get("SingleMuon_para_bin%d" % qTbin)
        norm_data = hist_data.Integral()
        hist_data.Scale(1./norm_data)
        
        
        # compute the total background norm (necessary for PDF normalization)
        norm_bkg = 0
        for bkg in bkgs: norm_bkg += fIn.Get("%s_para_bin%d" % (bkg, qTbin)).Integral()
 

        # do the background fits and construct PDF 
        w_bkg = ROOT.RooWorkspace("w_bin%d" % qTbin, "")
        pdfs_bkg = ROOT.RooArgList()
        norms_bkg = ROOT.RooArgList()
        for bkg in bkgs:
        
            hist_bkg = fIn.Get("%s_perp_bin%d" % (bkg, qTbin))
            norm = hist_bkg.Integral()
            hist_bkg.Rebin(2)
            doFit_bkg_perp(recoil, w_bkg, bkg, hist_bkg, qTbin, qT, qTmin, qTmax)
            
            pdf = w_bkg.pdf("pdf_bkg_%s_bin%d" % (bkg, qTbin))
            pdfs_bkg.add(pdf)
            
            norm_ = ROOT.RooRealVar("norm_bkg_%s_bin%d" % (bkg, qTbin), "", norm/norm_bkg) # fix internal norm
            norms_bkg.addOwned(norm_)

        # construct bkg PDF  
        pdfs_bkg.Print()
        norms_bkg.Print()
        pdf_bkg = ROOT.RooAddPdf("pdf_bkg_bin%d" % qTbin, '', pdfs_bkg, norms_bkg)
        pdf_bkg.Print()
        
        # construct bkg+sig PDF with constraint term on BKG normalization
        val = norm_bkg/norm_data
        #norm_bkg_var = ROOT.RooRealVar("norm_bkg_var_bin%d" % qTbin, "", val, 0, 1) # let it float
        norm_bkg_var.setVal(val)
        if bkg_constr <= 0: norm_bkg_var.setConstant(ROOT.kTRUE)
        norm_bkg_constr = ROOT.RooGaussian("norm_bkg_constr_bin%d" % qTbin, "", norm_bkg_var, ROOT.RooFit.RooConst(val), ROOT.RooFit.RooConst(val*bkg_constr))
        
        pdf_sig_bkg = ROOT.RooAddPdf("pdf_sig_bkg_bin%d" % qTbin, '', ROOT.RooArgList(pdf_bkg, pdf_sig), ROOT.RooArgList(norm_bkg_var)) # a*BKG + SIG
        pdf = ROOT.RooProdPdf("pdf_bin%d" % qTbin, "", ROOT.RooArgSet(pdf_sig_bkg, norm_bkg_constr)) # a*BKG + SIG with constraint term on a
        
        pdf_sig_bkg.Print()
        pdf.Print()
        
        # do the fit
        rdh_data = ROOT.RooDataHist("rdh_data_bin%d" % qTbin, "", ROOT.RooArgList(recoil), ROOT.RooFit.Import(hist_data))
        fitRes_data = pdf.fitTo(rdh_data, ROOT.RooFit.SumW2Error(ROOT.kTRUE), ROOT.RooFit.Save(ROOT.kTRUE), ROOT.RooFit.Constrain(norm_bkg_constr))

       
        mean2.setVal(mean1.getVal())

        #############
        # SIG+BKG FIT
        #############
        cfg['xmin'], cfg['xmax'] = -100, 100
        #tag = "data_para"
        
        fitValid = (fitRes_data.covQual() == 3 and fitRes_data.status() == 0)
     
        plotter.cfg = cfg
        canvas, padT, padB = plotter.canvasRatio()
        dummyT, dummyB, dummyL = plotter.dummyRatio(line=0)
        canvas.cd()
        padT.Draw()
        padT.cd()
        dummyT.Draw("HIST")
        
        ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.ERROR)
        plt = recoil.frame()
        
        if fitValid: pdf_sig.plotOn(plt, ROOT.RooFit.VisualizeError(fitRes_data, 1), ROOT.RooFit.Name("errorband"), ROOT.RooFit.FillColor(ROOT.kGreen+1))
        else: pdf_sig.plotOn(plt, ROOT.RooFit.VisualizeError(fitRes_data, 1), ROOT.RooFit.Name("errorband"), ROOT.RooFit.FillColor(ROOT.kOrange))
        rdh_data.plotOn(plt)
        gauss1.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kBlue), ROOT.RooFit.Normalization(norm1.getVal(), ROOT.RooAbsReal.NumEvent), ROOT.RooFit.LineStyle(ROOT.kDashed))
        gauss2.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kGreen), ROOT.RooFit.Normalization(norm2.getVal(), ROOT.RooAbsReal.NumEvent), ROOT.RooFit.LineStyle(ROOT.kDashed))
        pdf_sig.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kRed), ROOT.RooFit.LineStyle(ROOT.kDashed))
        pdf.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kRed))
        
        histpull = plt.pullHist()
        chi2 = plt.chiSquare() 
      
        latex = ROOT.TLatex()
        latex.SetNDC()
        latex.SetTextSize(0.040)
        latex.SetTextColor(1)
        latex.SetTextFont(42)
        latex.DrawLatex(0.20, 0.85, "%s, Data" % label)
        latex.DrawLatex(0.20, 0.80, "q_{T} = [%.1f, %.1f] GeV" % (qTmin, qTmax))
        latex.DrawLatex(0.20, 0.75, "Mean = %.3f #pm %.3f" % (hist_data.GetMean(), hist_data.GetMeanError()))
        latex.DrawLatex(0.20, 0.70, "RMS = %.3f #pm %.3f" % (hist_data.GetRMS(), hist_data.GetRMSError()))
        latex.DrawLatex(0.20, 0.65, "#chi^{2}/ndof = %.3f" % chi2)
        latex.DrawLatex(0.20, 0.60, "#color[%d]{Cov. matrix quality = %d}" % (2 if fitRes_data.covQual() != 3 else 1, fitRes_data.covQual()))
        latex.DrawLatex(0.20, 0.55, "#color[%d]{MINUIT status = %d}" % (2 if fitRes_data.status() != 0 else 1, fitRes_data.status()))

        latex.DrawLatex(0.60, 0.85, "#mu_{1} = %.3f #pm %.3f" % (mean1.getVal(), mean1.getError()))
        latex.DrawLatex(0.60, 0.80, "#mu_{2} = %.3f #pm %.3f" % (mean2.getVal(), mean2.getError()))
        latex.DrawLatex(0.60, 0.70, "#sigma_{1} = %.3f #pm %.3f" % (sigma1.getVal(), sigma1.getError()))
        latex.DrawLatex(0.60, 0.65, "#sigma_{2} = %.3f #pm %.3f" % (sigma2.getVal(), sigma2.getError()))
        latex.DrawLatex(0.60, 0.55, "N_{1} = %.3f #pm %.3f" % (norm1.getVal(), norm1.getError()))
        

        plt.Draw("SAME")
        plotter.auxRatio()
           
        ## BOTTOM PAD ##
        canvas.cd()
        padB.Draw()
        padB.cd()
        dummyB.Draw("HIST")
        plt = recoil.frame()
        plt.addPlotable(histpull, "P")
        plt.Draw("SAME")
        dummyL.Draw("SAME")

        canvas.Modify()
        canvas.Update()
        canvas.Draw()
        canvas.SaveAs("%s/%s/%03d_recoil.png" % (outDir, tag, qTbin))
        #canvas.SaveAs("%s/%03d_recoil_%s.pdf" % (outDir, qTbin, name))

        ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)   

        dummyB.Delete()
        dummyT.Delete()
        padT.Delete()
        padB.Delete()
        
        


        hOut.SetBinContent(qTbin, 1, 1, mean1.getVal())
        hOut.SetBinContent(qTbin, 1, 0, mean1.getError())
        hOut.SetBinContent(qTbin, 2, 1, sigma1.getVal())
        hOut.SetBinContent(qTbin, 2, 0, sigma1.getError())
        hOut.SetBinContent(qTbin, 3, 1, norm1.getVal())
        hOut.SetBinContent(qTbin, 3, 0, norm1.getError())
        
        hOut.SetBinContent(qTbin, 4, 1, mean2.getVal())
        hOut.SetBinContent(qTbin, 4, 0, mean2.getError())
        hOut.SetBinContent(qTbin, 5, 1, sigma2.getVal())
        hOut.SetBinContent(qTbin, 5, 0, sigma2.getError())
        hOut.SetBinContent(qTbin, 6, 1, norm2.getVal())
        hOut.SetBinContent(qTbin, 6, 0, 0)
        
        
        hOut.SetBinContent(qTbin, 10, 1, norm_data-norm_bkg)
        hOut.SetBinContent(qTbin, 0, 0, chi2)
        
        g_chi2.SetPoint(qTbin-1, qT, chi2)

        # diagonalize covariance matrix and store perturbations
        floatingParams = fitRes_data.floatParsFinal() # floating parameters
        variations = diagonalize(fitRes_data)
        # print(p.GetName(), p.isConstant(), type(p), isinstance(p, ROOT.RooFormulaVar))
        for iVar, var in enumerate(variations):
            for iPar, p in enumerate(params):
                if fitValid:
                    if getParamIdx(p.GetName(), floatingParams) > -1: val = var[getParamIdx(p.GetName(), floatingParams)]
                    else: val = p.getVal()
                    hOut.SetBinContent(qTbin, iPar+1, iVar+2, val)
                    print(iVar, iPar, p.GetName(), val)
                else: hOut.SetBinContent(qTbin, iPar+1, iVar+2, p.getVal())
        
        
    fOut = ROOT.TFile(fOut_.replace(".root", "_data_para.root"), "RECREATE")
    hOut.Write()
    fOut.Close()    
 
 
    # plot the chi2
    cfg_chi2 = {

        'logy'              : False,
        'logx'              : False,
        
        'xmin'              : 0,
        'xmax'              : 100,
        'ymin'              : 0,
        'ymax'              : 3,
            
        'xtitle'            : "q_{T} (GeV)",
        'ytitle'            : "#chi^{2}",
            
        'topRight'          : "199 pb^{#minus1} (13 TeV)", 
        'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",

    } 
    

    plotter.cfg = cfg_chi2
    canvas = plotter.canvas()
    dummy = plotter.dummy()   
    canvas.cd()
    dummy.Draw("HIST")
    g_chi2.Draw("SAME LP")
    plotter.aux()
    
    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.035)
    latex.SetTextColor(1)
    latex.SetTextFont(42)
    latex.DrawLatex(0.20, 0.90, "%s, MC" % label)
    latex.DrawLatex(0.20, 0.85, "U_{#parallel} + q_{T} ")
    
    canvas.Modify()
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs("%s/%s/chi2.png" % (outDir, tag))
    canvas.SaveAs("%s/%s/chi2.pdf" % (outDir, tag))
    canvas.Delete()




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

def prepareFile(fOut_):
    
    fOut = ROOT.TFile(fOut_, "RECREATE")

    # RECO level qT
    datagroups = Datagroups("lowPU_%s_%s.pkl.lz4" % (flavor, met))
    procs = []
    if flavor == "mumu": procs = ["SingleMuon", "DYmumu"] + bkgs
    if flavor == "ee": procs = ["SingleElectron", "DYee"] + bkgs

    
    for proc in procs:
    
        bhist_para_qT = readProc(datagroups, "recoil_corr_lep_para_qT_qTbinned", proc) # recoil_corr_xy_para_qT_qTbinned recoil_uncorr_para_qT_qTbinned
        bhist_perp = readProc(datagroups, "recoil_corr_lep_perp_qTbinned", proc) # recoil_corr_xy_perp_qTbinned recoil_uncorr_perp_qTbinned
        rhist_para_qT = narf.hist_to_root(bhist_para_qT)
        rhist_perp = narf.hist_to_root(bhist_perp)
        for iBin in range(1, rhist_para_qT.GetNbinsX()+1):

            hist_para = rhist_para_qT.ProjectionY("%s_para_bin%d" % (proc, iBin), iBin, iBin)
            hist_para.Write()
            print(iBin, hist_para.Integral())
            
            hist_perp = rhist_perp.ProjectionY("%s_perp_bin%d" % (proc, iBin), iBin, iBin)
            hist_perp.Write()
         
         
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
        
    fOut.ls()
    fOut.Close()
  






def doPlot(comp, param, jsIn, outDir, label, xMin=0, xMax=150, yMin=0, yMax=30, yTitle=""):


    f_base = ROOT.TF1("f_base", jsIn[param]['func'], 0, 300)
    for i in range(0, jsIn[param]['nParams']): f_base.SetParameter(i, jsIn[param]['p%d' % i])
    f_unc = ROOT.TF1("f_unc", jsIn[param]['func'], 0, 300)
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
        print(x, y, y_err)
        
        g.SetPoint(iPoint, x, y)   
        g.SetPointError(iPoint, 0, y_err)
        print(x, y)
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
            
        'topRight'          : "199 pb^{#minus1} (13 TeV)", 
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
    
    
    
def doFit(jsIn, jsOut, comp, param, fitF, iParams, outDir, label, cParams=[], fitMin=0, fitMax=200, xMin=0, xMax=150, yMin=0, yMax=30, yTitle="", doFit=True):

    fit = ROOT.TF1("fit", fitF, 0, 300) if fitF != "" else None
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
        x = 0.5*(recoil_qTbins[qTbin-1] + recoil_qTbins[qTbin])
        x_err = 0.5*(recoil_qTbins[qTbin] - recoil_qTbins[qTbin-1])
        if not param in jsIn[str(qTbin)]:
            print("WARNING: param %s not found for qTbin %d" % (param, qTbin))
            continue
        y = jsIn[str(qTbin)][param]
        y_err = jsIn[str(qTbin)][param + "_err"]
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
            
        'topRight'          : "199 pb^{#minus1} (13 TeV)", 
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

def doFit_mc_perp_param():

    fOut_ = "wremnants/data/lowPU/recoil_param_%s_%s.root" % (flavor, met)
    #outDir = "/eos/user/j/jaeyserm/www/wmass/lowPU/recoilCorrectionParametric/recoil_%s_%s_parametric/" % (flavor, met)
    fIn = ROOT.TFile("wremnants/data/lowPU/recoil_param_%s_%s_mc_perp.root" % (flavor, met))
  
    tag = "mc_perp"
    
    
    outDir_ = "%s/%s_parametric" % (outDir, tag)
    functions.prepareDir(outDir_)

    cfg = {

        'logy'              : False,
        'logx'              : False,
        
        'xmin'              : 0,
        'xmax'              : 150,
        'ymin'              : -200,
        'ymax'              : 20,
            
        'xtitle'            : "q_{T} (GeV)",
        'ytitle'            : "#mu(U_{#parallel}) (GeV)",
            
        'topRight'          : "199 pb^{#minus1} (13 TeV)", 
        'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",
            
        'ratiofraction'     : 0.25,
        'ytitleR'           : "Ratio",
        'yminR'             : -6, # 0.88
        'ymaxR'             : 6, # 1.12
    } 
    
    
    

    ###################
    #### para_data_mean
    ###################
    idx = 1
    hIn = copy.deepcopy(fIn.Get(tag))
    
    fit = ROOT.TF1("fit_%s" % tag, "[0]*x + [1]", 0, 200)
    cfg['ymin'], cfg['ymax'], cfg['ytitle'] = -3, 3, "#mu (GeV)"
    doPlot(hIn, fit, cfg, "%s/%s" % (outDir_, "mean"), "MC", 1, fitMin=0, fitMax=100)
    
    
    fit = ROOT.TF1("fit_%s" % tag, "[0]*TMath::Power(x+[1], [2])", 0, 200)
    fit.SetParameters(3.38525e+00, 8.13936e-01, 1.29584e-01)
    cfg['ymin'], cfg['ymax'], cfg['ytitle'] = 0, 20, "#sigma_{1} (GeV)"
    doPlot(hIn, fit, cfg, "%s/%s" % (outDir_, "sigma1"), "MC", 2, fitMin=0, fitMax=150)


    fit = ROOT.TF1("fit_%s" % tag, "[0]*TMath::Power(x+[1], [2])", 0, 200)
    fit.SetParameters(5.47523e+00, 8.14368e+00, 1.20125e-01)
    cfg['ymin'], cfg['ymax'], cfg['ytitle'] = 0, 20, "#sigma_{2} (GeV)"
    doPlot(hIn, fit, cfg, "%s/%s" % (outDir_, "sigma2"), "MC", 3, fitMin=0, fitMax=150)
    
    
    fit = ROOT.TF1("fit_%s" % tag, "[0]*TMath::Power(x+[1], [2])", 0, 200)
    #fit = ROOT.TF1("fit_%s" % tag, "[4]*x*x*x*x + [3]*x*x*x + [2]*x*x + [1]*x + [0]", 0, 200)
    fit.SetParameters(8.12452e+00, 7.04522e+00, 9.70946e-02)
    cfg['ymin'], cfg['ymax'], cfg['ytitle'] = 0, 20, "#sigma_{3} (GeV)"
    doPlot(hIn, fit, cfg, "%s/%s" % (outDir_, "sigma3"), "MC", 4, fitMin=0, fitMax=60)
    
    
    fit = ROOT.TF1("fit_%s" % tag, "[0]*TMath::Power(x+[1], [2])", 0, 200)
    fit.SetParameters(5, 10, 9.72398e-01)
    cfg['ymin'], cfg['ymax'], cfg['ytitle'] = 10, 40, "#sigma_{4} (GeV)"
    doPlot(hIn, fit, cfg, "%s/%s" % (outDir_, "sigma4"), "MC", 5, fitMin=0, fitMax=60)
 
 
def doFit_data_perp_param():

    fOut_ = "wremnants/data/lowPU/recoil_param_%s_%s.root" % (flavor, met)
    #outDir = "/eos/user/j/jaeyserm/www/wmass/lowPU/recoilCorrectionParametric/recoil_%s_%s_parametric/" % (flavor, met)
    fIn = ROOT.TFile("wremnants/data/lowPU/recoil_param_%s_%s_data_perp.root" % (flavor, met))
  
    tag = "data_perp"
    
    
    outDir_ = "%s/%s_parametric" % (outDir, tag)
    functions.prepareDir(outDir_)

    cfg = {

        'logy'              : False,
        'logx'              : False,
        
        'xmin'              : 0,
        'xmax'              : 150,
        'ymin'              : -200,
        'ymax'              : 20,
            
        'xtitle'            : "q_{T} (GeV)",
        'ytitle'            : "#mu(U_{#parallel}) (GeV)",
            
        'topRight'          : "199 pb^{#minus1} (13 TeV)", 
        'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",
            
        'ratiofraction'     : 0.25,
        'ytitleR'           : "Ratio",
        'yminR'             : -6, # 0.88
        'ymaxR'             : 6, # 1.12
    } 
    
    
    
    hIn = fIn.Get(tag)
    
    fit = ROOT.TF1("fit_%s_mean" % tag, "[0]*x + [1]", 0, 200)
    fit.SetParameters(0, 0)
    cfg['ymin'], cfg['ymax'], cfg['ytitle'] = -3, 3, "#mu (GeV)"
    doPlot(hIn, fit, cfg, "%s/%s" % (outDir_, "mean"), "Data", 1, fitMin=0, fitMax=50)
    
    
    fit = ROOT.TF1("fit_%s_sigma1" % tag, "[0]*TMath::Power(x+[1], [2])", 0, 200)
    fit.SetParameters(3.12720e+00, 4.43889e+00, 1.49295e-01)
    cfg['ymin'], cfg['ymax'], cfg['ytitle'] = 0, 20, "#sigma_{1} (GeV)"
    doPlot(hIn, fit, cfg, "%s/%s" % (outDir_, "sigma1"), "Data", 2, fitMin=0, fitMax=50)


    fit = ROOT.TF1("fit_%s_sigma2" % tag, "[0]*TMath::Power(x+[1], [2])", 0, 200)
    fit.SetParameters(4.68280e+00, 1.51109e+01, 1.58014e-01)
    cfg['ymin'], cfg['ymax'], cfg['ytitle'] = 0, 20, "#sigma_{2} (GeV)"
    doPlot(hIn, fit, cfg, "%s/%s" % (outDir_, "sigma2"), "Data", 3, fitMin=0, fitMax=50)
    
    
    fit = ROOT.TF1("fit_%s_sigma3" % tag, "[0]*TMath::Power(x+[1], [2])", 0, 200)
    fit.SetParameters(7.73471e+00, 8.64117e+00, 1.17306e-01)
    cfg['ymin'], cfg['ymax'], cfg['ytitle'] = 0, 20, "#sigma_{3} (GeV)"
    doPlot(hIn, fit, cfg, "%s/%s" % (outDir_, "sigma3"), "Data", 4, fitMin=0, fitMax=50)
    
    
    fit = ROOT.TF1("fit_%s_sigma4" % tag, "[0]*TMath::Power(x+[1], [2])", 0, 200)
    #fit = ROOT.TF1("fit_%s_sigma4" % tag, "[0]", 0, 200)
    #fit.SetParameters(0)
    fit.SetParameters(5.64193e-01, 1.59898e+02, 6.22697e-01)
    #fit.FixParameter(2, 1)
    cfg['ymin'], cfg['ymax'], cfg['ytitle'] = 10, 40, "#sigma_{4} (GeV)"
    doPlot(hIn, fit, cfg, "%s/%s" % (outDir_, "sigma4"), "Data", 5, fitMin=0, fitMax=60)
      
  
def doFit_ttbar_perp_param(bkg, tag, outDir_, name, label):
  
    jsOut = {}
    with open("%s/fit.json" % outDir_) as f: jsIn = json.load(f)
  
   
    # mean
    fitF = ""
    params = []
    doFit(jsIn, jsOut, "norm", fitF, params, outDir_, label, fitMin=0, fitMax=200, yMin=-5, yMax=5, yTitle = "#mu (GeV)")

    # sigma1
    fitF = "[0]*x + [1]"
    params = [-4.42073e-02, 4.68472e+01]
    doFit(jsIn, jsOut, "sigma1", fitF, params, outDir_, label, fitMin=0, fitMax=200, yMin=0, yMax=100, yTitle = "#sigma_{1} (GeV)")

    # sigma2
    fitF = "[0]*x*x + [1]*x + [2]"
    params = [0, 0, 0]
    doFit(jsIn, jsOut, "sigma2", fitF, params, outDir_, label, fitMin=0, fitMax=200, yMin=0, yMax=100, yTitle = "#sigma_{2} (GeV)")
    
    # norm
    fitF = ""
    params = []
    doFit(jsIn, jsOut, "norm", fitF, params, outDir_, label, fitMin=0, fitMax=200, yMin=0, yMax=1, yTitle = "n (/)")
    
    # yield
    fitF = ""
    params = []
    doFit(jsIn, jsOut, "yield", fitF, params, outDir_, label, fitMin=0, fitMax=200, yMin=0, yMax=5, yTitle = "Event yield")
    

    
  
def doFit_ewk_perp_param():

    fOut_ = "wremnants/data/lowPU/recoil_param_%s_%s.root" % (flavor, met)
    #outDir = "/eos/user/j/jaeyserm/www/wmass/lowPU/recoilCorrectionParametric/recoil_%s_%s_parametric/" % (flavor, met)
    fIn = ROOT.TFile("wremnants/data/lowPU/recoil_param_%s_%s_ewk_perp.root" % (flavor, met))
  
    tag = "ewk_perp"
    
    
    outDir_ = "%s/%s_parametric" % (outDir, tag)
    functions.prepareDir(outDir_)

    cfg = {

        'logy'              : False,
        'logx'              : False,
        
        'xmin'              : 0,
        'xmax'              : 150,
        'ymin'              : -200,
        'ymax'              : 20,
            
        'xtitle'            : "q_{T} (GeV)",
        'ytitle'            : "#mu(U_{#parallel}) (GeV)",
            
        'topRight'          : "199 pb^{#minus1} (13 TeV)", 
        'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",
            
        'ratiofraction'     : 0.25,
        'ytitleR'           : "Ratio",
        'yminR'             : -6, # 0.88
        'ymaxR'             : 6, # 1.12
    } 
    

    hIn = fIn.Get(tag)
    
    # mean
    fit = None
    cfg['ymin'], cfg['ymax'], cfg['ytitle'] = -10, 10, "#mu (GeV)"
    doPlot(hIn, fit, cfg, "%s/%s" % (outDir_, "mean"), "EWK", 1, fitMin=0, fitMax=100)
    
    # sigma1
    fit = ROOT.TF1("fit_%s_sigma1" % tag, "[0]*TMath::Power(x+[1], [2])", 0, 200)
    fit.SetParameters(6.76272e+00, -4.94509e-01, 1.33996e-01)
    cfg['ymin'], cfg['ymax'], cfg['ytitle'] = 0, 50, "#sigma_{1} (GeV)"
    doPlot(hIn, fit, cfg, "%s/%s" % (outDir_, "sigma1"), "EWK", 2, fitMin=0, fitMax=150)

    # sigma2
    fit = ROOT.TF1("fit_%s_sigma2" % tag, "[0]*TMath::Power(x+[1], [2])", 0, 200)
    fit.SetParameters(1.45688e+01, -4.99997e-01, 2.80827e-01)
    #fit = ROOT.TF1("fit_%s_sigma2" % tag, "[0]*TMath::Sqrt(x+[1])", 0, 200)
    #fit.SetParameters(6.76272e+00, -4.94509e-01, 1.33996e-01)
    cfg['ymin'], cfg['ymax'], cfg['ytitle'] = 0, 100, "#sigma_{2} (GeV)"
    doPlot(hIn, fit, cfg, "%s/%s" % (outDir_, "sigma2"), "EWK", 3, fitMin=0, fitMax=100)
    
    # norm
    fit = ROOT.TF1("fit_%s_norm" % tag, "[0]*x + [1]", 0, 200)
    fit.SetParameters(-1.25835e-03, 8.64469e-01)
    cfg['ymin'], cfg['ymax'], cfg['ytitle'] = 0, 1, "Norm (/)"
    doPlot(hIn, fit, cfg, "%s/%s" % (outDir_, "norm"), "EWK", 4, fitMin=0, fitMax=60)
    
    # yield
    fit = None
    cfg['ymin'], cfg['ymax'], cfg['ytitle'] = 0, 10, "Yields (/)"
    doPlot(hIn, fit, cfg, "%s/%s" % (outDir_, "yields"), "EWK", 5, fitMin=0, fitMax=60)
    
    
  
  
 
def doFit_mc_perp_autoFit():

    recoil = ROOT.RooRealVar("recoil", "Recoil parallel (GeV)", 0, -500, 500)
    
    w = ROOT.RooWorkspace("w", "workspace")
    cats = ROOT.RooCategory("category", "") # for each qT bin, define category
    hists = ROOT.std.map("string, RooDataHist*")() # container holding all RooDataHists
    
    

    # sigma 1
    s1_a = ROOT.RooRealVar("s1_a", "", 3.02414e+00, -100, 100)
    s1_b = ROOT.RooRealVar("s1_b", "", 4.85217e+00, -100, 100)
    s1_c = ROOT.RooRealVar("s1_c", "", 1.55760e-01, -100, 100)
    
    # sigma 2
    s2_a = ROOT.RooRealVar("s2_a", "", 5.47523e+00, -100, 100)
    s2_b = ROOT.RooRealVar("s2_b", "", 8.14375e+00, -100, 100)
    s2_c = ROOT.RooRealVar("s2_c", "", 1.20125e-01, -100, 100)
    
    # sigma 3
    s3_a = ROOT.RooRealVar("s3_a", "", 8.12452e+00, -100, 100)
    s3_b = ROOT.RooRealVar("s3_b", "", 7.04516e+00, -100, 100)
    s3_c = ROOT.RooRealVar("s3_c", "", 9.70948e-02, -100, 100)
    
    # sigma 4
    s4_a = ROOT.RooRealVar("s4_a", "", 9.58838e-02, -100, 100)
    s4_b = ROOT.RooRealVar("s4_b", "", 1.20719e+02, -100, 200)
    s4_c = ROOT.RooRealVar("s4_c", "", 1.03581e+00, -100, 100)

    # poly
    s1_z1 = ROOT.RooRealVar("s1_z1", "", 0, -1, 1)
    s1_z2 = ROOT.RooRealVar("s1_z2", "", 0, -1, 1)
    s2_z1 = ROOT.RooRealVar("s2_z1", "", 0, -1, 1)
    s2_z2 = ROOT.RooRealVar("s2_z2", "", 0, -1, 1)

    mean = ROOT.RooRealVar("mean", "", 0)
    n1 = ROOT.RooRealVar("n1", "", 0.22)
    n2 = ROOT.RooRealVar("n2", "", 0.5)
    n3 = ROOT.RooRealVar("n3", "", 0.25)
    
    nBins = 0
    for qTbin in range(1, len(recoil_qTbins)):
        
        qT = 0.5*(recoil_qTbins[qTbin-1] + recoil_qTbins[qTbin]) if qTbin < len(recoil_qTbins)-1 else (recoil_qTbins[qTbin-1] + 20)
        qTmin, qTmax = recoil_qTbins[qTbin-1], recoil_qTbins[qTbin]
        hName = "%s_perp_bin%d" % (sig, qTbin)
        
        if qT > 50: continue
        nBins += 1
        
        # import and store RDH
        hist = fIn.Get(hName)
        hist.Rebin(1)
        norm = hist.Integral()
        hist.Scale(1./norm)
        rdh = ROOT.RooDataHist("rdh", "", ROOT.RooArgList(recoil), ROOT.RooFit.Import(hist))
        rdh.SetName("rdh_bin%d" % (qTbin))
        catIDx = "cat_bin%s" % (qTbin)
        hists.insert(ROOT.std.pair("string, RooDataHist*")(catIDx, rdh))
        cats.defineType(catIDx, qTbin)    
        
        print("----------------------------------->", rdh.numEntries(), rdh.sumEntries())
        getattr(w, 'import')(rdh) # import before PDF construction

        
        sigma1 = ROOT.RooFormulaVar("sigma1_bin%d" % (qTbin), "@0*TMath::Power(%f+@1, @2)" % (qT), ROOT.RooArgList(s1_a, s1_b, s1_c))
        sigma2 = ROOT.RooFormulaVar("sigma2_bin%d" % (qTbin), "@0*TMath::Power(%f+@1, @2)" % (qT), ROOT.RooArgList(s2_a, s2_b, s2_c))
        
        #sigma1 = ROOT.RooFormulaVar("sigma1_bin%d" % (qTbin), "@0*TMath::Power(%f+@1, @2)*TMath::Exp(%f*@3)" % (qT, qT), ROOT.RooArgList(s1_a, s1_b, s1_c, s1_z1))
        #sigma2 = ROOT.RooFormulaVar("sigma2_bin%d" % (qTbin), "@0*TMath::Power(%f+@1, @2)*TMath::Exp(%f*@3)" % (qT, qT), ROOT.RooArgList(s2_a, s2_b, s2_c, s2_z1, s2_z2))
        #sigma1 = ROOT.RooFormulaVar("sigma1_bin%d" % (qTbin), "@0*TMath::Power(%f+@1, @2)*TMath::Exp(%f*@3 + %f*%f*@3)" % (qT, qT, qT, qT), ROOT.RooArgList(s1_a, s1_b, s1_c, s1_z1, s1_z2))
        #sigma2 = ROOT.RooFormulaVar("sigma2_bin%d" % (qTbin), "@0*TMath::Power(%f+@1, @2)*TMath::Exp(%f*@3 + %f*%f*@3)" % (qT, qT, qT, qT), ROOT.RooArgList(s2_a, s2_b, s2_c, s2_z1, s2_z2))
        
        sigma3 = ROOT.RooFormulaVar("sigma3_bin%d" % (qTbin), "@0*TMath::Power(%f+@1, @2)" % (qT), ROOT.RooArgList(s3_a, s3_b, s3_c))
        sigma4 = ROOT.RooFormulaVar("sigma4_bin%d" % (qTbin), "@0*TMath::Power(%f+@1, @2)" % (qT), ROOT.RooArgList(s4_a, s4_b, s4_c))
        
        

        gauss1 = ROOT.RooGaussian("gauss1_bin%d" % (qTbin), "", recoil, mean, sigma1)
        gauss2 = ROOT.RooGaussian("gauss2_bin%d" % (qTbin), "", recoil, mean, sigma2)
        gauss3 = ROOT.RooGaussian("gauss3_bin%d" % (qTbin), "", recoil, mean, sigma3)
        gauss4 = ROOT.RooGaussian("gauss4_bin%d" % (qTbin), "", recoil, mean, sigma4)
        
        
        pdf = ROOT.RooAddPdf("pdf_bin%d" % (qTbin), '', ROOT.RooArgList(gauss1, gauss2, gauss3, gauss4), ROOT.RooArgList(n1, n2, n3))
            

        getattr(w, 'import')(pdf) # import must happen after the initial guesses
        
        print("----------------------------------->", rdh.numEntries())       

  
    w.Print()

    print("#############################################################################################################################")
    
    recoil.setBins(600) # seems to be necessary...
    #qt = ROOT.RooRealVar("qt", "", 0, 0, 1) 
    
    # construct total RDH
    rdh_tot = ROOT.RooDataHist("rdh_tot", "", ROOT.RooArgList(recoil), cats, hists)
    
    

    cats.setIndex(1)
    print("Categories")
    for cat in cats:
        print(cat)
    print("------")

    # construct total PDF
    pdf_tot = ROOT.RooSimultaneous("pdf_tot", "", cats)
    for iBin in range(1, nBins+1):
        
        pdf = w.obj("pdf_bin%d" % (iBin))
        pdf.Print()
        pdf_tot.addPdf(pdf, "cat_bin%s" % (iBin))
        
    print("#############################################################################################################################")
    
    pdf_tot.Print()
    selectedPdf = pdf_tot.getPdf("cat_bin1")
    selectedPdf.Print()
    print(hists['cat_bin1'])
    #selectedPdf.fitTo(hists['qTbin0'], ROOT.RooFit.SumW2Error(ROOT.kTRUE))

    fitRes = pdf_tot.fitTo(rdh_tot, ROOT.RooFit.SumW2Error(ROOT.kTRUE), ROOT.RooFit.Save(ROOT.kTRUE))
    #pdfTot.fitTo(totrdh, ROOT.RooFit.Extended(ROOT.kTRUE), ROOT.RooFit.SumW2Error(ROOT.kTRUE))
    #fitRes = pdf.fitTo(rdh, ROOT.RooFit.Extended(ROOT.kTRUE), ROOT.RooFit.SumW2Error(ROOT.kTRUE), ROOT.RooFit.Save(ROOT.kTRUE))
 
    params = fitRes.floatParsFinal()  
  


def doFit_data_perp_autoFit():

    recoil = ROOT.RooRealVar("recoil", "Recoil parallel (GeV)", 0, -500, 500)
    
    w = ROOT.RooWorkspace("w", "workspace")
    cats = ROOT.RooCategory("category", "") # for each qT bin, define category
    hists = ROOT.std.map("string, RooDataHist*")() # container holding all RooDataHists
    

    

    # sigma 1
    s1_a = ROOT.RooRealVar("s1_a", "", 4.99982e+00, 0, 100)
    s1_b = ROOT.RooRealVar("s1_b", "", 7.67005e-01, 0, 1000)
    s1_c = ROOT.RooRealVar("s1_c", "", 5.53110e-02, 0, 100)
    
    # sigma 2
    s2_a = ROOT.RooRealVar("s2_a", "", 3.68495e-01, 0, 100)
    s2_b = ROOT.RooRealVar("s2_b", "", 1.13750e+02, 0, 1000)
    s2_c = ROOT.RooRealVar("s2_c", "", 6.69613e-01, 0, 100)
    
    # sigma 3
    s3_a = ROOT.RooRealVar("s3_a", "", 5.64193e-01, 0, 100)
    s3_b = ROOT.RooRealVar("s3_b", "", 1.59898e+02, 0, 1000)
    s3_c = ROOT.RooRealVar("s3_c", "", 6.22697e-01, 0, 100)
    
    # sigma 4
    s4_a = ROOT.RooRealVar("s4_a", "", 1.49073e+01)
    #s4_b = ROOT.RooRealVar("s4_b", "", 1.15485e+04, -100, 200)
    #s4_c = ROOT.RooRealVar("s4_c", "", 1) ## fixed

    # poly
    s1_z1 = ROOT.RooRealVar("s1_z1", "", 0, -1, 1)
    s1_z2 = ROOT.RooRealVar("s1_z2", "", 0, -1, 1)
    s2_z1 = ROOT.RooRealVar("s2_z1", "", 0, -1, 1)
    s2_z2 = ROOT.RooRealVar("s2_z2", "", 0, -1, 1)

    mean = ROOT.RooRealVar("mean", "", 0)
    n1 = ROOT.RooRealVar("n1", "", 0.3)
    n2 = ROOT.RooRealVar("n2", "", 0.6)
    #n3 = ROOT.RooRealVar("n3", "", 0.25)
    
    
    # ewk
    norm_ewk_func = ROOT.TF1("norm_ewk_func", "[0]*x + [1]", 0, 500)
    norm_ewk_func.SetParameters(-1.25835e-03, 8.64469e-01)
    sigma1_ewk_func = ROOT.TF1("sigma1_ewk_func", "[0]*TMath::Power(x+[1], [2])", 0, 500)
    sigma1_ewk_func.SetParameters(6.76272e+00, -4.94509e-01, 1.33996e-01)
    sigma2_ewk_func = ROOT.TF1("sigma2_ewk_func", "[0]*TMath::Power(x+[1], [2])", 0, 500)
    sigma2_ewk_func.SetParameters(1.45688e+01, -4.99997e-01, 2.80827e-01)

    
    # ttbar
    sigma1_ttbar_func = ROOT.TF1("sigma1_ttbar_func", "[0]*x + [1]", 0, 500)
    sigma1_ttbar_func.SetParameters(-4.42073e-02, 4.68472e+01)
    sigma2_ttbar_func = ROOT.TF1("sigma2_ttbar_func", "[0]*x*x + [1]*x + [2]", 0, 500)
    sigma2_ttbar_func.SetParameters(-1.08054e-03, 7.03511e-02, 7.01008e+01)
    
    
    
    
    
    nBins = 0
    for qTbin in range(1, len(recoil_qTbins)):
        
        qT = 0.5*(recoil_qTbins[qTbin-1] + recoil_qTbins[qTbin]) if qTbin < len(recoil_qTbins)-1 else (recoil_qTbins[qTbin-1] + 20)
        qTmin, qTmax = recoil_qTbins[qTbin-1], recoil_qTbins[qTbin]
        hName = "%s_perp_bin%d" % ("SingleMuon", qTbin)
        
        if qT > 50: continue
        nBins += 1
        
        hist_ttbar = fIn.Get("%s_perp_bin%d" % ("TTbar", qTbin))
        hist_ewk = fIn.Get("%s_perp_bin%d" % ("EWK", qTbin))
        
        # import and store RDH
        hist = fIn.Get(hName)
        hist.Rebin(1)
        norm = hist.Integral()
        hist.Scale(1./norm)
        rdh = ROOT.RooDataHist("rdh", "", ROOT.RooArgList(recoil), ROOT.RooFit.Import(hist))
        rdh.SetName("rdh_bin%d" % (qTbin))
        catIDx = "cat_bin%s" % (qTbin)
        hists.insert(ROOT.std.pair("string, RooDataHist*")(catIDx, rdh))
        cats.defineType(catIDx, qTbin)    
        
        print("----------------------------------->", rdh.numEntries(), rdh.sumEntries())
        getattr(w, 'import')(rdh) # import before PDF construction
        
        
        ## ewk
        yield_ewk = ROOT.RooRealVar("yield_ewk_bin%d" % (qTbin), "", hist_ewk.Integral()/norm)
        mean_ewk = ROOT.RooRealVar("mean_ewk_bin%d" % (qTbin), "", 0) # fixed
        norm_ewk = ROOT.RooRealVar("norm_ewk_bin%d" % (qTbin), "", norm_ewk_func.Eval(qT))
        sigma1_ewk = ROOT.RooRealVar("sigma1_ewk_bin%d" % (qTbin), "", sigma1_ewk_func.Eval(qT))
        sigma2_ewk = ROOT.RooRealVar("sigma2_ewk_bin%d" % (qTbin), "", sigma2_ewk_func.Eval(qT))
        gauss1_ewk = ROOT.RooGaussian("gauss1_ewk_bin%d" % (qTbin), "", recoil, mean_ewk, sigma1_ewk)
        gauss2_ewk = ROOT.RooGaussian("gauss2_ewk_bin%d" % (qTbin), "", recoil, mean_ewk, sigma2_ewk)
        pdf_ewk = ROOT.RooAddPdf("pdf_ewk_bin%d" % (qTbin), '', ROOT.RooArgList(gauss1_ewk, gauss2_ewk), ROOT.RooArgList(norm_ewk))        
            
        # ttbar
        yield_ttbar = ROOT.RooRealVar("yield_ttbar_bin%d" % (qTbin), "", hist_ttbar.Integral()/norm)
        mean_ttbar = ROOT.RooRealVar("mean_ttbar_bin%d" % (qTbin), "", 0)
        norm_ttbar = ROOT.RooRealVar("norm_ttbar_bin%d" % (qTbin), "", 0.7)
        sigma1_ttbar = ROOT.RooRealVar("sigma1_ttbar_bin%d" % (qTbin), "", sigma1_ttbar_func.Eval(qT))
        sigma2_ttbar = ROOT.RooRealVar("sigma2_ttbar_bin%d" % (qTbin), "", sigma2_ttbar_func.Eval(qT))
        gauss1_ttbar = ROOT.RooGaussian("gauss1_ttbar_bin%d" % (qTbin), "", recoil, mean_ttbar, sigma1_ttbar)
        gauss2_ttbar = ROOT.RooGaussian("gauss2_ttbar_bin%d" % (qTbin), "", recoil, mean_ttbar, sigma2_ttbar)
        pdf_ttbar = ROOT.RooAddPdf("pdf_ttbar_bin%d" % (qTbin), '', ROOT.RooArgList(gauss1_ttbar, gauss2_ttbar), ROOT.RooArgList(norm_ttbar))        

        
        sigma1 = ROOT.RooFormulaVar("sigma1_bin%d" % (qTbin), "@0*TMath::Power(%f+@1, @2)" % (qT), ROOT.RooArgList(s1_a, s1_b, s1_c))
        sigma2 = ROOT.RooFormulaVar("sigma2_bin%d" % (qTbin), "@0*TMath::Power(%f+@1, @2)" % (qT), ROOT.RooArgList(s2_a, s2_b, s2_c))
        
        #sigma1 = ROOT.RooFormulaVar("sigma1_bin%d" % (qTbin), "@0*TMath::Power(%f+@1, @2)*TMath::Exp(%f*@3)" % (qT, qT), ROOT.RooArgList(s1_a, s1_b, s1_c, s1_z1))
        #sigma2 = ROOT.RooFormulaVar("sigma2_bin%d" % (qTbin), "@0*TMath::Power(%f+@1, @2)*TMath::Exp(%f*@3)" % (qT, qT), ROOT.RooArgList(s2_a, s2_b, s2_c, s2_z1, s2_z2))
        #sigma1 = ROOT.RooFormulaVar("sigma1_bin%d" % (qTbin), "@0*TMath::Power(%f+@1, @2)*TMath::Exp(%f*@3 + %f*%f*@3)" % (qT, qT, qT, qT), ROOT.RooArgList(s1_a, s1_b, s1_c, s1_z1, s1_z2))
        #sigma2 = ROOT.RooFormulaVar("sigma2_bin%d" % (qTbin), "@0*TMath::Power(%f+@1, @2)*TMath::Exp(%f*@3 + %f*%f*@3)" % (qT, qT, qT, qT), ROOT.RooArgList(s2_a, s2_b, s2_c, s2_z1, s2_z2))
        
        sigma3 = ROOT.RooFormulaVar("sigma3_bin%d" % (qTbin), "@0*TMath::Power(%f+@1, @2)" % (qT), ROOT.RooArgList(s3_a, s3_b, s3_c))
        #sigma4 = ROOT.RooFormulaVar("sigma4_bin%d" % (qTbin), "@0", ROOT.RooArgList(s4_a))
        
        

        gauss1 = ROOT.RooGaussian("gauss1_bin%d" % (qTbin), "", recoil, mean, sigma1)
        gauss2 = ROOT.RooGaussian("gauss2_bin%d" % (qTbin), "", recoil, mean, sigma2)
        gauss3 = ROOT.RooGaussian("gauss3_bin%d" % (qTbin), "", recoil, mean, sigma3)
        #gauss4 = ROOT.RooGaussian("gauss4_bin%d" % (qTbin), "", recoil, mean, sigma4)
        
        
        pdf_sig = ROOT.RooAddPdf("pdf_sig_bin%d" % (qTbin), '', ROOT.RooArgList(gauss1, gauss2, gauss3), ROOT.RooArgList(n1, n2))
        
        
        # total data model
        pdf = ROOT.RooAddPdf("pdf_bin%d" % (qTbin), '', ROOT.RooArgList(pdf_ttbar, pdf_ewk, pdf_sig), ROOT.RooArgList(yield_ttbar, yield_ewk))
            
        
        getattr(w, 'import')(pdf) # import must happen after the initial guesses
        
        print("----------------------------------->", rdh.numEntries())       

    w.Print()

    print("#############################################################################################################################")
    
    #recoil.setBins(600) # seems to be necessary...
    #qt = ROOT.RooRealVar("qt", "", 0, 0, 1) 
    
    # construct total RDH
    rdh_tot = ROOT.RooDataHist("rdh_tot", "", ROOT.RooArgList(recoil), cats, hists)
    
    

    cats.setIndex(1)
    print("Categories")
    for cat in cats:
        print(cat)
    print("------")

    # construct total PDF
    pdf_tot = ROOT.RooSimultaneous("pdf_tot", "", cats)
    for iBin in range(1, nBins+1):
        
        pdf = w.obj("pdf_bin%d" % (iBin))
        pdf.Print()
        pdf_tot.addPdf(pdf, "cat_bin%s" % (iBin))
        
    print("#############################################################################################################################")
    
    pdf_tot.Print()
    selectedPdf = pdf_tot.getPdf("cat_bin1")
    selectedPdf.Print()
    print(hists['cat_bin1'])
    #selectedPdf.fitTo(hists['qTbin0'], ROOT.RooFit.SumW2Error(ROOT.kTRUE))

    fitRes = pdf_tot.fitTo(rdh_tot, ROOT.RooFit.SumW2Error(ROOT.kTRUE), ROOT.RooFit.Save(ROOT.kTRUE))
    #pdfTot.fitTo(totrdh, ROOT.RooFit.Extended(ROOT.kTRUE), ROOT.RooFit.SumW2Error(ROOT.kTRUE))
    #fitRes = pdf.fitTo(rdh, ROOT.RooFit.Extended(ROOT.kTRUE), ROOT.RooFit.SumW2Error(ROOT.kTRUE), ROOT.RooFit.Save(ROOT.kTRUE))
 
    params = fitRes.floatParsFinal()  
 
 
def combinedFit(proc, comp, jsIn, qTmin=0, qTmax=50, rebin=1, singleMean=False, bkgCfg={}):
    
    fIn = ROOT.TFile(fIn_)
    nGauss = jsIn['nGauss']
    recoil = ROOT.RooRealVar("recoil", "", 0, -300, 300)

    cats = ROOT.RooCategory("category", "") # for each qT bin, define category
    hists = ROOT.std.map("string, RooDataHist*")() # container holding all RooDataHists
    ptrs = {}
    fitFunctions = {}
    garbage = [] # need to store the variables for memory issues
    means_arglist, sigmas_arglist, norms_arglist = [], [], []
    for iGauss in range(1, nGauss+1):
        # prepare variables
        mean_arglist, sigma_arglist = ROOT.RooArgList(), ROOT.RooArgList()
        mean_arglist.setName("gauss%d_mean_arglist" % iGauss)
        sigma_arglist.setName("gauss%d_sigma_arglist" % iGauss)
        for iParam in range(0, jsIn['sigma%d' % iGauss]['nParams']):
            tmp = 'sigma%d_p%d' % (iGauss, iParam)
            ptrs[tmp] = ROOT.RooRealVar(tmp, "", jsIn['sigma%d' % iGauss]['p%d' % iParam], 0, 100)
            ptrs[tmp].SetName(tmp)
            if jsIn['sigma%d' % iGauss]['p%d_isCte' % iParam]: ptrs[tmp].setConstant(ROOT.kTRUE)
            sigma_arglist.addOwned(ptrs[tmp])

        for iParam in range(0, jsIn['mean%d' % iGauss]['nParams']):
            tmp = 'mean%d_p%d' % (iGauss, iParam)
            if singleMean and iGauss != 1:
                ptrs[tmp] = ROOT.RooFormulaVar(tmp, "@0", ROOT.RooArgList(ptrs['mean%d_p%d' % (1, iParam)]))
            else:
                ptrs[tmp] = ROOT.RooRealVar(tmp, "", jsIn['mean%d' % iGauss]['p%d' % iParam], -100, 100)
                if jsIn['mean%d' % iGauss]['p%d_isCte' % iParam]: ptrs[tmp].setConstant(ROOT.kTRUE)

            ptrs[tmp].SetName(tmp)
            mean_arglist.addOwned(ptrs[tmp])

        means_arglist.append(mean_arglist)
        sigmas_arglist.append(sigma_arglist)

        # construct the RooFormulaVar
        fitF = jsIn['sigma%d' % iGauss]['func'].replace("Exp", "ABC").replace("x", "{0}").replace("[", "x[").replace("ABC", "Exp")
        fitFunctions['sigma%d' % iGauss] = fitF
        fitF = jsIn['mean%d' % iGauss]['func'].replace("Exp", "ABC").replace("x", "{0}").replace("[", "x[").replace("ABC", "Exp")
        fitFunctions['mean%d' % iGauss] = fitF
    

    for iGauss in range(1, nGauss):
        # prepare variables
        norm_arglist = ROOT.RooArgList("gauss%d_mean_norm" % iGauss)
        for iParam in range(0, jsIn['norm%d' % iGauss]['nParams']):
            tmp = 'norm%d_p%d' % (iGauss, iParam)
            ptrs[tmp] = ROOT.RooRealVar(tmp, "", jsIn['norm%d' % iGauss]['p%d' % iParam], -100, 100)
            if jsIn['norm%d' % iGauss]['p%d_isCte' % iParam]: ptrs[tmp].setConstant(ROOT.kTRUE)
            norm_arglist.addOwned(ptrs[tmp])
        norms_arglist.append(norm_arglist)
        
        # construct the RooFormulaVar
        fitF = jsIn['norm%d' % iGauss]['func'].replace("Exp", "ABC").replace("x", "{0}").replace("[", "x[").replace("ABC", "Exp")
        fitFunctions['norm%d' % iGauss] = fitF

  
    nBins = 0
    pdf_tot = ROOT.RooSimultaneous("pdf_tot", "", cats) # total pdf, containing all the categories
    for qTbin in range(1, len(recoil_qTbins)):
        
        qT = 0.5*(recoil_qTbins[qTbin-1] + recoil_qTbins[qTbin]) if qTbin < len(recoil_qTbins)-1 else (recoil_qTbins[qTbin-1] + 20)
        hName = "%s_%s_bin%d" % (proc, comp, qTbin)
        
        if qT < qTmin: continue
        if qT > qTmax: continue
        nBins += 1
        
        # import and store RDH
        hist = fIn.Get(hName)
        hist.Rebin(rebin)
        yield_ = hist.Integral()
        hist.Scale(1./yield_)
        rdh = ROOT.RooDataHist("", "", ROOT.RooArgList(recoil), ROOT.RooFit.Import(hist))
        rdh.SetName("rdh_bin%d" % (qTbin))
        catIDx = "cat_bin%s" % (qTbin)
        hists.insert(ROOT.std.pair("string, RooDataHist*")(catIDx, rdh))
        cats.defineType(catIDx, qTbin)    
        
        print("----------------------------------->", rdh.numEntries(), rdh.sumEntries())
        ptrs[rdh.GetName()] = rdh

        pdfs, norms = ROOT.RooArgList(), ROOT.RooArgList()
        pdfs.setName("pdfs_bin%d" % qTbin)
        norms.setName("norms_bin%d" % qTbin)        
        for iGauss in range(1, nGauss+1):
            fu = fitFunctions['mean%d' % iGauss]
            if "+0+" in fu: # check for piece-wise function
                fus = fu.split("+0+")
                func = None
                for f in fus:
                    t = f.split("*1*")
                    if eval(t[0].format(qT)):
                        func = t[1]
                        break
                if func == None: sys.exit("No piecewise function condition found for qT=%.2f and %s" % (qT, fu))
                mean_ = ROOT.RooFormulaVar("mean%d_bin%d" % (iGauss, qTbin), func.format(qT), means_arglist[iGauss-1])
            else: mean_ = ROOT.RooFormulaVar("mean%d_bin%d" % (iGauss, qTbin), fitFunctions['mean%d' % iGauss].format(qT), means_arglist[iGauss-1])
            sigma_ = ROOT.RooFormulaVar("sigma%d_bin%d" % (iGauss, qTbin), fitFunctions['sigma%d' % iGauss].format(qT), sigmas_arglist[iGauss-1])
            gauss = ROOT.RooGaussian("gauss%d_bin%d" % (iGauss, qTbin), "", recoil, mean_, sigma_)
            pdfs.addOwned(gauss)
            garbage.append(mean_)
            garbage.append(sigma_)
            garbage.append(gauss)
        

        for iGauss in range(1, nGauss):
            norm_ = ROOT.RooFormulaVar("norm%d_bin%d" % (iGauss, qTbin), fitFunctions['norm%d' % iGauss].format(qT), norms_arglist[iGauss-1])
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
                mean = ROOT.RooRealVar("mean%d_bin%d_%s" % (iGauss, qTbin, bkg), "", jsIn_bkg[str(qTbin)]["mean%d" % iGauss], -100, 100)
                sigma = ROOT.RooRealVar("sigma%d_bin%d_%s" % (iGauss, qTbin, bkg), "", jsIn_bkg[str(qTbin)]["sigma%d" % iGauss], 0.1, 100)
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
    

    #recoil.setBins(600) # seems to be necessary ??
    rdh_tot = ROOT.RooDataHist("rdh_tot", "", ROOT.RooArgList(recoil), cats, hists) # total RDH   , ROOT.RooFit.Minos(ROOT.kTRUE) , , ROOT.RooFit.Hesse(ROOT.kTRUE) 
    fitRes = pdf_tot.fitTo(rdh_tot, ROOT.RooFit.SumW2Error(ROOT.kTRUE), ROOT.RooFit.Save(ROOT.kTRUE))
    fitValid = (fitRes.covQual() == 3 and fitRes.status() == 0)
    print("****************************")
    print("FIT STATUS")
    print("Covariance Quality = %d" % fitRes.covQual())
    print("Fit status = %d" % fitRes.status())
    print("****************************")
    
    cov = fitRes.covarianceMatrix()
    cov.Print()
    
    #sys.exit()
    
    
    floatingParams = fitRes.floatParsFinal()
    jsOut = copy.deepcopy(jsIn)
    for iGauss in range(1, nGauss+1):
        for iParam in range(0, jsIn['sigma%d' % iGauss]['nParams']):
            tmp = 'sigma%d_p%d' % (iGauss, iParam)
            p = getParamIdx(tmp, floatingParams)
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
            if singleMean:
                tmp_1 = 'mean%d_p%d' % (1, iParam)
                p = getParamIdx(tmp_1, floatingParams) # try first Gauss
                if p == -1: val, err = ptrs[tmp_1].getVal(), 0
                else: val, err = floatingParams[p].getVal(), floatingParams[p].getError()
            jsOut['mean%d' % iGauss]['p%d' % iParam] = val
            jsOut['mean%d' % iGauss]['p%d_err' % iParam] = err
            print("%s \t %.3e" % (tmp, val))

    for iGauss in range(1, nGauss):
        for iParam in range(0, jsIn['norm%d' % iGauss]['nParams']):
            tmp = 'norm%d_p%d' % (iGauss, iParam)
            p = getParamIdx(tmp, floatingParams)
            if p == -1: val, err = ptrs[tmp].getVal(), 0
            else: val, err = floatingParams[p].getVal(), floatingParams[p].getError()
            jsOut['norm%d' % iGauss]['p%d' % iParam] = val
            jsOut['norm%d' % iGauss]['p%d_err' % iParam] = err
            print("%s \t %.3e" % (tmp, val))
    
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
                if singleMean:
                    tmp_1 = 'mean%d_p%d' % (1, iParam)
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
 

def do_ewk_perp():

    bkg = "EWK"
    tag = "ewk_perp"
    comp = "perp"
    baseDir = "%s/%s" % (outDir, tag)
    name = "EWK #rightarrow %s" % flavortag
    label = "%s, %s" % (name, met)
    functions.prepareDir(baseDir, False)
    rebin = 2

    ###### STEP 1: floating fits
    iteration = 0
    outDir_fits = "%s/fits_v0" % baseDir
    fitCfg = {}
    fitCfg['mean'], fitCfg['mean_cfg'] = [0, 0], [1, 1]
    fitCfg['sigma'], fitCfg['sigma_cfg'] = [20, 50], [0, 0]
    fitCfg['norm'], fitCfg['norm_cfg'] = [0.9], [1]
    nGauss = len(fitCfg['mean'])
    
    if True: doFitMultiGauss(bkg, comp, fitCfg, label, outDir_fits, qTmax=200, rebin=rebin, xMin=-200, xMax=200)
    
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


def do_ewk_para():

    bkg = "EWK"
    tag = "ewk_para"
    comp = "para"
    baseDir = "%s/%s" % (outDir, tag)
    name = "EWK #rightarrow %s" % flavortag
    label = "%s, %s" % (name, met)
    functions.prepareDir(baseDir, False)
    rebin = 2

    ###### STEP 1: floating fits
    iteration = 0
    outDir_fits = "%s/fits_v0" % baseDir
    fitCfg = {}
    fitCfg['mean'], fitCfg['mean_cfg'] = [0, 0], [0, 0]
    fitCfg['sigma'], fitCfg['sigma_cfg'] = [10, 50], [0, 0]
    fitCfg['norm'], fitCfg['norm_cfg'] = [0.95], [1]
    nGauss = len(fitCfg['mean'])
    
    if False: doFitMultiGauss(bkg, comp, fitCfg, label, outDir_fits, qTmax=200, rebin=rebin, xMin=-200, xMax=200, singleMean=True)
    
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


def do_ttbar_perp():

    bkg = "TTbar"
    tag = "ttbar_perp"
    comp = "perp"
    baseDir = "%s/%s" % (outDir, tag)
    name = "TTbar #rightarrow %s" % flavortag
    label = "%s, %s" % (name, met)
    functions.prepareDir(baseDir, False)
    rebin = 2

    ###### STEP 1: floating fits
    iteration = 0
    outDir_fits = "%s/fits_v0" % baseDir
    fitCfg = {}
    fitCfg['mean'], fitCfg['mean_cfg'] = [0, 0], [1, 1]
    fitCfg['sigma'], fitCfg['sigma_cfg'] = [20, 50], [0, 0]
    fitCfg['norm'], fitCfg['norm_cfg'] = [0.7], [1]
    nGauss = len(fitCfg['mean'])
    
    if False: doFitMultiGauss(bkg, comp, fitCfg, label, outDir_fits, qTmax=200, rebin=rebin, xMin=-200, xMax=200)
    
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


def do_ttbar_para():

    bkg = "TTbar"
    tag = "ttbar_para"
    comp = "para"
    baseDir = "%s/%s" % (outDir, tag)
    name = "TTbar #rightarrow %s" % flavortag
    label = "%s, %s" % (name, met)
    functions.prepareDir(baseDir, False)
    rebin = 2

    ###### STEP 1: floating fits
    iteration = 0
    outDir_fits = "%s/fits_v0" % baseDir
    fitCfg = {}
    fitCfg['mean'], fitCfg['mean_cfg'] = [0, 0], [0, 0]
    fitCfg['sigma'], fitCfg['sigma_cfg'] = [40, 70], [0, 0]
    fitCfg['norm'], fitCfg['norm_cfg'] = [0.5], [1]
    nGauss = len(fitCfg['mean'])
    
    if False: 
        doFitMultiGauss(bkg, comp, fitCfg, label, outDir_fits, qTmax=200, rebin=rebin, xMin=-200, xMax=200, singleMean=True)
        sys.exit()

    
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


def do_dymumu_perp():

    proc = "DYmumu"
    tag = "dymumu_perp"
    comp = "perp"
    baseDir = "%s/%s" % (outDir, tag)
    name = "DY #rightarrow %s" % flavortag
    label = "%s, %s" % (name, met)
    functions.prepareDir(baseDir, False)
    nGauss = 5
    rebin = 1

    ###### STEP 1: floating fits
    iteration = 0
    outDir_fits = "%s/fits_v0" % baseDir
    fitCfg = {}
    fitCfg['mean'], fitCfg['mean_cfg'] = [0, 0, 0, 0], [1, 1, 1, 1] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
    fitCfg['sigma'], fitCfg['sigma_cfg'] = [5, 8, 11, 20], [0, 0, 0, 0]
    fitCfg['norm'], fitCfg['norm_cfg'] = [0.22, 0.5, 0.25], [1, 1, 1]
    
    fitCfg['mean'], fitCfg['mean_cfg'] = [0, 0, 0, 0, 0], [0, 0, 0, 0, 0] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
    fitCfg['sigma'], fitCfg['sigma_cfg'] = [4, 6, 8, 11, 20], [0, 0, 0, 0, 0]
    fitCfg['norm'], fitCfg['norm_cfg'] = [0.1, 0.15, 0.53, 0.20], [1, 1, 1, 1]
    
    fitCfg['mean'], fitCfg['mean_cfg'] = [0, 0, 0, 0, 0,], [1, 1, 1, 1, 1] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
    fitCfg['sigma'], fitCfg['sigma_cfg'] = [4, 6, 8, 10, 15], [0, 0, 0, 0, 0]
    fitCfg['norm'], fitCfg['norm_cfg'] = [0.15, 0.15, 0.3, 0.25], [1, 1, 1, 1]

    nGauss = len(fitCfg['mean'])
    
    if False: doFitMultiGauss(proc, comp, fitCfg, label, outDir_fits, qTmax=200, rebin=rebin)

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


def do_dymumu_para_5Gauss():

    proc = "DYmumu"
    tag = "dymumu_para"
    comp = "para"
    baseDir = "%s/%s" % (outDir, tag)
    name = "DY #rightarrow %s" % flavortag
    label = "%s, %s" % (name, met)
    functions.prepareDir(baseDir, False)
    rebin = 1

    ###### STEP 1: floating fits
    iteration = 0
    outDir_fits = "%s/fits_v0" % baseDir
    fitCfg = {}
    fitCfg['mean'], fitCfg['mean_cfg'] = [0, 0, 0, 0, 0], [0, 0, 0, 0, 0] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
    fitCfg['sigma'], fitCfg['sigma_cfg'] = [4, 6, 8, 10, 15], [0, 0, 0, 0, 0]
    fitCfg['norm'], fitCfg['norm_cfg'] = [0.15, 0.15, 0.3, 0.25], [1, 1, 1, 1]
    nGauss = len(fitCfg['mean'])
    
    if True: 
        doFitMultiGauss(proc, comp, fitCfg, label, outDir_fits, qTmax=150, rebin=rebin, singleMean=True)
        sys.exit()

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


def do_dymumu_para():

    proc = "DYmumu"
    tag = "dymumu_para"
    #proc = "DYee"
    #tag = "dyee_para"
    comp = "para"
    baseDir = "%s/%s" % (outDir, tag)
    name = "DY #rightarrow %s" % flavortag
    label = "%s, %s" % (name, met)
    functions.prepareDir(baseDir, False)
    rebin = 1

    ###### STEP 1: floating fits
    outDir_fits = "%s/fits_v0" % baseDir
    fitCfg = {}
    #fitCfg['mean'], fitCfg['mean_cfg'] = [0, 0, 0, 0, 0], [0, 0, 0, 0, 0] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
    #fitCfg['sigma'], fitCfg['sigma_cfg'] = [4, 6, 8, 10, 15], [0, 0, 0, 0, 0]
    #fitCfg['norm'], fitCfg['norm_cfg'] = [0.15, 0.15, 0.3, 0.25], [1, 1, 1, 1]
    
    fitCfg['mean'], fitCfg['mean_cfg'] = [0, 0, 0], [0, 0, 0] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
    fitCfg['sigma'], fitCfg['sigma_cfg'] = [4, 6, 15], [0, 0, 0]
    fitCfg['norm'], fitCfg['norm_cfg'] = [0.70, 0.25], [0, 0]
    
    fitCfg['mean'], fitCfg['mean_cfg'] = [0, 0, 0], [0, 0, 0] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
    fitCfg['sigma'], fitCfg['sigma_cfg'] = [8, 12, 20], [0, 0, 0]
    fitCfg['norm'], fitCfg['norm_cfg'] = [0.4, 0.4], [0, 0]
    
    nGauss = len(fitCfg['mean'])
    
    if True: 
        doFitMultiGauss(proc, comp, fitCfg, label, outDir_fits, qTmax=80, rebin=rebin, singleMean=True)
        sys.exit()

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


def do_singlemuon_perp_old():

    proc = "SingleMuon"
    tag = "singlemuon_perp"
    comp = "perp"
    baseDir = "%s/%s" % (outDir, tag)
    name = "SingleMuon #rightarrow %s" % flavortag
    label = "%s, %s" % (name, met)
    functions.prepareDir(baseDir, False)
    rebin = 1

    ###### STEP 1: floating fits
    iteration = 0
    outDir_fits_v0 = "%s/fits_v0" % baseDir
    fitCfg = {}
    fitCfg['mean'], fitCfg['mean_cfg'] = [0, 0, 0], [1, 1, 1] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
    fitCfg['sigma'], fitCfg['sigma_cfg'] = [6, 10, 15], [0, 0, 0]
    fitCfg['norm'], fitCfg['norm_cfg'] = [0.25, 0.5], [1, 1]

    bkgCfg = {}
    bkgCfg['TTbar'] = { "filename": "/eos/user/j/jaeyserm/www/wmass/lowPU/recoilCorrectionParametric/recoil_mumu_RawPFMET/ttbar_perp/refit_v0/results.json", "norm": 1.0, "float": False }
    bkgCfg['EWK'] = { "filename": "/eos/user/j/jaeyserm/www/wmass/lowPU/recoilCorrectionParametric/recoil_mumu_RawPFMET/ewk_perp/refit_v0/results.json", "norm": 1.0, "float": False }
    nGauss = len(fitCfg['mean'])
    if False: 
        doFitMultiGauss(proc, comp, fitCfg, label, outDir_fits_v0, qTmax=200, rebin=rebin, bkgCfg=bkgCfg)
        sys.exit()
    
    
    ###### STEP 2: parameterize sigma1
    outDir_param_v0 = "%s/param_v0" % baseDir
    if False:
        jsOut = {}
        jsOut['nGauss'] = nGauss
        functions.prepareDir(outDir_param_v0, True)
        with open("%s/results.json" % outDir_fits_v0) as f: jsIn = json.load(f)
        
        fitF, params, cParams = "([0]*TMath::Power(x, [1]) + [2])", [7.26314e-01, 2.49241e-01, 4.41560e+00], [False, False, False]
        #fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2] + [3]*x + [4]*x*x", [7.26314e-01, 2.49241e-01, 0, 0, 0], [False, False, False, False, False]
        doFit(jsIn, jsOut, comp, "sigma1", fitF, params, outDir_param_v0, label, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=20, xMax=200, yTitle = "#sigma_{1} (GeV)")
        #sys.exit()
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [2.44857e-01, 5.19971e-01, 4.91301e+00], [False, False, False]
        doFit(jsIn, jsOut, comp, "sigma2", fitF, params, outDir_param_v0, label, cParams=cParams, fitMin=0, fitMax=120, yMin=0, yMax=20, yTitle = "#sigma_{2} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [1, 0, 10], [False, False, False]
        doFit(jsIn, jsOut, comp, "sigma3", fitF, params, outDir_param_v0, label, cParams=cParams, fitMin=0, fitMax=120, yMin=0, yMax=30, yTitle = "#sigma_{3} (GeV)")
        
        
        with open("%s/results.json" % outDir_param_v0, "w") as outfile: json.dump(jsOut, outfile, indent=4)
        sys.exit()
        
        
    #sys.exit()
    ###### STEP 3: fit with parameterized sigma1
    outDir_fits_v1 = "%s/fits_v1" % baseDir
    if False:
        fitCfg['mean_cfg'] = [1, 1, 1] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
        fitCfg['sigma_cfg'] = [2, 0, 0]
        fitCfg['norm_cfg'] = [1, 1]
        with open("%s/results.json" % outDir_param_v0) as f: jsIn = json.load(f)
        doFitMultiGauss(proc, comp, fitCfg, label, outDir_fits_v1, funcJs=jsIn, qTmax=200, rebin=rebin, bkgCfg=bkgCfg)    
        sys.exit()
        
    #sys.exit()
    ###### STEP 4: parameterize other parameters
    outDir_param_v1 = "%s/param_v1" % baseDir
    if True:
        jsOut = {}
        jsOut['nGauss'] = nGauss
        functions.prepareDir(outDir_param_v1, True)
        with open("%s/results.json" % outDir_fits_v1) as f: jsIn = json.load(f)
        
        fitF, params, cParams = "([0]*TMath::Power(x, [1]) + [2])", [1.50003e-03, 1.65757e+00, 5.37455e+00], [False, True, False]
        doFit(jsIn, jsOut, comp, "sigma1", fitF, params, outDir_param_v1, label, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=20, xMax=200, yTitle = "#sigma_{1} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [1.63388e+00, 2.12662e-01, 5.90292e+00], [False, True, False]
        doFit(jsIn, jsOut, comp, "sigma2", fitF, params, outDir_param_v1, label, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=20, xMax=200, yTitle = "#sigma_{2} (GeV)")
        
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [-8.86748e-03, 1., 1.23786e+01], [False, True, False]
        doFit(jsIn, jsOut, comp, "sigma3", fitF, params, outDir_param_v1, label, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=30, xMax=200, yTitle = "#sigma_{3} (GeV)")
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
        #sys.exit()
        
        
    #sys.exit()
    ###### STEP 5: refit
    if True:
        with open("%s/results.json" % outDir_param_v1) as f: jsIn = json.load(f)
        jsOut = combinedFit(proc, comp, jsIn, rebin=rebin, bkgCfg=bkgCfg, qTmax=150)
        with open("%s/results_refit.json" % outDir_param_v1, "w") as outfile: json.dump(jsOut, outfile, indent=4)
        #sys.exit()
        
    ###### STEP 6: final validation
    if True:
        fitCfg['mean_cfg'] = [1, 1, 1] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
        fitCfg['sigma_cfg'] = [2, 2, 2]
        fitCfg['norm_cfg'] = [1, 1]

        outDir_refit = "%s/refit_v1" % baseDir
        with open("%s/results_refit.json" % outDir_param_v1) as f: jsIn = json.load(f)
        #doFitMultiGauss(proc, comp, fitCfg, label, outDir_refit, funcJs=jsIn, qTmax=200, rebin=rebin, bkgCfg=bkgCfg)
 
        outDir_refit = "%s/param_v1_refit" % baseDir
        functions.prepareDir(outDir_refit)
        doPlot(comp, "sigma1", jsIn, outDir_refit, label, yMin=0, yMax=20, yTitle = "#sigma_{1} (GeV)")
        doPlot(comp, "sigma2", jsIn, outDir_refit, label, yMin=0, yMax=30, yTitle = "#sigma_{2} (GeV)")
        doPlot(comp, "sigma3", jsIn, outDir_refit, label, yMin=0, yMax=30, yTitle = "#sigma_{3} (GeV)")
 


def do_singlemuon_perp():

    proc = "SingleMuon"
    tag = "singlemuon_perp"
    comp = "perp"
    baseDir = "%s/%s" % (outDir, tag)
    name = "SingleMuon #rightarrow %s" % flavortag
    label = "%s, %s" % (name, met)
    functions.prepareDir(baseDir, False)
    rebin = 1

    ###### STEP 1: floating fits
    iteration = 0
    outDir_fits_v0 = "%s/fits_v0" % baseDir
    fitCfg = {}
    fitCfg['mean'], fitCfg['mean_cfg'] = [0, 0, 0], [1, 1, 1] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
    fitCfg['sigma'], fitCfg['sigma_cfg'] = [6, 10, 15], [0, 0, 0]
    fitCfg['norm'], fitCfg['norm_cfg'] = [0.3, 0.6], [1, 1]

    bkgCfg = {}
    bkgCfg['TTbar'] = { "filename": "/eos/user/j/jaeyserm/www/wmass/lowPU/recoilCorrectionParametric/recoil_mumu_RawPFMET/ttbar_perp/refit_v0/results.json", "norm": 1.0, "float": False }
    bkgCfg['EWK'] = { "filename": "/eos/user/j/jaeyserm/www/wmass/lowPU/recoilCorrectionParametric/recoil_mumu_RawPFMET/ewk_perp/refit_v0/results.json", "norm": 1.0, "float": False }
    nGauss = len(fitCfg['mean'])
    if False: 
        doFitMultiGauss(proc, comp, fitCfg, label, outDir_fits_v0, qTmax=200, rebin=rebin, bkgCfg=bkgCfg)
        sys.exit()
    
    
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
 



def do_singlemuon_para_old_old():

    proc = "SingleMuon"
    tag = "singlemuon_para"
    comp = "para"
    baseDir = "%s/%s" % (outDir, tag)
    name = "SingleMuon #rightarrow %s" % flavortag
    label = "%s, %s" % (name, met)
    functions.prepareDir(baseDir, False)
    rebin = 1

    ###### STEP 1: floating fits
    iteration = 0
    outDir_fits_v0 = "%s/fits_v0" % baseDir
    fitCfg = {}
    fitCfg['mean'], fitCfg['mean_cfg'] = [0, 0, 0], [0, 0, 0] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
    fitCfg['sigma'], fitCfg['sigma_cfg'] = [6, 10, 15], [0, 0, 0]
    fitCfg['norm'], fitCfg['norm_cfg'] = [0.3, 0.6], [1, 1]

    bkgCfg = {}
    bkgCfg['TTbar'] = { "filename": "/eos/user/j/jaeyserm/www/wmass/lowPU/recoilCorrectionParametric/recoil_mumu_RawPFMET/ttbar_para/refit_v0/results.json", "norm": 1.0, "float": False }
    bkgCfg['EWK'] = { "filename": "/eos/user/j/jaeyserm/www/wmass/lowPU/recoilCorrectionParametric/recoil_mumu_RawPFMET/ewk_para/refit_v0/results.json", "norm": 1.0, "float": False }
    nGauss = len(fitCfg['mean'])
    if False: doFitMultiGauss(proc, comp, fitCfg, label, outDir_fits_v0, qTmax=200, rebin=rebin, bkgCfg=bkgCfg, singleMean=True)

    
    
    ###### STEP 2: parameterize sigma1
    outDir_param_v0 = "%s/param_v0" % baseDir
    if True:
        jsOut = {}
        jsOut['nGauss'] = nGauss
        functions.prepareDir(outDir_param_v0, True)
        with open("%s/results.json" % outDir_fits_v0) as f: jsIn = json.load(f)

        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [2.44857e-01, 5.19971e-01, 4.91301e+00], [False, False, False]
        doFit(jsIn, jsOut, comp, "sigma1", fitF, params, outDir_param_v0, label, cParams=cParams, fitMin=0, fitMax=60, yMin=0, yMax=20, yTitle = "#sigma_{1} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [2.44857e-01, 5.19971e-01, 4.91301e+00], [False, False, False]
        doFit(jsIn, jsOut, comp, "sigma2", fitF, params, outDir_param_v0, label, cParams=cParams, fitMin=0, fitMax=80, yMin=0, yMax=20, yTitle = "#sigma_{2} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [2.44857e-01, 5.19971e-01, 4.91301e+00], [False, False, False]
        doFit(jsIn, jsOut, comp, "sigma3", fitF, params, outDir_param_v0, label, cParams=cParams, fitMin=0, fitMax=60, yMin=0, yMax=30, yTitle = "#sigma_{3} (GeV)")
        
        # mean
        #fitF, params, cParams = "(x<20)*1*([0] + [1]*x + [2]*x*x + [3]*x*x*x) +0+ (x>20)*1*([4] + [5]*x)", [2.15851e-01, 3.79250e-01, -7.94738e-03, 5.30835e-05, 3.46010e+00, 9.16460e-02], [False, False, False, False, False, False]
        #fitF, params, cParams = "(x<25)*1*([0] + [1]*x + [2]*x*x + [3]*x*x*x) +0+ (x>25)*1*(    ([0] + [1]*25 + [2]*25*25 + [3]*25*25*25 - [4]*25) + [4]*x)", [1.54013e-01, 4.20263e-01, -1.38828e-02, 2.80192e-04, 9.44310e-02], [False, False, False, False, False, False]
        doFit(jsIn, jsOut, comp, "mean1", fitF, params, outDir_param_v0, label, cParams=cParams, fitMin=0, fitMax=80, yMin=0, yMax=50, yTitle = "#mu_{1} (GeV)", doFit=True)
        doFit(jsIn, jsOut, comp, "mean2", fitF, params, outDir_param_v0, label, cParams=cParams, fitMin=0, fitMax=80, yMin=0, yMax=50, yTitle = "#mu_{2} (GeV)", doFit=True)
        doFit(jsIn, jsOut, comp, "mean3", fitF, params, outDir_param_v0, label, cParams=cParams, fitMin=0, fitMax=80, yMin=0, yMax=50, yTitle = "#mu_{3} (GeV)", doFit=True)


        # norm
        fitF, params, cParams = "[0]", [fitCfg['norm'][0]], [True]
        doFit(jsIn, jsOut, comp, "norm1", fitF, params, outDir_param_v0, label, cParams=cParams, fitMin=0, fitMax=50, yMin=0, yMax=1, yTitle = "n_{1} (GeV)", doFit=False)
        fitF, params, cParams = "[0]", [fitCfg['norm'][1]], [True]
        doFit(jsIn, jsOut, comp, "norm2", fitF, params, outDir_param_v0, label, cParams=cParams, fitMin=0, fitMax=50, yMin=0, yMax=1, yTitle = "n_{2} (GeV)", doFit=False)
        
        
        with open("%s/results.json" % outDir_param_v0, "w") as outfile: json.dump(jsOut, outfile, indent=4)

    ###### STEP 5: refit
    if True:
        with open("%s/results.json" % outDir_param_v0) as f: jsIn = json.load(f)
        jsOut = combinedFit(proc, comp, jsIn, rebin=rebin, bkgCfg=bkgCfg, singleMean=True)
        with open("%s/results_refit.json" % outDir_param_v0, "w") as outfile: json.dump(jsOut, outfile, indent=4)


    ###### STEP 6: final validation
    if True:
        fitCfg['mean_cfg'] = [2, 2, 2] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
        fitCfg['sigma_cfg'] = [2, 2, 2]
        fitCfg['norm_cfg'] = [1, 1]

        outDir_refit = "%s/refit_v0" % baseDir
        with open("%s/results_refit.json" % outDir_param_v0) as f: jsIn = json.load(f)
        #doFitMultiGauss(proc, comp, fitCfg, label, outDir_refit, funcJs=jsIn, qTmax=200, rebin=rebin, bkgCfg=bkgCfg, singleMean=True)
 
        outDir_refit = "%s/param_v0_refit" % baseDir
        functions.prepareDir(outDir_refit)
        doPlot(comp, "sigma1", jsIn, outDir_refit, label, yMin=0, yMax=20, yTitle = "#sigma_{1} (GeV)")
        doPlot(comp, "sigma2", jsIn, outDir_refit, label, yMin=0, yMax=30, yTitle = "#sigma_{2} (GeV)")
        doPlot(comp, "sigma3", jsIn, outDir_refit, label, yMin=0, yMax=30, yTitle = "#sigma_{3} (GeV)")
        doPlot(comp, "mean1", jsIn, outDir_refit, label, yMin=0, yMax=20, yTitle = "#mu_{1} (GeV)")
        doPlot(comp, "mean2", jsIn, outDir_refit, label, yMin=0, yMax=20, yTitle = "#mu_{2} (GeV)")
        doPlot(comp, "mean3", jsIn, outDir_refit, label, yMin=0, yMax=20, yTitle = "#mu_{3} (GeV)")

 
def do_singlemuon_para_old_old_old():

    proc = "SingleMuon"
    tag = "singlemuon_para"
    comp = "para"
    baseDir = "%s/%s" % (outDir, tag)
    name = "SingleMuon #rightarrow %s" % flavortag
    label = "%s, %s" % (name, met)
    functions.prepareDir(baseDir, False)
    rebin = 1

    ###### STEP 1: floating fits
    iteration = 0
    outDir_fits_v0 = "%s/fits_v0" % baseDir
    fitCfg = {}
    fitCfg['mean'], fitCfg['mean_cfg'] = [0, 0, 0], [0, 0, 0] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
    fitCfg['sigma'], fitCfg['sigma_cfg'] = [6, 10, 15], [0, 0, 0]
    fitCfg['norm'], fitCfg['norm_cfg'] = [0.3, 0.6], [1, 1]

    bkgCfg = {}
    bkgCfg['TTbar'] = { "filename": "/eos/user/j/jaeyserm/www/wmass/lowPU/recoilCorrectionParametric/recoil_mumu_RawPFMET/ttbar_para/refit_v0/results.json", "norm": 1.0, "float": False }
    bkgCfg['EWK'] = { "filename": "/eos/user/j/jaeyserm/www/wmass/lowPU/recoilCorrectionParametric/recoil_mumu_RawPFMET/ewk_para/refit_v0/results.json", "norm": 1.0, "float": False }
    nGauss = len(fitCfg['mean'])
    if False: 
        doFitMultiGauss(proc, comp, fitCfg, label, outDir_fits_v0, qTmax=200, rebin=rebin, bkgCfg=bkgCfg, singleMean=True)
        sys.exit()
    
    
    ###### STEP 2: parameterize mean
    outDir_param_v0 = "%s/param_v0" % baseDir
    if False:
        jsOut = {}
        jsOut['nGauss'] = nGauss
        functions.prepareDir(outDir_param_v0, True)
        with open("%s/results.json" % outDir_fits_v0) as f: jsIn = json.load(f)

        # mean
        #fitF, params, cParams = "(x<20)*1*([0] + [1]*x + [2]*x*x + [3]*x*x*x) +0+ (x>20)*1*([4] + [5]*x)", [2.15851e-01, 3.79250e-01, -7.94738e-03, 5.30835e-05, 3.46010e+00, 9.16460e-02], [False, False, False, False, False, False]
        fitF, params, cParams = "(x<30)*1*([0] + [1]*x + [2]*x*x + [3]*x*x*x) +0+ (x>30)*1*(    ([0] + [1]*30 + [2]*30*30 + [3]*30*30*30 - [4]*30) + [4]*x)", [1.54013e-01, 4.20263e-01, -1.38828e-02, 2.80192e-04, 9.44310e-02], [False, False, False, False, False, False]
        doFit(jsIn, jsOut, comp, "mean1", fitF, params, outDir_param_v0, label, cParams=cParams, fitMin=0, fitMax=80, yMin=0, yMax=50, yTitle = "#mu_{1} (GeV)", doFit=True)
        doFit(jsIn, jsOut, comp, "mean2", fitF, params, outDir_param_v0, label, cParams=cParams, fitMin=0, fitMax=80, yMin=0, yMax=50, yTitle = "#mu_{2} (GeV)", doFit=True)
        doFit(jsIn, jsOut, comp, "mean3", fitF, params, outDir_param_v0, label, cParams=cParams, fitMin=0, fitMax=80, yMin=0, yMax=50, yTitle = "#mu_{3} (GeV)", doFit=True)

        # norm
        fitF, params, cParams = "[0]", [fitCfg['norm'][0]], [True]
        doFit(jsIn, jsOut, comp, "norm1", fitF, params, outDir_param_v0, label, cParams=cParams, fitMin=0, fitMax=50, yMin=0, yMax=1, yTitle = "n_{1} (GeV)", doFit=False)
        fitF, params, cParams = "[0]", [fitCfg['norm'][1]], [True]
        doFit(jsIn, jsOut, comp, "norm2", fitF, params, outDir_param_v0, label, cParams=cParams, fitMin=0, fitMax=50, yMin=0, yMax=1, yTitle = "n_{2} (GeV)", doFit=False)

        with open("%s/results.json" % outDir_param_v0, "w") as outfile: json.dump(jsOut, outfile, indent=4)
        sys.exit()
        
    ###### STEP 3: refit with fixed means
    outDir_fits_v1 = "%s/fits_v1" % baseDir
    if False:
        fitCfg['mean_cfg'] = [2, 2, 2] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
        fitCfg['sigma_cfg'] = [0, 0, 0]
        fitCfg['norm_cfg'] = [1, 1]

        with open("%s/results.json" % outDir_param_v0) as f: jsIn = json.load(f)
        doFitMultiGauss(proc, comp, fitCfg, label, outDir_fits_v1, funcJs=jsIn, qTmax=200, rebin=rebin, bkgCfg=bkgCfg, singleMean=True)
        sys.exit()

    ###### STEP 4: parameterize sigma1
    outDir_param_v1 = "%s/param_v1" % baseDir
    if False:
        jsOut = {}
        jsOut['nGauss'] = nGauss
        functions.prepareDir(outDir_param_v1, True)
        with open("%s/results.json" % outDir_fits_v1) as f: jsIn = json.load(f)

        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [2.44857e-01, 5.19971e-01, 4.91301e+00], [False, False, False]
        doFit(jsIn, jsOut, comp, "sigma1", fitF, params, outDir_param_v1, label, cParams=cParams, fitMin=0, fitMax=80, yMin=0, yMax=20, yTitle = "#sigma_{1} (GeV)")
        
        #fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [2.44857e-01, 5.19971e-01, 4.91301e+00], [False, False, False]
        #doFit(jsIn, jsOut, comp, "sigma2", fitF, params, outDir_param_v1, label, cParams=cParams, fitMin=0, fitMax=80, yMin=0, yMax=20, yTitle = "#sigma_{2} (GeV)")
        
        #fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [2.44857e-01, 5.19971e-01, 4.91301e+00], [False, False, False]
        #doFit(jsIn, jsOut, comp, "sigma3", fitF, params, outDir_param_v1, label, cParams=cParams, fitMin=0, fitMax=60, yMin=0, yMax=30, yTitle = "#sigma_{3} (GeV)")
        
        # mean
        fitF, params, cParams = "(x<30)*1*([0] + [1]*x + [2]*x*x + [3]*x*x*x) +0+ (x>30)*1*(    ([0] + [1]*30 + [2]*30*30 + [3]*30*30*30 - [4]*30) + [4]*x)", [1.54013e-01, 4.20263e-01, -1.38828e-02, 2.80192e-04, 9.44310e-02], [False, False, False, False, False, False]
        doFit(jsIn, jsOut, comp, "mean1", fitF, params, outDir_param_v1, label, cParams=cParams, fitMin=0, fitMax=80, yMin=0, yMax=50, yTitle = "#mu_{1} (GeV)", doFit=True)
        doFit(jsIn, jsOut, comp, "mean2", fitF, params, outDir_param_v1, label, cParams=cParams, fitMin=0, fitMax=80, yMin=0, yMax=50, yTitle = "#mu_{2} (GeV)", doFit=True)
        doFit(jsIn, jsOut, comp, "mean3", fitF, params, outDir_param_v1, label, cParams=cParams, fitMin=0, fitMax=80, yMin=0, yMax=50, yTitle = "#mu_{3} (GeV)", doFit=True)


        # norm
        fitF, params, cParams = "[0]", [fitCfg['norm'][0]], [True]
        doFit(jsIn, jsOut, comp, "norm1", fitF, params, outDir_param_v1, label, cParams=cParams, fitMin=0, fitMax=50, yMin=0, yMax=1, yTitle = "n_{1} (GeV)", doFit=False)
        fitF, params, cParams = "[0]", [fitCfg['norm'][1]], [True]
        doFit(jsIn, jsOut, comp, "norm2", fitF, params, outDir_param_v1, label, cParams=cParams, fitMin=0, fitMax=50, yMin=0, yMax=1, yTitle = "n_{2} (GeV)", doFit=False)

        with open("%s/results.json" % outDir_param_v1, "w") as outfile: json.dump(jsOut, outfile, indent=4)
        sys.exit()
    
    ###### STEP 5: refit with fixed means and fixed sigma1
    outDir_fits_v2 = "%s/fits_v2" % baseDir
    if False:
        fitCfg['mean_cfg'] = [2, 2, 2] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
        fitCfg['sigma_cfg'] = [2, 0, 0]
        fitCfg['norm_cfg'] = [1, 1]

        with open("%s/results.json" % outDir_param_v1) as f: jsIn = json.load(f)
        doFitMultiGauss(proc, comp, fitCfg, label, outDir_fits_v2, funcJs=jsIn, qTmax=200, rebin=rebin, bkgCfg=bkgCfg, singleMean=True)
        sys.exit()
    
    ###### STEP 5: parameterize sigma2 and sigma3
    outDir_param_v2 = "%s/param_v2" % baseDir
    if False:
        jsOut = {}
        jsOut['nGauss'] = nGauss
        functions.prepareDir(outDir_param_v2, True)
        with open("%s/results.json" % outDir_fits_v2) as f: jsIn = json.load(f)

        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [1.20272e-01, 8.76573e-01, 5.00362e+00], [False, False, False]
        doFit(jsIn, jsOut, comp, "sigma1", fitF, params, outDir_param_v2, label, cParams=cParams, fitMin=0, fitMax=80, yMin=0, yMax=20, yTitle = "#sigma_{1} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [3.19657e-01, 6.80485e-01, 8.10558e+00], [False, False, False]
        doFit(jsIn, jsOut, comp, "sigma2", fitF, params, outDir_param_v2, label, cParams=cParams, fitMin=0, fitMax=80, yMin=0, yMax=20, yTitle = "#sigma_{2} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [-6.04973e-03, 1, 1.36443e+01], [False, True, False]
        doFit(jsIn, jsOut, comp, "sigma3", fitF, params, outDir_param_v2, label, cParams=cParams, fitMin=0, fitMax=80, yMin=0, yMax=30, yTitle = "#sigma_{3} (GeV)")
        
        # mean
        fitF, params, cParams = "(x<30)*1*([0] + [1]*x + [2]*x*x + [3]*x*x*x) +0+ (x>30)*1*(    ([0] + [1]*30 + [2]*30*30 + [3]*30*30*30 - [4]*30) + [4]*x)", [2.55524e-01, 3.66526e-01, -7.32212e-03, 5.92873e-05, 8.55411e-02], [False, False, False, False, False, False]
        doFit(jsIn, jsOut, comp, "mean1", fitF, params, outDir_param_v2, label, cParams=cParams, fitMin=0, fitMax=80, yMin=0, yMax=50, yTitle = "#mu_{1} (GeV)", doFit=True)
        doFit(jsIn, jsOut, comp, "mean2", fitF, params, outDir_param_v2, label, cParams=cParams, fitMin=0, fitMax=80, yMin=0, yMax=50, yTitle = "#mu_{2} (GeV)", doFit=True)
        doFit(jsIn, jsOut, comp, "mean3", fitF, params, outDir_param_v2, label, cParams=cParams, fitMin=0, fitMax=80, yMin=0, yMax=50, yTitle = "#mu_{3} (GeV)", doFit=True)
      

        # norm
        fitF, params, cParams = "[0]", [fitCfg['norm'][0]], [True]
        doFit(jsIn, jsOut, comp, "norm1", fitF, params, outDir_param_v2, label, cParams=cParams, fitMin=0, fitMax=50, yMin=0, yMax=1, yTitle = "n_{1} (GeV)", doFit=False)
        fitF, params, cParams = "[0]", [fitCfg['norm'][1]], [True]
        doFit(jsIn, jsOut, comp, "norm2", fitF, params, outDir_param_v2, label, cParams=cParams, fitMin=0, fitMax=50, yMin=0, yMax=1, yTitle = "n_{2} (GeV)", doFit=False)

        with open("%s/results.json" % outDir_param_v2, "w") as outfile: json.dump(jsOut, outfile, indent=4)
        #sys.exit()


    
    ###### STEP 5: refit
    if True:
        with open("%s/results.json" % outDir_param_v2) as f: jsIn = json.load(f)
        jsOut = combinedFit(proc, comp, jsIn, rebin=rebin, bkgCfg=bkgCfg, singleMean=True, qTmax=95) # 95 was good
        with open("%s/results_refit.json" % outDir_param_v2, "w") as outfile: json.dump(jsOut, outfile, indent=4)
        #sys.exit()
   
    ###### STEP 6: final validation
    if True:
        fitCfg['mean_cfg'] = [2, 2, 2] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
        fitCfg['sigma_cfg'] = [2, 2, 2]
        fitCfg['norm_cfg'] = [1, 1]

        outDir_refit = "%s/refit_v2" % baseDir
        with open("%s/results_refit.json" % outDir_param_v2) as f: jsIn = json.load(f)
        #doFitMultiGauss(proc, comp, fitCfg, label, outDir_refit, funcJs=jsIn, qTmax=200, rebin=rebin, bkgCfg=bkgCfg, singleMean=True)
 
        outDir_refit = "%s/param_v2_refit" % baseDir
        functions.prepareDir(outDir_refit)
        doPlot(comp, "sigma1", jsIn, outDir_refit, label, yMin=0, yMax=20, yTitle = "#sigma_{1} (GeV)")
        doPlot(comp, "sigma2", jsIn, outDir_refit, label, yMin=0, yMax=20, yTitle = "#sigma_{2} (GeV)")
        doPlot(comp, "sigma3", jsIn, outDir_refit, label, yMin=10, yMax=40, yTitle = "#sigma_{3} (GeV)")
        doPlot(comp, "mean1", jsIn, outDir_refit, label, yMin=0, yMax=20, yTitle = "#mu_{1} (GeV)")
        doPlot(comp, "mean2", jsIn, outDir_refit, label, yMin=0, yMax=20, yTitle = "#mu_{2} (GeV)")
        doPlot(comp, "mean3", jsIn, outDir_refit, label, yMin=0, yMax=20, yTitle = "#mu_{3} (GeV)")


def do_singlemuon_para():

    proc = "SingleMuon"
    tag = "singlemuon_para"
    comp = "para"
    baseDir = "%s/%s" % (outDir, tag)
    name = "SingleMuon #rightarrow %s" % flavortag
    label = "%s, %s" % (name, met)
    functions.prepareDir(baseDir, False)
    rebin = 1
    #rebin = [-300, -100, -80, -70, -60, -50, -46, -42, -38, -34] + list(range(-30, 30, 1)) + [30, 34, 38, 42, 46, 50, 60, 70, 80, 100, 300]


    ###### STEP 1: floating fits
    iteration = 0
    outDir_fits_v0 = "%s/fits_v0" % baseDir
    fitCfg = {}
    fitCfg['mean'], fitCfg['mean_cfg'] = [0, 0, 0], [0, 0, 0] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
    fitCfg['sigma'], fitCfg['sigma_cfg'] = [6, 10, 15], [0, 0, 0]
    fitCfg['norm'], fitCfg['norm_cfg'] = [0.3, 0.6], [1, 1]
    
    fitCfg['mean'], fitCfg['mean_cfg'] = [0, 0], [0, 0] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
    fitCfg['sigma'], fitCfg['sigma_cfg'] = [5, 15], [0, 0]
    fitCfg['norm'], fitCfg['norm_cfg'] = [0.8], [0]
    
    fitCfg['mean'], fitCfg['mean_cfg'] = [0, 0], [0, 0] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
    fitCfg['sigma'], fitCfg['sigma_cfg'] = [5, 15], [0, 0]
    fitCfg['norm'], fitCfg['norm_cfg'] = [0.8], [0]

    bkgCfg = {}
    bkgCfg['TTbar'] = { "filename": "/eos/user/j/jaeyserm/www/wmass/lowPU/recoilCorrectionParametric/recoil_mumu_RawPFMET/ttbar_para/refit_v0/results.json", "norm": 1.0, "float": False, "subtract": False }
    bkgCfg['EWK'] = { "filename": "/eos/user/j/jaeyserm/www/wmass/lowPU/recoilCorrectionParametric/recoil_mumu_RawPFMET/ewk_para/refit_v0/results.json", "norm": 1.0, "float": False, "subtract": False }
    nGauss = len(fitCfg['mean'])
    if True: 
        doFitMultiGauss(proc, comp, fitCfg, label, outDir_fits_v0, qTmax=80, rebin=rebin, bkgCfg=bkgCfg, singleMean=True)
        sys.exit()

    
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


 
def do_singlemuon_para_3Gauss():

    proc = "SingleMuon"
    tag = "singlemuon_para"
    comp = "para"
    baseDir = "%s/%s" % (outDir, tag)
    name = "SingleMuon #rightarrow %s" % flavortag
    label = "%s, %s" % (name, met)
    functions.prepareDir(baseDir, False)
    rebin = 1

    ###### STEP 1: floating fits
    outDir_fits_v0 = "%s/fits_v0" % baseDir
    fitCfg = {}

    fitCfg['mean'], fitCfg['mean_cfg'] = [0, 0, 0], [0, 0, 0] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
    fitCfg['sigma'], fitCfg['sigma_cfg'] = [5, 10, 15], [0, 0, 0]
    fitCfg['norm'], fitCfg['norm_cfg'] = [0.6, 0.35], [0, 1]

    bkgCfg = {}
    bkgCfg['TTbar'] = { "filename": "/eos/user/j/jaeyserm/www/wmass/lowPU/recoilCorrectionParametric/recoil_mumu_RawPFMET/ttbar_para/refit_v0/results.json", "norm": 1.0, "float": False }
    bkgCfg['EWK'] = { "filename": "/eos/user/j/jaeyserm/www/wmass/lowPU/recoilCorrectionParametric/recoil_mumu_RawPFMET/ewk_para/refit_v0/results.json", "norm": 1.0, "float": False }
    nGauss = len(fitCfg['mean'])
    if False: 
        doFitMultiGauss(proc, comp, fitCfg, label, outDir_fits_v0, qTmax=150, rebin=rebin, bkgCfg=bkgCfg, singleMean=True)
        sys.exit()
    
    
    ###### STEP 2: parameterize means
    outDir_param_v0 = "%s/param_v0" % baseDir
    if True:
        jsOut = {}
        jsOut['nGauss'] = nGauss
        functions.prepareDir(outDir_param_v0, True)
        with open("%s/results.json" % outDir_fits_v0) as f: jsIn = json.load(f)

        # mean
        fitF, params, cParams = "(x<30)*1*([0] + [1]*x + [2]*x*x + [3]*x*x*x) +0+ (x>30)*1*(    ([0] + [1]*30 + [2]*30*30 + [3]*30*30*30 - ([1] + 2*[2]*30 + 3*[3]*30*30)*30) + ([1] + 2*[2]*30 + 3*[3]*30*30)*x )", [0, 0, 0, 0], [False, False, False, False]
        doFit(jsIn, jsOut, comp, "mean1", fitF, params, outDir_param_v0, label, cParams=cParams, fitMin=0, fitMax=80, yMin=0, yMax=50, xMax=150, yTitle = "#mu_{2} (GeV)", doFit=True)
        doFit(jsIn, jsOut, comp, "mean2", fitF, params, outDir_param_v0, label, cParams=cParams, fitMin=0, fitMax=80, yMin=0, yMax=50, xMax=150, yTitle = "#mu_{2} (GeV)", doFit=True)
        doFit(jsIn, jsOut, comp, "mean3", fitF, params, outDir_param_v0, label, cParams=cParams, fitMin=0, fitMax=80, yMin=0, yMax=50, xMax=150, yTitle = "#mu_{3} (GeV)", doFit=True)

        # norm
        fitF, params, cParams = "[0]", [fitCfg['norm'][0]], [True]
        doFit(jsIn, jsOut, comp, "norm1", fitF, params, outDir_param_v0, label, cParams=cParams, fitMin=0, fitMax=50, yMin=0, yMax=1, yTitle = "n_{1} (GeV)", doFit=False)
        fitF, params, cParams = "[0]", [fitCfg['norm'][1]], [True]
        doFit(jsIn, jsOut, comp, "norm2", fitF, params, outDir_param_v0, label, cParams=cParams, fitMin=0, fitMax=50, yMin=0, yMax=1, yTitle = "n_{2} (GeV)", doFit=False)
        
        with open("%s/results.json" % outDir_param_v0, "w") as outfile: json.dump(jsOut, outfile, indent=4)
        sys.exit()
        
    ###### STEP 3: refit with fixed means
    outDir_fits_v1 = "%s/fits_v1" % baseDir
    if False:
        fitCfg['mean_cfg'] = [2, 2, 2] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
        fitCfg['sigma_cfg'] = [0, 0, 0]
        fitCfg['norm_cfg'] = [1, 1]

        with open("%s/results.json" % outDir_param_v0) as f: jsIn = json.load(f)
        doFitMultiGauss(proc, comp, fitCfg, label, outDir_fits_v1, funcJs=jsIn, qTmax=200, rebin=rebin, bkgCfg=bkgCfg, singleMean=True)
        sys.exit()

    ###### STEP 4: parameterize sigma1
    outDir_param_v1 = "%s/param_v1" % baseDir
    if False:
        jsOut = {}
        jsOut['nGauss'] = nGauss
        functions.prepareDir(outDir_param_v1, True)
        with open("%s/results.json" % outDir_fits_v1) as f: jsIn = json.load(f)

        #fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [2.44857e-01, 5.19971e-01, 4.91301e+00], [False, False, False]
        #doFit(jsIn, jsOut, comp, "sigma1", fitF, params, outDir_param_v1, label, cParams=cParams, fitMin=0, fitMax=100, yMin=0, yMax=30, xMax=200, yTitle = "#sigma_{1} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [2.44857e-01, 5.19971e-01, 4.91301e+00], [False, False, False]
        #fitF, params, cParams = "(x<40)*1*([0] + [1]*x + [2]*x*x) +0+ (x>40)*1*([3] + [4]*TMath::Sqrt(x))", [9.00849e+00, 1.11356e-01, 2.40663e-05, 1, 1], [False, False, False, False, False]
        #fitF, params, cParams = "[0] + [1]*x + [2]*x*x", [9.00849e+00, 1.11356e-01, -1.57611e-03], [False, False, False]
        #fitF, params, cParams = "[0]*x + [1]", [1, 1], [False, False]
        doFit(jsIn, jsOut, comp, "sigma2", fitF, params, outDir_param_v1, label, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=30, xMax=200, yTitle = "#sigma_{2} (GeV)")
        #sys.exit()
        #fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [2.52429e+00, 1.29461e-011, 1.05909e+01], [False, False, False]
        #doFit(jsIn, jsOut, comp, "sigma3", fitF, params, outDir_param_v1, label, cParams=cParams, fitMin=0, fitMax=100, yMin=0, yMax=30, xMax=200, yTitle = "#sigma_{3} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [1.20272e-01, 8.76573e-01, 5.00362e+00], [False, False, False]
        doFit(jsIn, jsOut, comp, "sigma1", fitF, params, outDir_param_v1, label, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=30, xMax=200, yTitle = "#sigma_{1} (GeV)")
        
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [1.20272e-01, 8.76573e-01, 5.00362e+00], [False, False, False]
        doFit(jsIn, jsOut, comp, "sigma3", fitF, params, outDir_param_v1, label, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=30, xMax=200, yTitle = "#sigma_{3} (GeV)")
        
        
        # mean
        fitF, params, cParams = "(x<30)*1*([0] + [1]*x + [2]*x*x + [3]*x*x*x) +0+ (x>30)*1*(    ([0] + [1]*30 + [2]*30*30 + [3]*30*30*30 - ([1] + 2*[2]*30 + 3*[3]*30*30 - 2*[4]*30)*30 - [4]*30*30) + ([1] + 2*[2]*30 + 3*[3]*30*30 - 2*[4]*30)*x + [4]*x*x  )", [2.58151e-01, 3.62633e-01, -6.89350e-03, 4.93698e-05, 4.94332e-05], [False, False, False, False, False]
        doFit(jsIn, jsOut, comp, "mean1", fitF, params, outDir_param_v1, label, cParams=cParams, fitMin=0, fitMax=110, yMin=0, yMax=50, xMax=200, yTitle = "#mu_{1} (GeV)", doFit=True)
        doFit(jsIn, jsOut, comp, "mean2", fitF, params, outDir_param_v1, label, cParams=cParams, fitMin=0, fitMax=110, yMin=0, yMax=50, xMax=200, yTitle = "#mu_{2} (GeV)", doFit=True)
        doFit(jsIn, jsOut, comp, "mean3", fitF, params, outDir_param_v1, label, cParams=cParams, fitMin=0, fitMax=110, yMin=0, yMax=50, xMax=200, yTitle = "#mu_{3} (GeV)", doFit=True)


        # norm
        fitF, params, cParams = "[0]", [fitCfg['norm'][0]], [True]
        doFit(jsIn, jsOut, comp, "norm1", fitF, params, outDir_param_v1, label, cParams=cParams, fitMin=0, fitMax=50, yMin=0, yMax=1, yTitle = "n_{1} (GeV)", doFit=False)
        fitF, params, cParams = "[0]", [fitCfg['norm'][1]], [True]
        doFit(jsIn, jsOut, comp, "norm2", fitF, params, outDir_param_v1, label, cParams=cParams, fitMin=0, fitMax=50, yMin=0, yMax=1, yTitle = "n_{2} (GeV)", doFit=False)

        with open("%s/results.json" % outDir_param_v1, "w") as outfile: json.dump(jsOut, outfile, indent=4)
        sys.exit(0)

    ###### STEP 5: refit with fixed means and fixed sigma1
    outDir_fits_v2 = "%s/fits_v2" % baseDir
    if False:
        fitCfg['mean_cfg'] = [2, 2, 2] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
        fitCfg['sigma_cfg'] = [2, 0, 0]
        fitCfg['norm_cfg'] = [1, 1]

        with open("%s/results.json" % outDir_param_v1) as f: jsIn = json.load(f)
        doFitMultiGauss(proc, comp, fitCfg, label, outDir_fits_v2, funcJs=jsIn, qTmax=200, rebin=rebin, bkgCfg=bkgCfg, singleMean=True)
        sys.exit()
    
    ###### STEP 6: parameterize sigma3
    outDir_param_v2 = "%s/param_v2" % baseDir
    if False:
        jsOut = {}
        jsOut['nGauss'] = nGauss
        functions.prepareDir(outDir_param_v2, True)
        with open("%s/results.json" % outDir_fits_v2) as f: jsIn = json.load(f)

        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [1.20272e-01, 8.76573e-01, 5.00362e+00], [False, False, False]
        doFit(jsIn, jsOut, comp, "sigma1", fitF, params, outDir_param_v2, label, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=30, xMax=200, yTitle = "#sigma_{1} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [2.44857e-01, 5.19971e-01, 4.91301e+00], [False, False, False]
        doFit(jsIn, jsOut, comp, "sigma2", fitF, params, outDir_param_v2, label, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=30, xMax=200, yTitle = "#sigma_{2} (GeV)")
        
        #fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [-6.04973e-03, 1, 1.36443e+01], [False, True, False]
        #fitF, params, cParams = "[0] + [1]*x + [2]*x*x", [1, 1, 1], [False, False, False]
        #doFit(jsIn, jsOut, comp, "sigma3", fitF, params, outDir_param_v2, label, cParams=cParams, fitMin=0, fitMax=80, yMin=0, yMax=30, xMax=200, yTitle = "#sigma_{3} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [1.20272e-01, 8.76573e-01, 5.00362e+00], [False, False, False]
        doFit(jsIn, jsOut, comp, "sigma3", fitF, params, outDir_param_v2, label, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=30, xMax=200, yTitle = "#sigma_{3} (GeV)")
        
        # mean
        fitF, params, cParams = "(x<30)*1*([0] + [1]*x + [2]*x*x + [3]*x*x*x) +0+ (x>30)*1*(    ([0] + [1]*30 + [2]*30*30 + [3]*30*30*30 - ([1] + 2*[2]*30 + 3*[3]*30*30 - 2*[4]*30)*30 - [4]*30*30) + ([1] + 2*[2]*30 + 3*[3]*30*30 - 2*[4]*30)*x + [4]*x*x  )", [2.58151e-01, 3.62633e-01, -6.89350e-03, 4.93698e-05, 4.94332e-05], [False, False, False, False, False]
        doFit(jsIn, jsOut, comp, "mean1", fitF, params, outDir_param_v2, label, cParams=cParams, fitMin=0, fitMax=80, yMin=0, yMax=50, yTitle = "#mu_{1} (GeV)", doFit=True)
        doFit(jsIn, jsOut, comp, "mean2", fitF, params, outDir_param_v2, label, cParams=cParams, fitMin=0, fitMax=80, yMin=0, yMax=50, yTitle = "#mu_{2} (GeV)", doFit=True)
        doFit(jsIn, jsOut, comp, "mean3", fitF, params, outDir_param_v2, label, cParams=cParams, fitMin=0, fitMax=80, yMin=0, yMax=50, yTitle = "#mu_{3} (GeV)", doFit=True)
      

        # norm
        fitF, params, cParams = "[0]", [fitCfg['norm'][0]], [True]
        doFit(jsIn, jsOut, comp, "norm1", fitF, params, outDir_param_v2, label, cParams=cParams, fitMin=0, fitMax=50, yMin=0, yMax=1, yTitle = "n_{1} (GeV)", doFit=False)
        fitF, params, cParams = "[0]", [fitCfg['norm'][1]], [True]
        doFit(jsIn, jsOut, comp, "norm2", fitF, params, outDir_param_v2, label, cParams=cParams, fitMin=0, fitMax=50, yMin=0, yMax=1, yTitle = "n_{2} (GeV)", doFit=False)

        with open("%s/results.json" % outDir_param_v2, "w") as outfile: json.dump(jsOut, outfile, indent=4)
        sys.exit()

    ###### STEP 7: refit with sigma3
    outDir_fits_v3 = "%s/fits_v3" % baseDir
    if False:
        fitCfg['mean_cfg'] = [2, 2, 2] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
        fitCfg['sigma_cfg'] = [2, 0, 2]
        fitCfg['norm_cfg'] = [1, 1]

        with open("%s/results.json" % outDir_param_v2) as f: jsIn = json.load(f)
        doFitMultiGauss(proc, comp, fitCfg, label, outDir_fits_v3, funcJs=jsIn, qTmax=200, rebin=rebin, bkgCfg=bkgCfg, singleMean=True)
        sys.exit()
        
        
    ###### STEP 8: parameterize sigma3
    outDir_param_v3 = "%s/param_v3" % baseDir
    if True:
        jsOut = {}
        jsOut['nGauss'] = nGauss
        functions.prepareDir(outDir_param_v3, True)
        with open("%s/results.json" % outDir_fits_v3) as f: jsIn = json.load(f)

        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [1.48307e-01, 8.45809e-01, 5.31522e+00], [False, False, True]
        doFit(jsIn, jsOut, comp, "sigma1", fitF, params, outDir_param_v3, label, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=30, xMax=200, yTitle = "#sigma_{1} (GeV)")
        
        #fitF, params, cParams = "([0]*TMath::Power(x, [1]) + [2])*TMath::Exp(x*[3])", [1.62086e-01, 8.49236e-01, 7.23931e+00, 0], [False, False, False, False]
        fitF, params, cParams = "([0]*TMath::Power(x, [1]) + [2])", [1.67172e-01, 8.38949e-01, 7.26492e+00], [False, False, True]
        #fitF, params, cParams = "([0]*TMath::Power(x, [1]) + [2])*TMath::Exp(x*[3])", [1.67172e-01, 8.38949e-01, 7.26492e+00, 0], [False, False, False, False]
        doFit(jsIn, jsOut, comp, "sigma2", fitF, params, outDir_param_v3, label, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=30, xMax=200, yTitle = "#sigma_{2} (GeV)")
        
        fitF, params, cParams = "[0]*TMath::Power(x, [1]) + [2]", [8.16404e-01, 4.13643e-01, 9.87572e+00], [False, False, False]
        doFit(jsIn, jsOut, comp, "sigma3", fitF, params, outDir_param_v3, label, cParams=cParams, fitMin=0, fitMax=150, yMin=0, yMax=30, xMax=200, yTitle = "#sigma_{3} (GeV)")
        
        
        # mean
        fitF, params, cParams = "(x<30)*1*([0] + [1]*x + [2]*x*x + [3]*x*x*x) +0+ (x>30)*1*(    ([0] + [1]*30 + [2]*30*30 + [3]*30*30*30 - ([1] + 2*[2]*30 + 3*[3]*30*30 - 2*[4]*30)*30 - [4]*30*30) + ([1] + 2*[2]*30 + 3*[3]*30*30 - 2*[4]*30)*x + [4]*x*x  )", [2.58151e-01, 3.62633e-01, -6.89350e-03, 4.93698e-05, 4.94332e-05], [False, False, False, False, False]
        fitF, params, cParams = "(x<30)*1*([0] + [1]*x + [2]*x*x + [3]*x*x*x) +0+ (x>30)*1*(    ([0] + [1]*30 + [2]*30*30 + [3]*30*30*30 - ([1] + 2*[2]*30 + 3*[3]*30*30 - 2*[4]*30)*30 - [4]*30*30) + ([1] + 2*[2]*30 + 3*[3]*30*30 - 2*[4]*30)*x + [4]*x*x  )", [2.58151e-01, 3.62633e-01, -6.89350e-03, 4.93698e-05, 0], [True, False, False, False, True]
        doFit(jsIn, jsOut, comp, "mean1", fitF, params, outDir_param_v3, label, cParams=cParams, fitMin=0, fitMax=80, yMin=0, yMax=50, yTitle = "#mu_{1} (GeV)", doFit=True)
        doFit(jsIn, jsOut, comp, "mean2", fitF, params, outDir_param_v3, label, cParams=cParams, fitMin=0, fitMax=80, yMin=0, yMax=50, yTitle = "#mu_{2} (GeV)", doFit=True)
        doFit(jsIn, jsOut, comp, "mean3", fitF, params, outDir_param_v3, label, cParams=cParams, fitMin=0, fitMax=80, yMin=0, yMax=50, yTitle = "#mu_{3} (GeV)", doFit=True)


        # norm
        fitF, params, cParams = "[0]", [fitCfg['norm'][0]], [True]
        doFit(jsIn, jsOut, comp, "norm1", fitF, params, outDir_param_v3, label, cParams=cParams, fitMin=0, fitMax=50, yMin=0, yMax=1, yTitle = "n_{1} (GeV)", doFit=False)
        fitF, params, cParams = "[0]", [fitCfg['norm'][1]], [True]
        doFit(jsIn, jsOut, comp, "norm2", fitF, params, outDir_param_v3, label, cParams=cParams, fitMin=0, fitMax=50, yMin=0, yMax=1, yTitle = "n_{2} (GeV)", doFit=False)

        with open("%s/results.json" % outDir_param_v3, "w") as outfile: json.dump(jsOut, outfile, indent=4)
        #sys.exit()
    
    ###### STEP 9: refit
    if True:
        with open("%s/results.json" % outDir_param_v3) as f: jsIn = json.load(f)
        jsOut = combinedFit(proc, comp, jsIn, rebin=rebin, bkgCfg=bkgCfg, singleMean=True, qTmax=120)
        with open("%s/results_refit.json" % outDir_param_v3, "w") as outfile: json.dump(jsOut, outfile, indent=4)
        #sys.exit()
   
    ###### STEP 6: final validation
    if True:
        fitCfg['mean_cfg'] = [2, 2, 2] # 0 = floating, 1 = fixed, 2 = parametric (fixed), 3 = parametric (floating)
        fitCfg['sigma_cfg'] = [2, 2, 2]
        fitCfg['norm_cfg'] = [1, 1]

        outDir_refit = "%s/refit_v3" % baseDir
        with open("%s/results_refit.json" % outDir_param_v3) as f: jsIn = json.load(f)
        #doFitMultiGauss(proc, comp, fitCfg, label, outDir_refit, funcJs=jsIn, qTmax=200, rebin=rebin, bkgCfg=bkgCfg, singleMean=True)
 
        outDir_refit = "%s/param_v3_refit" % baseDir
        functions.prepareDir(outDir_refit)
        doPlot(comp, "sigma1", jsIn, outDir_refit, label, yMin=0, yMax=30, yTitle = "#sigma_{1} (GeV)")
        doPlot(comp, "sigma2", jsIn, outDir_refit, label, yMin=0, yMax=30, yTitle = "#sigma_{2} (GeV)")
        doPlot(comp, "sigma3", jsIn, outDir_refit, label, yMin=0, yMax=30, yTitle = "#sigma_{3} (GeV)")
        doPlot(comp, "mean1", jsIn, outDir_refit, label, yMin=0, yMax=20, yTitle = "#mu_{1} (GeV)")
        doPlot(comp, "mean2", jsIn, outDir_refit, label, yMin=0, yMax=20, yTitle = "#mu_{2} (GeV)")
        doPlot(comp, "mean3", jsIn, outDir_refit, label, yMin=0, yMax=20, yTitle = "#mu_{3} (GeV)")
     
if __name__ == "__main__":

    met = "RawPFMET" # RawPFMET DeepMETReso
    flavor = "mumu" # mu, e, mumu, ee

    if flavor == "mumu":
        flavortag = "#mu^{+}#mu^{#minus}"
        label = "DY #rightarrow #mu^{+}#mu^{#minus}, %s" % met
        sig = "DYmumu"
        data = "SingleMuon"
    if flavor == "ee": 
        flavortag = "e^{+}e^{#minus}"
        label = "DY #rightarrow e^{+}e^{#minus}, %s" % met
        sig = "DYee"
        data = "SingleElectron"
    if flavor == "mu": 
        label = "W #rightarrow #mu^{#pm}, %s" % met
        sig = "WjetsMuNu"
        
    ####################################################################
    outDir = "/eos/user/j/jaeyserm/www/wmass/lowPU/recoilCorrectionParametric/recoil_%s_%s/" % (flavor, met)
    fIn_ = "wremnants/data/lowPU/recoil_%s_%s.root" % (flavor, met)
    fOut_ = "wremnants/data/lowPU/recoil_param_%s_%s.root" % (flavor, met)
    
    bkgs = ["EWK", "TTbar"]
    bkg_constr = 1.1

    
    functions.prepareDir(outDir, False)

    cfg = {

        'logy'              : True,
        'logx'              : False,
    
        'xmin'              : -300,
        'xmax'              : 100,
        'ymin'              : 1e-1,
        'ymax'              : 1,
        
        'xtitle'            : "Recoil U_{#parallel} (GeV)",
        'ytitle'            : "Events / 1 GeV",
        
        'topRight'          : "199 pb^{#minus1} (13 TeV)", 
        'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",
        
        'ratiofraction'     : 0.25,
        'ytitleR'           : "Pull",
        'yminR'             : -2.5,
        'ymaxR'             : 2.5,
    }   
    

   
    recoil_qTbins = list(range(0, 30, 1)) + list(range(30, 50, 2)) + list(range(50, 100, 5)) + list(range(100, 150, 10)) + [150, 175, 200, 10000]
    recoil_qTbins = list(range(0, 30, 1)) + list(range(30, 50, 2)) + list(range(50, 70, 5)) + list(range(70, 150, 10)) + [150, 200, 300, 10000]
    recoil_qTbins = list(functions.drange(0, 30, 0.5)) + list(range(30, 50, 1)) + list(range(50, 70, 2)) + list(range(70, 100, 5)) + list(range(100, 150, 10)) + [150, 200, 300, 10000]
    recoil_qTbins = list(range(0, 50, 1)) + list(range(50, 70, 2)) + list(range(70, 100, 5)) + list(range(100, 150, 10)) + [150, 200, 300, 10000]
    #recoil_qTbins = [0, 0.5, 1, 1.5] + list(range(2, 50, 1)) + list(range(50, 70, 2)) + list(range(70, 100, 5)) + list(range(100, 150, 10)) + [150, 200, 300, 10000]
    
    recoil_qTbins = list(range(0, 300, 1))
    #recoil_qTbins = list(functions.drange(0, 300, 0.5))
    #recoil_qTbins = list(range(0, 50, 1)) + list(range(50, 70, 2)) + list(range(70, 100, 5)) + list(range(100, 150, 10)) + [150, 200, 300, 10000]
    #recoil_qTbins = list(range(0, 20, 1)) + list(range(20, 40, 2)) + list(range(40, 55, 3)) + list(range(55, 80, 5)) + list(range(80, 100, 10)) + [100, 125, 150, 10000]
    #recoil_qTbins = list(range(0, 200, 1))
    #sys.exit()
    
    

    prepareFile(fIn_)
    #sys.exit()
    #print(fIn_)
    #sys.exit()
    
    
    
    #doFit_ewk_perp()
    #doFit_ewk_perp_param()
    
    
    #do_ttbar()

    #do_dymumu_perp()
    do_dymumu_para()
    
    #do_ttbar_perp()
    #do_ttbar_para()
    
    #do_ewk_perp()
    #do_ewk_para()
    
    #do_singlemuon_perp()
    #do_singlemuon_para()
    
    #doFit_data_perp()
    #doFit_data_perp_param()
    #doFit_data_perp_autoFit()
    
    #doFit_mc_gen_para()
    #doFit_mc_gen_perp()
    #doFit_mc_para()
    #doFit_mc_perp()
    #doFit_mc_perp_param()
    #doFit_mc_perp_autoFit()
    sys.exit()
    #doFit_data_perp()
    #doFit_data_perp_2gauss()
    #doFit_data_para()
    #doFit_data_para_2gauss()
    
    ###os.system("hadd -f %s %s %s %s %s" % (fOut_, fOut_.replace(".root", "_data_para.root"), fOut_.replace(".root", "_data_perp.root"), fOut_.replace(".root", "_mc_para.root"), fOut_.replace(".root", "_mc_perp.root")))
    