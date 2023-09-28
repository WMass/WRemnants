
import sys,array,math,os,copy,decimal
import numpy as np

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


def drange(x, y, jump):
    while x < y:
        yield float(x)
        x += decimal.Decimal(jump)
        
def getParamIdx(name, params):
    
    for i in range(0, len(params)): 
       
        if name == params[i].GetName(): return i
            
    return -1
    


def diagonalize(fitRes):

    cov = fitRes.covarianceMatrix() # covariance matrix: diagonal elements are the variances of each of the parameters
    cov.Print()
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






def doFit_perp(tag = "gen_perp"):
    
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
    if "plus" in tag: 
        proc = procPlus
        label_ = label.replace("#pm", "#plus")
    elif "minus" in tag: 
        proc = procMinus
        label_ = label.replace("#pm", "#minus")
    else: 
        proc = sig
        label_ = label

    for qTbin in range(1, len(recoil_qTbins)):
        
        qT = 0.5*(recoil_qTbins[qTbin-1] + recoil_qTbins[qTbin]) if qTbin < len(recoil_qTbins)-1 else (recoil_qTbins[qTbin-1] + 20)
        qTmin, qTmax = recoil_qTbins[qTbin-1], recoil_qTbins[qTbin]
        hName = "%s_perp_bin%d" % (proc, qTbin)
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
        latex.DrawLatex(0.20, 0.85, "%s, GEN" % label_)
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
        '''
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
        '''

    fOut = ROOT.TFile(fOut_.replace(".root", "_%s.root" % tag), "RECREATE")
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
    latex.DrawLatex(0.20, 0.90, "%s, GEN" % label_)
    latex.DrawLatex(0.20, 0.85, "U_{#perp}")
    
    canvas.Modify()
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs("%s/%s/chi2.png" % (outDir, tag))
    canvas.SaveAs("%s/%s/chi2.pdf" % (outDir, tag))
    canvas.Delete()





def doFit_para(tag = "gen_para"):

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
    if "plus" in tag: 
        proc = procPlus
        label_ = label.replace("#pm", "#plus")
    elif "minus" in tag: 
        proc = procMinus
        label_ = label.replace("#pm", "#minus")
    else: 
        proc = sig
        label_ = label
        
    for qTbin in range(1, len(recoil_qTbins)):

        qT = 0.5*(recoil_qTbins[qTbin-1] + recoil_qTbins[qTbin]) if qTbin < len(recoil_qTbins)-1 else (recoil_qTbins[qTbin-1] + 20)
        qTmin, qTmax = recoil_qTbins[qTbin-1], recoil_qTbins[qTbin]
        hName = "%s_perp_bin%d" % (proc, qTbin)
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
        latex.DrawLatex(0.20, 0.85, "%s, GEN" % label_)
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
        '''
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
        '''

    fOut = ROOT.TFile(fOut_.replace(".root", "_%s.root" % tag), "RECREATE")
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
    latex.DrawLatex(0.20, 0.90, "%s, GEN" % label_)
    latex.DrawLatex(0.20, 0.85, "U_{#parallel} + q_{T} ")
    
    canvas.Modify()
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs("%s/%s/chi2.png" % (outDir, tag))
    canvas.SaveAs("%s/%s/chi2.pdf" % (outDir, tag))
    canvas.Delete()




def readProc(groups_mumu, hName, procName):

    label_ = "%s_%s" % (hName, procName) 
    groups_mumu.setHists(hName, "", label=label_, procsToRead=[procName], selectSignal=False)
    bhist = groups_mumu.groups[procName][label_]
    return bhist 

def prepareFile(fOut_):
    
    datagroups = Datagroups("lowPU_%s_%s.pkl.lz4" % (flavor, met))
    fOut = ROOT.TFile(fOut_, "RECREATE")

    if flavor == "mumu": procs =[ "DYmumu"]
    if flavor == "ee": procs = ["DYee"]
    if flavor == "mu": procs = ["WplusJetsToMuNu", "WminusJetsToMuNu"]
    if flavor == "e": procs = ["WplusJetsToENu", "WminusJetsToENu"]
    
    for proc in procs:
    
        bhist_para_qT = readProc(datagroups, "recoil_corr_xy_para_qT_qTbinned_gen", proc)
        bhist_perp = readProc(datagroups, "recoil_corr_xy_perp_qTbinned_gen", proc)
        rhist_para_qT = narf.hist_to_root(bhist_para_qT)
        rhist_perp = narf.hist_to_root(bhist_perp)
        for iBin in range(1, rhist_perp.GetNbinsX()+1):
                
            hist_para = rhist_para_qT.ProjectionY("%s_para_bin%d" % (proc, iBin), iBin, iBin)
            hist_para.Write()
                
            hist_perp = rhist_perp.ProjectionY("%s_perp_bin%d" % (proc, iBin), iBin, iBin)
            hist_perp.Write()
        
    fOut.ls()
    fOut.Close()
  

  
if __name__ == "__main__":

    met = "RawPFMET" # RawPFMET DeepMETReso
    flavor = "mumu" # mu, e, mumu, ee
    label = ""
    
    if flavor == "mumu": 
        label = "DY #rightarrow #mu^{+}#mu^{#minus}, %s" % met
        sig = "DYmumu"
        data = "SingleMuon"
    if flavor == "ee": 
        label = "DY #rightarrow e^{+}e^{#minus}, %s" % met
        sig = "DYee"
        data = "SingleElectron"
    if flavor == "mu": 
        label = "W #rightarrow #mu^{#pm}#nu, %s" % met
        procPlus = "WplusJetsToMuNu"
        procMinus = "WminusJetsToMuNu"
        
    ####################################################################
    outDir = "/eos/user/j/jaeyserm/www/wmass/lowPU/recoilCorrection/recoil_%s_%s/" % (flavor, met)
    fIn_ = "wremnants/data/lowPU/recoil/%s_%s_gen.root" % (flavor, met)
    fOut_ = "wremnants/data/lowPU/recoil/param_%s_%s_gen.root" % (flavor, met)


    
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
    

    recoil_qTbins = list(range(0, 50, 1)) + list(range(50, 70, 2)) + list(range(70, 100, 5)) + list(range(100, 150, 10)) + [150, 200, 300, 10000]
    print(recoil_qTbins)
    #sys.exit()
    
    

    #prepareFile(fIn_)
    #sys.exit()
    
    fIn = ROOT.TFile(fIn_)
    

    if flavor == "mu" or flavor == "e":
        doFit_perp(tag = "gen_plus_perp")
        doFit_perp(tag = "gen_minus_perp")
        doFit_para("gen_plus_para")
        doFit_para("gen_minus_para")
        os.system("hadd -f %s %s %s %s %s" % (fOut_, fOut_.replace(".root", "_gen_plus_perp.root"), fOut_.replace(".root", "_gen_minus_perp.root"), fOut_.replace(".root", "_gen_plus_para.root"), fOut_.replace(".root", "_gen_minus_para.root")))
    
    if flavor == "mumu" or flavor == "ee":
        doFit_perp(tag = "gen_perp")
        doFit_para("gen_para")
        os.system("hadd -f %s %s %s" % (fOut_, fOut_.replace(".root", "_gen_perp.root"), fOut_.replace(".root", "_gen_para.root")))
        
