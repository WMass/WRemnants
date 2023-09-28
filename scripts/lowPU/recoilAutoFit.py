
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

mirror = False

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

def mirrorx(h):

    hnew = h.Clone("%s_mirrorx" % h.GetName())
    for i in range(1, hnew.GetNbinsX()+1):
        idx = hnew.GetNbinsX()-i+1
        hnew.SetBinContent(i, h.GetBinContent(idx))
        hnew.SetBinError(i, h.GetBinError(idx))

    #sys.exit()
    return hnew



def doFitMC_3Gauss(tag, name, hOut, qTbin, qTmin, qTmax, mean1_, mean2_, mean3_, norm1_, norm2_, sigma1_, sigma2_, sigma3_):

    hist = fIn.Get(name)
    hist.Rebin(1)
    
    if qTmin < 30: xMin,xMax = -100, 100
    elif qTmin < 60: xMin,xMax = -150, 50
    elif qTmin < 150: xMin,xMax = -200, 50
    else: xMin,xMax = -500, 0
    if "perp" in name: xMin,xMax = -100, 100
    if "_qT" in name: xMin,xMax = -100, 100
    xMin,xMax = -100, 100
    
    if mirror:
        hist = mirrorx(hist)
        xMax = -xMax
        xMin = -xMin
        tmp = xMax
        xMax = xMin
        xMin = tmp
    
    #hist = hist.Rebin(1)
    norm = hist.Integral()
    hist.Scale(1./norm)
    
    qT = 0.5*(qTmin + qTmax)
    cfg['xmin'], cfg['xmax'] = xMin, xMax
    cfg['ymax'] = 1.75*hist.GetMaximum()
    cfg['ymax'] = 1e3
    cfg['ymin'] = 1e-5


    recoil = ROOT.RooRealVar("recoil", "Recoil parallel (GeV)", 0, -500, 500)   
    
    rdh = ROOT.RooDataHist("rdh", "", ROOT.RooArgList(recoil), ROOT.RooFit.Import(hist))
    
    if "para" in name:
    
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
        gauss2 = ROOT.RooGaussian("gauss2", "gauss2", recoil, mean2, sigma2)
        gauss3 = ROOT.RooGaussian("gauss3", "gauss3", recoil, mean3, sigma3)
        pdf = ROOT.RooAddPdf("pdf", '', ROOT.RooArgList(gauss1, gauss2, gauss3), ROOT.RooArgList(norm1, norm2))
        
        norm1.setConstant(ROOT.kTRUE)
        norm2.setConstant(ROOT.kTRUE)    
        #mean3.setConstant(ROOT.kTRUE)
        
        fitRes = pdf.fitTo(rdh, ROOT.RooFit.SumW2Error(ROOT.kTRUE), ROOT.RooFit.Save(ROOT.kTRUE)) # ROOT.RooFit.Extended(ROOT.kTRUE), 
    
        #mean3.setVal(mean2.getVal())

    if "perp" in name:
    
        
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
        #mean2.setConstant(ROOT.kTRUE)
        #mean3.setConstant(ROOT.kTRUE)

        fitRes = pdf.fitTo(rdh, ROOT.RooFit.SumW2Error(ROOT.kTRUE), ROOT.RooFit.Save(ROOT.kTRUE)) # ROOT.RooFit.Extended(ROOT.kTRUE), 
    
        #mean1.setVal(0)
        #mean1.setConstant(ROOT.kTRUE)
        #mean2.setVal(0)
        #mean2.setConstant(ROOT.kTRUE)
        #mean3.setVal(0)
        #mean3.setConstant(ROOT.kTRUE)
        

        mean2.setVal(mean1.getVal())
        mean3.setVal(mean1.getVal())    

    params = [mean1, mean2, mean3, norm1, norm2, norm3, sigma1, sigma2, sigma3] # alphabetically ordered
    params = [mean1, sigma1, norm1, mean2, sigma2, norm2, mean3, sigma3, norm3] # alphabetically ordered

    plotter.cfg = cfg
    canvas, padT, padB = plotter.canvasRatio()
    dummyT, dummyB, dummyL = plotter.dummyRatio(line=0)
    
    
    ## TOP PAD ##
    canvas.cd()
    padT.Draw()
    padT.cd()
    dummyT.Draw("HIST")
    
    ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.ERROR)
    plt = recoil.frame()
    rdh.plotOn(plt)
    pdf.plotOn(plt, ROOT.RooFit.VisualizeError(fitRes, 1), ROOT.RooFit.Name("errorband"), ROOT.RooFit.FillColor(ROOT.kOrange))
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
    
    hOut.SetBinContent(qTbin, 10, 1, norm)
    
    hOut.SetBinContent(qTbin, 0, 0, chi2)

    # diagonalize covariance matrix and store perturbations
    floatingParams = fitRes.floatParsFinal() # floating parameters
    variations = diagonalize(fitRes)
    # print(p.GetName(), p.isConstant(), type(p), isinstance(p, ROOT.RooFormulaVar))
    for iVar, var in enumerate(variations):
        for iPar, p in enumerate(params):
            if getParamIdx(p.GetName(), floatingParams) > -1: val = var[getParamIdx(p.GetName(), floatingParams)]
            else: val = p.getVal()
            hOut.SetBinContent(qTbin, iPar+1, iVar+2, val)
            print(iVar, iPar, p.GetName(), val)

    return mean1.getVal(), mean2.getVal(), mean3.getVal(), norm1.getVal(), norm2.getVal(), sigma1.getVal(), sigma2.getVal(), sigma3.getVal()
 



def doFitBkg_3Gauss(tag, name, qTbin, qTmin, qTmax, mean1_, mean2_, mean3_, norm1_, norm2_, sigma1_, sigma2_, sigma3_):

    hist = None
    for bkg in bkgs:
        nn = name.replace("bkg", bkg)
        h = fIn.Get(nn)
        print(h)
        if hist == None: hist = h
        else: hist.Add(h)
        
    hist.Rebin(2)
    
    if qTmin < 30: xMin,xMax = -100, 100
    elif qTmin < 60: xMin,xMax = -150, 50
    elif qTmin < 150: xMin,xMax = -200, 50
    else: xMin,xMax = -500, 0
    if "perp" in name: xMin,xMax = -300, 300
    if "_qT" in name: xMin,xMax = -300, 300
    
    #hist = hist.Rebin(1)
    hist_norm = hist.Integral()
    hist.Scale(1./hist_norm)    
    
    qT = 0.5*(qTmin + qTmax)
    


    cfg['xmin'], cfg['xmax'] = xMin, xMax
    cfg['ymax'] = 1.75*hist.GetMaximum()
    cfg['ymax'] = 1e5
    cfg['ymin'] = 1e-5
    


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
    gauss2 = ROOT.RooGaussian("gauss2", "gauss2", recoil, mean2, sigma2)
    gauss3 = ROOT.RooGaussian("gauss3", "gauss3", recoil, mean3, sigma3)
    
    #norm1.setConstant(ROOT.kTRUE)
    #norm2.setConstant(ROOT.kTRUE)
   
    pdf = ROOT.RooAddPdf("pdf", '', ROOT.RooArgList(gauss1, gauss2, gauss3), ROOT.RooArgList(norm1, norm2))
    #pdf = ROOT.RooAddPdf("pdf", '', ROOT.RooArgList(gauss1, gauss2), ROOT.RooArgList(norm1))
    rdh = ROOT.RooDataHist("rdh", "", ROOT.RooArgList(recoil), ROOT.RooFit.Import(hist))


    fitRes = pdf.fitTo(rdh, ROOT.RooFit.SumW2Error(ROOT.kTRUE), ROOT.RooFit.Save(ROOT.kTRUE))
    
    #norm2.setVal(1-norm1.getVal())
    #sigma3.setVal(0)
    #sigma3.setError(0)
    #mean3.setVal(0)
    #mean3.setError(0)
    
    #mean2.setVal(mean1.getVal())
    #mean3.setVal(mean1.getVal())

    plotter.cfg = cfg
    canvas, padT, padB = plotter.canvasRatio()
    dummyT, dummyB, dummyL = plotter.dummyRatio(line=0)
    
    
    ## TOP PAD ##
    canvas.cd()
    padT.Draw()
    padT.cd()
    dummyT.Draw("HIST")
    
    ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.ERROR)
    plt = recoil.frame()
    pdf.plotOn(plt, ROOT.RooFit.VisualizeError(fitRes, 1), ROOT.RooFit.Name("errorband"), ROOT.RooFit.FillColor(ROOT.kOrange))
    rdh.plotOn(plt)
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
    latex.DrawLatex(0.20, 0.85, "%s, Bkgs" % label)
    latex.DrawLatex(0.20, 0.80, "q_{T} = [%.1f, %.1f] GeV" % (qTmin, qTmax))
    latex.DrawLatex(0.20, 0.75, "Mean = %.3f #pm %.3f" % (hist.GetMean(), hist.GetMeanError()))
    latex.DrawLatex(0.20, 0.70, "RMS = %.3f #pm %.3f" % (hist.GetRMS(), hist.GetRMSError()))
    latex.DrawLatex(0.20, 0.65, "#chi^{2}/ndof = %.3f" % chi2)
        
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
    
    return mean1.getVal(), mean2.getVal(), mean3.getVal(), norm1.getVal(), norm2.getVal(), sigma1.getVal(), sigma2.getVal(), sigma3.getVal(), hist_norm







def doFitData_3Gauss(tag, name, hOut, qTbin, qTmin, qTmax, mean1_, mean2_, mean3_, norm1_, norm2_, sigma1_, sigma2_, sigma3_, mean1__bkg, mean2__bkg, mean3__bkg, norm1__bkg, norm2__bkg, sigma1__bkg, sigma2__bkg, sigma3__bkg, hist_norm_bkg):

    hist = fIn.Get(name)
    hist.Rebin(1)
    
    if qTmin < 30: xMin,xMax = -100, 100
    elif qTmin < 60: xMin,xMax = -150, 50
    elif qTmin < 150: xMin,xMax = -200, 50
    else: xMin,xMax = -500, 0
    if "perp" in name: xMin,xMax = -100, 100
    if "_qT" in name: xMin,xMax = -100, 100
    xMin,xMax = -100, 100
    
    hist_norm_data = hist.Integral()
    hist.Scale(1./hist_norm_data)    

 
    recoil = ROOT.RooRealVar("recoil", "Recoil parallel (GeV)", 0, -500, 500)
    cfg['xmin'], cfg['xmax'] = xMin, xMax
    cfg['ymax'] = 1.75*hist.GetMaximum()
    cfg['ymax'] = 1e3
    cfg['ymin'] = 1e-5
    
    
    ## background
    mean1_bkg = ROOT.RooRealVar("mean1_bkg", "", mean1__bkg)
    mean2_bkg = ROOT.RooRealVar("mean2_bkg", "", mean2__bkg)
    mean3_bkg = ROOT.RooRealVar("mean3_bkg", "", mean3__bkg)
    sigma1_bkg = ROOT.RooRealVar("sigma1_bkg", "", sigma1__bkg)
    sigma2_bkg = ROOT.RooRealVar("sigma2_bkg", "", sigma2__bkg)
    sigma3_bkg = ROOT.RooRealVar("sigma3_bkg", "", sigma3__bkg)
    norm1_bkg = ROOT.RooRealVar("norm1_bkg", "", norm1__bkg)
    norm2_bkg = ROOT.RooRealVar("norm2_bkg", "", norm2__bkg)
    gauss1_bkg = ROOT.RooGaussian("gauss1_bkg", "gauss1", recoil, mean1_bkg, sigma1_bkg)
    gauss2_bkg = ROOT.RooGaussian("gauss2_bkg", "gauss2", recoil, mean2_bkg, sigma2_bkg)
    gauss3_bkg = ROOT.RooGaussian("gauss3_bkg", "gauss3", recoil, mean3_bkg, sigma3_bkg)
    pdf_bkg = ROOT.RooAddPdf("pdf_bkg", '', ROOT.RooArgList(gauss1_bkg, gauss2_bkg, gauss3_bkg), ROOT.RooArgList(norm1_bkg, norm2_bkg))
    

    
    rdh = ROOT.RooDataHist("rdh", "", ROOT.RooArgList(recoil), ROOT.RooFit.Import(hist))
    if "para" in name:

        ## signal
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
        gauss2 = ROOT.RooGaussian("gauss2", "gauss2", recoil, mean2, sigma2)
        gauss3 = ROOT.RooGaussian("gauss3", "gauss3", recoil, mean3, sigma3)
        pdf_sig = ROOT.RooAddPdf("pdf_sig", '', ROOT.RooArgList(gauss1, gauss2, gauss3), ROOT.RooArgList(norm1, norm2))

        norm1.setConstant(ROOT.kTRUE)
        norm2.setConstant(ROOT.kTRUE)
        #mean2.setConstant(ROOT.kTRUE)
        #mean3.setConstant(ROOT.kTRUE)
        #norm_sig = ROOT.RooRealVar("norm_sig", "", (1. - hist_norm_bkg/hist_norm_data), 0, 1)
        norm_sig = ROOT.RooRealVar("norm_sig", "", 1)
        norm_sig.setConstant(ROOT.kTRUE)
        pdf = ROOT.RooAddPdf("pdf", '', ROOT.RooArgList(pdf_sig, pdf_bkg), ROOT.RooArgList(norm_sig))
        fitRes = pdf.fitTo(rdh, ROOT.RooFit.SumW2Error(ROOT.kTRUE), ROOT.RooFit.Save(ROOT.kTRUE))   
        
        #mean2.setVal(mean1.getVal())
        #mean3.setVal(mean2.getVal())
        




    if "perp" in name:
    
        ## signal
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
        pdf_sig = ROOT.RooAddPdf("pdf_sig", '', ROOT.RooArgList(gauss1, gauss2, gauss3), ROOT.RooArgList(norm1, norm2))    
       

        norm1.setConstant(ROOT.kTRUE)
        norm2.setConstant(ROOT.kTRUE)
        mean2.setConstant(ROOT.kTRUE)
        mean3.setConstant(ROOT.kTRUE)
        
        #sigma2.setConstant(ROOT.kTRUE)
        #sigma1.setConstant(ROOT.kTRUE)
        
        #norm_sig = ROOT.RooRealVar("norm_sig", "", (1. - hist_norm_bkg/hist_norm_data), 0, 1)
        norm_sig = ROOT.RooRealVar("norm_sig", "", 1)
        norm_sig.setConstant(ROOT.kTRUE)
        pdf = ROOT.RooAddPdf("pdf", '', ROOT.RooArgList(pdf_sig, pdf_bkg), ROOT.RooArgList(norm_sig))
        fitRes = pdf.fitTo(rdh, ROOT.RooFit.SumW2Error(ROOT.kTRUE), ROOT.RooFit.Save(ROOT.kTRUE))
        
        mean2.setVal(mean1.getVal())
        mean3.setVal(mean1.getVal())
        
        
    params = [mean1, sigma1, norm1, mean2, sigma2, norm2, mean3, sigma3, norm3]

    fitValid = True
    for p in params:
        if isinstance(p, ROOT.RooRealVar) and abs(p.getError()/p.getVal()) > 10:
            fitValid = False
            break

    fitValid = True

    plotter.cfg = cfg
    canvas, padT, padB = plotter.canvasRatio()
    dummyT, dummyB, dummyL = plotter.dummyRatio(line=0)
    canvas.cd()
    padT.Draw()
    padT.cd()
    dummyT.Draw("HIST")
    
    ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.ERROR)
    plt = recoil.frame()
    
    if fitValid: pdf_sig.plotOn(plt, ROOT.RooFit.VisualizeError(fitRes, 1), ROOT.RooFit.Name("errorband"), ROOT.RooFit.FillColor(ROOT.kOrange))
    else: pdf_sig.plotOn(plt, ROOT.RooFit.VisualizeError(fitRes, 1), ROOT.RooFit.Name("errorband"), ROOT.RooFit.FillColor(ROOT.kRed))
    rdh.plotOn(plt)
    gauss1.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kBlue), ROOT.RooFit.Normalization(norm1.getVal(), ROOT.RooAbsReal.NumEvent), ROOT.RooFit.LineStyle(ROOT.kDashed))
    gauss2.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kGreen), ROOT.RooFit.Normalization(norm2.getVal(), ROOT.RooAbsReal.NumEvent), ROOT.RooFit.LineStyle(ROOT.kDashed))
    gauss3.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kCyan), ROOT.RooFit.Normalization(norm3.getVal(), ROOT.RooAbsReal.NumEvent), ROOT.RooFit.LineStyle(ROOT.kDashed))
    pdf.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kRed))
    pdf_sig.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kRed), ROOT.RooFit.LineStyle(ROOT.kDashed))
    histpull = plt.pullHist()
    chi2 = plt.chiSquare() 
  
    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.040)
    latex.SetTextColor(1)
    latex.SetTextFont(42)
    latex.DrawLatex(0.20, 0.85, "%s, Data" % label)
    latex.DrawLatex(0.20, 0.80, "q_{T} = [%.1f, %.1f] GeV" % (qTmin, qTmax))
    latex.DrawLatex(0.20, 0.75, "Mean = %.3f #pm %.3f" % (hist.GetMean(), hist.GetMeanError()))
    latex.DrawLatex(0.20, 0.70, "RMS = %.3f #pm %.3f" % (hist.GetRMS(), hist.GetRMSError()))
    latex.DrawLatex(0.20, 0.65, "#chi^{2}/ndof = %.3f" % chi2)

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
    
    hOut.SetBinContent(qTbin, 10, 1, hist_norm_data-hist_norm_bkg)
    
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
            
    return mean1.getVal(), mean2.getVal(), mean3.getVal(), norm1.getVal(), norm2.getVal(), sigma1.getVal(), sigma2.getVal(), sigma3.getVal()
 
 
 
def doFitData_2Gauss(tag, name, hOut, qTbin, qTmin, qTmax, mean1_, mean2_, norm1_, sigma1_, sigma2_, mean1__bkg, mean2__bkg, mean3__bkg, norm1__bkg, norm2__bkg, sigma1__bkg, sigma2__bkg, sigma3__bkg, hist_norm_bkg):

    hist = fIn.Get(name)
    hist.Rebin(1)
    
    if qTmin < 30: xMin,xMax = -100, 100
    elif qTmin < 60: xMin,xMax = -150, 50
    elif qTmin < 150: xMin,xMax = -200, 50
    else: xMin,xMax = -500, 0
    if "perp" in name: xMin,xMax = -100, 100
    if "_qT" in name: xMin,xMax = -100, 100
    
    hist_norm_data = hist.Integral()
    hist.Scale(1./hist_norm_data)    

 
    recoil = ROOT.RooRealVar("recoil", "Recoil parallel (GeV)", 0, -500, 500)
    cfg['xmin'], cfg['xmax'] = xMin, xMax
    cfg['ymax'] = 1.75*hist.GetMaximum()
    cfg['ymax'] = 1e3
    cfg['ymin'] = 1e-6
    
    
    ## background
    mean1_bkg = ROOT.RooRealVar("mean1_bkg", "", mean1__bkg)
    mean2_bkg = ROOT.RooRealVar("mea2_bkg", "", mean2__bkg)
    mean3_bkg = ROOT.RooRealVar("mean3_bkg", "", mean3__bkg)
    sigma1_bkg = ROOT.RooRealVar("sigma1_bkg", "", sigma1__bkg)
    sigma2_bkg = ROOT.RooRealVar("sigma2_bkg", "", sigma2__bkg)
    sigma3_bkg = ROOT.RooRealVar("sigma3_bkg", "", sigma3__bkg)
    norm1_bkg = ROOT.RooRealVar("norm1_bkg", "", norm1__bkg)
    norm2_bkg = ROOT.RooRealVar("norm2_bkg", "", norm2__bkg)
    gauss1_bkg = ROOT.RooGaussian("gauss1_bkg", "gauss1", recoil, mean1_bkg, sigma1_bkg)
    gauss2_bkg = ROOT.RooGaussian("gauss2_bkg", "gauss2", recoil, mean2_bkg, sigma2_bkg)
    gauss3_bkg = ROOT.RooGaussian("gauss3_bkg", "gauss3", recoil, mean3_bkg, sigma3_bkg)
    pdf_bkg = ROOT.RooAddPdf("pdf_bkg", '', ROOT.RooArgList(gauss1_bkg, gauss2_bkg, gauss3_bkg), ROOT.RooArgList(norm1_bkg, norm2_bkg))
    

    
    rdh = ROOT.RooDataHist("rdh", "", ROOT.RooArgList(recoil), ROOT.RooFit.Import(hist))
    if "para" in name:

        ## signal
        mean1 = ROOT.RooRealVar("mean1", "", mean1_, -500, 500)
        mean2 = ROOT.RooRealVar("mean2", "", mean2_, -500, 500)
        sigma1 = ROOT.RooRealVar("sigma1", "", sigma1_, 0.1, 100)
        sigma2 = ROOT.RooRealVar("sigma2", "", sigma2_, 0.1, 100)
        norm1 = ROOT.RooRealVar("norm1", "", norm1_, 0, 1)
        norm2 = ROOT.RooFormulaVar("norm3", "(1. - @0)", ROOT.RooArgList(norm1))
        gauss1 = ROOT.RooGaussian("gauss1", "gauss1", recoil, mean1, sigma1)
        gauss2 = ROOT.RooGaussian("gauss2", "gauss2", recoil, mean2, sigma2)
        pdf_sig = ROOT.RooAddPdf("pdf_sig", '', ROOT.RooArgList(gauss1, gauss2), ROOT.RooArgList(norm1))

        norm1.setConstant(ROOT.kTRUE)
        
        norm_sig = ROOT.RooRealVar("norm_sig", "", (1. - hist_norm_bkg/hist_norm_data), 0, 1)
        norm_sig.setConstant(ROOT.kTRUE)
        pdf = ROOT.RooAddPdf("pdf", '', ROOT.RooArgList(pdf_sig, pdf_bkg), ROOT.RooArgList(norm_sig))
        fitRes = pdf.fitTo(rdh, ROOT.RooFit.SumW2Error(ROOT.kTRUE), ROOT.RooFit.Save(ROOT.kTRUE))           

        #mean2.setVal(mean1.getVal())

    if "perp" in name:
    
        ## signal
        mean1 = ROOT.RooRealVar("mean1", "", mean1_, -500, 500)
        mean2 = ROOT.RooRealVar("mean2", "", mean2_, -500, 500)
        sigma1 = ROOT.RooRealVar("sigma1", "", sigma1_, 0.1, 100)
        sigma2 = ROOT.RooRealVar("sigma2", "", sigma2_, 0.1, 100)
        norm1 = ROOT.RooRealVar("norm1", "", norm1_, 0, 1)
        norm2 = ROOT.RooFormulaVar("norm3", "(1. - @0)", ROOT.RooArgList(norm1))
        gauss1 = ROOT.RooGaussian("gauss1", "gauss1", recoil, mean1, sigma1)
        gauss2 = ROOT.RooGaussian("gauss2", "gauss2", recoil, mean2, sigma2)
        pdf_sig = ROOT.RooAddPdf("pdf_sig", '', ROOT.RooArgList(gauss1, gauss2), ROOT.RooArgList(norm1))    
        
        #mean1.setVal(0)
        #mean1.setConstant(ROOT.kTRUE)
        #mean2.setVal(0)
        #mean2.setConstant(ROOT.kTRUE)
        #mean3.setVal(0)
        #mean3.setConstant(ROOT.kTRUE)

        norm1.setConstant(ROOT.kTRUE)
        
        norm_sig = ROOT.RooRealVar("norm_sig", "", (1. - hist_norm_bkg/hist_norm_data), 0, 1)
        norm_sig.setConstant(ROOT.kTRUE)
        pdf = ROOT.RooAddPdf("pdf", '', ROOT.RooArgList(pdf_sig, pdf_bkg), ROOT.RooArgList(norm_sig))
        fitRes = pdf.fitTo(rdh, ROOT.RooFit.SumW2Error(ROOT.kTRUE), ROOT.RooFit.Save(ROOT.kTRUE))   

    params = [mean1, sigma1, norm1, mean2, sigma2, norm2]

   
    
    #mean2.setVal(mean1.getVal())
    #mean3.setVal(mean1.getVal())  

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
    pdf.plotOn(plt, ROOT.RooFit.VisualizeError(fitRes, 1), ROOT.RooFit.Name("errorband"), ROOT.RooFit.FillColor(ROOT.kOrange))
    gauss1.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kBlue), ROOT.RooFit.Normalization(norm1.getVal(), ROOT.RooAbsReal.NumEvent), ROOT.RooFit.LineStyle(ROOT.kDashed))
    gauss2.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kGreen), ROOT.RooFit.Normalization(norm2.getVal(), ROOT.RooAbsReal.NumEvent), ROOT.RooFit.LineStyle(ROOT.kDashed))
    pdf.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kRed))
    pdf_sig.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kRed), ROOT.RooFit.LineStyle(ROOT.kDashed))
    histpull = plt.pullHist()
    chi2 = plt.chiSquare() 
  
    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.040)
    latex.SetTextColor(1)
    latex.SetTextFont(42)
    latex.DrawLatex(0.20, 0.85, "%s, Data" % label)
    latex.DrawLatex(0.20, 0.80, "q_{T} = [%.1f, %.1f] GeV" % (qTmin, qTmax))
    latex.DrawLatex(0.20, 0.75, "Mean = %.3f #pm %.3f" % (hist.GetMean(), hist.GetMeanError()))
    latex.DrawLatex(0.20, 0.70, "RMS = %.3f #pm %.3f" % (hist.GetRMS(), hist.GetRMSError()))

    latex.DrawLatex(0.60, 0.85, "#mu_{1} = %.3f #pm %.3f" % (mean1.getVal(), mean1.getError()))
    latex.DrawLatex(0.60, 0.80, "#mu_{2} = %.3f #pm %.3f" % (mean2.getVal(), mean2.getError()))
    latex.DrawLatex(0.60, 0.75, "#sigma_{1} = %.3f #pm %.3f" % (sigma1.getVal(), sigma1.getError()))
    latex.DrawLatex(0.60, 0.70, "#sigma_{2} = %.3f #pm %.3f" % (sigma2.getVal(), sigma2.getError()))
    latex.DrawLatex(0.60, 0.65, "N_{1} = %.3f #pm %.3f" % (norm1.getVal(), norm1.getError()))
    latex.DrawLatex(0.60, 0.60, "#chi^{2}/ndof = %.3f" % chi2)

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
    
    hOut.SetBinContent(qTbin, 7, 1, 0)
    hOut.SetBinContent(qTbin, 7, 0, 0)
    hOut.SetBinContent(qTbin, 8, 1, 0)
    hOut.SetBinContent(qTbin, 8, 0, 0)
    hOut.SetBinContent(qTbin, 9, 1, 0)
    hOut.SetBinContent(qTbin, 9, 0, 0)
    

    # diagonalize covariance matrix and store perturbations
    floatingParams = fitRes.floatParsFinal() # floating parameters
    variations = diagonalize(fitRes)
    # print(p.GetName(), p.isConstant(), type(p), isinstance(p, ROOT.RooFormulaVar))
    for iVar, var in enumerate(variations):
        for iPar, p in enumerate(params):
            if getParamIdx(p.GetName(), floatingParams) > -1: val = var[getParamIdx(p.GetName(), floatingParams)]
            else: val = p.getVal()
            hOut.SetBinContent(qTbin, iPar+1, iVar+2, val)
            print(iVar, iPar, p.GetName(), val)

    return mean1.getVal(), mean2.getVal(), norm1.getVal(), sigma1.getVal(), sigma2.getVal()
 

def calcReweight(): 
 
    recoil_qTbins = list(range(0, 30, 1)) + list(range(30, 50, 2)) + list(range(50, 70, 5)) + list(range(70, 150, 10)) + [150, 200, 300, 10000]
    
    
    norm_data, norm_mc, norm_bkg = 0, 0, 0
    for i in range(1, len(recoil_qTbins)):
    
        h_para_data = fIn.Get("data_para_qT_bin%d" % i)
        h_para_mc = fIn.Get("mc_para_qT_bin%d" % i)
        h_para_bkg = fIn.Get("bkg_para_qT_bin%d" % i)
        
        norm_data += h_para_data.Integral()
        norm_mc += h_para_mc.Integral()
        norm_bkg += h_para_bkg.Integral()
        
    norm_data -= norm_bkg
    sf = norm_data / norm_mc
    


    for i in range(1, len(recoil_qTbins)):
    
        h_para_data = fIn.Get("data_para_qT_bin%d" % i)
        h_perp_data = fIn.Get("data_perp_bin%d" % i)
        h_para_mc = fIn.Get("mc_para_qT_bin%d" % i)
        h_perp_mc = fIn.Get("mc_perp_bin%d" % i)
        h_para_bkg = fIn.Get("bkg_para_qT_bin%d" % i)
        h_perp_bkg = fIn.Get("bkg_perp_bin%d" % i)
        
        #h_para_data.Add(h_para_bkg, -1)

        #print(i, h_para_data.Integral(), h_perp_data.Integral(), h_para_mc.Integral(), h_perp_mc.Integral(), h_para_bkg.Integral(), h_perp_bkg.Integral())
        a = h_para_data.Integral() / h_para_mc.Integral() / sf
        #b = (h_perp_data.Integral()-h_perp_bkg.Integral()) / h_perp_mc.Integral()
        #print(i, a, b)
        print("%.3f, " % a, end = '')
 
 
 
def doFit_mc_perp():

    tag = "mc_perp"
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
    
    func_sigma1 = ROOT.TF1("func_sigma1", "[0]*TMath::Power(x+[1], [2])", 0, 200)
    func_sigma1.SetParameters(3.29832e+00, 1.39192e+01, 2.36563e-01)
    
    func_sigma2 = ROOT.TF1("func_sigma2", "[0]*TMath::Power(x+[1], [2])", 0, 200)
    func_sigma2.SetParameters(1.90798e+00, 5.06144e+00, 2.35892e-01)
    func_sigma3 = ROOT.TF1("func_sigma3", "[0]*TMath::Power(x+[1], [2])", 0, 200)
    func_sigma3.SetParameters(3.01634e+00, 2.50830e+01, 3.43482e-01)
    
    func_mean1 = ROOT.TF1("func_mean1", "[0]*x + [1]", 0, 150)
    func_mean1.SetParameters(2.54771e-03, -1.41744e-03)
    
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
        hName = "%s_perp_bin%d" % (sig, qTbin)
        
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


    fOut = ROOT.TFile(fOut_.replace(".root", "_mc_perp.root"), "RECREATE")
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
    
    

def doFit_data_perp():

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
    datagroups = DatagroupsLowPU_Z("lowPU_%s_%s.pkl.lz4" % (flavor, met), flavor=flavor)
    procs = []
    if flavor == "mumu": procs = ["SingleMuon", "DYmumu"] + bkgs
    if flavor == "ee": procs = ["SingleElectron", "DYee"] + bkgs

    
    for proc in procs:
    
        bhist_para_qT = readProc(datagroups, "recoil_corr_xy_para_qT_qTbinned", proc)
        bhist_perp = readProc(datagroups, "recoil_corr_xy_perp_qTbinned", proc)
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
  

  
if __name__ == "__main__":

    met = "RawPFMET" # RawPFMET DeepMETReso
    flavor = "mu" # mu, e, mumu, ee

    if flavor == "mumu": 
        label = "DY #rightarrow #mu^{+}#mu^{#minus}, %s" % met
        sig = "DYmumu"
        data = "SingleMuon"
    if flavor == "ee": 
        label = "DY #rightarrow e^{+}e^{#minus}, %s" % met
        sig = "DYee"
        data = "SingleElectron"
    if flavor == "mu": 
        label = "W #rightarrow #mu^{#pm}, %s" % met
        sig = "WjetsMuNu"
        
    ####################################################################
    outDir = "/eos/user/j/jaeyserm/www/wmass/lowPU/recoilCorrection/recoil_%s_%s/" % (flavor, met)
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
    recoil_qTbins = list(drange(0, 30, 0.5)) + list(range(30, 50, 1)) + list(range(50, 70, 2)) + list(range(70, 100, 5)) + list(range(100, 150, 10)) + [150, 200, 300, 10000]
    recoil_qTbins = list(range(0, 50, 1)) + list(range(50, 70, 2)) + list(range(70, 100, 5)) + list(range(100, 150, 10)) + [150, 200, 300, 10000]
    #recoil_qTbins = [0, 0.5, 1, 1.5] + list(range(2, 50, 1)) + list(range(50, 70, 2)) + list(range(70, 100, 5)) + list(range(100, 150, 10)) + [150, 200, 300, 10000]
    
    #recoil_qTbins = list(range(0, 200, 1))
    #recoil_qTbins = list(range(0, 20, 1)) + list(range(20, 40, 2)) + list(range(40, 55, 3)) + list(range(55, 80, 5)) + list(range(80, 100, 10)) + [100, 125, 150, 10000]
    #recoil_qTbins = list(range(0, 200, 1))
    print(recoil_qTbins)
    #sys.exit()
    
    

    prepareFile(fIn_)
    #print(fIn_)
    #sys.exit()
    
    fIn = ROOT.TFile(fIn_)
    
    #doFit_mc_gen_para()
    doFit_mc_gen_perp()
    #doFit_mc_para()
    #doFit_mc_perp()
    sys.exit()
    #doFit_data_perp()
    #doFit_data_perp_2gauss()
    #doFit_data_para()
    #doFit_data_para_2gauss()
    
    os.system("hadd -f %s %s %s %s %s" % (fOut_, fOut_.replace(".root", "_data_para.root"), fOut_.replace(".root", "_data_perp.root"), fOut_.replace(".root", "_mc_para.root"), fOut_.replace(".root", "_mc_perp.root")))
    
    sys.exit()
    '''
    from threading import Thread
    threads = []
    t1 = threads.append(Thread(target=doFit_mc_perp))
    t2 = threads.append(Thread(target=doFit_data_perp))
    #t3 = threads.append(Thread(target=doFit_data_para))
    #t4 = threads.append(Thread(target=doFit_mc_para))
    
    for x in threads: x.start()
    for x in threads: x.join()
    
    print("DONE")
    '''
    
    
    
    
      

    # PERP MC
    functions.prepareDir(outDir + "/mc_perp/", True)
    cfg['xtitle'] = "U_{#perp}  (GeV)"
    h_perp_mc = ROOT.TH3D("mc_perp", "", 150, 0, 150, 20, 0, 20, 20, 0, 20) # qt, nominal, error
    mean1__, mean2__, mean3__, norm1__, norm2__, sigma1__, sigma2__, sigma3__ = 0, 0, 0,   0.6, 0.2,   8, 5, 18 # PF MET
    mean1__, mean2__, mean3__, norm1__, norm2__, sigma1__, sigma2__, sigma3__ = 0, 0, 0,   0.6, 0.35,   8, 5, 15 # PF MET
    #mean1__, mean2__, mean3__, norm1__, norm2__, sigma1__, sigma2__, sigma3__ = 0, 0, 0,   0.6, 0.3,   6, 4, 10 # DeepMET
    #mean1__, mean2__, mean3__, norm1__, norm2__, sigma1__, sigma2__, sigma3__ = 0, 0, 0,   0.6, 0.3,   7, 5, 10 # DeepMET
    
    #mean1__, mean2__, mean3__, norm1__, norm2__, sigma1__, sigma2__, sigma3__ = 0, 0, 0,   0.6, 0.3,   6, 5, 10 # DeepMET

    
    func_mean = ROOT.TF1("func_mean", "[0]*x + [1]", 0, 150)
    func_mean.SetParameters(2.39093e-03, -6.18303e-04)
    for i in range(1, len(recoil_qTbins)):
        continue
        qT = 0.5*(recoil_qTbins[i-1] + recoil_qTbins[i]) if i < len(recoil_qTbins)-1 else (recoil_qTbins[i-1] + 20)
        #sigma2__ = func_sigma2.Eval(qT)
        #sigma3__ = func_sigma3.Eval(qT)
        #sigma1__ = func_sigma1.Eval(qT)
        #mean1__ = func_mean.Eval(qT)
        #mean2__ = mean1__
        #mean3__ = mean1__
        mean1__, mean2__, mean3__, norm1__, norm2__, sigma1__, sigma2__, sigma3__ = doFitMC_3Gauss("mc_perp", "%s_perp_bin%d" % (sig, i), h_perp_mc, i, recoil_qTbins[i-1], recoil_qTbins[i], mean1__, mean2__, mean3__, norm1__, norm2__, sigma1__, sigma2__, sigma3__)

    #fOut = ROOT.TFile(fOut_.replace(".root", "_mc_perp.root"), "RECREATE")
    #h_perp_mc.Write()
    #fOut.Close()

    
    # PARA MC
    functions.prepareDir(outDir + "/mc_para/", False)
    cfg['xtitle'] = "U_{#parallel} + q_{T} (GeV)"
    h_para_mc = ROOT.TH3D("mc_para", "", 150, 0, 150, 20, 0, 20, 20, 0, 20) # qt, nominal, error
    mean1__, mean2__, mean3__, norm1__, norm2__, sigma1__, sigma2__, sigma3__ = 0.2, 0.2, 0.2,    0.5, 0.35,     8, 5, 15 # PF MET
    #mean1__, mean2__, mean3__, norm1__, norm2__, sigma1__, sigma2__, sigma3__ = 0, 0, 0,   0.6, 0.35,   10, 5, 15 # PF MET
    #mean1__, mean2__, mean3__, norm1__, norm2__, sigma1__, sigma2__, sigma3__ = 0.2, 0.2, 0.2,    0.55, 0.35,     6, 4, 10 # DeepMET
    #mean1__, mean2__, mean3__, norm1__, norm2__, sigma1__, sigma2__, sigma3__ = 0.2, 0.2, 0.2,    0.6, 0.3,     6, 4, 10 # DeepMET
    #mean1__, mean2__, mean3__, norm1__, norm2__, sigma1__, sigma2__, sigma3__ = 0, 0, 0,   0.6, 0.3,   6, 4, 10 # DeepMET
    for i in range(1, len(recoil_qTbins)):
        continue
        qT = 0.5*(recoil_qTbins[i-1] + recoil_qTbins[i]) if i < len(recoil_qTbins)-1 else (recoil_qTbins[i-1] + 20)
        mean1__, mean2__, mean3__, norm1__, norm2__, sigma1__, sigma2__, sigma3__ = doFitMC_3Gauss("mc_para", "%s_para_bin%d" % (sig, i), h_para_mc, i, recoil_qTbins[i-1], recoil_qTbins[i], mean1__, mean2__, mean3__, norm1__, norm2__, sigma1__, sigma2__, sigma3__)
    
    
    # PARA DATA
    functions.prepareDir(outDir + "/data_para/", False)
    functions.prepareDir(outDir + "/bkg_para/", False)
    cfg['xtitle'] = "U_{#parallel} + q_{T} (GeV)"    
    h_para_data = ROOT.TH3D("data_para", "", 150, 0, 150, 20, 0, 20, 20, 0, 20) # qt, nominal, error
    mean1__bkg, mean2__bkg, mean3__bkg, norm1__bkg, norm2__bkg, sigma1__bkg, sigma2__bkg, sigma3__bkg = 0, 0, 0, 0.5, 0.3, 5, 10, 15 # PF MET
    mean1__, mean2__, mean3__, norm1__, norm2__, sigma1__, sigma2__, sigma3__ = 0, 0, 0,   0.6, 0.35,   10, 5, 15 # PF MET
    mean1__, mean2__, mean3__, norm1__, norm2__, sigma1__, sigma2__, sigma3__ = 0.2, 0.2, 0.2,    0.5, 0.35,     8, 5, 15 # PF MET
    #mean1__bkg, mean2__bkg, mean3__bkg, norm1__bkg, norm2__bkg, sigma1__bkg, sigma2__bkg, sigma3__bkg = 0, 0, 0, 0.5, 0.3, 5, 10, 15 # DeepMET
    #mean1__, mean2__, mean3__, norm1__, norm2__, sigma1__, sigma2__, sigma3__ = 0, 0, 0,   0.6, 0.35,   6, 4, 10 # DeepMET
    #mean1__, mean2__, mean3__, norm1__, norm2__, sigma1__, sigma2__, sigma3__ = 0, 0, 0,   0.6, 0.3,   8, 5, 10 # DeepMET
    #mean1__, mean2__, norm1__, sigma1__, sigma2__ = 0, 0,   0.6,   6, 4
    for i in range(1, len(recoil_qTbins)):
        #continue
        qT = 0.5*(recoil_qTbins[i-1] + recoil_qTbins[i]) if i < len(recoil_qTbins)-1 else (recoil_qTbins[i-1] + 20)
        mean1__bkg, mean2__bkg, mean3__bkg,  norm1__bkg, norm2__bkg, sigma1__bkg, sigma2__bkg, sigma3__bkg, hist_norm_bkg = doFitBkg_3Gauss("bkg_para", "bkg_para_bin%d" % i, i, recoil_qTbins[i-1], recoil_qTbins[i], mean1__bkg, mean2__bkg, mean3__bkg, norm1__bkg, norm2__bkg, sigma1__bkg, sigma2__bkg, sigma3__bkg)
        mean1__, mean2__, mean3__, norm1__, norm2__, sigma1__, sigma2__, sigma3__ = doFitData_3Gauss("data_para", "%s_para_bin%d" % (data, i), h_para_data, i, recoil_qTbins[i-1], recoil_qTbins[i], mean1__, mean2__, mean3__, norm1__, norm2__, sigma1__, sigma2__, sigma3__, mean1__bkg, mean2__bkg, mean3__bkg, norm1__bkg, norm2__bkg, sigma1__bkg, sigma2__bkg, sigma3__bkg, hist_norm_bkg)
        #mean1__, mean2__, norm1__, sigma1__, sigma2__ = doFitData_2Gauss("data_para", "data_para_qT_bin%d" % i, h_para_data, i, recoil_qTbins[i-1], recoil_qTbins[i], mean1__, mean2__, norm1__, sigma1__, sigma2__, mean1__bkg, mean2__bkg, mean3__bkg, norm1__bkg, norm2__bkg, sigma1__bkg, sigma2__bkg, sigma3__bkg, hist_norm_bkg)

    sys.exit()
    # PERP DATA
    functions.prepareDir(outDir + "/bkg_perp/", False)
    functions.prepareDir(outDir + "/data_perp/", False)
    cfg['xtitle'] = "U_{#perp}  (GeV)"
    h_perp_data = ROOT.TH3D("data_perp", "", 150, 0, 150, 20, 0, 20, 20, 0, 20) # qt, nominal, error
    #mean1__bkg, mean2__bkg, mean3__bkg, norm1__bkg, norm2__bkg, sigma1__bkg, sigma2__bkg, sigma3__bkg = 0, 0, 0, 0.3, 0.0, 3, 15, 8 # PF MET
    mean1__, mean2__, mean3__, norm1__, norm2__, sigma1__, sigma2__, sigma3__ = 0, 0, 0,   0.5, 0.25,   10, 10, 20 # PF MET
    mean1__, mean2__, mean3__, norm1__, norm2__, sigma1__, sigma2__, sigma3__ = 0, 0, 0,   0.5, 0.35,   10, 5, 15 # PF MET
    mean1__, mean2__, mean3__, norm1__, norm2__, sigma1__, sigma2__, sigma3__ = 0, 0, 0,   0.6, 0.35,   8, 5, 15 # PF MET
    #mean1__bkg, mean2__bkg, mean3__bkg, norm1__bkg, norm2__bkg, sigma1__bkg, sigma2__bkg, sigma3__bkg = 0, 0, 0, 0.3, 0.0, 3, 15, 8 # DeepMET
    #mean1__, mean2__, mean3__, norm1__, norm2__, sigma1__, sigma2__, sigma3__ = 0, 0, 0,   0.6, 0.3,   8, 5, 10 # DeepMET
    #mean1__, mean2__, norm1__, sigma1__, sigma2__ = 0, 0,   0.6,   8, 5
    func_sigma2 = ROOT.TF1("data_perp_sigma2", "[0]*TMath::Power(x+[1], [2])", 0, 200)
    func_sigma2.SetParameters(2.46190e+00, 5.50556e+00, 2.02359e-01)
    func_sigma1 = ROOT.TF1("data_perp_sigma1", "[0]*TMath::Power(x+[1], [2])", 0, 200)
    func_sigma1.SetParameters(3.10493e+00, 2.12375e+01, 2.57209e-01)
    
    func_mean = ROOT.TF1("func_mean", "[0]*x + [1]", 0, 150)
    func_mean.SetParameters(1.62056e-03, -1.32873e-02)
    
    for i in range(1, len(recoil_qTbins)):
        #continue
        qT = 0.5*(recoil_qTbins[i-1] + recoil_qTbins[i]) if i < len(recoil_qTbins)-1 else (recoil_qTbins[i-1] + 20)
        #sigma2__ = func_sigma2.Eval(qT)
        #sigma1__ = func_sigma1.Eval(qT)
        #mean1__ = func_mean.Eval(qT)
        mean1__bkg, mean2__bkg, mean3__bkg,  norm1__bkg, norm2__bkg, sigma1__bkg, sigma2__bkg, sigma3__bkg, hist_norm_bkg = doFitBkg_3Gauss("bkg_perp", "bkg_perp_bin%d" % i, i, recoil_qTbins[i-1], recoil_qTbins[i], mean1__bkg, mean2__bkg, mean3__bkg, norm1__bkg, norm2__bkg, sigma1__bkg, sigma2__bkg, sigma3__bkg)
        mean1__, mean2__, mean3__, norm1__, norm2__, sigma1__, sigma2__, sigma3__ = doFitData_3Gauss("data_perp", "%s_perp_bin%d" % (data, i), h_perp_data, i, recoil_qTbins[i-1], recoil_qTbins[i], mean1__, mean2__, mean3__, norm1__, norm2__, sigma1__, sigma2__, sigma3__, mean1__bkg, mean2__bkg, mean3__bkg, norm1__bkg, norm2__bkg, sigma1__bkg, sigma2__bkg, sigma3__bkg, hist_norm_bkg)
        #mean1__, mean2__, norm1__, sigma1__, sigma2__ = doFitData_2Gauss("data_perp", "data_perp_bin%d" % i, h_perp_data, i, recoil_qTbins[i-1], recoil_qTbins[i], mean1__, mean2__, norm1__, sigma1__, sigma2__, mean1__bkg, mean2__bkg, mean3__bkg, norm1__bkg, norm2__bkg, sigma1__bkg, sigma2__bkg, sigma3__bkg, hist_norm_bkg)
        
      
    fOut = ROOT.TFile(fOut_, "RECREATE")
    h_para_mc.Write()
    h_perp_mc.Write()
    h_para_data.Write()
    h_perp_data.Write()
    fOut.Close()      
    fIn.Close()