
import sys,array,math,os,copy,fnmatch
from collections import OrderedDict

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptTitle(0)

import plotter
import functions

import lz4.frame
import pickle
import narf
import numpy as np

from wremnants.datasets.datagroups import Datagroups


w = ROOT.RooWorkspace("w", "")

narf.clingutils.Declare("""

void GaussianKernel(TH1D h, TH1D &h_sm, TH1D &h_pull, double sigma) {

    for(int i = 1; i <= h.GetNbinsX(); i++) {
        
        double num = 0.0;
        double sumw = 0.0;
        for(int j = 1; j <= h.GetNbinsX(); j++) {
        
            double weight = exp( -(i-j)*(i-j)/(2.*sigma));
            sumw += weight;
            num += h.GetBinContent(j)*weight;
        }
        
        h_sm.SetBinContent(i, num/sumw);
        h_sm.SetBinError(i, 0);
        if(h.GetBinContent(i) > 0) h_pull.SetBinContent(i, (num/sumw-h.GetBinContent(i))/h.GetBinError(i));
    } 
}


Double_t DoubleSidedCB2(double x, double mu, double width, double a1, double p1, double a2, double p2) {
  double u   = (x-mu)/width;
  double A1  = TMath::Power(p1/TMath::Abs(a1),p1)*TMath::Exp(-a1*a1/2);
  double A2  = TMath::Power(p2/TMath::Abs(a2),p2)*TMath::Exp(-a2*a2/2);
  double B1  = p1/TMath::Abs(a1) - TMath::Abs(a1);
  double B2  = p2/TMath::Abs(a2) - TMath::Abs(a2);

  double result(1);
  if      (u<-a1) result *= A1*TMath::Power(B1-u,-p1);
  else if (u<a2)  result *= TMath::Exp(-u*u/2);
  else            result *= A2*TMath::Power(B2+u,-p2);
  return result;
}


double DoubleSidedCB(double* x, double *par) {
  return(par[0] * DoubleSidedCB2(x[0], par[1],par[2],par[3],par[4],par[5],par[6]));
}


""")




def getqTBinInfo(qTbinIdx):

    xLow = qTbins[qTbinIdx]
    xHigh = qTbins[qTbinIdx+1]
    
    if xLow < 30: xMin,xMax = -100, 100
    elif xLow < 60: xMin,xMax = -150, 50
    elif xLow < 100: xMin,xMax = -200, 50
    else: xMin,xMax = -300, 0

    name = "%s_%sGeV" % (str(xLow).replace(".", "p"), str(xHigh).replace(".", "p"))
    return name, xLow, xHigh, xMin, xMax

 
def doFit_4Gauss(outDir, qTbinMinGeV, qTbinMaxGeV, hist, name, label, met, cfg, hOut):
                 
    if qTbinMinGeV < 30: xMin,xMax = -100, 100
    elif qTbinMinGeV < 60: xMin,xMax = -150, 50
    elif qTbinMinGeV < 100: xMin,xMax = -200, 50
    else: xMin,xMax = -300, 0
    
    qTlabel = "%.1f < q_{T} < %.1f GeV" % (qTbinMinGeV, qTbinMaxGeV)
    outName = "%s_%sGeV" % (str("%.1f" % qTbinMinGeV).replace(".", "p"), str("%.1f" % qTbinMaxGeV).replace(".", "p"))
    if "_gen" in outDir: qTlabel = qTlabel.replace("q_{T}", "q_{T}^{GEN}")
    
    if "para_" in name:
    
        cfg['xmin'] = xMin
        cfg['xmax'] = xMax
        
    else:
    
        cfg['xmin'] = -100
        cfg['xmax'] = 100

    recoil = ROOT.RooRealVar("recoil_%s" % name, "Recoil parallel (GeV)", 0, -300, 100) # independent var needed for CDF
    mean1 = ROOT.RooRealVar("mean1_%s" % name, "", 0, -300, 100)
    mean2 = ROOT.RooRealVar("mean2_%s" % name, "", 0, -300, 100)
    mean3 = ROOT.RooRealVar("mean3_%s" % name, "", 0, -300, 100)
    mean4 = ROOT.RooRealVar("mean4_%s" % name, "", 0, -300, 100)
    sigma1 = ROOT.RooRealVar("sigma1_%s" % name, "", 5, 0.1, 100)
    sigma2 = ROOT.RooRealVar("sigma2_%s" % name, "", 5, 0.1, 100)
    sigma3 = ROOT.RooRealVar("sigma3_%s" % name, "", 5, 0.1, 100)
    sigma4 = ROOT.RooRealVar("sigma4_%s" % name, "", 5, 0.1, 100)
    norm1 = ROOT.RooRealVar("norm1_%s" % name, "norm", 0, 0, 1e8)
    norm2 = ROOT.RooRealVar("norm2_%s" % name, "norm", 0, 0, 1e8)
    norm3 = ROOT.RooRealVar("norm3_%s" % name, "norm", 0, 0, 1e8)
    norm4 = ROOT.RooRealVar("norm4_%s" % name, "norm", 0, 0, 1e8)
    gauss1 = ROOT.RooGaussian("gauss1_%s" % name, "gauss1", recoil, mean1, sigma1)
    gauss2 = ROOT.RooGaussian("gauss2_%s" % name, "gauss2", recoil, mean2, sigma2)
    gauss3 = ROOT.RooGaussian("gauss3_%s" % name, "gauss2", recoil, mean3, sigma3)
    gauss4 = ROOT.RooGaussian("gauss4_%s" % name, "gauss4", recoil, mean4, sigma4)
    pdf = ROOT.RooAddPdf("pdf_%s" % name, '', ROOT.RooArgList(gauss1, gauss2, gauss3, gauss4), ROOT.RooArgList(norm1, norm2, norm3, norm4)) # sum of 2 Gaussians

    rdh = ROOT.RooDataHist("rdh_%s" % name, "", ROOT.RooArgList(recoil), ROOT.RooFit.Import(hist))
    norm1.setVal(0.25*hist.Integral())
    norm2.setVal(0.25*hist.Integral())
    norm3.setVal(0.25*hist.Integral())
    norm4.setVal(0.25*hist.Integral())
    mean1.setVal(hist.GetMean())
    mean2.setVal(hist.GetMean())
    sigma1.setVal(10)
    sigma2.setVal(5)
    pdf.fitTo(rdh, ROOT.RooFit.Extended(ROOT.kTRUE), ROOT.RooFit.SumW2Error(ROOT.kTRUE))   

    plotter.cfg = cfg
    canvas, padT, padB = plotter.canvasRatio()
    dummyT, dummyB, dummyL = plotter.dummyRatio()
        
    ## TOP PAD ##
    canvas.cd()
    padT.Draw()
    padT.cd()
    dummyT.Draw("HIST")
        
    plt = recoil.frame()
    #plt.SetTitle("ZH signal")
    rdh.plotOn(plt) # , ROOT.RooFit.Normalization(yield_zh, ROOT.RooAbsReal.NumEvent) , ROOT.RooFit.Binning(200)
    
    gauss1.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kBlue), ROOT.RooFit.Normalization(norm1.getVal(), ROOT.RooAbsReal.NumEvent), ROOT.RooFit.LineStyle(ROOT.kDashed))
    #gauss1.paramOn(plt)
        
    gauss2.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kBlue), ROOT.RooFit.Normalization(norm2.getVal(), ROOT.RooAbsReal.NumEvent), ROOT.RooFit.LineStyle(ROOT.kDashed))
    #gauss2.paramOn(plt)
    
    gauss3.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kBlue), ROOT.RooFit.Normalization(norm3.getVal(), ROOT.RooAbsReal.NumEvent), ROOT.RooFit.LineStyle(ROOT.kDashed))
    
    gauss4.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kBlue), ROOT.RooFit.Normalization(norm4.getVal(), ROOT.RooAbsReal.NumEvent), ROOT.RooFit.LineStyle(ROOT.kDashed))
        
    pdf.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kRed))
    #pdf.paramOn(plt, ROOT.RooFit.Format("NELU", ROOT.RooFit.AutoPrecision(2)), ROOT.RooFit.Layout(0.45, 0.9, 0.9))
        
    ratio = hist.Integral() / (norm1.getVal() + norm2.getVal() + norm3.getVal() + norm4.getVal())
    err = ratio * math.hypot(norm1.getError()+norm2.getError()) / (norm1.getVal() + norm2.getVal())
        
    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.040)
    latex.SetTextColor(1)
    latex.SetTextFont(42)
    latex.DrawLatex(0.20, 0.85, label)
    latex.DrawLatex(0.20, 0.80, qTlabel)
    latex.DrawLatex(0.20, 0.75, met_labels[met_estimators.index(met)])
    latex.DrawLatex(0.20, 0.70, "N_{hist}/N_{fit} = %.3f #pm %.3f" % (ratio, err))
    latex.DrawLatex(0.20, 0.65, "Mean = %.3f" % hist.GetMean())
    latex.DrawLatex(0.20, 0.60, "RMS = %.3f" % hist.GetRMS())
    
    latex.SetTextSize(0.0250)
    latex.DrawLatex(0.55, 0.86, "#mu_{1} = %.2f #pm %.2f" % (mean1.getVal(), mean1.getError()))
    latex.DrawLatex(0.55, 0.83, "#sigma_{1} = %.3f #pm %.2f" % (sigma1.getVal(), sigma1.getError()))
    latex.DrawLatex(0.55, 0.80, "N_{1} = %.2f #pm %.2f" % (norm1.getVal(), norm1.getError()))
    
    latex.DrawLatex(0.55, 0.77, "#mu_{2} = %.2f #pm %.2f" % (mean2.getVal(), mean2.getError()))
    latex.DrawLatex(0.55, 0.74, "#sigma_{2} = %.2f #pm %.2f" % (sigma2.getVal(), sigma2.getError()))
    latex.DrawLatex(0.55, 0.71, "N_{2} = %.2f #pm %.2f" % (norm2.getVal(), norm2.getError()))
    
    latex.DrawLatex(0.75, 0.86, "#mu_{3} = %.2f #pm %.2f" % (mean3.getVal(), mean3.getError()))
    latex.DrawLatex(0.75, 0.83, "#sigma_{3} = %.2f #pm %.2f" % (sigma3.getVal(), sigma3.getError()))
    latex.DrawLatex(0.75, 0.80, "N_{3} = %.2f #pm %.2f" % (norm3.getVal(), norm3.getError()))
    
    latex.DrawLatex(0.75, 0.77, "#mu_{4} = %.2f #pm %.2f" % (mean4.getVal(), mean4.getError()))
    latex.DrawLatex(0.75, 0.74, "#sigma_{4} = %.2f #pm %.2f" % (sigma4.getVal(), sigma4.getError()))
    latex.DrawLatex(0.75, 0.71, "N_{4} = %.2f #pm %.2f" % (norm4.getVal(), norm4.getError()))
    
        
    histpull = plt.pullHist()
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
        
    line = ROOT.TLine(120, 0, 140, 0)
    line.SetLineColor(ROOT.kBlue+2)
    line.SetLineWidth(2)
    line.Draw("SAME")
        
      
    canvas.Modify()
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs("%s/recoil_%s.png" % (outDir, outName))
    canvas.SaveAs("%s/recoil_%s.pdf" % (outDir, outName))
    

        
    del dummyB
    del dummyT
    del padT
    del padB
    del canvas
    
 
def doFit_3Gauss(outDir, qTbin, qTbinMinGeV, qTbinMaxGeV, hist, name, label, met, cfg, hOut):
              
    if qTbinMinGeV < 30: xMin,xMax = -100, 100
    elif qTbinMinGeV < 60: xMin,xMax = -150, 50
    elif qTbinMinGeV < 100: xMin,xMax = -200, 50
    else: xMin,xMax = -500, 0
    
    outDir += "/3Gauss/"
    prepareDir(outDir, remove=False)
    
    cfg['yminR'], cfg['ymaxR'] = -10, 10
    
    hist = hist.Rebin(2)
    
    cfg['ymax'] = 1.75*hist.GetMaximum()
    
    qTlabel = "%.1f < q_{T} < %.1f GeV" % (qTbinMinGeV, qTbinMaxGeV)
    outName = "%s_%sGeV" % (str("%.1f" % qTbinMinGeV).replace(".", "p"), str("%.1f" % qTbinMaxGeV).replace(".", "p"))
    if "_gen" in outDir: qTlabel = qTlabel.replace("q_{T}", "q_{T}^{GEN}")
    
    if "para_" in name:
    
        cfg['xmin'] = xMin
        cfg['xmax'] = xMax
        
    else:
    
        cfg['xmin'] = -100
        cfg['xmax'] = 100
        
        
    
    sMax = hist.GetRMS()*3
    sMin = hist.GetRMS()/3
    

    recoil = ROOT.RooRealVar("recoil_%s" % name, "Recoil parallel (GeV)", 0, -500, 500) # independent var needed for CDF
    mean1 = ROOT.RooRealVar("mean1_%s" % name, "", 0, -300, 100)
    mean2 = ROOT.RooRealVar("mean2_%s" % name, "", 0, -300, 100)
    mean3 = ROOT.RooRealVar("mean3_%s" % name, "", 0, -300, 100)
    sigma1 = ROOT.RooRealVar("sigma1_%s" % name, "", 5, 0, 100)
    sigma2 = ROOT.RooRealVar("sigma2_%s" % name, "", 5, 0, 100)
    sigma3 = ROOT.RooRealVar("sigma3_%s" % name, "", 5, 0, 100)
    norm1 = ROOT.RooRealVar("norm1_%s" % name, "", 0.25, 0, 1)
    norm2 = ROOT.RooRealVar("norm2_%s" % name, "", 0.25, 0, 1)
    norm3 = ROOT.RooRealVar("norm3_%s" % name, "", 0.25, 0, 1)
    gauss1 = ROOT.RooGaussian("gauss1_%s" % name, "", recoil, mean1, sigma1)
    gauss2 = ROOT.RooGaussian("gauss2_%s" % name, "", recoil, mean2, sigma2)
    gauss3 = ROOT.RooGaussian("gauss3_%s" % name, "", recoil, mean3, sigma3)
    pdf = ROOT.RooAddPdf("pdf_%s" % name, '', ROOT.RooArgList(gauss1, gauss2, gauss3), ROOT.RooArgList(norm1, norm2)) # sum of 2 Gaussians

    rdh = ROOT.RooDataHist("rdh_%s" % name, "", ROOT.RooArgList(recoil), ROOT.RooFit.Import(hist))
    
    
    ## init
    #norm1.setVal(0.4)
    #norm2.setVal(0.15) # wide low norm
    #norm3.setVal(0.10) # narrow high norm
    #norm1.setConstant(ROOT.kTRUE)
    #norm2.setConstant(ROOT.kTRUE)
    #norm3.setConstant(ROOT.kTRUE)
    #sigma3.setConstant(ROOT.kTRUE)
    #sigma3.setVal(0.0)
    
    # qTbins = [0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 60.0, 80, 100.0, 10000]
    
    #if qTbin < 6: norm2.setConstant(ROOT.kTRUE)
    #else: 
    
    if "data_" in name:
   
        mean1.setVal(hist.GetMean())
        #mean1.setConstant(ROOT.kTRUE)
        
        norm1.setVal(0.45)
        #norm1.setConstant(ROOT.kTRUE)
        norm2.setVal(0.45)
        #norm2.setConstant(ROOT.kTRUE)

        sigma1.setVal(hist.GetRMS()*2)
        sigma2.setVal(hist.GetRMS()/1.5)
        sigma3.setVal(hist.GetRMS()*5)
        #sigma3.setVal(0.0)    
        
    
        #chi2 = ROOT.RooChi2Var("chi2","chi2", pdf, rdh)  # ROOT.RooFit.RooAbsData.Poisson
        #m = ROOT.RooMinuit(chi2)
        #m.migrad()
        #m.hesse()
        
          

    
        fitRes = pdf.fitTo(rdh, ROOT.RooFit.SumW2Error(ROOT.kTRUE), ROOT.RooFit.Save(ROOT.kTRUE))   
    
        mean2.setVal(mean1.getVal()) 
        mean3.setVal(mean1.getVal()) 
        
        norm3.setVal(1-norm1.getVal()-norm2.getVal())  
        
        
    if "reco" in name:
    
        mean1.setVal(hist.GetMean())
        mean2.setVal(hist.GetMean())
        mean3.setVal(hist.GetMean())
        #mean1.setConstant(ROOT.kTRUE)
        
        norm1.setVal(0.45)
        #norm1.setConstant(ROOT.kTRUE)
        norm2.setVal(0.45)
        #norm2.setConstant(ROOT.kTRUE)

        sigma1.setVal(hist.GetRMS()*2)
        sigma2.setVal(hist.GetRMS()/1.5)
        sigma3.setVal(hist.GetRMS()*4)
        #sigma3.setVal(0.0)    
        
    
        #chi2 = ROOT.RooChi2Var("chi2","chi2", pdf, rdh)  # ROOT.RooFit.RooAbsData.Poisson
        #m = ROOT.RooMinuit(chi2)
        #m.migrad()
        #m.hesse()
        
          

    
        fitRes = pdf.fitTo(rdh, ROOT.RooFit.SumW2Error(ROOT.kTRUE), ROOT.RooFit.Save(ROOT.kTRUE))   
    
        #mean2.setVal(mean1.getVal()) 
        #mean3.setVal(mean1.getVal()) 
        
        norm3.setVal(1-norm1.getVal()-norm2.getVal())  
    
        
        

    # save parameters    
    hOut.SetBinContent(qTbin+1, 1, 1, mean1.getVal())
    hOut.SetBinContent(qTbin+1, 2, 1, sigma1.getVal())
    hOut.SetBinContent(qTbin+1, 3, 1, norm1.getVal())
    hOut.SetBinContent(qTbin+1, 4, 1, mean2.getVal())
    hOut.SetBinContent(qTbin+1, 5, 1, sigma2.getVal())
    hOut.SetBinContent(qTbin+1, 6, 1, norm2.getVal())
    hOut.SetBinContent(qTbin+1, 7, 1, mean3.getVal())
    hOut.SetBinContent(qTbin+1, 8, 1, sigma3.getVal())
    hOut.SetBinContent(qTbin+1, 9, 1, norm3.getVal())
    



    plotter.cfg = cfg
    canvas, padT, padB = plotter.canvasRatio()
    dummyT, dummyB, dummyL = plotter.dummyRatio()
        
    ## TOP PAD ##
    canvas.cd()
    padT.Draw()
    padT.cd()
    dummyT.Draw("HIST")
    
    ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.ERROR)
        
    plt = recoil.frame()
    #plt.SetTitle("ZH signal")
    
    
    rdh.plotOn(plt) # , ROOT.RooFit.Normalization(yield_zh, ROOT.RooAbsReal.NumEvent) , ROOT.RooFit.Binning(200)
    
    gauss1.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kRed), ROOT.RooFit.Normalization(norm1.getVal(), ROOT.RooAbsReal.NumEvent), ROOT.RooFit.LineStyle(ROOT.kDashed))
    gauss2.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kGreen), ROOT.RooFit.Normalization(norm2.getVal(), ROOT.RooAbsReal.NumEvent), ROOT.RooFit.LineStyle(ROOT.kDashed))
    gauss3.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kBlue), ROOT.RooFit.Normalization(norm3.getVal(), ROOT.RooAbsReal.NumEvent), ROOT.RooFit.LineStyle(ROOT.kDashed))
    
    pdf.plotOn(plt, ROOT.RooFit.VisualizeError(fitRes, 1), ROOT.RooFit.Name("errorband"), ROOT.RooFit.FillColor(ROOT.kOrange))
    
    pdf.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kRed), ROOT.RooFit.Name("pdfcurve"))

    
    '''
    pdf_curve = plt.getCurve("pdfcurve")
    
    upBound = ROOT.TGraph(pdf_curve.GetN())
    lowBound = ROOT.TGraph(pdf_curve.GetN())
    
    for i in range(0, pdf_curve.GetN()+1):
    
        print(pdf_curve.GetY()[i])


    
    for( int j = 0; j < curve->GetN(); ++j ){
      if( j < central->GetN() )
          upBound->SetPoint(j, curve->GetX()[j], curve->GetY()[j]);
      else
          loBound->SetPoint(j, curve->GetX()[j], curve->GetY()[j]);
    }
    '''

    ratio = hist.Integral() / (norm1.getVal() + norm2.getVal() + norm3.getVal())
    err = ratio * math.hypot(norm1.getError(), norm2.getError(), norm3.getError()) / (norm1.getVal() + norm2.getVal() + norm3.getVal())
        
    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.040)
    latex.SetTextColor(1)
    latex.SetTextFont(42)
    latex.DrawLatex(0.20, 0.85, label)
    latex.DrawLatex(0.20, 0.80, qTlabel)
    latex.DrawLatex(0.20, 0.75, met_labels[met_estimators.index(met)])
    latex.DrawLatex(0.20, 0.70, "N_{hist}/N_{fit} = %.3f #pm %.3f" % (ratio, err))
    latex.DrawLatex(0.20, 0.65, "Mean = %.3f" % hist.GetMean())
    latex.DrawLatex(0.20, 0.60, "RMS = %.3f" % hist.GetRMS())
    
    latex.SetTextSize(0.0250)
    latex.DrawLatex(0.55, 0.86, "#mu_{1} = %.2f #pm %.2f" % (mean1.getVal(), mean1.getError()))
    latex.DrawLatex(0.55, 0.83, "#sigma_{1} = %.3f #pm %.2f" % (sigma1.getVal(), sigma1.getError()))
    latex.DrawLatex(0.55, 0.80, "N_{1} = %.2f #pm %.2f" % (norm1.getVal(), norm1.getError()))
    
    latex.DrawLatex(0.55, 0.77, "#mu_{2} = %.2f #pm %.2f" % (mean2.getVal(), mean2.getError()))
    latex.DrawLatex(0.55, 0.74, "#sigma_{2} = %.2f #pm %.2f" % (sigma2.getVal(), sigma2.getError()))
    latex.DrawLatex(0.55, 0.71, "N_{2} = %.2f #pm %.2f" % (norm2.getVal(), norm2.getError()))
    
    latex.DrawLatex(0.75, 0.86, "#mu_{3} = %.2f #pm %.2f" % (mean3.getVal(), mean3.getError()))
    latex.DrawLatex(0.75, 0.83, "#sigma_{3} = %.2f #pm %.2f" % (sigma3.getVal(), sigma3.getError()))
    latex.DrawLatex(0.75, 0.80, "N_{3} = %.2f #pm %.2f" % (norm3.getVal(), norm3.getError()))
    
    
        
    histpull = plt.pullHist()
    plt.Draw("SAME")
    '''
    histRatio = hist.Clone("ratio")
    histRatio.SetMarkerStyle(20)
    histRatio.SetMarkerSize(0.7)
    histRatio.SetMarkerColor(ROOT.kBlack)
    histRatio.SetLineColor(ROOT.kBlack)
    histRatio.SetLineWidth(1)
    norm = ROOT.RooArgSet(recoil)
    for i in range(1, histRatio.GetNbinsX()+1):

        recoil.setVal(histRatio.GetBinCenter(i))
        pdf_eval = pdf.getVal(norm)
        r = 0
        if(histRatio.GetBinContent(i) > 0): r = (pdf_eval-histRatio.GetBinContent(i))/pdf_eval
        #print(i, histRatio.GetBinCenter(i), r) # hist.GetBinContent(i), histRatio.GetBinContent(i),
        histRatio.SetBinContent(i, r)
    '''    
    plotter.auxRatio()
        
    ## BOTTOM PAD ##
    canvas.cd()
    padB.Draw()
    padB.cd()
    dummyB.Draw("HIST")
        
    #histRatio.Draw("SAME P")
    plt = recoil.frame()
    plt.addPlotable(histpull, "P")
    plt.Draw("SAME")
        
    line = ROOT.TLine(cfg['xmin'], 0, cfg['xmax'], 0)
    line.SetLineColor(ROOT.kBlue+2)
    line.SetLineWidth(2)
    line.Draw("SAME")
        
      
    canvas.Modify()
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs("%s/recoil_%s.png" % (outDir, outName))
    canvas.SaveAs("%s/recoil_%s.pdf" % (outDir, outName))
    

    
    ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)
        
    del dummyB
    del dummyT
    del padT
    del padB



def doFit_Voigtian(outDir, qTbin, qTbinMinGeV, qTbinMaxGeV, hist, name, label, met, cfg, hOut):

    if qTbinMinGeV < 30: xMin,xMax = -100, 100
    elif qTbinMinGeV < 60: xMin,xMax = -150, 50
    elif qTbinMinGeV < 100: xMin,xMax = -200, 50
    else: xMin,xMax = -300, 0
    
    outDir += "/Voigtian/"
    prepareDir(outDir, remove=False)
    
    cfg['ymax'] = 1.75*hist.GetMaximum()
    #cfg['ymax'] = 10
    
    qTlabel = "%.1f < q_{T} < %.1f GeV" % (qTbinMinGeV, qTbinMaxGeV)
    outName = "%s_%sGeV" % (str("%.1f" % qTbinMinGeV).replace(".", "p"), str("%.1f" % qTbinMaxGeV).replace(".", "p"))
    if "_gen" in outDir: qTlabel = qTlabel.replace("q_{T}", "q_{T}^{GEN}")

    if "para_" in name:
    
        cfg['xmin'] = xMin
        cfg['xmax'] = xMax
        
    else:
    
        cfg['xmin'] = -100
        cfg['xmax'] = 100

    sMax = hist.GetRMS()*2
    sMin = hist.GetRMS()/2

    # alphabetic prefix for ordering of parameters
    recoil = ROOT.RooRealVar("recoil_%s" % name, "Recoil parallel (GeV)", 0, -300, 100) # independent var needed for CDF
    
    mean = ROOT.RooRealVar("mean", "", 0, -400, 100)
    mean1 = ROOT.RooRealVar("mean1", "", 0, -400, 100)
    mean2 = ROOT.RooRealVar("mean2", "", 0, -400, 100)
    mean_v = ROOT.RooRealVar("mean_v", "", 0, -400, 100)
    sigma_v = ROOT.RooRealVar("sigma_v", "", 5, 0, 100)
    gamma_v = ROOT.RooRealVar("gamma_v", "", 1, 0, 5)
    voigt = ROOT.RooVoigtian("voigt", "", recoil, mean, gamma_v, sigma_v)
    br = ROOT.RooBreitWigner("br", "", recoil, mean, sigma_v)
    #voigt = ROOT.RooGaussian("voigt", "", recoil, mean_v, sigma_v)
    
    mean_g = ROOT.RooRealVar("mean_g", "", 0, -400, 100)
    sigma_g = ROOT.RooRealVar("sigma_g", "", 5, 0, 100)
    gauss = ROOT.RooGaussian("gauss", "", recoil, mean, sigma_g)
    
    norm = ROOT.RooRealVar("norm", "", 0.1, 0, 1)
    pdf = ROOT.RooAddPdf("pdf_%s" % name, '', ROOT.RooArgList(br, gauss), ROOT.RooArgList(norm)) # sum of 2 Gaussians
    
    
    vars_ = [mean_v, sigma_v, gamma_v, mean_g, sigma_g, norm]
    
    rdh = ROOT.RooDataHist("rdh_%s" % name, "", ROOT.RooArgList(recoil), ROOT.RooFit.Import(hist))

    
    if "reco" in name or True:
    
        mean.setVal(hist.GetMean())    
        mean_v.setVal(hist.GetMean())
        sigma_v.setVal(hist.GetRMS()*1.2)
        gamma_v.setVal(1)
        #gamma_v.setConstant(ROOT.kTRUE)
        
        mean_g.setVal(hist.GetMean())
        sigma_g.setVal(hist.GetRMS()/1.2)
        
        #norm.setVal(0.45)
        #norm.setConstant(ROOT.kTRUE)
        
        # ML fit
        fitRes = pdf.fitTo(rdh, ROOT.RooFit.SumW2Error(ROOT.kFALSE), ROOT.RooFit.Save(ROOT.kTRUE)) #  , ROOT.RooFit.Extended(ROOT.kTRUE), 
    
        #norm2.setVal(1-norm1.getVal())    
        #mean2.setVal(mean1.getVal()) 

    
    # save nominal parameters
    for k, var in enumerate(vars_):

        val = var.getVal()
        hOut.SetBinContent(qTbin+1, k+1, 1, val) 
        print(var.GetName(), qTbin+1, k+1, 1, val)    
    
    #sys.exit()
    plotter.cfg = cfg
    canvas, padT, padB = plotter.canvasRatio()
    dummyT, dummyB, dummyL = plotter.dummyRatio()
    
    print("------------")
    
    
    '''
    # diagonalize covariance matrix 
    #print(mean1.getVal(), mean1.getVal())

    cov = copy.deepcopy(fitRes.covarianceMatrix()) # covariance matrix: diagonal elements are the variances of each of the parameters
    #cov.Print()


    #covInv = copy.deepcopy(res.covarianceMatrix()) # copy necessary... 
    #covInv.Invert()
   
    eigen = ROOT.TMatrixDEigen(cov) # sorts vectors/values by eigenvalues
    eigenVectors = copy.deepcopy(eigen.GetEigenVectors()) # TMatrixD
    eigenVectorsInv = copy.deepcopy(eigen.GetEigenVectors()) # TMatrixD
    eigenVectorsInv.Invert() # TMatrixD
    eigenValues = eigen.GetEigenValuesRe() # TVectorD
    
    
    npars = len(eigenValues)
    sigma = 1
    for iVar in range(0, npars):
    
        print("vary %s", iVar)
    
        eigenValuesShifted = ROOT.TMatrixD(npars, npars) 
        for i, eig in enumerate(eigen.GetEigenValuesRe()): 
        
            if i == iVar: eigenValuesShifted[i][i] = sigma*math.sqrt(eig)
            else: eigenValuesShifted[i][i] = eig
        
        # rotate the shifted eigenvalue matrix
        new = ROOT.TMatrixD(npars, npars)
        new.Mult(eigenVectors, eigenValuesShifted)
        
        new1 = ROOT.TMatrixD(npars, npars)
        new1.Mult(new, eigenVectorsInv)
        
        
        kk = 0
        for k, var in enumerate(vars_):
        
            if var.isConstant(): err = 0
            else: 
                err = math.sqrt(new1[kk][kk])
                kk += 1
                
            if k == 3: err = math.sqrt(new1[0][0]) # mean2 = mean1
            
            hOut.SetBinContent(qTbin+1, k+1, iVar+2, err) 
            print(var.GetName(), qTbin+1, k+1, iVar+2, err)
        

    '''

  
    ## TOP PAD ##
    canvas.cd()
    padT.Draw()
    padT.cd()
    dummyT.Draw("HIST")
    
    ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.ERROR)
    plt = recoil.frame()
    rdh.plotOn(plt)
    
    voigt.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kGreen), ROOT.RooFit.Normalization(norm.getVal(), ROOT.RooAbsReal.NumEvent),  ROOT.RooFit.LineStyle(ROOT.kDashed))
    gauss.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kBlue), ROOT.RooFit.Normalization(1.-norm.getVal(), ROOT.RooAbsReal.NumEvent),  ROOT.RooFit.LineStyle(ROOT.kDashed))
    pdf.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kRed))
        
    ratio = 1
    err = 0
        
    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.040)
    latex.SetTextColor(1)
    latex.SetTextFont(42)
    latex.DrawLatex(0.20, 0.85, label)
    latex.DrawLatex(0.20, 0.80, qTlabel)
    latex.DrawLatex(0.20, 0.75, met_labels[met_estimators.index(met)])
    latex.DrawLatex(0.20, 0.70, "N_{hist}/N_{fit} = %.3f #pm %.3f" % (ratio, err))
    latex.DrawLatex(0.20, 0.65, "Mean = %.3f #pm %.3f" % (hist.GetMean(), hist.GetMeanError()))
    latex.DrawLatex(0.20, 0.60, "RMS = %.3f #pm %.3f" % (hist.GetRMS(), hist.GetRMSError()))
        
    latex.DrawLatex(0.60, 0.85, "#mu_{v} = %.3f #pm %.3f" % (mean_v.getVal(), mean_v.getError()))
    latex.DrawLatex(0.60, 0.80, "#sigma_{v} = %.3f #pm %.3f" % (sigma_v.getVal(), sigma_v.getError()))
    latex.DrawLatex(0.60, 0.75, "#gamma_{v} = %.3f #pm %.3f" % (gamma_v.getVal(), gamma_v.getError()))
    
    latex.DrawLatex(0.60, 0.70, "#mu_{g} = %.3f #pm %.3f" % (mean_g.getVal(), mean_g.getError()))
    latex.DrawLatex(0.60, 0.65, "#sigma_{g} = %.3f #pm %.3f" % (sigma_g.getVal(), sigma_g.getError()))
    latex.DrawLatex(0.60, 0.60, "N = %.3f #pm %.3f" % (norm.getVal(), norm.getError()))
    
    histpull = plt.pullHist()
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
        
    line = ROOT.TLine(120, 0, 140, 0)
    line.SetLineColor(ROOT.kBlue+2)
    line.SetLineWidth(2)
    line.Draw("SAME")

    canvas.Modify()
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs("%s/recoil_%s.png" % (outDir, outName))
    canvas.SaveAs("%s/recoil_%s.pdf" % (outDir, outName))

    ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)   


 
    dummyB.Delete()
    dummyT.Delete()
    padT.Delete()
    padB.Delete()
    
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
        #print(dnom)
     
    return ret
    
def doFit_cumulative(outDir, qTbin, qTbinMinGeV, qTbinMaxGeV, hist, name, label, cfg, hOut=None, refittedPdf=None):
    

    if qTbinMinGeV < 30: xMin,xMax = -100, 100
    elif qTbinMinGeV < 60: xMin,xMax = -150, 50
    elif qTbinMinGeV < 100: xMin,xMax = -200, 50
    else: xMin,xMax = -500, 0
    if "perp" in name: xMin,xMax = -100, 100
    
    #hist = hist.Rebin(1)
    #hist.Scale(1./hist.Integral())    
    histCumFw = hist.GetCumulative(ROOT.kTRUE)
    histCumBw = hist.GetCumulative(ROOT.kFALSE)
    
    for i in range(0, histCumFw.GetNbinsX()+1):
        #if hist.GetBinCenter(i) > -100 and hist.GetBinCenter(i) < 100:
        print(i, hist.GetBinCenter(i), hist.GetBinContent(i), hist.GetBinError(i), histCumFw.GetBinContent(i), histCumBw.GetBinContent(i))
    
    return
    
    qT = 0.5*(qTbinMinGeV + qTbinMaxGeV)
    

    qTlabel = "%.1f < q_{T} < %.1f GeV" % (qTbinMinGeV, qTbinMaxGeV)
    outName = "%s_%sGeV" % (str("%.1f" % qTbinMinGeV).replace(".", "p"), str("%.1f" % qTbinMaxGeV).replace(".", "p"))
    if "_gen" in outDir: qTlabel = qTlabel.replace("q_{T}", "q_{T}^{GEN}")

    cfg['xmin'], cfg['xmax'] = xMin, xMax
    cfg['ymax'] = 1.75*hist.GetMaximum()
    


    recoil = ROOT.RooRealVar("recoil", "Recoil parallel (GeV)", 0, -500, 500)
    mean = ROOT.RooRealVar("mean", "", 0, -400, 100)
    sigma1 = ROOT.RooRealVar("sigma1", "", 5, 0.1, 100)
    sigma2 = ROOT.RooRealVar("sigma2", "", 5, 0.1, 100)
    norm = ROOT.RooRealVar("norm", "", 0, 0, 1)
    gauss1 = ROOT.RooGaussian("gauss1", "gauss1", recoil, mean, sigma1)
    gauss2 = ROOT.RooGaussian("gauss2", "gauss2", recoil, mean, sigma2)
    pdf = ROOT.RooAddPdf("pdf", '', ROOT.RooArgList(gauss1, gauss2), ROOT.RooArgList(norm)) # sum of 2 Gaussians
    rdh = ROOT.RooDataHist("rdh", "", ROOT.RooArgList(recoil), ROOT.RooFit.Import(hist))
    
    params = [mean, sigma1, sigma2, norm]
    
    mean.setVal(hist.GetMean())
    #mean.setConstant(ROOT.kTRUE)
        
    norm.setVal(0.6)
    norm.setConstant(ROOT.kTRUE)
        
    sigma1.setVal(hist.GetRMS()/1.5)
    #sigma1.setConstant(ROOT.kTRUE)
        
    sigma2.setVal(hist.GetRMS()*1.5)
    #sigma2.setConstant(ROOT.kTRUE)
        
    if "perp" in name:
        mean.setVal(0)
        mean.setConstant(ROOT.kTRUE)


    # ML fit
    fitRes = pdf.fitTo(rdh, ROOT.RooFit.SumW2Error(ROOT.kTRUE), ROOT.RooFit.Save(ROOT.kTRUE)) # ROOT.RooFit.Extended(ROOT.kTRUE), 
   
        
    print("***** Summary for qTbin %d  *****" % qTbin)
    fitRes.floatParsFinal().Print("s")

    
    if hOut != None:
        for iPar, p in enumerate(params):
            hOut.SetBinContent(qTbin, iPar+1, 1, p.getVal())
            hOut.SetBinContent(qTbin, iPar+1, 0, p.getError())


    plotter.cfg = cfg
    canvas, padT, padB = plotter.canvasRatio()
    dummyT, dummyB, dummyL = plotter.dummyRatio()
    
    
    ## TOP PAD ##
    canvas.cd()
    padT.Draw()
    padT.cd()
    dummyT.Draw("HIST")
    
    ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.ERROR)
    plt = recoil.frame()
    rdh.plotOn(plt)
    gauss1.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kBlue), ROOT.RooFit.Normalization(norm.getVal(), ROOT.RooAbsReal.NumEvent), ROOT.RooFit.LineStyle(ROOT.kDashed))
    gauss2.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kGreen), ROOT.RooFit.Normalization(1.-norm.getVal(), ROOT.RooAbsReal.NumEvent), ROOT.RooFit.LineStyle(ROOT.kDashed))
    
    pdf.plotOn(plt, ROOT.RooFit.VisualizeError(fitRes, 1), ROOT.RooFit.Name("errorband"), ROOT.RooFit.FillColor(ROOT.kOrange))
    pdf.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kRed))
    histpull = plt.pullHist()
    chi2 = plt.chiSquare() 
    
    if refittedPdf != None: 
        refittedPdf.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kCyan))
        histpull_refit = plt.pullHist()
        
        
    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.040)
    latex.SetTextColor(1)
    latex.SetTextFont(42)
    latex.DrawLatex(0.20, 0.85, label)
    latex.DrawLatex(0.20, 0.80, qTlabel)
    latex.DrawLatex(0.20, 0.75, "Mean = %.3f #pm %.3f" % (hist.GetMean(), hist.GetMeanError()))
    latex.DrawLatex(0.20, 0.70, "RMS = %.3f #pm %.3f" % (hist.GetRMS(), hist.GetRMSError()))
        
    latex.DrawLatex(0.60, 0.85, "#mu = %.3f #pm %.3f" % (mean.getVal(), mean.getError()))
    latex.DrawLatex(0.60, 0.80, "#sigma_{1} = %.3f #pm %.3f" % (sigma1.getVal(), sigma1.getError()))
    latex.DrawLatex(0.60, 0.75, "#sigma_{2} = %.3f #pm %.3f" % (sigma2.getVal(), sigma2.getError()))
    latex.DrawLatex(0.60, 0.70, "N = %.3f #pm %.3f" % (norm.getVal(), norm.getError()))
    latex.DrawLatex(0.60, 0.65, "#chi^{2}/ndof = %.3f" % chi2)

    plt.Draw("SAME")
    plotter.auxRatio()
       
    ## BOTTOM PAD ##
    canvas.cd()
    padB.Draw()
    padB.cd()
    dummyB.Draw("HIST")
    
    plt = recoil.frame()
    plt.addPlotable(histpull, "P")
    if refittedPdf != None: plt.addPlotable(histpull_refit, "P")

    plt.Draw("SAME")
        
    line = ROOT.TLine(120, 0, 140, 0)
    line.SetLineColor(ROOT.kBlue+2)
    line.SetLineWidth(2)
    line.Draw("SAME")

    canvas.Modify()
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs("%s/%03d_recoil_%s.png" % (outDir, qTbin, outName))
    canvas.SaveAs("%s/%03d_recoil_%s.pdf" % (outDir, qTbin, outName))

    ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)   

    dummyB.Delete()
    dummyT.Delete()
    padT.Delete()
    padB.Delete()
    
    if hOut == None: return
    
    # diagonalize covariance matrix and store perturbations
    floatingParams = fitRes.floatParsFinal() # floating parameters
    variations = diagonalize(fitRes)
    
    for iVar, var in enumerate(variations):
        for iPar, p in enumerate(params):
            if getParamIdx(p.GetName(), floatingParams) > -1: val = var[getParamIdx(p.GetName(), floatingParams)]
            else: val = p.getVal()
            hOut.SetBinContent(qTbin, iPar+1, iVar+2, val)
            print(iVar, iPar, p.GetName(), val)


def doFit_2Gauss_DY(outDir, qTbin, qTbinMinGeV, qTbinMaxGeV, hist, name, label, cfg, hOut=None, refittedPdf=None):
    

    if qTbinMinGeV < 30: xMin,xMax = -100, 100
    elif qTbinMinGeV < 60: xMin,xMax = -150, 50
    elif qTbinMinGeV < 100: xMin,xMax = -200, 50
    else: xMin,xMax = -500, 0
    if "perp" in name: xMin,xMax = -100, 100
    
    #hist = hist.Rebin(1)
    hist.Scale(1./hist.Integral())    
    
    for i in range(0, hist.GetNbinsX()+1):
        if hist.GetBinContent(i) <= 0:  hist.SetBinError(i, 0)
    
    
    qT = 0.5*(qTbinMinGeV + qTbinMaxGeV)
    

    qTlabel = "%.1f < q_{T} < %.1f GeV" % (qTbinMinGeV, qTbinMaxGeV)
    outName = "%s_%sGeV" % (str("%.1f" % qTbinMinGeV).replace(".", "p"), str("%.1f" % qTbinMaxGeV).replace(".", "p"))
    if "_gen" in outDir: qTlabel = qTlabel.replace("q_{T}", "q_{T}^{GEN}")

    cfg['xmin'], cfg['xmax'] = xMin, xMax
    cfg['ymax'] = 1.75*hist.GetMaximum()
    


    recoil = ROOT.RooRealVar("recoil", "Recoil parallel (GeV)", 0, -500, 500)
    mean = ROOT.RooRealVar("mean", "", 0, -400, 100)
    sigma1 = ROOT.RooRealVar("sigma1", "", 5, 0.1, 100)
    sigma2 = ROOT.RooRealVar("sigma2", "", 5, 0.1, 100)
    norm = ROOT.RooRealVar("norm", "", 0, 0, 1)
    gauss1 = ROOT.RooGaussian("gauss1", "gauss1", recoil, mean, sigma1)
    gauss2 = ROOT.RooGaussian("gauss2", "gauss2", recoil, mean, sigma2)
    pdf = ROOT.RooAddPdf("pdf", '', ROOT.RooArgList(gauss1, gauss2), ROOT.RooArgList(norm)) # sum of 2 Gaussians
    rdh = ROOT.RooDataHist("rdh", "", ROOT.RooArgList(recoil), ROOT.RooFit.Import(hist))
    
    params = [mean, sigma1, sigma2, norm]
    
    mean.setVal(hist.GetMean())
    #mean.setConstant(ROOT.kTRUE)
        
    norm.setVal(0.6)
    norm.setConstant(ROOT.kTRUE)
        
    sigma1.setVal(hist.GetRMS()/1.5)
    #sigma1.setConstant(ROOT.kTRUE)
        
    sigma2.setVal(hist.GetRMS()*1.5)
    #sigma2.setConstant(ROOT.kTRUE)
        
    if "perp" in name:
        mean.setVal(0)
        mean.setConstant(ROOT.kTRUE)


    # ML fit
    fitRes = pdf.fitTo(rdh, ROOT.RooFit.SumW2Error(ROOT.kTRUE), ROOT.RooFit.Save(ROOT.kTRUE)) # ROOT.RooFit.Extended(ROOT.kTRUE), 
   
        
    print("***** Summary for qTbin %d  *****" % qTbin)
    fitRes.floatParsFinal().Print("s")

    
    if hOut != None:
        for iPar, p in enumerate(params):
            hOut.SetBinContent(qTbin, iPar+1, 1, p.getVal())
            hOut.SetBinContent(qTbin, iPar+1, 0, p.getError())


    plotter.cfg = cfg
    canvas, padT, padB = plotter.canvasRatio()
    dummyT, dummyB, dummyL = plotter.dummyRatio()
    
    
    ## TOP PAD ##
    canvas.cd()
    padT.Draw()
    padT.cd()
    dummyT.Draw("HIST")
    
    ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.ERROR)
    plt = recoil.frame()
    rdh.plotOn(plt)
    gauss1.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kBlue), ROOT.RooFit.Normalization(norm.getVal(), ROOT.RooAbsReal.NumEvent), ROOT.RooFit.LineStyle(ROOT.kDashed))
    gauss2.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kGreen), ROOT.RooFit.Normalization(1.-norm.getVal(), ROOT.RooAbsReal.NumEvent), ROOT.RooFit.LineStyle(ROOT.kDashed))
    
    pdf.plotOn(plt, ROOT.RooFit.VisualizeError(fitRes, 1), ROOT.RooFit.Name("errorband"), ROOT.RooFit.FillColor(ROOT.kOrange))
    pdf.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kRed))
    histpull = plt.pullHist()
    chi2 = plt.chiSquare() 
    
    if refittedPdf != None: 
        refittedPdf.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kCyan))
        histpull_refit = plt.pullHist()
        
        
    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.040)
    latex.SetTextColor(1)
    latex.SetTextFont(42)
    latex.DrawLatex(0.20, 0.85, label)
    latex.DrawLatex(0.20, 0.80, qTlabel)
    latex.DrawLatex(0.20, 0.75, "Mean = %.3f #pm %.3f" % (hist.GetMean(), hist.GetMeanError()))
    latex.DrawLatex(0.20, 0.70, "RMS = %.3f #pm %.3f" % (hist.GetRMS(), hist.GetRMSError()))
        
    latex.DrawLatex(0.60, 0.85, "#mu = %.3f #pm %.3f" % (mean.getVal(), mean.getError()))
    latex.DrawLatex(0.60, 0.80, "#sigma_{1} = %.3f #pm %.3f" % (sigma1.getVal(), sigma1.getError()))
    latex.DrawLatex(0.60, 0.75, "#sigma_{2} = %.3f #pm %.3f" % (sigma2.getVal(), sigma2.getError()))
    latex.DrawLatex(0.60, 0.70, "N = %.3f #pm %.3f" % (norm.getVal(), norm.getError()))
    latex.DrawLatex(0.60, 0.65, "#chi^{2}/ndof = %.3f" % chi2)

    plt.Draw("SAME")
    plotter.auxRatio()
       
    ## BOTTOM PAD ##
    canvas.cd()
    padB.Draw()
    padB.cd()
    dummyB.Draw("HIST")
    
    plt = recoil.frame()
    plt.addPlotable(histpull, "P")
    if refittedPdf != None: plt.addPlotable(histpull_refit, "P")

    plt.Draw("SAME")
        
    line = ROOT.TLine(120, 0, 140, 0)
    line.SetLineColor(ROOT.kBlue+2)
    line.SetLineWidth(2)
    line.Draw("SAME")

    canvas.Modify()
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs("%s/%03d_recoil_%s.png" % (outDir, qTbin, outName))
    canvas.SaveAs("%s/%03d_recoil_%s.pdf" % (outDir, qTbin, outName))

    ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)   

    dummyB.Delete()
    dummyT.Delete()
    padT.Delete()
    padB.Delete()
    
    if hOut == None: return
    
    # diagonalize covariance matrix and store perturbations
    floatingParams = fitRes.floatParsFinal() # floating parameters
    variations = diagonalize(fitRes)
    
    for iVar, var in enumerate(variations):
        for iPar, p in enumerate(params):
            if getParamIdx(p.GetName(), floatingParams) > -1: val = var[getParamIdx(p.GetName(), floatingParams)]
            else: val = p.getVal()
            hOut.SetBinContent(qTbin, iPar+1, iVar+2, val)
            print(iVar, iPar, p.GetName(), val)

    
def doFit_2GaussConstrained(outDir, qTbin, qTbinMinGeV, qTbinMaxGeV, hist, name, label, met, cfg, hOut):

    if qTbinMinGeV < 30: xMin,xMax = -100, 100
    elif qTbinMinGeV < 60: xMin,xMax = -150, 50
    elif qTbinMinGeV < 100: xMin,xMax = -200, 50
    else: xMin,xMax = -500, 0
    
    outDir += "/2Gauss/"
    prepareDir(outDir, remove=False)
    
    qT = 0.5*(qTbinMinGeV + qTbinMaxGeV)

    
    cfg['ymax'] = 1.75*hist.GetMaximum()
    cfg['yminR'], cfg['ymaxR'] = -5.5, 5.5
    
    #cfg['yminR'], cfg['ymaxR'] = 0.9, 1.1
    

    
    qTlabel = "%.1f < q_{T} < %.1f GeV" % (qTbinMinGeV, qTbinMaxGeV)
    outName = "%s_%sGeV" % (str("%.1f" % qTbinMinGeV).replace(".", "p"), str("%.1f" % qTbinMaxGeV).replace(".", "p"))
    if "_gen" in outDir: qTlabel = qTlabel.replace("q_{T}", "q_{T}^{GEN}")

    if "para_" in name:
    
        cfg['xmin'] = xMin
        cfg['xmax'] = xMax
        
    else:
    
        cfg['xmin'] = -100
        cfg['xmax'] = 100

    

    # alphabetic prefix for ordering of parameters
    recoil = ROOT.RooRealVar("recoil", "Recoil parallel (GeV)", 0, -500, 500) # independent var needed for CDF
    
    sigma = ROOT.RooRealVar("sigma", "", hist.GetRMS(), 0, 100)
    sigma.setConstant(ROOT.kTRUE)
    
    mean = ROOT.RooRealVar("mean", "", hist.GetMean(), -300, 100)
    mean1 = ROOT.RooRealVar("mean1", "", hist.GetMean(), -300, 100)
    mean2 = ROOT.RooRealVar("mean2", "", hist.GetMean(), -300, 100)
    #mean.setConstant(ROOT.kTRUE)
    
    sigma1 = ROOT.RooRealVar("a_sigma1", "", 5, 5, 50)
    sigma2 = ROOT.RooRealVar("b_sigma2", "", 5, 5, 50)
    
    sigma1.setVal(hist.GetRMS()/2)
    sigma2.setVal(hist.GetRMS()*2) 
    
    if "data" in name:
        pass
        #m = (-5.91330e-02*qT**2 + -4.98983e-01*qT) / (6.28090e-02*qT + 1)
        #m = (-5.14250e-06*qT*qT*qT + -4.95061e-02*qT*qT + -5.07584e-01*qT)/(3.37032e-05*qT*qT + 5.00865e-02*qT + 1)
        #mean.setVal(m)
        #mean.setConstant(ROOT.kTRUE)
        
        #s = 6.27502e-01*math.pow(qT + 5.26520e+01, 6.51829e-01)
        #sigma.setVal(s)
        #sigma.setConstant(ROOT.kTRUE)
        
        #s1 =3.60998e+00*math.pow(qT + 4.48133e+00, 2.95892e-01)
        #sigma1.setVal(s1)
        #sigma1.setConstant(ROOT.kTRUE)
        
        #s2 = 73.1176*math.exp(- -0.00609622*qT) + -64.8956
        #sigma2.setVal(s2)
        #sigma2.setConstant(ROOT.kTRUE)
        
    if "reco" in name: 
        pass
        #m = (-1.05968e-01*qT**2 + -4.63625e-01*qT) / (1.09285e-01*qT + 1)
        #m = (-1.88706e-02*qT*qT*qT + -1.84548e-02*qT*qT + -6.26865e-01*qT)/(2.00519e-02*qT*qT + 7.34363e-02*qT + 1)
        #mean.setVal(m)
        #mean.setConstant(ROOT.kTRUE)
        
        #s = 5.17314e-01*math.pow(qT + 6.26437e+01, 6.53109e-01)
        #sigma.setVal(s)
        #sigma.setConstant(ROOT.kTRUE)
        
        #s1 = 1.25773e+00*math.pow(qT + 1.91805e+01, 4.89662e-01)
        #sigma1.setVal(s1)
        #sigma1.setConstant(ROOT.kTRUE)
    
        #s2 = 2.00780e+01*math.exp(- -5.60455e-03*qT) + -1.00487e+01
        #sigma2.setVal(s2)
        #sigma2.setConstant(ROOT.kTRUE)
    
    norm = ROOT.RooFormulaVar("norm", "(@0*@0 - @2*@2)/(@1*@1 - @2*@2)", ROOT.RooArgList(sigma, sigma1, sigma2))
    #norm = ROOT.RooRealVar("norm", "", 1, 0, 1)
    #norm.setConstant(ROOT.kTRUE)
    gauss1 = ROOT.RooGaussian("gauss1", "gauss1", recoil, mean1, sigma1)
    gauss2 = ROOT.RooGaussian("gauss2", "gauss2", recoil, mean1, sigma2)
    pdf = ROOT.RooAddPdf("pdf", '', ROOT.RooArgList(gauss1, gauss2), ROOT.RooArgList(norm)) # sum of 2 Gaussians
    
    rdh = ROOT.RooDataHist("rdh", "", ROOT.RooArgList(recoil), ROOT.RooFit.Import(hist))
    
    
    if "perp" in name:
    
        mean.setVal(0)
        mean.setConstant(ROOT.kTRUE)
        
        mean1.setVal(0)
        mean1.setConstant(ROOT.kTRUE)
        mean2.setVal(0)
        mean2.setConstant(ROOT.kTRUE)
        

    fitRes = pdf.fitTo(rdh, ROOT.RooFit.SumW2Error(ROOT.kTRUE), ROOT.RooFit.Save(ROOT.kTRUE)) # ROOT.RooFit.Extended(ROOT.kTRUE), 
    
    #norm2.setVal(1-norm1.getVal())    
    #mean2.setVal(mean1.getVal()) 
    
    ''' 
    s1 = sigma1.getVal()
    s2 = sigma2.getVal()
    s = hist.GetRMS()
    ds1 = sigma1.getError()
    ds2 = sigma2.getError()
    ds = hist.GetRMSError()
    
    #print(s1, s2, s, ds1, ds1**2, ds2, ds2**2)
    norm_err = norm.getVal() * math.sqrt(  (math.sqrt((2.*s*ds)**2 + (2.*s2*ds2)**2)/(s**2 - s2**2))**2 + (math.sqrt((2.*s1*ds1)**2 + (2.*s2*ds2)**2)/(s1**2 - s2**2))**2  )
    norm_err = 0
    if abs(s1-s) > 1e-4: norm_err = norm.getVal() * math.sqrt(  ((2.*s*ds)**2 + (2.*s2*ds2)**2)/(s**2 - s2**2)**2 + ((2.*s1*ds1)**2 + (2.*s2*ds2)**2)/(s1**2 - s2**2)**2  )
    print(norm_err)
    if fit: norm_err = norm.getPropagatedError(fitRes)
    else: norm_err = 0
    print(norm_err)
    '''
    
    norm_err = 0 # norm.getError()
    
    hOut.SetBinContent(qTbin+1, 1, 1, mean1.getVal())
    hOut.SetBinContent(qTbin+1, 2, 1, sigma1.getVal())
    hOut.SetBinContent(qTbin+1, 3, 1, norm.getVal())
    hOut.SetBinContent(qTbin+1, 4, 1, mean1.getVal())
    hOut.SetBinContent(qTbin+1, 5, 1, sigma2.getVal())
    hOut.SetBinContent(qTbin+1, 6, 1, (1.0-norm.getVal()))
    hOut.SetBinContent(qTbin+1, 7, 1, sigma.getVal())
    
    hOut.SetBinContent(qTbin+1, 1, 2, mean1.getError())
    hOut.SetBinContent(qTbin+1, 2, 2, sigma1.getError())
    hOut.SetBinContent(qTbin+1, 3, 2, norm_err)
    hOut.SetBinContent(qTbin+1, 4, 2, mean1.getError())
    hOut.SetBinContent(qTbin+1, 5, 2, sigma2.getError())
    hOut.SetBinContent(qTbin+1, 6, 2, norm_err)
    hOut.SetBinContent(qTbin+1, 7, 2, sigma.getError())

    plotter.cfg = cfg
    canvas, padT, padB = plotter.canvasRatio()
    dummyT, dummyB, dummyL = plotter.dummyRatio()
    
    
    ## TOP PAD ##
    canvas.cd()
    padT.Draw()
    padT.cd()
    dummyT.Draw("HIST")
    
    ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.ERROR)
    plt = recoil.frame()
    rdh.plotOn(plt, ROOT.RooFit.Name("data"), ROOT.RooFit.DataError(ROOT.RooAbsData.SumW2))
    gauss1.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kBlue), ROOT.RooFit.Normalization(norm.getVal(), ROOT.RooAbsReal.NumEvent), ROOT.RooFit.LineStyle(ROOT.kDashed))
    gauss2.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kGreen), ROOT.RooFit.Normalization(1.-norm.getVal(), ROOT.RooAbsReal.NumEvent), ROOT.RooFit.LineStyle(ROOT.kDashed))
    
    pdf.plotOn(plt, ROOT.RooFit.VisualizeError(fitRes, 1), ROOT.RooFit.Name("errorband"), ROOT.RooFit.FillColor(ROOT.kOrange))
    pdf.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kRed), ROOT.RooFit.Name("pdf"))
    
    # store chi2
    
    #if fit: chi2 = plt.chiSquare("pdf", "data", fitRes.floatParsFinal().getSize())
    #else: chi2 = plt.chiSquare("pdf", "data")
    #hOut.SetBinContent(qTbin+1, 0, 0, chi2)
    chi2 = 0

        
    ratio = hist.Integral() / (norm.getVal() + norm.getVal())
    #err = ratio * math.hypot(norm.getError()+norm.getError()) / (norm.getVal() + norm.getVal())
    err=0
        
    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.040)
    latex.SetTextColor(1)
    latex.SetTextFont(42)
    latex.DrawLatex(0.20, 0.85, label)
    latex.DrawLatex(0.20, 0.80, qTlabel)
    latex.DrawLatex(0.20, 0.75, met_labels[met_estimators.index(met)])
    latex.DrawLatex(0.20, 0.70, "N_{hist}/N_{fit} = %.3f #pm %.3f" % (ratio, err))
    latex.DrawLatex(0.20, 0.65, "Mean = %.3f #pm %.3f" % (hist.GetMean(), hist.GetMeanError()))
    latex.DrawLatex(0.20, 0.60, "RMS = %.3f #pm %.3f" % (hist.GetRMS(), hist.GetRMSError()))
        
    latex.DrawLatex(0.60, 0.85, "#mu = %.3f #pm %.3f" % (mean.getVal(), mean.getError()))
    latex.DrawLatex(0.60, 0.80, "#sigma = %.3f #pm %.3f" % (sigma.getVal(), sigma.getError()))
    latex.DrawLatex(0.60, 0.75, "#sigma_{1} = %.3f #pm %.3f" % (sigma1.getVal(), sigma1.getError()))
    latex.DrawLatex(0.60, 0.70, "#sigma_{2} = %.3f #pm %.3f" % (sigma2.getVal(), sigma2.getError()))
    latex.DrawLatex(0.60, 0.65, "n = %.3f" % (norm.getVal()))
    latex.DrawLatex(0.60, 0.60, "#chi^{2} = %.3f" % chi2)
    
    
    histpull = plt.pullHist()
    #histpull = plt.residHist()
    plt.Draw("SAME")
    

    histRatio = hist.Clone("ratio")
    histRatio.SetMarkerStyle(20)
    histRatio.SetMarkerSize(0.7)
    histRatio.SetMarkerColor(ROOT.kBlack)
    histRatio.SetLineColor(ROOT.kBlack)
    histRatio.SetLineWidth(1)
    

    norm_ = ROOT.RooArgSet(recoil)
    for i in range(1, histRatio.GetNbinsX()+1):

        recoil.setVal(histRatio.GetBinCenter(i))
        pdf_eval = pdf.getVal(norm_)
        r = 0
        errr = 0
        if(histRatio.GetBinContent(i) > 0): 
            r = pdf_eval/histRatio.GetBinContent(i)
            errr = r*histRatio  .GetBinError(i)/histRatio.GetBinContent(i)
        #if(histRatio.GetBinContent(i) > 0): errr = 0.5*(abs((pdf_eval-histRatio.GetBinContent(i)-histRatio.GetBinError(i))/pdf_eval) + abs((pdf_eval-histRatio.GetBinContent(i)+histRatio.GetBinError(i)))/pdf_eval)
        #print(i, histRatio.GetBinCenter(i), r) # hist.GetBinContent(i), histRatio.GetBinContent(i),
        histRatio.SetBinContent(i, r)
        histRatio.SetBinError(i, 0, errr)


    plotter.auxRatio()
       
    ## BOTTOM PAD ##
    canvas.cd()
    padB.Draw()
    padB.cd()
    dummyB.Draw("HIST")
      
    
    #histRatio.Draw("SAME P")
    
    plt = recoil.frame()
    plt.addPlotable(histpull, "P")
    plt.Draw("SAME")
        
    line = ROOT.TLine(cfg['xmin'], 0, cfg['xmax'], 0)
    line.SetLineColor(ROOT.kBlue+2)
    line.SetLineWidth(2)
    line.Draw("SAME")

    canvas.Modify()
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs("%s/recoil_%s.png" % (outDir, outName))
    canvas.SaveAs("%s/recoil_%s.pdf" % (outDir, outName))

    ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)   

    
    dummyB.Delete()
    dummyT.Delete()
    padT.Delete()
    padB.Delete()
    
    return
    
    # diagonalize covariance matrix 
    cov = fitRes.covarianceMatrix() # covariance matrix: diagonal elements are the variances of each of the parameters
    #cov.Print()
    npars = fitRes.floatParsFinal().getSize()
    sigma = 1
    
   
    nom = np.zeros(npars)
    cov_ = np.zeros((npars, npars))
    for i in range(0, npars):
        for j in range(0, npars):
            cov_[i][j] = cov[i][j]
     
    
    # fill the nominal
    i = 0
    for k, var in enumerate(vars_):
    
        if var.isConstant(): continue
        nom[i] = var.getVal()
        i += 1

   
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
    
    for iVar in range(0, npars):
    
        print("******** vary", iVar+1, "********")
        
        dnom = copy.deepcopy(eig_vals)
        for i, eig in enumerate(dnom): 
       
            if i == iVar: dnom[i] = sigma*math.sqrt(eig)
            else: dnom[i] = 0
           
        #print("PERTURBED NON-ROTATED")
        #print(dnom)
        
        # rotate perturbation back to nominal base
        dnom = np.dot(eig_vec, dnom)
        nom_pert = nom + dnom
        #print("PERTURBED ROTATED")
        #print(dnom)
  
        
        
        i = 0
        for k, var in enumerate(vars_):
        
            if var.isConstant(): continue
            var.setVal(nom_pert[i])
            i += 1

        mean2.setVal(mean1.getVal())
        norm2.setVal(1. - norm1.getVal())
        
        
        # save
        for k, var in enumerate(vars_):

            val = var.getVal()
            print(var.GetName(), "\t", qTbin+1, k+1, iVar+2, val) # iVar=1 == nominal
            hOut.SetBinContent(qTbin+1, k+1, iVar+2, val)
    
    
  
def doFit_2Gauss_data(outDir, qTbin, qTbinMinGeV, qTbinMaxGeV, hist_data, hist_bkg, name, label, cfg, hOut=None, refittedPdf=None):

    if qTbinMinGeV < 30: xMin,xMax = -100, 100
    elif qTbinMinGeV < 60: xMin,xMax = -150, 50
    elif qTbinMinGeV < 100: xMin,xMax = -200, 50
    else: xMin,xMax = -500, 0
    if "perp" in name: xMin,xMax = -100, 100
    
    qT = 0.5*(qTbinMinGeV + qTbinMaxGeV)
    
    # store the normalisations and normalize
    int_bkg = hist_bkg.Integral()
    int_data = hist_data.Integral()
    if int_bkg > 0: hist_bkg.Scale(1./int_bkg)
    hist_data.Scale(1./int_data)
    
    
    
    qTlabel = "%.1f < q_{T} < %.1f GeV" % (qTbinMinGeV, qTbinMaxGeV)
    outName = "%s_%sGeV" % (str("%.1f" % qTbinMinGeV).replace(".", "p"), str("%.1f" % qTbinMaxGeV).replace(".", "p"))
    if "_gen" in outDir: qTlabel = qTlabel.replace("q_{T}", "q_{T}^{GEN}")

    recoil = ROOT.RooRealVar("recoil", "Recoil parallel (GeV)", 0, -500, 500)
    
    # background-only fit (parameters are frozen after)
    doFit_bkg(outDir, qTbin, qTbinMinGeV, qTbinMaxGeV, hist_bkg, name, label, cfg)
    pdf_bkg = w.obj("pdf_bkg_%s_qTbin%d" % (name, qTbin))
    
    #recoil = ROOT.RooRealVar("recoil", "Recoil parallel (GeV)", 0, -500, 500) # independent var needed for CDF
    #rdh_bkg = ROOT.RooDataHist("rdh_bkg_qTbin%d" % qTbin, "", ROOT.RooArgList(recoil), ROOT.RooFit.Import(hist))
    #pdf_bkg = ROOT.RooNDKeysPdf("pdf_bkg_%s_qTbin%d" % (name, qTbin), "", ROOT.RooArgList(recoil), hist_bkg, "", 0.2, 2)
    
    #pdf_bkg.Print()
    

    # signal pdf
    norm = ROOT.RooRealVar("norm_qTbin%d" % qTbin, "", 0.5, 0, 1)
    norm.setVal(0.6)
    norm.setConstant(ROOT.kTRUE)
    
    mean = ROOT.RooRealVar("mean_qTbin%d" % qTbin, "", hist_data.GetMean(), -500, 500)
    
    sigma1 = ROOT.RooRealVar("sigma1_qTbin%d" % qTbin, "", 5, 0.1, 50)
    sigma2 = ROOT.RooRealVar("sigma2_qTbin%d" % qTbin, "", 5, 0.1, 50)
    
    sigma1.setVal(hist_data.GetRMS()/1.5)
    sigma2.setVal(hist_data.GetRMS()*1.5)
    
    if "perp" in name:
        mean.setVal(0)
        mean.setConstant(ROOT.kTRUE)

    gauss1 = ROOT.RooGaussian("gauss1_qTbin%d" % qTbin, "", recoil, mean, sigma1)
    gauss2 = ROOT.RooGaussian("gauss2_qTbin%d" % qTbin, "", recoil, mean, sigma2)
    pdf_sig = ROOT.RooAddPdf("pdf_sig_qTbin%d" % qTbin, '', ROOT.RooArgList(gauss1, gauss2), ROOT.RooArgList(norm))
   
    norm_sig = ROOT.RooRealVar("norm_sig_qTbin%d" % qTbin, "", (1. - int_bkg/int_data), 0, 1)
    #norm_sig.setConstant(ROOT.kTRUE)
    
    pdf_tot = ROOT.RooAddPdf("pdf_tot_qTbin%d" % qTbin, '', ROOT.RooArgList(pdf_sig, pdf_bkg), ROOT.RooArgList(norm_sig))
    rdh_tot = ROOT.RooDataHist("rdh_tot_qTbin%d" % qTbin, "", ROOT.RooArgList(recoil), ROOT.RooFit.Import(hist_data))
    fitRes = pdf_tot.fitTo(rdh_tot, ROOT.RooFit.SumW2Error(ROOT.kTRUE), ROOT.RooFit.Save(ROOT.kTRUE))
    
    params = [mean, sigma1, sigma2, norm]

    if hOut != None:
    
        hOut.SetBinContent(qTbin, 1, 1, mean.getVal())
        hOut.SetBinContent(qTbin, 2, 1, sigma1.getVal())
        hOut.SetBinContent(qTbin, 3, 1, sigma2.getVal())
        hOut.SetBinContent(qTbin, 4, 1, norm.getVal())
        
        hOut.SetBinContent(qTbin, 1, 0, mean.getError())
        hOut.SetBinContent(qTbin, 2, 0, sigma1.getError())
        hOut.SetBinContent(qTbin, 3, 0, sigma2.getError())
        hOut.SetBinContent(qTbin, 4, 0, norm.getError())

    
    cfg['ymax'] = 1.75*hist_data.GetMaximum()
    cfg['xmin'], cfg['xmax'] = xMin,xMax
    plotter.cfg = cfg
    canvas, padT, padB = plotter.canvasRatio()
    dummyT, dummyB, dummyL = plotter.dummyRatio()
    
    
    ## TOP PAD ##
    canvas.cd()
    padT.Draw()
    padT.cd()
    dummyT.Draw("HIST")
    
    ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.ERROR)
    plt = recoil.frame()
    rdh_tot.plotOn(plt, ROOT.RooFit.Name("data"), ROOT.RooFit.DataError(ROOT.RooAbsData.SumW2))
    gauss1.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kBlue), ROOT.RooFit.Normalization(norm.getVal(), ROOT.RooAbsReal.NumEvent), ROOT.RooFit.LineStyle(ROOT.kDashed))
    gauss2.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kGreen), ROOT.RooFit.Normalization(1.-norm.getVal(), ROOT.RooAbsReal.NumEvent), ROOT.RooFit.LineStyle(ROOT.kDashed))
    
    
    pdf_bkg.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kCyan), ROOT.RooFit.Name("pdf_bkg"))
    #pdf_tot.plotOn(plt, ROOT.RooFit.VisualizeError(fitRes, 1), ROOT.RooFit.Name("errorband"), ROOT.RooFit.FillColor(ROOT.kOrange))
    pdf_tot.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kRed), ROOT.RooFit.Name("pdf"))
   
    histpull = plt.pullHist()
    chi2 = plt.chiSquare()
    
    if refittedPdf != None: 
        refittedPdf.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kCyan))
        histpull_refit = plt.pullHist()
   

    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.040)
    latex.SetTextColor(1)
    latex.SetTextFont(42)
    latex.DrawLatex(0.20, 0.85, label)
    latex.DrawLatex(0.20, 0.80, qTlabel)
    latex.DrawLatex(0.20, 0.75, "Mean = %.3f #pm %.3f" % (hist_data.GetMean(), hist_data.GetMeanError()))
    latex.DrawLatex(0.20, 0.70, "RMS = %.3f #pm %.3f" % (hist_data.GetRMS(), hist_data.GetRMSError()))
    latex.DrawLatex(0.20, 0.65, "#chi^{2} = %.3f" % chi2)
        
    latex.DrawLatex(0.60, 0.85, "#mu = %.3f #pm %.3f" % (mean.getVal(), mean.getError()))
    latex.DrawLatex(0.60, 0.80, "#sigma_{1} = %.3f #pm %.3f" % (sigma1.getVal(), sigma1.getError()))
    latex.DrawLatex(0.60, 0.75, "#sigma_{2} = %.3f #pm %.3f" % (sigma2.getVal(), sigma2.getError()))
    latex.DrawLatex(0.60, 0.70, "n = %.3f #pm %.3f" % (norm.getVal(), norm.getError()))
    latex.DrawLatex(0.60, 0.65, "n_{sig} = %.3f #pm %.3f" % (norm_sig.getVal(), norm_sig.getError()))
    
    plt.Draw("SAME")
    plotter.auxRatio()
    canvas.cd()
    padB.Draw()
    padB.cd()
    dummyB.Draw("HIST")

    plt = recoil.frame()
    plt.addPlotable(histpull, "P")
    if refittedPdf != None: plt.addPlotable(histpull_refit, "P")
    plt.Draw("SAME")
        
    line = ROOT.TLine(cfg['xmin'], 0, cfg['xmax'], 0)
    line.SetLineColor(ROOT.kBlue+2)
    line.SetLineWidth(2)
    line.Draw("SAME")

    canvas.Modify()
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs("%s/%03d_recoil_%s.png" % (outDir, qTbin, outName))
    canvas.SaveAs("%s/%03d_recoil_%s.pdf" % (outDir, qTbin, outName))

    ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)   
    dummyB.Delete()
    dummyT.Delete()
    padT.Delete()
    padB.Delete()
 
    if hOut == None: return
    
    # diagonalize covariance matrix and store perturbations
    floatingParams = fitRes.floatParsFinal() # floating parameters
    variations = diagonalize(fitRes)
    
    for iVar, var in enumerate(variations):
        for iPar, p in enumerate(params):
            if getParamIdx(p.GetName(), floatingParams) > -1: val = var[getParamIdx(p.GetName(), floatingParams)]
            else: val = p.getVal()
            hOut.SetBinContent(qTbin, iPar+1, iVar+2, val)
            print(iPar+1, iVar+2, p.GetName(), val)


 
  
def doFit_bkg(outDir, qTbin, qTbinMinGeV, qTbinMaxGeV, hist, name, label, cfg):

    xMin, xMax = -500, 500
    if qTbinMinGeV < 30: xMin,xMax = -100, 100
    elif qTbinMinGeV < 60: xMin,xMax = -150, 50
    elif qTbinMinGeV < 100: xMin,xMax = -200, 50
    else: xMin,xMax = -500, 0
    qT = 0.5*(qTbinMinGeV + qTbinMaxGeV)
    outDir = outDir.replace("/data/", "/bkg/")
    label = "Backgrounds (simulation)"
    
    #hist.Rebin(2)
    if hist.Integral() > 0: hist.Scale(1./hist.Integral())
    cfg['ymax'] = 1.75*hist.GetMaximum()
    
    qTlabel = "%.1f < q_{T} < %.1f GeV" % (qTbinMinGeV, qTbinMaxGeV)
    outName = "%s_%sGeV" % (str("%.1f" % qTbinMinGeV).replace(".", "p"), str("%.1f" % qTbinMaxGeV).replace(".", "p"))
    
    
    '''
    recoil = ROOT.RooRealVar("recoil", "Recoil parallel (GeV)", 0, -500, 500) # independent var needed for CDF
    rdh_bkg = ROOT.RooDataHist("rdh_bkg_qTbin%d" % qTbin, "", ROOT.RooArgList(recoil), ROOT.RooFit.Import(hist))
    pdf = ROOT.RooNDKeysPdf("pdf_%s_qTbin%d" % (name, qTbin), "", ROOT.RooArgList(recoil), hist, "ma", 0.2, 2)
    
    
    #recoil = ROOT.RooRealVar("recoil", "Recoil parallel (GeV)", 0, -500, 500) # independent var needed for CDF
    mean1 = ROOT.RooRealVar("mean1_bkg_qTbin%d" % qTbin, "", hist.GetMean(), -400, 100)
    mean2 = ROOT.RooRealVar("mean2_bkg_qTbin%d" % qTbin, "", 0)
    sigma1 = ROOT.RooRealVar("sigma1_bkg_qTbin%d" % qTbin, "", hist.GetRMS(), 0.1, 100)
    sigma2 = ROOT.RooRealVar("sigma2_bkg_qTbin%d" % qTbin, "", hist.GetRMS(), 0.1, 100)
    norm = ROOT.RooRealVar("norm_bkg_qTbin%d" % qTbin, "", 0.5, 0, 1) # first Gauss = ttbar

    gauss1 = ROOT.RooGaussian("gauss1_bkg_qTbin%d" % qTbin, "", recoil, mean1, sigma1)
    gauss2 = ROOT.RooGaussian("gauss2_bkg_qTbin%d" % qTbin, "", recoil, mean2, sigma2)
    pdf = ROOT.RooAddPdf("pdf_bkg_qTbin%d" % qTbin, '', ROOT.RooArgList(gauss1, gauss2), ROOT.RooArgList(norm))
    rdh_bkg = ROOT.RooDataHist("rdh_bkg_qTbin%d" % qTbin, "", ROOT.RooArgList(recoil), ROOT.RooFit.Import(hist))
    fitRes = pdf.fitTo(rdh_bkg, ROOT.RooFit.SumW2Error(ROOT.kTRUE), ROOT.RooFit.Save(ROOT.kTRUE))
    '''
    '''
    recoil = ROOT.RooRealVar("recoil", "Recoil parallel (GeV)", 0, -500, 500) # independent var needed for CDF
    mean1 = ROOT.RooRealVar("mean1_bkg_qTbin%d" % qTbin, "", hist.GetMean(), -400, 100)
    mean2 = ROOT.RooRealVar("mean2_bkg_qTbin%d" % qTbin, "", hist.GetMean(), -400, 100)
    mean3 = ROOT.RooRealVar("mean3_bkg_qTbin%d" % qTbin, "", 0) # fixed mean at 0
    sigma1 = ROOT.RooRealVar("sigma1_bkg_qTbin%d" % qTbin, "", hist.GetRMS()/2, 0.1, 100)
    sigma2 = ROOT.RooRealVar("sigma2_bkg_qTbin%d" % qTbin, "", hist.GetRMS()*2, 0.1, 100)
    sigma3 = ROOT.RooRealVar("sigma3_bkg_qTbin%d" % qTbin, "", hist.GetRMS()/2, 0.1, 100)
    norm = ROOT.RooRealVar("norm_bkg_qTbin%d" % qTbin, "", 0.8, 0, 1) # first Gauss = ttbar
    norm1 = ROOT.RooRealVar("norm1_bkg_qTbin%d" % qTbin, "", 0.05, 0, 1) # first Gauss = ttbar
    gauss1 = ROOT.RooGaussian("gauss1_bkg_qTbin%d" % qTbin, "", recoil, mean1, sigma1)
    gauss2 = ROOT.RooGaussian("gauss2_bkg_qTbin%d" % qTbin, "", recoil, mean2, sigma2)
    gauss3 = ROOT.RooGaussian("gauss3_bkg_qTbin%d" % qTbin, "", recoil, mean3, sigma3)
    pdf = ROOT.RooAddPdf("pdf_bkg_qTbin%d" % qTbin, '', ROOT.RooArgList(gauss1, gauss2, gauss3), ROOT.RooArgList(norm, norm1))
    rdh_bkg = ROOT.RooDataHist("rdh_bkg_qTbin%d" % qTbin, "", ROOT.RooArgList(recoil), ROOT.RooFit.Import(hist))
    fitRes = pdf.fitTo(rdh_bkg, ROOT.RooFit.SumW2Error(ROOT.kTRUE), ROOT.RooFit.Save(ROOT.kTRUE))
    '''
    

    recoil = ROOT.RooRealVar("recoil", "Recoil parallel (GeV)", 0, -500, 500) # independent var needed for CDF
    mean1 = ROOT.RooRealVar("mean1_bkg_%s_qTbin%d" % (name, qTbin), "", hist.GetMean(), -400, 100)
    mean2 = ROOT.RooRealVar("mean2_bkg_%s_qTbin%d" % (name, qTbin), "", hist.GetMean(), -400, 100)
    sigma1 = ROOT.RooRealVar("sigma1_bkg_%s_qTbin%d" % (name, qTbin), "", hist.GetRMS()/2, 0.1, 100)
    sigma2 = ROOT.RooRealVar("sigma2_bkg_%s_qTbin%d" % (name, qTbin), "", hist.GetRMS()*2, 0.1, 100)
    norm = ROOT.RooRealVar("norm_bkg_%s_qTbin%d" % (name, qTbin), "", 0.9, 0, 1) # first Gauss = ttbar
    gauss1 = ROOT.RooGaussian("gauss1_bkg_%s_qTbin%d" % (name, qTbin), "", recoil, mean1, sigma1)
    gauss2 = ROOT.RooGaussian("gauss2_bkg_%s_qTbin%d" % (name, qTbin), "", recoil, mean2, sigma2)
    pdf = ROOT.RooAddPdf("pdf_bkg_%s_qTbin%d" % (name, qTbin), '', ROOT.RooArgList(gauss1, gauss2), ROOT.RooArgList(norm))
    rdh_bkg = ROOT.RooDataHist("rdh_%s_qTbin%d" % (name, qTbin), "", ROOT.RooArgList(recoil), ROOT.RooFit.Import(hist))
    fitRes = pdf.fitTo(rdh_bkg, ROOT.RooFit.SumW2Error(ROOT.kTRUE), ROOT.RooFit.Save(ROOT.kTRUE))

 
    # freeze the PDF
    mean1.setConstant(ROOT.kTRUE)
    mean2.setConstant(ROOT.kTRUE)
    sigma1.setConstant(ROOT.kTRUE)
    sigma2.setConstant(ROOT.kTRUE)
    norm.setConstant(ROOT.kTRUE)
 
    w.Import(pdf)
   

    plotter.cfg = cfg
    cfg['xmin'] = xMin
    cfg['xmax'] = xMax
    canvas, padT, padB = plotter.canvasRatio()
    dummyT, dummyB, dummyL = plotter.dummyRatio()
    
    
    ## TOP PAD ##
    canvas.cd()
    padT.Draw()
    padT.cd()
    dummyT.Draw("HIST")
    
    ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.ERROR)
    plt = recoil.frame()
    rdh_bkg.plotOn(plt, ROOT.RooFit.Name("bkg"), ROOT.RooFit.DataError(ROOT.RooAbsData.SumW2))
    pdf.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kRed), ROOT.RooFit.Name("pdf"))
    histpull = plt.pullHist()
    chi2 = plt.chiSquare() 
    
    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.040)
    latex.SetTextColor(1)
    latex.SetTextFont(42)
    latex.DrawLatex(0.20, 0.85, label)
    latex.DrawLatex(0.20, 0.80, qTlabel)
    latex.DrawLatex(0.20, 0.75, "Mean = %.3f #pm %.3f" % (hist.GetMean(), hist.GetMeanError()))
    latex.DrawLatex(0.20, 0.70, "RMS = %.3f #pm %.3f" % (hist.GetRMS(), hist.GetRMSError()))
    latex.DrawLatex(0.20, 0.65, "#chi^{2} = %.3f" % chi2)

    latex.DrawLatex(0.60, 0.85, "#mu_{1} = %.3f #pm %.3f" % (mean1.getVal(), mean1.getError()))
    latex.DrawLatex(0.60, 0.80, "#sigma_{1} = %.3f #pm %.3f" % (sigma1.getVal(), sigma1.getError()))
    latex.DrawLatex(0.60, 0.75, "#mu_{2} = %.3f #pm %.3f" % (mean2.getVal(), mean2.getError()))
    latex.DrawLatex(0.60, 0.70, "#sigma_{2} = %.3f #pm %.3f" % (sigma2.getVal(), sigma2.getError()))
    latex.DrawLatex(0.60, 0.65, "N = %.3f #pm %.3f" % (norm.getVal(), norm.getError()))
    

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
        
    line = ROOT.TLine(cfg['xmin'], 0, cfg['xmax'], 0)
    line.SetLineColor(ROOT.kBlue+2)
    line.SetLineWidth(2)
    line.Draw("SAME")

    canvas.Modify()
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs("%s/%03d_recoil_%s.png" % (outDir, qTbin, outName))
    canvas.SaveAs("%s/%03d_recoil_%s.pdf" % (outDir, qTbin, outName))

    ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)   

    dummyB.Delete()
    dummyT.Delete()
    padT.Delete()
    padB.Delete()



    
  
            
    
def doFit_GaussianKernel(outDir, qTbin, qTbinMinGeV, qTbinMaxGeV, hist, name, label, met, cfg):

    if qTbinMinGeV < 30: xMin,xMax = -100, 100
    elif qTbinMinGeV < 60: xMin,xMax = -150, 50
    elif qTbinMinGeV < 100: xMin,xMax = -200, 50
    else: xMin,xMax = -300, 0
    
    outDir += "/GaussianKernel/"
    prepareDir(outDir, remove=False)
    
    cfg['ymax'] = 1.75*hist.GetMaximum()
 
    qTlabel = "%.1f < q_{T} < %.1f GeV" % (qTbinMinGeV, qTbinMaxGeV)
    outName = "%s_%sGeV" % (str("%.1f" % qTbinMinGeV).replace(".", "p"), str("%.1f" % qTbinMaxGeV).replace(".", "p"))
    if "gen" in outDir: qTlabel = qTlabel.replace("q_{T}", "q_{T}^{GEN}")

    if "para_" in name:
    
        cfg['xmin'] = xMin
        cfg['xmax'] = xMax
        
    else:
    
        cfg['xmin'] = -100
        cfg['xmax'] = 100
 

    ### do Gaussian Kernel smoothing
    sigma = 10
    
    
    hist_gk = copy.deepcopy(hist)
    hist_gk.SetName("pdf_gk_%s_%s" % (name, met))
    
    hist_gk_pull = copy.deepcopy(hist)
    hist_gk_pull.SetName("pdf_gk_%s_%s_pull" % (name, met))
    
    ROOT.GaussianKernel(hist, hist_gk, hist_gk_pull, sigma)
    
    
    ### get cumulative
    hist_gk_cdf = hist_gk.GetCumulative()
    hist_gk_cdf.SetName("cdf_gk_%s_%s" % (name, met))

    ratio = hist.Integral() / hist_gk.Integral()
    
    
    hist.SetMarkerStyle(20)
    hist.SetMarkerColor(ROOT.kBlack)
    hist.SetLineColor(ROOT.kBlack)
    
    hist_gk.SetLineColor(ROOT.kBlue)
    #hist_gk.SetLineStyle(ROOT.kDashed)
    hist_gk.SetLineWidth(2)
    
    hist_gk_pull.SetMarkerStyle(20)
    hist_gk_pull.SetMarkerColor(ROOT.kBlack)
    hist_gk_pull.SetLineColor(ROOT.kBlack)

    plotter.cfg = cfg
    canvas, padT, padB = plotter.canvasRatio()
    dummyT, dummyB, dummyL = plotter.dummyRatio()
    
        
    ## TOP PAD ##
    canvas.cd()
    padT.Draw()
    padT.cd()
    dummyT.Draw("HIST")
    
    hist.Draw("PE SAME")
    hist_gk.Draw("HIST SAME")
    
    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.040)
    latex.SetTextColor(1)
    latex.SetTextFont(42)
    latex.DrawLatex(0.20, 0.85, label)
    latex.DrawLatex(0.20, 0.80, qTlabel)
    latex.DrawLatex(0.20, 0.75, met_labels[met_estimators.index(met)])
    latex.DrawLatex(0.20, 0.70, "N_{hist}/N_{fit} = %.3f" % (ratio))
    latex.DrawLatex(0.20, 0.65, "Mean = %.3f" % hist.GetMean())
    latex.DrawLatex(0.20, 0.60, "RMS = %.3f" % hist.GetRMS())
    
    latex.DrawLatex(0.60, 0.85, "Kernel width #sigma = %.2f " % sigma)

    plotter.auxRatio()
       
    ## BOTTOM PAD ##
    canvas.cd()
    padB.Draw()
    padB.cd()
    dummyB.Draw("HIST")
      
    hist_gk_pull.Draw("PE SAME")

    line = ROOT.TLine(120, 0, 140, 0)
    line.SetLineColor(ROOT.kBlue+2)
    line.SetLineWidth(2)
    line.Draw("SAME")

    canvas.Modify()
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs("%s/recoil_%s.png" % (outDir, outName))
    canvas.SaveAs("%s/recoil_%s.pdf" % (outDir, outName))

    return hist_gk, hist_gk_cdf
    


def doFit_DSCB(outDir, qTbin, qTbinMinGeV, qTbinMaxGeV, hist, name, label, met, cfg, hOut):

    if qTbinMinGeV < 30: xMin,xMax = -100, 100
    elif qTbinMinGeV < 60: xMin,xMax = -150, 50
    elif qTbinMinGeV < 100: xMin,xMax = -200, 50
    else: xMin,xMax = -300, 0
    
    outDir += "/DSCB/"
    prepareDir(outDir, remove=False)
    
    cfg['ymax'] = 1.75*hist.GetMaximum()
    #cfg['ymax'] = 10
    
    qTlabel = "%.1f < q_{T} < %.1f GeV" % (qTbinMinGeV, qTbinMaxGeV)
    outName = "%s_%sGeV" % (str("%.1f" % qTbinMinGeV).replace(".", "p"), str("%.1f" % qTbinMaxGeV).replace(".", "p"))
    if "_gen" in outDir: qTlabel = qTlabel.replace("q_{T}", "q_{T}^{GEN}")

    if "para_" in name:
    
        cfg['xmin'] = xMin
        cfg['xmax'] = xMax
        
    else:
    
        cfg['xmin'] = -100
        cfg['xmax'] = 100

    # alphabetic prefix for ordering of parameters
    recoil = ROOT.RooRealVar("recoil_%s" % name, "Recoil parallel (GeV)", 0, -300, 100) # independent var needed for CDF
    mean = ROOT.RooRealVar("a_mean_%s" % name, "", 0, -400, 100)
    sigma = ROOT.RooRealVar("b_sigma_%s" % name, "", 5, 0.1, 100)
    a1 = ROOT.RooRealVar("c_a1_%s" % name, "", 5, 0.1, 100)
    p1 = ROOT.RooRealVar("d_p1_%s" % name, "", 5, 0.1, 100)
    a2 = ROOT.RooRealVar("e_a2_%s" % name, "norm", 1, 0.1, 1000)
    p2 = ROOT.RooRealVar("f_p2_%s" % name, "norm", 1, 0.1, 1000)
    
    mean_g = ROOT.RooRealVar("a_mean_g_%s" % name, "", 0, -400, 100)
    sigma_g = ROOT.RooRealVar("c_sigma_g_%s" % name, "", 5, 0.1, 100)
    norm1 = ROOT.RooRealVar("e_norm1_%s" % name, "norm", 0, 0, 1000)
    norm2 = ROOT.RooRealVar("e_norm2_%s" % name, "norm", 0, 0, 1000)

    dscb = ROOT.RooCrystalBall("pdf_%s" % name, '', recoil, mean, sigma, a1, p1, a2, p2);
    gauss = ROOT.RooGaussian("gauss_%s" % name, "gauss", recoil, mean_g, sigma_g)
    pdf = ROOT.RooAddPdf("pdf_%s" % name, '', ROOT.RooArgList(dscb), ROOT.RooArgList(norm1)) # sum of 2 Gaussians
    rdh = ROOT.RooDataHist("rdh_%s" % name, "", ROOT.RooArgList(recoil), ROOT.RooFit.Import(hist))
    
    pdf = ROOT.RooAddPdf("pdf_%s" % name, '', ROOT.RooArgList(dscb, gauss), ROOT.RooArgList(norm1)) # sum of 2 Gaussians
    
    
    norm1.setVal(1)
    #norm1.setConstant(ROOT.kTRUE)
    #norm2.setVal(0.0)
    
    mean_g.setVal(hist.GetMean())
    #mean.setConstant(ROOT.kTRUE)
    
    sigma_g.setVal(hist.GetRMS())
    #width.setConstant(ROOT.kTRUE)    

    
    mean.setVal(hist.GetMean())
    mean.setConstant(ROOT.kTRUE)
    
    sigma.setVal(hist.GetRMS()*0.8)
    #width.setConstant(ROOT.kTRUE)
    
    a1.setVal(2.5)
    a2.setVal(2.5)
    
    p1.setVal(2.5)
    p2.setVal(2.5)
    #a2.setVal(0)
    #a2.setConstant(ROOT.kTRUE)
    #p1.setVal(0)
    #p1.setConstant(ROOT.kTRUE)
    fitRes = pdf.fitTo(rdh, ROOT.RooFit.SumW2Error(ROOT.kTRUE), ROOT.RooFit.Save(ROOT.kTRUE))   
    
    # 
    

    #sys.exit()
    plotter.cfg = cfg
    canvas, padT, padB = plotter.canvasRatio()
    dummyT, dummyB, dummyL = plotter.dummyRatio()
 
    # save nominal parameters
    '''
    for k, var in enumerate(vars_):

        val = var.getVal()
        hOut.SetBinContent(qTbin+1, k+1, 1, val) 
        print(var.GetName(), qTbin+1, k+1, 1, val)    
     
    
    
    print("------------")
    # diagonalize covariance matrix 
    #print(mean1.getVal(), mean1.getVal())

    cov = copy.deepcopy(fitRes.covarianceMatrix()) # covariance matrix: diagonal elements are the variances of each of the parameters
    #cov.Print()


    #covInv = copy.deepcopy(res.covarianceMatrix()) # copy necessary... 
    #covInv.Invert()
   
    eigen = ROOT.TMatrixDEigen(cov) # sorts vectors/values by eigenvalues
    eigenVectors = copy.deepcopy(eigen.GetEigenVectors()) # TMatrixD
    eigenVectorsInv = copy.deepcopy(eigen.GetEigenVectors()) # TMatrixD
    eigenVectorsInv.Invert() # TMatrixD
    eigenValues = eigen.GetEigenValuesRe() # TVectorD
    
    
    npars = len(eigenValues)
    sigma = 1
    for iVar in range(0, npars):
    
        print("vary %s", iVar)
    
        eigenValuesShifted = ROOT.TMatrixD(npars, npars) 
        for i, eig in enumerate(eigen.GetEigenValuesRe()): 
        
            if i == iVar: eigenValuesShifted[i][i] = sigma*math.sqrt(eig)
            else: eigenValuesShifted[i][i] = eig
        
        # rotate the shifted eigenvalue matrix
        new = ROOT.TMatrixD(npars, npars)
        new.Mult(eigenVectors, eigenValuesShifted)
        
        new1 = ROOT.TMatrixD(npars, npars)
        new1.Mult(new, eigenVectorsInv)
        
        
        kk = 0
        for k, var in enumerate(vars_):
        
            if var.isConstant(): err = 0
            else: 
                err = math.sqrt(new1[kk][kk])
                kk += 1
                
            if k == 3: err = math.sqrt(new1[0][0]) # mean2 = mean1
            
            hOut.SetBinContent(qTbin+1, k+1, iVar+2, err) 
            print(var.GetName(), qTbin+1, k+1, iVar+2, err)
        
    '''
    

  
    ## TOP PAD ##
    canvas.cd()
    padT.Draw()
    padT.cd()
    dummyT.Draw("HIST")
    
    ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.ERROR)
    plt = recoil.frame()
    rdh.plotOn(plt)
    #gauss1.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kBlue), ROOT.RooFit.Normalization(norm1.getVal(), ROOT.RooAbsReal.NumEvent), ROOT.RooFit.LineStyle(ROOT.kDashed))
    #gauss2.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kBlue), ROOT.RooFit.Normalization(norm2.getVal(), ROOT.RooAbsReal.NumEvent), ROOT.RooFit.LineStyle(ROOT.kDashed))
    pdf.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kRed))
        
    #ratio = hist.Integral() / (norm1.getVal() + norm2.getVal())
    #err = ratio * math.hypot(norm1.getError()+norm2.getError()) / (norm1.getVal() + norm2.getVal())
        
    ratio = 0
    err = 0
    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.040)
    latex.SetTextColor(1)
    latex.SetTextFont(42)
    latex.DrawLatex(0.20, 0.85, label)
    latex.DrawLatex(0.20, 0.80, qTlabel)
    latex.DrawLatex(0.20, 0.75, met_labels[met_estimators.index(met)])
    latex.DrawLatex(0.20, 0.70, "N_{hist}/N_{fit} = %.3f #pm %.3f" % (ratio, err))
    latex.DrawLatex(0.20, 0.65, "Mean = %.3f #pm %.3f" % (hist.GetMean(), hist.GetMeanError()))
    latex.DrawLatex(0.20, 0.60, "RMS = %.3f #pm %.3f" % (hist.GetRMS(), hist.GetRMSError()))
 
    latex.DrawLatex(0.60, 0.85, "#mu = %.3f #pm %.3f" % (mean.getVal(), mean.getError()))
    latex.DrawLatex(0.60, 0.80, "#sigma = %.3f #pm %.3f" % (sigma.getVal(), sigma.getError()))
    latex.DrawLatex(0.60, 0.75, "a_{1} = %.3f #pm %.3f" % (a1.getVal(), a1.getError()))
    latex.DrawLatex(0.60, 0.70, "p_{1} = %.3f #pm %.3f" % (p1.getVal(), p1.getError()))
    latex.DrawLatex(0.60, 0.65, "a{2} = %.3f #pm %.3f" % (a2.getVal(), a2.getError()))
    latex.DrawLatex(0.60, 0.60, "p_{2} = %.3f #pm %.3f" % (p2.getVal(), p2.getError()))
 
    
    histpull = plt.pullHist()
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
        
    line = ROOT.TLine(120, 0, 140, 0)
    line.SetLineColor(ROOT.kBlue+2)
    line.SetLineWidth(2)
    line.Draw("SAME")

    canvas.Modify()
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs("%s/recoil_%s.png" % (outDir, outName))
    canvas.SaveAs("%s/recoil_%s.pdf" % (outDir, outName))

    ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)   

    #sys.exit()
 
    dummyB.Delete()
    dummyT.Delete()
    padT.Delete()
    padB.Delete()
    
    
    #return mean1.getVal(), sigma1.getVal(), norm1.getVal(), mean2.getVal(), sigma2.getVal(), norm2.getVal()
  
def doFit_2Gauss_DY__(outDir, qTbin, qTbinMinGeV, qTbinMaxGeV, hist, name, label, cfg, hOut=None, refittedPdf=None):
    

    if qTbinMinGeV < 30: xMin,xMax = -100, 100
    elif qTbinMinGeV < 60: xMin,xMax = -150, 50
    elif qTbinMinGeV < 100: xMin,xMax = -200, 50
    else: xMin,xMax = -500, 0
    if "perp" in name: xMin,xMax = -100, 100
    
    #hist = hist.Rebin(1)
    hist.Scale(1./hist.Integral())    
    
    qT = 0.5*(qTbinMinGeV + qTbinMaxGeV)
    

    qTlabel = "%.1f < q_{T} < %.1f GeV" % (qTbinMinGeV, qTbinMaxGeV)
    outName = "%s_%sGeV" % (str("%.1f" % qTbinMinGeV).replace(".", "p"), str("%.1f" % qTbinMaxGeV).replace(".", "p"))
    if "_gen" in outDir: qTlabel = qTlabel.replace("q_{T}", "q_{T}^{GEN}")

    cfg['xmin'], cfg['xmax'] = xMin, xMax
    cfg['ymax'] = 1.75*hist.GetMaximum()
    


    recoil = ROOT.RooRealVar("recoil", "Recoil parallel (GeV)", 0, -500, 500)
    mean = ROOT.RooRealVar("mean", "", 0, -500, 500)
    sigma1 = ROOT.RooRealVar("sigma1", "", 5, 0.1, 100)
    sigma2 = ROOT.RooRealVar("sigma2", "", 5, 0.1, 100)
    norm = ROOT.RooRealVar("norm", "", 0, 0, 1)
    gauss1 = ROOT.RooGaussian("gauss1", "gauss1", recoil, mean, sigma1)
    gauss2 = ROOT.RooGaussian("gauss2", "gauss2", recoil, mean, sigma2)
    pdf = ROOT.RooAddPdf("pdf", '', ROOT.RooArgList(gauss1, gauss2), ROOT.RooArgList(norm)) # sum of 2 Gaussians
    rdh = ROOT.RooDataHist("rdh", "", ROOT.RooArgList(recoil), ROOT.RooFit.Import(hist))
    
    params = [mean, sigma1, sigma2, norm]
    
    mean.setVal(hist.GetMean())
    #mean.setConstant(ROOT.kTRUE)
        
    norm.setVal(0.6)
    #norm.setConstant(ROOT.kTRUE)
        
    sigma1.setVal(hist.GetRMS()/1.5)
    #sigma1.setConstant(ROOT.kTRUE)
        
    sigma2.setVal(hist.GetRMS()*1.5)
    #sigma2.setConstant(ROOT.kTRUE)
        
    if "perp" in name:
        mean.setVal(0)
        mean.setConstant(ROOT.kTRUE)


    # ML fit
    fitRes = pdf.fitTo(rdh, ROOT.RooFit.SumW2Error(ROOT.kTRUE), ROOT.RooFit.Save(ROOT.kTRUE)) # ROOT.RooFit.Extended(ROOT.kTRUE), 
   
        
    print("***** Summary for qTbin %d  *****" % qTbin)
    fitRes.floatParsFinal().Print("s")

    
    if hOut != None:
        for iPar, p in enumerate(params):
            hOut.SetBinContent(qTbin, iPar+1, 1, p.getVal())
            hOut.SetBinContent(qTbin, iPar+1, 0, p.getError())


    plotter.cfg = cfg
    canvas, padT, padB = plotter.canvasRatio()
    dummyT, dummyB, dummyL = plotter.dummyRatio()
    
    
    ## TOP PAD ##
    canvas.cd()
    padT.Draw()
    padT.cd()
    dummyT.Draw("HIST")
    
    ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.ERROR)
    plt = recoil.frame()
    rdh.plotOn(plt)
    gauss1.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kBlue), ROOT.RooFit.Normalization(norm.getVal(), ROOT.RooAbsReal.NumEvent), ROOT.RooFit.LineStyle(ROOT.kDashed))
    gauss2.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kGreen), ROOT.RooFit.Normalization(1.-norm.getVal(), ROOT.RooAbsReal.NumEvent), ROOT.RooFit.LineStyle(ROOT.kDashed))
    
    pdf.plotOn(plt, ROOT.RooFit.VisualizeError(fitRes, 1), ROOT.RooFit.Name("errorband"), ROOT.RooFit.FillColor(ROOT.kOrange))
    pdf.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kRed))
    histpull = plt.pullHist()
    chi2 = plt.chiSquare() 
    
    if refittedPdf != None: 
        refittedPdf.plotOn(plt, ROOT.RooFit.LineColor(ROOT.kCyan))
        histpull_refit = plt.pullHist()
        
        
    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.040)
    latex.SetTextColor(1)
    latex.SetTextFont(42)
    latex.DrawLatex(0.20, 0.85, label)
    latex.DrawLatex(0.20, 0.80, qTlabel)
    latex.DrawLatex(0.20, 0.75, "Mean = %.3f #pm %.3f" % (hist.GetMean(), hist.GetMeanError()))
    latex.DrawLatex(0.20, 0.70, "RMS = %.3f #pm %.3f" % (hist.GetRMS(), hist.GetRMSError()))
        
    latex.DrawLatex(0.60, 0.85, "#mu = %.3f #pm %.3f" % (mean.getVal(), mean.getError()))
    latex.DrawLatex(0.60, 0.80, "#sigma_{1} = %.3f #pm %.3f" % (sigma1.getVal(), sigma1.getError()))
    latex.DrawLatex(0.60, 0.75, "#sigma_{2} = %.3f #pm %.3f" % (sigma2.getVal(), sigma2.getError()))
    latex.DrawLatex(0.60, 0.70, "N = %.3f #pm %.3f" % (norm.getVal(), norm.getError()))
    latex.DrawLatex(0.60, 0.65, "#chi^{2}/ndof = %.3f" % chi2)

    plt.Draw("SAME")
    plotter.auxRatio()
       
    ## BOTTOM PAD ##
    canvas.cd()
    padB.Draw()
    padB.cd()
    dummyB.Draw("HIST")
    
    plt = recoil.frame()
    plt.addPlotable(histpull, "P")
    if refittedPdf != None: plt.addPlotable(histpull_refit, "P")

    plt.Draw("SAME")
        
    line = ROOT.TLine(120, 0, 140, 0)
    line.SetLineColor(ROOT.kBlue+2)
    line.SetLineWidth(2)
    line.Draw("SAME")

    canvas.Modify()
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs("%s/%03d_recoil_%s.png" % (outDir, qTbin, outName))
    canvas.SaveAs("%s/%03d_recoil_%s.pdf" % (outDir, qTbin, outName))

    ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)   

    dummyB.Delete()
    dummyT.Delete()
    padT.Delete()
    padB.Delete()
    
    if hOut == None: return
    
    # diagonalize covariance matrix and store perturbations
    floatingParams = fitRes.floatParsFinal() # floating parameters
    variations = diagonalize(fitRes)
    
    for iVar, var in enumerate(variations):
        for iPar, p in enumerate(params):
            if getParamIdx(p.GetName(), floatingParams) > -1: val = var[getParamIdx(p.GetName(), floatingParams)]
            else: val = p.getVal()
            hOut.SetBinContent(qTbin, iPar+1, iVar+2, val)
            print(iVar, iPar, p.GetName(), val)


def readProc(hName, procName):

    label = "%s_%s" % (hName, procName)
    print(groups_mumu.groups)
    groups_mumu.setHists(hName, "", label=label, procsToRead=[procName], selectSignal=False)
    bhist = groups_mumu.groups[procName][label]
    return bhist
    #print(bhist)
    #rhist = narf.hist_to_root(bhist)
    #rhist.SetName(label)
    #return rhist

def doRecoilFits_Z():
    
    if lowPU:
        outDir = "/eos/user/j/jaeyserm/www/wmass/lowPU/recoilCorrection/fits_Z_new/"
        fOut_ = "wremnants/data/lowPU/recoil_fits_Z.root"
        dataName = "SingleMuon"
        signalName = "DYmumu"
    else:
        outDir = "/eos/user/j/jaeyserm/www/wmass/highPU/recoilCorrection/fits_Z/"
        fOut_ = "wremnants/data/recoil_fits_Z.root"
        dataName = "Data"
        signalName = "Zmumu"

    b_data_para = readProc("recoil_uncorr_para_qTbinned", dataName) 
    b_data_para_qT = readProc("recoil_uncorr_para_qT_qTbinned", dataName)
    b_data_perp = readProc("recoil_uncorr_perp_qTbinned", dataName)
    b_mc_para = readProc("recoil_uncorr_para_qTbinned", signalName) 
    b_mc_para_qT = readProc("recoil_uncorr_para_qT_qTbinned", signalName) 
    b_mc_perp = readProc("recoil_uncorr_perp_qTbinned", signalName)
    
    #b_data_para = groups_mumu.readProc("recoil_uncorr_para_qt", dataName) # + groups_ee.readProc("recoil_uncorr_para_qt", "SingleElectron")
    #b_data_perp = groups_mumu.readProc("recoil_uncorr_perp_qt", dataName) # + groups_ee.readProc("recoil_uncorr_perp_qt", "SingleElectron")
    #b_mc_para = groups_mumu.readProc("recoil_uncorr_para_qt", signalName) # + groups_ee.readProc("recoil_uncorr_para_qt", "DYee")
    #b_mc_perp = groups_mumu.readProc("recoil_uncorr_perp_qt", signalName) # + groups_ee.readProc("recoil_uncorr_perp_qt", "DYee")
    

    b_bkg_para, b_bkg_para_qT, b_bkg_perp = None, None, None
    for bkg in bkg_procs:
    
        b_para = readProc("recoil_uncorr_para_qTbinned", bkg)
        if b_bkg_para == None: b_bkg_para = b_para
        else: b_bkg_para += b_para
        
        b_para_qT = readProc("recoil_uncorr_para_qT_qTbinned", bkg)
        if b_bkg_para_qT == None: b_bkg_para_qT = b_para_qT
        else: b_bkg_para_qT += b_para_qT
    
        b_perp = readProc("recoil_uncorr_perp_qTbinned", bkg)
        if b_bkg_perp == None: b_bkg_perp = b_perp
        else: b_bkg_perp += b_perp
      
      
    h_data_para = narf.hist_to_root(b_data_para)
    h_data_para_qT = narf.hist_to_root(b_data_para_qT)
    h_mc_para = narf.hist_to_root(b_mc_para)
    h_mc_para_qT = narf.hist_to_root(b_mc_para_qT)
    h_bkg_para = narf.hist_to_root(b_bkg_para)
    h_bkg_para_qT = narf.hist_to_root(b_bkg_para_qT)
    h_data_perp = narf.hist_to_root(b_data_perp)
    h_mc_perp = narf.hist_to_root(b_mc_perp)
    h_bkg_perp = narf.hist_to_root(b_bkg_perp)

    #h_data_para.Add(h_bkg_para, -1)
    #h_data_perp.Add(h_bkg_perp, -1)
    
    # normalize the data and total MC to 1
    '''
    h_data_para.Scale(1./h_data_para.Integral())
    mc_tot_int_para = h_mc_para.Integral() + h_bkg_para.Integral()
    h_mc_para.Scale(1./mc_tot_int_para)
    h_bkg_para.Scale(1./mc_tot_int_para)
    
    h_data_perp.Scale(1./h_data_perp.Integral())
    mc_tot_int_perp = h_mc_perp.Integral() + h_bkg_perp.Integral()
    h_mc_perp.Scale(1./mc_tot_int_perp)
    h_bkg_perp.Scale(1./mc_tot_int_perp)
    '''

    cfg = {

        'logy'              : False,
        'logx'              : False,
    
        'xmin'              : -300,
        'xmax'              : 100,
        'ymin'              : 0.0000001,
        'ymax'              : 1500,
        
        'xtitle'            : "Recoil U_{#parallel} (GeV)",
        'ytitle'            : "Events / 1 GeV",
        
        'topRight'          : "199 pb^{#minus1} (13 TeV)", 
        'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",
        
        'ratiofraction'     : 0.25,
        'ytitleR'           : "Pull",
        'yminR'             : -5.8,
        'ymaxR'             : 5.8,
    }   
    
    label_mc = "DY #rightarrow #mu^{+}#mu^{#minus} #plus e^{+}e^{#minus}"
    
    h_para_data = ROOT.TH3D("para_data", "", 100, 0, 100, 20, 0, 20, 20, 0, 20)
    h_para_mc = ROOT.TH3D("para_mc", "", 100, 0, 100, 20, 0, 20, 20, 0, 20)
    h_perp_data = ROOT.TH3D("perp_data", "", 100, 0, 100, 20, 0, 20, 20, 0, 20) 
    h_perp_mc = ROOT.TH3D("perp_mc", "", 100, 0, 100, 20, 0, 20, 20, 0, 20)
    
    functions.prepareDir(outDir, False)
    functions.prepareDir(outDir + "/data/", False)
    functions.prepareDir(outDir + "/mc/", False)
    functions.prepareDir(outDir + "/data/para/")
    functions.prepareDir(outDir + "/data/perp/")
    functions.prepareDir(outDir + "/mc/para/")
    functions.prepareDir(outDir + "/mc/perp/")
    functions.prepareDir(outDir + "/bkg/", False)
    functions.prepareDir(outDir + "/bkg/para/")
    functions.prepareDir(outDir + "/bkg/perp/")
    
    fOut = ROOT.TFile("/eos/user/j/jaeyserm/analysis/lowPU/recoil_mumu_DeepMET_lowPU_RoccoR_lowPU_v0.root", "RECREATE")
    #fOut = ROOT.TFile("/eos/user/j/jaeyserm/analysis/lowPU/recoil_mumu_PFMET_lowPU_RoccoR_lowPU_v0.root", "RECREATE")
    
    for iBin in range(1, h_data_para.GetNbinsX()+1):
    
        qTbinMinGeV, qTbinMaxGeV = h_data_para.GetXaxis().GetBinLowEdge(iBin), h_data_para.GetXaxis().GetBinLowEdge(iBin+1)
        qT = h_data_para.GetXaxis().GetBinCenter(iBin)
        
        print("***************** Do iBin %d [%f,%f]" % (iBin, qTbinMinGeV, qTbinMaxGeV))
        
        #hist = h_bkg_para.ProjectionY("bkg_para_bin%d" % iBin, iBin, iBin)
        #doFit_bkg(outDir + "/bkg/para/", iBin, qTbinMinGeV, qTbinMaxGeV, hist, "bkg_para", "Backgrounds", cfg)
        
        #hist = h_bkg_perp.ProjectionY("bkg_perp_bin%d" % iBin, iBin, iBin)
        #doFit_bkg(outDir + "/bkg/perp/", iBin, qTbinMinGeV, qTbinMaxGeV, hist, "bkg_perp", "Backgrounds", cfg)
        
        
        #hist = h_mc_para.ProjectionY("mc_para_bin%d" % iBin, iBin, iBin)
        #doFit_2Gauss_DY__(outDir + "/mc/para/", iBin, qTbinMinGeV, qTbinMaxGeV, hist, "mc_para", label_mc, cfg, h_para_mc)
        
        #hist = h_mc_perp.ProjectionY("mc_perp_bin%d" % iBin, iBin, iBin)
        #doFit_2Gauss_DY__(outDir + "/mc/perp/", iBin, qTbinMinGeV, qTbinMaxGeV, hist, "mc_perp", label_mc, cfg, h_perp_mc)
        
        #hist_data = h_data_para.ProjectionY("data_para_bin%d" % iBin, iBin, iBin)
        #doFit_2Gauss_DY__(outDir + "/data/para/", iBin, qTbinMinGeV, qTbinMaxGeV, hist_data, "data_para", "Data", cfg, h_para_data)
        
        #hist_data = h_data_perp.ProjectionY("data_perp_bin%d" % iBin, iBin, iBin)
        #doFit_2Gauss_DY__(outDir + "/data/perp/", iBin, qTbinMinGeV, qTbinMaxGeV, hist_data, "data_perp", "Data", cfg, h_perp_data)
        
        
        hist = h_mc_para.ProjectionY("mc_para_bin%d" % iBin, iBin, iBin)
        hist.Write()
        #doFit_2Gauss_DY(outDir + "/mc/para/", iBin, qTbinMinGeV, qTbinMaxGeV, hist, "mc_para", label_mc, cfg, h_para_mc)
        
        #doFit_cumulative(outDir + "/mc/para/", iBin, qTbinMinGeV, qTbinMaxGeV, hist, "mc_para", label_mc, cfg, h_para_mc)
        
        hist = h_mc_para_qT.ProjectionY("mc_para_qT_bin%d" % iBin, iBin, iBin)
        print(hist.GetName())
        hist.Write()
        
        
        hist = h_mc_perp.ProjectionY("mc_perp_bin%d" % iBin, iBin, iBin)
        hist.Write()
        #doFit_2Gauss_DY(outDir + "/mc/perp/", iBin, qTbinMinGeV, qTbinMaxGeV, hist, "mc_perp", label_mc, cfg, h_perp_mc)
        
        hist_data = h_data_para.ProjectionY("data_para_bin%d" % iBin, iBin, iBin)
        hist_data.Write()
        hist_bkg = h_bkg_para.ProjectionY("bkg_para_bin%d" % iBin, iBin, iBin)
        hist_bkg.Write()
        
        hist_data_qT = h_data_para_qT.ProjectionY("data_para_qT_bin%d" % iBin, iBin, iBin)
        hist_data_qT.Write()
        hist_bkg_qT = h_bkg_para_qT.ProjectionY("bkg_para_qT_bin%d" % iBin, iBin, iBin)
        hist_bkg_qT.Write()
        #doFit_2Gauss_data(outDir + "/data/para/", iBin, qTbinMinGeV, qTbinMaxGeV, hist_data, hist_bkg, "data_para", "Data", cfg, h_para_data)
        

        hist_data = h_data_perp.ProjectionY("data_perp_bin%d" % iBin, iBin, iBin)
        hist_data.Write()
        hist_bkg = h_bkg_perp.ProjectionY("bkg_perp_bin%d" % iBin, iBin, iBin)
        hist_bkg.Write()
        #doFit_2Gauss_data(outDir + "/data/perp/", iBin, qTbinMinGeV, qTbinMaxGeV, hist_data, hist_bkg, "data_perp", "Data", cfg, h_perp_data)
    
    fOut.ls()
    fOut.Close()
    sys.exit()
    fOut = ROOT.TFile(fOut_, "RECREATE")
    
    h_para_data.Write()
    h_para_mc.Write()
    h_perp_data.Write()
    h_perp_mc.Write()

    fOut.ls()
    fOut.Close()
    print("Written to %s" % fOut_)
    
    return

    sys.exit()

    

    

    prepareDir(outDir)
    prepareDir(outDir + "/data/")
    prepareDir(outDir + "/mc/")
    prepareDir(outDir + "/data/para/")
    prepareDir(outDir + "/data/perp/")
    prepareDir(outDir + "/mc/para/")
    prepareDir(outDir + "/mc/perp/")
    prepareDir(outDir + "/gen/")
    prepareDir(outDir + "/gen/para/")
    prepareDir(outDir + "/gen/perp/")



    

    histsToWrite = []
    for qTbin in range(0, len(qTbins)-1):
    
        #qTbinMinGeV, qTbinMaxGeV = qTbins[qTbin], qTbins[qTbin+1]
        #qTbinMin, qTbinMax = int(2*qTbinMinGeV)+1, int(2*qTbinMaxGeV) # binned in 0.5 GeV
        
        qTbinMinGeV, qTbinMaxGeV = qTbins[qTbin], qTbins[qTbin+1]
        qTbinMin, qTbinMax = int(qTbins[qTbin]+1), int(qTbins[qTbin+1]) # proper indexing!
        qTbinMin, qTbinMax = int(2*qTbinMinGeV)+1, int(2*qTbinMaxGeV) # proper indexing!
        #qT = 0.5*(qTbinMinGeV + qTbinMaxGeV)
        

        print("***************** Do qTbin=[%f,%f],[%d, %d]" % (qTbinMinGeV, qTbinMaxGeV, qTbinMin, qTbinMax))
        
        
        
        # gen - parallel
        '''
        thn_gen_para_muon.GetAxis(1).SetRange(qTbinMin, qTbinMax)
        thn_gen_para_electron.GetAxis(1).SetRange(qTbinMin, qTbinMax)
        hist_muon = thn_gen_para_muon.Projection(0, "E")
        hist_electron = thn_gen_para_electron.Projection(0, "E")
        hist_muon.Add(hist_electron)
        hist_muon.Scale(1./hist_muon.Integral())
        name = "para_gen_qTbinIdx%d" % (qTbin)
        cfg['xtitle'] = "Recoil U_{#parallel} (GeV)"
        #hist_gk_pdf, hist_gk_cdf = doFit_GaussianKernel(outDir + "/gen/para/", qTbin, qTbinMinGeV, qTbinMaxGeV, hist_muon, name, label + ", MiNNLO", met, cfg)
        #histsToWrite.append(hist_gk_pdf)
        #histsToWrite.append(hist_gk_cdf)
        ###doFit_2Gauss(outDir + "/gen/para/", qTbin, qTbinMinGeV, qTbinMaxGeV, hist_muon, name, label + ", MiNNLO", met, cfg, h_para_gen_2g)
        ###doFit_3Gauss(outDir + "/gen/para/", qTbin, qTbinMinGeV, qTbinMaxGeV, hist_muon, name, label + ", MiNNLO", met, cfg, h_para_gen_3g)
        hist_muon.Delete()
        hist_electron.Delete()



        # gen - perpendicular
        thn_gen_perp_muon.GetAxis(1).SetRange(qTbinMin, qTbinMax)
        thn_gen_perp_electron.GetAxis(1).SetRange(qTbinMin, qTbinMax)
        hist_muon = thn_gen_perp_muon.Projection(0, "E")
        hist_electron = thn_gen_perp_electron.Projection(0, "E")
        hist_muon.Add(hist_electron)
        hist_muon.Scale(1./hist_muon.Integral())
        name = "perp_gen_qTbinIdx%d" % (qTbin)
        cfg['xtitle'] = "Recoil U_{#perp}  (GeV)"
        #hist_gk_pdf, hist_gk_cdf = doFit_GaussianKernel(outDir + "/gen/perp/", qTbin, qTbinMinGeV, qTbinMaxGeV, hist_muon, name, label + ", MiNNLO", met, cfg)
        #histsToWrite.append(hist_gk_pdf)
        #histsToWrite.append(hist_gk_cdf)
        ###doFit_2Gauss(outDir + "/gen/perp/", qTbin, qTbinMinGeV, qTbinMaxGeV, hist_muon, name, label + ", MiNNLO", met, cfg, h_perp_gen_2g)
        ###doFit_3Gauss(outDir + "/gen/perp/", qTbin, qTbinMinGeV, qTbinMaxGeV, hist_muon, name, label + ", MiNNLO", met, cfg, h_perp_gen_3g)
        hist_muon.Delete()
        hist_electron.Delete()
        '''
    
        # data - parallel
        thn_data_para_muon.GetAxis(1).SetRange(qTbinMin, qTbinMax)
        hist_muon = thn_data_para_muon.Projection(0, "E")
        hist_muon.SetName("data_para_muon")
        
        thn_data_para_electron.GetAxis(1).SetRange(qTbinMin, qTbinMax)
        hist_electron = thn_data_para_electron.Projection(0, "E")
        hist_electron.SetName("data_para_electron")
        hist_muon.Add(hist_electron)
        
        hist_bkg = None
        for bkg in bkgs:
            
            thn_bkgs['para_%s_muon' % bkg].GetAxis(1).SetRange(qTbinMin, qTbinMax)
            hist_muon_bkg = thn_bkgs['para_%s_muon' % bkg].Projection(0, "E")
            hist_muon_bkg.SetName(hist_muon_bkg.GetName() + "_muon")
            
            thn_bkgs['para_%s_electron' % bkg].GetAxis(1).SetRange(qTbinMin, qTbinMax)
            hist_electron_bkg = thn_bkgs['para_%s_electron' % bkg].Projection(0, "E")
            hist_electron_bkg.SetName(hist_muon_bkg.GetName() + "_electron")
            
            if hist_bkg == None: 
                hist_bkg = copy.deepcopy(hist_muon_bkg)
                hist_bkg.SetName("para_bkg_%d" % qTbinMin)
                
            else: hist_bkg.Add(hist_muon_bkg)
            
            hist_bkg.Add(hist_electron_bkg)
            
            hist_muon_bkg.Delete()
            hist_electron_bkg.Delete()
            
            
        hist_bkg.Scale(lumi)
        '''
        hist_muon.Add(hist_bkg, -1)
        for i in range(0, hist_muon.GetNbinsX()+1):
            if hist_muon.GetBinContent(i) <= 0: hist_muon.SetBinContent(i, 0)
        '''
        name = "para_data_qTbinIdx%d" % (qTbin)


        cfg['xtitle'] = "Recoil U_{#parallel} (GeV)"
        #hist_gk_pdf, hist_gk_cdf = doFit_GaussianKernel(outDir + "/data/para/", qTbin, qTbinMinGeV, qTbinMaxGeV, hist_muon, name, label + ", MiNNLO", met, cfg)
        #histsToWrite.append(hist_gk_pdf)
        #histsToWrite.append(hist_gk_cdf)
        #doFit_2GaussConstrained(outDir + "/data/para/", qTbin, qTbinMinGeV, qTbinMaxGeV, hist_muon, name, label + ", MiNNLO", met, cfg, h_para_data_2g)
        #doFit_bkg(outDir + "/data/para/", qTbin, qTbinMinGeV, qTbinMaxGeV, hist_bkg, name, label + ", MiNNLO", met, cfg, h_para_data_2g)
        doFit_2GaussConstrainedBkg(outDir + "/data/para/", qTbin, qTbinMinGeV, qTbinMaxGeV, hist_muon, hist_bkg, name, label + ", MiNNLO", met, cfg, h_para_data_2g)
        #doFit_Voigtian(outDir + "/data/para/", qTbin, qTbinMinGeV, qTbinMaxGeV, hist_muon, name, label + ", MiNNLO", met, cfg, h_para_data_2g)
        ###doFit_DSCB(outDir + "/data/para/", qTbin, qTbinMinGeV, qTbinMaxGeV, hist_muon, name, label + ", MiNNLO", met, cfg, h_para_data_2g)
        #doFit_3Gauss(outDir + "/data/para/", qTbin, qTbinMinGeV, qTbinMaxGeV, hist_muon, name, label + ", MiNNLO", met, cfg, h_para_data_3g)
        #doFit_2Gauss(outDir + "/data/para/", qTbin, qTbinMinGeV, qTbinMaxGeV, hist_muon, name, label + ", MiNNLO", met, cfg, h_para_data_2g)
        hist_muon.Delete()
        hist_electron.Delete()
        

        # data - perpendicular
        thn_data_perp_muon.GetAxis(1).SetRange(qTbinMin, qTbinMax)
        thn_data_perp_electron.GetAxis(1).SetRange(qTbinMin, qTbinMax)
        hist_muon = thn_data_perp_muon.Projection(0, "E")
        hist_electron = thn_data_perp_electron.Projection(0, "E")
        hist_muon.Add(hist_electron)

        hist_bkg = None
        for bkg in bkgs:
        
            thn_bkgs['perp_%s_muon' % bkg].GetAxis(1).SetRange(qTbinMin, qTbinMax)
            hist_muon_bkg = thn_bkgs['perp_%s_muon' % bkg].Projection(0, "E")
            hist_muon_bkg.SetName(hist_muon_bkg.GetName() + "_muon")
            
            thn_bkgs['perp_%s_electron' % bkg].GetAxis(1).SetRange(qTbinMin, qTbinMax)
            hist_electron_bkg = thn_bkgs['perp_%s_electron' % bkg].Projection(0, "E")
            hist_electron_bkg.SetName(hist_muon_bkg.GetName() + "_electron")
            
            if hist_bkg == None: 
                hist_bkg = copy.deepcopy(hist_muon_bkg)
                hist_bkg.SetName("para_bkg_%d" % qTbinMin)
                
            else: hist_bkg.Add(hist_muon_bkg)
            
            hist_bkg.Add(hist_electron_bkg)
            
            hist_muon_bkg.Delete()
            hist_electron_bkg.Delete()        
        


        hist_bkg.Scale(lumi)
        '''
        hist_muon.Add(hist_bkg, -1)
        for i in range(0, hist_muon.GetNbinsX()+1):
            if hist_muon.GetBinContent(i) <= 0: hist_muon.SetBinContent(i, 0)
        '''
        name = "perp_data_qTbinIdx%d" % (qTbin)
        cfg['xtitle'] = "Recoil U_{#perp}  (GeV)"
        #hist_gk_pdf, hist_gk_cdf = doFit_GaussianKernel(outDir + "/data/perp/", qTbin, qTbinMinGeV, qTbinMaxGeV, hist_muon, name, label + ", MiNNLO", met, cfg)
        #histsToWrite.append(hist_gk_pdf)
        #histsToWrite.append(hist_gk_cdf)
        #doFit_2GaussConstrained(outDir + "/data/perp/", qTbin, qTbinMinGeV, qTbinMaxGeV, hist_muon, name, label + ", MiNNLO", met, cfg, h_perp_data_2g)
        doFit_2GaussConstrainedBkg(outDir + "/data/perp/", qTbin, qTbinMinGeV, qTbinMaxGeV, hist_muon, hist_bkg, name, label + ", MiNNLO", met, cfg, h_perp_data_2g)
        #doFit_bkg(outDir + "/data/perp/", qTbin, qTbinMinGeV, qTbinMaxGeV, hist_bkg, name, label + ", MiNNLO", met, cfg, h_perp_data_2g)
        #doFit_Voigtian(outDir + "/data/perp/", qTbin, qTbinMinGeV, qTbinMaxGeV, hist_muon, name, label + ", MiNNLO", met, cfg, h_perp_data_2g)
        #doFit_3Gauss(outDir + "/data/perp/", qTbin, qTbinMinGeV, qTbinMaxGeV, hist_muon, name, label + ", MiNNLO", met, cfg, h_perp_data_3g)
        ###doFit_DSCB(outDir + "/data/perp/", qTbin, qTbinMinGeV, qTbinMaxGeV, hist_muon, name, label + ", MiNNLO", met, cfg, h_perp_data_3g)
        #doFit_2Gauss(outDir + "/data/perp/", qTbin, qTbinMinGeV, qTbinMaxGeV, hist_muon, name, label + ", MiNNLO", met, cfg, h_perp_data_2g)
        hist_muon.Delete()
        hist_electron.Delete()
        
        
        # mc - parallel
        thn_reco_para_muon.GetAxis(1).SetRange(qTbinMin, qTbinMax)
        thn_reco_para_electron.GetAxis(1).SetRange(qTbinMin, qTbinMax)
        hist_muon = thn_reco_para_muon.Projection(0, "E")
        hist_electron = thn_reco_para_electron.Projection(0, "E")
        hist_muon.Add(hist_electron)

        name = "para_reco_qTbinIdx%d" % (qTbin)
        cfg['xtitle'] = "Recoil U_{#parallel} (GeV)"
        #hist_gk_pdf, hist_gk_cdf = doFit_GaussianKernel(outDir + "/reco/para/", qTbin, qTbinMinGeV, qTbinMaxGeV, hist_muon, name, label + ", MiNNLO", met, cfg)
        #histsToWrite.append(hist_gk_pdf)
        #histsToWrite.append(hist_gk_cdf)
        #doFit_2GaussConstrained(outDir + "/reco/para/", qTbin, qTbinMinGeV, qTbinMaxGeV, hist_muon, name, label + ", MiNNLO", met, cfg, h_para_mc_2g)
        #doFit_Voigtian(outDir + "/reco/para/", qTbin, qTbinMinGeV, qTbinMaxGeV, hist_muon, name, label + ", MiNNLO", met, cfg, h_para_mc_2g)
        #doFit_DSCB(outDir + "/reco/para/", qTbin, qTbinMinGeV, qTbinMaxGeV, hist_muon, name, label + ", MiNNLO", met, cfg, h_para_mc_2g)
        #doFit_3Gauss(outDir + "/reco/para/", qTbin, qTbinMinGeV, qTbinMaxGeV, hist_muon, name, label + ", MiNNLO", met, cfg, h_para_mc_2g)
        doFit_2Gauss(outDir + "/reco/para/", qTbin, qTbinMinGeV, qTbinMaxGeV, hist_muon, name, label + ", MiNNLO", met, cfg, h_para_mc_2g)
        hist_muon.Delete()
        hist_electron.Delete()

        # mc - perpendicular
        thn_reco_perp_muon.GetAxis(1).SetRange(qTbinMin, qTbinMax)
        thn_reco_perp_electron.GetAxis(1).SetRange(qTbinMin, qTbinMax)
        hist_muon = thn_reco_perp_muon.Projection(0, "E")
        hist_electron = thn_reco_perp_electron.Projection(0, "E")
        hist_muon.Add(hist_electron)

        name = "perp_reco_qTbinIdx%d" % (qTbin)
        cfg['xtitle'] = "Recoil U_{#perp}  (GeV)"
        #hist_gk_pdf, hist_gk_cdf = doFit_GaussianKernel(outDir + "/reco/perp/", qTbin, qTbinMinGeV, qTbinMaxGeV, hist_muon, name, label + ", MiNNLO", met, cfg)
        #histsToWrite.append(hist_gk_pdf)
        #histsToWrite.append(hist_gk_cdf)
        doFit_2Gauss(outDir + "/reco/perp/", qTbin, qTbinMinGeV, qTbinMaxGeV, hist_muon, name, label + ", MiNNLO", met, cfg, h_perp_mc_2g)
        #doFit_2GaussConstrained(outDir + "/reco/perp/", qTbin, qTbinMinGeV, qTbinMaxGeV, hist_muon, name, label + ", MiNNLO", met, cfg, h_perp_mc_2g)
        #doFit_3Gauss(outDir + "/reco/perp/", qTbin, qTbinMinGeV, qTbinMaxGeV, hist_muon, name, label + ", MiNNLO", met, cfg, h_perp_mc_3g)
        ###doFit_DSCB(outDir + "/reco/perp/", qTbin, qTbinMinGeV, qTbinMaxGeV, hist_muon, name, label + ", MiNNLO", met, cfg, h_perp_mc_3g)
        hist_muon.Delete()
        hist_electron.Delete()
        
        #sys.exit()
        
        
    fOut_ = "analyses/lowPU/recoilCorrections/recoil_Z_%s.root" % met
    fOut = ROOT.TFile(fOut_, "RECREATE")
    
    h_para_data_2g.Write()
    h_para_mc_2g.Write()
    h_para_gen_2g.Write()
    h_perp_data_2g.Write()
    h_perp_mc_2g.Write()
    h_perp_gen_2g.Write()

    h_para_data_3g.Write()
    h_para_mc_3g.Write()
    h_para_gen_3g.Write()
    h_perp_data_3g.Write()
    h_perp_mc_3g.Write()
    h_perp_gen_3g.Write()
    
    for h in histsToWrite: h.Write()

    fOut.ls()
    fOut.Close()
    print("Written to %s" % fOut_)
    
    return
    hNames = [x.GetName() for x in fOut.GetListOfKeys()]
    

    if h_para_data.GetName() in hNames: fOut.Delete(h_para_data.GetName()+";*")
    if h_perp_data.GetName() in hNames: fOut.Delete(h_perp_data.GetName()+";*")
    if h_para_mc.GetName() in hNames: fOut.Delete(h_para_mc.GetName()+";*")
    if h_perp_mc.GetName() in hNames: fOut.Delete(h_perp_mc.GetName()+";*")
    h_para_data.Write()
    h_perp_data.Write()
    h_para_mc.Write()
    h_perp_mc.Write()
        
    if h_para_gen.GetName() in hNames: fOut.Delete(h_para_gen.GetName()+";*")
    if h_perp_gen.GetName() in hNames: fOut.Delete(h_perp_gen.GetName()+";*")
    h_para_gen.Write()
    h_perp_gen.Write()
    
    
    
    fOut.ls()
    fOut.Close()
    print("Written to %s" % fOut_)



def doParameterizedFit(cat, hOut, hist, hist_bkg=None):

    cfg = {

        'logy'              : False,
        'logx'              : False,
    
        'xmin'              : -300,
        'xmax'              : 100,
        'ymin'              : 0.0000001,
        'ymax'              : 1500,
        
        'xtitle'            : "Recoil U_{#parallel} (GeV)",
        'ytitle'            : "Events / 1 GeV",
        
        'topRight'          : "199 pb^{#minus1} (13 TeV)", 
        'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",
        
        'ratiofraction'     : 0.25,
        'ytitleR'           : "Ratio",
        'yminR'             : -5,
        'ymaxR'             : 5,
    }   

    recoil = ROOT.RooRealVar("recoil", "Recoil parallel (GeV)", 0, -500, 500)
    
    w = ROOT.RooWorkspace("w", "workspace")
    cats = ROOT.RooCategory("category", "") # for each qT bin, define category
    hists = ROOT.std.map("string, RooDataHist*")() # container holding all RooDataHists
    
    f_param_init = ROOT.TFile("wremnants/data/lowPU/recoil_fits_Z_param_init.root")
    h_param_init = f_param_init.Get(cat)
    

    # mean
    if "perp" in cat:
    
        mean_a = ROOT.RooRealVar("mean_a", "", 0) # constant
    
    else:
    
        mean_a = ROOT.RooRealVar("mean_a", "", h_param_init.GetBinContent(1, 1), -10, 10)
        mean_b = ROOT.RooRealVar("mean_b", "", h_param_init.GetBinContent(1, 2), -10, 10)
        mean_c = ROOT.RooRealVar("mean_c", "", h_param_init.GetBinContent(1, 3), -10, 10)
        mean_d = ROOT.RooRealVar("mean_d", "", h_param_init.GetBinContent(1, 4), -10, 10)
        mean_e = ROOT.RooRealVar("mean_e", "", h_param_init.GetBinContent(1, 5), -10, 10)
        mean_f = ROOT.RooRealVar("mean_f", "", h_param_init.GetBinContent(1, 6), -10, 10)
        mean_g = ROOT.RooRealVar("mean_g", "", h_param_init.GetBinContent(1, 7), -10, 10)
    
    # sigma1
    sigma1_a = ROOT.RooRealVar("sigma1_a", "", h_param_init.GetBinContent(2, 1), -10, 200)
    sigma1_b = ROOT.RooRealVar("sigma1_b", "", h_param_init.GetBinContent(2, 2), -10, 200)
    sigma1_c = ROOT.RooRealVar("sigma1_c", "", h_param_init.GetBinContent(2, 3), -100, 200)
    
    # sigma2
    sigma2_a = ROOT.RooRealVar("sigma2_a", "", h_param_init.GetBinContent(3, 1), 0, 200)
    sigma2_b = ROOT.RooRealVar("sigma2_b", "", h_param_init.GetBinContent(3, 2), -10, 300)
    sigma2_c = ROOT.RooRealVar("sigma2_c", "", h_param_init.GetBinContent(3, 3), -10, 10)
    

    norm_a = ROOT.RooRealVar("norm_a", "", h_param_init.GetBinContent(4, 1), 0, 1)
    #norm_a.setConstant(ROOT.kTRUE)

    f_param_init.Close()

    

    nBins = hist.GetNbinsX()
    for iBin in range(1, nBins): # exclude last bin
    
        qTbinMinGeV, qTbinMaxGeV = hist.GetXaxis().GetBinLowEdge(iBin), hist.GetXaxis().GetBinLowEdge(iBin+1)
        qT = hist.GetXaxis().GetBinCenter(iBin)
        
        print("***************** Do iBin %d [%f,%f]" % (iBin, qTbinMinGeV, qTbinMaxGeV))
        
        
        h = hist.ProjectionY("%s_bin%d" % (cat, iBin), iBin, iBin)
        h_integral = h.Integral()
        h.Scale(1./h_integral)
        print(h.GetMean(), h.GetRMS())
        
        
        # import and store RDH
        rdh = ROOT.RooDataHist("rdh_%d" % iBin, "", ROOT.RooArgList(recoil), ROOT.RooFit.Import(h))
        rdh.SetName("rdh_%s_bin%d" % (cat, iBin))
        catIDx = "cat_%s_bin%s" % (cat, iBin)
        hists.insert(ROOT.std.pair("string, RooDataHist*")(catIDx, rdh))
        cats.defineType(catIDx, iBin)    
        
        #print("----------------------------------->", rdh.numEntries(), rdh.sumEntries())
        getattr(w, 'import')(rdh) # import before PDF construction


        sigma1 = ROOT.RooFormulaVar("%s_sigma1_bin%d" % (cat, iBin), "@0*TMath::Power(%f+@1, @2)" % (qT), ROOT.RooArgList(sigma1_a, sigma1_b, sigma1_c))
        sigma2 = ROOT.RooFormulaVar("%s_sigma2_bin%d" % (cat, iBin), "@0*TMath::Power(%f+@1, @2)" % (qT), ROOT.RooArgList(sigma2_a, sigma2_b, sigma2_c))
        if "perp" in cat: mean = mean_a
        #else: mean = ROOT.RooFormulaVar("%s_mean_bin%d" % (cat, iBin), "(@0*%f*%f*%f + @1*%f*%f + @2*%f)/(@3*%f*%f + @4*%f + 1.0)" % (qT, qT, qT, qT, qT, qT, qT, qT, qT), ROOT.RooArgList(mean_a, mean_b, mean_c, mean_d, mean_e))
        else: mean = ROOT.RooFormulaVar("%s_mean_bin%d" % (cat, iBin), "(@0*%f*%f*%f*%f + @1*%f*%f*%f + @2*%f*%f + @3*%f)/(@4*%f*%f*%f + @5*%f*%f + @6*%f + 1.0)" % (qT, qT, qT, qT, qT, qT, qT, qT, qT, qT, qT, qT, qT, qT, qT, qT), ROOT.RooArgList(mean_a, mean_b, mean_c, mean_d, mean_e, mean_f, mean_g))
        norm = ROOT.RooFormulaVar("%s_norm_bin%d" % (cat, iBin), "@0", ROOT.RooArgList(norm_a))

        gauss1 = ROOT.RooGaussian("%s_gauss1_bin%d" % (cat, iBin), "", recoil, mean, sigma1)
        gauss2 = ROOT.RooGaussian("%s_gauss2_bin%d" % (cat, iBin), "", recoil, mean, sigma2)
        
        if "data" in cat:
        
            # do background-fit
            if hist_bkg == None: sys.exit("Background histogram necessary for data")
            h_bkg = hist_bkg.ProjectionY("%s_bin%d" % (cat, iBin), iBin, iBin)
            h_bkg_integral = h_bkg.Integral()
            h_bkg.Scale(1./h_bkg_integral)
            
            mean1_bkg = ROOT.RooRealVar("%s_mean1_bkg_bin%d" % (cat, iBin), "", h_bkg.GetMean(), -400, 100)
            mean2_bkg = ROOT.RooRealVar("%s_mean2_bkg_bin%d" % (cat, iBin), "", h_bkg.GetMean(), -400, 100)
            sigma1_bkg = ROOT.RooRealVar("%s_sigma1_bkg_bin%d" % (cat, iBin), "", h_bkg.GetRMS()/2, 0.1, 100)
            sigma2_bkg = ROOT.RooRealVar("%s_sigma2_bkg_bin%d" % (cat, iBin), "", h_bkg.GetRMS()*2, 0.1, 100)
            norm_bkg = ROOT.RooRealVar("%s_norm_bkg_bin%d" % (cat, iBin), "", 0.9, 0, 1) # first Gauss = ttbar
            gauss1_bkg = ROOT.RooGaussian("%s_gauss1_bkg_bin%d" % (cat, iBin), "", recoil, mean1_bkg, sigma1_bkg)
            gauss2_bkg = ROOT.RooGaussian("%s_gauss2_bkg_bin%d" % (cat, iBin), "", recoil, mean2_bkg, sigma2_bkg)
            pdf_bkg = ROOT.RooAddPdf("%s_pdf_bkg_bin%d" % (cat, iBin), '', ROOT.RooArgList(gauss1_bkg, gauss2_bkg), ROOT.RooArgList(norm_bkg))

            rdh_bkg = ROOT.RooDataHist("%s_rdh_bkg_bin%d" % (cat, iBin), "", ROOT.RooArgList(recoil), ROOT.RooFit.Import(h_bkg))
            fitRes_bkg = pdf_bkg.fitTo(rdh_bkg, ROOT.RooFit.SumW2Error(ROOT.kTRUE), ROOT.RooFit.Save(ROOT.kTRUE))
    
            # freeze the background parameters for now
            sigma1_bkg.setConstant(ROOT.kTRUE)
            mean1_bkg.setConstant(ROOT.kTRUE)
            sigma2_bkg.setConstant(ROOT.kTRUE)
            mean2_bkg.setConstant(ROOT.kTRUE)
            norm_bkg.setConstant(ROOT.kTRUE)
          
            # create joint PDF
            pdf_sig = ROOT.RooAddPdf("%s_pdf_sig_bin%d" % (cat, iBin), '', ROOT.RooArgList(gauss1, gauss2), ROOT.RooArgList(norm))
            norm_sig = ROOT.RooRealVar("%s_norm_sig_bin%d" % (cat, iBin), "", (1. - h_bkg_integral/h_integral)) # fixed norm , 0.95, 1
            pdf = ROOT.RooAddPdf("%s_pdf_bin%d" % (cat, iBin), '', ROOT.RooArgList(pdf_sig, pdf_bkg), ROOT.RooArgList(norm_sig))


        else:
        
            pdf = ROOT.RooAddPdf("%s_pdf_bin%d" % (cat, iBin), '', ROOT.RooArgList(gauss1, gauss2), ROOT.RooArgList(norm))
            

        getattr(w, 'import')(pdf) # import must happen after the initial guesses
        
        #print("----------------------------------->", rdh.numEntries())
        
        
    
    w.Print()

    print("#############################################################################################################################")
    
    recoil.setBins(1000) # seems to be necessary...
    #qt = ROOT.RooRealVar("qt", "", 0, 0, 1) 
    
    # construct total RDH
    rdh_tot = ROOT.RooDataHist("rdh_tot", "", ROOT.RooArgList(recoil), cats, hists)
    
    
    '''
    cats.setIndex(1)
    print("Categories")
    for cat in cats:
        print(cat)
    print("------")
    '''
    
    # construct total PDF
    pdf_tot = ROOT.RooSimultaneous("pdf_tot", "", cats)
    for iBin in range(1, nBins):
        
        pdf =  w.obj("%s_pdf_bin%d" % (cat, iBin))
        pdf.Print()
        pdf_tot.addPdf(pdf, "cat_%s_bin%s" % (cat, iBin))
        
    
    
    #selectedPdf = pdf_tot.getPdf("iBin1")
    #selectedPdf.Print()
    #print(hists['iBin1'])
    #selectedPdf.fitTo(hists['qTbin0'], ROOT.RooFit.SumW2Error(ROOT.kTRUE))

    fitRes = pdf_tot.fitTo(rdh_tot, ROOT.RooFit.SumW2Error(ROOT.kTRUE), ROOT.RooFit.Save(ROOT.kTRUE))
    #pdfTot.fitTo(totrdh, ROOT.RooFit.Extended(ROOT.kTRUE), ROOT.RooFit.SumW2Error(ROOT.kTRUE))
    #fitRes = pdf.fitTo(rdh, ROOT.RooFit.Extended(ROOT.kTRUE), ROOT.RooFit.SumW2Error(ROOT.kTRUE), ROOT.RooFit.Save(ROOT.kTRUE))
 
    params = fitRes.floatParsFinal()
 

    #print("----------------------------------->", rdh.numEntries())
    #print("----------------------------------->", totrdh.numEntries())
    #print("----------------------------------->", hists['qTbin0'].numEntries())
    
    # do validation
    doValidation = False
    if doValidation:
    
        outDir = "/eos/user/j/jaeyserm/www/wmass/lowPU/recoilCorrection/fits_Z_new/refit/%s/" % cat
        functions.prepareDir(outDir)
        for iBin in range(1, nBins): # exclude last bin
        
            qTbinMinGeV, qTbinMaxGeV = hist.GetXaxis().GetBinLowEdge(iBin), hist.GetXaxis().GetBinLowEdge(iBin+1)
            qT = hist.GetXaxis().GetBinCenter(iBin)
            
            h = hist.ProjectionY("%s_sig_bin%d" % (cat, iBin), iBin, iBin)
            h_bkg = hist_bkg.ProjectionY("%s_bkg_bin%d" % (cat, iBin), iBin, iBin)
        
            pdf =  w.obj("%s_pdf_bin%d" % (cat, iBin))
            pdf.Print()
            if "data" in cat: doFit_2GaussConstrainedBkg(outDir, iBin, qTbinMinGeV, qTbinMaxGeV, h, h_bkg, cat, "Data", cfg, refittedPdf = pdf)
            else: doFit_2Gauss(outDir, iBin, qTbinMinGeV, qTbinMaxGeV, h, cat, "MC", cfg, refittedPdf = pdf)
            

    
    # save parameters
    if not "perp" in cat:
        ### need to go through the workspace, direct mean_a.getVal() returns the initial ones
        hOut.SetBinContent(1, 1, 1, w.obj("mean_a").getVal())
        hOut.SetBinContent(1, 2, 1, w.obj("mean_b").getVal())
        hOut.SetBinContent(1, 3, 1, w.obj("mean_c").getVal())
        hOut.SetBinContent(1, 4, 1, w.obj("mean_d").getVal())
        hOut.SetBinContent(1, 5, 1, w.obj("mean_e").getVal())
        hOut.SetBinContent(1, 6, 1, w.obj("mean_f").getVal())
        hOut.SetBinContent(1, 7, 1, w.obj("mean_g").getVal())
    
    hOut.SetBinContent(2, 1, 1, w.obj("sigma1_a").getVal())
    hOut.SetBinContent(2, 2, 1, w.obj("sigma1_b").getVal())
    hOut.SetBinContent(2, 3, 1, w.obj("sigma1_c").getVal())
    
    hOut.SetBinContent(3, 1, 1, w.obj("sigma2_a").getVal())
    hOut.SetBinContent(3, 2, 1, w.obj("sigma2_b").getVal())
    hOut.SetBinContent(3, 3, 1, w.obj("sigma2_c").getVal())
    
    hOut.SetBinContent(4, 1, 1, w.obj("norm_a").getVal())
    

    # do uncertainties        
    cov = fitRes.covarianceMatrix() # covariance matrix: diagonal elements are the variances of each of the parameters
    npars = fitRes.floatParsFinal().getSize()
    sigma = 1
        
    
    # convert to numpy arrays
    nom = np.zeros(npars)   
    cov_ = np.zeros((npars, npars))
    for i in range(0, npars):
    
        nom[i] = params[i].getVal()
        for j in range(0, npars):
            cov_[i][j] = cov[i][j]
            

    # eigenvalues and eigenvectors
    eig_vals, eig_vec = np.linalg.eig(cov_)
    eig_vec_inv = np.linalg.inv(eig_vec)
    
    print(eig_vals)
        
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
    
    def getIdx(name, params):
    
        for i in range(0, len(params)): 
       
            if name == params[i].GetName(): return i
            
        return -1
      


    print("%f " % nom[getIdx("mean_a", params)], end =" ")
    print("%f " % nom[getIdx("mean_b", params)], end =" ")
    print("%f " % nom[getIdx("mean_c", params)], end =" ")
    print("%f " % nom[getIdx("mean_d", params)], end =" ")
    print("%f " % nom[getIdx("mean_e", params)], end =" ")
    print("%f " % nom[getIdx("mean_f", params)], end =" ")
    print("%f " % nom[getIdx("mean_g", params)], end =" ")
    
    print("%f " % nom[getIdx("sigma1_a", params)], end =" ")
    print("%f " % nom[getIdx("sigma1_b", params)], end =" ")
    print("%f " % nom[getIdx("sigma1_c", params)], end =" ")
    
    print("%f " % nom[getIdx("sigma2_a", params)], end =" ")
    print("%f " % nom[getIdx("sigma2_b", params)], end =" ")
    print("%f " % nom[getIdx("sigma2_c", params)], end =" ")
    
    print()
        
    for iVar in range(0, npars):
        
        print("******** vary", iVar+1, "********")
            
        dnom = copy.deepcopy(eig_vals)
        for i, eig in enumerate(dnom): 
           
            if i == iVar: dnom[i] = sigma*math.sqrt(eig)
            else: dnom[i] = 0
               
        #print("PERTURBED NON-ROTATED")
        #print(dnom)
            
        # rotate perturbation back to nominal base
        dnom = np.dot(eig_vec, dnom)
        nom_pert = nom + dnom
        #print("PERTURBED ROTATED")
        #print(dnom)
      
           
        print("%f " % nom_pert[getIdx("mean_a", params)], end =" ")
        print("%f " % nom_pert[getIdx("mean_b", params)], end =" ")
        print("%f " % nom_pert[getIdx("mean_c", params)], end =" ")
        print("%f " % nom_pert[getIdx("mean_d", params)], end =" ")
        print("%f " % nom_pert[getIdx("mean_e", params)], end =" ")
        print("%f " % nom_pert[getIdx("mean_f", params)], end =" ")
        print("%f " % nom_pert[getIdx("mean_g", params)], end =" ")
        
        print("%f " % nom_pert[getIdx("sigma1_a", params)], end =" ")
        print("%f " % nom_pert[getIdx("sigma1_b", params)], end =" ")
        print("%f " % nom_pert[getIdx("sigma1_c", params)], end =" ")
        
        print("%f " % nom_pert[getIdx("sigma2_a", params)], end =" ")
        print("%f " % nom_pert[getIdx("sigma2_b", params)], end =" ")
        print("%f " % nom_pert[getIdx("sigma2_c", params)], end =" ")
        
        print()
            
        # the order of nom and nom_pert follows the order or "params"
        
        # save parameters
        if not "perp" in cat:

            hOut.SetBinContent(1, 1, iVar+2, nom_pert[getIdx("mean_a", params)])
            hOut.SetBinContent(1, 2, iVar+2, nom_pert[getIdx("mean_b", params)])
            hOut.SetBinContent(1, 3, iVar+2, nom_pert[getIdx("mean_c", params)])
            hOut.SetBinContent(1, 4, iVar+2, nom_pert[getIdx("mean_d", params)])
            hOut.SetBinContent(1, 5, iVar+2, nom_pert[getIdx("mean_e", params)])
            hOut.SetBinContent(1, 6, iVar+2, nom_pert[getIdx("mean_f", params)])
            hOut.SetBinContent(1, 7, iVar+2, nom_pert[getIdx("mean_g", params)])
        
        hOut.SetBinContent(2, 1, iVar+2, nom_pert[getIdx("sigma1_a", params)])
        hOut.SetBinContent(2, 2, iVar+2, nom_pert[getIdx("sigma1_b", params)])
        hOut.SetBinContent(2, 3, iVar+2, nom_pert[getIdx("sigma1_c", params)])
        
        hOut.SetBinContent(3, 1, iVar+2, nom_pert[getIdx("sigma2_a", params)])
        hOut.SetBinContent(3, 2, iVar+2, nom_pert[getIdx("sigma2_b", params)])
        hOut.SetBinContent(3, 3, iVar+2, nom_pert[getIdx("sigma2_c", params)])
        
        hOut.SetBinContent(4, 1, iVar+2, nom_pert[getIdx("norm_a", params)]) # cte
    

def autoFit():

    hOut_para_data = ROOT.TH3D("para_data", "", 10, 0, 10, 20, 0, 20, 20, 0, 20) # first idx = mean, sigma1, sigma2, norm; second idx = parameters of the fit
    hOut_para_mc = ROOT.TH3D("para_mc", "", 10, 0, 10, 20, 0, 20, 20, 0, 20)
    hOut_perp_data = ROOT.TH3D("perp_data", "", 10, 0, 10, 20, 0, 20, 20, 0, 20) # first idx = mean, sigma1, sigma2, norm; second idx = parameters of the fit
    hOut_perp_mc = ROOT.TH3D("perp_mc", "", 10, 0, 10, 20, 0, 20, 20, 0, 20)
    
    # load all histograms
    b_data_para = groups_mumu.readProc("recoil_uncorr_para_qt", "SingleMuon") # + groups_ee.readProc("recoil_uncorr_para_qt", "SingleElectron")
    b_data_perp = groups_mumu.readProc("recoil_uncorr_perp_qt", "SingleMuon") # + groups_ee.readProc("recoil_uncorr_perp_qt", "SingleElectron")
    b_mc_para = groups_mumu.readProc("recoil_uncorr_para_qt", "DYmumu") # + groups_ee.readProc("recoil_uncorr_para_qt", "DYee")
    b_mc_perp = groups_mumu.readProc("recoil_uncorr_perp_qt", "DYmumu") #+ groups_ee.readProc("recoil_uncorr_perp_qt", "DYee")
    
    b_bkg_para, b_bkg_perp = None, None
    for bkg in bkg_procs:
    
        b_para = groups_mumu.readProc("recoil_uncorr_para_qt", bkg)
        if b_bkg_para == None: b_bkg_para = b_para
        else: b_bkg_para += b_para
    
        b_perp = groups_mumu.readProc("recoil_uncorr_perp_qt", bkg)
        if b_bkg_perp == None: b_bkg_perp = b_perp
        else: b_bkg_perp += b_perp
      
      
    h_data_para = narf.hist_to_root(b_data_para)
    h_mc_para = narf.hist_to_root(b_mc_para)
    h_bkg_para = narf.hist_to_root(b_bkg_para)
    h_data_perp = narf.hist_to_root(b_data_perp)
    h_mc_perp = narf.hist_to_root(b_mc_perp)
    h_bkg_perp = narf.hist_to_root(b_bkg_perp)
    
    
    

    ############
    doParameterizedFit("para_mc", hOut_para_mc, h_mc_para)
    doParameterizedFit("para_data", hOut_para_data, h_data_para, hist_bkg=h_bkg_para)
    doParameterizedFit("perp_mc", hOut_perp_mc, h_mc_perp)
    doParameterizedFit("perp_data", hOut_perp_data, h_data_perp, hist_bkg=h_bkg_perp)

    fOut_ = "wremnants/data/lowPU/recoil_fits_Z_param_refit.root"
    fOut = ROOT.TFile(fOut_, "RECREATE")
    
    hOut_para_mc.Write()
    hOut_para_data.Write()
    hOut_perp_mc.Write()
    hOut_perp_data.Write()

    fOut.ls()
    fOut.Close()
    
        

def doRecoilFits_fine(cat, met, label, qTbins):
    
    outDir = "/eos/user/j/jaeyserm/www/wmass/lowPU/recoilCorrection/fits_Z/%s/" % met
    

    
    nGauss = 2
    # first index: recoil bin
    # second index: gauss component
    # third index: fit parameters (mean, sigma, norm)
    
    # third index = 1 --> nominal, the rest are the stat variations
    h_para_data_2g = ROOT.TH3D("para_data_%s_2gauss" % met, "", len(qTbins), 0, len(qTbins), nGauss*3, 0, nGauss*3, 20, 0, 20)
    h_para_mc_2g = ROOT.TH3D("para_reco_%s_2gauss" % met, "", len(qTbins), 0, len(qTbins), nGauss*3, 0, nGauss*3, 20, 0, 20)
    h_para_gen_2g = ROOT.TH3D("para_gen_%s_2gauss" % met, "", len(qTbins), 0, len(qTbins), nGauss*3, 0, nGauss*3, 20, 0, 20) 
    h_perp_data_2g = ROOT.TH3D("perp_data_%s_2gauss" % met, "", len(qTbins), 0, len(qTbins), nGauss*3, 0, nGauss*3, 20, 0, 20) 
    h_perp_mc_2g = ROOT.TH3D("perp_reco_%s_2gauss" % met, "", len(qTbins), 0, len(qTbins), nGauss*3, 0, nGauss*3, 20, 0, 20) 
    h_perp_gen_2g = ROOT.TH3D("perp_gen_%s_2gauss" % met, "", len(qTbins), 0, len(qTbins), nGauss*3, 0, nGauss*3, 20, 0, 20)
    
    nGauss = 3
    # first index: recoil bin
    # second index: gauss component
    # third index: fit parameters (mean, sigma, norm)
    h_para_data_3g = ROOT.TH3D("para_data_%s_3gauss" % met, "", len(qTbins), 0, len(qTbins), 20, 0, 20, 20, 0, 20)
    h_para_mc_3g = ROOT.TH3D("para_reco_%s_3gauss" % met, "", len(qTbins), 0, len(qTbins), 20, 0, 20, 20, 0, 20)
    h_para_gen_3g = ROOT.TH3D("para_gen_%s_3gauss" % met, "", len(qTbins), 0, len(qTbins), 20, 0, 20, 20, 0, 20) 
    h_perp_data_3g = ROOT.TH3D("perp_data_%s_3gauss" % met, "", len(qTbins), 0, len(qTbins), 20, 0, 20, 20, 0, 20) 
    h_perp_mc_3g = ROOT.TH3D("perp_reco_%s_3gauss" % met, "", len(qTbins), 0, len(qTbins), 20, 0, 20, 20, 0, 20) 
    h_perp_gen_3g = ROOT.TH3D("perp_gen_%s_3gauss" % met, "", len(qTbins), 0, len(qTbins), 20, 0, 20, 20, 0, 20)

    cfg = {

        'logy'              : False,
        'logx'              : False,
    
        'xmin'              : -300,
        'xmax'              : 100,
        'ymin'              : 0,
        'ymax'              : 1500,
        
        'xtitle'            : "Recoil U_{#parallel} (GeV)",
        'ytitle'            : "Events / 1 GeV",
        
        'topRight'          : "199 pb^{#minus1} (13 TeV)", 
        'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",
        
        'ratiofraction'     : 0.25,
        'ytitleR'           : "Pull",
        'yminR'             : -6,
        'ymaxR'             : 6,
    }
    

    prepareDir(outDir)
    prepareDir(outDir + "/data/")
    prepareDir(outDir + "/reco/")
    prepareDir(outDir + "/data/para/")
    prepareDir(outDir + "/data/perp/")
    prepareDir(outDir + "/reco/para/")
    prepareDir(outDir + "/reco/perp/")
    prepareDir(outDir + "/gen/")
    prepareDir(outDir + "/gen/para/")
    prepareDir(outDir + "/gen/perp/")

    bkgs = ["TTTo2L2Nu", "TTToSemiLeptonic", "WplusJetsToMuNu", "WminusJetsToMuNu", "WZTo3LNu", "WWTo2L2Nu", "WplusJetsToTauNu", "WminusJetsToTauNu"]


    # the following histograms are binned in 0.5 GeV for qT, from 0 to 500 GeV
    fIn_muon = ROOT.TFile("/eos/user/j/jaeyserm/analysis/lowPU/Zmumu/output.root")
    thn_gen_para_muon = fIn_muon.Get("recoil_uncorr_para_qtgen_DYmumu_MiNNLO")
    thn_gen_perp_muon = fIn_muon.Get("recoil_uncorr_perp_qtgen_DYmumu_MiNNLO")
    thn_reco_para_muon = fIn_muon.Get("recoil_uncorr_para_qt_DYmumu_MiNNLO")
    thn_reco_perp_muon = fIn_muon.Get("recoil_uncorr_perp_qt_DYmumu_MiNNLO")
    thn_data_para_muon = fIn_muon.Get("recoil_uncorr_para_qt_singlemuon")
    thn_data_perp_muon = fIn_muon.Get("recoil_uncorr_perp_qt_singlemuon")
    
    thn_bkgs = {}
    for bkg in bkgs:
    
        thn_bkgs['para_%s_muon' % bkg] = fIn_muon.Get("recoil_uncorr_para_qt_%s" % bkg)
        thn_bkgs['perp_%s_muon' % bkg] = fIn_muon.Get("recoil_uncorr_perp_qt_%s" % bkg)

    
 
    fIn_electron = ROOT.TFile("/eos/user/j/jaeyserm/analysis/lowPU/Zee/output.root")
    thn_gen_para_electron = fIn_electron.Get("recoil_uncorr_para_qtgen_DYee_MiNNLO")
    thn_gen_perp_electron = fIn_electron.Get("recoil_uncorr_perp_qtgen_DYee_MiNNLO")
    thn_reco_para_electron = fIn_electron.Get("recoil_uncorr_para_qt_DYee_MiNNLO")
    thn_reco_perp_electron = fIn_electron.Get("recoil_uncorr_perp_qt_DYee_MiNNLO")
    thn_data_para_electron = fIn_electron.Get("recoil_uncorr_para_qt_singleelectron")
    thn_data_perp_electron = fIn_electron.Get("recoil_uncorr_perp_qt_singleelectron")
    thn_ttbar_para_electron = fIn_electron.Get("recoil_uncorr_para_qt_TTTo2L2Nu")
    thn_ttbar_perp_electron = fIn_electron.Get("recoil_uncorr_perp_qt_TTTo2L2Nu")
    
    for bkg in bkgs:
    
        thn_bkgs['para_%s_electron' % bkg] = fIn_electron.Get("recoil_uncorr_para_qt_%s" % bkg)
        thn_bkgs['perp_%s_electron' % bkg] = fIn_electron.Get("recoil_uncorr_perp_qt_%s" % bkg)

    fIn_muon.Close()
    fIn_electron.Close()

    histsToWrite = []
    for qTbin in range(0, len(qTbins)-1):
    
        #qTbinMinGeV, qTbinMaxGeV = qTbins[qTbin], qTbins[qTbin+1]
        #qTbinMin, qTbinMax = int(2*qTbinMinGeV)+1, int(2*qTbinMaxGeV) # binned in 0.5 GeV
        
        qTbinMinGeV, qTbinMaxGeV = qTbins[qTbin], qTbins[qTbin+1]
        qTbinMin, qTbinMax = int(qTbins[qTbin]+1), int(qTbins[qTbin+1]) # proper indexing!
        #qT = 0.5*(qTbinMinGeV + qTbinMaxGeV)

        print("***************** Do qTbin=[%f,%f],[%d, %d]" % (qTbinMinGeV, qTbinMaxGeV, qTbinMin, qTbinMax))
        


        
        # data - parallel
        thn_data_para_muon.GetAxis(1).SetRange(qTbinMin, qTbinMax)
        thn_data_para_electron.GetAxis(1).SetRange(qTbinMin, qTbinMax)
        hist_muon = thn_data_para_muon.Projection(0, "E")
        hist_electron = thn_data_para_electron.Projection(0, "E")
        hist_muon.Add(hist_electron)

        hist_bkg = None
        for bkg in bkgs:
            
            thn_bkgs['para_%s_muon' % bkg].GetAxis(1).SetRange(qTbinMin, qTbinMax)
            hist_muon_bkg = thn_bkgs['para_%s_muon' % bkg].Projection(0, "E")
            hist_muon.SetName("muon_%s_%d" % (bkg, qTbinMin))
            
            thn_bkgs['para_%s_electron' % bkg].GetAxis(1).SetRange(qTbinMin, qTbinMax)
            hist_electron_bkg = thn_bkgs['para_%s_electron' % bkg].Projection(0, "E")
            hist_electron_bkg.SetName("muon_%s_%d" % (bkg, qTbinMin))
            
            if hist_bkg == None: hist_bkg = hist_muon_bkg
            else: hist_bkg.Add(hist_muon_bkg)
            
            hist_bkg.Add(hist_electron_bkg)

        hist_bkg.Scale(lumi)

        for iBin in range(0, hist_muon.GetNbinsX()+1):
            if hist_muon.GetBinContent(iBin) > 0:
                newVal = hist_muon.GetBinContent(iBin) - hist_bkg.GetBinContent(iBin)
                if newVal < 0: newVal = 0
                hist_muon.SetBinContent(iBin, newVal)
          
        hist_muon.Scale(1./hist_muon.Integral())        
        name = "para_data_qTbinIdx%d" % (qTbin)

        cfg['xtitle'] = "Recoil U_{#parallel} (GeV)"
        #hist_gk_pdf, hist_gk_cdf = doFit_GaussianKernel(outDir + "/data/para/", qTbin, qTbinMinGeV, qTbinMaxGeV, hist_muon, name, label + ", MiNNLO", met, cfg)
        #histsToWrite.append(hist_gk_pdf)
        #histsToWrite.append(hist_gk_cdf)
        doFit_2Gauss(outDir + "/data/para/", qTbin, qTbinMinGeV, qTbinMaxGeV, hist_muon, name, label + ", MiNNLO", met, cfg, h_para_data_2g)
        #doFit_DSCB(outDir + "/data/para/", qTbin, qTbinMinGeV, qTbinMaxGeV, hist_muon, name, label + ", MiNNLO", met, cfg, h_para_data_2g)
        
        
        #doFit_3Gauss(outDir + "/data/para/", qTbin, qTbinMinGeV, qTbinMaxGeV, hist_muon, name, label + ", MiNNLO", met, cfg, h_para_data_3g)
        hist_muon.Delete()
        hist_electron.Delete()
        
        continue

        # data - perpendicular
        thn_data_perp_muon.GetAxis(1).SetRange(qTbinMin, qTbinMax)
        thn_data_perp_electron.GetAxis(1).SetRange(qTbinMin, qTbinMax)
        hist_muon = thn_data_perp_muon.Projection(0, "E")
        hist_electron = thn_data_perp_electron.Projection(0, "E")
        hist_muon.Add(hist_electron)

        hist_bkg = None
        for bkg in bkgs:
            
            thn_bkgs['para_%s_muon' % bkg].GetAxis(1).SetRange(qTbinMin, qTbinMax)
            hist_muon_bkg = thn_bkgs['para_%s_muon' % bkg].Projection(0, "E")
            hist_muon.SetName("muon_%s_%d" % (bkg, qTbinMin))
            
            thn_bkgs['para_%s_electron' % bkg].GetAxis(1).SetRange(qTbinMin, qTbinMax)
            hist_electron_bkg = thn_bkgs['para_%s_electron' % bkg].Projection(0, "E")
            hist_electron_bkg.SetName("muon_%s_%d" % (bkg, qTbinMin))
            
            if hist_bkg == None: hist_bkg = hist_muon_bkg
            else: hist_bkg.Add(hist_muon_bkg)
            
            hist_bkg.Add(hist_electron_bkg)

        hist_bkg.Scale(lumi)

        for iBin in range(0, hist_muon.GetNbinsX()+1):
            if hist_muon.GetBinContent(iBin) > 0:
                newVal = hist_muon.GetBinContent(iBin) - hist_bkg.GetBinContent(iBin)
                if newVal < 0: newVal = 0
                hist_muon.SetBinContent(iBin, newVal)        
        
           
        hist_muon.Scale(1./hist_muon.Integral())    
        
        name = "perp_data_qTbinIdx%d" % (qTbin)
        cfg['xtitle'] = "Recoil U_{#perp}  (GeV)"
        #hist_gk_pdf, hist_gk_cdf = doFit_GaussianKernel(outDir + "/data/perp/", qTbin, qTbinMinGeV, qTbinMaxGeV, hist_muon, name, label + ", MiNNLO", met, cfg)
        #histsToWrite.append(hist_gk_pdf)
        #histsToWrite.append(hist_gk_cdf)
        doFit_2Gauss(outDir + "/data/perp/", qTbin, qTbinMinGeV, qTbinMaxGeV, hist_muon, name, label + ", MiNNLO", met, cfg, h_perp_data_2g)
        #doFit_3Gauss(outDir + "/data/perp/", qTbin, qTbinMinGeV, qTbinMaxGeV, hist_muon, name, label + ", MiNNLO", met, cfg, h_perp_data_3g)
        hist_muon.Delete()
        hist_electron.Delete()

        
        # mc - parallel
        thn_reco_para_muon.GetAxis(1).SetRange(qTbinMin, qTbinMax)
        thn_reco_para_electron.GetAxis(1).SetRange(qTbinMin, qTbinMax)
        hist_muon = thn_reco_para_muon.Projection(0, "E")
        hist_electron = thn_reco_para_electron.Projection(0, "E")
        hist_muon.Add(hist_electron)
        hist_muon.Scale(1./hist_muon.Integral())
        name = "para_reco_qTbinIdx%d" % (qTbin)
        cfg['xtitle'] = "Recoil U_{#parallel} (GeV)"
        #hist_gk_pdf, hist_gk_cdf = doFit_GaussianKernel(outDir + "/reco/para/", qTbin, qTbinMinGeV, qTbinMaxGeV, hist_muon, name, label + ", MiNNLO", met, cfg)
        #histsToWrite.append(hist_gk_pdf)
        #histsToWrite.append(hist_gk_cdf)
        doFit_2Gauss(outDir + "/reco/para/", qTbin, qTbinMinGeV, qTbinMaxGeV, hist_muon, name, label + ", MiNNLO", met, cfg, h_para_mc_2g)
        #doFit_DSCB(outDir + "/reco/para/", qTbin, qTbinMinGeV, qTbinMaxGeV, hist_muon, name, label + ", MiNNLO", met, cfg, h_para_mc_2g)
        #doFit_3Gauss(outDir + "/reco/para/", qTbin, qTbinMinGeV, qTbinMaxGeV, hist_muon, name, label + ", MiNNLO", met, cfg, h_para_mc_3g)
        hist_muon.Delete()
        hist_electron.Delete()

        # mc - perpendicular
        thn_reco_perp_muon.GetAxis(1).SetRange(qTbinMin, qTbinMax)
        thn_reco_perp_electron.GetAxis(1).SetRange(qTbinMin, qTbinMax)
        hist_muon = thn_reco_perp_muon.Projection(0, "E")
        hist_electron = thn_reco_perp_electron.Projection(0, "E")
        hist_muon.Add(hist_electron)
        hist_muon.Scale(1./hist_muon.Integral())
        name = "perp_reco_qTbinIdx%d" % (qTbin)
        cfg['xtitle'] = "Recoil U_{#perp}  (GeV)"
        #hist_gk_pdf, hist_gk_cdf = doFit_GaussianKernel(outDir + "/reco/perp/", qTbin, qTbinMinGeV, qTbinMaxGeV, hist_muon, name, label + ", MiNNLO", met, cfg)
        #histsToWrite.append(hist_gk_pdf)
        #histsToWrite.append(hist_gk_cdf)
        doFit_2Gauss(outDir + "/reco/perp/", qTbin, qTbinMinGeV, qTbinMaxGeV, hist_muon, name, label + ", MiNNLO", met, cfg, h_perp_mc_2g)
        #doFit_3Gauss(outDir + "/reco/perp/", qTbin, qTbinMinGeV, qTbinMaxGeV, hist_muon, name, label + ", MiNNLO", met, cfg, h_perp_mc_3g)
        hist_muon.Delete()
        hist_electron.Delete()
        
        
    fOut_ = "analyses/lowPU/recoilCorrections/recoil_Z_%s.root" % met
    fOut = ROOT.TFile(fOut_, "RECREATE")
    
    h_para_data_2g.Write()
    h_para_mc_2g.Write()
    h_para_gen_2g.Write()
    h_perp_data_2g.Write()
    h_perp_mc_2g.Write()
    h_perp_gen_2g.Write()

    h_para_data_3g.Write()
    h_para_mc_3g.Write()
    h_para_gen_3g.Write()
    h_perp_data_3g.Write()
    h_perp_mc_3g.Write()
    h_perp_gen_3g.Write()
    
    for h in histsToWrite: h.Write()

    fOut.ls()
    fOut.Close()
    print("Written to %s" % fOut_)
    
    return
    hNames = [x.GetName() for x in fOut.GetListOfKeys()]
    

    if h_para_data.GetName() in hNames: fOut.Delete(h_para_data.GetName()+";*")
    if h_perp_data.GetName() in hNames: fOut.Delete(h_perp_data.GetName()+";*")
    if h_para_mc.GetName() in hNames: fOut.Delete(h_para_mc.GetName()+";*")
    if h_perp_mc.GetName() in hNames: fOut.Delete(h_perp_mc.GetName()+";*")
    h_para_data.Write()
    h_perp_data.Write()
    h_para_mc.Write()
    h_perp_mc.Write()
        
    if h_para_gen.GetName() in hNames: fOut.Delete(h_para_gen.GetName()+";*")
    if h_perp_gen.GetName() in hNames: fOut.Delete(h_perp_gen.GetName()+";*")
    h_para_gen.Write()
    h_perp_gen.Write()
    
    
    
    fOut.ls()
    fOut.Close()
    print("Written to %s" % fOut_)

    

def doPlots_MET_qTbins():

    label = "MET_corr_rec_pt_qt"
    outDir = "/eos/user/j/jaeyserm/www/wmass/lowPU/recoilCorrection/MET_plots_qT/"
    
    functions.prepareDir(outDir, True)
    
    l_data = "SingleMuon"
    l_dy = "DYmumu"
    l_ttbar = "TTbar"
    l_ewk = "EWK"

    b_data = readProc(label, l_data) 
    b_dy = readProc(label, l_dy)
    b_ttbar = readProc(label, l_ttbar)
    b_ewk = readProc(label, l_ewk)

    h_data = narf.hist_to_root(b_data)
    h_dy = narf.hist_to_root(b_dy)
    h_ttbar = narf.hist_to_root(b_ttbar)
    h_ewk = narf.hist_to_root(b_ewk)

    yRatio = 1.3

    cfg = {

        'logy'              : True,
        'logx'              : False,
    
        'xmin'              : 0,
        'xmax'              : 100,
        'ymin'              : 1e-2,
        'ymax'              : 1e7,
        
        'xtitle'            : "MET (GeV)",
        'ytitle'            : "Events / 1 GeV",
        
        'topRight'          : "199 pb^{#minus1} (13 TeV)", 
        'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",
        
        'ratiofraction'     : 0.25,
        'ytitleR'           : "Ratio",
        'yminR'             : 1-(yRatio-1), # 0.7
        'ymaxR'             : yRatio, # 1.3
    }   
    
    label_mc = "DY #rightarrow #mu^{+}#mu^{#minus} #plus e^{+}e^{#minus}"
    

    
    for iBin in range(1, h_data.GetNbinsX()+1):
    
        qTbinMinGeV, qTbinMaxGeV = h_data.GetXaxis().GetBinLowEdge(iBin), h_data.GetXaxis().GetBinLowEdge(iBin+1)
        qT = h_data.GetXaxis().GetBinCenter(iBin)
        
        qTlabel = "%.1f < q_{T} < %.1f GeV" % (qTbinMinGeV, qTbinMaxGeV)
        outName = "%s_%sGeV" % (str("%.1f" % qTbinMinGeV).replace(".", "p"), str("%.1f" % qTbinMaxGeV).replace(".", "p"))
        
        print("***************** Do iBin %d [%f,%f]" % (iBin, qTbinMinGeV, qTbinMaxGeV))
        
        hist_data = h_data.ProjectionY("data_bin%d" % iBin, iBin, iBin)
        hist_dy = h_dy.ProjectionY("dy_bin%d" % iBin, iBin, iBin)
        hist_ttbar = h_ttbar.ProjectionY("ttbar_bin%d" % iBin, iBin, iBin)
        hist_ewk = h_ewk.ProjectionY("ewk_bin%d" % iBin, iBin, iBin)
        hist_data.Rebin(2)
        hist_dy.Rebin(2)
        hist_ttbar.Rebin(2)
        hist_ewk.Rebin(2)
        
        hist_procs = [hist_ewk, hist_ttbar, hist_dy]
        
        print(hist_ttbar.Integral())
        
        sf = hist_data.Integral() / (hist_dy.Integral() + hist_ttbar.Integral() + hist_ewk.Integral())

        leg = ROOT.TLegend(.20, 0.88-(3+2)*0.05, .5, .88)
        leg.SetBorderSize(0)
        leg.SetFillStyle(0)
        leg.SetTextSize(0.040)
        
        leg.AddEntry(hist_data, groups_mumu.groups[l_data]['label'], "PE")

        st = ROOT.THStack()
        st.SetName("stack")
        h_bkg = hist_data.Clone("bkg_nominal")
        h_bkg.Reset("ACE")
        for i,proc in enumerate([l_ewk, l_ttbar, l_dy]):
        
            hist = hist_procs[i]
            hist.Scale(sf)
            hist.SetFillColor(groups_mumu.groups[proc]['color'])
            hist.SetLineColor(ROOT.kBlack)
            hist.SetLineWidth(1)
            hist.SetLineStyle(1)
            
            leg.AddEntry(hist, groups_mumu.groups[proc]['label'], "F")
            st.Add(hist)
            h_bkg.Add(hist)
           
        h_bkg.SetLineColor(ROOT.kBlack)
        h_bkg.SetFillColor(0)
        h_bkg.SetLineWidth(2)
        
        int_bkg = h_bkg.Integral()

        hist_data.SetLineColor(ROOT.kBlack)
        hist_data.SetMarkerStyle(20)
        hist_data.SetMarkerColor(ROOT.kBlack)
        hist_data.SetLineColor(ROOT.kBlack)

        h_err = h_bkg.Clone("syst")
        h_err.SetFillColor(ROOT.kBlack)
        h_err.SetMarkerSize(0)
        h_err.SetLineWidth(0)
        h_err.SetFillStyle(3004)    
        leg.AddEntry(h_err, "Stat. + Syst. Unc.", "F")
        
        h_bkg_ratio = h_bkg.Clone("h_bkg_ratio") # nominal point, need to remove stat. error
        h_err_ratio = h_err.Clone("syst_ratio")
        h_err_ratio.Divide(h_bkg_ratio)
        
        
        # Data/MC ratio (the error bars represent the data uncertainty)
        h_ratio = hist_data.Clone("h_ratio")
        for i in range(h_bkg .GetNbinsX()+1): h_bkg.SetBinError(i, 0) # set MC errors to zero
        h_ratio.Divide(h_bkg)
        h_ratio.SetMarkerStyle(20)
        h_ratio.SetMarkerSize(0.7)
        h_ratio.SetMarkerColor(ROOT.kBlack)
        h_ratio.SetLineColor(ROOT.kBlack)
        h_ratio.SetLineWidth(1)

        

        plotter.cfg = cfg
        canvas, padT, padB = plotter.canvasRatio()
        dummyT, dummyB, dummyL = plotter.dummyRatio()
            
        ## top panel
        canvas.cd()
        padT.Draw()
        padT.cd()
        padT.SetGrid()
        dummyT.Draw("HIST")
            
        st.Draw("HIST SAME")
        
        h_err.Draw("E2 SAME")
        h_bkg.Draw("HIST SAME")
        hist_data.Draw("PE SAME")
        leg.Draw("SAME")
        
        plotter.auxRatio()  
        ROOT.gPad.SetTickx()
        ROOT.gPad.SetTicky()
        ROOT.gPad.RedrawAxis()  
        

        ## bottom panel
        canvas.cd()
        padB.Draw()
        padB.SetFillStyle(0)
        padB.cd()
        dummyB.Draw("HIST")
        dummyL.Draw("SAME")
        
        h_ratio.Draw("P SAME") # E2 SAME
        h_err_ratio.Draw("E2 SAME")

        ROOT.gPad.SetTickx()
        ROOT.gPad.SetTicky()
        ROOT.gPad.RedrawAxis()

        canvas.SaveAs("%s/%03d_qT_%s.png" % (outDir, iBin, outName))
        canvas.SaveAs("%s/%03d_qT_%s.pdf" % (outDir, iBin, outName))
        canvas.Close()        



import decimal
def drange(x, y, jump):
    while x < y:
        yield float(x)
        x += decimal.Decimal(jump)
        

def calcReweight(): 
 
    recoil_qTbins = list(range(0, 30, 1)) + list(range(30, 50, 2)) + list(range(50, 70, 5)) + list(range(70, 150, 10)) + [150, 200, 300]
    recoil_qTbins = list(drange(0, 30, 0.5)) + list(range(30, 50, 1)) + list(range(50, 70, 2)) + list(range(70, 100, 5)) + list(range(100, 150, 10)) + [150, 200, 300, 10000]
    recoil_qTbins = list(range(0, 50, 1)) + list(range(50, 70, 2)) + list(range(70, 100, 5)) + list(range(100, 150, 10)) + [150, 200, 300, 10000]
    recoil_qTbins = list(range(0, 20, 1)) + list(range(20, 40, 2)) + list(range(40, 55, 3)) + list(range(55, 80, 5)) + list(range(80, 100, 10)) + [100, 125, 150, 10000]

    
    b_data = readProc("qT", "SingleMuon")
    b_mc = readProc("qT", "DYmumu")
    b_bkg = None
    for bkg in bkg_procs:
    
        b_para = readProc("qT", bkg)
        if b_bkg == None: b_bkg = b_para
        else: b_bkg += b_para
    
    h_data = narf.hist_to_root(b_data)
    h_mc = narf.hist_to_root(b_mc)
    h_bkg = narf.hist_to_root(b_bkg)
    
    h_data = functions.Rebin(h_data, recoil_qTbins, binWidth=False)
    h_mc = functions.Rebin(h_mc, recoil_qTbins, binWidth=False)
    h_bkg = functions.Rebin(h_bkg, recoil_qTbins, binWidth=False)
    
    h_data.Add(h_bkg, -1)
    h_data.Scale(1./h_data.Integral())
    h_mc.Scale(1./h_mc.Integral())
    
    
    for i in range(1, len(recoil_qTbins)):
    
    
        a = h_data.GetBinContent(i) / h_mc.GetBinContent(i) if h_mc.GetBinContent(i) > 0 and h_data.GetBinContent(i) else 1
        print("%.3f, " % a, end = '')
        #print(i, a, h_data.GetBinCenter(i) )

    sys.exit()
    
    
    
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

if __name__ == "__main__":

    met = "RawPFMET" # RawPFMET DeepMETReso
    flavor = "mumu" # mu, e, mumu, ee
    
    groups_mumu = Datagroups("lowPU_%s_%s.pkl.lz4" % (flavor, met))
    bkg_procs = ['EWK', 'TTbar']
    '''
    lowPU = True
    if lowPU:
        groups_mumu = Datagroups("mz_lowPU_mumu.pkl.lz4")
        #groups_ee = Datagroups("mz_lowPU_ee.pkl.lz4")
        bkg_procs = ['EWK', 'TTbar']
    else:
        groups_mumu = Datagroups("mz_wlike_with_mu_eta_pt.pkl.lz4", wlike=True)
        bkg_procs = ['Other']
    '''
    label = "DY #rightarrow #mu^{+}#mu^{#minus} #plus e^{+}e^{#minus}"
    
    
    #doPlots_MET_qTbins()
    #doRecoilFits_Z()
    calcReweight()
    #autoFit()
    #doRecoilFits_fine(cat, met, label, qTbins)
    