import ROOT
import random

cfg = None

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptTitle(0)
#ROOT.gStyle.SetImageScaling(2.) 


def translate(word):
    if "METPT" in word: return word.replace("METPT", "p_{T}^{miss} (GeV)")
    if "METWLIKEPT" in word: return word.replace("METWLIKEPT", "p_{T}^{miss} (W-like) (GeV)")
    if "METPHI" in word: return word.replace("METPHI", "#phi^{miss} (GeV)")
    if "METX" in word: return word.replace("METX", "p_{T,x}^{miss} (GeV)")
    if "METY" in word: return word.replace("METY", "p_{T,y}^{miss} (GeV)")
    if "UPARAQT" in word: return word.replace("UPARAQT", "U_{#parallel} #plus q_{T} (GeV)")
    if "UPARA" in word: return word.replace("UPARA", "U_{#parallel} (GeV)")
    if "UPERP" in word: return word.replace("UPERP", "U_{#perp}   (GeV)")
    if "UMAGN" in word: return word.replace("UMAGN", "|U| (GeV)")
    if "MT" in word: return word.replace("MT", "m_{T} (GeV)")
    if "PT" in word: return word.replace("PT", "p_{T} (GeV)")
    if "QT" in word: return word.replace("QT", "q_{T} (GeV)")
    if "HT" in word: return word.replace("HT", "H_{T} (GeV)")
    if "ETA" in word: return word.replace("ETA", "#eta")
    return word

def dummy(nbins = 1):

    if cfg['logx']: 
        xmin = 0.999*float(cfg['xmin']) # hack to display lower/upper ticks on axis
        xmax = 1.001*float(cfg['xmax'])
    else:
        xmin = float(cfg['xmin'])
        xmax = float(cfg['xmax'])
        
    if cfg['logy']: 
        ymin = 0.999*float(cfg['ymin']) # hack to display lower/upper ticks on axis
        ymax = 1.001*float(cfg['ymax'])
    else:
        ymin = float(cfg['ymin'])
        ymax = float(cfg['ymax'])

    # dummy
    dummy = ROOT.TH1D("h", "h", nbins, xmin, xmax)
    for i in range(0, nbins+1): dummy.SetBinContent(i, -1e9)

    # x-axis
    dummy.GetXaxis().SetTitle(translate(cfg['xtitle']))

    dummy.GetXaxis().SetRangeUser(xmin, xmax)

    dummy.GetXaxis().SetTitleFont(43)
    dummy.GetXaxis().SetTitleSize(40)
    dummy.GetXaxis().SetLabelFont(43)
    dummy.GetXaxis().SetLabelSize(35)

    dummy.GetXaxis().SetTitleOffset(1.2*dummy.GetXaxis().GetTitleOffset())
    dummy.GetXaxis().SetLabelOffset(1.2*dummy.GetXaxis().GetLabelOffset())

    # y-axis
    dummy.GetYaxis().SetTitle(translate(cfg['ytitle']))

    dummy.GetYaxis().SetRangeUser(ymin, ymax)
    dummy.SetMaximum(ymax)
    dummy.SetMinimum(ymin)

    dummy.GetYaxis().SetTitleFont(43)
    dummy.GetYaxis().SetTitleSize(40)
    dummy.GetYaxis().SetLabelFont(43)
    dummy.GetYaxis().SetLabelSize(35)

    dummy.GetYaxis().SetTitleOffset(1.7*dummy.GetYaxis().GetTitleOffset())
    dummy.GetYaxis().SetLabelOffset(1.4*dummy.GetYaxis().GetLabelOffset())

    return dummy


def aux(c=None):

    if c==None:
    
        tr = 0.95
    
    else:
    
        tr = 1.-c.GetRightMargin()

    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.045)
    latex.SetTextColor(1)
    latex.SetTextFont(42)
    latex.SetTextAlign(30) # 0 special vertical aligment with subscripts
    if "_" in cfg['topRight']: latex.DrawLatex(tr, 0.945, cfg['topRight'])
    elif "^" in cfg['topRight']: latex.DrawLatex(tr, 0.945, cfg['topRight'])
    else: latex.DrawLatex(tr, 0.945, cfg['topRight']) # was 955 ??  945
    

    latex.SetTextAlign(13)
    latex.SetTextFont(42)
    latex.SetTextSize(0.045)
    latex.DrawLatex(0.15, 0.985, cfg['topLeft'])
    

def canvas():

    c = ROOT.TCanvas("%d" % random.randint(1e4, 1e5), "c", 1000, 1000)
    c.SetTopMargin(0.055)
    c.SetRightMargin(0.05)
    c.SetLeftMargin(0.15)
    c.SetBottomMargin(0.11)
    
    #c.SetFrameLineWidth(2)
    
    if cfg['logy']: c.SetLogy()
    if cfg['logx']: c.SetLogx()


    c.Modify()
    c.Update()

    return c
    
 


def auxRatio():

    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.055)
    latex.SetTextColor(1)
    latex.SetTextFont(42)
    latex.SetTextAlign(31)
    latex.DrawLatex(0.95, 0.935, cfg['topRight'])

    latex.SetTextAlign(13)
    latex.SetTextFont(42)
    latex.SetTextSize(0.06)
    latex.DrawLatex(0.15, 0.975, cfg['topLeft'])

   


def dummyRatio(nbins = 1, line=1):

    if cfg['logx']: 
        xmin = 0.999*float(cfg['xmin']) # hack to display lower/upper ticks on axis
        xmax = 1.001*float(cfg['xmax'])
    else:
        xmin = float(cfg['xmin'])
        xmax = float(cfg['xmax'])
        
    if cfg['logy']: 
        ymin = 0.999*float(cfg['ymin']) # hack to display lower/upper ticks on axis
        ymax = 1.001*float(cfg['ymax'])
    else:
        ymin = float(cfg['ymin'])
        ymax = float(cfg['ymax'])


    # dummy
    dummyT = ROOT.TH1D("h1_%d" % random.randint(1e4, 1e5), "", nbins, xmin, xmax)
    dummyB = ROOT.TH1D("h2_%d" % random.randint(1e4, 1e5), "", nbins, xmin, xmax)
    
    for i in range(0, nbins+1): dummyT.SetBinContent(i, -1e9)
    for i in range(0, nbins+1): dummyB.SetBinContent(i, -1e9)

    # x-axis
    dummyB.GetXaxis().SetTitle(translate(cfg['xtitle']))

    dummyT.GetXaxis().SetRangeUser(xmin, xmax)
    dummyB.GetXaxis().SetRangeUser(xmin, xmax)

    dummyT.GetXaxis().SetTitleFont(43)
    dummyT.GetXaxis().SetTitleSize(0)
    dummyT.GetXaxis().SetLabelFont(43)
    dummyT.GetXaxis().SetLabelSize(0)

    dummyT.GetXaxis().SetTitleOffset(1.2*dummyT.GetXaxis().GetTitleOffset())
    dummyT.GetXaxis().SetLabelOffset(1.2*dummyT.GetXaxis().GetLabelOffset())

    dummyB.GetXaxis().SetTitleFont(43)
    dummyB.GetXaxis().SetTitleSize(35)
    dummyB.GetXaxis().SetLabelFont(43)
    dummyB.GetXaxis().SetLabelSize(30)

    dummyB.GetXaxis().SetTitleOffset(1.0*dummyB.GetXaxis().GetTitleOffset()) ### was 4.0???
    dummyB.GetXaxis().SetLabelOffset(3.0*dummyB.GetXaxis().GetLabelOffset())

    # y-axis
    dummyT.GetYaxis().SetTitle(translate(cfg['ytitle']))
    dummyB.GetYaxis().SetTitle(translate(cfg['ytitleR']))

    dummyT.GetYaxis().SetRangeUser(ymin, ymax)
    dummyT.SetMaximum(ymax)
    dummyT.SetMinimum(ymin)

    dummyB.GetYaxis().SetRangeUser(cfg['yminR'], cfg['yminR'])
    dummyB.SetMaximum(cfg['ymaxR'])
    dummyB.SetMinimum(cfg['yminR'])

    dummyT.GetYaxis().SetTitleFont(43)
    dummyT.GetYaxis().SetTitleSize(35)
    dummyT.GetYaxis().SetLabelFont(43)
    dummyT.GetYaxis().SetLabelSize(30)

    dummyT.GetYaxis().SetTitleOffset(1.3) # 1.7*dummyT.GetYaxis().GetTitleOffset()
    dummyT.GetYaxis().SetLabelOffset(1.4*dummyT.GetYaxis().GetLabelOffset())
    
    dummyB.GetYaxis().SetTitleFont(43)
    dummyB.GetYaxis().SetTitleSize(35)
    dummyB.GetYaxis().SetLabelFont(43)
    dummyB.GetYaxis().SetLabelSize(30)

    dummyB.GetYaxis().SetTitleOffset(1.3) # 1.7*dummyB.GetYaxis().GetTitleOffset()
    dummyB.GetYaxis().SetLabelOffset(1.4*dummyB.GetYaxis().GetLabelOffset())
    dummyB.GetYaxis().SetNdivisions(505)
    
    line = ROOT.TLine(float(cfg['xmin']), line, float(cfg['xmax']), line)
    line.SetLineColor(ROOT.kRed)
    line.SetLineWidth(2)

    return dummyT, dummyB, line

def canvasRatio():

    
    epsilon = 0.025

    c = ROOT.TCanvas("canvas%d" % random.randint(1e4, 1e5), "", 1000, 1000)
    c.SetTopMargin(0.0)
    c.SetRightMargin(0.0)
    c.SetBottomMargin(0.0)
    c.SetLeftMargin(0.0)

    pad1 = ROOT.TPad("p1%d" % random.randint(1e4, 1e5), "", 0, cfg['ratiofraction'], 1, 1)
    pad2 = ROOT.TPad("p2%d" % random.randint(1e4, 1e5), "", 0, 0.0, 1, cfg['ratiofraction']-0.7*epsilon)

    pad1.SetBottomMargin(epsilon)
    pad1.SetTopMargin(0.055/(1.-cfg['ratiofraction']))
    pad1.SetRightMargin(0.05)
    pad1.SetLeftMargin(0.15)
    #pad1.SetFrameLineWidth(2)
                
    pad2.SetBottomMargin(0.37)
    pad2.SetTopMargin(0.0)
    pad2.SetRightMargin(0.05)
    pad2.SetLeftMargin(0.15)
    #pad2.SetFrameLineWidth(2)
                
                
    if cfg['logy']: pad1.SetLogy()
    if cfg['logx']: 
        pad2.SetLogx()                
        pad1.SetLogx()     

    c.Modify()
    c.Update()

    return c, pad1, pad2
    