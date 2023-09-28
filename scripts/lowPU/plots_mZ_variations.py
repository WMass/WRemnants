
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
import hist

from wremnants.datasets.datagroups import Datagroups


def Rebin(h, newbins):

    ''' Rebin with un-even bins '''
    mybins = array.array('d', newbins)
    h1 = h.Rebin(len(mybins)-1, h.GetName(), mybins)
    return h1


def doOverlow(h):

    n = h.GetNbinsX()
    h.SetBinContent(1, h.GetBinContent(0) + h.GetBinContent(1))
    h.SetBinContent(n, h.GetBinContent(n+1) + h.GetBinContent(n))
    h.SetBinError(1, math.hypot(h.GetBinError(0), h.GetBinError(1)))
    h.SetBinError(n, math.hypot(h.GetBinError(n+1), h.GetBinError(n)))
    h.SetBinContent(0, 0)
    h.SetBinContent(n+1, 0)
    h.SetBinContent(0, 0)
    h.SetBinContent(n+1, 0)
    
    return h
    
def getSyst(In, systName, yBins):

    pass
    
def parseHist(h, norm, rebin, projectionx=[]):

    if h.InheritsFrom("TH2") and projectionx != []: h = h.ProjectionX("", projectionx[0], projectionx[1])

    if isinstance(rebin, int): h.Rebin(rebin)
    else: h = Rebin(h, rebin)
    
    h.Scale(norm)
    h = doOverlow(h)
    
    return h
    
    
def computeEGammaScaleSmearingUnc(hName, proc, fIn, fNameIn):

    systName = "egamma_scale_smear"
    
    histName_MC = "%s_%s" % (hName, proc)
    hNom_MC = copy.deepcopy(fIn.Get(histName_MC))
 
    histName_DATA = "%s_%s" % (hName, "singleMuon")
    hNom_DATA = copy.deepcopy(fIn.Get(histName_DATA))
 
    hSigmaUp = copy.deepcopy(fIn.Get("%s_syst_%sUp" % (histName_MC, systName)))
    hSigmaDown = copy.deepcopy(fIn.Get("%s_syst_%sDown" % (histName_MC, systName)))
    
    hScaleUp_DATA = copy.deepcopy(fIn.Get("%s_syst_%sUp" % (histName_DATA, systName)))
    hScaleDown_DATA = copy.deepcopy(fIn.Get("%s_syst_%sDown" % (histName_DATA, systName)))
    
    
    # project DATA differences on MC
    hScaleUp = hNom_MC.Clone("hScaleUp")
    hScaleDown = hNom_MC.Clone("hScaleDown")
    for i in range(0, hScaleUp.GetNbinsX()+2):
    
        dN = hNom_DATA.GetBinContent(i) - hScaleUp_DATA.GetBinContent(i)
        hScaleUp.SetBinContent(i, hScaleUp.GetBinContent(i) + dN)
        
        dN = hNom_DATA.GetBinContent(i) - hScaleDown_DATA.GetBinContent(i)
        hScaleDown.SetBinContent(i, hScaleDown.GetBinContent(i) + dN)
    
    
    
    fIn.cd()
    
    
    
    ## do some plotting of the ratio
    
    if hName == "m_mumu":
    
        cfg = {

            'logy'              : False,
            'logx'              : False,
            
            'xmin'              : 60,
            'xmax'              : 120,
            'ymin'              : 1e1,
            'ymax'              : 60000, # 3e6
                
            'xtitle'            : "m(#mu^{#plus},#mu^{#minus}) (GeV)",
            'ytitle'            : "Events",
                
            'topRight'          : "199 pb^{#minus1} (13 TeV)", 
            'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",

            'ratiofraction'     : 0.3,
            'ytitleR'           : "Ratio",
            
            'yminR'             : 0.95,
            'ymaxR'             : 1.05,
        }
        
        
        leg = ROOT.TLegend(.20, 0.85-(len(bkgs)+2)*0.055, .5, .85)
        leg.SetBorderSize(0)
        leg.SetFillStyle(0)
        leg.SetTextSize(0.040)
        
        leg.AddEntry(hNom_MC, "Nominal", "L")
        leg.AddEntry(hScaleUp, "Scale up", "L")
        leg.AddEntry(hScaleDown, "Scale down", "L")

        plotter.cfg = cfg
        canvas, padT, padB = plotter.canvasRatio()
        dummyT, dummyB, dummyL = plotter.dummyRatio()
            
        ## top panel
        canvas.cd()
        padT.Draw()
        padT.cd()
        padT.SetGrid()
        dummyT.Draw("HIST")
           
        hNom_MC.SetLineColor(ROOT.kBlack)
        hNom_MC.SetLineWidth(2)
        hNom_MC.Draw("HIST SAME")
        
        hScaleUp.SetLineColor(ROOT.kRed)
        hScaleUp.SetLineWidth(2)
        hScaleUp.Draw("HIST SAME")
        
        hScaleDown.SetLineColor(ROOT.kBlue)
        hScaleDown.SetLineWidth(2)
        hScaleDown.Draw("HIST SAME")
        
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
        
        hScaleUp_ratio = hScaleUp.Clone("tmp1")
        hScaleDown_ratio = hScaleDown.Clone("tmp1")
        
        hScaleUp_ratio.Divide(hNom_MC)
        hScaleDown_ratio.Divide(hNom_MC)
         
        hScaleUp_ratio.SetLineColor(ROOT.kRed)
        hScaleUp_ratio.SetLineWidth(2)
        
        hScaleDown_ratio.SetLineColor(ROOT.kBlue)
        hScaleDown_ratio.SetLineWidth(2)

        hScaleUp_ratio.Draw("HIST SAME")
        hScaleDown_ratio.Draw("HIST SAME")

        ROOT.gPad.SetTickx()
        ROOT.gPad.SetTicky()
        ROOT.gPad.RedrawAxis()

        canvas.SaveAs("/eos/user/j/jaeyserm/www/wmass/lowPU/Zee_systs/scale.png")
        canvas.SaveAs("/eos/user/j/jaeyserm/www/wmass/lowPU/Zee_systs/scale.pdf")
        canvas.Close()    
        
    
    
    
        ##############################
        leg = ROOT.TLegend(.20, 0.85-(len(bkgs)+2)*0.055, .5, .85)
        leg.SetBorderSize(0)
        leg.SetFillStyle(0)
        leg.SetTextSize(0.040)
        
        leg.AddEntry(hNom_MC, "Nominal", "L")
        leg.AddEntry(hSigmaUp, "Smearing up", "L")
        leg.AddEntry(hSigmaDown, "Smearing down", "L")

        plotter.cfg = cfg
        canvas, padT, padB = plotter.canvasRatio()
        dummyT, dummyB, dummyL = plotter.dummyRatio()
            
        ## top panel
        canvas.cd()
        padT.Draw()
        padT.cd()
        padT.SetGrid()
        dummyT.Draw("HIST")
           
        hNom_MC.SetLineColor(ROOT.kBlack)
        hNom_MC.SetLineWidth(2)
        hNom_MC.Draw("HIST SAME")
        
        hSigmaUp.SetLineColor(ROOT.kRed)
        hSigmaUp.SetLineWidth(2)
        hSigmaUp.Draw("HIST SAME")
        
        hSigmaDown.SetLineColor(ROOT.kBlue)
        hSigmaDown.SetLineWidth(2)
        hSigmaDown.Draw("HIST SAME")
        
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
        
        hSigmaUp_ratio = hSigmaUp.Clone("tmp1")
        hSigmaDown_ratio = hSigmaDown.Clone("tmp1")
        
        hSigmaUp_ratio.Divide(hNom_MC)
        hSigmaDown_ratio.Divide(hNom_MC)
         
        hSigmaUp_ratio.SetLineColor(ROOT.kRed)
        hSigmaUp_ratio.SetLineWidth(2)
        
        hSigmaDown_ratio.SetLineColor(ROOT.kBlue)
        hSigmaDown_ratio.SetLineWidth(2)

        hSigmaUp_ratio.Draw("HIST SAME")
        hSigmaDown_ratio.Draw("HIST SAME")

        ROOT.gPad.SetTickx()
        ROOT.gPad.SetTicky()
        ROOT.gPad.RedrawAxis()

        canvas.SaveAs("/eos/user/j/jaeyserm/www/wmass/lowPU/Zee_systs/smearing.png")
        canvas.SaveAs("/eos/user/j/jaeyserm/www/wmass/lowPU/Zee_systs/smearing.pdf")
        canvas.Close()   
  
    
    
    
    return hScaleUp, hScaleDown, hSigmaUp, hSigmaDown


def addUnc(hTarget, hNom, hUp, hDw=None):

    for k in range(1, hTarget.GetNbinsX()+1):

        if hDw != None: sigma = 0.5*abs(abs(hUp.GetBinContent(k)-hNom.GetBinContent(k)) + abs(hDw.GetBinContent(k)-hNom.GetBinContent(k)))
        else: sigma = abs(hUp.GetBinContent(k)-hNom.GetBinContent(k))
        hTarget.SetBinError(k, math.sqrt(sigma*sigma + hTarget.GetBinError(k)*hTarget.GetBinError(k)))
                            


def recoilStatUncertainty(systName, sysHists, rebin):

    systHistName = systs[systName]['histName']
                    
    # get the nominal (only for DY)
    thn_base = fIn.Get("recoil_corr_magn_DYmumu_MiNNLO")
    h_base = thn_base.Projection(0, "E")
                    
    thn_nom = fIn.Get("RecoilSystStat_nom_DYmumu_MiNNLO") # 
    thn_pert = fIn.Get("%s_DYmumu_MiNNLO" % systHistName)
    
    groupIdx = groupSysts.index("total")        

    for iRecoilVar in range(1, 74):
                    
        thn_nom.GetAxis(1).SetRange(iRecoilVar, iRecoilVar)
        h_nom = thn_nom.Projection(0, "E")
        h_nom.SetTitle("h_nom%d" % iRecoilVar)
        h_nom.SetBinContent(0, 0) # remove underflow

        thn_pert.GetAxis(1).SetRange(iRecoilVar, iRecoilVar)
        h_pert = thn_pert.Projection(0, "E")
        h_pert.SetTitle("h_pert%d" % iRecoilVar)
        h_pert.SetBinContent(0, 0) # remove underflow
                        
        print(h_base.GetNbinsX(), h_nom.GetNbinsX(), h_pert.GetNbinsX())
                        
        hUp = copy.deepcopy(h_base)
        hUp.SetName("RecoilSystStat_para_data_m1_qTbin%dUp" % iRecoilVar)
        hUp.Add(h_nom, -1)
        hUp.Add(h_pert)
        
        hDw = copy.deepcopy(hUp)
        hDw.SetName("RecoilSystStat_para_data_m1_qTbin%dDown" % iRecoilVar)
                        
        for k in range(1, hUp.GetNbinsX()+1):
                        
            hDw.SetBinContent(k, 2*h_base.GetBinContent(k) - hUp.GetBinContent(k))
            #print(hUp.GetBinContent(k), h_base.GetBinContent(k), hUp.GetBinContent(k)/h_base.GetBinContent(k))
                        
                        
                        
        print(h_base.Integral(), h_nom.Integral(), h_pert.Integral(), h_nom.Integral()/h_pert.Integral(), hUp.Integral())
                    
        hUp = parseHist(hUp, lumi, rebin)
        hDw = parseHist(hDw, lumi, rebin)

        addUnc(sysHists[groupIdx], h_base, hUp, hDw)  
        

def parseProc(histCfg, procName, syst="", rebin=1):

    axis = histCfg['axis']
    hNames = histCfg['name'].split(",")

    
    bhist = None
    for hName in hNames:
        
        bhist = groups.readProc(hName, procName, axis=axis)
        if bhist == None: continue
        label = "%s_%s" % (hName, procName)
        break
        
    print(bhist)
    rhist = narf.hist_to_root(bhist)
    rhist.Rebin(rebin)
    rhist.SetName(label)
    rhist = doOverlow(rhist)
    
    if procName == "SingleMuon" or procName == "SingleElectron": pass
    else: rhist.Scale(MC_SF)

    print("Get histogram %s, yield=%d" % (label, rhist.Integral()))
    return rhist
                        
                        
def parseHists(histCfg, leg, rebin=1, projectionx=[], noData=False):

    h_data = parseProc(histCfg, data, rebin=rebin)
    leg.AddEntry(h_data, groups.groups[data]['label'], "PE")

    st = ROOT.THStack()
    st.SetName("stack")
    h_bkg = h_data.Clone("bkg_nominal")
    h_bkg.Reset("ACE")
    for i,proc in enumerate(procs):
    
        hist = parseProc(histCfg, proc, rebin=rebin)
        hist.SetFillColor(groups.groups[proc]['color'])
        hist.SetLineColor(ROOT.kBlack)
        hist.SetLineWidth(1)
        hist.SetLineStyle(1)
        
        leg.AddEntry(hist, groups.groups[proc]['label'], "F")
        st.Add(hist)
        h_bkg.Add(hist)
       
    h_bkg.SetLineColor(ROOT.kBlack)
    h_bkg.SetFillColor(0)
    h_bkg.SetLineWidth(2)

    h_data.SetLineColor(ROOT.kBlack)
    h_data.SetMarkerStyle(20)
    h_data.SetMarkerColor(ROOT.kBlack)
    h_data.SetLineColor(ROOT.kBlack)

    h_err = h_bkg.Clone("syst")
    h_err.SetFillColor(ROOT.kBlack)
    h_err.SetMarkerSize(0)
    h_err.SetLineWidth(0)
    h_err.SetFillStyle(3004)    
    leg.AddEntry(h_err, "Stat. + Syst. Unc.", "F")
    
    
    # do systematics
    for i,bkg in enumerate(procs):

        break
        hist = None
        for proc in plotcfg[bkg]['samples']:
        
            # get nominal
            thn = None
            for hName in hNames:
                
                histName = "%s_%s" % (hName, proc)
                thn = fIn.Get(histName)
                if thn == None: continue
                else: break
        
       
            hNom = thn.Projection(0, "E")
            hNom.SetTitle("hNom_%s" % histName)
            hNom = parseHist(hNom, lumi, rebin)

            for systName in systs:
            
                if "RecoilSystStat" in systName or "METSystStat" in systName:

                    
                    if "METSystStat" in systName and hName == "recoil_corr_magn": continue
                    if "RecoilSystStat" in systName and hName == "MET_corr_rec_pt": continue
                    if proc != "DYmumu_MiNNLO": continue
        
                    #continue
                    systHistName = systs[systName]['histName']
                    
                    # get the nominal (only for DY)
                    if "RecoilSystStat" in systName:
                    
                        thn_nom = fIn.Get("RecoilSystStat_nom_DYmumu_MiNNLO") # binned in qT
                        thn_base = fIn.Get("recoil_corr_magn_DYmumu_MiNNLO")
                        
                    else:
                    
                        thn_nom = fIn.Get("METSystStat_nom_DYmumu_MiNNLO") # binned in qT
                        thn_base = fIn.Get("MET_corr_rec_pt_DYmumu_MiNNLO")
                     
                    h_base = thn_base.Projection(0, "E")
                    h_base = parseHist(h_base, lumi, rebin)
                     
                    # get the perturbation
                    thn_pert = fIn.Get("%s_DYmumu_MiNNLO" % systHistName)
                    
                    

                    for iRecoilVar in range(1, 40):
                        #if iRecoilVar == 40: continue
                        thn_nom.GetAxis(1).SetRange(iRecoilVar, iRecoilVar)
                        h_nom = thn_nom.Projection(0, "E")
                        h_nom.SetTitle("h_nom%d" % iRecoilVar)
                        h_nom.SetBinContent(0, 0) # remove underflow
                        h_nom.SetBinContent(h_nom.GetNbinsX()+1, 0) # remove overflow
                        h_nom = parseHist(h_nom, lumi, rebin)
                        
                        if h_nom.Integral() <= 0: continue

                        thn_pert.GetAxis(1).SetRange(iRecoilVar, iRecoilVar)
                        h_pert = thn_pert.Projection(0, "E")
                        h_pert.SetTitle("h_pert%d" % iRecoilVar)
                        h_pert.SetBinContent(0, 0) # remove underflow
                        h_pert.SetBinContent(h_pert.GetNbinsX(), 0) # remove overflow
                        h_pert.SetBinContent(h_pert.GetNbinsX()+1, 0) # remove overflow
                        h_pert = parseHist(h_pert, lumi, rebin)
                        
                        #print(h_base.GetNbinsX(), h_nom.GetNbinsX(), h_pert.GetNbinsX())
                        
                        hUp = copy.deepcopy(h_base)
                        hUp.SetName("RecoilSystStat_para_data_m1_qTbin%dUp" % iRecoilVar)
                        hUp.Add(h_nom, -1)
                        hUp.Add(h_pert)
                        
                        hDw = copy.deepcopy(hUp)
                        hDw.SetName("RecoilSystStat_para_data_m1_qTbin%dDown" % iRecoilVar)
                        
                        for k in range(1, hUp.GetNbinsX()+1):
                        
                            hDw.SetBinContent(k, 2*h_base.GetBinContent(k) - hUp.GetBinContent(k))
                            #print(hUp.GetBinCenter(k), hUp.GetBinContent(k), h_base.GetBinContent(k), hUp.GetBinContent(k)/h_base.GetBinContent(k))
                        
                        
                        #print(h_base.Integral(), h_nom.Integral(), h_pert.Integral(), hUp.Integral(), hDw.Integral())                    
                        #hUp = parseHist(hUp, lumi, rebin)
                        #hDw = parseHist(hDw, lumi, rebin)
                        
                        print(hNom.Integral(), h_base.Integral(), h_nom.Integral(), h_pert.Integral(), hUp.Integral(), hDw.Integral())
                        print(hUp.Integral()/hNom.Integral()) #### SHOULD BE 1 (as recoil does not alter the normalization)
                        
                        for kk in range(0, hUp.GetNbinsX()+1):
                            pass
                            if hNom.GetBinContent(kk) <= 0: continue
                            #print(kk, hNom.GetBinContent(kk), h_nom.GetBinContent(kk), h_pert.GetBinContent(kk), hUp.GetBinContent(kk), hUp.GetBinContent(kk)/hNom.GetBinContent(kk))
                            
                    
                        groupIdx = groupSysts.index("total")        
                        addUnc(sysHists[groupIdx], h_base, hUp, hDw)  
                    
                        #break
                    #sys.exit()
                    continue    
                
                idx = systs[systName]['idx']
                systHistName = systs[systName]['histName']
                group = systs[systName]['group']
                
                #print(systName)
                
                
                thn = None
                for hName in hNames:

                    thn = fIn.Get(histName + "_" + systHistName)
                    if thn == None: continue
                    else: break

                if len(idx) == 2: # up/down specified
                            
                    idxUp = idx[0]
                    idxDw = idx[1]
                    thn.GetAxis(1).SetRange(idxUp, idxUp)
                    hUp = thn.Projection(0, "E")
                    hUp.SetName("%s_%sUp" % (histName, systName))
                    
                    thn.GetAxis(1).SetRange(idxDw, idxDw)    
                    hDw = thn.Projection(0, "E")
                    hDw.SetName("%s_%shDw" % (histName, systName))
                    
                    hUp = parseHist(hUp, lumi, rebin)
                    hDw = parseHist(hDw, lumi, rebin)
                            
                            
                    #print(systName, bkg, proc, hNom.Integral(), hUp.Integral(), hDw.Integral())
                            
                         
                    #for k in range(1, sysHists[groupIdx].GetNbinsX()+1):

                    #    sigma = 0.5*abs(abs(hUp.GetBinContent(k)-hNom.GetBinContent(k)) + abs(hDw.GetBinContent(k)-hNom.GetBinContent(k)))
                    #    sysHists[groupIdx].SetBinError(k, math.sqrt(sigma*sigma + sysHists[groupIdx].GetBinError(k)*sysHists[groupIdx].GetBinError(k)))
                            
                else: # only 1 specified: symmetrize
                
                    idxUp = idx[0]
                    thn.GetAxis(1).SetRange(idxUp, idxUp)    
                    hUp = thn.Projection(0, "E")
                    hUp.SetName("%s_%sUp" % (histName, systName))
                    hUp = parseHist(hUp, lumi, rebin)        
                    
                    #hUp = systHist.ProjectionX("%s_%sUp" % (histName, systName), idxUp, idxUp)
                    #hUp = parseHist(hUp, lumi, rebin, projectionx=projectionx)
                    hDw = None
                    
                    #for k in range(1, hUp.GetNbinsX()+1):
                    
                    #    print(k, hNom.GetBinContent(k), hUp.GetBinContent(k), 100.*hUp.GetBinContent(k)/hNom.GetBinContent(k))
         
                    #print(systName, bkg, proc, hNom.Integral(), hUp.Integral())
                    #addUnc(sysHists[groupIdx], hNom, hUp)                    
                    #for k in range(1, sysHists[groupIdx].GetNbinsX()+1):

                    #    sigma = abs(hUp.GetBinContent(k)-hNom.GetBinContent(k))
                    #    sysHists[groupIdx].SetBinError(k, math.sqrt(sigma*sigma + sysHists[groupIdx].GetBinError(k)*sysHists[groupIdx].GetBinError(k)))
   
                
                if group in groupSysts:
                    groupIdx = groupSysts.index(group)        
                    addUnc(sysHists[groupIdx], hNom, hUp, hDw)   
                
                if "total" in groupSysts:
                    groupIdx = groupSysts.index("total")        
                    addUnc(sysHists[groupIdx], hNom, hUp, hDw)   
          

                    ## do EGammaScaleSmearing
                    '''
                    hScaleUp, hScaleDown, hSigmaUp, hSigmaDown = computeEGammaScaleSmearingUnc(hName, proc, fIn, fNameIn)
                    
                    hScaleUp = parseHist(hScaleUp, lumi, rebin)
                    hScaleDown = parseHist(hScaleDown, lumi, rebin)
                    hSigmaUp = parseHist(hSigmaUp, lumi, rebin)
                    hSigmaDown = parseHist(hSigmaDown, lumi, rebin)
                    
                    print("EGammaScale", bkg, proc, hNom.Integral(), hScaleUp.Integral(), hScaleDown.Integral())
                    print("EGammaSmear", bkg, proc, hNom.Integral(), hSigmaUp.Integral(), hSigmaDown.Integral())
                    #sys.exit()
                    for k in range(1, h_bkg_err.GetNbinsX()+1):
                        #continue
                        sigma = 0.5*abs(abs(hSigmaUp.GetBinContent(k)-hNom.GetBinContent(k)) + abs(hSigmaDown.GetBinContent(k)-hNom.GetBinContent(k)))
                        h_bkg_err.SetBinError(k, math.sqrt(sigma*sigma + h_bkg_err.GetBinError(k)*h_bkg_err.GetBinError(k)))
                    
                    for k in range(1, h_bkg_err.GetNbinsX()+1):
                        #continue
                        sigma = 0.5*abs(abs(hScaleUp.GetBinContent(k)-hNom.GetBinContent(k)) + abs(hScaleDown.GetBinContent(k)-hNom.GetBinContent(k)))
                        h_bkg_err.SetBinError(k, math.sqrt(sigma*sigma + h_bkg_err.GetBinError(k)*h_bkg_err.GetBinError(k)))
                    '''

            
    # ratios (bands)
    h_bkg_ratio = h_bkg.Clone("h_bkg_ratio") # nominal point, need to remove stat. error
    h_err_ratio = h_err.Clone("syst_ratio")
    h_err_ratio.Divide(h_bkg_ratio)


    # Data/MC ratio (the error bars represent the data uncertainty)
    h_ratio = h_data.Clone("h_ratio")
    for i in range(h_bkg .GetNbinsX()+1): h_bkg.SetBinError(i, 0) # set MC errors to zero
    h_ratio.Divide(h_bkg)
    h_ratio.SetMarkerStyle(20)
    h_ratio.SetMarkerSize(0.7)
    h_ratio.SetMarkerColor(ROOT.kBlack)
    h_ratio.SetLineColor(ROOT.kBlack)
    h_ratio.SetLineWidth(1)
    
    return h_data, st, h_bkg, h_ratio, h_err, h_err_ratio



def singlePlot(histCfg, fOut, xMin, xMax, yMin, yMax, xLabel, yLabel, logY=True, rebin=1, legPos=[], projectionx=[], yRatio = 1.3):
    
    # default cfg
    cfg = {

        'logy'              : True,
        'logx'              : False,
        
        'xmin'              : 60,
        'xmax'              : 120,
        'ymin'              : 1e1,
        'ymax'              : 1e5, # 3e6
            
        'xtitle'            : "m(#mu,#mu) (GeV)",
        'ytitle'            : "Events",
            
        'topRight'          : "199 pb^{#minus1} (13 TeV)", 
        'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",

        'ratiofraction'     : 0.3,
        'ytitleR'           : "Data/MC",
        
        'yminR'             : 1-(yRatio-1), # 0.7
        'ymaxR'             : yRatio, # 1.3
    }
    
    
    leg = ROOT.TLegend(.20, 0.88-(len(procs)+2)*0.05, .5, .88)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.040)
    
    h_data, st, h_bkg, h_ratio, h_err, h_err_ratio = parseHists(histCfg, leg, rebin, projectionx=projectionx)

    cfg['logy'] = logY
    cfg['xmin'] = xMin
    cfg['xmax'] = xMax
    cfg['ymin'] = yMin
    cfg['ymax'] = yMax
    
    cfg['xtitle'] = xLabel
    cfg['ytitle'] = yLabel
    

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
    h_data.Draw("PE SAME")
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

    canvas.SaveAs("%s/%s.png" % (outDir, fOut))
    canvas.SaveAs("%s/%s.pdf" % (outDir, fOut))
    canvas.Close()



def singlePlot_noData(hName, fOut, xMin, xMax, yMin, yMax, xLabel, yLabel, logY=True, rebin=1, legPos=[], projectionx=[]):
    
    # default cfg
    cfg = {

        'logy'              : True,
        'logx'              : False,
        
        'xmin'              : 60,
        'xmax'              : 120,
        'ymin'              : 1e1,
        'ymax'              : 1e5, # 3e6
            
        'xtitle'            : "m(#mu,#mu) (GeV)",
        'ytitle'            : "Events",
            
        'topRight'          : "199 pb^{#minus1} (13 TeV)", 
        'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",

    }
    
    
    leg = ROOT.TLegend(.20, 0.90-(len(bkgs)+1)*0.05, .55, .90)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.040)
    
    h_data, st, h_bkg, sysHists, h_ratio, sysHists_ratio = parseHists(hName, leg, rebin, projectionx=projectionx, noData=True)
    #h_data, st, h_bkg, h_bkg_err, h_ratio, h_bkg_err_ratio = parseHists(hName, leg, rebin)

    cfg['logy'] = logY
    cfg['xmin'] = xMin
    cfg['xmax'] = xMax
    cfg['ymin'] = yMin
    cfg['ymax'] = yMax
    
    cfg['xtitle'] = xLabel
    cfg['ytitle'] = yLabel
    

    plotter.cfg = cfg
    canvas = plotter.canvas()
    dummy = plotter.dummy()
        
    
    canvas.SetGrid()
    dummy.Draw("HIST")
        
    st.Draw("HIST SAME")
    
    #for h in sysHists: h.Draw("E2 SAME")
    sysHists[0].Draw("E2 SAME")
    h_bkg.Draw("HIST SAME")
    leg.Draw("SAME")
    
    plotter.aux(canvas)  
    ROOT.gPad.SetTickx()
    ROOT.gPad.SetTicky()
    ROOT.gPad.RedrawAxis()  
    


    canvas.SaveAs("%s/%s.png" % (outDir, fOut))
    canvas.SaveAs("%s/%s.pdf" % (outDir, fOut))
    canvas.Close()

	
	
if __name__ == "__main__":

    flavor = "mumu"
    #flavor = "ee"
    
    MC_SF = 1.0
    if flavor == "mumu": MC_SF = 1.026

    outDir = "/eos/user/j/jaeyserm/www/wmass/lowPU/Z%s/plots_mZ_variation/" % flavor
    functions.prepareDir(outDir)
      
    groups = Datagroups("mz_lowPU_%s.pkl.lz4" % flavor)
    
   

    ############
    ### mT 
    ############
    bhist = groups.readProc("mt_massWeight", "DYmumu")
    hist_nom = narf.hist_to_root(bhist[:, 10])
    hist_plus_1 = narf.hist_to_root(bhist[:, 15])
    hist_minus_1 = narf.hist_to_root(bhist[:, 5])
    hist_plus_2 = narf.hist_to_root(bhist[:, 12])
    hist_minus_2 = narf.hist_to_root(bhist[:, 8])
    
    mT_bins = [0, 10, 15, 20, 25, 30, 35,] + list(range(40, 100, 2)) + [102, 104, 106, 108, 110, 115, 120, 125, 130, 140, 160, 200]
    hist_nom = functions.Rebin(hist_nom, mT_bins)
    hist_plus_1 = functions.Rebin(hist_plus_1, mT_bins)
    hist_minus_1 = functions.Rebin(hist_minus_1, mT_bins)
    hist_plus_2 = functions.Rebin(hist_plus_2, mT_bins)
    hist_minus_2 = functions.Rebin(hist_minus_2, mT_bins)

       
    hist_nom.SetLineColor(ROOT.kBlack)
    hist_nom.SetFillColor(0)
    hist_nom.SetLineWidth(2)
    
    hist_plus_1.SetLineColor(ROOT.kRed)
    hist_plus_1.SetFillColor(0)
    hist_plus_1.SetLineWidth(2)
    hist_minus_1.SetLineColor(ROOT.kRed)
    hist_minus_1.SetFillColor(0)
    hist_minus_1.SetLineWidth(2)
    
    hist_plus_2.SetLineColor(ROOT.kBlue)
    hist_plus_2.SetFillColor(0)
    hist_plus_2.SetLineWidth(2)
    hist_minus_2.SetLineColor(ROOT.kBlue)
    hist_minus_2.SetFillColor(0)
    hist_minus_2.SetLineWidth(2)
    
    hist_plus_err_1 = hist_plus_1.Clone("hist_plus_err_1")
    hist_plus_err_1.Divide(hist_nom)
    hist_minus_err_1 = hist_minus_1.Clone("hist_minus_err_1")
    hist_minus_err_1.Divide(hist_nom)
    
    hist_plus_err_2 = hist_plus_2.Clone("hist_plus_err_2")
    hist_plus_err_2.Divide(hist_nom)
    hist_minus_err_2 = hist_minus_2.Clone("hist_minus_err_2")
    hist_minus_err_2.Divide(hist_nom)
    

    h_err = hist_nom.Clone("syst")
    h_err.SetFillColor(ROOT.kBlack)
    h_err.SetMarkerSize(0)
    h_err.SetLineWidth(0)
    h_err.SetFillStyle(3004)    

    cfg = {

        'logy'              : False,
        'logx'              : False,
        
        'xmin'              : 0,
        'xmax'              : 200,
        'ymin'              : 0,
        'ymax'              : 1.1*hist_nom.GetMaximum(),
            
        'xtitle'            : "m_{T} (GeV)",
        'ytitle'            : "Events",
            
        'topRight'          : "199 pb^{#minus1} (13 TeV)", 
        'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",

        'ratiofraction'     : 0.3,
        'ytitleR'           : "Ratio",
        
        'yminR'             : 0.991,
        'ymaxR'             : 1.009,
    }
    
    
    leg = ROOT.TLegend(.55, 0.70, .9, .88)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.045)
    
    leg.AddEntry(hist_nom, "Nominal (DY #rightarrow #mu^{#plus}#mu^{#minus})", "L")
    leg.AddEntry(hist_plus_1, "#pm 50 MeV", "L")
    leg.AddEntry(hist_plus_2, "#pm 20 MeV", "L")
    
    plotter.cfg = cfg
    canvas, padT, padB = plotter.canvasRatio()
    dummyT, dummyB, dummyL = plotter.dummyRatio()
    canvas.cd()
    padT.Draw()
    padT.cd()
    padT.SetGrid()
    dummyT.GetYaxis().SetTitleOffset(1.3*dummyT.GetYaxis().GetTitleOffset())
    dummyT.Draw("HIST")
    hist_nom.Draw("HIST SAME")
    hist_plus_1.Draw("HIST SAME")
    hist_minus_1.Draw("HIST SAME")
    hist_plus_2.Draw("HIST SAME")
    hist_minus_2.Draw("HIST SAME")
    h_err.Draw("E2 SAME")
    leg.Draw("SAME")
    
    plotter.auxRatio()  
    ROOT.gPad.SetTickx()
    ROOT.gPad.SetTicky()
    ROOT.gPad.RedrawAxis()  
    
    canvas.cd()
    padB.Draw()
    padB.SetFillStyle(0)
    padB.cd()
    dummyB.GetYaxis().SetTitleOffset(1.3*dummyB.GetYaxis().GetTitleOffset())
    dummyB.Draw("HIST")
    dummyL.Draw("SAME")
    
    hist_plus_err_1.Draw("HIST SAME")
    hist_minus_err_1.Draw("HIST SAME")
    hist_plus_err_2.Draw("HIST SAME")
    hist_minus_err_2.Draw("HIST SAME")
    
    ROOT.gPad.SetTickx()
    ROOT.gPad.SetTicky()
    ROOT.gPad.RedrawAxis()
    canvas.SaveAs("%s/mT_mass_weights.png" % (outDir))
    canvas.SaveAs("%s/mT_mass_weights.pdf" % (outDir))
    canvas.Close()



    ############
    ### recoil 
    ############
    bhist = groups.readProc("gen_reco_mll_massWeight", "DYmumu")
    #s = hist.tag.Slicer()
    hist_nom = narf.hist_to_root(bhist[::hist.sum, :, ::hist.sum, 10])
    hist_plus_1 = narf.hist_to_root(bhist[::hist.sum, :, ::hist.sum, 15])
    hist_minus_1 = narf.hist_to_root(bhist[::hist.sum, :, ::hist.sum, 5])
    hist_plus_2 = narf.hist_to_root(bhist[::hist.sum, :, ::hist.sum, 12])
    hist_minus_2 = narf.hist_to_root(bhist[::hist.sum, :, ::hist.sum, 8])
    
    hist_nom.Scale(1, "width")
    hist_plus_1.Scale(1, "width")
    hist_minus_1.Scale(1, "width")
    hist_plus_2.Scale(1, "width")
    hist_minus_2.Scale(1, "width")
       
    hist_nom.SetLineColor(ROOT.kBlack)
    hist_nom.SetFillColor(0)
    hist_nom.SetLineWidth(2)
    
    hist_plus_1.SetLineColor(ROOT.kRed)
    hist_plus_1.SetFillColor(0)
    hist_plus_1.SetLineWidth(2)
    hist_minus_1.SetLineColor(ROOT.kRed)
    hist_minus_1.SetFillColor(0)
    hist_minus_1.SetLineWidth(2)
    
    hist_plus_2.SetLineColor(ROOT.kBlue)
    hist_plus_2.SetFillColor(0)
    hist_plus_2.SetLineWidth(2)
    hist_minus_2.SetLineColor(ROOT.kBlue)
    hist_minus_2.SetFillColor(0)
    hist_minus_2.SetLineWidth(2)
    
    hist_plus_err_1 = hist_plus_1.Clone("hist_plus_err_1")
    hist_plus_err_1.Divide(hist_nom)
    hist_minus_err_1 = hist_minus_1.Clone("hist_minus_err_1")
    hist_minus_err_1.Divide(hist_nom)
    
    hist_plus_err_2 = hist_plus_2.Clone("hist_plus_err_2")
    hist_plus_err_2.Divide(hist_nom)
    hist_minus_err_2 = hist_minus_2.Clone("hist_minus_err_2")
    hist_minus_err_2.Divide(hist_nom)
    

    h_err = hist_nom.Clone("syst")
    h_err.SetFillColor(ROOT.kBlack)
    h_err.SetMarkerSize(0)
    h_err.SetLineWidth(0)
    h_err.SetFillStyle(3004)    

    cfg = {

        'logy'              : False,
        'logx'              : False,
        
        'xmin'              : 0,
        'xmax'              : 200,
        'ymin'              : 0,
        'ymax'              : 1.1*hist_nom.GetMaximum(),
            
        'xtitle'            : "Recoil (GeV)",
        'ytitle'            : "Events",
            
        'topRight'          : "199 pb^{#minus1} (13 TeV)", 
        'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",

        'ratiofraction'     : 0.3,
        'ytitleR'           : "Ratio",
        
        'yminR'             : 0.991,
        'ymaxR'             : 1.009,
    }
    
    
    leg = ROOT.TLegend(.55, 0.70, .9, .88)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.045)
    
    leg.AddEntry(hist_nom, "Nominal (DY #rightarrow #mu^{#plus}#mu^{#minus})", "L")
    leg.AddEntry(hist_plus_1, "#pm 50 MeV", "L")
    leg.AddEntry(hist_plus_2, "#pm 20 MeV", "L")
    
    plotter.cfg = cfg
    canvas, padT, padB = plotter.canvasRatio()
    dummyT, dummyB, dummyL = plotter.dummyRatio()
    canvas.cd()
    padT.Draw()
    padT.cd()
    padT.SetGrid()
    dummyT.GetYaxis().SetTitleOffset(1.3*dummyT.GetYaxis().GetTitleOffset())
    dummyT.Draw("HIST")
    hist_nom.Draw("HIST SAME")
    hist_plus_1.Draw("HIST SAME")
    hist_minus_1.Draw("HIST SAME")
    hist_plus_2.Draw("HIST SAME")
    hist_minus_2.Draw("HIST SAME")
    h_err.Draw("E2 SAME")
    leg.Draw("SAME")
    
    plotter.auxRatio()  
    ROOT.gPad.SetTickx()
    ROOT.gPad.SetTicky()
    ROOT.gPad.RedrawAxis()  
    
    canvas.cd()
    padB.Draw()
    padB.SetFillStyle(0)
    padB.cd()
    dummyB.GetYaxis().SetTitleOffset(1.3*dummyB.GetYaxis().GetTitleOffset())
    dummyB.Draw("HIST")
    dummyL.Draw("SAME")
    
    hist_plus_err_1.Draw("HIST SAME")
    hist_minus_err_1.Draw("HIST SAME")
    hist_plus_err_2.Draw("HIST SAME")
    hist_minus_err_2.Draw("HIST SAME")
    
    ROOT.gPad.SetTickx()
    ROOT.gPad.SetTicky()
    ROOT.gPad.RedrawAxis()
    canvas.SaveAs("%s/recoil_mass_weights.png" % (outDir))
    canvas.SaveAs("%s/recoil_mass_weights.pdf" % (outDir))
    canvas.Close()




    ############
    ### mll 
    ############
    bhist = groups.readProc("gen_reco_mll_massWeight", "DYmumu")
    hist_nom = narf.hist_to_root(bhist[::hist.sum, ::hist.sum, :, 10])
    hist_plus_1 = narf.hist_to_root(bhist[::hist.sum, ::hist.sum, :, 15])
    hist_minus_1 = narf.hist_to_root(bhist[::hist.sum, ::hist.sum, :, 5])
    hist_plus_2 = narf.hist_to_root(bhist[::hist.sum, ::hist.sum, :, 12])
    hist_minus_2 = narf.hist_to_root(bhist[::hist.sum, ::hist.sum, :, 8])
    
    hist_nom.Scale(1, "width")
    hist_plus_1.Scale(1, "width")
    hist_minus_1.Scale(1, "width")
    hist_plus_2.Scale(1, "width")
    hist_minus_2.Scale(1, "width")
       
    hist_nom.SetLineColor(ROOT.kBlack)
    hist_nom.SetFillColor(0)
    hist_nom.SetLineWidth(2)
    
    hist_plus_1.SetLineColor(ROOT.kRed)
    hist_plus_1.SetFillColor(0)
    hist_plus_1.SetLineWidth(2)
    hist_minus_1.SetLineColor(ROOT.kRed)
    hist_minus_1.SetFillColor(0)
    hist_minus_1.SetLineWidth(2)
    
    hist_plus_2.SetLineColor(ROOT.kBlue)
    hist_plus_2.SetFillColor(0)
    hist_plus_2.SetLineWidth(2)
    hist_minus_2.SetLineColor(ROOT.kBlue)
    hist_minus_2.SetFillColor(0)
    hist_minus_2.SetLineWidth(2)
    
    hist_plus_err_1 = hist_plus_1.Clone("hist_plus_err_1")
    hist_plus_err_1.Divide(hist_nom)
    hist_minus_err_1 = hist_minus_1.Clone("hist_minus_err_1")
    hist_minus_err_1.Divide(hist_nom)
    
    hist_plus_err_2 = hist_plus_2.Clone("hist_plus_err_2")
    hist_plus_err_2.Divide(hist_nom)
    hist_minus_err_2 = hist_minus_2.Clone("hist_minus_err_2")
    hist_minus_err_2.Divide(hist_nom)
    

    h_err = hist_nom.Clone("syst")
    h_err.SetFillColor(ROOT.kBlack)
    h_err.SetMarkerSize(0)
    h_err.SetLineWidth(0)
    h_err.SetFillStyle(3004)    

    cfg = {

        'logy'              : False,
        'logx'              : False,
        
        'xmin'              : 60,
        'xmax'              : 120,
        'ymin'              : 0,
        'ymax'              : 1.1*hist_nom.GetMaximum(),
            
        'xtitle'            : "m(#mu^{#plus}, #mu^{#minus}) (GeV)",
        'ytitle'            : "Events",
            
        'topRight'          : "199 pb^{#minus1} (13 TeV)", 
        'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",

        'ratiofraction'     : 0.3,
        'ytitleR'           : "Ratio",
        
        'yminR'             : 0.965,
        'ymaxR'             : 1.035,
    }
    
    
    leg = ROOT.TLegend(.55, 0.70, .9, .88)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.045)
    
    leg.AddEntry(hist_nom, "Nominal (DY #rightarrow #mu^{#plus}#mu^{#minus})", "L")
    leg.AddEntry(hist_plus_1, "#pm 50 MeV", "L")
    leg.AddEntry(hist_plus_2, "#pm 20 MeV", "L")
    
    plotter.cfg = cfg
    canvas, padT, padB = plotter.canvasRatio()
    dummyT, dummyB, dummyL = plotter.dummyRatio()
    canvas.cd()
    padT.Draw()
    padT.cd()
    padT.SetGrid()
    dummyT.GetYaxis().SetTitleOffset(1.3*dummyT.GetYaxis().GetTitleOffset())
    dummyT.Draw("HIST")
    hist_nom.Draw("HIST SAME")
    hist_plus_1.Draw("HIST SAME")
    hist_minus_1.Draw("HIST SAME")
    hist_plus_2.Draw("HIST SAME")
    hist_minus_2.Draw("HIST SAME")
    h_err.Draw("E2 SAME")
    leg.Draw("SAME")
    
    plotter.auxRatio()  
    ROOT.gPad.SetTickx()
    ROOT.gPad.SetTicky()
    ROOT.gPad.RedrawAxis()  
    
    canvas.cd()
    padB.Draw()
    padB.SetFillStyle(0)
    padB.cd()
    dummyB.GetYaxis().SetTitleOffset(1.3*dummyB.GetYaxis().GetTitleOffset())
    dummyB.Draw("HIST")
    dummyL.Draw("SAME")
    
    hist_plus_err_1.Draw("HIST SAME")
    hist_minus_err_1.Draw("HIST SAME")
    hist_plus_err_2.Draw("HIST SAME")
    hist_minus_err_2.Draw("HIST SAME")
    
    ROOT.gPad.SetTickx()
    ROOT.gPad.SetTicky()
    ROOT.gPad.RedrawAxis()
    canvas.SaveAs("%s/mZ_mass_weights.png" % (outDir))
    canvas.SaveAs("%s/mZ_mass_weights.pdf" % (outDir))
    canvas.Close()
