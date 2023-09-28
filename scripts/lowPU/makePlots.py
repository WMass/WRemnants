import sys,array,math,os,copy,fnmatch
from collections import OrderedDict

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptTitle(0)

import hist
from utilities import boostHistHelpers as hh
import numpy as np


import plotter
import functions

import lz4.frame
import pickle
import narf

import wremnants.datasets.datagroups as datagroups
#from wremnants.datasets.datagroups2016 import make_datagroups_2016
#from wremnants.datasets.datagroupsLowPU import make_datagroups_lowPU
#from wremnants.datasets.datagroupsLowPU import datagroupsLowPU
#from wremnants.datasets.datagroups import datagroups2016
from wremnants import histselections as sel


def parseHist(h, norm=1, rebin=1): # parse 1D histogram

    h = functions.Rebin(h, rebin)
    h.Scale(norm)

    # overflow
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



def addUnc(hTarget, hNom, hUp, hDw=None):

    for k in range(1, hTarget.GetNbinsX()+1):

        #if hDw != None: sigma = 0.5*abs(abs(hUp.GetBinContent(k)-hNom.GetBinContent(k)) + abs(hDw.GetBinContent(k)-hNom.GetBinContent(k)))
        if hDw != None: sigma = 0.5*abs(hUp.GetBinContent(k)-hDw.GetBinContent(k))
        else: sigma = abs(hUp.GetBinContent(k)-hNom.GetBinContent(k))
        hTarget.SetBinError(k, math.sqrt(sigma*sigma + hTarget.GetBinError(k)*hTarget.GetBinError(k)))




def parseProc_Z(histCfg, procName, syst="", rebin=1):

    axis = histCfg['axis']
    hName = histCfg['name']

    label = "%s_%s" % (hName, procName)
    bhist = functions.readBoostHistProc(groups, hName, [procName])
    rhist = narf.hist_to_root(bhist)
    rhist = functions.Rebin(rhist, rebin)
    rhist.SetName(label)
    rhist = doOverlow(rhist)

    if procName == "SingleMuon" or procName == "SingleElectron": pass
    else: rhist.Scale(MC_SF)

    print("Get histogram %s, yield=%.2f" % (label, rhist.Integral()))
    return rhist

def getHist(histCfg, procName, syst="", boost=False):
    axis = histCfg['axis']
    hName = histCfg['name']
    charge = histCfg['charge'] if 'charge' in histCfg else False

    if syst != "":
        hName += f"_{syst}"
    #label = "%s_%s" % (hName, procName)
    #groups.setHists(hName, "", label=label, procsToRead=[procName])
    #bhist = groups.groups[procName].hists[label]

    groups.setNominalName(hName)
    groups.loadHistsForDatagroups(hName, syst="", procsToRead=[procName]) #, procsToRead=datasets, applySelection=applySelection)
    hists = groups.getDatagroups()
    bhist = groups.groups[procName].hists[hName]
    
    s = hist.tag.Slicer()
    axes = [ax.name for ax in bhist.axes]
    if "eta" in axes and "pt" in axes:
        bhist = bhist[{"eta" : s[::hist.sum], "pt" : s[::hist.sum]}]

    if charge and charge == "combined": bhist = bhist[{"charge" : s[::hist.sum]}]
    elif charge and charge == "plus": bhist = bhist[{"charge" : bhist.axes["charge"].index(+1)}]
    elif charge and charge == "minus": bhist = bhist[{"charge" : bhist.axes["charge"].index(-1)}]

    if boost:
        return bhist
    rhist = narf.hist_to_root(bhist)
    rhist.SetName(f"{procName}_{hName}")
    return rhist


def doRecoilStatSyst(histCfg, procName, hNom, h_syst, rebin):
    tags = ["target_para", "target_perp", "source_para", "source_perp", "target_para_bkg", "target_perp_bkg"]
    tags = ["para", "perp"]
    doubleSide = False
    for tag in tags:
        try:
            bhist_pert = getHist(histCfg, procName, syst=f"recoilUnc_{tag}", boost=True)
        except:
            continue
        ax_entries = list(bhist_pert.axes[1])
        s = hist.tag.Slicer()
        print(f"Include {tag} recoil uncertainty for process {procName} and histogram {histCfg['name']}")
        if doubleSide:
            for i in range(int(len(ax_entries)/2)):

                hDw = bhist_pert[{"recoilVar" : 2*(i-1)-1}]
                hUp = bhist_pert[{"recoilVar" : 2*i-1}]

                hDw = narf.hist_to_root(hDw)
                hUp = narf.hist_to_root(hUp)

                hUp = parseHist(hUp, rebin=rebin)
                hDw = parseHist(hDw, rebin=rebin)
                print(f" variation {i}, nom={hNom.Integral()}, up={hUp.Integral()}, dw={hDw.Integral()}")
                #if tag == "target_para" and i == 3:
                #    for k in range(0, hNom.GetNbinsX()): print(" ", tag, i, hNom.GetBinCenter(k), hNom.GetBinContent(k), hUp.GetBinContent(k), hDw.GetBinContent(k), hNom.GetBinContent(k)/hUp.GetBinContent(k) if hUp.GetBinContent(k) > 0 else 1)
                addUnc(h_syst, hNom, hUp, hDw)
        else:
            for i in range(int(len(ax_entries))):
                hUp = bhist_pert[{"recoilVar" : i}]
                hUp = narf.hist_to_root(hUp)
                hUp = parseHist(hUp, rebin=rebin)
                #print(f" variation {i}, nom={hNom.Integral()}, up={hUp.Integral()}")
                #if tag == "target_para" and i == 3:
                #    for k in range(0, hNom.GetNbinsX()): print(" ", tag, i, hNom.GetBinCenter(k), hNom.GetBinContent(k), hUp.GetBinContent(k), hDw.GetBinContent(k), hNom.GetBinContent(k)/hUp.GetBinContent(k) if hUp.GetBinContent(k) > 0 else 1)
                addUnc(h_syst, hNom, hUp)


def parseHists(histCfg, leg, rebin=1, noData=False):

    h_data = getHist(histCfg, data)
    h_data = parseHist(h_data, rebin=rebin)
    leg.AddEntry(h_data, groups.groups[data].label, "PE")

    st = ROOT.THStack()
    st.SetName("stack")
    h_bkg = h_data.Clone("bkg_nominal")
    h_bkg.Reset("ACE")
    bkg_hists = {}
    normMC = 0
    for i,proc in enumerate(procs):

        hist = getHist(histCfg, proc)
        hist = parseHist(hist, rebin=rebin)
        hist.SetFillColor(ROOT.TColor.GetColor(groups.groups[proc].color))
        hist.SetLineColor(ROOT.kBlack)
        hist.SetLineWidth(1)
        hist.SetLineStyle(1)
        bkg_hists[proc] = hist
        if proc != dataNormProc:
            normMC += hist.Integral()

    if dataNormProc != "":
        bkg_hists[dataNormProc].Scale((h_data.Integral()-normMC)/bkg_hists[dataNormProc].Integral())
    for i,proc in enumerate(procs):
        leg.AddEntry(bkg_hists[proc], groups.groups[proc].label if proc != "WJetsToMuNu" else "W^{{#{q}}} #rightarrow #mu^{{#{q}}}#nu".format(q=charge if charge != "combined" else "pm"), "F")
        st.Add(bkg_hists[proc])
        h_bkg.Add(bkg_hists[proc])


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
    leg.AddEntry(h_err, "Stat. Unc." if not doSyst else "Stat. + Syst. Unc.", "F")

    h_syst = h_bkg.Clone("syst_group%d" % i)
    h_syst.SetFillColor(ROOT.kBlack)
    h_syst.SetFillColorAlpha(ROOT.kBlack, 1)
    h_syst.SetMarkerSize(0)
    h_syst.SetLineWidth(0)
    h_syst.SetFillStyle(3004)
    for iBin in range(0, h_syst.GetNbinsX()+1): h_syst.SetBinError(iBin, 0) # reset errors for non-stat ones

    systs = {}

    # do systematics
    for i,proc in enumerate(procs):

        if not doSyst:
            continue

        hNom = getHist(histCfg, proc)
        hNom = parseHist(hNom, rebin=rebin)
        hPert = doRecoilStatSyst(histCfg, proc, hNom, h_err, rebin)




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



def singlePlot(histCfg, fOut, xMin, xMax, yMin, yMax, xLabel, yLabel, logY=True, rebin=1, legPos=[], yRatio = 1.3, dataNorm=False):

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

        'topRight'          : lumi_header,
        'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",

        'ratiofraction'     : 0.3,
        'ytitleR'           : "Data/MC",

        'yminR'             : 1-(yRatio-1), # 0.7
        'ymaxR'             : yRatio, # 1.3
    }




    leg = ROOT.TLegend(.50, 0.88-(len(procs)+2)*0.05, .8, .88)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.040)


    h_data, st, h_bkg, h_ratio, h_err, h_err_ratio = parseHists(histCfg, leg, rebin)

    if dataNorm:
        for i in range(0, h_data.GetNbinsX()+1):
            h_data.SetBinContent(i, h_bkg.GetBinContent(i))
            h_ratio.SetBinContent(i, 1)

    cfg['logy'] = logY
    cfg['xmin'] = xMin
    cfg['xmax'] = xMax
    cfg['ymin'] = yMin
    cfg['ymax'] = yMax

    cfg['xtitle'] = xLabel
    cfg['ytitle'] = yLabel

    if not logY:
        ROOT.TGaxis.SetMaxDigits(3)
        ROOT.TGaxis.SetExponentOffset(-0.075, 0.01, "y")


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

    h_ratio.Draw("PE0 SAME") # E2 SAME
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

        'topRight'          : lumi_header,
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


def MET2d(type_="ratio"):

    histCfg = {"name": "recoil_corr_rec_para_perp_2d", "axis": "recoil_perp" }
    h_data = parseProc(histCfg, data)
    hist_mc = None
    for i,proc in enumerate(procs):

        hist = parseProc(histCfg, proc)
        if hist_mc == None: hist_mc = hist
        else: hist_mc.Add(hist)

    h_ratio = h_data.Clone("ratio")
    h_ratio.Divide(hist_mc)

    '''
    for i in range(1, h_data.GetNbinsX()+1):
        for j in range(1, h_data.GetNbinsX()+1):

            val = h_data.GetBinContent(i, j)
            if val >= 1 and val < 1.05: newval = 1.025
            elif val >= 1.05 and val < 1.1: newval = 1.075
            elif val >= 1.1 and val < 1.15: newval = 1.125
            elif val >= 1.15 and val < 1.2: newval = 1.175
            elif val >= 1.2: newval = 1.3
            elif val <= 1 and val > 0.95: newval = 0.975
            elif val <= 0.95 and val > 0.9: newval = 0.925
            elif val <= 0.9 and val > 0.85: newval = 0.875
            elif val <= 0.85 and val > 0.8: newval = 0.825
            elif val <= 0.8: newval = 0.7
            h_data.SetBinContent(i, j, newval)
            #print("%.3f " % h_data.GetBinContent(i, j), end="")
    '''

    cfg = {

        'logy'              : False,
        'logx'              : False,

        'xmin'              : -10,
        'xmax'              : 10,
        'ymin'              : -10,
        'ymax'              : 10,

        'xtitle'            : "Recoil parallel",
        'ytitle'            : "Recoil perp",

        'topRight'          : lumi_header,
        'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",
    }



    #ROOT.gStyle.SetPalette(6, [ROOT.kRed+2, ROOT.kOrange+2 , ROOT.kGreen +2 , ROOT.kGreen +2 , ROOT.kAzure +2, ROOT.kBlue  +2])
    #ROOT.gStyle.SetNumberContours(6)
    #ROOT.gStyle.SetPalette(ROOT.kVisibleSpectrum)


    plotter.cfg = cfg
    canvas = plotter.canvas()
    canvas.SetRightMargin(0.12)
    #canvas.SetLogz()
    dummy = plotter.dummy()
    canvas.cd()
    dummy.Draw("HIST")

    ROOT.gStyle.SetPaintTextFormat("4.3f")
    h_data.SetMarkerSize(0.4)

    if type_ == "data":
        #h_data.GetZaxis().SetRangeUser(0.8, 1.2)
        #h_data.Draw("SAME COLZ TEXTE")
        h_data.Draw("SAME COLZ")
    elif type_ == "mc":
        #h_mc.GetZaxis().SetRangeUser(0.8, 1.2)
        #h_data.Draw("SAME COLZ TEXTE")
        hist_mc.Draw("SAME COLZ")
    else:
        #h_ratio.GetZaxis().SetRangeUser(0.8, 1.2)
        #h_data.Draw("SAME COLZ TEXTE")
        h_ratio.Draw("SAME COLZ")

    ells = []
    for r in [1, 2, 3]:
        met1 = ROOT.TEllipse(0., 0., r, r)
        met1.SetFillStyle(4000)
        met1.SetLineWidth(1)
        met1.Draw("SAME")
        ells.append(met1)


    plotter.aux(canvas)

    canvas.Modify()
    canvas.Update()
    canvas.Draw()
    canvas.RedrawAxis()
    canvas.SaveAs("%s/MET_2d_%s.png" % (outDir, type_))
    canvas.SaveAs("%s/MET_2d_%s.pdf" % (outDir, type_))
    canvas.Delete()

    for i in range(1, h_data.GetNbinsX()+1):
        for j in range(1, h_data.GetNbinsX()+1):

            print("%.3f " % h_data.GetBinContent(i, j), end="")
        print()

        #hx = h_data.ProjectionX("x", i, i)
        #hy = h_data.ProjectionY("y", i, i)

        #print(hx.Integral()/hx.GetNbinsX(), hy.Integral()/hy.GetNbinsX())

def doYields_Z(hName):

    with open("%s/yields.txt" % outDir, 'w') as f:
        sys.stdout = f

        formatted_row = '{:<15} {:<25}'
        print(formatted_row.format(*(["Process", "Yields"])))
        print(formatted_row.format(*(["---------------"]+["-----------------------"]*3)))
        for proc in procs+[data]:
            row = [proc]
            h = getHist({"name": hName, "axis": ""}, proc, boost=True)
            
            yield_, err = sum(h.values()), math.sqrt(sum(h.variances()))
            row.append("%.3e +/- %.3e" % (yield_, err))
            print(formatted_row.format(*row))
    sys.stdout = sys.__stdout__

def doYields_W(hName):

    with open("%s/yields.txt" % outDir, 'w') as f:
        sys.stdout = f

        formatted_row = '{:<15} {:<25} {:<25} {:<25}'
        print(formatted_row.format(*(["Process", "Yields plus", "Yields minus", "Yields combined"])))
        print(formatted_row.format(*(["---------------"]+["-----------------------"]*3)))
        for proc in procs+[data]:
            row = [proc]
            for q in ["plus", "minus", "combined"]:
                h = getHist({"name": hName, "axis": "", "charge": q}, proc, boost=True)
                
                #s = hist.tag.Slicer()
                #h = h[{"mt": s[complex(0,40):complex(0,120)]}]
                #print(h)
                #print(h.values())
                #print(h.variances())
                #print(h.values())
                yield_, err = sum(h.values()), math.sqrt(sum(h.variances()))
                row.append("%.3e +/- %.3e" % (yield_, err))
            print(formatted_row.format(*row))
    sys.stdout = sys.__stdout__




def lowPU_Z():

    doYields_Z("mZ")
    singlePlot({"name": "mZ", "axis": "mll" }, "mZ", 60, 120, 1e0, 1e7, "m(l, l) (GeV)", "Events", yRatio=1.15)

    ## recoil plots
    bins_recoil_para_perp = [-100, -80, -70, -60, -50, -46, -42, -38, -34] + list(range(-30, 30, 2)) + [30, 34, 38, 42, 46, 50, 60, 70, 80, 100]
    bins_recoil_magn = list(range(0, 50, 5)) + list(range(50, 100, 5)) + [100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
    singlePlot({"name": "recoil_uncorr_para", "axis": "recoil_perp" }, "recoil_uncorr_para", -100, 100, 1e-1, 1e7, "U_{#parallel} (GeV) (uncorrected)", "Events", rebin=bins_recoil_para_perp, yRatio=1.15)
    singlePlot({"name": "recoil_uncorr_perp", "axis": "recoil_perp" }, "recoil_uncorr_perp", -100, 100, 1e-1, 1e7, "U_{#perp}  (GeV) (uncorrected)", "Events", rebin=bins_recoil_para_perp, yRatio=1.15)
    singlePlot({"name": "recoil_uncorr_magn", "axis": "recoil_magn" }, "recoil_uncorr_magn", 0, 200, 1e-1, 1e7, "|U| (GeV) (uncorrected)", "Events", rebin=bins_recoil_magn, yRatio=1.15)

    singlePlot({"name": "recoil_corr_lep_para", "axis": "recoil_perp" }, "recoil_corr_lep_para", -100, 100, 1e-1, 1e7, "U_{#parallel} (GeV) (lepton corrected)", "Events", rebin=bins_recoil_para_perp, yRatio=1.15)
    singlePlot({"name": "recoil_corr_lep_perp", "axis": "recoil_perp" }, "recoil_corr_lep_perp", -100, 100, 1e-1, 1e7, "U_{#perp}   (GeV) (lepton corrected)", "Events", rebin=bins_recoil_para_perp, yRatio=1.15)
    singlePlot({"name": "recoil_corr_lep_magn", "axis": "recoil_magn" }, "recoil_corr_lep_magn", 0, 200, 1e-1, 1e7, "|U| (GeV) (lepton corrected)", "Events", rebin=bins_recoil_magn, yRatio=1.15)

    singlePlot({"name": "recoil_corr_xy_para", "axis": "recoil_perp" }, "recoil_corr_xy_para", -100, 100, 1e-1, 1e7, "U_{#parallel} (GeV) (XY corrected)", "Events", rebin=bins_recoil_para_perp, yRatio=1.15)
    singlePlot({"name": "recoil_corr_xy_perp", "axis": "recoil_perp" }, "recoil_corr_xy_perp", -100, 100, 1e-1, 1e7, "U_{#perp}   (GeV) (XY corrected)", "Events", rebin=bins_recoil_para_perp, yRatio=1.15)
    singlePlot({"name": "recoil_corr_xy_magn", "axis": "recoil_magn" }, "recoil_corr_xy_magn", 0, 200, 1e-1, 1e7, "|U| (GeV) (XY corrected)", "Events", rebin=bins_recoil_magn, yRatio=1.15)

    singlePlot({"name": "recoil_corr_rec_para", "axis": "recoil_para_qT" }, "recoil_corr_rec_para", -100, 100, 1e-1, 1e7, "U_{#parallel} (GeV) (recoil corrected)", "Events", rebin=bins_recoil_para_perp, yRatio=1.15)
    singlePlot({"name": "recoil_corr_rec_para_qTrw", "axis": "recoil_para_qT" }, "recoil_corr_rec_para_qTrw", -100, 100, 1e-1, 1e7, "U_{#parallel} (GeV) (recoil corrected, q_{T} rw)", "Events", rebin=bins_recoil_para_perp, yRatio=1.15)

    singlePlot({"name": "recoil_corr_rec_perp", "axis": "recoil_perp" }, "recoil_corr_rec_perp", -100, 100, 1e-1, 1e7, "U_{#perp}   (GeV) (recoil corrected)", "Events", rebin=bins_recoil_para_perp, yRatio=1.15)
    singlePlot({"name": "recoil_corr_rec_perp_qTrw", "axis": "recoil_perp" }, "recoil_corr_rec_perp_qTrw", -100, 100, 1e-1, 1e7, "U_{#perp}   (GeV) (recoil corrected, q_{T} rw)", "Events", rebin=bins_recoil_para_perp, yRatio=1.15)

    singlePlot({"name": "recoil_corr_rec_magn", "axis": "recoil_magn" }, "recoil_corr_rec_magn", 0, 200, 1e-1, 1e7, "|U| (GeV) (recoil corrected)", "Events", rebin=bins_recoil_magn, yRatio=1.15)
    singlePlot({"name": "recoil_corr_rec_magn_qTrw", "axis": "recoil_magn_qTrw" }, "recoil_corr_rec_magn_qTrw", 0, 200, 1e-1, 1e7, "|U| (GeV) (recoil corrected, q_{T} rw)", "Events", rebin=bins_recoil_magn, yRatio=1.15)


    singlePlot({"name": "recoil_corr_xy_para_qT", "axis": "recoil_para" }, "recoil_corr_xy_para_qT", -200, 100, 1e-1, 1e7, "U_{#parallel} #minus q_{T} (GeV)", "Events", rebin=5, yRatio=1.15)
    singlePlot({"name": "recoil_corr_rec_para_qT", "axis": "recoil_para" }, "recoil_corr_rec_para_qT", -200, 100, 1e-1, 1e7, "U_{#parallel} #minus q_{T} (GeV) (recoil corrected)", "Events", rebin=5, yRatio=1.15)
    singlePlot({"name": "recoil_corr_rec_para_qT_qTrw", "axis": "recoil_para_qTrw" }, "recoil_corr_rec_para_qT_qTrw", -200, 100, 1e-1, 1e7, "U_{#parallel} #minus q_{T} (GeV) (recoil corrected, q_{T} rw)", "Events", rebin=5, yRatio=1.15)


    ##### MET PLOTS
    bins_MET = list(range(0, 50, 2)) + list(range(50, 70, 4)) + [70, 80, 90, 100] # was 2 before
    bins_MET = [0, 0.5, 1, 1.5] + list(range(2, 20, 1)) + list(range(20, 50, 2)) + list(range(50, 70, 4)) + [70, 80, 90, 100] # was 2 before
    bins_MET = list(range(0, 20, 2)) + list(range(20, 50, 5)) + [50, 60, 70, 80, 90, 100] # was 2 before
    singlePlot({"name": "MET_uncorr_pt", "axis": "MET_pt" }, "MET_uncorr_pt", 0, 100, 1, 1e5, "MET p_{T} (uncorrected)", "Events", rebin=bins_MET, yRatio=1.15)
    singlePlot({"name": "MET_corr_lep_pt", "axis": "MET_pt" }, "MET_corr_lep_pt", 0, 100, 1, 1e5, "MET p_{T} (lepton corrected)", "Events", rebin=bins_MET, yRatio=1.15)
    singlePlot({"name": "MET_corr_xy_pt", "axis": "MET_pt" }, "MET_corr_xy_pt", 0, 100, 1, 1e5, "MET p_{T} (XY corrected)", "Events", rebin=bins_MET, yRatio=1.15)
    singlePlot({"name": "MET_corr_rec_pt", "axis": "MET_pt" }, "MET_corr_rec_pt", 0, 100, 1, 1e5, "MET p_{T} (recoil corrected)", "Events", rebin=bins_MET, yRatio=1.15)
    singlePlot({"name": "MET_corr_rec_pt_qTrw", "axis": "MET_pt" }, "MET_corr_rec_pt_qTrw", 0, 100, 1, 1e5, "MET p_{T} (recoil corrected, q_{T} rw)", "Events", rebin=bins_MET, yRatio=1.15)

    singlePlot({"name": "MET_uncorr_phi", "axis": "recoil_MET_phi" }, "MET_uncorr_phi", -4, 4, 1e1, 1e7, "MET #phi (uncorrected)", "Events", rebin=1, yRatio=1.15)
    singlePlot({"name": "MET_corr_lep_phi", "axis": "recoil_MET_phi" }, "MET_corr_lep_phi", -4, 4, 1e1, 1e7, "MET #phi (lepton corrected)", "Events", rebin=1, yRatio=1.15)
    singlePlot({"name": "MET_corr_xy_phi", "axis": "recoil_MET_phi" }, "MET_corr_xy_phi", -4, 4, 1e1, 1e7, "MET #phi (XY corrected)", "Events", rebin=1, yRatio=1.15)
    singlePlot({"name": "MET_corr_rec_phi", "axis": "recoil_MET_phi" }, "MET_corr_rec_phi", -4, 4, 1e1, 1e7, "MET #phi (recoil corrected)", "Events", rebin=1, yRatio=1.15)

    mT_bins = [0, 10, 15, 20, 25, 30, 35,] + list(range(40, 100, 2)) + [100, 102, 104, 106, 108, 110, 115, 120, 125, 130, 140, 160, 200]
    singlePlot({"name": "mT_uncorr", "axis": "mt" }, "mT_uncorr", 0, 200, 1e-1, 1e6, "m_{T} (GeV) (uncorrected)", "Events", rebin=mT_bins, yRatio=1.15)
    singlePlot({"name": "mT_corr_lep", "axis": "mt" }, "mT_corr_lep", 0, 200, 1e-1, 1e6, "m_{T} (GeV) (lepton corrected)", "Events", rebin=mT_bins, yRatio=1.15)
    singlePlot({"name": "mT_corr_xy", "axis": "mt" }, "mT_corr_xy", 0, 200, 1e-1, 1e6, "m_{T} (GeV) (XY corrected)", "Events", rebin=mT_bins, yRatio=1.15)
    singlePlot({"name": "mT_corr_rec", "axis": "mt" }, "mT_corr_rec", 0, 200, 1e-1, 1e6, "m_{T} (GeV)  (recoil corrected)", "Events", rebin=mT_bins, yRatio=1.15)
    singlePlot({"name": "mT_corr_rec_qTrw", "axis": "mt" }, "mT_corr_rec_qTrw", 0, 200, 1e-1, 1e6, "m_{T} (GeV)  (recoil corrected, q_{T} rw)", "Events", rebin=mT_bins, yRatio=1.15)



    recoil_qTbins = list(range(0, 30, 1)) + list(range(30, 50, 2)) + list(range(50, 70, 5)) + list(range(70, 100, 10)) + [100]
    recoil_qTbins = list(range(0, 30, 1)) + list(range(30, 50, 2)) + list(range(50, 70, 5)) + list(range(70, 120, 10)) + [120, 150]
    singlePlot({"name": "qT", "axis": "qT" }, "qT", 0, 100, 0.1, 1e6, "q_{T} (GeV)", "Events", rebin=recoil_qTbins, yRatio=1.15)
    singlePlot({"name": "qT_qTrw", "axis": "qT" }, "qT_qTrw", 0, 100, 0.1, 1e6, "q_{T} (GeV) (q_{T} rw)", "Events", rebin=recoil_qTbins, yRatio=1.15)

    singlePlot({"name": "METx_uncorr", "axis": "MET_xy" }, "METx_uncorr", -100, 100, 1e1, 1e6, "MET x (uncorrected)", "Events", rebin=1)
    singlePlot({"name": "METy_uncorr", "axis": "MET_xy" }, "METy_uncorr", -100, 100, 1e1, 1e6, "MET y (uncorrected)", "Events", rebin=1)

    singlePlot({"name": "METx_corr_lep", "axis": "MET_xy" }, "METx_corr_lep", -100, 100, 1e1, 1e6, "MET x (lepton corrected)", "Events", rebin=1)
    singlePlot({"name": "METy_corr_lep", "axis": "MET_xy" }, "METy_corr_lep", -100, 100, 1e1, 1e6, "MET y (lepton corrected)", "Events", rebin=1)

    singlePlot({"name": "METx_corr_xy", "axis": "MET_xy" }, "METx_corr_xy", -100, 100, 1e1, 1e6, "MET x (XY corrected)", "Events", rebin=1)
    singlePlot({"name": "METy_corr_xy", "axis": "MET_xy" }, "METy_corr_xy", -100, 100, 1e1, 1e6, "MET y (XY corrected)", "Events", rebin=1)

    singlePlot({"name": "METx_corr_rec", "axis": "MET_xy" }, "METx_corr_rec", -100, 100, 1e1, 1e6, "MET x (recoil corrected)", "Events", rebin=1)
    singlePlot({"name": "METy_corr_rec", "axis": "MET_xy" }, "METy_corr_rec", -100, 100, 1e1, 1e6, "MET y (recoil corrected)", "Events", rebin=1)

    singlePlot({"name": "MET_corr_lep_ll_dPhi", "axis": "recoil_MET_phi" }, "MET_corr_lep_ll_dPhi", -4, 4, 1e1, 1e6, "#Delta#phi(MET lep corr, ll) (lepton corrected)", "Events", rebin=1)
    singlePlot({"name": "MET_corr_xy_ll_dPhi", "axis": "recoil_MET_phi" }, "MET_corr_xy_ll_dPhi", -4, 4, 1e1, 1e6, "#Delta#phi(MET xy corr, ll) (lepton corrected)", "Events", rebin=1)
    singlePlot({"name": "MET_corr_rec_ll_dPhi", "axis": "recoil_MET_phi" }, "MET_corr_rec_ll_dPhi", -4, 4, 1e1, 1e6, "#Delta#phi(MET rec corr, ll) (lepton corrected)", "Events", rebin=1)
    #singlePlot({"name": "ll_phi", "axis": "recoil_MET_phi" }, "ll_phi", -4, 4, 1e1, 1e6, "ll #phi (lepton corrected)", "Events", rebin=1)
    #singlePlot({"name": "MET_corr_rec_xy_dPhi", "axis": "recoil_MET_phi" }, "MET_corr_rec_xy_dPhi", -4, 4, 1e1, 1e6, "#phi(MET rec corr) - #phi(MET xy corr)", "Events", rebin=1)

    singlePlot({"name": "npv", "axis": "recoil_npv" }, "npv", 0, 15, 1e1, 1e7, "Number of primary vertices", "Events", rebin=1, yRatio=1.8)

def highPU_Z():
    #mT_bins = [0, 10, 15, 20, 25, 30, 35,] + list(range(40, 100, 2)) + [100, 102, 104, 106, 108, 110, 115, 120, 125, 130, 140, 160, 200]
    #singlePlot({"name": "mT_corr_rec", "axis": "mt" }, "mT_corr_rec", 0, 200, 1e0, 1e8, "m_{T} (GeV)  (recoil corrected)", "Events", rebin=mT_bins, yRatio=1.15)
    #singlePlot({"name": "mT_corr_rec_qTrw", "axis": "mt" }, "mT_corr_rec_qTrw", 0, 200, 1e0, 1e8, "m_{T} (GeV)  (recoil corrected, q_{T} rw)", "Events", rebin=mT_bins, yRatio=1.15)
    #quit()
    doYields_Z("mT_corr_lep")
    #quit()
    bins_recoil_para_perp = [-150, -120, -110, -100, -90, -80, -70, -60, -50, -46, -42, -38, -34] + list(range(-30, 30, 2)) + [30, 34, 38, 42, 46, 50, 60, 70, 80, 90, 100, 110, 120, 150]
    bins_recoil_magn = list(range(0, 100, 4)) + [100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
    singlePlot({"name": "recoil_uncorr_para", "axis": "recoil_para" }, "recoil_uncorr_para", -150, 150, 1e0, 1e8, "U_{#parallel} (GeV) (uncorrected)", "Events", rebin=bins_recoil_para_perp, yRatio=1.15)

    singlePlot({"name": "recoil_uncorr_perp", "axis": "recoil_perp" }, "recoil_uncorr_perp", -150, 150, 1e0, 1e8, "U_{#perp}  (GeV) (uncorrected)", "Events", rebin=bins_recoil_para_perp, yRatio=1.15)
    singlePlot({"name": "recoil_uncorr_magn", "axis": "recoil_magn" }, "recoil_uncorr_magn", 0, 200, 1e0, 1e8, "|U| (GeV) (uncorrected)", "Events", rebin=bins_recoil_magn, yRatio=1.15)

    singlePlot({"name": "recoil_corr_lep_para", "axis": "recoil_para" }, "recoil_corr_lep_para", -150, 150, 1e0, 1e8, "U_{#parallel} (GeV) (lepton corrected)", "Events", rebin=bins_recoil_para_perp, yRatio=1.15)
    singlePlot({"name": "recoil_corr_lep_perp", "axis": "recoil_perp" }, "recoil_corr_lep_perp", -150, 150, 1e0, 1e8, "U_{#perp}  (GeV) (lepton corrected)", "Events", rebin=bins_recoil_para_perp, yRatio=1.15)
    singlePlot({"name": "recoil_corr_lep_magn", "axis": "recoil_magn" }, "recoil_corr_lep_magn", 0, 200, 1e0, 1e8, "|U| (GeV) (lepton corrected)", "Events", rebin=bins_recoil_magn, yRatio=1.15)


    singlePlot({"name": "recoil_corr_xy_para", "axis": "recoil_para" }, "recoil_corr_xy_para", -150, 150, 1e0, 1e8, "U_{#parallel} (GeV) (XY corrected)", "Events", rebin=bins_recoil_para_perp, yRatio=1.15)
    singlePlot({"name": "recoil_corr_xy_perp", "axis": "recoil_perp" }, "recoil_corr_xy_perp", -150, 150, 1e0, 1e8, "U_{#perp}  (GeV) (XY corrected)", "Events", rebin=bins_recoil_para_perp, yRatio=1.15)
    singlePlot({"name": "recoil_corr_xy_magn", "axis": "recoil_magn" }, "recoil_corr_xy_magn", 0, 200, 1e0, 1e8, "|U| (GeV) (XY corrected)", "Events", rebin=bins_recoil_magn, yRatio=1.15)



    singlePlot({"name": "recoil_corr_rec_para", "axis": "recoil_para" }, "recoil_corr_rec_para", -150, 150, 1e0, 1e8, "U_{#parallel} (GeV) (recoil corrected)", "Events", rebin=bins_recoil_para_perp, yRatio=1.15)
    singlePlot({"name": "recoil_corr_rec_para_qTrw", "axis": "recoil_para_qT" }, "recoil_corr_rec_para_qTrw", -150, 150, 1e0, 1e8, "U_{#parallel} (GeV) (recoil corrected, q_{T} rw)", "Events", rebin=bins_recoil_para_perp, yRatio=1.15)

    singlePlot({"name": "recoil_corr_rec_perp", "axis": "recoil_perp" }, "recoil_corr_rec_perp", -150, 150, 1e0, 1e8, "U_{#perp}  (GeV) (recoil corrected)", "Events", rebin=bins_recoil_para_perp, yRatio=1.15)
    singlePlot({"name": "recoil_corr_rec_perp_qTrw", "axis": "recoil_perp" }, "recoil_corr_rec_perp_qTrw", -150, 150, 1e0, 1e8, "U_{#perp}  (GeV) (recoil corrected, q_{T} rw)", "Events", rebin=bins_recoil_para_perp, yRatio=1.15)


    singlePlot({"name": "recoil_corr_rec_magn", "axis": "recoil_magn" }, "recoil_corr_rec_magn", 0, 200, 1e0, 1e8, "|U| (GeV) (recoil corrected)", "Events", rebin=bins_recoil_magn, yRatio=1.15)
    singlePlot({"name": "recoil_corr_rec_magn_qTrw", "axis": "recoil_magn" }, "recoil_corr_rec_magn_qTrw", 0, 200, 1e0, 1e8, "|U| (GeV) (recoil corrected, q_{T} rw)", "Events", rebin=bins_recoil_magn, yRatio=1.15)

    singlePlot({"name": "recoil_corr_xy_para_qT", "axis": "recoil_para" }, "recoil_corr_xy_para_qT", -200, 100, 1e0, 1e8, "U_{#parallel} #minus q_{T} (GeV) (XY corrected)", "Events", rebin=5, yRatio=1.15)
    singlePlot({"name": "recoil_corr_rec_para_qT", "axis": "recoil_para" }, "recoil_corr_rec_para_qT", -200, 100, 1e0, 1e8, "U_{#parallel} #minus q_{T} (GeV) (recoil corrected)", "Events", rebin=5, yRatio=1.15)
    singlePlot({"name": "recoil_corr_rec_para_qT_qTrw", "axis": "recoil_para_qTrw" }, "recoil_corr_rec_para_qT_qTrw", -200, 100, 1e0, 1e8, "U_{#parallel} #minus q_{T} (GeV) (recoil corrected, q_{T} rw)", "Events", rebin=5, yRatio=1.15)



    ##### MET PLOTS
    bins_MET = list(range(0, 20, 2)) + list(range(20, 50, 2)) + list(range(50, 70, 4)) + [70, 80, 90, 110, 130, 150] # was 2 before
    singlePlot({"name": "MET_uncorr_pt", "axis": "recoil_MET_pt" }, "MET_uncorr_pt", 0, 150, 10, 1e8, "MET p_{T} (uncorrected)", "Events", rebin=bins_MET, yRatio=1.15)
    singlePlot({"name": "MET_corr_lep_pt", "axis": "recoil_MET_pt" }, "MET_corr_lep_pt", 0, 150, 10, 1e8, "MET p_{T} (lepton corrected)", "Events", rebin=bins_MET, yRatio=1.15)
    singlePlot({"name": "MET_corr_xy_pt", "axis": "recoil_MET_pt" }, "MET_corr_xy_pt", 0, 150, 10, 1e8, "MET p_{T} (XY corrected)", "Events", rebin=bins_MET, yRatio=1.15)
    singlePlot({"name": "MET_corr_rec_pt", "axis": "recoil_MET_pt" }, "MET_corr_rec_pt", 0, 150, 10, 1e8, "MET p_{T} (recoil corrected)", "Events", rebin=bins_MET, yRatio=1.15)
    singlePlot({"name": "MET_corr_rec_pt_qTrw", "axis": "recoil_MET_pt" }, "MET_corr_rec_pt_qTrw", 0, 150, 10, 1e8, "MET p_{T} (recoil corrected, q_{T} rw)", "Events", rebin=bins_MET, yRatio=1.15)


    singlePlot({"name": "MET_uncorr_phi", "axis": "recoil_MET_phi" }, "MET_uncorr_phi", -4, 4, 1e3, 1e9, "MET #phi (uncorrected)", "Events", rebin=1)
    singlePlot({"name": "MET_corr_lep_phi", "axis": "recoil_MET_phi" }, "MET_corr_lep_phi", -4, 4, 1e3, 1e9, "MET #phi (lepton corrected)", "Events", rebin=1)
    singlePlot({"name": "MET_corr_xy_phi", "axis": "recoil_MET_phi" }, "MET_corr_xy_phi", -4, 4, 1e3, 1e9, "MET #phi (XY corrected)", "Events", rebin=1, yRatio=1.15)
    singlePlot({"name": "MET_corr_rec_phi", "axis": "recoil_MET_phi" }, "MET_corr_rec_phi", -4, 4, 1e3, 1e9, "MET #phi (recoil corrected)", "Events", rebin=1, yRatio=1.15)





    recoil_qTbins = list(range(0, 50, 1)) + list(range(50, 80, 2)) + list(range(80, 120, 5)) + list(range(120, 160, 10)) + list(range(160, 200, 20)) + [200]
    singlePlot({"name": "qT", "axis": "qT" }, "qT", 0, 120, 10, 1e9, "q_{T} (GeV)", "Events", rebin=recoil_qTbins, yRatio=1.15)
    singlePlot({"name": "qT_qTrw", "axis": "qT" }, "qT_qTrw", 0, 120, 10, 1e9, "q_{T} (GeV), (q_{T} rw)", "Events", rebin=recoil_qTbins, yRatio=1.15)

    singlePlot({"name": "METx_uncorr", "axis": "MET_xy" }, "METx_uncorr", -100, 100, 1e1, 1e8, "MET x (uncorrected)", "Events", rebin=1)
    singlePlot({"name": "METy_uncorr", "axis": "MET_xy" }, "METy_uncorr", -100, 100, 1e1, 1e8, "MET y (uncorrected)", "Events", rebin=1)

    singlePlot({"name": "METx_corr_lep", "axis": "MET_xy" }, "METx_corr_lep", -100, 100, 1e1, 1e8, "MET x (lepton corrected)", "Events", rebin=1)
    singlePlot({"name": "METy_corr_lep", "axis": "MET_xy" }, "METy_corr_lep", -100, 100, 1e1, 1e8, "MET y (lepton corrected)", "Events", rebin=1)

    singlePlot({"name": "METx_corr_xy", "axis": "MET_xy" }, "METx_corr_xy", -100, 100, 1e1, 1e8, "MET x (XY corrected)", "Events", rebin=1)
    singlePlot({"name": "METy_corr_xy", "axis": "MET_xy" }, "METy_corr_xy", -100, 100, 1e1, 1e8, "MET y (XY corrected)", "Events", rebin=1)

    singlePlot({"name": "METx_corr_rec", "axis": "MET_xy" }, "METx_corr_rec", -100, 100, 1e1, 1e8, "MET x (recoil corrected)", "Events", rebin=1)
    singlePlot({"name": "METy_corr_rec", "axis": "MET_xy" }, "METy_corr_rec", -100, 100, 1e1, 1e8, "MET y (recoil corrected)", "Events", rebin=1)

    singlePlot({"name": "npv", "axis": "recoil_npv" }, "npv", 0, 50, 1e2, 1e8, "Number of primary vertices", "Events", rebin=1)
    singlePlot({"name": "njets", "axis": "recoil_njets" }, "njets", 0, 20, 1e3, 1e8, "Number of jets", "Events", rebin=1)
    singlePlot({"name": "RawMET_sumEt", "axis": "recoil_sumEt" }, "RawMET_sumEt", 0, 3000, 10, 1e6, "RawMET sumEt", "Events", rebin=2)

    mT_bins = [0, 10, 15, 20, 25, 30, 35,] + list(range(40, 100, 2)) + [100, 102, 104, 106, 108, 110, 115, 120, 125, 130, 140, 160, 200]
    singlePlot({"name": "mT_uncorr", "axis": "mt" }, "mT_uncorr", 0, 200, 1e0, 1e8, "m_{T} (GeV) (uncorrected)", "Events", rebin=mT_bins, yRatio=1.15)
    singlePlot({"name": "mT_corr_lep", "axis": "mt" }, "mT_corr_lep", 0, 200, 1e0, 1e8, "m_{T} (GeV) (lepton corrected)", "Events", rebin=mT_bins, yRatio=1.15)
    singlePlot({"name": "mT_corr_xy", "axis": "mt" }, "mT_corr_xy", 0, 200, 1e0, 1e8, "m_{T} (GeV) (XY corrected)", "Events", rebin=mT_bins, yRatio=1.15)
    singlePlot({"name": "mT_corr_rec", "axis": "mt" }, "mT_corr_rec", 0, 200, 1e0, 1e8, "m_{T} (GeV)  (recoil corrected)", "Events", rebin=mT_bins, yRatio=1.06)
    singlePlot({"name": "mT_corr_rec_qTrw", "axis": "mt" }, "mT_corr_rec_qTrw", 0, 200, 1e0, 1e8, "m_{T} (GeV)  (recoil corrected, q_{T} rw)", "Events", rebin=mT_bins, yRatio=1.06)
    
    # response corrected profiles
    singlePlot({"name": "mT_corr_rec_resp_qTrw", "axis": "mt" }, "mT_corr_rec_resp_qTrw", 0, 200, 1e0, 1e8, "m_{T} (GeV)  (recoil, resp. corrected, q_{T} rw)", "Events", rebin=mT_bins, yRatio=1.15)
    singlePlot({"name": "mT_corr_rec_resp_qTrw", "axis": "mt" }, "mT_corr_rec_resp_qTrw", 0, 200, 1e0, 1e8, "m_{T} (GeV)  (recoil, resp. corrected, q_{T} rw)", "Events", rebin=mT_bins, yRatio=1.15)
    
    singlePlot({"name": "MET_corr_rec_resp_pt", "axis": "recoil_MET_pt" }, "MET_corr_rec_resp_pt", 0, 150, 10, 1e8, "MET p_{T} (recoil, resp. corrected)", "Events", rebin=bins_MET, yRatio=1.15)
    singlePlot({"name": "MET_corr_rec_resp_pt_qTrw", "axis": "recoil_MET_pt" }, "MET_corr_rec_resp_pt_qTrw", 0, 150, 10, 1e8, "MET p_{T} (recoil, resp. corrected, q_{T} rw)", "Events", rebin=bins_MET, yRatio=1.15)

    singlePlot({"name": "recoil_corr_rec_resp_para", "axis": "recoil_para" }, "recoil_corr_rec_resp_para", -150, 150, 1e0, 1e8, "U_{#parallel} (GeV) (recoil, resp. corrected)", "Events", rebin=bins_recoil_para_perp, yRatio=1.15)
    singlePlot({"name": "recoil_corr_rec_resp_para_qTrw", "axis": "recoil_para_qT" }, "recoil_corr_rec_resp_para_qTrw", -150, 150, 1e0, 1e8, "U_{#parallel} (GeV) (recoil, resp. corrected, q_{T} rw)", "Events", rebin=bins_recoil_para_perp, yRatio=1.15)

def lowPU_W():

    doYields_W("mT_corr_rec")
    yRatio = 1.15

    # lepton
    singlePlot({"name": "lep_pt", "axis": "", "charge": charge }, "lep_pt", 25, 100, 1e1, 1e6, "Lepton p_{T} (GeV)", "Events", rebin=1, yRatio=yRatio)
    singlePlot({"name": "lep_eta", "axis": "", "charge": charge }, "lep_eta", -2.5, 2.5, 1e4, 1e7, "Lepton #eta (GeV)", "Events", rebin=1, yRatio=yRatio)
    singlePlot({"name": "lep_phi", "axis": "", "charge": charge }, "lep_phi", -4, 4, 1e4, 1e7, "Lepton #phi (GeV)", "Events", rebin=1, yRatio=yRatio)

    # mT
    bins_mT = list(range(40, 110, 1)) + [110, 112, 114, 116, 118, 120, 125, 130, 140, 160, 180, 200]
    singlePlot({"name": "mT_corr_rec", "axis": "mt", "charge": charge }, "mT_corr_rec", 40, 200, 1e0, 1e7, "m_{T} (GeV) (recoil corrected)", "Events", rebin=bins_mT, yRatio=yRatio)
    singlePlot({"name": "mT_corr_xy", "axis": "mt", "charge": charge }, "mT_corr_xy", 40, 200, 1e0, 1e7, "m_{T} (GeV) (XY corrected)", "Events", rebin=bins_mT, yRatio=yRatio)
    singlePlot({"name": "mT_corr_rec_qTrw", "axis": "mt", "charge": charge }, "mT_corr_rec_qTrw", 40, 200, 1e0, 1e7, "m_{T} (GeV) (recoil corrected, q_{T} rw)", "Events", rebin=bins_mT, yRatio=yRatio)

    # MET
    bins_MET = list(range(0, 10, 2)) + list(range(10, 70, 1)) + list(range(70, 120, 2)) + list(range(120, 150, 5)) + list(range(150, 200, 10))  + [200]
    singlePlot({"name": "MET_corr_rec_pt", "axis": "MET_pt", "charge": charge }, "MET_corr_rec_pt", 0, 200, 1e0, 1e7, "MET p_{T} (GeV) (recoil corrected)", "Events", rebin=bins_MET, yRatio=yRatio)
    singlePlot({"name": "MET_corr_xy_pt", "axis": "MET_pt", "charge": charge }, "MET_corr_xy_pt", 0, 200, 1e0, 1e7, "MET p_{T} (GeV) (XY corrected)", "Events", rebin=bins_MET, yRatio=yRatio)
    singlePlot({"name": "MET_corr_rec_pt_qTrw", "axis": "MET_pt", "charge": charge }, "MET_corr_rec_pt_qTrw", 0, 200, 1e0, 1e7, "MET p_{T} (GeV) (recoil corrected, q_{T} rw)", "Events", rebin=bins_MET, yRatio=yRatio)

    singlePlot({"name": "MET_corr_lep_phi", "axis": "recoil_MET_phi", "charge": charge }, "MET_corr_lep_phi", -4, 4, 1e4, 1e7, "MET #phi (lepton corrected)", "Events", rebin=1, yRatio=yRatio)
    singlePlot({"name": "MET_corr_xy_phi", "axis": "recoil_MET_phi", "charge": charge }, "MET_corr_xy_phi", -4, 4, 1e4, 1e7, "MET #phi (XY corrected)", "Events", rebin=1, yRatio=yRatio)
    singlePlot({"name": "MET_corr_rec_phi", "axis": "recoil_MET_phi", "charge": charge }, "MET_corr_rec_phi", -4, 4, 1e4, 1e7, "MET #phi (recoil corrected)", "Events", rebin=1, yRatio=yRatio)


    # recoil
    bins_recoil_magn = list(range(0, 100, 2)) + list(range(100, 150, 5)) + [150, 160, 170, 180, 190, 200]
    singlePlot({"name": "recoil_corr_rec_magn", "axis": "recoil_magn", "charge": charge  }, "recoil_corr_rec_magn", 0, 200, 1e0, 1e7, "Recoil |U| (GeV)", "Events", rebin=bins_recoil_magn, dataNorm=True, yRatio=yRatio) # blind!
    singlePlot({"name": "recoil_corr_rec_magn_qTrw", "axis": "recoil_magn", "charge": charge  }, "recoil_corr_rec_magn_qTrw", 0, 200, 1e0, 1e7, "Recoil |U| (GeV)", "Events", rebin=bins_recoil_magn, yRatio=yRatio)

def highPU_W():

    doYields_W("mT_corr_lep")
    #quit()
    yRatio = 1.15
    # mT
    bins_mT = list(range(40, 110, 1)) + [110, 112, 114, 116, 118, 120, 125, 130, 140, 160, 180, 200]
    singlePlot({"name": "mT_corr_lep", "axis": "mt", "charge": charge }, "mT_corr_lep", 40, 200, 1e2, 1e9, "m_{T} (GeV) (lepton corrected)", "Events", rebin=bins_mT, yRatio=yRatio)
    singlePlot({"name": "mT_corr_rec", "axis": "mt", "charge": charge }, "mT_corr_rec", 40, 200, 1e2, 1e9, "m_{T} (GeV) (recoil corrected)", "Events", rebin=bins_mT, yRatio=1.06)
    singlePlot({"name": "mT_corr_xy", "axis": "mt", "charge": charge }, "mT_corr_xy", 40, 200, 1e2, 1e9, "m_{T} (GeV) (XY corrected)", "Events", rebin=bins_mT, yRatio=yRatio)
    singlePlot({"name": "mT_corr_xy", "axis": "mt", "charge": charge }, "mT_corr_xy_noLogY", 40, 140, 0, 5e6, "m_{T} (GeV) (XY corrected)", "Events", rebin=bins_mT, yRatio=yRatio, logY=False)
    singlePlot({"name": "mT_corr_rec_qTrw", "axis": "mt", "charge": charge }, "mT_corr_rec_qTrw", 40, 200, 1e2, 1e9, "m_{T} (GeV) (recoil corrected, q_{T} rw)", "Events", rebin=bins_mT, yRatio=1.06)
    singlePlot({"name": "mT_corr_rec", "axis": "mt", "charge": charge }, "mT_corr_rec_noLogY", 40, 140, 0, 5e6, "m_{T} (GeV) (recoil corrected)", "Events", rebin=bins_mT, yRatio=1.06, logY=False)

    # MET
    bins_MET = list(range(0, 10, 2)) + list(range(10, 70, 1)) + list(range(70, 120, 2)) + list(range(120, 150, 5)) + list(range(150, 200, 10))  + [200]
    singlePlot({"name": "MET_corr_lep_pt", "axis": "MET_pt", "charge": charge }, "MET_corr_lep_pt", 0, 200, 1e2, 1e9, "MET p_{T} (GeV) (lepton corrected)", "Events", rebin=bins_MET, yRatio=yRatio)
    singlePlot({"name": "MET_corr_rec_pt", "axis": "MET_pt", "charge": charge }, "MET_corr_rec_pt", 0, 200, 1e2, 1e9, "MET p_{T} (GeV) (recoil corrected)", "Events", rebin=bins_MET, yRatio=yRatio)
    singlePlot({"name": "MET_corr_xy_pt", "axis": "MET_pt", "charge": charge }, "MET_corr_xy_pt", 0, 200, 1e2, 1e9, "MET p_{T} (GeV) (XY corrected)", "Events", rebin=bins_MET, yRatio=yRatio)
    singlePlot({"name": "MET_corr_rec_pt_qTrw", "axis": "MET_pt", "charge": charge }, "MET_corr_rec_pt_qTrw", 0, 200, 1e2, 1e9, "MET p_{T} (GeV) (recoil corrected, q_{T} rw)", "Events", rebin=bins_MET, yRatio=yRatio)


    singlePlot({"name": "MET_corr_lep_phi", "axis": "recoil_MET_phi", "charge": charge }, "MET_corr_lep_phi", -4, 4, 1e5, 1e9, "MET #phi (lepton corrected)", "Events", rebin=1, yRatio=yRatio)
    singlePlot({"name": "MET_corr_xy_phi", "axis": "recoil_MET_phi", "charge": charge }, "MET_corr_xy_phi", -4, 4, 1e5, 1e9, "MET #phi (XY corrected)", "Events", rebin=1, yRatio=yRatio)
    singlePlot({"name": "MET_corr_rec_phi", "axis": "recoil_MET_phi", "charge": charge }, "MET_corr_rec_phi", -4, 4, 1e5, 1e9, "MET #phi (recoil corrected)", "Events", rebin=1, yRatio=yRatio)


    # recoil
    bins_recoil_magn = list(range(0, 100, 2)) + list(range(100, 150, 5)) + [150, 160, 170, 180, 190, 200]
    singlePlot({"name": "recoil_corr_rec_magn", "axis": "recoil_magn", "charge": charge  }, "recoil_corr_rec_magn", 0, 200, 1e2, 1e9, "Recoil |U| (GeV)", "Events", rebin=bins_recoil_magn, dataNorm=True, yRatio=yRatio) # blind!
    singlePlot({"name": "recoil_corr_rec_magn_qTrw", "axis": "recoil_magn", "charge": charge  }, "recoil_corr_rec_magn_qTrw", 0, 200, 1e2, 1e9, "Recoil |U| (GeV)", "Events", rebin=bins_recoil_magn, yRatio=yRatio)


if __name__ == "__main__":

    print("Start")
    flavor = "mumu"
    met = "DeepMETReso" # DeepMETReso RawPFMET
    charge = "combined" # combined plus minus
    lowPU = False
    doSyst = True

    suffix = "scetlib_dyturbo"
    suffix = ""
    #suffix = "_scetlibCorr"

    if lowPU:

        lumi_header = "199 pb^{#minus1} (13 TeV)"
        groups = datagroupsLowPU(f"lowPU_{flavor}_{met}.hdf5", flavor=flavor)

        if flavor == "mumu":
            procs, data = ['EWK', 'Top', 'Zmumu'], 'SingleMuon'
            outDir = f"/eos/user/j/jaeyserm/www/wmass/lowPU/Z{flavor}_{met}/plots{suffix}/"
            dataNormProc = 'Zmumu'
            functions.prepareDir(outDir, remove=True)
            lowPU_Z()
        elif flavor == "mu":
            procs, data = ['EWK', 'Top', 'Fake', 'WJetsToMuNu'], 'SingleMuon'
            outDir = f"/eos/user/j/jaeyserm/www/wmass/lowPU/W{flavor}_{met}/plots_{charge}{suffix}/"
            dataNormProc = 'WJetsToMuNu'
            #dataNormProc = ''
            label = "W^{#%s}, %s" % (charge if charge != "combined" else "pm", met)
            functions.prepareDir(outDir, remove=True)
            lowPU_W()
        elif flavor == "ee":
            procs, data = ['EWK', 'TTbar', 'DYee'], 'SingleElectron'



    else:
        lumi_header = "16.8 fb^{#minus1} (13 TeV)"

        def fakeHistABCD(h):

            axes = [ax.name for ax in h.axes]
            if "mt" in axes:
                s = hist.tag.Slicer()
                sf = h[{"passIso" : True, "passMT" : False}].sum().value / h[{"passIso" : False, "passMT" : False}].sum().value
                return h[{"passIso" : False, "passMT" : True}]*sf

            ret = hh.multiplyHists(
                hh.divideHists(h[{"passIso" : True, "passMT" : False}],
                    h[{"passIso" : False, "passMT" : False}],
                        cutoff=1
                    ),
                        #where=h[{"passIso" : False, "passMT" : True}].values(flow=True)>1),
                h[{"passIso" : False, "passMT" : True}],
            )
            return ret


        

        if flavor == "mumu":
            groups = datagroups.Datagroups(f"mz_wlike_with_mu_eta_pt_{met}.hdf5")
            #groups = datagroups.Datagroups(f"mz_wlike_with_mu_eta_pt_scetlib_dyturboCorr.hdf5")
            
            groups.groups['Zmumu'].color = "#F8CE68"
            groups.groups['Zmumu'].label = "DY #rightarrow #mu^{#plus}#mu^{#minus} (MiNNLO)"
            
            groups.addGroup("Top",
                members = list(filter(lambda y: y.group == "Top", groups.datasets.values())),
                label = "Top",
                color = "#DE5A6A",
                selectOp = sel.signalHistWmass if flavor == "mu" else None,
            )
        else:
            groups = datagroups.Datagroups(f"mw_with_mu_eta_pt_{met}.hdf5")
            #groups = datagroups.Datagroups(f"mw_with_mu_eta_pt_scetlib_dyturboCorr.hdf5")
            
            groups.groups['Fake'].selectOp = fakeHistABCD
            groups.groups['Fake'].color = "#A9A9A9"
            groups.groups['Wmunu'].color = "#F8CE68"
            groups.groups['Wmunu'].label = "W^{#pm} #rightarrow #mu^{#pm}#nu"
            groups.groups['Top'].color = "#DE5A6A"


        groups.addGroup("EWK_Z",
            members = [x for x in groups.datasets.values() if not x.is_data and x.group not in ["Zmumu", "Top"] and x.group != "QCD"],
            label = r"EWK (#tau^{#plus}#tau^{#minus}, VV)",
            color = "#64C0E8",
            selectOp = sel.signalHistWmass if flavor == "mu" else None,
        )
        
        groups.addGroup("EWK_W",
            members = [x for x in groups.datasets.values() if not x.is_data and x.group not in ["Wmunu", "Top"] and x.group != "QCD"],
            label = r"EWK (#tau^{#plus}#tau^{#minus}, VV)",
            color = "#64C0E8",
            selectOp = sel.signalHistWmass if flavor == "mu" else None,
        )

        groups.addGroup("Topa",
            members = list(filter(lambda y: y.group == "Top", groups.datasets.values())),
            label = "Top",
            color = "#DE5A6A",
            selectOp = sel.signalHistWmass if flavor == "mu" else None,
        )

        if flavor == "mumu":
            procs, data = ['EWK_Z', 'Top', 'Zmumu'], 'Data'
            #procs, data = ['Ztautau', 'Other', 'Zmumu'], 'Data'
            #outDir = f"/eos/user/j/jaeyserm/www/wmass/highPU/Z{flavor}_{met}/plots{suffix}/"
            outDir = f"/home/submit/jaeyserm/public_html/wmass/highPU/Z{flavor}_{met}/plots{suffix}/"
            dataNormProc = 'Zmumu'
            functions.prepareDir(outDir, remove=True)
            highPU_Z()

        if flavor == "mu":
            procs, data = ['EWK_W', 'Top', 'Fake', 'Wmunu'], 'Data'
            outDir = f"/home/submit/jaeyserm/public_html/wmass/highPU/W{flavor}_{met}/plots{suffix}/"
            dataNormProc = 'Wmunu'
            #dataNormProc = ''
            functions.prepareDir(outDir, remove=True)
            highPU_W()



