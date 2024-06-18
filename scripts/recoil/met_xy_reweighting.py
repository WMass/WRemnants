
import sys,argparse

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptTitle(0)

import functions
import plotter
import narf

from wremnants.datasets import datagroups

def makePlot(hist_data, hist_mc, fOut, xLabel, npv, outDir_):

    hist_data.Scale(1./hist_data.Integral())
    hist_mc.Scale(1./hist_mc.Integral())

    hist_data.SetLineColor(ROOT.kBlack)
    hist_data.SetLineWidth(2)

    hist_mc.SetLineColor(ROOT.kRed)
    hist_mc.SetLineWidth(2)

    ## sigmas
    cfg = {

        'logy'              : True,
        'logx'              : False,

        'xmin'              : -100,
        'xmax'              : 100,
        'ymin'              : 1e-5,
        'ymax'              : 1e0,

        'xtitle'            : xLabel,
        'ytitle'            : "Events",

        'topRight'          : lumi_header, 
        'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",

    } 


    plotter.cfg = cfg
    canvas = plotter.canvas()
    dummy = plotter.dummy()   
    canvas.cd()
    dummy.Draw("HIST")
    hist_data.Draw("SAME")
    hist_mc.Draw("SAME")
    plotter.aux()

    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.030)
    latex.SetTextColor(1)
    latex.SetTextFont(42)

    latex.DrawLatex(0.20, 0.90, "NPV = %d" % npv)

    latex.DrawLatex(0.20, 0.85, "Data")
    latex.DrawLatex(0.20, 0.82, "Mean %.2f #pm %.2f GeV" % (hist_data.GetMean(), hist_data.GetMeanError()))

    latex.DrawLatex(0.60, 0.85, "#color[2]{MC}")
    latex.DrawLatex(0.60, 0.82, "#color[2]{Mean %.2f #pm %.2f GeV}" % (hist_mc.GetMean(), hist_mc.GetMeanError()))

    canvas.Modify()
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs("%s/%s.png" % (outDir_, fOut))
    canvas.SaveAs("%s/%s.pdf" % (outDir_, fOut))
    canvas.Delete()


def makePlot_fit(hist_data, hist_mc, fOut, xLabel, npv, outDir_):

    hist_data.Scale(1./hist_data.Integral())
    hist_mc.Scale(1./hist_mc.Integral())

    hist_data.SetLineColor(ROOT.kBlack)
    hist_data.SetLineWidth(2)
    
    hist_mc.SetLineColor(ROOT.kRed)
    hist_mc.SetLineWidth(2)

    ## sigmas
    cfg = {

        'logy'              : True,
        'logx'              : False,

        'xmin'              : -100,
        'xmax'              : 100,
        'ymin'              : 1e-5,
        'ymax'              : 1e0,

        'xtitle'            : xLabel,
        'ytitle'            : "Events",

        'topRight'          : lumi_header, 
        'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",

    } 

    fit_mc = ROOT.TF1("fit_mc", "gaus", -100, 100)
    hist_mc.Fit("fit_mc")

    fit_data = ROOT.TF1("fit_data", "gaus", -100, 100)
    hist_data.Fit("fit_data")

    plotter.cfg = cfg
    canvas = plotter.canvas()
    dummy = plotter.dummy()   
    canvas.cd()
    dummy.Draw("HIST")
    hist_data.Draw("SAME")
    hist_mc.Draw("SAME")

    fit_mc.Draw("SAME")
    fit_data.Draw("SAME")
    plotter.aux()

    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.030)
    latex.SetTextColor(1)
    latex.SetTextFont(42)

    latex.DrawLatex(0.20, 0.90, "NPV = %d" % npv)

    latex.DrawLatex(0.20, 0.85, "Data")
    latex.DrawLatex(0.20, 0.82, "Mean %.2f #pm %.2f GeV" % (hist_data.GetMean(), hist_data.GetMeanError()))
    latex.DrawLatex(0.20, 0.79, "#color[2]{Mean %.2f #pm %.2f GeV}" % (fit_data.GetParameter(1), fit_data.GetParError(1)))

    latex.DrawLatex(0.60, 0.85, "#color[2]{MC}")
    latex.DrawLatex(0.60, 0.82, "#color[2]{Mean %.2f #pm %.2f GeV}" % (hist_mc.GetMean(), hist_mc.GetMeanError()))
    latex.DrawLatex(0.60, 0.79, "#color[2]{Mean %.2f #pm %.2f GeV}" % (fit_mc.GetParameter(1), fit_mc.GetParError(1)))

    canvas.Modify()
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs("%s/%s.png" % (outDir_, fOut))
    canvas.SaveAs("%s/%s.pdf" % (outDir_, fOut))
    canvas.Delete()

    return fit_data.GetParameter(1), fit_data.GetParError(1), fit_mc.GetParameter(1), fit_mc.GetParError(1)

def met_xy_reweighting(direction = "x", corrType="uncorr", polyOrderData=-1, polyOrderMC=-1, procs=[], data="", yMin=-5, yMax=5):

    # corrType uncorr or corr_xy
    outDir_ = "%s/%s" % (outDir, corrType)
    functions.prepareDir(outDir_, False)

    h_data = functions.readBoostHist(groups, f"met_{corrType}_{direction}_npv", [data], abcd=False)
    h_mc = functions.readBoostHist(groups, f"met_{corrType}_{direction}_npv", procs, abcd=False)

    g_data = ROOT.TGraphErrors()
    g_data.SetLineColor(ROOT.kBlack)
    g_data.SetMarkerStyle(20)
    g_data.SetMarkerSize(1)
    g_data.SetMarkerColor(ROOT.kBlack)

    g_mc = ROOT.TGraphErrors()
    g_mc.SetLineColor(ROOT.kRed)
    g_mc.SetMarkerStyle(21)
    g_mc.SetMarkerSize(1)
    g_mc.SetMarkerColor(ROOT.kRed)

    outDict = {"data": { 'nom': [] }, "mc": { 'nom': [] }}
    for iBin in range(1, npv_max):

        hist_data = h_data.ProjectionY("data_bin%d" % iBin, iBin, iBin)
        hist_mc = h_mc.ProjectionY("mc_bin%d" % iBin, iBin, iBin)

        mean_data, mean_data_err = hist_data.GetMean(), hist_data.GetMeanError()
        mean_mc, mean_mc_err = hist_mc.GetMean(), hist_mc.GetMeanError()
        print(iBin, mean_mc, mean_mc_err)

        makePlot(hist_data, hist_mc, "npv_%d_%d_MET%s" % (iBin, iBin, direction), "MET %s (GeV)" % direction, iBin, outDir_)
        #mean_data, mean_data_err, mean_mc, mean_mc_err = makePlot_fit(hist_data, hist_mc, "npv_%d_%d_MET%s" % (iBin, iBin, direction), "MET %s (GeV)" % direction, iBin, outDir_)

        g_data.SetPoint(iBin-1, iBin, mean_data)
        g_data.SetPointError(iBin-1, 0, mean_data_err)
        g_mc.SetPoint(iBin-1, iBin, mean_mc)
        g_mc.SetPointError(iBin-1, 0, mean_mc_err)

    if polyOrderData > 0:
        fit_data = ROOT.TF1("fit_data_%s" % direction, "pol%d" % polyOrderData, 0, npv_max)
        result = g_data.Fit(fit_data.GetName(), "NSE", "", npv_fit_min, npv_fit_max) 
        fit_data.SetLineColor(ROOT.kBlack)
        fit_data.GetXaxis().SetRangeUser(0, npv_max)
        fit_data.SetLineWidth(2)
        for iP in range(0, polyOrderData+1): outDict['data']['nom'].append(fit_data.GetParameter(iP))

    if polyOrderMC > 0:
        fit_mc = ROOT.TF1("fit_mc_%s" % direction, "pol%d" % polyOrderMC, 0, npv_max)
        result = g_mc.Fit(fit_mc.GetName(), "NSE", "", npv_fit_min, npv_fit_max) 
        fit_mc.SetLineColor(ROOT.kRed)
        fit_mc.SetLineStyle(ROOT.kDashed)
        fit_mc.GetXaxis().SetRangeUser(0, npv_max)
        fit_mc.SetLineWidth(2)
        for iP in range(0, polyOrderMC+1): outDict['mc']['nom'].append(fit_mc.GetParameter(iP))

    ## sigmas
    cfg = {

        'logy'              : False,
        'logx'              : False,

        'xmin'              : 0,
        'xmax'              : npv_max,
        'ymin'              : yMin,
        'ymax'              : yMax,

        'xtitle'            : "Number of primary vertices",
        'ytitle'            : "#LT MET_{%s} #GT (Gev)" % direction,

        'topRight'          : lumi_header, 
        'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",

    } 

    plotter.cfg = cfg
    canvas = plotter.canvas()
    dummy = plotter.dummy()   
    canvas.cd()
    dummy.Draw("HIST")
    g_data.Draw("PE SAME")
    if polyOrderData > 0: fit_data.Draw("L SAME")
    g_mc.Draw("PE SAME")
    if polyOrderMC > 0: fit_mc.Draw("L SAME")
    plotter.aux()

    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.035)
    latex.SetTextColor(1)
    latex.SetTextFont(42)

    if flavor == "mumu": label = "Z #rightarrow #mu^{#plus}#mu^{#minus}"
    elif flavor == "ee": label = "Z #rightarrow e^{#plus}e^{#minus}"
    elif flavor == "mu": label = "W^{#pm} #rightarrow #mu^{#pm}#nu"
    elif flavor == "e": label = "W^{#pm} #rightarrow e^{#pm}#nu"
    else: label = ""
    latex.DrawLatex(0.20, 0.90, label)
    latex.DrawLatex(0.20, 0.85, "#LT MET_{%s} #GT, %s" % (direction, met))

    leg = ROOT.TLegend(.2, 0.7, .5, .80)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.035)
    leg.AddEntry(g_data, "Data (order=%d)"%polyOrderData, "LP")
    leg.AddEntry(g_mc, "MC (order=%d)"%polyOrderMC, "LP")
    leg.Draw("SAME")

    line = ROOT.TLine(0, 0, npv_max, 0)
    line.SetLineColor(ROOT.kBlue+2)
    line.SetLineWidth(2)
    line.Draw("SAME")

    canvas.Modify()
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs(f"{outDir_}/MET_{direction}_npv.png")
    canvas.SaveAs(f"{outDir_}/MET_{direction}_npv.pdf")
    canvas.Delete()
    
    return outDict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="Input hdf5 file")
    parser.add_argument("-s", "--save", action='store_true', help="Save to wremnants-data")
    args = parser.parse_args()

    groups = datagroups.Datagroups(args.input)
    met, analysis, flavor = functions.get_meta(groups)
    fOut = f"wremnants-data/data/recoil/{analysis}_{met}/met_xy_{flavor}.json"
    outDir = f"/home/submit/jaeyserm/public_html/recoil/{analysis}_{met}/met_xy_{flavor}/"
    functions.prepareDir(outDir, True)
    lumi_header = functions.getLumiLabel(groups)

    if analysis == "lowPU":
        if flavor == "mumu":
            procs, data = ["Zmumu", "Ztautau", "Other"], "Data"

            npv_max, npv_fit_min, npv_fit_max = 10, 0, 10
            yMin_x, yMax_x, yMin_y, yMax_y = -5, 5, -5, 5
            polyOrderData_x, polyOrderData_y = 2, 2
            polyOrderMC_x, polyOrderMC_y = 2, 2
        else:
            procs, data = ["Zmumu", "Ztautau", "Wtaunu", "Wmunu", "Top", "Diboson"], "Data"

            npv_max, npv_fit_min, npv_fit_max = 10, 0, 10
            yMin_x, yMax_x, yMin_y, yMax_y = -5, 5, -5, 5
            polyOrderData_x, polyOrderData_y = 3, 2
            polyOrderMC_x, polyOrderMC_y = 3, 2

            for g in groups.groups:
                groups.groups[g].selectOp = None

    else:
        npv_max, npv_fit_min, npv_fit_max = 60, 5, 55

        if flavor == "mumu":
            procs, data = ["Zmumu", "Ztautau", "Other"], "Data"

            npv_max, npv_fit_min, npv_fit_max = 60, 5, 55
            yMin_x, yMax_x, yMin_y, yMax_y = -10, 6, -5, 5
            polyOrderData_x, polyOrderData_y = 3, 3
            polyOrderMC_x, polyOrderMC_y = 3, 3

        else:
            procs, data = ["Zmumu", "Ztautau", "Wtaunu", "Wmunu", "Top", "Diboson"], "Data"

            npv_max, npv_fit_min, npv_fit_max = 60, 0, 55
            yMin_x, yMax_x, yMin_y, yMax_y = -10, 6, -5, 5
            polyOrderData_x, polyOrderData_y = 4, 6
            polyOrderMC_x, polyOrderMC_y = 3, 4

            for g in groups.groups:
                groups.groups[g].selectOp = None


    outDict = {}
    dictX = met_xy_reweighting(direction="x", corrType="corr_lep", polyOrderData=polyOrderData_x, polyOrderMC=polyOrderMC_x, procs=procs, data=data, yMin=yMin_x, yMax=yMax_x)
    dictY = met_xy_reweighting(direction="y", corrType="corr_lep", polyOrderData=polyOrderData_y, polyOrderMC=polyOrderMC_y, procs=procs, data=data, yMin=yMin_y, yMax=yMax_y)

    if args.save:
        outDict['x'] = dictX
        outDict['y'] = dictY
        functions.writeJSON(fOut, outDict)

    met_xy_reweighting(direction="x", corrType="corr_xy", polyOrderData=polyOrderData_x, polyOrderMC=polyOrderMC_x, procs=procs, data=data, yMin=yMin_x, yMax=yMax_x)
    met_xy_reweighting(direction="y", corrType="corr_xy", polyOrderData=polyOrderData_y, polyOrderMC=polyOrderMC_y, procs=procs, data=data, yMin=yMin_y, yMax=yMax_y)
    print(fOut)
