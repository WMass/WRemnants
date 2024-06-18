
import ROOT
import functions
import plotter
import hist as bh
import narf
import math

def stacked_plot_ratio(groups, hName, procs, outDir, suffix="", xMin=0, xMax=100, yMin=0, yMax=-1.75, xLabel="xLabel", yLabel="Events", logX=False, logY=False, rebin=1, legPos=[], yRatio=1.15, blind=False, dataNormProc="", labels=[], charge=None, extraRatios=[]):

    functions.prepareDir(outDir, remove=False)

    leg = ROOT.TLegend(.50, 0.88-(len(procs)+2)*0.05, .8, .88)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.040)

    h_data = functions.readBoostHist(groups, hName, [procs[0]], charge=charge)
    h_data = functions.rebin(h_data, rebin)
    print("Data", h_data.Integral())
    leg.AddEntry(h_data, groups.groups[procs[0]].label, "PE")

    st = ROOT.THStack()
    st.SetName("stack")
    h_bkg = h_data.Clone("bkg_nominal")
    h_bkg.Reset("ACE")
    bkg_hists = {}
    normMC = 0
    for i,proc in enumerate(procs[1:]):
        hist = functions.readBoostHist(groups, hName, [proc], charge=charge)
        hist = functions.rebin(hist, rebin)
        hist.SetFillColor(ROOT.TColor.GetColor(groups.groups[proc].color))
        hist.SetLineColor(ROOT.kBlack)
        hist.SetLineWidth(1)
        hist.SetLineStyle(1)
        bkg_hists[proc] = hist
        if proc != dataNormProc:
            normMC += hist.Integral()
        print(proc, hist.Integral())

    if dataNormProc != "":
        bkg_hists[dataNormProc].Scale((h_data.Integral()-normMC)/bkg_hists[dataNormProc].Integral())
    for i,proc in enumerate(procs[1:]):
        leg.AddEntry(bkg_hists[proc], groups.groups[proc].label if proc != "WJetsToMuNu" else "W^{{#{q}}} #rightarrow #mu^{{#{q}}}#nu".format(q=charge if charge != "combined" else "pm"), "F")
        st.Add(bkg_hists[proc])
        h_bkg.Add(bkg_hists[proc])

    hists_extraRatios = []
    colors = [ROOT.kRed, ROOT.kBlue, ROOT.kGreen+1, ROOT.kOrange]
    for ig,gr_er in enumerate(extraRatios):
        n = gr_er[0]
        h_tot = h_data.Clone(f"h_tot_{n}")
        h_tot.Reset("ACE")
        for i,proc in enumerate(procs[1:]):
            hist = functions.readBoostHist(gr_er[1], hName, [proc], charge=charge)
            hist = functions.rebin(hist, rebin)
            if proc == dataNormProc:
                hist.Scale((h_data.Integral()-normMC)/hist.Integral())
            h_tot.Add(hist)

        h_ratio = h_data.Clone(f"h_extraratio_{n}")
        for i in range(h_tot .GetNbinsX()+1): h_tot.SetBinError(i, 0) # set MC errors to zero
        h_ratio.Divide(h_tot)
        h_ratio.SetMarkerStyle(20)
        h_ratio.SetMarkerSize(0.5)
        h_ratio.SetMarkerColor(colors[ig])
        h_ratio.SetLineColor(colors[ig])
        h_ratio.SetLineWidth(1)
        hists_extraRatios.append(h_ratio)

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
    leg.AddEntry(h_err, "Stat. Unc.", "F")


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


    cfg = {

        'logy'              : logY,
        'logx'              : logX,

        'xmin'              : xMin,
        'xmax'              : xMax,
        'ymin'              : yMin,
        'ymax'              : yMax if yMax > 0 else -yMax*h_data.GetMaximum(),

        'xtitle'            : xLabel,
        'ytitle'            : yLabel,

        'topRight'          : functions.getLumiLabel(groups),
        'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",

        'ratiofraction'     : 0.3,
        'ytitleR'           : "Data/MC",

        'yminR'             : 1-(yRatio-1),
        'ymaxR'             : yRatio,
    }

    if blind:
        for i in range(0, h_data.GetNbinsX()+1):
            h_data.SetBinContent(i, h_bkg.GetBinContent(i))
            h_ratio.SetBinContent(i, 1)

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

    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.040)
    latex.SetTextColor(1)
    latex.SetTextFont(42)
    for il, label in enumerate(labels):
        latex.DrawLatex(0.20, 0.85-il*0.05, label)

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

    if len(hists_extraRatios) > 0:
        leg_extraRatios = ROOT.TLegend(.35, 0.83, 0.35+0.175*len(hists_extraRatios), .93)
        leg_extraRatios.SetBorderSize(0)
        leg_extraRatios.SetFillStyle(0)
        leg_extraRatios.SetTextSize(0.1)
        leg_extraRatios.SetNColumns(len(hists_extraRatios))

    for ig,h in enumerate(hists_extraRatios):
        leg_extraRatios.AddEntry(h, extraRatios[ig][0], "PE")
        h.Draw("PE0 SAME")
    h_ratio.Draw("PE0 SAME") # E2 SAME
    h_err_ratio.Draw("E2 SAME")

    if len(hists_extraRatios) > 0:
        leg_extraRatios.Draw("SAME")

    ROOT.gPad.SetTickx()
    ROOT.gPad.SetTicky()
    ROOT.gPad.RedrawAxis()

    canvas.SaveAs(f"{outDir}/{hName}{suffix}.png")
    canvas.SaveAs(f"{outDir}/{hName}{suffix}.pdf")
    canvas.Close()





def plot_ratio(groups, histNames, procs, labels, fOut, outDir, suffix="", xMin=0, xMax=100, yMin=0, yMax=100, xLabel="xLabel", yLabel="Events", logX=False, logY=False, rebin=1, legPos=[], yRatio=1.15, extralabels=[], lumi_label="", doRatio=True, ytitleR="Ratio", norm=False):

    colors = [ROOT.kBlack, ROOT.kBlue, ROOT.kRed, ROOT.kGreen+2, ROOT.kOrange]
    functions.prepareDir(outDir, remove=False)

    leg = ROOT.TLegend(.60, 0.88-(len(procs)+2)*0.05, .9, .88)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.040)

    hists, ratios = [], []
    yMaxHists = -1e99
    for i, group in enumerate(groups):
        hist = functions.readBoostHist(group, histNames[i], [procs[i]])
        hist = functions.rebin(hist, rebin)
        if norm:
            hist.Scale(1./hist.Integral())
        hist.SetLineColor(colors[i])
        hist.SetLineWidth(2)
        hist.SetLineStyle(1)
        leg.AddEntry(hist, labels[i], "L")
        hists.append(hist)
        if i==0:
            ratios.append(hist)
        h_ratio = hist.Clone(hist.GetName() + "_ratio")
        h_ratio.Divide(ratios[0])
        ratios.append(h_ratio)
        if hist.GetMaximum() > yMaxHists:
            yMaxHists = hist.GetMaximum()

    cfg = {

        'logy'              : logY,
        'logx'              : logX,

        'xmin'              : xMin,
        'xmax'              : xMax,
        'ymin'              : yMin,
        'ymax'              : yMax if yMax != -1 else 1.2*yMaxHists,

        'xtitle'            : xLabel,
        'ytitle'            : yLabel,

        'topRight'          : lumi_label,
        'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",

        'ratiofraction'     : 0.3,
        'ytitleR'           : ytitleR,

        'yminR'             : 1-(yRatio-1),
        'ymaxR'             : yRatio,
    }

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
    for hist in hists:
        hist.Draw("SAME HIST")
        print(hist.GetName(), hist.GetMean(), hist.GetRMS())
    leg.Draw("SAME")
    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.040)
    latex.SetTextColor(1)
    latex.SetTextFont(42)
    for il, label in enumerate(extralabels):
        latex.DrawLatex(0.20, 0.85-il*0.05, label)

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
    if doRatio:
        for hist in ratios:
            hist.Draw("SAME HIST")

    ROOT.gPad.SetTickx()
    ROOT.gPad.SetTicky()
    ROOT.gPad.RedrawAxis()

    canvas.SaveAs(f"{outDir}/{fOut}{suffix}.png")
    canvas.SaveAs(f"{outDir}/{fOut}{suffix}.pdf")
    canvas.Close()



def stacked_2d(groups, hName, proc, outDir, suffix="", xMin=0, xMax=100, yMin=0, yMax=100, xLabel="xLabel", yLabel="yLabel", logX=False, logY=False, logZ=False):

    functions.prepareDir(outDir, remove=False)
    hist = functions.readBoostHist(groups, hName, [proc])


    cfg = {

        'logy'              : logY,
        'logx'              : logX,

        'xmin'              : xMin,
        'xmax'              : xMax,
        'ymin'              : yMin,
        'ymax'              : yMax,

        'xtitle'            : xLabel,
        'ytitle'            : yLabel,

        'topRight'          : functions.getLumiLabel(groups),
        'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",

    }



    plotter.cfg = cfg
    canvas = plotter.canvas()
    canvas.SetRightMargin(0.12)
    dummy = plotter.dummy()
    canvas.cd()
    if logZ:
        canvas.SetLogz()
    #h_mc.GetZaxis().SetRangeUser(0.8, 1.2)
    #h_data.Draw("SAME COLZ TEXTE")
    dummy.Draw("HIST")
    hist.Draw("SAME COLZ")
    


    canvas.SaveAs(f"{outDir}/{hName}{suffix}.png")
    canvas.SaveAs(f"{outDir}/{hName}{suffix}.pdf")
    canvas.Close()


def plot_systs(groups, histName, systName, proc, outDir, suffix="", xMin=0, xMax=100, yMin=0, yMax=100, xLabel="xLabel", yLabel="Events", logX=False, logY=False, rebin=1, legPos=[], yRatio=1.15, extralabels=[], doRatio=True, plotSysts=False):

    functions.prepareDir(outDir, remove=False)
    hist_nom = functions.readBoostHist(groups, histName, [proc], boost=False)
    hist_nom = functions.rebin(hist_nom, rebin)

    hist_unc = functions.readBoostHist(groups, f"{histName}_{systName}", [proc], boost=True)
    try:
    #    #hist_unc = functions.readBoostHist(groups, f"{histName}_{systName}", [proc], boost=True, abcd=False, applySelection=False)
        hist_unc = functions.readBoostHist(groups, f"{histName}_{systName}", [proc], boost=True)
    except:
        return
    ax_entries = list(hist_unc.axes[1])
    s = bh.tag.Slicer()

    hist_syst_tot = hist_nom.Clone("syst_tot")
    hist_systs = []
    doubleSide = False

    def addUnc(hTarget, hNom, hUp, hDw=None):
        for k in range(1, hTarget.GetNbinsX()+1):
            c_orig = hNom.GetBinContent(k)
            err_orig = abs(hTarget.GetBinContent(k)-hNom.GetBinContent(k))
            if hDw != None: sigma = 0.5*abs(hUp.GetBinContent(k)-hDw.GetBinContent(k))
            else: sigma = abs(hUp.GetBinContent(k)-hNom.GetBinContent(k))
            err_new = math.sqrt(sigma*sigma + err_orig*err_orig)
            hTarget.SetBinContent(k, c_orig + err_new)

    if doubleSide:
        for i in range(int(len(ax_entries)/2)):
            hDw = hist_unc[{"recoil_unc" : 2*(i-1)-1}]
            hUp = hist_unc[{"recoil_unc" : 2*i-1}]
            hDw = narf.hist_to_root(hDw)
            hUp = narf.hist_to_root(hUp)
            hUp = functions.rebin(hUp, rebin)
            hDw = functions.rebin(hDw, rebin)
            addUnc(h_syst, hNom, hUp, hDw)
    else:
        for i in range(int(len(ax_entries))):
            hUp = hist_unc[{"recoil_unc" : i*1.j}]
            hUp = narf.hist_to_root(hUp)
            hUp.SetName(f"syst_{i}")
            hUp = functions.rebin(hUp, rebin)
            addUnc(hist_syst_tot, hist_nom, hUp)
            hist_systs.append(hUp)
            #if i == 0:
            #    hist_nom = hUp.Clone("hist_nom")
    #hist_systs.append(hist_syst_tot)



    cfg = {

        'logy'              : logY,
        'logx'              : logX,

        'xmin'              : xMin,
        'xmax'              : xMax,
        'ymin'              : yMin,
        'ymax'              : yMax if yMax != -1 else 1.3*hist_nom.GetMaximum(),

        'xtitle'            : xLabel,
        'ytitle'            : yLabel,

        'topRight'          : functions.getLumiLabel(groups),
        'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",

        'ratiofraction'     : 0.3,
        'ytitleR'           : "Data/MC",

        'yminR'             : 1-(yRatio-1),
        'ymaxR'             : yRatio,
    }

    if plotSysts:
        for iSyst,hist_syst in enumerate(hist_systs):
            leg = ROOT.TLegend(.50, 0.88-(3)*0.05, .8, .88)
            leg.SetBorderSize(0)
            leg.SetFillStyle(0)
            leg.SetTextSize(0.040)
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

            hist_nom.SetLineColor(ROOT.kBlack)
            hist_nom.SetLineWidth(2)
            hist_nom.SetLineStyle(1)
            leg.AddEntry(hist_nom, "Nominal", "L")
            hist_nom.Draw("SAME HIST")

            hist_syst.SetLineColor(ROOT.kBlue)
            hist_syst.SetLineWidth(2)
            hist_syst.SetLineStyle(1)
            leg.AddEntry(hist_syst, "Up", "L")
            hist_syst.Draw("SAME HIST")

            h_ratio = hist_syst.Clone(hist_syst.GetName() + "_ratio")
            h_ratio.Divide(hist_nom)

            rMin, rMax = functions.getMinMaxRange(h_ratio, xMin, xMax)
            ratio_range = 1.02*(1. + max(rMax-1, 1-rMin))
            leg.Draw("SAME")

            latex = ROOT.TLatex()
            latex.SetNDC()
            latex.SetTextSize(0.040)
            latex.SetTextColor(1)
            latex.SetTextFont(42)
            for il, label in enumerate(extralabels):
                latex.DrawLatex(0.20, 0.85-il*0.05, label)
            latex.DrawLatex(0.20, 0.85-(il+1)*0.05, f"Total uncertainty" if iSyst == len(hist_systs)-1 else f"Unc idx = {iSyst}")


            plotter.auxRatio()
            ROOT.gPad.SetTickx()
            ROOT.gPad.SetTicky()
            ROOT.gPad.RedrawAxis()


            ## bottom panel
            canvas.cd()
            padB.Draw()
            padB.SetFillStyle(0)
            padB.cd()
            dummyB.GetYaxis().SetRangeUser(2-ratio_range, ratio_range)
            dummyB.Draw("HIST")
            dummyL.Draw("SAME")
            h_ratio.Draw("SAME HIST")
            ROOT.gPad.SetTickx()
            ROOT.gPad.SetTicky()
            ROOT.gPad.RedrawAxis()

            canvas.SaveAs(f"{outDir}/{histName}_{systName}_{iSyst}{suffix}.png")
            canvas.SaveAs(f"{outDir}/{histName}_{systName}_{iSyst}{suffix}.pdf")
            canvas.Close()
            del canvas, padB, padT, dummyB, dummyT, dummyL, leg


    ## envelope
    leg = ROOT.TLegend(.60, 0.88-(3)*0.05, .9, .88)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.040)
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

    hist_nom.SetLineColor(ROOT.kBlack)
    hist_nom.SetLineWidth(2)
    hist_nom.SetLineStyle(1)
    leg.AddEntry(hist_nom, "Nominal", "L")
    hist_nom.Draw("SAME HIST")

    hist_syst_tot_dw = hist_nom.Clone("hist_syst_tot_dw")
    hist_syst_tot_dw.Scale(2)
    hist_syst_tot_dw.Add(hist_syst_tot, -1)
    hist_syst_tot_dw.SetLineColor(ROOT.kBlue)
    hist_syst_tot_dw.SetLineWidth(2)
    hist_syst_tot_dw.SetLineStyle(1)
    #leg.AddEntry(hist_syst_tot_dw, "Down", "L")
    hist_syst_tot_dw.Draw("SAME HIST")

    hist_syst_tot.SetLineColor(ROOT.kBlue)
    hist_syst_tot.SetLineWidth(2)
    hist_syst_tot.SetLineStyle(1)
    leg.AddEntry(hist_syst_tot, "Uncertainty", "L")
    hist_syst_tot.Draw("SAME HIST")

    h_ratio = hist_syst_tot.Clone(hist_syst_tot.GetName() + "_ratio")
    h_ratio.Divide(hist_nom)
    h_ratio_dw = hist_syst_tot_dw.Clone(hist_syst_tot_dw.GetName() + "_ratio_dw")
    h_ratio_dw.Divide(hist_nom)

    rMin, rMax = functions.getMinMaxRange(h_ratio, xMin, xMax)
    ratio_range = 1.02*(1. + max(rMax-1, 1-rMin))
    if ratio_range > 1.1:
        ratio_range = 1.1
    leg.Draw("SAME")

    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.040)
    latex.SetTextColor(1)
    latex.SetTextFont(42)

    for il, label in enumerate(extralabels):
        latex.DrawLatex(0.20, 0.85-il*0.05, label)
    latex.DrawLatex(0.20, 0.85-(il+1)*0.05, f"Total uncertainty")


    plotter.auxRatio()
    ROOT.gPad.SetTickx()
    ROOT.gPad.SetTicky()
    ROOT.gPad.RedrawAxis()


    ## bottom panel
    canvas.cd()
    padB.Draw()
    padB.SetFillStyle(0)
    padB.cd()
    dummyB.GetYaxis().SetRangeUser(2-ratio_range, ratio_range)
    dummyB.Draw("HIST")
    dummyL.Draw("SAME")
    h_ratio.Draw("SAME HIST")
    h_ratio_dw.Draw("SAME HIST")
    ROOT.gPad.SetTickx()
    ROOT.gPad.SetTicky()
    ROOT.gPad.RedrawAxis()

    canvas.SaveAs(f"{outDir}/{histName}_{systName}_envelope{suffix}.png")
    canvas.SaveAs(f"{outDir}/{histName}_{systName}_envelope{suffix}.pdf")
    canvas.Close()
