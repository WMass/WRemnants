#!/usr/bin/env python


import ROOT

centerOfMassEnergy = 13


def setTDRStyle():

    ROOT.gStyle.SetCanvasBorderMode(0)
    ROOT.gStyle.SetCanvasColor(0)
    ROOT.gStyle.SetCanvasDefH(600)
    ROOT.gStyle.SetCanvasDefW(600)
    ROOT.gStyle.SetCanvasDefX(0)
    ROOT.gStyle.SetCanvasDefY(0)

    ROOT.gStyle.SetPadBorderMode(0)
    ROOT.gStyle.SetPadColor(0)
    ROOT.gStyle.SetPadGridX(0)
    ROOT.gStyle.SetPadGridY(0)
    ROOT.gStyle.SetGridColor(0)
    ROOT.gStyle.SetGridStyle(3)
    ROOT.gStyle.SetGridWidth(1)

    ROOT.gStyle.SetFrameBorderMode(0)
    ROOT.gStyle.SetFrameBorderSize(1)
    ROOT.gStyle.SetFrameFillColor(0)
    ROOT.gStyle.SetFrameFillStyle(0)
    ROOT.gStyle.SetFrameLineColor(1)
    ROOT.gStyle.SetFrameLineStyle(1)
    ROOT.gStyle.SetFrameLineWidth(1)
    ROOT.gStyle.SetHistLineColor(1)
    ROOT.gStyle.SetHistLineStyle(0)
    ROOT.gStyle.SetHistLineWidth(1)

    ROOT.gStyle.SetEndErrorSize(2)
    ROOT.gStyle.SetFuncColor(2)
    ROOT.gStyle.SetFuncStyle(1)
    ROOT.gStyle.SetFuncWidth(1)
    ROOT.gStyle.SetOptDate(0)

    ROOT.gStyle.SetOptFile(0)
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetStatColor(0)
    ROOT.gStyle.SetStatFont(42)
    ROOT.gStyle.SetStatFontSize(0.04)
    ROOT.gStyle.SetStatTextColor(1)
    ROOT.gStyle.SetStatFormat("6.4g")
    ROOT.gStyle.SetStatBorderSize(0)
    ROOT.gStyle.SetStatH(0.19)
    ROOT.gStyle.SetStatW(0.24)
    ROOT.gStyle.SetStatX(0.9)
    ROOT.gStyle.SetStatY(0.9)

    ROOT.gStyle.SetPadTopMargin(0.07)
    ROOT.gStyle.SetPadBottomMargin(0.13)
    ROOT.gStyle.SetPadLeftMargin(0.12)
    ROOT.gStyle.SetPadRightMargin(0.05)

    ROOT.gStyle.SetOptTitle(0)
    ROOT.gStyle.SetTitleFont(42)
    ROOT.gStyle.SetTitleColor(1)
    ROOT.gStyle.SetTitleTextColor(1)
    ROOT.gStyle.SetTitleFillColor(10)
    ROOT.gStyle.SetTitleFontSize(0.05)

    ROOT.gStyle.SetTitleColor(1, "XYZ")
    ROOT.gStyle.SetTitleFont(42, "XYZ")
    ROOT.gStyle.SetTitleSize(0.05, "XYZ")
    ROOT.gStyle.SetTitleXOffset(0.9)
    ROOT.gStyle.SetTitleYOffset(1.05)

    ROOT.gStyle.SetLabelColor(1, "XYZ")
    ROOT.gStyle.SetLabelFont(42, "XYZ")
    ROOT.gStyle.SetLabelOffset(0.007, "XYZ")
    ROOT.gStyle.SetLabelSize(0.04, "XYZ")

    ROOT.gStyle.SetAxisColor(1, "XYZ")
    ROOT.gStyle.SetStripDecimals(1)
    ROOT.gStyle.SetTickLength(0.025, "XYZ")
    ROOT.gStyle.SetNdivisions(510, "XYZ")
    ROOT.gStyle.SetPadTickX(1)
    ROOT.gStyle.SetPadTickY(1)

    ROOT.gStyle.SetOptLogx(0)
    ROOT.gStyle.SetOptLogy(0)
    ROOT.gStyle.SetOptLogz(0)

    ROOT.gStyle.SetPaperSize(20.0, 20.0)
    ROOT.gStyle.SetPaintTextFormat(".2f")


def CMS_lumi(
    pad=0,
    lumi="",
    up=False,
    skipPreliminary=True,
    reduceSize=False,
    offset=0,
    offsetLumi=0,
):

    setTDRStyle()

    lm = pad.GetLeftMargin() - 0.15  ## 0.15 should be the default
    rm = min(0, 0.16 - pad.GetRightMargin())  ## 0.16 should be the default

    latex2 = ROOT.TLatex()
    latex2.SetNDC()
    latex2.SetTextSize(0.6 * pad.GetTopMargin())
    latex2.SetTextFont(42)
    latex2.SetTextAlign(31)
    if reduceSize:
        latex2.SetTextSize(0.5 * pad.GetTopMargin())

    if lumi != "":
        latex2.DrawLatex(
            rm + 0.94 + offsetLumi,
            0.95,
            "%s fb^{-1} (%d TeV)" % (str(lumi), centerOfMassEnergy),
        )
    else:
        latex2.DrawLatex(rm + 0.88 + offsetLumi, 0.95, "(%d TeV)" % centerOfMassEnergy)

    if up:
        latex2.SetTextSize(0.65 * pad.GetTopMargin())
        if reduceSize:
            latex2.SetTextSize(0.5 * pad.GetTopMargin())
        latex2.SetTextFont(62)
        latex2.SetTextAlign(11)
        latex2.DrawLatex(lm + 0.15 + offset, 0.95, "CMS")
    else:
        latex2.SetTextSize(0.6 * pad.GetTopMargin())
        if reduceSize:
            latex2.SetTextSize(0.45 * pad.GetTopMargin())
            # latex2.SetTextSize(0.40*pad.GetTopMargin())

        latex2.SetTextFont(62)
        latex2.SetTextAlign(11)
        latex2.DrawLatex(lm + 0.175 + offset, 0.85, "CMS")

    if not skipPreliminary:

        if up:
            latex2.SetTextSize(0.65 * pad.GetTopMargin())
            latex2.SetTextFont(52)
            latex2.SetTextAlign(11)
            if reduceSize:
                latex2.SetTextSize(0.5 * pad.GetTopMargin())
            latex2.DrawLatex(lm + 0.25 + offset, 0.95, "Preliminary")

        else:
            latex2.SetTextSize(0.6 * pad.GetTopMargin())
            if reduceSize:
                latex2.SetTextSize(0.45 * pad.GetTopMargin())
            latex2.SetTextFont(52)
            latex2.SetTextAlign(11)
            if reduceSize:
                latex2.DrawLatex(lm + 0.235 + offset, 0.85, "Preliminary")
            else:
                latex2.DrawLatex(lm + 0.28 + offset, 0.85, "Preliminary")
