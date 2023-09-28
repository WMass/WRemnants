
import sys,array,math,os,copy,json
import numpy as np

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptTitle(0)

import functions
import plotter

import lz4.frame
import pickle
import narf
import numpy as np

import wremnants.datasets.datagroups as datagroups


def makePlot(xMin=0, xMax=200, yMin=1, yMax=1e5):

    b_resp = functions.readBoostHistProc(groups, "mT_corr_rec_qTrw", [proc])
    b_no_resp = functions.readBoostHistProc(groups, "mT_corr_rec_resp_qTrw", [proc])

    h_resp = narf.hist_to_root(b_resp)
    h_no_resp = narf.hist_to_root(b_no_resp)

    h_resp.SetLineColor(ROOT.kBlue)
    h_resp.SetLineWidth(2)

    h_no_resp.SetLineColor(ROOT.kRed)
    h_no_resp.SetLineWidth(2)

    cfg = {

        'logy'              : False,
        'logx'              : False,

        'xmin'              : xMin,
        'xmax'              : xMax,
        'ymin'              : yMin,
        'ymax'              : yMax,

        'xtitle'            : "m_{T} (GeV)",
        'ytitle'            : "Events",

        'topRight'          : lumi_header, 
        'topLeft'           : "#bf{CMS} #scale[0.7]{#it{Preliminary}}",

    }

    leg = ROOT.TLegend(.60, 0.75, .8, .90)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.035)
    leg.SetHeader(met)
    leg.AddEntry(h_resp, "Response corrected", "L")
    leg.AddEntry(h_no_resp, "Nominal", "L")

    plotter.cfg = cfg
    canvas = plotter.canvas()
    dummy = plotter.dummy()   
    canvas.cd()
    dummy.Draw("HIST")
    h_resp.Draw("HIST SAME")
    h_no_resp.Draw("HIST SAME")
    plotter.aux()

    leg.Draw("SAME")

    canvas.SetGrid()
    canvas.SetTickx()
    canvas.SetTicky()
    canvas.Modify()
    canvas.Update()
    canvas.Draw()
    canvas.SaveAs(f"{outDir}/response_comparison.png")
    canvas.SaveAs(f"{outDir}/response_comparison.pdf")
    canvas.Delete()


if __name__ == "__main__":

    met = "DeepMETReso" # PFMET, RawPFMET DeepMETReso
    flavor = "mumu" # mu, e, mumu, ee
    lowPU = False

    ####################################################################
    if lowPU:
        npv_max, npv_fit_min, npv_fit_max = 10, 0, 10
        lumi_header = "199 pb^{#minus1} (13 TeV)"
        
        groups = datagroupsLowPU("lowPU_%s_%s.pkl.lz4" % (flavor, met), flavor=flavor)
        procs = ['EWK', 'Top', 'Zmumu'] 
        data = "SingleMuon" if "mu" in flavor else "SingleElectron"

        outDir = "/eos/user/j/jaeyserm/www/wmass/lowPU/METxy_correction/METxy_%s_%s/" % (flavor, met)
        fOut = "wremnants/data/recoil/lowPU/%s_%s/met_xy_correction.json" % (flavor, met)
        functions.prepareDir(outDir, True)
        
        dictout = {}
        dictX = METxyCorrection(direction="x", corrType="corr_lep", polyOrderData=1, polyOrderMC=1, procs=procs, data=data)
        dictY = METxyCorrection(direction="y", corrType="corr_lep", polyOrderData=1, polyOrderMC=1, procs=procs, data=data)
        
        
        dictout['x'] = dictX
        dictout['y'] = dictY
        jsOut = json.dumps(dictout, indent = 4)
        with open(fOut, "w") as outfile: outfile.write(jsOut)
        os.system("cp %s %s" % (fOut, outDir)) # make copy to web dir

        
        METxyCorrection(direction="x", corrType="corr_xy", polyOrderData=1, polyOrderMC=1, procs=procs, data=data)
        METxyCorrection(direction="y", corrType="corr_xy", polyOrderData=1, polyOrderMC=1, procs=procs, data=data)
    
    else:
        lumi_header = "16.8 fb^{#minus1} (13 TeV)"

        if flavor == "mumu":
            groups = datagroups.Datagroups(f"mz_wlike_with_mu_eta_pt_{met}.hdf5")
            proc = "Zmumu"
        else:
            npv_max, npv_fit_min, npv_fit_max = 60, 0, 55
            polyOrderDataX, polyOrderMCX = 3, 3
            polyOrderDataY, polyOrderMCY = 6, 3
            datagroups = datagroups2016(f"mw_with_mu_eta_pt_{met}.hdf5")
            procs = ["Zmumu", "Ztautau", "Wtaunu", "Wmunu", "Top", "Diboson"]
            data = "Data"

            for g in datagroups.groups:
                datagroups.groups[g]['selectOp'] = None


        #outDir = "/eos/user/j/jaeyserm/www/wmass/highPU/METxy_correction/METxy_%s_%s/" % (flavor, met)
        outDir = f"/home/submit/jaeyserm/public_html/wmass/highPU/Z{flavor}_{met}/plots/"
        makePlot(xMin=0, xMax=200, yMin=1, yMax=250000)

