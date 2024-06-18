
import sys,argparse

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptTitle(0)

import functions
import plotter
import narf

from wremnants.datasets import datagroups

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="Input hdf5 file")
    parser.add_argument("-m", "--mode", type=str, help="Mode", default="mc_data")
    args = parser.parse_args()

    groups = datagroups.Datagroups(args.input)
    met, analysis, flavor = functions.get_meta(groups)
    fOut = f"wremnants-data/data/recoil/{analysis}_{met}/vptrw_{args.mode}_{flavor}.json"
    print(flavor)
    ####################################################################
    if analysis == "lowPU":
        
        if args.mode == "mc_data":
            h_source = functions.readBoostHist(groups, "v_pt", ['Zmumu' if flavor=='mumu' else 'Zee'])
            h_target =functions.readBoostHist(groups, "v_pt", ['Data'])
            h_bkgs = functions.readBoostHist(groups, "v_pt", ['Ztautau', 'Other'])
            h_target.Add(h_bkgs, -1)
            vpt_bins = list(functions.readBoostHist(groups, "v_pt", ['Zmumu' if flavor=='mumu' else 'Zee'], boost=True).axes[0].edges)

    else:
        if args.mode == "mc_data":
            h_source = functions.readBoostHist(groups, "v_pt", ['Zmumu'])
            h_target =functions.readBoostHist(groups, "v_pt", ['Data'])
            h_bkgs = functions.readBoostHist(groups, "v_pt", ['Ztautau', 'Other'])
            h_target.Add(h_bkgs, -1)
            vpt_bins = list(functions.readBoostHist(groups, "v_pt", ['Zmumu'], boost=True).axes[0].edges)

        elif args.mode == "gen_data":
            h_source = functions.readBoostHist(groups, "v_gen_pt", ['Zmumu'])
            h_target =functions.readBoostHist(groups, "v_pt", ['Data'])
            h_bkgs = functions.readBoostHist(groups, "v_pt", ['Ztautau', 'Other'])
            h_target.Add(h_bkgs, -1)
            vpt_bins = list(functions.readBoostHist(groups, "v_gen_pt", ['Zmumu'], boost=True).axes[0].edges)



    # normalize histograms, to preserve total normalization
    h_source.Scale(1./h_source.Integral())
    h_target.Scale(1./h_target.Integral())

    weights = []
    for i in range(1, h_source.GetNbinsX()+1):
        w = h_target.GetBinContent(i)/h_source.GetBinContent(i) if h_source.GetBinContent(i) > 0 and h_target.GetBinContent(i) else 1
        weights.append(w)
        print(h_target.GetBinLowEdge(i), h_source.GetBinLowEdge(i+1), w)

    outDict = {}
    outDict['vpt_bins'] = vpt_bins
    outDict['weights'] = weights
    functions.writeJSON(fOut, outDict)
    print(fOut)
