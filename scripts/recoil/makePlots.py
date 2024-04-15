
import sys,argparse

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptTitle(0)

import functions
import plotutils
import narf
import wremnants.histselections as sel

from wremnants.datasets import datagroups

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="Input hdf5 file")
    parser.add_argument("-s", "--suffix", type=str, help="Suffix", default="")
    args = parser.parse_args()
    suffix = "" if args.suffix == "" else f"_{args.suffix}"

    groups = datagroups.Datagroups(args.input)
    met, analysis, flavor = functions.get_meta(groups)
    outDir = f"/home/submit/jaeyserm/public_html/recoil/{analysis}_{met}/plots_{flavor}/"
    functions.prepareDir(outDir, remove=False)
    datasets = groups.getNames()

    groups.addGroup("Top",
        members = groups.get_members_from_results(startswith=["Top", "SingleT", "TT"]),
        label = "Top",
        color = "#DE5A6A",
    )
    groups.addGroup("EWK_Z",
        members = groups.get_members_from_results(startswith=["Wplusmunu", "Wminusmunu", "Wmunu", "Wplustaunu", "Wminustaunu", "Ztautau", "Diboson", "WW", "WZ", "ZZ", "PhotonInduced"]),
        label = r"EWK (#tau^{#plus}#tau^{#minus}, VV)",
        color = "#64C0E8",
    )
    groups.addGroup("EWK_W",
        members = groups.get_members_from_results(startswith=["Wplustaunu", "Wminustaunu", "DYlowMass", "DYJetsToMuMuMass10to50", "Zmumu", "Ztautau", "Diboson", "WW", "WZ", "ZZ", "PhotonInduced"]),
        label = r"EWK (#tau^{#plus}#tau^{#minus}, VV)",
        color = "#64C0E8",
    )

    groups.groups['Zmumu'].color = "#F8CE68"
    groups.groups['Zmumu'].label = "DY #rightarrow #mu^{#plus}#mu^{#minus} (MiNNLO)"


    from utilities import boostHistHelpers as hh
    import hist
    def fakeHistABCD_(h, thresholdMT=40.0, fakerate_integration_axes=[], axis_name_mt="mt", integrateLowMT=True, integrateHighMT=False):
        # integrateMT=False keeps the mT axis in the returned histogram (can be used to have fakes vs mT)

        passIsoName = "passIso"
        passMTName = "passMT"
        passIso = {passIsoName: True}
        failIso = {passIsoName: False}
        passMT = {passMTName: True}
        failMT = {passMTName: False}

        nameMT, failMT, passMT = get_mt_selection(h, thresholdMT, axis_name_mt, integrateLowMT, integrateHighMT)

        if any(a in h.axes.name for a in fakerate_integration_axes):
            fakerate_axes = [n for n in h.axes.name if n not in [*fakerate_integration_axes, passIsoName, nameMT]]
            hPassIsoFailMT = h[{**passIso, nameMT: failMT}].project(*fakerate_axes)
            hFailIsoFailMT = h[{**failIso, nameMT: failMT}].project(*fakerate_axes)
        else:
            hPassIsoFailMT = h[{**passIso, nameMT: failMT}]
            hFailIsoFailMT = h[{**failIso, nameMT: failMT}]

        hFRF = hh.divideHists(hPassIsoFailMT, hFailIsoFailMT, cutoff=1, createNew=True)   

        return hh.multiplyHists(hFRF, h[{**failIso, nameMT: passMT}])

    def get_mt_selection(h, thresholdMT=40.0, axis_name_mt="mt", integrateLowMT=True, integrateHighMT=False):
        if axis_name_mt in h.axes.name:
            s = hist.tag.Slicer()
            high = h.axes[axis_name_mt].index(thresholdMT)
            failMT = s[:high:hist.sum] if integrateLowMT else s[:high:]
            passMT = s[high:hist.sum] if integrateHighMT else s[high:]
            nameMT = axis_name_mt
        else:
            failMT = 0
            passMT = 1
            nameMT = "passMT"

        return nameMT, failMT, passMT


    def fakeHistABCD(h, thresholdMT=40.0, fakerate_integration_axes=[], axis_name_mt="mt", integrateMT=False):

        
        axes = [ax.name for ax in h.axes]
        if "mt" in axes:
            s = hist.tag.Slicer()
            
            low = hist.underflow if h.axes[axis_name_mt].traits.underflow else 0
            failMT = {axis_name_mt : s[low:complex(0,thresholdMT):hist.sum]}
            passMT = {axis_name_mt : s[complex(0,thresholdMT):hist.overflow]}
            
            sf = h[{"passIso" : True, **failMT}].sum().value / h[{"passIso" : False, **failMT}].sum().value
            return h[{"passIso" : False, **passMT}]*sf
            
            #sf = h[{"passIso" : True, "passMT" : False}].sum().value / h[{"passIso" : False, "passMT" : False}].sum().value
            #return h[{"passIso" : False, "passMT" : True}]*sf

        ret = hh.multiplyHists(
            hh.divideHists(h[{"passIso" : True, "passMT" : False}],
                h[{"passIso" : False, "passMT" : False}],
                    cutoff=1
                ),
                    #where=h[{"passIso" : False, "passMT" : True}].values(flow=True)>1),
            h[{"passIso" : False, "passMT" : True}],
        )
        return ret

    if analysis == "lowPU":
        if flavor == "mumu":
            procs = ['Data', 'EWK_Z', 'Top', 'Zmumu']
            dataNormProc = 'Zmumu'
        elif flavor == "mu":
            procs = ['Data', 'EWK', 'Top', 'Fake', 'WJetsToMuNu'], 'SingleMuon'
            outDir = f"/eos/user/j/jaeyserm/www/wmass/lowPU/W{flavor}_{met}/plots_{charge}{suffix}/"
        elif flavor == "ee":
            procs, data = ['EWK', 'TTbar', 'DYee'], 'SingleElectron'

    else:
        if flavor == "mumu":
            groups.groups['Zmumu'].color = "#F8CE68"
            groups.groups['Zmumu'].label = "DY #rightarrow #mu^{#plus}#mu^{#minus} (MiNNLO)"
            procs, data = ['Data', 'EWK_Z', 'Top', 'Zmumu'], 'Data'
            dataNormProc = 'Zmumu'
        else:
            #groups.groups['Fake'].selectOp = fakeHistABCD
            groups.groups['Fake'].color = "#A9A9A9"
            groups.groups['Wmunu'].color = "#F8CE68"
            groups.groups['Wmunu'].label = "W^{#pm} #rightarrow #mu^{#pm}#nu"
            groups.groups['Top'].color = "#DE5A6A"
            procs, data = ['Data', 'EWK_W', 'Top', 'Fake', 'Wmunu'], 'Data'
            dataNormProc = 'Wmunu'

    if flavor == "mumu":
        
        if analysis == "lowPU":
            bins_met = list(range(0, 20, 1)) + list(range(20, 50, 2)) + list(range(50, 70, 4)) + [70, 80, 90, 100]
            xMin, xMax, yMin, yMax = 0, 100, 1e0, 1e5
        else:
            bins_met = 2
            bins_met = list(range(0, 20, 2)) + list(range(20, 50, 2)) + list(range(50, 70, 4)) + [70, 80, 90, 110, 120]
            xMin, xMax, yMin, yMax = 0, 120, 1e1, 1e8
        outDir_ = f"{outDir}/met_pt"
        plotutils.stacked_plot_ratio(groups, "met_uncorr_pt", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="METPT", rebin=bins_met, dataNormProc=dataNormProc, labels=[met, "Uncorrected"])
        plotutils.stacked_plot_ratio(groups, "met_corr_lep_pt", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="METPT", rebin=bins_met, dataNormProc=dataNormProc, labels=[met, "Lepton corrected"])
        plotutils.stacked_plot_ratio(groups, "met_corr_xy_pt", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="METPT", rebin=bins_met, dataNormProc=dataNormProc, labels=[met, "XY corrected"])
        plotutils.stacked_plot_ratio(groups, "met_corr_xy_pt_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="METPT", rebin=bins_met, dataNormProc=dataNormProc, labels=[met, "XY corrected", "q_{T} reweighted"])
        plotutils.stacked_plot_ratio(groups, "met_corr_rec_pt", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="METPT", rebin=bins_met, dataNormProc=dataNormProc, labels=[met, "Recoil corrected"], yRatio=1.06)
        plotutils.stacked_plot_ratio(groups, "met_corr_rec_pt_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="METPT", rebin=bins_met, dataNormProc=dataNormProc, labels=[met, "Recoil corrected", "q_{T} reweighted"], yRatio=1.06)
        plotutils.plot_systs(groups, "met_corr_rec_pt", "recoil_syst", 'Zmumu', f"{outDir_}/unc_syst/", xMin=xMin, xMax=xMax, yMin=0, yMax=-1, xLabel="MT", rebin=bins_met, extralabels=[met, "Recoil corrected"])
        #plotutils.plot_systs(groups, "met_corr_rec_pt", "recoil_stat", 'Zmumu', f"{outDir_}/unc_stat/", xMin=xMin, xMax=xMax, yMin=0, yMax=-1, xLabel="MT", rebin=bins_met, extralabels=[met, "Recoil corrected"])

        outDir_ = f"{outDir}/met_pt_wlike"
        plotutils.stacked_plot_ratio(groups, "met_uncorr_pt_wlike", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="METPT", rebin=bins_met, dataNormProc=dataNormProc, labels=[met, "Uncorrected"])
        plotutils.stacked_plot_ratio(groups, "met_corr_lep_pt_wlike", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="METPT", rebin=bins_met, dataNormProc=dataNormProc, labels=[met, "Lepton corrected"])
        plotutils.stacked_plot_ratio(groups, "met_corr_xy_pt_wlike", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="METPT", rebin=bins_met, dataNormProc=dataNormProc, labels=[met, "XY corrected"])
        plotutils.stacked_plot_ratio(groups, "met_corr_xy_pt_wlike_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="METPT", rebin=bins_met, dataNormProc=dataNormProc, labels=[met, "XY corrected", "q_{T} reweighted"])
        plotutils.stacked_plot_ratio(groups, "met_corr_rec_pt_wlike", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="METPT", rebin=bins_met, dataNormProc=dataNormProc, labels=[met, "Recoil corrected"], yRatio=1.06)
        plotutils.stacked_plot_ratio(groups, "met_corr_rec_pt_wlike_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="METPT", rebin=bins_met, dataNormProc=dataNormProc, labels=[met, "Recoil corrected", "q_{T} reweighted"], yRatio=1.06)


        if analysis == "lowPU": xMin, xMax, yMin, yMax = -4, 4, 1e0, 1e9
        else: xMin, xMax, yMin, yMax = -4, 4, 1e2, 1e9
        outDir_ = f"{outDir}/met_phi"
        plotutils.stacked_plot_ratio(groups, "met_uncorr_phi", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="METPHI", dataNormProc=dataNormProc, labels=[met, "Uncorrected"])
        plotutils.stacked_plot_ratio(groups, "met_corr_lep_phi", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="METPHI", dataNormProc=dataNormProc, labels=[met, "Lepton corrected"])
        plotutils.stacked_plot_ratio(groups, "met_corr_xy_phi", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="METPHI", dataNormProc=dataNormProc, labels=[met, "XY corrected"])
        plotutils.stacked_plot_ratio(groups, "met_corr_xy_phi_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="METPHI", dataNormProc=dataNormProc, labels=[met, "XY corrected", "q_{T} reweighted"])
        plotutils.stacked_plot_ratio(groups, "met_corr_rec_phi", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="METPHI", dataNormProc=dataNormProc, labels=[met, "Recoil corrected"])
        plotutils.stacked_plot_ratio(groups, "met_corr_rec_phi_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="METPHI", dataNormProc=dataNormProc, labels=[met, "Recoil corrected", "q_{T} reweighted"])


        if analysis == "lowPU":
            bins_recoil_para_perp = [-60, -50, -46, -42, -38, -34] + list(range(-30, 30, 2)) + [30, 34, 38, 42, 46, 50, 60, 70, 80, 90, 100]
            xMin, xMax, yMin, yMax = -60, 100, 1e0, 1e7
        else:
            bins_recoil_para_perp = [-150, -120, -110, -100, -90, -80, -70, -60, -50, -46, -42, -38, -34] + list(range(-30, 30, 2)) + [30, 34, 38, 42, 46, 50, 60, 70, 80, 90, 100, 110, 120, 150]
            xMin, xMax, yMin, yMax = -150, 150, 1e0, 1e9
        outDir_ = f"{outDir}/recoil_para"
        plotutils.stacked_plot_ratio(groups, "recoil_uncorr_para", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="UPARA", rebin=bins_recoil_para_perp, dataNormProc=dataNormProc, labels=[met, "Uncorrected"])
        plotutils.stacked_plot_ratio(groups, "recoil_corr_lep_para", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="UPARA", rebin=bins_recoil_para_perp, dataNormProc=dataNormProc, labels=[met, "Lepton corrected"])
        plotutils.stacked_plot_ratio(groups, "recoil_corr_xy_para", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="UPARA", rebin=bins_recoil_para_perp, dataNormProc=dataNormProc, labels=[met, "XY corrected"])
        plotutils.stacked_plot_ratio(groups, "recoil_corr_xy_para_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="UPARA", rebin=bins_recoil_para_perp, dataNormProc=dataNormProc, labels=[met, "XY corrected", "q_{T} reweighted"])
        plotutils.stacked_plot_ratio(groups, "recoil_corr_rec_para", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="UPARA", rebin=bins_recoil_para_perp, dataNormProc=dataNormProc, labels=[met, "Recoil corrected"])
        plotutils.stacked_plot_ratio(groups, "recoil_corr_rec_para_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="UPARA", rebin=bins_recoil_para_perp, dataNormProc=dataNormProc, labels=[met, "Recoil corrected", "q_{T} reweighted"])
        plotutils.plot_systs(groups, "recoil_corr_rec_para", "recoil_syst", 'Zmumu', f"{outDir_}/unc_syst/", xMin=xMin, xMax=xMax, yMin=0, yMax=-1, xLabel="UPARA", rebin=bins_recoil_para_perp, extralabels=[met, "Recoil corrected"])
        #plotutils.plot_systs(groups, "recoil_corr_rec_para", "recoil_stat", 'Zmumu', f"{outDir_}/unc_stat/", xMin=xMin, xMax=xMax, yMin=0, yMax=-1, xLabel="UPARA", rebin=bins_recoil_para_perp, extralabels=[met, "Recoil corrected"])

        if analysis == "lowPU":
            bins_recoil_para_perp = [-60, -50, -46, -42, -38, -34] + list(range(-30, 30, 2)) + [30, 34, 38, 42, 46, 50, 60]
            xMin, xMax, yMin, yMax = -60, 60, 1e0, 1e7
        else:
            bins_recoil_para_perp = [-150, -120, -110, -100, -90, -80, -70, -60, -50, -46, -42, -38, -34] + list(range(-30, 30, 2)) + [30, 34, 38, 42, 46, 50, 60, 70, 80, 90, 100, 110, 120, 150]
            xMin, xMax, yMin, yMax = -150, 150, 1e0, 1e9
        outDir_ = f"{outDir}/recoil_perp"
        plotutils.stacked_plot_ratio(groups, "recoil_uncorr_perp", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="UPERP", rebin=bins_recoil_para_perp, dataNormProc=dataNormProc, labels=[met, "Uncorrected"])
        plotutils.stacked_plot_ratio(groups, "recoil_corr_lep_perp", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="UPERP", rebin=bins_recoil_para_perp, dataNormProc=dataNormProc, labels=[met, "Lepton corrected"])
        plotutils.stacked_plot_ratio(groups, "recoil_corr_xy_perp", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="UPERP", rebin=bins_recoil_para_perp, dataNormProc=dataNormProc, labels=[met, "XY corrected"])
        plotutils.stacked_plot_ratio(groups, "recoil_corr_xy_perp_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="UPERP", rebin=bins_recoil_para_perp, dataNormProc=dataNormProc, labels=[met, "XY corrected", "q_{T} reweighted"])
        plotutils.stacked_plot_ratio(groups, "recoil_corr_rec_perp", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="UPERP", rebin=bins_recoil_para_perp, dataNormProc=dataNormProc, labels=[met, "Recoil corrected"])
        plotutils.stacked_plot_ratio(groups, "recoil_corr_rec_perp_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="UPERP", rebin=bins_recoil_para_perp, dataNormProc=dataNormProc, labels=[met, "Recoil corrected", "q_{T} reweighted"])
        plotutils.plot_systs(groups, "recoil_corr_rec_perp", "recoil_syst", 'Zmumu', f"{outDir_}/unc_syst/", xMin=xMin, xMax=xMax, yMin=0, yMax=-1, xLabel="UPERP", rebin=bins_recoil_para_perp, extralabels=[met, "Recoil corrected"])
        #plotutils.plot_systs(groups, "recoil_corr_rec_perp", "recoil_stat", 'Zmumu', f"{outDir_}/unc_stat/", xMin=xMin, xMax=xMax, yMin=0, yMax=-1, xLabel="UPERP", rebin=bins_recoil_para_perp, extralabels=[met, "Recoil corrected"])
        

        if analysis == "lowPU":
            xMin, xMax, yMin, yMax = -60, 100, 1e0, 1e7
        else:
            xMin, xMax, yMin, yMax = -200, 100, 1e0, 1e9
        outDir_ = f"{outDir}/recoil_para_qt"
        plotutils.stacked_plot_ratio(groups, "recoil_uncorr_para_qt", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="UPARAQT", rebin=5, dataNormProc=dataNormProc, labels=[met, "Uncorrected"])
        plotutils.stacked_plot_ratio(groups, "recoil_corr_lep_para_qt", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="UPARAQT", rebin=5, dataNormProc=dataNormProc, labels=[met, "Lepton corrected"])
        plotutils.stacked_plot_ratio(groups, "recoil_corr_xy_para_qt", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="UPARAQT", rebin=5, dataNormProc=dataNormProc, labels=[met, "XY corrected"])
        plotutils.stacked_plot_ratio(groups, "recoil_corr_xy_para_qt_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="UPARAQT", rebin=5, dataNormProc=dataNormProc, labels=[met, "XY corrected", "q_{T} reweighted"])
        plotutils.stacked_plot_ratio(groups, "recoil_corr_rec_para_qt", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="UPARAQT", rebin=5, dataNormProc=dataNormProc, labels=[met, "Recoil corrected"])
        plotutils.stacked_plot_ratio(groups, "recoil_corr_rec_para_qt_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="UPARAQT", rebin=5, dataNormProc=dataNormProc, labels=[met, "Recoil corrected", "q_{T} reweighted"])
        plotutils.plot_systs(groups, "recoil_corr_rec_para_qt", "recoil_syst", 'Zmumu', f"{outDir_}/unc_syst/", xMin=xMin, xMax=xMax, yMin=0, yMax=-1, xLabel="UPARAQT", rebin=5, extralabels=[met, "Recoil corrected"])
        #plotutils.plot_systs(groups, "recoil_corr_rec_para_qt", "recoil_stat", 'Zmumu', f"{outDir_}/unc_stat/", xMin=xMin, xMax=xMax, yMin=0, yMax=-1, xLabel="UPARAQT", rebin=5, extralabels=[met, "Recoil corrected"])


        if analysis == "lowPU":
            xMin, xMax, yMin, yMax = -60, 100, 1e0, 1e7
        else:
            recoil_magn_bins = 2
            xMin, xMax, yMin, yMax = 0, 200, 1e0, 1e9
        outDir_ = f"{outDir}/recoil_magn"
        plotutils.stacked_plot_ratio(groups, "recoil_uncorr_magn", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="UMAGN", rebin=recoil_magn_bins, dataNormProc=dataNormProc, labels=[met, "Uncorrected"])
        plotutils.stacked_plot_ratio(groups, "recoil_corr_lep_magn", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="UMAGN", rebin=recoil_magn_bins, dataNormProc=dataNormProc, labels=[met, "Lepton corrected"])
        plotutils.stacked_plot_ratio(groups, "recoil_corr_xy_magn", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="UMAGN", rebin=recoil_magn_bins, dataNormProc=dataNormProc, labels=[met, "XY corrected"])
        plotutils.stacked_plot_ratio(groups, "recoil_corr_xy_magn_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="UMAGN", rebin=recoil_magn_bins, dataNormProc=dataNormProc, labels=[met, "XY corrected", "q_{T} reweighted"])
        plotutils.stacked_plot_ratio(groups, "recoil_corr_rec_magn", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="UMAGN", rebin=recoil_magn_bins, dataNormProc=dataNormProc, labels=[met, "Recoil corrected"])
        plotutils.stacked_plot_ratio(groups, "recoil_corr_rec_magn_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="UMAGN", rebin=recoil_magn_bins, dataNormProc=dataNormProc, labels=[met, "Recoil corrected", "q_{T} reweighted"])
        plotutils.plot_systs(groups, "recoil_corr_rec_magn", "recoil_syst", 'Zmumu', f"{outDir_}/unc_syst/", xMin=xMin, xMax=xMax, yMin=0, yMax=-1, xLabel="UMAGN", rebin=recoil_magn_bins, extralabels=[met, "Recoil corrected"])
        #plotutils.plot_systs(groups, "recoil_corr_rec_magn", "recoil_stat", 'Zmumu', f"{outDir_}/unc_stat/", xMin=xMin, xMax=xMax, yMin=0, yMax=-1, xLabel="UMAGN", rebin=recoil_magn_bins, extralabels=[met, "Recoil corrected"])


        if analysis == "lowPU":
            xMin, xMax, yMin, yMax = -60, 100, 1e0, 1e7
        else:
            mt_bins = 2
            mt_bins = [0, 10, 15, 20, 25, 30, 35,] + list(range(40, 100, 2)) + [100, 102, 104, 106, 108, 110, 115, 120, 125, 130, 140, 160, 200]
            xMin, xMax, yMin, yMax = 0, 200, 1e0, 1e9
        outDir_ = f"{outDir}/mt"
        plotutils.stacked_plot_ratio(groups, "mt_uncorr", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="MT", rebin=mt_bins, dataNormProc=dataNormProc, labels=[met, "Uncorrected"])
        plotutils.stacked_plot_ratio(groups, "mt_corr_lep", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="MT", rebin=mt_bins, dataNormProc=dataNormProc, labels=[met, "Lepton corrected"])
        plotutils.stacked_plot_ratio(groups, "mt_corr_xy", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="MT", rebin=mt_bins, dataNormProc=dataNormProc, labels=[met, "XY corrected"])
        plotutils.stacked_plot_ratio(groups, "mt_corr_xy_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="MT", rebin=mt_bins, dataNormProc=dataNormProc, labels=[met, "XY corrected", "q_{T} reweighted"])
        plotutils.stacked_plot_ratio(groups, "mt_corr_rec", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="MT", rebin=mt_bins, dataNormProc=dataNormProc, labels=[met, "Recoil corrected"], yRatio=1.06)
        plotutils.stacked_plot_ratio(groups, "mt_corr_rec_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="MT", rebin=mt_bins, dataNormProc=dataNormProc, labels=[met, "Recoil corrected", "q_{T} reweighted"], yRatio=1.06)
        plotutils.stacked_plot_ratio(groups, "mt_corr_rec", procs, outDir_, xMin=40, xMax=140, yMin=0, yMax=4e5, logY=False, xLabel="MT", rebin=1, dataNormProc=dataNormProc, labels=[met, "Recoil corrected"], yRatio=1.06, suffix="_noLog")
        plotutils.stacked_plot_ratio(groups, "mt_corr_rec_qtrw", procs, outDir_, xMin=40, xMax=140, yMin=0, yMax=4e5, logY=False, xLabel="MT", rebin=1, dataNormProc=dataNormProc, labels=[met, "Recoil corrected", "q_{T} reweighted"], yRatio=1.06, suffix="_noLog")


        plotutils.plot_systs(groups, "mt_corr_rec", "recoil_syst", 'Zmumu', f"{outDir_}/unc_syst/", xMin=xMin, xMax=xMax, yMin=0, yMax=-1, xLabel="MT", rebin=mt_bins, extralabels=[met, "Recoil corrected"])
        #plotutils.plot_systs(groups, "mt_corr_rec", "recoil_stat", 'Zmumu', f"{outDir_}/unc_stat/", xMin=xMin, xMax=xMax, yMin=0, yMax=-1, xLabel="MT", rebin=mt_bins, extralabels=[met, "Recoil corrected"])

        if analysis == "lowPU":
            xMin, xMax, yMin, yMax = -60, 100, 1e0, 1e7
        else:
            vpt_bins = 2
            xMin, xMax, yMin, yMax = 0, 100, 1e0, 1e9
        plotutils.stacked_plot_ratio(groups, "v_pt", procs, outDir, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="QT", rebin=vpt_bins, dataNormProc=dataNormProc, labels=[met])
        plotutils.stacked_plot_ratio(groups, "v_pt_qtrw", procs, outDir, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="QT", rebin=vpt_bins, dataNormProc=dataNormProc, labels=[met, "q_{T} reweighted"])


        if analysis == "lowPU":
            xMin, xMax, yMin, yMax = -60, 100, 1e0, 1e7
        else:
            x_y_bins = 2
            xMin, xMax, yMin, yMax = -100, 100, 1e0, 1e9
        outDir_ = f"{outDir}/met_x"
        plotutils.stacked_plot_ratio(groups, "met_uncorr_x", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="METX", rebin=x_y_bins, dataNormProc=dataNormProc, labels=[met, "Uncorrected"])
        plotutils.stacked_plot_ratio(groups, "met_corr_lep_x", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="METX", rebin=x_y_bins, dataNormProc=dataNormProc, labels=[met, "Lepton corrected"])
        plotutils.stacked_plot_ratio(groups, "met_corr_xy_x", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="METX", rebin=x_y_bins, dataNormProc=dataNormProc, labels=[met, "XY corrected"])
        plotutils.stacked_plot_ratio(groups, "met_corr_xy_x_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="METX", rebin=x_y_bins, dataNormProc=dataNormProc, labels=[met, "XY corrected", "q_{T} reweighted"])
        plotutils.stacked_plot_ratio(groups, "met_corr_rec_x", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="METX", rebin=x_y_bins, dataNormProc=dataNormProc, labels=[met, "Recoil corrected"])
        plotutils.stacked_plot_ratio(groups, "met_corr_rec_x_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="METX", rebin=x_y_bins, dataNormProc=dataNormProc, labels=[met, "Recoil corrected", "q_{T} reweighted"])

        outDir_ = f"{outDir}/met_y"
        plotutils.stacked_plot_ratio(groups, "met_uncorr_y", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="METY", rebin=x_y_bins, dataNormProc=dataNormProc, labels=[met, "Uncorrected"])
        plotutils.stacked_plot_ratio(groups, "met_corr_lep_y", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="METY", rebin=x_y_bins, dataNormProc=dataNormProc, labels=[met, "Lepton corrected"])
        plotutils.stacked_plot_ratio(groups, "met_corr_xy_y", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="METY", rebin=x_y_bins, dataNormProc=dataNormProc, labels=[met, "XY corrected"])
        plotutils.stacked_plot_ratio(groups, "met_corr_xy_y_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="METY", rebin=x_y_bins, dataNormProc=dataNormProc, labels=[met, "XY corrected", "q_{T} reweighted"])
        plotutils.stacked_plot_ratio(groups, "met_corr_rec_y", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="METY", rebin=x_y_bins, dataNormProc=dataNormProc, labels=[met, "Recoil corrected"])
        plotutils.stacked_plot_ratio(groups, "met_corr_rec_y_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="METY", rebin=x_y_bins, dataNormProc=dataNormProc, labels=[met, "Recoil corrected", "q_{T} reweighted"])



    else:
        
        if analysis == "lowPU":
            bins_met = list(range(0, 20, 1)) + list(range(20, 50, 2)) + list(range(50, 70, 4)) + [70, 80, 90, 100]
            xMin, xMax, yMin, yMax = 0, 100, 1e0, 1e5
        else:
            bins_met = 2
            bins_met = list(range(0, 20, 2)) + list(range(20, 50, 2)) + list(range(50, 70, 4)) + [70, 80, 90, 110, 120]
            xMin, xMax, yMin, yMax = 0, 120, 1e1, 1e8
        outDir_ = f"{outDir}/met_pt"
        groups.set_histselectors(datasets, histToRead="met_uncorr_pt", fake_processes=None, mode="simple", smoothen=False, simultaneousABCD=False)
        plotutils.stacked_plot_ratio(groups, "met_uncorr_pt", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="METPT", rebin=bins_met, dataNormProc=dataNormProc, labels=[met, "Uncorrected"], yRatio=1.06, charge="combined")
        #quit()
        plotutils.stacked_plot_ratio(groups, "met_corr_lep_pt", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="METPT", rebin=bins_met, dataNormProc=dataNormProc, labels=[met, "Lepton corrected"], yRatio=1.06, charge="combined")
        plotutils.stacked_plot_ratio(groups, "met_corr_xy_pt", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="METPT", rebin=bins_met, dataNormProc=dataNormProc, labels=[met, "XY corrected"], yRatio=1.06, charge="combined")
        plotutils.stacked_plot_ratio(groups, "met_corr_xy_pt_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="METPT", rebin=bins_met, dataNormProc=dataNormProc, labels=[met, "XY corrected", "q_{T} reweighted"], yRatio=1.06, charge="combined")
        plotutils.stacked_plot_ratio(groups, "met_corr_rec_pt", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="METPT", rebin=bins_met, dataNormProc=dataNormProc, labels=[met, "Recoil corrected"], yRatio=1.06, charge="combined")
        plotutils.stacked_plot_ratio(groups, "met_corr_rec_pt_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="METPT", rebin=bins_met, dataNormProc=dataNormProc, labels=[met, "Recoil corrected", "q_{T} reweighted"], yRatio=1.06, charge="combined")
        #plotutils.plot_systs(groups, "met_corr_rec_pt", "recoil_syst", 'Zmumu', f"{outDir_}/unc_syst/", xMin=xMin, xMax=xMax, yMin=0, yMax=-1, xLabel="MT", rebin=bins_met, extralabels=[met, "Recoil corrected"])
        #plotutils.plot_systs(groups, "met_corr_rec_pt", "recoil_stat", 'Zmumu', f"{outDir_}/unc_stat/", xMin=xMin, xMax=xMax, yMin=0, yMax=-1, xLabel="MT", rebin=bins_met, extralabels=[met, "Recoil corrected"])


        if analysis == "lowPU":
            xMin, xMax, yMin, yMax = -60, 100, 1e0, 1e7
        else:
            mt_bins = list(range(40, 110, 1)) + [110, 112, 114, 116, 118, 120, 125, 130, 140, 160, 180, 200]
            #mt_bins = [0, 10, 15, 20, 25, 30, 35,] + list(range(40, 100, 2)) + [100, 102, 104, 106, 108, 110, 115, 120, 125, 130, 140, 160, 200]
            xMin, xMax, yMin, yMax = 40, 120, 0, 5e6
        outDir_ = f"{outDir}/mt"
        plotutils.stacked_plot_ratio(groups, "mt_uncorr", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabel="MT", rebin=mt_bins, dataNormProc=dataNormProc, labels=[met, "Uncorrected"], yRatio=1.06, charge="combined")
        plotutils.stacked_plot_ratio(groups, "mt_corr_lep", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabel="MT", rebin=mt_bins, dataNormProc=dataNormProc, labels=[met, "Lepton corrected"], yRatio=1.06, charge="combined")
        plotutils.stacked_plot_ratio(groups, "mt_corr_xy", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabel="MT", rebin=mt_bins, dataNormProc=dataNormProc, labels=[met, "XY corrected"], yRatio=1.06, charge="combined")
        plotutils.stacked_plot_ratio(groups, "mt_corr_xy_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabel="MT", rebin=mt_bins, dataNormProc=dataNormProc, labels=[met, "XY corrected", "q_{T} reweighted"], yRatio=1.06, charge="combined")
        plotutils.stacked_plot_ratio(groups, "mt_corr_rec", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabel="MT", rebin=mt_bins, dataNormProc=dataNormProc, labels=[met, "Recoil corrected"], yRatio=1.06, charge="combined")
        plotutils.stacked_plot_ratio(groups, "mt_corr_rec_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabel="MT", rebin=mt_bins, dataNormProc=dataNormProc, labels=[met, "Recoil corrected", "q_{T} reweighted"], yRatio=1.06, charge="combined")


        if analysis == "lowPU":
            xMin, xMax, yMin, yMax = -60, 100, 1e0, 1e7
        else:
            xMin, xMax, yMin, yMax = 20, 70, 0, 1e7
        outDir_ = f"{outDir}/lep_pt"
        plotutils.stacked_plot_ratio(groups, "lep_uncorr_pt", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabel="Lepton PT", rebin=1, dataNormProc=dataNormProc, labels=[met, "Uncorrected"], yRatio=1.06, charge="combined")
        plotutils.stacked_plot_ratio(groups, "lep_corr_lep_pt", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabel="Lepton PT", rebin=1, dataNormProc=dataNormProc, labels=[met, "Lepton corrected"], yRatio=1.06, charge="combined")
        plotutils.stacked_plot_ratio(groups, "lep_corr_rec_pt", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabel="Lepton PT", rebin=1, dataNormProc=dataNormProc, labels=[met, "Lepton corrected"], yRatio=1.06, charge="combined")
        plotutils.stacked_plot_ratio(groups, "lep_corr_rec_pt_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabel="Lepton PT", rebin=1, dataNormProc=dataNormProc, labels=[met, "Lepton corrected"], yRatio=1.06, charge="combined")

        if analysis == "lowPU": xMin, xMax, yMin, yMax = -4, 4, 1e0, 1e9
        else: xMin, xMax, yMin, yMax = -4, 4, 1e2, 1e9
        outDir_ = f"{outDir}/met_phi"
        plotutils.stacked_plot_ratio(groups, "met_uncorr_phi", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="METPHI", dataNormProc=dataNormProc, labels=[met, "Uncorrected"], yRatio=1.06, charge="combined")
        plotutils.stacked_plot_ratio(groups, "met_corr_lep_phi", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="METPHI", dataNormProc=dataNormProc, labels=[met, "Lepton corrected"], yRatio=1.06, charge="combined")
        plotutils.stacked_plot_ratio(groups, "met_corr_xy_phi", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="METPHI", dataNormProc=dataNormProc, labels=[met, "XY corrected"], yRatio=1.06, charge="combined")
        plotutils.stacked_plot_ratio(groups, "met_corr_xy_phi_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="METPHI", dataNormProc=dataNormProc, labels=[met, "XY corrected", "q_{T} reweighted"], yRatio=1.06, charge="combined")
        plotutils.stacked_plot_ratio(groups, "met_corr_rec_phi", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="METPHI", dataNormProc=dataNormProc, labels=[met, "Recoil corrected"], yRatio=1.06, charge="combined")
        plotutils.stacked_plot_ratio(groups, "met_corr_rec_phi_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="METPHI", dataNormProc=dataNormProc, labels=[met, "Recoil corrected", "q_{T} reweighted"], yRatio=1.06, charge="combined")

        if analysis == "lowPU": xMin, xMax, yMin, yMax = -4, 4, 1e0, 1e9
        else: xMin, xMax, yMin, yMax = 0, 100, 0, 1e7
        outDir_ = f"{outDir}/recoil_magn"
        plotutils.stacked_plot_ratio(groups, "recoil_uncorr_magn", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=False, xLabel="UMAGN", dataNormProc=dataNormProc, labels=[met, "Uncorrected"], yRatio=1.06, charge="combined")
        plotutils.stacked_plot_ratio(groups, "recoil_corr_lep_magn", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=False, xLabel="UMAGN", dataNormProc=dataNormProc, labels=[met, "Lepton corrected"], yRatio=1.06, charge="combined")
        plotutils.stacked_plot_ratio(groups, "recoil_corr_xy_magn", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=False, xLabel="UMAGN", dataNormProc=dataNormProc, labels=[met, "XY corrected"], yRatio=1.06, charge="combined")
        plotutils.stacked_plot_ratio(groups, "recoil_corr_xy_magn_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=False, xLabel="UMAGN", dataNormProc=dataNormProc, labels=[met, "XY corrected", "q_{T} reweighted"], yRatio=1.06, charge="combined")
        plotutils.stacked_plot_ratio(groups, "recoil_corr_rec_magn", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=False, xLabel="UMAGN", dataNormProc=dataNormProc, labels=[met, "Recoil corrected"], yRatio=1.06, charge="combined")
        plotutils.stacked_plot_ratio(groups, "recoil_corr_rec_magn_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=False, xLabel="UMAGN", dataNormProc=dataNormProc, labels=[met, "Recoil corrected", "q_{T} reweighted"], yRatio=1.06, charge="combined")
