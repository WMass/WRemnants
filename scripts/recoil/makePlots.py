
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
    outDir = f"/home/submit/jaeyserm/public_html/recoil/{analysis}_{met}/plots_{flavor}{suffix}/"
    functions.prepareDir(outDir, remove=False)
    datasets = groups.getNames()
    doSysts = False

    def expand_groups(groups):
        groups.addGroup("Top",
            members = groups.get_members_from_results(startswith=["Top", "SingleT", "TT"]),
            label = "Top",
            color = "#DE5A6A",
        )
        groups.addGroup("EWK_Z",
            members = groups.get_members_from_results(startswith=["Wplusmunu", "Wminusmunu", "Wmunu", "Wplustaunu", "Wminustaunu", "Ztautau", "Diboson", "WW", "WZ", "ZZ", "PhotonInduced"]),
            label = r"EWK (#mu^{#plus}#mu^{#minus}, #tau^{#plus}#tau^{#minus}, VV)",
            color = "#64C0E8",
        )
        groups.addGroup("EWK_W",
            members = groups.get_members_from_results(startswith=["Wplustaunu", "Wminustaunu", "DYlowMass", "DYJetsToMuMuMass10to50", "Zmumu", "Ztautau", "Diboson", "WW", "WZ", "ZZ", "PhotonInduced"]),
            label = r"EWK (#tau^{#plus}#tau^{#minus}, VV)",
            color = "#64C0E8",
        )

        th_corr = "MiNNLO+scet+DY" if "scetlib_dyturbo" in suffix else "MiNNLO"
        if flavor == "mumu":
            groups.groups['Zmumu'].color = "#F8CE68"
            groups.groups['Zmumu'].label = f"DY #rightarrow #mu^{{#plus}}#mu^{{#minus}} ({th_corr})"

        if flavor == "mu":
            groups.groups['Fake_mu' if analysis == "lowPU" else 'Fake'].color = "#A9A9A9"
            groups.groups['Fake_mu' if analysis == "lowPU" else 'Fake'].label = "Nonprompt"
            groups.groups['Wmunu'].color = "#F8CE68"
            groups.groups['Wmunu'].label = f"W^{{#pm}} #rightarrow #mu^{{#pm}}#nu ({th_corr})"
            groups.groups['Top'].color = "#DE5A6A"

        return groups

    groups = expand_groups(groups)

    if analysis == "lowPU":
        if flavor == "mumu":
            procs = ['Data', 'EWK_Z', 'Top', 'Zmumu']
            dataNormProc = 'Zmumu'
        elif flavor == "mu":
            procs, data = ['Data', 'EWK_W', 'Top', 'Fake_mu' if analysis == "lowPU" else 'Fake', 'Wmunu'], 'SingleMuon'
            dataNormProc = 'Wmunu'
        elif flavor == "ee":
            procs, data = ['EWK_W', 'TTbar', 'DYee'], 'SingleElectron'

    else:
        if flavor == "mumu":
            #groups.groups['Zmumu'].color = "#F8CE68"
            #groups.groups['Zmumu'].label = "DY #rightarrow #mu^{#plus}#mu^{#minus} (MiNNLO)"
            procs, data = ['Data', 'EWK_Z', 'Top', 'Zmumu'], 'Data'
            dataNormProc = 'Zmumu'
        else:
            #groups.groups['Fake'].color = "#A9A9A9"
            #groups.groups['Wmunu'].color = "#F8CE68"
            #groups.groups['Wmunu'].label = "W^{#pm} #rightarrow #mu^{#pm}#nu"
            #groups.groups['Top'].color = "#DE5A6A"
            procs, data = ['Data', 'EWK_W', 'Top', 'Fake', 'Wmunu'], 'Data'
            dataNormProc = 'Wmunu'


    if flavor == "mumu":

        extraRatios_ = False
        if extraRatios_:
            extraRatios = []
            extraRatios.append(("pre-fsr", expand_groups(datagroups.Datagroups("/scratch/submit/cms/jaeyserm/Recoil/mz_wlike_with_mu_eta_pt_DeepMETPVRobust_gen_prefsr.hdf5"))))
            extraRatios.append(("post-fsr", expand_groups(datagroups.Datagroups("/scratch/submit/cms/jaeyserm/Recoil/mz_wlike_with_mu_eta_pt_DeepMETPVRobust_gen_postfsr.hdf5"))))
            extraRatios.append(("proxy-reco", expand_groups(datagroups.Datagroups("/scratch/submit/cms/jaeyserm/Recoil/mz_wlike_with_mu_eta_pt_DeepMETPVRobust_gen_proxy_postfsr.hdf5"))))

        if analysis == "lowPU":
            yRatio = 1.15
        else:
            yRatio = 1.06


        ################
        ## MET
        ################
        if analysis == "lowPU": xMin, xMax, yMin, yMax = 0, 50, 0, 1.2e4
        else: xMin, xMax, yMin, yMax = 0, 50, 0, 0.6e6
        outDir_ = f"{outDir}/met_pt"
        bins_met = 1

        plotutils.stacked_plot_ratio(groups, "met_uncorr_pt", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabel="METPT", rebin=bins_met, dataNormProc=dataNormProc, labels=[met, "Uncorrected"], yRatio=yRatio)
        plotutils.stacked_plot_ratio(groups, "met_corr_lep_pt", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabel="METPT", rebin=bins_met, dataNormProc=dataNormProc, labels=[met, "Lepton corrected"], yRatio=yRatio)
        plotutils.stacked_plot_ratio(groups, "met_corr_xy_pt", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabel="METPT", rebin=bins_met, dataNormProc=dataNormProc, labels=[met, "XY corrected"], yRatio=yRatio)
        plotutils.stacked_plot_ratio(groups, "met_corr_xy_pt_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabel="METPT", rebin=bins_met, dataNormProc=dataNormProc, labels=[met, "XY corrected", "q_{T} reweighted"], yRatio=yRatio)
        plotutils.stacked_plot_ratio(groups, "met_corr_rec_pt", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabel="METPT", rebin=bins_met, dataNormProc=dataNormProc, labels=[met, "Recoil corrected"], yRatio=yRatio)
        plotutils.stacked_plot_ratio(groups, "met_corr_rec_pt_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabel="METPT", rebin=bins_met, dataNormProc=dataNormProc, labels=[met, "Recoil corrected", "q_{T} reweighted"], yRatio=yRatio)

        if doSysts:
            plotutils.plot_systs(groups, "met_corr_rec_pt", "recoil_syst", 'Zmumu', f"{outDir_}/unc_syst/", xMin=xMin, xMax=xMax, yMin=yMin, yMax=-1, xLabel="METPT", rebin=bins_met, extralabels=[met, "Recoil corrected"])
            plotutils.plot_systs(groups, "met_corr_rec_pt", "recoil_stat", 'Zmumu', f"{outDir_}/unc_stat/", xMin=xMin, xMax=xMax, yMin=yMin, yMax=-1, xLabel="METPT", rebin=bins_met, extralabels=[met, "Recoil corrected"])

        if extraRatios_:
            plotutils.stacked_plot_ratio(groups, "met_corr_rec_pt_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabel="METPT", rebin=bins_met, dataNormProc=dataNormProc, labels=[met, "Recoil corrected", "q_{T} reweighted"], yRatio=yRatio, suffix="_extraRatios", extraRatios=extraRatios)

        ################
        ## MET - log
        ################
        if analysis == "lowPU": xMin, xMax, yMin, yMax = 0, 100, 1e0, 1e6
        else: xMin, xMax, yMin, yMax = 0, 120, 1e1, 1e8
        outDir_ = f"{outDir}/met_pt_log"
        bins_met = 2

        plotutils.stacked_plot_ratio(groups, "met_uncorr_pt", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="METPT", rebin=bins_met, dataNormProc=dataNormProc, labels=[met, "Uncorrected"], yRatio=yRatio)
        plotutils.stacked_plot_ratio(groups, "met_corr_lep_pt", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="METPT", rebin=bins_met, dataNormProc=dataNormProc, labels=[met, "Lepton corrected"], yRatio=yRatio)
        plotutils.stacked_plot_ratio(groups, "met_corr_xy_pt", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="METPT", rebin=bins_met, dataNormProc=dataNormProc, labels=[met, "XY corrected"], yRatio=yRatio)
        plotutils.stacked_plot_ratio(groups, "met_corr_xy_pt_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="METPT", rebin=bins_met, dataNormProc=dataNormProc, labels=[met, "XY corrected", "q_{T} reweighted"], yRatio=yRatio)
        plotutils.stacked_plot_ratio(groups, "met_corr_rec_pt", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="METPT", rebin=bins_met, dataNormProc=dataNormProc, labels=[met, "Recoil corrected"], yRatio=yRatio)
        plotutils.stacked_plot_ratio(groups, "met_corr_rec_pt_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="METPT", rebin=bins_met, dataNormProc=dataNormProc, labels=[met, "Recoil corrected", "q_{T} reweighted"], yRatio=yRatio)

        if doSysts:
            plotutils.plot_systs(groups, "met_corr_rec_pt", "recoil_syst", 'Zmumu', f"{outDir_}/unc_syst/", xMin=xMin, xMax=xMax, logY=True, yMin=yMin, yMax=yMax, xLabel="METPT", rebin=bins_met, extralabels=[met, "Recoil corrected"])
            plotutils.plot_systs(groups, "met_corr_rec_pt", "recoil_stat", 'Zmumu', f"{outDir_}/unc_stat/", xMin=xMin, xMax=xMax, logY=True, yMin=yMin, yMax=yMax, xLabel="METPT", rebin=bins_met, extralabels=[met, "Recoil corrected"])

        if extraRatios_:
            plotutils.stacked_plot_ratio(groups, "met_corr_rec_pt_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, logY=True, yMin=yMin, yMax=yMax, xLabel="METPT", rebin=bins_met, dataNormProc=dataNormProc, labels=[met, "Recoil corrected", "q_{T} reweighted"], yRatio=yRatio, suffix="_extraRatios", extraRatios=extraRatios)


        ################
        ## WLIKE MET
        ################
        if analysis == "lowPU": xMin, xMax, yMin, yMax = 0, 100, 0, 1e4
        else: xMin, xMax, yMin, yMax = 0, 100, 0, 0.6e6
        outDir_ = f"{outDir}/met_pt_wlike"
        bins_met = 1

        plotutils.stacked_plot_ratio(groups, "met_uncorr_pt_wlike", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabel="METWLIKEPT", rebin=bins_met, dataNormProc=dataNormProc, labels=[met, "Uncorrected"], yRatio=yRatio)
        plotutils.stacked_plot_ratio(groups, "met_corr_lep_pt_wlike", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabel="METWLIKEPT", rebin=bins_met, dataNormProc=dataNormProc, labels=[met, "Lepton corrected"], yRatio=yRatio)
        plotutils.stacked_plot_ratio(groups, "met_corr_xy_pt_wlike", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabel="METWLIKEPT", rebin=bins_met, dataNormProc=dataNormProc, labels=[met, "XY corrected"], yRatio=yRatio)
        plotutils.stacked_plot_ratio(groups, "met_corr_xy_pt_wlike_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabel="METWLIKEPT", rebin=bins_met, dataNormProc=dataNormProc, labels=[met, "XY corrected", "q_{T} reweighted"], yRatio=yRatio)
        plotutils.stacked_plot_ratio(groups, "met_corr_rec_pt_wlike", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabel="METWLIKEPT", rebin=bins_met, dataNormProc=dataNormProc, labels=[met, "Recoil corrected"], yRatio=yRatio)
        plotutils.stacked_plot_ratio(groups, "met_corr_rec_pt_wlike_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabel="METWLIKEPT", rebin=bins_met, dataNormProc=dataNormProc, labels=[met, "Recoil corrected", "q_{T} reweighted"], yRatio=yRatio)

        if extraRatios_:
            plotutils.stacked_plot_ratio(groups, "met_corr_rec_pt_wlike_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabel="METWLIKEPT", rebin=bins_met, dataNormProc=dataNormProc, labels=[met, "Recoil corrected", "q_{T} reweighted"], yRatio=yRatio, suffix="_extraRatios", extraRatios=extraRatios)


        ################
        ## WLIKE MET - log
        ################
        if analysis == "lowPU": xMin, xMax, yMin, yMax = 0, 120, 1e0, 1e6
        else: xMin, xMax, yMin, yMax = 0, 120, 1e1, 1e8
        outDir_ = f"{outDir}/met_pt_wlike_log"
        bins_met = 2

        plotutils.stacked_plot_ratio(groups, "met_uncorr_pt_wlike", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="METWLIKEPT", rebin=bins_met, dataNormProc=dataNormProc, labels=[met, "Uncorrected"], yRatio=yRatio)
        plotutils.stacked_plot_ratio(groups, "met_corr_lep_pt_wlike", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="METWLIKEPT", rebin=bins_met, dataNormProc=dataNormProc, labels=[met, "Lepton corrected"], yRatio=yRatio)
        plotutils.stacked_plot_ratio(groups, "met_corr_xy_pt_wlike", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="METWLIKEPT", rebin=bins_met, dataNormProc=dataNormProc, labels=[met, "XY corrected"], yRatio=yRatio)
        plotutils.stacked_plot_ratio(groups, "met_corr_xy_pt_wlike_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="METWLIKEPT", rebin=bins_met, dataNormProc=dataNormProc, labels=[met, "XY corrected", "q_{T} reweighted"], yRatio=yRatio)
        plotutils.stacked_plot_ratio(groups, "met_corr_rec_pt_wlike", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="METWLIKEPT", rebin=bins_met, dataNormProc=dataNormProc, labels=[met, "Recoil corrected"], yRatio=yRatio)
        plotutils.stacked_plot_ratio(groups, "met_corr_rec_pt_wlike_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="METWLIKEPT", rebin=bins_met, dataNormProc=dataNormProc, labels=[met, "Recoil corrected", "q_{T} reweighted"], yRatio=yRatio)



        ################
        ## MET PHI
        ################
        if analysis == "lowPU": xMin, xMax, yMin, yMax = -4, 4, 1e0, 1e9
        else: xMin, xMax, yMin, yMax = -4, 4, 1e2, 1e9
        outDir_ = f"{outDir}/met_phi_log"
        plotutils.stacked_plot_ratio(groups, "met_uncorr_phi", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="METPHI", dataNormProc=dataNormProc, labels=[met, "Uncorrected"], yRatio=yRatio)
        plotutils.stacked_plot_ratio(groups, "met_corr_lep_phi", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="METPHI", dataNormProc=dataNormProc, labels=[met, "Lepton corrected"], yRatio=yRatio)
        plotutils.stacked_plot_ratio(groups, "met_corr_xy_phi", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="METPHI", dataNormProc=dataNormProc, labels=[met, "XY corrected"], yRatio=yRatio)
        plotutils.stacked_plot_ratio(groups, "met_corr_xy_phi_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="METPHI", dataNormProc=dataNormProc, labels=[met, "XY corrected", "q_{T} reweighted"], yRatio=yRatio)
        plotutils.stacked_plot_ratio(groups, "met_corr_rec_phi", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="METPHI", dataNormProc=dataNormProc, labels=[met, "Recoil corrected"], yRatio=yRatio)
        plotutils.stacked_plot_ratio(groups, "met_corr_rec_phi_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="METPHI", dataNormProc=dataNormProc, labels=[met, "Recoil corrected", "q_{T} reweighted"], yRatio=yRatio)


        ################
        ## RECOIL PARALLEL
        ################
        if analysis == "lowPU": xMin, xMax, yMin, yMax = -100, 50, 0, 0.8e4
        else: xMin, xMax, yMin, yMax = -100, 50, 0, 0.6e6
        rebin = 2
        outDir_ = f"{outDir}/recoil_para"

        plotutils.stacked_plot_ratio(groups, "recoil_uncorr_para", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabel="UPARA", rebin=rebin, dataNormProc=dataNormProc, labels=[met, "Uncorrected"], yRatio=yRatio)
        plotutils.stacked_plot_ratio(groups, "recoil_corr_lep_para", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabel="UPARA", rebin=rebin, dataNormProc=dataNormProc, labels=[met, "Lepton corrected"], yRatio=yRatio)
        plotutils.stacked_plot_ratio(groups, "recoil_corr_xy_para", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabel="UPARA", rebin=rebin, dataNormProc=dataNormProc, labels=[met, "XY corrected"], yRatio=yRatio)
        plotutils.stacked_plot_ratio(groups, "recoil_corr_xy_para_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabel="UPARA", rebin=rebin, dataNormProc=dataNormProc, labels=[met, "XY corrected", "q_{T} reweighted"], yRatio=yRatio)
        plotutils.stacked_plot_ratio(groups, "recoil_corr_rec_para", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabel="UPARA", rebin=rebin, dataNormProc=dataNormProc, labels=[met, "Recoil corrected"], yRatio=yRatio)
        plotutils.stacked_plot_ratio(groups, "recoil_corr_rec_para_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabel="UPARA", rebin=rebin, dataNormProc=dataNormProc, labels=[met, "Recoil corrected", "q_{T} reweighted"], yRatio=yRatio)

        if doSysts:
            plotutils.plot_systs(groups, "recoil_corr_rec_para", "recoil_syst", 'Zmumu', f"{outDir_}/unc_syst/", xMin=xMin, xMax=xMax, yMin=yMin, yMax=-1, xLabel="UPARA", rebin=rebin, extralabels=[met, "Recoil corrected"], yRatio=yRatio)
            plotutils.plot_systs(groups, "recoil_corr_rec_para", "recoil_stat", 'Zmumu', f"{outDir_}/unc_stat/", xMin=xMin, xMax=xMax, yMin=yMin, yMax=-1, xLabel="UPARA", rebin=rebin, extralabels=[met, "Recoil corrected"], yRatio=yRatio)

        if extraRatios_:
            plotutils.stacked_plot_ratio(groups, "recoil_corr_rec_para_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabel="UPARA", rebin=rebin, dataNormProc=dataNormProc, labels=[met, "Recoil corrected", "q_{T} reweighted"], yRatio=yRatio, suffix="_extraRatios", extraRatios=extraRatios)


        ################
        ## RECOIL PARALLEL - log
        ################
        if analysis == "lowPU":
            xMin, xMax, yMin, yMax = -200, 100, 1e-1, 1e7
        else:
            bins_recoil_para_perp = [-150, -120, -110, -100, -90, -80, -70, -60, -50, -46, -42, -38, -34] + list(range(-30, 30, 2)) + [30, 34, 38, 42, 46, 50, 60, 70, 80, 90, 100, 110, 120, 150]
            xMin, xMax, yMin, yMax = -150, 100, 1e0, 1e9
        rebin = 2
        outDir_ = f"{outDir}/recoil_para_log"

        plotutils.stacked_plot_ratio(groups, "recoil_uncorr_para", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="UPARA", rebin=5, dataNormProc=dataNormProc, labels=[met, "Uncorrected"], yRatio=yRatio)
        plotutils.stacked_plot_ratio(groups, "recoil_corr_lep_para", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="UPARA", rebin=5, dataNormProc=dataNormProc, labels=[met, "Lepton corrected"], yRatio=yRatio)
        plotutils.stacked_plot_ratio(groups, "recoil_corr_xy_para", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="UPARA", rebin=5, dataNormProc=dataNormProc, labels=[met, "XY corrected"], yRatio=yRatio)
        plotutils.stacked_plot_ratio(groups, "recoil_corr_xy_para_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="UPARA", rebin=5, dataNormProc=dataNormProc, labels=[met, "XY corrected", "q_{T} reweighted"], yRatio=yRatio)
        plotutils.stacked_plot_ratio(groups, "recoil_corr_rec_para", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="UPARA", rebin=5, dataNormProc=dataNormProc, labels=[met, "Recoil corrected"], yRatio=yRatio)
        plotutils.stacked_plot_ratio(groups, "recoil_corr_rec_para_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="UPARA", rebin=5, dataNormProc=dataNormProc, labels=[met, "Recoil corrected", "q_{T} reweighted"], yRatio=yRatio)

        if doSysts:
            plotutils.plot_systs(groups, "recoil_corr_rec_para", "recoil_syst", 'Zmumu', f"{outDir_}/unc_syst/", xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="UPARA", rebin=5, extralabels=[met, "Recoil corrected"], yRatio=yRatio)
            plotutils.plot_systs(groups, "recoil_corr_rec_para", "recoil_stat", 'Zmumu', f"{outDir_}/unc_stat/", xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="UPARA", rebin=5, extralabels=[met, "Recoil corrected"], yRatio=yRatio)

        if extraRatios_:
            plotutils.stacked_plot_ratio(groups, "recoil_corr_rec_para_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="UPARA", rebin=5, dataNormProc=dataNormProc, labels=[met, "Recoil corrected", "q_{T} reweighted"], yRatio=yRatio, suffix="_extraRatios", extraRatios=extraRatios)


        ################
        ## RECOIL PERPENDICULAR
        ################
        if analysis == "lowPU": xMin, xMax, yMin, yMax = -50, 50, 0, 1.3e4
        else: xMin, xMax, yMin, yMax = -50, 50, 0, 0.6e6
        outDir_ = f"{outDir}/recoil_perp"
        rebin = 2

        plotutils.stacked_plot_ratio(groups, "recoil_uncorr_perp", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabel="UPERP", rebin=rebin, dataNormProc=dataNormProc, labels=[met, "Uncorrected"], yRatio=yRatio)
        plotutils.stacked_plot_ratio(groups, "recoil_corr_lep_perp", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabel="UPERP", rebin=rebin, dataNormProc=dataNormProc, labels=[met, "Lepton corrected"], yRatio=yRatio)
        plotutils.stacked_plot_ratio(groups, "recoil_corr_xy_perp", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabel="UPERP", rebin=rebin, dataNormProc=dataNormProc, labels=[met, "XY corrected"], yRatio=yRatio)
        plotutils.stacked_plot_ratio(groups, "recoil_corr_xy_perp_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabel="UPERP", rebin=rebin, dataNormProc=dataNormProc, labels=[met, "XY corrected", "q_{T} reweighted"], yRatio=yRatio)
        plotutils.stacked_plot_ratio(groups, "recoil_corr_rec_perp", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabel="UPERP", rebin=rebin, dataNormProc=dataNormProc, labels=[met, "Recoil corrected"], yRatio=yRatio)
        plotutils.stacked_plot_ratio(groups, "recoil_corr_rec_perp_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabel="UPERP", rebin=rebin, dataNormProc=dataNormProc, labels=[met, "Recoil corrected", "q_{T} reweighted"], yRatio=yRatio)

        if doSysts:
            plotutils.plot_systs(groups, "recoil_corr_rec_perp", "recoil_syst", 'Zmumu', f"{outDir_}/unc_syst/", xMin=xMin, xMax=xMax, yMin=yMin, yMax=-1, xLabel="UPERP", rebin=rebin, extralabels=[met, "Recoil corrected"], yRatio=yRatio)
            plotutils.plot_systs(groups, "recoil_corr_rec_perp", "recoil_stat", 'Zmumu', f"{outDir_}/unc_stat/", xMin=xMin, xMax=xMax, yMin=yMin, yMax=-1, xLabel="UPERP", rebin=rebin, extralabels=[met, "Recoil corrected"], yRatio=yRatio)

        if extraRatios_:
            plotutils.stacked_plot_ratio(groups, "recoil_corr_rec_perp_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabel="UPERP", rebin=rebin, dataNormProc=dataNormProc, labels=[met, "Recoil corrected", "q_{T} reweighted"], yRatio=yRatio, suffix="_extraRatios", extraRatios=extraRatios)


        ################
        ## RECOIL PERPENDICULAR - log
        ################
        if analysis == "lowPU": xMin, xMax, yMin, yMax = -100, 100, 1e-1, 1e7
        else: xMin, xMax, yMin, yMax = -100, 100, 1e0, 1e9
        outDir_ = f"{outDir}/recoil_perp_log"
        rebin = 2

        plotutils.stacked_plot_ratio(groups, "recoil_uncorr_perp", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="UPERP", rebin=rebin, dataNormProc=dataNormProc, labels=[met, "Uncorrected"], yRatio=yRatio)
        plotutils.stacked_plot_ratio(groups, "recoil_corr_lep_perp", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="UPERP", rebin=rebin, dataNormProc=dataNormProc, labels=[met, "Lepton corrected"], yRatio=yRatio)
        plotutils.stacked_plot_ratio(groups, "recoil_corr_xy_perp", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="UPERP", rebin=rebin, dataNormProc=dataNormProc, labels=[met, "XY corrected"], yRatio=yRatio)
        plotutils.stacked_plot_ratio(groups, "recoil_corr_xy_perp_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="UPERP", rebin=rebin, dataNormProc=dataNormProc, labels=[met, "XY corrected", "q_{T} reweighted"], yRatio=yRatio)
        plotutils.stacked_plot_ratio(groups, "recoil_corr_rec_perp", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="UPERP", rebin=rebin, dataNormProc=dataNormProc, labels=[met, "Recoil corrected"], yRatio=yRatio)
        plotutils.stacked_plot_ratio(groups, "recoil_corr_rec_perp_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="UPERP", rebin=rebin, dataNormProc=dataNormProc, labels=[met, "Recoil corrected", "q_{T} reweighted"], yRatio=yRatio)

        if doSysts:
            plotutils.plot_systs(groups, "recoil_corr_rec_perp", "recoil_syst", 'Zmumu', f"{outDir_}/unc_syst/", xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="UPERP", rebin=rebin, extralabels=[met, "Recoil corrected"], yRatio=yRatio)
            plotutils.plot_systs(groups, "recoil_corr_rec_perp", "recoil_stat", 'Zmumu', f"{outDir_}/unc_stat/", xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="UPERP", rebin=rebin, extralabels=[met, "Recoil corrected"], yRatio=yRatio)

        if extraRatios_:
            plotutils.stacked_plot_ratio(groups, "recoil_corr_rec_perp_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="UPERP", rebin=rebin, dataNormProc=dataNormProc, labels=[met, "Recoil corrected", "q_{T} reweighted"], yRatio=yRatio, suffix="_extraRatios", extraRatios=extraRatios)


        ################
        ## RECOIL PARALLEL+QT
        ################
        if analysis == "lowPU": xMin, xMax, yMin, yMax = -50, 50, 0, 1.2e4
        else: xMin, xMax, yMin, yMax = -50, 50, 0, 0.6e6
        outDir_ = f"{outDir}/recoil_para_qt"
        rebin = 2

        plotutils.stacked_plot_ratio(groups, "recoil_uncorr_para_qt", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabel="UPARAQT", rebin=rebin, dataNormProc=dataNormProc, labels=[met, "Uncorrected"], yRatio=yRatio)
        plotutils.stacked_plot_ratio(groups, "recoil_corr_lep_para_qt", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabel="UPARAQT", rebin=rebin, dataNormProc=dataNormProc, labels=[met, "Lepton corrected"], yRatio=yRatio)
        plotutils.stacked_plot_ratio(groups, "recoil_corr_xy_para_qt", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabel="UPARAQT", rebin=rebin, dataNormProc=dataNormProc, labels=[met, "XY corrected"], yRatio=yRatio)
        plotutils.stacked_plot_ratio(groups, "recoil_corr_xy_para_qt_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabel="UPARAQT", rebin=rebin, dataNormProc=dataNormProc, labels=[met, "XY corrected", "q_{T} reweighted"], yRatio=yRatio)
        plotutils.stacked_plot_ratio(groups, "recoil_corr_rec_para_qt", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabel="UPARAQT", rebin=rebin, dataNormProc=dataNormProc, labels=[met, "Recoil corrected"], yRatio=yRatio)
        plotutils.stacked_plot_ratio(groups, "recoil_corr_rec_para_qt_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabel="UPARAQT", rebin=rebin, dataNormProc=dataNormProc, labels=[met, "Recoil corrected", "q_{T} reweighted"], yRatio=yRatio)

        if doSysts:
            plotutils.plot_systs(groups, "recoil_corr_rec_para_qt", "recoil_syst", 'Zmumu', f"{outDir_}/unc_syst/", xMin=xMin, xMax=xMax, yMin=yMin, yMax=-1, xLabel="UPARAQT", rebin=rebin, extralabels=[met, "Recoil corrected"], yRatio=yRatio)
            plotutils.plot_systs(groups, "recoil_corr_rec_para_qt", "recoil_stat", 'Zmumu', f"{outDir_}/unc_stat/", xMin=xMin, xMax=xMax, yMin=yMin, yMax=-1, xLabel="UPARAQT", rebin=rebin, extralabels=[met, "Recoil corrected"], yRatio=yRatio)

        if extraRatios_:
            plotutils.stacked_plot_ratio(groups, "recoil_corr_rec_para_qt_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabel="UPARAQT", rebin=rebin, dataNormProc=dataNormProc, labels=[met, "Recoil corrected", "q_{T} reweighted"], yRatio=yRatio, suffix="_extraRatios", extraRatios=extraRatios)


        ################
        ## RECOIL PARALLEL+QT - log
        ################
        if analysis == "lowPU": xMin, xMax, yMin, yMax = -100, 100, 1e-1, 1e7
        else: xMin, xMax, yMin, yMax = -100, 100, 1e0, 1e9
        outDir_ = f"{outDir}/recoil_para_qt_log"
        rebin = 2

        plotutils.stacked_plot_ratio(groups, "recoil_uncorr_para_qt", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="UPARAQT", rebin=rebin, dataNormProc=dataNormProc, labels=[met, "Uncorrected"], yRatio=yRatio)
        plotutils.stacked_plot_ratio(groups, "recoil_corr_lep_para_qt", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="UPARAQT", rebin=rebin, dataNormProc=dataNormProc, labels=[met, "Lepton corrected"], yRatio=yRatio)
        plotutils.stacked_plot_ratio(groups, "recoil_corr_xy_para_qt", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="UPARAQT", rebin=rebin, dataNormProc=dataNormProc, labels=[met, "XY corrected"], yRatio=yRatio)
        plotutils.stacked_plot_ratio(groups, "recoil_corr_xy_para_qt_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="UPARAQT", rebin=rebin, dataNormProc=dataNormProc, labels=[met, "XY corrected", "q_{T} reweighted"], yRatio=yRatio)
        plotutils.stacked_plot_ratio(groups, "recoil_corr_rec_para_qt", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="UPARAQT", rebin=rebin, dataNormProc=dataNormProc, labels=[met, "Recoil corrected"], yRatio=yRatio)
        plotutils.stacked_plot_ratio(groups, "recoil_corr_rec_para_qt_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="UPARAQT", rebin=rebin, dataNormProc=dataNormProc, labels=[met, "Recoil corrected", "q_{T} reweighted"], yRatio=yRatio)

        if doSysts:
            plotutils.plot_systs(groups, "recoil_corr_rec_para_qt", "recoil_syst", 'Zmumu', f"{outDir_}/unc_syst/", xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="UPARAQT", rebin=rebin, extralabels=[met, "Recoil corrected"], yRatio=yRatio)
            plotutils.plot_systs(groups, "recoil_corr_rec_para_qt", "recoil_stat", 'Zmumu', f"{outDir_}/unc_stat/", xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="UPARAQT", rebin=rebin, extralabels=[met, "Recoil corrected"], yRatio=yRatio)

        if extraRatios_:
            plotutils.stacked_plot_ratio(groups, "recoil_corr_rec_para_qt_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="UPARAQT", rebin=rebin, dataNormProc=dataNormProc, labels=[met, "Recoil corrected", "q_{T} reweighted"], yRatio=yRatio, suffix="_extraRatios", extraRatios=extraRatios)


        ################
        ## RECOIL MAGNITUDE
        ################
        if analysis == "lowPU": xMin, xMax, yMin, yMax = 0, 100, 0, 1e4
        else: xMin, xMax, yMin, yMax = 0, 100, 0, 6e5
        outDir_ = f"{outDir}/recoil_magn"
        plotutils.stacked_plot_ratio(groups, "recoil_uncorr_magn", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabel="UMAGN", rebin=rebin, dataNormProc=dataNormProc, labels=[met, "Uncorrected"], yRatio=yRatio)
        plotutils.stacked_plot_ratio(groups, "recoil_corr_lep_magn", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabel="UMAGN", rebin=rebin, dataNormProc=dataNormProc, labels=[met, "Lepton corrected"], yRatio=yRatio)
        plotutils.stacked_plot_ratio(groups, "recoil_corr_xy_magn", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabel="UMAGN", rebin=rebin, dataNormProc=dataNormProc, labels=[met, "XY corrected"], yRatio=yRatio)
        plotutils.stacked_plot_ratio(groups, "recoil_corr_xy_magn_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabel="UMAGN", rebin=rebin, dataNormProc=dataNormProc, labels=[met, "XY corrected", "q_{T} reweighted"], yRatio=yRatio)
        plotutils.stacked_plot_ratio(groups, "recoil_corr_rec_magn", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabel="UMAGN", rebin=rebin, dataNormProc=dataNormProc, labels=[met, "Recoil corrected"], yRatio=yRatio)
        plotutils.stacked_plot_ratio(groups, "recoil_corr_rec_magn_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabel="UMAGN", rebin=rebin, dataNormProc=dataNormProc, labels=[met, "Recoil corrected", "q_{T} reweighted"], yRatio=yRatio)

        if doSysts:
            plotutils.plot_systs(groups, "recoil_corr_rec_magn", "recoil_syst", 'Zmumu', f"{outDir_}/unc_syst/", xMin=xMin, xMax=xMax, yMin=yMin, yMax=-1, xLabel="UMAGN", rebin=rebin, extralabels=[met, "Recoil corrected"], yRatio=yRatio)
            plotutils.plot_systs(groups, "recoil_corr_rec_magn", "recoil_stat", 'Zmumu', f"{outDir_}/unc_stat/", xMin=xMin, xMax=xMax, yMin=yMin, yMax=-1, xLabel="UMAGN", rebin=rebin, extralabels=[met, "Recoil corrected"], yRatio=yRatio)

        if extraRatios_:
            plotutils.stacked_plot_ratio(groups, "recoil_corr_rec_magn_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabel="UMAGN", rebin=rebin, dataNormProc=dataNormProc, labels=[met, "Recoil corrected", "q_{T} reweighted"], yRatio=yRatio, suffix="_extraRatios", extraRatios=extraRatios)

        ################
        ## RECOIL MAGNITUDE - log
        ################
        if analysis == "lowPU": xMin, xMax, yMin, yMax = 0, 200, 1e-1, 1e7
        else: xMin, xMax, yMin, yMax = 0, 200, 1e0, 1e9
        outDir_ = f"{outDir}/recoil_magn_log"
        rebin = 2

        plotutils.stacked_plot_ratio(groups, "recoil_uncorr_magn", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="UMAGN", rebin=rebin, dataNormProc=dataNormProc, labels=[met, "Uncorrected"], yRatio=yRatio)
        plotutils.stacked_plot_ratio(groups, "recoil_corr_lep_magn", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="UMAGN", rebin=rebin, dataNormProc=dataNormProc, labels=[met, "Lepton corrected"], yRatio=yRatio)
        plotutils.stacked_plot_ratio(groups, "recoil_corr_xy_magn", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="UMAGN", rebin=rebin, dataNormProc=dataNormProc, labels=[met, "XY corrected"], yRatio=yRatio)
        plotutils.stacked_plot_ratio(groups, "recoil_corr_xy_magn_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="UMAGN", rebin=rebin, dataNormProc=dataNormProc, labels=[met, "XY corrected", "q_{T} reweighted"], yRatio=yRatio)
        plotutils.stacked_plot_ratio(groups, "recoil_corr_rec_magn", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="UMAGN", rebin=rebin, dataNormProc=dataNormProc, labels=[met, "Recoil corrected"], yRatio=yRatio)
        plotutils.stacked_plot_ratio(groups, "recoil_corr_rec_magn_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="UMAGN", rebin=rebin, dataNormProc=dataNormProc, labels=[met, "Recoil corrected", "q_{T} reweighted"], yRatio=yRatio)

        if doSysts:
            plotutils.plot_systs(groups, "recoil_corr_rec_magn", "recoil_syst", 'Zmumu', f"{outDir_}/unc_syst/", xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="UMAGN", rebin=rebin, extralabels=[met, "Recoil corrected"], yRatio=yRatio)
            plotutils.plot_systs(groups, "recoil_corr_rec_magn", "recoil_stat", 'Zmumu', f"{outDir_}/unc_stat/", xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="UMAGN", rebin=rebin, extralabels=[met, "Recoil corrected"], yRatio=yRatio)

        if extraRatios_:
            plotutils.stacked_plot_ratio(groups, "recoil_corr_rec_magn_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="UMAGN", rebin=rebin, dataNormProc=dataNormProc, labels=[met, "Recoil corrected", "q_{T} reweighted"], yRatio=yRatio, suffix="_extraRatios", extraRatios=extraRatios)

        ################
        ## TRANSVERSE MASS
        ################
        if analysis == "lowPU": xMin, xMax, yMin, yMax = 40, 120, 0, 0.8e4
        else: xMin, xMax, yMin, yMax = 40, 120, 0, 4e5
        outDir_ = f"{outDir}/mt"
        rebin = 1

        plotutils.stacked_plot_ratio(groups, "mt_uncorr", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabel="MT", rebin=rebin, dataNormProc=dataNormProc, labels=[met, "Uncorrected"], yRatio=yRatio)
        plotutils.stacked_plot_ratio(groups, "mt_corr_lep", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabel="MT", rebin=rebin, dataNormProc=dataNormProc, labels=[met, "Lepton corrected"], yRatio=yRatio)
        plotutils.stacked_plot_ratio(groups, "mt_corr_xy", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabel="MT", rebin=rebin, dataNormProc=dataNormProc, labels=[met, "XY corrected"], yRatio=yRatio)
        plotutils.stacked_plot_ratio(groups, "mt_corr_xy_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabel="MT", rebin=rebin, dataNormProc=dataNormProc, labels=[met, "XY corrected", "q_{T} reweighted"], yRatio=yRatio)
        plotutils.stacked_plot_ratio(groups, "mt_corr_rec", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabel="MT", rebin=rebin, dataNormProc=dataNormProc, labels=[met, "Recoil corrected"], yRatio=yRatio)
        plotutils.stacked_plot_ratio(groups, "mt_corr_rec_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabel="MT", rebin=rebin, dataNormProc=dataNormProc, labels=[met, "Recoil corrected", "q_{T} reweighted"], yRatio=yRatio)

        if doSysts:
            plotutils.plot_systs(groups, "mt_corr_rec", "recoil_syst", 'Zmumu', f"{outDir_}/unc_syst/", xMin=xMin, xMax=xMax, yMin=yMin, yMax=-1, xLabel="MT", rebin=rebin, extralabels=[met, "Recoil corrected"])
            plotutils.plot_systs(groups, "mt_corr_rec", "recoil_stat", 'Zmumu', f"{outDir_}/unc_stat/", xMin=xMin, xMax=xMax, yMin=yMin, yMax=-1, xLabel="MT", rebin=rebin, extralabels=[met, "Recoil corrected"])

        if extraRatios_:
            plotutils.stacked_plot_ratio(groups, "mt_corr_rec_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabel="MT", rebin=rebin, dataNormProc=dataNormProc, labels=[met, "XY corrected", "q_{T} reweighted"], yRatio=yRatio, suffix="_extraRatios", extraRatios=extraRatios)

        ################
        ## TRANSVERSE MASS - log
        ################
        if analysis == "lowPU": xMin, xMax, yMin, yMax = 0, 150, 1e-1, 1e7
        else: xMin, xMax, yMin, yMax = 0, 150, 1e0, 1e9
        outDir_ = f"{outDir}/mt_log"
        rebin = 2

        plotutils.stacked_plot_ratio(groups, "mt_uncorr", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="MT", rebin=rebin, dataNormProc=dataNormProc, labels=[met, "Uncorrected"], yRatio=yRatio)
        plotutils.stacked_plot_ratio(groups, "mt_corr_lep", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="MT", rebin=rebin, dataNormProc=dataNormProc, labels=[met, "Lepton corrected"], yRatio=yRatio)
        plotutils.stacked_plot_ratio(groups, "mt_corr_xy", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="MT", rebin=rebin, dataNormProc=dataNormProc, labels=[met, "XY corrected"], yRatio=yRatio)
        plotutils.stacked_plot_ratio(groups, "mt_corr_xy_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="MT", rebin=rebin, dataNormProc=dataNormProc, labels=[met, "XY corrected", "q_{T} reweighted"], yRatio=yRatio)
        plotutils.stacked_plot_ratio(groups, "mt_corr_rec", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="MT", rebin=rebin, dataNormProc=dataNormProc, labels=[met, "Recoil corrected"], yRatio=yRatio)
        plotutils.stacked_plot_ratio(groups, "mt_corr_rec_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="MT", rebin=rebin, dataNormProc=dataNormProc, labels=[met, "Recoil corrected", "q_{T} reweighted"], yRatio=yRatio)

        if doSysts:
            plotutils.plot_systs(groups, "mt_corr_rec", "recoil_syst", 'Zmumu', f"{outDir_}/unc_syst/", xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="MT", rebin=rebin, extralabels=[met, "Recoil corrected"])
            plotutils.plot_systs(groups, "mt_corr_rec", "recoil_stat", 'Zmumu', f"{outDir_}/unc_stat/", xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="MT", rebin=rebin, extralabels=[met, "Recoil corrected"])

        if extraRatios_:
            plotutils.stacked_plot_ratio(groups, "mt_corr_rec_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="MT", rebin=rebin, dataNormProc=dataNormProc, labels=[met, "XY corrected", "q_{T} reweighted"], yRatio=yRatio, suffix="_extraRatios", extraRatios=extraRatios)


        ################
        ## MET XY
        ################
        if analysis == "lowPU": xMin, xMax, yMin, yMax = -100, 100, 1e-1, 1e7
        else: xMin, xMax, yMin, yMax = -100, 100, 1e0, 1e9
        outDir_ = f"{outDir}/met_x_log"
        rebin = 2

        plotutils.stacked_plot_ratio(groups, "met_uncorr_x", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="METX", rebin=rebin, dataNormProc=dataNormProc, labels=[met, "Uncorrected"], yRatio=yRatio)
        plotutils.stacked_plot_ratio(groups, "met_corr_lep_x", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="METX", rebin=rebin, dataNormProc=dataNormProc, labels=[met, "Lepton corrected"], yRatio=yRatio)
        plotutils.stacked_plot_ratio(groups, "met_corr_xy_x", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="METX", rebin=rebin, dataNormProc=dataNormProc, labels=[met, "XY corrected"], yRatio=yRatio)
        plotutils.stacked_plot_ratio(groups, "met_corr_xy_x_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="METX", rebin=rebin, dataNormProc=dataNormProc, labels=[met, "XY corrected", "q_{T} reweighted"], yRatio=yRatio)
        plotutils.stacked_plot_ratio(groups, "met_corr_rec_x", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="METX", rebin=rebin, dataNormProc=dataNormProc, labels=[met, "Recoil corrected"], yRatio=yRatio)
        plotutils.stacked_plot_ratio(groups, "met_corr_rec_x_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="METX", rebin=rebin, dataNormProc=dataNormProc, labels=[met, "Recoil corrected", "q_{T} reweighted"], yRatio=yRatio)

        outDir_ = f"{outDir}/met_y_log"
        plotutils.stacked_plot_ratio(groups, "met_uncorr_y", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="METY", rebin=rebin, dataNormProc=dataNormProc, labels=[met, "Uncorrected"], yRatio=yRatio)
        plotutils.stacked_plot_ratio(groups, "met_corr_lep_y", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="METY", rebin=rebin, dataNormProc=dataNormProc, labels=[met, "Lepton corrected"], yRatio=yRatio)
        plotutils.stacked_plot_ratio(groups, "met_corr_xy_y", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="METY", rebin=rebin, dataNormProc=dataNormProc, labels=[met, "XY corrected"], yRatio=yRatio)
        plotutils.stacked_plot_ratio(groups, "met_corr_xy_y_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="METY", rebin=rebin, dataNormProc=dataNormProc, labels=[met, "XY corrected", "q_{T} reweighted"], yRatio=yRatio)
        plotutils.stacked_plot_ratio(groups, "met_corr_rec_y", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="METY", rebin=rebin, dataNormProc=dataNormProc, labels=[met, "Recoil corrected"], yRatio=yRatio)
        plotutils.stacked_plot_ratio(groups, "met_corr_rec_y_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="METY", rebin=rebin, dataNormProc=dataNormProc, labels=[met, "Recoil corrected", "q_{T} reweighted"], yRatio=yRatio)


        ################
        # qt
        ################
        if analysis == "lowPU": xMin, xMax, yMin, yMax = 0, 100, 1e-1, 1e7
        else: xMin, xMax, yMin, yMax = 0, 100, 1e0, 1e9
        outDir_ = f"{outDir}/plots/"
        vpt_bins = 1
        plotutils.stacked_plot_ratio(groups, "v_pt", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="QT", rebin=vpt_bins, dataNormProc=dataNormProc, labels=[met])
        plotutils.stacked_plot_ratio(groups, "v_pt_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="QT", rebin=vpt_bins, dataNormProc=dataNormProc, labels=[met, "q_{T} reweighted"])

        ################
        # number of (good) primary vertices
        ################
        if analysis == "lowPU":
            xMin, xMax, yMin, yMax = 0, 10, 10, 1e7
        else:
            xMin, xMax, yMin, yMax = 0, 50, 1e2, 1e8
        outDir_ = f"{outDir}/plots/"
        yRatio = 1.25
        plotutils.stacked_plot_ratio(groups, "npv", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="Number of primary vertices", dataNormProc=dataNormProc, yRatio=yRatio)

    else:

        extraRatios_ = False
        if extraRatios_:
            extraRatios = []
            extraRatios.append(("pre-fsr", expand_groups(datagroups.Datagroups("/scratch/submit/cms/jaeyserm/Recoil/mw_with_mu_eta_pt_DeepMETPVRobust_prefsr.hdf5"))))
            extraRatios.append(("post-fsr", expand_groups(datagroups.Datagroups("/scratch/submit/cms/jaeyserm/Recoil/mw_with_mu_eta_pt_DeepMETPVRobust_postfsr.hdf5"))))

        if analysis == "lowPU":
            yRatio = 1.15
        else:
            yRatio = 1.06

        ################
        # MET - no log
        ################
        if analysis == "lowPU": xMin, xMax, yMin, yMax = 0, 100, 1e1, 1e5
        else: xMin, xMax, yMin, yMax = 0, 100, 1e1, 0.7e7
        outDir_ = f"{outDir}/met_pt"
        bins_met = 1

        plotutils.stacked_plot_ratio(groups, "met_uncorr_pt", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabel="METPT", rebin=bins_met, dataNormProc=dataNormProc, labels=[met, "Uncorrected"], yRatio=yRatio, charge="combined")
        plotutils.stacked_plot_ratio(groups, "met_corr_lep_pt", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabel="METPT", rebin=bins_met, dataNormProc=dataNormProc, labels=[met, "Lepton corrected"], yRatio=yRatio, charge="combined")
        plotutils.stacked_plot_ratio(groups, "met_corr_xy_pt", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabel="METPT", rebin=bins_met, dataNormProc=dataNormProc, labels=[met, "XY corrected"], yRatio=yRatio, charge="combined")
        plotutils.stacked_plot_ratio(groups, "met_corr_xy_pt_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabel="METPT", rebin=bins_met, dataNormProc=dataNormProc, labels=[met, "XY corrected", "q_{T} reweighted"], yRatio=yRatio, charge="combined")
        plotutils.stacked_plot_ratio(groups, "met_corr_rec_pt", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabel="METPT", rebin=bins_met, dataNormProc=dataNormProc, labels=[met, "Recoil corrected"], yRatio=yRatio, charge="combined")
        plotutils.stacked_plot_ratio(groups, "met_corr_rec_pt_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabel="METPT", rebin=bins_met, dataNormProc=dataNormProc, labels=[met, "Recoil corrected", "q_{T} reweighted"], yRatio=yRatio, charge="combined")

        if doSysts:
            plotutils.plot_systs(groups, "met_corr_rec_pt", "recoil_syst", 'Wmunu', f"{outDir_}/unc_syst/", xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabel="METPT", rebin=bins_met, extralabels=[met, "Recoil corrected"])
            plotutils.plot_systs(groups, "met_corr_rec_pt", "recoil_stat", 'Wmunu', f"{outDir_}/unc_stat/", xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabel="METPT", rebin=bins_met, extralabels=[met, "Recoil corrected"])

        if extraRatios_:
            plotutils.stacked_plot_ratio(groups, "met_corr_rec_pt", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabel="METPT", rebin=bins_met, dataNormProc=dataNormProc, labels=[met, "Recoil corrected"], yRatio=yRatio, suffix="_extraRatios", extraRatios=extraRatios, charge="combined") # , extraRatios=extraRatios


        ################
        # MET - log
        ################
        if analysis == "lowPU": xMin, xMax, yMin, yMax = 0, 150, 1e1, 1e7
        else: xMin, xMax, yMin, yMax = 0, 200, 1e1, 1e9
        outDir_ = f"{outDir}/met_pt_log"
        bins_met = 2

        plotutils.stacked_plot_ratio(groups, "met_uncorr_pt", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="METPT", rebin=bins_met, dataNormProc=dataNormProc, labels=[met, "Uncorrected"], yRatio=yRatio, charge="combined")
        plotutils.stacked_plot_ratio(groups, "met_corr_lep_pt", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="METPT", rebin=bins_met, dataNormProc=dataNormProc, labels=[met, "Lepton corrected"], yRatio=yRatio, charge="combined")
        plotutils.stacked_plot_ratio(groups, "met_corr_xy_pt", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="METPT", rebin=bins_met, dataNormProc=dataNormProc, labels=[met, "XY corrected"], yRatio=yRatio, charge="combined")
        plotutils.stacked_plot_ratio(groups, "met_corr_xy_pt_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="METPT", rebin=bins_met, dataNormProc=dataNormProc, labels=[met, "XY corrected", "q_{T} reweighted"], yRatio=yRatio, charge="combined")
        plotutils.stacked_plot_ratio(groups, "met_corr_rec_pt", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="METPT", rebin=bins_met, dataNormProc=dataNormProc, labels=[met, "Recoil corrected"], yRatio=yRatio, charge="combined")
        plotutils.stacked_plot_ratio(groups, "met_corr_rec_pt_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="METPT", rebin=bins_met, dataNormProc=dataNormProc, labels=[met, "Recoil corrected", "q_{T} reweighted"], yRatio=yRatio, charge="combined")

        if doSysts:
            plotutils.plot_systs(groups, "met_corr_rec_pt", "recoil_syst", 'Wmunu', f"{outDir_}/unc_syst/", xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="METPT", rebin=bins_met, extralabels=[met, "Recoil corrected"])
            plotutils.plot_systs(groups, "met_corr_rec_pt", "recoil_stat", 'Wmunu', f"{outDir_}/unc_stat/", xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="METPT", rebin=bins_met, extralabels=[met, "Recoil corrected"])


        if extraRatios_:
            plotutils.stacked_plot_ratio(groups, "met_corr_rec_pt", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="METPT", rebin=bins_met, dataNormProc=dataNormProc, labels=[met, "Recoil corrected"], yRatio=yRatio, suffix="_extraRatios", extraRatios=extraRatios, charge="combined")



        ################
        # transverse mass - no log
        ################
        if analysis == "lowPU": xMin, xMax, yMin, yMax = 40, 120, 0, 1e5
        else: xMin, xMax, yMin, yMax = 40, 120, 0, 6e6
        outDir_ = f"{outDir}/mt"
        mt_bins = 1

        plotutils.stacked_plot_ratio(groups, "mt_uncorr", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabel="MT", rebin=mt_bins, dataNormProc=dataNormProc, labels=[met, "Uncorrected"], yRatio=yRatio, charge="combined")
        plotutils.stacked_plot_ratio(groups, "mt_corr_lep", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabel="MT", rebin=mt_bins, dataNormProc=dataNormProc, labels=[met, "Lepton corrected"], yRatio=yRatio, charge="combined")
        plotutils.stacked_plot_ratio(groups, "mt_corr_xy", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabel="MT", rebin=mt_bins, dataNormProc=dataNormProc, labels=[met, "XY corrected"], yRatio=yRatio, charge="combined")
        plotutils.stacked_plot_ratio(groups, "mt_corr_xy_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabel="MT", rebin=mt_bins, dataNormProc=dataNormProc, labels=[met, "XY corrected", "q_{T} reweighted"], yRatio=yRatio, charge="combined")
        plotutils.stacked_plot_ratio(groups, "mt_corr_rec", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabel="MT", rebin=mt_bins, dataNormProc=dataNormProc, labels=[met, "Recoil corrected"], yRatio=yRatio, charge="combined")
        plotutils.stacked_plot_ratio(groups, "mt_corr_rec_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabel="MT", rebin=mt_bins, dataNormProc=dataNormProc, labels=[met, "Recoil corrected", "q_{T} reweighted"], yRatio=yRatio, charge="combined")

        if doSysts:
            plotutils.plot_systs(groups, "mt_corr_rec", "recoil_syst", 'Wmunu', f"{outDir_}/unc_syst/", xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabel="MT", rebin=bins_met, extralabels=[met, "Recoil corrected"])
            plotutils.plot_systs(groups, "mt_corr_rec", "recoil_stat", 'Wmunu', f"{outDir_}/unc_stat/", xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabel="MT", rebin=bins_met, extralabels=[met, "Recoil corrected"])

        if extraRatios_:
            plotutils.stacked_plot_ratio(groups, "mt_corr_rec", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabel="MT", rebin=mt_bins, dataNormProc=dataNormProc, labels=[met, "Recoil corrected"], yRatio=yRatio, suffix="_extraRatios", extraRatios=extraRatios, charge="combined")

        ################
        # transverse mass - log
        ################
        if analysis == "lowPU": xMin, xMax, yMin, yMax = 40, 150, 1e1, 1e7
        else: xMin, xMax, yMin, yMax = 40, 150, 1e2, 1e10
        outDir_ = f"{outDir}/mt_log"
        mt_bins = 2

        plotutils.stacked_plot_ratio(groups, "mt_uncorr", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="MT", rebin=mt_bins, dataNormProc=dataNormProc, labels=[met, "Uncorrected"], yRatio=yRatio, charge="combined")
        plotutils.stacked_plot_ratio(groups, "mt_corr_lep", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="MT", rebin=mt_bins, dataNormProc=dataNormProc, labels=[met, "Lepton corrected"], yRatio=yRatio, charge="combined")
        plotutils.stacked_plot_ratio(groups, "mt_corr_xy", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="MT", rebin=mt_bins, dataNormProc=dataNormProc, labels=[met, "XY corrected"], yRatio=yRatio, charge="combined")
        plotutils.stacked_plot_ratio(groups, "mt_corr_xy_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="MT", rebin=mt_bins, dataNormProc=dataNormProc, labels=[met, "XY corrected", "q_{T} reweighted"], yRatio=yRatio, charge="combined")
        plotutils.stacked_plot_ratio(groups, "mt_corr_rec", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="MT", rebin=mt_bins, dataNormProc=dataNormProc, labels=[met, "Recoil corrected"], yRatio=yRatio, charge="combined")
        plotutils.stacked_plot_ratio(groups, "mt_corr_rec_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="MT", rebin=mt_bins, dataNormProc=dataNormProc, labels=[met, "Recoil corrected", "q_{T} reweighted"], yRatio=yRatio, charge="combined")


        if doSysts:
            plotutils.plot_systs(groups, "mt_corr_rec", "recoil_syst", 'Wmunu', f"{outDir_}/unc_syst/", xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="MT", rebin=bins_met, extralabels=[met, "Recoil corrected"])
            plotutils.plot_systs(groups, "mt_corr_rec", "recoil_stat", 'Wmunu', f"{outDir_}/unc_stat/", xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="MT", rebin=bins_met, extralabels=[met, "Recoil corrected"])


        if extraRatios_:
            plotutils.stacked_plot_ratio(groups, "mt_corr_rec", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="MT", rebin=mt_bins, dataNormProc=dataNormProc, labels=[met, "Recoil corrected"], yRatio=yRatio, suffix="_extraRatios", extraRatios=extraRatios, charge="combined")


        ################
        # lepton pt - log
        ################
        if analysis == "lowPU": xMin, xMax, yMin, yMax = 20, 70, 1e1, 1e7
        else: xMin, xMax, yMin, yMax = 20, 70, 0, 1e7
        outDir_ = f"{outDir}/lep_pt_log"

        plotutils.stacked_plot_ratio(groups, "lep_uncorr_pt", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabel="Lepton PT", rebin=1, dataNormProc=dataNormProc, labels=[met, "Uncorrected"], yRatio=yRatio, charge="combined")
        plotutils.stacked_plot_ratio(groups, "lep_corr_lep_pt", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabel="Lepton PT", rebin=1, dataNormProc=dataNormProc, labels=[met, "Lepton corrected"], yRatio=yRatio, charge="combined")
        plotutils.stacked_plot_ratio(groups, "lep_corr_rec_pt", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabel="Lepton PT", rebin=1, dataNormProc=dataNormProc, labels=[met, "Lepton corrected"], yRatio=yRatio, charge="combined")
        plotutils.stacked_plot_ratio(groups, "lep_corr_rec_pt_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabel="Lepton PT", rebin=1, dataNormProc=dataNormProc, labels=[met, "Lepton corrected"], yRatio=yRatio, charge="combined")


        ################
        # met phi - log
        ################
        if analysis == "lowPU": xMin, xMax, yMin, yMax = -4, 4, 1e2, 1e9
        else: xMin, xMax, yMin, yMax = -4, 4, 1e5, 1e10
        outDir_ = f"{outDir}/met_phi_log"
        plotutils.stacked_plot_ratio(groups, "met_uncorr_phi", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="METPHI", dataNormProc=dataNormProc, labels=[met, "Uncorrected"], yRatio=yRatio, charge="combined")
        plotutils.stacked_plot_ratio(groups, "met_corr_lep_phi", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="METPHI", dataNormProc=dataNormProc, labels=[met, "Lepton corrected"], yRatio=yRatio, charge="combined")
        plotutils.stacked_plot_ratio(groups, "met_corr_xy_phi", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="METPHI", dataNormProc=dataNormProc, labels=[met, "XY corrected"], yRatio=yRatio, charge="combined")
        plotutils.stacked_plot_ratio(groups, "met_corr_xy_phi_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="METPHI", dataNormProc=dataNormProc, labels=[met, "XY corrected", "q_{T} reweighted"], yRatio=yRatio, charge="combined")
        plotutils.stacked_plot_ratio(groups, "met_corr_rec_phi", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="METPHI", dataNormProc=dataNormProc, labels=[met, "Recoil corrected"], yRatio=yRatio, charge="combined")
        plotutils.stacked_plot_ratio(groups, "met_corr_rec_phi_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="METPHI", dataNormProc=dataNormProc, labels=[met, "Recoil corrected", "q_{T} reweighted"], yRatio=yRatio, charge="combined")



        # recoil magnitude
        if analysis == "lowPU": xMin, xMax, yMin, yMax = 0, 100, 0, 1.2e5
        else: xMin, xMax, yMin, yMax = 0, 100, 0, 1e7
        outDir_ = f"{outDir}/recoil_magn"

        plotutils.stacked_plot_ratio(groups, "recoil_uncorr_magn", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabel="UMAGN", dataNormProc=dataNormProc, labels=[met, "Uncorrected"], yRatio=yRatio, charge="combined")
        plotutils.stacked_plot_ratio(groups, "recoil_corr_lep_magn", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabel="UMAGN", dataNormProc=dataNormProc, labels=[met, "Lepton corrected"], yRatio=yRatio, charge="combined")
        plotutils.stacked_plot_ratio(groups, "recoil_corr_xy_magn", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabel="UMAGN", dataNormProc=dataNormProc, labels=[met, "XY corrected"], yRatio=yRatio, charge="combined")
        plotutils.stacked_plot_ratio(groups, "recoil_corr_xy_magn_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabel="UMAGN", dataNormProc=dataNormProc, labels=[met, "XY corrected", "q_{T} reweighted"], yRatio=yRatio, charge="combined")
        plotutils.stacked_plot_ratio(groups, "recoil_corr_rec_magn", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabel="UMAGN", dataNormProc=dataNormProc, labels=[met, "Recoil corrected"], yRatio=yRatio, charge="combined", blind=True)
        plotutils.stacked_plot_ratio(groups, "recoil_corr_rec_magn_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabel="UMAGN", dataNormProc=dataNormProc, labels=[met, "Recoil corrected", "q_{T} reweighted"], yRatio=yRatio, charge="combined")

        if extraRatios_:
            plotutils.stacked_plot_ratio(groups, "recoil_corr_rec_magn_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, xLabel="UMAGN", rebin=bins_met, dataNormProc=dataNormProc, labels=[met, "Recoil corrected"], yRatio=yRatio, suffix="_extraRatios", extraRatios=extraRatios, charge="combined")


        ################
        # recoil magnitude - log
        ################
        if analysis == "lowPU": xMin, xMax, yMin, yMax = 0, 150, 1e1, 1e7
        else: xMin, xMax, yMin, yMax = 0, 150, 1e2, 1e9
        outDir_ = f"{outDir}/recoil_magn_log"
        rebin = 2

        plotutils.stacked_plot_ratio(groups, "recoil_uncorr_magn", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="UMAGN", rebin=rebin, dataNormProc=dataNormProc, labels=[met, "Uncorrected"], yRatio=yRatio, charge="combined")
        plotutils.stacked_plot_ratio(groups, "recoil_corr_lep_magn", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="UMAGN", rebin=rebin, dataNormProc=dataNormProc, labels=[met, "Lepton corrected"], yRatio=yRatio, charge="combined")
        plotutils.stacked_plot_ratio(groups, "recoil_corr_xy_magn", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="UMAGN", rebin=rebin, dataNormProc=dataNormProc, labels=[met, "XY corrected"], yRatio=yRatio, charge="combined")
        plotutils.stacked_plot_ratio(groups, "recoil_corr_xy_magn_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="UMAGN", rebin=rebin, dataNormProc=dataNormProc, labels=[met, "XY corrected", "q_{T} reweighted"], yRatio=yRatio, charge="combined")
        plotutils.stacked_plot_ratio(groups, "recoil_corr_rec_magn", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="UMAGN", rebin=rebin, dataNormProc=dataNormProc, labels=[met, "Recoil corrected"], yRatio=yRatio, charge="combined", blind=True)
        plotutils.stacked_plot_ratio(groups, "recoil_corr_rec_magn_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="UMAGN", rebin=rebin, dataNormProc=dataNormProc, labels=[met, "Recoil corrected", "q_{T} reweighted"], yRatio=yRatio, charge="combined")

        if extraRatios_:
            plotutils.stacked_plot_ratio(groups, "recoil_corr_rec_magn_qtrw", procs, outDir_, xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, logY=True, xLabel="UMAGN", rebin=bins_met, dataNormProc=dataNormProc, labels=[met, "Recoil corrected"], yRatio=yRatio, suffix="_extraRatios", extraRatios=extraRatios, charge="combined")
