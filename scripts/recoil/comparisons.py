import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptTitle(0)

import functions
import plotutils

from wremnants.datasets import datagroups

if __name__ == "__main__":

    outDir = f"/home/submit/jaeyserm/public_html/met_lepton_correction_RawPFMET/"

    group = datagroups.Datagroups("mw_with_mu_eta_pt.hdf5")
    lumi_label = functions.getLumiLabel(group)

    proc = "Data"  # Wmunu Data

    xMin, xMax, yMin, yMax = 0.98, 1.02, 0, -1
    hists = ["lep_pt_uncorr_over_corr", "lep_pt_uncorr_over_corr"]
    labels = ["Data", "MC"]
    procs = ["Data", "Wmunu"]
    plotutils.plot_ratio(
        [group, group],
        hists,
        procs,
        labels,
        "lep_pt_uncorr_over_corr",
        outDir,
        xMin=xMin,
        xMax=xMax,
        yMin=yMin,
        yMax=yMax,
        xLabel="Uncorr/corr",
        lumi_label=lumi_label,
        yRatio=1.08,
    )

    xMin, xMax, yMin, yMax = 0.98, 1.02, 0, -1
    hists = ["met_pt_uncorr_over_corr_lep", "met_pt_uncorr_over_corr_lep"]
    labels = ["Data", "MC"]
    procs = ["Data", "Wmunu"]
    plotutils.plot_ratio(
        [group, group],
        hists,
        procs,
        labels,
        "met_pt_uncorr_over_corr_lep",
        outDir,
        xMin=xMin,
        xMax=xMax,
        yMin=yMin,
        yMax=yMax,
        xLabel="Uncorr/corr",
        lumi_label=lumi_label,
        yRatio=1.08,
    )

    xMin, xMax, yMin, yMax = -0.5, 0.5, 0, -1
    hists = ["lep_pt_uncorr_minus_corr", "lep_pt_uncorr_minus_corr"]
    labels = ["Data", "MC"]
    procs = ["Data", "Wmunu"]
    plotutils.plot_ratio(
        [group, group],
        hists,
        procs,
        labels,
        "lep_pt_uncorr_minus_corr",
        outDir,
        xMin=xMin,
        xMax=xMax,
        yMin=yMin,
        yMax=yMax,
        xLabel="#Deltap_{T}(uncorr-corr)",
        lumi_label=lumi_label,
        yRatio=1.08,
        rebin=100,
    )

    xMin, xMax, yMin, yMax = -0.5, 0.5, 0, -1
    hists = ["met_pt_uncorr_minus_corr_lep", "met_pt_uncorr_minus_corr_lep"]
    labels = ["Data", "MC"]
    procs = ["Data", "Wmunu"]
    plotutils.plot_ratio(
        [group, group],
        hists,
        procs,
        labels,
        "met_pt_uncorr_minus_corr_lep",
        outDir,
        xMin=xMin,
        xMax=xMax,
        yMin=yMin,
        yMax=yMax,
        xLabel="#Deltap_{T}(uncorr-corr)",
        lumi_label=lumi_label,
        yRatio=1.08,
        rebin=100,
    )

    xMin, xMax, yMin, yMax = -0.001, 0.001, 0, -1
    hists = ["lep_phi_uncorr_minus_corr", "lep_phi_uncorr_minus_corr"]
    labels = ["Data", "MC"]
    procs = ["Data", "Wmunu"]
    plotutils.plot_ratio(
        [group, group],
        hists,
        procs,
        labels,
        "lep_phi_uncorr_minus_corr",
        outDir,
        xMin=xMin,
        xMax=xMax,
        yMin=yMin,
        yMax=yMax,
        xLabel="#Delta#phi(uncorr-corr) (-0.001, 0.001)",
        lumi_label=lumi_label,
        yRatio=1.08,
    )

    xMin, xMax, yMin, yMax = -0.001, 0.001, 0, -1
    hists = ["met_phi_uncorr_minus_corr_lep", "met_phi_uncorr_minus_corr_lep"]
    labels = ["Data", "MC"]
    procs = ["Data", "Wmunu"]
    plotutils.plot_ratio(
        [group, group],
        hists,
        procs,
        labels,
        "met_phi_uncorr_minus_corr_lep",
        outDir,
        xMin=xMin,
        xMax=xMax,
        yMin=yMin,
        yMax=yMax,
        xLabel="#Delta#phi(uncorr-corr) (-0.001, 0.001)",
        lumi_label=lumi_label,
        yRatio=1.08,
    )

    xMin, xMax, yMin, yMax = 0, 100, 0, -1
    hists = ["met_uncorr_pt", "met_corr_lep_pt"]
    labels = ["Uncorrected (mc)", "Lepton-corrected (mc)"]
    procs = ["Wmunu", "Wmunu"]
    plotutils.plot_ratio(
        [group, group],
        hists,
        procs,
        labels,
        "met_mc",
        outDir,
        xMin=xMin,
        xMax=xMax,
        yMin=yMin,
        yMax=yMax,
        xLabel="METPT",
        yRatio=1.08,
        norm=True,
    )

    xMin, xMax, yMin, yMax = 0, 100, 0, -1
    hists = ["met_uncorr_pt", "met_corr_lep_pt"]
    labels = ["Uncorrected (data)", "Lepton-corrected (data)"]
    procs = ["Data", "Data"]
    plotutils.plot_ratio(
        [group, group],
        hists,
        procs,
        labels,
        "met_data",
        outDir,
        xMin=xMin,
        xMax=xMax,
        yMin=yMin,
        yMax=yMax,
        xLabel="METPT",
        yRatio=1.08,
        norm=True,
    )

    xMin, xMax, yMin, yMax = 40, 120, 0, -1
    hists = ["mt_uncorr", "mt_corr_lep"]
    labels = ["Uncorrected (mc)", "Lepton-corrected (mc)"]
    procs = ["Wmunu", "Wmunu"]
    plotutils.plot_ratio(
        [group, group],
        hists,
        procs,
        labels,
        "mt_mc",
        outDir,
        xMin=xMin,
        xMax=xMax,
        yMin=yMin,
        yMax=yMax,
        xLabel="MT",
        yRatio=1.08,
        norm=True,
    )

    xMin, xMax, yMin, yMax = 40, 120, 0, -1
    hists = ["mt_uncorr", "mt_corr_lep"]
    labels = ["Uncorrected (data)", "Lepton-corrected (data)"]
    procs = ["Data", "Data"]
    plotutils.plot_ratio(
        [group, group],
        hists,
        procs,
        labels,
        "mt_data",
        outDir,
        xMin=xMin,
        xMax=xMax,
        yMin=yMin,
        yMax=yMax,
        xLabel="MT",
        yRatio=1.08,
        norm=True,
    )

    quit()

    outDir = f"/home/submit/jaeyserm/public_html/test/"

    group_new = datagroups.Datagroups("mw_with_mu_eta_pt.hdf5")
    group_old = datagroups.Datagroups("../old/WRemnants/mw_with_mu_eta_pt.hdf5")
    # group_old = datagroups.Datagroups("mw_with_mu_eta_pt.hdf5")

    xMin, xMax, yMin, yMax = 40, 120, 0, -1
    hists = ["mt_uncorr", "mT_uncorr"]
    labels = ["New", "Old"]
    procs = ["Wmunu", "Wmunu"]
    plotutils.plot_ratio(
        [group_new, group_old],
        hists,
        procs,
        labels,
        "mt",
        outDir,
        xMin=xMin,
        xMax=xMax,
        yMin=yMin,
        yMax=yMax,
        xLabel="MT",
        yRatio=1.08,
        norm=True,
    )

    xMin, xMax, yMin, yMax = 0, 100, 0, -1
    hists = ["MET_corr_lep_pt_test", "MET_corr_lep_pt_test"]
    labels = ["New", "Old"]
    procs = ["Wmunu", "Wmunu"]
    plotutils.plot_ratio(
        [group_new, group_old],
        hists,
        procs,
        labels,
        "met",
        outDir,
        xMin=xMin,
        xMax=xMax,
        yMin=yMin,
        yMax=yMax,
        xLabel="MT",
        yRatio=1.08,
        norm=True,
    )

    xMin, xMax, yMin, yMax = -4, 4, 0, -1
    hists = ["MET_corr_lep_phi_test", "MET_corr_lep_phi_test"]
    labels = ["New", "Old"]
    procs = ["Wmunu", "Wmunu"]
    plotutils.plot_ratio(
        [group_new, group_old],
        hists,
        procs,
        labels,
        "phi",
        outDir,
        xMin=xMin,
        xMax=xMax,
        yMin=yMin,
        yMax=yMax,
        xLabel="METPHI",
        yRatio=1.08,
        norm=True,
    )

    quit()

    outDir = f"/home/submit/jaeyserm/public_html/recoil/highPU_DeepMETReso/gen_reco/"

    group_def = datagroups.Datagroups("mz_wlike_with_mu_eta_pt_DeepMETReso.hdf5")
    group = datagroups.Datagroups("mz_wlike_with_mu_eta_pt_postfsr.hdf5")
    lumi_label = functions.getLumiLabel(group)

    xMin, xMax, yMin, yMax = 0, 50, 0, 6e5
    hists = [
        "v_pt",
        "v_gen_pt_prefsr",
        "v_gen_pt_postfsr",
        "v_gen_pt_proxy_prefsr",
        "v_gen_pt_proxy_postfsr",
    ]
    labels = [
        "Reco",
        "Gen (preFSR)",
        "Gen (postFSR)",
        "Gen (proxy preFSR)",
        "Gen (proxy postFSR)",
    ]
    procs = ["Zmumu", "Zmumu", "Zmumu", "Zmumu", "Zmumu"]
    plotutils.plot_ratio(
        [group, group, group, group, group],
        hists,
        procs,
        labels,
        "v_pt_gen_reco",
        outDir,
        xMin=xMin,
        xMax=xMax,
        yMin=yMin,
        yMax=yMax,
        xLabel="QT",
        lumi_label=lumi_label,
        yRatio=1.08,
    )

    xMin, xMax, yMin, yMax = 0.9, 1.1, 0, -1
    hists = [
        "reco_over_gen_v_pt_prefsr",
        "reco_over_gen_v_pt_postfsr",
        "reco_over_gen_v_pt_proxy_prefsr",
        "reco_over_gen_v_pt_proxy_postfsr",
    ]
    labels = [
        "Gen (preFSR)",
        "Gen (postFSR)",
        "Gen (proxy preFSR)",
        "Gen (proxy postFSR)",
    ]
    procs = ["Zmumu", "Zmumu", "Zmumu", "Zmumu"]
    plotutils.plot_ratio(
        [group, group, group, group],
        hists,
        procs,
        labels,
        "reco_over_gen_v_pt",
        outDir,
        xMin=xMin,
        xMax=xMax,
        yMin=yMin,
        yMax=yMax,
        xLabel="q_{T}(reco)/q_{T}(gen)",
        lumi_label=lumi_label,
        rebin=10,
        doRatio=False,
    )

    xMin, xMax, yMin, yMax = -1, 1, 0, -1
    hists = [
        "reco_minus_gen_v_pt_prefsr",
        "reco_minus_gen_v_pt_postfsr",
        "reco_minus_gen_v_pt_proxy_prefsr",
        "reco_minus_gen_v_pt_proxy_postfsr",
    ]
    labels = [
        "Gen (preFSR)",
        "Gen (postFSR)",
        "Gen (proxy preFSR)",
        "Gen (proxy postFSR)",
    ]
    procs = ["Zmumu", "Zmumu", "Zmumu", "Zmumu"]
    plotutils.plot_ratio(
        [group, group, group, group],
        hists,
        procs,
        labels,
        "reco_minus_gen_v_pt",
        outDir,
        xMin=xMin,
        xMax=xMax,
        yMin=yMin,
        yMax=yMax,
        xLabel="q_{T}(reco) #minus q_{T}(gen)",
        lumi_label=lumi_label,
        rebin=20,
        doRatio=False,
    )

    xMin, xMax, yMin, yMax = -0.1, 0.1, 0, -1
    hists = [
        "reco_minus_gen_v_phi_prefsr",
        "reco_minus_gen_v_phi_postfsr",
        "reco_minus_gen_v_phi_proxy_prefsr",
        "reco_minus_gen_v_phi_proxy_postfsr",
    ]
    labels = [
        "Gen (preFSR)",
        "Gen (postFSR)",
        "Gen (proxy preFSR)",
        "Gen (proxy postFSR)",
    ]
    procs = ["Zmumu", "Zmumu", "Zmumu", "Zmumu"]
    plotutils.plot_ratio(
        [group, group, group, group],
        hists,
        procs,
        labels,
        "reco_minus_gen_v_phi",
        outDir,
        xMin=xMin,
        xMax=xMax,
        yMin=yMin,
        yMax=yMax,
        xLabel="#phi(reco) #minus #phi(gen)",
        lumi_label=lumi_label,
        rebin=10,
        doRatio=False,
    )

    plotutils.stacked_2d(
        group,
        "v_gen_reco_pt_prefsr",
        "Zmumu",
        outDir,
        xMin=0,
        xMax=30,
        yMin=0,
        yMax=30,
        xLabel="q_{T}(gen, preFSR)",
        yLabel="q_{T}(reco)",
        logZ=True,
    )
    plotutils.stacked_2d(
        group,
        "v_gen_reco_pt_postfsr",
        "Zmumu",
        outDir,
        xMin=0,
        xMax=30,
        yMin=0,
        yMax=30,
        xLabel="q_{T}(gen, postFSR)",
        yLabel="q_{T}(reco)",
        logZ=True,
    )
    plotutils.stacked_2d(
        group,
        "v_gen_reco_pt_proxy_prefsr",
        "Zmumu",
        outDir,
        xMin=0,
        xMax=30,
        yMin=0,
        yMax=30,
        xLabel="q_{T}(gen, proxy preFSR)",
        yLabel="q_{T}(reco)",
        logZ=True,
    )
    plotutils.stacked_2d(
        group,
        "v_gen_reco_pt_proxy_postfsr",
        "Zmumu",
        outDir,
        xMin=0,
        xMax=30,
        yMin=0,
        yMax=30,
        xLabel="q_{T}(gen, proxy postFSR)",
        yLabel="q_{T}(reco)",
        logZ=True,
    )

    xMin, xMax, yMin, yMax = 0, 200, 0, 0.3e6
    hists = ["mt_corr_rec", "mt_corr_rec"]
    labels = ["Reco", "Gen (proxy postFSR)"]
    procs = ["Zmumu", "Zmumu"]
    plotutils.plot_ratio(
        [group_def, group],
        hists,
        procs,
        labels,
        "mt_corr_rec",
        outDir,
        xMin=xMin,
        xMax=xMax,
        yMin=yMin,
        yMax=yMax,
        xLabel="MT",
        lumi_label=lumi_label,
        yRatio=1.08,
    )

    xMin, xMax, yMin, yMax = -80, 80, 0, 0.5e6
    hists = ["recoil_corr_rec_perp", "recoil_corr_rec_perp"]
    labels = ["Reco", "Gen (proxy postFSR)"]
    procs = ["Zmumu", "Zmumu"]
    plotutils.plot_ratio(
        [group_def, group],
        hists,
        procs,
        labels,
        "recoil_corr_rec_perp",
        outDir,
        xMin=xMin,
        xMax=xMax,
        yMin=yMin,
        yMax=yMax,
        xLabel="UPERP",
        lumi_label=lumi_label,
        yRatio=1.08,
    )

    xMin, xMax, yMin, yMax = -80, 80, 0, 0.5e6
    hists = ["recoil_corr_rec_para", "recoil_corr_rec_para"]
    labels = ["Reco", "Gen (proxy postFSR)"]
    procs = ["Zmumu", "Zmumu"]
    plotutils.plot_ratio(
        [group_def, group],
        hists,
        procs,
        labels,
        "recoil_corr_rec_para",
        outDir,
        xMin=xMin,
        xMax=xMax,
        yMin=yMin,
        yMax=yMax,
        xLabel="UPARA",
        lumi_label=lumi_label,
        yRatio=1.08,
    )
