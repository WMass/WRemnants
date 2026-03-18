"""
python scripts/histmakers/kaon_response.py --era 2018 --dataPath '/scratch/submit/cms/zmass/' --filterProcs BuToJpsiK
"""

import os

import hist
import matplotlib
import numpy as np
import ROOT

import narf
from wums import logging

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wremnants.production import btojpsik_selections

narf.clingutils.Declare('#include "muon_calibration.hpp"')
from wremnants.production.datasets.dataset_tools import getDatasets
from wremnants.production.histmaker_tools import write_analysis_output
from wremnants.utilities import common, parsing

analysis_label = common.analysis_label(os.path.basename(__file__))
parser, initargs = parsing.common_parser(analysis_label)

logger = logging.setup_logger(__file__)

parser.add_argument(
    "--ptCutoff",
    type=float,
    default=-1.0,
    help="pT cutoff for input histograms to make_muon_response_maps.py",
)
parser.add_argument(
    "--testHelpers", action="store_true", help="test diff weights helper"
)


plot_dir = "/home/submit/pmlugato/public_html/mz/calibration/dweightdscale_checks/"
tflite_postfix = ""
corr_file = common.calib_filepaths["data_corrfile"]["lbl_massfit"]
plot_all_kin_bins = False
weight_hist_bins = 200
weight_hist_range = (-0.2, 0.2)


args = parser.parse_args()

era = args.era

datasets = getDatasets(
    maxFiles=args.maxFiles,
    filt=args.filterProcs,
    excl=args.excludeProcs,
    nanoVersion="v9",
    base_path=args.dataPath,
    # extended="msht20an3lo" not in args.pdfs,
    era=era,
)

axis_genPt = hist.axis.Regular(7, 1.0, 8.0, name="genPt")
axis_genEta = hist.axis.Regular(28, -1.4, 1.4, name="genEta")
axis_genCharge = hist.axis.Regular(
    2, -2.0, 2.0, underflow=False, overflow=False, name="genCharge"
)
axis_qopr = hist.axis.Regular(501, 0.0, 2.0, name="qopr")
axis_curvature = hist.axis.Regular(50, 0.0, 1.0, name="curvature")
axis_dweightdqop = hist.axis.Regular(200, -100.0, 100.0, name="dweightdqop")

response_axes = [axis_genPt, axis_genEta, axis_genCharge, axis_qopr]

sigmarel = 5e-3
scalerel = 5e-4
nreps = 100

smearing_helper_simple_multi = ROOT.wrem.SmearingHelperSimpleMulti[nreps](sigmarel)

if args.testHelpers:
    response_helper = ROOT.wrem.SplinesDifferentialWeightsHelper(
        f"/ceph/submit/data/user/p/pmlugato/mz/calibration/kaon_response_{tflite_postfix}.tflite"
    )


def plot_weight_variation_hists(resultdict, plot_dir):
    hist_dweight = None
    for dataset_name, payload in resultdict.items():
        output = payload.get("output", {})
        if "hist_dweightdqop" in output:
            hist_dweight = output["hist_dweightdqop"].get()
            logger.info("Plotting weight variation hists from dataset %s", dataset_name)
            break
    if hist_dweight is None:
        logger.warning("hist_dweightdqop not found in outputs; skipping weight plots.")
        return

    if not corr_file:
        logger.warning("No corrFile provided; skipping weight plots.")
        return
    f = ROOT.TFile.Open(corr_file)
    if not f or f.IsZombie():
        logger.warning("Could not open corrFile %s; skipping weight plots.", corr_file)
        return

    A = f.Get("A")
    e = f.Get("e")
    M = f.Get("M")
    if not A or not e or not M:
        logger.warning("Missing A/e/M in corrFile; skipping weight plots.")
        f.Close()
        return
    A = narf.root_to_hist(A, axis_names=["scale_eta"])
    e = narf.root_to_hist(e, axis_names=["scale_eta"])
    M = narf.root_to_hist(M, axis_names=["scale_eta"])
    f.Close()

    A_unc = (
        np.sqrt(A.variances())
        if A.variances() is not None
        else np.zeros_like(A.values())
    )
    e_unc = (
        np.sqrt(e.variances())
        if e.variances() is not None
        else np.zeros_like(e.values())
    )
    M_unc = (
        np.sqrt(M.variances())
        if M.variances() is not None
        else np.zeros_like(M.values())
    )

    axis_pt = hist_dweight.axes["genPt"]
    axis_eta = hist_dweight.axes["genEta"]
    axis_charge = hist_dweight.axes["genCharge"]
    axis_curv = hist_dweight.axes["curvature"]
    axis_dw = hist_dweight.axes["dweightdqop"]

    eta_centers = axis_eta.centers
    corr_eta_centers = A.axes["scale_eta"].centers
    A_unc_interp = np.interp(eta_centers, corr_eta_centers, A_unc)
    e_unc_interp = np.interp(eta_centers, corr_eta_centers, e_unc)
    M_unc_interp = np.interp(eta_centers, corr_eta_centers, M_unc)

    theta = 2.0 * np.arctan(np.exp(-eta_centers))
    sin_theta = np.sin(theta)

    curv_centers = axis_curv.centers
    dw_centers = axis_dw.centers
    weight_edges = np.linspace(
        weight_hist_range[0], weight_hist_range[1], weight_hist_bins + 1
    )

    values = hist_dweight.values()
    # shape: [pt, eta, charge, curvature, dweightdqop]
    pt_indices = range(axis_pt.size) if plot_all_kin_bins else [0]
    eta_indices = range(axis_eta.size) if plot_all_kin_bins else [0]
    charge_indices = range(axis_charge.size) if plot_all_kin_bins else [0]

    os.makedirs(plot_dir, exist_ok=True)

    dw_sum = np.sum(values, axis=4)
    dw_mean = np.divide(
        np.sum(values * dw_centers[None, None, None, None, :], axis=4),
        dw_sum,
        out=np.zeros_like(dw_sum, dtype=float),
        where=dw_sum != 0,
    )

    for pt_idx in pt_indices:
        for eta_idx in eta_indices:
            sin_eta = sin_theta[eta_idx]
            k = curv_centers
            kunc_A = A_unc_interp[eta_idx] * k
            kunc_e = -e_unc_interp[eta_idx] * k * k
            kunc_M = M_unc_interp[eta_idx] * np.ones_like(k)

            for charge_idx in charge_indices:
                charge_val = axis_charge.centers[charge_idx]
                slice_vals = values[pt_idx, eta_idx, charge_idx, :, :]
                if not np.any(slice_vals):
                    continue

                fig, ax = plt.subplots()
                ax.plot(curv_centers, dw_mean[pt_idx, eta_idx, charge_idx, :])
                ax.set_xlabel("curvature (1/pt)")
                ax.set_ylabel("mean dweight/dqop")
                ax.set_title(
                    f"dweight/dqop vs curvature (ptidx={pt_idx}, etaidx={eta_idx}, charge={charge_val:g})"
                )
                fig.tight_layout()
                fig.savefig(
                    os.path.join(
                        plot_dir,
                        f"bin{pt_idx}_{eta_idx}_{charge_idx}_dweightdqop_mean.png",
                    )
                )
                plt.close(fig)

                for label, kunc in [("A", kunc_A), ("e", kunc_e), ("M", kunc_M)]:
                    kunc_use = kunc * charge_val if label == "M" else kunc
                    kprime_over_k = 1.0 + np.divide(
                        kunc_use,
                        k,
                        out=np.zeros_like(kunc_use, dtype=float),
                        where=k != 0,
                    )
                    fig, ax = plt.subplots()
                    ax.plot(curv_centers, kprime_over_k)
                    ax.set_xlabel("curvature (1/pt)")
                    ax.set_ylabel("k'/k")
                    ax.set_title(
                        f"k'/k {label} (ptidx={pt_idx}, etaidx={eta_idx}, charge={charge_val:g})"
                    )
                    fig.tight_layout()
                    fig.savefig(
                        os.path.join(
                            plot_dir,
                            f"bin{pt_idx}_{eta_idx}_{charge_idx}_kprime_over_k_{label}.png",
                        )
                    )
                    plt.close(fig)

                    delta_qop = charge_val * sin_eta * kunc
                    if label == "M":
                        delta_qop = sin_eta * kunc_M

                    counts = np.zeros(weight_hist_bins, dtype=float)
                    for curv_idx, dq in enumerate(delta_qop):
                        weights = dw_centers * dq
                        counts += np.histogram(
                            weights,
                            bins=weight_edges,
                            weights=slice_vals[curv_idx, :],
                        )[0]

                    fig, ax = plt.subplots()
                    centers = 0.5 * (weight_edges[1:] + weight_edges[:-1])
                    ax.step(centers, counts, where="mid")
                    ax.set_xlabel("dweight/dscale * delta(qop)")
                    ax.set_ylabel("counts")
                    ax.set_title(
                        f"weight distribution {label} (ptidx={pt_idx}, etaidx={eta_idx}, charge={charge_val:g})"
                    )
                    fig.tight_layout()
                    fig.savefig(
                        os.path.join(
                            plot_dir,
                            f"bin{pt_idx}_{eta_idx}_{charge_idx}_weight_dist_{label}.png",
                        )
                    )
                    plt.close(fig)


def build_graph(df, dataset):
    logger.info(f"build graph for dataset: {dataset.name}")
    results = []

    if dataset.is_data:
        df = df.DefinePerSample("weight", "1.0")
    else:
        df = df.Define("weight", "std::copysign(1.0, genWeight)")
        df = df.Define("nominal_weight", "static_cast<double>(weight)")

    weightsum = df.SumAndCount("weight")

    if dataset.is_data:
        df = df.DefinePerSample("nominal_weight", "1.0")

    df, _ = btojpsik_selections.define_jpsi_triggers(
        df, trigger_name="DoubleMu4_3_Jpsi"
    )

    bkmm_selections = [
        # (
        #    "dimuon cand neutral",
        #    lambda d: btojpsik_selections.select_opposite_sign_dimuon(d),
        # ),
        # (
        #    "muon |eta| < 1.4",
        #    lambda d: btojpsik_selections.select_muon_eta(d, 1.4),
        # ),
        # (
        #    "muon pT > 4",
        #    lambda d: btojpsik_selections.select_muon_pt(d, 4),
        # ),
        # (
        #    "muon softMVA > 0.45",
        #    lambda d: btojpsik_selections.select_muon_softmva(d, 0.45),
        # ),
        # (
        #    "dimuon pT > 7",
        #    lambda d: btojpsik_selections.select_dimuon_pt(d, 7.0),
        # ),
        # (
        #    "dimuon alphaBS < 0.4",
        #    lambda d: btojpsik_selections.select_dimuon_alphabs(d, 0.4),
        # ),  # og 0.4
        # (
        #    "dimuon vtx prob > 0.1",
        #    lambda d: btojpsik_selections.select_dimuon_vtx_prob(d, 0.1),
        # ),  # og 0.1
        # (
        #    "dimuon sl3d > 4",
        #    lambda d: btojpsik_selections.select_dimuon_sl3d(d, 4),
        # ),  # og 4
        # (
        #    "bkmm vtx prob > 0.3",
        #    lambda d: btojpsik_selections.select_bkmm_vtx_prob(d, 0.3),
        # ),  # og 0.025
        # (
        #    "bkmm mass window",
        #    lambda d: btojpsik_selections.select_bkmm_mass_window(d, 5.3, 0.1),
        # ),  # og 5.4, 0.5
        (
            "kaon pT < 8",
            lambda d: btojpsik_selections.select_kaon_pt(d, 8),
        ),
        # (
        #   "bkmm bmm bdt output > 0.10",
        #    lambda d: btojpsik_selections.select_bkmm_bmm_bdt(d, 0.10)
        # ) # NOTE: this doesn't touch kaon
    ]

    df, _, _ = btojpsik_selections.bkmm_selections(df, dataset.name, bkmm_selections)

    df = btojpsik_selections.select_only_passing_bkmm_candidates(
        df,
        signal=dataset.name == "signalBuToJpsiK_2018",
        select_best=True,
        gen_match_nonsignal=not dataset.is_data
        and dataset.name != "signalBuToJpsiK_2018",
    )

    reco_sel_GF = "selKaon"
    if not dataset.is_data:
        # temp vars defined during selection of best cand
        df = df.Alias(f"{reco_sel_GF}_genPt", "temp_kaon_genPt")
        df = df.Alias(f"{reco_sel_GF}_genEta", "temp_kaon_genEta")
        df = df.Alias(f"{reco_sel_GF}_genCharge", "temp_kaon_genCharge")
        df = df.Alias(f"{reco_sel_GF}_recoPt", "bkmm_kaon_pt")
        df = df.Alias(f"{reco_sel_GF}_recoEta", "bkmm_kaon_eta")
        df = df.Alias(f"{reco_sel_GF}_recoCharge", "bkmm_kaon_charge")

    df = df.Define(
        f"{reco_sel_GF}_qop",
        f"{reco_sel_GF}_recoCharge*1.0/({reco_sel_GF}_recoPt*cosh({reco_sel_GF}_recoEta))",
    )
    df = df.Define(
        f"{reco_sel_GF}_genQop",
        f"{reco_sel_GF}_genCharge*1.0/({reco_sel_GF}_genPt*cosh({reco_sel_GF}_genEta))",
    )
    df = df.Define(f"{reco_sel_GF}_qopr", f"{reco_sel_GF}_qop/{reco_sel_GF}_genQop")
    # import pdb
    # pdb.set_trace()

    # nominal
    response_cols = [
        f"{reco_sel_GF}_genPt",
        f"{reco_sel_GF}_genEta",
        f"{reco_sel_GF}_genCharge",
        f"{reco_sel_GF}_qopr",
    ]
    hist_qopr = df.HistoBoost(
        "hist_qopr", response_axes, [*response_cols, "nominal_weight"]
    )
    results.append(hist_qopr)

    # shift
    df = df.Define(
        f"{reco_sel_GF}_shiftedqopr", f"(1. + {scalerel})*{reco_sel_GF}_qopr"
    )
    response_cols_shifted = [
        f"{reco_sel_GF}_genPt",
        f"{reco_sel_GF}_genEta",
        f"{reco_sel_GF}_genCharge",
        f"{reco_sel_GF}_shiftedqopr",
    ]
    hist_qopr_shifted = df.HistoBoost(
        "hist_qopr_shifted",
        response_axes,
        [*response_cols_shifted, "nominal_weight"],
    )
    hist_qopr_shifted._hist.metadata = {"scalerel": scalerel}
    results.append(hist_qopr_shifted)

    # smear
    df = df.Define(
        f"{reco_sel_GF}Multi_smearedmqop",
        smearing_helper_simple_multi,
        [
            "run",
            "luminosityBlock",
            "event",
            f"{reco_sel_GF}_recoPt",
            f"{reco_sel_GF}_recoEta",
            f"{reco_sel_GF}_recoCharge",
        ],
    )
    df = df.Define(
        f"{reco_sel_GF}Multi_genPt",
        f"wrem::replicate_rvec({reco_sel_GF}_genPt, {nreps})",
    )
    df = df.Define(
        f"{reco_sel_GF}Multi_genEta",
        f"wrem::replicate_rvec({reco_sel_GF}_genEta, {nreps})",
    )
    df = df.Define(
        f"{reco_sel_GF}Multi_genCharge",
        f"wrem::replicate_rvec({reco_sel_GF}_genCharge, {nreps})",
    )
    df = df.Define(
        f"{reco_sel_GF}Multi_genQop",
        f"wrem::replicate_rvec({reco_sel_GF}_genQop, {nreps})",
    )
    df = df.Define(
        f"{reco_sel_GF}Multi_smearedqopr",
        f"{reco_sel_GF}Multi_smearedmqop/{reco_sel_GF}Multi_genQop",
    )

    response_cols_smeared_multi = [
        f"{reco_sel_GF}Multi_genPt",
        f"{reco_sel_GF}Multi_genEta",
        f"{reco_sel_GF}Multi_genCharge",
        f"{reco_sel_GF}Multi_smearedqopr",
    ]
    hist_qopr_smearedmulti = df.HistoBoost(
        "hist_qopr_smearedmulti",
        response_axes,
        [*response_cols_smeared_multi, "nominal_weight"],
    )
    hist_qopr_smearedmulti._hist.metadata = {"sigmarel": sigmarel}
    results.append(hist_qopr_smearedmulti)

    if args.testHelpers:
        df = df.Define(
            f"{reco_sel_GF}_response_weight",
            response_helper,
            [
                f"{reco_sel_GF}_recoPt",
                f"{reco_sel_GF}_recoEta",
                f"{reco_sel_GF}_recoCharge",
                f"{reco_sel_GF}_genPt",
                f"{reco_sel_GF}_genEta",
                f"{reco_sel_GF}_genCharge",
            ],
        )
        df = df.Define(
            f"{reco_sel_GF}_dweightdqop",
            f"""
            ROOT::VecOps::RVec<double> out;
            out.reserve({reco_sel_GF}_response_weight.size());
            for (const auto& pair : {reco_sel_GF}_response_weight) {{
                out.push_back(pair.first);
            }}
            return out;
            """,
        )
        df = df.Define(
            f"{reco_sel_GF}_curvature",
            f"1.0/{reco_sel_GF}_recoPt",
        )
        hist_dweightdqop = df.HistoBoost(
            "hist_dweightdqop",
            [axis_genPt, axis_genEta, axis_genCharge, axis_curvature, axis_dweightdqop],
            [
                f"{reco_sel_GF}_genPt",
                f"{reco_sel_GF}_genEta",
                f"{reco_sel_GF}_genCharge",
                f"{reco_sel_GF}_curvature",
                f"{reco_sel_GF}_dweightdqop",
                "nominal_weight",
            ],
        )
        results.append(hist_dweightdqop)

    return results, weightsum


resultdict = narf.build_and_run(datasets, build_graph)

write_analysis_output(
    resultdict, f"{os.path.basename(__file__).replace('py', 'hdf5')}", args
)

if args.testHelpers:
    os.makedirs(plot_dir, exist_ok=True)
    plot_weight_variation_hists(resultdict, plot_dir)
