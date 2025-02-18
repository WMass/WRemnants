import boost_histogram as bh
import hist
import ROOT

import narf
from utilities import common
from utilities.io_tools import input_tools
from wums import logging

logger = logging.child_logger(__name__)

narf.clingutils.Declare('#include "muon_efficiencies_veto.hpp"')

data_dir = common.data_dir


def make_muon_efficiency_helpers_veto(useGlobalOrTrackerVeto=False, era=None):

    logger.debug("Make efficiency helper veto")

    effSyst_decorrEtaEdges = [round(-2.4 + 0.1 * i, 1) for i in range(49)]
    Nsyst = 1 + (
        len(effSyst_decorrEtaEdges) - 1
    )  # 1 inclusive variation + all decorrelated bins

    eradir = era if era in ["2017", "2018"] else ""

    if useGlobalOrTrackerVeto:
        # in this way we are hardcoding the file names for the veto SFs, but I don't think we are going to change them in the helpers anyways
        filename_plus = f"{data_dir}/muonSF/{eradir}/smoothedSFandEffi_newveto_globalortracker_regular_GtoH_plus.root"
        filename_minus = f"{data_dir}/muonSF/{eradir}/smoothedSFandEffi_newveto_globalortracker_regular_GtoH_minus.root"
    else:
        filename_plus = f"{data_dir}/muonSF/{eradir}/smoothedSFandEffi_newveto_regular_GtoH_plus.root"
        filename_minus = f"{data_dir}/muonSF/{eradir}/smoothedSFandEffi_newveto_regular_GtoH_minus.root"

    if not useGlobalOrTrackerVeto:
        Steps = 3  # we decided to compute the syst variations on the veto SFs independently for each of the tnp fits (using only global muons in the muon definition we fit reco, "tracking", looseID + dxybs)
    else:
        Steps = 5  # we decided to compute the syst variations on the veto SFs independently for each of the tnp fits (for global-or-tracker in the muon definition we fit reco, "tracking", P(tracker-seeded track|standalone), P(tracker muon and not global|tracker-seeded track), looseID + dxybs)

    file_plus = input_tools.safeOpenRootFile(f"{filename_plus}")
    plus_histo = input_tools.safeGetRootObject(
        file_plus, "SF_nomiAndAlt_GtoH_newveto_plus"
    )
    file_plus.Close()
    file_minus = input_tools.safeOpenRootFile(f"{filename_minus}")
    minus_histo = input_tools.safeGetRootObject(
        file_minus, "SF_nomiAndAlt_GtoH_newveto_minus"
    )
    file_minus.Close()

    hist_plus = narf.root_to_hist(
        plus_histo, axis_names=["SF eta", "SF pt", "nomi-statUpDown-syst"]
    )
    hist_minus = narf.root_to_hist(
        minus_histo, axis_names=["SF eta", "SF pt", "nomi-statUpDown-syst"]
    )

    # note: eta-pt axes have overflow bins
    axis_eta_eff = hist_plus.axes[0]
    axis_pt_eff = hist_plus.axes[1]
    axis_originalStatSystVars = hist_plus.axes[2]
    nBins_nomiAndSyst = 1 + Steps
    NEtaBins = axis_eta_eff.size
    NPtEigenBins = int(
        int(axis_originalStatSystVars.size - nBins_nomiAndSyst) / 2
    )  # use only stat Up, and do mirroring later on at analysis level

    # categorical axes in python bindings always have an overflow bin, so use a regular
    # axis for the charge
    axis_charge = hist.axis.Regular(
        2, -2.0, 2.0, underflow=False, overflow=False, name="SF charge"
    )

    # use a boost histogram with both charges to create the helpers
    # first create a single one from the two charge histograms
    hist_charge = hist.Hist(
        axis_eta_eff,
        axis_pt_eff,
        axis_charge,
        axis_originalStatSystVars,
        name="hist_charge",
        storage=hist.storage.Weight(),
    )
    hist_charge.view(flow=False)[:, :, axis_charge.index(-1), :] = hist_minus.view(
        flow=False
    )[:, :, :]
    hist_charge.view(flow=False)[:, :, axis_charge.index(1), :] = hist_plus.view(
        flow=False
    )[:, :, :]

    # create nomi-stat-syst axis with 1 + NPtEigenBins + Nsyst bins (here the stat uncertainty is not split by step)
    # will remove Down stat variations, keeping only the up ones
    axis_nom_syst = hist.axis.Integer(
        0, 1 + NPtEigenBins + Steps, underflow=False, overflow=False, name="nom-systs"
    )  # nominal in first bin
    # now create histogram to feed the helpers
    sf_syst_2D = hist.Hist(
        axis_eta_eff,
        axis_pt_eff,
        axis_charge,
        axis_nom_syst,
        name="sf_syst_2D",
        storage=hist.storage.Weight(),
    )

    charges = [-1, 1]
    for charge in charges:
        # copy nominal
        chIdx = axis_charge.index(charge)
        sf_syst_2D.view(flow=False)[:, :, chIdx, 0] = hist_charge.view(flow=False)[
            :, :, chIdx, 0
        ]
        # copy stat vars (only up ones, for a total of NPtEigenBins)
        for ipt in range(NPtEigenBins):
            sf_syst_2D.view(flow=False)[:, :, chIdx, 1 + ipt] = hist_charge.view(
                flow=False
            )[:, :, chIdx, 1 + ipt]
        # copy syst (eta decorrelation dealt with inside helper directly for now)
        for istep in range(Steps):
            # reminder: hist_charge stores all up/down stat variations, sf_syst_2D stores only the up ones
            sf_syst_2D.view(flow=False)[:, :, chIdx, 1 + NPtEigenBins + istep] = (
                hist_charge.view(flow=False)[:, :, chIdx, 1 + 2 * NPtEigenBins + istep]
            )

    # set overflow and underflow eta-pt bins equal to adjacent bins
    sf_syst_2D.view(flow=True)[0, ...] = sf_syst_2D.view(flow=True)[1, ...]
    sf_syst_2D.view(flow=True)[axis_eta_eff.extent - 1, ...] = sf_syst_2D.view(
        flow=True
    )[axis_eta_eff.extent - 2, ...]
    sf_syst_2D.view(flow=True)[:, 0, ...] = sf_syst_2D.view(flow=True)[:, 1, ...]
    sf_syst_2D.view(flow=True)[:, axis_pt_eff.extent - 1, ...] = sf_syst_2D.view(
        flow=True
    )[:, axis_pt_eff.extent - 2, ...]

    hist_pyroot = narf.hist_to_pyroot_boost(sf_syst_2D)

    helper = ROOT.wrem.muon_efficiency_veto_helper[
        type(hist_pyroot), NEtaBins, NPtEigenBins, Nsyst, Steps
    ](
        ROOT.std.move(hist_pyroot),
    )

    helper_syst = ROOT.wrem.muon_efficiency_veto_helper_syst[
        type(hist_pyroot), NEtaBins, NPtEigenBins, Nsyst, Steps
    ](helper)

    if not useGlobalOrTrackerVeto:
        axis_all = hist.axis.Integer(
            0,
            Steps,
            underflow=False,
            overflow=False,
            name="veto_reco-veto_tracking-veto_idip",
        )
    else:
        axis_all = hist.axis.Integer(
            0,
            Steps,
            underflow=False,
            overflow=False,
            name="veto_reco-veto_tracking-veto_idip-veto_trackerreco-veto_trackertracking",
        )
    axis_nsyst = hist.axis.Integer(
        0, Nsyst, underflow=False, overflow=False, name="n_syst_variations"
    )

    helper_syst.tensor_axes = [axis_all, axis_nsyst]

    helper_stat = ROOT.wrem.muon_efficiency_veto_helper_stat[
        type(hist_pyroot), NEtaBins, NPtEigenBins, Nsyst, Steps
    ](helper)

    # make new versions of these axes without overflow/underflow to index the tensor
    if isinstance(axis_eta_eff, bh.axis.Regular):
        axis_eta_eff_tensor = hist.axis.Regular(
            axis_eta_eff.size,
            axis_eta_eff.edges[0],
            axis_eta_eff.edges[-1],
            name=axis_eta_eff.name,
            overflow=False,
            underflow=False,
        )
    elif isinstance(axis_eta_eff, bh.axis.Variable):
        axis_eta_eff_tensor = hist.axis.Variable(
            axis_eta_eff.edges, name=axis_eta_eff.name, overflow=False, underflow=False
        )

    axis_ptEigen_eff_tensor = hist.axis.Integer(
        0, NPtEigenBins, underflow=False, overflow=False, name="nPtEigenBins"
    )

    effStatTensorAxes = [axis_eta_eff_tensor, axis_ptEigen_eff_tensor, axis_charge]
    helper_stat.tensor_axes = effStatTensorAxes

    return helper, helper_syst, helper_stat
