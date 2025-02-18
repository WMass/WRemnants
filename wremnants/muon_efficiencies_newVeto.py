import os.path
import pickle

import boost_histogram as bh
import hist
import lz4.frame
import ROOT

import narf
from utilities import common
from wums import logging

logger = logging.child_logger(__name__)

narf.clingutils.Declare('#include "muon_efficiencies_newVeto.hpp"')

data_dir = common.data_dir


def make_muon_efficiency_helpers_newVeto(antiveto=False, era=None):

    logger.debug("Make efficiency helper for veto (with newer approach)")

    axis_eta_eff = None
    axis_pt_eff = None
    # categorical axes in python bindings always have an overflow bin, so use a regular
    # axis for the charge
    axis_charge = hist.axis.Regular(
        2, -2.0, 2.0, underflow=False, overflow=False, name="SF charge"
    )

    veto_tag = "antiVeto" if antiveto else "veto"
    charges = {-1.0: "minus", 1.0: "plus"}
    steps = ["vetoreco", "vetotracking", "vetoidip"]
    eff_types_2D = [x for x in steps]
    logger.info(f"{veto_tag} SF steps in 2D (eta-pt): {eff_types_2D}")
    axis_eff_type_2D = hist.axis.StrCategory(eff_types_2D, name="eff_types_2D_etapt")

    eradir = era if era in ["2017", "2018"] else ""

    fileVetoSF = {
        "plus": f"{data_dir}/muonSF/{eradir}/veto_global_SF/allVetoSF_global_plus.pkl.lz4",
        "minus": f"{data_dir}/muonSF/{eradir}/veto_global_SF/allVetoSF_global_minus.pkl.lz4",
    }
    effSyst_decorrEtaEdges = [round(-2.4 + 0.1 * i, 1) for i in range(49)]
    NsystDecorr = len(effSyst_decorrEtaEdges) - 1
    Nsyst = 1 + NsystDecorr  # 1 inclusive variation + all decorrelated bins
    Nstat = 4  # from smoothing with pol3
    inputHist_systBin = -1  # set later

    axis_nom_syst = hist.axis.Integer(
        0, 1 + Nsyst + Nstat, underflow=False, overflow=False, name="nom-systs"
    )  # nominal in first bin

    ### first the syst part
    sf_syst_2D = None
    for charge, charge_tag in charges.items():
        fileSF = fileVetoSF[charge_tag]
        if not os.path.isfile(fileSF):
            raise IOError(
                f"Couldn't read veto/antiveto SF file {fileSF}, make sure you have it."
            )
        logger.info(f"Veto/antiveto SF read from {fileSF} for charge {charge_tag}")
        with lz4.frame.open(fileSF) as fveto:
            dict_veto = pickle.load(fveto)
        # for veto or antiveto the histogram is nomi-stat-syst, but there are three of such histograms
        # nomi is always the same (product already of all steps), but stat/syst variations are for only one step variation with respect to the product (hence the three histograms)
        logger.debug(f"Objects in dict_veto: {[x for x in dict_veto.keys()]}")
        # axes are
        # Regular(48, -2.4, 2.4, name='SF eta')
        # Regular(250, 15, 65, name='SF pt')
        # Regular(10, 0.5, 10.5, name='nomi-statUpDown-syst'))
        for step in steps:
            hist_hist = dict_veto[f"{veto_tag}SF_global_{step}_{charge_tag}"]

            nstat = int(
                (hist_hist.axes[2].size - 2) / 2
            )  # use only statUp, and do mirroring later on at analysis level
            inputHist_systBin = hist_hist.axes[2].size - 1
            if nstat != Nstat:
                logger.warning(
                    f"Step {step}: expected {Nstat} stat variations but {nstat} were found. Please check"
                )
                raise IOError(
                    f"Found inconsistent number of SF stat variations for {step} step"
                )

            if sf_syst_2D is None:
                axis_eta_eff = hist_hist.axes[0]
                axis_pt_eff = hist_hist.axes[1]
                # store all systs (currently only 1) with the nominal, for all efficiency steps
                sf_syst_2D = hist.Hist(
                    axis_eta_eff,
                    axis_pt_eff,
                    axis_charge,
                    axis_eff_type_2D,
                    axis_nom_syst,
                    name="sf_syst_2D",
                    storage=hist.storage.Weight(),
                )
            # extract nominal (first bin that is not underflow) and put in corresponding bin of destination
            sf_syst_2D.view(flow=False)[
                :, :, axis_charge.index(charge), axis_eff_type_2D.index(step), 0
            ] = hist_hist.view(flow=False)[:, :, 0]
            # extract syst (last bin except overflow) and put in corresponding bin of destination (bin 1 is the second bin because no underflow)
            sf_syst_2D.view(flow=False)[
                :, :, axis_charge.index(charge), axis_eff_type_2D.index(step), 1
            ] = hist_hist.view(flow=False)[:, :, inputHist_systBin]
            for isyst in range(len(effSyst_decorrEtaEdges) - 1):
                # first copy the nominal
                sf_syst_2D.view(flow=False)[
                    :,
                    :,
                    axis_charge.index(charge),
                    axis_eff_type_2D.index(step),
                    2 + isyst,
                ] = hist_hist.view(flow=False)[:, :, 0]
                # now update with actual syst all eta bins inside interval [effSyst_decorrEtaEdges[isyst], effSyst_decorrEtaEdges[isyst+1]]
                # add epsilon to ensure picking the bin on the right of the edge (for the right edge given by
                # effSyst_decorrEtaEdges[isyst+1]] the range selection in boost later on will stop at the left
                # edge of the chosen bin number, e.g. h[b:b+1] will pick the range containing the single bin b, unlike in ROOT
                indexEtaLow = axis_eta_eff.index(
                    effSyst_decorrEtaEdges[isyst] + 0.001
                )  # add epsilon to ensure picking the bin on the right of the edge
                indexEtaHigh = axis_eta_eff.index(
                    effSyst_decorrEtaEdges[isyst + 1] + 0.001
                )
                sf_syst_2D.view(flow=False)[
                    indexEtaLow:indexEtaHigh,
                    :,
                    axis_charge.index(charge),
                    axis_eff_type_2D.index(step),
                    2 + isyst,
                ] = hist_hist.view(flow=False)[
                    indexEtaLow:indexEtaHigh, :, inputHist_systBin
                ]
            # now the stat part
            for istat in range(Nstat):
                sf_syst_2D.view(flow=False)[
                    :,
                    :,
                    axis_charge.index(charge),
                    axis_eff_type_2D.index(step),
                    1 + Nsyst + istat,
                ] = hist_hist.view(flow=False)[:, :, 1 + istat]

    # set overflow and underflow eta-pt bins equal to adjacent bins
    sf_syst_2D.view(flow=True)[0, ...] = sf_syst_2D.view(flow=True)[1, ...]
    sf_syst_2D.view(flow=True)[axis_eta_eff.extent - 1, ...] = sf_syst_2D.view(
        flow=True
    )[axis_eta_eff.extent - 2, ...]
    sf_syst_2D.view(flow=True)[:, 0, ...] = sf_syst_2D.view(flow=True)[:, 1, ...]
    sf_syst_2D.view(flow=True)[:, axis_pt_eff.extent - 1, ...] = sf_syst_2D.view(
        flow=True
    )[:, axis_pt_eff.extent - 2, ...]

    NetaBinsStat = axis_eta_eff.size

    # now creating the tensors
    sf_syst_2D_pyroot = narf.hist_to_pyroot_boost(sf_syst_2D)
    # nomi and syst are stored in the same histogram, just use different helpers to override the () operator for now, until RDF is improved
    helper = ROOT.wrem.muon_efficiency_newVeto_helper[
        type(sf_syst_2D_pyroot), len(steps), Nsyst, NetaBinsStat, Nstat
    ](ROOT.std.move(sf_syst_2D_pyroot))
    helper_syst = ROOT.wrem.muon_efficiency_newVeto_helper_syst[
        type(sf_syst_2D_pyroot), len(steps), Nsyst, NetaBinsStat, Nstat
    ](helper)
    # define axis for syst variations with all steps
    axis_all = hist.axis.Integer(
        0, len(steps), underflow=False, overflow=False, name="-".join(steps)
    )
    axis_nsyst = hist.axis.Integer(
        0, Nsyst, underflow=False, overflow=False, name="n_syst_variations"
    )
    helper_syst.tensor_axes = [axis_all, axis_nsyst]
    #
    ### now the stat part
    helper_stat = ROOT.wrem.muon_efficiency_newVeto_helper_stat[
        type(sf_syst_2D_pyroot), len(steps), Nsyst, NetaBinsStat, Nstat
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
        0, Nstat, underflow=False, overflow=False, name="nPtEigenBins"
    )
    helper_stat.tensor_axes = [
        axis_all,
        axis_eta_eff_tensor,
        axis_ptEigen_eff_tensor,
        axis_charge,
    ]

    logger.debug(f"Return veto efficiency helpers!")

    return helper, helper_syst, helper_stat
