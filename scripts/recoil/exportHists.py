
import sys,array,math,os
import numpy as np
import pickle
import wremnants.datasets.datagroups as datagroups


def readProc(groups, hName, procs):
    groups.setNominalName(hName)
    groups.loadHistsForDatagroups(hName, syst="")
    return sum([groups.groups[p].hists[hName] for p in procs])
    #return groups.groups[procs[0]].hists[hName]


if __name__ == "__main__":

    met = "DeepMETReso"
    flavor = "mumu"

    #groups = datagroups.Datagroups(f"mz_wlike_with_mu_eta_pt_{met}.hdf5")
    groups = datagroups.Datagroups(f"mz_wlike_with_mu_eta_pt_DeepMETReso_nnpdf31_noPTcut.hdf5")
    groups = datagroups.Datagroups(f"mz_wlike_with_mu_eta_pt_DeepMETReso_nnpdf31.hdf5")
    
    

    zmumu_para = readProc(groups, "recoil_corr_xy_para_qTbinned", ["Zmumu"])
    zmumu_perp = readProc(groups, "recoil_corr_xy_perp_qTbinned", ["Zmumu"])

    bkg_para = readProc(groups, "recoil_corr_xy_para_qTbinned", ["Ztautau", "Other"])
    bkg_perp = readProc(groups, "recoil_corr_xy_perp_qTbinned", ["Ztautau", "Other"])

    singlemuon_para = readProc(groups, "recoil_corr_xy_para_qTbinned", ["Data"])
    singlemuon_perp = readProc(groups, "recoil_corr_xy_perp_qTbinned", ["Data"])

    savedict = {}
    savedict['zmumu_para'] = zmumu_para
    savedict['zmumu_perp'] = zmumu_perp
    savedict['bkg_para'] = bkg_para
    savedict['bkg_perp'] = bkg_perp
    savedict['singlemuon_para'] = singlemuon_para
    savedict['singlemuon_perp'] = singlemuon_perp

    with open(f"recoil_mz_wlike_with_mu_eta_pt_{met}.pkl", "wb") as f:
        pickle.dump(savedict, f)
