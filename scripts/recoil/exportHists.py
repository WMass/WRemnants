
import sys,argparse
import pickle
import wremnants.datasets.datagroups as datagroups

import functions

def readProc(groups, hName, procs):
    groups.setNominalName(hName)
    groups.loadHistsForDatagroups(hName, syst="", procsToRead=procs)
    
    bhist = sum([groups.groups[p].hists[hName] for p in procs])
    k = bhist.values()
    k[bhist.values()<0] = 0 # remove negative values
    bhist.view().value = k
    return bhist


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="Input hdf5 file")
    args = parser.parse_args()

    groups = datagroups.Datagroups(args.input)
    met, analysis, flavor = functions.get_meta(groups)

    savedict = {}
    savedict['lumi_header'] = functions.getLumiLabel(groups)
    if flavor == "mumu" or flavor == "ee":

        savedict['z_para'] = readProc(groups, "recoil_corr_xy_para_v_pt", ['Zmumu' if flavor=='mumu' else 'Zee'])
        savedict['z_perp'] = readProc(groups, "recoil_corr_xy_perp_v_pt", ['Zmumu' if flavor=='mumu' else 'Zee'])

        savedict['z_para_gen'] = readProc(groups, "recoil_corr_xy_para_gen_v_gen_pt", ['Zmumu' if flavor=='mumu' else 'Zee'])
        savedict['z_perp_gen'] = readProc(groups, "recoil_corr_xy_perp_gen_v_gen_pt", ['Zmumu' if flavor=='mumu' else 'Zee'])

        savedict['bkg_para'] = readProc(groups, "recoil_corr_xy_para_v_pt", ['Ztautau', 'Other', 'PhotonInduced'])
        savedict['bkg_perp'] = readProc(groups, "recoil_corr_xy_perp_v_pt", ['Ztautau', 'Other', 'PhotonInduced'])

        savedict['data_para'] = readProc(groups, "recoil_corr_xy_para_v_pt", ['Data'])
        savedict['data_perp'] = readProc(groups, "recoil_corr_xy_perp_v_pt", ['Data'])

    else:

        savedict['w_para_gen'] = readProc(groups, "recoil_corr_xy_para_gen_v_gen_pt", ['Wmunu' if flavor=='mu' else 'Wenu'])
        savedict['w_perp_gen'] = readProc(groups, "recoil_corr_xy_perp_gen_v_gen_pt", ['Wmunu' if flavor=='mu' else 'Wenu'])

        savedict['z_para_gen'] = readProc(groups, "recoil_corr_xy_para_gen_v_gen_pt", ['Zmumu' if flavor=='mu' else 'Zee'])
        savedict['z_perp_gen'] = readProc(groups, "recoil_corr_xy_perp_gen_v_gen_pt", ['Zmumu' if flavor=='mu' else 'Zee'])

    with open(f"recoil/{analysis}_{met}/input_{flavor}.pkl", "wb") as f:
        pickle.dump(savedict, f)
