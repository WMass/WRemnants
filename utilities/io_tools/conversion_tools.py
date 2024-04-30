
import itertools
import hist
import numpy as np

from narf import ioutils

from utilities import logging
from utilities.io_tools import combinetf_input

logger = logging.child_logger(__name__)

def fitresult_pois_to_hist(infile, result=None, poi_types = ["mu", "pmaskedexp", "pmaskedexpnorm", "sumpois", "sumpoisnorm", ], translate_poi_types=True,
    merge_channels=True, grouped=True, uncertainties=None, expected=False,
):
    # convert POIs in fitresult into histograms
    # uncertainties, use None to get all, use [] to get none
    # grouped=True for grouped uncertainties 
    # Different channels can have different year, flavor final state, particle final state, sqrt(s), 
    #   if merge_channels=True the lumi is added up for final states with different flavors or eras with same sqrt(s)
    
    # translate the name of the keys to be written out
    target_keys={
        "pmaskedexp": "xsec",
        "sumpois": "xsec",
        "pmaskedexpnorm": "xsec_normalized",
        "sumpoisnorm": "xsec_normalized",
    }
    
    channel_energy={
        "2017G": "5TeV",
        "2017H": "13TeV",
        "2016preVFP": "13TeV", 
        "2016postVFP":"13TeV", 
        "2017": "13TeV", 
        "2018": "13TeV",
    }
    channel_flavor ={
        "e": "l", 
        "mu": "l",
        "ee": "ll", 
        "mumu": "ll",
    }

    fitresult = combinetf_input.get_fitresult(infile.replace(".root",".hdf5"))
    meta = ioutils.pickle_load_h5py(fitresult["meta"])
    meta_info = meta["meta_info"]

    if merge_channels:
        channel_info = {}
        for chan, info in meta["channel_info"].items():
            if chan.endswith("masked"):
                continue
            channel = f"chan_{channel_energy[info['era']]}"
            if info['flavor'] in channel_flavor:
                channel += f"_{channel_flavor[info['flavor']]}"

            logger.debug(f"Merge channel {chan} into {channel}")

            lumi = info["lumi"]    
            gen_axes = info["gen_axes"]    

            if channel not in channel_info:
                channel_info[channel] = {
                    "gen_axes": gen_axes,
                    "lumi": lumi,
                }
            else:
                if gen_axes != channel_info[channel]["gen_axes"]:
                    raise RuntimeError(f"The gen axes are different among channels {channel_info}, so they can't be merged")
                channel_info[channel]["lumi"] += lumi
    else:
        channel_info = meta["channel_info"]

    if result is None: 
        result = {}
    for poi_type in poi_types:
        logger.debug(f"Now at POI type {poi_type}")

        df = combinetf_input.read_impacts_pois(fitresult, poi_type=poi_type, group=grouped, uncertainties=uncertainties)
        if df is None:
            logger.warning(f"POI type {poi_type} not found in histogram, continue with next one")
            continue

        scale = 1 
        if poi_type in ["nois"]:
            scale = 1./(imeta["args"]["scaleNormXsecHistYields"]*imeta["args"]["priorNormXsec"])

        poi_key = target_keys.get(poi_type, poi_type) if translate_poi_types else poi_type
        if poi_key not in result:
            result[poi_key] = {}
        for channel, info in channel_info.items():
            logger.debug(f"Now at channel {channel}")

            channel_scale = scale
            if poi_type in ["pmaskedexp", "sumpois"]:
                channel_scale = info["lumi"]*1000
            if channel not in result[poi_key]:
                result[poi_key][channel] = {}
            for proc, gen_axes_proc in info["gen_axes"].items():
                logger.debug(f"Now at proc {proc}")

                if poi_type.startswith("sum"):
                    if len(gen_axes_proc)==1:
                        logger.info(f"Skip POI type {poi_type} since there is only one gen axis")
                        continue
                    # make all possible lower dimensional gen axes combinations; wmass only combinations including qGen
                    gen_axes_permutations = [list(k) for n in range(1, len(gen_axes_proc)) for k in itertools.combinations(gen_axes_proc, n)]
                else:
                    gen_axes_permutations = [gen_axes_proc[:],]

                if proc not in result[poi_key][channel]:
                    result[poi_key][channel][proc] = {}
                for axes in gen_axes_permutations:
                    shape = [a.extent for a in axes]
                    axes_names = [a.name for a in axes]

                    data = combinetf_input.select_pois(df, axes_names, base_processes=proc, flow=True)
                    logger.debug(f"The values for the hist in poi {poi_key} are {data['value'].values}")

                    values = np.reshape(data["value"].values/channel_scale, shape)
                    variances = np.reshape( (data["err_total"].values/channel_scale)**2, shape)

                    h_ = hist.Hist(*axes, storage=hist.storage.Weight())
                    h_.view(flow=True)[...] = np.stack([values, variances], axis=-1)

                    hist_name = "hist_" + "_".join(axes_names)
                    if expected:
                        hist_name+= "_expected"
                    logger.info(f"Save histogram {hist_name}")
                    if hist_name in result[poi_key][channel][proc]:
                        logger.warning(f"Histogram {hist_name} already in result, it will be overridden")
                    result[poi_key][channel][proc][hist_name] = h_

                    if "err_stat" in data.keys():
                        # save stat only hist
                        variances = np.reshape( (data["err_stat"].values/channel_scale)**2, shape)
                        h_stat = hist.Hist(*axes, storage=hist.storage.Weight())
                        h_stat.view(flow=True)[...] = np.stack([values, variances], axis=-1)
                        hist_name_stat = f"{hist_name}_stat"
                        if hist_name_stat in result[poi_key][channel][proc]:
                            logger.warning(f"Histogram {hist_name_stat} already in result, it will be overridden")
                        result[poi_key][channel][proc][hist_name_stat] = h_stat

                    # save other systematic uncertainties as separately varied histograms
                    labels = [u.replace("err_","") for u in filter(lambda x: x.startswith("err_") and x not in ["err_total", "err_stat"], data.keys())]
                    if labels:
                        # string category always has an overflow bin, we set it to 0
                        systs = np.stack([values, *[values + np.reshape(data[f"err_{u}"].values/channel_scale, shape) for u in labels], np.zeros_like(values)], axis=-1)
                        h_syst = hist.Hist(*axes, hist.axis.StrCategory(["nominal", *labels], name="syst"), storage=hist.storage.Double())
                        h_syst.values(flow=True)[...] = systs
                        hist_name_syst = f"{hist_name}_syst"
                        if hist_name_syst in result[poi_key][channel][proc]:
                            logger.warning(f"Histogram {hist_name_syst} already in result, it will be overwritten")
                        result[poi_key][channel][proc][hist_name_syst] = h_syst

    return result, meta
