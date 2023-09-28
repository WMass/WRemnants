from narf.ioutils import H5PickleProxy
import time
from utilities import logging

logger = logging.child_logger(__name__)

def scale_to_data(result_dict, data_name = "dataPostVFP"):
    # scale histograms by lumi*xsec/sum(gen weights)
    time0 = time.time()

    lumi = [result["lumi"] for result in result_dict.values() if result["dataset"]["is_data"]]
    if len(lumi) == 0:
        lumi = 1
    else:
        lumi = sum(lumi)

    logger.warning(f"Scale histograms with luminosity = {lumi} /fb")
    for d_name, result in result_dict.items():
        if result["dataset"]["is_data"]:
            continue

        xsec = result["dataset"]["xsec"]

        logger.debug(f"For dataset {d_name} with xsec={xsec}")

        scale = lumi * 1000 * xsec / result["weight_sum"]

        result["weight_sum"] = result["weight_sum"]*scale

        for h_name, histogram in result["output"].items():

            histo = histogram.get()

            histo *= scale

    logger.info(f"Scale to data: {time.time() - time0}")


def aggregate_groups(datasets, result_dict, groups_to_aggregate):
    # add members of groups together
    time0 = time.time()

    for group in groups_to_aggregate:

        dataset_names = [d.name for d in datasets if d.group == group]
        if len(dataset_names) == 0:
            continue

        logger.debug(f"Aggregate group {group}")

        resdict = None
        members = {}
        to_del = []
        for name, result in result_dict.items():
            if result["dataset"]["name"] not in dataset_names:
                continue

            logger.debug(f"Add {name}")

            for h_name, histogram in result["output"].items():
                if h_name in members.keys():
                    members[h_name].append(histogram.get())
                else:
                    members[h_name] = [histogram.get()]

            if resdict is None:
                resdict = {
                    "n_members": 1,
                    "dataset": {
                        "name": group,
                        "xsec": result["dataset"]["xsec"],
                        "filepaths": result["dataset"]["filepaths"],
                    },
                    "weight_sum": float(result["weight_sum"]),
                    "event_count": float(result["event_count"])
                }
            else:
                resdict["dataset"]["xsec"] += result["dataset"]["xsec"]
                resdict["dataset"]["filepaths"] += result["dataset"]["filepaths"]
                resdict["n_members"] += 1
                resdict["weight_sum"] += float(result["weight_sum"])
                resdict["event_count"] += float(result["event_count"])

            to_del.append(name)

        output = {}
        for h_name, histograms in members.items():

            if len(histograms) != resdict["n_members"]:
                logger.warning(f"There is a different number of histograms ({len(histograms)}) than original members ("+str(dsetresult_group[g_name]["n_members"])+f") for {h_name} from group {g_name}")
                logger.warning("Summing them up probably leads to wrong behaviour")

            output[h_name] = H5PickleProxy(sum(histograms))

        result_dict[group] = resdict
        result_dict[group]["output"] = output

        # delete individual datasets
        for name in to_del:
            del result_dict[name]

    logger.info(f"Aggregate groups: {time.time() - time0}")
