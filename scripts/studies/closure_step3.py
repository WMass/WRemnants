import argparse
import json
import os

import pandas as pd
import uproot

os.sys.path.append(os.path.expandvars("/home/submit/tyjyang/analysis/wmass/WRemnants"))

from utilities.io_tools import input_tools
from wums import logging

logger = logging.setup_logger(__file__, 3, True)

# Write LATEX table for results of biastests.
#   The shift in the mass parameter is the bias
# This script is step 3 and done within WRemnants singularity, it collects the biastests results and creates a latex table

parser = argparse.ArgumentParser()
parser.add_argument(
    "-i", "--input", type=str, help="json file with paths to fit results"
)
parser.add_argument(
    "-m",
    "--mode",
    type=str,
    default="pdf",
    help="Uncertainty to consider",
    choices=["pdf", "scale"],
)
parser.add_argument(
    "-f", "--folder", type=str, default="Impacts", help="output folder for impact plots"
)
parser.add_argument(
    "-u", "--update", action="store_true", help="Override existing json file"
)
parser.add_argument("--doImpacts", action="store_true", help="Produce impact plots")
parser.add_argument(
    "--impactTypes",
    default=["group"],
    nargs="+",
    choices=["group", "ungroup"],
    help="Group or ungroup nuisances in impact plot, or both",
)
args = parser.parse_args()

outDir = "/".join(args.input.split("/")[:-1])

output_dir_impacts = f"/eos/home-d/dwalter/www/WMassAnalysis/{args.folder}"
if args.doImpacts and not os.path.isdir(output_dir_impacts):
    os.mkdir(output_dir_impacts)


def EXE(command):
    logger.info(command)
    # os.system(command)  # for testing comment out this line


def read_result(rootfile, nominal):
    logger.info(f"read {rootfile}")
    with uproot.open(rootfile) as rtfile:

        impacts_group = {
            l: i
            for i, l in zip(*input_tools.readImpacts(rtfile, True, normalize=False)[:2])
        }

        pull_mass = rtfile["fitresults"]["massShift100MeV"].array()[0]
        print(impacts_group.keys())
        uncertainty_total = impacts_group["Total"]

        if args.mode == "scale":
            uncertainty = impacts_group["muonScale"]
        elif args.mode == "pdf":
            uncertainty = impacts_group[f"pdf{nominal.upper()}"]

        return pull_mass, uncertainty, uncertainty_total


### Collecting fit results
with open(args.input, "r") as rfile:
    results = json.load(rfile)

nominals = []
pseudos = []
channels = []
masses = []
uncertainties = []
totals = []

for nominal, r_n in results.items():

    for pseudodata, r_np in r_n.items():

        for channel, r_c in r_np.items():

            if isinstance(r_c, dict):
                filename = r_c["filename"]
            else:
                filename = r_c

            if (
                not args.update
                and isinstance(r_c, dict)
                and "mass" in r_c.keys()
                and "unc" in r_c.keys()
                and "tot" in r_c.keys()
            ):
                m = r_c["mass"]
                u = r_c["unc"]
                t = r_c["tot"]
            else:
                m, u, t = read_result(filename, nominal)

                results[nominal][pseudodata][channel] = {
                    "filename": filename,
                    "mass": m,
                    "unc": u,
                    "tot": t,
                }
                # save result.json so that next time the information does not have to be read again
                with open(args.input, "w") as rfile:
                    json.dump(results, rfile, indent=4)

            for group in args.impactTypes:

                output_impacts = f"{output_dir_impacts}/{channel}_{group}_{nominal}_vs_{pseudodata}.html"

                if args.doImpacts and not os.path.isfile(output_impacts):

                    group_str = "-g" if group == "group" else ""

                    EXE(
                        f"python3 scripts/combine/pullsAndImpacts.py -f {filename} {group_str} -s impact output -o {output_impacts}"
                    )

            if args.mode == "pdf":
                nominals.append(nominal.upper())
                pseudos.append(pseudodata.upper())
            else:
                nominals.append(nominal)
                pseudos.append(pseudodata)

            channels.append(channel)
            masses.append(m)
            uncertainties.append(u)
            totals.append(t)


### Make latex table

# some settings
boson_str = "W"
fit_str = r"\P" + boson_str + r" $(\pt^{\ell},\eta^{\ell})$"


data = pd.DataFrame(
    {
        "nominal": nominals,
        "pseudo": pseudos,
        "channel": channels,
        "mass": masses,
        "unc": uncertainties,
        "tot": totals,
    }
)


def plot_table_pdf(data, uncertainty="unc"):

    mode_str = "pdf sets"
    uncertainty_str1 = "only pdf and bin-by-bin statistical uncertainties."
    uncertainty_str2 = r"pdf uncertainty on $m_\P" + boson_str + "$"

    for channel, df in data.groupby("channel"):

        df.sort_values(by=["nominal", "pseudo"])

        pseudo = list(sorted(set(df["pseudo"].values)))
        nominal = list(sorted(set(df["nominal"].values)))

        pseudo = list(filter(lambda x: x in pseudo, nominal)) + list(
            filter(lambda x: x not in nominal, pseudo)
        )

        outfile = f"{outDir}/result_table_{channel}_{uncertainty}.txt"
        logger.info(f"write {outfile}")
        with open(outfile, "w") as outfile:

            outfile.write(r"\begin{table}" + "\n")
            outfile.write(
                r"\topcaption{\label{table:pulls_pdf_"
                + nominal
                + "_"
                + uncertainty
                + "}"
                + r"\n"
            )
            outfile.write(
                r""" Pulls table for different pdf sets. The fit is performed on """
                + fit_str
                + """ in the """
                + channel
                + r""" channel with """
                + uncertainty_str1
                + r"""
    Entries read (pull on the $m_\P"""
                + boson_str
                + r"""$ central value) $\pm$ ("""
                + uncertainty_str2
                + r""").}"""
                + r"\n"
            )
            outfile.write(r"\centering" + "\n")

            columns = "l|"
            columns += "".join(["c" for c in range(len(pseudo))])
            outfile.write(r"\begin{tabular}{" + columns + "}" + r"\n")

            outfile.write(
                r"  Model + & \multicolumn{"
                + str(len(pseudo))
                + "}{c}{Pseudodata} "
                + r" \\"
                + r"\n"
            )
            outfile.write("  Uncertainty & " + " & ".join(pseudo) + r" \\" + r"\n")

            outfile.write(r"  \hline " + r"\n")

            for nominal, df_n in df.groupby("nominal"):
                entries = []
                for p in pseudo:
                    df_p = df_n.loc[df_n["pseudo"] == p][["mass", uncertainty]]
                    if len(df_p) == 1:
                        m, u = df_p.values[0]
                        m = round(100 * m, 2)
                        u = round(100 * u, 2)
                        colorstring = (
                            r"\cellcolor{red!25}" if abs(m) > u else ""
                        )  # highlight background color of cell if uncertainty does not cover mass shift
                        entries.append(rf"{colorstring} ${m} \pm {u}$")
                    else:
                        entries.append(r" \NA ")

                outfile.write(f"  {nominal} & " + " & ".join(entries) + r" \\" + r"\n")

            outfile.write(r"  \end{tabular}" + r"\n")
            outfile.write(r"\end{table} " + r"\n")


def plot_table_scale(data, uncertainty="unc"):
    translate = {
        "smearing": "smeared",
        "biasCalibrationA": "bias A",
        "biasCalibrationM": "bias M",
        "biasCalibrationParameterized": "bias A,M",
        "biasCalibrationBinned": r"bias $\pt,\eta$",
        "smearing_biasCalibrationA": "+ bias A",
        "smearing_biasCalibrationM": "+ bias M",
        "smearing_biasCalibrationParameterized": "+ bias A,M",
        "smearing_biasCalibrationBinned": r"+ bias $\pt,\eta$",
        # fit types
        "reducedUncertainties": "only scale and bin-by-bin MC statistical uncertainties.",
        "fullUncertainties": "the full uncertainty model.",
    }

    mode_str = "muon momentum scale and resoltion"
    uncertainty_str2 = "full" if uncertainty == "tot" else "momentum scale"
    uncertainty_str2 += r" uncertainty on $m_\P" + boson_str + "$"

    for fittype, df_fittype in data.groupby("fittype"):

        uncertainty_str1 = translate[fittype]

        for nominal, df in df_fittype.groupby("nominal"):

            closure_str = (
                "different muon momentum scale closures"
                if nominal == "smearing"
                else "muon momentum scale resolution"
            )

            df.sort_values(by=["pseudo"])

            pseudo = list(sorted(set(df["pseudo"].values)))
            pseudo = list(filter(lambda x: x in pseudo, nominal)) + list(
                filter(lambda x: x not in nominal, pseudo)
            )

            outfile = f"{outDir}/result_table_{nominal}_{fittype}_{uncertainty}.txt"
            logger.info(f"write {outfile}")
            with open(outfile, "w") as outfile:

                outfile.write(r"\begin{table}" + r"\n")
                outfile.write(
                    r"\topcaption{\label{table:pulls_scale_"
                    + nominal
                    + "_"
                    + uncertainty
                    + "}"
                    + r"\n"
                )
                outfile.write(
                    r""" Pulls table for """
                    + closure_str
                    + """. The fit is performed on """
                    + fit_str
                    + """ in different channels with """
                    + uncertainty_str1
                    + r"""
        Entries read (pull on the $m_\P"""
                    + boson_str
                    + r"""$ central value) $\pm$ ("""
                    + uncertainty_str2
                    + r""").}"""
                    + "\n"
                )
                outfile.write(r"\centering" + "\n")

                columns = "l|"
                columns += "".join(["c" for c in range(len(pseudo))])
                outfile.write(r"\begin{tabular}{" + columns + "}" + "\n")

                outfile.write(
                    r"  Model + & \multicolumn{"
                    + str(len(pseudo))
                    + "}{c}{Pseudodata} "
                    + r" \\"
                    + r"\n"
                )
                outfile.write(
                    "  Uncertainty & "
                    + " & ".join([translate.get(p, p) for p in pseudo])
                    + r" \\"
                    + r"\n"
                )

                outfile.write(r"  \hline " + "\n")

                for channel, df_n in df.groupby("channel"):
                    entries = []
                    for p in pseudo:
                        df_p = df_n.loc[df_n["pseudo"] == p][["mass", uncertainty]]
                        if len(df_p) == 1:
                            m, u = df_p.values[0]
                            m = round(100 * m, 2)
                            u = round(100 * u, 2)
                            colorstring = (
                                r"\cellcolor{red!25}" if abs(m) > u else ""
                            )  # highlight background color of cell if uncertainty does not cover mass shift
                            entries.append(rf"{colorstring} ${m} \pm {u}$")
                        else:
                            entries.append(r" \NA ")

                    outfile.write(
                        f"  {translate.get(nominal,nominal)} ({channel}) & "
                        + " & ".join(entries)
                        + r" \\"
                        + "\n"
                    )

                outfile.write(r"  \end{tabular}" + r"\n")
                outfile.write(r"\end{table} " + r"\n")


data["fittype"] = data["pseudo"].apply(lambda x: x.split("_")[-1])
data["pseudo"] = data["pseudo"].apply(lambda x: "_".join(x.split("_")[:-1]))

if args.mode == "scale":
    plot_table_scale(data)
    plot_table_scale(data, "tot")

if args.mode == "pdf":
    plot_table_pdf(data)
