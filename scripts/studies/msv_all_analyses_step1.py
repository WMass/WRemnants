import argparse
import os

from wums import logging

logger = logging.setup_logger(__file__, 3, True)

parser = argparse.ArgumentParser()
parser.add_argument(
    "-o", "--outfolder", type=str, default="./hdf5Files/splines", help="Output folder"
)
parser.add_argument(
    "--combineOutFolder",
    type=str,
    default="./CombineStudies/msv_all_analyses",
    help="output folder for combine files",
)
parser.add_argument(
    "--defaultPostfix",
    type=str,
    default="scetlib_dyturboCorr",
    help="the postfix string added to the hdf5 file by default by the histmaker",
)
parser.add_argument(
    "--infoOnly",
    action="store_true",
    help="only print the commands without actually running them",
)

args = parser.parse_args()

histmakers = ["mw_with_mu_eta_pt", "mz_wlike_with_mu_eta_pt", "mz_dilepton"]

if not os.path.isdir(args.outfolder):
    os.mkdir(args.outfolder)
if not os.path.isdir(args.combineOutFolder):
    os.mkdir(args.combineOutFolder)


def EXE(command):
    logger.info(command)
    if not args.infoOnly:
        os.system(command)  # for testing comment out this line


def make_appendix(name):

    parts = []
    for p in [pa.replace("-", " ").replace("_", " ") for pa in name.split("--")]:
        if p == "":
            continue
        ps = p.split(" ")
        ps = "".join(
            [
                ps[0],
            ]
            + [p.capitalize() for p in ps[1:]]
        )
        parts.append(ps)

    return "_".join(parts)


for histmaker in histmakers:
    file_hdf5 = f"{histmaker}_{args.defaultPostfix}.hdf5"
    if not os.path.isfile(f"{args.outfolder}/{file_hdf5}"):
        # run histmaker for nominal
        if histmaker == "mz_dilepton":
            EXE(
                f"python3 scripts/histmakers/{histmaker}.py -o {args.outfolder} --axes ptll mll yll"
            )
        else:
            EXE(f"python3 scripts/histmakers/{histmaker}.py -o {args.outfolder}")
    else:
        logger.info(f"Found file for {file_hdf5}")

    # make combine input
    dir_combine = f"{args.combineOutFolder}/{histmaker}"
    sep = "--sepImpactForNC --sepImpactForReso"
    if histmaker == "mz_dilepton":
        EXE(
            f"python3 scripts/combine/setupRabbit.py -o {dir_combine} -i {args.outfolder}/{file_hdf5} --fitvar mll {sep}"
        )
    else:
        EXE(
            f"python3 scripts/combine/setupRabbit.py -o {dir_combine} -i {args.outfolder}/{file_hdf5} {sep}"
        )
