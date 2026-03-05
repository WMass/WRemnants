"""
Convenience script to generate a the PDF gen histograms for a list of theory corrections.
"""

import argparse
import os
from datetime import datetime

THEORY_PREDS = {
    "scetlib_dyturbo_CT18Z_N3p0LL_N2LO_pdfvars": {"pdf": "ct18z"},
    "scetlib_dyturbo_CT18Z_N3p1LL_N2LO_pdfvars": {"pdf": "ct18z"},
    "scetlib_dyturbo_CT18Z_N4p0LL_N2LO_pdfvars": {"pdf": "ct18z"},
    "scetlib_dyturbo_MSHT20_N3p0LL_N2LO_pdfvars": {"pdf": "msht20"},
    "scetlib_dyturbo_MSHT20an3lo_N3p0LL_N2LO_pdfvars": {"pdf": "msht20an3lo"},
    "scetlib_dyturbo_LatticeNP_CT18Z_N2p1LL_N2LO_pdfvars": {"pdf": "ct18z"},
    "scetlib_dyturbo_LatticeNP_CT18Z_N3p0LL_N2LO_pdfvars": {"pdf": "ct18z"},
    "scetlib_dyturbo_LatticeNP_CT18Z_N3p1LL_N2LO_pdfvars": {"pdf": "ct18z"},
    "scetlib_dyturbo_LatticeNP_CT18Z_N4p0LL_N2LO_pdfvars": {"pdf": "ct18z"},
    "scetlib_dyturbo_LatticeNP_CT18_N3p0LL_N2LO_pdfvars": {"pdf": "ct18"},
    "scetlib_dyturbo_LatticeNP_HERAPDF20_N3p0LL_N2LO_pdfvars": {
        "pdf": "herapdf20 herapdf20ext"
    },
    "scetlib_dyturbo_LatticeNP_MSHT20_N3p0LL_N2LO_pdfvars": {"pdf": "msht20"},
    "scetlib_dyturbo_LatticeNP_MSHT20aN3LO_N3p0LL_N2LO_pdfvars": {"pdf": "msht20an3lo"},
    "scetlib_dyturbo_LatticeNP_NNPDF40_N3p0LL_N2LO_pdfvars": {"pdf": "nnpdf40"},
    "scetlib_dyturbo_LatticeNP_PDF4LHC21_N3p0LL_N2LO_pdfvars": {"pdf": "pdf4lhc21"},
    "scetlib_dyturbo_LatticeNP_NNPDF31_N3p0LL_N2LO_pdfvars": {"pdf": "nnpdf31"},
}


def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preds",
        nargs="+",
        help="List of theory preds to process. (default: %(default)s)",
        default=list(THEORY_PREDS.keys()),
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=str,
        default=f"{os.environ.get('MY_OUT_DIR', '.')}/{datetime.now().strftime('%y%m%d')}_pdfsFromCorrByHelicity/",
    )
    parser.add_argument(
        "--skim",
        action="store_true",
        help="If set, will run a skimming step to only keep the PDF histograms in the file, saving a new output file.",
    )
    parser.add_argument(
        "--bosons",
        nargs="+",
        choices=["W", "Z"],
        default=["W", "Z"],
        help="Bosons to process. Choose one or both of: W Z (default: %(default)s)",
    )

    return parser.parse_args()


def main():

    args = parse_arguments()

    print("Generating histograms by helicity for the following theory preds:")
    print(args.preds)
    print("Processing bosons:")
    print(args.bosons)
    print("Will output to directory:")
    print(args.outdir)

    filter_procs = []
    aggregate_groups = []
    if "Z" in args.bosons:
        filter_procs.append("Zmumu_13TeVGen")
        aggregate_groups.append("Zmumu")
    if "W" in args.bosons:
        filter_procs.extend(["Wplusmunu_13TeVGen", "Wminusmunu_13TeVGen"])
        aggregate_groups.append("Wmunu")

    filter_procs_arg = " ".join(f"'{proc}'" for proc in filter_procs)
    aggregate_groups_arg = " ".join(aggregate_groups)

    for pred in args.preds:

        pdf = THEORY_PREDS[pred]["pdf"]

        command = f"""
        python {os.environ['WREM_BASE']}/scripts/histmakers/w_z_gen_dists.py --theoryCorr {pred} \
        --filterProcs {filter_procs_arg} --aggregateGroups {aggregate_groups_arg} \
        -o {args.outdir} --addHelicityAxis --pdf {pdf} --maxFiles '-1' -j 300
        """
        print(f"Running command: {command}")
        os.system(command)

        if args.skim:
            pred_corr = f"{pred}_Corr"
            pdf_replace = f"_{pdf.split(' ')[0]}" if pdf != "ct18z" else ""
            input_file = (
                f"{args.outdir}/w_z_gen_dists_{pred_corr}_maxFiles_m1{pdf_replace}.hdf5"
            )
            output_file = input_file.replace(".hdf5", "_skimmed.hdf5")
            skim_command = (
                f"python {os.environ['WREM_BASE']}/utilities/open_narf_h5py.py "
                f"{input_file} "
                f"--filterHistsRegex '^(.*pdfvars_Corr.*|nominal_gen_pdf_uncorr)$' "
                f"--outfile {output_file}"
            )
            print(f"Running skimming command: {skim_command}")
            os.system(skim_command)

    print("All done!")


if __name__ == "__main__":
    main()
