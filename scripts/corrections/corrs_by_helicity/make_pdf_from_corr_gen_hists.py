"""
Convenience script to generate a the PDF gen histograms for a list of theory corrections.
"""

import argparse
import os
from datetime import datetime

THEORY_PREDS = {
    "scetlib_dyturbo_CT18Z_N3p0LL_N2LO_pdfvars": {"pdf": "ct18z"},
    "scetlib_dyturbo_LatticeNP_CT18Z_N3p0LL_N2LO_pdfvars": {"pdf": "ct18z"},
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

    return parser.parse_args()


def main():

    args = parse_arguments()

    print("Generating histograms by helicity for the following theory preds:")
    print(args.preds)
    print("Will output to directory:")
    print(args.outdir)

    for pred in args.preds:

        command = f"""
        python {os.environ['WREM_BASE']}/scripts/histmakers/w_z_gen_dists.py --theoryCorr {pred} \
        --filterProcs 'Zmumu_MiNNLO' 'Wplusmunu_MiNNLO' 'Wminusmunu_MiNNLO' --aggregateGroups Zmumu Wmunu \
        -o {args.outdir} --addHelicityAxis --pdf {THEORY_PREDS[pred]['pdf']} --maxFiles '-1' -j 300
        """
        print(f"Running command: {command}")
        os.system(command)

        if args.skim:
            skim_command = f"python {os.environ['WREM_BASE']}/utilities/open_narf_h5py.py {args.outdir}/w_z_gen_dists_{pred + "_Corr"}_maxFiles_m1.hdf5 --filterHistsRegex '^(.*pdfvars_Corr.*|nominal_gen_pdf_uncorr)$' --outfile {args.outdir}/w_z_gen_dists_{pred + "_Corr"}_maxFiles_m1_skimmed.hdf5"
            print(f"Running skimming command: {skim_command}")
            os.system(skim_command)

    print("All done!")


if __name__ == "__main__":
    main()
