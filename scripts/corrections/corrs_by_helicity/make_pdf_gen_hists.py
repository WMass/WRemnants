"""
Convenience script to generate a the PDF gen histograms for a list of PDF sets.
"""

import argparse
import os
from datetime import datetime

PDF_SETS = [
    "ct18z",
    "msht20mcrange_renorm",
    "msht20mbrange_renorm",
    "nnpdf31",
    "ct18",
    "nnpdf30",
    "nnpdf40",
    "pdf4lhc21",
    "msht20",
    "msht20an3lo",
    "atlasWZj20",
    "herapdf20",
    "herapdf20ext",
]


def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pdf", nargs="+", help="List of PDFs to process", default=PDF_SETS
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=str,
        default=f"{os.environ.get('MY_OUT_DIR', '.')}/{datetime.now().strftime('%y%m%d')}_pdfsByHelicity/",
    )
    parser.add_argument(
        "--skim",
        action="store_true",
        help="If set, will run a skimming step to only keep the PDF histograms in the file, saving a new output file.",
    )
    parser.add_argument(
        "-j",
        "--njobs",
        type=int,
        default=300,
        help="Number of parallel threads",
    )
    return parser.parse_args()


def main():

    args = parse_arguments()

    print("Generating histograms by helicity for the following PDFs:")
    print(args.pdf)
    print("Will output to directory:")
    print(args.outdir)

    for pdf in args.pdf:

        command = f"""
        python {os.environ['WREM_BASE']}/scripts/histmakers/w_z_gen_dists.py --pdf {pdf} -o {args.outdir} --maxFiles '-1' -j {args.njobs} \
        --filterProcs 'Zmumu_13TeVGen' 'Wplusmunu_13TeVGen' 'Wminusmunu_13TeVGen' --aggregateGroups Zmumu Wmunu \
        --addHelicityAxis --postfix pdfByHelicity
        """
        print(f"Running command: {command}")
        os.system(command)

        if args.skim:
            pdf_replace = f"_{pdf}" if pdf != "ct18z" else ""
            skim_command = f"""
            python {os.environ['WREM_BASE']}/utilities/open_narf_h5py.py {args.outdir}/w_z_gen_dists_maxFiles_m1{pdf_replace}_pdfByHelicity.hdf5 \
            --filterHists nominal_gen_pdf --excludeHists alpha --outfile {args.outdir}/w_z_gen_dists_maxFiles_m1_{pdf}_pdfByHelicity_skimmed.hdf5
            """
            print(f"Running skimming command: {skim_command}")
            os.system(skim_command)

    print("All done!")


if __name__ == "__main__":
    main()
