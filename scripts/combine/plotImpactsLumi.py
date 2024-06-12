import argparse
from utilities.io_tools import combinetf_input #intput_tools, output_tools?
import pandas as pd
import matplotlib.pyplot as plt
import os
import mplhep as hep
import re

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-u", "--ungroup", action='store_true', help="Use ungrouped nuisances")
    parser.add_argument("-n", "--nuisance", type=str, help="Only print value for specific nuisance")
    parser.add_argument("-s", "--sort", action='store_true', help="Sort nuisances by impact")
    parser.add_argument("--inputfolder", type=str, help="Folder with fitresults, output ROOT file from combinetf")
    parser.add_argument("-o", "--outfile", default = 'luminosity_uncertainty_projection.png', type=str, help="name of output file")
    parser.add_argument("--addPDF", default=0, type=int, nargs="*", help="Also output in PDF if == 1")
    parser.add_argument("-lsc", "--lumiscales", default = [1.0], type = float, nargs='+', help="Additional luminosity scales to examine (default 1.0)")
    return parser.parse_args()

def get_file_names(folder_path):
    return os.listdir(folder_path)

def updateImpactsDict(args, fitresult, df, lumiscale, poi='Wmass'):
    """
    Update dictionary for the labels and values, for one lumiscale
    """
    impacts,labels,_ = combinetf_input.read_impacts_poi(fitresult, not args.ungroup, sort=args.sort, poi=poi, normalize = False)
    print(labels)

    lumiVal = lumiscale * 0.200181002
    if args.nuisance:
        if args.nuisance not in labels:
            raise ValueError(f"Invalid nuisance {args.nuisance}. Options are {labels}")
        nuisanceImpact = impacts[list(labels).index(args.nuisance)]*100
        new_row = {args.nuisance: nuisanceImpact, 'Lumi': lumiVal}
        df.loc[len(df)] = new_row

    else:
        # plot_labels = ['Total', 'Background', 'Theory', 'PDF', 'Data stat.', 'Luminosity']
        # plot_vals = [impacts[list(labels).index('Total')], impacts[list(labels).index('CMS_background')], \
        #             impacts[list(labels).index('theory')], impacts[list(labels).index('pdfCT18Z')], \
        #             impacts[list(labels).index('stat')], lumiVal]
        plot_labels = ['Total', 'Theory', 'Theory EW', 'PT Modeling', 'QCD Scale', 'PDF', 'Data Stat.', 'Luminosity']
        plot_vals = [impacts[list(labels).index('Total')], impacts[list(labels).index('theory')], \
                    impacts[list(labels).index('theory_ew')], impacts[list(labels).index('pTModeling')], \
                    impacts[list(labels).index('QCDscale')], impacts[list(labels).index('pdfCT18Z')], \
                    impacts[list(labels).index('stat')], lumiVal]
        print(impacts)

        # new_row = {k: v for k, v in zip(plot_labels, plot_vals)}
        # df.loc[len(df)] = new_row

def sortFileNames(inputFiles): # natural sorting
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(inputFiles, key=alphanum_key)

if __name__ == '__main__':
    args = parseArgs()
    lumiscales = sorted(args.lumiscales)
    inFolder = args.inputfolder
    inputFiles = sortFileNames(get_file_names(inFolder))


    if args.nuisance:
        df = pd.DataFrame(columns=[args.nuisance, 'Luminosity'])
    else:
        df = pd.DataFrame(columns=['Total', 'Theory', 'Theory EW', 'PT Modeling', 'QCD Scale', 'PDF', 'Data Stat.', 'Luminosity'])

    for i in range(len(lumiscales)):
        inputFile = inFolder + '/' + inputFiles[i]
        fitresult = combinetf_input.get_fitresult(inputFile)
        for poi in combinetf_input.get_poi_names(fitresult):
            updateImpactsDict(args, fitresult, df, lumiscales[i], poi)


    # plt.figure(figsize=(8, 8))
    # hep.cms.label(fontsize=20, data=False, label="Projection", com=13.6)
    # for column in df.columns:
    #     if column != 'Luminosity':  # Exclude the Luminosity column from plotting
    #         plt.plot(df['Luminosity'], df[column], label=column, marker='o')

    # plt.xlabel("Integrated luminosity (fb$^{-1})$")
    # plt.ylabel("Uncertainty in $m_{W}$ (MeV)")
    # plt.legend()

    # plt.savefig(args.outfile)
    # if args.addPDF:
    #     pdfFileName = (args.outfile).split('.')[0] + ".pdf"
    #     plt.savefig(pdfFileName, format = 'pdf')
