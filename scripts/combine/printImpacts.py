import uproot
import argparse
from utilities import input_tools

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-u", "--ungroup", action='store_true', help="Use ungrouped nuisances")
    parser.add_argument("-n", "--nuisance", type=str, help="Only print value for specific nuiance")
    parser.add_argument("-s", "--sort", action='store_true', help="Sort nuisances by impact")
    parser.add_argument("-f", "--inputFile", 
        default="/Users/kenneth/cernbox/CombineStudies/WGen/etal_ptl_smear_unrolled_scetlib/fitresults_123456789.root", 
        help="fitresults output ROOT file from combinetf")
    return parser.parse_args()


def printImpacts(args,fitresult,POI='Wmass'):
    impacts,labels,_ = input_tools.readImpacts(fitresult, not args.ungroup, sort=args.sort, POI=POI, normalize = False)
    unit = 'MeV' if POI=='Wmass' else 'n.u. %'
    if args.nuisance:
        if args.nuisance not in labels:
            raise ValueError(f"Invalid nuisance {args.nuisance}. Options are {labels}")
        print(f"Impact of nuisance {args.nuisance} is {impacts[list(labels).index(args.nuisance)]*100} {unit}")
    else:
        print(f"Impact of all systematics (in {unit})")
        print("\n".join([f"   {k}: {round(v*100, 2)}" for k,v in zip(labels, impacts)]))


if __name__ == '__main__':
    args = parseArgs()
    fitresult = input_tools.getFitresult(args.inputFile)
    for poi in input_tools.getPOInames(fitresult):
        printImpacts(args,fitresult,poi)
