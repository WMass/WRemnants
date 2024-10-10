import argparse

from utilities.io_tools import combinetf_input


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-u", "--ungroup", action="store_true", help="Use ungrouped nuisances"
    )
    parser.add_argument(
        "-n", "--nuisance", type=str, help="Only print value for specific nuiance"
    )
    parser.add_argument(
        "-s", "--sort", action="store_true", help="Sort nuisances by impact"
    )
    parser.add_argument(
        "inputFile",
        default="/Users/kenneth/cernbox/CombineStudies/WGen/etal_ptl_smear_unrolled_scetlib/fitresults_123456789.root",
        help="fitresults output ROOT file from combinetf",
    )
    return parser.parse_args()


def printImpacts(args, fitresult, poi="Wmass"):
    impacts, labels, _ = combinetf_input.read_impacts_poi(
        fitresult, not args.ungroup, sort=args.sort, poi=poi, normalize=False
    )
    unit = "MeV" if poi.startswith("mass") else "n.u. %"
    if args.nuisance:
        if args.nuisance not in labels:
            raise ValueError(f"Invalid nuisance {args.nuisance}. Options are {labels}")
        print(
            f"Impact of nuisance {args.nuisance} is {impacts[list(labels).index(args.nuisance)]*100} {unit}"
        )
    else:
        print(f"Impact of all systematics (in {unit})")
        print(
            "\n".join([f"   {k}: {round(v*100, 2)}" for k, v in zip(labels, impacts)])
        )


if __name__ == "__main__":
    args = parseArgs()
    fitresult = combinetf_input.get_fitresult(args.inputFile)
    for poi in combinetf_input.get_poi_names(fitresult):
        printImpacts(args, fitresult, poi)
