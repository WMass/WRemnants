import os

from utilities import parsing
from wremnants.datasets.datagroups import Datagroups
from wums import logging

analysis_label = Datagroups.analysisLabel(os.path.basename(__file__))
parser, initargs = parsing.common_parser(analysis_label)

args = parser.parse_args()

logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)

import hist

import narf
from wremnants.datasets.dataset_tools import getDatasets
from wremnants.histmaker_tools import write_analysis_output

datasets = getDatasets(
    maxFiles=args.maxFiles,
    filt=args.filterProcs,
    excl=args.excludeProcs,
    base_path=args.dataPath,
    mode=analysis_label,
    era=args.era,
)

# define histogram axes, see: https://hist.readthedocs.io/en/latest/index.html
axis_nLepton = hist.axis.Integer(0, 5, name="nLepton", underflow=False)


def build_graph(df, dataset):
    logger.info(f"build graph for dataset: {dataset.name}")

    results = []

    if dataset.is_data:
        df = df.DefinePerSample("weight", "1.0")
    else:
        df = df.Define("weight", "std::copysign(1.0, genWeight)")

    weightsum = df.SumAndCount("weight")

    # filter events
    df = df.Filter("HLT_HIMu17")

    # available columns, see: https://cms-xpog.docs.cern.ch/autoDoc/

    # define new columns
    df = df.Define("nLepton", "nElectron + nMuon")

    # fill histogram
    hist_nLepton = df.HistoBoost("nLepton", [axis_nLepton], ["nLepton"])

    results.append(hist_nLepton)

    return results, weightsum


resultdict = narf.build_and_run(datasets, build_graph)

fout = f"{os.path.basename(__file__).replace('py', 'hdf5')}"
write_analysis_output(resultdict, fout, args)
