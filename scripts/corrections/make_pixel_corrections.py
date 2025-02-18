import argparse
import pickle

import lz4.frame

from wremnants.datasets.datagroups import Datagroups
from wums import logging

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--inputFile", type=str, required=True)
parser.add_argument("--debug", action="store_true", help="Print debug output")
args = parser.parse_args()

logger = logging.setup_logger("make_pixel_correctons", 4 if args.debug else 3)

nominalName = "nominal"

histnames = []
histnames.append("hNValidPixelHitsTrig")
histnames.append("hNValidPixelHitsNonTrig")


datagroups = Datagroups(args.inputFile)

if datagroups.mode != "z_dilepton":
    raise ValueError("Expected input is the output from the dilepton histmaker")

for histname in histnames:
    datagroups.loadHistsForDatagroups(histname, syst="")


groups = datagroups.getDatagroups()

hNValidPixelHitsTrig_mc = (
    groups["Zmumu"].hists["hNValidPixelHitsTrig"]
    + groups["Ztautau"].hists["hNValidPixelHitsTrig"]
)

hNValidPixelHitsNonTrig_mc = (
    groups["Zmumu"].hists["hNValidPixelHitsNonTrig"]
    + groups["Ztautau"].hists["hNValidPixelHitsNonTrig"]
)


print(hNValidPixelHitsTrig_mc)

res = {
    "hNValidPixelHitsTrig_data": groups["Data"].hists["hNValidPixelHitsTrig"],
    "hNValidPixelHitsNonTrig_data": groups["Data"].hists["hNValidPixelHitsNonTrig"],
    "hNValidPixelHitsTrig_mc": hNValidPixelHitsTrig_mc,
    "hNValidPixelHitsNonTrig_mc": hNValidPixelHitsNonTrig_mc,
}

with lz4.frame.open("pixelcorr.pkl.lz4", "wb") as fout:
    pickle.dump(res, fout, protocol=pickle.HIGHEST_PROTOCOL)
