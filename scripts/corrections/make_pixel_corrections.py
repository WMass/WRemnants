import argparse
import pickle

import lz4.frame

from wremnants.postprocessing.datagroups.datagroups import Datagroups
from wums import logging

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--inputFile", type=str, required=True)
parser.add_argument("--debug", action="store_true", help="Print debug output")
args = parser.parse_args()

logger = logging.setup_logger("make_pixel_correctons", 4 if args.debug else 3)

# The dilepton histmaker writes the per-leg valid-pixel-hit spectra named by
# muon position (first/second) per category. The pixel-multiplicity correction
# distinguishes triggering vs non-triggering muons; here the first (second) muon
# fills the triggering (nonTriggering) slot positionally (kept for backward
# compatibility, see the note in mz_dilepton.py).
nominalName = "nominal"
firstHist = f"{nominalName}_hNValidPixelHitsFirst"
secondHist = f"{nominalName}_hNValidPixelHitsSecond"

datagroups = Datagroups(args.inputFile)

if datagroups.mode != "z_dilepton":
    raise ValueError("Expected input is the output from the dilepton histmaker")

for histname in [firstHist, secondHist]:
    datagroups.loadHistsForDatagroups(histname, syst="")


groups = datagroups.groups

hNValidPixelHitsTrig_mc = (
    groups["Zmumu"].hists[firstHist] + groups["Ztautau"].hists[firstHist]
)

hNValidPixelHitsNonTrig_mc = (
    groups["Zmumu"].hists[secondHist] + groups["Ztautau"].hists[secondHist]
)


print(hNValidPixelHitsTrig_mc)

res = {
    "hNValidPixelHitsTrig_data": groups["Data"].hists[firstHist],
    "hNValidPixelHitsNonTrig_data": groups["Data"].hists[secondHist],
    "hNValidPixelHitsTrig_mc": hNValidPixelHitsTrig_mc,
    "hNValidPixelHitsNonTrig_mc": hNValidPixelHitsNonTrig_mc,
}

with lz4.frame.open("pixelcorr.pkl.lz4", "wb") as fout:
    pickle.dump(res, fout, protocol=pickle.HIGHEST_PROTOCOL)
