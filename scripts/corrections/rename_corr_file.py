"""
Rename a correction file in .pkl.lz4 format by changing substrings in the keys.
"""

import argparse
import pickle
import sys

import lz4.frame

parser = argparse.ArgumentParser()
parser.add_argument("input_file", type=str, help="Path to the input file to be renamed")
parser.add_argument(
    "output_file", type=str, help="Path to the output file with the new name"
)
parser.add_argument(
    "-a",
    "--automatic",
    action="store_true",
    help="Automatically rename keys based on input/output filenames",
)
parser.add_argument(
    "--oldstring",
    type=str,
    default="",
    help="Old string to be replaced in keys, for use without --automatic",
)
parser.add_argument(
    "--newstring",
    type=str,
    default="",
    help="New string to replace with in keys, for use without --automatic",
)
args = parser.parse_args()

with lz4.frame.open(args.input_file, "rb") as f:
    data = pickle.load(f)

# define mapping
if args.automatic:
    old_string = args.input_file.split("/")[-1].replace(".pkl.lz4", "").split("Corr")[0]
    new_string = (
        args.output_file.split("/")[-1].replace(".pkl.lz4", "").split("Corr")[0]
    )
    # sanitize new_string to remove trailing underscores
    if new_string.endswith("_"):
        new_string = new_string[:-1]
else:
    if not args.oldstring or not args.newstring:
        raise ValueError(
            "When not using --automatic, both --oldstring and --newstring must be provided."
        )
    old_string = args.oldstring
    new_string = args.newstring

print(f"Renaming keys: {old_string} -> {new_string}")

newdata = {}
for key in data.keys():

    if "meta_data" in key or "metadata" in key:
        # no renaming needed, just deep copy the whole thing
        newdata[key] = data[key]
    else:
        # for each subkey inside this dictionary, apply the rule
        for subkey in data[key].keys():
            new_subkey = subkey.replace(old_string, new_string)
            print(f"Renaming key: {key}/{subkey} -> {key}/{new_subkey}")
            if key not in newdata:
                newdata[key] = {}
            newdata[key][new_subkey] = data[key][subkey]

print(f"Writing renamed data to {args.output_file}")
# append the command used for renaming
newdata["meta_data"]["command"] += "; " + sys.executable + " " + " ".join(sys.argv)
with lz4.frame.open(args.output_file, "wb") as f:
    pickle.dump(newdata, f)
