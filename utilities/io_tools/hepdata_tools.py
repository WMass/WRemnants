import ROOT

import narf
from utilities.io_tools import input_tools
from wums import logging

logger = logging.child_logger(__name__)


def make_mass_summary_histogram(
    dfs,
    outfile_root,
    column_labels,
    outhist_name="mass_summary",
    outhist_title="",
):

    nRows = dfs.shape[0]
    rf = input_tools.safeOpenRootFile(outfile_root, mode="recreate")
    n_columns = len(column_labels)
    hr2 = ROOT.TH2D(
        outhist_name,
        outhist_title,
        nRows,
        0.0,
        float(nRows),
        n_columns,
        -0.5,
        float(n_columns) - 0.5,
    )
    for ic in range(n_columns):
        hr2.GetYaxis().SetBinLabel(ic + 1, column_labels[ic])
    for ix, (k, v) in enumerate(dfs.iterrows()):
        hr2.GetXaxis().SetBinLabel(ix + 1, v.iloc[0])
        for iy in range(n_columns):
            hr2.SetBinContent(ix + 1, iy + 1, v.iloc[iy + 1])
    print(f"Saving histogram {hr2.GetName()} for HEPData in {outfile_root}")
    hr2.Write()
    rf.Close()


def make_postfit_pulls_and_impacts(
    df,
    outfile_root,
    columns_to_save,
    outhist_name="nuisanceInfo",
    outhist_title="",
):

    nRows = int(df.shape[0])
    rf = input_tools.safeOpenRootFile(outfile_root, mode="recreate")
    hr2 = ROOT.TH2D(
        outhist_name,
        outhist_title,
        nRows,
        0.0,
        float(nRows),
        len(columns_to_save),
        0.0,
        float(len(columns_to_save)),
    )
    logger.warning(f"Preparing histogram {hr2.GetName()} for HEPData")
    for ix in range(nRows):
        latexLabel = df.at[df.index[ix], "label_hepdata"]
        if "mass" in latexLabel:
            logger.warning(f"{ix+1} {latexLabel}")
        hr2.GetXaxis().SetBinLabel(ix + 1, latexLabel)
        logger.debug(f"{ix+1} {latexLabel}")
        for iy, y in enumerate(columns_to_save):
            hr2.GetYaxis().SetBinLabel(iy + 1, y)
            hr2.SetBinContent(ix + 1, iy + 1, df.at[df.index[ix], y])
    logger.warning(f"Saving histogram {hr2.GetName()} for HEPData in {outfile_root}")
    hr2.Write()
    rf.Close()


def save_histograms_to_root(
    hists,
    names,
    labels,
    outfile_root,
    xlabel,
    ylabel,
):

    rf = input_tools.safeOpenRootFile(outfile_root, mode="recreate")
    logger.warning(f"Saving histograms for HEPData in {outfile_root}")
    for ih, h in enumerate(hists):
        hroot = narf.hist_to_root(h)
        htitle = labels[ih]
        hname = names[ih].replace(" ", "_").replace("-", "_")
        hroot.SetName(hname)
        hroot.SetTitle(htitle)
        # divide by bin width
        hroot.Scale(1.0, "width")
        hroot.GetXaxis().SetTitle(xlabel)
        hroot.GetYaxis().SetTitle(ylabel)
        hroot.Write()
        logger.info(f"Saving histogram {hname}")
    rf.Close()
