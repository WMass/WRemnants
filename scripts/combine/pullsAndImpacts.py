import argparse
import json
import os
import re

import dash
import dash_daq as daq
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# prevent MathJax from bein loaded
import plotly.io as pio
import ROOT
from dash import dcc, html
from dash.dependencies import Input, Output
from plotly.subplots import make_subplots

from narf import ioutils
from utilities import logging
from utilities.common import base_dir
from utilities.io_tools import (
    combinetf_input,
    conversion_tools,
    input_tools,
    output_tools,
)
from utilities.styles.styles import nuisance_groupings as groupings
from wremnants import plot_tools

pio.kaleido.scope.mathjax = None

logger = logging.child_logger(__name__)


def writeOutput(fig, outfile, extensions=[], postfix=None, args=None, meta_info=None):
    name, _ = os.path.splitext(outfile)

    if postfix:
        name += f"_{postfix}"

    for ext in extensions:
        if ext[0] != ".":
            ext = "." + ext
        output = name + ext
        logger.info(f"Write output file {output}")
        if ext == ".html":
            fig.write_html(output, include_mathjax=False)
        else:
            fig.write_image(output)

        output = name.rsplit("/", 1)
        output[1] = os.path.splitext(output[1])[0]
        if len(output) == 1:
            output = (None, *output)
    if args is None and meta_info is None:
        return
    plot_tools.write_index_and_log(
        *output,
        args=args,
        analysis_meta_info={"AnalysisOutput": meta_info},
    )


def get_marker(filled=True, color="#377eb8", opacity=1.0):
    if filled:
        marker = {
            "marker": {
                "color": color,  # Fill color for the filled bars
                "opacity": opacity,  # Opacity for the filled bars (adjust as needed)
            }
        }
    else:
        marker = {
            "marker": {
                "color": "rgba(0, 0, 0, 0)",  # Transparent fill color
                "opacity": opacity,
                "line": {"color": color, "width": 2},  # Border color  # Border width
            }
        }
    return marker


def plotImpacts(
    df,
    impact_title="",
    pulls=False,
    normalize=False,
    oneSidedImpacts=False,
    pullrange=None,
    cmsDecor=None,
    impacts=True,
    asym_pulls=False,
    include_ref=False,
    ref_name="ref.",
    show_numbers=False,
    show_legend=True,
    legend_pos="bottom",
):
    impacts = bool(np.count_nonzero(df["absimpact"])) and impacts
    ncols = pulls + impacts
    fig = make_subplots(rows=1, cols=ncols, horizontal_spacing=0.1, shared_yaxes=True)

    if cmsDecor == "Supplementary":
        loffset = 140
    elif cmsDecor == "Preliminary":
        loffset = 110
    else:
        loffset = 50

    if legend_pos == "bottom":
        legend = dict(
            orientation="h",
            xanchor="left",
            yanchor="top",
            x=0.0,
            y=0.0,
        )
    elif legend_pos == "right":
        legend = dict(
            orientation="v",
            xanchor="left",
            yanchor="top",
            x=1.0,
            y=1.0,
        )
    else:
        raise NotImplementedError("Supported legend positions are ['bottom', 'left']")

    ndisplay = len(df)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title=impact_title if impacts else "Pull",
        margin=dict(l=loffset, r=20, t=50, b=20),
        yaxis=dict(range=[-1, ndisplay]),
        showlegend=show_legend,
        legend=legend,
        legend_itemsizing="constant",
        height=100 * (ndisplay < 100) + ndisplay * 20.5,
        width=800 if show_legend and legend_pos == "right" else 640,
        font=dict(
            color="black",
        ),
    )

    gridargs = dict(
        showgrid=True,
        gridwidth=1,
        gridcolor="Gray",
        griddash="dash",
        zeroline=True,
        zerolinewidth=2,
        zerolinecolor="Gray",
    )
    tickargs = dict(
        tick0=0.0,
        tickmode="linear",
        tickangle=0,
        side="top",
    )

    text_on_bars = False
    labels = df["label"]
    if impacts and show_numbers:
        if include_ref:
            # append numerical values of impacts on nuisance name; fill up empty room with spaces to align numbers
            frmt = (
                "{:0"
                + str(
                    int(
                        np.log10(max(df["absimpact"]))
                        if max(df[f"absimpact_ref"]) > 0
                        else 0
                    )
                    + 2
                )
                + ".2f}"
            )
            nval = df["absimpact"].apply(
                lambda x, frmt=frmt: frmt.format(x)
            )  # .astype(str)
            nspace = nval.apply(
                lambda x, n=nval.apply(len).max(): " " * (n - len(x) + 1)
            )
            if include_ref:
                frmt_ref = (
                    "{:0"
                    + str(
                        int(
                            np.log10(max(df[f"absimpact_ref"]))
                            if max(df[f"absimpact_ref"]) > 0
                            else 0
                        )
                        + 2
                    )
                    + ".2f}"
                )
                nval_ref = df[f"absimpact_ref"].apply(
                    lambda x, frmt=frmt_ref: " (" + frmt.format(x) + ")"
                )
                nspace_ref = nval_ref.apply(
                    lambda x, n=nval_ref.apply(len).max(): " " * (n - len(x))
                )
                nval = nval + nspace_ref + nval_ref
            labels = labels + nspace + nval
        else:
            text_on_bars = True

    if impacts:

        def make_bar(
            key="impact",
            sign=1,
            color="#377eb8",
            name="+1σ impact",
            text_on_bars=False,
            filled=True,
            opacity=1,
        ):
            x = (
                np.where(sign * df[key] < 0, np.nan, sign * df[key])
                if oneSidedImpacts
                else sign * df[key]
            )

            if text_on_bars:
                text = np.where(np.isnan(x), None, [f"{value:.2f}" for value in x])
            else:
                text = None

            return go.Bar(
                orientation="h",
                x=x,
                y=labels,
                text=text,
                textposition="outside",
                **get_marker(filled=filled, color=color, opacity=opacity),
                name=name,
            )

        fig.add_trace(
            make_bar(text_on_bars=text_on_bars, opacity=0.5 if include_ref else 1),
            row=1,
            col=1,
        )
        if include_ref:
            fig.add_trace(
                make_bar(
                    key="impact_ref", name=f"+1σ impact ({ref_name})", filled=False
                ),
                row=1,
                col=1,
            )

        fig.add_trace(
            make_bar(
                name="-1σ impact",
                sign=-1,
                color="#e41a1c",
                text_on_bars=text_on_bars,
                opacity=0.5 if include_ref else 1,
            ),
            row=1,
            col=1,
        )
        if include_ref:
            fig.add_trace(
                make_bar(
                    key="impact_ref",
                    name=f"-1σ impact ({ref_name})",
                    sign=-1,
                    color="#e41a1c",
                    filled=False,
                ),
                row=1,
                col=1,
            )

        # impact range in steps of 0.5
        impact_range = np.ceil(df["impact"].max() * 2) / 2
        if include_ref:
            impact_range = max(impact_range, np.ceil(df[f"impact_ref"].max() * 2) / 2)
        impact_spacing = min(impact_range, 2 if pulls else 3)
        if impact_range % impact_spacing:
            impact_range += impact_spacing - (impact_range % impact_spacing)
        tick_spacing = impact_range / impact_spacing
        if pulls and oneSidedImpacts:
            tick_spacing /= 2.0
        if tick_spacing > 0.5 * impact_range:  # make sure to have at least two ticks
            tick_spacing /= 2.0
        fig.update_layout(barmode="overlay")
        fig.update_layout(
            xaxis=dict(
                range=[
                    -impact_range * 1.1 if not oneSidedImpacts else -impact_range / 20,
                    impact_range * 1.1,
                ],
                dtick=tick_spacing,
                **gridargs,
                **tickargs,
            ),
        )

    if pulls:
        fig.add_trace(
            go.Scatter(
                x=df["pull"],
                y=labels,
                mode="markers",
                marker=dict(
                    color="black",
                    size=8,
                ),
                error_x=dict(
                    array=df["constraint"],
                    color="black",
                    thickness=1.5,
                    width=5,
                ),
                name="Pulls ± Constraints",
                showlegend=include_ref,
            ),
            row=1,
            col=ncols,
        )
        if include_ref:
            fig.add_trace(
                go.Bar(
                    base=df["pull_ref"] - df["constraint_ref"],
                    x=2 * df["constraint_ref"],
                    y=labels,
                    orientation="h",
                    **get_marker(filled=False, color="black"),
                    name=f"Pulls ± Constraints ({ref_name})",
                    showlegend=True,
                ),
                row=1,
                col=ncols,
            )

        if asym_pulls:
            fig.add_trace(
                go.Scatter(
                    x=df["newpull"],
                    y=labels,
                    mode="markers",
                    marker=dict(
                        color="green",
                        symbol="x",
                        size=8,
                        # line=dict(width=1),  # Adjust the thickness of the marker lines
                    ),
                    name="Asym. pulls",
                    showlegend=include_ref,
                ),
                row=1,
                col=ncols,
            )

            if include_ref:
                fig.add_trace(
                    go.Scatter(
                        x=df["newpull_ref"],
                        y=labels,
                        mode="markers",
                        marker=dict(
                            color="green",
                            symbol="circle-open",
                            size=8,
                            line=dict(
                                width=1
                            ),  # Adjust the thickness of the marker lines
                        ),
                        name=f"Asym. pulls ({ref_name})",
                        showlegend=include_ref,
                    ),
                    row=1,
                    col=ncols,
                )
        max_pull = np.max(df["abspull"])
        if pullrange is None:
            # Round up to nearest 0.5, add 1.1 for display
            pullrange = 0.5 * np.ceil(max_pull) + 1.1
        # Keep it a factor of 0.5, but no bigger than 1
        spacing = min(1, np.ceil(pullrange) / 2.0)
        if spacing > 0.5 * pullrange:  # make sure to have at least two ticks
            spacing /= 2.0
        xaxis_title = "Nuisance parameter"
        #  (
        #     "θ - θ<sub>0</sub> <span style='color:blue'>θ - θ<sub>0</sub> / √(σ<sup>2</sup>-σ<sub>0</sub><sup>2</sup>) </span>"
        #     if asym_pulls
        #     else "Nuisance parameter"  # "θ - θ<sub>0</sub>"
        # )
        info = dict(
            xaxis=dict(
                range=[-pullrange, pullrange], dtick=spacing, **gridargs, **tickargs
            ),
            xaxis_title=xaxis_title,
            yaxis=dict(range=[-1, ndisplay]),
            yaxis_visible=not impacts,
        )
        if impacts:
            new_info = {}
            for k in info.keys():
                new_info[k.replace("axis", "axis2")] = info[k]
            info = new_info
        fig.update_layout(barmode="overlay", **info)

    if cmsDecor is not None:
        # add CMS decor
        fig.add_annotation(
            x=0,
            y=1,
            xshift=-loffset,
            yshift=50,
            xref="paper",
            yref="paper",
            showarrow=False,
            text="CMS",
            font=dict(size=24, color="black", family="Arial", weight="bold"),
        )
        if cmsDecor != "":
            fig.add_annotation(
                x=0,
                y=1,
                xshift=-loffset,
                yshift=25,
                xref="paper",
                yref="paper",
                showarrow=False,
                text=f"<i>{cmsDecor}</i>",
                font=dict(
                    size=20,
                    color="black",
                    family="Arial",
                ),
            )

    return fig


def readFitInfoFromFile(
    rf,
    filename,
    poi,
    group=False,
    grouping=None,
    filters=[],
    stat=0.0,
    normalize=False,
    scale=1,
    saveForHepdata=False,
):
    logger.debug("Read impacts for poi from file")
    impacts, labels, norm = combinetf_input.read_impacts_poi(
        rf, group, add_total=group, stat=stat, poi=poi, normalize=normalize
    )

    if (group and grouping) or filters:
        filtimpacts = []
        filtlabels = []
        for impact, label in zip(impacts, labels):
            if group and grouping and label not in grouping:
                continue
            if filters and not any(re.search(f, label) for f in filters):
                continue
            filtimpacts.append(impact)
            filtlabels.append(label)
        impacts = filtimpacts
        labels = filtlabels

    df = pd.DataFrame(np.array(impacts, dtype=np.float64).T * scale, columns=["impact"])
    df["label"] = [translate_label.get(l, l) for l in labels]
    if saveForHepdata and not group:
        convert_hepdata_json = (
            base_dir + "/utilities/styles/nuisance_translate_hepdata.json"
        )
        logger.warning(
            f"Using file {convert_hepdata_json} to convert names of nuisance parameters for HEPData"
        )
        with open(convert_hepdata_json) as hepf:
            translate_label_hepdata = json.load(hepf)
        labels_hepdata = []
        for l in labels:
            new_label = conversion_tools.get_hepdata_label(
                l, input_dict=translate_label_hepdata
            )
            labels_hepdata.append(new_label)
        df["label_hepdata"] = labels_hepdata
        logger.warning("HEPData labels were created")

    df["absimpact"] = np.abs(df["impact"])
    if not group:
        df["pull"], df["constraint"], df["pull_prefit"] = (
            combinetf_input.get_pulls_and_constraints(filename, labels)
        )
        df["pull"] = df["pull"] - df["pull_prefit"]
        df["abspull"] = np.abs(df["pull"])
        df["newpull"] = df["pull"] / (1 - df["constraint"] ** 2) ** 0.5
        df["newpull"] = df["newpull"].replace([np.inf, -np.inf, np.nan], 999)
        if poi:
            df = df.drop(
                df.loc[
                    df["label"].str.contains(poi.replace("_noi", ""), regex=True)
                ].index
            )

    return df


def parseArgs():
    sort_choices = ["label", "pull", "abspull", "constraint", "absimpact"]
    sort_choices += [
        *[
            f"{c}_diff" for c in sort_choices
        ],  # possibility to sort based on largest difference between input and referencefile
        *[
            f"{c}_ref" for c in sort_choices
        ],  # possibility to sort based on reference file
        *[f"{c}_both" for c in sort_choices],
    ]  # possibility to sort based on the largest/smallest of both input and reference file

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--inputFile",
        type=str,
        required=True,
        help="fitresults output ROOT/hdf5 file from combinetf",
    )
    parser.add_argument(
        "-r",
        "--referenceFile",
        type=str,
        help="fitresults output ROOT/hdf5 file from combinetf for reference",
    )
    parser.add_argument(
        "--refName",
        type=str,
        help="Name of reference input for legend",
    )
    parser.add_argument(
        "-s",
        "--sort",
        default="absimpact",
        type=str,
        help="Sort mode for nuisances",
        choices=sort_choices,
    )
    parser.add_argument(
        "--stat",
        default=0.0,
        type=float,
        help="Overwrite stat. uncertainty with this value",
    )
    parser.add_argument(
        "-d",
        "--sortDescending",
        dest="ascending",
        action="store_false",
        help="Sort mode for nuisances",
    )
    parser.add_argument(
        "-m",
        "--mode",
        choices=["group", "ungrouped", "both"],
        default="both",
        help="Impact mode",
    )
    parser.add_argument(
        "--absolute",
        action="store_true",
        help="Not normalize impacts on cross sections and event numbers.",
    )
    parser.add_argument("--debug", action="store_true", help="Print debug output")
    parser.add_argument(
        "--diffPullAsym",
        action="store_true",
        help="Also add the pulls after the diffPullAsym definition",
    )
    parser.add_argument(
        "--oneSidedImpacts", action="store_true", help="Make impacts one-sided"
    )
    parser.add_argument(
        "--filters",
        nargs="*",
        type=str,
        help="Filter regexes to select nuisances by name",
    )
    parser.add_argument(
        "--grouping",
        type=str,
        default=None,
        help="Select nuisances by a predefined grouping",
        choices=groupings.keys(),
    )
    parser.add_argument(
        "-t",
        "--translate",
        type=str,
        default=None,
        help="Specify .json file to translate labels",
    )
    parser.add_argument(
        "--cmsDecor",
        default="Preliminary",
        nargs="?",
        type=str,
        choices=[
            None,
            "",
            "Preliminary",
            "Work in progress",
            "Internal",
            "Supplementary",
        ],
        help="CMS label",
    )
    parser.add_argument("--noImpacts", action="store_true", help="Don't show impacts")
    parser.add_argument(
        "--showNumbers", action="store_true", help="Show values of impacts"
    )
    parser.add_argument(
        "--poi",
        type=str,
        default=None,
        help="Specify POI to make impacts for, otherwise use all",
    )
    parser.add_argument(
        "--poiType", type=str, default=None, help="POI type to make impacts for"
    )
    parser.add_argument(
        "--pullrange", type=float, default=None, help="POI type to make impacts for"
    )
    parsers = parser.add_subparsers(dest="output_mode")
    interactive = parsers.add_parser(
        "interactive", help="Launch and interactive dash session"
    )
    interactive.add_argument(
        "-i",
        "--interface",
        default="localhost",
        help="The network interface to bind to.",
    )
    output = parsers.add_parser(
        "output", help="Produce plots as output (not interactive)"
    )
    output.add_argument(
        "-o",
        "--outputFile",
        default="test.html",
        type=str,
        help="Output file (extension specifies if html or pdf/png)",
    )
    output.add_argument(
        "--outFolder",
        type=str,
        default="",
        help="Output folder (created if it doesn't exist)",
    )
    output.add_argument(
        "--otherExtensions",
        default=[],
        type=str,
        nargs="*",
        help="Additional output file types to write",
    )
    output.add_argument("-n", "--num", type=int, help="Number of nuisances to plot")
    output.add_argument(
        "--noPulls",
        action="store_true",
        help="Don't show pulls (not defined for groups)",
    )
    output.add_argument(
        "--eoscp",
        action="store_true",
        help="Use of xrdcp for eos output rather than the mount",
    )
    parser.add_argument(
        "--saveForHepdata",
        action="store_true",
        help="Save output as ROOT to prepare HEPData",
    )

    return parser.parse_args()


app = dash.Dash(__name__)


@app.callback(
    Output("scatter-plot", "figure"),
    [Input("maxShow", "value")],
    [Input("sortBy", "value")],
    [Input("sortDescending", "on")],
    [Input("filterLabels", "value")],
    [Input("groups", "on")],
)
def producePlots(
    fitresult,
    args,
    poi,
    group=False,
    normalize=False,
    fitresult_ref=None,
    grouping=None,
    pullrange=None,
):
    poi_type = poi.split("_")[-1] if poi else None

    if poi is not None and "MeV" in poi:
        scale = float(re.findall(r"\d+", poi.split("MeV")[0].replace("p", "."))[-1])
        if "Diff" in poi:
            scale *= 2  # take diffs by 2 as up and down pull in opposite directions
    else:
        scale = 1

    if poi and poi.startswith("massShift"):
        label = poi.replace("massShift", "")[0]
        impact_title = f"Impact on <i>m</i><sub>{label}</sub> (MeV)"
    elif poi and poi.startswith("massDiff"):
        if poi.startswith("massDiffCharge"):
            impact_title = "Impact on mass diff. (charge) (MeV)"
        elif poi.startswith("massDiffEta"):
            impact_title = "Impact on mass diff. η (MeV)"
        else:
            impact_title = "Impact on mass diff. (MeV)"
    elif poi and poi.startswith("width"):
        impact_title = "Impact on width (MeV)"
    elif poi_type in ["pmaskedexp", "pmaskedexpnorm", "sumxsec", "sumxsecnorm"]:
        if poi_type in ["pmaskedexp", "sumxsec"]:
            meta = ioutils.pickle_load_h5py(fitresult["meta"])
            channel_info = conversion_tools.combine_channels(meta, True)
            if len(channel_info.keys()) == 1:
                lumi = channel_info["chan_13TeV"]["lumi"]
            else:
                raise NotImplementedError(
                    f"Found channels {[k for k in channel_info.keys()]} but only one channel is supported."
                )
            scale = 1.0 / (lumi * 1000)
            poi_name = "_".join(poi.split("_")[:-1])
            impact_title = "σ<sub>fid</sub>(" + poi_name + ") (pb)"
        else:
            impact_title = "1/σ<sub>fid</sub> dσ"
    elif poi_type in ["ratiometaratio"]:
        poi_name = "_".join(poi.split("_")[:-1]).replace("r_", "")
        impact_title = f"Impact on ratio {poi_name} *1000"
        scale = 1000
    elif poi in ["pdfAlphaS_noi"]:
        scale = 1.5
        impact_title = "Impact on <i>α</i><sub>S</sub> in 10<sup>-3</sup>"
    else:
        impact_title = poi

    if not (group and args.output_mode == "output"):
        df = readFitInfoFromFile(
            fitresult,
            args.inputFile,
            poi,
            False,
            filters=args.filters,
            stat=args.stat / 100.0,
            normalize=normalize,
            scale=scale,
            saveForHepdata=args.saveForHepdata,
        )
    elif group:
        df = readFitInfoFromFile(
            fitresult,
            args.inputFile,
            poi,
            True,
            filters=args.filters,
            stat=args.stat / 100.0,
            normalize=normalize,
            scale=scale,
            grouping=grouping,
            saveForHepdata=args.saveForHepdata,
        )

    if fitresult_ref:
        df_ref = readFitInfoFromFile(
            fitresult_ref,
            args.referenceFile,
            poi,
            group,
            filters=args.filters,
            stat=args.stat / 100.0,
            normalize=normalize,
            scale=scale,
            grouping=grouping,
            saveForHepdata=False,  # can stay false here, it would only create new labels
        )
        df = df.merge(df_ref, how="outer", on="label", suffixes=("", "_ref"))

    if df.empty:
        logger.warning("Empty dataframe")
        if group and grouping:
            logger.warning(
                f"This can happen if no group is found that belongs to {grouping}"
            )
            logger.warning(
                "Try a different mode for --grouping or use '--mode ungrouped' to skip making impacts for groups"
            )
        logger.warning("Skipping this part")
        return

    if args.sort:
        logger.debug("Sort impacts")
        if args.sort.endswith("diff"):
            logger.debug("Sort impacts")
            key = args.sort.replace("_diff", "")
            df[f"{key}_diff"] = abs(df[key] - df[f"{key}_ref"])
        elif args.sort.endswith("both"):
            key = args.sort.replace("_both", "")
            if args.ascending:
                df[f"{key}_both"] = df[[key, f"{key}_ref"]].max(axis=1)
            else:
                df[f"{key}_both"] = df[[key, f"{key}_ref"]].min(axis=1)

        df = df.sort_values(by=args.sort, ascending=args.ascending)

    df = df.fillna(0)

    logger.debug("Make plots")
    if args.output_mode == "interactive":
        app.layout = html.Div(
            [
                dcc.Input(
                    id="maxShow",
                    type="number",
                    placeholder="maxShow",
                    min=0,
                    max=10000,
                    step=1,
                ),
                dcc.Input(
                    id="filterLabels",
                    type="text",
                    placeholder="filter labels (comma-separated list)",
                    style={"width": "25%"},
                ),
                html.Br(),
                html.Label("Sort by"),
                dcc.Dropdown(
                    id="sortBy",
                    options=[
                        {"label": v, "value": v.lower()}
                        for v in ["Impact", "Pull", "Constraint", "Label"]
                    ],
                    placeholder="select sort criteria...",
                    style={"width": "50%"},
                    value=args.sort,
                ),
                daq.BooleanSwitch(
                    "sortDescending",
                    label="Decreasing order",
                    labelPosition="top",
                    on=True,
                ),
                daq.BooleanSwitch(
                    "groups",
                    label="Show nuisance groups",
                    labelPosition="top",
                    on=False,
                ),
                dcc.Graph(id="scatter-plot", style={"width": "100%", "height": "100%"}),
            ],
            style={
                "width": "100%",
                "height": "100%",
                "display": "inline-block",
                "padding-top": "10px",
                "padding-left": "1px",
                "overflow": "hidden",
            },
        )

        app.run_server(debug=True, port=3389, host=args.interface)
    elif args.output_mode == "output":
        postfix = poi
        meta = input_tools.get_metadata(args.inputFile)

        outdir = output_tools.make_plot_dir(args.outFolder, "", eoscp=args.eoscp)
        if group:
            outfile = os.path.splitext(args.outputFile)
            outfile = "".join([outfile[0] + "_group", outfile[1]])
        else:
            outfile = args.outputFile
        outfile = os.path.join(outdir, outfile)
        extensions = [outfile.split(".")[-1], *args.otherExtensions]

        include_ref = "impact_ref" in df.keys() or "constraint_ref" in df.keys()

        kwargs = dict(
            pulls=not args.noPulls and not group,
            impact_title=impact_title,
            normalize=not args.absolute,
            oneSidedImpacts=args.oneSidedImpacts,
            pullrange=pullrange,
            cmsDecor=args.cmsDecor,
            impacts=not args.noImpacts,
            asym_pulls=args.diffPullAsym,
            include_ref=include_ref,
            ref_name=args.refName,
            show_numbers=args.showNumbers,
            show_legend=not group and not args.noImpacts,
        )

        if args.num and args.num < int(df.shape[0]):
            # in case multiple extensions are given including html, don't do the skimming on html but all other formats
            if "html" in extensions and len(extensions) > 1:
                fig = plotImpacts(df, legend_pos="right", **kwargs)
                outfile_html = outfile.replace(outfile.split(".")[-1], "html")
                writeOutput(fig, outfile_html, [".html"], postfix=postfix)
                extensions = [e for e in extensions if e != "html"]
                outfile = outfile.replace(outfile.split(".")[-1], extensions[0])
            df = df[-args.num :]

        # utility output to prepare hepdata
        if args.saveForHepdata and not group:
            keysToSave = ["impact", "pull", "constraint"]
            if args.referenceFile:
                keysToSave.extend(["impact_ref", "pull_ref", "constraint_ref"])
            nRows = int(df.shape[0])
            outfile_root = outfile.replace(outfile.split(".")[-1], "root")
            outfile_root = outfile_root.replace(".root", f"_{postfix}.root")
            rf = input_tools.safeOpenRootFile(outfile_root, mode="recreate")
            hr2 = ROOT.TH2D(
                "nuisanceInfo",
                "",
                nRows,
                0.0,
                float(nRows),
                len(keysToSave),
                0.0,
                float(len(keysToSave)),
            )
            logger.warning(f"Preparing histogram {hr2.GetName()} for HEPData")
            for ix in range(nRows):
                latexLabel = df.at[df.index[ix], "label_hepdata"]
                if "mass" in latexLabel:
                    logger.warning(f"{ix+1} {latexLabel}")
                hr2.GetXaxis().SetBinLabel(ix + 1, latexLabel)
                logger.debug(f"{ix+1} {latexLabel}")
                for iy, y in enumerate(keysToSave):
                    hr2.GetYaxis().SetBinLabel(iy + 1, y)
                    hr2.SetBinContent(ix + 1, iy + 1, df.at[df.index[ix], y])
            logger.warning(
                f"Saving histogram {hr2.GetName()} for HEPData in {outfile_root}"
            )
            hr2.Write()
            rf.Close()

        fig = plotImpacts(df, **kwargs)

        writeOutput(
            fig, outfile, extensions, postfix=postfix, args=args, meta_info=meta
        )
        if args.eoscp and output_tools.is_eosuser_path(args.outFolder):
            output_tools.copy_to_eos(outdir, args.outFolder, "")
    else:
        raise ValueError("Must select mode 'interactive' or 'output'")


if __name__ == "__main__":
    args = parseArgs()

    logger = logging.setup_logger("pullsAndImpacts", 4 if args.debug else 3)

    grouping = groupings[args.grouping] if args.grouping else None

    translate_label = {}
    if args.translate:
        with open(args.translate) as f:
            translate_label = json.load(f)

    fitresult = combinetf_input.get_fitresult(args.inputFile)
    fitresult_ref = (
        combinetf_input.get_fitresult(args.referenceFile)
        if args.referenceFile
        else None
    )

    if args.noImpacts:
        # do one pulls plot, ungrouped
        producePlots(
            fitresult, args, None, fitresult_ref=fitresult_ref, pullrange=args.pullrange
        )
        exit()

    if args.poi:
        pois = [args.poi]
    else:
        pois = combinetf_input.get_poi_names(fitresult, poi_type=args.poiType)

    for poi in pois:
        logger.info(f"Now at {poi}")
        if args.mode in ["both", "ungrouped"]:
            logger.debug(f"Make impact per nuisance")
            producePlots(
                fitresult,
                args,
                poi,
                fitresult_ref=fitresult_ref,
                pullrange=args.pullrange,
            )
        if args.mode in ["both", "group"]:
            logger.debug(f"Make impact by group")
            producePlots(
                fitresult,
                args,
                poi,
                group=True,
                fitresult_ref=fitresult_ref,
                grouping=grouping,
                pullrange=args.pullrange,
            )
