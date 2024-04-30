import uproot
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import argparse
import dash
from dash import dcc
import dash_daq as daq
from dash import html
from dash.dependencies import Input, Output
from utilities import logging
from utilities.io_tools import input_tools, output_tools, combinetf_input
from wremnants import plot_tools
import os
import re
import json

from utilities.styles.styles import nuisance_groupings as groupings

def writeOutput(fig, outfile, extensions=[], postfix=None, args=None, meta_info=None):
    name, ext = os.path.splitext(outfile)
    if ext not in extensions:
        extensions.append(ext)

    if postfix:
        name += f"_{postfix}"

    for ext in extensions:
        if ext[0] != ".":
            ext = "."+ext
        output = name+ext
        logger.debug(f"Write output file {output}")
        if ext == ".html":
            fig.write_html(output, include_mathjax='cdn')
        else:
            fig.write_image(output)
        
        output = name.rsplit("/", 1)
        output[1] = os.path.splitext(output[1])[0]
        if len(output) == 1:
            output = (None, *output)
    if args is None and meta_info is None:
        return
    plot_tools.write_index_and_log(*output, 
        args=args,
        analysis_meta_info={"AnalysisOutput" : meta_info},
    )

def get_marker(filled=True, color='#377eb8', opacity=1.0):
    if filled:
        marker={"marker": {
                "color":color,  # Fill color for the filled bars
                "opacity":opacity  # Opacity for the filled bars (adjust as needed)        
            }
        }
    else:
        marker= {"marker": {
                "color":'rgba(0, 0, 0, 0)',  # Transparent fill color
                "opacity":opacity,
                "line":{
                    "color":color,  # Border color
                    "width":2  # Border width
                }
            }
        }
    return marker

def plotImpacts(df, impact_title="", pulls=False, normalize=False, oneSidedImpacts=False):
    impacts = bool(np.count_nonzero(df['absimpact'])) and not args.noImpacts
    ncols = pulls+impacts
    fig = make_subplots(rows=1,cols=ncols,
            horizontal_spacing=0.1, shared_yaxes=True)#ncols > 1)
   
    ndisplay = len(df)
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis_title=impact_title if impacts else "Pull",
        margin=dict(l=20,r=20,t=50,b=20),
        yaxis=dict(range=[-1, ndisplay]),
        showlegend=False,
        height=100*(ndisplay<100)+ndisplay*20.5,width=800,
    )

    include_ref = "impact_ref" in df.keys() or "constraint_ref" in df.keys()
    impact_str = 'impact' if not oneSidedImpacts else 'absimpact'

    if impacts and include_ref:
        # append numerical values of impacts on nuisance name; fill up empty room with spaces to align numbers
        frmt = "{:0"+str(int(np.log10(max(df[impact_str])))+2)+".2f}"
        nval = df[impact_str].apply(lambda x,frmt=frmt: frmt.format(x)) #.astype(str)
        nspace = nval.apply(lambda x, n=nval.apply(len).max(): " "*(n - len(x))) 
        if include_ref:
            frmt_ref = "{:0"+str(int(np.log10(max(df[f"{impact_str}_ref"])))+2)+".2f}"
            nval_ref = df[f'{impact_str}_ref'].apply(lambda x,frmt=frmt_ref: " ("+frmt.format(x)+")") #.round(2).astype(str)
            nspace_ref = nval_ref.apply(lambda x, n=nval_ref.apply(len).max(): " "*(n - len(x))) 
            nval = nval+nspace_ref+nval_ref 
        labels = df["label"].apply(lambda x: x[:-1] if x.endswith("$") else r"$\text{"+x+"}")
        labels = labels+r"\ \ \text{"+nspace+nval+"}$"
        textargs = dict()
    else:
        labels = df["label"]
        textargs = dict(
            texttemplate="%{x:0.2f}",
            textposition="outside",
            textfont_size=12,
            textangle=0,
        )

    if impacts:
        fig.add_trace(
            go.Bar(
                x=df[impact_str],
                y=labels,
                width=0.2 if include_ref else None,
                orientation='h',
                **get_marker(filled=True, color=df['impact_color'] if oneSidedImpacts else '#377eb8'),
                **textargs,
                name="impacts_down",
            ),
            row=1,col=1,

        )
        if include_ref:
            fig.add_trace(
                go.Bar(
                    x=df[f'{impact_str}_ref'],
                    y=labels,
                    orientation='h',
                    **get_marker(filled=True, color=df['impact_color'] if oneSidedImpacts else '#377eb8', opacity=0.5),
                ),
                row=1,col=1,
            )
        if not oneSidedImpacts:
            fig.add_trace(
                go.Bar(
                    x=-1*df['impact'],
                    y=labels,
                    width=0.2 if include_ref else None,
                    orientation='h',
                    **get_marker(filled=True, color='#e41a1c'),
                    name="impacts_up",
                ),
                row=1,col=1,
            )
            if include_ref:
                fig.add_trace(
                    go.Bar(
                        x=-1*df['impact_ref'],
                        y=labels,
                        orientation='h',
                        **get_marker(filled=True, color='#e41a1c', opacity=0.5),
                    ),
                    row=1,col=1,
                )
        impact_range = np.ceil(df[impact_str].max())
        if include_ref:
            impact_range = max(impact_range,np.ceil(df[f'{impact_str}_ref'].max()))
        impact_spacing = min(impact_range, 2 if pulls else 3)
        if impact_range % impact_spacing:
            impact_range += impact_spacing - (impact_range % impact_spacing)
        tick_spacing = impact_range/impact_spacing
        if pulls and oneSidedImpacts:
            tick_spacing /= 2.
        fig.update_layout(barmode='overlay')
        fig.update_layout(
            xaxis=dict(range=[-impact_range*1.1 if not oneSidedImpacts else -impact_range/20, impact_range*1.1],
                    showgrid=True, gridwidth=1, gridcolor='Gray', griddash='dash',
                    zeroline=True, zerolinewidth=2, zerolinecolor='Gray',
                    tickmode='linear',
                    tickangle=0,
                    tick0=0.,
                    side='top',
                    dtick=tick_spacing,
                ),
        )

    if pulls:
        fig.add_trace(
            go.Scatter(
                x=df['pull'],
                y=labels,
                mode="markers",
                marker=dict(color='black', size=8,),
                error_x=dict(
                    array=df['constraint'],
                    color="black",
                    thickness=1.5,
                    width=5,
                ),
                name="pulls",
            ),
            row=1,col=ncols,
        )
        if args.diffPullAsym:
            fig.add_trace(
                go.Scatter(
                    x=df['newpull'],
                    y=labels,
                    mode="markers",
                    marker=dict(color='blue', symbol="x", size=8, 
                        line=dict(
                            width=1  # Adjust the thickness of the marker lines
                    )),
                    name="newpulls",
                ),
                row=1,col=ncols,
            )

        if include_ref:
            fig.add_trace(
                go.Bar(
                    x=df['constraint_ref'],
                    y=labels,
                    orientation='h',
                    **get_marker(filled=True, color='grey', opacity=0.5),
                    name="constraint_ref",
                ),
                row=1,col=ncols,
            )
            fig.add_trace(
                go.Bar(
                    x=-1*df['constraint_ref'],
                    y=labels,
                    orientation='h',
                    **get_marker(filled=True, color='grey', opacity=0.5),
                    name="constraint_ref",
                ),
                row=1,col=ncols,
            )
        max_pull = np.max(df["abspull"])
        # Round up to nearest 0.25, add 1.1 for display
        pullrange = .5*np.ceil(max_pull/0.5)+1.1
        # Keep it a factor of 0.25, but no bigger than 1
        spacing = min(1, np.ceil(pullrange)/4.)
        xaxis_title = r'$\Theta - \Theta_0 \ \color{blue}{(\Theta - \Theta_0) / \sqrt{\sigma^2 -\sigma_0^2}}$' if args.diffPullAsym else r'$\Theta - \Theta_0$'
        info = dict(
            xaxis=dict(range=[-pullrange, pullrange],
                    showgrid=True, gridwidth=2, gridcolor='LightBlue',
                    zeroline=True, zerolinewidth=4, zerolinecolor='Gray',
                    tickmode='linear',
                    tick0=0.,
                    dtick=spacing,
                    side='top',
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
        fig.update_layout(barmode='overlay', **info)

    return fig

def readFitInfoFromFile(rf, filename, poi, group=False, stat=0.0, normalize=False, scale=1):    
    impacts, labels, _ = combinetf_input.read_impacts_poi(rf, group, add_total=group, stat=stat, poi=poi, normalize=normalize)
    
    if (group and grouping) or args.filters:
        filtimpacts = []
        filtlabels = []
        for impact,label in zip(impacts,labels):
            if group and grouping and label not in grouping:
                continue
            if args.filters and not any(re.match(f, label) for f in args.filters):
                continue
            filtimpacts.append(impact)
            filtlabels.append(label)
        impacts = filtimpacts
        labels = filtlabels

    df = pd.DataFrame(np.array(impacts, dtype=np.float64).T*scale, columns=["impact"])
    df['label'] = [translate_label.get(l, l) for l in labels]
    df['absimpact'] = np.abs(df['impact'])
    if not group:
        df["pull"], df["constraint"], df["pull_prefit"] = combinetf_input.get_pulls_and_constraints(filename, labels)
        df["pull"] = df['pull'] - df["pull_prefit"]
        df['abspull'] = np.abs(df['pull'])
        df['newpull'] = df['pull'] / (1-df["constraint"]**2)**0.5
        df['newpull'].replace([np.inf, -np.inf, np.nan], 999, inplace=True)
        if poi:
            df.drop(df.loc[df['label'].str.contains(poi.replace("_noi",""), regex=True)].index, inplace=True)
    colors = np.full(len(df), '#377eb8')
    if not group:
        colors[df['impact'] > 0.] = '#e41a1c'
    df['impact_color'] = colors

    return df

def parseArgs():
    sort_choices = ["label", "abspull", "constraint", "absimpact"]
    sort_choices += [
        *[f"{c}_diff" for c in sort_choices],  # possibility to sort based on largest difference between inputfile and referencefile
        *[f"{c}_ref" for c in sort_choices] ]  # possibility to sort based on referencefile

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--inputFile", type=str, required=True, help="fitresults output ROOT/hdf5 file from combinetf")
    parser.add_argument("-r", "--referenceFile", type=str, help="fitresults output ROOT/hdf5 file from combinetf for reference")
    parser.add_argument("-s", "--sort", default="absimpact", type=str, help="Sort mode for nuisances", choices=sort_choices)
    parser.add_argument("--stat", default=0.0, type=float, help="Overwrite stat. uncertainty with this value")
    parser.add_argument("-d", "--sortDescending", dest='ascending', action='store_false', help="Sort mode for nuisances")
    parser.add_argument("-m", "--mode", choices=["group", "ungrouped", "both"], default="both", help="Impact mode")
    parser.add_argument("--absolute", action='store_true', help="Not normalize impacts on cross sections and event numbers.")
    parser.add_argument("--debug", action='store_true', help="Print debug output")
    parser.add_argument("--diffPullAsym", action='store_true', help="Also add the pulls after the diffPullAsym definition")
    parser.add_argument("--oneSidedImpacts", action='store_true', help="Make impacts one-sided")
    parser.add_argument("--filters", nargs="*", type=str, help="Filter regexes to select nuisances by name")
    parser.add_argument("--grouping", type=str, default=None, help="Select nuisances by a predefined grouping", choices=groupings.keys())
    parser.add_argument("-t","--translate", type=str, default=None, help="Specify .json file to translate labels")
    parser.add_argument("--noImpacts", action='store_true', help="Don't show impacts")
    parsers = parser.add_subparsers(dest='output_mode')
    interactive = parsers.add_parser("interactive", help="Launch and interactive dash session")
    interactive.add_argument("-i", "--interface", default="localhost", help="The network interface to bind to.")
    output = parsers.add_parser("output", help="Produce plots as output (not interactive)")
    output.add_argument("-o", "--outputFile", default="test.html", type=str, help="Output file (extension specifies if html or pdf/png)")
    output.add_argument("--outFolder", type=str, default="", help="Output folder (created if it doesn't exist)")
    output.add_argument("--otherExtensions", default=[], type=str, nargs="*", help="Additional output file types to write")
    output.add_argument("-n", "--num", type=int, help="Number of nuisances to plot")
    output.add_argument("--noPulls", action='store_true', help="Don't show pulls (not defined for groups)")
    output.add_argument("--eoscp", action='store_true', help="Use of xrdcp for eos output rather than the mount")
    
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

def producePlots(fitresult, args, poi, group=False, normalize=False, fitresult_ref=None):
    poi_type = poi.split("_")[-1] if poi else None

    if poi is not None and "MeV" in poi:
        scale = float(re.search(r'\d+(\.\d+)?', poi.split("MeV")[0].replace("p",".")).group())
        if "Diff" in poi:
            scale *= 2 # take diffs by 2 as up and down pull in opposite directions
    else:
        scale = 1

    if poi and poi.startswith("massShift"):
        impact_title = "Impact on mass (MeV)"
    elif poi and poi.startswith("massDiff"):
        if poi.startswith("massDiffCharge"):
            impact_title = "Impact on mass diff. (charge) (MeV)"
        elif poi.startswith("massDiffEta"):
            impact_title = "$\\mathrm{Impact\\ on\\ mass\\ diff. }(\\eta)\\ (\\mathrm{MeV})$"
        else:
            impact_title = "Impact on mass diff. (MeV)"
    elif poi and poi.startswith("Width"):
        impact_title = "Impact on width (MeV)"
    else:
        impact_title=poi

    if not (group and args.output_mode == 'output'):
        df = readFitInfoFromFile(fitresult, args.inputFile, poi, False, stat=args.stat/100., normalize=normalize, scale=scale)
    elif group:
        df = readFitInfoFromFile(fitresult, args.inputFile, poi, True, stat=args.stat/100., normalize=normalize, scale=scale)

    if fitresult_ref:
        df_ref = readFitInfoFromFile(fitresult_ref, args.referenceFile, poi, group, stat=args.stat/100., normalize=normalize, scale=scale)
        df = df.merge(df_ref, how="left", on="label", suffixes=("","_ref"))
    
    if group and fitresult_ref and set(fitresult_ref["hsysts"]) != set(fitresult["hsysts"]):
        # add another group for the uncertainty from all systematics that are not common among the two groups; 
        # defined as err = sqrt( variance(total) - variance(common nuisances) )
        np_common = [s in fitresult_ref["hsysts"][...] for s in fitresult["hsysts"][...]]
        np_common_ref = [s in fitresult["hsysts"][...] for s in fitresult_ref["hsysts"][...]]
        cov = fitresult["cov"][np_common, :][:, np_common]
        cov_ref = fitresult_ref["cov"][np_common_ref, :][:, np_common_ref]

        jac = fitresult["nuisance_impact_nois"][:,np_common]
        jac_ref = fitresult_ref["nuisance_impact_nois"][:,np_common_ref]

        df_total = df.loc[df["label"] == "Total"]
        impact_others = np.sqrt(df_total["impact"].values[0]**2 - scale**2 * (jac @ (cov @ jac.T))[0][0])
        impact_others_ref = np.sqrt(df_total["impact_ref"].values[0]**2 - scale**2 * (jac_ref @ (cov_ref @ jac_ref.T))[0][0])

        new_row = {"label": "Others",
            "impact": impact_others, "absimpact": abs(impact_others), "impact_color": df_total["impact_color"].values[0],
            "impact_ref": impact_others_ref, "absimpact_ref": abs(impact_others_ref), "impact_color_ref": df_total["impact_color"].values[0],
        }

        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        # remove rows with nuisance groups that are only included in one group but not in the other, since those are included in "Others" already
        df = df.dropna()

    if args.sort:
        if args.sort.endswith("diff"):
            key = args.sort.replace("_diff","")
            df[f"{key}_diff"] = df[key] - df[f"{key}_ref"]

        df = df.sort_values(by=args.sort, ascending=args.ascending)

    df = df.fillna(0)

    if args.output_mode == "interactive":
        app.layout = html.Div([
                dcc.Input(
                    id="maxShow", type="number", placeholder="maxShow",
                    min=0, max=10000, step=1,
                ),
                dcc.Input(
                    id="filterLabels", type="text", placeholder="filter labels (comma-separated list)",
                    style={
                        'width': '25%'
                    },
                ),
                html.Br(),
                html.Label('Sort by'),
                dcc.Dropdown(
                    id='sortBy',
                    options=[{'label' : v, "value" : v.lower()} for v in ["Impact", "Pull", "Constraint", "Label"]
                    ],
                    placeholder='select sort criteria...',
                    style={
                        'width': '50%'
                    },
                    value=args.sort
                ),
                daq.BooleanSwitch('sortDescending', label="Decreasing order", labelPosition="top", on=True),
                daq.BooleanSwitch('groups', label="Show nuisance groups", labelPosition="top", on=False),
                dcc.Graph(id="scatter-plot", style={'width': '100%', 'height': '100%'}),
            ],
            style={
                "width": "100%",
                "height": "100%",
                "display": "inline-block",
                "padding-top": "10px",
                "padding-left": "1px",
                "overflow": "hidden"
            },
        )

        app.run_server(debug=True, port=3389, host=args.interface)
    elif args.output_mode == 'output':
        postfix = poi
        meta = input_tools.get_metadata(args.inputFile)

        outdir = output_tools.make_plot_dir(args.outFolder, "", eoscp=args.eoscp)
        if group:
            outfile = os.path.splitext(args.outputFile)
            outfile = "".join([outfile[0]+"_group", outfile[1]])
        else:
            outfile = args.outputFile
        outfile = os.path.join(outdir, outfile)
        extensions = [outfile.split(".")[-1], *args.otherExtensions]

        df = df.sort_values(by=args.sort, ascending=args.ascending)
        if args.num and args.num < df.size:
            # in case multiple extensions are given including html, don't do the skimming on html but all other formats
            if "html" in extensions and len(extensions)>1:
                fig = plotImpacts(df, pulls=not args.noPulls and not group, impact_title=impact_title, normalize=not args.absolute, oneSidedImpacts=args.oneSidedImpacts)
                outfile_html = outfile.replace(outfile.split(".")[-1], "html")
                writeOutput(fig, outfile_html, postfix=postfix)
                extensions = [e for e in extensions if e != "html"]
                outfile = outfile.replace(outfile.split(".")[-1], extensions[0])
            df = df[-args.num:]

        fig = plotImpacts(df, pulls=not args.noPulls and not group, impact_title=impact_title, normalize=not args.absolute, oneSidedImpacts=args.oneSidedImpacts)
        writeOutput(fig, outfile, extensions[0:], postfix=postfix, args=args, meta_info=meta)      
        if args.eoscp and output_tools.is_eosuser_path(args.outFolder):
            output_tools.copy_to_eos(args.outFolder, "")
    else:
        raise ValueError("Must select mode 'interactive' or 'output'")


if __name__ == '__main__':
    args = parseArgs()

    logger = logging.setup_logger("pullsAndImpacts", 4 if args.debug else 3)

    grouping = groupings[args.grouping] if args.grouping else None

    translate_label = {}
    if args.translate:
        with open(args.translate) as f:
            translate_label = json.load(f)

    fitresult = combinetf_input.get_fitresult(args.inputFile)
    fitresult_ref = combinetf_input.get_fitresult(args.referenceFile) if args.referenceFile else None

    if args.noImpacts:
        # do one pulls plot, ungrouped
        producePlots(fitresult, args, None, fitresult_ref=fitresult_ref)
        exit()

    pois = combinetf_input.get_poi_names(fitresult, poi_type=None)
    for poi in pois:
        if args.mode in ["both", "ungrouped"]:
            producePlots(fitresult, args, poi, fitresult_ref=fitresult_ref)
        if args.mode in ["both", "group"]:
            producePlots(fitresult, args, poi, group=True, fitresult_ref=fitresult_ref)
