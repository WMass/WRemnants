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
from utilities import input_tools, output_tools, logging
from wremnants import plot_tools
import os
import re

translate_label = {
    # "stat": "Unfolding",
    "resumNonpert": "Non perturbative QCD",
    "pdfMSHT20": "pdf",
    "pdfMSHT20AlphaS": r"$\mathrm{pdf}\ \alpha_\mathrm{S}$",
    "QCDscaleWPtHelicityMiNNLO": "Perturbative QCD",
    "theory_ew": "EW",
}

def writeOutput(fig, outfile, extensions=[], postfix=None, args=None, meta_info=None):
    name, ext = os.path.splitext(outfile)
    if ext not in extensions:
        extensions.append(ext)

    if postfix:
        name += f"_{postfix}"

    for ext in extensions:
        if ext[0] != ".":
            ext = "."+ext
        func = "write_html" if ext == ".html" else "write_image"
        output = name+ext
        logger.debug(f"Write output file {output}")

        getattr(fig, func)(output)
        
        output = outfile.rsplit("/", 1)
        output[1] = os.path.splitext(output[1])[0]
        if len(output) == 1:
            output = (None, *output)
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

def plotImpacts(df, pulls=False, POI='Wmass', normalize=False, oneSidedImpacts=False):
    poi_type = POI.split("_")[-1] if POI else None

    if poi_type == "Wmass":
        impact_title="Impact on mass (MeV)"
    elif poi_type == "mu":
        impact_title=r"$\Delta \mu$"
    elif poi_type == "pmaskedexp":
        impact_title=r"$\delta N$" if normalize else r"$\Delta N$"
    elif poi_type == "pmaskedexpnorm":
        impact_title=r"$\delta (\mathrm{d} \sigma / \sigma)$" if normalize else r"$\Delta(\mathrm{d} \sigma / \sigma)$"

    pulls = pulls 
    impacts = bool(df['impact'].sum()) and not args.noImpacts
    ncols = pulls+impacts
    fig = make_subplots(rows=1,cols=ncols,
            horizontal_spacing=0.1, shared_yaxes=True)#ncols > 1)

    max_pull = np.max(df["abspull"])
    # Round up to nearest 0.25, add 1.1 for display
    pullrange = .5*np.ceil(max_pull/0.5)+1.1
    
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

    if impacts:
        fig.add_trace(
            go.Bar(
                x=df['impact' if not oneSidedImpacts else 'absimpact'],
                y=df['label'],
                orientation='h',
                **get_marker(filled=True, color=df['impact_color'] if oneSidedImpacts else '#377eb8', opacity=0.5 if include_ref else 1.0),
                texttemplate="%{x:0.2f}",
                textposition="outside",
                textfont_size=12,
                textangle=0,
                name="impacts_down",
            ),
            row=1,col=1,
        )
        if include_ref:
            fig.add_trace(
                go.Bar(
                    x=df['impact_ref' if not oneSidedImpacts else 'absimpact_ref'],
                    y=df['label'],
                    orientation='h',
                    **get_marker(filled=False, color=df['impact_color'] if oneSidedImpacts else '#377eb8'),
                ),
                row=1,col=1,
            )
        if not oneSidedImpacts:
            fig.add_trace(
                go.Bar(
                    x=-1*df['impact'],
                    y=df['label'],
                    orientation='h',
                    **get_marker(filled=True, color='#e41a1c', opacity=0.5 if include_ref else 1.0),
                    name="impacts_up",
                ),
                row=1,col=1,
            )
            if include_ref:
                fig.add_trace(
                    go.Bar(
                        x=-1*df['impact_ref'],
                        y=df['label'],
                        orientation='h',
                        **get_marker(filled=False, color='#e41a1c'),
                    ),
                    row=1,col=1,
                )
        impact_range = np.ceil(df['impact' if not oneSidedImpacts else 'absimpact'].max())
        if include_ref:
            impact_range = max(impact_range,np.ceil(df['impact_ref' if not oneSidedImpacts else 'absimpact_ref'].max()))
        impact_spacing = min(impact_range, 2 if pulls else 3)
        if impact_range % impact_spacing:
            impact_range += impact_spacing - (impact_range % impact_spacing)
        tick_spacing = impact_range/impact_spacing
        if pulls and oneSidedImpacts:
            tick_spacing /= 2.
        fig.update_layout(barmode='overlay')
        fig.update_layout(
            xaxis=dict(range=[-impact_range if not oneSidedImpacts else -impact_range/20, impact_range*1.1],
                    showgrid=True, gridwidth=1,gridcolor='Gray', griddash='dash',
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
                y=df['label'],
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
        if include_ref:
            fig.add_trace(
                go.Bar(
                    x=df['constraint_ref'],
                    y=df['label'],
                    orientation='h',
                    **get_marker(filled=False, color='black'),
                    name="constraint_ref",
                ),
                row=1,col=ncols,
            )
            fig.add_trace(
                go.Bar(
                    x=-1*df['constraint_ref'],
                    y=df['label'],
                    orientation='h',
                    **get_marker(filled=False, color='black'),
                    name="constraint_ref",
                ),
                row=1,col=ncols,
            )
        # Keep it a factor of 0.25, but no bigger than 1
        spacing = min(1, np.ceil(pullrange)/4.)
        info = dict(
            xaxis=dict(range=[-pullrange, pullrange],
                    showgrid=True, gridwidth=2,gridcolor='LightBlue',
                    zeroline=True, zerolinewidth=4, zerolinecolor='Gray',
                    tickmode='linear',
                    tick0=0.,
                    dtick=spacing,
                    side='top',
                ),
            xaxis_title="pull+constraint",
            yaxis=dict(range=[-1, ndisplay]),
            yaxis_visible=not impacts,
        )
        if impacts:
            new_info = {}
            for k in info.keys():
                new_info[k.replace("axis", "axis2")] = info[k]
            info = new_info
        fig.update_layout(barmode='overlay',**info)

    return fig

def readFitInfoFromFile(rf, filename, group=False, stat=0.0, POI='Wmass', normalize=False):    
    # TODO: Make add_total configurable
    add_total = group
    impacts, labels, _ = input_tools.readImpacts(rf, group, add_total=add_total, stat=stat, POI=POI, normalize=normalize)
    # TODO: Make configurable
    if True:
        impacts = impacts*100

    # skip POIs in case of unfolding, want only nuisances
    pois = input_tools.getPOInames(rf) if POI is None else []

    if args.filters or len(pois) > 0:
        filtimpacts = []
        filtlabels = []
        for impact,label in zip(impacts,labels):
            if label in pois:
                continue
            if args.filters and not any(re.match(f, label) for f in args.filters):
                continue
            filtimpacts.append(impact)
            filtlabels.append(label)
        impacts = filtimpacts
        labels = filtlabels

    pulls = np.zeros_like(labels, dtype=float)
    constraints = np.zeros_like(labels, dtype=float)
    if not group:
        import ROOT
        fitresult = ROOT.TFile.Open(filename.replace(".hdf5",".root"))
        rtree = fitresult.Get("fitresults")
        rtree.GetEntry(0)
        for i, label in enumerate(labels):
            if not hasattr(rtree, label):
                logger.warning(f"Failed to find syst {label} in tree")
                continue
                
            pulls[i] = getattr(rtree, label)
            constraints[i] = getattr(rtree, label+"_err")
    
    labels = [translate_label.get(l, l) for l in labels]

    df = pd.DataFrame(np.array((pulls, impacts, constraints), dtype=np.float64).T, columns=["pull", "impact", "constraint"])
    df['label'] = labels
    df['absimpact'] = np.abs(df['impact'])
    df['abspull'] = np.abs(df['pull'])
    if not group:
        df.drop(df.loc[df['label'].str.contains('massShift.*100MeV', regex=True)].index, inplace=True)
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
    parser.add_argument("--oneSidedImpacts", action='store_true', help="Make impacts one-sided")
    parser.add_argument("--filters", nargs="*", type=str, help="Filter regexes to select nuisances by name")
    parsers = parser.add_subparsers(dest='output_mode')
    interactive = parsers.add_parser("interactive", help="Launch and interactive dash session")
    interactive.add_argument("-i", "--interface", default="localhost", help="The network interface to bind to.")
    output = parsers.add_parser("output", help="Produce plots as output (not interactive)")
    output.add_argument("-o", "--outputFile", default="test.html", type=str, help="Output file (extension specifies if html or pdf/png)")
    output.add_argument("--outFolder", type=str, default="", help="Output folder (created if it doesn't exist)")
    output.add_argument("--otherExtensions", default=[], type=str, nargs="*", help="Additional output file types to write")
    output.add_argument("-n", "--num", type=int, help="Number of nuisances to plot")
    output.add_argument("--noPulls", action='store_true', help="Don't show pulls (not defined for groups)")
    output.add_argument("--noImpacts", action='store_true', help="Don't show impacts")
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

def producePlots(fitresult, args, POI='Wmass', normalize=False, fitresult_ref=None):

    group = args.mode == "group"
    if not (group and args.output_mode == 'output'):
        df = readFitInfoFromFile(fitresult, args.inputFile, False, stat=args.stat/100., POI=POI, normalize=normalize)
    elif group:
        df = readFitInfoFromFile(fitresult, args.inputFile, True, stat=args.stat/100., POI=POI, normalize=normalize)

    if fitresult_ref:
        df_ref = readFitInfoFromFile(fitresult_ref, args.referenceFile, group, stat=args.stat/100., POI=POI, normalize=normalize)
        df = df.merge(df_ref, how="left", on="label", suffixes=("","_ref"))
        
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
        postfix = POI if POI and "mass" not in POI else None
        meta = input_tools.get_metadata(args.inputFile)
        outdir = output_tools.make_plot_dir(args.outFolder, "", eoscp=args.eoscp)
        if outdir and not os.path.isdir(outdir):
            os.path.makedirs(outdir)
        outfile = os.path.join(outdir, args.outputFile)
        extensions = [outfile.split(".")[-1], *args.otherExtensions]

        df = df.sort_values(by=args.sort, ascending=args.ascending)
        if args.num and args.num < df.size:
            # in case multiple extensions are given including html, don't do the skimming on html but all other formats
            if "html" in extensions and len(extensions)>1:
                fig = plotImpacts(df, pulls=not args.noPulls and not group, POI=POI, normalize=not args.absolute, oneSidedImpacts=args.oneSidedImpacts)
                outfile_html = outfile.replace(outfile.split(".")[-1], "html")
                writeOutput(fig, outfile_html, postfix=postfix)
                extensions = [e for e in extensions if e != "html"]
                outfile = outfile.replace(outfile.split(".")[-1], extensions[0])
            df = df[-args.num:]

        fig = plotImpacts(df, pulls=not args.noPulls and not group, POI=POI, normalize=not args.absolute, oneSidedImpacts=args.oneSidedImpacts)
        writeOutput(fig, outfile, extensions[0:], postfix=postfix, args=args, meta_info=meta)      
        if args.eoscp and output_tools.is_eosuser_path(args.outFolder):
            output_tools.copy_to_eos(args.outFolder, "")
    else:
        raise ValueError("Must select mode 'interactive' or 'output'")


if __name__ == '__main__':
    args = parseArgs()

    logger = logging.setup_logger("pullsAndImpacts", 4 if args.debug else 3)

    fitresult = input_tools.getFitresult(args.inputFile)
    fitresult_ref = input_tools.getFitresult(args.referenceFile) if args.referenceFile else None

    if args.noImpacts:
        # do one pulls plot, ungrouped
        args.mode = "ungrouped"
        producePlots(fitresult, args, None, fitresult_ref=fitresult_ref)
        exit()

    POIs = input_tools.getPOInames(fitresult, poi_type=None)

    for POI in POIs:
        do_both = args.mode == "both"
        if args.mode == "both":
            args.mode = "ungrouped"
        producePlots(fitresult, args, POI, fitresult_ref=fitresult_ref)
        if do_both:
            args.mode = "group"
            outfile = os.path.splitext(args.outputFile)
            args.outputFile = "".join([outfile[0]+"_group", outfile[1]])
            producePlots(fitresult, args, POI, fitresult_ref=fitresult_ref)
    
    if POIs[0] and not (POIs[0]=="Wmass" and len(POIs) == 1):
        # masked channel
        for POI in input_tools.getPOInames(fitresult, poi_type="pmaskedexp"):
            producePlots(fitresult, args, POI, normalize=not args.absolute, fitresult_ref=fitresult_ref)

        # masked channel normalized
        for POI in input_tools.getPOInames(fitresult, poi_type="pmaskedexpnorm"):
            producePlots(fitresult, args, POI, normalize=not args.absolute, fitresult_ref=fitresult_ref)
