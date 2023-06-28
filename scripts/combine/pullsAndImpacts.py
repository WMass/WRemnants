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

def writeOutput(fig, outfile, extensions=[], postfix=None, args=None):
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
        )


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
    impacts = bool(df['impact'].sum())
    ncols = pulls+impacts
    fig = make_subplots(rows=1,cols=ncols,
            horizontal_spacing=0.1, shared_yaxes=ncols > 1)

    max_pull = np.max(df["abspull"])
    # Round up to nearest 0.5
    pullrange = np.ceil(max_pull*1.1+1)
    
    ndisplay = len(df)
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis_title=impact_title if impacts else "Pull",
        margin=dict(l=20,r=20,t=50,b=20),
        yaxis=dict(range=[-1, ndisplay]),
        showlegend=False,
        height=100*(ndisplay<100)+ndisplay*20.5,width=800,
    )

    if impacts:
        fig.add_trace(
            go.Bar(
                x=df['impact' if not oneSidedImpacts else 'absimpact'],
                y=df['label'],
                marker_color=df['impact_color'] if oneSidedImpacts else '#377eb8',
                texttemplate="%{x:0.2f}",
                textposition="outside",
                textfont_size=12,
                textangle=0,
                orientation='h',
                name="impacts_down",
            ),
            row=1,col=1,
        )
        if not oneSidedImpacts:
            fig.add_trace(
                go.Bar(
                    x=-1*df['impact'],
                    y=df['label'],
                    marker_color='#e41a1c',
                    name="impacts_up",
                    orientation='h',
                ),
                row=1,col=1,
        )
        impact_range = np.ceil(df['impact'].max())
        impact_spacing = min(impact_range, 2 if pulls else 3)
        if impact_range % impact_spacing:
            impact_range += impact_spacing - (impact_range % impact_spacing)
        tick_spacing = impact_range/impact_spacing
        if pulls and oneSidedImpacts:
            tick_spacing /= 2.
        fig.update_layout(barmode='relative')
        fig.update_layout(
            xaxis=dict(range=[-impact_range if not oneSidedImpacts else -impact_range/20, impact_range],
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
        fig.update_layout(**info)

    return fig

def readFitInfoFromFile(rf,filename, group=False, sort=None, ascending=True, stat=0.0, POI='Wmass', normalize=False):    
    treename = "fitresults"
    tree = rf[treename]
    # TODO: Make add_total configurable
    add_total = group
    impacts, labels, _ = input_tools.readImpacts(rf, group, sort=sort, add_total=add_total, stat=stat, POI=POI, normalize=normalize)
    # TODO: Make configurable
    if True:
        impacts = impacts*100
    
    if args.filters:
        filtimpacts = []
        filtlabels = []
        for impact,label in zip(impacts,labels):
            if any(re.match(f, label) for f in args.filters):
                filtimpacts.append(impact)
                filtlabels.append(label)
        impacts = filtimpacts
        labels = filtlabels

    pulls = np.zeros_like(labels, dtype=float)
    constraints = np.zeros_like(labels, dtype=float)
    if not group:
        import ROOT
        rtfile = ROOT.TFile.Open(filename)
        rtree = rtfile.Get(treename)
        rtree.GetEntry(0)
        for i, label in enumerate(labels):
            if not hasattr(rtree, label):
                logger.warning(f"Failed to find syst {label} in tree")
                continue
                
            pulls[i] = getattr(rtree, label)
            constraints[i] = getattr(rtree, label+"_err")
    
    df = pd.DataFrame(np.array((pulls, impacts, constraints), dtype=np.float64).T, columns=["pull", "impact", "constraint"])
    df['label'] = labels
    df['absimpact'] = np.abs(df['impact'])
    df['abspull'] = np.abs(df['pull'])
    if not group:
        df.drop(df.loc[df['label']=='WmassShift100MeV'].index, inplace=True)
        df.drop(df.loc[df['label']=='ZmassShift100MeV'].index, inplace=True)
    colors = np.full(len(df), '#377eb8')
    if not group:
        colors[df['impact'] > 0.] = '#e41a1c'
    df['impact_color'] = colors

    if sort:
        df = df.sort_values(by=sort, ascending=ascending)

    return df

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--inputFile", type=str, required=True,
        help="fitresults output ROOT file from combinetf")
    parser.add_argument("-s", "--sort", default="absimpact", type=str, choices=["label", "abspull", "constraint", "absimpact"], help="Sort mode for nuisances")
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
def draw_figure(maxShow, sortBy, sortDescending, filterLabels, groups, oneSidedImpacts=False):
    df = groupsdataframe if groups else dataframe
    df = df[0 if not maxShow else -1*maxShow:].copy()
    df = df.sort_values(by=sortBy, ascending=sortDescending)
    if filterLabels:
        filt = np.full(len(df["label"]), False)
        for label in filterLabels.split(","):
            filt = filt | (df["label"].str.find(label.strip()) >= 0)
        df = df[filt]
    return plotImpacts(df, pulls=True, oneSidedImpacts=oneSidedImpacts)

dataframe = pd.DataFrame()
groupsdataframe = pd.DataFrame()

def producePlots(rtfile, args, POI='Wmass', normalize=False):

    group = args.mode == "group"
    if not (group and args.output_mode == 'output'):
        dataframe = readFitInfoFromFile(rtfile, args.inputFile, False, sort=args.sort, ascending=args.ascending, stat=args.stat/100., POI=POI, normalize=normalize)
    elif group:
        groupsdataframe = readFitInfoFromFile(rtfile, args.inputFile, True, sort=args.sort, ascending=args.ascending, stat=args.stat/100., POI=POI, normalize=normalize)

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
        df = dataframe if not group else groupsdataframe
        df = df.sort_values(by=args.sort, ascending=args.ascending)
        if args.num and args.num < df.size:
            df = df[-args.num:]
        fig = plotImpacts(df, pulls=not args.noPulls and not group, POI=POI, normalize=not args.absolute, oneSidedImpacts=args.oneSidedImpacts)

        postfix = POI if POI and "mass" not in POI else None
        
        outdir = output_tools.make_plot_dir(args.outFolder, "", eoscp=args.eoscp)
        if outdir and not os.path.isdir(outdir):
            os.path.makedirs(outdir)

        outfile = os.path.join(outdir, args.outputFile)
        writeOutput(fig, outfile, args.otherExtensions, postfix=postfix, args=args)
        if args.eoscp and output_tools.is_eosuser_path(args.outFolder):
            output_tools.copy_to_eos(args.outFolder, "")
    else:
        raise ValueError("Must select mode 'interactive' or 'output'")


if __name__ == '__main__':
    args = parseArgs()

    logger = logging.setup_logger("pullsAndImpacts", 4 if args.debug else 3)

    rtfile = uproot.open(args.inputFile)
    POIs = input_tools.getPOInames(rtfile)
    for POI in POIs:
        do_both = args.mode == "both"
        if args.mode == "both":
            args.mode = "ungrouped"
        producePlots(rtfile, args, POI)
        if do_both:
            args.mode = "group"
            outfile = os.path.splitext(args.outputFile)
            args.outputFile = "".join([outfile[0]+"_group", outfile[1]])
            producePlots(rtfile, args, POI)
    
    if POIs[0] and not (POIs[0]=="Wmass" and len(POIs) == 1):
        # masked channel
        for POI in input_tools.getPOInames(rtfile, poi_type="pmaskedexp"):
            producePlots(rtfile, args, POI, normalize=not args.absolute)

        # masked channel normalized
        for POI in input_tools.getPOInames(rtfile, poi_type="pmaskedexpnorm"):
            producePlots(rtfile, args, POI, normalize=not args.absolute)
