import pandas as pd
from scipy.stats import chi2

import rabbit.io_tools
from utilities import parsing
from wums import logging, output_tools, tex_tools

translate = {
    "asimov": "Asimov",
    "data": "Data",
    "uncorr": "MiNNLO",
    "dyturbo": "DYTurbo",
    "dyturboN3LLp": "DYTurbo N$^{3}$LL' (1D)",
    "dyturboN3LLp2d": "DYTurbo N$^{3}$LL' (2D)",
    "matrix_radish": "MATRIX$+$RadISH",
    "scetlibNP": "SCETLib NP",
    "scetlibN4LL": "SCETLib N$^{4}$LL",
}


def read_fitresult(filename):
    try:
        fitresult = rabbit.io_tools.get_fitresult(filename)
        ndf = fitresult[f"ndf"]
        chi2_prefit = fitresult[f"chi2_prefit"]
        chi2_postfit = fitresult[f"chi2_postfit"]
        p_prefit = (1 - chi2.cdf(chi2_prefit, ndf)) * 100
        p_postft = (1 - chi2.cdf(chi2_postfit, ndf)) * 100

    except IOError as e:
        return 0, 0, 0, 0, 0

    return chi2_postfit, chi2_postfit, p_prefit, p_postft, ndf


parser = parsing.plot_parser()
parser.add_argument(
    "inputs", nargs="+", type=str, help="Paths to fitresult .hdf5 files"
)
args = parser.parse_args()

logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)
outdir = output_tools.make_plot_dir(args.outpath, args.outfolder)

df = pd.DataFrame(args.inputs, columns=["path"])

df[["chi2_prefit", "chi2_postfit", "p_prefit", "p_postfit", "ndf"]] = (
    df["path"].apply(read_fitresult).apply(pd.Series)
)

df["name_parts"] = df["path"].apply(
    lambda x: [y for y in filter(lambda z: z, x.split("/"))]
)

df["channel"] = df["name_parts"].apply(lambda x: x[-2].split("_")[0])

df["name_parts"] = df["name_parts"].apply(
    lambda x: [y.replace(".hdf5", "") for y in x[-1].split("_")[2:]]
)

df["axes"] = df["name_parts"].apply(
    lambda x: [y for y in x if y in ["ptll", "yll", "pt", "eta", "charge"]]
)
df["dataset"] = df["name_parts"].apply(
    lambda x: "_".join(
        [y for y in x if y not in ["ptll", "yll", "pt", "eta", "charge"]]
    )
)
df["dataset"] = df["dataset"].apply(
    lambda x: translate.get(x.replace("Corr", ""), x.replace("Corr", ""))
)

df["column_name"] = df["axes"].apply(lambda x: "-".join(x))
df["column_name_ndf"] = df["column_name"] + df["ndf"].apply(lambda x: f" ({x})")


for channel, df_c in df.groupby("channel"):
    tex_tools.make_latex_table(
        df_c,
        output_dir=outdir,
        output_name=f"table_rabbit_prefit_{channel}",
        column_title="Axes (bins)",
        caption=r"Resulting prefit $\chi^2$ values (and p-values) from WLike fits on different data, and pseudodata sets.",
        label="Pseudodata",
        sublabel="",
        column_name="column_name_ndf",
        row_name="dataset",
        cell_columns=["chi2_prefit", "p_prefit"],
        color_condition=lambda x, y: y < 5,
        cell_format=lambda x, y: rf"${round(x,1)} ({round(y,1)}\%)$",
    )

    tex_tools.make_latex_table(
        df_c,
        output_dir=outdir,
        output_name=f"table_rabbit_postfit_{channel}",
        column_title="Axes (bins)",
        caption=r"Resulting postfit $\chi^2$ values (and p-values) from WLike fits on different data, and pseudodata sets.",
        label="Pseudodata",
        sublabel="",
        column_name="column_name_ndf",
        row_name="dataset",
        cell_columns=["chi2_postfit", "p_postfit"],
        color_condition=lambda x, y: y < 5,
        cell_format=lambda x, y: rf"${round(x,1)} ({round(y,1)}\%)$",
    )
