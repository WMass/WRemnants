
from utilities import logging

logger = logging.child_logger(__name__)

def make_latex_table(df, output_dir="./", output_name="table", column_name="column_name", row_name="dataset",
    caption="Table", label="Model", sublabel="Uncertainty", column_title="Configuration", 
    cell_columns=["chi2", "pvalue"], color_condition=None, cell_format=None,
    sort=[],
):
    # df: pandas dataframe
    # cell_columns: columns to fill the cells
    # color_condition: function with condition to color the cell, has to return a boolean (e.g. lambda x, y: y > x)
    # cell_format: function with the cell formatting, has to return a string (e.g. lambda x, y: f"${round(x)} ({round(y)})$")      

    if len(sort)>0:
        df.sort_values(by=sort)

    column_names = df[column_name].values
    if hasattr(column_names, "categories"):
        column_names = [x for x in column_names.categories]
    else:
        column_names = sorted(set(column_names))

    outfile=f"{output_dir}/{output_name}.tex"
    logger.info(f"write {outfile}")
    with open(outfile, "w") as outfile:
        outfile.write(r"\documentclass{article}" +"\n")
        outfile.write(r"\usepackage{caption}" +"\n")
        outfile.write(r"\usepackage[table]{xcolor}" +"\n")
        outfile.write(r"\begin{document}" +"\n")
        outfile.write(r"\newcommand{\NA}{---}" +"\n")

        outfile.write(r"\begin{table}" +"\n")
        outfile.write(r"\caption{\label{table:"+output_name+"}"+caption+r"""}"""+"\n")
        outfile.write(r"\centering"+"\n")

        columns = "l|"
        columns += "".join(["c" for c in range(len(column_names))])
        outfile.write(r"\begin{tabular}{"+columns+"}"+"\n")
        
        if column_title:
            outfile.write("  "+label+" & \multicolumn{"+str(len(column_names))+"}{c}{"+column_title+"} " + r" \\"+"\n")
        else:
            sublabel = f"{label} {sublabel}"
        outfile.write("  "+sublabel+" & " + " & ".join(column_names) + r" \\"+"\n")

        outfile.write(r"  \hline "+"\n")

        row_names = df.loc[df[column_name]==column_names[0]][row_name].values[::-1]
        for nominal in row_names:
            df_n = df.loc[df[row_name]==nominal]
            entries = []
            for p in column_names:         
                df_p = df_n.loc[df_n[column_name] == p][cell_columns]        
                if len(df_p) == 1:
                    vals = df_p.values[0]
                    colorstring = "\cellcolor{red!25}" if color_condition and color_condition(*vals) else ""    # highlight background color where test fails
                    entries.append(f"{colorstring} {cell_format(*vals) if cell_format else vals}")
                else:
                    entries.append(r" \NA ")

            outfile.write(f"  {nominal} & " + " & ".join(entries) + r" \\"+"\n")


        outfile.write(r"  \end{tabular}"+"\n")
        outfile.write(r"\end{table} "+"\n")
        outfile.write(r"\end{document}" +"\n")
