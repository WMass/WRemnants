"""
INPUT -------------------------------------------------------------------------
|* (str) sort_key: the name of the RDF column to be used as the key to sort cols_to_sort
|* (list(str)) cols_to_sort: the RDF columns to be sorted by sort_key
|* (str) order: ascending or descending sort
|* (str) sorted_cols_suffix: output cols have names cols_to_sort + suffix
|
ROUTINE -----------------------------------------------------------------------
|* sort a list of RDF columns by one RDF column as the key, in the order specified
|
OUTPUT ------------------------------------------------------------------------
|* sorted columns appended to the RDF, with namee cols_to_sort + suffix
+------------------------------------------------------------------------------
"""


def sort_rdf_cols(
    df,
    sort_key=None,
    cols_to_sort=None,
    order="descending",
    sorted_cols_suffix="_sorted",
):
    if not (sort_key and cols_to_sort):
        raise ValueError("Please specify sort key and columns to be sorted")
    if type(cols_to_sort) != list:
        raise TypeError("Please put the columns to be sorted in a list of strings")
    if type(sort_key) != str:
        raise TypeError("Please pass in the sort key as a string")
    order_key = "std::greater()" if order == "descending" else ""

    df = df.Define(
        f"sort_idx_{sort_key}", f"ROOT::VecOps::Argsort({sort_key}, {order_key})"
    )
    for col in cols_to_sort:
        df = df.Define(
            f"{col}{sorted_cols_suffix}",
            f"ROOT::VecOps::Take({col}, sort_idx_{sort_key})",
        )
    return df
