import itertools
import re

import pandas as pd

import rabbit.io_tools
from wums import logging

logger = logging.child_logger(__name__)


def read_groupunc_df(filename, uncs, rename_cols={}, name=None):
    ref_massw = 80379
    ref_massz = 91187.6

    fitresult = rabbit.io_tools.get_fitresult(filename)
    df = rabbit.io_tools.read_impacts_pois(
        fitresult, poi_type="nois", group=True, uncertainties=uncs
    )

    df.iloc[0, 1:] = df.iloc[0, 1:] * 100
    df.iloc[0, 1] += ref_massz if df.loc[0, "Name"] == "massShiftZ100MeV" else ref_massw

    if rename_cols:
        df.rename(columns=rename_cols, inplace=True)
    if name:
        df.loc[0, "Name"] = name

    return df


def read_all_groupunc_df(filenames, uncs, rename_cols={}, names=[]):
    dfs = [
        read_groupunc_df(f, uncs, rename_cols, n)
        for f, n in itertools.zip_longest(filenames, names)
    ]

    return pd.concat(dfs)


def decode_poi_bin(name, var):
    name_split = name.split(var)
    if len(name_split) == 1:
        return None
    else:
        # capture one or more consecutive digits; filter out empty strings
        return next(filter(None, re.split(r"(\d+)", name_split[-1])))


def filter_poi_bins(names, gen_axes, selections={}, base_processes=[], flow=False):
    if isinstance(gen_axes, str):
        gen_axes = [gen_axes]
    if isinstance(base_processes, str):
        base_processes = [base_processes]
    df = pd.DataFrame({"Name": names})
    for axis in gen_axes:
        df[axis] = df["Name"].apply(lambda x, a=axis: decode_poi_bin(x, a))
        df.dropna(inplace=True)
        if flow:
            # set underflow to -1, overflow to max bin number+1
            max_bin = pd.to_numeric(df[axis], errors="coerce").max()
            df[axis] = df[axis].apply(
                lambda x, iu=max_bin: (
                    -1
                    if x[0] == "U"
                    else iu + 1 if x[0] == "O" else int(x) if x is not None else None
                )
            )
        else:
            # set underflow and overflow to None
            df[axis] = df[axis].apply(
                lambda x: None if x is None or x[0] in ["U", "O"] else int(x)
            )

    # filter out rows with NaNs
    mask = ~df.isna().any(axis=1)
    # select rows from base process
    if len(base_processes):
        mask = mask & df["Name"].apply(
            lambda x, b=base_processes: any([x.startswith(p) for p in b])
        )
    # remove rows that have additional axes that are not required (strip off process prefix and poi type postfix and compare length of gen axes assuming they are separated by '_')
    mask = mask & df["Name"].apply(
        lambda x, a=gen_axes, b=base_processes: any(
            (
                len(x.replace(p, "").split("_")[1:-1]) == len(a)
                if x.startswith(p)
                else False
            )
            for p in b
        )
    )
    # gen bin selections
    for k, v in selections.items():
        mask = mask & (df[k] == v)

    filtered_df = df[mask]

    return filtered_df.sort_values(list(gen_axes)).index.to_list()


def select_pois(df, gen_axes=[], selections={}, base_processes=[], flow=False):
    return df.iloc[
        filter_poi_bins(
            df["Name"].values,
            gen_axes,
            selections=selections,
            base_processes=base_processes,
            flow=flow,
        )
    ]
