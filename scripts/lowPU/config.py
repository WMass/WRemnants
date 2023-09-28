
## common configuration for lowPU analysis


import decimal
def drange(x, y, jump):
    while x < y:
        yield float(x)
        x += decimal.Decimal(jump)

bins_recoil_reco = [0, 5, 10, 15, 20, 30, 40, 50, 60, 75, 90, 10000]
bins_recoil_gen = [0.0, 10.0, 20.0, 40.0, 60.0, 90.0, 10000]

bins_recoil_qT = list(drange(0, 30, 0.5)) + list(range(30, 60, 2)) + list(range(60, 100, 5)) + list(range(100, 210, 10)) + [10000]



def lepSF_systs(df, results, sName, sVars, defineExpr, baseName, baseAxes, baseCols):

    if sName not in df.GetColumnNames():
        df = df.Define(sName, defineExpr)
        df = df.Define(f"{sName}_tensor", f"Eigen::TensorFixedSize<double, Eigen::Sizes<{sVars}>> res; auto w = nominal_weight*{sName}; std::copy(std::begin(w), std::end(w), res.data()); return res;")
    results.append(df.HistoBoost(f"{baseName}_{sName}", [*baseAxes], [*baseCols, f"{sName}_tensor"]))
    return df