#!/usr/bin/env python3
"""Checks of setupRabbit --pdfCorr/--asCorr/--mbCorr/--mcCorr in TheoryHelper, against a
stub datagroups that only carries the histmaker metadata and records addSystematic calls:
with a flag the correction hist name comes from it, without the flags it is unchanged.

    python scripts/tests/test_theory_corr_name_flags.py     (or: pytest <this file>)

Exits 1 on any failure.
"""

import sys
from types import SimpleNamespace

from wremnants.postprocessing.rabbit_theory_helper import TheoryHelper

NOMINAL = "scetlib_dyturbo_CT18Z_N3p0LL_N2LO"
NEWNP_MB = "scetlib_dyturbo_LatticeNP_MSHT20mbrange_N3p0LL_N2LO_pdfvars"
NEWNP_MC = "scetlib_dyturbo_LatticeNP_MSHT20mcrange_N3p0LL_N2LO_pdfvars"
FV = "scetlib_dyturbo_LatticeNPLambda4Bugfix_FranksValsVars"
MB = f"{FV}_MSHT20mbrange_N3p0LL_N2LO_pdfvars"
MC = f"{FV}_MSHT20mcrange_N3p0LL_N2LO_pdfvars"
PDFVARS = f"{FV}_CT18Z_N3p0LL_N2LO_pdfvars"
PDFAS = f"{FV}_CT18Z_N3p0LL_N2LO_pdfas"


class StubDatagroups:
    def __init__(self, theory_corrs, pdfs):
        self.meta = {"theoryCorr": theory_corrs, "pdfs": pdfs}
        self.procGroups = {
            g: [("Zmumu", None)]
            for g in ("signal_samples", "signal_samples_inctau", "single_v_samples")
        }
        self.calls = []

    def args_from_metadata(self, key):
        return self.meta.get(key)

    def addSystematic(self, histname=None, **kwargs):
        self.calls.append(histname)


def make_helper(theory_corrs, pdfs=("ct18z",), **flags):
    args = SimpleNamespace(
        noi=["alphaS"],
        **{k: flags.get(k) for k in ("pdfCorr", "asCorr", "mbCorr", "mcCorr")},
    )
    dg = StubDatagroups(list(theory_corrs), list(pdfs))
    th = TheoryHelper("Z", dg, args)
    th.configure(
        resumUnc="none",
        np_model="none",
        pdf_from_corr=True,
        as_from_corr=True,
        samples=["signal_samples_inctau"],
        from_hels=True,
    )
    return th, dg


def test_pdf_as_default_unchanged():
    corrs = [NOMINAL, f"{NOMINAL}_pdfvars", f"{NOMINAL}_pdfas"]
    th, dg = make_helper(corrs)
    th.add_pdf_uncertainty()
    th.add_pdf_alphas_variation(noi=True)
    assert dg.calls == [
        f"{NOMINAL}_pdfvars_CorrByHelicity",
        f"{NOMINAL}_pdfas_CorrByHelicity",
    ]


def test_pdf_as_from_flags():
    corrs = [NOMINAL, f"{NOMINAL}_pdfvars", f"{NOMINAL}_pdfas", PDFVARS, PDFAS]
    th, dg = make_helper(corrs, pdfCorr=PDFVARS, asCorr=PDFAS)
    th.add_pdf_uncertainty()
    th.add_pdf_alphas_variation(noi=True)
    assert dg.calls == [f"{PDFVARS}_CorrByHelicity", f"{PDFAS}_CorrByHelicity"]


def test_quark_mass_default_unchanged():
    pdfs = ["ct18z", "msht20mcrange_renorm", "msht20mbrange_renorm"]
    th, dg = make_helper([NOMINAL, MB, MC], pdfs)
    th.add_quark_mass_vars()
    assert dg.calls == ["pdfMSHT20mbrangeByHelicity", "pdfMSHT20mcrangeByHelicity"]
    th, dg = make_helper([NOMINAL, NEWNP_MB, NEWNP_MC], pdfs)
    th.add_quark_mass_vars()
    assert dg.calls == [
        f"{NEWNP_MB}_CorrByHelicity",
        f"{NEWNP_MC}_CorrByHelicity",
    ]


def test_quark_mass_from_flags():
    pdfs = ["ct18z", "msht20mcrange_renorm", "msht20mbrange_renorm"]
    th, dg = make_helper([NOMINAL, MB, MC], pdfs, mbCorr=MB, mcCorr=MC)
    th.add_quark_mass_vars()
    assert dg.calls == [f"{MB}_CorrByHelicity", f"{MC}_CorrByHelicity"]


def test_separator_always_underscore():
    # a suffixed nominal and a single-entry theoryCorr list: the histmaker writes
    # f"{generator}_Corr", so both resolve to <name>_Corr
    suffixed = "scetlib_dyturbo_LatticeMAP22_CT18Z_N4p0LL_N2LO_adcorrY35"
    th, _ = make_helper([suffixed, PDFVARS, PDFAS, MC, MB])
    assert th.corr_hist_name == f"{suffixed}_Corr"
    th, _ = make_helper([suffixed])
    assert th.corr_hist_name == f"{suffixed}_Corr"
    # --pdfCorr/--asCorr NAME resolve to NAME_Corr next to a suffixed nominal
    th, dg = make_helper([suffixed, PDFVARS, PDFAS], pdfCorr=PDFVARS, asCorr=PDFAS)
    th.set_from_hels([])
    th.add_pdf_uncertainty()
    th.add_pdf_alphas_variation(noi=True)
    assert dg.calls == [f"{PDFVARS}_Corr", f"{PDFAS}_Corr"]


if __name__ == "__main__":
    failed = 0
    tests = [(k, v) for k, v in dict(globals()).items() if k.startswith("test_")]
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:
            failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
