"""SCETlib autodiff postprocessing package.

``SCETlibADParamModel`` is a rabbit ParamModel whose prediction is differentiable
in every parameter SCETlib exposes -- alpha_s, the nonperturbative lambdas, the
theory nuisance parameters, and PDF eigenvector coefficients -- via the cached
matched cross section of the SCETlib ``autodiff-sigmaul`` branch.

Imported lazily: the heavy dependencies (TensorFlow and the compiled
``scetlib_qT`` extension) should not be pulled in by a tool that merely wants the
parameter-name map or the response helpers. The package-level re-export
``wremnants.postprocessing.scetlib_ad.SCETlibADParamModel`` -- the name rabbit's
``--paramModel`` loader resolves -- still works, via PEP 562.
"""

__all__ = ["SCETlibADParamModel", "ScetlibADXsec"]


def __getattr__(name):
    if name == "SCETlibADParamModel":
        from wremnants.postprocessing.scetlib_ad.param_model import (
            SCETlibADParamModel,
        )

        return SCETlibADParamModel
    if name == "ScetlibADXsec":
        from wremnants.postprocessing.scetlib_ad.xsec_backend import ScetlibADXsec

        return ScetlibADXsec
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
