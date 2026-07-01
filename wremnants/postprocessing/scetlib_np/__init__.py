"""SCETlib-NP postprocessing package.

``SCETlibNPParamModel`` (rabbit adapter) and ``SigmaGenModel`` (datacard-free
σ_gen physics core), with their TensorFlow / btgrid dependencies, are imported
lazily so lightweight submodules (e.g. :mod:`response_matrix`, used by setupRabbit
to embed the response matrix in the datacard) import without pulling in
TensorFlow. The package-level re-exports
``wremnants.postprocessing.scetlib_np.SCETlibNPParamModel`` (rabbit's
``--paramModel`` loader) and ``…​.SigmaGenModel`` still work, resolved on first
access via PEP 562.
"""

__all__ = ["SCETlibNPParamModel", "SigmaGenModel"]


def __getattr__(name):
    if name == "SCETlibNPParamModel":
        from wremnants.postprocessing.scetlib_np.param_model import (
            SCETlibNPParamModel,
        )

        return SCETlibNPParamModel
    if name == "SigmaGenModel":
        from wremnants.postprocessing.scetlib_np.sigma_gen import SigmaGenModel

        return SigmaGenModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
