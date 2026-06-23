"""SCETlib-NP postprocessing package.

``SCETlibNPParamModel`` (and its TensorFlow / btgrid dependencies) is imported
lazily so that lightweight submodules — e.g. :mod:`response_matrix`, used by
setupRabbit to embed the response matrix in the datacard — can be imported
without pulling in TensorFlow. The package-level re-export
``wremnants.postprocessing.scetlib_np.SCETlibNPParamModel`` (used by rabbit's
``--paramModel`` loader) still works, resolved on first access via PEP 562.
"""

__all__ = ["SCETlibNPParamModel"]


def __getattr__(name):
    if name == "SCETlibNPParamModel":
        from wremnants.postprocessing.scetlib_np.param_model import (
            SCETlibNPParamModel,
        )

        return SCETlibNPParamModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
