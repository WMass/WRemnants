"""SCETlib NP continuous-λ param model and its bT-grid factorisation port.

The fit-time rabbit ParamModel lives in :mod:`.param_model`; the supporting
bT-grid numpy/TF code, Q-integration, caching, response-matrix loader and
λ_central reader are the sibling modules. ``SCETlibNPParamModel`` is
re-exported here so it can be referenced by the short dotted path:

    --paramModel wremnants.postprocessing.scetlib_np.SCETlibNPParamModel
"""

from wremnants.postprocessing.scetlib_np.param_model import SCETlibNPParamModel

__all__ = ["SCETlibNPParamModel"]
