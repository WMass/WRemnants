#!/usr/bin/env python3
"""Dump per-region smoothing polynomial coefficients for SmoothExtendedABCD.

Refit the linearized smoothing polynomial from a nominal fake histogram and
save the per-region polynomial coefficients in the SmoothExtendedABCD
power-series basis.  The output can be loaded as initial parameter values in a
rabbit fit via ``params:PATH`` in the ``--paramModel SmoothExtendedABCDIsoMT``
CLI token.

Typical use::

    python scripts/rabbit/regen_smoothing_params.py \\
        -i mw_with_mu_eta_pt_scetlib_dyturbo.hdf5 \\
        -o /path/to/params.hdf5
"""

import os

import h5py
import numpy as np

from wremnants.postprocessing.datagroups.datagroups import Datagroups
from wremnants.postprocessing.histselections import FakeSelectorSimpleABCD
from wremnants.postprocessing.regression import Regressor
from wremnants.utilities import common, parsing
from wums import ioutils, logging, output_tools

logger = logging.child_logger(__name__)


def make_parser():
    parser = parsing.base_parser()
    parser.description = __doc__
    parser.add_argument(
        "-i",
        "--inputFile",
        required=True,
        type=str,
        help="Input HDF5 histogram file (output of a histmaker).",
    )
    parser.add_argument(
        "-o",
        "--outpath",
        required=True,
        type=str,
        help="Output path for the params HDF5 file (.hdf5 appended if missing).",
    )
    parser.add_argument(
        "--inputBaseName",
        default="nominal",
        type=str,
        help="Name of the nominal histogram inside the input file.",
    )
    parser.add_argument(
        "--fakerateAxes",
        nargs="+",
        default=["eta", "pt", "charge"],
        help="Axes for the fakerate binning.",
    )
    parser.add_argument(
        "--fakeEstimation",
        type=str,
        default="extended1D",
        choices=["simple", "extrapolate", "extended1D", "extended2D"],
        help="Fake estimation mode (must match what will be used in setupRabbit).",
    )
    parser.add_argument(
        "--fakeSmoothingMode",
        type=str,
        default="full",
        choices=FakeSelectorSimpleABCD.smoothing_modes,
        help="Smoothing mode for fake estimate.",
    )
    parser.add_argument(
        "--fakeSmoothingOrder",
        type=int,
        default=3,
        help="Polynomial order for the spectrum smoothing.",
    )
    parser.add_argument(
        "--fakeSmoothingPolynomial",
        type=str,
        default="chebyshev",
        choices=Regressor.polynomials,
        help="Polynomial type for the spectrum smoothing.",
    )
    parser.add_argument(
        "--excludeProcGroups",
        type=str,
        nargs="*",
        default=["QCD"],
        help="Process groups to exclude when building Datagroups.",
    )
    parser.add_argument(
        "--filterProcGroups",
        type=str,
        nargs="*",
        default=None,
        help="If set, keep only these process groups when building Datagroups.",
    )
    return parser


def dump_smoothing_params(
    outpath, fakeselector, datagroups, inputBaseName, meta_data_dict=None, postfix=""
):
    """
    Dump the per-region Chebyshev polynomial coefficients of the nominal fake
    histogram smoothing fit, in the layout expected by SmoothExtendedABCD as
    ``initial_params``.

    Both the WRemnants spectrum regressor and SmoothExtendedABCD now use the
    same Chebyshev basis (T_k of the first kind, with x̃ ∈ [-1, 1] via the
    axis edges), so the regressor coefficients are passed through directly,
    with a projection of ``log(bin_width)`` added so the exported coefficients
    predict yields per bin, not rates per unit of the smoothing axis (the
    regressor fits ``log(data_rate) = log(data_yield / bin_width)``). In the
    simultaneous (extended)ABCD fit the nonprompt process is filled with
    ``OnesSelector``, so ``mc_template = 1`` and the polynomial must carry the
    full absolute scale; the Chebyshev intercept (T_0) is therefore kept.

    The saved array has shape (5 * n_outer * (order+1),) with the layout
    [A_params, B_params, C_params, Ax_params, Bx_params].

    Flat ABCD index ordering for the 5 regions with signal_region=False
    (FakeSelector1DExtendedABCD, with flow=True, after y-axis flip so tight iso is last):
        0=Ax (low mt, fail iso), 1=Bx (low mt, pass iso),
        2=A  (mid mt, fail iso), 3=B  (mid mt, pass iso),
                                 4=C  (signal mt, fail iso)  ← application region, rabbit's free parameter
    Note: signal_region=False drops flat index 5 = D (signal mt, pass iso = signal
    region), which is rabbit's predicted region.

    Histselections ↔ SmoothExtendedABCD model name mapping:
        histsel "application region" (signal mt + fail iso) = C_model (free)
        histsel "signal region"      (signal mt + pass iso) = D_model (predicted)

    Mapping to model order [A=0, B=1, C=2, Ax=3, Bx=4]:
        model_A  ← sideband flat 2
        model_B  ← sideband flat 3
        model_C  ← sideband flat 4
        model_Ax ← sideband flat 0
        model_Bx ← sideband flat 1
    """
    g = datagroups.fakeName
    # Build the combined fake histogram (data - prompt MC) the same way the normal
    # histogram loading does, but without applying the histselector.
    datagroups.loadHistsForDatagroups(
        inputBaseName, syst="", procsToRead=[g], label="_dump_tmp", applySelection=False
    )
    h_fakes = datagroups.groups[g].hists["_dump_tmp"]

    # Single call with signal_region=False: returns 5 regions [Ax=0, Bx=1, A=2, B=3, C=4]
    # The dropped 6th flat element (D = signal mt + pass iso) is the predicted region.
    fakeselector.calculate_fullABCD_smoothed(h_fakes, signal_region=False)
    if not hasattr(fakeselector, "_params_before_reduce"):
        raise RuntimeError(
            "dump_smoothing_params: _params_before_reduce not found on fakeselector. "
            "Ensure fakeSmoothingMode='full' is used."
        )
    # Shape: (*outer_dims, 5, order+1) — indices [Ax=0, Bx=1, A=2, B=3, C=4]
    params_5d = fakeselector._params_before_reduce.copy()

    reg = fakeselector.spectrum_regressor
    order = reg.order

    # Flatten outer dims → (n_outer_flat, n_abcd, order+1)
    outer_shape = params_5d.shape[:-2]
    n_outer_flat = int(np.prod(outer_shape))
    params_5d_flat = params_5d.reshape(n_outer_flat, 5, order + 1)

    # Assemble the 5 model regions in model order [A, B, C, Ax, Bx]
    # A ← flat 2, B ← flat 3, C ← flat 4, Ax ← flat 0, Bx ← flat 1
    params_model = np.stack(
        [
            params_5d_flat[:, 2, :],  # A  (mid mt, fail iso)
            params_5d_flat[:, 3, :],  # B  (mid mt, pass iso)
            params_5d_flat[:, 4, :],  # C  (signal mt, fail iso) = application region
            params_5d_flat[:, 0, :],  # Ax (low mt, fail iso)
            params_5d_flat[:, 1, :],  # Bx (low mt, pass iso)
        ],
        axis=1,
    )  # (n_outer, 5, order+1)

    # Both bases now agree (Chebyshev T_k, x̃ ∈ [-1, 1] via the axis edges).
    # The regressor fits log(data_rate) = log(data_yield / bin_width); the model
    # evaluates yield = exp(poly) * mc. With mc = 1 (OnesSelector in the
    # simultaneous ABCD fit) the target coefficients are those of
    # log(data_yield) = log_rate + log(bin_width). The log(bin_width)
    # contribution is projected onto the Chebyshev basis and added.
    smooth_ax = h_fakes.axes[fakeselector.smoothing_axis_name]
    bin_widths = np.array(smooth_ax.widths)
    x_cheby = reg.transform_x(np.array(smooth_ax.centers))
    n_smooth = len(x_cheby)

    T = np.zeros((n_smooth, order + 1))
    T[:, 0] = 1.0
    if order >= 1:
        T[:, 1] = x_cheby
    for k in range(2, order + 1):
        T[:, k] = 2.0 * x_cheby * T[:, k - 1] - T[:, k - 2]

    # Chebyshev coefficients of log(bin_width) on the smoothing grid, shape (order+1,)
    log_bw_coeffs = np.linalg.lstsq(T, np.log(bin_widths), rcond=None)[0]

    q_model = params_model + log_bw_coeffs[np.newaxis, np.newaxis, :]

    # Flatten to model's layout [A_block, B_block, C_block, Ax_block, Bx_block]
    # Each block: n_outer × (order+1) in C-order
    params_out = q_model.transpose(1, 0, 2).reshape(-1)  # (5 * n_outer * (order+1),)
    if outpath and not os.path.isdir(outpath):
        os.makedirs(outpath)

    outfile = f"{outpath}/params"
    if postfix:
        outfile += f"_{postfix}"

    outfile += ".hdf5"

    with h5py.File(outfile, mode="w") as f:
        f.create_dataset("params", data=params_out)
        f.create_dataset("order", data=np.array(order))
        f.create_dataset(
            "smoothing_axis_name",
            data=np.array(fakeselector.smoothing_axis_name, dtype=h5py.string_dtype()),
        )
        f.create_dataset("n_outer", data=np.array(n_outer_flat))
        f.create_dataset("outer_shape", data=np.array(outer_shape, dtype="int64"))
        if meta_data_dict is not None:
            ioutils.pickle_dump_h5py("meta", meta_data_dict, f)

    logger.info(
        f"Saved smoothing initial params to {outfile}  "
        f"(shape {params_out.shape}, n_outer={n_outer_flat}, n_abcd=5, order={order})"
    )


def main():
    parser = make_parser()
    args = parser.parse_args()
    logging.setup_logger(__file__, args.verbose, args.noColorLogger)

    dg = Datagroups(
        args.inputFile,
        excludeGroups=args.excludeProcGroups if args.excludeProcGroups else None,
        filterGroups=args.filterProcGroups if args.filterProcGroups else None,
    )

    dg.fakerate_axes = args.fakerateAxes
    dg.set_histselectors(
        dg.getNames(),
        args.inputBaseName,
        mode=args.fakeEstimation,
        smoothing_mode=args.fakeSmoothingMode,
        smoothingOrderSpectrum=args.fakeSmoothingOrder,
        smoothingPolynomialSpectrum=args.fakeSmoothingPolynomial,
        mcCorr=None,
        integrate_x=True,
        forceGlobalScaleFakes=False,
        abcdExplicitAxisEdges={},
        fakeTransferAxis="",
        fakeTransferCorrFileName=None,
        histAxesRemovedBeforeFakes=[],
    )

    fakeselector = dg.groups[dg.fakeName].histselector
    logger.info(f"fakeselector type: {type(fakeselector).__name__}")

    meta_data_dict = {
        "meta_info": output_tools.make_meta_info_dict(
            args=args,
            wd=common.base_dir,
        ),
        "meta_info_input": dg.getMetaInfo(),
    }

    dump_smoothing_params(
        args.outpath,
        fakeselector,
        dg,
        args.inputBaseName,
        meta_data_dict=meta_data_dict,
    )


if __name__ == "__main__":
    main()
