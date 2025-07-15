import hist
import numpy as np
from scipy import interpolate

from utilities import common
from wremnants.regression import Regressor, Regressor2D
from wums import boostHistHelpers as hh
from wums import logging

logger = logging.child_logger(__name__)

# thresholds,
abcd_thresholds = {
    "pt": [26, 28, 30],
    "mt": [0, 20, 40],
    "iso": [0, 4, 8, 12],
    "relIso": [0, 0.15, 0.3, 0.45],
    "relJetLeptonDiff": [0, 0.2, 0.35, 0.5],
    "dxy": [0, 0.01, 0.02, 0.03],
}

# default abcd_variables to look for
abcd_variables = (
    ("mt", "passMT"),
    ("relIso", "passIso"),
    ("iso", "passIso"),
    ("relJetLeptonDiff", "passIso"),
    ("dxy", "passDxy"),
)


def get_selection_edges(axis_name, upper_bound=False):
    # returns edges from pass to fail regions [x0, x1, x2, x3] e.g. x=[x0,x1], dx=[x1,x2], d2x=[x2,x3]
    if axis_name in abcd_thresholds:
        ts = abcd_thresholds[axis_name]
        if axis_name in ["mt", "pt"]:
            # low: failing, high: passing, no upper bound
            return None, complex(0, ts[2]), complex(0, ts[1]), complex(0, ts[0])
        if axis_name in ["dxy", "iso", "relIso", "relJetLeptonDiff"]:
            # low: passing, high: failing, no upper bound
            return (
                complex(0, ts[0]),
                complex(0, ts[1]),
                complex(0, ts[2]),
                (complex(0, ts[3]) if upper_bound else None),
            )
    else:
        raise RuntimeError(f"Can not find threshold for abcd axis {axis_name}")


def extend_edges(traits, x):
    # extend array for underflow/overflow with distance from difference of two closest values
    if traits.underflow:
        new_x = x[0] - x[1] + x[0]
        x = np.array([new_x, *x])
        logger.debug(f"Extend array by underflow bin using {new_x}")
    if traits.overflow:
        new_x = x[-1] + x[-1] - x[-2]
        x = np.array([*x, new_x])
        logger.debug(f"Extend array by overflow bin using {new_x}")
    return x


def get_rebinning(edges, axis_name):
    if axis_name == "mt":
        target_edges = common.get_binning_fakes_mt(high_mt_bins=True)
    elif axis_name == "pt":
        target_edges = common.get_binning_fakes_pt(edges[0], edges[-1])
    else:
        raise RuntimeError(f"No automatic rebinning known for axis {axis_name}")

    if len(edges) == len(target_edges) and all(edges == target_edges):
        logger.debug(f"Axis has already in target edges, no rebinning needed")
        return None

    i = 0
    rebin = np.zeros_like(target_edges)
    for e in edges:
        if e >= target_edges[i]:
            rebin[i] = e
            i += 1
        if i >= len(target_edges):
            break
    rebin = rebin[:i]

    if len(rebin) <= 2:
        logger.warning(f"No automatic rebinning possible for axis {axis_name}")
        return None

    logger.debug(f"Rebin axis {axis_name} to {rebin}")
    return rebin


def divide_arrays(num, den, cutoff=1, replace=1):
    r = num / den
    # criteria = abs(den) <= cutoff
    criteria = den <= cutoff
    if np.sum(criteria) > 0:
        logger.warning(
            f"Found {np.sum(criteria)} / {criteria.size} values in denominator less than or equal to {cutoff}, the ratio will be set to {replace} for those"
        )
        r[criteria] = (
            replace  # if denumerator is close to 0 set ratio to 1 to avoid large negative/positive values
        )
    return r


def spline_smooth_nominal(binvals, edges, edges_out, axis):
    # interpolation of CDF
    cdf = np.cumsum(binvals, axis=axis)
    padding = binvals.ndim * [(0, 0)]
    padding[axis] = (1, 0)
    cdf = np.pad(cdf, padding)

    x = edges
    y = cdf
    xout = edges_out

    spline = interpolate.make_interp_spline(x, y, axis=axis, bc_type=None)
    yout = spline(xout)

    binvalsout = np.diff(yout, axis=axis)
    binvalsout = np.maximum(binvalsout, 0.0)

    return binvalsout


def spline_smooth(binvals, edges, edges_out, axis, binvars=None, syst_variations=False):
    ynom = spline_smooth_nominal(binvals, edges, edges_out, axis=axis)

    if not syst_variations:
        return ynom, None

    nvars = binvals.shape[axis]

    yvars = np.zeros((*ynom.shape, nvars, 2), dtype=ynom.dtype)

    binerrs = np.sqrt(binvars)

    # fluctuate the bin contents one by one to build the variations
    for ivar in range(nvars):
        for iupdown in range(2):
            scale = 1.0 if iupdown == 0 else -1.0

            binvalsvar = binvals.copy()
            varsel = binvals.ndim * [slice(None)]
            varsel[axis] = ivar
            binvalsvar[*varsel] += scale * binerrs[*varsel]

            yvars[..., ivar, iupdown] = spline_smooth_nominal(
                binvalsvar, edges, edges_out, axis=axis
            )

    return ynom, yvars


class HistselectorABCD(object):
    def __init__(
        self,
        h,
        name_x=None,
        name_y=None,
        fakerate_axes=["eta", "pt", "charge"],
        smoothing_axis_name="pt",
        rebin_smoothing_axis="automatic",  # can be a list of bin edges, "automatic", or None
        upper_bound_y=None,  # using an upper bound on the abcd y-axis (e.g. isolation)
        integrate_x=True,  # integrate the abcd x-axis in final histogram (allows simplified procedure e.g. for extrapolation method)
    ):
        self.upper_bound_y = upper_bound_y
        self.integrate_x = integrate_x

        self.name_x = name_x
        self.name_y = name_y
        if name_x is None or name_y is None:
            self.set_abcd_axes(h)

        self.axis_x = h.axes[self.name_x]
        self.axis_y = h.axes[self.name_y]

        self.sel_x = None
        self.sel_dx = None
        self.sel_y = None
        self.sel_dy = None
        self.set_selections_x()
        self.set_selections_y()

        if fakerate_axes is not None:
            self.fakerate_axes = fakerate_axes
            self.fakerate_integration_axes = [
                n
                for n in h.axes.name
                if n not in [self.name_x, self.name_y, *fakerate_axes]
            ]
            logger.debug(
                f"Setting fakerate integration axes to {self.fakerate_integration_axes}"
            )

        self.smoothing_axis_name = smoothing_axis_name
        edges = h.axes[smoothing_axis_name].edges
        if rebin_smoothing_axis == "automatic":
            self.rebin_smoothing_axis = get_rebinning(edges, self.smoothing_axis_name)
        else:
            self.rebin_smoothing_axis = rebin_smoothing_axis

        edges = (
            edges if self.rebin_smoothing_axis is None else self.rebin_smoothing_axis
        )
        edges = extend_edges(h.axes[self.smoothing_axis_name].traits, edges)
        self.smoothing_axis_min = edges[0]
        self.smoothing_axis_max = edges[-1]

    # A
    def get_hist_failX_failY(self, h):
        return h[{self.name_x: self.sel_dx, self.name_y: self.sel_dy}]

    # B
    def get_hist_failX_passY(self, h):
        return h[{self.name_x: self.sel_dx, self.name_y: self.sel_y}]

    # C
    def get_hist_passX_failY(self, h):
        return h[{self.name_x: self.sel_x, self.name_y: self.sel_dy}]

    # D
    def get_hist_passX_passY(self, h):
        return h[{self.name_x: self.sel_x, self.name_y: self.sel_y}]

    def set_abcd_axes(self, h):
        for a, b in abcd_variables:
            if self.name_x is None and a in h.axes.name:
                self.name_x = a
                continue
            if self.name_x is None and b in h.axes.name:
                self.name_x = b
                continue
            if self.name_y is None and a in h.axes.name:
                self.name_y = a
                continue
            if self.name_y is None and b in h.axes.name:
                self.name_y = b
                continue
        if self.name_x is None or self.name_y is None:
            raise RuntimeError(
                f"Can not find ABCD axes among histogram axes {h.axes.name}"
            )
        logger.debug(f"Found ABCD axes {self.name_x} and {self.name_y}")

    # set slices object for selection of signal and sideband regions
    def set_selections_x(self):
        if self.name_x.startswith("pass"):
            self.sel_x = 1
            self.sel_dx = 0
        else:
            x0, x1, x2, x3 = get_selection_edges(self.name_x)
            s = hist.tag.Slicer()
            do = hist.sum if self.integrate_x else None
            self.sel_x = (
                s[x0:x1:do] if x0 is not None and x1.imag > x0.imag else s[x1:x0:do]
            )
            self.sel_dx = (
                s[x1 : x3 : hist.sum]
                if x3 is None or x3.imag > x1.imag
                else s[x3 : x1 : hist.sum]
            )

    def set_selections_y(self):
        if self.name_y.startswith("pass"):
            self.sel_y = 1
            self.sel_dy = 0
        else:
            y0, y1, y2, y3 = get_selection_edges(
                self.name_y, upper_bound=self.upper_bound_y
            )
            s = hist.tag.Slicer()
            self.sel_y = (
                s[y0 : y1 : hist.sum]
                if y0 is not None and y1.imag > y0.imag
                else s[y1 : y0 : hist.sum]
            )
            self.sel_dy = (
                s[y1 : y3 : hist.sum]
                if y3 is None or y3.imag > y1.imag
                else s[y3 : y1 : hist.sum]
            )


class SignalSelectorABCD(HistselectorABCD):
    # simple ABCD method
    def __init__(self, h, *args, **kwargs):
        super().__init__(h, *args, **kwargs)

    # signal region selection
    def get_hist(self, h, is_nominal=False):
        return self.get_hist_passX_passY(h)


class FakeSelectorSimpleABCD(HistselectorABCD):
    # supported smoothing mode choices
    smoothing_modes = ["binned", "fakerate", "hybrid", "full"]

    # simple ABCD method
    def __init__(
        self,
        h,
        *args,
        smoothing_mode="full",
        smoothing_order_fakerate=3,
        smoothing_order_spectrum=3,
        smoothing_polynomial_spectrum="power",
        throw_toys=None,  # "normal", # None, 'normal' or 'poisson'
        global_scalefactor=1,  # apply global correction factor on prediction
        **kwargs,
    ):
        super().__init__(h, *args, **kwargs)

        # nominal histogram to be used to transfer variances for systematic variations
        self.h_nominal = None
        self.global_scalefactor = global_scalefactor

        if smoothing_mode not in FakeSelectorSimpleABCD.smoothing_modes:
            raise NotImplementedError(
                f"Smoothing mode {smoothing_mode} is not implemented. Supported smoothing modes are {FakeSelectorSimpleABCD.smoothing_modes}"
            )

        self.smoothing_mode = smoothing_mode

        # select appropriate regressor objects depending on type of smoothing
        if self.smoothing_mode in ["fakerate", "hybrid"]:
            self.fakerate_regressor = Regressor(
                "bernstein",
                smoothing_order_fakerate,
                min_x=self.smoothing_axis_min,
                max_x=self.smoothing_axis_max,
            )
            # swap the A and C regions for better numerical behaviour (only makes sense for fakerate and hybrid smoothing)
            self.swap_regions = True
        else:
            self.fakerate_regressor = None
            self.swap_regions = False

        if self.smoothing_mode in ["fakerate", "hybrid", "full"]:
            self.spectrum_regressor = Regressor(
                smoothing_polynomial_spectrum,
                smoothing_order_spectrum,
                min_x=self.smoothing_axis_min,
                max_x=self.smoothing_axis_max,
                # nnls = self.smoothing_mode not in ["full"], # constraint is handled elsewhere in the full smoothing case
                nnls=False,
            )
        else:
            self.spectrum_regressor = None

        if self.smoothing_mode in ["binned"]:
            # rebinning doesn't make sense for binned estimation
            self.rebin_smoothing_axis = None

        if hasattr(self, "fakerate_integration_axes"):
            if smoothing_mode == "full" and self.fakerate_integration_axes:
                raise NotImplementedError(
                    "Smoothing of full fake prediction is not currently supported together with integration axes."
                )

        self.throw_toys = throw_toys

        # histogram with nonclosure corrections
        self.hCorr = None

    def set_correction(self, hQCD, axes_names=False, mirror_axes=["eta"], flow=True):
        # hQCD is QCD MC histogram before selection (should contain variances)
        # axes_names: the axes names to bin the correction in. If empty make an inclusive correction (i.e. a single number)
        hQCD_rebin = (
            hh.rebinHist(hQCD, self.smoothing_axis_name, self.rebin_smoothing_axis)
            if self.rebin_smoothing_axis is not None
            else hQCD
        )

        s = hist.tag.Slicer()

        keep_axes = [*axes_names, self.name_x, self.name_y, self.smoothing_axis_name]
        if any(n not in keep_axes for n in hQCD_rebin.axes.name):
            # rebin instead of integrate to keep axes to allow for easy post processing
            hQCD_rebin = hQCD_rebin[
                {
                    a.name: s[:: hist.rebin(a.size)]
                    for a in hQCD_rebin.axes
                    if a.name not in keep_axes
                }
            ]

        # mirror eta axes to cope with limited QCD MC stat
        for n in mirror_axes:
            if n in hQCD_rebin.axes.name and hQCD_rebin.axes[n].size > 1:
                hQCD_rebin = hh.mirrorAxis(hQCD_rebin, n)

        # prediction without smoothing
        d, dvar = self.calculate_fullABCD(hQCD_rebin, flow=flow)
        hPred = hist.Hist(
            *hQCD_rebin[
                {
                    self.name_x: self.sel_x if not self.integrate_x else hist.sum,
                    self.name_y: self.sel_y,
                }
            ].axes,
            storage=hQCD_rebin.storage_type(),
        )
        hPred.values(flow=flow)[...] = d
        if hPred.storage_type == hist.storage.Weight:
            hPred.variances(flow=flow)[...] = dvar

        # truth histogram
        hTruth = self.get_hist_passX_passY(hQCD_rebin)

        if any(n not in axes_names for n in hTruth.axes.name):
            hTruth = hTruth[
                {
                    a.name: (
                        s[:: hist.rebin(a.size)]
                        if a.name in hPred.axes.name
                        else hist.sum
                    )
                    for a in hTruth.axes
                    if a.name not in axes_names
                }
            ]
        if any(n not in axes_names for n in hPred.axes.name):
            hPred = hPred[
                {
                    a.name: s[:: hist.rebin(a.size)]
                    for a in hPred.axes
                    if a.name not in axes_names
                }
            ]

        sel = {n: hist.sum for n in self.fakerate_integration_axes}
        self.hCorr = hh.divideHists(hTruth[sel], hPred[sel])

        if axes_names is not None and len(axes_names) == 0:
            logger.info(f"Got QCD MC corrections of {self.hCorr.values()}")
        else:
            logger.debug(f"Got QCD MC corrections of {self.hCorr.values()}")

        if np.any(~np.isfinite(self.hCorr.values(flow=flow))):
            logger.warning(
                f"{sum(~np.isfinite(self.hCorr.values(flow=flow)))} Inf or NaN values in QCD MC nonclosure correction"
            )
        if np.any(~np.isfinite(self.hCorr.variances(flow=flow))):
            logger.warning(
                f"{sum(~np.isfinite(self.hCorr.values(flow=flow)))} Inf or NaN variances in QCD MC nonclosure correction"
            )

    def apply_correction(self, y, yvar=None, flow=True):
        # apply QCD MC nonclosure correction and account for variance of correction
        cval = self.hCorr.values(flow=flow)
        y = y * cval
        if yvar is not None:
            cvar = self.hCorr.variances(flow=flow)
            yvar = y**2 * cvar + cval**2 * yvar
        return y, yvar

    def transfer_variances(self, h, set_nominal=False):
        if set_nominal:
            self.h_nominal = h.copy()
        elif self.h_nominal is not None:
            h = hh.transfer_variances(h, self.h_nominal)
        elif h.storage_type == hist.storage.Weight:
            logger.warning(
                "Nominal histogram is not set but current histogram has variances, use those"
            )
        else:
            raise RuntimeError(f"Failed to transfer variances")
        return h

    def get_smoothing_syst(self, h):
        # TODO this might not be safe in certain future parallelization schemes
        smoothing_mode_old = self.smoothing_mode
        self.smoothing_mode = "fakerate"
        halt = self.get_hist(h)
        self.smoothing_mode = smoothing_mode_old

        axis_var = hist.axis.Integer(0, 1, underflow=False, overflow=False, name="var")
        hout = hist.Hist(*halt.axes, axis_var)
        hout[{"var": 0}] = halt.values(flow=True)

        return hout

    def get_hist(
        self,
        h,
        is_nominal=False,
        variations_frf=False,
        variations_smoothing=False,
        flow=True,
        use_spline=False,
    ):
        if self.smoothing_mode in ["fakerate", "hybrid"]:
            h = self.transfer_variances(h, set_nominal=is_nominal)
            y_frf, y_frf_var = self.compute_fakeratefactor(
                h, smoothing=True, syst_variations=variations_frf
            )

            if not self.integrate_x and self.swap_regions:
                logger.warning(
                    f"Regions for fakerate estiamation can only be swapped if abcd-x axis is integrated"
                )
            if self.swap_regions and self.integrate_x:
                if type(self) == FakeSelectorSimpleABCD:
                    # replace C region with B region
                    hC = self.get_hist_failX_passY(h)
                elif type(self) == FakeSelector1DExtendedABCD:
                    # replace C region with Ax region
                    hC = h[{self.name_x: self.sel_d2x, self.name_y: self.sel_dy}]
            else:
                hC = self.get_hist_passX_failY(h)

            if use_spline and self.smoothing_mode in ["hybrid"]:
                hCNew = hC[{self.smoothing_axis_name: hist.rebin(3)}]
            else:
                hCNew = hC

            cval = hCNew.values(flow=flow)
            cvar = hCNew.variances(flow=flow)
            if self.smoothing_mode in ["hybrid"]:
                cval, cvar = self.smoothen_spectrum(
                    hC,
                    hCNew.axes[self.smoothing_axis_name].edges,
                    cval,
                    cvar,
                    syst_variations=variations_smoothing,
                    use_spline=use_spline,
                    flow=flow,
                )

            d = cval * y_frf

            if variations_smoothing:
                dvar = cvar[..., :, :] * y_frf[..., np.newaxis, np.newaxis]
            elif variations_frf:
                dvar = cval[..., np.newaxis, np.newaxis] * y_frf_var[..., :, :]
            elif self.smoothing_mode in ["hybrid"]:
                # noo bin by bin statistical uncertainty, all regions covered by smoothing
                dvar = np.zeros_like(cvar)
            else:
                # only take bin by bin uncertainty from c region
                dvar = y_frf**2 * cvar

        elif self.smoothing_mode == "full":
            h = self.transfer_variances(h, set_nominal=is_nominal)
            d, dvar = self.calculate_fullABCD_smoothed(
                h, flow=flow, syst_variations=variations_smoothing, use_spline=False
            )
        elif self.smoothing_mode == "binned":
            # no smoothing of rates
            d, dvar = self.calculate_fullABCD(h, flow=flow)
        else:
            raise ValueError("invalid choice of smoothing_mode")

        # set histogram in signal region
        hSignal = hist.Hist(
            *h[
                {
                    self.name_x: self.sel_x if not self.integrate_x else hist.sum,
                    self.name_y: self.sel_y,
                }
            ].axes,
            storage=hist.storage.Double() if variations_smoothing else h.storage_type(),
        )
        hSignal.values(flow=flow)[...] = d
        if (variations_smoothing or variations_frf) and self.smoothing_mode != "binned":
            hSignal = self.get_syst_hist(hSignal, d, dvar, flow=flow)
        elif hSignal.storage_type == hist.storage.Weight:
            hSignal.variances(flow=flow)[...] = dvar

        if self.global_scalefactor != 1:
            hSignal = hh.scaleHist(hSignal, self.global_scalefactor)

        return hSignal

    def get_yields_applicationregion(self, h, flow=True):
        hC = self.get_hist_passX_failY(h)
        c = hC.values(flow=flow)
        if h.storage_type == hist.storage.Weight:
            cvar = hC.variances(flow=flow)
            return c, cvar
        return c, None

    def compute_fakeratefactor(
        self, h, smoothing=False, syst_variations=False, flow=True
    ):
        # rebin in smoothing axis to have stable ratios
        sel = {n: hist.sum for n in self.fakerate_integration_axes}
        hNew = (
            hh.rebinHist(h[sel], self.smoothing_axis_name, self.rebin_smoothing_axis)
            if self.rebin_smoothing_axis is not None
            else h[sel]
        )

        # select sideband regions
        ha = self.get_hist_failX_failY(hNew)
        if not self.integrate_x and self.swap_regions:
            logger.warning(
                f"Regions for fakerate estiamation can only be swapped if abcd-x axis is integrated"
            )
        if self.swap_regions and self.integrate_x:
            # replace B region with C region
            hb = self.get_hist_passX_failY(hNew)
        else:
            hb = self.get_hist_failX_passY(hNew)

        a = ha.values(flow=flow)
        b = hb.values(flow=flow)
        # fakerate factor
        y = divide_arrays(b, a, cutoff=1)
        if h.storage_type == hist.storage.Weight:
            avar = ha.variances(flow=flow)
            bvar = hb.variances(flow=flow)
            y_var = bvar / a**2 + b**2 * avar / a**4
            y_var[a <= 1] = 1e10
        else:
            y_var = None

        if self.hCorr:
            y, y_var = self.apply_correction(y, y_var)

        if smoothing:
            x = self.get_bin_centers_smoothing(
                hNew, flow=True
            )  # the bins where the smoothing is performed (can be different to the bins in h)
            y, y_var = self.smoothen(
                h,
                x,
                y,
                y_var,
                regressor=self.fakerate_regressor,
                syst_variations=syst_variations,
                flow=flow,
            )

        # broadcast abcd-x axis and application axes
        slices = [
            slice(None) if n in ha.axes.name else np.newaxis
            for n in h[{self.name_x: self.sel_x}].axes.name
            if n != self.name_y
        ]
        y = y[*slices]
        y_var = y_var[*slices] if y_var is not None else None

        return y, y_var

    def smoothen(
        self, h, x, y, y_var, regressor, syst_variations=False, reduce=False, flow=True
    ):
        if h.storage_type == hist.storage.Weight:
            # transform with weights
            w = 1 / np.sqrt(y_var)
        else:
            logger.warning(
                "Smoothing extended ABCD on histogram without uncertainties, make an unweighted linear squared solution."
            )
            w = np.ones_like(y)

        # move smoothing axis to last
        axes = [
            n
            for n in h.axes.name
            if n not in [self.name_x, self.name_y, *self.fakerate_integration_axes]
        ]
        idx_ax_smoothing = axes.index(self.smoothing_axis_name)
        if idx_ax_smoothing != len(axes) - 1:
            y = np.moveaxis(y, idx_ax_smoothing, -1)
            w = np.moveaxis(w, idx_ax_smoothing, -1)

        # smoothen
        regressor.solve(x, y, w)

        if reduce:
            # add up parameters from smoothing of individual sideband regions
            if type(self) == FakeSelectorSimpleABCD:
                # exp(-a + b + c)
                # ['a', 'b', 'c']
                w_region = np.array([-1, 1, 1], dtype=int)
            elif type(self) == FakeSelector1DExtendedABCD:
                # exp(ax + 2*b - bx -2*a + c)
                # ['ax', 'a', 'bx', 'b', 'c']
                w_region = np.array([1, -1, -2, 2, 1], dtype=int)
            elif type(self) == FakeSelector2DExtendedABCD:
                # exp(2*c + 2*ax + 2*ay + 2*b - cy - axy - bx - 4*a)
                # ['axy', 'ax', 'bx', 'ay', 'a', 'b', 'cy', 'c']
                w_region = np.array([-1, 2, -1, 2, -4, 2, -1, 2], dtype=int)

            regressor.reduce_parameters(w_region)

            if regressor.polynomial == "monotonic":
                regressor.force_positive(exclude_idx=0)

        # evaluate in range of original histogram
        x_smooth_orig = self.get_bin_centers_smoothing(h, flow=True)
        y_smooth_orig = regressor.evaluate(x_smooth_orig)

        if syst_variations:
            y_smooth_var_orig = regressor.get_eigenvector_predictions(x_smooth_orig)
        else:
            y_smooth_var_orig = None

        # move smoothing axis to original positon again
        if idx_ax_smoothing != len(axes) - 1:
            y_smooth_orig = np.moveaxis(y_smooth_orig, -1, idx_ax_smoothing)
            y_smooth_var_orig = (
                np.moveaxis(y_smooth_var_orig, -3, idx_ax_smoothing)
                if syst_variations
                else None
            )

        # check for negative rates
        if np.sum(y_smooth_orig < 0) > 0:
            logger.warning(
                f"Found {np.sum(y_smooth_orig<0)} bins with negative values from smoothing"
            )
        if y_smooth_var_orig is not None and np.sum(y_smooth_var_orig < 0) > 0:
            logger.warning(
                f"Found {np.sum(y_smooth_var_orig<0)} bins with negative values from smoothing variations"
            )

        return y_smooth_orig, y_smooth_var_orig

    def smoothen_spectrum(
        self,
        h,
        edges,
        sval,
        svar,
        syst_variations=False,
        use_spline=False,
        reduce=False,
        flow=True,
    ):
        smoothidx = [
            n for n in h.axes.name if n not in [self.name_x, self.name_y]
        ].index(self.smoothing_axis_name)
        smoothing_axis = h.axes[self.smoothing_axis_name]
        nax = sval.ndim

        # underflow and overflow are left unchanged along the smoothing axis
        # so we need to exclude them if they have been otherwise included
        if flow:
            smoothstart = 1 if smoothing_axis.traits.underflow else 0
            smoothstop = -1 if smoothing_axis.traits.overflow else None
            smoothslice = slice(smoothstart, smoothstop)
        else:
            smoothslice = slice(None)

        sel = nax * [slice(None)]
        sel[smoothidx] = smoothslice

        sval = sval[*sel]
        svar = svar[*sel]

        if use_spline:
            if reduce:
                raise NotImplementedError(
                    "splines with reduction over regions not implemented yet."
                )

            print("splines")

            sval, svar = spline_smooth(
                sval,
                edges=edges,
                edges_out=h.axes[self.smoothing_axis_name].edges,
                axis=smoothidx,
                binvars=svar,
                syst_variations=syst_variations,
            )
        else:
            xwidth = h.axes[self.smoothing_axis_name].widths

            xwidthtgt = xwidth[
                *smoothidx * [None],
                :,
                *(nax - smoothidx - 2 + (0 if reduce else 1)) * [None],
            ]
            xwidth = xwidth[*smoothidx * [None], :, *(nax - smoothidx - 1) * [None]]

            sval *= 1.0 / xwidth
            svar *= 1.0 / xwidth**2

            goodbin = (sval > 0.0) & (svar > 0.0)
            if goodbin.size - np.sum(goodbin) > 0:
                logger.warning(
                    f"Found {goodbin.size-np.sum(goodbin)} of {goodbin.size} bins with 0 or negative bin content, those will be set to 0 and a large error"
                )

            logd = np.where(goodbin, np.log(sval), 0.0)
            logdvar = np.where(goodbin, svar / sval**2, np.inf)

            x = self.get_bin_centers_smoothing(
                h, flow=flow
            )  # the bins where the smoothing is performed (can be different to the bins in h)

            sval, svar = self.smoothen(
                h,
                x,
                logd,
                logdvar,
                regressor=self.spectrum_regressor,
                syst_variations=syst_variations,
                reduce=reduce,
            )

            sval = np.exp(sval) * xwidthtgt
            sval = np.where(np.isfinite(sval), sval, 0.0)
            if syst_variations:
                svar = np.exp(svar) * xwidthtgt[..., None, None] ** 2
                svar = np.where(
                    (sval[..., None, None] > 0.0) & np.isfinite(svar),
                    svar,
                    sval[..., None, None],
                )

        # get output shape from original hist axes, but as for result histogram
        hOut = (
            h[{self.name_x: self.sel_x if not self.integrate_x else hist.sum}]
            if self.name_x in h.axes.name
            else h
        )
        out = np.zeros(
            [a.extent if flow else a.shape for a in hOut.axes if a.name != self.name_y],
            dtype=sval.dtype,
        )
        # leave the underflow and overflow unchanged if present
        out[*sel[:-1]] = sval
        if syst_variations:
            outvar = np.zeros_like(out)
            outvar = outvar[..., None, None] * np.ones(
                (*outvar.shape, *svar.shape[-2:]), dtype=outvar.dtype
            )
            # leave the underflow and overflow unchanged if present
            outvar[*sel[:-1], :, :] = svar
        else:
            # with full smoothing all of the statistical uncertainty is included in the
            # explicit variations, so the remaining binned uncertainty is zero
            outvar = np.zeros_like(out)

        return out, outvar

    def get_syst_hist(self, hNominal, values, alternate, flow=True):
        # return systematic histogram with nominal and nominal+variation on diagonal elements for systematic axes of parameters and up/down variations
        hsyst = hist.Hist(
            *hNominal.axes,
            hist.axis.Integer(
                0, alternate.shape[-2], name="_param", overflow=False, underflow=False
            ),
            common.down_up_axis,
            storage=hist.storage.Double(),
        )

        # check alternate lower than 0
        # vcheck = alternate < (-1*values[...,np.newaxis,np.newaxis])
        # if np.sum(vcheck) > 0:
        #     logger.warning(f"Found {np.sum(vcheck)} bins with alternate giving negative yields, set these to 0")
        #     alternate[vcheck] = 0

        hsyst.values(flow=flow)[...] = alternate - values[..., np.newaxis, np.newaxis]

        # decorrelate in fakerate axes
        axes_names = [
            n
            for n in self.fakerate_axes
            if self.smoothing_mode == "binned" or n != self.smoothing_axis_name
        ]
        hsyst = hh.expand_hist_by_duplicate_axes(
            hsyst, axes_names, [f"_{n}" for n in axes_names]
        )

        # add nominal hist and broadcast
        hNominal = hh.addHists(hNominal, hsyst)
        return hNominal

    def get_bin_centers_smoothing(self, h, flow=True):
        return self.get_bin_centers(h, self.smoothing_axis_name, flow=flow)

    def get_bin_edges_smoothing(self, h, flow=True):
        return self.get_bin_edges(h, self.smoothing_axis_name, flow=flow)

    def get_bin_edges(self, h, axis_name, flow=True):
        x = h.axes[axis_name].edges
        if flow:
            x = extend_edges(h.axes[axis_name].traits, x)
        return x

    def get_bin_centers(self, h, axis_name, flow=True):
        x = h.axes[axis_name].centers
        if flow:
            x = extend_edges(h.axes[axis_name].traits, x)
        return x

    def calculate_fullABCD(self, h, flow=True):
        if len(self.fakerate_integration_axes) > 0:
            logger.warning(
                f"Binned fake estimation is performed but fakerate integration axes {self.fakerate_integration_axes} are set, the bin-by-bin stat uncertainties are not correct along this axis."
            )
        if not self.integrate_x:
            logger.warning(
                f"Binned fake estimation is performed but ABCD x-axis is not integrated, the bin-by-bin stat uncertainties are not correct along this axis."
            )

        c, cvar = self.get_yields_applicationregion(h)
        frf, frf_var = self.compute_fakeratefactor(h)

        d = c * frf
        if h.storage_type == hist.storage.Weight:
            dvar = frf**2 * cvar + c**2 * frf_var

        return d, dvar

    def calculate_fullABCD_smoothed(
        self, h, syst_variations=False, use_spline=False, signal_region=False, flow=True
    ):

        if type(self) in [FakeSelectorSimpleABCD, FakeSelector1DExtendedABCD]:
            # sum up high abcd-y axis bins
            h = hh.rebinHist(h, self.name_y, h.axes[self.name_y].edges[:2])
        if type(self) == FakeSelectorSimpleABCD:
            h = hh.rebinHist(h, self.name_x, [0, h.axes[self.name_x].edges[2]])
        else:
            # sum high mT bins
            h = hh.rebinHist(h, self.name_x, h.axes[self.name_x].edges[:3])

        if use_spline:
            hNew = h[{self.smoothing_axis_name: hist.rebin(3)}]
        else:
            hNew = h

        # get values and variances of all sideband regions (this assumes signal region is at high abcd-x and low abcd-y axis bins)
        sval = hNew.values(flow=flow)
        svar = hNew.variances(flow=flow)
        # move abcd axes last
        idx_x = hNew.axes.name.index(self.name_x)
        idx_y = hNew.axes.name.index(self.name_y)

        sval = np.moveaxis(sval, [idx_x, idx_y], [-2, -1])
        svar = np.moveaxis(svar, [idx_x, idx_y], [-2, -1])

        # invert y-axis to get signal region last
        sval = np.flip(sval, axis=-1)
        svar = np.flip(svar, axis=-1)

        if signal_region:
            sval = sval.reshape((*sval.shape[:-2], sval.shape[-2] * sval.shape[-1]))[
                ..., -1
            ]
            svar = svar.reshape((*svar.shape[:-2], svar.shape[-2] * svar.shape[-1]))[
                ..., -1
            ]
        else:
            # make abcd axes flat, take all but last bin (i.e. signal region D)
            sval = sval.reshape((*sval.shape[:-2], sval.shape[-2] * sval.shape[-1]))[
                ..., :-1
            ]
            svar = svar.reshape((*svar.shape[:-2], svar.shape[-2] * svar.shape[-1]))[
                ..., :-1
            ]

        return self.smoothen_spectrum(
            h,
            hNew.axes[self.smoothing_axis_name].edges,
            sval,
            svar,
            syst_variations=syst_variations,
            use_spline=use_spline,
            reduce=not signal_region,
            flow=flow,
        )


class FakeSelector1DExtendedABCD(FakeSelectorSimpleABCD):
    # extended ABCD method with 5 control regions as desribed in https://arxiv.org/abs/1906.10831 equation 16
    def __init__(self, h, *args, **kwargs):
        super().__init__(h, *args, **kwargs)
        self.sel_d2x = None
        self.set_selections_x(integrate_x=self.integrate_x)

    # set slices object for selection of sideband regions
    def set_selections_x(self, integrate_x=True):
        x0, x1, x2, x3 = get_selection_edges(self.name_x)
        s = hist.tag.Slicer()
        do = hist.sum if integrate_x else None
        self.sel_x = (
            s[x0:x1:do] if x0 is not None and x1.imag > x0.imag else s[x1:x0:do]
        )
        self.sel_dx = (
            s[x1 : x2 : hist.sum] if x2.imag > x1.imag else s[x2 : x1 : hist.sum]
        )
        self.sel_d2x = (
            s[x2 : x3 : hist.sum] if x3.imag > x2.imag else s[x3 : x2 : hist.sum]
        )

    def calculate_fullABCD(self, h, flow=True, syst_variations=False):
        if len(self.fakerate_integration_axes) > 0:
            logger.warning(
                f"Binned fake estimation is performed but fakerate integration axes {self.fakerate_integration_axes} are set, the bin-by-bin stat uncertainties are not correct along this axis."
            )
        if not self.integrate_x:
            logger.warning(
                f"Binned fake estimation is performed but ABCD x-axis is not integrated, the bin-by-bin stat uncertainties are not correct along this axis."
            )

        sel = {n: hist.sum for n in self.fakerate_integration_axes}

        ha = h[{**sel, self.name_x: self.sel_dx, self.name_y: self.sel_dy}]
        hax = h[{**sel, self.name_x: self.sel_d2x, self.name_y: self.sel_dy}]
        hb = h[{**sel, self.name_x: self.sel_dx, self.name_y: self.sel_y}]
        hbx = h[{**sel, self.name_x: self.sel_d2x, self.name_y: self.sel_y}]

        hc = h[{self.name_x: self.sel_x, self.name_y: self.sel_dy}]

        # calculate extended ABCD method without smoothing and in a single bin in x
        a = ha.values(flow=flow)
        ax = hax.values(flow=flow)
        b = hb.values(flow=flow)
        bx = hbx.values(flow=flow)
        c = hc.values(flow=flow)

        frf = (b / a) ** 2 * (ax / bx)
        slices = [slice(None) if a in ha.axes else np.newaxis for a in hc.axes]

        frf = frf[*slices]
        d = c * frf

        dvar = None
        if h.storage_type == hist.storage.Weight:
            avar = ha.variances(flow=flow)
            axvar = hax.variances(flow=flow)
            bvar = hb.variances(flow=flow)
            bxvar = hbx.variances(flow=flow)
            cvar = hc.variances(flow=flow)
            dvar = (
                frf**2 * cvar
                + d**2
                * (
                    4 * bvar / b**2 + 4 * avar / a**2 + axvar / ax**2 + bxvar / bx**2
                )[*slices]
            )

        return d, dvar

    def compute_fakeratefactor(
        self, h, smoothing=False, syst_variations=False, flow=True
    ):
        # rebin in smoothing axis to have stable ratios
        sel = {n: hist.sum for n in self.fakerate_integration_axes}
        hNew = (
            hh.rebinHist(h[sel], self.smoothing_axis_name, self.rebin_smoothing_axis)
            if self.rebin_smoothing_axis is not None
            else h[sel]
        )

        # select sideband regions
        ha = hNew[{self.name_x: self.sel_dx, self.name_y: self.sel_dy}]
        hb = hNew[{self.name_x: self.sel_dx, self.name_y: self.sel_y}]
        hbx = hNew[{self.name_x: self.sel_d2x, self.name_y: self.sel_y}]

        if not self.integrate_x and self.swap_regions:
            logger.warning(
                f"Regions for fakerate estiamation can only be swapped if abcd-x axis is integrated"
            )
        if self.swap_regions and self.integrate_x:
            # replace Ax region with C region
            hax = self.get_hist_passX_failY(hNew)
        else:
            hax = hNew[{self.name_x: self.sel_d2x, self.name_y: self.sel_dy}]

        a = ha.values(flow=flow)
        ax = hax.values(flow=flow)
        b = hb.values(flow=flow)
        bx = hbx.values(flow=flow)

        # fakerate factor
        y_num = ax * b**2
        y_den = bx * a**2
        # fakerate factor
        y = divide_arrays(y_num, y_den, cutoff=1)

        if h.storage_type == hist.storage.Weight:
            # full variances
            avar = ha.variances(flow=flow)
            axvar = hax.variances(flow=flow)
            bvar = hb.variances(flow=flow)
            bxvar = hbx.variances(flow=flow)
            y_var = (
                b**4 / (bx**2 * a**4) * axvar
                + ax**2 * b**2 / (bx**2 * a**4) * 4 * bvar
                + y**2 * 4 * avar / a**2
                + y**2 * bxvar / bx**2
            )
            y_var[y_den <= 1] = 1e10

        if self.hCorr:
            # apply QCD MC nonclosure correction and account for variance of correction
            cval = self.hCorr.values(flow=flow)
            y *= cval
            if h.storage_type == hist.storage.Weight:
                cvar = self.hCorr.variances(flow=flow)
                y_var = y**2 * cvar + cval**2 * y_var

        if self.throw_toys:
            logger.info("Throw toys")
            # throw toys for each parameter separately
            values = np.stack([a, ax, b, bx], axis=-1)
            variances = np.stack([avar, axvar, bvar, bxvar], axis=-1)

            nsamples = 10000
            seed = 42

            if self.throw_toys == "normal":
                # throw gaussian toys
                toy_shape = [*values.shape, nsamples]
                toy_size = np.prod(toy_shape)
                np.random.seed(seed)  # For reproducibility
                toys = np.random.normal(0, 1, size=toy_size)
                toys = np.reshape(toys, toy_shape)
                toys = (
                    toys * np.sqrt(variances)[..., np.newaxis] + values[..., np.newaxis]
                )
                # toy = toys[...,0,:] / toys[...,1,:]
                toy = (toys[..., 1, :] * toys[..., 2, :] ** 2) / (
                    toys[..., 0, :] ** 2 * toys[..., 3, :]
                )
                toy_mean = np.mean(toy, axis=-1)
                toy_var = np.var(toy, ddof=1, axis=-1)
            elif self.throw_toys == "poisson":
                # throw posson toys
                toy_shape = [nsamples, *values.shape]
                rng = np.random.default_rng(seed)
                toys = rng.poisson(values, size=toy_shape)
                toy = (toys[..., 1] * toys[..., 2] ** 2) / (
                    toys[..., 0] ** 2 * toys[..., 3]
                )
                toy_mean = np.mean(toy, axis=0)
                toy_var = np.var(toy, ddof=1, axis=0)

            y = toy_mean
            y_var = toy_var

            logger.info("Done with toys")

        if smoothing:
            x = self.get_bin_centers_smoothing(
                hNew, flow=True
            )  # the bins where the smoothing is performed (can be different to the bin in h)
            y, y_var = self.smoothen(
                h,
                x,
                y,
                y_var,
                regressor=self.fakerate_regressor,
                syst_variations=syst_variations,
                flow=flow,
            )

        # broadcast abcd-x axis and application axes
        slices = [
            slice(None) if n in ha.axes.name else np.newaxis
            for n in h[{self.name_x: self.sel_x}].axes.name
            if n != self.name_y
        ]
        y = y[*slices]
        y_var = y_var[*slices] if y_var is not None else None

        return y, y_var


class FakeSelector2DExtendedABCD(FakeSelector1DExtendedABCD):
    # extended ABCD method with 8 control regions as desribed in https://arxiv.org/abs/1906.10831 equation 15
    def __init__(
        self,
        h,
        *args,
        interpolate_x=True,
        interpolation_order=2,
        rebin_x="automatic",  # can be a list of bin edges, "automatic", or None
        integrate_shapecorrection_x=False,
        smooth_shapecorrection=True,
        smoothing_order_shapecorrection=[2, 2, 2],
        **kwargs,
    ):
        super().__init__(h, *args, **kwargs)
        self.sel_d2y = None
        self.set_selections_y()
        self.set_selections_x(integrate_x=False)

        self.interpolate_x = interpolate_x

        if rebin_x == "automatic":
            edges = h.axes[self.name_x].edges
            self.rebin_x = get_rebinning(edges, self.name_x)
        else:
            self.rebin_x = rebin_x

        # shape correction, can be interpolated in the abcd x-axis in 1D, in the x-axis and smoothing axis in 2D, or in the smoothing axis integrating out the abcd x-axis in 1D
        self.integrate_shapecorrection_x = integrate_shapecorrection_x  # if the shape correction factor for the abcd x-axis should be inclusive or differential
        if self.integrate_shapecorrection_x and self.interpolate_x:
            raise RuntimeError("Can not integrate and interpolate x at the same time")
        self.smooth_shapecorrection = smooth_shapecorrection

        if not self.integrate_shapecorrection_x:
            # initialiaze shape correction regressor (can be 1D or 2D)
            mins_x = []
            maxs_x = []
            orders = []
            if self.interpolate_x:
                # min and max (in application region) for transformation of bernstain polynomials into interval [0,1]
                axis_x_min = h[{self.name_x: self.sel_x}].axes[self.name_x].edges[0]
                if self.name_x == "mt":
                    # mt does not have an upper bound, cap at 100
                    edges = (
                        self.rebin_x
                        if self.rebin_x is not None
                        else h.axes[self.name_x].edges
                    )
                    axis_x_max = extend_edges(h.axes[self.name_x].traits, edges)[-1]
                elif self.name_x in ["iso", "relIso", "relJetLeptonDiff", "dxy"]:
                    # iso and dxy have a finite lower and upper bound in the application region
                    axis_x_max = abcd_thresholds[self.name_x][1]
                else:
                    axis_x_max = self.axis_x.edges[-1]
                mins_x.append(axis_x_min)
                maxs_x.append(axis_x_max)
                orders.append(interpolation_order)
            if self.smooth_shapecorrection:
                mins_x.append(self.smoothing_axis_min)
                maxs_x.append(self.smoothing_axis_max)
                orders.append(smoothing_order_shapecorrection)

            if len(smoothing_order_shapecorrection) > 1:
                self.shapecorrection_regressor = Regressor2D(
                    "bernstein",
                    orders,
                    min_x=mins_x,
                    max_x=maxs_x,
                )
            elif len(smoothing_order_shapecorrection) == 1:
                self.shapecorrection_regressor = Regressor(
                    "bernstein",
                    orders[0],
                    min_x=mins_x[0],
                    max_x=maxs_x[0],
                )
            else:
                self.shapecorrection_regressor = None

    # set slices object for selection of sideband regions
    def set_selections_y(self):
        y0, y1, y2, y3 = get_selection_edges(
            self.name_y, upper_bound=self.upper_bound_y
        )
        s = hist.tag.Slicer()
        self.sel_y = (
            s[y0 : y1 : hist.sum]
            if y0 is not None and y1.imag > y0.imag
            else s[y1 : y0 : hist.sum]
        )
        self.sel_dy = (
            s[y1 : y2 : hist.sum] if y2.imag > y1.imag else s[y2 : y1 : hist.sum]
        )
        self.sel_d2y = (
            s[y2 : y3 : hist.sum]
            if y3 is None or y3.imag > y2.imag
            else s[y3 : y2 : hist.sum]
        )

    def get_hist(
        self,
        h,
        is_nominal=False,
        variations_scf=False,
        variations_smoothing=False,
        variations_full=False,
        flow=True,
    ):
        if variations_scf and variations_smoothing:
            raise RuntimeError(
                f"Can only calculate vairances for fakerate factor or shape correction factor but not both"
            )

        if self.smoothing_mode == "fakerate":
            h = self.transfer_variances(h, set_nominal=is_nominal)

            y_frf, y_frf_var = self.compute_fakeratefactor(
                h, smoothing=True, syst_variations=variations_smoothing
            )
            if not self.integrate_shapecorrection_x:
                y_scf, y_scf_var = self.compute_shapecorrection(
                    h, smoothing=True, syst_variations=variations_scf
                )
                y_frf = y_scf * y_frf
                y_frf_var = (
                    y_scf[..., np.newaxis, np.newaxis] * y_frf_var[..., :, :]
                    if y_frf_var is not None
                    else None
                )
                y_scf_var = (
                    y_frf[..., np.newaxis, np.newaxis] * y_scf_var[..., :, :]
                    if y_scf_var is not None
                    else None
                )

            c, cvar = self.get_yields_applicationregion(h)
            d = c * y_frf

            if variations_scf and (self.interpolate_x or self.smooth_shapecorrection):
                dvar = c[..., np.newaxis, np.newaxis] * y_scf_var[..., :, :]
            elif variations_smoothing:
                dvar = c[..., np.newaxis, np.newaxis] * y_frf_var[..., :, :]
            else:
                # only take bin by bin uncertainty from c region
                dvar = y_frf**2 * cvar

            if self.integrate_x:
                idx_x = [n for n in h.axes.name if n != self.name_y].index(self.name_x)
                d = d.sum(axis=idx_x)
                dvar = dvar.sum(axis=idx_x)
        elif self.smoothing_mode == "full":
            h = self.transfer_variances(h, set_nominal=is_nominal)
            d, dvar = self.calculate_fullABCD_smoothed(
                h, flow=flow, syst_variations=variations_smoothing, use_spline=False
            )
        elif self.smoothing_mode == "binned":
            # no smoothing of rates
            d, dvar = self.calculate_fullABCD(h)
        else:
            raise ValueError("invalid choice of smoothing mode")

        # set histogram in signal region
        axes = [
            a
            for a in h[
                {self.name_x: self.sel_x if not self.integrate_x else hist.sum}
            ].axes
            if a.name != self.name_y
        ]
        hSignal = hist.Hist(
            *axes,
            storage=hist.storage.Double() if variations_smoothing else h.storage_type(),
        )
        hSignal.values(flow=flow)[...] = d
        if variations_scf or variations_smoothing or variations_full:
            hSignal = self.get_syst_hist(hSignal, d, dvar, flow=flow)
        elif hSignal.storage_type == hist.storage.Weight:
            hSignal.variances(flow=flow)[...] = dvar

        if self.global_scalefactor != 1:
            hSignal = hh.scaleHist(hSignal, self.global_scalefactor)

        return hSignal

    def compute_shapecorrection(
        self, h, smoothing=False, syst_variations=False, apply=False, flow=True
    ):
        # if apply=True, shape correction is multiplied to application region for correct statistical uncertainty, only allowed if not smoothing
        if apply and smoothing and (self.interpolate_x or self.smooth_shapecorrection):
            raise NotImplementedError(
                f"Direct application of shapecorrection only supported when no smoothing is performed"
            )
        # rebin in smoothing axis to have stable ratios
        sel = {n: hist.sum for n in self.fakerate_integration_axes}
        hNew = (
            hh.rebinHist(h[sel], self.smoothing_axis_name, self.rebin_smoothing_axis)
            if self.rebin_smoothing_axis is not None
            else h[sel]
        )
        if self.rebin_x is not None and (
            self.interpolate_x or self.smooth_shapecorrection
        ):
            hNew = hh.rebinHist(
                hNew, self.name_x, self.rebin_x
            )  # only rebin for regression
        hNew = hNew[{self.name_x: self.sel_x}]

        hc = hNew[{self.name_y: self.sel_dy}]
        hcy = hNew[{self.name_y: self.sel_d2y}]

        c = hc.values(flow=flow)
        cy = hcy.values(flow=flow)
        if h.storage_type == hist.storage.Weight:
            cvar = hc.variances(flow=flow)
            cyvar = hcy.variances(flow=flow)

        if self.integrate_shapecorrection_x:
            idx_x = hNew.axes.name.index(self.name_x)
            c = c.sum(axis=idx_x)
            cy = cy.sum(axis=idx_x)
            cvar = cvar.sum(axis=idx_x)
            cyvar = cyvar.sum(axis=idx_x)

        # shape correction factor
        y_num = c**2 if apply else c
        y_den = cy
        y = divide_arrays(y_num, y_den, cutoff=1)

        if h.storage_type == hist.storage.Weight:
            if apply:
                y_var = (
                    4 * c**2 / cy**2 * cvar + c**4 / cy**4 * cyvar
                )  # multiply out for better numerical stability (avoid NaNs from divisions)
            else:
                y_var = (
                    cvar / cy**2 + (c**2 * cyvar) / cy**4
                )  # multiply out for better numerical stability (avoid NaNs from divisions)
            y_var[cy <= 1] = 1e10

        if self.throw_toys:
            logger.info("Throw toys")
            # throw toys for nominator and denominator
            y_num_var = cvar
            y_den_var = cyvar
            values = np.stack([y_num, y_den], axis=-1)
            variances = np.stack([y_num_var, y_den_var], axis=-1)

            nsamples = 10000
            seed = 42

            if self.throw_toys == "normal":
                # throw gaussian toys
                toy_shape = [*values.shape, nsamples]
                toy_size = np.prod(toy_shape)
                np.random.seed(seed)  # For reproducibility
                toys = np.random.normal(0, 1, size=toy_size)
                toys = np.reshape(toys, toy_shape)
                toys = (
                    toys * np.sqrt(variances)[..., np.newaxis] + values[..., np.newaxis]
                )
                toy = toys[..., 0, :] / toys[..., 1, :]
                toy_mean = np.mean(toy, axis=-1)
                toy_var = np.var(toy, ddof=1, axis=-1)
            elif self.throw_toys == "poisson":
                # throw posson toys
                toy_shape = [nsamples, *values.shape]
                rng = np.random.default_rng(seed)
                toys = rng.poisson(values, size=toy_shape)
                toy = toys[..., 0] / toys[..., 1]
                toy_mean = np.mean(toy, axis=0)
                toy_var = np.var(toy, ddof=1, axis=0)

            y = toy_mean
            y_var = toy_var

            logger.info("Done with toys")

        if smoothing and (self.interpolate_x or self.smooth_shapecorrection):
            if h.storage_type == hist.storage.Weight:
                w = 1 / np.sqrt(y_var)
            else:
                logger.warning(
                    "Smoothing extended ABCD on histogram without uncertainties, make an unweighted linear squared solution."
                )
                w = np.ones_like(y)

        if smoothing and self.interpolate_x:
            axes = [
                n
                for n in h.axes.name
                if n not in [*self.fakerate_integration_axes, self.name_y]
            ]

            idx_ax_interpol = axes.index(self.name_x)

            # constract x points in application region
            x_interpol = self.get_bin_centers_interpolation(hNew, flow=flow)
            # x axis in original binning in passing region
            x_interpol_orig = self.get_bin_centers_interpolation(
                h[{self.name_x: self.sel_x}], flow=flow, cap=True
            )

            if self.smooth_shapecorrection:
                # interpolate scf in mT and smooth in pT (i.e. 2D)
                idx_ax_smoothing = hNew.axes.name.index(self.smoothing_axis_name)
                if (
                    idx_ax_smoothing != len(axes) - 2
                    or idx_ax_interpol != len(axes) - 1
                ):
                    y = np.moveaxis(y, (idx_ax_smoothing, idx_ax_interpol), (-2, -1))
                    w = np.moveaxis(w, (idx_ax_smoothing, idx_ax_interpol), (-2, -1))

                x_smoothing = self.get_bin_centers_smoothing(hNew, flow=True)
                self.shapecorrection_regressor.solve(
                    x_interpol, x_smoothing, y, w, flatten=True
                )

                x_smooth_orig = self.get_bin_centers_smoothing(h, flow=True)
                y_smooth_orig = self.shapecorrection_regressor.evaluate(
                    x_interpol_orig, x_smooth_orig
                )

                if syst_variations:
                    y_smooth_var_orig = (
                        self.shapecorrection_regressor.get_eigenvector_predictions(
                            x_interpol_orig, x_smooth_orig
                        )
                    )
                else:
                    y_smooth_var_orig = None

                # move interpolation axis to original positon again
                if (
                    idx_ax_smoothing != len(axes) - 2
                    or idx_ax_interpol != len(axes) - 1
                ):
                    y_smooth_orig = np.moveaxis(
                        y_smooth_orig, (-2, -1), (idx_ax_smoothing, idx_ax_interpol)
                    )
                    y_smooth_var_orig = (
                        np.moveaxis(
                            y_smooth_var_orig,
                            (-4, -3),
                            (idx_ax_smoothing, idx_ax_interpol),
                        )
                        if syst_variations
                        else None
                    )

            else:
                # interpolate scf in mT in 1D
                if idx_ax_interpol != len(axes) - 1:
                    y = np.moveaxis(y, idx_ax_interpol, -1)
                    w = np.moveaxis(w, idx_ax_interpol, -1)

                self.shapecorrection_regressor.solve(x_interpol, y, w)

                y_smooth_orig = self.shapecorrection_regressor.evaluate(x_interpol_orig)

                if syst_variations:
                    y_smooth_var_orig = (
                        self.shapecorrection_regressor.get_eigenvector_predictions(
                            x_interpol_orig
                        )
                    )
                else:
                    y_smooth_var_orig = None

                # move interpolation axis to original positon again
                if idx_ax_interpol != len(axes) - 1:
                    y_smooth_orig = np.moveaxis(y_smooth_orig, -1, idx_ax_interpol)
                    y_smooth_var_orig = (
                        np.moveaxis(y_smooth_var_orig, -3, idx_ax_smoothing)
                        if syst_variations
                        else None
                    )

            # check for negative rates
            if np.sum(y_smooth_orig < 0) > 0:
                logger.warning(
                    f"Found {np.sum(y_smooth_orig<0)} bins with negative shape correction factors"
                )
            if y_smooth_var_orig is not None and np.sum(y_smooth_var_orig < 0) > 0:
                logger.warning(
                    f"Found {np.sum(y_smooth_var_orig<0)} bins with negative shape correction factor variations"
                )
                y_smooth_var_orig[y_smooth_var_orig < 0] = 0

            y, y_var = y_smooth_orig, y_smooth_var_orig

        elif smoothing and self.smooth_shapecorrection:
            # don't interpolate in mT, but smooth in pT in 1D

            # move smoothing axis to last
            axes = [
                n
                for n in h.axes.name
                if n not in [*self.fakerate_integration_axes, self.name_x, self.name_y]
            ]
            idx_ax_smoothing = axes.index(self.smoothing_axis_name)
            if idx_ax_smoothing != len(axes) - 1:
                y = np.moveaxis(y, idx_ax_smoothing, -1)
                w = np.moveaxis(w, idx_ax_smoothing, -1)

            # smooth scf (e.g. in pT)
            x_smoothing = self.get_bin_centers_smoothing(hNew, flow=True)
            self.shapecorrection_regressor.solve(x_smoothing, y, w)

            # evaluate in range of original histogram
            x_smooth_orig = self.get_bin_centers_smoothing(h, flow=True)
            y_smooth_orig = self.shapecorrection_regressor.evaluate(x_smooth_orig)

            if syst_variations:
                y_smooth_var_orig = (
                    self.shapecorrection_regressor.get_eigenvector_predictions(
                        x_smooth_orig
                    )
                )
            else:
                y_smooth_var_orig = None

            # move smoothing axis to original positon again
            if idx_ax_smoothing != len(axes) - 1:
                y_smooth_orig = np.moveaxis(y_smooth_orig, -1, idx_ax_smoothing)
                y_smooth_var_orig = (
                    np.moveaxis(y_smooth_var_orig, -3, idx_ax_smoothing)
                    if syst_variations
                    else None
                )

            # check for negative rates
            if np.sum(y_smooth_orig < 0) > 0:
                logger.warning(
                    f"Found {np.sum(y_smooth_orig<0)} bins with negative shape correction factors"
                )
            if y_smooth_var_orig is not None and np.sum(y_smooth_var_orig < 0) > 0:
                logger.warning(
                    f"Found {np.sum(y_smooth_var_orig<0)} bins with negative shape correction factor variations"
                )

            y, y_var = y_smooth_orig, y_smooth_var_orig

        # broadcast abcd-x axis and application axes
        if self.integrate_shapecorrection_x:
            slices = [
                np.newaxis if n == self.name_x or n not in hc.axes.name else slice(None)
                for n in h.axes.name
                if n not in [self.name_y]
            ]
        else:
            slices = [
                slice(None) if n in hc.axes.name else np.newaxis
                for n in h.axes.name
                if n not in [self.name_x, self.name_y]
            ]

        y = y[*slices]
        y_var = y_var[*slices] if y_var is not None else None

        return y, y_var

    def compute_fakeratefactor(
        self, h, smoothing=False, syst_variations=False, flow=True
    ):
        # rebin in smoothing axis to have stable ratios
        sel = {n: hist.sum for n in self.fakerate_integration_axes}
        hNew = (
            hh.rebinHist(h[sel], self.smoothing_axis_name, self.rebin_smoothing_axis)
            if self.rebin_smoothing_axis is not None
            else h[sel]
        )

        # select sideband regions
        ha = hNew[{self.name_x: self.sel_dx, self.name_y: self.sel_dy}]
        hax = hNew[{self.name_x: self.sel_d2x, self.name_y: self.sel_dy}]
        hay = hNew[{self.name_x: self.sel_dx, self.name_y: self.sel_d2y}]
        haxy = hNew[{self.name_x: self.sel_d2x, self.name_y: self.sel_d2y}]
        hb = hNew[{self.name_x: self.sel_dx, self.name_y: self.sel_y}]
        hbx = hNew[{self.name_x: self.sel_d2x, self.name_y: self.sel_y}]

        a = ha.values(flow=flow)
        ax = hax.values(flow=flow)
        ay = hay.values(flow=flow)
        axy = haxy.values(flow=flow)
        b = hb.values(flow=flow)
        bx = hbx.values(flow=flow)

        # fakerate factor
        y_num = (ax * ay * b) ** 2
        y_den = a**4 * axy * bx

        if h.storage_type == hist.storage.Weight:
            # full variances
            avar = ha.variances(flow=flow)
            axvar = hax.variances(flow=flow)
            ayvar = hay.variances(flow=flow)
            axyvar = haxy.variances(flow=flow)
            bvar = hb.variances(flow=flow)
            bxvar = hbx.variances(flow=flow)
            yvarrel = (
                4 * axvar / ax**2
                + 4 * ayvar / ay**2
                + axyvar / axy**2
                + 4 * bvar / b**2
                + 16 * avar / a**2
                + bxvar / bx**2
            )

        if self.integrate_shapecorrection_x and smoothing:
            # in case we integrate abcd-x axis we can multiply the fakerate factor and the shape correction factor and smooth once
            hc = hNew[{self.name_x: self.sel_x, self.name_y: self.sel_dy}]
            hcy = hNew[{self.name_x: self.sel_x, self.name_y: self.sel_d2y}]

            idx_x = hNew.axes.name.index(self.name_x)
            c = hc.values(flow=flow).sum(axis=idx_x)
            cy = hcy.values(flow=flow).sum(axis=idx_x)

            # shape correction factor
            y_num *= c
            y_den *= cy

            if h.storage_type == hist.storage.Weight:
                cvar = hc.variances(flow=flow).sum(axis=idx_x)
                cyvar = hcy.variances(flow=flow).sum(axis=idx_x)

                yvarrel += cvar / c**2 + cyvar / cy**2

        y = divide_arrays(y_num, y_den, cutoff=1)

        if h.storage_type == hist.storage.Weight:
            y_var = y**2 * yvarrel
            y_var[y_den <= 1] = 1e10
        else:
            y_var = None

        if self.hCorr:
            y, y_var = self.apply_correction(y, y_var)

        if self.throw_toys:
            logger.info("Throw toys")
            # throw toys for each parameter separately
            values = np.stack([a, ax, ay, axy, b, bx], axis=-1)
            variances = np.stack([avar, axvar, ayvar, axyvar, bvar, bxvar], axis=-1)

            nsamples = 10000
            seed = 42  # For reproducibility

            if self.throw_toys == "normal":
                # throw gaussian toys
                toy_shape = [*values.shape, nsamples]
                toy_size = np.prod(toy_shape)
                np.random.seed(seed)
                toys = np.random.normal(0, 1, size=toy_size)
                toys = np.reshape(toys, toy_shape)
                toys = (
                    toys * np.sqrt(variances)[..., np.newaxis] + values[..., np.newaxis]
                )
                toy = (toys[..., 1, :] * toys[..., 2, :] * toys[..., 4, :]) ** 2 / (
                    toys[..., 0, :] ** 4 * toys[..., 3, :] * toys[..., 5, :]
                )
                toy_mean = np.mean(toy, axis=-1)
                toy_var = np.var(toy, ddof=1, axis=-1)
            elif self.throw_toys == "poisson":
                # throw posson toys
                toy_shape = [nsamples, *values.shape]
                rng = np.random.default_rng(seed)
                toys = rng.poisson(values, size=toy_shape)
                toys = toys.astype(np.double)
                toy = (toys[..., 1] * toys[..., 2] * toys[..., 4]) ** 2 / (
                    toys[..., 0] ** 4 * toys[..., 3] * toys[..., 5]
                )
                toy_mean = np.mean(toy, axis=0)
                toy_var = np.var(toy, ddof=1, axis=0)

            y = toy_mean
            y_var = toy_var

            logger.info("Done with toys")

        if smoothing:
            x = self.get_bin_centers_smoothing(
                hNew, flow=True
            )  # the bins where the smoothing is performed (can be different to the bin in h)
            y, y_var = self.smoothen(
                h,
                x,
                y,
                y_var,
                regressor=self.fakerate_regressor,
                syst_variations=syst_variations,
            )

        # broadcast abcd-x axis and application axes
        slices = [
            slice(None) if n in ha.axes.name else np.newaxis
            for n in h[{self.name_x: self.sel_x}].axes.name
            if n != self.name_y
        ]
        y = y[*slices]
        y_var = y_var[*slices] if y_var is not None else None

        return y, y_var

    def get_bin_centers_interpolation(self, h, flow=True, cap=False):
        return self.get_bin_centers(h, self.name_x, flow=flow)

    def calculate_fullABCD(self, h, flow=True):
        if len(self.fakerate_integration_axes) > 0:
            logger.warning(
                f"Binned fake estimation is performed but fakerate integration axes {self.fakerate_integration_axes} are set, the bin-by-bin stat uncertainties are not correct along this axis."
            )
        if not self.integrate_x:
            logger.warning(
                f"Binned fake estimation is performed but ABCD x-axis is not integrated, the bin-by-bin stat uncertainties are not correct along this axis."
            )

        frf, frf_var = self.compute_fakeratefactor(h, smoothing=False)
        c_scf, c_scf_var = self.compute_shapecorrection(h, smoothing=False, apply=True)

        d = frf * c_scf

        if self.integrate_x:
            idx_x = [n for n in h.axes.name if n != self.name_y].index(self.name_x)
            d = d.sum(axis=idx_x)

            if h.storage_type == hist.storage.Weight:
                # the fakerate factor is independent of the abcd x-axis and treated fully correlated
                dvar = c_scf * frf_var**0.5
                dvar = np.moveaxis(dvar, idx_x, -1)
                dvar = np.einsum("...ni,...nj->...n", dvar, dvar)
                dvar += (frf**2 * c_scf_var).sum(axis=idx_x)

        elif h.storage_type == hist.storage.Weight:
            dvar = c_scf**2 * frf_var + frf**2 * c_scf_var

        return d, dvar
