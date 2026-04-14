import copy
import pprint
import re

from rabbit.tensorwriter import TensorWriter
from wums import boostHistHelpers as hh
from wums import logging


class SigmaULTheoryFitWriter(TensorWriter):
    """Tensor writer for the direct-theory sigmaUL fit."""

    def __init__(
        self,
        exclude_nuisances="",
        keep_nuisances="",
        process_name="Zmumu",
        sigmaul_channel="chSigmaUL",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.logger = logging.child_logger(__name__)
        self.ref = {}
        self.process_name = process_name
        self.sigmaul_channel = sigmaul_channel
        self._exclude_nuisances = (
            re.compile(exclude_nuisances) if exclude_nuisances else None
        )
        self._keep_nuisances = re.compile(keep_nuisances) if keep_nuisances else None

    def _keep_systematic(self, name):
        if self._exclude_nuisances and self._exclude_nuisances.search(name):
            return False
        if self._keep_nuisances and not self._keep_nuisances.search(name):
            return False
        return True

    def set_reference(self, channel, h, lumi=1.0, scale=1.0, postOp=None):
        self.ref[channel] = {
            "h": h,
            "lumi": lumi,
            "scale": scale,
            "postOp": postOp,
            "ptV_name": self.get_ptV_axis_name(h),
            "absYV_name": self.get_absYV_axis_name(h),
            "chargeV_name": self.get_charge_axis_name(h),
            "ptV_bins": h.axes[self.get_ptV_axis_name(h)].edges,
            "absYV_bins": h.axes[self.get_absYV_axis_name(h)].edges,
        }
        self.logger.debug("Initialized channel %s with parameters", channel)
        self.logger.debug(pprint.pformat(self.ref[channel]))

    def add_systematic(
        self,
        h,
        name,
        process,
        channel,
        rebin_pt=True,
        rebin_y=True,
        normalize=True,
        apply_postOp=True,
        format=True,
        **kwargs,
    ):
        if not self._keep_systematic(name):
            self.logger.info(
                "Skipping systematic '%s' for process '%s' in channel '%s' due to nuisance filtering.",
                name,
                process,
                channel,
            )
            return

        if format:
            if isinstance(h, (list, tuple)):
                h[0] = self.format(
                    h[0],
                    channel,
                    process,
                    rebin_pt=rebin_pt,
                    rebin_y=rebin_y,
                    normalize=normalize,
                    apply_postOp=apply_postOp,
                )
                h[1] = self.format(
                    h[1],
                    channel,
                    process,
                    rebin_pt=rebin_pt,
                    rebin_y=rebin_y,
                    normalize=normalize,
                    apply_postOp=apply_postOp,
                )
            elif kwargs.get("mirror"):
                h = self.format(
                    h,
                    channel,
                    process,
                    rebin_pt=rebin_pt,
                    rebin_y=rebin_y,
                    normalize=normalize,
                    apply_postOp=apply_postOp,
                )

        super().add_systematic(h, name, process, channel, **kwargs)

    def add_shape_systematic(
        self,
        h,
        name,
        process,
        channel,
        rebin_pt=True,
        rebin_y=True,
        normalize=False,
        apply_postOp=True,
        format=True,
        **kwargs,
    ):
        if not self._keep_systematic(name):
            self.logger.info(
                "Skipping systematic '%s' for process '%s' in channel '%s' due to nuisance filtering.",
                name,
                process,
                channel,
            )
            return

        if not kwargs.get("mirror"):
            if format:
                h[0] = self.format(
                    h[0],
                    channel,
                    process,
                    rebin_pt=rebin_pt,
                    rebin_y=rebin_y,
                    normalize=normalize,
                    apply_postOp=apply_postOp,
                )
                h[1] = self.format(
                    h[1],
                    channel,
                    process,
                    rebin_pt=rebin_pt,
                    rebin_y=rebin_y,
                    normalize=normalize,
                    apply_postOp=apply_postOp,
                )
                h[2] = self.format(
                    h[2],
                    channel,
                    process,
                    rebin_pt=rebin_pt,
                    rebin_y=rebin_y,
                    normalize=normalize,
                    apply_postOp=apply_postOp,
                )

            hup = hh.divideHists(h[0], h[2])
            hdown = hh.divideHists(h[1], h[2])
            hup = hh.multiplyHists(hup, self.ref[channel][process])
            hdown = hh.multiplyHists(hdown, self.ref[channel][process])
            super().add_systematic([hup, hdown], name, process, channel, **kwargs)
        else:
            if format:
                h[0] = self.format(
                    h[0],
                    channel,
                    process,
                    rebin_pt=rebin_pt,
                    rebin_y=rebin_y,
                    normalize=normalize,
                    apply_postOp=apply_postOp,
                )
                h[1] = self.format(
                    h[1],
                    channel,
                    process,
                    rebin_pt=rebin_pt,
                    rebin_y=rebin_y,
                    normalize=normalize,
                    apply_postOp=apply_postOp,
                )

            mirrored = hh.divideHists(h[0], h[1])
            mirrored = hh.multiplyHists(mirrored, self.ref[channel][process])
            super().add_systematic(mirrored, name, process, channel, **kwargs)

    def add_process(
        self,
        h,
        name,
        channel,
        rebin_pt=True,
        rebin_y=True,
        normalize=True,
        apply_postOp=True,
        **kwargs,
    ):
        h = self.format(
            h,
            channel,
            name,
            rebin_pt=rebin_pt,
            rebin_y=rebin_y,
            normalize=normalize,
            apply_postOp=apply_postOp,
        )
        super().add_process(h, name, channel, **kwargs)
        self.ref[channel][name] = h

    def format(
        self,
        h,
        channel,
        process,
        rebin_pt=True,
        rebin_y=True,
        normalize=True,
        apply_postOp=True,
    ):
        h = copy.deepcopy(h)
        h = self.apply_selections(h, process, channel)

        pt_axis_name = self.get_ptV_axis_name(h)
        absY_axis_name = self.get_absYV_axis_name(h)
        charge_axis_name = self.get_charge_axis_name(h)

        hh.renameAxis(h, pt_axis_name, self.ref[channel]["ptV_name"])
        hh.renameAxis(h, absY_axis_name, self.ref[channel]["absYV_name"])
        if charge_axis_name:
            hh.renameAxis(h, charge_axis_name, self.ref[channel]["chargeV_name"])

        h = hh.setFlow(h, self.ref[channel]["ptV_name"], under=False, over=True)

        if rebin_pt:
            h = hh.rebinHist(
                h, self.ref[channel]["ptV_name"], self.ref[channel]["ptV_bins"]
            )
        if rebin_y:
            h = hh.rebinHist(
                h, self.ref[channel]["absYV_name"], self.ref[channel]["absYV_bins"]
            )
        if normalize:
            h *= self.ref[channel]["lumi"] * self.ref[channel]["scale"]

        remaining_axes = list(h.axes.name)
        remaining_axes.remove(self.ref[channel]["ptV_name"])
        remaining_axes.remove(self.ref[channel]["absYV_name"])
        h = h.project(
            self.ref[channel]["ptV_name"],
            self.ref[channel]["absYV_name"],
            *remaining_axes,
        )

        if self.ref[channel]["postOp"] is not None and apply_postOp:
            h = self.ref[channel]["postOp"](h)

        return h

    def get_ptV_axis_name(self, h):
        for name in ["ptVgen", "ptVGen", "qT"]:
            if name in h.axes.name:
                return name
        self.logger.debug("Did not find pT axis. Available axes: %s", h.axes.name)

    def get_absYV_axis_name(self, h):
        for name in ["absYVgen", "absYVGen", "absY"]:
            if name in h.axes.name:
                return name
        self.logger.debug("Did not find absY axis. Available axes: %s", h.axes.name)

    def get_charge_axis_name(self, h):
        for name in ["chargeVgen", "charge", "qGen"]:
            if name in h.axes.name:
                return name
        return None

    def get_mass_axis_name(self, h):
        for name in ["massVgen", "Q"]:
            if name in h.axes.name:
                return name
        return None

    def apply_selections(self, h, process, channel):
        if process != self.process_name:
            raise ValueError(
                f"Unsupported process '{process}' for sigmaUL writer; expected '{self.process_name}'"
            )

        mass_axis_name = self.get_mass_axis_name(h)
        if mass_axis_name:
            h = h[{mass_axis_name: 90.0j}]

        charge_axis_name = self.get_charge_axis_name(h)
        if charge_axis_name:
            h = h[{charge_axis_name: 0.0j}]

        if channel == self.sigmaul_channel and "helicity" in h.axes.name:
            h = h[{"helicity": -1.0j}]

        return h
