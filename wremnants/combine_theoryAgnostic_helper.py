import h5py
import hist

import wums.ioutils
from utilities import common
from wums import boostHistHelpers as hh
from wums import logging

logger = logging.child_logger(__name__)


class TheoryAgnosticHelper(object):
    def __init__(self, datagroups, externalArgs=None):
        self.datagroups = datagroups
        toCheck = ["signal_samples", "signal_samples_noOutAcc"]
        for group in toCheck:
            if group not in self.datagroups.procGroups:
                raise ValueError(
                    f"Must define '{group}' procGroup in datagroups for theory agnostic fit"
                )
        self.args = externalArgs
        self.label = "Label"
        self.passSystToFakes = True
        self.separateOutOfAccSignal = False
        self.poi_axes = []

    def configure_polVar(
        self, label, passSystToFakes, hasSeparateOutOfAcceptanceSignal
    ):
        self.label = label
        self.passSystToFakes = passSystToFakes
        self.separateOutOfAccSignal = hasSeparateOutOfAcceptanceSignal

    def configure_normVar(self, label, passSystToFakes, poi_axes):
        self.label = label
        self.passSystToFakes = passSystToFakes
        self.poi_axes = poi_axes

    def add_theoryAgnostic_polVar_uncertainty(self):
        coeffs = ["UL"] + [f"A{i}" for i in range(5)]
        groupName = f"polVar{self.label}"
        for genVcharge in ["minus", "plus"]:
            for coeffKey in coeffs:
                self.datagroups.addSystematic(
                    f"theoryAgnosticWithPol_{coeffKey}_{genVcharge}",
                    group=groupName,
                    mirror=False,
                    symmetrize="average" if self.args.symmetrizePolVar else None,
                    passToFakes=(
                        False if self.args.noPolVarOnFake else self.passSystToFakes
                    ),
                    processes=[
                        (
                            "signal_samples_noOutAcc"
                            if self.separateOutOfAccSignal
                            else "signal_samples"
                        )
                    ],
                    baseName=f"{groupName}_{coeffKey}_{genVcharge}_",
                    noConstraint=True,
                    systAxes=["nPolVarSyst", "downUpVar"],
                    labelsByAxis=["v", "downUpVar"],
                    # splitGroup={f"{groupName}_{coeffKey}" : f"{groupName}_{coeffKey}"}
                )

    def add_muRmuF_polVar_uncertainty(self):
        coeffs = [f"A{i}" for i in range(8)]
        signalGroupName = "muRmuFPolVarW" if self.label == "W" else "muRmuFPolVarZ"
        nonSignalGroupName = "muRmuFPolVarZ" if self.label == "W" else "muRmuFPolVarW"
        for coeffKey in coeffs:
            self.datagroups.addSystematic(
                f"{signalGroupName}_{coeffKey}",
                group=signalGroupName,
                mirror=False,
                passToFakes=self.passSystToFakes,
                processes=[
                    (
                        "signal_samples_inctau_noOutAcc"
                        if self.separateOutOfAccSignal
                        else "signal_samples_inctau"
                    )
                ],
                baseName=f"{signalGroupName}_{coeffKey}_",
                noConstraint=False,
                systAxes=["nPolVarSyst", "downUpVar"],
                labelsByAxis=["v", "downUpVar"],
                # splitGroup={f"{groupName}_{coeffKey}" : f"{groupName}_{coeffKey}"}
            )

        for coeffKey in coeffs:
            self.datagroups.addSystematic(
                f"{nonSignalGroupName}_{coeffKey}",
                group=nonSignalGroupName,
                mirror=False,
                passToFakes=self.passSystToFakes,
                processes=[
                    (
                        "nonsignal_samples_inctau_noOutAcc"
                        if self.separateOutOfAccSignal
                        else "nonsignal_samples_inctau"
                    )
                ],
                baseName=f"{nonSignalGroupName}_{coeffKey}_",
                noConstraint=False,
                systAxes=["nPolVarSyst", "downUpVar"],
                labelsByAxis=["v", "downUpVar"],
                # splitGroup={f"{groupName}_{coeffKey}" : f"{groupName}_{coeffKey}"}
            )

    def apply_theoryAgnostic_normVar_uncertainty(
        self, scale_hists, sign, sum_axes=[], rebin_axes=[], helicities=[], scale=1.0
    ):

        def apply_transformations(h, scale_hist):
            sum2nom = {ax: hist.tag.Slicer()[:: hist.sum] for ax in self.poi_axes}
            nom_hist = h[sum2nom]

            values = scale_hist.values()

            # rescale if necessary
            for helicity in helicities:
                values[:-1, :-1, helicity + 1] = (
                    values[:-1, :-1, helicity + 1] * scale
                )  # don't rescale OOA
            scale_hist.values()[...] = values

            scaled_hist = hh.multiplyHists(h, scale_hist, flow=True)
            scaled_hist = scaled_hist[
                {ax: hist.tag.Slicer()[:: hist.sum] for ax in sum_axes}
            ]

            for rebin_axis in rebin_axes:
                edges = [
                    scaled_hist.axes[rebin_axis].edges[0],
                    scaled_hist.axes[rebin_axis].edges[-1],
                ]
                scaled_hist = hh.rebinHist(scaled_hist, rebin_axis, edges)
                # scaled_hist = hh.scaleHist(scaled_hist, 1./len(scaled_hist.axes[rebin_axis].edges))

            summed_hist = hh.addHists(nom_hist, scaled_hist)

            return summed_hist

        def slice_histogram(h):
            transformed_hists = {
                ax: hist.tag.Slicer()[:: hist.sum] for ax in self.poi_axes
            }
            return h[transformed_hists]

        result = {}

        for g in self.datagroups.procGroups["signal_samples"]:
            if sign != "":
                if sign is not None:
                    for m in self.datagroups.groups[g].members:
                        if sign in m.name:
                            scale_hist = scale_hists[m.name]
                            result[m.name] = (
                                lambda h, scale_hist=scale_hist: apply_transformations(
                                    h, scale_hist
                                )
                            )
                        else:
                            result[m.name] = lambda h: slice_histogram(h)
                else:
                    scale_hist = scale_hists["WplusmunuPostVFP"]
                    result["WplusmunuPostVFP"] = (
                        lambda h, scale_hist=scale_hist: apply_transformations(
                            h, scale_hist
                        )
                    )
                    scale_hist = scale_hists["WminusmunuPostVFP"]
                    result["WminusmunuPostVFP"] = (
                        lambda h, scale_hist=scale_hist: apply_transformations(
                            h, scale_hist
                        )
                    )
            else:
                scale_hist = scale_hists["ZmumuPostVFP"]
                result["ZmumuPostVFP"] = (
                    lambda h, scale_hist=scale_hist: apply_transformations(
                        h, scale_hist
                    )
                )
        return result

    def add_theoryAgnostic_normVar_uncertainty(self, flow=True):

        common_noi_args = dict(
            group=f"normXsec{self.label}", passToFakes=self.passSystToFakes
        )

        # open file with theory bands
        with h5py.File(
            f"{common.data_dir}/angularCoefficients/theoryband_variations_corr.hdf5",
            "r",
        ) as ff:
            scale_hists = wums.ioutils.pickle_load_h5py(ff["theorybands"])

        # First do in acceptance bins, then OOA later (for OOA we need to group bins into macro regions)
        nuisanceBaseName = f"norm{self.label}"
        if self.label == "Z":
            sign_list = [""]
        else:
            sign_list = ["plus", "minus"]
        for sign in sign_list:
            self.datagroups.addSystematic(
                "yieldsTheoryAgnostic",
                name=f"{nuisanceBaseName}{sign}",
                **common_noi_args,
                mirror=True,
                symmetrize=None,
                systAxes=self.poi_axes,
                processes=["signal_samples"],
                baseName=f"{nuisanceBaseName}{sign}_",
                noConstraint=True if self.args.priorNormXsec < 0 else False,
                scale=1,
                formatWithValue=[None, None, "low"],
                labelsByAxis=["PtV", "YVBin", "Helicity"],
                systAxesFlow=[
                    "ptVgenSig",
                    "absYVgenSig",
                ],  # only bins in acceptance in this call
                skipEntries=[
                    {"helicitySig": [6, 7, 8]}
                ],  # removing last three indices out of 9 (0,1,...,7,8) corresponding to A5,6,7
                splitGroup={
                    f"{nuisanceBaseName}_Helicity{ihel}": f".*{nuisanceBaseName}{sign}.*Helicity{ihel}"
                    for ihel in [-1, 0, 1, 2, 3, 4]
                },
                # splitGroup={f"{nuisanceBaseName}{sign}" : f".*{nuisanceBaseName}{sign}"},
                preOpMap=self.apply_theoryAgnostic_normVar_uncertainty(
                    scale_hists,
                    sign,
                    helicities=self.args.helicitiesToInflate,
                    scale=self.args.theoryAgnosticBandSize,
                ),
            ),
            if sign == "":  # only for Z
                self.datagroups.addSystematic(
                    "yieldsTheoryAgnostic",
                    name=f"{nuisanceBaseName}{sign}CorrAll",
                    **common_noi_args,
                    mirror=True,
                    symmetrize=None,
                    systAxes=["ptVgenSig", "absYVgenSig"],
                    processes=["signal_samples"],
                    baseName=f"{nuisanceBaseName}{sign}CorrAll_",
                    noConstraint=True if self.args.priorNormXsec < 0 else False,
                    scale=1,
                    formatWithValue=[None, None, "low"],
                    # customizeNuisanceAttributes={".*AngCoeff4" : {"scale" : 1, "shapeType": "shapeNoConstraint"}},
                    labelsByAxis=["PtV", "YVBin", "Helicity"],
                    systAxesFlow=[],  # only bins in acceptance in this call
                    skipEntries=[],  # removing last three indices out of 9 (0,1,...,7,8) corresponding to A5,6,7
                    splitGroup={
                        f"{nuisanceBaseName}_Helicity{ihel}": f".*{nuisanceBaseName}{sign}.*Helicity{ihel}"
                        for ihel in [-1, 0, 1, 2, 3, 4]
                    },
                    # splitGroup={f"{nuisanceBaseName}{sign}CorrAll" : f".*{nuisanceBaseName}{sign}CorrAll"},
                    preOpMap=self.apply_theoryAgnostic_normVar_uncertainty(
                        scale_hists,
                        sign=sign,
                        helicities=self.args.helicitiesToInflate,
                        scale=self.args.theoryAgnosticBandSize,
                        rebin_axes=["ptVgenSig", "absYVgenSig"],
                        sum_axes=["helicitySig"],
                    ),
                ),

        if not sign_list == [""]:
            self.datagroups.addSystematic(
                "yieldsTheoryAgnostic",
                name=f"{nuisanceBaseName}CorrAllQ",
                **common_noi_args,
                mirror=True,
                symmetrize=None,
                systAxes=["ptVgenSig", "absYVgenSig"],
                processes=["signal_samples"],
                baseName=f"{nuisanceBaseName}CorrAllQ_",
                noConstraint=True if self.args.priorNormXsec < 0 else False,
                scale=1.0,
                formatWithValue=[None, None, "low"],
                # customizeNuisanceAttributes={".*AngCoeff4" : {"scale" : 1, "shapeType": "shapeNoConstraint"}},
                labelsByAxis=["PtV", "YVBin", "Helicity"],
                systAxesFlow=[],  # only bins in acceptance in this call
                skipEntries=[],  # removing last three indices out of 9 (0,1,...,7,8) corresponding to A5,6,7
                splitGroup={
                    f"{nuisanceBaseName}_Helicity{ihel}": f".*{nuisanceBaseName}.*Helicity{ihel}"
                    for ihel in [-1, 0, 1, 2, 3, 4]
                },
                # splitGroup={f"{nuisanceBaseName}CorrAllQ" : f".*{nuisanceBaseName}CorrAllQ"},
                preOpMap=self.apply_theoryAgnostic_normVar_uncertainty(
                    scale_hists,
                    sign=None,
                    helicities=self.args.helicitiesToInflate,
                    scale=self.args.theoryAgnosticBandSize,
                    rebin_axes=["ptVgenSig", "absYVgenSig"],
                    sum_axes=["helicitySig"],
                ),
            ),

    def add_theoryAgnostic_uncertainty(self):
        if self.args.analysisMode == "theoryAgnosticPolVar":
            self.add_theoryAgnostic_polVar_uncertainty()
        elif self.args.muRmuFPolVar == True:
            self.add_muRmuF_polVar_uncertainty()
        else:
            self.add_theoryAgnostic_normVar_uncertainty()
