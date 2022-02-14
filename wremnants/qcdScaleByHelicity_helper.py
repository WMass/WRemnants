import uproot
import ROOT
import pathlib
import hist
import narf
from .correctionsTensor_helper import makeCorrectionsTensor

data_dir = f"{pathlib.Path(__file__).parent}/data/"

def makeQCDScaleByHelicityHelper(input_path=f"{data_dir}/angularCoefficients"):
    charge_axis = hist.axis.Regular(
        2, -2, 2, name="chargeVgen", underflow=False, overflow=False
    )
    # this puts the bin centers at 0.5, 1.0, 2.0
    mur_axis = hist.axis.Variable(
        [0.25, 0.75, 1.25, 2.75], name="muRfact", underflow=False, overflow=False
    )
    muf_axis = hist.axis.Variable(
        [0.25, 0.75, 1.25, 2.75], name="muFfact", underflow=False, overflow=False
    )
    coeff_axis = hist.axis.Integer(
        0, 9, name="coeff", overflow=False, underflow=False
    )
    f = uproot.open(f"{input_path}/fractions_minus_2022.root")
    nom = f["unpol_minus_nominal"].to_hist()
    axes = [hist.axis.Variable(ax.edges, name=name) for ax,name in zip(nom.axes, ["yVgen", "ptVgen"])]
    corrh = hist.Hist(*axes, charge_axis, mur_axis, muf_axis, coeff_axis)

    for i,charge in enumerate(["minus", "plus"]):
        f = uproot.open(f"{input_path}/fractions_{charge}_2022.root")
        for k,coeff in enumerate(["unpol"] + [f"a{m}" for m in range(8)]):
            corrh.view(flow=True)[...,i,1,1,k] = f[f"{coeff}_{charge}_nominal"].to_hist().values(flow=True)
            for j,muR in enumerate(["muRDown", "", "muRUp"]):
                for l,muF in enumerate(["muFDown", "", "muFUp"]):
                    if muR != "" and muF == "":
                        name = muR
                    elif muF != "" and muR == "":
                        name = muF
                    elif "Up" in muR and "Up" in muF:
                        name = "muRmuFUp"
                    elif "Down" in muR and "Down" in muF:
                        name = "muRmuFDown"
                    else:
                        continue

                    rthist = f[f"{coeff}_{charge}_{name}"].to_hist()
                    corrh.view(flow=True)[...,i,j,l,k] = rthist.values(flow=True)

    #should be uniformly 1.0 for 1+cos^2 theta term
    corrh.values(flow=True)[...,0] = 1.0
    corrh.variances(flow=True)[...,0] = 0.0

    return makeCorrectionsTensor(corrh, ROOT.wrem.QCDScaleByHelicityCorrectionsHelper, tensor_rank=3)
