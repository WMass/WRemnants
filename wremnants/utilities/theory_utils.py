import glob
import os
import re

from wremnants.utilities import common, samples

pdfMap = {
    "nnpdf31": {
        "name": "pdfNNPDF31",
        "branch": "LHEPdfWeight",
        "combine": "symHessian",
        "entries": 101,
        "alphas": ["LHEPdfWeight[0]", "LHEPdfWeight[101]", "LHEPdfWeight[102]"],
        "alphasRange": "002",
        "inflation_factor_wmass": 3.0,
        "inflation_factor_alphaS": 3.0,
    },
    "ct18": {
        "name": "pdfCT18",
        "branch": "LHEPdfWeightAltSet11",
        "combine": "asymHessian",
        "entries": 59,
        "alphas": [
            "LHEPdfWeightAltSet11[0]",
            "LHEPdfWeightAltSet11[59]",
            "LHEPdfWeightAltSet11[62]",
        ],
        "alphasRange": "002",
        "scale": 1 / 1.645,  # Convert from 90% CL to 68%
        "inflation_factor_wmass": 1.0,
        "inflation_factor_alphaS": 1.2,
    },
    "nnpdf30": {
        "name": "pdfNNPDF30",
        "branch": "LHEPdfWeightAltSet7",
        "combine": "symHessian",
        "entries": 101,
        "alphas": [
            "LHEPdfWeightAltSet13[0]",
            "LHEPdfWeightAltSet15[0]",
            "LHEPdfWeightAltSet16[0]",
        ],
        "alphasRange": "001",
        "inflation_factor_wmass": 1.0,  # not determined
        "inflation_factor_alphaS": 1.0,
    },
    "nnpdf40": {
        "name": "pdfNNPDF40",
        "branch": "LHEPdfWeightAltSet3",
        "combine": "symHessian",
        "entries": 51,
        "alphas": [
            "LHEPdfWeightAltSet3[0]",
            "LHEPdfWeightAltSet3[51]",
            "LHEPdfWeightAltSet3[52]",
        ],
        "alphasRange": "001",
        "inflation_factor_wmass": 5.0,
        "inflation_factor_alphaS": 5.0,
    },
    "pdf4lhc21": {
        "name": "pdfPDF4LHC21",
        "branch": "LHEPdfWeightAltSet10",
        "combine": "symHessian",
        "entries": 41,
        "alphas": [
            "LHEPdfWeightAltSet10[0]",
            "LHEPdfWeightAltSet10[41]",
            "LHEPdfWeightAltSet10[42]",
        ],
        "alphasRange": "001",
        "inflation_factor_wmass": 1.0,
        "inflation_factor_alphaS": 2.0,
    },
    "msht20": {
        "name": "pdfMSHT20",
        "branch": "LHEPdfWeightAltSet12",
        "combine": "asymHessian",
        "entries": 65,
        "alphas": [
            "LHEPdfWeightAltSet12[0]",
            "LHEPdfWeightAltSet12[67]",
            "LHEPdfWeightAltSet12[70]",
        ],
        "alphasRange": "002",
        "inflation_factor_wmass": 1.5,
        "inflation_factor_alphaS": 2.2,
    },
    "msht20mcrange": {
        "name": "pdfMSHT20mcrange",
        "branch": "LHEPdfWeightAltSet12",
        "combine": "asymHessian",
        "entries": 9,
        "first_entry": 72,
    },
    "msht20mbrange": {
        "name": "pdfMSHT20mbrange",
        "branch": "LHEPdfWeightAltSet12",
        "combine": "asymHessian",
        "entries": 7,
        "first_entry": 81,
    },
    "msht20mcrange_renorm": {
        "name": "pdfMSHT20mcrange",
        "branch": "LHEPdfWeightAltSet12",
        "combine": "asymHessian",
        "entries": 9,
        "first_entry": 72,
        "renorm": True,
    },
    "msht20mbrange_renorm": {
        "name": "pdfMSHT20mbrange",
        "branch": "LHEPdfWeightAltSet12",
        "combine": "asymHessian",
        "entries": 7,
        "first_entry": 81,
        "renorm": True,
    },
    "msht20an3lo": {
        "name": "pdfMSHT20an3lo",
        "branch": "LHEPdfWeightAltSet24",
        "combine": "asymHessian",
        "entries": 105,
        "alphas": [
            "LHEPdfWeightAltSet24[0]",
            "LHEPdfWeightAltSet24[108]",
            "LHEPdfWeightAltSet24[111]",
        ],
        "alphasRange": "002",
        "inflation_factor_wmass": 1.5,
        "inflation_factor_alphaS": 1.5,
    },
    "ct18z": {
        "name": "pdfCT18Z",
        "branch": "LHEPdfWeightAltSet11",
        "combine": "asymHessian",
        "entries": 59,
        "first_entry": 63,
        "alphas": [
            "LHEPdfWeightAltSet11[63]",
            "LHEPdfWeightAltSet11[122]",
            "LHEPdfWeightAltSet11[125]",
        ],
        "alphasRange": "002",
        "scale": 1 / 1.645,  # Convert from 90% CL to 68%
        "inflation_factor_wmass": 1.0,
        "inflation_factor_alphaS": 1.0,
    },
    "atlasWZj20": {
        "name": "pdfATLASWZJ20",
        "branch": "LHEPdfWeightAltSet19",
        "combine": "asymHessian",
        "entries": 60,
        "alphas": ["LHEPdfWeight[0]", "LHEPdfWeight[41]", "LHEPdfWeight[42]"],
        "alphasRange": "002",
        "inflation_factor_wmass": 1.0,  # not determined
        "inflation_factor_alphaS": 1.0,  # not determined
    },
    "herapdf20": {
        "name": "pdfHERAPDF20",
        "branch": "LHEPdfWeightAltSet20",
        "combine": "asymHessian",
        "entries": 29,
        "alphas": [
            "LHEPdfWeightAltSet20[0]",
            "LHEPdfWeightAltSet22[0]",
            "LHEPdfWeightAltSet23[0]",
        ],  # alphas 116-120
        "alphasRange": "002",
        "inflation_factor_wmass": 4.0,
        "inflation_factor_alphaS": 3.5,
    },
    "herapdf20ext": {
        "name": "pdfHERAPDF20ext",
        "branch": "LHEPdfWeightAltSet21",
        "combine": "asymHessian",
        "entries": 14,
        "alphas": [
            "LHEPdfWeightAltSet20[0]",
            "LHEPdfWeightAltSet22[0]",
            "LHEPdfWeightAltSet23[0]",
        ],  # dummy AS
        "alphasRange": "002",
        "inflation_factor_wmass": 4.0,
        "inflation_factor_alphaS": 3.5,
    },
}


only_central_pdf_datasets = [
    "Wplusmunu_bugfix",
    "Wminusmunu_bugfix",
    "Zmumu_bugfix",
    "Zmumu_bugfix_slc7",
]


def pdf_info_map(dataset, pdfset):
    infoMap = pdfMap

    # Just ignore PDF variations for non W/Z samples
    if (
        pdfset is None
        or not (dataset[0] in ["W", "Z"] and dataset[1] not in ["W", "Z"])
        or "horace" in dataset
        or (pdfset != "nnpdf31" and dataset in only_central_pdf_datasets)
        or pdfset not in infoMap
    ):
        raise ValueError(f"Skipping PDF {pdfset} for dataset {dataset}")
    return infoMap[pdfset]


def pdfNamesSymHessian(entries, pdfset=""):
    return [f"pdf{i+1}{pdfset.replace('pdf', '')}" for i in range(entries)]


def pdfNamesAsymHessian(entries, pdfset=""):
    pdfNames = ["pdf0" + pdfset.replace("pdf", "")]
    if pdfset == "pdfHERAPDF20ext":
        entries -= 3
    pdfNames.extend(
        [
            f"pdf{int((j+2)/2)}{pdfset.replace('pdf', '')}{'Up' if j % 2 else 'Down'}"
            for j in range(entries - 1)
        ]
    )
    if pdfset == "pdfHERAPDF20ext":
        pdfNames.extend(
            [f"pdf{entries//2 + j}{pdfset.replace('pdf', '')}" for j in range(1, 4)]
        )
    return pdfNames


def valid_theory_corrections():
    corr_files = glob.glob(common.data_dir + "TheoryCorrections/*Corr*.pkl.lz4")
    matches = [
        re.match(r"(^.*)_Corr[W|Z|BSM]\.pkl\.lz4", os.path.basename(c))
        for c in corr_files
    ]
    return [m[1] for m in matches if m] + ["none"]


def valid_ew_theory_corrections():
    corr_files = glob.glob(
        common.data_dir + "TheoryCorrections/*[eE][wW]*Corr*.pkl.lz4"
    )
    matches = [
        re.match(r"(^.*)Corr[W|Z]\.pkl\.lz4", os.path.basename(c)) for c in corr_files
    ]
    return [m[1] for m in matches if m] + ["none"]


def massWeightNames(matches=None, proc="", exclude=[]):
    if isinstance(exclude, (int, float)):
        exclude = [
            exclude,
        ]
    central = 10
    nweights = 21
    names = [
        f"massShift{proc[0] if len(proc) else proc}{int(abs(central-i)*10)}MeV{'' if i == central else ('Down' if i < central else 'Up')}"
        for i in range(nweights)
        if int(abs(central - i) * 10) not in exclude
    ]
    if proc and (proc in samples.zprocs or proc == "Z") and 2.1 not in exclude:
        # This is the PDG uncertainty (turned off for now since it doesn't seem to have been read into the nano)
        names.extend(["massShiftZ2p1MeVDown", "massShiftZ2p1MeVUp"])

    # If name is "" it won't be stored
    return [x if not matches or any(y in x for y in matches) else "" for x in names]


def widthWeightNames(matches=None, proc="", exclude=[]):
    if isinstance(exclude, (int, float)):
        exclude = [
            exclude,
        ]
    if proc[0] == "Z":
        widths = (2.49333, 2.49493, 2.4929, 2.4952, 2.4975)
    elif proc[0] == "W":
        widths = (2.09053, 2.09173, 2.043, 2.085, 2.127)
    else:
        raise RuntimeError(f"No width found for process {proc}")
    # 0 and 1 are Up, Down from mass uncertainty EW fit (already accounted for in mass variations)
    # 2, 3, and 4 are PDG width Down, Central, Up
    names = [
        f"width{proc[0]}{str(width).replace('.','p')}GeV"
        for width in widths
        if width not in exclude
    ]

    return [x if not matches or any(y in x for y in matches) else "" for x in names]


def sin2thetaWeightNames(matches=None, proc=""):
    if proc[0] != "Z":
        raise RuntimeError("sin2theta weights are only defined for Z")

    sin2thetas = (
        0.23151,
        0.23154,
        0.23157,
        0.2230,
        0.2300,
        0.2305,
        0.2310,
        0.2315,
        0.2320,
        0.2325,
        0.2330,
    )

    # 1 is the central value
    # 0 and 2 are Down, Up from uncertainty in EW fit
    names = [
        f"sin2theta{proc[0]}{str(sin2theta).replace('.','p')}"
        for sin2theta in sin2thetas
    ]

    return [x if not matches or any(y in x for y in matches) else "" for x in names]
