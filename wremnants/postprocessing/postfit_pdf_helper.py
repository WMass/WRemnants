import os
import sys

import h5py
import lhapdf
import numpy as np
from mc2hlib import lh
from mc2hlib.common import load_pdf

from rabbit import io_tools
from wremnants.postprocessing import syst_tools
from wremnants.utilities import theory_utils
from wums import logging

logger = logging.child_logger(__name__)


class PostfitPdfHelper(object):
    def __init__(self):
        pass

    def get_pdf_covariance(self):
        """
        Abstract-like method: Each child class must implement how
        to extract the raw covariance and parameter names.
        """
        raise NotImplementedError("Subclasses must implement get_pdf_covariance")

    def get_postfit_eigenvectors(self):
        r"""
        Common logic for all fitters to compute scaled eigenvectors.
        Calculates $V \sqrt{\max(\lambda, 0)}$.
        """
        cov_matrix, _ = self.get_pdf_covariance()

        # Linear algebra: eigh is optimized for symmetric matrices (covariance)
        eigv, V = np.linalg.eigh(cov_matrix)

        # Apply sqrt(eigenvalues) to the rotation matrix
        return V * np.sqrt(np.maximum(eigv, 0))

    def get_pdf_matrix(self, Q=100):
        """
        Load the base PDF grids, build the scaled Hessian difference matrix,
        and apply symmetrization if needed. Returns (matrix, grids, central_pdf_path).
        """
        central_pdf_path = "/".join([lhapdf.paths()[0], self.pdf_name, self.pdf_name])
        base_pdf, fl, xgrid = load_pdf(self.pdf_name, Q, self.max_nf, self.photon)
        headers, grids = lh.load_all_replicas(base_pdf, central_pdf_path)

        # Big matrix is hessian_i - central, scaled up by pdf_scale
        matrix = lh.big_matrix(grids[: self.n_hessian + 1]) * self.pdf_scale

        if not self.symm_errors:
            logger.info(
                f"Applying symmetrization {self.pdf_symm} to PDF uncertainties."
            )
            matrix = syst_tools.symmetrize_unc_matrix(
                matrix, self.pdf_nuisances, self.pdf_symm
            )

        return matrix, grids, central_pdf_path

    def compute_postfit_matrix(self, Q=100):
        """Compute the postfit PDF matrix and new central value."""
        K = self.get_postfit_eigenvectors()
        matrix, grids, central_pdf_path = self.get_pdf_matrix(Q)
        new_central = grids[0] + np.sum(self.pdf_pulls * matrix, axis=1)
        postfit_matrix = matrix.dot(K).add(new_central, axis=0)
        return postfit_matrix, new_central, central_pdf_path

    def write_grids(
        self, central_pdf_path, outfolder, fitlabel, lhaid, postfit_matrix, central_grid
    ):
        """Write postfit PDF grids to disk in LHAPDF format."""
        scale_label = (
            "unscaled"
            if self.pdf_scale == 1
            else f"uncx{self.pdf_scale:.1f}".replace(".", "p")
        )
        new_pdf = f"{os.path.basename(central_pdf_path)}_{fitlabel}_{scale_label}"

        outdir = os.path.join(outfolder, new_pdf)
        if not os.path.exists(outdir):
            logger.info(f"Creating output folder {outdir}")
            os.makedirs(outdir)

        outbase = "/".join([outdir, new_pdf])
        with open(central_pdf_path + ".info", "r") as inn, open(
            outbase + ".info", "w"
        ) as out:
            for l in inn.readlines():
                if l.find("SetDesc:") >= 0:
                    out.write(
                        f'SetDesc: "{self.pdf_name} modified by CMS mW postfit covariance, with prefit pdf unc scaled by {self.pdf_scale}. Produced by the command {" ".join(sys.argv)}"\n'
                    )
                elif l.find("SetIndex:") >= 0:
                    out.write(f"SetIndex: {lhaid}\n")
                elif l.find("NumMembers:") >= 0:
                    out.write(f"NumMembers: {postfit_matrix.shape[-1] + 1}\n")
                elif l.find("ErrorType") >= 0:
                    out.write(f"ErrorType: symmhessian\n")
                elif l.find("ErrorConfLevel") >= 0:
                    out.write(f"ErrorConfLevel: 68.26894921370858\n")
                else:
                    out.write(l)

        lh.write_replica(
            0, outbase, b"PdfType: 'central'\nFormat: lhagrid1\n", central_grid
        )
        for column in postfit_matrix.columns:
            header = b"PdfType: 'error'\nFormat: lhagrid1\n"
            lh.write_replica(column + 1, outbase, header, postfit_matrix[column])
        logger.info(f"Wrote PDF grids to {outbase}")


class RabbitPostfitPdfHelper(PostfitPdfHelper):
    def __init__(self, fitresult_path, args=None, pseudoData=None):
        super().__init__()
        self.fitresult = None
        self.pdf_nuisances = None
        self.pdf_pulls = None
        self.path = fitresult_path
        self.pseudoData = args.pseudoData if args is not None else pseudoData

        # Rabbit-specific attributes
        self.meta = None
        self.pdf_name = None
        self.pdf_scale = 1.0
        self.pdf_symm = None
        self.symm_errors = False
        self.n_hessian = 0
        self.max_nf = 5
        self.photon = False

        self._load_and_parse()

    def _load_and_parse(self):
        """Handles the specific I/O for Rabbit fit results."""
        self.fitresult, self.meta = io_tools.get_fitresult(
            self.path, meta=True, result=self.pseudoData
        )

        # Extract PDF metadata (the nested dictionary traversal)
        input_meta = self.meta.get("meta_info_input", {})
        input_args = input_meta.get("meta_info_input", {}).get("args", {})

        if input_args and "pdfs" in input_args:
            pdf_input = input_args["pdfs"][0]
            pdf_info = theory_utils.pdf_info_map("Zmumu_2016PostVFP", pdf_input)
            self.pdf_name = pdf_info["lha_name"]

            # Validation with LHAPDF
            pdf_set = lhapdf.getPDFSet(self.pdf_name)
            error_info = pdf_set.errorInfo
            if error_info.coreType not in ["hessian", "symmhessian"]:
                raise ValueError(
                    f"Unsupported PDF error type: {error_info.coreType}. Only Hessian PDFs are supported."
                )
            self.symm_errors = error_info.coreType == "symmhessian"
            self.n_hessian = error_info.nmemCore

            flavors = pdf_set.mkPDF(0).flavors()
            self.max_nf = max((abs(f) for f in flavors if abs(f) <= 6), default=5)
            self.photon = 22 in flavors

            # Read scaling and symmetrization settings from the fit metadata
            fit_args = input_meta.get("meta_info", {}).get("args", {})
            self.pdf_symm = fit_args.get("symmetrizePdfUnc")
            raw_scale = fit_args.get("scalePdf", 1.0)
            if raw_scale == -1:
                self.pdf_scale = theory_utils.pdf_inflation_factor(
                    pdf_info, fit_args.get("noi")
                )
                logger.info(
                    f"Using default inflation factor from theory_utils: {self.pdf_scale}"
                )
            else:
                self.pdf_scale = raw_scale
            self.pdf_scale *= pdf_info.get("scale", 1)
            logger.info(f"Scaling PDF uncertainties by {self.pdf_scale}")

        # Determine the nuisance labels and pulls (handling the pdf1 duplicate)
        self.pdf_nuisances, self.pdf_pulls = self._extract_nuisance_labels_and_pulls()

    def _extract_nuisance_labels_and_pulls(self):
        """Rabbit-specific logic for filtering PDF nuisances, returns labels and pulls."""
        regex = r"pdf\d+"
        labels, pulls, _ = io_tools.get_pulls_and_constraints(
            self.fitresult, keep_nuisances=regex
        )

        if pulls.size - 1 == self.n_hessian:
            logger.warning("Duplicate pdf1 detected. Filtering.")
            labels, pulls, _ = io_tools.get_pulls_and_constraints(
                self.fitresult, keep_nuisances=r"pdf(?![1][^\d])\d+"
            )
        return labels, pulls

    def get_pdf_covariance(self):
        """Implementation of the base class requirement for Rabbit."""
        cov_obj = self.fitresult["cov"].get()
        var_names = np.array(cov_obj.axes["parms_x"])

        mask = np.isin(var_names, self.pdf_nuisances)
        cov_values = cov_obj.values()[np.ix_(mask, mask)]

        return cov_values, var_names

    def get_channel_vals_and_errors(self, fit_types, chan):
        """Read PDF x*f(x) values and symmetric errors from fit result channels."""
        values = []
        errors = []
        for fit in fit_types:
            h = self.fitresult["mappings"]["BaseMapping"]["channels"][chan][
                f"hist_{fit}_inclusive"
            ].get()
            val = h.values()
            err = np.sqrt(h.variances())
            values.append(val)
            errors.append([err, err])
        return values, errors


class SimplePostfitPdfHelper(PostfitPdfHelper):
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.pdf_name = None
        self.pdf_scale = 1.0
        self.pdf_symm = None
        self.pdf_nuisances = None
        self.pdf_pulls = None
        self.symm_errors = False
        self.n_hessian = 0
        self.max_nf = 5
        self.photon = False
        self._cov = None

        self._load_and_parse()

    def _load_and_parse(self):
        """Load covariance, pulls, labels, and metadata from the simple HDF5 format."""
        with h5py.File(self.path, "r") as f:
            self._cov = f["covariance"][:]
            self.pdf_pulls = f["pulls"][:]
            self.pdf_nuisances = f["labels"][:].astype(str)
            self.pdf_name = f.attrs["pdf_name"]
            self.pdf_scale = f.attrs["pdf_scale"]
            self.pdf_symm = f.attrs["pdf_symm"]

        # Derive LHAPDF-dependent attributes from pdf_name
        pdf_set = lhapdf.getPDFSet(self.pdf_name)
        error_info = pdf_set.errorInfo
        if error_info.coreType not in ["hessian", "symmhessian"]:
            raise ValueError(
                f"Unsupported PDF error type: {error_info.coreType}. Only Hessian PDFs are supported."
            )
        self.symm_errors = error_info.coreType == "symmhessian"
        self.n_hessian = error_info.nmemCore

        flavors = pdf_set.mkPDF(0).flavors()
        self.max_nf = max((abs(f) for f in flavors if abs(f) <= 6), default=5)
        self.photon = 22 in flavors

    def get_pdf_covariance(self):
        """Return the stored PDF covariance matrix and nuisance labels."""
        return self._cov, self.pdf_nuisances
