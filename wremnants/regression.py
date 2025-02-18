import numpy as np
from scipy import stats
from scipy.optimize import nnls
from scipy.special import comb

from wums import logging

logger = logging.child_logger(__name__)


def compute_chi2(y, y_pred, w=None, nparams=1):
    # goodness of fit from parameter matrix 'X', values 'y' and parateters 'params'
    residuals = y - y_pred
    if w is not None:
        residuals *= w
    chi2 = np.sum((residuals**2), axis=-1)  # per fit chi2
    chi2_total = np.sum((residuals**2))  # chi2 of all fits together

    # Degrees of freedom calculation
    ndf = y.shape[-1] - nparams
    ndf_total = y.size - chi2.size * nparams

    logger.debug(
        f"Total chi2/ndf = {chi2_total}/{ndf_total} = {chi2_total/ndf_total} (p = {stats.chi2.sf(chi2_total, ndf_total)})"
    )
    logger.debug(
        f"Min chi2/ndf = {chi2.min()}/{ndf} (p = {stats.chi2.sf(chi2.min(), ndf)})"
    )
    logger.debug(
        f"Max chi2/ndf = {chi2.max()}/{ndf} (p = {stats.chi2.sf(chi2.max(), ndf)})"
    )
    logger.debug(f"Mean chi2 = {chi2.mean()}")
    logger.debug(f"Std chi2 = {chi2.std()}")
    return chi2, ndf


def get_parameter_eigenvectors(params, cov, sign=1, force_positive=False):
    # diagonalize and get eigenvalues and eigenvectors
    e, v = np.linalg.eigh(
        cov
    )  # The column eigenvectors[:, i] is the normalized eigenvector corresponding to the eigenvalue eigenvalues[i],
    # protect against negative eigenvalues
    e = np.maximum(e, 0.0)
    vT = np.transpose(
        v, axes=(*np.arange(v.ndim - 2), v.ndim - 1, v.ndim - 2)
    )  # transpose v to have row eigenvectors[i, :] for easier computations
    mag = np.sqrt(e)[..., np.newaxis] * vT * sign
    # duplicate params vector to have duplicated rows
    params_brd = np.broadcast_to(params[..., np.newaxis], mag.shape)
    paramsT = np.transpose(
        params_brd,
        axes=(
            *np.arange(params_brd.ndim - 2),
            params_brd.ndim - 1,
            params_brd.ndim - 2,
        ),
    )
    mask = (paramsT + mag < 0).any(axis=-1)
    if force_positive and mask.sum() > 0:
        logger.info(f"Force {mask.sum()} eigenvectors to be positive")
        # scaling the magnitude of the eigenvector to avoid negative coefficients
        min_idxs = (
            paramsT + mag == np.min(paramsT + mag, axis=-1)[..., np.newaxis]
        )  # find indices with minimum value of each eigenvector
        min_paramsT = paramsT[min_idxs]
        min_mag = mag[min_idxs]
        factors = np.reshape(-1 * min_paramsT / min_mag, mask.shape)
        factors = np.where(mask, factors, np.ones_like(mask))
        factors = np.broadcast_to(factors[..., np.newaxis], mag.shape)
        mag = factors * mag
    if np.sum(mag.sum(axis=-1) == 0):
        logger.warning(
            f"Found {np.sum(mag.sum(axis=-1) == 0)} eigenvector shifts where all coefficients are 0"
        )
    params_var = paramsT + mag
    if np.sum(params_var.sum(axis=-1) == 0):
        logger.warning(
            f"Found {np.sum(params_var.sum(axis=-1) == 0)} variations where all coefficients are 0"
        )
    return params_var


def make_eigenvector_predictons(params, cov, func, x1, x2=None, force_positive=False):
    # return alternate values i.e. nominal+/-variation
    params_up = get_parameter_eigenvectors(
        params, cov, sign=1, force_positive=force_positive
    )
    y_pred_up = func(x1, params_up) if x2 is None else func(x1, x2, params_up)
    y_pred_up = np.moveaxis(
        y_pred_up, params.ndim - 1, -1
    )  # put parameter variations last
    params_dn = get_parameter_eigenvectors(
        params, cov, sign=-1, force_positive=force_positive
    )
    y_pred_dn = func(x1, params_dn) if x2 is None else func(x1, x2, params_dn)
    y_pred_dn = np.moveaxis(
        y_pred_dn, params.ndim - 1, -1
    )  # put parameter variations last
    return np.stack((y_pred_up, y_pred_dn), axis=-1)


def chebycshev(n, x):
    # chebychev polynomials of first kind (see https://en.wikipedia.org/wiki/Chebyshev_polynomials)
    if n == 0:
        return 1
    if n == 1:
        return x
    return 2 * x * chebycshev(n - 1, x) - chebycshev(n - 2, x)


def poly(pol, order, order2=None):
    if order2 is None:
        if pol == "power":
            return lambda x, n, p=1: p * x**n
        elif pol == "chebyshev":
            return lambda x, n, p=1, o=order: p * chebycshev(n, x)
        elif pol == "bernstein":
            return lambda x, n, p=1, o=order: p * comb(o, n) * x**n * (1 - x) ** (o - n)
        elif pol == "monotonic":
            # integral of bernstein (see https://en.wikipedia.org/wiki/Bernstein_polynomial#Properties)
            # for n>o return one additional parameter for the constant term from the integration
            return lambda x, n, p=1, o=order: (
                p * x**0
                if n == 0
                else -1
                * p
                * 1
                / o
                * np.array(
                    [comb(o, k) * x**k * (1 - x) ** (o - k) for k in range(n, o + 1)]
                ).sum(axis=0)
            )
    else:
        return (
            lambda x, x2, n, m, p=1, o1=order, o2=order2: p
            * poly(pol, o1)(x, n)
            * poly(pol, o2)(x2, m)
        )


def get_parameter_matrices(x, y, w, order, pol="power"):
    if x.shape != y.shape:
        x = np.broadcast_to(x, y.shape)
    stackX = []  # parameter matrix X
    stackXTY = []  # and X.T @ Y
    f = poly(pol, order)
    for n in range(order + 1):
        p = f(x, n)
        stackX.append(w * p)
        stackXTY.append((w**2 * p * y).sum(axis=(-1)))
    X = np.stack(stackX, axis=-1)
    XTY = np.stack(stackXTY, axis=-1)
    return X, XTY


def get_parameter_matrices_from2D(
    x, x2, y, w, order, order2=None, pol="power", flatten=False
):
    x, x2 = np.broadcast_arrays(x[np.newaxis, ...], x2[..., np.newaxis])
    x = np.broadcast_to(x, y.shape)
    x2 = np.broadcast_to(x2, y.shape)
    if order2 is None:
        order2 = [
            0,
        ] * (order + 1)
    elif type(order2) == int:
        order2 = [
            order2,
        ] * (order + 1)
    elif type(order2) == list:
        if len(order2) < order + 1:
            order2.append(
                [
                    0,
                ]
                * (len(order2) - order + 1)
            )
    else:
        raise RuntimeError(f"Input 'order2' requires type 'None', 'int' or 'list'")
    stackX = []  # parameter matrix X
    stackXTY = []  # and X.T @ Y
    for n in range(order + 1):
        f = poly(pol, order, order2[n])
        for m in range(order2[n] + 1):
            p = f(x, x2, n, m)
            stackX.append(w * p)
            stackXTY.append((w**2 * p * y).sum(axis=(-2, -1)))
    X = np.stack(stackX, axis=-1)
    XTY = np.stack(stackXTY, axis=-1)
    if flatten:
        # flatten the 2D array into 1D
        newshape = (*y.shape[:-2], np.prod(y.shape[-2:]))
        y = y.reshape(newshape)
        X = X.reshape(*newshape, X.shape[-1])

    return X, y, XTY


def get_regression_function(orders, pol="power"):
    if not isinstance(orders, list):
        orders = [orders]

    if len(orders) == 1:

        def fsum(x, ps, o=orders[0]):
            f = poly(pol, o)
            if hasattr(ps, "ndim") and ps.ndim > 1:
                return sum([f(x, n, ps[..., n, np.newaxis]) for n in range(o + 1)])
            else:
                return sum([f(x, n, ps[n]) for n in range(o + 1)])

    else:

        def fsum(x1, x2, ps, o1=orders[0], o2=orders[1]):
            idx = 0
            psum = 0
            x1, x2 = np.broadcast_arrays(x1[np.newaxis, ...], x2[..., np.newaxis])
            if hasattr(ps, "ndim") and ps.ndim > 1:
                x1 = np.broadcast_to(x1, [*ps.shape[:-1], *x1.shape])
                x2 = np.broadcast_to(x2, [*ps.shape[:-1], *x2.shape])
                for n in range(o1 + 1):
                    f = poly(pol, o1, o2[n])
                    for m in range(o2[n] + 1):
                        psum += f(x1, x2, n, m, ps[..., idx, np.newaxis, np.newaxis])
                        idx += 1
            else:
                for n in range(o1 + 1):
                    f = poly(pol, o1, o2[n])
                    for m in range(o2[n] + 1):
                        psum += f(x1, x2, n, m, ps[idx])
                        idx += 1
            return psum

    return fsum


def solve_leastsquare(X, XTY):
    # compute the transpose of X for the mt and parameter axes
    XT = np.transpose(X, axes=(*np.arange(X.ndim - 2), X.ndim - 1, X.ndim - 2))
    XTX = XT @ X
    # compute the inverse of the matrix in each bin (reshape to make last two axes contiguous, reshape back after inversion),
    # this term is also the covariance matrix for the parameters
    XTXinv = np.linalg.inv(XTX.reshape(-1, *XTX.shape[-2:]))
    XTXinv = XTXinv.reshape((*XT.shape[:-2], *XTXinv.shape[-2:]))
    params = np.einsum("...ij,...j->...i", XTXinv, XTY)
    return params, XTXinv


def solve_nonnegative_leastsquare(X, XTY, exclude_idx=None):
    # exclude_idx to exclude the non negative constrained for one parameter by evaluating the nnls twice and flipping the sign
    XT = np.transpose(X, axes=(*np.arange(X.ndim - 2), X.ndim - 1, X.ndim - 2))
    XTX = XT @ X
    XTXinv = np.linalg.inv(XTX.reshape(-1, *XTX.shape[-2:]))
    XTXinv = XTXinv.reshape((*XT.shape[:-2], *XTXinv.shape[-2:]))
    orig_shape = XTY.shape
    nBins = np.prod(orig_shape[:-1])
    XTY_flat = XTY.reshape(nBins, XTY.shape[-1])
    XTX_flat = XTX.reshape(nBins, XTX.shape[-2], XTX.shape[-1])
    # params = [fnnls(xtx, xty) for xtx, xty in zip(XTX_flat, XTY_flat)] # use fast nnls (for some reason slower, even though it should be faster ...)
    params = [
        nnls(xtx, xty)[0] for xtx, xty in zip(XTX_flat, XTY_flat)
    ]  # use scipy implementation of nnls
    params = np.reshape(params, orig_shape)
    if exclude_idx is not None and np.sum(params[..., exclude_idx] == 0):
        mask = params[..., exclude_idx] == 0
        mask_flat = mask.flatten()
        w_flip = np.ones(XTY.shape[-1])
        w_flip[exclude_idx] = -1
        params_negative = [
            nnls(xtx, xty)[0]
            for xtx, xty in zip(XTX_flat[mask_flat], XTY_flat[mask_flat] * w_flip)
        ]
        params[mask] = np.array(params_negative) * w_flip
        logger.info(
            f"Found {mask.sum()} parameters that are excluded in nnls and negative"
        )
    return params, XTXinv


def get_solver(polynomial):
    # solve with non negative least squares
    if polynomial in [
        "bernstein",
    ]:
        return solve_nonnegative_leastsquare
    elif polynomial in [
        "monotonic",
    ]:
        # exclude first parameter (integration constant) from non negative constraint
        return lambda x, y: solve_nonnegative_leastsquare(x, y, 0)
    else:
        return solve_leastsquare


def transform_bernstein(x, min_x, max_x, cap_x=False):
    # transform x to [0,1] (where bernstein polinomials are defined)
    # get x axes values for interpolation/smoothing with transformation
    x = (x - min_x) / (max_x - min_x)
    if np.sum(x < 0) or np.sum(x > 1):
        if cap_x:
            logger.info("Values outside [0,1] found, those will be capped to [0,1]")
            x[x < 0] = 0
            x[x > 1] = 1
        else:
            raise RuntimeError(
                f"All values need to be within [0,1] but {np.sum(x < 0)} values smaller 0 ({x[x < 0]}) and {np.sum(x > 1)} larger 1 ({x[x > 1]}) found after transformation with xmin={min_x} and xmax={max_x}"
            )
    return x


def transform_chebyshev(x, *args, **kwargs):
    # transform x to [-1,1] (where chebyshev polinomials are defined)
    return transform_bernstein(x, *args, **kwargs) * 2 - 1


class Regressor(object):
    # supported polynomials
    polynomials = ["power", "bernstein", "monotonic", "chebyshev"]

    def __init__(
        self,
        polynomial,
        order=None,
        order2=None,
        cap_x=False,
        min_x=0,
        max_x=1,
        nnls=None,
    ):
        if polynomial not in Regressor.polynomials:
            raise NotImplementedError(
                f"Polynomial {polynomial} not implemented. Supported polynomials are: {Regressor.polynomials}"
            )
        self.polynomial = polynomial
        self.order = order

        if nnls is None:
            self.solver = get_solver(polynomial)
        else:
            self.solver = solve_nonnegative_leastsquare if nnls else solve_leastsquare

        self.evaluator = get_regression_function(order, pol=polynomial)

        self.cap_x = cap_x
        self.min_x = min_x
        self.max_x = max_x

        self.params = None
        self.cov = None

        # external parameters and covariance matrix added for evaluation
        self.external_params = None
        self.external_cov = None

    def transform_x(self, x):
        if self.polynomial in ["bernstein", "monotonic"]:
            x = transform_bernstein(x, self.min_x, self.max_x, self.cap_x)
        else:
            x = transform_chebyshev(x, self.min_x, self.max_x, self.cap_x)
        return x

    def solve(self, x, y, w, chi2_info=True):
        x = self.transform_x(x)
        X, XTY = get_parameter_matrices(x, y, w, self.order, pol=self.polynomial)
        self.params, self.cov = self.solver(X, XTY)

        if chi2_info:
            ypred = self.evaluator(x, self.params)
            compute_chi2(y, ypred, w, self.params.shape[-1])

    def evaluate(self, x):
        x = self.transform_x(x)
        params = self.params
        if self.external_params is not None:
            params += self.external_params[
                ...,
                *[
                    np.newaxis
                    for n in range(self.params.ndim - self.external_params.ndim)
                ],
                :,
            ]
        return self.evaluator(x, params)

    def get_eigenvector_predictions(self, x1, x2=None):
        x1 = self.transform_x(x1)
        if x2 is not None:
            x2 = self.transform_x(x2)
        cov = self.cov
        params = self.params
        if self.external_cov:
            cov += self.external_cov[
                ...,
                *[np.newaxis for n in range(self.cov.ndim - self.external_cov.ndim)],
                :,
                :,
            ]
        if self.external_params is not None:
            params += self.external_params[
                ...,
                *[
                    np.newaxis
                    for n in range(self.params.ndim - self.external_params.ndim)
                ],
                :,
            ]
        return make_eigenvector_predictons(
            params, cov, func=self.evaluator, x1=x1, x2=x2, force_positive=False
        )

    def reduce_parameters(self, weight_vector=None, axis=-2):
        # linear parameter combination using weight_vector
        if weight_vector is None:
            weight_vector = np.ones_like(self.params.shape[axis])
        if axis < 0:
            axis = self.params.ndim + axis
        if axis == self.params.ndim - 1:
            raise RuntimeError(
                "Last axis can not be reduced since it's the parameter axis"
            )
        slices = [
            np.newaxis if i != axis else slice(None) for i in range(self.params.ndim)
        ]

        self.params = np.sum(self.params * weight_vector[*slices], axis=axis)
        self.cov = np.sum(self.cov * weight_vector[*slices, np.newaxis] ** 2, axis=axis)

    def force_positive(self, exclude_idx=None):
        # performing a nnls to enforce monotonicity for the signal region (using generalized least squares)
        Y = self.params
        W = np.linalg.inv(self.cov.reshape(-1, *self.cov.shape[-2:]))
        W = W.reshape((*self.cov.shape[:-2], *W.shape[-2:]))
        WY = np.einsum("...ij,...j->...i", W, Y)
        # the design matrix X is just a 1xn unity matrix and can thus be ignored
        XTWY = WY
        XTWX = W

        orig_shape = XTWY.shape
        nBins = np.prod(orig_shape[:-1])
        XTWY_flat = XTWY.reshape(nBins, XTWY.shape[-1])
        XTWX_flat = XTWX.reshape(nBins, XTWX.shape[-2], XTWX.shape[-1])
        self.params = [nnls(xtwx, xtwy)[0] for xtwx, xtwy in zip(XTWX_flat, XTWY_flat)]
        self.params = np.reshape(self.params, orig_shape)

        # allow the integration constaint to be negative
        if exclude_idx is not None and np.sum(self.params[..., exclude_idx] == 0) > 0:
            mask = self.params[..., exclude_idx] == 0
            mask_flat = mask.flatten()
            w_flip = np.ones(XTWY.shape[-1])
            w_flip[exclude_idx] = -1
            self.params_negative = [
                nnls(xtx, xty)[0]
                for xtx, xty in zip(XTWX_flat[mask_flat], XTWY_flat[mask_flat] * w_flip)
            ]
            self.params[mask] = np.array(self.params_negative) * w_flip
            logger.info(
                f"Found {mask.sum()} parameters that are excluded in nnls and negative"
            )

        # modify covariance matrix by fixing parameters at boundary
        # this should lead to zero eigenvalues/null variations in
        # the corresponding cases
        self.cov = np.where(self.params[..., None] == 0.0, 0.0, self.cov)
        self.cov = np.where(self.params[..., None, :] == 0.0, 0.0, self.cov)


class Regressor2D(Regressor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not isinstance(self.cap_x, list):
            self.order = [self.order] * 2
        if not isinstance(self.cap_x, list):
            self.cap_x = [self.cap_x] * 2
        if not isinstance(self.min_x, list):
            self.min_x = [self.min_x] * 2
        if not isinstance(self.max_x, list):
            self.max_x = [self.max_x] * 2

    def transform_x(self, x1, x2):
        if self.polynomial in ["bernstein", "monotonic"]:
            x1 = transform_bernstein(x1, self.min_x[0], self.max_x[0], self.cap_x[0])
            x2 = transform_bernstein(x2, self.min_x[1], self.max_x[1], self.cap_x[1])
        else:
            x1 = transform_chebyshev(x1, self.min_x[0], self.max_x[0], self.cap_x[0])
            x2 = transform_chebyshev(x2, self.min_x[1], self.max_x[1], self.cap_x[1])
        return x1, x2

    def solve(self, x1, x2, y, w, flatten=False):
        x1, x2 = self.transform_x(x1, x2)
        X, XTY = get_parameter_matrices_from2D(
            x1, x2, y, w, *self.order, pol=self.polynomial, flatten=flatten
        )
        self.params, self.cov = self.solver(X, XTY)
