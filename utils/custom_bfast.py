from functools import partial

import numpy as np
import statsmodels.api as sm
from scipy.stats import norm

sc_me = np.array(
    [
        0.7552,
        0.9809,
        1.1211,
        1.217,
        1.2811,
        1.3258,
        1.3514,
        1.3628,
        1.361,
        1.3751,
        0.7997,
        1.0448,
        1.203,
        1.3112,
        1.387,
        1.4422,
        1.4707,
        1.4892,
        1.4902,
        1.5067,
        0.825,
        1.0802,
        1.2491,
        1.3647,
        1.4449,
        1.5045,
        1.5353,
        1.5588,
        1.563,
        1.5785,
        0.8414,
        1.1066,
        1.2792,
        1.3973,
        1.4852,
        1.5429,
        1.5852,
        1.6057,
        1.6089,
        1.6275,
        0.8541,
        1.1247,
        1.304,
        1.425,
        1.5154,
        1.5738,
        1.6182,
        1.646,
        1.6462,
        1.6644,
        0.8653,
        1.1415,
        1.3223,
        1.4483,
        1.5392,
        1.6025,
        1.6462,
        1.6697,
        1.6802,
        1.6939,
        0.8017,
        1.0483,
        1.2059,
        1.3158,
        1.392,
        1.4448,
        1.4789,
        1.4956,
        1.4976,
        1.5115,
        0.8431,
        1.1067,
        1.2805,
        1.4042,
        1.4865,
        1.5538,
        1.59,
        1.6105,
        1.6156,
        1.6319,
        0.8668,
        1.1419,
        1.3259,
        1.4516,
        1.5421,
        1.6089,
        1.656,
        1.6751,
        1.6828,
        1.6981,
        0.8828,
        1.1663,
        1.3533,
        1.4506,
        1.5791,
        1.6465,
        1.6927,
        1.7195,
        1.7245,
        1.7435,
        0.8948,
        1.1846,
        1.3765,
        1.5069,
        1.6077,
        1.677,
        1.7217,
        1.754,
        1.7574,
        1.7777,
        0.9048,
        1.1997,
        1.3938,
        1.5305,
        1.6317,
        1.7018,
        1.7499,
        1.7769,
        1.7889,
        1.8052,
        0.8444,
        1.1119,
        1.2845,
        1.4053,
        1.4917,
        1.5548,
        1.5946,
        1.6152,
        1.621,
        1.6341,
        0.8838,
        1.1654,
        1.3509,
        1.4881,
        1.5779,
        1.653,
        1.6953,
        1.7206,
        1.7297,
        1.7455,
        0.904,
        1.1986,
        1.3951,
        1.5326,
        1.6322,
        1.7008,
        1.751,
        1.7809,
        1.7901,
        1.8071,
        0.9205,
        1.2217,
        1.4212,
        1.5593,
        1.669,
        1.742,
        1.7941,
        1.8212,
        1.8269,
        1.8495,
        0.9321,
        1.2395,
        1.444,
        1.5855,
        1.6921,
        1.7687,
        1.8176,
        1.8553,
        1.8615,
        1.8816,
        0.9414,
        1.253,
        1.4596,
        1.61,
        1.7139,
        1.793,
        1.8439,
        1.8763,
        1.8932,
        1.9074,
        0.8977,
        1.1888,
        1.3767,
        1.5131,
        1.6118,
        1.6863,
        1.7339,
        1.7572,
        1.7676,
        1.7808,
        0.9351,
        1.2388,
        1.4362,
        1.5876,
        1.693,
        1.7724,
        1.8223,
        1.8559,
        1.8668,
        1.8827,
        0.9519,
        1.27,
        1.482,
        1.6302,
        1.747,
        1.8143,
        1.8756,
        1.9105,
        1.919,
        1.9395,
        0.9681,
        1.2918,
        1.5013,
        1.6536,
        1.7741,
        1.8573,
        1.914,
        1.945,
        1.9592,
        1.9787,
        0.9799,
        1.3088,
        1.5252,
        1.6791,
        1.7967,
        1.8837,
        1.9377,
        1.9788,
        1.9897,
        2.0085,
        0.988,
        1.622,
        1.5392,
        1.7014,
        1.8154,
        1.9061,
        1.9605,
        1.9986,
        2.0163,
        2.0326,
    ]
).reshape(
    (60, 4), order="F"  # apparently, R uses Fortran order
)


def r_style_interval(from_tuple, end_tuple, frequency):
    """
    Create time interval using R-style double-tuple notation
    """
    from_year, from_seg = from_tuple
    end_year, end_seg = end_tuple
    n = (end_year - from_year + 1) * frequency
    full_range = np.linspace(from_year, end_year + 1, num=n, endpoint=False)
    real_range = full_range[(from_seg - 1) : n - (frequency - end_seg)]
    return real_range


def _Xinv0(x, coeffs):
    """
    Approximate (X'X)^-1 using QR decomposition.

    NOTE: This version also stacks the results N times, with
    N being the number of pixels (inferred from coeffs).
    """
    r = np.linalg.qr(x)[1]
    qr_rank = np.linalg.matrix_rank(r)

    r = r[:qr_rank, :qr_rank]

    k, n_pixels = coeffs.shape
    rval = np.zeros((k, k))
    rval[:qr_rank, :qr_rank] = np.linalg.inv(r.T @ r)
    return np.repeat(rval[np.newaxis, ...], n_pixels, axis=0)


def recresid(x: np.ndarray, y: np.ndarray):
    """
    Function for computing the recursive residuals (standardized one step
    prediction errors) of a linear regression model.

    NOTE: This version is modified to work with batches of pixels, with the
    batch dimension on the last axis. All the matmul ops have been
    adjusted to work in parallel on the batch. This modification leverage
    on the vector optimizations of numpy.
    """
    nrow, ncol = x.shape

    start = ncol + 1
    end = nrow
    tol = np.sqrt(np.finfo(float).eps / ncol)

    # checks and data dimensions
    assert start > ncol and start <= nrow

    n = end
    q = start - 1
    rval = np.zeros((n - q, y.shape[1]))

    # initialize recursion
    y1 = y[:q]

    x_q = x[:q]

    model = sm.OLS(y1, x_q).fit()
    coeffs = model.params
    if np.ndim(coeffs) == 1:
        coeffs = coeffs[:, np.newaxis]

    X1 = _Xinv0(x_q, coeffs)
    betar = np.nan_to_num(coeffs).T

    xr = np.repeat([x[q]], coeffs.shape[1], axis=0)
    fr = 1 + (xr[:, np.newaxis, :] @ X1 @ xr[..., np.newaxis])
    rval[0] = (
        y[q] - np.squeeze(xr[:, np.newaxis, :] @ betar[..., np.newaxis])
    ) / np.squeeze(np.sqrt(fr))

    # check recursion against full QR decomposition?
    check = np.ones(y.shape[1], dtype=bool)

    if (q + 1) >= n:
        return rval

    for r in range(q + 1, n):
        # check for NAs in coefficients
        nona = np.logical_not(np.isnan(coeffs).any(axis=0))

        # recursion formula
        X1 = X1 - (X1 @ xr[..., np.newaxis] @ xr[:, np.newaxis, :] @ X1) / fr

        betar += (
            np.squeeze(X1 @ xr[..., np.newaxis])
            * rval[r - q - 1 : r - q].T
            * np.sqrt(fr)[..., 0]
        )

        # full QR decomposition
        if np.any(check):
            y2 = y[:r, check]
            x_i = x[:r]
            model = sm.OLS(y2, x_i, missing="drop").fit()
            mparams = model.params
            if np.ndim(mparams) == 1:  # batch of 1 pixel case, remove the auto squeeze
                mparams = mparams[:, np.newaxis]
            coeffs[:, check] = mparams
            nona[check] = np.logical_and(
                nona[check],
                np.logical_and(
                    np.logical_not(np.isnan(betar[check]).any(axis=1)),
                    np.logical_not(np.isnan(mparams).any(axis=0)),
                ),
            )

            X1[check] = _Xinv0(x_i, coeffs)
            betar[check] = np.nan_to_num(mparams).T
            # keep checking?
            check[check] = np.logical_not(
                np.logical_and(
                    nona[check], np.allclose(mparams.T, betar[check], atol=tol)
                )
            )

        # residual
        xr = np.repeat([x[r]], coeffs.shape[1], axis=0)
        fr = 1 + (xr[:, np.newaxis, :] @ X1 @ xr[..., np.newaxis])
        val = np.nan_to_num(xr * betar)
        v = (y[r] - np.sum(val, axis=1)) / np.squeeze(np.sqrt(fr))
        rval[r - q] = v

    rval = np.around(rval, 8)
    return rval


def sctest(X: np.ndarray, y: np.ndarray, h: float, verbosity: int = 0):
    """
    Performs a generalized fluctuation test.

    Parameters
    ----------
    X : ndarray
        Input vectors of regressors - design matrix:
        ndarray of shape (nobs, f) with f number of features.
    y : ndarray
        Input time series of shape (nobs, np), where nobs
        is the number of time steps and np the number of
        pixels to analyse.
    h : float, default 0.15
        Float in the interval (0,1) specifying the
        bandwidth relative to the sample size in
        the MOSUM/ME monitoring processes.
    verbosity : int, optional (default=0)
        The verbosity level (0=no output, 1=output)

    Returns
    -------
    (stat, p_v) : tuple of floats
        A tuple of arrays containing applied 'max' functional and
        p value of each pixel
    """
    nobs = len(X)

    fm = sm.OLS(
        y, X
    ).fit()  # If a batch of pixels is given in 'y' in the last axis, multiple linear models are fitted in parallel, one for each pixel

    e = np.squeeze(y) - fm.predict(exog=X)  # residuals of the model
    if (
        np.ndim(e) == 1
    ):  # In case a single time series is modelled, make its shape coherent with the code
        e = e[:, np.newaxis]

    sigma = np.sqrt(np.sum(e**2, axis=0) / (nobs - 2)).reshape(1, -1)

    nh = np.floor(nobs * h).astype(int)

    e_zero = np.insert(e, 0, 0, axis=0)

    process = np.cumsum(e_zero, axis=0)
    process = process[nh:, :] - process[: (nobs - nh + 1), :]
    process = process / (sigma * np.sqrt(nobs))

    stat = np.max(np.abs(process), axis=0)
    p_v = p_value(stat, h, 1)

    return (stat, p_v)


def p_value(x: np.ndarray, h: float, k: int, max_k: int = 6, table_dim: int = 10):
    """
    Returns the p value for the process.

    Parameters
    ----------
    x : ndarray
        Result of application of the functional.
    h : float
        Bandwidth parameter.
    k : int
        Number of rows of process matrix x

    Returns
    -------
    p : float
        p value for the process.
    """

    k = min(k, max_k)

    crit_table = sc_me[((k - 1) * table_dim) : (k * table_dim), :]
    tablen = crit_table.shape[1]
    tableh = np.arange(1, table_dim + 1) * 0.05
    tablep = np.array((0.1, 0.05, 0.025, 0.01))
    tableipl = np.zeros(tablen)

    for i in range(tablen):
        tableipl[i] = np.interp(h, tableh, crit_table[:, i])

    tableipl = np.insert(tableipl, 0, 0)
    tablep = np.insert(tablep, 0, 1)

    p = np.interp(x, tableipl, tablep)

    return p


def SSRi(i: int, X: np.ndarray, y: np.ndarray, k: int):
    """
    Compute i'th row of the SSR diagonal matrix, i.e,
    the recursive residuals for segments starting at i = 1:(n-h+1)

    NOTE: This version is modified to work with batches of pixels, with the
    batch dimension on the last axis.
    """
    ssr = recresid(X[i:], y[i:])
    rval = np.concatenate(
        (np.full((k, ssr.shape[-1]), np.nan), np.cumsum(ssr**2, axis=0))
    )
    return rval


def ssr_triang(n: int, h: float, X: np.ndarray, y: np.ndarray, k: int):
    """
    Calculates the upper triangular matrix of squared residuals
    """
    my_SSRi = partial(SSRi, X=X, y=y, k=k)
    # return np.vectorize(my_SSRi)(np.arange(n-h+1))
    return np.array([my_SSRi(i) for i in range(n - h + 1)], dtype=object)


def breakpoints(X: np.ndarray, y: np.ndarray, h: float = 0.15, verbosity: int = 0):
    """
    Computation of optimal breakpoints in regression relationships.

    NOTE: Maximum number of breakpoints is assumed to be 1.
    TODO: General case with m breaks.

    Parameters
    ----------
    X : ndarray
        Input vectors of regressors - design matrix:
        ndarray of shape (nobs, f) with f number of features.
    y : ndarray
        Input time series of shape (nobs, np), where nobs
        is the number of time steps and np the number of
        pixels to analyse.
    h : float, default 0.15
        Float in the interval (0,1) specifying the
        bandwidth relative to the sample size in
        the MOSUM/ME monitoring processes.
    verbosity : int, optional (default=0)
        The verbosity level (0=no output, 1=output)

    Returns
    -------
    bp : ndarray of ints
        Array of indices of the breaks detected. It has the
        a shape (np,). A value equals to -1 means no break detected
    """
    n = len(X)
    k = 2
    h = np.floor(n * h).astype(int)

    ## compute optimal previous partner if observation i is the mth break
    ## store results together with SSRs in SSR_table
    SSR_triang = ssr_triang(n, h, X, y, k)  # Never considere intercept only
    index = np.arange((h - 1), (n - h))

    ## 1 break
    break_SSR = np.array([SSR_triang[0][i] for i in index])

    # SSR_table = np.column_stack((index, break_SSR))
    SSR_table = np.stack(
        (np.repeat(index.reshape(-1, 1), break_SSR.shape[1], axis=1), break_SSR), axis=1
    )

    _, BIC_table = break_summary(n, k, SSR_triang, SSR_table, n, h)

    # find the optimal number of breakpoints using Bayesian Information Criterion
    breaks = np.argmin(BIC_table, axis=0)

    _, bp = breakpoints_for_m(n, h, SSR_triang, SSR_table, breaks, n)

    return np.squeeze(bp)


def break_summary(
    n: int, k: int, SSR_triang: np.ndarray, SSR_table: np.ndarray, nobs: int, h: float
):
    """
    Calculates Sums of Squared Residuals and BIC for m in 0..max_breaks

    NOTE:
        - This version is modified to work with batches of pixels, with the
        batch dimension on the last axis.
        - max_breaks is always 1 in this version.
    """
    SSR = np.concatenate(
        ([SSR_triang[0][nobs - 1]], [np.repeat(np.nan, len(SSR_triang[0][nobs - 1]))])
    )
    not_to_minus_inf = np.logical_not(np.isclose(SSR[0], 0.0))
    BIC_val = np.full(SSR.shape[-1], -np.inf)
    BIC_val[not_to_minus_inf] = n * (
        np.log(SSR[0][not_to_minus_inf]) + 1 - np.log(n) + np.log(2 * np.pi)
    ) + np.log(n) * (k + 1)
    BIC = np.concatenate(([BIC_val], [np.repeat(np.nan, SSR.shape[-1])]))
    SSR1, breakpoints = breakpoints_for_m(n, h, SSR_triang, SSR_table, 1, nobs)
    SSR[1] = SSR1
    BIC[1] = BIC_fun(n, k, SSR1, breakpoints)
    retval = (SSR, BIC)
    return retval


def BIC_fun(n: int, k: int, SSR: np.ndarray, breakpoints: np.ndarray):
    """
    Bayesian Information Criterion

    NOTE: This version is modified to work with batches of pixels.
    """
    bic = np.full(SSR.shape, -np.inf)
    to_process = np.logical_not(np.isclose(SSR, 0.0))
    df = (k + 1) * (
        np.sum(
            ~np.isnan(breakpoints[to_process].reshape(-1, to_process.shape[-1])), axis=0
        )
        + 1
    )
    # log-likelihood
    logL = n * (np.log(SSR[to_process]) + 1 - np.log(n) + np.log(2 * np.pi))
    bic[to_process] = df * np.log(n) + logL
    return bic


def breakpoints_for_m(
    n: int,
    h: float,
    SSR_triang: np.ndarray,
    SSR_table: np.ndarray,
    m: np.ndarray,
    nobs: int,
):
    """
    Extract the required number of breakpoints

    NOTE: This version is modified to work with batches of pixels, with the
    batch dimension on the last axis.
    """
    if np.isscalar(m) and m == 1:  # Assume we want to estimate 1 break for every pixel
        m = np.ones(SSR_table.shape[-1])

    # defaults for no breaks (-1)
    breakpoints = np.full((1, len(m)), -1)

    SSRs = SSR_triang[0][nobs - 1].reshape(1, -1)

    # Extract break only where required
    breakpoints[:, m > 0] = extract_break(n, h, SSR_triang, SSR_table, m > 0)
    # map reduce
    bp = np.concatenate(
        (
            [np.zeros(breakpoints[:, m > 0].shape[-1])],
            breakpoints[:, m > 0],
            [np.full(breakpoints[:, m > 0].shape[-1], nobs - 1)],
        )
    )
    # shape (segment_number, segment_end_points, n_pixels)
    cb = np.stack((bp[:-1] + 1, bp[1:]), axis=1).astype(int)
    # change to a view with shape (n_pixels, segment_number, segment_end_points)
    cb = np.swapaxes(np.swapaxes(cb, 0, -1), 1, 2)

    def fun(x: np.ndarray, j: int):
        """
        Extract segment value from the SSR diagonal matrix for the segment defined
        by the two bounds in x and the required selected pixel j.
        """
        return SSR_triang[x[0]][x[1] - x[0]][j]

    SSRs[:, m > 0] = np.array(
        [np.sum([fun(i, j) for i in elem]) for j, elem in enumerate(cb)]
    )
    return SSRs, breakpoints


def extract_break(
    n: int,
    h: float,
    SSR_triang: np.ndarray,
    SSR_table: np.ndarray,
    m: np.ndarray = None,
):
    """
    Extract optimal breaks
    """
    _, ncol, _ = SSR_table.shape
    if 2 > ncol:
        raise ValueError("compute SSR_table with enough breaks before")

    index = SSR_table[:, 0, 0].astype(int)
    fun = lambda i: SSR_table[i - h + 1, 1][m > 0] + SSR_triang[i + 1][n - i - 2][m > 0]
    # parallel map
    break_SSR = np.stack([fun(i) for i in index], axis=0)
    opt = index[np.nanargmin(break_SSR, axis=0)].reshape(1, -1)
    return opt


def partition_matrix(part, mat):
    """
    Create a partition matrix, given a partition vector and a matrix
    """
    if part.shape[0] != mat.shape[0]:
        raise ValueError("Partition length must equal Matrix nrows")
    if mat.ndim != 2:
        raise TypeError("mat must be a 2D matrix")

    n_rows, n_cols = mat.shape
    n_pixels = part.shape[1]
    # number of partitions
    n_parts = 2  # here only detected breaks are processed and the only 1 break is considered, thus always 2 parts
    ret_val = np.zeros((n_rows, n_parts * n_cols, n_pixels)).astype(float)
    for j in range(n_parts):
        i1, i2 = np.where(part == j)
        ret_val[i1, (j * n_cols) : ((j + 1) * n_cols), i2] = mat[i1, :]
    return ret_val


def breakfactor(breaks, nobs):
    fac = np.ones((nobs, len(breaks)))
    # scan
    for i, b in enumerate(breaks):
        if not np.isnan(b):
            fac[: int(b), i] = 0
    return fac


def pargmaxV(x, xi=1, phi1=1, phi2=1):
    phi = xi * (phi2 / phi1) ** 2

    def G1(x, xi=1, phi=1):
        x = np.abs(x)
        frac = xi / phi
        rval = (
            -np.exp(np.log(x) / 2 - x / 8 - np.log(2 * np.pi) / 2)
            - (phi / xi * (phi + 2 * xi) / (phi + xi))
            * np.exp(
                (frac * (1 + frac) * x / 2) + norm.logcdf(-(0.5 + frac) * np.sqrt(x))
            )
            + np.exp(
                np.log(x / 2 - 2 + ((phi + 2 * xi) ** 2) / ((phi + xi) * xi))
                + norm.logcdf(-np.sqrt(x) / 2)
            )
        )
        return rval

    def G2(x, xi=1, phi=1):
        x = np.abs(x)
        frac = xi**2 / phi
        rval = (
            1
            + np.sqrt(frac)
            * np.exp(np.log(x) / 2 - (frac * x) / 8 - np.log(2 * np.pi) / 2)
            + (xi / phi * (2 * phi + xi) / (phi + xi))
            * np.exp(
                ((phi + xi) * x / 2)
                + norm.logcdf(-(phi + xi / 2) / np.sqrt(phi) * np.sqrt(x))
            )
            - np.exp(
                np.log(((2 * phi + xi) ** 2) / ((phi + xi) * phi) - 2 + frac * x / 2)
                + norm.logcdf(-np.sqrt(frac) * np.sqrt(x) / 2)
            )
        )
        return rval

    return G1(x, xi=xi, phi=phi) if x < 0 else G2(x, xi=xi, phi=phi)


def bp_confidence(
    X: np.ndarray,
    res: np.ndarray,
    Vt_bp: np.ndarray,
    coeffs: np.ndarray,
    interval: int = 3,
):
    """
    Given a interval lenght for the breakpoints, returns the
    confidence of the interval (opposite behaviour of confint).

    This function is derived from the R confint function of package strucchangeRcpp
    https://github.com/bfast2/strucchangeRcpp/blob/master/R/breakpoints.R#L503

    NOTE: This version is modified to work with batches of pixels, with the
    batch dimension on the FIRST axis.

    Parameters
    ----------
    X : ndarray of floats
        Input dates in r-style format of shape (nobs, 2). First axis
        should be filled with ones (i.e., X is the design matrix).
    res : ndarray of floats
        Residuals of the segment-wise fitted models on the time series
        with shape (nobs, np), where nobs is the number of time steps
        and np the number of pixels to analyse.
    Vt_bp : ndarray of ints
        List of (single) trend breakpoint indices for each pixel with
        shape (np,).
    coeffs : ndarray of floats
        Coefficients (intercepts and slopes) of fitted models with shape (4,np).
        coeffs[:2] are the coefficients of the linear models fitted
        on the first segment of the time series (from 0 to breakpoint),
        whereas coeffs[2:] are the coefficients of the linear models
        fitted on the second segment of the time series (from breakpoint to nobs)
    interval : interval, default 3
        Width of the interval surrounding the breakpoint for which to
        estimate the confidence of the breakpoint estimation.

    Returns
    -------
    confidence : ndarray of floats
        Array of values in the interval (0,1) indicating the confidence
        of the predicted break. Where a break is not detected a confidence
        of 0 is returned.
    """
    nobs, n_pixels = X.shape[0], len(Vt_bp)

    res = res.T  # transposed to have N pixels on first axis, nobs on second axis
    sigma = (np.squeeze(res[:, np.newaxis, :] @ res[..., np.newaxis]) / nobs).reshape(-1)  # not hetereogeneous errors (sigma1 = sigma2 = sigma)

    Q_temp = np.repeat([X.T @ X], n_pixels, axis=0)
    Q1 = np.stack(
        [(X[:bp].T @ X[:bp]) / bp for bp in Vt_bp], axis=0
    )  # hetereogeneous regressors (Q1 != Q2)
    Q2 = (Q_temp - Q1 * Vt_bp.reshape(-1, 1, 1)) / (-(Vt_bp - nobs)).reshape(-1, 1, 1)  # hetereogeneous regressors (Q1 != Q2)

    delta = (coeffs[2:] - coeffs[:2]).T  # params of second segment model minus the params of first segment model

    Qprod1 = np.squeeze(delta[:, np.newaxis, :] @ Q1 @ delta[..., np.newaxis]).reshape(-1)
    Qprod2 = np.squeeze(delta[:, np.newaxis, :] @ Q2 @ delta[..., np.newaxis]).reshape(-1)

    xi = Qprod2 / Qprod1

    phi = np.sqrt(sigma)  # not hetereogeneous errors (phi1 = phi2 = phi)

    # n=interval-1 possible intervals around the breakpoint are tested. The
    # confidence of the interval with best balanced right-left probability
    # is returned (based on the fact that confint searches for intervals
    # with (1-level)/2) probability on both sides).
    a2_sum = np.zeros((interval - 1, n_pixels))
    a2_abs_diff = np.zeros((interval - 1, n_pixels))
    for j, (lb, ub) in enumerate(zip(range(1 - interval, 0), range(1, interval))):
        # NOTE: Shouldn't one of these use Qprod2? Following github implementation
        lower = lb * Qprod1 / sigma
        upper = ub * Qprod1 / sigma
        a2_lower = np.array(
            [
                pargmaxV(l, phi1=phi[i], phi2=phi[i], xi=xi[i])
                for i, l in enumerate(lower)  # loop over the pixels
            ]
        )
        a2_upper = np.array(
            [
                1 - pargmaxV(u, phi1=phi[i], phi2=phi[i], xi=xi[i])
                for i, u in enumerate(upper)  # loop over the pixels
            ]
        )
        a2_sum[j] = a2_lower + a2_upper
        a2_abs_diff[j] = np.abs(a2_lower + a2_upper)
    confidence = 1 - a2_sum[np.argmin(a2_abs_diff, axis=0), np.arange(n_pixels)]  # choose the interval with most symmetric probs
    return confidence


def bfast_cci(
    Yt: np.ndarray,
    ti: np.ndarray,
    frequency: int = 12,
    h: float = 0.15,
    season: str = "none",
    max_iter: int = 1,
    max_breaks: int = 1,
    level: float = 0.05,
    verbosity: int = 0,
):
    """
    Iterative break detection in seasonal and trend
    component of a time series. This implementation uses
    only a trend model, performs only 1 bfast iteration
    and detects at most 1 break. Also, cases with nodata in
    the time series are not considered. This python implementation
    partly follows the R implementation of bfast and
    strucchangeRcpp at https://github.com/bfast2/.


    NOTE: The code is written to process multiple time series
    simultaneously, given that they share the same parameters
    and time steps.

    Parameters
    ----------
    Yt : ndarray
        Input time series of shape (nobs, np), where nobs
        is the number of time steps and np the number of
        pixels to analyse.
    ti : ndarray
        Input dates in r-style format of shape (nobs, 1).
    frequency : int default=52
        The frequency for the seasonal model.
    h : float, default 0.15
        Float in the interval (0,1) specifying the
        bandwidth relative to the sample size in
        the MOSUM/ME monitoring processes.
    season : str default='none'
        The seasonal model used to fit the seasonal component and
        detect seasonal breaks (i.e. significant phenological change).
        There are three options: "dummy", "harmonic", or "none" where
        "dummy" is the model proposed in the first Remote Sensing of
        Environment paper and "harmonic" is the model used in the second
        Remote Sensing of Environment paper (See paper for more details)
        and where "none" indicates that no seasonal model will be fitted
        (i.e. St = 0 ). If there is no seasonal cycle (e.g. frequency of
        the time series is 1) "none" can be selected to avoid fitting a
        seasonal model.
        NOTE: Not yet implemented
    max_iter : int default=1
        Maximum amount of iterations allowed for estimation of breakpoints
        in seasonal and trend component.
        NOTE: Not yet implemented, only 1 iteration is currently performed.
    max_breaks : int default=1
        Integer specifying the maximal number of breaks to be calculated.
        NOTE: Not yet implemented, only 1 break is currently estimated.
    level : float, default 0.05
        Threshold value for the generalized fluctuation test (sctest) on
        the Empirical Fluctuation Process (EFP). Only the Ordinary Least
        Squares MOving SUM (OLS-MOSUM) is supported.
    verbosity : int, optional (default=0)
        The verbosity level (0=no output, 1=output)

    Returns
    -------
    bp_Vt : ndarray of ints
        Array of indices of the breaks detected. It has the
        a shape (np,). A value equals to -1 means no break detected
    confidence : ndarray of floats
        Array of values in the interval (0,1) indicating the confidence
        of the predicted break. Where a break is not detected a confidence
        of 0 is returned.
    """
    ti = sm.add_constant(ti)  # design matrix, a column of ones is added before the time steps (for intercept + slope linear models)
    nobs, n_pixels = Yt.shape
    Tt = np.full((nobs, n_pixels), np.nan)  # Initialize the Trend component

    if season == "harmonic":
        raise NotImplementedError("Seasonal models not yet implemented")
    elif season == "dummy":
        raise NotImplementedError("Seasonal models not yet implemented")
    elif season == "none":
        St = 0
    else:
        raise ValueError("Seasonal model is unknown, use 'harmonic', 'dummy' or 'none'")

    bp_Vt = np.full(n_pixels, -1)  # Initialize array of break indices

    #### First and only bfast iteration will follow

    with np.errstate(invalid="ignore"):
        Vt = Yt - St  # Deseasonalized Time series
    ### Change in trend component
    p_Vt = sctest(ti, Vt, h, verbosity=verbosity)  # tuple (stat,p_value)
    maybe_break = p_Vt[1] <= level  # mask for pixels with possible breaks

    if maybe_break.any():
        bp_Vt[maybe_break] = breakpoints(ti, Vt[:, maybe_break], h=h, verbosity=verbosity)
    nobp_Vt = bp_Vt == -1
    yesbp_Vt = np.logical_not(nobp_Vt)

    ## No Trend Change models
    fm0 = sm.OLS(Vt[:, nobp_Vt], ti).fit()
    prediction = fm0.predict()
    Tt[:, nobp_Vt] = (
        prediction
        if prediction.ndim != 1
        else prediction.reshape(
            -1, 1
        )  # statsmodel automatically squeeze the data, reverting this
    )

    ## Trend Change models
    part = breakfactor(bp_Vt[yesbp_Vt], nobs)
    X1 = partition_matrix(part, ti)
    fm1s = (
        [  # each pixel has a different partition matrix, thus each pixel needs its own model
            sm.OLS(Vt[..., pixel], X1[..., i], missing="drop").fit()
            for i, pixel in enumerate(np.where(yesbp_Vt)[0])
        ]
    )
    Tt[:, yesbp_Vt] = np.array([fm1.predict() for fm1 in fm1s]).T

    if season != "none":
        raise NotImplementedError("Seasonal models not yet implemented")

    with np.errstate(invalid="ignore"):
        Nt = Yt - Tt - St  # remainder

    ## UNUSED STAFF IS COMMENTED
    # output = [
    #     "Trend:\n{}\n\n".format(Tt[:, i])
    #     + "Season:\n{}\n\n".format(St[:,i])
    #     + "Remainder:\n{}\n\n".format(Nt[:, i])
    #     + "Trend Breakpoints:\n{}\n\n".format(Vt_bp[i])
    #     + "Season Breakpoints:\n{}\n".format(Wt_bp[i])
    #     for i in range(n_pixels)
    # ]

    ### Here ends the first and only bfast iteration

    # defaults for no breaks
    # m_x = np.full((2, n_pixels), np.nan)
    # m_y = np.full((2, n_pixels), np.nan)
    # Magnitude = np.zeros(n_pixels)
    # Time = np.full(n_pixels, np.nan)
    # Mag = np.full((1, 3, n_pixels), np.nan)

    # We compute the confidence as function of an interval width
    confidence = np.zeros(n_pixels)

    if fm1s:  # if empty do not process
        co = np.array([fm1.params for fm1 in fm1s]).T
        confidence[yesbp_Vt] = bp_confidence(ti, Nt[:, yesbp_Vt], bp_Vt[yesbp_Vt], co, interval=3)
        # y1 = co[0] + co[1] * ti[bp_Vt[yesbp_Vt], 1]
        # y2 = co[2] + co[3] * ti[bp_Vt[yesbp_Vt] + 1, 1]
        # Mag[r, 0, yesbp_Vt] = y1
        # Mag[r, 1, yesbp_Vt] = y2
        # Mag[r, 2, yesbp_Vt] = y2 - y1
        # m_x[:, yesbp_Vt] = np.array([bp_Vt[yesbp_Vt]] * 2)
        # m_y[:, yesbp_Vt] = Mag[0, :2][:, yesbp_Vt]  # Magnitude position
        # Magnitude[yesbp_Vt] = Mag[0, 2, yesbp_Vt]  # Magnitude of biggest change
        # Time[yesbp_Vt] = bp_Vt[yesbp_Vt]

    # t = Yt
    # nobp = (nobp_Vt, True)
    # magnitude = Magnitude
    # mags = Mag
    # time = Time
    # jump = ti[np.nan_to_num(m_x, -1).astype(int)], m_y

    return bp_Vt, confidence
