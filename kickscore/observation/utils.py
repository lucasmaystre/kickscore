import numba
import numpy as np

from math import erfc, exp, log, pi, sqrt  # Faster than numpy equivalents.
from scipy.special import roots_hermitenorm


# Some magic constants for a stable computation of logphi(z).
CS = np.array([
  0.00048204, -0.00142906, 0.0013200243174, 0.0009461589032, -0.0045563339802,
  0.00556964649138, 0.00125993961762116, -0.01621575378835404,
  0.02629651521057465, -0.001829764677455021, 2*(1-pi/3), (4-pi)/3, 1, 1,])
RS = np.array([
  1.2753666447299659525, 5.019049726784267463450, 6.1602098531096305441,
  7.409740605964741794425, 2.9788656263939928886,])
QS = np.array([
  2.260528520767326969592, 9.3960340162350541504, 12.048951927855129036034,
  17.081440747466004316, 9.608965327192787870698, 3.3690752069827527677,])

SQRT2 = sqrt(2.0)
SQRT2PI = sqrt(2.0 * pi)


@numba.jit(nopython=True)
def normpdf(x):
    """Normal probability density function."""
    return exp(-x*x / 2.0) / SQRT2PI


@numba.jit(nopython=True)
def normcdf(x):
    """Normal cumulative density function."""
    # If X ~ N(0,1), returns P(X < x).
    return erfc(-x / SQRT2) / 2.0


@numba.jit(nopython=True)
def logphi(z):
    """Compute the log of the normal cumulative density function."""
    # Adapted from the GPML function `logphi.m`.
    # We cannot use `scipy.special.log_ndtr` because of numba.
    if z * z < 0.0492:
        # First case: z close to zero.
        coef = -z / SQRT2PI
        val = 0
        for c in CS:
            val = coef * (c + val)
        res = -2 * val - log(2)
        dres = exp(-(z * z) / 2 - res) / SQRT2PI
    elif z < -11.3137:
        # Second case: z very small.
        num = 0.5641895835477550741
        for r in RS:
            num = -z * num / SQRT2 + r
        den = 1.0
        for q in QS:
            den = -z * den / SQRT2 + q
        res = log(num / (2 * den)) - (z * z) / 2
        dres = abs(den / num) * sqrt(2.0 / pi)
    else:
        res = log(normcdf(z))
        dres = exp(-(z * z) / 2 - res) / SQRT2PI
    return res, dres


@numba.jit(nopython=True)
def logsumexp(xs):
    a = np.max(xs)
    return a + log(np.sum(np.exp(xs - a)))


@numba.jit(nopython=True)
def logsumexp2(xs, bs):
    a = np.max(xs)
    return a + log(np.sum(bs * np.exp(xs - a)))


def cvi_expectations(ll_fct):
    """Add a function that computes the exp. log-lik. and its derivatives."""
    n = 30  # Order of Gauss-Hermite quadrature.
    xs, ws = roots_hermitenorm(n)
    @numba.jit(nopython=True)
    def integrals(mean, var, *args):
        std = sqrt(var)
        exp_ll, alpha, beta = 0.0, 0.0, 0.0
        for i in range(n):
            val = (ws[i] / SQRT2PI) * ll_fct(std*xs[i] + mean, *args)
            exp_ll += val
            alpha += (xs[i] / std) * val
            beta += ((xs[i]*xs[i] - 1) / (2*var)) * val
        return exp_ll, alpha, beta
    ll_fct.cvi_expectations = integrals
    return ll_fct


def match_moments(ll_fct):
    """Add a function that computes the log-part. fct and its derivatives."""
    n = 30  # Order of Gauss-Hermite quadrature.
    xs, ws = roots_hermitenorm(n)
    lws = np.log(ws) - log(SQRT2PI)
    @numba.jit(nopython=True)
    def integrals(mean, var, *args):
        std = sqrt(var)
        loglike = np.array([ll_fct(std*x + mean, *args) for x in xs])
        logpart = logsumexp(lws + loglike)
        vals = np.exp(lws + loglike - logpart)
        dlogpart = np.dot(vals, xs / std)
        d2logpart = np.dot(vals, (xs*xs - 1) / var) - dlogpart*dlogpart
        return logpart, dlogpart, d2logpart
    ll_fct.match_moments = integrals
    return ll_fct


_LF_CACHE = np.cumsum([0] + [log(i) for i in range(1, 500)])

@numba.jit(nopython=True)
def log_factorial(k):
    if k < len(_LF_CACHE):
        return _LF_CACHE[k]
    else:
        tot = _LF_CACHE[-1]
        for i in range(len(_LF_CACHE), k+1):
            tot += log(i)
        return tot
