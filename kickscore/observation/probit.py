import numba
import numpy as np

from .observation import Observation
from math import erfc, exp, log, pi, sqrt  # Faster than numpy equivalents.
from scipy.special import log_ndtr


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
def _normpdf(x):
    """Normal probability density function."""
    return exp(-x*x / 2.0) / SQRT2PI


@numba.jit(nopython=True)
def _normcdf(x):
    """Normal cumulative density function."""
    # If X ~ N(0,1), returns P(X < x).
    return erfc(-x / SQRT2) / 2.0


@numba.jit(nopython=True)
def _logphi(z):
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
        res = log(_normcdf(z))
        dres = exp(-(z * z) / 2 - res) / SQRT2PI
    return res, dres


@numba.jit(nopython=True)
def _match_moments_probit(mean_cav, cov_cav):
    # Adapted from the GPML function `likErf.m`.
    z = mean_cav / sqrt(1 + cov_cav)
    logpart, val = _logphi(z)
    dlogpart = val / sqrt(1 + cov_cav)  # 1st derivative w.r.t. mean.
    d2logpart = -val * (z + val) / (1 + cov_cav)
    return logpart, dlogpart, d2logpart


@numba.jit(nopython=True)
def _match_moments_probit_tie(mean_cav, cov_cav, margin):
    # TODO This is probably numerically unstable.
    denom = sqrt(1 + cov_cav)
    z1 = (mean_cav + margin) / denom
    z2 = (mean_cav - margin) / denom
    Phi1 = _normcdf(z1)
    Phi2 = _normcdf(z2)
    v1 = _normpdf(z1)
    v2 = _normpdf(z2)
    logpart = log(Phi1 - Phi2)
    dlogpart = (v1 - v2) / (denom * (Phi1 - Phi2))
    d2logpart = (-z1 * v1 + z2 * v2) / ((1 + cov_cav) * (Phi1 - Phi2)) - dlogpart**2
    return logpart, dlogpart, d2logpart


class ProbitObservation(Observation):

    def __init__(self, elems, t, margin=0):
        super().__init__(elems, t)
        self._margin = margin

    def match_moments(self, mean_cav, cov_cav):
        return _match_moments_probit(mean_cav - self._margin, cov_cav)

    def log_likelihood(self, x):
        return log_ndtr(x - self._margin)

    @staticmethod
    def probability(elems, t, margin=0):
        ts = np.array([t])
        m, v = 0.0, 0.0
        for item, coeff in elems:
            ms, vs = item.predict(ts)
            m += coeff * ms[0]
            v += coeff * coeff * vs[0]
        logpart, _, _ = _match_moments_probit(m - margin, v)
        return exp(logpart)


class ProbitTieObservation(Observation):

    def __init__(self, elems, t, margin):
        super().__init__(elems, t)
        self._margin = margin

    def match_moments(self, mean_cav, cov_cav):
        return _match_moments_probit_tie(mean_cav, cov_cav, self._margin)

    def log_likelihood(self, x):
        return log(_normcdf(x + self._margin) - _normcdf(x - self._margin))

    @staticmethod
    def probability(elems, t, margin):
        ts = np.array([t])
        m, v = 0.0, 0.0
        for item, coeff in elems:
            ms, vs = item.predict(ts)
            m += coeff * ms[0]
            v += coeff * coeff * vs[0]
        logpart, _, _ = _match_moments_probit_tie(m, v, margin)
        return exp(logpart)
