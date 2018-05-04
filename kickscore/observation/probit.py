import numba
import numpy as np

from .observation import Observation
from math import erfc, exp, log, pi, sqrt  # Faster than numpy equivalents.


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
def _match_moments_probit(em, ev, mm, mv):
    # Adapted from the GPML function `likErf.m`.
    z = (em - mm) / sqrt(1 + ev + mv)
    logpart, val = _logphi(z)
    dlogpart = val / sqrt(1 + ev + mv)  # 1st derivative w.r.t. elems mean.
    d2logpart = -val * (z + val) / (1 + ev + mv)
    return logpart, dlogpart, d2logpart, -dlogpart, d2logpart


@numba.jit(nopython=True)
def _match_moments_probit_tie(em, ev, mm1, mv1, mm2, mv2):
    # TODO This is probably numerically unstable.
    denom1 = sqrt(1 + ev + mv1)
    denom2 = sqrt(1 + ev + mv2)
    z1 = (em + mm1) / denom1
    z2 = (em - mm2) / denom2
    Phi1 = _normcdf(z1)
    Phi2 = _normcdf(z2)
    v1 = _normpdf(z1)
    v2 = _normpdf(z2)
    logpart = log(Phi1 - Phi2)
    dlogpart_e = (v1 / denom1 - v2 / denom2) / (Phi1 - Phi2)
    d2logpart_e = ((-z1 * v1 / (1 + ev + mv1) + z2 * v2 / (1 + ev + mv2))
            / (Phi1 - Phi2) - dlogpart_e * dlogpart_e)
    _, dlogpart_m1, d2logpart_m1, _, _ = _match_moments_probit(
            mm1, mv1, -em, ev)
    _, _, _, dlogpart_m2, d2logpart_m2 = _match_moments_probit(
            em, ev, mm2, mv2)
    return (logpart,
            dlogpart_e, d2logpart_e,
            dlogpart_m1, d2logpart_m1,
            -dlogpart_m2, -d2logpart_m2)


class ProbitWinObservation(Observation):

    def __init__(self, elems, margin, t, base_margin):
        self.base_margin = base_margin
        assert len(elems) > 0, "need at least one item per observation"
        self._M = [None, None]
        self._items = [None, None]
        self._coeffs = [None, None]
        self._indices = [None, None]
        self._ns_cav = [None, None]
        self._xs_cav = [None, None]
        for i, what in enumerate((elems, margin)):
            self._M[i] = len(what)
            self._items[i] = np.zeros(self._M[i], dtype=object)
            self._coeffs[i] = np.zeros(self._M[i], dtype=float)
            self._indices[i] = np.zeros(self._M[i], dtype=int)
            self._ns_cav[i] = np.zeros(self._M[i], dtype=float)
            self._xs_cav[i] = np.zeros(self._M[i], dtype=float)
            for j, (item, coeff) in enumerate(what):
                self._items[i][j] = item
                self._coeffs[i][j] = coeff
                self._indices[i][j] = item.fitter.add_sample(t)
        self.t = t
        self._logpart = 0

    @staticmethod
    def match_moments(em, ev, mm, mv):
        return _match_moments_probit(em, ev, mm, mv)

    @classmethod
    def probability(cls, elems, margin, t, base_margin):
        ts = np.array([t])
        em, ev = 0.0, 0.0
        for item, coeff in elems:
            ms, vs = item.predict(ts)
            em += coeff * ms[0]
            ev += coeff * coeff * vs[0]
        mm, mv = 0.0, 0.0
        for item, coeff in margin:
            ms, vs = item.predict(ts)
            mm += coeff * ms[0]
            mv += coeff * coeff * vs[0]
        logpart, _, _, _, _ = cls.match_moments(
                em, ev, mm + base_margin, mv)
        return exp(logpart)

    def ep_update(self, damping=1.0):
        f_mean_cav = [0, 0]
        f_var_cav = [0, 0]
        for i in (0, 1):
            # Mean and variance of the cavity distribution in function space.
            for j in range(self._M[i]):
                item = self._items[i][j]
                idx = self._indices[i][j]
                coeff = self._coeffs[i][j]
                # Compute the natural parameters of the cavity distribution.
                x_tot = 1.0 / item.fitter.vs[idx]
                n_tot = x_tot * item.fitter.ms[idx]
                x_cav = x_tot - item.fitter.xs[idx]
                n_cav = n_tot - item.fitter.ns[idx]
                self._xs_cav[i][j] = x_cav
                self._ns_cav[i][j] = n_cav
                # Adjust the function-space cavity mean & variance.
                f_mean_cav[i] += coeff * n_cav / x_cav
                f_var_cav[i] += coeff * coeff / x_cav
        dlp = [None, None]
        d2lp = [None, None]
        # Moment matching.
        logpart, dlp[0], d2lp[0], dlp[1], d2lp[1] = self.match_moments(
                f_mean_cav[0], f_var_cav[0],
                f_mean_cav[1] + self.base_margin, f_var_cav[1])
        for i in (0, 1):
            for j in range(self._M[i]):
                item = self._items[i][j]
                idx = self._indices[i][j]
                coeff = self._coeffs[i][j]
                x_cav = self._xs_cav[i][j]
                n_cav = self._ns_cav[i][j]
                # Update the elements' parameters.
                denom = (1 + coeff * coeff * d2lp[i] / x_cav)
                x = -coeff * coeff * d2lp[i] / denom
                n = (coeff * (dlp[i] - coeff * (n_cav / x_cav) * d2lp[i])
                        / denom)
                item.fitter.xs[idx] = ((1 - damping) * item.fitter.xs[idx]
                        + damping * x)
                item.fitter.ns[idx] = ((1 - damping) * item.fitter.ns[idx]
                        + damping * n)
        diff = abs(self._logpart - logpart)
        # Save log partition function value for the log-likelihood.
        self._logpart = logpart
        return diff

    @property
    def log_likelihood_contrib(self):
        """Contribution to the log-marginal likelihood of the model."""
        loglik = self._logpart
        for i in (0, 1):
            for j in range(self._M[i]):
                item = self._items[i][j]
                idx = self._indices[i][j]
                x_cav = self._xs_cav[i][j]
                n_cav = self._ns_cav[i][j]
                x = item.fitter.xs[idx]
                n = item.fitter.ns[idx]
                # Adding the contribution of the factor to the log-likelihood.
                loglik += (0.5 * log(x / x_cav + 1)
                        + (-n**2 - 2 * n * n_cav + x * n_cav**2 / x_cav)
                        / (2 * (x + x_cav)))
        return loglik


class ProbitTieObservation(Observation):

    def __init__(self, elems, margin1, margin2, t, base_margin):
        self.base_margin = base_margin
        assert len(elems) > 0, "need at least one item per observation"
        self._M = [None, None, None]
        self._items = [None, None, None]
        self._coeffs = [None, None, None]
        self._indices = [None, None, None]
        self._ns_cav = [None, None, None]
        self._xs_cav = [None, None, None]
        for i, what in enumerate((elems, margin1, margin2)):
            self._M[i] = len(what)
            self._items[i] = np.zeros(self._M[i], dtype=object)
            self._coeffs[i] = np.zeros(self._M[i], dtype=float)
            self._indices[i] = np.zeros(self._M[i], dtype=int)
            self._ns_cav[i] = np.zeros(self._M[i], dtype=float)
            self._xs_cav[i] = np.zeros(self._M[i], dtype=float)
            for j, (item, coeff) in enumerate(what):
                self._items[i][j] = item
                self._coeffs[i][j] = coeff
                self._indices[i][j] = item.fitter.add_sample(t)
        self.t = t
        self._logpart = 0

    @staticmethod
    def match_moments(em, ev, mm1, mv1, mm2, mv2):
        return _match_moments_probit_tie(em, ev, mm1, mv1, mm2, mv2)

    @classmethod
    def probability(cls, elems, margin1, margin2, t, base_margin):
        ts = np.array([t])
        em, ev = 0.0, 0.0
        for item, coeff in elems:
            ms, vs = item.predict(ts)
            em += coeff * ms[0]
            ev += coeff * coeff * vs[0]
        mm1, mv1 = 0.0, 0.0
        for item, coeff in margin1:
            ms, vs = item.predict(ts)
            mm1 += coeff * ms[0]
            mv1 += coeff * coeff * vs[0]
        mm2, mv2 = 0.0, 0.0
        for item, coeff in margin2:
            ms, vs = item.predict(ts)
            mm2 += coeff * ms[0]
            mv2 += coeff * coeff * vs[0]
        logpart, _, _, _, _, _, _ = cls.match_moments(
                em, ev,
                mm1 + base_margin, mv1,
                mm2 + base_margin, mv2)
        return exp(logpart)

    def ep_update(self, damping=1.0):
        f_mean_cav = [0, 0, 0]
        f_var_cav = [0, 0, 0]
        for i in (0, 1, 2):
            # Mean and variance of the cavity distribution in function space.
            for j in range(self._M[i]):
                item = self._items[i][j]
                idx = self._indices[i][j]
                coeff = self._coeffs[i][j]
                # Compute the natural parameters of the cavity distribution.
                x_tot = 1.0 / item.fitter.vs[idx]
                n_tot = x_tot * item.fitter.ms[idx]
                x_cav = x_tot - item.fitter.xs[idx]
                n_cav = n_tot - item.fitter.ns[idx]
                self._xs_cav[i][j] = x_cav
                self._ns_cav[i][j] = n_cav
                # Adjust the function-space cavity mean & variance.
                f_mean_cav[i] += coeff * n_cav / x_cav
                f_var_cav[i] += coeff * coeff / x_cav
        dlp = [None, None, None]
        d2lp = [None, None, None]
        # Moment matching.
        logpart, dlp[0], d2lp[0], dlp[1], d2lp[1], dlp[2], d2lp[2] = self.match_moments(
                f_mean_cav[0], f_var_cav[0],
                f_mean_cav[1] + self.base_margin, f_var_cav[1],
                f_mean_cav[2] + self.base_margin, f_var_cav[2])
        for i in (0, 1, 2):
            for j in range(self._M[i]):
                item = self._items[i][j]
                idx = self._indices[i][j]
                coeff = self._coeffs[i][j]
                x_cav = self._xs_cav[i][j]
                n_cav = self._ns_cav[i][j]
                # Update the elements' parameters.
                denom = (1 + coeff * coeff * d2lp[i] / x_cav)
                x = -coeff * coeff * d2lp[i] / denom
                n = (coeff * (dlp[i] - coeff * (n_cav / x_cav) * d2lp[i])
                        / denom)
                item.fitter.xs[idx] = ((1 - damping) * item.fitter.xs[idx]
                        + damping * x)
                item.fitter.ns[idx] = ((1 - damping) * item.fitter.ns[idx]
                        + damping * n)
        diff = abs(self._logpart - logpart)
        # Save log partition function value for the log-likelihood.
        self._logpart = logpart
        return diff

    @property
    def log_likelihood_contrib(self):
        """Contribution to the log-marginal likelihood of the model."""
        loglik = self._logpart
        for i in (0, 1, 2):
            for j in range(self._M[i]):
                item = self._items[i][j]
                idx = self._indices[i][j]
                x_cav = self._xs_cav[i][j]
                n_cav = self._ns_cav[i][j]
                x = item.fitter.xs[idx]
                n = item.fitter.ns[idx]
                # Adding the contribution of the factor to the log-likelihood.
                loglik += (0.5 * log(x / x_cav + 1)
                        + (-n**2 - 2 * n * n_cav + x * n_cav**2 / x_cav)
                        / (2 * (x + x_cav)))
        return loglik
