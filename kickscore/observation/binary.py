import functools

from .observation import Observation
from math import erfc, exp, log, pi, sqrt  # Faster than numpy equivalents.


class BinaryObservation(Observation):

    def __init__(self, winner, loser, t):
        self.t = t
        self._winner = winner
        self._wid = winner.fitter.add_sample(t)
        self._loser = loser
        self._lid = loser.fitter.add_sample(t)
        self._tau = 0
        self._nu = 0

    def ep_update(self, threshold=1e-4):
        # Mean and variance in function space.
        f_var = (self._winner.fitter.vars[self._wid]
                + self._loser.fitter.vars[self._lid])
        f_mean = (self._winner.fitter.means[self._wid]
                - self._loser.fitter.means[self._lid])
        # Cavity distribution.
        tau_tot = 1.0 / f_var
        nu_tot = tau_tot * f_mean
        tau_cav = tau_tot - self._tau
        nu_cav = nu_tot - self._nu
        cov_cav = 1.0 / tau_cav
        mean_cav = cov_cav * nu_cav
        # Moment matching.
        logpart, dlogpart, d2logpart = _match_moments_probit(mean_cav, cov_cav)
        # Update factor params in the function space.
        tau = -d2logpart / (1 + d2logpart / tau_cav)
        nu = ((dlogpart - (nu_cav / tau_cav) * d2logpart)
                 / (1 + d2logpart / tau_cav))
        # Update factor params in the weight space.
        self._winner.fitter.nus[self._wid] = +nu
        self._loser.fitter.nus[self._lid] = -nu
        self._winner.fitter.taus[self._wid] = tau
        self._loser.fitter.taus[self._lid] = tau
        # Check for convergence.
        converged = False
        if (abs(tau - self._tau) < threshold
                and abs(nu - self._nu) < threshold):
            converged = True
        # Save new parameters.
        self._nu = nu
        self._tau = tau
        return converged


# Some magic constants for a stable computation of logphi(z).
CS = [
  0.00048204, -0.00142906, 0.0013200243174, 0.0009461589032, -0.0045563339802,
  0.00556964649138, 0.00125993961762116, -0.01621575378835404,
  0.02629651521057465, -0.001829764677455021, 2*(1-pi/3), (4-pi)/3, 1, 1,]
RS = [
  1.2753666447299659525, 5.019049726784267463450, 6.1602098531096305441,
  7.409740605964741794425, 2.9788656263939928886,]
QS = [
  2.260528520767326969592, 9.3960340162350541504, 12.048951927855129036034,
  17.081440747466004316, 9.608965327192787870698, 3.3690752069827527677,]

SQRT2 = sqrt(2.0)
SQRT2PI = sqrt(2.0 * pi)


def _normcdf(x):
    """Normal cumulative density function."""
    # If X ~ N(0,1), returns P(X < x).
    return erfc(-x / SQRT2) / 2.0


def _logphi(z):
    # Adapted from the GPML function `logphi.m`.
    if z * z < 0.0492:
        # First case: z close to zero.
        coef = -z / SQRT2PI
        val = functools.reduce(lambda acc, c: coef * (c + acc), CS, 0)
        res = -2 * val - log(2)
        dres = exp(-(z * z) / 2 - res) / SQRT2PI
    elif z < -11.3137:
        # Second case: z very small.
        num = functools.reduce(lambda acc, r: -z * acc / SQRT2 + r, RS,
                     0.5641895835477550741)
        den = functools.reduce(lambda acc, q: -z * acc / SQRT2 + q, QS, 1.0)
        res = log(num / (2 * den)) - (z * z) / 2
        dres = abs(den / num) * sqrt(2.0 / pi)
    else:
        res = log(_normcdf(z))
        dres = exp(-(z * z) / 2 - res) / SQRT2PI
    return res, dres


def _match_moments_probit(mean_cav, cov_cav):
    # Adapted from the GPML function `likErf.m`.
    z = mean_cav / sqrt(1 + cov_cav)
    logpart, val = _logphi(z)
    dlogpart = val / sqrt(1 + cov_cav)  # 1st derivative w.r.t. mean.
    d2logpart = -val * (z + val) / (1 + cov_cav)
    return logpart, dlogpart, d2logpart
