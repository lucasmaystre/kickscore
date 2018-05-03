import abc
import numpy as np

from math import log, exp


class Observation(metaclass=abc.ABCMeta):

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

    @abc.abstractstaticmethod
    def match_moments(em, ev, mm, mv):
        """Compute statistics of the hybrid distribution."""

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
