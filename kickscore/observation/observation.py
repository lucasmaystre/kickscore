import abc
import numpy as np

from math import log


class Observation(metaclass=abc.ABCMeta):

    def __init__(self, elems, t):
        assert len(elems) > 0, "need at least one item per observation"
        self._M = len(elems)
        self._items = np.zeros(self._M, dtype=object)
        self._coeffs = np.zeros(self._M, dtype=float)
        self._indices = np.zeros(self._M, dtype=int)
        self._ns_cav = np.zeros(self._M, dtype=float)
        self._xs_cav = np.zeros(self._M, dtype=float)
        for i, (item, coeff) in enumerate(elems):
            self._items[i] = item
            self._coeffs[i] = coeff
            self._indices[i] = item.fitter.add_sample(t)
        self.t = t
        self._logpart = 0

    @abc.abstractmethod
    def match_moments(self, mean_cav, cov_cav):
        """Compute statistics of the hybrid distribution."""

    @abc.abstractstaticmethod
    def probability(*args, **kwargs):
        """Compute the probability of the outcome described by `elems`."""

    def ep_update(self, damping=1.0, threshold=1e-4):
        # Mean and variance of the cavity distribution in function space.
        f_mean_cav = 0
        f_var_cav = 0
        for i in range(self._M):
            item = self._items[i]
            idx = self._indices[i]
            coeff = self._coeffs[i]
            # Compute the natural parameters of the cavity distribution.
            x_tot = 1.0 / item.fitter.vs[idx]
            n_tot = x_tot * item.fitter.ms[idx]
            x_cav = x_tot - item.fitter.xs[idx]
            n_cav = n_tot - item.fitter.ns[idx]
            self._xs_cav[i] = x_cav
            self._ns_cav[i] = n_cav
            # Adjust the function-space cavity mean & variance.
            f_mean_cav += coeff * n_cav / x_cav
            f_var_cav += coeff * coeff / x_cav
        # Moment matching.
        logpart, dlogpart, d2logpart = self.match_moments(
                f_mean_cav, f_var_cav)
        converged = True
        for i in range(self._M):
            item = self._items[i]
            idx = self._indices[i]
            coeff = self._coeffs[i]
            x_cav = self._xs_cav[i]
            n_cav = self._ns_cav[i]
            # Update the elements' parameters.
            denom = (1 + coeff * coeff * d2logpart / x_cav)
            x = -coeff * coeff * d2logpart / denom
            n = (coeff * (dlogpart - coeff * (n_cav / x_cav) * d2logpart)
                    / denom)
            if (abs(x - item.fitter.xs[idx]) > threshold
                    or abs(n - item.fitter.ns[idx]) > threshold):
                converged = False
            item.fitter.xs[idx] = ((1 - damping) * item.fitter.xs[idx]
                    + damping * x)
            item.fitter.ns[idx] = ((1 - damping) * item.fitter.ns[idx]
                    + damping * n)
        # Save log partition function value for the log-likelihood.
        self._logpart = logpart
        return converged

    @property
    def log_likelihood_contrib(self):
        """Contribution to the log-marginal likelihood of the model."""
        loglik = self._logpart
        for i in range(self._M):
            item = self._items[i]
            idx = self._indices[i]
            x_cav = self._xs_cav[i]
            n_cav = self._ns_cav[i]
            x = item.fitter.xs[idx]
            n = item.fitter.ns[idx]
            # Adding the contribution of the factor to the log-likelihood.
            loglik += (0.5 * log(x / x_cav + 1)
                    + (-n**2 - 2 * n * n_cav + x * n_cav**2 / x_cav)
                    / (2 * (x + x_cav)))
        return loglik
