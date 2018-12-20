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
        self._logpart = 0  # Value of log-partition function, used with EP.
        self._exp_ll = 0  # Expected log-likelihood, used with CVI.

    @abc.abstractmethod
    def match_moments(self, mean_cav, cov_cav):
        """Compute statistics of the hybrid distribution."""

    @abc.abstractmethod
    def cvi_expectations(self, mean, var):
        """Compute the expected log-likelihood and its derivatives."""

    @abc.abstractstaticmethod
    def probability(*args, **kwargs):
        """Compute the probability of the outcome described by `elems`."""

    def ep_update(self, lr=1.0):
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
            item.fitter.xs[idx] = (1 - lr) * item.fitter.xs[idx] + lr * x
            item.fitter.ns[idx] = (1 - lr) * item.fitter.ns[idx] + lr * n
        diff = abs(self._logpart - logpart)
        # Save log partition function value for the log-likelihood.
        self._logpart = logpart
        return diff

    def kl_update(self, lr=0.3):
        # Mean and variance in function space.
        f_mean = 0
        f_var = 0
        for i in range(self._M):
            item = self._items[i]
            idx = self._indices[i]
            coeff = self._coeffs[i]
            # Adjust the function-space mean & variance.
            f_mean += coeff * item.fitter.ms[idx]
            f_var += coeff * coeff * item.fitter.vs[idx]
        # Compute the derivatives of the exp. log-lik. w.r.t. mean parameters.
        exp_ll, alpha, beta = self.cvi_expectations(f_mean, f_var)
        for i in range(self._M):
            item = self._items[i]
            idx = self._indices[i]
            coeff = self._coeffs[i]
            # Update the elements' parameters.
            x = -2.0 * coeff * coeff * beta
            n = coeff * (alpha - 2 * item.fitter.ms[idx] * coeff * beta)
            item.fitter.xs[idx] = (1 - lr) * item.fitter.xs[idx] + lr * x
            item.fitter.ns[idx] = (1 - lr) * item.fitter.ns[idx] + lr * n
        diff = abs(self._exp_ll - exp_ll)
        # Save the expected log-likelihood.
        self._exp_ll = exp_ll
        return diff

    @property
    def ep_log_likelihood_contrib(self):
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

    @property
    def kl_log_likelihood_contrib(self):
        """Contribution to the log-marginal likelihood of the model."""
        return self._exp_ll

    @staticmethod
    def f_params(elems, t):
        """Compute function-space mean and variance."""
        ts = np.array([t])
        m, v = 0.0, 0.0
        for item, coeff in elems:
            ms, vs = item.predict(ts)
            m += coeff * ms[0]
            v += coeff * coeff * vs[0]
        return m, v
