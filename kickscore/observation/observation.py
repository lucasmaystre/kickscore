import abc
import numpy as np

from math import log


class Observation(metaclass=abc.ABCMeta):

    def __init__(self, elems, t):
        assert len(elems) > 0, "need at least one item per observation"
        self._M = len(elems)
        self._elems_item = np.zeros(self._M, dtype=object)
        self._elems_coeff = np.zeros(self._M, dtype=float)
        self._elems_idx = np.zeros(self._M, dtype=int)
        self._elems_tau_cav = np.zeros(self._M, dtype=float)
        self._elems_nu_cav = np.zeros(self._M, dtype=float)
        for i, (item, coeff) in enumerate(elems):
            idx = item.fitter.add_sample(t)
            self._elems_item[i] = item
            self._elems_idx[i] = idx
            self._elems_coeff[i] = coeff
        self.t = t
        self._logpart = 0

    @abc.abstractmethod
    def match_moments(self, mean_cav, cov_cav):
        """Compute statistics of the hybrid distribution."""

    def ep_update(self, threshold=1e-4):
        # Mean and variance of the cavity distribution in function space.
        f_mean_cav = 0
        f_var_cav = 0
        for i in range(self._M):
            item = self._elems_item[i]
            idx = self._elems_idx[i]
            coeff = self._elems_coeff[i]
            # Compute the natural parameters of the cavity distribution.
            tau_tot = 1.0 / item.fitter.vars[idx]
            nu_tot = tau_tot * item.fitter.means[idx]
            tau_cav = tau_tot - item.fitter.taus[idx]
            nu_cav = nu_tot - item.fitter.nus[idx]
            self._elems_tau_cav[i] = tau_cav
            self._elems_nu_cav[i] = nu_cav
            # Adjust the function-space cavity mean & variance.
            f_mean_cav += coeff * nu_cav / tau_cav
            f_var_cav += coeff * coeff / tau_cav
        # Moment matching.
        logpart, dlogpart, d2logpart = self.match_moments(
                f_mean_cav, f_var_cav)
        converged = True
        for i in range(self._M):
            item = self._elems_item[i]
            idx = self._elems_idx[i]
            coeff = self._elems_coeff[i]
            tau_cav = self._elems_tau_cav[i]
            nu_cav = self._elems_nu_cav[i]
            # Update the elements' parameters.
            denom = (1 + coeff * coeff * d2logpart / tau_cav)
            tau = -coeff * coeff * d2logpart / denom
            nu = (coeff * (dlogpart - coeff * (nu_cav / tau_cav) * d2logpart)
                    / denom)
            if (abs(tau - item.fitter.taus[idx]) > threshold
                    or abs(nu - item.fitter.nus[idx]) > threshold):
                converged = False
            item.fitter.taus[idx] = tau
            item.fitter.nus[idx] = nu
        # Save log partition function value for the log-likelihood.
        self._logpart = logpart
        return converged

    @property
    def log_likelihood_contrib(self):
        """Contribution to the log-marginal likelihood of the model."""
        loglik = self._logpart
        for i in range(self._M):
            item = self._elems_item[i]
            idx = self._elems_idx[i]
            tau_cav = self._elems_tau_cav[i]
            nu_cav = self._elems_nu_cav[i]
            tau = item.fitter.taus[idx]
            nu = item.fitter.nus[idx]
            # Adding the contribution of the factor to the log-likelihood.
            loglik += (0.5 * log(tau / tau_cav + 1)
                    + (-nu**2 - 2 * nu * nu_cav + tau * nu_cav**2 / tau_cav)
                    / (2 * (tau + tau_cav)))
        return loglik

