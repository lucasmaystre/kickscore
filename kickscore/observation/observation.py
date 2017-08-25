import abc

from math import log


class Observation(metaclass=abc.ABCMeta):

    def __init__(self, winners, losers, t):
        self._winners = list()
        for item, coeff in winners.items():
            idx = item.fitter.add_sample(t)
            self._winners.append((item, idx, coeff))
        self._losers = list()
        for item, coeff in losers.items():
            idx = item.fitter.add_sample(t)
            self._losers.append((item, idx, coeff))
        self.t = t
        self._tau = 0
        self._nu = 0
        self._logpart = 0
        self._mean_cav = 0
        self._cov_cav = 0

    @abc.abstractmethod
    def match_moments(self, mean_cav, cov_cav):
        pass

    def ep_update(self, threshold=1e-4):
        # Mean and variance in function space.
        f_mean = 0
        f_var = 0
        for item, idx, coeff in self._winners:
            f_mean += coeff * item.fitter.means[idx]
            f_var += coeff * item.fitter.vars[idx]
        for item, idx, coeff in self._losers:
            f_mean -= coeff * item.fitter.means[idx]
            f_var += coeff * item.fitter.vars[idx]
        # Cavity distribution.
        tau_tot = 1.0 / f_var
        nu_tot = tau_tot * f_mean
        tau_cav = tau_tot - self._tau
        nu_cav = nu_tot - self._nu
        cov_cav = 1.0 / tau_cav
        mean_cav = cov_cav * nu_cav
        # Moment matching.
        logpart, dlogpart, d2logpart = self.match_moments(mean_cav, cov_cav)
        # Update factor params in the function space.
        tau = -d2logpart / (1 + d2logpart / tau_cav)
        nu = ((dlogpart - (nu_cav / tau_cav) * d2logpart)
                 / (1 + d2logpart / tau_cav))
        # Update factor params in the weight space.
        for item, idx, coeff in self._winners:
            item.fitter.nus[idx] = +coeff * nu
            item.fitter.taus[idx] = (coeff * coeff) * tau
        for item, idx, coeff in self._losers:
            item.fitter.nus[idx] = -coeff * nu
            item.fitter.taus[idx] = (coeff * coeff) * tau
        # Check for convergence.
        converged = False
        if (abs(tau - self._tau) < threshold
                and abs(nu - self._nu) < threshold):
            converged = True
        # Save the new parameters & info for the log-likelihood.
        self._nu = nu
        self._tau = tau
        self._logpart = logpart
        self._mean_cav = mean_cav
        self._cov_cav = cov_cav
        return converged

    @property
    def log_likelihood_contrib(self):
        """Contribution to the log-marginal likelihood of the model."""
        nu, tau = self._nu, self._tau
        m, s = self._mean_cav, self._cov_cav
        return (self._logpart + 0.5 * log(tau * s + 1)
                + (-nu**2 * s - 2 * nu * m + tau * m**2) / (2 * (tau * s + 1)))

