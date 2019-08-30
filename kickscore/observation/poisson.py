import numba
import numpy as np

from .observation import Observation
from .utils import log_factorial, iv, cvi_expectations, match_moments
from math import exp, log  # Faster than numpy equivalents.


@match_moments
@cvi_expectations
@numba.jit(nopython=True)
def _ll_poisson(x, count):
    """Compute the log-likelihood of x under the Poisson model."""
    return x * count - log_factorial(count) - exp(x)


class PoissonObservation(Observation):

    def __init__(self, items, count, t):
        super().__init__(items, t)
        self._count = count

    def match_moments(self, mean_cav, var_cav):
        return _ll_poisson.match_moments(mean_cav, var_cav, self._count)

    def cvi_expectations(self, mean, var):
        return _ll_poisson.cvi_expectations(mean, var, self._count)

    @staticmethod
    def probability(items, count, t):
        m, v = Observation.f_params(items, t)
        logpart, _, _ = _ll_poisson.match_moments(m, v, count)
        return exp(logpart)


@match_moments
@cvi_expectations
@numba.jit(nopython=True)
def _ll_skellam(x, diff, base_rate):
    # TODO The call to `iv` slows this function down by a factor 30x.
    """Compute the log-likelihood of x under the Skellam model."""
    return (-(exp(x + base_rate) + exp(-x + base_rate)) + x * diff
            + log(iv(abs(diff), 2 * exp(base_rate))))


class SkellamObservation(Observation):

    def __init__(self, items, diff, base_rate, t):
        super().__init__(items, t)
        self._diff = diff
        self._base_rate = base_rate

    def match_moments(self, mean_cav, var_cav):
        return _ll_skellam.match_moments(
                mean_cav, var_cav, self._diff, self._base_rate)

    def cvi_expectations(self, mean, var):
        return _ll_skellam.cvi_expectations(
                mean, var, self._diff, self._base_rate)

    @staticmethod
    def probability(items, diff, base_rate, t):
        m, v = Observation.f_params(items, t)
        logpart, _, _ = _ll_skellam.match_moments(m, v, diff, base_rate)
        return exp(logpart)
