import numba
import numpy as np

from .observation import Observation
from .utils import log_factorial, cvi_expectations, match_moments
from math import exp  # Faster than numpy equivalents.


@match_moments
@cvi_expectations
@numba.jit(nopython=True)
def _ll_poisson(x, count):
    """Compute the log-likelihood of x under the poisson model."""
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
