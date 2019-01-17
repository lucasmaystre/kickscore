import numba
import numpy as np

from .observation import Observation
from .utils import normcdf
from math import log, sqrt, pi  # Faster than numpy equivalents.


@numba.jit(nopython=True)
def _mm_gaussian(mean_cav, var_cav, diff, var_obs):
    logpart = -0.5 * (log(2 * pi *(var_obs + var_cav))
            + (diff - mean_cav)**2 / (var_obs + var_cav))
    dlogpart = (diff - mean_cav) / (var_obs + var_cav)
    d2logpart = -1.0 / (var_obs + var_cav)
    return logpart, dlogpart, d2logpart


class GaussianObservation(Observation):

    def __init__(self, items, diff, var, t):
        super().__init__(items, t)
        self._diff = diff
        self._var = var

    def match_moments(self, mean_cav, var_cav):
        return _mm_gaussian(mean_cav, var_cav, self._diff, self._var)

    def cvi_expectations(self, mean, var):
        raise NotImplementedError

    @staticmethod
    def probability(items, threshold, var, t):
        m, v = Observation.f_params(items, t)
        return normcdf((m - threshold) / sqrt(var + v))
