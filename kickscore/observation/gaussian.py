from collections.abc import Sequence
from math import log, pi, sqrt  # Faster than numpy equivalents.

import numba

from ..item import Item
from .observation import Observation
from .utils import normcdf


@numba.jit(nopython=True)
def _mm_gaussian(
    mean_cav: float,
    var_cav: float,
    diff: float,
    var_obs: float,
) -> tuple[float, float, float]:
    logpart = -0.5 * (
        log(2 * pi * (var_obs + var_cav)) + (diff - mean_cav) ** 2 / (var_obs + var_cav)
    )
    dlogpart = (diff - mean_cav) / (var_obs + var_cav)
    d2logpart = -1.0 / (var_obs + var_cav)
    return logpart, dlogpart, d2logpart


class GaussianObservation(Observation):
    def __init__(self, items: Sequence[tuple[Item, float]], diff: float, var: float, t: float):
        super().__init__(items, t)
        self._diff = diff
        self._var = var

    def match_moments(self, mean_cav: float, var_cav: float) -> tuple[float, float, float]:
        return _mm_gaussian(mean_cav, var_cav, self._diff, self._var)

    def cvi_expectations(self, mean: float, var: float) -> tuple[float, float, float]:
        raise NotImplementedError

    @staticmethod
    def probability(
        items: Sequence[tuple[Item, float]], threshold: float, var: float, t: float
    ) -> float:
        m, v = Observation.f_params(items, t)
        return normcdf((m - threshold) / sqrt(var + v))
