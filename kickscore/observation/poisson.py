from collections.abc import Sequence
from math import exp, log  # Faster than numpy equivalents.

import numba

from ..item import Item
from .observation import Observation
from .utils import cvi_expectations, iv, log_factorial, match_moments


@match_moments
@cvi_expectations
@numba.jit(nopython=True)
def _ll_poisson(x: float, count: int) -> float:
    """Compute the log-likelihood of x under the Poisson model."""
    return float(x * count - log_factorial(count) - exp(x))


class PoissonObservation(Observation):
    def __init__(self, items: Sequence[tuple[Item, float]], count: float, t: float):
        super().__init__(items, t)
        self._count = count

    def match_moments(self, mean_cav: float, var_cav: float) -> tuple[float, float, float]:
        return _ll_poisson.match_moments(mean_cav, var_cav, self._count)  # pyright: ignore[reportFunctionMemberAccess]

    def cvi_expectations(self, mean: float, var: float) -> tuple[float, float, float]:
        return _ll_poisson.cvi_expectations(mean, var, self._count)  # pyright: ignore[reportFunctionMemberAccess]

    @staticmethod
    def probability(items: Sequence[tuple[Item, float]], count: float, t: float) -> float:
        m, v = Observation.f_params(items, t)
        logpart, _, _ = _ll_poisson.match_moments(m, v, count)  # pyright: ignore[reportFunctionMemberAccess]
        return exp(logpart)


@match_moments
@cvi_expectations
@numba.jit(nopython=True)
def _ll_skellam(x: float, diff: int, base_rate: float) -> float:
    # TODO The call to `iv` slows this function down by a factor 30x.
    """Compute the log-likelihood of x under the Skellam model."""
    return (
        -(exp(x + base_rate) + exp(-x + base_rate))
        + x * diff
        + log(iv(abs(diff), 2 * exp(base_rate)))
    )


class SkellamObservation(Observation):
    def __init__(self, items: Sequence[tuple[Item, float]], diff: int, base_rate: float, t: float):
        super().__init__(items, t)
        self._diff = diff
        self._base_rate = base_rate

    def match_moments(self, mean_cav: float, var_cav: float) -> tuple[float, float, float]:
        return _ll_skellam.match_moments(mean_cav, var_cav, self._diff, self._base_rate)  # pyright: ignore[reportFunctionMemberAccess]

    def cvi_expectations(self, mean: float, var: float) -> tuple[float, float, float]:
        return _ll_skellam.cvi_expectations(mean, var, self._diff, self._base_rate)  # pyright: ignore[reportFunctionMemberAccess]

    @staticmethod
    def probability(
        items: Sequence[tuple[Item, float]], diff: int, base_rate: float, t: float
    ) -> float:
        m, v = Observation.f_params(items, t)
        logpart, _, _ = _ll_skellam.match_moments(m, v, diff, base_rate)  # pyright: ignore[reportFunctionMemberAccess]
        return exp(logpart)
