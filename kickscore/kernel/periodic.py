from math import pi

import numpy as np
from numpy.typing import NDArray

from .kernel import Kernel


class PeriodicExponential(Kernel):
    def __init__(self, var: float, lscale: float, period: float):
        self.var = var
        self.lscale = lscale
        self.period = period

    def k_mat(self, ts1: NDArray, ts2: NDArray | None = None) -> NDArray:
        if ts2 is None:
            ts2 = ts1
        r = 2 * np.abs(np.sin(Kernel.distances(ts1, ts2) * pi / self.period)) / self.lscale
        return self.var * np.exp(-r)

    def k_diag(self, ts: NDArray) -> NDArray:
        return self.var * np.ones(len(ts))

    @property
    def feedback(self) -> NDArray:
        raise NotImplementedError()

    @property
    def measurement_vector(self) -> NDArray:
        raise NotImplementedError()

    @property
    def noise_density(self) -> NDArray:
        raise NotImplementedError()

    @property
    def noise_effect(self) -> NDArray:
        raise NotImplementedError()

    @property
    def order(self) -> int:
        raise NotImplementedError()

    def state_cov(self, t: float) -> NDArray:
        raise NotImplementedError()

    def state_mean(self, t: float) -> NDArray:
        raise NotImplementedError()
