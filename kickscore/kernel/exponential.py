from math import exp

import numpy as np
from numpy.typing import NDArray

from .kernel import Kernel

VEC_ZERO = np.zeros(1)
VEC_ONE = np.ones(1)
MAT_ONE = np.ones((1, 1))


class Exponential(Kernel):
    def __init__(self, var: float, lscale: float):
        self.var = var
        self.lscale = lscale

    def k_mat(self, ts1: NDArray, ts2: NDArray | None = None) -> NDArray:
        if ts2 is None:
            ts2 = ts1
        r = Kernel.distances(ts1, ts2) / self.lscale
        return self.var * np.exp(-r)

    def k_diag(self, ts: NDArray) -> NDArray:
        return self.var * np.ones(len(ts))

    @property
    def order(self) -> int:
        return 1

    def transition(self, t1: float, t2: float) -> NDArray:
        return exp(-(t2 - t1) / self.lscale) * MAT_ONE

    def noise_cov(self, t1: float, t2: float) -> NDArray:
        return self.var * (1 - exp(-2 * (t2 - t1) / self.lscale)) * MAT_ONE

    def state_mean(self, t: float) -> NDArray:
        return VEC_ZERO

    def state_cov(self, t: float) -> NDArray:
        return self.var * MAT_ONE

    @property
    def measurement_vector(self) -> NDArray:
        return VEC_ONE

    @property
    def feedback(self) -> NDArray:
        return (-1 / self.lscale) * MAT_ONE

    @property
    def noise_effect(self) -> NDArray:
        return MAT_ONE

    @property
    def noise_density(self) -> NDArray:
        return np.array([[2 * self.var / self.lscale]])

    @property
    def stationary_mean(self) -> NDArray:
        return VEC_ZERO

    @property
    def stationary_cov(self) -> NDArray:
        return self.var * MAT_ONE
