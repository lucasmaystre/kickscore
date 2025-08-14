from math import exp, sqrt

import numpy as np
from numpy.typing import NDArray

from .kernel import Kernel

MEASUREMENT_VECTOR = np.array([1.0, 0.0])
NOISE_EFFECT = np.transpose([[0.0, 1.0]])
STATIONARY_MEAN = np.zeros(2)


class Matern32(Kernel):
    def __init__(self, var: float, lscale: float):
        self.var = var
        self.lscale = lscale
        self.lambda_ = sqrt(3) / lscale

    def k_mat(self, ts1: NDArray, ts2: NDArray | None = None) -> NDArray:
        if ts2 is None:
            ts2 = ts1
        r = Kernel.distances(ts1, ts2) / self.lscale
        sqrt3 = sqrt(3)
        return self.var * (1 + sqrt3 * r) * np.exp(-sqrt3 * r)

    def k_diag(self, ts: NDArray) -> NDArray:
        return self.var * np.ones(len(ts))

    @property
    def order(self) -> int:
        return 2

    def transition(self, t1: float, t2: float) -> NDArray:
        d = t2 - t1
        a = self.lambda_
        A = np.array(
            [
                [d * a + 1, d],
                [-d * a * a, 1 - d * a],
            ]
        )
        return exp(-d * a) * A

    def noise_cov(self, t1: float, t2: float) -> NDArray:
        d = t2 - t1
        a = self.lambda_
        da = d * a
        c = exp(-2 * da)
        x11 = 1 - c * (2 * da * da + 2 * da + 1)
        x12 = c * (2 * da * da * a)
        x22 = a * a * (1 - c * (2 * da * da - 2 * da + 1))
        mat = np.array(
            [
                [x11, x12],
                [x12, x22],
            ]
        )
        return self.var * mat

    def state_mean(self, t: float) -> NDArray:
        return STATIONARY_MEAN

    def state_cov(self, t: float) -> NDArray:
        return self.stationary_cov

    @property
    def measurement_vector(self) -> NDArray:
        return MEASUREMENT_VECTOR

    @property
    def feedback(self) -> NDArray:
        a = self.lambda_
        mat = np.array(
            [
                [0, 1],
                [-(a**2), -2 * a],
            ]
        )
        return mat

    @property
    def noise_effect(self) -> NDArray:
        return NOISE_EFFECT

    @property
    def noise_density(self) -> NDArray:
        return np.array([[4 * self.var * self.lambda_**3]])

    @property
    def stationary_mean(self) -> NDArray:
        return STATIONARY_MEAN

    @property
    def stationary_cov(self) -> NDArray:
        a = self.lambda_
        mat = np.array(
            [
                [1, 0],
                [0, a * a],
            ]
        )
        return self.var * mat
