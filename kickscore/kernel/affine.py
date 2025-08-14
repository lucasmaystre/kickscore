import numpy as np
from numpy.typing import NDArray

from .kernel import Kernel

NOISE_COV = np.zeros((2, 2))
MEASUREMENT_VECTOR = np.array([1.0, 0.0])
FEEDBACK = np.array([[0.0, 1.0], [0.0, 0.0]])
NOISE_EFFECT = np.transpose([[0.0, 1.0]])
STATE_MEAN = np.zeros(2)


class Affine(Kernel):
    """Affine kernel.

    A combination of a linear kernel with a constant offset. The offset leads
    to a positive-definite state covariance matrix, which avoids numerical
    issues for SSM inference.
    """

    def __init__(self, var_offset: float, var_slope: float, t0: float):
        self.var_offset = var_offset
        self.var_slope = var_slope
        self.t0 = t0

    def k_mat(self, ts1: NDArray, ts2: NDArray | None = None) -> NDArray:
        if ts2 is None:
            ts2 = ts1
        ts1 = np.asarray(ts1)
        ts2 = np.asarray(ts2)
        k_linear = self.var_slope * np.outer(ts1 - self.t0, ts2 - self.t0)
        return k_linear + self.var_offset * np.ones_like(k_linear)

    def k_diag(self, ts: NDArray) -> NDArray:
        ts = np.asarray(ts)
        return self.var_slope * np.square(ts - self.t0) + self.var_offset * np.ones_like(ts)

    @property
    def order(self) -> int:
        return 2

    def transition(self, t1: float, t2: float) -> NDArray:
        return np.array([[1.0, t2 - t1], [0.0, 1.0]])

    def noise_cov(self, t1: float, t2: float) -> NDArray:
        return NOISE_COV

    def state_mean(self, t: float) -> NDArray:
        return STATE_MEAN

    def state_cov(self, t: float) -> NDArray:
        t = t - self.t0
        return self.var_slope * np.array([[t * t, t], [t, 1]]) + self.var_offset * np.array(
            [[1.0, 0.0], [0.0, 0.0]]
        )

    @property
    def measurement_vector(self) -> NDArray:
        return MEASUREMENT_VECTOR

    @property
    def feedback(self) -> NDArray:
        return FEEDBACK

    @property
    def noise_effect(self) -> NDArray:
        return NOISE_EFFECT

    @property
    def noise_density(self) -> NDArray:
        return np.zeros((1, 1))
