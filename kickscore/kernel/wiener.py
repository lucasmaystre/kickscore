import numpy as np
from numpy.typing import NDArray

from .kernel import Kernel

VEC_ZERO = np.zeros(1)
VEC_ONE = np.ones(1)
MAT_ZERO = np.zeros((1, 1))
MAT_ONE = np.ones((1, 1))


class Wiener(Kernel):
    """Kernel of a Wiener process.

    For convenience, it is also possible to specify an additional positive
    variance at t0 (this allows to circumvent some numerical issues).
    """

    def __init__(self, var: float, t0: float, var_t0: float = 0.0):
        self.var = var
        self.t0 = t0
        self.var_t0 = var_t0

    def k_mat(self, ts1: NDArray, ts2: NDArray | None = None) -> NDArray:
        if ts2 is None:
            ts2 = ts1
        ts1 = np.asarray(ts1)
        ts2 = np.asarray(ts2)
        return self.var * (np.fmin(ts1[:, None], ts2[None, :]) - self.t0) + self.var_t0

    def k_diag(self, ts: NDArray) -> NDArray:
        ts = np.asarray(ts)
        return self.var * (ts - self.t0) + self.var_t0

    @property
    def order(self) -> int:
        return 1

    def transition(self, t1: float, t2: float) -> NDArray:
        return MAT_ONE

    def noise_cov(self, t1: float, t2: float) -> NDArray:
        return self.var * (t2 - t1) * MAT_ONE

    def state_mean(self, t: float) -> NDArray:
        return VEC_ZERO

    def state_cov(self, t: float) -> NDArray:
        return (self.var * (t - self.t0) + self.var_t0) * MAT_ONE

    @property
    def measurement_vector(self) -> NDArray:
        return VEC_ONE

    @property
    def feedback(self) -> NDArray:
        return MAT_ZERO

    @property
    def noise_effect(self) -> NDArray:
        return MAT_ONE

    @property
    def noise_density(self) -> NDArray:
        return np.array([[self.var]])
