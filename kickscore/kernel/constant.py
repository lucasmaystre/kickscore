import numpy as np
from numpy.typing import NDArray

from .kernel import Kernel

MAT_ZERO = np.array([[0.0]])
MAT_ONE = np.array([[1.0]])
VEC_ZERO = np.array([0.0])
VEC_ONE = np.array([1.0])


class Constant(Kernel):
    def __init__(self, var: float):
        self.var = var

    def k_mat(self, ts1: NDArray, ts2: NDArray | None = None) -> NDArray:
        n = len(ts1)
        m = len(ts2) if ts2 is not None else len(ts1)
        return self.var * np.ones((n, m))

    def k_diag(self, ts: NDArray) -> NDArray:
        return self.var * np.ones(len(ts))

    @property
    def order(self) -> int:
        return 1

    def transition(self, t1: float, t2: float) -> NDArray:
        return MAT_ONE

    def noise_cov(self, t1: float, t2: float) -> NDArray:
        return MAT_ZERO

    def state_mean(self, t: float) -> NDArray:
        return VEC_ZERO

    def state_cov(self, t: float) -> NDArray:
        return self.var * MAT_ONE

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
        return MAT_ZERO


class PiecewiseConstant(Kernel):
    def __init__(self, var: float, bounds: NDArray):
        self.var = var
        self.bounds = np.sort(bounds)

    def k_mat(self, ts1: NDArray, ts2: NDArray | None = None) -> NDArray:
        if ts2 is None:
            ts2 = ts1
        idx1 = np.searchsorted(self.bounds, ts1)
        idx2 = np.searchsorted(self.bounds, ts2)
        return self.var * np.equal.outer(idx1, idx2).astype(float)

    def k_diag(self, ts: NDArray) -> NDArray:
        return self.var * np.ones(len(ts))

    @property
    def order(self) -> int:
        return 1

    def transition(self, t1: float, t2: float) -> NDArray:
        idx1 = np.searchsorted(self.bounds, t1)
        idx2 = np.searchsorted(self.bounds, t2)
        if idx1 == idx2:
            return MAT_ONE
        else:
            return MAT_ZERO

    def noise_cov(self, t1: float, t2: float) -> NDArray:
        idx1 = np.searchsorted(self.bounds, t1)
        idx2 = np.searchsorted(self.bounds, t2)
        if idx1 == idx2:
            return MAT_ZERO
        else:
            return self.var * MAT_ONE

    def state_mean(self, t: float) -> NDArray:
        return VEC_ZERO

    def state_cov(self, t: float) -> NDArray:
        return self.var * MAT_ONE

    @property
    def measurement_vector(self) -> NDArray:
        return VEC_ONE

    @property
    def feedback(self) -> NDArray:
        raise NotImplementedError()

    @property
    def noise_effect(self) -> NDArray:
        raise NotImplementedError()

    @property
    def noise_density(self) -> NDArray:
        raise NotImplementedError()
