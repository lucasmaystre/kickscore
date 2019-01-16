import numpy as np

from .kernel import Kernel


MAT_ZERO = np.array([[0.]])
MAT_ONE = np.array([[1.]])
VEC_ZERO = np.array([0.])
VEC_ONE = np.array([1.])


class Constant(Kernel):

    def __init__(self, var):
        self.var = var

    def k_mat(self, ts1, ts2=None):
        n = len(ts1)
        m = len(ts2) if ts2 is not None else len(ts1)
        return self.var * np.ones((n, m))

    def k_diag(self, ts):
        return self.var * np.ones(len(ts))

    @property
    def order(self):
        return 1

    def transition(self, t1, t2):
        return MAT_ONE

    def noise_cov(self, t1, t2):
        return MAT_ZERO

    def state_mean(self, t):
        return VEC_ZERO

    def state_cov(self, t):
        return self.var * MAT_ONE

    @property
    def measurement_vector(self):
        return VEC_ONE

    @property
    def feedback(self):
        return MAT_ZERO

    @property
    def noise_effect(self):
        return MAT_ONE

    @property
    def noise_density(self):
        return MAT_ZERO


class PiecewiseConstant(Kernel):

    def __init__(self, var, bounds):
        self.var = var
        self.bounds = np.sort(bounds)

    def k_mat(self, ts1, ts2=None):
        if ts2 is None:
            ts2 = ts1
        idx1 = np.searchsorted(self.bounds, ts1)
        idx2 = np.searchsorted(self.bounds, ts2)
        return self.var * np.equal.outer(idx1, idx2).astype(float)

    def k_diag(self, ts):
        return self.var * np.ones(len(ts))

    @property
    def order(self):
        return 1

    def transition(self, t1, t2):
        idx1 = np.searchsorted(self.bounds, t1)
        idx2 = np.searchsorted(self.bounds, t2)
        if idx1 == idx2:
            return MAT_ONE
        else:
            return MAT_ZERO

    def noise_cov(self, t1, t2):
        idx1 = np.searchsorted(self.bounds, t1)
        idx2 = np.searchsorted(self.bounds, t2)
        if idx1 == idx2:
            return MAT_ZERO
        else:
            return self.var * MAT_ONE

    def state_mean(self, t):
        return VEC_ZERO

    def state_cov(self, t):
        return self.var * MAT_ONE

    @property
    def measurement_vector(self):
        return VEC_ONE

    @property
    def feedback(self):
        raise NotImplementedError()

    @property
    def noise_effect(self):
        raise NotImplementedError()

    @property
    def noise_density(self):
        raise NotImplementedError()
