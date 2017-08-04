import numpy as np

from .kernel import Kernel


ARRAY_ZERO = np.array([[0.]])
ARRAY_ONE = np.array([[1.]])
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

    def transition(self, delta):
        return ARRAY_ONE

    def noise_cov(self, delta):
        return ARRAY_ZERO

    def state_mean(self, t):
        return VEC_ZERO

    def state_cov(self, t):
        return self.var * ARRAY_ONE

    @property
    def measurement_vector(self):
        return VEC_ONE

    @property
    def feedback(self):
        return ARRAY_ZERO

    @property
    def noise_effect(self):
        return VEC_ONE

    @property
    def noise_density(self):
        return 0.0
