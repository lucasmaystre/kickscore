import numpy as np

from math import exp, sqrt
from .kernel import Kernel


VEC_ZERO = np.zeros(1)
VEC_ONE = np.ones(1)
ARRAY_ONE = np.ones((1, 1))


class Exponential(Kernel):

    def __init__(self, var, lscale):
        self.var = var
        self.lscale = lscale

    def k_mat(self, ts1, ts2=None):
        if ts2 is None:
            ts2 = ts1
        r = Kernel.distances(ts1, ts2) / self.lscale
        return self.var * np.exp(-r)

    def k_diag(self, ts):
        return self.var * np.ones(len(ts))

    @property
    def order(self):
        return 1

    def transition(self, delta):
        return exp(-delta / self.lscale) * ARRAY_ONE

    def noise_cov(self, delta):
        return self.var * (1 - exp(-2 * delta / self.lscale)) * ARRAY_ONE

    def state_mean(self, t):
        return VEC_ZERO

    def state_cov(self, t):
        return self.var * ARRAY_ONE

    @property
    def measurement_vector(self):
        return VEC_ONE

    @property
    def feedback(self):
        return (-1 / self.lscale) * ARRAY_ONE

    @property
    def noise_effect(self):
        return VEC_ONE

    @property
    def noise_density(self):
        return 2 * self.var / self.lscale

    @property
    def stationary_mean(self):
        return VEC_ZERO

    @property
    def stationary_cov(self):
        return self.var * ARRAY_ONE
