import numpy as np

from math import exp, sqrt, pi
from .kernel import Kernel


class PeriodicExponential(Kernel):

    def __init__(self, var, lscale, period):
        self.var = var
        self.lscale = lscale
        self.period = period

    def k_mat(self, ts1, ts2=None):
        if ts2 is None:
            ts2 = ts1
        r = 2 * np.abs(np.sin(Kernel.distances(ts1, ts2) * pi / self.period)) / self.lscale
        return self.var * np.exp(-r)

    def k_diag(self, ts):
        return self.var * np.ones(len(ts))

    @property
    def feedback(self):
        raise NotImplementedError()

    @property
    def measurement_vector(self):
        raise NotImplementedError()

    @property
    def noise_density(self):
        raise NotImplementedError()

    @property
    def noise_effect(self):
        raise NotImplementedError()

    @property
    def order(self):
        raise NotImplementedError()

    def state_cov(self):
        raise NotImplementedError()

    def state_mean(self):
        raise NotImplementedError()
