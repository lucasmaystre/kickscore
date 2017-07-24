import numpy as np

from math import exp, sqrt
from .kernel import Kernel


class Matern52(Kernel):

    def __init__(self, var, lscale):
        self.var = var
        self.lscale = lscale

    def k_mat(self, ts1, ts2=None, diag=False):
        if ts2 is None:
            ts2 = ts1
        r = Kernel.distances(ts1, ts2) / self.lscale
        sqrt5 = sqrt(5)
        return self.var * (1 + sqrt5 * r + (5/3) * r**2) * np.exp(-sqrt5 * r)

    def k_diag(self, ts):
        return self.var * np.ones(len(ts))
