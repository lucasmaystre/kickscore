import abc
import math
import numpy as np

from abc import ABCMeta


class Kernel(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def compute(ts1, ts2=None):
        pass

    @staticmethod
    def distances(ts1, ts2):
        # mat[i, j] = |ts1[i] - ts2[j]|
        n_rows = len(ts1)
        n_cols = len(ts2)
        return np.abs(np.tile(ts1, (n_cols, 1)).T - np.tile(ts2, (n_rows, 1)))


class Gaussian(Kernel):

    def __init__(self, var, lscale):
        self._var = var
        self._lscale = lscale

    def compute(self, ts1, ts2=None):
        if ts2 is None:
            ts2 = ts1
        dist = Kernel.distances(ts1, ts2)
        return self._var * np.exp(-0.5 * (dist / self._lscale)**2)


class Matern52(Kernel):

    def __init__(self, var, lscale):
        self._var = var
        self._lscale = lscale

    def compute(self, ts1, ts2=None):
        if ts2 is None:
            ts2 = ts1
        r = Kernel.distances(ts1, ts2) / self._lscale
        sqrt5 = math.sqrt(5)
        return self._var * (1 + sqrt5 * r + (5/3) * r**2) * np.exp(-sqrt5 * r)
