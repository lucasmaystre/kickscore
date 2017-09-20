"""Abstract base class for score process fitters."""

import abc
import numpy as np


class Fitter(metaclass=abc.ABCMeta):

    def __init__(self, kernel):
        self.ts_new = list()
        self.kernel = kernel
        # Arrays to be allocated later.
        self.ts = np.zeros(0)
        self.means = np.zeros(0)
        self.vars = np.zeros(0)
        self.nus = np.zeros(0)
        self.taus = np.zeros(0)
        # State of the fitter.
        self.is_fitted = False

    def add_sample(self, t):
        idx = len(self.ts) + len(self.ts_new)
        self.ts_new.append(t)
        self.is_fitted = False
        return idx

    def allocate(self):
        n_new = len(self.ts_new)
        self.ts = np.concatenate((self.ts, self.ts_new))
        self.means = np.concatenate((self.means, np.zeros(n_new)))
        self.vars = np.concatenate(
                (self.vars, self.kernel.k_diag(self.ts_new)))
        self.nus = np.concatenate((self.nus, np.zeros(n_new)))
        self.taus = np.concatenate((self.taus, np.zeros(n_new)))
        # Clear the list of pending samples.
        self.ts_new = list()

    @property
    def is_allocated(self):
        return len(self.ts_new) == 0

    @property
    def posterior(self):
        if not self.is_fitted:
            raise RuntimeError("new data since last call to `fit()`")
        return (self.ts, self.means, self.vars)

    @abc.abstractmethod
    def fit(self):
        """Fit the score model given the available data."""

    @abc.abstractmethod
    def predict(self, ts):
        """Predict score at new time locations."""
