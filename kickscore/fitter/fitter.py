"""Abstract base class for score process fitters."""

import abc

import numpy as np
from numpy.typing import NDArray

from ..kernel import Kernel


class Fitter(metaclass=abc.ABCMeta):
    def __init__(self, kernel: Kernel):
        self.ts_new = list()
        self.kernel = kernel
        # Arrays to be allocated later.
        self.ts = np.zeros(0)  # Timestamps.
        self.ms = np.zeros(0)  # Means.
        self.vs = np.zeros(0)  # Variances.
        self.ns = np.zeros(0)  # Precision-adjusted means of pseudo-obs.
        self.xs = np.zeros(0)  # Precision of pseudo-obs.
        # State of the fitter.
        self.is_fitted = True  # Zero samples -> model is "fitted".

    def add_sample(self, t: float) -> int:
        idx = len(self.ts) + len(self.ts_new)
        self.ts_new.append(t)
        self.is_fitted = False
        return idx

    def allocate(self) -> None:
        n_new = len(self.ts_new)
        zeros = np.zeros(n_new)
        self.ts = np.concatenate((self.ts, self.ts_new))
        self.ms = np.concatenate((self.ms, zeros))
        self.vs = np.concatenate((self.vs, self.kernel.k_diag(np.asarray(self.ts_new))))
        self.ns = np.concatenate((self.ns, zeros))
        self.xs = np.concatenate((self.xs, zeros))
        # Clear the list of pending samples.
        self.ts_new = list()

    @property
    def is_allocated(self) -> bool:
        return len(self.ts_new) == 0

    @property
    def posterior(self) -> tuple[NDArray, NDArray, NDArray]:
        if not self.is_fitted:
            raise RuntimeError("new data since last call to `fit()`")
        return (self.ts, self.ms, self.vs)

    @abc.abstractmethod
    def fit(self) -> None:
        """Fit the score model given the available data."""

    @property
    @abc.abstractmethod
    def ep_log_likelihood_contrib(self) -> float:
        """Contribution to the log marginal likelihood of the model."""

    @property
    @abc.abstractmethod
    def kl_log_likelihood_contrib(self) -> float:
        """Contribution to the log marginal likelihood of the model."""

    @abc.abstractmethod
    def predict(self, ts: NDArray) -> tuple[NDArray, NDArray]:
        """Predict score at new time locations."""
