import abc

import numpy as np
import scipy as sp
from numpy.typing import NDArray
from scipy.linalg import block_diag


class Kernel(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def k_mat(self, ts1: NDArray, ts2: NDArray | None = None) -> NDArray:
        """Compute the covariance matrix."""

    @abc.abstractmethod
    def k_diag(self, ts: NDArray) -> NDArray:
        """Compute the variances (diagonal of the covariance matrix)."""

    @property
    @abc.abstractmethod
    def order(self) -> int:
        """Order of the SDE :math:`m`."""

    @abc.abstractmethod
    def state_mean(self, t: float) -> NDArray:
        r"""Prior mean of the state vector, :math:`\mathbf{m}_0(t)`."""

    @abc.abstractmethod
    def state_cov(self, t: float) -> NDArray:
        r"""Prior covariance of the state vector, :math:`\mathbf{P}_0(t)`."""

    @property
    @abc.abstractmethod
    def measurement_vector(self) -> NDArray:
        r"""Measurement vector :math:`\mathbf{h}`."""

    @property
    @abc.abstractmethod
    def feedback(self) -> NDArray:
        r"""Feedback matrix :math:`\mathbf{F}`."""

    @property
    @abc.abstractmethod
    def noise_effect(self) -> NDArray:
        r"""Noise effect matrix :math:`\mathbf{L}`."""

    @property
    @abc.abstractmethod
    def noise_density(self) -> NDArray:
        r"""Power spectral density of the noise :math:`\mathbf{Q}`."""
        # Note: usually a scalar, except for combination kernels (e.g., Add).

    def transition(self, t1: float, t2: float) -> NDArray:
        r"""Transition matrix :math:`\mathbf{A}` for a given time interval.

        Note that this default implementation assumes that the feedback matrix
        is independent of time.
        """
        F = self.feedback
        return sp.linalg.expm(F * (t2 - t1))

    def noise_cov(self, t1: float, t2: float) -> NDArray:
        r"""Noise covariance matrix :math:`\mathbf{Q}` for a given time interval.

        Note that this default implementations assumes that the feedback
        matrix, the noise density and the noise effect are independent of time.
        """
        # Solution via matrix fraction decomposition, see:
        # - <https://github.com/SheffieldML/GPy/blob/devel/GPy/models/state_space.py#L715>
        # - Särkka's thesis (2006).
        mat = self.noise_effect.dot(self.noise_density).dot(self.noise_effect.T)
        # print(g)
        print(mat)
        Phi = np.vstack(
            (np.hstack((self.feedback, mat)), np.hstack((np.zeros_like(mat), -self.feedback.T)))
        )
        print(Phi)
        m = self.order
        AB = np.dot(sp.linalg.expm(Phi * (t2 - t1)), np.eye(2 * m, m, k=-m))
        print(AB)
        return sp.linalg.solve(AB[m:, :].T, AB[:m, :].T)

    def __add__(self, other: "Kernel") -> "Kernel":
        return Add(self, other)

    @staticmethod
    def distances(ts1: NDArray, ts2: NDArray) -> NDArray:
        # mat[i, j] = |ts1[i] - ts2[j]|
        return np.abs(ts1[:, np.newaxis] - ts2[np.newaxis, :])

    def simulate(self, ts: NDArray) -> NDArray:
        """Sample from a Gaussian process with the corresponding kernel."""
        ts = np.sort(ts)
        xs = np.zeros((len(ts), self.order))
        mean = self.state_mean(ts[0])
        cov = self.state_cov(ts[0])
        xs[0, :] = np.random.multivariate_normal(mean, cov)
        for i in range(1, len(ts)):
            mean = np.dot(self.transition(ts[i - 1], ts[i]), xs[i - 1])
            cov = self.noise_cov(ts[i - 1], ts[i])
            xs[i, :] = np.random.multivariate_normal(mean, cov)
        return np.dot(xs, self.measurement_vector)


class Add(Kernel):
    def __init__(self, first: Kernel, second: Kernel):
        self.parts: list[Kernel] = list()
        for k in (first, second):
            if isinstance(k, Add):
                self.parts.extend(k.parts)
            else:
                self.parts.append(k)

    def k_mat(self, ts1: NDArray, ts2: NDArray | None = None) -> NDArray:
        return sum(k.k_mat(ts1, ts2) for k in self.parts)  # pyright: ignore[reportReturnType]

    def k_diag(self, ts: NDArray) -> NDArray:
        return sum(k.k_diag(ts) for k in self.parts)  # pyright: ignore[reportReturnType]

    @property
    def order(self) -> int:
        return sum(k.order for k in self.parts)

    def transition(self, t1: float, t2: float) -> NDArray:
        mats = [k.transition(t1, t2) for k in self.parts]
        return block_diag(*mats)

    def noise_cov(self, t1: float, t2: float) -> NDArray:
        mats = [k.noise_cov(t1, t2) for k in self.parts]
        return block_diag(*mats)

    def state_mean(self, t: float) -> NDArray:
        vecs = [k.state_mean(t) for k in self.parts]
        return np.concatenate(vecs)

    def state_cov(self, t: float) -> NDArray:
        mats = [k.state_cov(t) for k in self.parts]
        return block_diag(*mats)

    @property
    def measurement_vector(self) -> NDArray:
        vecs = [k.measurement_vector for k in self.parts]
        return np.concatenate(vecs)

    @property
    def feedback(self) -> NDArray:
        mats = [k.feedback for k in self.parts]
        return block_diag(*mats)

    @property
    def noise_effect(self) -> NDArray:
        mats = [k.noise_effect for k in self.parts]
        return block_diag(*mats)

    @property
    def noise_density(self) -> NDArray:
        mats = [k.noise_density for k in self.parts]
        return block_diag(*mats)
