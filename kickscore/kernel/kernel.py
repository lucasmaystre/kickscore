import abc
import numpy as np
import scipy as sp


class Kernel(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def k_mat(self, ts1, ts2=None):
        """Compute the covariance matrix."""

    @abc.abstractmethod
    def k_diag(self, ts):
        """Compute the variances (diagonal of the covariance matrix)."""

    @abc.abstractproperty
    def order(self):
        """Order of the SDE :math:`m`."""

    @abc.abstractmethod
    def state_mean(self, t):
        """Prior mean of the state vector, :math:`\mathbf{m}_0(t)`."""

    @abc.abstractmethod
    def state_cov(self, t):
        """Prior covariance of the state vector, :math:`\mathbf{P}_0(t)`."""

    @abc.abstractproperty
    def measurement_vector(self):
        """Measurement vector :math:`\mathbf{h}`."""

    @abc.abstractproperty
    def feedback(self):
        """Feedback matrix :math:`\mathbf{F}`."""

    @abc.abstractproperty
    def noise_effect(self):
        """Noise effect matrix :math:`\mathbf{L}`."""

    @abc.abstractproperty
    def noise_density(self):
        """Power spectral density of the noise :math:`\mathbf{Q}`."""
        # Note: usually a scalar, except for combination kernels (e.g., Add).

    def transition(self, delta):
        """Transition matrix :math:`\mathbf{A}` for a given time interval."""
        F = self.feedback
        return sp.linalg.expm(F * delta)

    def noise_cov(self, delta):
        """Noise covariance matrix :math:`\mathbf{Q}` for a given time interval."""
        # Solution via matrix fraction decomposition, see:
        # - <https://github.com/SheffieldML/GPy/blob/devel/GPy/models/state_space.py#L715>
        # - Särkka's thesis (2006).
        mat = self.noise_effect.dot(self.noise_density).dot(self.noise_effect.T)
        #print(g)
        print(mat)
        Phi = np.vstack((
                np.hstack((self.feedback, mat)),
                np.hstack((np.zeros_like(mat), -self.feedback.T))))
        print(Phi)
        m = self.order
        AB = np.dot(sp.linalg.expm(Phi * delta), np.eye(2*m, m, k=-m))
        print(AB)
        return sp.linalg.solve(AB[m:,:].T, AB[:m,:].T)

    def __add__(self, other):
        # Delayed import to avoid circular dependencies.
        from .add import Add
        return Add(self, other)

    @staticmethod
    def distances(ts1, ts2):
        # mat[i, j] = |ts1[i] - ts2[j]|
        return np.abs(ts1[:,np.newaxis] - ts2[np.newaxis,:])

    # TODO simlulate()
