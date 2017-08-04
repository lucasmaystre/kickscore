import abc
import numpy as np
import scipy as sp


class Kernel(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def k_mat(self, ts1, ts2=None):
        """Compute the covariance matrix."""
        pass

    @abc.abstractmethod
    def k_diag(self, ts):
        """Compute the variances (diagonal of the covariance matrix)."""
        pass

    @abc.abstractproperty
    def order(self):
        """Order of the SDE :math:`m`."""
        return 3

    @abc.abstractmethod
    def state_mean(self, t):
        """Prior mean of the state vector, :math:`\mathbf{m}_0(t)`."""
        pass

    @abc.abstractmethod
    def state_cov(self, t):
        """Prior covariance of the state vector, :math:`\mathbf{P}_0(t)`."""
        pass

    @abc.abstractproperty
    def measurement_vector(self):
        """Measurement vector :math:`\mathbf{h}`."""
        pass

    @abc.abstractproperty
    def feedback(self):
        """Feedback matrix :math:`\mathbf{F}`."""
        pass

    @abc.abstractproperty
    def noise_effect(self):
        """Noise effect vector :math:`\mathbf{g}`."""
        pass

    @abc.abstractproperty
    def noise_density(self):
        """Power spectral density of the noise :math:`q`."""
        pass

    def transition(self, delta):
        """Transition matrix :math:`\mathbf{A}` for a given time interval."""
        F = self.feedback()
        return sp.linalg.expm(F * delta)

    def noise_cov(self, delta):
        """Noise covariance matrix :math:`\mathbf{Q}` for a given time interval."""
        # TODO: Matrix Riccati equations, see:
        # <https://github.com/SheffieldML/GPy/blob/devel/GPy/models/state_space.py#L715>
        raise NotImplementedError()

    def __add__(self, other):
        # Delayed import to avoid circular dependencies.
        from .add import Add
        return Add(self, other)

    @staticmethod
    def distances(ts1, ts2):
        # mat[i, j] = |ts1[i] - ts2[j]|
        return np.abs(ts1[:,np.newaxis] - ts2[np.newaxis,:])

    # TODO simlulate()
