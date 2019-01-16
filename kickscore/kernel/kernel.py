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

    def transition(self, t1, t2):
        """Transition matrix :math:`\mathbf{A}` for a given time interval.

        Note that this default implementation assumes that the feedback matrix
        is independent of time.
        """
        F = self.feedback
        return sp.linalg.expm(F * (t2 - t1))

    def noise_cov(self, t1, t2):
        """Noise covariance matrix :math:`\mathbf{Q}` for a given time interval.

        Note that this default implementations assumes that the feedback
        matrix, the noise density and the noise effect are independent of time.
        """
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
        AB = np.dot(sp.linalg.expm(Phi * (t2 - t1)), np.eye(2*m, m, k=-m))
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

    def simulate(self, ts):
        """Sample from a Gaussian process with the corresponding kernel."""
        ts = np.sort(ts)
        xs = np.zeros((len(ts), self.order))
        mean = self.state_mean(ts[0])
        cov = self.state_cov(ts[0])
        xs[0,:] = np.random.multivariate_normal(mean, cov)
        for i in range(1, len(ts)):
            mean = np.dot(self.transition(ts[i-1], ts[i]), xs[i-1])
            cov = self.noise_cov(ts[i-1], ts[i])
            xs[i,:] = np.random.multivariate_normal(mean, cov)
        return np.dot(xs, self.measurement_vector)
