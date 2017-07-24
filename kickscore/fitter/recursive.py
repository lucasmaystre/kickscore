import numpy as np

from .fitter import Fitter
from scipy.linalg import cho_factor, cho_solve


class RecursiveFitter(Fitter):

    def __init__(self, kernel):
        super().__init__(kernel)
        m = kernel.order
        self._h = kernel.measurement_vector
        self._I = np.eye(m)
        self._A = np.zeros((0, m, m))  # Transition matrices.
        self._Q = np.zeros((0, m, m))  # Noise covariance matrices.
        self._m_p = np.zeros((0, m))  # Predictive mean.
        self._P_p = np.zeros((0, m, m))  # Predictive covariance.
        self._m_f = np.zeros((0, m))  # Filtering mean.
        self._P_f = np.zeros((0, m, m))  # Filtering covariance.
        self._m_s = np.zeros((0, m))  # Smoothing mean.
        self._P_s = np.zeros((0, m, m))  # Smoothing covariance.

    def allocate(self):
        """Overrides `Fitter.allocate` to allocate the SSM-related matrices."""
        n_new = len(self.ts_new)
        # Usual variables.
        self.ts = np.concatenate((self.ts, self.ts_new))
        self.means = np.concatenate((self.means, np.zeros(n_new)))
        self.vars = np.concatenate(
                (self.vars, self.kernel.k_diag(self.ts_new)))
        self.nus = np.concatenate((self.nus, np.zeros(n_new)))
        self.taus = np.concatenate((self.taus, np.zeros(n_new)))
        # Initialize the predictive, filtering and smoothing distributions.
        mean = np.tile(self.kernel.initial_mean, (n_new, 1))
        cov = np.tile(self.kernel.initial_cov, (n_new, 1, 1))
        self._m_p = np.concatenate((self._m_p, mean))
        self._P_p = np.concatenate((self._P_p, cov))
        self._m_f = np.concatenate((self._m_f, mean))
        self._P_f = np.concatenate((self._P_f, cov))
        self._m_s = np.concatenate((self._m_s, mean))
        self._P_s = np.concatenate((self._P_s, cov))
        # Compute the new transition and noise covariance matrices.
        m = self.kernel.order
        self._A = np.concatenate((self._A, np.zeros((n_new, m, m))))
        self._Q = np.concatenate((self._Q, np.zeros((n_new, m, m))))
        for i in range(len(self.ts) - n_new, len(self.ts)):
            if i == 0:
                # Very first sample, no need to compute anything.
                continue
            dt = self.ts[i] - self.ts[i - 1]
            self._A[i-1] = self.kernel.transition(dt)
            self._Q[i-1] = self.kernel.noise_cov(dt)
        # Clear the list of pending samples.
        self.ts_new = list()

    def fit(self):
        if not self.is_allocated:
            raise RuntimeError("new data since last call to `allocate()`")
        if len(self.ts) == 0:
            self.is_fitted = True
            return
        # Rename variables for conciseness.
        mean, var = self.means, self.vars
        nu, tau = self.nus, self.taus
        h, I, A, Q = self._h, self._I, self._A, self._Q
        m_p, P_p = self._m_p, self._P_p
        m_f, P_f = self._m_f, self._P_f
        m_s, P_s = self._m_s, self._P_s
        # Forward pass (Kalman filter).
        for i in range(len(self.ts)):
            if i > 0:
                m_p[i] = A[i-1].dot(m_f[i-1])
                P_p[i] = A[i-1].dot(P_f[i-1]).dot(A[i-1].T) + Q[i-1]
            # These are slightly modified equations to work with tau and nu.
            k = P_p[i].dot(h) / (1 + tau[i] * h.dot(P_p[i]).dot(h))
            m_f[i] = m_p[i] + k * (nu[i] - tau[i] * np.dot(h, m_p[i]))
            P_f[i] = (I - np.outer(tau[i] * k, h)).dot(P_p[i])
        # Backward pass (RTS smoother).
        for i in range(len(self.ts) - 1, -1, -1):
            if i == len(self.ts) - 1:
                m_s[i] = m_f[i]
                P_s[i] = P_f[i]
            else:
                cho_fact = cho_factor(P_p[i+1])
                G = cho_solve(cho_fact, A[i].dot(P_f[i])).T
                m_s[i] = m_f[i] + G.dot(m_s[i+1] - m_p[i+1])
                P_s[i] = P_f[i] + G.dot(P_s[i+1] - P_p[i+1]).dot(G.T)
            mean[i] = h.dot(m_s[i])
            var[i] = h.dot(P_s[i]).dot(h)
        # set self.means and self.vars
        self.is_fitted = True

    def predict(self, ts):
        if not self.is_fitted:
            raise RuntimeError("new data since last call to `fit()`")
        mean = np.zeros(len(ts))
        var = np.zeros(len(ts))
        h = self._h
        m_p, P_p = self._m_p, self._P_p
        m_f, P_f = self._m_f, self._P_f
        m_s, P_s = self._m_s, self._P_s
        locations = np.searchsorted(self.ts, ts)
        for i, nxt in enumerate(locations):
            if nxt == len(self.ts):
                # new point is *after* last observation
                dt = ts[i] - self.ts[-1]
                A = self.kernel.transition(dt)
                Q = self.kernel.noise_cov(dt)
                mean[i] = h.dot(np.dot(A, m_s[-1]))
                var[i] = h.dot(A.dot(P_s[-1]).dot(A.T) + Q).dot(h)
            else:
                j = nxt - 1
                if j < 0:
                    m = self.kernel.initial_mean
                    P = self.kernel.initial_cov
                else:
                    # Predictive mean and cov for new point based on left
                    # neighbor.
                    dt = ts[i] - self.ts[j]
                    A = self.kernel.transition(dt)
                    Q = self.kernel.noise_cov(dt)
                    P = A.dot(P_f[j]).dot(A.T) + Q
                    m = A.dot(m_f[j])
                # RTS update using the right neighbor.
                dt = self.ts[j+1] - ts[i]
                A = self.kernel.transition(dt)
                cho_fact = cho_factor(P_p[j+1])
                G = cho_solve(cho_fact, A.dot(P)).T
                mean[i] = h.dot(m + G.dot(m_s[j+1] - m_p[j+1]))
                var[i] = h.dot(P + G.dot(P_s[j+1] - P_p[j+1]).dot(G.T)).dot(h)
        return (mean, var)
