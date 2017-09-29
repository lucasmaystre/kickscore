import numba
import numpy as np

from .fitter import Fitter
from math import log


@numba.jit(nopython=True)
def _fit(ts, ms, vs, ns, xs, h, I, A, Q, m_p, P_p, m_f, P_f, m_s, P_s):
    # Forward pass (Kalman filter).
    for i in range(len(ts)):
        if i > 0:
            m_p[i] = np.dot(A[i-1], m_f[i-1])
            P_p[i] = np.dot(np.dot(A[i-1], P_f[i-1]), A[i-1].T) + Q[i-1]
        # These are slightly modified equations to work with tau and nu.
        k = np.dot(P_p[i], h) / (1 + xs[i] * np.dot(np.dot(h, P_p[i]), h))
        m_f[i] = m_p[i] + k * (ns[i] - xs[i] * np.dot(h, m_p[i]))
        P_f[i] = np.dot(I - np.outer(xs[i] * k, h), P_p[i])
    # Backward pass (RTS smoother).
    for i in range(len(ts) - 1, -1, -1):
        if i == len(ts) - 1:
            m_s[i] = m_f[i]
            P_s[i] = P_f[i]
        else:
            G = np.linalg.solve(P_p[i+1], np.dot(A[i], P_f[i])).T
            m_s[i] = m_f[i] + np.dot(G, m_s[i+1] - m_p[i+1])
            P_s[i] = P_f[i] + np.dot(np.dot(G, P_s[i+1] - P_p[i+1]), G.T)
        ms[i] = np.dot(h, m_s[i])
        vs[i] = np.dot(np.dot(h, P_s[i]), h)


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
        if n_new == 0:
            return
        # Usual variables.
        zeros = np.zeros(n_new)
        self.ts = np.concatenate((self.ts, self.ts_new))
        self.ms = np.concatenate((self.ms, zeros))
        self.vs = np.concatenate((self.vs, self.kernel.k_diag(self.ts_new)))
        self.ns = np.concatenate((self.ns, zeros))
        self.xs = np.concatenate((self.xs, zeros))
        # Initialize the predictive, filtering and smoothing distributions.
        mean = np.array([self.kernel.state_mean(t) for t in self.ts_new])
        cov = np.array([self.kernel.state_cov(t) for t in self.ts_new])
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
        _fit(
                ts=self.ts, ms=self.ms, vs=self.vs, ns=self.ns, xs=self.xs,
                h=self._h, I=self._I, A=self._A, Q=self._Q,
                m_p=self._m_p, P_p=self._P_p,
                m_f=self._m_f, P_f=self._P_f,
                m_s=self._m_s, P_s=self._P_s)
        self.is_fitted = True

    @property
    def log_likelihood_contrib(self):
        """Contribution to the log-marginal likelihood of the model."""
        # Note: this is *not* equal to the log of the marginal likelihood of the
        # regression model. See "stable computation of the marginal likelihood"
        # in the notes.
        if not self.is_fitted:
            raise RuntimeError("new data since last call to `fit()`")
        h = self._h
        m_p, P_p = self._m_p, self._P_p
        ns, xs = self.ns, self.xs
        val = 0.0
        for i in range(len(self.ts)):
            o = h.dot(m_p[i])
            v = h.dot(P_p[i]).dot(h)
            val += -0.5 * (log(xs[i] * v  + 1.0)
                    + (-ns[i]**2 * v - 2 * ns[i] * o + xs[i] * o**2)
                    / (xs[i] * v + 1))
        return val

    def predict(self, ts):
        if not self.is_fitted:
            raise RuntimeError("new data since last call to `fit()`")
        ms = np.zeros(len(ts))
        vs = np.zeros(len(ts))
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
                ms[i] = h.dot(np.dot(A, m_s[-1]))
                vs[i] = h.dot(A.dot(P_s[-1]).dot(A.T) + Q).dot(h)
            else:
                j = nxt - 1
                if j < 0:
                    m = self.kernel.state_mean(ts[i])
                    P = self.kernel.state_cov(ts[i])
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
                G = np.linalg.solve(P_p[j+1], A.dot(P)).T
                ms[i] = h.dot(m + G.dot(m_s[j+1] - m_p[j+1]))
                vs[i] = h.dot(P + G.dot(P_s[j+1] - P_p[j+1]).dot(G.T)).dot(h)
        return (ms, vs)
