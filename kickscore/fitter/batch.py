"""Score process fitter using the standard GP batch equations."""

import numpy as np

from .fitter import Fitter
from scipy.linalg import solve_triangular


class BatchFitter(Fitter):

    def __init__(self, kernel):
        super().__init__(kernel)
        self._cov = None
        self._b_cholesky = None
        self._woodbury_inv = None
        self._woodbury_vec = None

    def allocate(self):
        """Overrides `Fitter.allocate` to compute the kernel matrix."""
        super().allocate()
        self._k_mat = self.kernel.k_mat(self.ts)

    def fit(self):
        # Woodbury inverse and Woodbury vector - used for prediction. Idea
        # taken from GPy (`latent_function_inference.posterior`).
        if not self.is_allocated:
            raise RuntimeError("new data since last call to `allocate()`")
        if len(self.ts) == 0:
            self.is_fitted = True
            return
        # Stable computation of the woodbury inverse.
        xs_sqrt = np.sqrt(self.xs)
        b_cho = np.linalg.cholesky(
                np.eye(len(self.ts))
                + np.outer(xs_sqrt, xs_sqrt) * self._k_mat)
        mat = solve_triangular(
                b_cho, np.diag(xs_sqrt), lower=True, overwrite_b=True)
        w_inv = np.transpose(mat).dot(mat)
        # Recompute mean and covariance.
        cov = self._k_mat - self._k_mat.dot(w_inv).dot(self._k_mat)
        self.ms = np.dot(cov, self.ns)
        self.vs = np.diag(cov)
        # Store some quantities for prediction and marginal likelihood).
        self._cov = cov
        self._b_cholesky = b_cho
        self._woodbury_inv = w_inv
        self._woodbury_vec = self.ns - w_inv.dot(self._k_mat).dot(self.ns)
        self.is_fitted = True

    @property
    def ep_log_likelihood_contrib(self):
        """Contribution to the log-marginal likelihood of the model."""
        # Note: this is *not* equal to the log of the marginal likelihood of the
        # regression model. See "stable computation of the marginal likelihood"
        # in the notes.
        if not self.is_fitted:
            raise RuntimeError("new data since last call to `fit()`")
        if len(self.ts) == 0:
            return 0.0
        # C.f. Rasmussen and Williams' GPML book, eqs. (3.73) and (3.74).
        return (-np.sum(np.log(np.diag(self._b_cholesky)))
                + 0.5 * np.dot(self.ns, np.dot(self._cov, self.ns)))

    @property
    def kl_log_likelihood_contrib(self):
        """Contribution to the log-marginal likelihood of the model."""
        raise NotImplementedError()

    def predict(self, ts):
        if not self.is_fitted:
            raise RuntimeError("new data since last call to `fit()`")
        if len(self.ts) == 0:
            return (np.zeros_like(ts), self.kernel.k_diag(ts))
        # (3.60) and (3.61) in the GPML book.
        k_mat1 = self.kernel.k_mat(ts, self.ts)
        k_mat2 = self.kernel.k_mat(ts, ts)
        mean = np.dot(k_mat1, self._woodbury_vec)
        cov = k_mat2 - k_mat1.dot(self._woodbury_inv).dot(k_mat1.T)
        return (mean, np.diag(cov))
