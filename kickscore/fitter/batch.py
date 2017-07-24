"""Score process fitter using the standard GP batch equations."""

import numpy as np

from .fitter import Fitter
from scipy.linalg import solve_triangular


def inv_posdef(mat):
    """Stable inverse of a positive definite matrix."""
    # See:
    # - http://www.seas.ucla.edu/~vandenbe/103/lectures/chol.pdf
    # - http://scicomp.stackexchange.com/questions/3188
    chol = np.linalg.cholesky(mat)
    ident = np.eye(mat.shape[0])
    res = solve_triangular(chol, ident, lower=True, overwrite_b=True)
    return np.transpose(res).dot(res)


class BatchFitter(Fitter):

    def __init__(self, kernel):
        super().__init__(kernel)
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
            raise RuntimeError("new data since last call to `allocate()`.")
        n_obs = len(self.ts)
        if n_obs > 0:
            # TODO The next two lines can be improved, see (3.67) and (3.68) in
            # GPML.
            sigmas = 1 / self.taus
            mat = inv_posdef(self._k_mat + np.diag(sigmas))
            self._woodbury_inv = mat
            self._woodbury_vec = mat.dot(sigmas * self.nus)
            # Recompute mean and covariance.
            cov = self._k_mat - self._k_mat.dot(mat).dot(self._k_mat)
            self.means = np.dot(cov, self.nus)
            self.vars = np.diag(cov)
        self.is_fitted = True

    @property
    def posterior(self):
        if not self.is_fitted:
            raise RuntimeError("new data since last call to `fit()`.")
        return (self.ts, self.mean, self.var)

    def predict(self, ts):
        if not self.is_fitted:
            raise RuntimeError("new data since last call to `fit()`.")
        # (3.60) and (3.61) in the GPML book.
        k_mat1 = self.kernel.k_mat(ts, self.ts)
        k_mat2 = self.kernel.k_mat(ts, ts)
        mean = np.dot(k_mat1, self._woodbury_vec)
        cov = k_mat2 - k_mat1.dot(self._woodbury_inv).dot(k_mat1.T)
        return (mean, np.diag(cov))
