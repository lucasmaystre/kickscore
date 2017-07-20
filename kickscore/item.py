import numpy as np

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


class Sample:
    
    def __init__(self, item, idx):
        self._item = item
        self._idx = idx

    @property
    def mean(self):
        return self._item.mean[self._idx]

    @property
    def var(self):
        return self._item.var[self._idx]

    def set_natural_params(self, nu, tau):
        self._item._fitter._nus[self._idx] = nu
        self._item._fitter._taus[self._idx] = tau


class ItemFitter:

    def __init__(self, parent):
        self._parent = parent

    def init(self):
        self._ts = np.array(self._parent._ts)
        self._kmat = self._parent._k.compute(self._ts)
        self._taus = np.zeros(len(self._ts))
        self._nus = np.zeros(len(self._ts))
        self._parent._mean = np.zeros(len(self._ts))
        self._parent._cov = self._kmat

    def recompute(self):
        # Woodbury inverse and Woodbury vector - used for prediction. Idea
        # taken from GPy (`latent_function_inference.posterior`).
        sigmas = 1 / self._taus
        # TODO This can be improved, see (3.67) and (3.68) in GPML.
        mat = inv_posdef(self._kmat + np.diag(sigmas))
        self._parent._woodbury_inv = mat
        self._parent._woodbury_vec = mat.dot(sigmas * self._nus)
        # Recompute mean and covariance.
        cov = self._kmat - self._kmat.dot(mat).dot(self._kmat)
        mean = np.dot(cov, self._nus)
        self._parent._cov = cov
        self._parent._mean = mean


class Item:

    def __init__(self, kernel):
        self._k = kernel
        self._fitter = ItemFitter(self)
        self._ts = list()
        self._woodbury_inv = None
        self._woodbury_vec = None
        self._mean = None
        self._cov = None

    @property
    def fitter(self):
        return self._fitter

    @property
    def mean(self):
        return self._mean

    @property
    def var(self):
        return np.diag(self._cov)

    @property
    def ts(self):
        return self._ts

    def predict(self, ts):
        if self._woodbury_vec is None or self._woodbury_inv is None:
            raise RuntimeError("item not fitted yet")
        # (3.60) and (3.61) in the GPML book.
        kmat1 = self._k.compute(ts, self._ts)
        kmat2 = self._k.compute(ts, ts)
        mean = np.dot(kmat1, self._woodbury_vec)
        cov = kmat2 - kmat1.dot(self._woodbury_inv).dot(kmat1.T)
        return mean, np.diag(cov)

    def add_sample(self, t):
        self._ts.append(t)
        idx = len(self._ts) - 1
        return Sample(self, idx)
