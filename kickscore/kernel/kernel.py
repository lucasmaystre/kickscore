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

    @staticmethod
    def distances(ts1, ts2):
        # mat[i, j] = |ts1[i] - ts2[j]|
        n_rows = len(ts1)
        n_cols = len(ts2)
        return np.abs(np.tile(ts1, (n_cols, 1)).T - np.tile(ts2, (n_rows, 1)))


# Future kernels to implement:
#class Wiener(Kernel):
#    pass
#
#class Constant(Kernel):
#    pass
#
#class Linear(Kernel);
#    pass
#
#class Exponential(Kernel):
#    pass
#
#class Matern32(Kernel):
#    pass
#
#class Sum(Kernel):
#    pass
