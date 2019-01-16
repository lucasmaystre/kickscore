import numpy as np

from .kernel import Kernel
from scipy.linalg import block_diag


class Add(Kernel):

    def __init__(self, first, second):
        self.parts = list()
        for k in (first, second):
            if isinstance(k, Add):
                self.parts.extend(k.parts)
            else:
                self.parts.append(k)

    def k_mat(self, ts1, ts2=None):
        return sum(k.k_mat(ts1, ts2) for k in self.parts)

    def k_diag(self, ts):
        return sum(k.k_diag(ts) for k in self.parts)

    @property
    def order(self):
        return sum(k.order for k in self.parts)

    def transition(self, t1, t2):
        mats = [k.transition(t1, t2) for k in self.parts]
        return block_diag(*mats)

    def noise_cov(self, t1, t2):
        mats = [k.noise_cov(t1, t2) for k in self.parts]
        return block_diag(*mats)

    def state_mean(self, t):
        vecs = [k.state_mean(t) for k in self.parts]
        return np.concatenate(vecs)

    def state_cov(self, t):
        mats = [k.state_cov(t) for k in self.parts]
        return block_diag(*mats)

    @property
    def measurement_vector(self):
        vecs = [k.measurement_vector for k in self.parts]
        return np.concatenate(vecs)

    @property
    def feedback(self):
        mats = [k.feedback for k in self.parts]
        return block_diag(*mats)

    @property
    def noise_effect(self):
        mats = [k.noise_effect for k in self.parts]
        return block_diag(*mats)

    @property
    def noise_density(self):
        mats = [k.noise_density for k in self.parts]
        return block_diag(*mats)
