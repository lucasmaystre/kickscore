from math import log, pi, sqrt

import numpy as np
from scipy.stats import norm

from kickscore.item import Item
from kickscore.kernel import Constant
from kickscore.observation.observation import Observation


class DummyObservation(Observation):
    def match_moments(self, mean_cav: float, var_cav: float) -> tuple[float, float, float]:
        return (0.0, 0.0, 0.0)

    def cvi_expectations(self, mean: float, var: float) -> tuple[float, float, float]:
        return (0.0, 0.0, 0.0)

    def probability(self) -> float:
        return 0.0


def test_ep_log_likelihood_contrib():
    mean = 0.5
    var = 1.5
    mean_cav = 0.7
    cov_cav = 0.6
    logpart = 1.4
    # Setup the observation.
    item = Item(Constant(1.0), "batch")
    obs = DummyObservation([(item, 1.0)], 0.0)
    item.fitter.allocate()
    item.fitter.xs[0] = 1 / var
    item.fitter.ns[0] = mean / var
    obs._xs_cav[0] = 1 / cov_cav
    obs._ns_cav[0] = mean_cav / cov_cav
    obs._logpart = logpart
    # Ground truth is log( Z / N(m1 | m2, v1 + v2) ), see notes.
    ground_truth = logpart - log(norm.pdf(mean, loc=mean_cav, scale=sqrt(var + cov_cav)))
    # The contribution to the log-likelihood doesn't include unstable terms,
    # they need to be added back to be compared to the ground truth.
    ll = obs.ep_log_likelihood_contrib
    ll += 0.5 * log(2 * pi * var)
    ll += 0.5 * mean * mean / var
    assert np.allclose(ll, ground_truth)
