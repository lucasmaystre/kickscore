import numpy as np

from kickscore.observation.observation import Observation
from math import log, pi, sqrt
from scipy.stats import norm


class DummyObservation(Observation):

    def match_moments(self, mean_cav, cov_cav):
        return 1.0, 2.0, 3.0


def test_log_likelihood_contrib():
    mean = 0.5
    var = 1.5
    mean_cav = 0.7
    cov_cav = 0.6
    logpart = 1.4
    # Setup the observation.
    obs = DummyObservation({}, {}, 0.0)
    obs._tau = 1 / var
    obs._nu = mean / var
    obs._mean_cav = mean_cav
    obs._cov_cav = cov_cav
    obs._logpart = logpart
    # Ground truth is log( Z / N(m1 | m2, v1 + v2) ), see notes.
    ground_truth = logpart - log(norm.pdf(
            mean, loc=mean_cav, scale=sqrt(var + cov_cav)))
    # The contribution to the log-likelihood doesn't include unstable terms,
    # they need to be added back to be compared to the ground truth.
    ll = obs.log_likelihood_contrib
    ll += 0.5 * log(2 * pi * var)
    ll += 0.5 * mean*mean / var
    assert np.allclose(ll, ground_truth)
