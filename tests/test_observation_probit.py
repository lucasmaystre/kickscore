import numpy as np

from kickscore.observation.probit import (
        _logphi, _match_moments_probit, _match_moments_probit_tie,
        _normpdf, _normcdf)
from math import log, pi, sqrt
from scipy.stats import norm


MEAN_CAV = 1.23
COV_CAV = 4.56
MARGIN = 0.98

# Ground-truth computed using kickscore at revision `66beee380`.
MM_PROBIT = (
        -0.35804993126636214, 0.21124433823827732, -0.09135628123504448)
MM_PROBIT_TIE = (
        -1.2606613196803695, -0.20881357057246397, -0.16982734815171024)


def test_normpdf():
    """``_normpdf`` should work as expected."""
    for v in np.linspace(-4, 4, num=21):
        assert np.allclose(norm.pdf(v), _normpdf(v))


def test_normcdf():
    """``_normcdf`` should work as expected."""
    for v in np.linspace(-4, 4, num=21):
        assert np.allclose(norm.cdf(v), _normcdf(v))


def test_logphi():
    """``_logphi`` should work as expected."""
    # First case: z close to 0.
    res, dres = _logphi(0)
    assert np.allclose(res, -log(2))
    assert np.allclose(dres, 2 / sqrt(2 * pi))
    # Second case: z small. Ground truth computed with GPy-1.7.7.
    res, dres = _logphi(-15)
    assert np.allclose(res, -116.13138484571169)
    assert np.allclose(dres, 15.066086827167823)
    # Third case: z positive, large.
    res, dres = _logphi(100)
    assert np.allclose(res, 0)
    assert np.allclose(dres, 0)


def test_match_moments_probit():
    """``_match_moments_probit`` should work as expected."""
    assert np.allclose(_match_moments_probit(MEAN_CAV, COV_CAV), MM_PROBIT)


def test_match_moments_probit_tie():
    """``_match_moments_probit_tie`` should work as expected."""
    assert np.allclose(
            _match_moments_probit_tie(MEAN_CAV, COV_CAV, MARGIN), MM_PROBIT_TIE)
