import numba
import numpy as np
import scipy.special

from kickscore.observation.ordinal import _mm_probit_win, _ll_probit_win
from kickscore.observation.utils import *
from math import log, pi, sqrt
from scipy.stats import norm


def test_normpdf():
    """``normpdf`` should work as expected."""
    for v in np.linspace(-4, 4, num=21):
        assert np.allclose(norm.pdf(v), normpdf(v))


def test_normcdf():
    """``normcdf`` should work as expected."""
    for v in np.linspace(-4, 4, num=21):
        assert np.allclose(norm.cdf(v), normcdf(v))


def test_logphi():
    """``logphi`` should work as expected."""
    # First case: z close to 0.
    res, dres = logphi(0)
    assert np.allclose(res, -log(2))
    assert np.allclose(dres, 2 / sqrt(2 * pi))
    # Second case: z small. Ground truth computed with GPy-1.7.7.
    res, dres = logphi(-15)
    assert np.allclose(res, -116.13138484571169)
    assert np.allclose(dres, 15.066086827167823)
    # Third case: z positive, large.
    res, dres = logphi(100)
    assert np.allclose(res, 0)
    assert np.allclose(dres, 0)


def test_cvi_expectations():
    """Basic test for ``cvi_expectations``."""
    @cvi_expectations
    @numba.jit(nopython=True)
    def ll(x):
        return logphi(x)[0]
    vals = ll.cvi_expectations(0.3, 2.7)
    assert np.allclose(vals, [-1.19810974, 0.89703901, -0.25653925])


def test_match_moments():
    """Basic test for ``match_moments``"""
    ll = match_moments(_ll_probit_win)
    for mean in (0.0, -2.0, 18):
        for var in (1e-3, 1.0, 5.0):
            assert np.allclose(
                    ll.match_moments(mean, var, 0.0),
                    _mm_probit_win(mean, var),
                    atol=1e-08, rtol=1e-04)


def test_logsumexp():
    """``logsumexp`` should match scipy's output."""
    for xs in np.random.randn(10, 5):
        assert logsumexp(xs) == scipy.special.logsumexp(xs)


def test_log_factorial():
    """``log_factorial`` should match scipy's ``gammaln``."""
    for n in (0, 1, 5, 123, 574):
        assert np.allclose(log_factorial(n), scipy.special.gammaln(n + 1))
