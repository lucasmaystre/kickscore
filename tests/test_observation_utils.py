import numpy as np

from kickscore.observation.utils import logphi, normpdf, normcdf
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
