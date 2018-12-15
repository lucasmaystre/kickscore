import numpy as np
import pytest

from kickscore.fitter import BatchFitter, RecursiveFitter
from kickscore.kernel import Matern32
from math import log, pi


KERNEL = Matern32(var=2.0, lscale=1.0)

# GPy code used to generate the ground-truth values.
#
#     kernel = GPy.kern.Matern32(
#             input_dim=1, variance=2.0, lengthscale=1.0)
#     m = GPy.models.GPHeteroscedasticRegression(
#             DATA["ts_train"][:,None], DATA["ys"][:,None], kernel)
#     m['.*het_Gauss.variance'] = DATA["vs"][:,None]
#     mean, var = m.predict_noiseless(DATA["ts_train"][:,None])
#     mean_pred, var_pred = m.predict_noiseless(DATA["ts_test"][:,None])
#     loglik = m.log_likelihood()
DATA = {
    "ts_train": np.array([
        0.11616722, 0.31198904, 0.31203728, 0.74908024, 1.19731697,
        1.20223002, 1.41614516, 1.46398788, 1.73235229, 1.90142861]),
    "ys": np.array([
        -1.10494786, -0.07702044, -0.25473925, 3.22959111, 0.90038114,
        0.30686385, 1.70281621, -1.717506, 0.63707278, -1.40986299]),
    "vs": np.array([
        0.55064619, 0.3540315, 0.34114585, 2.21458142, 7.40431354,
        0.35093921, 0.91847147, 4.50764809, 0.43440729, 1.3308561]),
    "mean": np.array([
        -0.52517486, -0.18391072, -0.18381275, 0.59905936, 0.62923813,
        0.6280899, 0.56576719, 0.53663651, 0.26874937, 0.04892406]),
    "var": np.array([
        0.20318775, 0.12410961, 0.12411533, 0.32855394, 0.19538865,
        0.19410925, 0.18676754, 0.19074449, 0.22105848, 0.33534931]),
    "loglik": -17.357282245711051,
    "ts_pred": np.array([0.0, 1.0, 2.0]),
    "mean_pred": np.array([-0.63981819, 0.67552349, -0.04684169]),
    "var_pred": np.array([0.33946081, 0.28362645, 0.45585554]),
}


@pytest.mark.parametrize(
        "fitter", (BatchFitter(KERNEL), RecursiveFitter(KERNEL)))
def test_allocation(fitter):
    # No data, hence fitter defined to be allocated.
    assert fitter.is_allocated
    # Add some data.
    for i in range(8):
        fitter.add_sample(i)
    assert not fitter.is_allocated
    # Allocate the arrays.
    fitter.allocate()
    assert fitter.is_allocated
    # Add some more data.
    for i in range(8):
        fitter.add_sample(i)
    assert not fitter.is_allocated
    # Re-allocate the arrays.
    fitter.allocate()
    assert fitter.is_allocated
    # Check that arrays have the appropriate size.
    for attr in ("ts", "ms", "vs", "ns", "xs"):
        assert len(getattr(fitter, attr)) == 16


@pytest.mark.parametrize(
        "fitter", (BatchFitter(KERNEL), RecursiveFitter(KERNEL)))
def test_against_gpy(fitter):
    """The output of the fitter should match that of GPy."""
    for t in DATA["ts_train"]:
        fitter.add_sample(t)
    fitter.allocate()
    fitter.xs[:] = 1 / DATA["vs"]
    fitter.ns[:] = DATA["ys"] / DATA["vs"]
    fitter.fit()
    # Estimation.
    print(fitter.ms)
    assert np.allclose(fitter.ms, DATA["mean"])
    assert np.allclose(fitter.vs, DATA["var"])
    # Prediction.
    ms, vs = fitter.predict(DATA["ts_pred"])
    assert np.allclose(ms, DATA["mean_pred"])
    assert np.allclose(vs, DATA["var_pred"])
    # Log-likelihood.
    ll = fitter.ep_log_likelihood_contrib
    # We need to add the unstable terms that cancel out with the EP
    # contributions to the log-likelihood. See appendix of the report.
    ll += sum(-0.5 * log(2 * pi * v) for v in DATA["vs"])
    ll += sum(-0.5 * y*y / v for y, v in zip(DATA["ys"], DATA["vs"]))
    assert np.allclose(ll, DATA["loglik"])


@pytest.mark.parametrize(
        "fitter", (BatchFitter(KERNEL), RecursiveFitter(KERNEL)))
def test_stability(fitter):
    """The fitter should handle infinite-variance observations."""
    ns = np.array([0.0, 0.0, 0.0])
    xs = np.array([100.0, 0.001, 0.0])
    for t in range(3):
        fitter.add_sample(t)
    fitter.allocate()
    fitter.xs[:] = xs
    fitter.ns[:] = ns
    fitter.fit()
    assert all(np.isfinite(fitter.ms))
    assert all(np.isfinite(fitter.vs))
    assert np.isfinite(fitter.ep_log_likelihood_contrib)
    if isinstance(fitter, RecursiveFitter):
        assert np.isfinite(fitter.kl_log_likelihood_contrib)


@pytest.mark.parametrize(
        "fitter", (BatchFitter(KERNEL), RecursiveFitter(KERNEL)))
def test_no_data(fitter):
    """The fitter should correctly handle the "no-data" case."""
    fitter.fit()
    fitter.predict(np.array([1.0, 2.0]))
    assert fitter.ep_log_likelihood_contrib == 0
    if isinstance(fitter, RecursiveFitter):
        assert fitter.kl_log_likelihood_contrib == 0
