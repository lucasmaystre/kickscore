"""Test that SSM and GP-batch produce the same results.

The emphasis of this test module is on the *kernels*, even though it implicitly
also tests the fitters.
"""

import numpy as np
import pytest

from kickscore.fitter import BatchFitter, RecursiveFitter
from kickscore.kernel import *


TRAIN_TS = np.array([1.32, 3.95, 4.47, 5.14, 5.49, 5.54, 6.41, 9.50])
OBS_NS = np.array([0.65, 0.53, 0.64, 0.99, -0.99, -0.64, 0.00, 0.00])
OBS_XS = np.array([0.20, 0.16, 0.21, 1.45, 1.00, 0.21, 0.00, 0.20])
TEST_TS = np.array([1.0, 5.14, 7.0, 11.0])

KERNELS = (
    Constant(2.5),
    Exponential(1.1, 2.2),
    Matern32(1.5, 0.7),
    Matern52(0.2, 5.0),
    Affine(var_offset=1.0, var_slope=2.0, t0=-0.3),
    Wiener(1.2, 0.3),
    Wiener(1.32, 1.0, var_t0=1.0),
    Constant(0.3) + Matern32(1.5, 0.7) + Matern32(0.2, 5.0),
)


@pytest.mark.parametrize("kernel", KERNELS)
def test_equivalence(kernel):
    """The SSM and GP-batch approaches should produce the same results."""
    # Setting up the fitters.
    batch = BatchFitter(kernel)
    recur = RecursiveFitter(kernel)
    for t in TRAIN_TS:
        _ = batch.add_sample(t)
        _ = recur.add_sample(t)
    batch.allocate()
    recur.allocate()
    for i, (n, x) in enumerate(zip(OBS_NS, OBS_XS)):
        batch.ns[i] = n
        batch.xs[i] = x
        recur.ns[i] = n
        recur.xs[i] = x
    batch.fit()
    recur.fit()
    # Estimated mean and variance at training points.
    assert np.allclose(batch.ms, recur.ms)
    assert np.allclose(batch.vs, recur.vs)
    # Predicted mean and var at new test points.
    ms_b, vs_b = batch.predict(TEST_TS)
    ms_r, vs_r = recur.predict(TEST_TS)
    assert np.allclose(ms_b, ms_r)
    assert np.allclose(vs_b, vs_r)
