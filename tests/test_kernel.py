import numpy as np
import pytest

from kickscore.kernel import (
    Affine,
    Constant,
    Exponential,
    Matern32,
    Matern52,
    Wiener,
)
from kickscore.kernel.kernel import Kernel

TS = np.array([1.26, 1.46, 2.67])
KERNEL = {
    "constant": Constant(2.5),
    "exponential": Exponential(1.1, 2.2),
    "matern32": Matern32(1.5, 0.7),
    "matern52": Matern52(0.2, 5.0),
    # In this case it's actually linear.
    "affine": Affine(var_offset=0.0, var_slope=2.0, t0=0.0),
    "wiener": Wiener(1.2, 0.0),
    "add": Matern32(1.5, 0.7) + Matern52(0.2, 5.0),
}

# GPy code used to generate comparison values.
#
#     import numpy as np
#     from GPy.kern import Bias, Exponential, Matern32, Matern52, Linear, Brownian
#     ts = np.array([1.26, 1.46, 2.67]).reshape(-1, 1)
#
#     kernel = {
#         "constant": Bias(input_dim=1, variance=2.5),
#         "exponential": Exponential(input_dim=1, variance=1.1, lengthscale=2.2),
#         "matern32": Matern32(input_dim=1, variance=1.5, lengthscale=0.7),
#         "matern52": Matern52(input_dim=1, variance=0.2, lengthscale=5.0),
#         "affine": Linear(input_dim=1, variances=2.0),
#         "wiener": Brownian(input_dim=1, variance=1.2),
#         "add": Matern32(input_dim=1, variance=1.5, lengthscale=0.7)
#                 + Matern52(input_dim=1, variance=0.2, lengthscale=5.0),
#     }
#
#     for name, k in kernel.items():
#         print(name)
#         print(k.K(ts))
GROUND_TRUTH = {
    "constant": np.array(
        [
            [2.5, 2.5, 2.5],
            [2.5, 2.5, 2.5],
            [2.5, 2.5, 2.5],
        ]
    ),
    "exponential": np.array(
        [
            [1.1, 1.00441079, 0.57949461],
            [1.00441079, 1.1, 0.63464479],
            [0.57949461, 0.63464479, 1.1],
        ]
    ),
    "matern32": np.array(
        [
            [1.5, 1.36702084, 0.20560784],
            [1.36702084, 1.5, 0.3000753],
            [0.20560784, 0.3000753, 1.5],
        ]
    ),
    "matern52": np.array(
        [
            [0.2, 0.19973384, 0.18769647],
            [0.19973384, 0.2, 0.1907786],
            [0.18769647, 0.1907786, 0.2],
        ]
    ),
    "affine": np.array(
        [
            [3.1752, 3.6792, 6.7284],
            [3.6792, 4.2632, 7.7964],
            [6.7284, 7.7964, 14.2578],
        ]
    ),
    "wiener": np.array(
        [
            [1.512, 1.512, 1.512],
            [1.512, 1.752, 1.752],
            [1.512, 1.752, 3.204],
        ]
    ),
    "add": np.array(
        [
            [1.7, 1.56675469, 0.39330431],
            [1.56675469, 1.7, 0.4908539],
            [0.39330431, 0.4908539, 1.7],
        ]
    ),
}


@pytest.mark.parametrize("name", KERNEL.keys())
def test_kernel_matrix(name: str):
    """`k_mat` should match the output of GPy."""
    assert np.allclose(KERNEL[name].k_mat(TS), GROUND_TRUTH[name])


@pytest.mark.parametrize("kernel", KERNEL.values())
def test_kernel_diag(kernel: Kernel):
    """`k_diag` should match the diagonal of `k_mat`."""
    ts = 10 * np.random.random(10)
    assert np.allclose(np.diag(kernel.k_mat(ts)), kernel.k_diag(ts))


@pytest.mark.parametrize("kernel", KERNEL.values())
def test_kernel_order(kernel: Kernel):
    """The SSM matrices & vectors should have the correct dims."""
    m = kernel.order
    assert kernel.state_mean(0.0).shape == (m,)
    assert kernel.state_cov(0.0).shape == (m, m)
    assert kernel.measurement_vector.shape == (m,)
    assert kernel.feedback.shape == (m, m)
    assert kernel.noise_effect.shape[0] == m
    assert kernel.transition(0.0, 1.0).shape == (m, m)
    assert kernel.noise_cov(0.0, 1.0).shape == (m, m)


@pytest.mark.parametrize("kernel", KERNEL.values())
def test_ssm_variance(kernel: Kernel):
    """The measured state variance should match `k_diag`."""
    ts = 10 * np.random.random(10)
    h = kernel.measurement_vector
    vars_ = [h.dot(kernel.state_cov(t)).dot(h) for t in ts]
    assert np.allclose(vars_, kernel.k_diag(ts))


@pytest.mark.parametrize("kernel", KERNEL.values())
def test_ssm_matrices(kernel: Kernel):
    """`transition` and `noise_cov` should match the numerical solution.`"""
    deltas = [0.01, 1.0, 10.0]
    for delta in deltas:
        assert np.allclose(Kernel.transition(kernel, 0.0, delta), kernel.transition(0.0, delta))
        assert np.allclose(Kernel.noise_cov(kernel, 0.0, delta), kernel.noise_cov(0.0, delta))


def test_simulate_constant():
    """A sample from a constant GP should be constant."""
    kernel = Constant(1.0)
    ts = np.linspace(-2, 7, num=10)
    xs = kernel.simulate(ts)
    assert all(xs == xs[0])


def test_simulate_affine():
    """A sample from an affine GP should be affine."""
    kernel = Affine(var_offset=1.0, t0=0.0, var_slope=1.0)
    ts = np.linspace(0, 10, num=10)
    xs = kernel.simulate(ts)
    slope = (xs[-1] - xs[0]) / 10
    assert np.allclose(xs, xs[0] + slope * ts)
