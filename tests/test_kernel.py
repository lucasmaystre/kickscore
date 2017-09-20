import numpy as np

from kickscore.kernel import *
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
    "constant": np.array([
        [ 2.5,  2.5,  2.5],
        [ 2.5,  2.5,  2.5],
        [ 2.5,  2.5,  2.5]]),
    "exponential": np.array([
        [ 1.1,         1.00441079,  0.57949461],
        [ 1.00441079,  1.1,         0.63464479],
        [ 0.57949461,  0.63464479,  1.1       ]]),
    "matern32": np.array([
        [ 1.5,         1.36702084,  0.20560784],
        [ 1.36702084,  1.5,         0.3000753 ],
        [ 0.20560784,  0.3000753,   1.5       ]]),
    "matern52": np.array([
        [ 0.2,         0.19973384,  0.18769647],
        [ 0.19973384,  0.2,         0.1907786 ],
        [ 0.18769647,  0.1907786,   0.2       ]]),
    "affine": np.array([
        [  3.1752,   3.6792,   6.7284],
        [  3.6792,   4.2632,   7.7964],
        [  6.7284,   7.7964,  14.2578]]),
    "wiener": np.array([
        [ 1.512,  1.512,  1.512],
        [ 1.512,  1.752,  1.752],
        [ 1.512,  1.752,  3.204]]),
    "add": np.array([
        [ 1.7,         1.56675469,  0.39330431],
        [ 1.56675469,  1.7,         0.4908539 ],
        [ 0.39330431,  0.4908539,   1.7       ]]),
}


def test_kernel_matrix():
    """`k_mat` should match the output of GPy."""
    for name in KERNEL:
        assert np.allclose(KERNEL[name].k_mat(TS), GROUND_TRUTH[name])


def test_kernel_diag():
    """`k_diag` should match the diagonal of `k_mat`."""
    ts = 10 * np.random.random(10)
    for k in KERNEL.values():
        assert np.allclose(np.diag(k.k_mat(ts)), k.k_diag(ts))


def test_kernel_order():
    """The SSM matrices & vectors should have the correct dims."""
    for k in KERNEL.values():
        m = k.order
        assert k.state_mean(0.0).shape == (m,)
        assert k.state_cov(0.0).shape == (m, m)
        assert k.measurement_vector.shape == (m,)
        assert k.feedback.shape == (m, m)
        assert k.noise_effect.shape[0] == m
        assert k.transition(1.0).shape == (m, m)
        assert k.noise_cov(1.0).shape == (m, m)


def test_ssm_variance():
    """The measured state variance should match `k_diag`."""
    ts = 10 * np.random.random(10)
    for k in KERNEL.values():
        h = k.measurement_vector
        vars_ = [h.dot(k.state_cov(t)).dot(h) for t in ts]
        assert np.allclose(vars_, k.k_diag(ts))


def test_ssm_matrices():
    """`transition` and `noise_cov` should match the numerical solution.`"""
    deltas = [0.01, 1.0, 10.0]
    for k in KERNEL.values():
        for delta in deltas:
            assert np.allclose(
                    Kernel.transition(k, delta), k.transition(delta))
            assert np.allclose(
                    Kernel.noise_cov(k, delta), k.noise_cov(delta))
