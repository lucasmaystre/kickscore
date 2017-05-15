"""
Example usage:

    import kickscore as ks

    model = ks.KickScore()
    k = ks.kernel.SquaredExponential()

    model.add_static_param("audrey", mean=2.0, var=3.0)
    model.add_dynamic_param("benjamin", kernel=k, mean_fct=None)

    model.observe_binary(winners="audrey", losers="benjamin", t=2.37)
    model.fit()

    model.param["audrey"].mean
    model.param["audrey"].var
    model.param["benjamin"].means
    model.param["benjamin"].vars
"""
import numpy as np

from .observation import BinaryObservation
from .parameter import DynamicParam


class KickScore():
    
    def __init__(self):
        self._param = dict()
        self._obs = list()

    @property
    def param(self):
        return self._param

    def add_static_param(self, name, var, mean=0.0):
        raise NotImplementedError()

    def add_dynamic_param(self, name, kernel, mean_fct=lambda t: 0):
        self._param[name] = DynamicParam(kernel=kernel, mean_fct=mean_fct)

    def observe_binary(self, winner, loser, t):
        for name in (winner, loser):
            if name not in self._param:
                raise ValueError("parameter {!r} not found".format(name))
        self._obs.append(BinaryObservation(
                winner=self._param[winner], loser=self._param[loser], t=t))

    def observe_ternary(self, params1, params2, outcome, t, margin=0.5):
        # Outcome can be "params1", "params2" or "tie".
        raise NotImplementedError()

    def observe_count(self, count, attack, defense, t):
        raise NotImplementedError()

    def observe_diff(self, diff, winners, losers, t):
        raise NotImplementedError()

    def fit(self, max_iter=100, verbose=False):
        for param in self._param.values():
            param.fitter.init()
        n_obs = len(self._obs)
        for _ in range(max_iter):
            if verbose:
                print(".", end="", flush=True)
            converged = list()
            for i in np.random.permutation(n_obs):
                c = self._obs[i].ep_update()
                converged.append(c)
            # Recompute mean and covariance for stability.
            for param in self._param.values():
                param.fitter.recompute()
            if all(converged):
                if verbose:
                    print()
                return
        raise RuntimeError(
                "did not converge after {} iterations".format(max_iter))
