"""
Example usage:

    import kickscore as ks

    model = ks.BinaryModel()
    k = ks.kernel.Matern52()

    model.add_item("audrey", kernel=k)
    model.add_item("benjamin", kernel=k)

    model.observe(winner="audrey", loser="benjamin", t=2.37)
    model.fit()

    model.item["audrey"].means
    model.item["audrey"].vars
"""

import abc
import numpy as np

from .observation import BinaryObservation
from .item import Item


class Model(metaclass=abc.ABCMeta):

    def __init__(self):
        self._item = dict()
        self._obs = list()

    @property
    def item(self):
        return self._item

    def add_item(self, name, kernel):
        self._item[name] = Item(kernel=kernel)

    @abc.abstractmethod
    def observe(self, *args, **kwargs):
        pass

    def fit(self, max_iter=100, verbose=False):
        for item in self._item.values():
            item.fitter.init()
        n_obs = len(self._obs)
        for _ in range(max_iter):
            if verbose:
                print(".", end="", flush=True)
            converged = list()
            for i in np.random.permutation(n_obs):
                c = self._obs[i].ep_update()
                converged.append(c)
            # Recompute mean and covariance for stability.
            for item in self._item.values():
                item.fitter.recompute()
            if all(converged):
                if verbose:
                    print()
                return
        raise RuntimeError(
                "did not converge after {} iterations".format(max_iter))


class BinaryModel(Model):

    def __init__(self):
        super().__init__()

    def observe(self, winner, loser, t):
        for name in (winner, loser):
            if name not in self._item:
                raise ValueError("item {!r} not found".format(name))
        self._obs.append(BinaryObservation(
                winner=self._item[winner], loser=self._item[loser], t=t))

# Future models
#def observe_ternary(self, params1, params2, outcome, t, margin=0.5):
#    # Outcome can be "params1", "params2" or "tie".
#    raise NotImplementedError()
#
#def observe_count(self, count, attack, defense, t):
#    raise NotImplementedError()
#
#def observe_diff(self, diff, winners, losers, t):
#    raise NotImplementedError()
