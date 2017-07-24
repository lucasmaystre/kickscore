import abc

from .observation import BinaryObservation
from .item import Item


class Model(metaclass=abc.ABCMeta):

    def __init__(self):
        self._item = dict()
        self.last_t = -float("inf")
        self.observations = list()

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
            item.fitter.allocate()
        for _ in range(max_iter):
            if verbose:
                print(".", end="", flush=True)
            converged = list()
            # Recompute the Gaussian pseudo-observations.
            for obs in self.observations:
                c = obs.ep_update()
                converged.append(c)
            # Recompute the posterior of the score processes.
            for item in self.item.values():
                item.fitter.fit()
            if all(converged):
                if verbose:
                    print()
                return True
        return False  # Did not converge after `max_iter`.


class BinaryModel(Model):

    def __init__(self):
        super().__init__()

    def observe(self, winner, loser, t):
        if t < self.last_t:
            raise ValueError(
                    "observations must be added in chronological order")
        winner = self.item[winner]
        loser = self.item[loser]
        obs = BinaryObservation(winner=winner, loser=loser, t=t)
        self.observations.append(obs)
        winner.link_observation(obs)
        loser.link_observation(obs)

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
