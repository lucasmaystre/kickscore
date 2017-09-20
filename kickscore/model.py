import abc
import itertools

from .observation import ProbitObservation, ProbitTieObservation
from .item import Item


class Model(metaclass=abc.ABCMeta):

    def __init__(self):
        self._item = dict()
        self.last_t = -float("inf")
        self.observations = list()

    @property
    def item(self):
        return self._item

    def add_item(self, name, kernel, fitter="batch"):
        self._item[name] = Item(kernel=kernel, fitter=fitter)

    @abc.abstractmethod
    def observe(self, *args, **kwargs):
        """Add a new observation to the dataset."""

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

    @property
    def log_likelihood(self):
        """Log-marginal likelihood of the model."""
        return (sum(o.log_likelihood_contrib for o in self.observations)
                + sum(i.fitter.log_likelihood_contrib
                        for i in self.item.values()))

    def process_items(self, items):
        if isinstance(items, dict):
            return {self.item[k]: float(v) for k, v in items.items()}
        if isinstance(items, list) or isinstance(items, tuple):
            return {self.item[k]: 1.0 for k in items}
        else:
            raise ValueError("items should be a list, a tuple or a dict")


class BinaryModel(Model):

    def __init__(self):
        super().__init__()

    def observe(self, winners, losers, t):
        if t < self.last_t:
            raise ValueError(
                    "observations must be added in chronological order")
        winners = self.process_items(winners)
        losers = self.process_items(losers)
        if len(winners) + len(losers) == 0:
            raise ValueError(
                    "at least one winner or one loser is required")
        obs = ProbitObservation(winners=winners, losers=losers, t=t)
        self.observations.append(obs)
        for item in itertools.chain(winners.keys(), losers.keys()):
            item.link_observation(obs)


class TernaryModel(Model):

    def __init__(self, margin=0.1):
        super().__init__()
        self.margin = margin

    def observe(self, winners, losers, t, tie=False, margin=None):
        if t < self.last_t:
            raise ValueError(
                    "observations must be added in chronological order")
        if margin is None:
            margin = self.margin
        winners = self.process_items(winners)
        losers = self.process_items(losers)
        if len(winners) + len(losers) == 0:
            raise ValueError(
                    "at least one winner or one loser is required")
        if tie:
            obs = ProbitTieObservation(
                    items1=winners, items2=losers, t=t, margin=margin)
        else:
            obs = ProbitObservation(
                    winners=winners, losers=losers, t=t, margin=margin)
        self.observations.append(obs)
        for item in itertools.chain(winners.keys(), losers.keys()):
            item.link_observation(obs)

# Future models
#def observe_count(self, count, attack, defense, t):
#    raise NotImplementedError()
#
#def observe_diff(self, diff, winners, losers, t):
#    raise NotImplementedError()
