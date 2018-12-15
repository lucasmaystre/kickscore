import abc

from .item import Item
from .observation import ProbitObservation, ProbitTieObservation


class Model(metaclass=abc.ABCMeta):

    def __init__(self):
        self._item = dict()
        self.last_t = -float("inf")
        self.observations = list()
        self._last_method = None  # Last method used to fit the model.

    @property
    def item(self):
        return self._item

    def add_item(self, name, kernel, fitter="recursive"):
        if name in self._item:
            raise ValueError("item '{}' already added".format(name))
        self._item[name] = Item(kernel=kernel, fitter=fitter)

    @abc.abstractmethod
    def observe(self, *args, **kwargs):
        """Add a new observation to the dataset."""

    def fit(self, method="ep", lr=1.0, tol=1e-3, max_iter=100, verbose=False):
        if method == "ep":
            update = lambda obs: obs.ep_update(lr=lr)
        elif method == "kl":
            update = lambda obs: obs.kl_update(lr=lr)
        else:
            raise ValueError("'method' should be one of: 'ep', 'kl'")
        self._last_method = method
        for item in self._item.values():
            item.fitter.allocate()
        for i in range(max_iter):
            max_diff = 0.0
            # Recompute the Gaussian pseudo-observations.
            for obs in self.observations:
                diff = update(obs)
                max_diff = max(max_diff, diff)
            # Recompute the posterior of the score processes.
            for item in self.item.values():
                item.fitter.fit()
            if verbose:
                print("iteration {}, max diff: {:.5f}".format(
                        i+1, max_diff), flush=True)
            if max_diff < tol:
                return True
        return False  # Did not converge after `max_iter`.

    @abc.abstractmethod
    def probabilities(self, *args, **kwargs):
        """Compute the probability of outcomes."""

    @property
    def log_likelihood(self):
        """Estimate of log-marginal likelihood of the model."""
        if self._last_method == "ep":
            contrib = lambda x: x.ep_log_likelihood_contrib
        else:  # self._last_method == "kl"
            contrib = lambda x: x.kl_log_likelihood_contrib
        return (sum(contrib(o) for o in self.observations)
                + sum(contrib(i.fitter) for i in self.item.values()))

    def process_items(self, items, sign=+1):
        if isinstance(items, dict):
            return [(self.item[k], sign * float(v)) for k, v in items.items()]
        if isinstance(items, list) or isinstance(items, tuple):
            return [(self.item[k], sign) for k in items]
        else:
            raise ValueError("items should be a list, a tuple or a dict")


class BinaryModel(Model):

    def __init__(self):
        super().__init__()

    def observe(self, winners, losers, t):
        if t < self.last_t:
            raise ValueError(
                    "observations must be added in chronological order")
        elems = (self.process_items(winners, sign=+1)
                + self.process_items(losers, sign=-1))
        obs = ProbitObservation(elems, t=t)
        self.observations.append(obs)
        for item, _ in elems:
            item.link_observation(obs)
        self.last_t = t

    def probabilities(self, team1, team2, t):
        elems = (self.process_items(team1, sign=+1)
                + self.process_items(team2, sign=-1))
        prob = ProbitObservation.probability(elems, t)
        return (prob, 1 - prob)


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
        elems = (self.process_items(winners, sign=+1)
                + self.process_items(losers, sign=-1))
        if tie:
            obs = ProbitTieObservation(elems, t=t, margin=margin)
        else:
            obs = ProbitObservation(elems, t=t, margin=margin)
        self.observations.append(obs)
        for item, _ in elems:
            item.link_observation(obs)
        self.last_t = t

    def probabilities(self, team1, team2, t, margin=None):
        if margin is None:
            margin = self.margin
        elems = (self.process_items(team1, sign=+1)
                + self.process_items(team2, sign=-1))
        prob1 = ProbitObservation.probability(elems, t, margin)
        prob2 = ProbitTieObservation.probability(elems, t, margin)
        return (prob1, prob2, 1 - prob1 - prob2)

# Future models
#def observe_count(self, count, attack, defense, t):
#    raise NotImplementedError()
#
#def observe_diff(self, diff, winners, losers, t):
#    raise NotImplementedError()
