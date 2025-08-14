import abc
from collections.abc import Sequence
from typing import Any, Literal

from .item import Item
from .kernel import Kernel
from .observation import (
    GaussianObservation,
    LogitTieObservation,
    LogitWinObservation,
    Observation,
    PoissonObservation,
    ProbitTieObservation,
    ProbitWinObservation,
    SkellamObservation,
)


class Model(metaclass=abc.ABCMeta):
    def __init__(self):
        self._item: dict[str, Item] = dict()
        self.last_t: float = -float("inf")
        self.observations: list[Observation] = list()
        self._last_method: Literal["ep", "kl"] | None = None  # Last method used to fit the model.

    @property
    def item(self) -> dict[str, Item]:
        return self._item

    def add_item(
        self,
        name: str,
        kernel: Kernel,
        fitter: Literal["batch", "recursive"] = "recursive",
    ) -> None:
        if name in self._item:
            raise ValueError("item '{}' already added".format(name))
        self._item[name] = Item(kernel=kernel, fitter=fitter)

    @abc.abstractmethod
    def observe(self, *args: Any, **kwargs: Any) -> None:
        """Add a new observation to the dataset."""

    def fit(
        self,
        method: Literal["ep", "kl"] = "ep",
        lr: float = 1.0,
        tol: float = 1e-3,
        max_iter: int = 100,
        verbose: bool = False,
    ) -> bool:
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
                print("iteration {}, max diff: {:.5f}".format(i + 1, max_diff), flush=True)
            if max_diff < tol:
                return True
        return False  # Did not converge after `max_iter`.

    @abc.abstractmethod
    def probabilities(self, *args: Any, **kwargs: Any) -> Any:
        """Compute the probability of outcomes."""

    @property
    def log_likelihood(self) -> float:
        """Estimate of log-marginal likelihood of the model."""
        if self._last_method == "ep":
            contrib = lambda x: x.ep_log_likelihood_contrib
        else:  # self._last_method == "kl"
            contrib = lambda x: x.kl_log_likelihood_contrib
        return sum(contrib(o) for o in self.observations) + sum(
            contrib(i.fitter) for i in self.item.values()
        )

    def process_items(
        self,
        items: dict[str, Any] | list[str],
        sign: Literal[-1, +1] = +1,
    ) -> list[tuple[Item, float]]:
        if isinstance(items, dict):
            return [(self.item[k], sign * float(v)) for k, v in items.items()]
        if isinstance(items, list) or isinstance(items, tuple):
            return [(self.item[k], sign) for k in items]
        else:
            raise ValueError("items should be a list, a tuple or a dict")

    def plot_scores(
        self,
        items: Sequence[str],
        resolution: float | None = None,
        figsize: float | None = None,
        timestamps: bool = False,
    ) -> Any:
        # Delayed import in order to avoid a hard dependency on Matplotlib.
        from .plotting import plot_scores

        return plot_scores(self, items, resolution, figsize, timestamps)


class BinaryModel(Model):
    def __init__(self, obs_type: Literal["probit", "logit"] = "probit"):
        super().__init__()
        if obs_type == "probit":
            self._win_obs = ProbitWinObservation
        elif obs_type == "logit":
            self._win_obs = LogitWinObservation
        else:
            raise ValueError("unknown observation type: '{}'".format(obs_type))

    def observe(
        self,
        winners: dict[str, Any] | list[str],
        losers: dict[str, Any] | list[str],
        t: float,
    ) -> None:
        if t < self.last_t:
            raise ValueError("observations must be added in chronological order")
        elems = self.process_items(winners, sign=+1) + self.process_items(losers, sign=-1)
        obs = self._win_obs(elems, t=t)
        self.observations.append(obs)
        self.last_t = t

    def probabilities(
        self,
        team1: dict[str, Any] | list[str],
        team2: dict[str, Any] | list[str],
        t: float,
    ) -> tuple[float, float]:
        elems = self.process_items(team1, sign=+1) + self.process_items(team2, sign=-1)
        prob = self._win_obs.probability(elems, t)
        return (prob, 1 - prob)


class TernaryModel(Model):
    def __init__(self, margin: float = 0.1, obs_type: Literal["probit", "logit"] = "probit"):
        super().__init__()
        if obs_type == "probit":
            self._win_obs = ProbitWinObservation
            self._tie_obs = ProbitTieObservation
        elif obs_type == "logit":
            self._win_obs = LogitWinObservation
            self._tie_obs = LogitTieObservation
        else:
            raise ValueError("unknown observation type: '{}'".format(obs_type))
        self.margin = margin

    def observe(
        self,
        winners: dict[str, Any] | list[str],
        losers: dict[str, Any] | list[str],
        t: float,
        tie: bool = False,
        margin: float | None = None,
    ) -> None:
        if t < self.last_t:
            raise ValueError("observations must be added in chronological order")
        if margin is None:
            margin = self.margin
        elems = self.process_items(winners, sign=+1) + self.process_items(losers, sign=-1)
        if tie:
            obs = self._tie_obs(elems, t=t, margin=margin)
        else:
            obs = self._win_obs(elems, t=t, margin=margin)
        self.observations.append(obs)
        self.last_t = t

    def probabilities(
        self,
        team1: dict[str, Any] | list[str],
        team2: dict[str, Any] | list[str],
        t: float,
        margin: float | None = None,
    ) -> tuple[float, float, float]:
        if margin is None:
            margin = self.margin
        elems = self.process_items(team1, sign=+1) + self.process_items(team2, sign=-1)
        prob1 = self._win_obs.probability(elems, t, margin)
        prob2 = self._tie_obs.probability(elems, t, margin)
        return (prob1, prob2, 1 - prob1 - prob2)


class DifferenceModel(Model):
    def __init__(self, var: float = 1.0):
        super().__init__()
        self.var = var

    def observe(
        self,
        items1: dict[str, Any] | list[str],
        items2: dict[str, Any] | list[str],
        diff: float,
        var: float | None = None,
        t: float = 0.0,
    ) -> None:
        if t < self.last_t:
            raise ValueError("observations must be added in chronological order")
        if var is None:
            var = self.var
        items = self.process_items(items1, sign=+1) + self.process_items(items2, sign=-1)
        obs = GaussianObservation(items, diff, var, t=t)
        self.observations.append(obs)
        self.last_t = t

    def probabilities(
        self,
        items1: dict[str, Any] | list[str],
        items2: dict[str, Any] | list[str],
        threshold: float = 0.0,
        var: float | None = None,
        t: float = 0.0,
    ) -> tuple[float, float]:
        if var is None:
            var = self.var
        items = self.process_items(items1, sign=+1) + self.process_items(items2, sign=-1)
        prob = GaussianObservation.probability(items, threshold, var, t=t)
        return (prob, 1 - prob)


class CountModel(Model):
    def observe(
        self,
        items1: dict[str, Any] | list[str],
        items2: dict[str, Any] | list[str],
        count: int,
        t: float = 0.0,
    ) -> None:
        assert isinstance(count, int) and count >= 0
        if t < self.last_t:
            raise ValueError("observations must be added in chronological order")
        items = self.process_items(items1, sign=+1) + self.process_items(items2, sign=-1)
        obs = PoissonObservation(items, count, t=t)
        self.observations.append(obs)
        self.last_t = t

    def probabilities(
        self,
        items1: dict[str, Any],
        items2: dict[str, Any],
        t: float = 0.0,
    ) -> tuple[float, ...]:
        items = self.process_items(items1, sign=+1) + self.process_items(items2, sign=-1)
        probs = list()
        while sum(probs) < 0.999:
            probs.append(PoissonObservation.probability(items, count=len(probs), t=t))
        return tuple(probs)


class CountDiffModel(Model):
    def __init__(self, base_rate: float = 0.0):
        super().__init__()
        self._base_rate = base_rate

    def observe(
        self,
        items1: dict[str, Any] | list[str],
        items2: dict[str, Any] | list[str],
        diff: int,
        t: float = 0.0,
    ) -> None:
        if t < self.last_t:
            raise ValueError("observations must be added in chronological order")
        items = self.process_items(items1, sign=+1) + self.process_items(items2, sign=-1)
        obs = SkellamObservation(items, diff, self._base_rate, t=t)
        self.observations.append(obs)
        self.last_t = t

    def probabilities(
        self,
        items1: dict[str, Any] | list[str],
        items2: dict[str, Any] | list[str],
        t: float = 0.0,
    ) -> tuple[float, ...]:
        items = self.process_items(items1, sign=+1) + self.process_items(items2, sign=-1)
        k = 0
        probs = [SkellamObservation.probability(items, k, self._base_rate, t=t)]
        while sum(probs) < 0.999:
            k += 1
            probs.append(SkellamObservation.probability(items, k, self._base_rate, t=t))
            probs.insert(0, SkellamObservation.probability(items, -k, self._base_rate, t=t))
        return tuple(probs)
