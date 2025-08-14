"""A dynamic skill rating system.

Example usage:

    import kickscore as ks

    model = ks.BinaryModel()
    k = ks.kernel.Matern52(var=1.0, lscale=1.0)

    model.add_item("audrey", kernel=k)
    model.add_item("benjamin", kernel=k)

    model.observe(winners=["audrey"], losers=["benjamin"], t=2.37)
    model.fit()

    model.item["audrey"].scores
"""

from . import kernel
from .item import (
    Item,
)
from .model import (
    BinaryModel,
    CountDiffModel,
    CountModel,
    DifferenceModel,
    TernaryModel,
)

__all__ = [
    "BinaryModel",
    "TernaryModel",
    "DifferenceModel",
    "CountModel",
    "CountDiffModel",
    "Item",
    "kernel",
]
