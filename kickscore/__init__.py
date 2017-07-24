"""A dynamic skill rating system.

Example usage:

    import kickscore as ks

    model = ks.BinaryModel()
    k = ks.kernel.Matern52()

    model.add_item("audrey", kernel=k)
    model.add_item("benjamin", kernel=k)

    model.observe(winner="audrey", loser="benjamin", t=2.37)
    model.fit()

    model.item["audrey"].scores
"""

from . import kernel

from .model import (
    BinaryModel,
)

from .item import (
    Item,
)
