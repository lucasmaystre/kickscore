import pytest

from kickscore.item import Item
from kickscore.kernel import Matern32


def test_fitter_types():
    """Items can be constructed with batch or recursive fitters."""
    kernel = Matern32(1.0, 1.0)
    Item(kernel, fitter="batch")
    Item(kernel, fitter="recursive")
    with pytest.raises(ValueError):
        Item(kernel, fitter="parallel")  # pyright: ignore[reportArgumentType]
