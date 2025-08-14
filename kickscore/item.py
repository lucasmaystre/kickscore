from typing import Literal

from numpy.typing import NDArray

from .fitter import BatchFitter, RecursiveFitter
from .kernel import Kernel


class Item:
    def __init__(self, kernel: Kernel, fitter: Literal["batch", "recursive"]):
        if fitter == "batch":
            self.fitter = BatchFitter(kernel)
        elif fitter == "recursive":
            self.fitter = RecursiveFitter(kernel)
        else:
            raise ValueError("invalid fitter type '{}'".format(fitter))

    @property
    def kernel(self) -> Kernel:
        return self.fitter.kernel

    @property
    def scores(self) -> tuple[NDArray, NDArray, NDArray]:
        return self.fitter.posterior

    def predict(self, ts: NDArray) -> tuple[NDArray, NDArray]:
        return self.fitter.predict(ts)
