import numpy as np
from .fitter import BatchFitter, RecursiveFitter


class Item:

    def __init__(self, kernel, fitter):
        if fitter == "batch":
            self.fitter = BatchFitter(kernel)
        elif fitter == "recursive":
            self.fitter = RecursiveFitter(kernel)
        else:
            raise ValueError("invalid fitter type '{}'".format(fitter))
        self.observations = list()

    @property
    def kernel(self):
        return self.fitter.kernel

    @property
    def scores(self):
        return self.fitter.posterior

    def predict(self, ts):
        return self.fitter.predict(ts)

    def link_observation(self, obs):
        self.observations.append(obs)
