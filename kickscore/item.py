import numpy as np
from .fitter import BatchFitter


class Item:

    def __init__(self, kernel, fitter="batch"):
        if fitter == "batch":
            self.fitter = BatchFitter(kernel)
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
