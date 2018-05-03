import abc
import numpy as np

from math import log, exp


class Observation(metaclass=abc.ABCMeta):

    @abc.abstractstaticmethod
    def match_moments(em, ev, mm, mv):
        """Compute statistics of the hybrid distribution."""
