import abc


class Observation(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def ep_update(self):
        """EP site update."""
        pass
