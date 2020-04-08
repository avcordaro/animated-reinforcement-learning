from abc import ABC, abstractmethod


class Agent(ABC):
    """
    Abstract Base Class for agents. Defines the compulsory methods to be included
    in concrete implementations which inherit from this class.
    """

    @abstractmethod
    def restart_matrices(self):
        pass

    @abstractmethod
    def step(self):
        pass
