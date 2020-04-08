from abc import ABC, abstractmethod


class Environment(ABC):
    """
    Abstract Base Class for environments. Defines the compulsory methods to be included
    in concrete implementations which inherit from this class.
    """

    @abstractmethod
    def execute_action(self):
        pass

    @abstractmethod
    def random_action(self):
        pass

    @abstractmethod
    def restart_environment(self):
        pass
