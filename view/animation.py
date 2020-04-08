from abc import ABC, abstractmethod


class Animation(ABC):
    """
    Abstract Base Class for animations. Defines the compulsory methods to be included
    in concrete implementations which inherit from this class.
    """

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def restart_environment(self):
        pass
