from abc import ABC, abstractmethod


class Controller(ABC):
    """
    Abstract Base Class for controllers. Defines the compulsory methods to be included
    in concrete implementations which inherit from this class.
    """

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def update_timescale(self):
        pass

    @abstractmethod
    def toggle_animation(self):
        pass

    @abstractmethod
    def stop_and_reset(self):
        pass
