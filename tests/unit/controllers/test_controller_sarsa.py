import pytest
import sys
sys.path.extend([".", "..", "../..", "../../.."])
from model.environment_frozenlake import FrozenLake
from controller.controller_sarsa import ControllerSARSA
from view.gui import GUI
from tkinter import Tk


@pytest.fixture(scope="module")
def controller():
    env = FrozenLake(False)
    pytest.gui = GUI(Tk(), None)
    pytest.controller = ControllerSARSA(env, None, pytest.gui)
    return pytest.controller


def test_update_timescale(controller):
    pytest.controller.update_timescale(50)
    assert pytest.controller.timescale == 50


def test_toggle_animation(controller):
    pytest.gui.animation_toggle.set(0)
    assert pytest.controller.timescale == 0
