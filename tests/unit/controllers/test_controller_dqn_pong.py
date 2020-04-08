import pytest
import sys
sys.path.extend([".", "..", "../..", "../../.."])
from model.environment_pong import Pong
from controller.controller_dqn_pong import ControllerDQNPong
from view.gui import GUI
from tkinter import Tk


@pytest.fixture(scope="module")
def controller():
    env = Pong()
    pytest.gui = GUI(Tk(), None)
    pytest.controller = ControllerDQNPong(env, None, pytest.gui)
    return pytest.controller


def test_update_timescale(controller):
    pytest.controller.update_timescale(50)
    assert pytest.controller.timescale == 50


def test_toggle_animation(controller):
    pytest.gui.animation_toggle.set(0)
    assert pytest.controller.timescale == 0
