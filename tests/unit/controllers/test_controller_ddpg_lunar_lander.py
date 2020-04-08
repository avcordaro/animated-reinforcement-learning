import pytest
import sys
sys.path.extend([".", "..", "../..", "../../.."])
from model.environment_lunar_lander import LunarLander
from controller.controller_ddpg_lunar_lander import ControllerDDPGLunarLander
from view.gui import GUI
from tkinter import Tk


@pytest.fixture(scope="module")
def controller():
    env = LunarLander()
    pytest.gui = GUI(Tk(), None)
    pytest.controller = ControllerDDPGLunarLander(env, None, pytest.gui)
    return pytest.controller


def test_update_timescale(controller):
    pytest.controller.update_timescale(50)
    assert pytest.controller.timescale == 50


def test_toggle_animation(controller):
    pytest.gui.animation_toggle.set(0)
    assert pytest.controller.timescale == 0
