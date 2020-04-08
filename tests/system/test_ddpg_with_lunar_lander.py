import pytest
import sys
sys.path.extend([".", "..", "../..", "../../.."])
from model.environment_lunar_lander import LunarLander
from model.agent_ddpg_lunar_lander import AgentDDPGLunarLander
from controller.controller_ddpg_lunar_lander import ControllerDDPGLunarLander
from view.animation_lunar_lander import AnimationLunarLander
from view.gui import GUI
from tkinter import Tk


@pytest.fixture(scope="module")
def controller():
    pytest.env = LunarLander()
    pytest.agent = AgentDDPGLunarLander(pytest.env)
    root = Tk()
    animation = AnimationLunarLander(root, pytest.env, update_animation=False)
    pytest.gui = GUI(root, animation)
    pytest.controller = ControllerDDPGLunarLander(pytest.env, pytest.agent, pytest.gui)
    return pytest.controller


def test_start(controller):
    pytest.gui.animation.update_animation = False
    pytest.controller.start()
