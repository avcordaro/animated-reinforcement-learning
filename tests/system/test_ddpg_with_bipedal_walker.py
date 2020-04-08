import pytest
import sys
sys.path.extend([".", "..", "../..", "../../.."])
from model.environment_bipedal_walker import BipedalWalker
from model.agent_ddpg_bipedal_walker import AgentDDPGBipedalWalker
from controller.controller_ddpg_bipedal_walker import ControllerDDPGBipedalWalker
from view.animation_bipedal_walker import AnimationBipedalWalker
from view.gui import GUI
from tkinter import Tk


@pytest.fixture(scope="module")
def controller():
    pytest.env = BipedalWalker()
    pytest.agent = AgentDDPGBipedalWalker(pytest.env)
    root = Tk()
    animation = AnimationBipedalWalker(root, pytest.env, update_animation=False)
    pytest.gui = GUI(root, animation)
    pytest.controller = ControllerDDPGBipedalWalker(pytest.env, pytest.agent, pytest.gui)
    return pytest.controller


def test_start(controller):
    pytest.gui.animation.update_animation = False
    pytest.controller.start()
