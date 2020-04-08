import pytest
import sys
import random
sys.path.extend([".", "..", "../..", "../../.."])
from model.environment_frozenlake import FrozenLake
from model.agent_sarsa import AgentSARSA
from controller.controller_sarsa import ControllerSARSA
from view.animation_frozenlake import AnimationFrozenLake
from view.gui import GUI
from tkinter import Tk


@pytest.fixture(scope="module")
def controller():
    pytest.env = FrozenLake(False)
    pytest.agent = AgentSARSA(pytest.env)
    root = Tk()
    animation = AnimationFrozenLake(root, pytest.env.GRID_MAP, update_animation=False)
    pytest.gui = GUI(root, animation)
    pytest.controller = ControllerSARSA(pytest.env, pytest.agent, pytest.gui)
    return pytest.controller


def test_start(controller):
    pytest.gui.animation.update_animation = False
    pytest.controller.start()
