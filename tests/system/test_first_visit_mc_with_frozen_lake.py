import pytest
import sys
sys.path.extend([".", "..", "../..", "../../.."])
from model.environment_frozenlake import FrozenLake
from model.agent_first_visit_mc import AgentFirstVisitMC
from controller.controller_first_visit_mc import ControllerFirstVisitMC
from view.animation_frozenlake import AnimationFrozenLake
from view.gui import GUI
from tkinter import Tk


@pytest.fixture(scope="module")
def controller():
    pytest.env = FrozenLake(False)
    pytest.agent = AgentFirstVisitMC(pytest.env)
    root = Tk()
    animation = AnimationFrozenLake(root, pytest.env.GRID_MAP, update_animation=False)
    pytest.gui = GUI(root, animation)
    pytest.controller = ControllerFirstVisitMC(pytest.env, pytest.agent, pytest.gui)
    return pytest.controller


def test_start(controller):
    pytest.gui.animation.update_animation = False
    pytest.controller.start()
