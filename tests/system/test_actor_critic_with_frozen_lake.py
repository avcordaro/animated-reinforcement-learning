import pytest
import sys
sys.path.extend([".", "..", "../..", "../../.."])
from model.environment_frozenlake import FrozenLake
from model.agent_actor_critic import AgentActorCritic
from controller.controller_actor_critic import ControllerActorCritic
from view.animation_frozenlake import AnimationFrozenLake
from view.gui import GUI
from tkinter import Tk


@pytest.fixture(scope="module")
def controller():
    pytest.env = FrozenLake(False)
    pytest.agent = AgentActorCritic(pytest.env)
    root = Tk()
    animation = AnimationFrozenLake(root, pytest.env.GRID_MAP, update_animation=False)
    pytest.gui = GUI(root, animation)
    pytest.controller = ControllerActorCritic(pytest.env, pytest.agent, pytest.gui)
    return pytest.controller


def test_start(controller):
    pytest.gui.animation.update_animation = False
    pytest.controller.start()
