import pytest
import sys
sys.path.extend([".", "..", "../..", "../../.."])
from model.environment_frozenlake import FrozenLake
from model.agent_reinforce import AgentREINFORCE
from controller.controller_reinforce import ControllerREINFORCE
from view.animation_frozenlake import AnimationFrozenLake
from view.gui import GUI
from tkinter import Tk


@pytest.fixture(scope="module")
def controller():
    pytest.env = FrozenLake(False)
    pytest.agent = AgentREINFORCE(pytest.env)
    root = Tk()
    animation = AnimationFrozenLake(root, pytest.env.GRID_MAP, update_animation=False)
    pytest.gui = GUI(root, animation)
    pytest.controller = ControllerREINFORCE(pytest.env, pytest.agent, pytest.gui)
    return pytest.controller


def test_stop_and_reset(controller):
    pytest.gui.animation.update_animation = False
    pytest.controller.stop_and_reset()
    env_current_state = pytest.env.current_state
    num_of_weights = len(pytest.agent.theta)
    assert (env_current_state, num_of_weights) == ((0, 0), 2)


def test_episode(controller):
    pytest.gui.animation.update_animation = False
    pytest.controller.run_episode()
