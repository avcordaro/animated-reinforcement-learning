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


def test_stop_and_reset(controller):
    pytest.gui.animation.update_animation = False
    pytest.controller.stop_and_reset()
    env_current_state = pytest.env.current_state
    agent_sample_value_of_action = pytest.agent.Q[((0, 0), "Right")]
    assert (env_current_state, agent_sample_value_of_action) == ((0, 0), 0)


def test_episode(controller):
    pytest.gui.animation.update_animation = False
    pytest.controller.next_action = pytest.agent.policy[
        self.env.current_state] if random.random() > pytest.controller.epsilon else pytest.env.random_action()
    pytest.controller.run_episode()
