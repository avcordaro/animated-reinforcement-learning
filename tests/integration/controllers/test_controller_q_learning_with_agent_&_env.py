import pytest
import sys
sys.path.extend([".", "..", "../..", "../../.."])
from model.environment_frozenlake import FrozenLake
from model.agent_q_learning import AgentQLearning
from controller.controller_q_learning import ControllerQLearning
from view.animation_frozenlake import AnimationFrozenLake
from view.gui import GUI
from tkinter import Tk


@pytest.fixture(scope="module")
def controller():
    pytest.env = FrozenLake(False)
    pytest.agent = AgentQLearning(pytest.env)
    root = Tk()
    animation = AnimationFrozenLake(root, pytest.env.GRID_MAP, update_animation=False)
    pytest.gui = GUI(root, animation)
    pytest.controller = ControllerQLearning(pytest.env, pytest.agent, pytest.gui)
    return pytest.controller


def test_stop_and_reset(controller):
    pytest.gui.animation.update_animation = False
    pytest.controller.stop_and_reset()
    env_current_state = pytest.env.current_state
    agent_sample_value_of_action = pytest.agent.Q[((0, 0), "Right")]
    assert (env_current_state, agent_sample_value_of_action) == ((0, 0), 0)


def test_episode(controller):
    pytest.gui.animation.update_animation = False
    pytest.controller.run_episode()
