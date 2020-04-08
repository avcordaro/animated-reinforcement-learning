import pytest
import sys
sys.path.extend([".", "..", "../..", "../../.."])
from model.environment_frozenlake import FrozenLake
from model.agent_value_iter import AgentValueIter
from controller.controller_value_iter import ControllerValueIter
from view.animation_frozenlake import AnimationFrozenLake
from view.gui import GUI
from tkinter import Tk


@pytest.fixture(scope="module")
def controller():
    pytest.env = FrozenLake(False)
    pytest.agent = AgentValueIter(pytest.env, gamma=0.9)
    root = Tk()
    animation = AnimationFrozenLake(root, pytest.env.GRID_MAP, update_animation=False)
    pytest.gui = GUI(root, animation)
    pytest.controller = ControllerValueIter(pytest.env, pytest.agent, pytest.gui)
    return pytest.controller


def test_stop_and_reset(controller):
    pytest.gui.animation.update_animation = False
    pytest.controller.stop_and_reset()
    env_current_state = pytest.env.current_state
    agent_transitions = pytest.agent.transitions_counter
    agent_sample_value_of_state = pytest.agent.values_of_state[(0, 0)]
    assert (env_current_state, agent_transitions, agent_sample_value_of_state) == ((0, 0), {}, 0)


def test_greedy_episodes(controller):
    pytest.gui.animation.update_animation = False
    pytest.agent.policy[(0, 0)] = "Right"
    pytest.agent.policy[(0, 1)] = "Down"
    pytest.controller.episode_counter = 0
    pytest.controller.no_of_greedy_episodes = 5
    pytest.controller.REWARD_THRESHOLD = 0
    pytest.controller.run_greedy_episode()


def test_exploration_episodes(controller):
    pytest.gui.animation.update_animation = False
    pytest.controller.episode_counter = 0
    pytest.controller.no_of_exploration_episodes = 50
    pytest.controller.run_exploration_episode()
