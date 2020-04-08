import pytest
import sys
sys.path.extend([".", "..", "../..", "../../.."])
from model.environment_cartpole import CartPole
from model.agent_dqn_cartpole import AgentDQNCartPole
from controller.controller_dqn_cartpole import ControllerDQNCartPole
from view.animation_cartpole import AnimationCartPole
from view.gui import GUI
from tkinter import Tk


@pytest.fixture(scope="module")
def controller():
    pytest.env = CartPole()
    pytest.agent = AgentDQNCartPole(pytest.env)
    root = Tk()
    animation = AnimationCartPole(root, pytest.env, update_animation=False)
    pytest.gui = GUI(root, animation)
    pytest.controller = ControllerDQNCartPole(pytest.env, pytest.agent, pytest.gui)
    return pytest.controller


def test_start(controller):
    pytest.gui.animation.update_animation = False
    pytest.controller.start()
