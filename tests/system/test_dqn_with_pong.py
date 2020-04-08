import pytest
import sys
sys.path.extend([".", "..", "../..", "../../.."])
from model.environment_pong import Pong
from model.agent_dqn_pong import AgentDQNPong
from controller.controller_dqn_pong import ControllerDQNPong
from view.animation_pong import AnimationPong
from view.gui import GUI
from tkinter import Tk


@pytest.fixture(scope="module")
def controller():
    pytest.env = Pong()
    pytest.agent = AgentDQNPong(pytest.env)
    root = Tk()
    animation = AnimationPong(root, pytest.env, update_animation=False)
    pytest.gui = GUI(root, animation)
    pytest.controller = ControllerDQNPong(pytest.env, pytest.agent, pytest.gui)
    return pytest.controller


def test_start(controller):
    pytest.gui.animation.update_animation = False
    pytest.controller.start()
