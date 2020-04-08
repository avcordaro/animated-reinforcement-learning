import pytest
import sys
sys.path.extend([".", "..", "../..", "../../.."])
from model.environment_pong import Pong


@pytest.fixture(scope="module")
def env():
    pytest.env = Pong()
    return pytest.env


def test_execute_action(env):
    _, reward, _, episode_done = pytest.env.execute_action(1)
    assert reward == 0 and not episode_done


def test_random_action(env):
    action = pytest.env.random_action()
    assert action in [0, 1, 2, 3, 4, 5]


def test_restart_environment(env):
    pytest.env.restart_environment()
    assert pytest.env.current_state.shape == (4, 84, 84)
