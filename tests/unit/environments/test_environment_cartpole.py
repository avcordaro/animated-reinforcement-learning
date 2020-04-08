import pytest
import sys
sys.path.extend([".", "..", "../..", "../../.."])
from model.environment_cartpole import CartPole


@pytest.fixture(scope="module")
def env():
    pytest.env = CartPole()
    return pytest.env


def test_execute_action(env):
    _, reward, _, episode_done = pytest.env.execute_action(0)
    assert reward == 1 and not episode_done


def test_random_action(env):
    action = pytest.env.random_action()
    assert action in [0, 1]


def test_restart_environment(env):
    pytest.env.restart_environment()
    assert len(pytest.env.current_state) == 4
