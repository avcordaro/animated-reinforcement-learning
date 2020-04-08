import pytest
import sys
import numpy as np
sys.path.extend([".", "..", "../..", "../../.."])
from model.environment_bipedal_walker import BipedalWalker


@pytest.fixture(scope="module")
def env():
    pytest.env = BipedalWalker()
    return pytest.env


def test_execute_action(env):
    _, _, _, episode_done = pytest.env.execute_action(np.array([0.0, 0.0, 0.0, 0.0]))
    assert not episode_done


def test_restart_environment(env):
    pytest.env.restart_environment()
    assert len(pytest.env.current_state) == 24
