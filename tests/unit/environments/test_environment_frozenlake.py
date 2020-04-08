import pytest
import sys
import numpy as np
sys.path.extend([".", "..", "../..", "../../.."])
from model.environment_frozenlake import FrozenLake


@pytest.fixture(scope="module")
def env():
    pytest.env = FrozenLake(False)
    pytest.env.current_state = (2, 1)
    return pytest.env


def test_state_space_size(env):
    assert len(pytest.env.state_space) == 16


def test_goal_and_start_state(env):
    start = pytest.env.start_state
    goal = pytest.env.goal_state
    assert (start, goal) == ((0, 0), (3, 3))


def test_random_action(env):
    action = pytest.env.random_action()
    assert action in ["Left", "Up", "Right", "Down"]


@pytest.mark.parametrize("action, expected_state",
                         [("Right", (2, 2)), ("Down", (3, 2)), ("Left", (3, 1)), ("Up", (2, 1))])
def test_movement_actions(env, action, expected_state):
    new_state, reward, _ = pytest.env.execute_action(action)
    assert (new_state, reward) == (expected_state, -1)


@pytest.mark.parametrize("start_state, action",
                         [((0, 1), "Up",), ((3, 2), "Down"), ((2, 0), "Left"), ((0, 3), "Right")])
def test_movement_actions_at_border(env, start_state, action):
    pytest.env.current_state = start_state
    new_state, reward, _ = pytest.env.execute_action(action)
    assert (new_state, reward) == (start_state, -1)


@pytest.mark.parametrize("start_state, action", [((0, 1), "Down",), ((1, 2), "Right"), ((3, 1), "Left")])
def test_hole_detection(env, start_state, action):
    pytest.env.current_state = start_state
    _, reward, episode_done = pytest.env.execute_action(action)
    assert (reward, episode_done) == (-100, True)


def test_goal_reached(env):
    pytest.env.current_state = (3, 2)
    _, reward, episode_done = pytest.env.execute_action("Right")
    assert (reward, episode_done) == (200, True)


def test_restart_environment(env):
    pytest.env.current_state = (3, 3)
    pytest.env.restart_environment()
    assert pytest.env.current_state == (0, 0)


@pytest.fixture
def env_slippery():
    pytest.env_slippery = FrozenLake(True)
    return pytest.env_slippery


def test_movement_action_when_slippery(env_slippery):
    pytest.env_slippery.current_state = (2, 1)
    new_state, _, _ = pytest.env_slippery.execute_action("Down")
    assert new_state in [(3, 1), (2, 0), (2, 2)]


def test_get_feature_vector(env):
    state = (0, 0)
    action = "Left"
    feature_vector = pytest.env.get_feature_vector(state, action)
    assert np.allclose(feature_vector, np.array([1, 0]))
