import pytest
import sys
sys.path.extend([".", "..", "../..", "../../.."])
from model.environment_cliff_walking import CliffWalking


@pytest.fixture(scope="module")
def env():
    pytest.env = CliffWalking()
    pytest.env.current_state = (1, 0)
    return pytest.env


def test_state_space_size(env):
    assert len(pytest.env.state_space) == 15


def test_goal_and_start_state(env):
    start = pytest.env.start_state
    goal = pytest.env.goal_state
    assert (start, goal) == ((0, 0), (0, 4))


def test_random_action(env):
    action = pytest.env.random_action()
    assert action in ["Left", "Up", "Right", "Down"]


@pytest.mark.parametrize("action, expected_state",
                         [("Right", (1, 1)), ("Down", (2, 1)), ("Left", (2, 0)), ("Up", (1, 0))])
def test_movement_actions(env, action, expected_state):
    new_state, reward, _ = pytest.env.execute_action(action)
    assert (new_state, reward) == (expected_state, -1)


@pytest.mark.parametrize("start_state, action",
                         [((0, 0), "Up",), ((2, 0), "Down"), ((1, 0), "Left"), ((2, 4), "Right")])
def test_movement_actions_at_border(env, start_state, action):
    pytest.env.current_state = start_state
    new_state, reward, _ = pytest.env.execute_action(action)
    assert (new_state, reward) == (start_state, -1)


@pytest.mark.parametrize("start_state, action", [((0, 0), "Right",), ((1, 1), "Up"), ((1, 2), "Up")])
def test_hole_detection(env, start_state, action):
    pytest.env.current_state = start_state
    _, reward, episode_done = pytest.env.execute_action(action)
    assert (reward, episode_done) == (-100, True)


def test_goal_reached(env):
    pytest.env.current_state = (1, 4)
    _, reward, episode_done = pytest.env.execute_action("Up")
    assert (reward, episode_done) == (-1, True)


def test_restart_environment(env):
    pytest.env.current_state = (2, 3)
    pytest.env.restart_environment()
    assert pytest.env.current_state == (0, 0)
