import pytest
import sys
sys.path.extend([".", "..", "../..", "../../.."])
from model.agent_policy_iter import AgentPolicyIter
from model.environment_frozenlake import FrozenLake


@pytest.fixture(scope="module")
def agent():
    pytest.env = FrozenLake(False)
    pytest.agent = AgentPolicyIter(pytest.env, gamma=0.9)
    return pytest.agent


def test_episode_with_random_steps(agent):
    new_state = None
    reward = 0
    episode_done = False
    while not episode_done:
        new_state, reward, episode_done = pytest.agent.random_step()
    terminal_state = True if new_state in [(1, 1), (1, 3), (2, 3), (3, 0), (3, 3)] else False
    reward_valid = True if reward in [-100, -1, 200] else False
    assert (terminal_state, reward_valid) == (True, True)


def test_episode_with_greedy_steps(agent):
    pytest.env.restart_environment()
    pytest.agent.restart_matrices()
    pytest.agent.policy[(0, 0)] = "Right"
    pytest.agent.policy[(0, 1)] = "Down"  # so the agent doesn't just go Left in an infinite loop
    new_state = None
    reward = 0
    episode_done = False
    while not episode_done:
        new_state, reward, episode_done = pytest.agent.step()
    terminal_state = True if new_state in [(1, 1), (1, 3), (2, 3), (3, 0), (3, 3)] else False
    reward_valid = True if reward in [-100, -1, 200] else False
    assert (terminal_state, reward_valid) == (True, True)
