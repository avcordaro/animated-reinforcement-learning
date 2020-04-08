import pytest
import sys
import numpy as np
sys.path.extend([".", "..", "../..", "../../.."])
from model.agent_reinforce import AgentREINFORCE
from model.environment_frozenlake import FrozenLake


@pytest.fixture(scope="module")
def agent():
    pytest.env = FrozenLake(False)
    pytest.agent = AgentREINFORCE(pytest.env)
    return pytest.agent


def test_episode_with_random_policy(agent):
    pytest.agent.restart_matrices()
    state = pytest.env.current_state
    reward = 0
    episode_done = False
    while not episode_done:
        softmax_probs = pytest.agent.policy(state)
        action_choice = np.random.choice(pytest.env.NUM_ACTIONS, p=softmax_probs)
        action = pytest.env.action_space[action_choice]
        state, reward, episode_done = pytest.agent.step(action)
    terminal_state = True if state in [(1, 1), (1, 3), (2, 3), (3, 0), (3, 3)] else False
    reward_valid = True if reward in [-100, -1, 200] else False
    assert (terminal_state, reward_valid) == (True, True)
