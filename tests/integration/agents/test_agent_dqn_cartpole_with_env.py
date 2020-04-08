import pytest
import sys
sys.path.extend([".", "..", "../..", "../../.."])
from model.agent_dqn_cartpole import AgentDQNCartPole
from model.environment_cartpole import CartPole


@pytest.fixture(scope="module")
def agent():
    pytest.env = CartPole()
    pytest.agent = AgentDQNCartPole(pytest.env)
    return pytest.agent


def test_episode_with_random_step(agent):
    pytest.agent.restart_matrices()
    new_state = None
    total_reward = 0
    episode_done = False
    while not episode_done:
        new_state, reward, episode_done = pytest.agent.random_step()
        total_reward += reward
    assert new_state is not None and total_reward > 0


def test_episode_with_step(agent):
    pytest.agent.restart_matrices()
    pytest.env.restart_environment()
    new_state = None
    total_reward = 0
    episode_done = False
    while not episode_done:
        new_state, reward, episode_done = pytest.agent.step()
        total_reward += reward
    assert new_state is not None and total_reward > 0
