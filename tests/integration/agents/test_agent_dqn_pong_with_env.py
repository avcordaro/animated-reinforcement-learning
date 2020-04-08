import pytest
import sys
sys.path.extend([".", "..", "../..", "../../.."])
from model.agent_dqn_pong import AgentDQNPong
from model.environment_pong import Pong


@pytest.fixture(scope="module")
def agent():
    pytest.env = Pong()
    pytest.agent = AgentDQNPong(pytest.env)
    return pytest.agent


def test_episode_with_random_step(agent):
    pytest.agent.restart_matrices()
    new_state = None
    total_reward = 0
    for _ in range(10):
        new_state, reward, _ = pytest.agent.random_step()
        total_reward += reward
    assert new_state is not None and total_reward in [0, -1]


def test_episode_with_step(agent):
    pytest.agent.restart_matrices()
    pytest.env.restart_environment()
    new_state = None
    total_reward = 0
    for _ in range(10):
        new_state, reward, _ = pytest.agent.step()
        total_reward += reward
    assert new_state is not None and total_reward in [0, -1]
