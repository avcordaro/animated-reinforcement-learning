import pytest
import sys
sys.path.extend([".", "..", "../..", "../../.."])
from model.agent_dqn_cartpole import AgentDQNCartPole
from model.environment_cartpole import CartPole


@pytest.fixture(scope="module")
def agent():
    env = CartPole()
    pytest.agent = AgentDQNCartPole(env, gamma=0.99, alpha=0.001, batch_size=8)
    return pytest.agent


def test_correct_initialisation(agent):
    pytest.agent.restart_matrices()
    gamma = pytest.agent.GAMMA
    alpha = pytest.agent.ALPHA
    batch_size = pytest.agent.BATCH_SIZE
    buffer_size = len(pytest.agent.replay_buffer)
    assert (gamma, alpha, batch_size, buffer_size) == (0.99, 0.001, 8, 0)


def test_restart_matrices(agent):
    pytest.agent.restart_matrices()
    buffer_size = len(pytest.agent.replay_buffer)
    assert buffer_size == 0
