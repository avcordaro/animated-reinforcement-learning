import pytest
import sys
sys.path.extend([".", "..", "../..", "../../.."])
from model.agent_ddpg_bipedal_walker import AgentDDPGBipedalWalker
from model.environment_bipedal_walker import BipedalWalker


@pytest.fixture(scope="module")
def agent():
    env = BipedalWalker()
    pytest.agent = AgentDDPGBipedalWalker(env, gamma=0.99, alpha_actor=0.001, alpha_critic=0.001)
    return pytest.agent


def test_correct_initialisation(agent):
    pytest.agent.restart_matrices()
    gamma = pytest.agent.GAMMA
    alpha_actor = pytest.agent.ALPHA_ACTOR
    alpha_critic = pytest.agent.ALPHA_CRITIC
    buffer_size = len(pytest.agent.replay_buffer)
    assert (gamma, alpha_actor, alpha_critic, buffer_size) == (0.99, 0.001, 0.001, 0)


def test_restart_matrices(agent):
    pytest.agent.restart_matrices()
    buffer_size = len(pytest.agent.replay_buffer)
    assert buffer_size == 0
