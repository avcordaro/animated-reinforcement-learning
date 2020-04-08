import pytest
import sys
sys.path.extend([".", "..", "../..", "../../.."])
from model.agent_ddpg_bipedal_walker import AgentDDPGBipedalWalker
from model.environment_bipedal_walker import BipedalWalker


@pytest.fixture(scope="module")
def agent():
    pytest.env = BipedalWalker()
    pytest.agent = AgentDDPGBipedalWalker(pytest.env)
    return pytest.agent


def test_episode_with_step(agent):
    pytest.agent.restart_matrices()
    pytest.env.restart_environment()
    new_state = None
    total_reward = 0
    for _ in range(10):
        new_state, reward, _ = pytest.agent.step(epsilon=0.1)
        total_reward += reward
    assert new_state is not None and total_reward < 0
