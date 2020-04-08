import pytest
import sys
sys.path.extend([".", "..", "../..", "../../.."])
from model.agent_q_learning import AgentQLearning
from model.environment_frozenlake import FrozenLake


@pytest.fixture(scope="module")
def agent():
    env = FrozenLake(False)
    pytest.agent = AgentQLearning(env, gamma=0.9, alpha=0.2)
    return pytest.agent


def test_correct_initialisation(agent):
    pytest.agent.restart_matrices()
    Q_values_size = len(pytest.agent.Q)
    policy_size = len(pytest.agent.policy)
    gamma = pytest.agent.GAMMA
    alpha = pytest.agent.alpha
    assert (Q_values_size, policy_size, gamma, alpha) == (64, 16, 0.9, 0.2)


def test_restart_matrices(agent):
    pytest.agent.restart_matrices()
    Q_values_size = len(pytest.agent.Q)
    policy_size = len(pytest.agent.policy)
    assert (Q_values_size, policy_size) == (64, 16)


def test_update_Q(agent):
    state = (0, 0)
    action = "Right"
    new_state = (0, 1)
    reward = 1
    pytest.agent._update_Q(state, action, new_state, reward)
    q_value = pytest.agent.Q[(state, action)]
    policy = pytest.agent.policy[state]
    assert (q_value, policy) == (0.2, "Right")


def test_best_action(agent):
    state = (0, 0)
    best_action = pytest.agent._best_action(state)
    assert best_action == "Right"
