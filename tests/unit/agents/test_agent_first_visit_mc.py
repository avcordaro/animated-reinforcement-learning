import pytest
import sys
sys.path.extend([".", "..", "../..", "../../.."])
from model.agent_first_visit_mc import AgentFirstVisitMC
from model.environment_frozenlake import FrozenLake


@pytest.fixture(scope="module")
def agent():
    env = FrozenLake(False)
    pytest.agent = AgentFirstVisitMC(env)
    return pytest.agent


def test_correct_initialisation(agent):
    Q_values_size = len(pytest.agent.Q)
    policy_size = len(pytest.agent.policy)
    returns_size = len(pytest.agent.returns)
    assert (Q_values_size, policy_size, returns_size) == (64, 16, 0)


def test_restart_matrices(agent):
    pytest.agent.restart_matrices()
    Q_values_size = len(pytest.agent.Q)
    policy_size = len(pytest.agent.policy)
    returns_size = len(pytest.agent.returns)
    assert (Q_values_size, policy_size, returns_size) == (64, 16, 0)


def test_appears_in_sequence(agent):
    state = (0, 0)
    episode_sequence = [{"state": state, "action": "Right", "reward": 1}]
    test_1 = pytest.agent._appears_in_sequence(state, "Right", episode_sequence)
    test_2 = pytest.agent._appears_in_sequence(state, "Left", episode_sequence)
    assert (test_1, test_2) == (True, False)


def test_update_policy(agent):
    state = (0, 0)
    episode_sequence = [{"state": state, "action": "Right", "reward": 1}]
    policy_before = pytest.agent.policy[state]
    pytest.agent.update_policy(episode_sequence)
    policy_after = pytest.agent.policy[state]
    assert (policy_before, policy_after) == ("Left", "Right")


def test_best_action(agent):
    state = (0, 0)
    best_action = pytest.agent._best_action(state)
    assert best_action == "Right"
