import pytest
import sys
sys.path.extend([".", "..", "../..", "../../.."])
from model.agent_policy_iter import AgentPolicyIter
from model.environment_frozenlake import FrozenLake


@pytest.fixture(scope="module")
def agent():
    env = FrozenLake(False)
    pytest.agent = AgentPolicyIter(env, gamma=0.9)
    return pytest.agent


def test_correct_initialisation(agent):
    vos_size = len(pytest.agent.values_of_state)
    policy_size = len(pytest.agent.policy)
    assert (vos_size, policy_size, pytest.agent.GAMMA) == (16, 16, 0.9)


def test_restart_matrices(agent):
    pytest.agent.restart_matrices()
    vos_size = len(pytest.agent.values_of_state)
    policy_size = len(pytest.agent.policy)
    transitions_size = len(pytest.agent.transitions_counter)
    rewards_size = len(pytest.agent.rewards)
    assert (vos_size, policy_size, transitions_size, rewards_size) == (16, 16, 0, 0)


def test_calculate_value_of_action(agent):
    state = (0, 0)
    action = "Right"
    new_state = (0, 1)
    pytest.agent.rewards[(state, action, new_state)] = 1
    pytest.agent.transitions_counter[(state, action)] = {}
    pytest.agent.transitions_counter[(state, action)][(new_state)] = 1
    value_of_action = pytest.agent._calculate_value_of_action(state, action)
    assert value_of_action == 1


def test_increment_transitions_counter(agent):
    pytest.agent.restart_matrices()
    state = (0, 0)
    action = "Right"
    new_state = (0, 1)
    pytest.agent._increment_transitions_counter(state, action, new_state)
    counter1 = pytest.agent.transitions_counter[(state, action)][(new_state)]
    pytest.agent._increment_transitions_counter(state, action, new_state)
    counter2 = pytest.agent.transitions_counter[(state, action)][(new_state)]
    new_state = (0, 0)
    pytest.agent._increment_transitions_counter(state, action, new_state)
    counter3 = pytest.agent.transitions_counter[(state, action)][(new_state)]
    assert (counter1, counter2, counter3) == (1, 2, 1)


def test_best_action(agent):
    state = (0, 0)
    action = "Right"
    new_state = (0, 1)
    pytest.agent.rewards[(state, action, new_state)] = 1
    best_action = pytest.agent._best_action(state)
    assert best_action == "Right"


def test_evaluate_policy(agent):
    pytest.agent.restart_matrices()
    state = (0, 0)
    action = "Right"
    new_state = (0, 1)
    pytest.agent.rewards[(state, action, new_state)] = 1
    pytest.agent._increment_transitions_counter(state, action, new_state)
    pytest.agent.policy[state] = "Right"
    value_before = pytest.agent.values_of_state[state]
    pytest.agent.evaluate_policy()
    value_after = pytest.agent.values_of_state[state]
    assert (value_before, value_after) == (0, 1)


def test_improve_policy(agent):
    state = (0, 0)
    pytest.agent.policy[state] = "Left"
    policy_before = pytest.agent.policy[state]
    pytest.agent.improve_policy()
    policy_after = pytest.agent.policy[state]
    assert (policy_before, policy_after) == ("Left", "Right")
