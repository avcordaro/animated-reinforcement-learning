import pytest
import sys
import numpy as np
sys.path.extend([".", "..", "../..", "../../.."])
from model.agent_actor_critic import AgentActorCritic
from model.environment_frozenlake import FrozenLake


@pytest.fixture(scope="module")
def agent():
    pytest.env = FrozenLake(False)
    pytest.agent = AgentActorCritic(pytest.env, gamma=0.9, alpha=0.1, beta=0.1)
    return pytest.agent


def test_correct_initialisation(agent):
    num_of_theta_weights = len(pytest.agent.theta)
    num_of_w_weights = len(pytest.agent.w)
    gamma = pytest.agent.GAMMA
    alpha = pytest.agent.ALPHA
    beta = pytest.agent.BETA
    assert (num_of_theta_weights, num_of_w_weights, gamma, alpha, beta) == (2, 2, 0.9, 0.1, 0.1)


def test_restart_matrices(agent):
    pytest.agent.restart_matrices()
    num_of_theta_weights = len(pytest.agent.theta)
    num_of_w_weights = len(pytest.agent.w)
    assert (num_of_theta_weights, num_of_w_weights) == (2, 2)


def test_policy(agent):
    state = (0, 0)
    softmax_probabilities = pytest.agent.policy(state)
    length_of_softmax_probs = len(softmax_probabilities)
    assert length_of_softmax_probs == 4


def test_Q(agent):
    state = (0, 0)
    action = "Left"
    pytest.agent.w = np.ones(2)
    Q_value = pytest.agent.Q(state, action)
    assert Q_value == 1


def test_compute_gradient(agent):
    state = (0, 0)
    action = "Left"
    gradient = pytest.agent.compute_gradient(state, action)
    length_of_gradient = len(gradient)
    assert length_of_gradient == 2


def test_update_weights(agent):
    pytest.agent.theta = np.zeros(2)
    pytest.agent.w = np.ones(2)
    state = (0, 0)
    action = "Left"
    reward = 1
    new_state = (0, 0)
    new_action = "Left"
    gradient = np.array([1, 2])
    pytest.agent.update_weights(gradient, state, action, reward, new_state, new_action)
    new_theta_weights = pytest.agent.theta
    new_w_weights = pytest.agent.w
    assert np.allclose(new_theta_weights, np.array([0.1, 0.2])) and np.allclose(new_w_weights, np.array([1.09, 1.0]))


def test_get_all_softmax_probabilities(agent):
    all_softmax_probs = pytest.agent.get_all_softmax_probabilities()
    assert len(all_softmax_probs) == 16
