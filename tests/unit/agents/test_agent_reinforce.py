import pytest
import sys
import numpy as np
sys.path.extend([".", "..", "../..", "../../.."])
from model.agent_reinforce import AgentREINFORCE
from model.environment_frozenlake import FrozenLake


@pytest.fixture(scope="module")
def agent():
    pytest.env = FrozenLake(False)
    pytest.agent = AgentREINFORCE(pytest.env, gamma=0.9, alpha=0.1)
    return pytest.agent


def test_correct_initialisation(agent):
    num_of_weights = len(pytest.agent.theta)
    gamma = pytest.agent.GAMMA
    alpha = pytest.agent.ALPHA
    assert (num_of_weights, gamma, alpha) == (2, 0.9, 0.1)


def test_restart_matrices(agent):
    pytest.agent.restart_matrices()
    num_of_weights = len(pytest.agent.theta)
    assert num_of_weights == 2


def test_policy(agent):
    state = (0, 0)
    softmax_probabilities = pytest.agent.policy(state)
    length_of_softmax_probs = len(softmax_probabilities)
    assert length_of_softmax_probs == 4


def test_compute_gradient(agent):
    state = (0, 0)
    action = "Left"
    gradient = pytest.agent.compute_gradient(state, action)
    length_of_gradient = len(gradient)
    assert length_of_gradient == 2


def test_update_weights(agent):
    pytest.agent.theta = np.zeros(2)
    rewards = [1]
    gradients = np.array([[1, 2]])
    pytest.agent.update_weights(gradients, rewards)
    new_weights = pytest.agent.theta
    assert np.allclose(new_weights, np.array([0.1, 0.2]))


def test_get_all_softmax_probabilities(agent):
    all_softmax_probs = pytest.agent.get_all_softmax_probabilities()
    assert len(all_softmax_probs) == 16
