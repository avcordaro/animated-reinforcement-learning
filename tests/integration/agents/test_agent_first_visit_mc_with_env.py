import pytest, sys

sys.path.extend([".", "..", "../..", "../../.."])
from model.agent_first_visit_mc import AgentFirstVisitMC
from model.environment_frozenlake import FrozenLake


@pytest.fixture(scope="module")
def agent():
    pytest.env = FrozenLake(False)
    pytest.agent = AgentFirstVisitMC(pytest.env)
    return pytest.agent


def test_episode_with_random_policy(agent):
    new_state = None
    reward = 0
    episode_done = False
    while not episode_done:
        _, new_state, reward, episode_done = pytest.agent.step(pytest.env.random_action())
    terminal_state = True if new_state in [(1, 1), (1, 3), (2, 3), (3, 0), (3, 3)] else False
    reward_valid = True if reward in [-100, -1, 200] else False
    assert (terminal_state, reward_valid) == (True, True)
