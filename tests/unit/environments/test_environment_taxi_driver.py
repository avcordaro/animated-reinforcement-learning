import pytest
import sys
import numpy as np
sys.path.extend([".", "..", "../..", "../../.."])
from model.environment_taxi_driver import TaxiDriver


@pytest.fixture(scope="module")
def env():
    pytest.env = TaxiDriver()
    return pytest.env


def test_state_space_size(env):
    assert len(pytest.env.state_space) == 500


def test_taxi_state(env):
    taxi = pytest.env.taxi_state
    location_in_range = True if 0 <= taxi[0] <= 4 else False
    location_in_range = True if 0 <= taxi[1] <= 4 else False
    pytest.env.taxi_state = (0, 0)  # setup for testing movement actions
    assert location_in_range


def test_destination_state(env):
    destination = pytest.env.destination_state
    assert destination in [(0, 0), (0, 4), (4, 0), (4, 3)]


def test_passenger_state(env):
    passenger = pytest.env.passenger_state
    valid_location = True if passenger in [(0, 0), (0, 4), (4, 0), (4, 3)] else False
    in_taxi = pytest.env.passenger_in_taxi
    assert (valid_location, in_taxi) == (True, False)


def test_random_action(env):
    action = pytest.env.random_action()
    assert action in ["Left", "Up", "Right", "Down", "Pickup", "Dropoff"]


@pytest.mark.parametrize("action, expected_state",
                         [("Right", (0, 1)), ("Down", (1, 1)), ("Left", (1, 0)), ("Up", (0, 0))])
def test_movement_actions(env, action, expected_state):
    new_state, reward, _ = pytest.env.execute_action(action)
    assert (new_state[0], reward) == (expected_state, -1)


@pytest.mark.parametrize("start_state, action",
                         [((0, 0), "Up",), ((4, 0), "Down"), ((0, 0), "Left"), ((4, 0), "Right")])
def test_movement_actions_at_border(env, start_state, action):
    pytest.env.taxi_state = start_state
    new_state, reward, _ = pytest.env.execute_action(action)
    assert (new_state[0], reward) == (start_state, -1)


@pytest.mark.parametrize("start_state, action", [((0, 1), "Right",), ((1, 2), "Left"), ((3, 2), "Right")])
def test_illegal_movement_actions(env, start_state, action):
    pytest.env.taxi_state = start_state
    new_state, reward, _ = pytest.env.execute_action(action)
    assert (new_state[0], reward) == (start_state, -1)


def test_dropoff_with_passenger_at_destination(env):
    pytest.env.taxi_state = pytest.env.destination_state = (0, 0)
    pytest.env.passenger_in_taxi = True
    _, reward, episode_done = pytest.env.execute_action("Dropoff")
    assert (reward, episode_done) == (20, True)


def test_dropoff_with_no_passenger(env):
    pytest.env.passenger_in_taxi = False
    _, reward, episode_done = pytest.env.execute_action("Dropoff")
    assert (reward, episode_done) == (-10, False)


def test_dropoff_with_passenger_but_wrong_destination(env):
    pytest.env.passenger_in_taxi = True
    pytest.env.taxi_state = (3, 0)
    pytest.env.destination_state = (4, 0)
    _, reward, episode_done = pytest.env.execute_action("Dropoff")
    assert (reward, episode_done) == (-10, False)


def test_pickup_with_no_passenger_at_taxi_location(env):
    pytest.env.passenger_in_taxi = False
    pytest.env.taxi_state = (0, 0)
    pytest.env.passenger_state = (4, 0)
    _, reward, episode_done = pytest.env.execute_action("Pickup")
    assert (reward, episode_done) == (-10, False)


def test_pickup_with_passenger_at_taxi_location(env):
    pytest.env.passenger_in_taxi = False
    pytest.env.taxi_state = (0, 0)
    pytest.env.passenger_state = (0, 0)
    _, reward, episode_done = pytest.env.execute_action("Pickup")
    assert (reward, episode_done, pytest.env.passenger_in_taxi) == (-1, False, True)


def test_restart_environment(env):
    pytest.env.restart_environment()
    taxi = pytest.env.taxi_state
    taxi_valid = True if 0 <= taxi[0] <= 4 else False
    taxi_valid = True if 0 <= taxi[1] <= 4 else False
    passenger_valid = True if pytest.env.passenger_state in [(0, 0), (0, 4), (4, 0), (4, 3)] else False
    passenger_in_taxi = pytest.env.passenger_in_taxi
    destination_valid = True if pytest.env.destination_state in [(0, 0), (0, 4), (4, 0), (4, 3)] else False
    assert (taxi_valid, passenger_valid, passenger_in_taxi, destination_valid) == (True, True, False, True)
