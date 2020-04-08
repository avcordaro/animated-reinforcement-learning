from model.environment import Environment
import random


class TaxiDriver(Environment):
    """
    TaxiDriver is a 5x5 grid world environment, featuring a taxi, a passenger and their destination.
    Both the passenger spawn state and their destination always belong to one of four locations in
    the grid. The taxi can spawn anywhere in the grid, and the agent must move the taxi towards
    the passenger, pick them up, move to their destination, and drop them off.
    """

    def __init__(self):
        self.name = "Taxi Driver"
        self.MAX_REWARD = 20
        self.MIN_REWARD = -250
        self.REWARD_THRESHOLD = 5
        self.MAX_EPISODE_STEPS = 1000
        self.GRID_ROWS = 5
        self.GRID_COLUMNS = 5
        self.GRID_MAP = [" : | : : ",
                         " : | : : ",
                         " : : : : ",
                         " | : | : ",
                         " | : | : "
                         ]
        self.action_space = ["Left", "Up", "Right", "Down", "Pickup", "Dropoff"]
        self.NUM_ACTIONS = 6
        self.illegal_actions = [(0, 1, "Right"), (0, 2, "Left"), (1, 1, "Right"),
                                (1, 2, "Left"), (3, 0, "Right"), (3, 1, "Left"),
                                (3, 2, "Right"), (3, 3, "Left"), (4, 0, "Right"),
                                (4, 1, "Left"), (4, 2, "Right"), (4, 3, "Left")
                                ]
        self.locations = [(0, 0), (0, 4), (4, 0), (4, 3)]
        self.state_space = []
        self.NUM_STATE_FEATURES = 6
        self.passenger_locations = self.locations + ["In Taxi"]
        for row in range(self.GRID_ROWS):
            for col in range(self.GRID_COLUMNS):
                for passenger_location in self.passenger_locations:
                    for destination in self.locations:
                        self.state_space.append(((row, col), passenger_location, destination))

        self.passenger_state = random.choice(self.locations)
        self.passenger_in_taxi = False
        self.destination_state = random.choice(self.locations)
        self.taxi_state = (random.randrange(0, 5), random.randrange(0, 5))
        self.start_state = (self.taxi_state, self.passenger_state, self.destination_state)
        self.current_state = self.start_state

    def execute_action(self, action):
        """
        Updates the current state based on the given action. Dropping off a passenger at the correct
        destinations gives a reward of 20. Incorrect pickup and dropoff actions give a reward of -10.
        All other steps give a reward of -1.

        @param action: the action chosen by the agent
        @return: the observation to the agent, including the new stae and reward
        """

        row, col = self.taxi_state

        if action in ["Left", "Up", "Right", "Down"]:
            if (row, col, action) not in self.illegal_actions:
                if action == "Up" and not row == 0:
                    self.taxi_state = (row - 1, col)
                elif action == "Left" and not col == 0:
                    self.taxi_state = (row, col - 1)
                elif action == "Right" and not col == 4:
                    self.taxi_state = (row, col + 1)
                elif action == "Down" and not row == 4:
                    self.taxi_state = (row + 1, col)

        reward = -1
        episode_done = False
        if action == "Pickup":
            if self.passenger_state == self.taxi_state and not self.passenger_in_taxi:
                self.passenger_in_taxi = True
                self.passenger_state = "In Taxi"
            else:
                reward = -10
        if action == "Dropoff":
            if self.passenger_in_taxi and self.taxi_state == self.destination_state:
                reward = 20
                episode_done = True
            elif self.taxi_state in self.locations and self.passenger_in_taxi:
                self.passenger_in_taxi = False
                self.passenger_state = self.taxi_state
            else:
                reward = -10

        self.current_state = (self.taxi_state, self.passenger_state, self.destination_state)
        return self.current_state, reward, episode_done

    def random_action(self):
        """
        Chooses a random action from the environment's action space

        @return: a random action
        """

        return random.choice(self.action_space)

    def restart_environment(self):
        """
        Resets the various environment state variables, by randomly generating a new spawn location
        for the passenger, destination and taxi.
        """

        self.passenger_state = random.choice(self.locations)
        self.passenger_in_taxi = False
        self.destination_state = random.choice(self.locations)
        self.taxi_state = (random.randrange(0, 5), random.randrange(0, 5))
        self.start_state = (self.taxi_state, self.passenger_state, self.destination_state)
        self.current_state = self.start_state
