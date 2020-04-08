from model.environment import Environment
import numpy as np
import random


class FrozenLake(Environment):
    """
    FrozenLake is a grid world environment, with a start and goal state for the agent.
    There are two variants of the lake map, 4x4 and 8x8, each with holes in various
    locations that the agent can fall into, terminating the episode.

    This frozen lake can also be made slippery, which changes the environment from
    deterministic to stochastic, where each movement action has an equal chance to
    move in one of three directions.
    """

    def __init__(self, is_slippery, grid_size=4):
        self.name = "Frozen Lake"
        self.is_slippery = is_slippery
        self.MAX_REWARD = 200
        self.MIN_REWARD = -150
        self.REWARD_THRESHOLD = 70 if grid_size == 8 else 140
        self.MAX_EPISODE_STEPS = 200
        self.GRID_ROWS = grid_size
        self.GRID_COLUMNS = grid_size
        map_4x4 = ["SFFF",
                   "FHFH",
                   "FFFH",
                   "HFFG"
                   ]
        map_8x8 = ["SFFFFFFF",
                   "FFFFFFFF",
                   "FFFHFFFF",
                   "FFFFFHFF",
                   "FFFHFFFF",
                   "FHHFFFHF",
                   "FHFFHFHF",
                   "FFFHFFFG"
                   ]
        self.GRID_MAP = map_4x4 if self.GRID_ROWS == 4 else map_8x8
        self.action_space = ["Left", "Up", "Right", "Down"]
        self.NUM_ACTIONS = 4
        self.state_space = []
        self.NUM_FEATURES = 2
        for row in range(self.GRID_ROWS):
            for col in range(self.GRID_COLUMNS):
                self.state_space.append((row, col))
                if self.GRID_MAP[row][col] == "S":
                    self.start_state = (row, col)
                    self.current_state = (row, col)
                if self.GRID_MAP[row][col] == "G":
                    self.goal_state = (row, col)

    def execute_action(self, action):
        """
        Updates the current state based on the given action. The reward is 0 for every step,
        except for when the agent reaches the goal state, in which case the reward is 1.

        @param action: the action chosen by the agent
        @return: the observation to the agent, including the new state and reward
        """

        row, col = self.current_state
        action = self._randomise_action(action) if self.is_slippery else action

        if action == "Up" and not row == 0:
            self.current_state = (row - 1, col)
        elif action == "Left" and not col == 0:
            self.current_state = (row, col - 1)
        elif action == "Right" and not col == (self.GRID_ROWS - 1):
            self.current_state = (row, col + 1)
        elif action == "Down" and not row == (self.GRID_ROWS - 1):
            self.current_state = (row + 1, col)

        reward = 200 if self.current_state == self.goal_state else -100 if self._hole_detected() else -1
        episode_done = True if self.current_state == self.goal_state or self._hole_detected() else False

        return self.current_state, reward, episode_done

    def random_action(self):
        """
        Chooses a random action from the environment's action space

        @return: a random action
        """

        return random.choice(self.action_space)

    def restart_environment(self):
        """
        Sets the current state back to the start state, restarting the environment
        """

        self.current_state = self.start_state

    def get_feature_vector(self, state, action):
        """
        Gets the feature vector for the given state and action.

        @param state: the given state
        @param action: the given action
        @return: the feature vector for the given state and action
        """

        row, col = state
        wall_ahead = 0
        hole_ahead = 0

        if action == "Up":
            wall_ahead = 1 if row == 0 else 0
            if not row == 0:
                hole_ahead = 1 if self.GRID_MAP[row - 1][col] == "H" else 0
        if action == "Left":
            wall_ahead = 1 if col == 0 else 0
            if not col == 0:
                hole_ahead = 1 if self.GRID_MAP[row][col - 1] == "H" else 0
        if action == "Right":
            wall_ahead = 1 if col == (self.GRID_ROWS - 1) else 0
            if not col == (self.GRID_ROWS - 1):
                hole_ahead = 1 if self.GRID_MAP[row][col + 1] == "H" else 0
        if action == "Down":
            wall_ahead = 1 if row == (self.GRID_ROWS - 1) else 0
            if not row == (self.GRID_ROWS - 1):
                hole_ahead = 1 if self.GRID_MAP[row + 1][col] == "H" else 0

        feature_vector = np.array([wall_ahead, hole_ahead])
        return feature_vector

    def _randomise_action(self, action):
        """
        Each action on a slippery lake results in moving in one of three directions

        @param: the action chosen by the agent
        @return: the actual action to be taken
        """

        if action == "Up":
            return random.choice(["Up", "Left", "Right"])
        elif action == "Left":
            return random.choice(["Left", "Up", "Down"])
        elif action == "Right":
            return random.choice(["Right", "Up", "Down"])
        elif action == "Down":
            return random.choice(["Down", "Left", "Right"])

    def _hole_detected(self):
        """
        Detects whether the agent has fallen into a hole in the lake

        @return: true or false whether there is a hole under the agent
        """

        row, col = self.current_state
        return self.GRID_MAP[row][col] == "H"