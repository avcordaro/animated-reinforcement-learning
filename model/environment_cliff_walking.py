from model.environment import Environment
import random


class CliffWalking(Environment):
    """
    CliffWalking is a 3x5 grid world environment, with a start and goal state for the agent.
    The top row of the environment includes a cliff edge that the agent can fall off,
    terminating the episode.
    """

    def __init__(self):
        self.name = "Cliff Walking"
        self.MAX_REWARD = 0
        self.MIN_REWARD = -100
        self.REWARD_THRESHOLD = -20
        self.MAX_EPISODE_STEPS = 200
        self.GRID_ROWS = 3
        self.GRID_COLUMNS = 5
        self.GRID_MAP = ["SHHHG",
                         "FFFFF",
                         "FFFFF"
                         ]
        self.action_space = ["Left", "Up", "Right", "Down"]
        self.NUM_ACTIONS = 4
        self.state_space = []
        self.NUM_STATE_FEATURES = 2
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
        Updates the current state based on the given action. Each step results in a reward of -1,
        while falling off the cliff edge results in a reward of -100.

        @param action: the action chosen by the agent
        @return: the observation to the agent, including the new state and reward
        """

        row, col = self.current_state

        if action == "Up" and not row == 0:
            self.current_state = (row - 1, col)
        elif action == "Left" and not col == 0:
            self.current_state = (row, col - 1)
        elif action == "Right" and not col == 4:
            self.current_state = (row, col + 1)
        elif action == "Down" and not row == 2:
            self.current_state = (row + 1, col)

        reward = -1
        if self._hole_detected():
            reward = -100
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

    def _hole_detected(self):
        """
        Detects whether the agent has fallen off the cliff edge

        @return: true or false whether the agent is over the cliff edge
        """

        row, col = self.current_state
        return self.GRID_MAP[row][col] == "H"
