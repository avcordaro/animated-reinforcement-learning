from model.agent import Agent
import random


class AgentSARSA(Agent):
    """
    This Agent class implements the SARSA algorithm, to solve the problem of the given environment.
    The algorithm learns on the fly without any need for knowledge of the environment dynamics.
    SARSA is the on-policy alternative to Q-Learning, which is off-policy.
    """

    def __init__(self, env, gamma=1.0, alpha=0.1, min_init_Q=0, max_init_Q=0):
        self.GAMMA = gamma
        self.alpha = alpha
        self.MIN_INIT_Q = min_init_Q
        self.MAX_INIT_Q = max_init_Q
        self.env = env
        self.Q = {}
        self.policy = {}

    def restart_matrices(self):
        """
        Resets the various matrices used by the agent back to their default values.
        """

        self.Q = {}
        for state in self.env.state_space:
            for action in self.env.action_space:
                self.Q[(state, action)] = random.uniform(self.MIN_INIT_Q, self.MAX_INIT_Q)
                if len(state) < 3:
                    row, col = state
                    if self.env.GRID_MAP[row][col] == "G" or self.env.GRID_MAP[row][col] == "H":
                        self.Q[(state, action)] = 0

        self.policy = {state: self._best_action(state) for state in self.env.state_space}

    def step(self, action_chosen):
        """
        Performs a step of the environment, by executing the provided action and returning the
        original state, new state, reward, and whether the episode has terminated.

        @param action_chosen: the chosen action to perform
        @return: the original state, new state, reward and whether the episode has terminated
        """

        origin_state = self.env.current_state
        new_state, reward, episode_done = self.env.execute_action(action_chosen)

        return origin_state, new_state, reward, episode_done

    def update_Q(self, origin_state, action, new_state, reward, new_action):
        """
        Uses the SARSA update rule to update the Q-value for the provided state, action,
        new state and reward.

        @param origin_state: the starting state
        @param action: the action taken
        @param new_state: the resulting new state
        @param reward: the reward received
        @param new_action: the next chosen action
        """

        self.Q[(origin_state, action)] += self.alpha * (reward + self.GAMMA * self.Q[(new_state, new_action)]
                                                        - self.Q[(origin_state, action)])
        self.policy[origin_state] = self._best_action(origin_state)

    def _best_action(self, state):
        """
        Utility method which returns the action with the highest value of action for a given state.

        @param state: the given state
        @return: action with the highest value of action for the given state
        """

        best_action, best_value = None, None
        for action in self.env.action_space:
            value_of_action = self.Q[(state, action)]
            if best_value is None or best_value < value_of_action:
                best_value = value_of_action
                best_action = action
        return best_action
