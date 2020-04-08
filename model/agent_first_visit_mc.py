from model.agent import Agent
import collections


class AgentFirstVisitMC(Agent):
    """
    This Agent class implements the First-visit Monte Carlo Control algorithm, to solve the problem
    of the given environment. This algorithm does not use exploring starts as we are assuming the
    controller for this agent will use an epsilon-greedy policy, thus guranteeing exploration.
    """

    def __init__(self, env):
        self.env = env
        self.Q = {}
        for state in self.env.state_space:
            for action in self.env.action_space:
                self.Q[(state, action)] = 0
        self.policy = {state: self._best_action(state) for state in self.env.state_space}
        self.returns = collections.defaultdict(lambda: [])

    def restart_matrices(self):
        """
        Resets the various matrices used by the agent back to their default values.
        """

        self.Q = {}
        for state in self.env.state_space:
            for action in self.env.action_space:
                self.Q[(state, action)] = 0
        self.policy = {state: self._best_action(state) for state in self.env.state_space}
        self.returns = collections.defaultdict(lambda: [])

    def step(self, action):
        """
        Performs a step of the environment, by executing the provided action and returning the
        original state, new state, reward, and whether the episode has terminated. The original
        state is included so that it can be used by the controller for building the episode sequence.

        @param action: the chosen action to perform
        @return: the original state, new state, reward and whether the episode has terminated
        """

        origin_state = self.env.current_state
        new_state, reward, episode_done = self.env.execute_action(action)

        return origin_state, new_state, reward, episode_done

    def update_policy(self, episode_sequence):
        """
        Updates the policy for the agent using the First-visit Monte Carlo Control algorithm, with
        the provided episode sequence. The policy is updated on a episode-by-episode basis.

        @param episode_sequence: the sequence of state-action-rewards for the episode
        """

        G = 0
        for t in range(len(episode_sequence) - 1, -1, -1):
            G += episode_sequence[t]["reward"]
            state_t = episode_sequence[t]["state"]
            action_t = episode_sequence[t]["action"]
            if not self._appears_in_sequence(state_t, action_t, episode_sequence[0:t]):
                self.returns[(state_t, action_t)].append(G)
                self.Q[(state_t, action_t)] = sum(self.returns[(state_t, action_t)]) / len(
                    self.returns[(state_t, action_t)])
                self.policy[state_t] = self._best_action(state_t)

    def _appears_in_sequence(self, state, action, episode_sequence):
        """
        Utility method used by update_policy() to determine whether a state-action pair have
        already appeared in the episode sequence seen so far.

        @param state: the state of the state-action pair
        @param action: the action of the state-action pair
        @param episode_sequence: a sequence of state-action-rewards
        @return: true or false whether the state-action pair already exists in the given sequence
        """

        pair_appears_in_sequence = False
        for obs in episode_sequence:
            if obs["state"] == state and obs["action"] == action:
                pair_appears_in_sequence = True
        return pair_appears_in_sequence

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
