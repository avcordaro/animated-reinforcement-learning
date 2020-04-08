from model.agent import Agent
import collections


class AgentValueIter(Agent):
    """
    This Agent class implements the Value Iteration algorithm, to solve the problem of the given environment.
    Following from Maxim Lapan's book, we assume the agent is not given the environment dynamics like rewards
    and transition probabilities, and instead keeps its own history of transitions and rewards received through
    exploration, in order to estimate the actual dynamics.
    """

    def __init__(self, env, gamma=1.0):
        self.env = env
        self.rewards = collections.defaultdict(float)
        self.values_of_state = {state: 0.0 for state in self.env.state_space}
        self.policy = {state: self.env.random_action() for state in self.env.state_space}
        self.transitions_counter = {}
        self.GAMMA = gamma

    def update_values(self):
        """
        Performs the value update stage of Value Iteration. Automatically calls the update_policy() method once
        the value of states have changed by a very small amount during update iterations.
        """

        epsilon = 0.001
        delta = 0
        for state in self.env.state_space:
            old_val = self.values_of_state[state]
            calculated_values = []
            for action in self.env.action_space:
                if (state, action) in self.transitions_counter:
                    calculated_values.append(self._calculate_value_of_action(state, action))
            new_val = self.values_of_state[state] = max(calculated_values) if len(calculated_values) > 0 else 0.0
            delta = max(delta, abs(old_val - new_val))
        if delta > epsilon:
            self.update_values()
        else:
            self.update_policy()

    def update_policy(self):
        """
        A greedy policy method, that uses the values of states to calculate the value of each action at each state,
        to output the best action as the policy for each state.
        """

        for state in self.env.state_space:
            best_action, best_value = None, None
            for action in self.env.action_space:
                if (state, action) in self.transitions_counter:
                    value_of_action = self._calculate_value_of_action(state, action)
                    if best_value is None or best_value < value_of_action:
                        best_value = value_of_action
                        best_action = action
            if best_action is not None:
                self.policy[state] = best_action

    def restart_matrices(self):
        """
        Resets the various matrices used by the agent back to their default values.
        """

        self.rewards = collections.defaultdict(float)
        self.values_of_state = {state: 0.0 for state in self.env.state_space}
        self.policy = {state: self.env.random_action() for state in self.env.state_space}
        self.transitions_counter = {}

    def random_step(self):
        """
        Performs an exploration step of the environment, by selecting a random action to perform.

        @return: the new state and reward after performing the selected action, as well as if the episode has terminated
        """

        origin_state = self.env.current_state
        action_chosen = self.env.random_action()
        new_state, reward, episode_done = self.env.execute_action(action_chosen)

        self._increment_transitions_counter(origin_state, action_chosen, new_state)
        self.rewards[(origin_state, action_chosen, new_state)] = reward

        return new_state, reward, episode_done

    def step(self):
        """
        Performs an exploitation step of the environment, using the current policy to select the best action for
        the current state.

        @return: the new state and reward after performing the selected action, as well as if the episode has terminated
        """

        origin_state = self.env.current_state
        action_chosen = self.policy[origin_state]
        new_state, reward, episode_done = self.env.execute_action(action_chosen)

        self._increment_transitions_counter(origin_state, action_chosen, new_state)
        self.rewards[(origin_state, action_chosen, new_state)] = reward

        return new_state, reward, episode_done

    def _calculate_value_of_action(self, state, action):
        """
        This method uses the Bellman equation to calculate the value of action for the given state and action.

        @param state: the given origin state
        @param action: the action selected
        @return: the calculated value of action
        """

        transition_counts = self.transitions_counter[(state, action)]
        total_counts = sum(transition_counts.values())
        value_of_action = 0.0
        for new_state, counter in transition_counts.items():
            reward = self.rewards[(state, action, new_state)]
            value_of_action += (counter / total_counts) * (reward + self.GAMMA * self.values_of_state[new_state])
        return value_of_action

    def _increment_transitions_counter(self, state, action, new_state):
        """
        Updates the transition counter with the information passed to the method. Initialises a new counter if the
        state/action, or new state has never been seen before.

        @param state: the given origin state
        @param action: the action selected
        @param new_state: the new state after performing the action
        """

        if (state, action) in self.transitions_counter:
            if new_state in self.transitions_counter[(state, action)]:
                self.transitions_counter[(state, action)][new_state] += 1
            else:
                self.transitions_counter[(state, action)][new_state] = 1
        else:
            self.transitions_counter[(state, action)] = {new_state: 1}
