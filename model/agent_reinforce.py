from model.agent import Agent
import numpy as np


class AgentREINFORCE(Agent):
    """
    This Agent class implements the REINFORCE algorithm, to solve the problem of the given environment. 
    The algorithm learns on the fly without any need for knowledge of the environment dynamics.
    """

    def __init__(self, env, gamma=1.0, alpha=0.1):
        self.env = env
        self.GAMMA = gamma
        self.ALPHA = alpha
        self.theta = np.random.rand(self.env.NUM_FEATURES)

    def restart_matrices(self):
        """
        Resets the theta weights.
        """

        self.theta = np.random.rand(self.env.NUM_FEATURES)

    def step(self, action):
        """
        Performs a step of the environment, by executing the provided action and returning the
        new state, reward, and whether the episode has terminated.

        @param action: the chosen action to perform
        @return: new state, reward and whether the episode has terminated
        """

        new_state, reward, episode_done = self.env.execute_action(action)
        return new_state, reward, episode_done

    def policy(self, state):
        """
        Returns the softmax probability of taking each action for the given state.

        @param state: the given state
        @return: the softmax probabilities
        """

        probs = np.array([])
        for action in self.env.action_space:
            feature_vector = self.env.get_feature_vector(state, action)
            p = feature_vector.dot(self.theta)
            probs = np.append(probs, p)
        e = np.exp(probs)
        softmax_probs = e / np.sum(e)
        return softmax_probs

    def compute_gradient(self, state, action):
        """
        Computes the gradient for the log of the policy of the given state and action.

        @param state: the given state
        @param action: the given action
        @return: the computed gradient
        """

        feature_vector = self.env.get_feature_vector(state, action)
        probs = self.policy(state)
        sum = np.zeros(len(feature_vector))
        for i in range(len(probs)):
            sum += probs[i] * self.env.get_feature_vector(state, self.env.action_space[i])
        gradient = feature_vector - sum
        return gradient

    def update_weights(self, gradients, rewards):
        """
        Updates the theta weights using the provided list of gradients and rewards received
        during the episode.

        @param gradients: list of calculated gradients
        @param rewards list of rewards received
        """

        for i in range(len(gradients)):
            self.theta += self.ALPHA * gradients[i] * sum([r * (self.GAMMA ** t) for t, r in enumerate(rewards[i:])])

    def get_all_softmax_probabilities(self):
        """
        Calculates the softmax probabilities of every action in every state.

        @return: mapping of state to action softmax probabilities
        """

        all_probs = {}
        for state in self.env.state_space:
            probs = self.policy(state)
            all_probs[state] = probs
        return all_probs
