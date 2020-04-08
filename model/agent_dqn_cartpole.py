from model.agent import Agent
import torch
import collections
import numpy as np
import torch.nn as nn


class AgentDQNCartPole(Agent):
    """
    This Agent class implements the Deep Q-Networks algorithm, to solve the problem of the given environment.
    This is a deep learning algorithm which uses neural networks from the PyTorch package.
    """

    def __init__(self, env, gamma=1.0, alpha=0.1, batch_size=8):
        self.GAMMA = gamma
        self.ALPHA = alpha
        self.BATCH_SIZE = batch_size
        self.HIDDEN_NEURONS = 128
        self.env = env
        self.net = nn.Sequential(
            nn.Linear(self.env.gym_env.observation_space.shape[0], self.HIDDEN_NEURONS),
            nn.ReLU(),
            nn.Linear(self.HIDDEN_NEURONS, self.env.gym_env.action_space.n)
        )

        self.target_net = nn.Sequential(
            nn.Linear(self.env.gym_env.observation_space.shape[0], self.HIDDEN_NEURONS),
            nn.ReLU(),
            nn.Linear(self.HIDDEN_NEURONS, self.env.gym_env.action_space.n)
        )
        self.loss_function = nn.MSELoss()
        self.optimiser = torch.optim.Adam(self.net.parameters(), lr=self.ALPHA)
        self.replay_buffer = collections.deque(maxlen=1000)

    def restart_matrices(self):
        """
        Resets the neural networks and replay buffer.
        """

        self.net = nn.Sequential(
            nn.Linear(self.env.gym_env.observation_space.shape[0], self.HIDDEN_NEURONS),
            nn.ReLU(),
            nn.Linear(self.HIDDEN_NEURONS, self.env.gym_env.action_space.n)
        )

        self.target_net = nn.Sequential(
            nn.Linear(self.env.gym_env.observation_space.shape[0], self.HIDDEN_NEURONS),
            nn.ReLU(),
            nn.Linear(self.HIDDEN_NEURONS, self.env.gym_env.action_space.n)
        )
        self.optimiser = torch.optim.Adam(self.net.parameters(), lr=self.ALPHA)
        self.replay_buffer = collections.deque(maxlen=1000)

    def step(self):
        """
        Uses the agent's neural network to choose the best action, then executes it in the environment,
        appending the transition information to the replay buffer.

        @return: the resulting new state, reward received, and whether the episode has finished.
        """

        state_t = torch.FloatTensor([self.env.current_state])
        q_vals_t = self.net(state_t)
        _, action_t = torch.max(q_vals_t, dim=1)
        action = action_t.item()

        origin_state, reward, new_state, episode_done = self.env.execute_action(action)

        self.replay_buffer.append((origin_state, action, reward, new_state, episode_done))

        if len(self.replay_buffer) > self.BATCH_SIZE:
            self._backwards_pass()

        return new_state, reward, episode_done

    def random_step(self):
        """
        Chooses a random action from the environment's action space, then executes it in the environment,
        appending the transition information to the replay buffer.

        @return: the resulting new state, reward received, and whether the episode has finished.
        """

        action = self.env.random_action()

        origin_state, reward, new_state, episode_done = self.env.execute_action(action)

        self.replay_buffer.append((origin_state, action, reward, new_state, episode_done))

        if len(self.replay_buffer) > self.BATCH_SIZE:
            self._backwards_pass()

        return new_state, reward, episode_done

    def test_step(self):
        """
        Uses the agent's neural network to choose the best action, then executes it in the environment.
        Does not perform backwards pass to update network parameters.

        @return: the resulting new state, reward received, and whether the episode has finished.
        """

        state_t = torch.FloatTensor([self.env.current_state])
        q_vals_t = self.net(state_t)
        _, action_t = torch.max(q_vals_t, dim=1)
        action = action_t.item()

        _, reward, new_state, episode_done = self.env.execute_action(action)

        return new_state, reward, episode_done

    def _backwards_pass(self):
        """
        Code taken and modified from:
        https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On/blob/master/Chapter06/02_dqn_pong.py

        Samples a batch of transitions from the replay buffer, and performs stochastic gradient descent
        to update the weights of the agent's neural network.
        """

        samples = np.random.choice(len(self.replay_buffer), self.BATCH_SIZE, replace=False)
        states, actions, rewards, new_states, dones = zip(*[self.replay_buffer[sample] for sample in samples])
        states_t = torch.FloatTensor(np.array(states))
        actions_t = torch.tensor(np.array(actions))
        rewards_t = torch.FloatTensor(np.array(rewards))
        new_states_t = torch.FloatTensor(np.array(new_states))
        dones = torch.ByteTensor(np.array(dones, dtype=np.uint8))

        self.optimiser.zero_grad()

        q_vals = self.net(states_t).gather(1, actions_t.unsqueeze(-1)).squeeze(-1)
        new_state_vals = self.target_net(new_states_t).max(1)[0].detach()
        new_state_vals[dones] = 0
        expected_q_vals = rewards_t + self.GAMMA * new_state_vals

        loss = self.loss_function(q_vals, expected_q_vals)
        loss.backward()
        self.optimiser.step()
