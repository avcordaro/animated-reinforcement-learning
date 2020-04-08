from model.agent import Agent
import torch
import collections
import numpy as np
import torch.nn as nn


class Actor(nn.Module):
	"""
	This class defines the actor neural network model for the DDPG algorithm.
	"""

	def __init__(self, input_shape, output_shape):
		super(Actor, self).__init__()

		self.net = nn.Sequential(
			nn.Linear(input_shape, 128),
			nn.ReLU(),
			nn.Linear(128, 64),
			nn.ReLU(),
			nn.Linear(64, output_shape),
			nn.Tanh()
		)

	def forward(self, state):
		return self.net(state)


class Critic(nn.Module):
	"""
	This class defines the actor neural network model for the DDPG algorithm.
	"""

	def __init__(self, input_shape):
		super(Critic, self).__init__()

		self.net = nn.Sequential(
			nn.Linear(input_shape, 128),
			nn.ReLU(),
			nn.Linear(128, 64),
			nn.ReLU(),
			nn.Linear(64, 1)
		)

	def forward(self, state, action):
		x = torch.cat([state, action], 1)
		return self.net(x)


class OUNoise:
	"""
	Code taken and modified from:
	https://github.com/tobiassteidle/Reinforcement-Learning/blob/master/OpenAI/LunarLander-v2/ddpg_agent.py

	This class implements the Ornstein–Uhlenbeck process, which is used for exploration in the
	DDPG algorithm.
	"""

	def __init__(self):
		self.mu = 0
		self.theta = 0.1
		self.sigma = 0.2
		self.action_dim = 2
		self.reset()

	def reset(self):
		self.state = np.ones(self.action_dim) * self.mu

	def get_noise(self):
		x = self.state
		dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
		self.state = x + dx
		return self.state


class AgentDDPGLunarLander(Agent):
	"""
	This Agent class implements the Deep Deterministic Policy Gradient algorithm, to solve the problem of
	the Lunar Lander environment. This is a deep learning algorithm which uses neural networks from the PyTorch package.
	"""

	def __init__(self, env, gamma=0.99, alpha_actor=0.0001, alpha_critic=0.001, tau=0.001, batch_size=32):
		self.GAMMA = gamma
		self.ALPHA_ACTOR = alpha_actor
		self.ALPHA_CRITIC = alpha_critic
		self.TAU = tau
		self.BATCH_SIZE = batch_size
		self.env = env
		num_obs = self.env.gym_env.observation_space.shape[0]
		num_acts = self.env.gym_env.action_space.shape[0]
		self.actor = Actor(num_obs, num_acts)
		self.actor_target = Actor(num_obs, num_acts)
		self.actor_target.load_state_dict(self.actor.state_dict())
		self.actor_optimiser = torch.optim.Adam(self.actor.parameters(), lr=self.ALPHA_ACTOR)
		self.critic = Critic(num_obs + num_acts)
		self.critic_target = Critic(num_obs + num_acts)
		self.critic_target.load_state_dict(self.critic.state_dict())
		self.critic_loss_function = nn.MSELoss()
		self.critic_optimiser = torch.optim.Adam(self.critic.parameters(), lr=self.ALPHA_CRITIC)
		self.ounoise = OUNoise()
		self.replay_buffer = collections.deque(maxlen=1000000)

	def restart_matrices(self):
		"""
		Resets the neural networks and replay buffer.
		"""
		num_obs = self.env.gym_env.observation_space.shape[0]
		num_acts = self.env.gym_env.action_space.shape[0]
		self.actor = Actor(num_obs, num_acts)
		self.actor_target = Actor(num_obs, num_acts)
		self.actor_target.load_state_dict(self.actor.state_dict())
		self.actor_optimiser = torch.optim.Adam(self.actor.parameters(), lr=self.ALPHA_ACTOR)
		self.critic = Critic(num_obs + num_acts)
		self.critic_target = Critic(num_obs + num_acts)
		self.critic_target.load_state_dict(self.critic.state_dict())
		self.critic_optimiser = torch.optim.Adam(self.critic.parameters(), lr=self.ALPHA_CRITIC)
		self.replay_buffer = collections.deque(maxlen=1000000)

	def step(self, epsilon):
		"""
		Uses the agent's actor neural network to choose the best action, then executes it in the environment,
		appending the transition information to the replay buffer.

		@return: the resulting new state, reward received, and whether the episode has finished.
		"""

		state_t = torch.FloatTensor([self.env.current_state])
		action = self.actor(state_t).cpu().detach().numpy()[0]
		noise = epsilon * self.ounoise.get_noise()
		action = np.clip(action + noise, -1.0, 1.0)

		origin_state, reward, new_state, episode_done = self.env.execute_action(action)

		self.replay_buffer.append((origin_state, action, reward, new_state, episode_done))

		if len(self.replay_buffer) > self.BATCH_SIZE:
			self._backwards_pass()

		return new_state, reward, episode_done

	def test_step(self):
		"""
		Uses the agent's actor neural network to choose the best action, then executes it in the environment.
		Does not perform backwards pass to update network parameters.

		@return: the resulting new state, reward received, and whether the episode has finished.
		"""

		state_t = torch.FloatTensor([self.env.current_state])
		action = self.actor(state_t).cpu().detach().numpy()[0]

		origin_state, reward, new_state, episode_done = self.env.execute_action(action)

		return new_state, reward, episode_done

	def _backwards_pass(self):
		"""
		Code taken and modified from:
		https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On/blob/master/Chapter06/02_dqn_pong.py

		Code for soft-update of target networks modified from:
		https://github.com/tobiassteidle/Reinforcement-Learning/blob/master/OpenAI/LunarLander-v2/ddpg_agent.py

		Samples a batch of transitions from the replay buffer, and performs stochastic gradient descent
		to update the weights of the agent's neural network.
		"""

		samples = np.random.choice(len(self.replay_buffer), self.BATCH_SIZE, replace=False)
		states, actions, rewards, new_states, dones = zip(*[self.replay_buffer[sample] for sample in samples])
		states_t = torch.FloatTensor(np.array(states))
		actions_t = torch.FloatTensor(np.array(actions))
		rewards_t = torch.FloatTensor(np.array(rewards).reshape(self.BATCH_SIZE, 1))
		new_states_t = torch.FloatTensor(np.array(new_states))
		dones_t = 1 - torch.ByteTensor(np.array(dones, dtype="uint8").reshape(self.BATCH_SIZE, 1))
	
		q_vals = self.critic(states_t, actions_t)
		next_actions = self.actor_target(new_states_t).detach()
		next_q_vals = self.critic_target(new_states_t, next_actions)
		next_q_vals = dones_t.float() * next_q_vals
		target_q_vals = rewards_t + self.GAMMA * next_q_vals
		critic_loss = self.critic_loss_function(q_vals, target_q_vals)

		actor_loss = -self.critic(states_t, self.actor(states_t)).mean()

		self.actor_optimiser.zero_grad()
		actor_loss.backward()
		self.actor_optimiser.step()

		self.critic_optimiser.zero_grad()
		critic_loss.backward()
		self.critic_optimiser.step()

		for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
			target_param.data.copy_(param.data * self.TAU + target_param.data * (1.0 - self.TAU))

		for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
			target_param.data.copy_(param.data * self.TAU + target_param.data * (1.0 - self.TAU))
