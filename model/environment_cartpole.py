from model.environment import Environment
import gym


class CartPole(Environment):
    """
    Wrapper class for the pre-existing implementation of the CartPole environment, from the OpenAI gym package.

    https://gym.openai.com/envs/CartPole-v0/
    """

    def __init__(self):
        self.name = "CartPole"
        self.gym_env = gym.make("CartPole-v0")
        self.current_state = self.gym_env.reset()
        self.MAX_REWARD = 200
        self.MIN_REWARD = 0

    def execute_action(self, action):
        """
        Takes the given action and passes it to the gym environment.

        @param action: the given action
        @return: the original state, reward received, new state and whether the episode has finished
        """

        origin_state = self.current_state
        new_state, reward, episode_done, _ = self.gym_env.step(action)
        self.current_state = new_state

        return origin_state, reward, new_state, episode_done

    def random_action(self):
        """
        Chooses a random action from the environment's action space.

        @return: a random action
        """

        return self.gym_env.action_space.sample()

    def restart_environment(self):
        """
        Resets the gym environment, and updates the current state attribute.
        """

        self.current_state = self.gym_env.reset()
