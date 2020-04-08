from model.environment import Environment
import gym


class BipedalWalker(Environment):
    """
    Wrapper class for the pre-existing implementation of the Lunar Lander environment, from the OpenAI gym package.

    https://gym.openai.com/envs/BipedalWalker-v2/
    """

    def __init__(self):
        self.name = "Bipedal Walker"
        self.gym_env = gym.make("BipedalWalker-v3")
        self.current_state = self.gym_env.reset()
        self.MAX_REWARD = 300
        self.MIN_REWARD = -150

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
        pass

    def restart_environment(self):
        """
        Resets the gym environment, and updates the current state attribute.
        """

        self.current_state = self.gym_env.reset()
