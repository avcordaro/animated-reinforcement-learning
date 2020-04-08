from model.environment import Environment
import model.environment_pong_wrappers as wrappers


class Pong(Environment):
    """
    Wrapper class for the pre-existing implementation of the Atari 2000 Pong environment, from the OpenAI gym package.
    The environment's state representation has been modified using custom environment wrappers.

    https://gym.openai.com/envs/Pong-v0/
    """

    def __init__(self):
        self.name = "Pong"
        self.gym_env = wrappers.make_env("PongNoFrameskip-v4")
        self.current_state = self.gym_env.reset()
        self.MAX_REWARD = 21
        self.MIN_REWARD = -21

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
