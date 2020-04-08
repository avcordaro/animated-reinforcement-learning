from controller.controller import Controller
import matplotlib.pyplot as plt
import random
import numpy
import time
import configparser
import os.path


class ControllerSARSA(Controller):
    """
    The Controller class for running a SARSA agent in a chosen environment. The Controller
    observes user input from the GUI and calls the appropriate methods from the Model (Agent
    and Environment objects) before updating the GUI with changes returned from the Model.

    In the case of SARSA, the Controller will run a given number of episodes using the
    Agent methods, and display statistics of the episodes on the GUI.
    """

    def __init__(self, env, agent, gui, epsilon=1.0, episodes=100):
        self.env = env
        self.agent = agent
        self.gui = gui
        self.gui.scale.configure(command=self.update_timescale)
        self.gui.start_button.configure(command=self.start)
        self.gui.animation_toggle.trace("w", self.toggle_animation)
        self.gui.update_episode_labels(0, 0, 0)

        self.canvas_after_variable = None
        self.timescale = self.gui.scale.get()
        self.toggle_animation()
        self.next_action = None

        self.EPSILON = epsilon
        self.epsilon = epsilon
        self.epsilon_decay = False
        self.no_of_episodes = episodes

        self.episode_counter = 0
        self.total_reward = 0
        self.total_reward_history = []

    def read_config_file(self):
        """
        Uses Python's in-built configparser module to read the program's configuration file, which is
        in INI format. Sets the relevant parameters to the config file values.
        """

        config = configparser.ConfigParser()
        if os.path.exists("config.txt"):
            config.read("config.txt")
            self.agent.GAMMA = config["temporal-difference"].getfloat("gamma")
            self.agent.alpha = config["temporal-difference"].getfloat("alpha")
            self.alpha_decay_amount = config["temporal-difference"].getfloat("alpha_decay_amount")
            self.EPSILON = config["temporal-difference"].getfloat("epsilon")
            self.epsilon = self.EPSILON
            self.epsilon_decay = config["temporal-difference"].getboolean("epsilon_decay")
            self.no_of_episodes = config["temporal-difference"].getint("episodes")
            self.agent.MIN_INIT_Q = config["temporal-difference"].getfloat("min_initial_Q")
            self.agent.MAX_INIT_Q = config["temporal-difference"].getfloat("max_initial_Q")

    def start(self):
        """
        The method called when the "Run" button is pressed on the GUI. It resets many of the
        various class attributes, and then initiates running the episodes.
        """

        self.read_config_file()
        if self.canvas_after_variable is not None:
            self.gui.animation.canvas.after_cancel(self.canvas_after_variable)
        self.episode_counter = 0
        self.total_reward = 0
        self.total_reward_history = []
        self.stop_and_reset()
        self.next_action = self.agent.policy[
            self.env.current_state] if random.random() > self.epsilon else self.env.random_action()
        self.gui.update_episode_labels(self.episode_counter, self.epsilon, self.agent.alpha)
        self.gui.draw_values_of_action(self.agent.Q)
        self.gui.draw_policy(self.agent.policy)
        self.gui.add_to_listbox("=============================================")
        self.gui.add_to_listbox("SARSA")
        self.gui.add_to_listbox("Running epsilon-greedy episodes.")
        self.start_time = time.time()
        self.run_episode()

    def run_episode(self):
        """
        Runs a single episode of the environment, learning via SARSA on the fly. This method will
        continually call itself until the desired number of episodes have been completed. Then
        the reward statistics from the episodes are displayed on the GUI.
        """

        action = self.next_action
        origin_state, new_state, reward, episode_done = self.agent.step(action)
        self.total_reward += reward
        self.next_action = self.agent.policy[new_state] if random.random() > self.epsilon else self.env.random_action()
        self.agent.update_Q(origin_state, action, new_state, reward, self.next_action)
        self.gui.update(new_state)

        if episode_done:
            self.episode_counter += 1
            self.env.restart_environment()
            self.gui.restart_environment(self.env.start_state)
            self.next_action = self.agent.policy[
                self.env.current_state] if random.random() > self.epsilon else self.env.random_action()
            self.gui.update_episode_labels(self.episode_counter, self.epsilon, self.agent.alpha)
            self.epsilon = self.EPSILON ** self.episode_counter if self.epsilon_decay else self.EPSILON
            self.agent.alpha = self.agent.alpha - self.alpha_decay_amount
            if self.episode_counter % 100 == 0:
                self.gui.add_to_listbox("(" + str(self.episode_counter - 99) + "-" + str(
                    self.episode_counter) + ") Average reward: {0:.2f}".format(self.total_reward / 100))
                self.gui.draw_values_of_action(self.agent.Q)
                self.gui.draw_policy(self.agent.policy)
                self.total_reward_history.append(self.total_reward)
                self.total_reward = 0

        if self.episode_counter < self.no_of_episodes:
            if self.timescale == 0:
                self.gui.root.update_idletasks()
            self.canvas_after_variable = self.gui.animation.canvas.after(self.timescale, self.run_episode)
        else:
            self.end_time = time.time()
            self.gui.animation.canvas.tag_raise(self.gui.animation.agent)
            self.gui.add_to_listbox("Finished {0} episodes. ({1:.2f} secs)".format(self.no_of_episodes, float(
                self.end_time - self.start_time)))
            self.gui.add_to_listbox(
                "Best average reward: {0:.2f}".format(numpy.amax(numpy.array(self.total_reward_history) / 100)))
            if self.gui.animation.update_animation is False:
                self.gui.animation.update_animation = True
                self.gui.draw_values_of_action(self.agent.Q)
                self.gui.draw_policy(self.agent.policy)
                self.gui.restart_environment(self.env.start_state)
                self.gui.animation.update_animation = False
            self.gui.root.update()
            plt.plot(list(range(1, int((self.no_of_episodes / 100) + 1))), numpy.array(self.total_reward_history) / 100)
            plt.title("Result")
            plt.xlabel("Episode Batch (100 episodes)")
            plt.ylabel("Average Reward")
            plt.ylim(self.env.MIN_REWARD, self.env.MAX_REWARD)
            plt.show()

    def update_timescale(self, var):
        """
        Sets the timescale used by the animation canvas to the given parameter value.

        @param var: the new timescale value
        """

        self.timescale = int(var)

    def toggle_animation(self, *args):
        """
        If the animation has been toggled off, sets the timescale to 0 and tells the gui animation
        to stop updating itself.
        """

        if self.gui.animation_toggle.get() == 0:
            self.timescale = 0
            self.gui.scale.configure(state='disabled')
            if self.gui.animation is not None:
                self.gui.animation.update_animation = False
        if self.gui.animation_toggle.get() == 1:
            self.timescale = self.gui.scale.get()
            self.gui.scale.configure(state='active')
            if self.gui.animation is not None:
                self.gui.animation.update_animation = True

    def stop_and_reset(self):
        """
        The Agent and Environment objects have their respective reset methods called, and the animation
        on the GUI is reset, ready for another run.
        """

        self.agent.restart_matrices()
        self.env.restart_environment()
        self.gui.animation.canvas.delete("actions")
        self.gui.animation.canvas.delete("policy")
        if self.gui.animation is not None:
            self.gui.restart_environment(self.env.start_state)
