from controller.controller import Controller
import matplotlib.pyplot as plt
import time
import configparser
import os.path


class ControllerPolicyIter(Controller):
    """
    The Controller class for running a Policy Iteration agent in a chosen environment. The Controller
    observes user input from the GUI and calls the appropriate methods from the Model (Agent and
    Environment objects) before updating the GUI with changes returned from the Model.

    In the case of Policy Iteration, the Controller will run exploration and greedy episodes
    separately using the Agent methods, and display statistics of the episodes on the GUI.
    """

    def __init__(self, env, agent, gui):
        self.env = env
        self.agent = agent
        self.gui = gui
        self.gui.scale.configure(command=self.update_timescale)
        self.gui.start_button.configure(command=self.start)
        self.gui.animation_toggle.trace("w", self.toggle_animation)
        self.gui.update_episode_labels(0)

        self.canvas_after_variable = None
        self.timescale = self.gui.scale.get()
        self.toggle_animation()

        self.iteration_counter = 1
        self.episode_counter = 0
        self.episode_steps = 0

        self.episode_total_reward = 0
        self.total_reward = 0
        self.avg_rewards = []

        self.no_of_greedy_episodes = 100
        self.no_of_exploration_episodes = 10

    def read_config_file(self):
        """
        Uses Python's in-built configparser module to read the program's configuration file, which is
        in INI format. Sets the relevant parameters to the config file values.
        """

        config = configparser.ConfigParser()
        if os.path.exists("config.txt"):
            config.read("config.txt")
            self.agent.GAMMA = config["dynamic-programming"].getfloat("gamma")
            self.no_of_greedy_episodes = config["dynamic-programming"].getint("greedy_episodes")
            self.no_of_exploration_episodes = config["dynamic-programming"].getint("exploration_episodes")

    def start(self):
        """
        The method called when the "Run" button is pressed on the GUI. It resets many of the
        various class attributes, and then initiates running the exploration episodes.
        """

        self.read_config_file()
        if self.canvas_after_variable is not None:
            self.gui.animation.canvas.after_cancel(self.canvas_after_variable)
        self.episode_counter = 0
        self.episode_steps = 0
        self.iteration_counter = 1
        self.episode_total_reward = 0
        self.total_reward = 0
        self.avg_rewards = []
        self.stop_and_reset()
        self.gui.update_episode_labels(self.episode_counter)
        self.gui.draw_values_of_states(self.agent.values_of_state)
        self.gui.draw_policy(self.agent.policy)
        self.gui.add_to_listbox("=============================================")
        self.gui.add_to_listbox("POLICY ITERATION")
        self.gui.add_to_listbox("Running exploration episodes.")
        self.start_time = time.time()
        self.run_exploration_episode()

    def run_exploration_episode(self):
        """
        Exploration episodes are ones in which the agent chooses random actions at each step. This method
        will continually call itself until the desired number of episodes have been completed. No
        statistics are displayed after the episodes are completed. The agent uses its experience
        from exploring to evaluate/improve its policy of state using Policy Iteration.
        """

        new_state, reward, episode_done = self.agent.random_step()
        self.gui.update(new_state)

        if episode_done:
            self.episode_counter += 1
            self.env.restart_environment()
            self.gui.restart_environment(self.env.start_state)
            self.gui.update_episode_labels(self.episode_counter)

        if self.episode_counter != self.no_of_exploration_episodes:
            if self.timescale == 0:
                self.gui.root.update_idletasks()
            self.canvas_after_variable = self.gui.animation.canvas.after(self.timescale, self.run_exploration_episode)
        else:
            self.episode_counter = 0
            self.agent.evaluate_policy()
            self.gui.draw_values_of_states(self.agent.values_of_state)
            self.gui.draw_policy(self.agent.policy)
            self.gui.add_to_listbox("Running greedy episodes.")
            self.run_greedy_episode()

    def run_greedy_episode(self):
        """
        Greedy episodes are ones in which the agent chooses the best action at each step, according to
        its policy. This method will continually call itself until the desired number of episodes have
        been completed. Then the reward statistics from the episodes are displayed on the GUI. If
        the average reward from the episodes does not meet a desired threshold, then exploration
        episodes are run again.
        """

        new_state, reward, episode_done = self.agent.step()
        self.episode_total_reward += reward
        self.episode_steps += 1
        self.gui.update(new_state)

        if episode_done or self.episode_steps >= self.env.MAX_EPISODE_STEPS:
            self.total_reward += self.episode_total_reward
            self.episode_total_reward = 0
            self.episode_counter += 1
            self.episode_steps = 0
            self.env.restart_environment()
            self.gui.restart_environment(self.env.start_state)
            self.gui.update_episode_labels(self.episode_counter)

        if self.episode_counter != self.no_of_greedy_episodes:
            if self.timescale == 0:
                self.gui.root.update_idletasks()
            self.canvas_after_variable = self.gui.animation.canvas.after(self.timescale, self.run_greedy_episode)
        else:
            avg_reward = self.total_reward / self.no_of_greedy_episodes
            self.avg_rewards.append(avg_reward)
            self.total_reward = 0
            self.gui.add_to_listbox("Average reward for greedy episodes: {0:.3f}".format(avg_reward))
            if avg_reward < self.env.REWARD_THRESHOLD:
                self.iteration_counter += 1
                self.episode_counter = 0
                self.gui.add_to_listbox("Running exploration episodes.")
                self.run_exploration_episode()
            else:
                self.end_time = time.time()
                self.gui.add_to_listbox("Completed in {0} iterations. ({1:.2f} secs)"
                                        .format(self.iteration_counter, (self.end_time - self.start_time)))
                if self.gui.animation.update_animation is False:
                    self.gui.animation.update_animation = True
                    self.gui.draw_values_of_states(self.agent.values_of_state)
                    self.gui.draw_policy(self.agent.policy)
                    self.gui.restart_environment(self.env.start_state)
                    self.gui.animation.update_animation = False
                self.gui.root.update()
                if self.iteration_counter > 1:
                    plt.plot(list(range(1, self.iteration_counter + 1)), self.avg_rewards)
                    plt.title("Result")
                    plt.xlabel("Iteration")
                    plt.ylabel("Average Reward")
                    plt.ylim(self.env.MIN_REWARD, self.env.MAX_REWARD)
                    plt.show()

    def update_timescale(self, var):
        """
        Sets the timescale used by the animation canvas to the given parameter value.

        @param var: the new timescale value
        """

        self.timescale = var

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
        self.gui.animation.canvas.delete("values")
        self.gui.animation.canvas.delete("policy")
        if self.gui.animation is not None:
            self.gui.restart_environment(self.env.start_state)
