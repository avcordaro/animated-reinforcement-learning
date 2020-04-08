from controller.controller import Controller
from tkinter import messagebox
import matplotlib.pyplot as plt
import numpy
import time
import configparser
import os.path
import torch


class ControllerDQNPong(Controller):
    """
    The Controller class for running a Deep Q-Network agent in a chosen environment. The Controller
    observes user input from the GUI and calls the appropriate methods from the Model (Agent
    and Environment objects) before updating the GUI with changes returned from the Model.

    In the case of Deep Q-Network, the Controller will run a given number of episodes using the
    Agent methods, and display statistics of the episodes on the GUI.
    """

    def __init__(self, env, agent, gui, epsilon=1.0, episodes=100):
        self.env = env
        self.agent = agent
        self.gui = gui
        self.gui.scale.configure(command=self.update_timescale)
        self.gui.start_button.configure(command=self.start)
        self.gui.animation_toggle.trace("w", self.toggle_animation)
        self.gui.load_best_model.trace("w", self.load_best_model)
        self.loaded_best_model = False
        self.gui.update_episode_labels(0, 0, 0, None, 0)

        self.canvas_after_variable = None
        self.timescale = self.gui.scale.get()
        self.toggle_animation()

        self.epsilon = epsilon
        self.epsilon_decay = 0
        self.FINAL_EPSILON = 0
        self.no_of_episodes = episodes
        self.sync_target_net_frames = 1000

        self.frame_counter = 0
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
            self.agent.GAMMA = config["deep-q-network-pong"].getfloat("gamma")
            self.agent.ALPHA = config["deep-q-network-pong"].getfloat("alpha")
            self.agent.BATCH_SIZE = config["deep-q-network-pong"].getint("batch_size")
            self.epsilon = config["deep-q-network-pong"].getfloat("epsilon")
            self.epsilon_decay = config["deep-q-network-pong"].getfloat("epsilon_decay_amount")
            self.FINAL_EPSILON = config["deep-q-network-pong"].getfloat("final_epsilon")
            self.no_of_episodes = config["deep-q-network-pong"].getint("episodes")
            self.sync_target_net_frames = config["deep-q-network-pong"].getint("sync_target_net_frames")

    def start(self):
        """
        The method called when the "Run" button is pressed on the GUI. It resets many of the
        various class attributes, and then initiates running the episodes.
        """

        self.read_config_file()
        if self.canvas_after_variable is not None:
            self.gui.animation.canvas.after_cancel(self.canvas_after_variable)
        self.episode_counter = 0
        self.frame_counter = 0
        self.total_reward = 0
        self.total_reward_history = []
        self.stop_and_reset()
        self.gui.update_episode_labels(self.episode_counter, self.epsilon, self.agent.ALPHA, None, self.frame_counter)
        self.gui.add_to_listbox("=============================================")
        self.gui.add_to_listbox("DEEP Q-NETWORK FOR PONG")
        self.gui.add_to_listbox("Running episodes.")
        self.start_time = time.time()
        self.run_episode()

    def run_episode(self):
        """
        Runs a single episode of the environment, learning via Deep Q-Network on the fly. This method
        will continually call itself until the desired number of episodes have been completed. Then
        the reward statistics from the episodes are displayed on the GUI.
        """
        self.frame_counter += 1
        if self.loaded_best_model:
            self.gui.update_episode_labels(self.episode_counter, 0, 0, None, self.frame_counter)
        else:
            self.gui.update_episode_labels(self.episode_counter, self.epsilon, self.agent.ALPHA, None,
                                           self.frame_counter)
        reward = episode_done = None
        if self.loaded_best_model:
            _, reward, episode_done = self.agent.test_step()
        elif numpy.random.random() < self.epsilon:
            _, reward, episode_done = self.agent.random_step()
        else:
            _, reward, episode_done = self.agent.step()
        self.total_reward += reward
        self.gui.animation.update()
        self.epsilon = max(self.FINAL_EPSILON, self.epsilon - self.epsilon_decay)

        if episode_done:
            self.episode_counter += 1
            self.env.restart_environment()
            if self.episode_counter % 100 == 0:
                self.gui.add_to_listbox("(" + str(self.episode_counter - 99) + "-" + str(
                    self.episode_counter) + ") Average reward: {0:.2f}".format(self.total_reward / 100))
                self.total_reward_history.append(self.total_reward)
                self.total_reward = 0

        if not self.loaded_best_model and self.frame_counter % self.sync_target_net_frames == 0:
            self.agent.target_net.load_state_dict(self.agent.net.state_dict())

        if self.episode_counter < self.no_of_episodes:
            if self.timescale == 0:
                self.gui.root.update_idletasks()
            self.canvas_after_variable = self.gui.animation.canvas.after(self.timescale, self.run_episode)
        else:
            self.end_time = time.time()
            self.gui.add_to_listbox("Finished {0} episodes. ({1:.2f} secs)".format(self.no_of_episodes, float(
                self.end_time - self.start_time)))
            self.gui.add_to_listbox(
                "Best average reward: {0:.2f}".format(numpy.amax(numpy.array(self.total_reward_history) / 100)))
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
                if self.env.name == "Pong":
                    self.gui.animation.close_animation()
        if self.gui.animation_toggle.get() == 1:
            self.timescale = self.gui.scale.get()
            self.gui.scale.configure(state='active')
            if self.gui.animation is not None:
                self.gui.animation.update_animation = True

    def load_best_model(self, *args):
        """
        If the load best model has been toggled on, the agent's network is loaded with the saved state dictionary
        of the best model previously trained. The agent will also stop performing backwards passes or using
        epsilon, as it just wants to test out the best model parameters on the environment.

        """
        if self.gui.load_best_model.get() == 1:
            if os.path.exists("neural_network_models/pong_best_parameters.pth"):
                self.loaded_best_model = True
                self.agent.net.load_state_dict(torch.load("neural_network_models/pong_best_parameters.pth"))
            else:
                self.gui.load_best_model.set(0)
                messagebox.showinfo("No Model Found", "Cannot find file \'pong_best_parameters.pth\' "
                                                      "which contains the best model parameters.")
        else:
            self.loaded_best_model = False
            if self.env.name == "Pong":
                self.agent.restart_matrices()

    def stop_and_reset(self):
        """
        The Agent and Environment objects have their respective reset methods called, ready for another run.
        """

        if not self.loaded_best_model:
            self.agent.restart_matrices()
        self.env.restart_environment()
        if self.env.name == "Pong":
            self.gui.animation.close_animation()
