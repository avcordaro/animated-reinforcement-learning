from tkinter import *


class GUI:
    """
    The GUI for the program, which contains all the buttons and controls. Also displays the
    animations of the environment simulations.
    """

    def __init__(self, root, animation):
        self.root = root
        self.animation = animation

        self.listbox = Listbox(root, height=25, width=60)
        self.listbox.grid(row=0, column=2, padx=(0, 10), pady=(10, 0))

        self.episode_number = StringVar()
        self.episode_number.set("Episode: 0")
        self.label_episode_number = Label(root, textvariable=self.episode_number)
        self.label_episode_number.grid(row=1, column=0, columnspan=3, padx=(10, 0), sticky=W)

        self.epsilon_value = StringVar()
        self.epsilon_value.set("")
        self.label_epsilon = Label(root, textvariable=self.epsilon_value)
        self.label_epsilon.grid(row=1, column=0, columnspan=3, padx=(110, 0), sticky=W)

        self.alpha_value = StringVar()
        self.alpha_value.set("")
        self.label_alpha = Label(root, textvariable=self.alpha_value)
        self.label_alpha.grid(row=1, column=0, columnspan=3, padx=(210, 0), sticky=W)

        self.beta_value = StringVar()
        self.beta_value.set("")
        self.label_beta = Label(root, textvariable=self.beta_value)
        self.label_beta.grid(row=1, column=0, columnspan=3, padx=(310, 0), sticky=W)

        self.label_environment = Label(root, text="Environment:")
        self.label_environment.grid(row=2, column=0, padx=(10, 0), sticky=W)
        environment_list = ["Frozen Lake 4x4", "Frozen Lake 4x4 - Slippery", "Frozen Lake 8x8",
                            "Frozen Lake 8x8 - Slippery", "Cliff Walking", "Taxi Driver", "CartPole", "Pong",
                            "Lunar Lander", "Bipedal Walker"]
        self.selected_environment = StringVar()
        self.selected_environment.set(environment_list[0])
        self.environment_menu = OptionMenu(root, self.selected_environment, *environment_list)
        self.environment_menu.config(width=30)
        self.environment_menu.grid(row=2, column=1, padx=(0, 80), sticky=W)

        self.label_algorithm = Label(root, text="Algorithm:")
        self.label_algorithm.grid(row=3, column=0, padx=(10, 0), pady=(5, 0), sticky=NW)
        algorithm_list = ["Value Iteration", "Policy Iteration", "First-visit Monte Carlo Control", "Q-Learning",
                          "SARSA", "REINFORCE", "Actor-Critic"]
        self.selected_algorithm = StringVar()
        self.selected_algorithm.set(algorithm_list[0])
        self.algorithm_menu = OptionMenu(root, self.selected_algorithm, *algorithm_list)
        self.algorithm_menu.config(width=30)
        self.algorithm_menu.grid(row=3, column=1, padx=(0, 80), sticky=NW)

        self.scale = Scale(root, orient="horizontal", from_=100, to=1, length=400, label="Timescale (ms)")
        self.scale.grid(row=2, column=2, padx=(0, 20), pady=(5, 10), sticky=E)
        self.scale.set(100)
        self.start_button = Button(root, text="    Run    ")
        self.start_button.grid(row=3, column=2, padx=(0, 20), pady=(10, 20), sticky=NE)
        self.animation_toggle = IntVar()
        self.animation_toggle.set(1)
        self.animation_chkbtn = Checkbutton(root, text=" Animation", variable=self.animation_toggle)
        self.animation_chkbtn.grid(row=3, column=2, padx=(0, 125), pady=(15, 20), sticky=NE)
        self.load_best_model = IntVar()
        self.load_best_model_chkbtn = Checkbutton(root, text=" Load best model", variable=self.load_best_model)
        self.load_best_model_chkbtn.grid(row=3, column=2, padx=(0, 250), pady=(15, 20), sticky=NE)
        self.load_best_model_chkbtn.grid_remove()

    def add_to_listbox(self, str):
        """
        Adds a string to the listbox on screen, useful for display statistical information

        @param str: the string to be added
        """

        self.listbox.insert(END, str)
        self.listbox.yview(END)

    def update_episode_labels(self, episode_no, epsilon=None, alpha=None, beta=None, frame=None):
        """
        Updates various labels with provided information about the running episode

        @param episode_no: the number of the current episode
        @param epsilon: current value of epsilon being used
        @param alpha: current value of alpha being used
        @param beta: current value of beta being used
        @param frame: current frame number of the episode
        """

        self.episode_number.set("Episode: " + str(episode_no))
        if epsilon is not None:
            self.epsilon_value.set("\u03B5 : {0:.4f}".format(epsilon))
        else:
            self.epsilon_value.set("")
        if alpha is not None:
            self.alpha_value.set("\u03B1 : {0:.4f}".format(alpha))
        else:
            self.alpha_value.set("")
        if beta is not None:
            self.beta_value.set("\u03B2 : {0:.4f}".format(beta))
        elif frame is not None:
            self.beta_value.set("Frame: {}".format(frame))
        else:
            self.beta_value.set("")

    def update(self, new_state):
        """
        Tells the animation class to update canvas using the new state information.

        @param new_state: the new state information
        """

        self.animation.update(new_state)

    def restart_environment(self, start_state):
        """
        Tells the animation class to reset the episode number label to 0, and update the canvas with the
        starting state information.

        @param start_state: the start state information
        """

        self.animation.restart_environment(start_state)

    def draw_values_of_states(self, values):
        """
        Tells the animation class to draw the provided values of state on the canvas.

        @param values: the values of state
        """

        self.animation.draw_values_of_states(values)

    def draw_values_of_action(self, values):
        """
        Tells the animation class to draw the provided values of action on the canvas.

        @param values: the values of action
        """

        self.animation.draw_values_of_action(values)

    def draw_policy(self, policy):
        """
        Tells the animation class to draw arrows indicating the current policy for movement on the grid world.

        @param policy: the current policy
        """

        self.animation.draw_policy(policy)

    def draw_softmax_probabilities(self, values):
        """
        Tells the animation class to draw the provided softmax probabilities on the canvas.

        @param values: the values of the softmax probabilities
        """

        self.animation.draw_softmax_probabilities(values)
