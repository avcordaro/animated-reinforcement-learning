from controller.environment_animation_factory import EnvironmentAnimationFactory
from controller.controller_agent_factory import ControllerAgentFactory
from view.gui import GUI
from tkinter import *


class MainDriver:
    """
    Main driver class for the program, which connects up the controller, agent, gui, environment and animation
    instances. Also handles user changes to the selected algorithm or selected environment on the gui.
    """

    def __init__(self):
        self.root = Tk()
        self.root.title("Animated Reinforcement Learning")
        self.gui = GUI(self.root, None)
        self.env, self.animation, algorithms = EnvironmentAnimationFactory.create_environment_and_animation(self.gui.selected_environment.get(), self.root)
        self.gui.animation = self.animation
        algorithm_menu = self.gui.algorithm_menu["menu"]
        algorithm_menu.delete(0, "end")
        for algorithm in algorithms:
            algorithm_menu.add_command(label=algorithm, command=lambda value=algorithm: self.gui.selected_algorithm.set(value))
        self.gui.selected_algorithm.set(algorithms[0])
        self.controller, self.agent = ControllerAgentFactory.create_controller_and_agent(self.gui.selected_algorithm.get(), self.env, self.gui)
        self.gui.selected_algorithm.trace("w", self.change_controller_agent)
        self.gui.selected_environment.trace("w", self.change_environment_animation)
        self.root.mainloop()

    def change_controller_agent(self, *args):
        """
        Changes the controller and agent objects to the correct implementation, based on the user's new algorithm
        selection.
        """

        if self.controller.canvas_after_variable is not None:
            self.gui.animation.canvas.after_cancel(self.controller.canvas_after_variable)
        self.gui.animation.canvas.delete("values")
        self.gui.animation.canvas.delete("actions")
        self.gui.animation.canvas.delete("policy")
        self.gui.animation.canvas.delete("probabilities")
        self.env.restart_environment()
        self.gui.restart_environment(self.env.current_state)
        self.controller, self.agent = ControllerAgentFactory.create_controller_and_agent(self.gui.selected_algorithm.get(), self.env, self.gui)

    def change_environment_animation(self, *args):
        """
        Changes the environment and animation objects to the correct implementation, based on the user's new environment
        selection.
        """

        if self.controller.canvas_after_variable is not None:
            self.gui.animation.canvas.after_cancel(self.controller.canvas_after_variable)
        self.controller.stop_and_reset()
        self.gui.animation.canvas.destroy()
        self.env, self.animation, algorithms = EnvironmentAnimationFactory.create_environment_and_animation(self.gui.selected_environment.get(), self.root)
        self.gui.animation = self.animation
        self.agent.env = self.env
        self.controller.env = self.env
        self.controller.MAX_REWARD = self.env.MAX_REWARD
        self.controller.MIN_REWARD = self.env.MIN_REWARD
        if hasattr(self.controller, "REWARD_THRESHOLD"):
            self.controller.REWARD_THRESHOLD = self.env.REWARD_THRESHOLD
        algorithm_menu = self.gui.algorithm_menu["menu"]
        algorithm_menu.delete(0, "end")
        for algorithm in algorithms:
            algorithm_menu.add_command(label=algorithm, command=lambda value=algorithm: self.gui.selected_algorithm.set(value))
        self.gui.selected_algorithm.set(algorithms[0])


# Runs the program.
MainDriver()