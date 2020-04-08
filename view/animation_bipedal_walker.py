from tkinter import *
from view.animation import Animation


class AnimationBipedalWalker(Animation):
    """
    Wrapper class for animating the Bipedal Walker environment. The OpenAI gym package from which
    the environment is from already has it's own rendering methods.
    """

    def __init__(self, root, env, update_animation):
        self.env = env
        self.canvas = Canvas(root, width=400, height=400)
        self.canvas.grid(row=0, column=0, columnspan=2)
        self.update_animation = update_animation
        if self.update_animation:
            self.canvas.create_text(200, 200, font="Arial 12 bold",
                                    text="Animation will be displayed in a separate window.")

    def update(self):
        """
        Calls the gym environment render method.
        """

        if self.update_animation:
            self.env.gym_env.render()

    def restart_environment(self, start_state):
        pass

    def close_animation(self):
        """
        Closes the window containing the Pong animation.
        """

        self.env.gym_env.close()
