from tkinter import *
from view.animation import Animation


class AnimationFrozenLake(Animation):
    """
    Animation class for the FrozenLake environment. Builds a 4x4 grid on a Tkinter canvas
    and uses graphics to display the lake, and the agent as a penguin trying to reach
    a fish at the goal state.
    """

    def __init__(self, root, grid_map, update_animation):
        self.GRID_MAP = grid_map
        self.GRID_ROWS = 4
        self.GRID_ROW_HEIGHT = 100
        self.canvas = Canvas(root, width=400, height=400)
        self.canvas.grid(row=0, column=0, columnspan=2)
        self.update_animation = update_animation
        if self.update_animation:
            self.ice_block = PhotoImage(file="view/graphics/iceblock.gif")
            self.ice_hole = PhotoImage(file="view/graphics/icehole.gif")
            self.penguin = PhotoImage(file="view/graphics/penguin.gif")
            self.fish = PhotoImage(file="view/graphics/fish.gif")

            y1 = 0
            for row in range(self.GRID_ROWS):
                x1 = 0
                for col in range(self.GRID_ROWS):
                    if self.GRID_MAP[row][col] == "F":
                        self.canvas.create_image(x1 + self.GRID_ROW_HEIGHT / 2, y1 + self.GRID_ROW_HEIGHT / 2,
                                                 image=self.ice_block)
                    elif self.GRID_MAP[row][col] == "H":
                        self.canvas.create_image(x1 + self.GRID_ROW_HEIGHT / 2, y1 + self.GRID_ROW_HEIGHT / 2,
                                                 image=self.ice_hole)
                    elif self.GRID_MAP[row][col] == "S":
                        self.canvas.create_image(x1 + self.GRID_ROW_HEIGHT / 2, y1 + self.GRID_ROW_HEIGHT / 2,
                                                 image=self.ice_block)
                        self.canvas.create_text(x1 + 25, y1 + 15, font="Arial 12 bold", text="Start")
                        self.agent = self.canvas.create_image(x1 + self.GRID_ROW_HEIGHT / 2,
                                                              y1 + self.GRID_ROW_HEIGHT / 2, image=self.penguin)
                    else:
                        self.canvas.create_image(x1 + self.GRID_ROW_HEIGHT / 2, y1 + self.GRID_ROW_HEIGHT / 2,
                                                 image=self.ice_block)
                        self.canvas.create_image(x1 + self.GRID_ROW_HEIGHT / 2, y1 + self.GRID_ROW_HEIGHT / 2,
                                                 image=self.fish)
                        self.canvas.create_text(x1 + 75, y1 + 85, fill="green", font="Arial 12 bold", text="Goal")
                    x1 += self.GRID_ROW_HEIGHT
                y1 += self.GRID_ROW_HEIGHT

    def update(self, new_state):
        """
        Deletes the agent element on the canvas, and redraws it at the new state location

        @param new_state: the new location to draw the agent
        """

        if self.update_animation:
            self.canvas.delete(self.agent)
            row, col = new_state
            x1 = col * self.GRID_ROW_HEIGHT
            y1 = row * self.GRID_ROW_HEIGHT
            self.agent = self.canvas.create_image(x1 + self.GRID_ROW_HEIGHT / 2, y1 + self.GRID_ROW_HEIGHT / 2,
                                                  image=self.penguin)

    def restart_environment(self, start_state):
        """
        Deletes the agent element on the canvas, and redraws it at the start location

        @param start_state: the environment's start location
        """

        if self.update_animation:
            self.canvas.delete(self.agent)
            row, col = start_state
            x1 = col * self.GRID_ROW_HEIGHT
            y1 = row * self.GRID_ROW_HEIGHT
            self.agent = self.canvas.create_image(x1 + self.GRID_ROW_HEIGHT / 2, y1 + self.GRID_ROW_HEIGHT / 2,
                                                  image=self.penguin)

    def draw_values_of_states(self, values):
        """
        Draws the provided values of state on the canvas.

        @param values: the values of state
        """

        if self.update_animation:
            self.canvas.delete("values")
            for state, value in values.items():
                row, col = state
                x1 = col * self.GRID_ROW_HEIGHT
                y1 = row * self.GRID_ROW_HEIGHT
                self.canvas.create_text(x1 + 85, y1 + 10, font="Arial 8 bold", text="{0:.2f}".format(value),
                                        tag="values")

    def draw_values_of_action(self, values):
        """
        Draws the provided values of action on the canvas.

        @param values: the values of action
        """

        if self.update_animation:
            self.canvas.delete("actions")
            for key, value in values.items():
                state, action = key
                row, col = state
                x1 = col * self.GRID_ROW_HEIGHT
                y1 = row * self.GRID_ROW_HEIGHT
                if action == "Up":
                    self.canvas.create_text(x1 + 50, y1 + 10, font="Arial 8 bold", text="{0:.2f}".format(value),
                                            tag="actions")
                elif action == "Down":
                    self.canvas.create_text(x1 + 50, y1 + 90, font="Arial 8 bold", text="{0:.2f}".format(value),
                                            tag="actions")
                elif action == "Left":
                    self.canvas.create_text(x1 + 10, y1 + 50, font="Arial 8 bold", text="{0:.2f}".format(value),
                                            angle=-90, tag="actions")
                elif action == "Right":
                    self.canvas.create_text(x1 + 90, y1 + 50, font="Arial 8 bold", text="{0:.2f}".format(value),
                                            angle=-90, tag="actions")

    def draw_policy(self, policy):
        """
        Draws arrows indicating the current policy for movement on the grid world.

        @param policy: the current policy
        """

        if self.update_animation:
            self.canvas.delete("policy")
            for state, action in policy.items():
                row, col = state
                if self.GRID_MAP[row][col] != "H" and self.GRID_MAP[row][col] != "G":
                    x1 = (col * self.GRID_ROW_HEIGHT) + (self.GRID_ROW_HEIGHT / 2)
                    y1 = (row * self.GRID_ROW_HEIGHT) + (self.GRID_ROW_HEIGHT / 2)
                    if action == "Left":
                        x2 = x1 + 15
                        x1 -= 15
                        self.canvas.create_line(x1, y1, x2, y1, arrow=FIRST, tag="policy")
                    elif action == "Up":
                        y2 = y1 + 15
                        y1 -= 15
                        self.canvas.create_line(x1, y1, x1, y2, arrow=FIRST, tag="policy")
                    elif action == "Right":
                        x2 = x1 + 15
                        x1 -= 15
                        self.canvas.create_line(x1, y1, x2, y1, arrow=LAST, tag="policy")
                    elif action == "Down":
                        y2 = y1 + 15
                        y1 -= 15
                        self.canvas.create_line(x1, y1, x1, y2, arrow=LAST, tag="policy")

    def draw_softmax_probabilities(self, values):
        """
        Draws the provided softmax probabilities values on canvas

        @param values: the softmax probability values
        """

        if self.update_animation:
            self.canvas.delete("probabilities")
            for state, probs in values.items():
                row, col = state
                x1 = col * self.GRID_ROW_HEIGHT
                y1 = row * self.GRID_ROW_HEIGHT
                for i, p in enumerate(probs):
                    if i == 1:
                        self.canvas.create_text(x1 + 50, y1 + 10, font="Arial 8 bold", text="{0:.2f}".format(p),
                                                tag="probabilities")
                    elif i == 3:
                        self.canvas.create_text(x1 + 50, y1 + 90, font="Arial 8 bold", text="{0:.2f}".format(p),
                                                tag="probabilities")
                    elif i == 0:
                        self.canvas.create_text(x1 + 10, y1 + 50, font="Arial 8 bold", text="{0:.2f}".format(p),
                                                angle=-90, tag="probabilities")
                    elif i == 2:
                        self.canvas.create_text(x1 + 90, y1 + 50, font="Arial 8 bold", text="{0:.2f}".format(p),
                                                angle=-90, tag="probabilities")
