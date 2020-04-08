from tkinter import *
from view.animation import Animation


class AnimationFrozenLake8x8(Animation):
    """
    Animation class for the FrozenLake environment. Builds a 8x8 grid on a Tkinter canvas
    and uses graphics to display the lake, and the agent as a penguin trying to reach
    a fish at the goal state.
    """

    def __init__(self, root, grid_map, update_animation):
        self.GRID_MAP = grid_map
        self.GRID_ROWS = 8
        self.GRID_ROW_HEIGHT = 50
        self.canvas = Canvas(root, width=400, height=400)
        self.canvas.grid(row=0, column=0, columnspan=2, padx=(5, 0), pady=(5, 0))
        self.update_animation = update_animation
        if self.update_animation:
            self.ice_block = PhotoImage(file="view/graphics/iceblock_small.gif")
            self.ice_hole = PhotoImage(file="view/graphics/icehole_small.gif")
            self.penguin = PhotoImage(file="view/graphics/penguin_small.gif")
            self.fish = PhotoImage(file="view/graphics/fish_small.gif")

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
                        self.canvas.create_text(x1 + 12.5, y1 + 7.5, font="Arial 10 bold", text="Start")
                        self.agent = self.canvas.create_image(x1 + self.GRID_ROW_HEIGHT / 2,
                                                              y1 + self.GRID_ROW_HEIGHT / 2, image=self.penguin)
                    else:
                        self.canvas.create_image(x1 + self.GRID_ROW_HEIGHT / 2, y1 + self.GRID_ROW_HEIGHT / 2,
                                                 image=self.ice_block)
                        self.canvas.create_image(x1 + self.GRID_ROW_HEIGHT / 2, y1 + self.GRID_ROW_HEIGHT / 2,
                                                 image=self.fish)
                        self.canvas.create_text(x1 + 37.5, y1 + 42.5, fill="green", font="Arial 10 bold", text="Goal")
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
                self.canvas.create_text(x1 + 42.5, y1 + 5, font="Arial 6 bold", text="{0:.2f}".format(value),
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
                    self.canvas.create_text(x1 + 25, y1 + 5, font="Arial 6 bold", text="{0:.2f}".format(value),
                                            tag="actions")
                elif action == "Down":
                    self.canvas.create_text(x1 + 25, y1 + 45, font="Arial 6 bold", text="{0:.2f}".format(value),
                                            tag="actions")
                elif action == "Left":
                    self.canvas.create_text(x1 + 5, y1 + 25, font="Arial 6 bold", text="{0:.2f}".format(value),
                                            angle=-90, tag="actions")
                elif action == "Right":
                    self.canvas.create_text(x1 + 45, y1 + 25, font="Arial 6 bold", text="{0:.2f}".format(value),
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
                        x2 = x1 + 7.5
                        x1 -= 7.5
                        self.canvas.create_line(x1, y1, x2, y1, arrow=FIRST, tag="policy")
                    elif action == "Up":
                        y2 = y1 + 7.5
                        y1 -= 7.5
                        self.canvas.create_line(x1, y1, x1, y2, arrow=FIRST, tag="policy")
                    elif action == "Right":
                        x2 = x1 + 7.5
                        x1 -= 7.5
                        self.canvas.create_line(x1, y1, x2, y1, arrow=LAST, tag="policy")
                    elif action == "Down":
                        y2 = y1 + 7.5
                        y1 -= 7.5
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
                        self.canvas.create_text(x1 + 25, y1 + 5, font="Arial 6 bold", text="{0:.2f}".format(p),
                                                tag="probabilities")
                    elif i == 3:
                        self.canvas.create_text(x1 + 25, y1 + 45, font="Arial 6 bold", text="{0:.2f}".format(p),
                                                tag="probabilities")
                    elif i == 0:
                        self.canvas.create_text(x1 + 5, y1 + 25, font="Arial 6 bold", text="{0:.2f}".format(p),
                                                angle=-90, tag="probabilities")
                    elif i == 2:
                        self.canvas.create_text(x1 + 45, y1 + 25, font="Arial 6 bold", text="{0:.2f}".format(p),
                                                angle=-90, tag="probabilities")
