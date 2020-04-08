from tkinter import *
from view.animation import Animation


class AnimationTaxiDriver(Animation):
    """
    Animation class for the TaxiDriver environment. Builds a 5x5 grid on a Tkinter canvas
    and uses graphics to display the taxi, passenger, destination and tarmac scenery.
    """

    def __init__(self, root, env, update_animation):
        self.env = env
        self.GRID_MAP = self.env.GRID_MAP
        self.GRID_ROWS = 5
        self.GRID_COLUMNS = 5
        self.GRID_ROW_HEIGHT = 80
        self.canvas = Canvas(root, width=400, height=400)
        self.canvas.grid(row=0, column=0, columnspan=2)
        self.update_animation = update_animation
        if self.update_animation:
            self.road_block = PhotoImage(file="view/graphics/roadblock.gif")
            self.wall = PhotoImage(file="view/graphics/wall.gif")
            self.taxi = PhotoImage(file="view/graphics/taxi.gif")
            self.passenger_man = PhotoImage(file="view/graphics/passenger.gif")
            self.finish_flag = PhotoImage(file="view/graphics/finish.gif")

            y1 = 0
            for row in range(self.GRID_ROWS):
                x1 = 0
                for col in range(0, 2 * self.GRID_COLUMNS, 2):
                    if self.GRID_MAP[row][col] == " ":
                        self.canvas.create_image(x1 + self.GRID_ROW_HEIGHT / 2, y1 + self.GRID_ROW_HEIGHT / 2,
                                                 image=self.road_block)
                    if col > 0 and self.GRID_MAP[row][col - 1] == "|":
                        self.canvas.create_image(x1, y1 + self.GRID_ROW_HEIGHT / 2, image=self.wall)
                    x1 += self.GRID_ROW_HEIGHT
                y1 += self.GRID_ROW_HEIGHT

            row, col = self.env.taxi_state
            y1 = row * self.GRID_ROW_HEIGHT
            x1 = col * self.GRID_ROW_HEIGHT
            self.agent = self.canvas.create_image(x1 + self.GRID_ROW_HEIGHT / 2, y1 + self.GRID_ROW_HEIGHT / 2,
                                                  image=self.taxi)

            row, col = self.env.passenger_state
            y1 = row * self.GRID_ROW_HEIGHT
            x1 = col * self.GRID_ROW_HEIGHT
            self.passenger = self.canvas.create_image(x1 + self.GRID_ROW_HEIGHT / 2, y1 + self.GRID_ROW_HEIGHT / 2,
                                                      image=self.passenger_man)

            row, col = self.env.destination_state
            y1 = row * self.GRID_ROW_HEIGHT
            x1 = col * self.GRID_ROW_HEIGHT
            self.destination = self.canvas.create_image(x1 + self.GRID_ROW_HEIGHT / 2, y1 + self.GRID_ROW_HEIGHT / 2,
                                                        image=self.finish_flag)

    def update(self, new_state):
        """
        Deletes the taxi and passenger elements on the canvas, then redraws them at their new state location

        @param new_state: the new state including the location of the passenger and taxi
        """

        if self.update_animation:
            self.canvas.delete(self.agent)
            row, col = new_state[0]
            y1 = row * self.GRID_ROW_HEIGHT
            x1 = col * self.GRID_ROW_HEIGHT
            self.agent = self.canvas.create_image(x1 + self.GRID_ROW_HEIGHT / 2, y1 + self.GRID_ROW_HEIGHT / 2,
                                                  image=self.taxi)

            self.canvas.delete(self.passenger)
            if not self.env.passenger_in_taxi:
                row, col = new_state[1]
                y1 = row * self.GRID_ROW_HEIGHT
                x1 = col * self.GRID_ROW_HEIGHT
                self.passenger = self.canvas.create_image(x1 + self.GRID_ROW_HEIGHT / 2, y1 + self.GRID_ROW_HEIGHT / 2,
                                                          image=self.passenger_man)

    def restart_environment(self, start_state):
        """
        Deletes the taxi, passenger and destination element on the canvas, then redraws them at their new start location

        @param start_state: the environment's start state, with locations of taxi, passenger and destination
        """

        if self.update_animation:
            self.update(start_state)
            self.canvas.delete(self.destination)
            row, col = start_state[2]
            y1 = row * self.GRID_ROW_HEIGHT
            x1 = col * self.GRID_ROW_HEIGHT
            self.destination = self.canvas.create_image(x1 + self.GRID_ROW_HEIGHT / 2, y1 + self.GRID_ROW_HEIGHT / 2,
                                                        image=self.finish_flag)

    def draw_values_of_states(self, values):
        pass

    def draw_values_of_action(self, values):
        pass

    def draw_policy(self, policy):
        pass

    def draw_softmax_probabilities(self, values):
        pass
