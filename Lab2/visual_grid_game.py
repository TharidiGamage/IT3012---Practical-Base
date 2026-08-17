# visual_grid_game.py
import random
import tkinter as tk


class VisualGridHuntGame:
    """Grid environment for IT3012 Practical 02."""

    def __init__(
        self,
        width=12,
        height=12,
        num_food=15,
        num_opponents=0,
        custom_walls=None
    ):
        self.width = width
        self.height = height
        self.agent_pos = [0, 0]
        self.direction = "Right"

        if custom_walls is not None:
            self.walls = set(custom_walls)
        else:
            self.walls = {
                (2, 2),
                (2, 3),
                (5, 5),
                (6, 5),
                (3, 7)
            }

        self.food_positions = set()

        while len(self.food_positions) < num_food:
            fx = random.randint(0, self.width - 1)
            fy = random.randint(0, self.height - 1)
            pos = (fx, fy)

            if pos != (0, 0) and pos not in self.walls:
                self.food_positions.add(pos)

        self.opponents = []

        while len(self.opponents) < num_opponents:
            ox = random.randint(0, self.width - 1)
            oy = random.randint(0, self.height - 1)
            pos = (ox, oy)

            if (
                pos != (0, 0)
                and pos not in self.walls
                and pos not in self.food_positions
            ):
                self.opponents.append([ox, oy])

        self.toxic_traps = set()

        while len(self.toxic_traps) < 3:
            tx = random.randint(0, self.width - 1)
            ty = random.randint(0, self.height - 1)
            pos = (tx, ty)

            if (
                pos != (0, 0)
                and pos not in self.walls
                and pos not in self.food_positions
            ):
                self.toxic_traps.add(pos)

        self.score = 0
        self.steps = 0
        self.collision = False

    def _next_position(self, direction):
        x, y = self.agent_pos

        if direction == "Up":
            return (x, y + 1)
        if direction == "Down":
            return (x, y - 1)
        if direction == "Left":
            return (x - 1, y)
        return (x + 1, y)

    def _is_blocked(self, position):
        x, y = position

        return (
            position in self.walls
            or x < 0
            or x >= self.width
            or y < 0
            or y >= self.height
        )

    def get_percept(self) -> dict:
        """
        Local percept for the Model-Based Agent.

        The agent does not receive the full environment.
        It only receives information about nearby cells.
        """

        directions = {
            "Up": "Up",
            "Right": "Right",
            "Down": "Down",
            "Left": "Left"
        }

        left_of = {
            "Up": "Left",
            "Left": "Down",
            "Down": "Right",
            "Right": "Up"
        }

        right_of = {
            "Up": "Right",
            "Right": "Down",
            "Down": "Left",
            "Left": "Up"
        }

        ahead = self._next_position(directions[self.direction])
        left = self._next_position(left_of[self.direction])
        right = self._next_position(right_of[self.direction])

        return {
            "wall_ahead": self._is_blocked(ahead),
            "food_here": tuple(self.agent_pos) in self.food_positions,

            # Local information used by the model-based agent.
            "left_blocked": self._is_blocked(left),
            "right_blocked": self._is_blocked(right),

            # The positions themselves are local percept information.
            "left_position": left,
            "right_position": right,
            "ahead_position": ahead
        }

    def execute_action(self, action: str):
        self.steps += 1

        # Eat food.
        if action == "Suck":
            current = tuple(self.agent_pos)

            if current in self.food_positions:
                self.food_positions.remove(current)
                self.score += 20

            return

        # Turn left.
        if action == "TurnLeft":
            order = ["Up", "Left", "Down", "Right"]
            index = order.index(self.direction)
            self.direction = order[(index + 1) % 4]
            return

        # Turn right.
        if action == "TurnRight":
            order = ["Up", "Right", "Down", "Left"]
            index = order.index(self.direction)
            self.direction = order[(index + 1) % 4]
            return

        # Move forward.
        if action == "Forward":
            new_pos = self._next_position(self.direction)

            if self._is_blocked(new_pos):
                self.score -= 5
            else:
                self.agent_pos = list(new_pos)

            current = tuple(self.agent_pos)

            if current in self.toxic_traps:
                self.score -= 15

            # Move opponents if any exist.
            for op in self.opponents:
                move = random.choice(
                    ["Up", "Down", "Left", "Right", "Stay"]
                )

                if move == "Up" and op[1] < self.height - 1:
                    op[1] += 1
                elif move == "Down" and op[1] > 0:
                    op[1] -= 1
                elif move == "Left" and op[0] > 0:
                    op[0] -= 1
                elif move == "Right" and op[0] < self.width - 1:
                    op[0] += 1

                if op == self.agent_pos:
                    self.score -= 50
                    self.collision = True

    def is_done(self):
        return (
            len(self.food_positions) == 0
            or self.steps >= 100
            or self.collision
        )


class SimpleReflexAgent:
    """Agent for Step 1.2."""

    def sense_and_act(self, percept):
        if percept["food_here"]:
            return "Suck"

        if percept["wall_ahead"]:
            return "TurnLeft"

        return "Forward"


class ModelBasedAgent:
    """
    Model-Based Agent for Step 1.3.

    It maintains an internal state containing cells it has already visited.
    It uses that state when deciding whether to turn or move forward.
    """

    def __init__(self):
        self.visited_cells = set()
        self.last_action = None
        self.current_position = None

    def sense_and_act(self, percept, position):
        self.current_position = tuple(position)
        self.visited_cells.add(self.current_position)

        # Rule 1: food_here -> suck
        if percept["food_here"]:
            action = "Suck"

        else:
            left_pos = tuple(percept["left_position"])
            right_pos = tuple(percept["right_position"])
            ahead_pos = tuple(percept["ahead_position"])

            left_is_visited = left_pos in self.visited_cells
            right_is_visited = right_pos in self.visited_cells
            ahead_is_visited = ahead_pos in self.visited_cells

            # Rule 2:
            # IF wall_ahead AND left_is_visited
            # THEN turn_right
            if (
                percept["wall_ahead"]
                and left_is_visited
                and not percept["right_blocked"]
            ):
                action = "TurnRight"

            # If ahead is already visited and the left path
            # is available and less familiar, turn left.
            elif (
                not percept["wall_ahead"]
                and ahead_is_visited
                and not percept["left_blocked"]
                and not left_is_visited
            ):
                action = "TurnLeft"

            # If ahead is blocked, choose an available alternative.
            elif percept["wall_ahead"]:
                if not percept["right_blocked"]:
                    action = "TurnRight"
                elif not percept["left_blocked"]:
                    action = "TurnLeft"
                else:
                    action = "TurnLeft"

            # Prefer unexplored forward paths.
            elif not ahead_is_visited:
                action = "Forward"

            # If forward has been visited, try another direction.
            elif not percept["left_blocked"] and not left_is_visited:
                action = "TurnLeft"

            elif not percept["right_blocked"] and not right_is_visited:
                action = "TurnRight"

            else:
                # All nearby choices are familiar.
                # Turn left to avoid blindly repeating forward.
                action = "TurnLeft"

        self.last_action = action
        return action


class GridGameGUI:
    def __init__(
        self,
        root,
        width=12,
        height=12,
        num_food=15,
        num_opponents=0,
        walls=None
    ):
        self.root = root
        self.root.title(
            "IT3012 - Practical 02: Model-Based Agent"
        )

        self.env = VisualGridHuntGame(
            width=width,
            height=height,
            num_food=num_food,
            num_opponents=num_opponents,
            custom_walls=walls
        )

        self.agent = ModelBasedAgent()

        max_canvas_dim = 600

        self.cell_size = max(
            20,
            min(
                max_canvas_dim // self.env.width,
                max_canvas_dim // self.env.height
            )
        )

        self.canvas = tk.Canvas(
            root,
            width=self.env.width * self.cell_size,
            height=self.env.height * self.cell_size,
            bg="white"
        )
        self.canvas.pack()

        self.label = tk.Label(
            root,
            text=(
                "Score: 0 | Steps: 0 | "
                "Direction: Right | Memory: 0"
            ),
            font=("Arial", 14)
        )
        self.label.pack(pady=10)

        self.btn = tk.Button(
            root,
            text="Start Simulation",
            command=self.run_loop,
            font=("Arial", 12)
        )
        self.btn.pack(pady=5)

        self.draw_grid()

    def draw_grid(self):
        self.canvas.delete("all")

        # Grid and walls.
        for x in range(self.env.width):
            for y in range(self.env.height):
                x1 = x * self.cell_size
                y1 = (
                    self.env.height - 1 - y
                ) * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size

                is_wall = (x, y) in self.env.walls

                self.canvas.create_rectangle(
                    x1,
                    y1,
                    x2,
                    y2,
                    fill="#64748b" if is_wall else "#f1f5f9",
                    outline="#cbd5e1"
                )

                if is_wall:
                    self.canvas.create_text(
                        x1 + self.cell_size / 2,
                        y1 + self.cell_size / 2,
                        text="W",
                        fill="white",
                        font=("Arial", 8, "bold")
                    )

        # Food.
        for fx, fy in self.env.food_positions:
            offset = self.cell_size * 0.25
            x1 = fx * self.cell_size + offset
            y1 = (
                self.env.height - 1 - fy
            ) * self.cell_size + offset

            self.canvas.create_oval(
                x1,
                y1,
                x1 + self.cell_size * 0.5,
                y1 + self.cell_size * 0.5,
                fill="#f59e0b",
                outline="#d97706"
            )

        # Toxic traps.
        for tx, ty in self.env.toxic_traps:
            offset = self.cell_size * 0.25
            x1 = tx * self.cell_size + offset
            y1 = (
                self.env.height - 1 - ty
            ) * self.cell_size + offset

            self.canvas.create_oval(
                x1,
                y1,
                x1 + self.cell_size * 0.5,
                y1 + self.cell_size * 0.5,
                fill="purple",
                outline="purple"
            )

        # Opponents.
        for ox, oy in self.env.opponents:
            offset = self.cell_size * 0.2
            x1 = ox * self.cell_size + offset
            y1 = (
                self.env.height - 1 - oy
            ) * self.cell_size + offset

            self.canvas.create_rectangle(
                x1,
                y1,
                x1 + self.cell_size * 0.6,
                y1 + self.cell_size * 0.6,
                fill="#990000",
                outline="#7a0000"
            )

        # Agent.
        ax, ay = self.env.agent_pos
        offset = self.cell_size * 0.15

        x1 = ax * self.cell_size + offset
        y1 = (
            self.env.height - 1 - ay
        ) * self.cell_size + offset

        self.canvas.create_oval(
            x1,
            y1,
            x1 + self.cell_size * 0.7,
            y1 + self.cell_size * 0.7,
            fill="#000066",
            outline="#1e3a8a"
        )

    def run_loop(self):
        self.btn.config(state="disabled")

        def step():
            if not self.env.is_done():
                percept = self.env.get_percept()

                action = self.agent.sense_and_act(
                    percept,
                    self.env.agent_pos
                )

                self.env.execute_action(action)
                self.draw_grid()

                self.label.config(
                    text=(
                        f"Score: {self.env.score} | "
                        f"Steps: {self.env.steps} | "
                        f"Action: {action} | "
                        f"Direction: {self.env.direction} | "
                        f"Memory: "
                        f"{len(self.agent.visited_cells)}"
                    )
                )

                self.root.after(250, step)

            else:
                self.label.config(
                    text=(
                        f"Finished! Final Score: "
                        f"{self.env.score} | "
                        f"Steps: {self.env.steps} | "
                        f"Memory: "
                        f"{len(self.agent.visited_cells)}"
                    )
                )

                self.btn.config(state="normal")

        step()


if __name__ == "__main__":
    root = tk.Tk()

    app = GridGameGUI(
        root,
        width=12,
        height=12,
        num_food=15,
        num_opponents=0
    )

    root.mainloop()