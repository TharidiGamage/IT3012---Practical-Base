# agent.py
import random
import heapq
from collections import deque
import math


class GreedyGridAgent:
    """A simple agent that tries to move around systematically to clear the grid."""

    def __init__(self):
        self.actions_pool = ["Up", "Down", "Left", "Right"]

    def sense_and_act(self, percept: dict) -> str:
        return random.choice(self.actions_pool)


class SearchAgent:
    """
    Goal-Based/Search Agent for IT3012 Practical 03.

    The agent receives the global grid state and creates an offline plan
    to the closest food using BFS, DFS, or UCS.
    """

    def __init__(self):
        self.plan = []
        self.active_algo = "AStar"

    def get_neighbors(self, position, walls, grid_size):
        """
        Return valid neighboring states as:
            (action, next_position)

        Each movement has a cost of 1, so BFS and UCS produce an
        optimal path in this environment.
        """
        x, y = position
        width, height = grid_size

        candidates = [
            ("Up", (x, y + 1)),
            ("Right", (x + 1, y)),
            ("Down", (x, y - 1)),
            ("Left", (x - 1, y)),
        ]

        neighbors = []

        for action, next_position in candidates:
            nx, ny = next_position

            if (
                0 <= nx < width
                and 0 <= ny < height
                and next_position not in walls
            ):
                neighbors.append((action, next_position))

        return neighbors

    def bfs_search(self, start, goal, walls, grid_size):
        """Breadth-First Search using a FIFO queue."""
        frontier = deque()
        frontier.append((start, []))

        reached = {start}

        while frontier:
            current, path = frontier.popleft()

            if current == goal:
                return path

            for action, next_state in self.get_neighbors(
                current, walls, grid_size
            ):
                if next_state not in reached:
                    reached.add(next_state)
                    frontier.append(
                        (next_state, path + [action])
                    )

        return []

    def dfs_search(self, start, goal, walls, grid_size):
        """Depth-First Search using a LIFO stack."""
        frontier = [(start, [])]
        reached = {start}

        while frontier:
            current, path = frontier.pop()

            if current == goal:
                return path

            for action, next_state in self.get_neighbors(
                current, walls, grid_size
            ):
                if next_state not in reached:
                    reached.add(next_state)
                    frontier.append(
                        (next_state, path + [action])
                    )

        return []

    def ucs_search(self, start, goal, walls, grid_size):
        """Uniform-Cost Search using a priority queue ordered by g(n)."""
        frontier = []
        counter = 0

        # (total_cost, tie_breaker, state, path)
        heapq.heappush(frontier, (0, counter, start, []))

        reached = {start: 0}

        while frontier:
            cost, _, current, path = heapq.heappop(frontier)

            if current == goal:
                return path

            # Ignore an outdated priority-queue entry.
            if cost > reached.get(current, float("inf")):
                continue

            for action, next_state in self.get_neighbors(
                current, walls, grid_size
            ):
                new_cost = cost + 1

                if (
                    next_state not in reached
                    or new_cost < reached[next_state]
                ):
                    reached[next_state] = new_cost
                    counter += 1

                    heapq.heappush(
                        frontier,
                        (
                            new_cost,
                            counter,
                            next_state,
                            path + [action],
                        )
                    )

        return []

    def _find_closest_food(self, start, food_positions):
        """Select the food with the smallest Manhattan distance."""
        return min(
            food_positions,
            key=lambda food: (
                abs(start[0] - food[0])
                + abs(start[1] - food[1])
            )
        )

    def _make_plan(self, percept):
        """Create a new plan using the currently selected algorithm."""
        start = tuple(percept["agent_pos"])
        food_positions = [
            tuple(food) for food in percept["all_food"]
        ]

        if not food_positions:
            return []

        goal = self._find_closest_food(start, food_positions)

        walls = {
            tuple(wall) for wall in percept["walls"]
        }
        grid_size = tuple(percept["grid_size"])

        if self.active_algo == "BFS":
            return self.bfs_search(
                start, goal, walls, grid_size
            )

        if self.active_algo == "DFS":
            return self.dfs_search(
                start, goal, walls, grid_size
            )

        if self.active_algo == "UCS":
            return self.ucs_search(
                start, goal, walls, grid_size
            )

        if self.active_algo == "AStar":
            return self.astar_search(
                start, goal, walls, grid_size,
                heuristic_type="manhattan"
            )

        raise ValueError(
            f"Unknown search algorithm: {self.active_algo}"
        )

    def sense_and_act(self, percept: dict) -> str:
        """
        If there is no current plan, create one.

        The plan contains direct grid movement actions. The environment
        executes those actions one at a time.
        """
        if percept["food_here"]:
            self.plan = []
            return "Suck"

        if not self.plan:
            self.plan = self._make_plan(percept)

        if self.plan:
            return self.plan.pop(0)

        # This should only occur if no reachable food remains.
        return "TurnLeft"

    def manhattan_distance(self, pos, goal):
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    def euclidean_distance(self, pos, goal):
        return math.sqrt((pos[0] - goal[0]) ** 2 + (pos[1] - goal[1]) ** 2)

    def astar_search(
    self,
    start_pos,
    goal_pos,
    walls,
    grid_size,
    heuristic_type="manhattan"
    ):

        frontier = []
        reached_states = set()

        # Calculate heuristic for the starting position
        if heuristic_type == "manhattan":
            h_cost = self.manhattan_distance(start_pos, goal_pos)
        else:
            h_cost = self.euclidean_distance(start_pos, goal_pos)

        # (f_cost, g_cost, current_pos, path_taken)
        heapq.heappush(
            frontier,
            (h_cost, 0, start_pos, [])
        )

        while frontier:
            f_cost, g_cost, current_pos, path_taken = heapq.heappop(
                frontier
            )

            if current_pos == goal_pos:
                return path_taken

            if current_pos in reached_states:
                continue

            reached_states.add(current_pos)

            for action, next_pos in self.get_neighbors(
                current_pos,
                walls,
                grid_size
            ):
                if next_pos in reached_states:
                    continue

                new_g_cost = g_cost + 1

                if heuristic_type == "manhattan":
                    new_h_cost = self.manhattan_distance(
                        next_pos,
                        goal_pos
                    )
                else:
                    new_h_cost = self.euclidean_distance(
                        next_pos,
                        goal_pos
                    )

                new_f_cost = new_g_cost + new_h_cost

                heapq.heappush(
                    frontier,
                (
                    new_f_cost,
                    new_g_cost,
                    next_pos,
                    path_taken + [action]
                )
            )

        return [] 