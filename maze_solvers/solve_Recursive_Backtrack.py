import random
from maze_core.constants import DIRECTIONS, OPPOSITE


class RecursiveBacktrackMazeSolver:
    def __init__(self, maze):
        """Initializes the solver with a pre-generated maze object."""
        self.maze = maze
        self.reset()

    def reset(self):
        self.stack = [(0, 0)]
        self.visited = set([(0, 0)])
        self.generator = self.solve_step_by_step()

    def get_valid_neighbors(self, x, y):
        """
        Finds all reachable, unvisited neighbors from a given cell.

        A neighbor is valid if there is no wall between it and the current cell.
        """
        neighbors = []
        for direction, (dx, dy) in DIRECTIONS.items():
            nx, ny = x + dx, y + dy

            # Check if the neighbor is within the maze boundaries.
            if 0 <= nx < self.maze.width and 0 <= ny < self.maze.height:

                # Only move if there is NO wall in the direction to the neighbor
                # and the neighbor has not been visited yet.
                if not self.maze.grid[(x, y)][direction] and (nx, ny) not in self.visited:
                    neighbors.append((nx, ny))
        return neighbors

    def solve_step_by_step(self):
        goal = (self.maze.width - 1, self.maze.height - 1)
        while self.stack:
            current = self.stack[-1]
            if current == goal:
                break
            neighbors = self.get_valid_neighbors(*current)
            if neighbors:
                # If there are valid neighbors, pick one random to explore next
                next_cell = random.choice(neighbors)
                self.stack.append(next_cell)
                self.visited.add(next_cell)
            else:
                # If there are no valid moves, hit dead end.
                # Backtrack by removing current cell from the path.
                self.stack.pop()
            yield

    def step(self):
        """Advances the solver by one step."""
        next(self.generator)

    def get_draw_data(self):
        """Returns data for visualization."""
        solution_path = set(self.stack)
        return {
             "walls": self.maze.grid,
            "visited": self.visited,   
            "solution": solution_path,
            # --- ADD THIS DEBUG INFO ---
            "current_cell": self.stack[-1] if self.stack else "Done",
            "stack_size": list(self.stack),
            "visited_count": list(self.visited),
        }
