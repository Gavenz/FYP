import random
import matplotlib.pyplot as plt
from maze_core.constants import DIRECTIONS, OPPOSITE


class RecursiveBacktrackMazeGenerator:
    """Generates a maze using depth-first search (DFS)"""
    
    def __init__(self, width, height):
        """Initialize the maze with given width and height"""
        self.width = width
        self.height = height
        # Set up the maze grid and initial state
        self.reset()

    def reset(self):
        """Reset the maze to its initial state (grid with all walls)"""
        self.grid = {
            (x, y): {"N": True, "S": True, "E": True, "W": True}
            for x in range(self.width) for y in range(self.height)
        }

        # The stack holds the current path of the DFS traversal
        self.stack = [(0, 0)]
        
        # The visited set prevents cycles and ensures all cells is visited once
        self.visited = set([(0, 0)])

        # Generator object that produces the maze step by step
        self.generator = self.generate_step_by_step()

    def get_unvisited_neighbors(self, x, y, visited):
        neighbors = []
        # Check all four directions for unvisited neighbors
        for direction, (dx, dy) in DIRECTIONS.items():
            nx, ny = x + dx, y + dy

            # Ensure the neighbor is within bounds and not visited
            if 0 <= nx < self.width and 0 <= ny < self.height:
                # Check if the neighbor is unvisited
                if (nx, ny) not in visited:
                    neighbors.append(((nx, ny), direction))
        return neighbors
    
    def step(self):
        next(self.generator)
        

    def generate_step_by_step(self):
        """
        A generator function that yields control after each step of the maze creation.
        
        This allows for visualizing the algorithm's progress. It runs as long
        as there are cells to process in the stack.
        """

        # Loop until the stack is empty, which means we have backtracked to the start.
        while self.stack:
            # Get the current cell from the top of the stack without removing it
            current = self.stack[-1]
            x, y = current

            # --- ADD THIS PRINT ---
            print(f"\nProcessing cell: {current}")

            # Get unvisited neighbors of the current cell
            neighbors = self.get_unvisited_neighbors(x, y, self.visited)
            
            # If there are unvisited neighbors, choose one randomly
            if neighbors:
                # --- ADD THIS PRINT ---
                print(f"Found neighbors: {[n[0] for n in neighbors]}")
                (nx, ny), direction = random.choice(neighbors)
                
                # --- ADD THIS PRINT ---
                print(f"--> Moving {direction} to ({nx}, {ny})")

                (nx, ny), direction = random.choice(neighbors)

                # Remove the wall from the current cell in the direction of the chosen neighbor
                self.grid[(x, y)][direction] = False
                # Remove the wall from the neighbor cell in the opposite direction
                self.grid[(nx, ny)][OPPOSITE[direction]] = False
                
                # Add the new cell to the visited set.
                self.visited.add((nx, ny))
                # Push the new cell onto the stack to continue the path.
                self.stack.append((nx, ny))
            else:
                # If there are no unvisited neighbors, hit a dead end.
                # Backtrack by popping the current cell from the stack.
                print("--> Dead end. Backtracking.")
                self.stack.pop()

            yield  # yield after every step

    def get_draw_data(self):
        """Return the data needed for drawing the maze."""
        return {
            "walls": self.grid,
            "visited": self.visited,
            "stack": self.stack,
            "is_done": not self.stack, #should be True because stack is empty
            #ADDED DEBUG TO SEE IF ALGORITHM WORKS CORRECTLY
            "current_cell": self.stack[-1] if self.stack else "Done",
            "stack_size": list(self.stack),
            "visited_count": list(self.visited),
        }
