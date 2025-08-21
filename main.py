# main.py
from maze_core.utils import MazeVisualizer
from maze_core.constants import DIRECTIONS, OPPOSITE
from maze_generation.gen_recursive_backtrack import RecursiveBacktrackMazeGenerator 
from maze_solvers.solve_recursive_backtrack import RecursiveBacktrackMazeSolver

WIDTH, HEIGHT = 5, 5

# 1. Create dictionaries that map a display name to the algorithm class
generator_algorithms = {
    "Recursive Backtrack": RecursiveBacktrackMazeGenerator,
}

solver_algorithms = {
    "Recursive Solver": RecursiveBacktrackMazeSolver,
}

# 2. Pass these dictionaries when you create the MazeVisualizer
visualizer = MazeVisualizer(
    WIDTH, 
    HEIGHT, 
    generator_classes=generator_algorithms, 
    solver_classes=solver_algorithms
)

# 3. Show the visualization
visualizer.show()