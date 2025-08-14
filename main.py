from maze_core.utils import visualize_maze_with_buttons
from maze_generation.recursive_matplot import RecursiveBacktrackMaze

if __name__ == "__main__":
    m = RecursiveBacktrackMaze(4, 4)
    visualize_maze_with_buttons(m)
