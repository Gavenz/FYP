import random
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from maze_core.constants import DIRECTIONS, OPPOSITE
DIRECTIONS = {
    "N": (0, 1),
    "S": (0, -1),
    "E": (1, 0),
    "W": (-1, 0),
}
OPPOSITE = {
    "N": "S",
    "S": "N",
    "E": "W",
    "W": "E",
}


class RecursiveBacktrackMaze:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.reset()

    def reset(self):
        self.grid = {
            (x, y): {"N": True, "S": True, "E": True, "W": True}
            for x in range(self.width) for y in range(self.height)
        }
        self.stack = [(0, 0)]
        self.visited = set([(0, 0)])
        self.generator = self.generate_step_by_step()

    def get_unvisited_neighbors(self, x, y, visited):
        neighbors = []
        for direction, (dx, dy) in DIRECTIONS.items():
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                if (nx, ny) not in visited:
                    neighbors.append(((nx, ny), direction))
        return neighbors

    def generate_step_by_step(self):
        while self.stack:
            current = self.stack[-1]
            x, y = current

            neighbors = self.get_unvisited_neighbors(x, y, self.visited)
            if neighbors:
                (nx, ny), direction = random.choice(neighbors)
                self.grid[(x, y)][direction] = False
                self.grid[(nx, ny)][OPPOSITE[direction]] = False
                self.stack.append((nx, ny))
            else:
                self.stack.pop()

            yield  # yield after every step


def draw_maze(ax, maze):
    ax.clear()
    ax.set_aspect("equal")
    ax.axis("off")

    grid = maze.grid
    width = maze.width
    height = maze.height

    for x in range(width):
        for y in range(height):
            cell = grid[(x, y)]
            if cell["N"]:
                ax.plot([x, x + 1], [y + 1, y + 1], color="black")
            if cell["S"]:
                ax.plot([x, x + 1], [y, y], color="black")
            if cell["W"]:
                ax.plot([x, x], [y, y + 1], color="black")
            if cell["E"]:
                ax.plot([x + 1, x + 1], [y, y + 1], color="black")

    ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor="green", alpha=0.3))

    # Highlight visited
    for x, y in maze.visited:
        ax.add_patch(plt.Rectangle((x, y), 1, 1, facecolor="gray", alpha=0.1))

    # Highlight stack
    for x, y in maze.stack:
        ax.add_patch(plt.Rectangle((x, y), 1, 1, facecolor="yellow", alpha=0.3))

    # Text box
    debug_text = f"Stack: {maze.stack}\nVisited: {sorted(maze.visited)}"
    ax.text(-0.5, -1, debug_text, fontsize=8, family="monospace", verticalalignment="top")


def visualize_maze_with_buttons(maze):
    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title("Recursive Backtracking")  # Set window title
    plt.subplots_adjust(bottom=0.25)

    draw_maze(ax, maze)

    def on_next(event):
        try:
            next(maze.generator)
            draw_maze(ax, maze)
            plt.draw()
        except StopIteration:
            ax.set_title("Maze Complete!")
            plt.draw()

    def on_restart(event):
        maze.reset()
        draw_maze(ax, maze)
        ax.set_title("")
        plt.draw()

    # Buttons
    ax_next = plt.axes([0.8, 0.05, 0.075, 0.060])
    btn_next = Button(ax_next, "Next")
    btn_next.on_clicked(on_next)

    ax_restart = plt.axes([0.87, 0.05, 0.075, 0.060])
    btn_restart = Button(ax_restart, "Restart")
    btn_restart.on_clicked(on_restart)

    plt.show()


if __name__ == "__main__":
    m = RecursiveBacktrackMaze(4, 4)
    visualize_maze_with_buttons(m)
