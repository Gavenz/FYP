# Utility functions for maze generation and visualization
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

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

    for x, y in maze.visited:
        ax.add_patch(plt.Rectangle((x, y), 1, 1, facecolor="gray", alpha=0.1))

    for x, y in maze.stack:
        ax.add_patch(plt.Rectangle((x, y), 1, 1, facecolor="yellow", alpha=0.3))

    debug_text = f"Stack: {maze.stack}\nVisited: {sorted(maze.visited)}"
    ax.text(-0.5, -1, debug_text, fontsize=8, family="monospace", verticalalignment="top")

def visualize_maze_with_buttons(maze):
    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title("Recursive Backtracking")
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

    ax_next = plt.axes([0.8, 0.05, 0.075, 0.060])
    btn_next = Button(ax_next, "Next")
    btn_next.on_clicked(on_next)

    ax_restart = plt.axes([0.87, 0.05, 0.075, 0.060])
    btn_restart = Button(ax_restart, "Restart")
    btn_restart.on_clicked(on_restart)

    plt.show()