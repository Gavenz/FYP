# utils.py
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons
from matplotlib.collections import LineCollection
from types import SimpleNamespace
import textwrap

class MazeVisualizer:
    """
    A generic class to visualize maze generation and solving algorithms
    that use a dictionary with (x, y) tuple keys for the grid.
    """

    def __init__(self, width, height, generator_classes, solver_classes):
        self.width = width
        self.height = height
        self.generator_classes = generator_classes
        self.solver_classes = solver_classes
        
        self.generator_names = list(generator_classes.keys())
        self.solver_names = list(solver_classes.keys())

        self.selected_generator_class = list(generator_classes.values())[0]
        self.selected_solver_class = list(solver_classes.values())[0]

        self.maze_data = None 
        self.algorithm = None 
        
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        plt.subplots_adjust(bottom=0.2, left=0.25)
        self.ax.set_aspect("equal")
        self.ax.axis("off")
        # --- ADD THIS LINE ---
        self.ax.set_facecolor('white') # Explicitly set background to white
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(0, self.height)
        self.ax.invert_yaxis()
        self.fig.canvas.manager.set_window_title("Maze Algorithms")
        
        self._init_artists()
        self._init_widgets()
        
        self.switch_to_generator()

    def _init_artists(self):
        """Create visual elements once to be updated later for performance."""
        # CHANGED: Use a dictionary for patches instead of a list of lists.
        # The keys will be (x, y) tuples, matching the maze grid structure.
        self.cell_patches = {} 
        for x in range(self.width):
            for y in range(self.height):
                patch = plt.Rectangle((x, y), 1, 1, facecolor="white", alpha=0.3)
                self.cell_patches[(x, y)] = patch
                self.ax.add_patch(patch)
        
        self.wall_collection = LineCollection([], color="black", linewidths=2)
        self.ax.add_collection(self.wall_collection)

        # --- ADD THIS ---
        # Create a single text object for debug info.
        # Position it just below the maze grid.
        self.debug_text = self.ax.text(
            0, -1, "", fontsize=9, family="monospace", va="top"
        )

    def _init_widgets(self):
        """Create buttons and other UI controls."""
        # This section remains unchanged.
        self.ax_next = plt.axes([0.7, 0.05, 0.15, 0.075])
        self.btn_next = Button(self.ax_next, "Next Step")
        self.btn_next.on_clicked(self.on_next)

        self.ax_gen_algo = plt.axes([0.05, 0.7, 0.18, 0.15])
        self.radio_gen = RadioButtons(self.ax_gen_algo, self.generator_names)
        self.radio_gen.on_clicked(self.on_select_generator)

        self.ax_solve_algo = plt.axes([0.05, 0.5, 0.18, 0.15])
        self.radio_solve = RadioButtons(self.ax_solve_algo, self.solver_names)
        self.radio_solve.on_clicked(self.on_select_solver)
        
        self.ax_mode = plt.axes([0.05, 0.3, 0.18, 0.1])
        self.radio_mode = RadioButtons(self.ax_mode, ("Generate", "Solve"))
        self.radio_mode.on_clicked(self.on_mode_change)

    def on_select_generator(self, label):
        self.selected_generator_class = self.generator_classes[label]
        if self.radio_mode.value_selected == "Generate":
            self.switch_to_generator()

    def on_select_solver(self, label):
        self.selected_solver_class = self.solver_classes[label]
        if self.radio_mode.value_selected == "Solve":
            self.switch_to_solver()

    def on_mode_change(self, label):
        if label == "Generate":
            self.switch_to_generator()
        elif label == "Solve":
            self.switch_to_solver()
    
    def on_next(self, event):
        print("--- 'Next Step' button clicked! ---")
        if not self.algorithm: 
            # --- ADD THIS PRINT ---
            print("No active algorithm. (This is normal if it just finished.")
            return
        try:
            self.algorithm.step()
        except StopIteration:
            self.ax.set_title("Algorithm Complete!", color="green")
            if self.radio_mode.value_selected == "Generate":
                self.maze_data = self.algorithm.grid
            self.algorithm = None
        self.draw()

    def switch_to_generator(self):
        self.ax.set_title("Mode: Generate")
        self.algorithm = self.selected_generator_class(self.width, self.height)
        self.maze_data = None 
        self.draw()

    def switch_to_solver(self):
        if self.maze_data is None:
            self.ax.set_title("Generate a maze first!", color="red")
            self.radio_mode.set_active(0)
            self.fig.canvas.draw_idle()
            return
        self.ax.set_title("Mode: Solve")
        maze_object = SimpleNamespace(grid=self.maze_data, width=self.width, height=self.height)
        self.algorithm = self.selected_solver_class(maze_object)
        self.draw()

    def draw(self):
        if not self.algorithm: 
            self.fig.canvas.draw_idle()
            return

        data = self.algorithm.get_draw_data()
         # --- ADD THIS PRINT ---
        # Print the data being used for drawing.
        print(f"Drawing with Stack Size: {data.get('stack_size')}, Visited Count: {data.get('visited_count')}")

        walls, visited, stack, solution = (
            data.get("walls"), data.get("visited", set()),  
            data.get("stack", []), data.get("solution", [])
        )

        for x in range(self.width):
            for y in range(self.height):
                # CHANGED: Access patches using (x, y) tuple keys.
                patch = self.cell_patches[(x, y)]
                if (x, y) in solution:
                    patch.set_facecolor("red")
                elif (x, y) in stack:
                    patch.set_facecolor("yellow")
                elif (x, y) in visited:
                    patch.set_facecolor("lightblue")
                else:
                    patch.set_facecolor("white")
        
        visible_walls = []
        if walls:
            for x in range(self.width):
                for y in range(self.height):
                    # CHANGED: Access walls using (x, y) tuple keys.
                    if walls[(x, y)]["N"]:
                        visible_walls.append([(x, y + 1), (x + 1, y + 1)])
                    if walls[(x, y)]["E"]:
                        visible_walls.append([(x + 1, y), (x + 1, y + 1)])
            visible_walls.extend([((x, 0), (x + 1, 0)) for x in range(self.width)])
            visible_walls.extend([((0, y), (0, y + 1)) for y in range(self.height)])

        self.wall_collection.set_segments(visible_walls)
    
        
        # --- ADD THIS ---
        # Update the debug text on every draw call.
        debug_string = (
            f"Mode: {self.radio_mode.value_selected}\n"
            f"Current Cell:  {data.get('current_cell', 'N/A')}\n"
            f"Stack Size:    {data.get('stack_size', 'N/A')}\n"
            f"Visited Cells: {data.get('visited_count', 'N/A')}\n"
        )
        self.debug_text.set_text(debug_string)
        self.fig.canvas.draw_idle()
    
    

    def show(self):
        plt.show()