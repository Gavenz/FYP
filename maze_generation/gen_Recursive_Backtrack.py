import random
# Recursive Backtracking Maze Generation Algorithm with a Stack

DIRECTIONS = {
    "N": (0, 1),
    "S": (0, -1),
    "E": (1, 0),
    "W": (-1, 0)
}
# Dictionary for opposite directions for wall removal between the cells.
OPPOSITE = {
    "N": "S",
    "S": "N",
    "E": "W",
    "W": "E"
}

class Maze:
    # initialize the maze with given width and height and set walls to True
    def __init__(self, width, height):
        self.width = width
        self.height = height
        # Each cell has 4 walls: True means wall is present
        self.grid = {
            (x, y): {"N": True, "S": True, "E": True, "W": True}
            for x in range(width) for y in range(height)
        }

    def get_unvisited_neighbors(self, x, y, visited):
        neighbors = []
        for direction, (dx, dy) in DIRECTIONS.items():
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                if (nx, ny) not in visited:
                    neighbors.append(((nx, ny), direction))
        return neighbors

    def generate(self, start_x=0, start_y=0):
        stack = [(start_x, start_y)]
        visited = set()
        visited.add((start_x, start_y))

        while stack:
            current = stack[-1] #pop the last element in the stack
            x, y = current

            neighbors = self.get_unvisited_neighbors(x, y, visited) # find the available neighbors and the direction that leads to them
            if neighbors:
                # Choose a random unvisited neighbor
                (nx, ny), direction = random.choice(neighbors)

                # Remove wall between current and neighbor
                self.grid[(x, y)][direction] = False
                self.grid[(nx, ny)][OPPOSITE[direction]] = False

                # Mark neighbor as visited and push to stack
                visited.add((nx, ny))
                stack.append((nx, ny))
            else:
                # Dead end → backtrack
                stack.pop()

    def display(self, start_x=0, start_y=0): #prints the maze with ASCII characters
        output = ""
        for y in reversed(range(self.height)):
            top = ""
            mid = ""
            for x in range(self.width):
                top += "+---" if self.grid[(x, y)]["N"] else "+   "
                if (x, y) == (start_x, start_y):
                    mid += "| S " if self.grid[(x, y)]["W"] else " S  "
                else:
                    mid += "|   " if self.grid[(x, y)]["W"] else "    "
            output += top + "+\n"
            output += mid + "|\n"
        output += "+---" * self.width + "+\n"
        print(output)


if __name__ == "__main__":
    m = Maze(4, 4)
    m.generate()
    m.display(0,0)
