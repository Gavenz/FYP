def gen_dfs(start, goal):

    stack = [start]                 # stack for DFS exploration
    visited = set()                 # track visited cells
    parent = {start: None}          # record parent for path reconstruction

    while stack:

        current = stack.pop()       # pop top of stack

        if current == goal:
            return parent           # goal reached

        if current in visited:
            continue

        visited.add(current)

        # explore neighbours (Right, Down, Left, Up)
        for neighbour in neighbors(current):

            if neighbour not in visited and passable(neighbour):

                parent[neighbour] = current
                stack.append(neighbour)

def gen_bfs(start, goal):

    queue = [start]
    visited = {start}
    parent = {start: None}

    while queue:                       # while queue is not empty
        current = queue.pop(0)         # dequeue from front

        if current == goal:
            return parent              # goal reached

        # check neighbours (RIGHT, DOWN, LEFT, UP)
        for neighbour in neighbours(current):

            if neighbour not in visited and passable(neighbour):

                visited.add(neighbour)
                parent[neighbour] = current
                queue.append(neighbour)   # enqueue

import heapq

def gen_dijkstra(start, goal):

    # initialise distance and parent dictionaries
    dist = {}
    parent = {}

    # set all cells to infinity initially
    for cell in all_cells():               # e.g. (0,0), (1,0), (2,0), ...
        dist[cell] = float("inf")          # dist = {(0,0):∞, (1,0):∞, (2,0):∞, (3,0):∞, ...}
        parent[cell] = None                # parent = {(0,0):None, (1,0):None, (2,0):None, ...}

    dist[start] = 0                       # dist = {(0,0):0, (1,0):∞, (2,0):∞, (3,0):∞, ...}
    pq = [(0, start)]                     # priority queue storing (cost, cell)
    while pq:                               

        current_cost, current = heapq.heappop(pq) #lowest cumulative cost is dequeued

        if current == goal:
            return parent, dist
        # if current !=goal:
        for neighbour in neighbours(current):
            if passable(neighbour):
                # cost of entering neighbour cell
                new_cost = current_cost + cell_cost(neighbour) #cell_cost can be (1 or 4)
                
                if new_cost < dist[neighbour]:
                    dist[neighbour] = new_cost
                    parent[neighbour] = current
                    # Example update:
                    # dist[(1,0)] = 1
                    # parent[(1,0)] = (0,0)
                    heapq.heappush(pq, (new_cost, neighbour))

def solve(board):
        #-Base Case -
        # find next empty cell in board, if there are no empty cell, return solve(board) = True
        pos = select_empty_cell(board)
        if pos is None:
            return True

        r,c = pos #row, column of selected empty cell
        
        #-Recursive Case-
        # generate possible candidate digits for the empty cell by checking constraints
        cands = candidates_for(board, r, c)

        
        for d in cands:

            # place the candidate digit on the board
            board[r][c] = d

            # recursively attempt to solve the rest
            if solve(board):
                return True
            else: #solve(board) returns False, backtracks 
                board[r][c] = 0
        
        return False

def solve(board):
    # stack to store previous decisions:
    # entry: (row, col, allowed_candidates, next_candidate_index)
    decision_stack = []                         

    while count_empty_cells(board) > 0:
        
        # select next empty cell
        pos = select_empty_cell(board)
        r,c = pos
        # generate possible candidates
        cands = candidates_for(board, r, c)
        #index of next candidate to try
        next_cand_index = 0

        while True:
            # all candidates have been tried then backtrack
            if next_cand_index == len(cands):
                
                #if backtrack all the way to the first decision and is empty now, means puzzle is unsolvable
                if not decision_stack:
                    return False

                # restore previous decision state
                r, c, cands, next_cand_index = decision_stack.pop()

                # remove previously placed digit if any
                board[r][c] = 0
                continue

            # select the next candidate digit    
            cand = cands[next_cand_index]
            next_idx += 1

            # Place candidate digit
            board[r][c] = cand

            # Save current decision state for backtracking
            decision_stack.append((r, c, cands, next_cand_index))

            # Move forward to next empty cell
            break

    # No empty cells remain → puzzle solved
    return True