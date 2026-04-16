# wordsearch_gen_solve_trie_viz.py
# Heaps (priority queue) for generation + Trie+DFS for solving, HV only.
# Full-trie visualization (orange = prefix, green = found), DFS Stack panel,
# animated heap deletion on the actual heap, canonical words, early stop.
# Buttons: Step, Play/Pause, Mode, Restart, Undo. Close the window to exit.
# Requires: matplotlib

import heapq, random, string, math, sys, copy
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button

# -------------------------
# Config
# -------------------------
GRID_H, GRID_W = 12, 12
WORDS = ["HEAP", "SENSORY", "ALLIGATOR", "ARTIFICIAL", "QUEUE"]
INSERT_WORD = "SUSTAINABLE"                # insertion word
ALLOW_REVERSE = True
RNG_SEED = 42
EMPTY = "."
DIRS_HV = [(0, 1), (1, 0), (0, -1), (-1, 0)]   # E,S,W,N
JITTER = 0.001                                  # tiny tiebreaker for equal overlap
JUMP_AFTER_FOUND = True                         # keep False to show DFS backtracking
# -------------------------
# Small helpers
# -------------------------
def empty_grid(h, w): return [[EMPTY for _ in range(w)] for _ in range(h)]
def in_bounds(r, c): return 0 <= r < GRID_H and 0 <= c < GRID_W
def canon(word: str) -> str: return min(word, word[::-1])

# -------------------------
# Grid ops
# Algorithm Engine
# -------------------------
def fits_and_overlap(grid, word, r, c, dr, dc):
    overlap, cells = 0, []
    for i, ch in enumerate(word):
        rr, cc = r + dr*i, c + dc*i
        if not in_bounds(rr, cc): return None
        g = grid[rr][cc]
        if g != EMPTY and g != ch: return None
        if g == ch: overlap += 1
        cells.append((rr, cc))
    return overlap, cells

def place(grid, cells, word):
    for i, (r, c) in enumerate(cells):
        if grid[r][c] == EMPTY:
            grid[r][c] = word[i]

def fill_random(grid):
    letters = string.ascii_uppercase
    for r in range(GRID_H):
        for c in range(GRID_W):
            if grid[r][c] == EMPTY:
                grid[r][c] = random.choice(letters)

# -------------------------
# Algorithm Engine
# Heap-guided generation (priority queue)
# -------------------------
def best_placement(grid, word):
    best = None
    variants = [word]
    if ALLOW_REVERSE and len(word) > 1:
        variants.append(word[::-1])
    for v in variants:
        for dr, dc in [(0, 1), (1, 0)]:  # HV only
            for r in range(GRID_H):
                for c in range(GRID_W):
                    res = fits_and_overlap(grid, v, r, c, dr, dc)
                    if res is None: continue
                    ov, cells = res
                    sc = ov + random.random() * JITTER
                    if best is None or sc > best[0]:
                        best = (sc, ov, cells, (dr, dc), v)
    return best

def _sift_down_minheap(H, i, n, steps):
    """Sift-down for a min-heap; records snapshots for compare + swap phases."""
    while True:
        l = 2*i + 1
        r = 2*i + 2
        if l >= n:
            # no children
            break

        # pick the better child (smaller key = higher priority)
        smallest = l
        if r < n and H[r][0] < H[l][0]:
            smallest = r

        # --- comparison snapshot (decision phase) ---
        steps.append((f"compare parent={H[i][1]} vs child={H[smallest][1]}",
                      list(H), i, smallest))

        # if child should rise, swap
        if H[smallest][0] < H[i][0]:
            H[i], H[smallest] = H[smallest], H[i]
            # --- swap snapshot (action phase) ---
            steps.append((f"sift-down swap {H[smallest][1]} <-> {H[i][1]}", list(H), smallest, None))
            i = smallest
        else:
            # optional: show that no swap is needed after comparison
            steps.append((f"stops as maxheap property holds at node{i}", list(H), i, smallest))
            break


def push_with_snapshots(heap_arr, item):
    """
    Demo insertion on a COPY of heap_arr.
    Returns: steps, final_heap
      steps = [(label, snapshot, active_index, secondary_index)]
    """
    H = list(heap_arr)  # ✅ copy
    steps = []

    steps.append(("initiate maxheap", list(H), None, None))

    H.append(item)
    i = len(H) - 1
    steps.append((f"append new word ({H[i][1]}) at index {i}", list(H), i, None))

    while i > 0:
        p = (i - 1) // 2
        steps.append((
            f"compare parent={H[p][1]} vs child= {H[i][1]}",
            list(H), i, p
        ))

        if H[i][0] < H[p][0]:
            H[i], H[p] = H[p], H[i]
            steps.append((f"sift-up swap {H[i][1]} <-> {H[p][1]}", list(H), p, None))
            i = p
        else:
            steps.append(("stops as maxheap property holds at node {i}", list(H), i, p))
            break

    steps.append(("insertion complete (maxheap satisfied)", list(H), i, None))
    return steps, H


def heapify_with_snapshots(arr):
    """
    Manual heapify that mimics heapq.heapify, but records snapshots.
    Assumes arr is a list of tuples (key, item) where smaller key = higher priority (min-heap).
    """
    H = list(arr)
    steps = [("start with an unsorted array", list(H), None, None)]
    n = len(H)
    for i in range(n//2 - 1, -1, -1):
        steps.append((f"sift-down from node {i}({H[i][1]})", list(H), i, None))
        _sift_down_minheap(H, i, n, steps)
    steps.append(("heapify complete (maxheap satisfied)", list(H), None, None))
    return H, steps

def pop_with_snapshots(heap_arr):
    """
    Demo delete-max on a COPY of heap_arr (min-heap storing (-len, word)).
    Returns: popped_word, steps, final_heap
    """
    H = list(heap_arr)  # ✅ copy
    steps = []
    if not H:
        return None, [("maxheap empty", [], None, None)], []

    steps.append(("Initiate maxheap", list(H), 0, None))

    last = len(H) - 1
    H[0], H[last] = H[last], H[0]
    steps.append((f"swap root {H[last][1]} with last node {H[0][1]}", list(H), 0, last))

    popped = H.pop()
    steps.append((f"popped last node {popped[1]}", list(H), 0, None))

    i = 0
    n = len(H)
    while True:
        l = 2*i + 1
        r = 2*i + 2
        if l >= n:
            steps.append((f"stop as no children at node={i}", list(H), i, None))
            break

        smallest = l
        if r < n and H[r][0] < H[l][0]:
            smallest = r

        steps.append((f"compare parent={H[i][1]} vs child={H[smallest][1]}",
                      list(H), i, smallest))

        if H[smallest][0] < H[i][0]:
            H[i], H[smallest] = H[smallest], H[i]
            steps.append((f"sift-down swap {H[i][1]} <-> {H[smallest][1]}", list(H), smallest, None))
            i = smallest
        else:
            steps.append((f"stop as maxheap property holds at node={i}", list(H), i, smallest))
            break

    steps.append(("maxheap restored", list(H), None, None))
    return popped[1], steps, H


class HeapGenerator:
    # Algorithm Engine: emits stepwise generation states for teaching.
    """
    build_heap -> pop_word -> show_delete (animated on tree) ->
    choose_placement -> commit -> (repeat) -> fill -> done
    """
    def __init__(self, words):
        self.insert_word = INSERT_WORD
        self.heap_insert_demo = []
        self.words = list(words)
        self.grid = empty_grid(GRID_H, GRID_W)
        self.phase = "show_pq_array"
        self.heap_build_demo = []
        self.heap_arr = []           # ACTUAL heap array of (-len, word)
        self.current = None          # (word, best)
        self.remaining = list(words) # just the *names* left to place
        self.heap_delete_demo = []   # list of (label, array) for animation
        self.heap_snapshot = None    # array to draw on tree during show_delete
        self.heap_after_insert = None
        self.heap_after_delete = None

    def clone(self):
        cp = HeapGenerator(self.words)
        cp.grid = copy.deepcopy(self.grid)
        cp.phase = self.phase
        cp.heap_arr = copy.deepcopy(self.heap_arr)
        cp.current = copy.deepcopy(self.current)
        cp.remaining = list(self.remaining)
        cp.heap_delete_demo = copy.deepcopy(self.heap_delete_demo)
        cp.heap_snapshot = copy.deepcopy(self.heap_snapshot)
        cp.heap_build_demo = copy.deepcopy(self.heap_build_demo)
        cp.insert_word = self.insert_word
        cp.heap_insert_demo = copy.deepcopy(self.heap_insert_demo)
        cp.heap_after_insert = copy.deepcopy(self.heap_after_insert)
        cp.heap_after_delete = copy.deepcopy(self.heap_after_delete)

        return cp

    def step(self):

        if self.phase == "show_pq_array":
            self.heap_arr = [(-len(w), w) for w in self.words]  # negative length for max-heap behavior
            heapified, steps = heapify_with_snapshots(self.heap_arr)
            self.heap_build_demo = steps[:]
            self.heap_arr = heapified[:]
            self.phase = "show_heapify"
            label, arr, active, secondary = self.heap_build_demo.pop(0)
            return {
                "state": "maxheapify_demo",
                "pq_msg": f"Build MaxHeap: {label}",
                "heap_msg": "",
                "grid": self.grid,
                "remaining": self.remaining,
                "heap_array": arr,
                "heap_view": "array",
                "heap_array_view": arr,
                "active_index": active,
                "secondary_index": secondary,
            }

        if self.phase == "show_heapify":
            # keep showing heapify snapshots until finished
            if self.heap_build_demo:
                label, arr, active, secondary = self.heap_build_demo.pop(0)
                return {
                    "state": "maxheapify_demo",
                    "grid": self.grid,
                    "remaining": self.remaining,
                    "pq_msg": f"Build MaxHeap: {label}",
                    "heap_msg": "",
                    "heap_array": arr,
                    "heap_view": "array",
                    "heap_array_view": arr,
                    "active_index": active,
                    "secondary_index": secondary,
                }
            self.phase = "show_insert"

            w_ins = self.insert_word
            if w_ins not in self.remaining:
                self.remaining.append(w_ins)
            
            item = (-len(w_ins), w_ins)
            steps, final_heap = push_with_snapshots(self.heap_arr, item)
            self.heap_insert_demo = steps
            self.heap_after_insert = final_heap   # store pending commit

            label, arr, active, secondary = self.heap_insert_demo.pop(0)
            return {
                "state": "insert_demo",
                "grid": self.grid,
                "remaining": self.remaining,
                "pq_msg": f"Insertion: {label}",
                "heap_msg": "",
                "heap_array": arr,
                "heap_view": "tree",
                "heap_array_view": arr,
                "active_index": active,
                "secondary_index": secondary,
            }
            
        if self.phase == "show_insert":
            if not self.heap_insert_demo:
                if self.heap_after_insert is not None:
                    self.heap_arr = self.heap_after_insert
                    self.heap_after_insert = None
                self.phase = "pop_word"
                return {
                    "state": "insert_done",
                    "grid": self.grid,
                    "remaining": self.remaining,
                    "pq_msg": "Begin Deletion (extract longest word for placement)",
                    "heap_msg": "",
                    "heap_array": list(self.heap_arr),
                    "heap_view": "tree",
                    "heap_array_view": list(self.heap_arr),
                    "active_index": None,
                    "secondary_index": None,
                }
            
            label, arr, active, secondary = self.heap_insert_demo.pop(0)
            return {
                "state": "insert_demo",
                "grid": self.grid,
                "remaining": self.remaining,
                "pq_msg": f"Insertion: {label}",
                "heap_msg": "",
                "heap_array": arr,
                "heap_view": "tree",
                "heap_array_view": arr,
                "active_index": active,
                "secondary_index": secondary,
            }

        if self.phase == "pop_word":
            if not self.heap_arr:
                self.phase = "fill"
                return {
                    "state": "pop_word",
                    "grid": self.grid,
                    "remaining": self.remaining,
                    "pq_msg": "MaxHeap empty → fill board with random letters",
                    "heap_msg": "",
                    "heap_view": "tree",
                    "heap_array": [],
                    "heap_array_view": [],
                }

            w, steps, final_heap = pop_with_snapshots(self.heap_arr)  # mutates heap_arr to AFTER pop
            self.current = (w, None)
            self.heap_delete_demo = steps[:]
            self.heap_after_delete = final_heap
            self.phase = "show_delete"

            label, arr, active, secondary = self.heap_delete_demo.pop(0)
            return {
                "state": "heap_demo",
                "grid": self.grid,
                "remaining": self.remaining,
                "pq_msg": f"Deletion: {label}",
                "heap_msg": "",
                "heap_view": "tree",
                "heap_array": arr,
                "heap_array_view": arr,
                "active_index": active,
                "secondary_index": secondary,
            }

        if self.phase == "show_delete":
            if not self.heap_delete_demo:
                if self.heap_after_delete is not None:
                    self.heap_arr = self.heap_after_delete
                    self.heap_after_delete = None

                self.phase = "choose_placement"
                return {
                    "state": "heap_demo_done",
                    "grid": self.grid,
                    "remaining": self.remaining,
                    "pq_msg": f"Deletion complete — next place '{self.current[0]}' on grid",
                    "heap_msg": "",
                    "heap_view": "tree",
                    "heap_array": list(self.heap_arr),
                    "heap_array_view": list(self.heap_arr),
                    "active_index": None,
                    "secondary_index": None,
                }

            label, arr, active, secondary = self.heap_delete_demo.pop(0)
            return {
                "state": "heap_demo",
                "grid": self.grid,
                "remaining": self.remaining,
                "pq_msg": f"Deletion: {label}",
                "heap_msg": "",
                "heap_view": "tree",
                "heap_array": arr,
                "heap_array_view": arr,
                "active_index": active,
                "secondary_index": secondary,
            }

        if self.phase == "choose_placement":
            w, _ = self.current
            best = best_placement(self.grid, w)
            self.current = (w, best)
            self.heap_snapshot = None
            if best is None:
                self.phase = "commit"
                return {
                    "state": "choose_placement",
                    "grid": self.grid,
                    "remaining": self.remaining,
                    "highlight": None,
                    "pq_msg": f"No valid position for '{w}'",
                    "heap_msg": "",
                    "heap_array": list(self.heap_arr),
                }
            _, ov, cells, _d, _v = best
            self.phase = "commit"
            return {
                "state": "choose_placement",
                "grid": self.grid,
                "remaining": self.remaining,
                "highlight": cells,
                "pq_msg": f"Place word at position with greatest overlap → overlap = {ov}",
                "heap_msg": "",
                "heap_array": list(self.heap_arr),
            }

        if self.phase == "commit":
            w, best = self.current
            if w in self.remaining: self.remaining.remove(w)
            if best is not None:
                _, ov, cells, _d, v = best
                place(self.grid, cells, v)
                note = f"Placed '{w}' (overlap={ov})"
            else:
                note = f"Skipped '{w}'"
            self.current = None
            self.heap_snapshot = None
            self.phase = "pop_word"
            return {
                "state": "commit",
                "grid": self.grid,
                "remaining": self.remaining,
                "pq_msg": note,
                "heap_array": list(self.heap_arr),
                "heap_msg": "",
            }

        if self.phase == "fill":
            fill_random(self.grid)
            self.phase = "done"
            return {
                "state": "fill",
                "grid": self.grid,
                "remaining": self.remaining,
                "pq_msg": "Filled the board with random letters",
                "heap_msg": "",
                "heap_array": list(self.heap_arr),
            }

        if self.phase == "done":
            return {
                "state": "done",
                "grid": self.grid,
                "remaining": self.remaining,
                "pq_msg": "Generation complete. Proceed to change mode to Solve",
                "heap_msg": "",
                "heap_array": list(self.heap_arr),
            }

# -------------------------
# Trie (tree) + DFS solver (global, single pass)
# -------------------------
class TrieNode:
    __slots__ = ("child", "word_end", "end_found", "rem")
    def __init__(self):
        self.child = {}
        self.word_end = False
        self.end_found = False  #whether this exact word endpoint has been found
        self.rem = 0


class Trie:
    def __init__(self):
        self.root = TrieNode()
    def insert(self, w: str):
        node = self.root
        node.rem += 1
        for ch in w:
            node = node.child.setdefault(ch, TrieNode())
            node.rem +=1
        node.word_end = True
    def has_prefix(self, pref):
        node = self.root
        for ch in pref:
            node = node.child.get(ch)
            if node is None:
                return None
        return node
    
    def is_word(self, pref:str) -> bool:
        node = self.has_prefix(pref)
        return (node is not None) and node.word_end
    def mark_found_word(self, w:str) ->bool:
        #mark exact spelling w found (if it exists in trie and hasn't been marked). Decrement rem along the path only once per spelling
        #return true if it successfully marked a new found word-end, else False
        node = self.root
        path = [node]  # store nodes along path so we can decrement rem
        for ch in w:
            node = node.child.get(ch)
            if node is None:
                return False
            path.append(node)

        if not node.word_end or node.end_found:
            return False  # not a word, or already marked found

        node.end_found = True
        for nd in path:
            nd.rem -= 1
        return True
    
def build_trie(words):
    t = Trie()
    seen = set()
    for w in words:
        w = w.upper()
        if w not in seen:
            t.insert(w)
            seen.add(w)
        if ALLOW_REVERSE:
            rw = w[::-1]
            if rw not in seen:
                t.insert(rw)
                seen.add(rw)
    return t

class GlobalDFSSolver:
    # Algorithm Engine: trie-guided DFS solver that yields state snapshots.
    """
    Start from every cell once; DFS with trie pruning.
    Records canonical words; stops early when all canon targets found.
    Emits steps: 'explore' (with dfs_note), 'found_word', 'done_all'
    """
    def __init__(self, grid, words):
        self.grid = grid
        self.targets = set(canon(w) for w in words)
        self.trie = build_trie(words)
        self.vis = [[False]*GRID_W for _ in range(GRID_H)]
        self.starts = [(r, c) for r in range(GRID_H) for c in range(GRID_W)]
        self.stack = []   # (r,c,next_dir)
        self.path = []    # [(r,c), ...]
        self.pref = ""
        self.found = {}   # canon_word -> cells
        self.mode = "seek_start"

    def clone(self):
        cp = GlobalDFSSolver(self.grid, [])
        cp.grid = self.grid
        cp.targets = set(self.targets)
        cp.trie = self.trie
        cp.vis = copy.deepcopy(self.vis)
        cp.starts = list(self.starts)
        cp.stack = copy.deepcopy(self.stack)
        cp.path = list(self.path)
        cp.pref = str(self.pref)
        cp.found = copy.deepcopy(self.found)
        cp.mode = self.mode
        return cp

    def _push(self, r, c):
        self.stack.append((r, c, 0))
        self.path.append((r, c))
        self.pref += self.grid[r][c]
        self.vis[r][c] = True

    def _pop(self):
        r, c = self.path.pop()
        self.pref = self.pref[:-1]
        self.vis[r][c] = False
        self.stack.pop()

    def _done_if_all_found(self):
        if len(self.found) == len(self.targets):
            self.mode = "done"
            return True
        return False
    def step(self):
        # done state
        if self.mode == "done":
            return {"state": "done_all", "found": self.found, "prefix": "", "stack": [], "path": [], "probe": None,
                    "dfs_note": "DONE"}

        # -----------------------
        # pick a new start cell
        # -----------------------
        if self.mode == "seek_start":
            if self._done_if_all_found():
                return {"state": "done_all", "found": self.found, "prefix": "", "stack": [], "path": [], "probe": None,
                        "dfs_note": "DONE (all target words found)"}

            if not self.starts:
                self.mode = "done"
                return {"state": "done_all", "found": self.found, "prefix": "", "stack": [], "path": [], "probe": None,
                        "dfs_note": "DONE (no more start cells)"}

            # reset per-start DFS state
            self.stack.clear()
            self.path.clear()
            self.pref = ""
            self.vis = [[False] * GRID_W for _ in range(GRID_H)]

            r, c = self.starts.pop(0)
            ch = self.grid[r][c]

            # “select start” step (always show probe for clarity)
            node = self.trie.has_prefix(ch)
            if node is not None and node.rem > 0:
                self._push(r, c)
                self.mode = "dfs"
                return {
                    "state": "explore",
                    "path": list(self.path),
                    "prefix": self.pref,
                    "dfs_note": f"SELECT starting letter ({r},{c})='{ch}' → PUSH TO STACK (depth={len(self.path)})",
                    "stack": list(self.stack),
                    "probe": (r, c),
                }
            else:
                return {
                    "state": "explore",
                    "path": [],
                    "prefix": ch,
                    "dfs_note": f"SELECT starting letter ({r},{c})='{ch}' → REJECT (no word starts with '{ch}')",
                    "stack": list(self.stack),
                    "probe": (r, c),
                }

        # -----------------------
        # DFS mode
        # -----------------------
        if self.mode == "dfs":
            # found a word?
            if self.trie.is_word(self.pref):
                cw = canon(self.pref)
                if cw in self.targets and cw not in self.found:
                    self.found[cw] = list(self.path)

                    # ✅ NEW: prune future searches for this word (and its reverse)
                    self.trie.mark_found_word(self.pref)
                    if ALLOW_REVERSE:
                        self.trie.mark_found_word(self.pref[::-1])

                    # optional jump (your existing behaviour)
                    if JUMP_AFTER_FOUND:
                        self.stack.clear()
                        self.path.clear()
                        self.pref = ""
                        self.vis = [[False] * GRID_W for _ in range(GRID_H)]
                        self.mode = "seek_start"

                    if self._done_if_all_found():
                        return {"state": "done_all", "found": self.found, "prefix": "", "stack": [], "path": [], "probe": None,
                                "dfs_note": "DONE (all target words found)"}

                    return {
                        "state": "found_word",
                        "word": cw,
                        "cells": list(self.found[cw]),
                        "stack": list(self.stack),
                        "prefix": self.pref,
                        "path": list(self.path),
                        "probe": None,
                        "dfs_note": f"FOUND WORD '{cw}'",
                    }

            # if stack emptied, go to next start
            if not self.stack:
                self.mode = "seek_start"
                return {
                    "state": "explore",
                    "path": [],
                    "prefix": "",
                    "dfs_note": "FINISHED EXPLORING this cell → move to next cell",
                    "stack": [],
                    "probe": None,
                }

            # expand from top-of-stack
            r, c, next_dir = self.stack[-1]

            for k in range(next_dir, len(DIRS_HV)):
                dr, dc = DIRS_HV[k]
                rr, cc = r + dr, c + dc

                # advance the direction pointer on the stack frame
                self.stack[-1] = (r, c, k + 1)

                # skip invalid / visited
                if not in_bounds(rr, cc) or self.vis[rr][cc]:
                    continue

                ch2 = self.grid[rr][cc]
                candidate = self.pref + ch2

                # ALWAYS show probe when considering a neighbor
                node = self.trie.has_prefix(candidate)
                if node is None or node.rem == 0:
                    return {
                        "state": "explore",
                        "path": list(self.path),
                        "prefix": candidate,
                        "dfs_note": f"CONSIDER ({rr},{cc})='{ch2}' → REJECT (no word starts with '{candidate}')",
                        "stack": list(self.stack),
                        "probe": (rr, cc),
                    }

                # accept and push
                self._push(rr, cc)
                return {
                    "state": "explore",
                    "path": list(self.path),
                    "prefix": self.pref,
                    "dfs_note": f"CONSIDER ({rr},{cc})='{ch2}' → PUSH (depth={len(self.path)})",
                    "stack": list(self.stack),
                    "probe": (rr, cc),
                }

            # no more directions -> backtrack
            popped_r, popped_c = self.path[-1]
            popped_ch = self.grid[popped_r][popped_c]
            self._pop()
            return {
                "state": "explore",
                "path": list(self.path),
                "prefix": self.pref,
                "dfs_note": f"BACKTRACK from ({popped_r},{popped_c})='{popped_ch}' → depth={len(self.path)}",
                "stack": list(self.stack),
                "probe": (popped_r, popped_c),
            }


# -------------------------
# Visualization
# -------------------------
class Viz:
    # Visualisation Module: all matplotlib drawing routines.
    def __init__(self):
        self.fig = plt.figure(figsize=(13.5, 8.2))
        self.fig.canvas.mpl_connect('close_event', lambda e: sys.exit(0))

        self.ax_grid   = self.fig.add_axes([0.05, 0.22, 0.48, 0.74])
        self.ax_trie   = self.fig.add_axes([0.56, 0.45, 0.39, 0.51])
        self.ax_log    = self.fig.add_axes([0.56, 0.18, 0.39, 0.22])

        bx = 0.05; by = 0.05; bw = 0.12; bh = 0.06; gap = 0.02
        self.ax_b_step    = self.fig.add_axes([bx + 0*(bw+gap), by, bw, bh])
        self.ax_b_play    = self.fig.add_axes([bx + 1*(bw+gap), by, bw, bh])
        self.ax_b_mode    = self.fig.add_axes([bx + 2*(bw+gap), by, bw, bh])
        self.ax_b_restart = self.fig.add_axes([bx + 3*(bw+gap), by, bw, bh])
        self.ax_b_undo    = self.fig.add_axes([bx + 4*(bw+gap), by, bw, bh])

        self.btn_step    = Button(self.ax_b_step,    "Step")
        self.btn_play    = Button(self.ax_b_play,    "Play/Pause")
        self.btn_mode    = Button(self.ax_b_mode,    "Mode: Generate")
        self.btn_restart = Button(self.ax_b_restart, "Restart")
        self.btn_undo    = Button(self.ax_b_undo,    "Undo")

        for ax in (self.ax_trie, self.ax_log):
            ax.axis('off')
        self.ax_grid.set_xticks([]); self.ax_grid.set_yticks([])

    def draw_grid(self, grid, title="", path=None, found=None, probe=None):
        ax = self.ax_grid; ax.clear()
        ax.set_xlim(0, GRID_W); ax.set_ylim(0, GRID_H)
        ax.set_xticks([]); ax.set_yticks([])
        for r in range(GRID_H):
            for c in range(GRID_W):
                ax.add_patch(Rectangle((c, GRID_H-1-r), 1, 1, fill=False))
                color = None
                if found and (r, c) in found: color = "black"
                ax.text(c+0.5, GRID_H-1-r+0.5, grid[r][c], ha='center', va='center',
                        fontsize=14, color=color or "black")
        if path:
            for (rr, cc) in path:
                ax.add_patch(Rectangle((cc, GRID_H-1-rr), 1, 1, fill=False, linewidth=3))
        if probe:
            pr, pc = probe
            ax.add_patch(Rectangle((pc, GRID_H-1-pr), 1, 1, fill=False, linewidth=3, edgecolor="red"))
        self.fig.canvas.draw_idle()

    def draw_maxheap_array_blocks(self, heap_array, footnote="", active_index=None, secondary_index=None):
        """
        Top panel: Binary tree view of the current array-as-heap.
        - Orange border = active move / swap target (active_index) when secondary_index is None
        - Grey shading  = comparison phase when secondary_index is not None (shade BOTH active+secondary)
        """
        ax = self.ax_trie
        ax.clear()
        ax.axis('off')
        ax.set_title("MaxHeap — tree view", fontsize=12)

        arr = list(heap_array) if heap_array is not None else []
        if not arr:
            ax.text(0.5, 0.5, "(empty)", ha='center', va='center')
            if footnote:
                ax.text(0.02, 0.04, footnote, transform=ax.transAxes, fontsize=10)
            self.fig.canvas.draw_idle()
            return

        # Convert stored (-len, word) → (len, word) for display
        disp = [(-s, str(w).upper()) for (s, w) in arr]
        n = len(disp)
        depth = max(1, math.ceil(math.log2(n + 1)))

        # layout positions in axes coords
        xs, ys = [], []
        TOP, BOTTOM = 0.92, 0.24
        for i in range(n):
            d = int(math.log2(i + 1))
            pos = i - (2**d - 1)
            level_n = 2**d
            xs.append((pos + 1) / (level_n + 1))
            ys.append(TOP - (TOP - BOTTOM) * (d / depth))

        # edges
        for i in range(1, n):
            p = (i - 1) // 2
            ax.plot([xs[p], xs[i]], [ys[p], ys[i]], color="0.7", linewidth=1.0, zorder=1)

        # state helpers
        in_compare = (secondary_index is not None)
        compare_set = set()
        if in_compare:
            if active_index is not None: compare_set.add(active_index)
            compare_set.add(secondary_index)

        for i, (length, w) in enumerate(disp):
            is_active = (active_index is not None and i == active_index)
            is_secondary = (secondary_index is not None and i == secondary_index)

            # default
            fc, ec, lw = "white", "black", 1.2

            # comparison phase: shade BOTH compared nodes grey
            if in_compare and i in compare_set:
                fc, ec, lw = "0.90", "0.45", 2.0

            # active move: orange border (dominates)
            if (secondary_index is None) and is_active:
                fc, ec, lw = "white", "tab:orange", 3.0

            ax.text(
                xs[i], ys[i],
                f"{w}\n{length}",
                ha='center', va='center',
                fontsize=10, family='monospace',
                bbox=dict(boxstyle="round,pad=0.28", fc=fc, ec=ec, lw=lw),
                zorder=2
            )

        if footnote:
            ax.text(0.02, 0.04, footnote, transform=ax.transAxes, fontsize=10)

        self.fig.canvas.draw_idle()
    def draw_heap_tree(self, remaining_words, footnote="", heap_array=None, active_index = None, secondary_index = None):
        ax = self.ax_trie; ax.clear(); ax.axis('off')
        ax.set_title("MaxHeap — tree view", fontsize=12)
        arr = list(heap_array) if heap_array is not None else [(-len(w), w) for w in remaining_words] or []
        if heap_array is None and arr:
            heapq.heapify(arr)
        if not arr:
            ax.text(0.5, 0.5, "(empty)", ha='center', va='center')
        else:
            disp = [(-s, str(w).upper()) for (s, w) in arr]   # (length, word)
            # --- visual state helpers (comparison vs move) ---
            in_compare = (secondary_index is not None)
            compare_set = set()
            if in_compare:
                if active_index is not None:
                    compare_set.add(active_index)
                compare_set.add(secondary_index)

            n = len(disp); depth = max(1, math.ceil(math.log2(n+1)))
            xs, ys = [], []
            TOP = 0.95
            BOTTOM = 0.25

            for i in range(n):
                d = int(math.log2(i+1))
                pos = i - (2**d - 1)
                level_n = 2**d
                xs.append((pos + 1)/(level_n + 1))
                ys.append(TOP - (TOP - BOTTOM) * (d / depth))

            for i in range(1, n):
                p = (i-1)//2; ax.plot([xs[p], xs[i]], [ys[p], ys[i]], color="0.7")
            for i, (score, w) in enumerate(disp):
                is_active = (active_index is not None and i == active_index)
                is_secondary = (secondary_index is not None and i == secondary_index)

                fc = "white"
                ec = "black"
                lw = 1.0

                # Comparison phase: shade BOTH compared nodes grey
                if in_compare and (i in compare_set):
                    fc = "0.90"
                    ec = "0.45"
                    lw = 2.0

                # Active move/swap phase: orange border for moved node (dominates)
                if (secondary_index is None) and is_active:
                    fc = "white"
                    ec = "tab:orange"
                    lw = 3.0

                ax.text(
                    xs[i], ys[i],
                    f"{w}\n{int(score)}",
                    ha='center', va='center',
                    bbox=dict(boxstyle="round", fc=fc, ec=ec, lw=lw)
                )
        if footnote:
            ax.text(0.02, 0.04, footnote, transform=ax.transAxes, fontsize=10)
        self.fig.canvas.draw_idle()

    def draw_trie_full(self, trie: "Trie", current_prefix: str, found_words: list):
        ax = self.ax_trie; ax.clear(); ax.axis('off')
        ax.set_title("Letters Tree", fontsize=14)

        nodes, edges, pos = [], [], {}
        leaf_counter = [0]

        def collect(node, depth=0, edge_ch=""):
            node_id = id(node)
            nodes.append((node, edge_ch, depth, node_id))
            items = sorted(node.child.items())
            if not items:
                x = leaf_counter[0]; leaf_counter[0] += 1
            else:
                xs = []
                for ch, child in items:
                    child_id = id(child)
                    edges.append((node_id, child_id, ch))
                    xs.append(collect(child, depth+1, ch))
                x = sum(xs)/len(xs)
            pos[node_id] = (x, depth)
            return x

        collect(trie.root, 0, "∅")
        span = max(1, (leaf_counter[0]-1) if leaf_counter[0] else 1)
        max_depth = max(d for (_,_,d,_) in nodes) if nodes else 1
        for _, _, depth, node_id in nodes:
            x_raw, _ = pos[node_id]
            x = 0.05 + 0.9*(x_raw/span)
            y = 0.95 - 0.85*(depth/max(1, max_depth))
            pos[node_id] = (x, y)

        def node_path_for_word(w):
            node = trie.root
            ids = [id(node)]
            for ch in w:
                node = node.child.get(ch)
                if node is None: return []
                ids.append(id(node))
            return ids

        orange_ids = set(node_path_for_word(current_prefix)) if current_prefix else set()
        green_ids = set()
        for w in found_words:
            for nid in node_path_for_word(w): green_ids.add(nid)
            rw = w[::-1]
            for nid in node_path_for_word(rw): green_ids.add(nid)

        for pid, cid, ch in edges:
            x1,y1 = pos[pid]; x2,y2 = pos[cid]
            color="0.7"; lw=1.0
            if pid in green_ids and cid in green_ids: color, lw = "tab:green", 2.5
            if pid in orange_ids and cid in orange_ids: color, lw = "darkorange", 2.5
            ax.plot([x1,x2],[y1,y2], color=color, linewidth=lw)
            # ax.text((x1+x2)/2, (y1+y2)/2+0.02, ch, ha="center", va="bottom", fontsize=9, color="0.3")

        for node, ch, depth, nid in nodes:
            x,y = pos[nid]
            label = "∅" if node is trie.root else ch
            face="white"; lw=1.0
            if nid in green_ids: face, lw = "palegreen", 2.0
            if nid in orange_ids: face, lw = "moccasin", max(lw, 2.0)
            ax.text(x, y, label, ha="center", va="center",
                    bbox=dict(boxstyle="round", fc=face, ec="black", lw=lw))
            if node.word_end:
                ax.text(x, y-0.04, "●", ha="center", va="center", color="tab:green", fontsize=10)

        ax.set_xlim(0,1); ax.set_ylim(0,1)
        self.fig.canvas.draw_idle()

    def draw_dfs_log(self, stack, prefix, dfs_note, prev_note=""):
        """
        Solve log panel (like generation log):
        - prev_note (muted gray): previous action
        - dfs_note (orange): current action
        - show prefix + top-of-stack direction in a compact way
        """
        ax = self.ax_log
        ax.clear()
        ax.axis("off")
        ax.set_title("Solve — DFS log", fontsize=12)

        # show previous note (gray)
        if prev_note:
            ax.text(0.02, 0.20, prev_note, transform=ax.transAxes,
                    fontsize=10, color="0.55")

        # show current note (orange)
        if dfs_note:
            ax.text(0.02, 0.10, dfs_note, transform=ax.transAxes,
                    fontsize=12, color="tab:orange", fontweight="bold")

        # compact status line
        prefix_line = f"prefix = '{prefix}'"
        ax.text(0.02, 0.92, prefix_line, transform=ax.transAxes,
                fontsize=11, family="monospace", va="top")

        # show only the TOP stack frame (most relevant)
        if stack:
            r, c, next_dir = stack[-1]
            DIR_NAMES = ["RIGHT", "DOWN", "LEFT", "UP"]
            nd = DIR_NAMES[next_dir] if 0 <= next_dir < len(DIR_NAMES) else "DONE"
            ax.text(0.02, 0.80, f"top = ({r},{c}) next = {nd}",
                    transform=ax.transAxes, fontsize=11, family="monospace", va="top")
        else:
            ax.text(0.02, 0.80, "top = (empty)",
                    transform=ax.transAxes, fontsize=11, family="monospace", va="top")

        self.fig.canvas.draw_idle()


    def draw_pq_array_in_log(self, heap_array, title="PQ array", footnote="", prev_footnote="", active_index=None, secondary_index=None):
        """
        PQ panel: shows array blocks + TWO message lines:
          - prev_footnote (muted gray) = previous action
          - footnote (bigger + attention color) = current action
        """
        ax = self.ax_log
        ax.clear()
        ax.axis('off')
        ax.set_title(title, fontsize=12)

        arr = list(heap_array) if heap_array is not None else []
        n = len(arr)

        if n == 0:
            ax.text(0.5, 0.55, "(empty)", ha='center', va='center')

        else:
            left, right = 0.02, 0.98
            width = right - left

            # Fill the horizontal space better (wider boxes, less dead space)
            box_w = (width / n) * 0.985
            rect_w = box_w * 0.94

            # Taller boxes to comfortably fit 2-line text
            y0 = 0.26
            h = 0.62

            in_compare = (secondary_index is not None)
            compare_set = set()
            if in_compare:
                if active_index is not None:
                    compare_set.add(active_index)
                compare_set.add(secondary_index)

            for i, (score, w) in enumerate(arr):
                length = -score  # stored (-len, word)
                x0 = left + i * box_w

                # Defaults
                face = "white"
                edge = "black"
                lw = 2

                # Comparison phase (before swap): grey shading for BOTH compared nodes
                if in_compare and i in compare_set:
                    face = "0.90"
                    edge = "0.45"
                    lw = 2.5

                # Active move/swap phase: orange border for the active moved node
                if (secondary_index is None) and (active_index is not None) and (i == active_index):
                    face = "white"
                    edge = "tab:orange"
                    lw = 3.0

                ax.add_patch(Rectangle((x0, y0), rect_w, h, fill=True,
                                       facecolor=face, edgecolor=edge, linewidth=lw))
                ax.text(x0 + rect_w*0.5, y0 + h + 0.08, f"[{i}]",
                        ha='center', va='center', fontsize=9)

                ax.text(x0 + rect_w*0.5, y0 + h*0.55, f"{str(w).upper()}\n{length}",
                        ha='center', va='center', fontsize=9, family='monospace')

        if prev_footnote:
            ax.text(0.02, 0.14, prev_footnote,
                    transform=ax.transAxes, fontsize=10, color="0.55")

        if footnote:
            ax.text(0.02, 0.06, footnote,
                    transform=ax.transAxes, fontsize=12, color="tab:orange",
                    fontweight="bold")

        self.fig.canvas.draw_idle()

# -------------------------
# Controller (state + undo history)
# -------------------------
class Controller:
    # UI / Controller Module: input events, autoplay loop, undo/restart orchestration.
    def __init__(self, fig):
        self.fig = fig
        self.step_flag = False
        self.autoplay = False
        self.delay_ms = 600
        self.restart = False
        self.mode = "generate"
        self.timer = fig.canvas.new_timer(interval=self.delay_ms)
        self.timer.add_callback(self._tick)
        self.history = []   # list of snapshots of *displayed* frames

    def bind(self, viz: Viz):
        viz.btn_step.on_clicked(lambda e: self._step())
        viz.btn_play.on_clicked(lambda e: self._toggle())
        viz.btn_restart.on_clicked(lambda e: self._restart())
        viz.btn_mode.on_clicked(lambda e: self._toggle_mode(viz))
        viz.btn_undo.on_clicked(lambda e: self._undo())

    def _toggle_mode(self, viz):
        self.mode = "solve" if self.mode == "generate" else "generate"
        viz.btn_mode.label.set_text(f"Mode: {self.mode.capitalize()}")

    def _tick(self):
        if self.autoplay:
            self.step_flag = True
            self._arm()

    def _arm(self):
        try: self.timer.stop()
        except Exception: pass
        if self.autoplay:
            self.timer.interval = self.delay_ms
            self.timer.start()

    def _step(self): self.step_flag = True

    def _toggle(self):
        self.autoplay = not self.autoplay
        self._arm()

    def _restart(self):
        self.autoplay = False
        self._arm()
        self.restart = True
        self.history.clear()

    def _undo(self):
        self.autoplay = False
        self._arm()
        if len(self.history) >= 2:
            self.history.pop()  # drop current
            snap = self.history[-1]
            self._restore_snapshot(snap)

    def wait(self):
        self.step_flag = False; self._arm()
        while not (self.step_flag or self.restart):
            plt.pause(0.02)
        self.step_flag = False; self._arm()
        return True

    def set_state_refs(self, refs):
        self.refs = refs

    def _make_snapshot(self):
        gen = self.refs["gen"]
        solver = self.refs["solver"]
        gen_done = self.refs["gen_done"]
        found = self.refs["found"]
        mode = self.mode

        return {
            "mode": mode,
            "gen": gen.clone(),
            "solver": (solver.clone() if solver else None),
            "gen_done": gen_done,
            "found": copy.deepcopy(found),

            #Generation
            "heap_view": self.refs.get("heap_view", "tree"),
            "heap_array_view": copy.deepcopy(self.refs.get("heap_array_view", [])),
            "heap_msg": self.refs.get("heap_msg", ""),
            "pq_msg": self.refs.get("pq_msg", ""),
            "prev_pq_msg": self.refs.get("prev_pq_msg", ""),
            "heap_array_for_trie": copy.deepcopy(self.refs.get("heap_array_for_trie", [])),
            "active_index": self.refs.get("active_index", None),
            "secondary_index": self.refs.get("secondary_index", None),

            #Solving
            "solve_path": copy.deepcopy(self.refs.get("solve_path", None)),
            "solve_probe": copy.deepcopy(self.refs.get("solve_probe", None)),
            "solve_prefix": self.refs.get("solve_prefix", ""),
            "solve_note": self.refs.get("solve_note", ""),
            "prev_solve_note": self.refs.get("prev_solve_note", ""),
        }

    def _restore_snapshot(self, snap):
        self.mode = snap["mode"]
        self.refs["gen"] = snap["gen"]
        self.refs["solver"] = snap["solver"]
        self.refs["gen_done"] = snap["gen_done"]
        self.refs["found"] = snap["found"]

        #generation
        self.refs["heap_view"] = snap.get("heap_view", "tree")
        self.refs["heap_array_view"] = snap.get("heap_array_view", [])
        self.refs["heap_msg"] = snap.get("heap_msg", "")
        self.refs["pq_msg"] = snap.get("pq_msg", "")
        self.refs["prev_pq_msg"] = snap.get("prev_pq_msg", "")
        self.refs["heap_array_for_trie"] = snap.get("heap_array_for_trie", [])
        self.refs["active_index"] = snap.get("active_index", None)
        self.refs["secondary_index"] = snap.get("secondary_index", None)
        
        #solving
        self.refs["solve_path"] = snap.get("solve_path",None)
        self.refs["solve_probe"] = snap.get("solve_probe", None)
        self.refs["solve_prefix"] = snap.get("solve_prefix", "")
        self.refs["solve_note"] = snap.get("solve_note", "")
        self.refs["prev_solve_note"] = snap.get("prev_solve_note", "")

        viz = self.refs["viz"]
        gen = self.refs["gen"]
        solver = self.refs["solver"]
        found_cells = self.refs["found"]

        if self.mode == "generate":
            viz.draw_grid(gen.grid, title="")
            hv = snap.get("heap_view", "tree")

            if hv == "array":
                viz.draw_maxheap_array_blocks(
                    snap.get("heap_array_view", snap.get("heap_array_for_trie", [])),
                    footnote=snap.get("heap_msg",""),
                    active_index=snap.get("active_index", None),
                    secondary_index=snap.get("secondary_index", None),
                )
            else:
                snap_heap = snap.get("heap_array")
                if snap_heap is None:
                    snap_heap = snap.get("heap_array_view", [])
                
                viz.draw_heap_tree(
                    gen.remaining,
                    footnote=snap.get("heap_msg",""),
                    heap_array=list(snap_heap),
                    active_index=snap.get("active_index", None),
                    secondary_index=snap.get("secondary_index", None),
                )

            viz.draw_pq_array_in_log(
                snap.get("heap_array_view", []),
                title="PQ array",
                footnote=snap.get("pq_msg",""),
                prev_footnote=snap.get("prev_pq_msg",""),
                active_index=snap.get("active_index", None),
                secondary_index=snap.get("secondary_index", None),
            )
            return

        # Solve mode restore
        solve_path = snap.get("solve_path", None)
        solve_probe = snap.get("solve_probe", None)
        solve_prefix = snap.get("solve_prefix", "")
        solve_note = snap.get("solve_note", "")
        prev_solve_note = snap.get("prev_solve_note", "")

        # recompute found_strings from solver (safe)
        found_strings = []
        if solver is not None:
            for cw, cells in solver.found.items():
                s = "".join(gen.grid[r][c] for (r, c) in cells)
                found_strings += [s, s[::-1]]

        # redraw grid WITH highlights
        viz.draw_grid(
            gen.grid,
            title="Solve",
            path=solve_path,
            found=found_cells,
            probe=solve_probe
        )

        # redraw trie + dfs log
        if solver is None:
            viz.draw_trie_full(build_trie(WORDS), current_prefix="", found_words=[])
            viz.draw_dfs_log([], "", solve_note, prev_note=prev_solve_note)
        else:
            viz.draw_trie_full(solver.trie, current_prefix=solve_prefix, found_words=found_strings)
            viz.draw_dfs_log(
                stack=list(solver.stack),
                prefix=solve_prefix,
                dfs_note=solve_note,
                prev_note=prev_solve_note
        )
# -------------------------
# Runner
# -------------------------
def run():
    random.seed(RNG_SEED)
    viz = Viz()
    ctl = Controller(viz.fig)
    ctl.bind(viz)

    gen = HeapGenerator(WORDS)
    solver = None
    gen_done = False
    found_cells_total = set()

    # ---------- initial draw ----------
    viz.draw_grid(gen.grid, title="Generate — empty grid")
    viz.draw_heap_tree(gen.remaining, footnote="Click Step to begin (heapify)", heap_array=[])

    initial_arr = [(-len(w), w) for w in WORDS]
    prev_msg = ""
    cur_msg = "Click Step to begin (heapify)"

    viz.draw_pq_array_in_log(
        initial_arr,
        title="PQ array",
        footnote=cur_msg,
        prev_footnote=prev_msg,
        active_index=None,
        secondary_index=None,
    )

    ctl.set_state_refs({
        "gen": gen, "solver": solver, "gen_done": gen_done,
        "found": found_cells_total, "viz": viz,
        "heap_view": "tree",
        "heap_array_view": initial_arr,
        "heap_msg": "Click Step to begin",
        "heap_array_for_trie": [],
        "prev_pq_msg": prev_msg,
        "pq_msg": cur_msg,
    })
    ctl.history.append(ctl._make_snapshot())

    while True:
        ctl.wait()
        if hasattr(ctl, "refs"):
            gen = ctl.refs.get("gen", gen)
            solver = ctl.refs.get("solver", solver)
            gen_done = ctl.refs.get("gen_done", gen_done)
            found_cells_total = ctl.refs.get("found", found_cells_total)
        # ---------- Restart ----------
        if ctl.restart:
            random.seed(RNG_SEED)
            gen = HeapGenerator(WORDS)
            solver = None
            gen_done = False
            found_cells_total = set()

            viz.draw_grid(gen.grid, title="Generate — empty grid")
            viz.draw_heap_tree(gen.remaining, footnote="Click Step to begin (heapify)", heap_array=[])

            initial_arr = [(-len(w), w) for w in WORDS]
            prev_msg = ""
            cur_msg = "Click Step to begin (heapify)"
            viz.draw_pq_array_in_log(initial_arr, title="PQ array", footnote=cur_msg, prev_footnote=prev_msg, active_index=None, secondary_index=None)

            ctl.set_state_refs({
                "gen": gen, "solver": solver, "gen_done": gen_done,
                "found": found_cells_total, "viz": viz,
                "heap_view": "tree",
                "heap_array_view": initial_arr,
                "heap_msg": "Click Step to begin (heapify)",
                "heap_array_for_trie": [],
                "prev_pq_msg": prev_msg,
                "pq_msg": cur_msg,
            })

            ctl.history = [ctl._make_snapshot()]
            ctl.restart = False
            continue

        # ---------- Generate mode ----------
        if ctl.mode == "generate":
            gen = ctl.refs["gen"]   # always step the current restored state
            info = gen.step()

            # previous message is the one that was just on screen
            prev_msg = ctl.refs.get("pq_msg", "") if hasattr(ctl, "refs") else ""

            viz.draw_grid(
                info.get("grid", gen.grid),
                title="",
                path=info.get("highlight"),
                found=None,
                probe=None
            )

            if info.get("heap_view") == "array":
                viz.draw_maxheap_array_blocks(
                    info.get("heap_array_view", info.get("heap_array", [])),
                    footnote=info.get("heap_msg", ""),
                    active_index=info.get("active_index", None),
                    secondary_index=info.get("secondary_index", None),
                )
            else:
                viz.draw_heap_tree(
                    info.get("remaining", []),
                    footnote=info.get("heap_msg", ""),
                    heap_array=info.get("heap_array_view", info.get("heap_array", [])),
                    active_index=info.get("active_index"),
                    secondary_index=info.get("secondary_index")
                )

            viz.draw_pq_array_in_log(
                info.get("heap_array_view", info.get("heap_array", [])),
                title="PQ array",
                footnote=info.get("pq_msg", ""),
                prev_footnote=prev_msg,
                active_index=info.get("active_index"),
                secondary_index=info.get("secondary_index"),
            )

            if info.get("state") == "done":
                gen_done = True

            ctl.set_state_refs({
                "gen": gen, "solver": solver, "gen_done": gen_done,
                "found": found_cells_total, "viz": viz,

                "heap_view": info.get("heap_view", "tree"),
                "heap_array_view": info.get("heap_array_view", info.get("heap_array", [])),
                "heap_array_for_trie": info.get("heap_array", []),
                "heap_msg": info.get("heap_msg", ""),

                "active_index": info.get("active_index", None),
                "secondary_index": info.get("secondary_index", None),

                "prev_pq_msg": prev_msg,
                "pq_msg": info.get("pq_msg", ""),
            })
            ctl.history.append(ctl._make_snapshot())
            continue

        # ---------- Solve mode ----------
        if not gen_done:
            viz.draw_grid(gen.grid, title="Solve — finish generation first (Step in Generate)")
            viz.draw_heap_tree(gen.remaining, footnote="Generation not finished", heap_array=list(gen.heap_arr))
            viz.draw_dfs_log([], "", "Finish generation first", prev_note="")

            ctl.set_state_refs({
                "gen": gen, "solver": solver, "gen_done": gen_done,
                "found": found_cells_total, "viz": viz,

                # for undo restore
                "solve_path": None,
                "solve_probe": None,
                "solve_prefix": "",
                "solve_note": "Finish generation first",
                "prev_solve_note": "",
            })
            ctl.history.append(ctl._make_snapshot())
            continue

        if solver is None:
            solve_words = list(set(WORDS + [INSERT_WORD]))
            solver = GlobalDFSSolver(gen.grid, solve_words)
            found_cells_total = set()

            viz.draw_grid(gen.grid, title="Solve — Letter Tree + DFS")
            viz.draw_trie_full(solver.trie, current_prefix="", found_words=[])
            viz.draw_dfs_log([], "", "Ready to start DFS", prev_note="")

            ctl.set_state_refs({
                "gen": gen, "solver": solver, "gen_done": gen_done,
                "found": found_cells_total, "viz": viz,

                "solve_path": None,
                "solve_probe": None,
                "solve_prefix": "",
                "solve_note": "Ready to start DFS",
                "prev_solve_note": "",
            })
            ctl.history.append(ctl._make_snapshot())
            continue

        solver = ctl.refs["solver"]

        prev_solve_note = ctl.refs.get("solve_note", "")
        info = solver.step()

        # build found_strings for trie coloring
        found_strings = []
        for cw, cells in solver.found.items():
            s = "".join(gen.grid[r][c] for (r, c) in cells)
            found_strings += [s, s[::-1]]

        if info["state"] == "done_all":
            found_cells_total = set(cell for cells in solver.found.values() for cell in cells)
            viz.draw_grid(gen.grid, title="Solve complete", found=found_cells_total)
            viz.draw_trie_full(solver.trie, current_prefix="", found_words=found_strings)
            viz.draw_dfs_log([], "", "DONE", prev_note=prev_solve_note)

            ctl.set_state_refs({
                "gen": gen, "solver": solver, "gen_done": gen_done,
                "found": found_cells_total, "viz": viz,

                "solve_path": None,
                "solve_probe": None,
                "solve_prefix": "",
                "solve_note": "DONE",
                "prev_solve_note": prev_solve_note,
            })
            ctl.history.append(ctl._make_snapshot())
            continue

        if info["state"] == "found_word":
            for cell in info["cells"]:
                found_cells_total.add(cell)

            viz.draw_grid(gen.grid, title=f"Found '{info['word']}'",
                        path=info["cells"], found=found_cells_total, probe=None)
            viz.draw_trie_full(solver.trie, current_prefix=info.get("prefix", ""), found_words=found_strings)
            viz.draw_dfs_log(info.get("stack", []), info.get("prefix", ""), "FOUND WORD", prev_note=prev_solve_note)

            ctl.set_state_refs({
                "gen": gen, "solver": solver, "gen_done": gen_done,
                "found": found_cells_total, "viz": viz,

                "solve_path": info["cells"],
                "solve_probe": None,
                "solve_prefix": info.get("prefix", ""),
                "solve_note": "FOUND WORD",
                "prev_solve_note": prev_solve_note,
            })
            ctl.history.append(ctl._make_snapshot())
            continue

        # explore
        if info["state"] == "explore":
            prefix = info.get("prefix", "")
            note = info.get("dfs_note", "")

            viz.draw_grid(gen.grid, title="Solve",
                        path=info.get("path"), found=found_cells_total, probe=info.get("probe"))
            viz.draw_trie_full(solver.trie, current_prefix=prefix, found_words=found_strings)
            viz.draw_dfs_log(info.get("stack", []), prefix, note, prev_note=prev_solve_note)

            ctl.set_state_refs({
                "gen": gen, "solver": solver, "gen_done": gen_done,
                "found": found_cells_total, "viz": viz,

                "solve_path": info.get("path"),
                "solve_probe": info.get("probe"),
                "solve_prefix": prefix,
                "solve_note": note,
                "prev_solve_note": prev_solve_note,
            })
            ctl.history.append(ctl._make_snapshot())
            continue


if __name__ == "__main__":
    run()
