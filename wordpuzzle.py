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
WORDS = ["PYTHON", "HEAP", "TRIE", "SEARCH", "QUEUE"]
ALLOW_REVERSE = True
RNG_SEED = 42
EMPTY = "."
DIRS_HV = [(0, 1), (1, 0), (0, -1), (-1, 0)]   # E,S,W,N
JITTER = 0.001                                  # tiny tiebreaker for equal overlap
JUMP_AFTER_FOUND = False                        # keep False to show DFS backtracking

# -------------------------
# Small helpers
# -------------------------
def empty_grid(h, w): return [[EMPTY for _ in range(w)] for _ in range(h)]
def in_bounds(r, c): return 0 <= r < GRID_H and 0 <= c < GRID_W
def canon(word: str) -> str: return min(word, word[::-1])

# -------------------------
# Grid ops
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

def heapify_min(arr):
    heapq.heapify(arr)

def pop_with_snapshots(minheap):
    """
    Teaching delete-max on a MIN-heap that stores (-len, word).
    Operates on `minheap` IN PLACE, returning:
      popped_word, steps
    where steps = [(label, array_snapshot)]
    """
    H = minheap
    steps = []
    if not H:
        return None, [("heap empty", [])]

    steps.append(("heap start", list(H)))

    # swap root with last
    last = len(H) - 1
    H[0], H[last] = H[last], H[0]
    steps.append(("swap root with last", list(H)))

    # remove last
    popped = H.pop()
    steps.append((f"remove last → popped {popped[1]} (len={-popped[0]})", list(H)))

    # sift-down from root
    i, n = 0, len(H)
    while True:
        l = 2*i + 1
        r = 2*i + 2
        smallest = i
        if l < n and H[l][0] < H[smallest][0]: smallest = l
        if r < n and H[r][0] < H[smallest][0]: smallest = r
        if smallest == i: break
        H[i], H[smallest] = H[smallest], H[i]
        i = smallest
        steps.append(("sift-down step", list(H)))

    steps.append(("heap restored", list(H)))
    return popped[1], steps

class HeapGenerator:
    """
    build_heap -> pop_word -> show_delete (animated on tree) ->
    choose_placement -> commit -> (repeat) -> fill -> done
    """
    def __init__(self, words):
        self.words = list(words)
        self.grid = empty_grid(GRID_H, GRID_W)
        self.phase = "build_heap"
        self.heap_arr = []           # ACTUAL heap array of (-len, word)
        self.current = None          # (word, best)
        self.remaining = list(words) # just the *names* left to place
        self.heap_delete_demo = []   # list of (label, array) for animation
        self.heap_snapshot = None    # array to draw on tree during show_delete

    def clone(self):
        cp = HeapGenerator(self.words)
        cp.grid = copy.deepcopy(self.grid)
        cp.phase = self.phase
        cp.heap_arr = copy.deepcopy(self.heap_arr)
        cp.current = copy.deepcopy(self.current)
        cp.remaining = list(self.remaining)
        cp.heap_delete_demo = copy.deepcopy(self.heap_delete_demo)
        cp.heap_snapshot = copy.deepcopy(self.heap_snapshot)
        return cp

    def step(self):
        if self.phase == "build_heap":
            self.heap_arr = [(-len(w), w) for w in self.words]
            heapify_min(self.heap_arr)
            self.phase = "pop_word"
            self.heap_snapshot = None
            return {"state":"build_heap", "grid":self.grid, "remaining":self.remaining,
                    "msg":"Step 1: built max-heap by word length (root = longest)",
                    "heap_array": list(self.heap_arr)}

        if self.phase == "pop_word":
            if not self.heap_arr:
                self.phase = "fill"
                self.heap_snapshot = None
                return {"state":"pop_word", "grid":self.grid, "remaining":self.remaining,
                        "msg":"Heap empty → fill random letters", "heap_array": []}

            w, steps = pop_with_snapshots(self.heap_arr)   # updates self.heap_arr in place
            self.current = (w, None)
            self.heap_delete_demo = steps[:]
            self.phase = "show_delete"
            return {"state":"pop_word", "grid":self.grid, "remaining":self.remaining,
                    "msg":f"Step 2: pop root → '{w}' (highest priority)", "heap_array": list(self.heap_arr)}

        if self.phase == "show_delete":
            if not self.heap_delete_demo:
                self.heap_snapshot = None
                self.phase = "choose_placement"
            else:
                label, arr = self.heap_delete_demo.pop(0)
                self.heap_snapshot = arr[:]  # draw this exact array snapshot
                return {"state":"heap_demo", "grid":self.grid, "remaining":self.remaining,
                        "msg":f"Step 2a: {label}", "heap_array": self.heap_snapshot}
            # fallthrough next step

        if self.phase == "choose_placement":
            w, _ = self.current
            best = best_placement(self.grid, w)
            self.current = (w, best)
            self.heap_snapshot = None
            if best is None:
                self.phase = "commit"
                return {"state":"choose_placement", "grid":self.grid, "remaining":self.remaining,
                        "highlight":None, "msg":f"Step 3: evaluate placements → none valid for '{w}'",
                        "heap_array": list(self.heap_arr)}
            _, ov, cells, _d, _v = best
            self.phase = "commit"
            return {"state":"choose_placement", "grid":self.grid, "remaining":self.remaining,
                    "highlight":cells, "msg":f"Step 3: evaluate placements → choose max overlap = {ov}",
                    "heap_array": list(self.heap_arr)}

        if self.phase == "commit":
            w, best = self.current
            if w in self.remaining: self.remaining.remove(w)
            if best is not None:
                _, ov, cells, _d, v = best
                place(self.grid, cells, v)
                note = f"Step 4: placed '{w}' (overlap={ov})"
            else:
                note = f"Step 4: skipped '{w}'"
            self.current = None
            self.heap_snapshot = None
            self.phase = "pop_word"
            return {"state":"commit", "grid":self.grid, "remaining":self.remaining,
                    "msg":note, "heap_array": list(self.heap_arr)}

        if self.phase == "fill":
            fill_random(self.grid)
            self.phase = "done"
            return {"state":"fill", "grid":self.grid, "remaining":self.remaining,
                    "msg":"Step 5: filled remaining cells with random letters",
                    "heap_array": list(self.heap_arr)}

        if self.phase == "done":
            return {"state":"done", "grid":self.grid, "remaining":self.remaining,
                    "msg":"Generation complete", "heap_array": list(self.heap_arr)}

# -------------------------
# Trie (tree) + DFS solver (global, single pass)
# -------------------------
class TrieNode:
    __slots__ = ("child", "word_end")
    def __init__(self):
        self.child = {}
        self.word_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()
    def insert(self, w):
        node = self.root
        for ch in w:
            node = node.child.setdefault(ch, TrieNode())
        node.word_end = True
    def has_prefix(self, pref):
        node = self.root
        for ch in pref:
            node = node.child.get(ch)
            if node is None: 
                return None
        return node
    def is_word(self, pref):
        node = self.has_prefix(pref)
        return (node is not None) and node.word_end

def build_trie(words):
    t = Trie()
    for w in words:
        t.insert(w)
        if ALLOW_REVERSE: t.insert(w[::-1])  # include reverse explicitly
    return t

class GlobalDFSSolver:
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
        self.path = []    # [(r,c), ...] current path
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
        if self.mode == "done":
            return {"state":"done_all", "found": self.found, "prefix": "", "stack": []}

        if self.mode == "seek_start":
            if self._done_if_all_found():
                return {"state":"done_all", "found": self.found, "prefix": "", "stack": []}
            if not self.starts:
                self.mode = "done"
                return {"state":"done_all", "found": self.found, "prefix": "", "stack": []}

            self.stack.clear(); self.path.clear(); self.pref = ""
            self.vis = [[False]*GRID_W for _ in range(GRID_H)]
            r, c = self.starts.pop(0)
            ch = self.grid[r][c]
            if self.trie.has_prefix(ch):
                self._push(r, c); self.mode = "dfs"
                return {"state":"explore", "path": list(self.path), "prefix": self.pref,
                        "dfs_note": f"PUSH → ({r},{c}) depth={len(self.path)}",
                        "stack": list(self.stack)}
            else:
                # invalid start; show the probed cell as well
                return {"state":"explore", "path": [], "prefix": ch, "note":"invalid start",
                        "dfs_note":"PRUNE start", "stack": list(self.stack),
                        "probe": (r, c)}

        if self.mode == "dfs":
            if self.trie.is_word(self.pref):
                cw = canon(self.pref)
                if cw in self.targets and cw not in self.found:
                    self.found[cw] = list(self.path)
                    if JUMP_AFTER_FOUND:
                        self.stack.clear(); self.path.clear(); self.pref = ""
                        self.vis = [[False]*GRID_W for _ in range(GRID_H)]
                        self.mode = "seek_start"
                    if self._done_if_all_found():
                        return {"state":"done_all", "found": self.found, "prefix": self.pref, "stack": list(self.stack)}
                    return {"state":"found_word", "word": cw, "cells": list(self.found[cw]),
                            "stack": list(self.stack), "prefix": self.pref}

            if not self.stack:
                self.mode = "seek_start"
                return {"state":"explore", "path": [], "prefix": self.pref,
                        "dfs_note": "finished start", "stack": list(self.stack)}

            r, c, next_dir = self.stack[-1]
            for k in range(next_dir, len(DIRS_HV)):
                dr, dc = DIRS_HV[k]
                rr, cc = r + dr, c + dc
                self.stack[-1] = (r, c, k + 1)
                if not (in_bounds(rr, cc) and not self.vis[rr][cc]):
                    continue
                ch2 = self.grid[rr][cc]
                if self.trie.has_prefix(self.pref + ch2) is None:
                    # show the probed neighbor even if pruned
                    return {"state":"explore", "path": list(self.path), "prefix": self.pref + ch2,
                            "note":"pruned", "dfs_note": f"PRUNE at ({rr},{cc})",
                            "stack": list(self.stack), "probe": (rr, cc)}
                self._push(rr, cc)
                return {"state":"explore", "path": list(self.path), "prefix": self.pref,
                        "dfs_note": f"PUSH → ({rr},{cc}) depth={len(self.path)}",
                        "stack": list(self.stack)}

            # no more neighbors → backtrack
            self._pop()
            return {"state":"explore", "path": list(self.path), "prefix": self.pref,
                    "note":"backtrack", "dfs_note": f"BACKTRACK → depth={len(self.path)}",
                    "stack": list(self.stack)}

# -------------------------
# Visualization
# -------------------------
class Viz:
    def __init__(self):
        self.fig = plt.figure(figsize=(13.5, 8.2))
        self.fig.canvas.mpl_connect('close_event', lambda e: sys.exit(0))

        # Layout: grid (left), trie/heap (top-right), dfs-log (bottom-right)
        self.ax_grid   = self.fig.add_axes([0.05, 0.22, 0.48, 0.74])
        self.ax_trie   = self.fig.add_axes([0.56, 0.45, 0.39, 0.51])
        self.ax_log    = self.fig.add_axes([0.56, 0.18, 0.39, 0.22])

        # Buttons row
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
        self.btn_undo    = Button(self.ax_b_undo,   "Undo")

        # Right panels: no axes
        for ax in (self.ax_trie, self.ax_log):
            ax.axis('off')
        self.ax_grid.set_xticks([]); self.ax_grid.set_yticks([])

    def draw_grid(self, grid, title="", path=None, found=None, probe=None):
        ax = self.ax_grid; ax.clear()
        ax.set_title(title, fontsize=12)
        ax.set_xlim(0, GRID_W); ax.set_ylim(0, GRID_H)
        ax.set_xticks([]); ax.set_yticks([])
        for r in range(GRID_H):
            for c in range(GRID_W):
                ax.add_patch(Rectangle((c, GRID_H-1-r), 1, 1, fill=False))
                color = None
                if found and (r, c) in found: color = "black"
                ax.text(c+0.5, GRID_H-1-r+0.5, grid[r][c], ha='center', va='center',
                        fontsize=12, color=color or "black")
        if path:
            for (rr, cc) in path:
                ax.add_patch(Rectangle((cc, GRID_H-1-rr), 1, 1, fill=False, linewidth=3))
        if probe:
            pr, pc = probe
            ax.add_patch(Rectangle((pc, GRID_H-1-pr), 1, 1, fill=False, linewidth=3, edgecolor="red"))
        self.fig.canvas.draw_idle()

    def draw_heap_tree(self, remaining_words, footnote="", heap_array=None):
        ax = self.ax_trie; ax.clear(); ax.axis('off')
        ax.set_title("Heap — tree view", fontsize=12)
        arr = list(heap_array) if heap_array is not None else [(-len(w), w) for w in remaining_words] or []
        if heap_array is None and arr:
            heapq.heapify(arr)
        if not arr:
            ax.text(0.5, 0.5, "(empty)", ha='center', va='center')
        else:
            disp = [(-s, w) for (s, w) in arr]   # (length, word)
            n = len(disp); depth = max(1, math.ceil(math.log2(n+1)))
            xs, ys = [], []
            for i in range(n):
                d = int(math.log2(i+1))
                pos = i - (2**d - 1); level_n = 2**d
                xs.append((pos + 1)/(level_n + 1)); ys.append(1.0 - d/depth)
            for i in range(1, n):
                p = (i-1)//2; ax.plot([xs[p], xs[i]], [ys[p], ys[i]], color="0.7")
            for i, (score, w) in enumerate(disp):
                ax.text(xs[i], ys[i], f"{w}\n(len={int(score)})", ha='center', va='center',
                        bbox=dict(boxstyle="round", fc="white"))
        if footnote:
            ax.text(0.02, 0.04, footnote, transform=ax.transAxes, fontsize=10)
        self.fig.canvas.draw_idle()

    def draw_trie_full(self, trie: "Trie", current_prefix: str, found_words: list):
        ax = self.ax_trie; ax.clear(); ax.axis('off')
        ax.set_title("Trie — full tree (orange = prefix, green = found)", fontsize=12)

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
            ax.text((x1+x2)/2, (y1+y2)/2+0.02, ch, ha="center", va="bottom", fontsize=9, color="0.3")

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

    def draw_dfs_log(self, stack, prefix, dfs_note):
        ax = self.ax_log; ax.clear(); ax.axis('off')
        ax.set_title("DFS stack (top → bottom)", fontsize=12)
        lines = []
        if dfs_note: lines.append(dfs_note)
        lines.append(f"prefix = '{prefix}'")
        if stack:
            lines.append("---")
            for (r,c,next_dir) in stack[::-1]:
                lines.append(f"({r},{c}) next_dir={next_dir}")
        else:
            lines.append("(empty)")
        ax.text(0.02, 0.9, "\n".join(lines[:18]), va="top", family="monospace", fontsize=10)
        self.fig.canvas.draw_idle()

# -------------------------
# Controller (state + undo history)
# -------------------------
class Controller:
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
        # Undo restores the previous *displayed* frame:
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

    # These are assigned later by run() so we can redraw and hold state
    def set_state_refs(self, refs):
        self.refs = refs  # dict with keys: gen, solver, gen_done, found_cells, viz

    def _make_snapshot(self):
        gen, solver, gen_done, found, mode = self.refs["gen"], self.refs["solver"], self.refs["gen_done"], self.refs["found"], self.mode
        return {
            "mode": mode,
            "gen": gen.clone(),
            "solver": (solver.clone() if solver else None),
            "gen_done": gen_done,
            "found": copy.deepcopy(found),
        }

    def _restore_snapshot(self, snap):
        self.mode = snap["mode"]
        self.refs["gen"] = snap["gen"]
        self.refs["solver"] = snap["solver"]
        self.refs["gen_done"] = snap["gen_done"]
        self.refs["found"] = snap["found"]
        # Redraw appropriate frame
        viz = self.refs["viz"]
        gen = self.refs["gen"]; solver = self.refs["solver"]; found_cells = self.refs["found"]
        if self.mode == "generate":
            viz.draw_grid(gen.grid, title="Generate")
            viz.draw_heap_tree(gen.remaining, footnote="", heap_array=list(gen.heap_arr))
            viz.draw_dfs_log([], "", "")
        else:
            viz.draw_grid(gen.grid, title="Solve", found=found_cells)
            if solver:
                found_strings = []
                for cw, cells in solver.found.items():
                    s = "".join(gen.grid[r][c] for (r,c) in cells)
                    found_strings += [s, s[::-1]]
                viz.draw_trie_full(solver.trie, current_prefix=solver.pref, found_words=found_strings)
                viz.draw_dfs_log(solver.stack, solver.pref, "")

# -------------------------
# Runner
# -------------------------
def run():
    random.seed(RNG_SEED)
    viz = Viz()
    ctl = Controller(viz.fig)
    ctl.bind(viz)

    # initial state
    gen = HeapGenerator(WORDS)
    solver = None
    gen_done = False
    found_cells_total = set()

    # draw initial and save snapshot
    viz.draw_grid(gen.grid, title="Generate — empty grid")
    viz.draw_heap_tree(gen.remaining, footnote="Click Step to begin (build heap)", heap_array=[])
    viz.draw_dfs_log([], "", "")
    ctl.set_state_refs({"gen": gen, "solver": solver, "gen_done": gen_done, "found": found_cells_total, "viz": viz})
    ctl.history.append(ctl._make_snapshot())

    while True:
        ctl.wait()

        if ctl.restart:
            random.seed(RNG_SEED)
            gen = HeapGenerator(WORDS); solver = None; gen_done = False
            found_cells_total = set()
            viz.draw_grid(gen.grid, title="Generate — empty grid")
            viz.draw_heap_tree(gen.remaining, footnote="Click Step to begin (build heap)", heap_array=[])
            viz.draw_dfs_log([], "", "")
            ctl.set_state_refs({"gen": gen, "solver": solver, "gen_done": gen_done, "found": found_cells_total, "viz": viz})
            ctl.history = [ctl._make_snapshot()]
            ctl.restart = False
            continue

        # normal step
        if ctl.mode == "generate":
            info = gen.step()
            viz.draw_grid(info["grid"], title=info.get("msg",""),
                          path=info.get("highlight"))
            viz.draw_heap_tree(info["remaining"], footnote=info.get("msg",""),
                               heap_array=info.get("heap_array"))
            viz.draw_dfs_log([], "", "")
            if info["state"] == "done":
                gen_done = True

            # save snapshot AFTER drawing
            ctl.set_state_refs({"gen": gen, "solver": solver, "gen_done": gen_done, "found": found_cells_total, "viz": viz})
            ctl.history.append(ctl._make_snapshot())
            continue

        # Solve mode
        if not gen_done:
            viz.draw_grid(gen.grid, title="Solve — finish generation first (Step in Generate)")
            viz.draw_heap_tree(gen.remaining, footnote="Generation not finished", heap_array=list(gen.heap_arr))
            viz.draw_dfs_log([], "", "")
            ctl.set_state_refs({"gen": gen, "solver": solver, "gen_done": gen_done, "found": found_cells_total, "viz": viz})
            ctl.history.append(ctl._make_snapshot())
            continue

        if solver is None:
            solver = GlobalDFSSolver(gen.grid, WORDS)
            found_cells_total = set()
            viz.draw_grid(gen.grid, title="Solve — Trie + DFS (single pass)")
            viz.draw_trie_full(solver.trie, current_prefix="", found_words=[])
            viz.draw_dfs_log([], "", "")
            ctl.set_state_refs({"gen": gen, "solver": solver, "gen_done": gen_done, "found": found_cells_total, "viz": viz})
            ctl.history.append(ctl._make_snapshot())
            continue

        info = solver.step()

        # Strings to shade in trie (both orientations for each found)
        found_strings = []
        for cw, cells in solver.found.items():
            s = "".join(gen.grid[r][c] for (r,c) in cells)
            found_strings += [s, s[::-1]]

        if info["state"] == "done_all":
            found_cells_total = set(cell for cells in solver.found.values() for cell in cells)
            viz.draw_grid(gen.grid, title="Solve complete", found=found_cells_total)
            viz.draw_trie_full(solver.trie, current_prefix="", found_words=found_strings)
            viz.draw_dfs_log([], "", "DONE")
            ctl.set_state_refs({"gen": gen, "solver": solver, "gen_done": gen_done, "found": found_cells_total, "viz": viz})
            ctl.history.append(ctl._make_snapshot())
            continue

        if info["state"] == "found_word":
            for cell in info["cells"]: found_cells_total.add(cell)
            viz.draw_grid(gen.grid, title=f"Found '{info['word']}'",
                          path=info["cells"], found=found_cells_total)
            viz.draw_trie_full(solver.trie, current_prefix=info.get("prefix",""), found_words=found_strings)
            viz.draw_dfs_log(info.get("stack", []), info.get("prefix",""), "FOUND")
            ctl.set_state_refs({"gen": gen, "solver": solver, "gen_done": gen_done, "found": found_cells_total, "viz": viz})
            ctl.history.append(ctl._make_snapshot())
            continue

        if info["state"] == "explore":
            prefix = info.get("prefix","")
            title = f"Exploring (prefix='{prefix}')"
            if "dfs_note" in info: title += f" — {info['dfs_note']}"
            viz.draw_grid(gen.grid, title=title, path=info.get("path"),
                          found=found_cells_total, probe=info.get("probe"))
            viz.draw_trie_full(solver.trie, current_prefix=prefix, found_words=found_strings)
            viz.draw_dfs_log(info.get("stack", []), prefix, info.get("dfs_note",""))
            ctl.set_state_refs({"gen": gen, "solver": solver, "gen_done": gen_done, "found": found_cells_total, "viz": viz})
            ctl.history.append(ctl._make_snapshot())
            continue

if __name__ == "__main__":
    run()
