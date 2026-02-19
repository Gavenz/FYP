# maze_teach_viz_layout.py
# Visualgo-inspired maze visualizer with DFS/BFS/Dijkstra.
# Left panel (pseudocode + help + status), right panel (maze),
# bold-label message panel above maze (same width as maze),
# compact legend under the maze (same width).
# Responsive layout; DFS/BFS yield micro-steps for teaching.
# U = Retract one step (undo), with redo when stepping forward again.
#
# NEW: 12x12 Tree maze (hard), 12x12 Weighted maze (loops + mud).
# NEW: Totals at goal (steps + path cost).
# NEW: Cost counters (Weighted only) in left panel and (space-permitting) message panel.

import time
import random
import heapq
from collections import deque
import textwrap
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

# --------- Terrain & costs ----------
# 1 = Wall (impassable)
# 0 = Open (cost 1)
# 2 = Mud  (cost MUD_COST)
MUD_COST = 4
COSTS = {0: 1, 2: MUD_COST}  # fallback to 1 if missing


def cell_cost(val: int) -> int:
    return COSTS.get(val, 1)

def new_counters():
    return {
        'push':0, 'pop':0, 'consider':0, 'relax':0, 'stale_pop':0, 'visited_add':0
    }

def graph_sizes():
    """Rough |V| and |E| of current passable grid under 4-neighborhood."""
    V = sum(1 for y in range(H) for x in range(W) if passable(x,y))
    E = 0
    for y in range(H):
        for x in range(W):
            if passable(x,y):
                if x+1 < W and passable(x+1,y): E += 1
                if y+1 < H and passable(x,y+1): E += 1
    E *= 2  # undirected edges, count both ways like adjacency
    return V, E

# --------------------------------------------------------------------------------------
# Utility: generate a 12x12 perfect (tree) maze + a weighted/loopy variant
# --------------------------------------------------------------------------------------
def _neighbors_cells(cx, cy, cw, ch):
    for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
        nx, ny = cx+dx, cy+dy
        if 0 <= nx < cw and 0 <= ny < ch:
            yield nx, ny, dx, dy


def generate_tree_maze_grid(W=12, H=12, seed=42):
    """
    Return a WxH grid (list[list[int]]), 1=wall, 0=open, perfect maze (no loops).
    Uses DFS backtracker over a cell grid of size cw x ch where cw=(W-1)//2.
    """
    rng = random.Random(seed)
    G = [[1 for _ in range(W)] for _ in range(H)]
    cw, ch = (W - 1) // 2, (H - 1) // 2
    start_cx, start_cy = cw // 2, 0

    visited = [[False]*ch for _ in range(cw)]
    stack = [(start_cx, start_cy)]
    visited[start_cx][start_cy] = True

    def open_cell(cx, cy):
        xw, yw = 2*cx + 1, 2*cy + 1
        G[yw][xw] = 0

    open_cell(start_cx, start_cy)

    while stack:
        cx, cy = stack[-1]
        nbrs = [(nx, ny, dx, dy) for nx, ny, dx, dy in _neighbors_cells(cx, cy, cw, ch) if not visited[nx][ny]]
        if not nbrs:
            stack.pop(); continue
        nx, ny, dx, dy = rng.choice(nbrs)
        xw, yw = 2*cx + 1, 2*cy + 1
        xw2, yw2 = 2*nx + 1, 2*ny + 1
        xm, ym = (xw + xw2)//2, (yw + yw2)//2
        G[ym][xm] = 0
        G[yw2][xw2] = 0
        visited[nx][ny] = True
        stack.append((nx, ny))

    return G


def add_loops(G, openings=8, seed=7):
    """Open 'openings' random walls that sit between two open cells (to create loops)."""
    rng = random.Random(seed)
    H = len(G); W = len(G[0])
    candidates = []
    for y in range(1, H-1):
        for x in range(1, W-1):
            if G[y][x] != 1: continue
            horiz = (G[y][x-1] != 1 and G[y][x+1] != 1)
            vert  = (G[y-1][x] != 1 and G[y+1][x] != 1)
            if horiz or vert:
                candidates.append((x,y))
    rng.shuffle(candidates)
    for (x,y) in candidates[:openings]:
        G[y][x] = 0


def add_mud_at_junctions(G, prob_at_branch=0.55, prob_center_band=0.35, seed=9):
    """
    Convert some open cells (0) to mud (2), biased towards junctions (deg>=3)
    and towards a central vertical band to encourage alternative detours.
    """
    rng = random.Random(seed)
    H = len(G); W = len(G[0]); cx = W // 2
    for y in range(1, H-1):
        for x in range(1, W-1):
            if G[y][x] != 0: continue
            deg = sum(G[y+dy][x+dx] != 1 for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)))
            p = 0.0
            if deg >= 3: p += prob_at_branch
            if abs(x - cx) <= 1: p += prob_center_band
            if rng.random() < min(0.85, p):
                G[y][x] = 2  # mud


def deep_copy_grid(src):
    return [row[:] for row in src]


# Build the two teaching mazes
TREE12 = generate_tree_maze_grid(W=12, H=12, seed=123)   # perfect maze (hard, dead-ends)
WEIGHTED12 = deep_copy_grid(TREE12)
add_loops(WEIGHTED12, openings=10, seed=17)             # introduce cycles (alt routes)
add_mud_at_junctions(WEIGHTED12, prob_at_branch=0.65, prob_center_band=0.40, seed=27)

# Bank lets us cycle with 'L'
MAZE_BANK = [
    ("Tree-12", TREE12),
    ("Weighted-12", WEIGHTED12),
]

# Start/Goal auto-centered (top/bottom middle)
START = (6, 0)
GOAL  = (6, 11)
MAZE = None
W = H = 12


def set_maze(src):
    """Deep copy + enforce border walls + auto-center Start/Goal."""
    global MAZE, W, H, START, GOAL
    MAZE = [row[:] for row in src]
    H = len(MAZE); W = len(MAZE[0])

    for x in range(W):
        MAZE[0][x]   = 1
        MAZE[H-1][x] = 1
    for y in range(H):
        MAZE[y][0]   = 1
        MAZE[y][W-1] = 1

    START = (W//2, 0)
    GOAL  = (W//2, H-1)
    MAZE[START[1]][START[0]] = 0
    MAZE[GOAL[1]][GOAL[0]]   = 0
    if MAZE[1][START[0]] == 1:  MAZE[1][START[0]] = 0
    if MAZE[H-2][GOAL[0]] == 1: MAZE[H-2][GOAL[0]] = 0


set_maze(TREE12)


def in_bounds(x, y): return 0 <= x < W and 0 <= y < H
def passable(x, y):  return MAZE[y][x] != 1
NEIGHBOR_DIRS = [(1,0),(0,1),(-1,0),(0,-1)]  # R, D, L, U

def neighbors(x, y):
    for dx, dy in NEIGHBOR_DIRS:
        nx, ny = x+dx, y+dy
        if in_bounds(nx, ny) and passable(nx, ny):
            yield (nx, ny)


# --------------------------------------------------------------------------------------
# Generators (micro-steps for DFS/BFS so students see pushes/enqueues)
# action ∈ {'init','pop','visit','consider','push','deadend','goal'}
# --------------------------------------------------------------------------------------
def gen_dfs(start, goal):
    stack = [start]
    visited = set()
    parent = {start: None}

    ops = new_counters()
    V, E = graph_sizes()

    def snap(action, current, neighbors_list=[]):
        return {
            'algo':'DFS','action':action,'current':current,
            'frontier':list(stack),'visited':set(visited),'parent':dict(parent),
            'dist':{},'neighbors':neighbors_list,'ops':dict(ops),'sizes':(V,E)
        }

    # init
    yield snap('init', start)

    while stack:
        u = stack.pop(); ops['pop'] += 1
        yield snap('pop', u)

        if u == goal:
            yield snap('goal', u)
            return

        if u in visited:
            continue

        visited.add(u); ops['visited_add'] += 1
        nbrs_all = [v for v in neighbors(*u)]
        yield snap('visit', u, nbrs_all)

        pushed_any = False
        for v in nbrs_all:
            ops['consider'] += 1
            yield snap('consider', u, [v])
            if v not in visited and v not in stack:
                parent.setdefault(v, u)
                stack.append(v); ops['push'] += 1; pushed_any = True
                yield snap('push', u, [v])

        if not pushed_any:
            yield snap('deadend', u)


def gen_bfs(start, goal):
    q = deque([start])
    visited = {start}
    parent = {start: None}
    ops = new_counters()
    V, E = graph_sizes()

    def snap(action, current, neighbors_list=[]):
        return {'algo':'BFS','action':action,'current':current,'frontier':list(q),
                'visited':set(visited),'parent':dict(parent),'dist':{},
                'neighbors':neighbors_list,'ops':dict(ops),'sizes':(V,E)}

    yield snap('init', start)
    while q:
        u = q.popleft(); ops['pop'] += 1
        yield snap('pop', u)
        if u == goal:
            yield snap('goal', u)
            return
        for v in neighbors(*u):
            ops['consider'] += 1
            yield snap('consider', u, [v])
            if v not in visited:
                visited.add(v); ops['visited_add'] += 1
                parent[v] = u; q.append(v); ops['push'] += 1
                yield snap('push', u, [v])


def gen_dijkstra(start, goal):
    pq = [(0, start)]            # (cost_so_far, node)
    dist = {start: 0}
    parent = {start: None}
    visited = set()

    ops = new_counters()
    V, E = graph_sizes()

    def snap(action, current, neighbors_list=[]):
        return {
            'algo':'Dijkstra','action':action,'current':current,
            'frontier':list(pq),'visited':set(visited),'parent':dict(parent),
            'dist':dict(dist),'neighbors':neighbors_list,'ops':dict(ops),'sizes':(V,E)
        }

    # initial snapshot
    yield snap('init', start)

    while pq:
        du, u = heapq.heappop(pq)

        # stale extract (worse than current best) → show & skip
        if du != dist.get(u, float('inf')):
            ops['stale_pop'] += 1
            yield snap('stale_pop', u)
            continue

        visited.add(u)
        ops['pop'] += 1
        yield snap('pop_min', u)

        if u == goal:
            yield snap('goal', u)
            return

        for v in neighbors(*u):
            vx, vy = v
            nd = du + cell_cost(MAZE[vy][vx])  # cost to ENTER v

            ops['consider'] += 1
            yield snap('consider', u, [v])

            if nd < dist.get(v, float('inf')):
                dist[v] = nd
                parent[v] = u
                heapq.heappush(pq, (nd, v))
                ops['relax'] += 1
                ops['push']  += 1
                yield snap('relax', u, [v])


# --------------------------------------------------------------------------------------
# Viz
# --------------------------------------------------------------------------------------
class MazeViz:
    def __init__(self):
        self.fig = plt.figure(figsize=(10, 6.9))
        try:
            self.fig.canvas.manager.set_window_title(
                'Maze Visualizer — 1:DFS  2:BFS  3:Dijkstra   SPACE:Step   P:Play/Pause   U:Retract   R:Reset   L:Cycle Maze')
        except Exception:
            pass

        # Left panel (pseudocode + help + status)
        self.left_ax = self.fig.add_axes([0.05, 0.10, 0.32, 0.80]); self.left_ax.axis('off')

        # Maze on the right
        self.ax = self.fig.add_axes([0.45, 0.22, 0.46, 0.60])
        self.ax.set_aspect('equal'); self.ax.axis('off')

        # Message panel above the maze — same width as maze
        self.msg_ax = self.fig.add_axes([0.45, 0.88, 0.46, 0.12]); self.msg_ax.axis('off')

        # Compact legend panel directly under the maze — same width as maze
        self.legend_ax = self.fig.add_axes([0.45, 0.17, 0.46, 0.045]); self.legend_ax.axis('off')

        # Status line (under pseudocode)
        self.info_text = self.left_ax.text(0.02, 0.34, "", transform=self.left_ax.transAxes,
                                           va='bottom', ha='left', family='monospace', color='#333333')
        # NEW: Cost/ops readout
        self.cost_text = self.left_ax.text(0.02, 0.31, "", transform=self.left_ax.transAxes,
                                           va='bottom', ha='left', family='monospace', color='#444444')
        # (kept for spacing; shows ops/cost summary)
        self.complexity_text = self.left_ax.text(0.02, 0.28, "", transform=self.left_ax.transAxes,
                                                 va='bottom', ha='left', family='monospace', color='#444444')

        self.algo_name=None; self.generator=None; self.last_state=None
        self.playing=False
        self.maze_idx = 0
        self.maze_name = MAZE_BANK[self.maze_idx][0]

        # --- History for retract/redo ---
        self.history = []
        self.hist_idx = -1
        self.exhausted = False

        # Help text
        self.help_text = self.left_ax.text(
            0.02, 0.25, self._help_message(), transform=self.left_ax.transAxes,
            va='top', ha='left', family='monospace'
        )

        # responsive state
        self.wrap_cols = 90
        self.msg_fs = 10.0
        self.line_spacing = 0.20

        self.draw_static_maze()

        # Pseudocode support
        self.code_text_artists = []
        our_highlight = None
        self.code_highlight = our_highlight
        self._init_code_blocks()

        # Delta memory
        self.prev_sets = {'visited': set(), 'frontier': set(), 'parents': set()}

        self.set_algo('BFS')

        self.fig.canvas.mpl_connect('resize_event', self.on_resize)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        self.apply_responsive_layout(initial=True)

    # ----------------- panels & legend
    def _help_message(self):
        return (
            "Instructions\n"
            "───────────\n"
            "1: DFS (stack)\n"
            "2: BFS (queue)\n"
            "3: Dijkstra (priority queue, weighted)\n"
            "SPACE: Step  |  P: Autoplay/Pause  |  U: Undo\n"
            "R: Reset     |  L: Cycle Maze (unweighted / weighted)\n"
            "\n"
            f"Move cost = cost(cell you ENTER): Open=1, Mud={MUD_COST}\n"
        )

    def draw_custom_legend(self):
        self.legend_ax.clear(); self.legend_ax.axis('off')
        items = [
            ("Start",   'limegreen', 'gray'),
            ("End",     '#FFF176',  'gray'),
            ("Current", '#FFC000',  '#8d6e00'),
            ("Visited", '#dcdcdc',  'gray'),
            ("Mud",  '#BCAAA4',  'gray'),
            ("Wall",    'black',    'gray'),
            ("Open",    'white',    'gray'),
        ]
        n = len(items)
        y_center = 0.55
        sw_h = 0.32
        sw_w_frac = 0.18

        for i, (label, face, edge) in enumerate(items):
            x0 = (i + 0.02) / n
            w  = (1.0 - 0.04) / n
            self.legend_ax.add_patch(Rectangle((x0, y_center - sw_h/2), w*sw_w_frac, sw_h,
                                               transform=self.legend_ax.transAxes,
                                               facecolor=face, edgecolor=edge, linewidth=0.8))
            self.legend_ax.text(x0 + w*sw_w_frac + w*0.06, y_center, label, transform=self.legend_ax.transAxes,
                                va='center', ha='left', family='monospace',
                                fontsize=max(8, self.msg_fs - 1.6), color='#222')

        self.legend_ax.add_patch(Rectangle((0,0),1,1, transform=self.legend_ax.transAxes,
                                           facecolor='none', edgecolor='#d0d7e2', linewidth=0.8))
        self.fig.canvas.draw_idle()

    # ----------------- maze drawing
    def draw_static_maze(self):
        self.ax.clear()
        self.ax.set_aspect('equal'); self.ax.axis('off')
        self.ax.set_xlim(0, W); self.ax.set_ylim(H, 0)

        for y in range(H):
            for x in range(W):
                v = MAZE[y][x]
                if v == 1:   color = 'black'
                elif v == 2: color = '#BCAAA4'  # mud
                else:        color = 'white'
                self.ax.add_patch(Rectangle((x,y),1,1,facecolor=color,edgecolor='gray',linewidth=0.5))
        self.ax.add_patch(Rectangle(START,1,1,facecolor='limegreen',edgecolor='gray',linewidth=1.0,alpha=0.9))
        self.ax.add_patch(Rectangle(GOAL, 1,1,facecolor='#FFF176',edgecolor='gray',linewidth=1.0,alpha=0.9))
        self.fig.canvas.draw_idle()

    # ----------------- algorithm plumbing
    def set_algo(self, name):
        self.algo_name = name
        if name=='DFS': self.generator = gen_dfs(START, GOAL)
        elif name=='BFS': self.generator = gen_bfs(START, GOAL)
        else: self.generator = gen_dijkstra(START, GOAL)
        self.last_state=None; self.playing=False
        self.exhausted=False
        self._t0 = time.perf_counter()
        self.elapsed_ms = None
        self.history=[]; self.hist_idx=-1
        self.prev_sets = {'visited': set(), 'frontier': set(), 'parents': set()}
        self.draw_static_maze()
        self.update_overlay(None)

    def cycle_maze(self):
        self.maze_idx = (self.maze_idx + 1) % len(MAZE_BANK)
        self.maze_name = MAZE_BANK[self.maze_idx][0]
        set_maze(MAZE_BANK[self.maze_idx][1])
        self.set_algo(self.algo_name or 'BFS')

    def reconstruct_path(self, parent, goal):
        if goal not in parent: return []
        path=[]; cur=goal
        while cur is not None:
            path.append(cur); cur = parent[cur]
        return list(reversed(path))

    def path_cost(self, path):
        """Total cost of a path (sum of cells you enter)."""
        if not path: return 0
        total = 0
        for (x,y) in path[1:]:
            total += cell_cost(MAZE[y][x])
        return total

    def depth_of(self, node, parent):
        if node is None or node not in parent: return -1
        d = 0; cur = node
        while cur is not None and cur in parent:
            cur = parent[cur]
            if cur is not None: d += 1
        return d

    # --- history helpers ---
    def _snapshot(self, s):
        if s is None: return None
        return {
            'algo': s['algo'],
            'action': s.get('action'),
            'current': s.get('current'),
            'frontier': list(s.get('frontier', [])),
            'visited': set(s.get('visited', set())),
            'parent': dict(s.get('parent', {})),
            'dist': dict(s.get('dist', {})),
            'neighbors': list(s.get('neighbors', [])),
        }

    def _sets_from_state(self, s):
        if s is None:
            return {'visited': set(), 'frontier': set(), 'parents': set()}
        if s['algo'] == 'Dijkstra':
            fr_nodes = {node for (_d, node) in (s.get('frontier') or [])}
        else:
            fr_nodes = set(s.get('frontier') or [])
        return {
            'visited': set(s.get('visited', set())),
            'frontier': set(fr_nodes),
            'parents': set(s.get('parent', {}).keys())
        }

    # ----------------- overlay drawing
    def draw_graph_overlay(self, parent, visited, frontier, current, path):
        for v in visited:
            p = parent.get(v)
            if p is None: continue
            x1,y1 = p[0]+0.5, p[1]+0.5
            x2,y2 = v[0]+0.5, v[1]+0.5
            self.ax.plot([x1,x2],[y1,y2],linewidth=1.3,color='#9e9e9e',alpha=0.9,zorder=5)
        if path and len(path)>1:
            xs=[x+0.5 for x,_ in path]; ys=[y+0.5 for _,y in path]
            self.ax.plot(xs,ys,linewidth=3.0,color='#FFD54F',alpha=1.0,zorder=6)
        is_pq = (self.algo_name == 'Dijkstra')
        fr_nodes = [node for (_d, node) in (frontier or [])] if is_pq else list(frontier or [])
        for (x,y) in fr_nodes:
            if (x,y) not in (START,GOAL):
                self.ax.add_patch(Circle((x+0.5,y+0.5),0.22,facecolor='none',
                                         edgecolor='#4472C4',linewidth=2.0,zorder=8))
        if current and current not in (START,GOAL):
            self.ax.add_patch(Circle((current[0]+0.5,current[1]+0.5),0.18,facecolor='#FFC000',
                                     edgecolor='#8d6e00',linewidth=1.1,zorder=9))

    # ----------------- message helpers (bold labels + deltas)
    def _msg_clear(self, kind='info'):
        face = {'info':'#eef3ff','ok':'#f0fff0','warn':'#fff8e1','err':'#ffebee'}.get(kind,'#eef3ff')
        edge = {'info':'#c7d2ff','ok':'#cfe8cf','warn':'#ffe0a3','err':'#ffcdd2'}.get(kind,'#c7d2ff')
        self.msg_ax.clear(); self.msg_ax.axis('off')
        self.msg_ax.add_patch(Rectangle((0,0),1,1, transform=self.msg_ax.transAxes,
                                        facecolor=face, edgecolor=edge, linewidth=1.0))

    def _draw_section(self, y, label, text, bold=True):
        x_label = 0.015
        x_text  = 0.18
        avail_frac = max(0.05, 1.0 - x_text - 0.02)
        cols_for_content = max(12, int(self.wrap_cols * avail_frac / (1.0 - x_label - 0.02)))
        wrapped = textwrap.wrap(text, width=cols_for_content) or [""]
        self.msg_ax.text(x_label, y, f"{label}:", transform=self.msg_ax.transAxes,
                         va='top', ha='left', family='monospace',
                         fontsize=self.msg_fs, fontweight=('bold' if bold else 'normal'), color='#222')
        self.msg_ax.text(x_text,  y, wrapped[0], transform=self.msg_ax.transAxes,
                         va='top', ha='left', family='monospace',
                         fontsize=self.msg_fs, color='#222')
        y_cur = y
        for cont in wrapped[1:]:
            y_cur -= self.line_spacing
            self.msg_ax.text(x_text, y_cur, cont, transform=self.msg_ax.transAxes,
                             va='top', ha='left', family='monospace',
                             fontsize=self.msg_fs, color='#222')
        return y_cur - (self.line_spacing + 0.02)

    def set_message(self, sections, kind='info'):
        self._msg_clear(kind)
        y = 0.92
        for (label, text) in sections:
            y = self._draw_section(y, label, text, bold=True)
            if y < 0.10:
                self.msg_ax.text(0.015, 0.08, "...", transform=self.msg_ax.transAxes,
                                 va='top', ha='left', family='monospace',
                                 fontsize=self.msg_fs, color='#222')
                break
        self.fig.canvas.draw_idle()

    def fmt_add(self, name, items, max_items=6):
        items = list(items)
        if not items: return ""
        show = items[:max_items]
        tail = "" if len(items) <= max_items else " ..."
        return f"{name} +{show}{tail}"

    # ----------------- stepping / retract -----------------
    def step_once(self):
        if self.hist_idx + 1 < len(self.history):
            self.hist_idx += 1
            prev_idx = self.hist_idx - 1
            self.prev_sets = self._sets_from_state(self.history[prev_idx] if prev_idx >= 0 else None)
            state = self.history[self.hist_idx]
            self.last_state = state
            finished = (self.exhausted and self.hist_idx == len(self.history)-1) or state.get('action')=='goal'
            self.update_overlay(state, finished=finished)
            return
        try:
            state = next(self.generator)
            self.last_state = state
            self.prev_sets = self._sets_from_state(self.history[-1] if self.history else None)
            self.history.append(self._snapshot(state))
            self.hist_idx = len(self.history)-1
            self.update_overlay(state)
        except StopIteration:
            self.playing=False
            self.exhausted=True
            if self.last_state:
                self.prev_sets = self._sets_from_state(self.history[-2] if len(self.history) >= 2 else None)
                self.update_overlay(self.last_state, finished=True)

    def retract_once(self):
        self.playing = False
        if self.hist_idx == -1:
            self.prev_sets = {'visited': set(), 'frontier': set(), 'parents': set()}
            self.update_overlay(None); return
        self.hist_idx -= 1
        if self.hist_idx == -1:
            self.prev_sets = {'visited': set(), 'frontier': set(), 'parents': set()}
            self.update_overlay(None); return
        prev_idx = self.hist_idx - 1
        self.prev_sets = self._sets_from_state(self.history[prev_idx] if prev_idx >= 0 else None)
        state = self.history[self.hist_idx]
        self.last_state = state
        finished = (self.exhausted and self.hist_idx == len(self.history)-1) or state.get('action')=='goal'
        self.update_overlay(state, finished=finished)

    # ----------------- core render
    def update_overlay(self, state, finished=False):
        self.draw_static_maze()
        self.draw_custom_legend()

        if state is None:
            self.set_message([("READY",
                               "SPACE = step. 1/2/3 = DFS/BFS/Dijkstra. U = retract. L = cycle maze.")], kind='info')
            self.info_text.set_text(f"Algorithm: {self.algo_name}   Maze: {self.maze_name}")
            self.cost_text.set_text("" if not self.maze_name.startswith("Weighted") else
                                    f"Cost — Open=1, Mud={MUD_COST}.")
            self._draw_code_panel(self.algo_name, current_line=0)
            self.fig.canvas.draw_idle()
            return

        visited  = state['visited']
        frontier = state['frontier']
        parent   = state['parent']
        current  = state['current']
        dist_map = state.get('dist', {})
        ops = state.get('ops', {}) if state else {}
        V,E = state.get('sizes', graph_sizes()) if state else graph_sizes()

        # shaded layers
        for (x,y) in visited:
            if (x,y) not in (START,GOAL):
                self.ax.add_patch(Rectangle((x,y),1,1,facecolor='#dcdcdc',edgecolor='gray',alpha=0.9))
        fr_nodes = [node for (_d, node) in (frontier or [])] if self.algo_name=='Dijkstra' else list(frontier or [])
        for (x,y) in fr_nodes:
            if (x,y) not in (START,GOAL):
                self.ax.add_patch(Rectangle((x,y),1,1,facecolor='#4472C4',edgecolor='gray',alpha=0.8))
        if current not in (START,GOAL):
            self.ax.add_patch(Rectangle(current,1,1,facecolor='#FFC000',edgecolor='gray',alpha=0.95))

        # final path + totals (at goal)
        path=[]; totals_text=""
        if finished or state.get('action') == 'goal' or current == GOAL:
            path = self.reconstruct_path(parent, GOAL)
            for (x,y) in path:
                if (x,y) not in (START,GOAL):
                    self.ax.add_patch(Rectangle((x,y),1,1,facecolor='#FFF176',edgecolor='gray',alpha=0.95))
            steps = max(0, len(path)-1)
            cost  = self.path_cost(path)
            totals_text = f"steps={steps}, path cost={cost}"

        self.draw_graph_overlay(parent, visited, frontier, current, path)

        # status (left)
        depth = self.depth_of(current, parent) if current in parent else -1
        if self.algo_name in ('DFS','BFS'):
            extra = f"depth {depth if depth>=0 else '-'}"
        else:
            gn = dist_map.get(current, '-')
            extra = f"g(n) {gn}"
        self.info_text.set_text(''
           # f"{self.algo_name} — current {current} | frontier {len(fr_nodes)} | visited {len(visited)} | {extra} | Maze: {self.maze_name}"
        )
        if self.algo_name in ('BFS', 'DFS'):
            theo = f"Theory: O(V+E)  |  Here: V={V}, E={E}"
        else:
            theo = f"Theory: O((V+E)·log V)  |  Here: V={V}, E={E}"

        # friendlier operation labels (keep internal counters as-is)
        ops_line = (
            f"work: added_to_frontier={ops.get('push', 0)}, "
            f"removed_from_frontier={ops.get('pop', 0)}, "
            f"neighbors_examined={ops.get('consider', 0)}"
        )
        if self.algo_name == 'Dijkstra':
            ops_line += (
                f", edge_relaxed={ops.get('relax', 0)}, "
                f"stale_extracts={ops.get('stale_pop', 0)}"
            )
        # --- path_so_far (robust for BOTH weighted & unweighted) ----------------------
        def _safe_reconstruct_to_start(cur, parent_map, start_node):
            """
            Return [start ... cur] if there's a clean chain; else [].
            Robust to transient/partial parents during Dijkstra relaxations.
            """
            if cur is None or not parent_map:
                return []
            path, u, seen = [], cur, set()
            while True:
                path.append(u)
                if u == start_node:
                    path.reverse()
                    return path
                if u in seen or u not in parent_map:
                    return []  # loop or broken chain
                seen.add(u)
                u = parent_map[u]

        def _current_path_cost(algo_name, cur, parent_map, dist_map, start_node):
            # 1) Dijkstra: prefer the distance map if available (true g(cur))
            if algo_name == 'Dijkstra' and dist_map and cur in dist_map:
                return dist_map[cur]
            # 2) Otherwise try a safe reconstruction and compute cost
            path = _safe_reconstruct_to_start(cur, parent_map, start_node)
            if len(path) >= 2:
                return self.path_cost(path)  # works for weighted & unweighted
            # 3) Not ready yet
            return '-'

        # pull what we need from 'state' (these names already exist in update_overlay)
        parent_map = parent          # from the args you already have
        cur_node   = current
        dist_map   = state.get('dist', {})   # Dijkstra provides this; others won't

        cur_cost = _current_path_cost(self.algo_name, cur_node, parent_map, dist_map, START)
        # ------------------------------------------------------------------------------

        # --- empirical proxies for "time complexity" ---
        nodes_visited = len(visited)  
        steps_taken = (
            ops.get('push', 0)
            + ops.get('pop', 0)
            + ops.get('consider', 0)
            + (ops.get('relax', 0) if self.algo_name == 'Dijkstra' else 0)
            + (ops.get('stale_pop', 0) if self.algo_name == 'Dijkstra' else 0)
        )
        t_ms = getattr(self, 'elapsed_ms', None)
        empirical = f"Nodes Visited={nodes_visited} | Steps Taken={steps_taken} | Path Cost = {cur_cost}"
        if t_ms is not None:
            empirical += f"  |  Runtime={t_ms:.0f} ms"

        # deltas (relative to previous state)
        cur_frontier_set = set(fr_nodes)
        cur_parents_keys = set(parent.keys())
        add_frontier = list(cur_frontier_set - self.prev_sets['frontier'])
        add_visited  = list(visited - self.prev_sets['visited'])
        add_parents  = list(cur_parents_keys - self.prev_sets['parents'])
        self.prev_sets = {'visited': set(visited), 'frontier': set(cur_frontier_set), 'parents': set(cur_parents_keys)}

        p = parent.get(current)
        where = (f"start at {current}" if p is None else f"at {current} (parent {p})")
        action = state.get('action', '')

        if action == 'init':
            action_text = "initialize data structures."
        elif action == 'pop':
            action_text = ("pop()" if self.algo_name == 'DFS' else "dequeue()")
        elif action == 'visit':
            action_text = "mark current as visited."
        elif action == 'consider':
            nbrs_str = ", ".join(str(t) for t in state.get('neighbors', [])) or "none"
            action_text = f"consider neighbor(s): {nbrs_str}"
        elif action == 'push':
            nbrs_str = ", ".join(str(t) for t in state.get('neighbors', [])) or "none"
            action_text = (f"push onto stack: {nbrs_str}"
                           if self.algo_name == 'DFS' else f"enqueue: {nbrs_str}")
        elif action == 'deadend':
            action_text = "no unvisited neighbors -> dead end."
        elif action == 'goal':
            action_text = f"reached GOAL {GOAL}."
        elif action == 'pop_min':
            action_text = "extract-min from PQ."
        elif action == 'relax':
            nbrs_str = ", ".join(str(t) for t in state.get('neighbors', [])) or "none"
            action_text = f"relax (decrease-key) → push/update: {nbrs_str}"
        elif action == 'stale_pop':
            action_text = "discard stale PQ entry."
        else:
            action_text = action

        deltas = []
        s1 = self.fmt_add("frontier", add_frontier, max_items=6)
        s2 = self.fmt_add("visited",  add_visited,  max_items=8)
        s3 = self.fmt_add("parents",  add_parents,  max_items=8)
        for s in (s1,s2,s3):
            if s: deltas.append(s)
        delta_line = " | ".join(deltas) if deltas else "(no changes)"
        sections = [("WHERE", where), ("ACTION", action_text), ("Δ", delta_line), ("COMPLEXITY", empirical)]
        if totals_text:
            sections.append(("TOTAL", totals_text))
        kind = ('ok' if action=='goal' else ('warn' if action=='deadend' else 'info'))
        self.set_message(sections, kind=kind)

        line = self._which_line(state['algo'], state, finished)
        self._draw_code_panel(state['algo'], current_line=line)
        self.fig.canvas.draw_idle()

    # ----------------- keys
    def on_key(self, event):
        k = (event.key or '').lower()
        if k=='1':
            self.set_algo('DFS')
        elif k=='2':
            self.set_algo('BFS')
        elif k=='3':
            self.set_algo('Dijkstra')
        elif k==' ':
            self.step_once()
        elif k=='p':
            self.playing = not self.playing
            if self.playing: self.autoplay()
        elif k=='u':
            self.retract_once()
        elif k=='r':
            self.set_algo(self.algo_name)
        elif k=='l':
            self.cycle_maze()

    def autoplay(self):
        while self.playing:
            self.step_once()
            self.fig.canvas.flush_events()
            plt.pause(0.08)

    # ----------------- RESIZE HANDLER -----------------
    def on_resize(self, event):
        self.apply_responsive_layout()

    def apply_responsive_layout(self, initial=False):
        fig_w_in, fig_h_in = self.fig.get_size_inches()
        dpi = self.fig.dpi or 100
        fig_w_px = fig_w_in * dpi
        fig_h_px = fig_h_in * dpi
        aspect = fig_w_in / (fig_h_in or 1e-6)

        small_screen = (fig_h_px < 800 or fig_w_px < 1200)

        if aspect >= 1.5:  # wide
            left_rect  = [0.05, 0.10, 0.32, 0.80]
            maze_rect  = [0.45, 0.22, 0.46, 0.60]
        elif 1.15 <= aspect < 1.5:  # medium
            left_rect  = [0.04, 0.10, 0.36, 0.80]
            maze_rect  = [0.43, 0.22, 0.50, 0.58]
        else:  # narrow/tall
            left_rect  = [0.05, 0.05, 0.90, 0.30]
            maze_rect  = [0.08, 0.44, 0.84, 0.50]

        msg_rect    = [maze_rect[0], maze_rect[1] + maze_rect[3] + 0.012,
                       maze_rect[2], 0.18 if small_screen else 0.16]
        legend_h    = 0.045
        legend_rect = [maze_rect[0], max(0.04, maze_rect[1] - (legend_h + 0.012)),
                       maze_rect[2], legend_h]

        self.left_ax.set_position(left_rect)
        self.ax.set_position(maze_rect)
        self.msg_ax.set_position(msg_rect)
        self.legend_ax.set_position(legend_rect)

        # keep status lines where we expect them
        self.info_text.set_position((0.02, 0.34))
        self.cost_text.set_position((0.02, 0.31))
        self.complexity_text.set_position((0.02, 0.28))

        msg_px = fig_w_px * msg_rect[2]
        self.wrap_cols = max(42, min(120, int(msg_px / 8.4)))
        self.msg_fs = (9.6 if small_screen else 10.2) + 0.0020 * msg_px
        self.help_text.set_fontsize(self.msg_fs - (1.0 if small_screen else 0.8))
        self.line_spacing = 0.18 if small_screen else 0.20

        self.draw_custom_legend()
        if self.last_state is None and not initial:
            self.update_overlay(None)
        elif self.last_state is not None:
            cur_state = self.history[self.hist_idx] if 0 <= self.hist_idx < len(self.history) else self.last_state
            finished = (self.exhausted and self.hist_idx == len(self.history)-1) or cur_state.get('action')=='goal'
            self.prev_sets = self._sets_from_state(self.history[self.hist_idx-1] if self.hist_idx-1 >= 0 else None)
            self.update_overlay(cur_state, finished=finished)

        self.fig.canvas.draw_idle()

    # ---------- pseudocode blocks & highlighting ----------
    def _init_code_blocks(self):
        self.PSEUDOCODE = {
            'DFS': [
                "push(start)                      # stack = LIFO",
                "visited = ∅; parent[start]=None",
                "while stack not empty:",
                "    u = pop()                    # remove last",
                "    if u == goal: break",
                "    if u ∉ visited:",
                "        visited.add(u)",
                "        for v in neighbors(u):",
                "            if v ∉ visited and v ∉ stack:",
                "                parent[v] = u; push(v)",
            ],
            'BFS': [
                "enqueue(start)                   # queue = FIFO, layer 0",
                "visited = {start}; parent[start]=None",
                "while queue not empty:",
                "    u = dequeue()                # remove front",
                "    if u == goal: break",
                "    for v in neighbors(u):",
                "        if v ∉ visited:",
                "            visited.add(v)       # discover next layer",
                "            parent[v] = u; enqueue(v)",
            ],
            'Dijkstra': [
                "push(0,start)  # Priority queue on cost",
                "dist[start]=0; parent[start]=None",
                "while Priority_Queue not empty:",
                "    (du,u) = pop_min()",
                "    if u == goal: break",
                "    for v in neighbors(u):",
                "        nd = du + cost(v)",
                "        if nd < dist[v]:",
                "            dist[v]=nd; parent[v]=u; push(nd,v)",
            ],
        }

    def _draw_code_panel(self, algo, current_line=None):
        for a in getattr(self, "code_text_artists", []):
            try: a.remove()
            except Exception: pass
        self.code_text_artists = []
        if self.code_highlight is not None:
            try: self.code_highlight.remove()
            except Exception: pass
            self.code_highlight = None

        lines = self.PSEUDOCODE.get(algo, [])
        ax = self.left_ax
        x0, y0 = 0.02, 0.98
        dy = 0.06

        if current_line is not None and 0 <= current_line < len(lines):
            y_bar = y0 - dy * current_line - dy*0.85
            self.code_highlight = ax.add_patch(
                Rectangle((x0-0.01, y_bar), 0.94, dy*0.9,
                          transform=ax.transAxes, facecolor='#FFF7CC',
                          edgecolor='#E6D58E', linewidth=0.6, zorder=0)
            )

        fs = max(8, self.msg_fs-1)
        for i, s in enumerate(lines):
            y = y0 - dy * i
            if y < 0.30: break
            t = ax.text(x0, y, s, transform=ax.transAxes,
                        va='top', ha='left', family='monospace', fontsize=fs)
            self.code_text_artists.append(t)

        self.fig.canvas.draw_idle()

    def _which_line(self, algo, state, finished):
        a = state.get('action') if state else None
        if finished or a == 'goal': return 4
        if algo == 'DFS':
            return {'init':0,'pop':3,'visit':6,'consider':7,'push':9,'deadend':7}.get(a,3)
        if algo == 'BFS':
            return {'init':0,'pop':3,'consider':5,'push':8,'deadend':5}.get(a,3)
        if algo == 'Dijkstra':
            return {'init':0, 'pop_min':3, 'consider':6, 'relax':8, 'stale_pop':3, 'deadend':5}.get(a, 3)
        return 6 if a == 'expand' else (5 if a=='deadend' else 3)


if __name__ == '__main__':
    # start on Tree-12; press L to cycle to Weighted-12
    set_maze(MAZE_BANK[0][1])
    viz = MazeViz()
    plt.show()
