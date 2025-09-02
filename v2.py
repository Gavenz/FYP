# maze_teach_viz_layout.py
# Visualgo-inspired maze visualizer with DFS/BFS/Dijkstra.
# Left panel (pseudocode + help + status), right panel (maze),
# bold-label message panel above maze (same width as maze),
# compact custom legend directly under maze (same width as maze) in requested order.
# Responsive layout; DFS/BFS yield micro-steps for teaching.

import heapq
from collections import deque
import textwrap
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

# --------------------------------------------------------------------------------------
# Mazes (7x7). Start = (3,0), Goal = (3,6).
# --------------------------------------------------------------------------------------
TREE_MAZE = [
    [1,1,1,0,1,1,1],
    [1,0,1,0,1,0,1],
    [1,0,1,0,0,0,1],
    [1,0,1,1,1,0,1],
    [1,0,0,0,1,0,1],
    [1,1,1,0,0,0,1],
    [1,1,1,0,1,1,1],
]
LOOPY_MAZE = [
    [1,1,1,0,1,1,1],
    [1,0,0,0,1,0,1],
    [1,0,1,0,0,0,1],
    [1,0,1,1,0,1,1],
    [1,0,0,0,0,0,1],
    [1,1,0,1,1,0,1],
    [1,1,1,0,1,1,1],
]

START = (3, 0)
GOAL  = (3, 6)

MAZE = None
W = H = 7

def set_maze(src):
    global MAZE, W, H
    MAZE = [row[:] for row in src]
    H = len(MAZE); W = len(MAZE[0])
    for x in range(W):
        MAZE[0][x]   = 1
        MAZE[H-1][x] = 1
    for y in range(H):
        MAZE[y][0]   = 1
        MAZE[y][W-1] = 1
    MAZE[START[1]][START[0]] = 0
    MAZE[GOAL[1]][GOAL[0]]   = 0

set_maze(TREE_MAZE)

def in_bounds(x,y): return 0 <= x < W and 0 <= y < H
def passable(x,y):  return MAZE[y][x] == 0
NEIGHBOR_DIRS = [(1,0),(0,1),(-1,0),(0,-1)]  # R, D, L, U

def neighbors(x,y):
    for dx,dy in NEIGHBOR_DIRS:
        nx, ny = x+dx, y+dy
        if in_bounds(nx,ny) and passable(nx,ny):
            yield (nx,ny)

# --------------------------------------------------------------------------------------
# Generators (micro-steps for DFS/BFS so students see pushes/enqueues)
# action ∈ {'init','pop','visit','consider','push','deadend','goal'}
# --------------------------------------------------------------------------------------
def gen_dfs(start, goal):
    stack = [start]
    visited = set()
    parent = {start: None}
    yield {'algo':'DFS','action':'init','current':start,'frontier':list(stack),
           'visited':set(visited),'parent':dict(parent),'dist':{},'neighbors':[]}
    while stack:
        u = stack.pop()
        yield {'algo':'DFS','action':'pop','current':u,'frontier':list(stack),
               'visited':set(visited),'parent':dict(parent),'dist':{},'neighbors':[]}
        if u == goal:
            yield {'algo':'DFS','action':'goal','current':u,'frontier':list(stack),
                   'visited':set(visited),'parent':dict(parent),'dist':{},'neighbors':[]}
            return
        if u in visited:
            continue
        visited.add(u)
        nbrs_all = [v for v in neighbors(*u)]
        yield {'algo':'DFS','action':'visit','current':u,'frontier':list(stack),
               'visited':set(visited),'parent':dict(parent),'dist':{},'neighbors':nbrs_all}
        pushed_any = False
        for v in nbrs_all:
            yield {'algo':'DFS','action':'consider','current':u,'frontier':list(stack),
                   'visited':set(visited),'parent':dict(parent),'dist':{},'neighbors':[v]}
            if v not in visited and v not in stack:
                parent.setdefault(v,u)
                stack.append(v)
                pushed_any = True
                yield {'algo':'DFS','action':'push','current':u,'frontier':list(stack),
                       'visited':set(visited),'parent':dict(parent),'dist':{},'neighbors':[v]}
        if not pushed_any:
            yield {'algo':'DFS','action':'deadend','current':u,'frontier':list(stack),
                   'visited':set(visited),'parent':dict(parent),'dist':{},'neighbors':[]}

def gen_bfs(start, goal):
    q = deque([start])
    visited = {start}
    parent = {start: None}
    yield {'algo':'BFS','action':'init','current':start,'frontier':list(q),
           'visited':set(visited),'parent':dict(parent),'dist':{},'neighbors':[]}
    while q:
        u = q.popleft()
        yield {'algo':'BFS','action':'pop','current':u,'frontier':list(q),
               'visited':set(visited),'parent':dict(parent),'dist':{},'neighbors':[]}
        if u == goal:
            yield {'algo':'BFS','action':'goal','current':u,'frontier':list(q),
                   'visited':set(visited),'parent':dict(parent),'dist':{},'neighbors':[]}
            return
        for v in neighbors(*u):
            yield {'algo':'BFS','action':'consider','current':u,'frontier':list(q),
                   'visited':set(visited),'parent':dict(parent),'dist':{},'neighbors':[v]}
            if v not in visited:
                visited.add(v); parent[v]=u; q.append(v)
                yield {'algo':'BFS','action':'push','current':u,'frontier':list(q),
                       'visited':set(visited),'parent':dict(parent),'dist':{},'neighbors':[v]}

def gen_dijkstra(start, goal):
    pq = [(0,start)]
    dist = {start:0}
    parent = {start:None}
    visited = set()
    while pq:
        du,u = heapq.heappop(pq)
        if u in visited: continue
        visited.add(u)
        nbrs=[]
        for v in neighbors(*u):
            nd = du+1
            if nd < dist.get(v, float('inf')):
                dist[v]=nd; parent[v]=u; heapq.heappush(pq,(nd,v)); nbrs.append(v)
        action = "expand" if nbrs else "deadend"
        yield {'algo':'Dijkstra','action':action,'current':u,'frontier':list(pq),
               'visited':set(visited),'parent':dict(parent),'dist':dict(dist),'neighbors':nbrs}
        if u == goal:
            yield {'algo':'Dijkstra','action':'goal','current':u,'frontier':list(pq),
                   'visited':set(visited),'parent':dict(parent),'dist':dict(dist),'neighbors':[]}
            return

# --------------------------------------------------------------------------------------
# Viz
# --------------------------------------------------------------------------------------
class MazeViz:
    def __init__(self):
        self.fig = plt.figure(figsize=(10, 6.9))
        try:
            self.fig.canvas.manager.set_window_title(
                'Maze Visualizer — 1:DFS  2:BFS  3:Dijkstra   SPACE:Step   P:Play/Pause   R:Reset   L:Toggle Maze')
        except Exception:
            pass

        # Left panel (pseudocode + help + status)
        self.left_ax = self.fig.add_axes([0.05, 0.10, 0.32, 0.80]); self.left_ax.axis('off')

        # Maze on the right
        self.ax = self.fig.add_axes([0.45, 0.24, 0.46, 0.58])
        self.ax.set_aspect('equal'); self.ax.axis('off')

        # Message panel above the maze — same width as maze
        self.msg_ax = self.fig.add_axes([0.45, 0.88, 0.46, 0.12]); self.msg_ax.axis('off')

        # Compact legend panel directly under the maze — same width as maze
        self.legend_ax = self.fig.add_axes([0.45, 0.17, 0.46, 0.045]); self.legend_ax.axis('off')

        # Status line now LIVES IN LEFT PANEL (under pseudocode)
        self.info_text = self.left_ax.text(0.02, 0.32, "", transform=self.left_ax.transAxes,
                                           va='bottom', ha='left', family='monospace', color='#333333')

        self.algo_name=None; self.generator=None; self.last_state=None; self.playing=False; self.use_loopy=False

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
        self.code_highlight = None
        self._init_code_blocks()

        # Delta memory
        self.prev_sets = {'visited': set(), 'frontier': set(), 'parents': set()}

        self.set_algo('BFS')

        # handlers
        self.fig.canvas.mpl_connect('resize_event', self.on_resize)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        # responsive init
        self.apply_responsive_layout(initial=True)

    # ----------------- panels & legend
    def _help_message(self):
        return (
            "How to Play\n"
            "───────────\n"
            "1: DFS (stack)\n"
            "2: BFS (queue)\n"
            "3: Dijkstra (PQ)\n"
            "SPACE: Step  |  P: Play/Pause\n"
            "R: Reset     |  L: Toggle Maze\n"
            "\n"
            "nodes + lines = search tree"
        )

    def draw_custom_legend(self):
        """Compact single-row legend; same width as message/maze; order: Start, End, Current, Visited, Goal, Wall, Open."""
        self.legend_ax.clear(); self.legend_ax.axis('off')
        items = [
            ("Start",   'limegreen', 'gray'),
            ("End",     '#FFF176',  'gray'),
            ("Current", '#FFC000',  '#8d6e00'),
            ("Visited", '#dcdcdc',  'gray'),
            ("Goal",    'crimson',  'gray'),
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

        # thin border
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
                color = 'white' if MAZE[y][x]==0 else 'black'
                self.ax.add_patch(Rectangle((x,y),1,1,facecolor=color,edgecolor='gray',linewidth=0.5))
        self.ax.add_patch(Rectangle(START,1,1,facecolor='limegreen',edgecolor='gray',linewidth=1.0,alpha=0.9))
        self.ax.add_patch(Rectangle(GOAL, 1,1,facecolor='crimson',  edgecolor='gray',linewidth=1.0,alpha=0.9))
        self.fig.canvas.draw_idle()

    # ----------------- algorithm plumbing
    def set_algo(self, name):
        self.algo_name = name
        if name=='DFS': self.generator = gen_dfs(START, GOAL)
        elif name=='BFS': self.generator = gen_bfs(START, GOAL)
        else: self.generator = gen_dijkstra(START, GOAL)
        self.last_state=None; self.playing=False
        self.prev_sets = {'visited': set(), 'frontier': set(), 'parents': set()}
        self.draw_static_maze()
        self.update_overlay(None)

    def reconstruct_path(self, parent, goal):
        if goal not in parent: return []
        path=[]; cur=goal
        while cur is not None:
            path.append(cur); cur = parent[cur]
        return list(reversed(path))

    def depth_of(self, node, parent):
        if node is None or node not in parent: return -1
        d = 0; cur = node
        while cur is not None and cur in parent:
            cur = parent[cur]
            if cur is not None: d += 1
        return d

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
        """
        Two-column layout: fixed label column + content column to avoid character clipping.
        """
        # fixed columns in axes fraction
        x_label = 0.015
        x_text  = 0.18            # <- content starts here; tweak if you want wider labels
        # compute wrap width as "how much of the panel width is available for content"
        avail_frac = max(0.05, 1.0 - x_text - 0.02)
        cols_for_content = max(12, int(self.wrap_cols * avail_frac / (1.0 - x_label - 0.02)))
        wrapped = textwrap.wrap(text, width=cols_for_content) or [""]

        # draw label
        self.msg_ax.text(x_label, y, f"{label}:", transform=self.msg_ax.transAxes,
                         va='top', ha='left', family='monospace',
                         fontsize=self.msg_fs, fontweight=('bold' if bold else 'normal'), color='#222')
        # first line
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

    # ----------------- core update
    def step_once(self):
        try:
            state = next(self.generator)
            self.last_state = state
            self.update_overlay(state)
        except StopIteration:
            self.playing=False
            if self.last_state:
                self.update_overlay(self.last_state, finished=True)

    def update_overlay(self, state, finished=False):
        self.draw_static_maze()
        self.draw_custom_legend()

        if state is None:
            self.set_message([("READY",
                               "Press SPACE to step. Use 1/2/3 to switch algorithms. L toggles Tree/Loopy maze.")], kind='info')
            # status now in left panel
            self.info_text.set_text(f"Algorithm: {self.algo_name}   Maze: {'Loopy' if self.use_loopy else 'Tree'}")
            self._draw_code_panel(self.algo_name, current_line=0)
            self.fig.canvas.draw_idle()
            return

        visited  = state['visited']
        frontier = state['frontier']
        parent   = state['parent']
        current  = state['current']
        dist_map = state.get('dist', {})

        for (x,y) in visited:
            if (x,y) not in (START,GOAL):
                self.ax.add_patch(Rectangle((x,y),1,1,facecolor='#dcdcdc',edgecolor='gray',alpha=0.9))

        fr_nodes = [node for (_d, node) in (frontier or [])] if self.algo_name=='Dijkstra' else list(frontier or [])
        for (x,y) in fr_nodes:
            if (x,y) not in (START,GOAL):
                self.ax.add_patch(Rectangle((x,y),1,1,facecolor='#4472C4',edgecolor='gray',alpha=0.8))

        if current not in (START,GOAL):
            self.ax.add_patch(Rectangle(current,1,1,facecolor='#FFC000',edgecolor='gray',alpha=0.95))

        path=[]
        if finished or state.get('action') == 'goal' or current == GOAL:
            path = self.reconstruct_path(parent, GOAL)
            for (x,y) in path:
                if (x,y) not in (START,GOAL):
                    self.ax.add_patch(Rectangle((x,y),1,1,facecolor='#FFF176',edgecolor='gray',alpha=0.95))

        self.draw_graph_overlay(parent, visited, frontier, current, path)

        # status text (moved to pseudocode area)
        depth = self.depth_of(current, parent) if current in parent else -1
        if self.algo_name in ('DFS','BFS'):
            extra = f"depth {depth if depth>=0 else '-'}"
        else:
            gn = dist_map.get(current, '-')
            extra = f"g(n) {gn}"
        self.info_text.set_text(
            f"{self.algo_name} — current {current} | frontier {len(fr_nodes)} | visited {len(visited)} | {extra} | Maze: {'Loopy' if self.use_loopy else 'Tree'}"
        )

        cur_frontier_set = set(fr_nodes)
        cur_parents_keys = set(parent.keys())
        add_frontier = list(cur_frontier_set - self.prev_sets['frontier'])
        add_visited  = list(visited - self.prev_sets['visited'])
        add_parents  = list(cur_parents_keys - self.prev_sets['parents'])
        self.prev_sets = {'visited': set(visited), 'frontier': set(cur_frontier_set), 'parents': set(cur_parents_keys)}

        p = parent.get(current)
        where = (f"start at {current}" if p is None else f"at {current} (parent {p})")
        action = state.get('action','')
        if action == 'init':    action_text = "initialize data structures."
        elif action == 'pop':   action_text = ("pop()" if self.algo_name=='DFS' else "dequeue()")
        elif action == 'visit': action_text = "mark current as visited."
        elif action == 'consider':
            nbrs_str = ", ".join(str(t) for t in state.get('neighbors', [])) or "none"
            action_text = f"consider neighbor(s): {nbrs_str}"
        elif action == 'push':
            nbrs_str = ", ".join(str(t) for t in state.get('neighbors', [])) or "none"
            action_text = (f"push onto stack: {nbrs_str}" if self.algo_name=='DFS' else f"enqueue: {nbrs_str}")
        elif action == 'deadend': action_text = "no unvisited neighbors -> dead end."
        elif action == 'goal':    action_text = f"reached GOAL {GOAL}."
        else:                     action_text = action

        deltas = []
        s1 = self.fmt_add("frontier", add_frontier, max_items=6)
        s2 = self.fmt_add("visited",  add_visited,  max_items=8)
        s3 = self.fmt_add("parents",  add_parents,  max_items=8)
        for s in (s1,s2,s3):
            if s: deltas.append(s)
        delta_line = " | ".join(deltas) if deltas else "(no changes)"
        data_line = f"frontier={len(fr_nodes)}, visited={len(visited)}"

        sections = [("WHERE", where), ("ACTION", action_text), ("Δ", delta_line), ("DATA", data_line)]
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
        elif k=='r':
            self.set_algo(self.algo_name)
        elif k=='l':
            self.use_loopy = not self.use_loopy
            set_maze(LOOPY_MAZE if self.use_loopy else TREE_MAZE)
            self.set_algo(self.algo_name)

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

        # Layout: legend sits right beneath the maze with a tiny, fixed gap.
        if aspect >= 1.5:  # wide
            left_rect  = [0.05, 0.10, 0.32, 0.80]
            maze_rect  = [0.45, 0.22, 0.46, 0.60]
        elif 1.15 <= aspect < 1.5:  # medium
            left_rect  = [0.04, 0.10, 0.36, 0.80]
            maze_rect  = [0.43, 0.22, 0.50, 0.58]
        else:  # narrow/tall
            left_rect  = [0.05, 0.05, 0.90, 0.30]
            maze_rect  = [0.08, 0.44, 0.84, 0.50]

        # message panel above maze; legend just below
        msg_rect    = [maze_rect[0], maze_rect[1] + maze_rect[3] + 0.012,
                       maze_rect[2], 0.16 if small_screen else 0.12]
        legend_h    = 0.045
        legend_rect = [maze_rect[0], max(0.04, maze_rect[1] - (legend_h + 0.012)),
                       maze_rect[2], legend_h]

        # Apply positions
        self.left_ax.set_position(left_rect)
        self.ax.set_position(maze_rect)
        self.msg_ax.set_position(msg_rect)
        self.legend_ax.set_position(legend_rect)

        # status line position stays under pseudocode
        self.info_text.set_position((0.02, 0.32))

        # Wrap width & fonts
        msg_px = fig_w_px * msg_rect[2]
        self.wrap_cols = max(42, min(120, int(msg_px / 8.4)))
        self.msg_fs = (9.6 if small_screen else 10.2) + 0.0020 * msg_px
        self.help_text.set_fontsize(self.msg_fs - (1.0 if small_screen else 0.8))
        self.line_spacing = 0.18 if small_screen else 0.20

        # Redraw legend and current state
        self.draw_custom_legend()
        if self.last_state is None and not initial:
            self.update_overlay(None)
        elif self.last_state is not None:
            self.update_overlay(self.last_state)

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
                "                parent[v] = u; push(v)",  # explore deeper
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
                "push(0,start)  # PQ on dist",
                "dist[start]=0; parent[start]=None",
                "while PQ not empty:",
                "    (du,u) = pop_min()",
                "    if u == goal: break",
                "    for v in neighbors(u):",
                "        nd = du + 1",
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
            if y < 0.30: break  # leave space for status/help
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
        return 6 if a == 'expand' else (5 if a=='deadend' else 3)

if __name__ == '__main__':
    viz = MazeViz()
    plt.show()
