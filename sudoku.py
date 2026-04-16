# sudoku_teach_left_right_stack_only_v17.py
# v17 = v14 + (safe) Mode toggle + two extra pseudocodes (While, For).
# NOTE: For maximum stability, all modes use the same proven recursive
#       event generator. Students still see different pseudocode/labels.

import copy
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button

# ---- Appearance & Layout Knobs ---------------------------------------------
GIVEN_NUM_COLOR   = 'black'
FILLED_NUM_COLOR  = 'tab:blue'
CELL_HL_COLOR     = 'tab:green'
CONFLICT_COLOR    = 'red'
STACK_TRUE_COLOR  = 'tab:green'
STACK_FALSE_COLOR = 'tab:red'
STACK_NEUTRAL     = 'black'

COMMENT_COLOR = '#708238'
PC_FONTSIZE_BASE  = 12

# >>> Call stack spacing controls:
STACK_AUTO_FIT = False   # ← original
STACK_Y0       = 0.68    # ← original
STACK_DY       = 0.20    # ← original
STACK_BOTTOM   = 0.10    # ← original
STACK_FONTSIZE = 10      # ← original
# ---------------------------------------------------------------------------

PUZZLE = [
    [0,0,0, 2,6,0, 7,0,1],
    [6,8,0, 0,7,0, 0,9,0],
    [1,9,0, 0,0,4, 5,0,0],
    [8,2,0, 1,0,0, 0,4,0],
    [0,0,4, 6,0,2, 9,0,0],
    [0,5,0, 0,0,3, 0,2,8],
    [0,0,9, 3,0,0, 0,7,4],
    [0,4,0, 0,5,0, 0,3,6],
    [7,0,3, 0,1,8, 0,0,0],
]

# ---------- Core helpers ----------
def count_empty(board):
    return sum(1 for r in range(9) for c in range(9) if board[r][c] == 0)

def select_cell_first_empty(board):
    for r in range(9):
        for c in range(9):
            if board[r][c] == 0:
                return (r, c)
    return None

def check_reason(board, r, c, d):  
    # #checks whether placing digit d at position (r,c) is valid. 
    # returns (True, None) if valid, (False, reason) if invalid.
    
    for j in range(9):
        if board[r][j] == d: #check row by trying out all columns in fixed row
            return False, {'rule':'row', 'conflicts':[(r,j)]}
    for i in range(9):
        if board[i][c] == d: #check column by trying out all rows in fixed column
            return False, {'rule':'col', 'conflicts':[(i,c)]}
    
    br, bc = (r//3)*3, (c//3)*3 #identify beginning row and column for the subsquare. // is a floor operator.
    for i in range(br, br+3):
        for j in range(bc, bc+3): #iterate columns then row first
            if board[i][j] == d:
                return False, {'rule':'box', 'conflicts':[(i,j)]}
    return True, None

def candidates_for(board, r, c): 
    #test number from 1 to 9, and only keep them if they satisfy the constraints, check_reason[0] = true
    return [d for d in range(1,10) if check_reason(board, r, c, d)[0]]

# ---------- Event generator (unchanged recursive; stable) ----------
def solve_steps(board, select_cell=select_cell_first_empty):
    def _solve(depth=0):
        k = count_empty(board)
        pos = select_cell(board)
        if pos is None:
            yield {'type':'base_case','k':k,'depth':depth}
            yield {'type':'solved','k':k,'depth':depth}
            return

        r,c = pos
        yield {'type':'select','cell':(r,c),'k':k,'depth':depth}
        cands = candidates_for(board, r, c)
        yield {'type':'build_candidates','cell':(r,c), 'cands':cands,'k':k,'depth':depth}
        yield {'type':'candidates','cell':(r,c),'cands':cands,'k':k,'depth':depth}
        yield {'type':'init_candidates', 'cell':(r,c), 'cands':cands, 'next_idx':0, 'k':k, 'depth':depth}
        for idx, d in enumerate(cands):
            yield {'type':'advance_candidate_index','cell':(r,c), 'digit': d, 'next_idx':idx, 'k':k, 'depth':depth}
            ok, reason = check_reason(board, r, c, d)
            yield {'type':'try','cell':(r,c),'digit':d,'valid':ok,'reason':reason,'k':k,'depth':depth}
            if not ok:
                continue

            board[r][c] = d
            yield {'type':'place','cell':(r,c),'digit':d,'k':k-1,'depth':depth}
            yield {'type':'push_decision', 'cell':(r,c), 'digit':d, 'cands':cands, 'next_idx':idx+1, 'k':k-1,'depth':depth}
            yield {'type':'recurse_enter','k':k-1,'depth':depth+1}
            outcome = None
            for ev in _solve(depth+1):
                yield ev
                if ev.get('type') == 'solved':
                    outcome = True
            yield {'type':'recurse_return','ok':bool(outcome),'k':count_empty(board),'depth':depth}

            if outcome:
                return True

            board[r][c] = 0
            yield {'type':'reset_cell', 'cell':(r,c), 'k':count_empty(board), 'depth':depth}
            yield {'type':'backtrack','cell':(r,c),'k':count_empty(board),'depth':depth}

        yield {'type':'dead_end','cell':(r,c),'cands':cands,'k':k,'depth':depth}
        return False
    yield from _solve()

def generate_event_trace(puzzle, select_cell=select_cell_first_empty):
    return list(solve_steps(copy.deepcopy(puzzle), select_cell))

# ---------- Viz ----------
class SudokuTeachViz:
    def __init__(self, puzzle):
        self.orig  = copy.deepcopy(puzzle)
        self.board = copy.deepcopy(puzzle)
        # NEW: mode cycle (purely visual, solver stays the same for stability)
        self.mode_cycle = ['Recur', 'While']
        self.mode_idx   = 0
        self.mode       = self.mode_cycle[self.mode_idx]

        self.fig = plt.figure(figsize=(13.4, 8.6))
        plt.subplots_adjust(left=0.03, right=0.99, top=0.95, bottom=0.12, wspace=0.06)

        outer = GridSpec(nrows=1, ncols=2, figure=self.fig, width_ratios=[1.45, 1.45])

        left = GridSpecFromSubplotSpec(3, 1, subplot_spec=outer[0,0],
                                       height_ratios=[0.82, 0.06, 0.12], hspace=0.02)
        self.ax_pc    = self.fig.add_subplot(left[0,0])
        self.ax_info  = self.fig.add_subplot(left[1,0])
        self.ax_stack = self.fig.add_subplot(left[2,0])

        self.ax_grid  = self.fig.add_subplot(outer[0,1])

        def silence(ax):
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_visible(False)
        for ax in (self.ax_pc, self.ax_info, self.ax_stack):
            silence(ax); ax.set_xlim(0,1); ax.set_ylim(0,1)

        # Grid
        self.ax_grid.set_xlim(0,9); self.ax_grid.set_ylim(0,9)
        self.ax_grid.set_aspect('equal'); self.ax_grid.invert_yaxis(); self.ax_grid.axis('off')
        for i in range(10):
            lw = 2.4 if i % 3 == 0 else 0.8
            self.ax_grid.plot([0,9],[i,i], linewidth=lw, color='black')
            self.ax_grid.plot([i,i],[0,9], linewidth=lw, color='black')

        self.status_text = self.fig.text(0.50, 0.965, "Ready: Press Next",
                                         ha='center', va='center', fontsize=14, fontweight='bold')

        # --- Pseudocode sets (right-arrow assignment; clearer names)
        self.pc_sets = {
            'Recur': [
                "solve(board):",
                "   -- Base Case -- ",
                "  (row, col) = select_next_empty_cell(board)",
                "  if (row, col) == None: return True            # board solved",
                "",
                "  cands = candidates_for_cell(board, row, col)",
                "",
                "  for num in cands:",
                "      board[row][col] = num                  # place guess on board",
                "      -- Recursive Case -- ",
                "      if solve(board):                       # recursive solve next cell",
                "           return True            ",
                "      else board[row][col] = 0               # reset & try next cand",
                "  return False                                 # none worked",
            ],

            'While': [
                "solve(board):                     # cand = candidates, idx = index", 
                "  decision_stack = []             # entry: (row, col, cands, next_cand_idx)",
                "  while count_empty_cells(board) > 0:",
                "       (row, col) = select_next_empty_cell(board)",
                "       cands = candidates_for_cell(board, row, col); & next_cand_idx = 0",
                "       while True:",
                "       if next_cand_idx == len(cands):          # tried all candidates so backtrack",
                "           if decision_stack is empty: return False   # unsolvable",
                "              else pop(decision_stack);            # restore previous entry",
                "              board[row][col] → 0; continue        # reset cell",
                "      cand = cands[next_cand_idx]   # next_cand_idx is a pointer for the candidate list",       
                "      next_cand_idx +=1",
                "      board[row][col] = cand                # place guess",
                "      push(decision_stack, (row, col, cands, next_cand_idx))  # remember where to resume",
                "      break                                   # move to next blank",
                "  return True                                   # solved",
            ],
        }


        # Draw initial pseudocode
        self._draw_pseudocode()

        # k badge
        self.k_text = self.ax_pc.text(0.965, 0.102, "k: —",
            fontsize=PC_FONTSIZE_BASE - 1, ha='right', va='center', family='monospace',
            bbox=dict(boxstyle="round,pad=0.25", fc=(0.95,0.95,0.95), ec="0.7"))
        self.k_text.set_visible(False)   # ← hide permanently
        # Info
        self.left_info = self.ax_info.text(0.02, 0.5, "", fontsize=10, ha='left', va='center', wrap=True, fontweight ='bold')

        # Stack header
        self.stack_header = self.ax_stack.text(0.02, 0.985, "Call stack (bottom = root):",
                           fontsize=STACK_FONTSIZE, family='monospace', ha='left', va='top')
        self.stack_drawn = []
        self.stack_frames = []  # {'depth','r','c','d','ret'}

        # Grid overlays
        self.cell_hl = Rectangle((0,0),1,1, fill=True, alpha=0.18, color=CELL_HL_COLOR)
        self.ax_grid.add_patch(self.cell_hl); self.cell_hl.set_visible(False)
        self.conflict_patches = []
        self.number_texts = [[None]*9 for _ in range(9)]
        self.draw_numbers()

        # Playback
        self.finished = False
        self.events  = []
        self.history = []
        self.index = -1
        self.autoplay = False
        self.timer = self.fig.canvas.new_timer(interval=180)
        self.timer.add_callback(self.auto_tick)

        self._init_buttons_under_grid()
        self.new_run()
    def set_k(self, k): #num_empty cells
        if hasattr(self, 'k_text'):
            self.k_text.set_text(f"k: {k}")
            self.k_text.set_visible(False)

    # ---- Pseudocode drawing & highlighting
    def _draw_pseudocode(self):
        # clear old texts
        for artist in getattr(self, "pc_texts", []):
            try: artist.remove()
            except Exception: pass
        self.pc_texts = []

        # clear highlight rect
        if hasattr(self, "pc_hl") and self.pc_hl in getattr(self.ax_pc, "patches", []):
            try: self.pc_hl.remove()
            except Exception: pass
        self.pc_hl = None

        # clear previous divider line(s)
        for ln in list(self.ax_pc.lines):
            try: ln.remove()
            except Exception: pass

        # draw current pseudocode
        lines = self.pc_sets.get(self.mode, self.pc_sets['Recur'])
        pc_fs = PC_FONTSIZE_BASE
        if self.mode == 'While':
            pc_fs -= 2         # smaller while-loop pseudocode
        n = len(lines)
        self.pc_y0 = 0.97
        bottom_margin = 0.02
        self.pc_dy = min(0.0666, (self.pc_y0 - bottom_margin) / max(1, n))
        for i, line in enumerate(lines):
            y = self.pc_y0 - i*self.pc_dy
            color = COMMENT_COLOR if '#' in line else 'black'
            t = self.ax_pc.text(0.02, y, line, fontsize=pc_fs, family='monospace',
                                ha='left', va='center', color=color)
            self.pc_texts.append(t)

        # highlight block (hidden until we select a line)
        self.pc_hl = Rectangle((0,0),0.001,0.001,
                               facecolor='yellow', alpha=0.18, zorder=0)
        self.ax_pc.add_patch(self.pc_hl); self.pc_hl.set_visible(False)

        # divider under last line
        last_y = self.pc_y0 - (n-1)*self.pc_dy
        divider_y = last_y - self.pc_dy*0.55
        self.ax_pc.plot([0.02, 0.98], [divider_y, divider_y], color='0.85', lw=1)

        self.fig.canvas.draw_idle()

    # ---- Buttons under grid (figure coords) ----
    def _init_buttons_under_grid(self):
        gb = self.ax_grid.get_position(self.fig)
        h = 0.06
        y = max(0.03, gb.y0 - 0.08)
        total_w = gb.x1 - gb.x0
        pad = 0.015
        bw = (total_w - pad*4) / 5.0
        xs = [gb.x0 + i*(bw+pad) for i in range(5)]
        def silence(ax):
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_visible(False)
        ax_next  = self.fig.add_axes([xs[0], y, bw, h]); silence(ax_next)
        ax_prev  = self.fig.add_axes([xs[1], y, bw, h]); silence(ax_prev)
        ax_auto  = self.fig.add_axes([xs[2], y, bw, h]); silence(ax_auto)
        ax_mode  = self.fig.add_axes([xs[3], y, bw, h]); silence(ax_mode)
        ax_reset = self.fig.add_axes([xs[4], y, bw, h]); silence(ax_reset)
        self.btn_next = Button(ax_next,  'Next')
        self.btn_prev = Button(ax_prev,  'Prev')
        self.btn_auto = Button(ax_auto,  'Auto')
        self.btn_mode = Button(ax_mode,  f"Mode: {self.mode}")
        self.btn_reset= Button(ax_reset, 'Reset')
        self.btn_next.on_clicked(self.on_next)
        self.btn_prev.on_clicked(self.on_prev)
        self.btn_auto.on_clicked(self.on_auto)
        self.btn_mode.on_clicked(self.on_mode)
        self.btn_reset.on_clicked(self.on_reset)

    # ---- Draw helpers ----
    def draw_numbers(self):
        for r in range(9):
            for c in range(9):
                t = self.number_texts[r][c]
                if t: t.remove()
                self.number_texts[r][c] = None
        for r in range(9):
            for c in range(9):
                v = self.board[r][c]
                if v != 0:
                    color = GIVEN_NUM_COLOR if self.orig[r][c] != 0 else FILLED_NUM_COLOR
                    self.number_texts[r][c] = self.ax_grid.text(
                        c+0.5, r+0.65, str(v), ha='center', va='center',
                        fontsize=16, fontweight='bold' if self.orig[r][c] != 0 else 'normal',
                        color=color
                    )
        self.fig.canvas.draw_idle()

    def set_status(self, headline, info=""):
        self.status_text.set_text(headline)
        self.left_info.set_text(info)

    def set_cell_highlight_color(self, color):
        self.cell_hl.set_facecolor(color)
        self.cell_hl.set_edgecolor(color)

    def set_cell_highlight(self, cell):
        if cell is None: self.cell_hl.set_visible(False)
        else:
            r,c = cell; self.cell_hl.set_xy((c,r))
            self.set_cell_highlight_color(CELL_HL_COLOR)   # always default to green
            self.cell_hl.set_visible(True)

    def clear_conflicts(self):
        for p in self.conflict_patches: p.remove()
        self.conflict_patches = []; self.fig.canvas.draw_idle()

    def show_conflicts(self, cells):
        self.clear_conflicts()
        for (r,c) in cells or []:
            rect = Rectangle((c,r),1,1, fill=True, alpha=0.14, color=CONFLICT_COLOR)
            self.ax_grid.add_patch(rect); self.conflict_patches.append(rect)
        self.fig.canvas.draw_idle()

    # ---- replace the whole highlight_pc() with:
    def highlight_pc(self, idx):
        # reset all text styles
        for i, t in enumerate(self.pc_texts):
            line = self.pc_sets.get(self.mode, self.pc_sets['Recur'])[i]
            t.set_color(COMMENT_COLOR if '#' in line else 'black')
            t.set_fontweight('normal')

        if idx is None or idx < 0 or idx >= len(self.pc_texts):
            self.pc_hl.set_visible(False)
            self.fig.canvas.draw_idle()
            return

        t = self.pc_texts[idx]
        t.set_color('tab:blue'); t.set_fontweight('bold')

        # compute tight bbox of the text in AXES coordinates
        renderer = self.fig.canvas.get_renderer()
        bbox_disp = t.get_window_extent(renderer=renderer)               # display coords (pixels)
        bbox_axes = bbox_disp.transformed(self.ax_pc.transAxes.inverted())  # → axes coords

        pad_x = 0.006  # small padding around the line (axes units)
        pad_y = 0.004

        x0 = max(0.02 - pad_x, 0.0)   # align with left margin you used for text
        x1 = min(bbox_axes.x1 + pad_x, 1.0)
        y0 = max(bbox_axes.y0 - pad_y, 0.0)
        y1 = min(bbox_axes.y1 + pad_y, 1.0)

        self.pc_hl.set_xy((x0, y0))
        self.pc_hl.set_width(x1 - x0)
        self.pc_hl.set_height(y1 - y0)
        self.pc_hl.set_visible(True)

        self.fig.canvas.draw_idle()


    def render_stack(self):
        for t in getattr(self, "stack_drawn", []):
            t.remove()
        self.stack_drawn = []

        rows = self.stack_frames[-7:]

        if STACK_AUTO_FIT:
            max_rows = max(1, min(7, len(rows)))
            top, bottom = 0.86, STACK_BOTTOM
            dy = (top - bottom) / max_rows
            y0 = top
        else:
            y0, dy = STACK_Y0, STACK_DY

        # ---- WHILE MODE: show decision_stack style ----
        if self.mode == 'While':
            self.stack_header.set_text("Decision stack (bottom = root):")

            for i, fr in enumerate(reversed(rows)):
                y = y0 - i * dy

                r = fr.get('r', 0) + 1
                c = fr.get('c', 0) + 1
                cands = fr.get('cands', [])
                next_idx = fr.get('next_idx', 0)

                disp = f"[{len(rows)-1-i}] (r{r},c{c}, cands={cands}, next_idx={next_idx})"

                self.stack_drawn.append(
                    self.ax_stack.text(
                        0.02, y, disp,
                        fontsize=STACK_FONTSIZE,
                        family='monospace',
                        ha='left', va='center',
                        color=STACK_NEUTRAL
                    )
                )

            self.fig.canvas.draw_idle()
            return

        # ---- RECUR MODE: original recursive call stack ----
        self.stack_header.set_text("Call stack (bottom = root):")

        for i, fr in enumerate(reversed(rows)):
            y = y0 - i * dy
            d = fr.get('d')
            ret = fr.get('ret')
            disp = f"[{fr['depth']}] r{fr['r']+1},c{fr['c']+1} = {'?' if d is None else d}"
            color = STACK_NEUTRAL
            if ret is True:
                disp += "  ← True"
                color = STACK_TRUE_COLOR
            if ret is False:
                disp += "  ← False"
                color = STACK_FALSE_COLOR

            self.stack_drawn.append(
                self.ax_stack.text(
                    0.02, y, disp,
                    fontsize=STACK_FONTSIZE,
                    family='monospace',
                    ha='left', va='center',
                    color=color
                )
            )

        self.fig.canvas.draw_idle()

    # ---- State/reset ----
    def reset_visual_state_only(self):
        self.finished = False
        self.board = copy.deepcopy(self.orig)
        self.draw_numbers()
        self.set_cell_highlight(None)
        self.clear_conflicts()
        self.set_status("Ready: Press Next","")
        self.highlight_pc(None)
        self.set_k(count_empty(self.board))
        self.history = []; self.index = -1
        self.stack_frames = []; self.render_stack()


    def new_run(self):
        self.events = generate_event_trace(self.orig)
        self.reset_visual_state_only()

    # ---- Event → UI ----
    def describe_event(self, event, replay_info=None):
        et = event['type']; k = event.get('k', count_empty(self.board))
        self.set_k(k)

        # Map events to an approximate line index for each pseudocode
        if self.mode == 'Recur':
            pc_map = {'base_case':3,'solved':3,'select':2,'build_candidates':5,'candidates':5,'try':99,'place':8,
                      'recurse_enter':10,'recurse_return':13,'backtrack':12,'dead_end':13}
        elif self.mode == 'While':
            pc_map = {'base_case':15,'solved':15,'select':3,'build_candidates':4,'candidates':4,'init_candidates':4, 'advance_candidate_index':10, 'try':99,'place':12,
                      'push_decision':13, 'recurse_enter':14,'recurse_return':8, 'reset_cell':9,'backtrack':9,'dead_end':6}

        pc_idx = pc_map.get(et, None)
        # ------------------ Recurs Mode -------------------------------------------
        if self.mode == 'Recur':
            if et == 'base_case':
                self.set_status("BASE CASE reached", "k == 0 → no empty cells remain.")
            elif et == 'solved':
                self.set_status("Solved!", "All cells filled consistently.")
            elif et == 'select':
                r,c = event['cell']
                self.set_status(f"Select next empty cell → r{r+1}, c{c+1}",
                                "We found a position to fill (problem shrinks by one cell).")
            elif et == 'build_candidates':
                r, c = event['cell']
                cands = event['cands']
                self.set_status(
                    f"Build candidate list for r{r+1}, c{c+1}",
                    f"Check digits 1–9 against constraints → valid candidates: {cands}."
                )
            elif et == 'candidates':
                r,c = event['cell']; cands = event['cands']
                self.set_status(f"Candidates for r{r+1}, c{c+1}: {cands}",
                            "Digits allowed by row, column, and 3×3 box rules.")
            
            elif et == 'try':
                r,c = event['cell']; d = event['digit']
                if event['valid']:
                    self.set_status(f"Try {d} at r{r+1}, c{c+1} ✓ valid",
                                    "No conflicts found → we may place and continue.")
                else:
                    reason = event['reason']; cells = reason['conflicts']
                    where = ", ".join([f"(r{ri+1},c{cj+1})" for (ri, cj) in cells])
                    self.set_status(f"Try {d} at r{r+1}, c{c+1} ✗ invalid",
                                f"Violates {reason['rule']}: {where}.")
            elif et == 'place':
                r,c = event['cell']; d = event['digit']
                self.set_status(f"Place {d} at r{r+1}, c{c+1}",
                            "Commit choice; solve the smaller subproblem.")

            elif et == 'recurse_enter':
                if self.mode == 'Recur':
                    self.set_status("RECURSIVE CALL", "Solve the smaller subproblem.")
                elif self.mode == 'While':
                    self.set_status("Push frame (conceptually)", "Same effect as descending once.")
            elif et == 'recurse_return':
                ok = event.get('ok', False)
                self.set_status("Return step",
                                "→ success bubbles up" if ok else "→ dead end, try next candidate")
            elif et == 'backtrack':
                r,c = event['cell']
                self.set_status(f"Backtrack: clear r{r+1}, c{c+1}",
                                "Undo the last placement; explore other options.")
            elif et == 'dead_end':
                r,c = event['cell']; cands = event['cands']
                self.set_status(f"Dead end at r{r+1}, c{c+1}",
                                f"No candidate leads to a solution (legal set was {cands}).")
        # ------------------ While Mode -------------------------------------------
        elif self.mode == 'While':
            if et == 'base_case':
                self.set_status("Loop Termination reached", "No empty cells remain, loop stops.")
            elif et == 'solved':
                self.set_status("Solved!", "All cells filled consistently.")
            elif et == 'select':
                r,c = event['cell']
                self.set_status(f"Select next empty cell → r{r+1}, c{c+1}",
                                "Loop scans board and chooses next empty cell.")
            elif et == 'build_candidates':
                r, c = event['cell']
                cands = event['cands']
                self.set_status(
                    f"Build candidate list for r{r+1}, c{c+1}",
                    f"Scan digits 1–9 using Sudoku rules. Valid candidates are {cands}."
                )
            elif et == 'candidates':
                r,c = event['cell']; cands = event['cands']
                self.set_status(f"Candidates for r{r+1}, c{c+1}: {cands}",
                            "Build candidate list for cell before iterating through.")
            elif et == 'init_candidates':
                r, c = event['cell']
                cands = event['cands']
                next_idx = event['next_idx']
                if self.mode == 'While':
                    self.set_status(
                        f"Initialize candidate list at r{r+1}, c{c+1}",
                        f"Set cands = {cands} and next_cand_idx = {next_idx}."
                    )
                else:
                    self.set_status(
                        f"Candidates ready for r{r+1}, c{c+1}",
                        f"Available candidates are {cands}."
                    )
            elif et == 'advance_candidate_index':
                r, c = event['cell']
                d = event['digit']
                idx = event['next_idx']
                if self.mode == 'While':
                    self.set_status(
                        f"Read candidate index {idx} at r{r+1}, c{c+1}",
                        f"Take cand = {d} from cands[{idx}] for checking."
                    )
                else:
                    self.set_status(
                        f"Try next candidate {d} at r{r+1}, c{c+1}",
                        "Proceed to validate this candidate."
                    )
            elif et == 'try':
                r,c = event['cell']; d = event['digit']
                if event['valid']:
                    self.set_status(f"Check {d} at r{r+1}, c{c+1} ✓ valid",
                                    "Candidate is valid, place guess.")
                else:
                    reason = event['reason']; cells = reason['conflicts']
                    where = ", ".join([f"(r{ri+1},c{cj+1})" for (ri, cj) in cells])
                    self.set_status(f"Try {d} at r{r+1}, c{c+1} ✗ invalid",
                                f"Violates {reason['rule']}: {where}.")
            elif et == 'place':
                r,c = event['cell']; d = event['digit']
                self.set_status(f"Place {d} at r{r+1}, c{c+1}",
                            "Store this decision in stack and continue to next cell.")
            elif et == 'push_decision':
                r, c = event['cell']
                d = event['digit']
                next_idx = event['next_idx']
                if self.mode == 'While':
                    self.set_status(
                        f"Push decision for r{r+1}, c{c+1}",
                        f"Store placed value {d} and resume later from next_cand_idx = {next_idx}."
                    )
                else:
                    self.set_status(
                        f"Record placement at r{r+1}, c{c+1}",
                        "Save the current decision before continuing."
                )
            elif et == 'recurse_enter':
                    self.set_status("Continue loop", "Continue to next cell.")

            elif et == 'recurse_return':
                ok = event.get('ok', False)
                self.set_status("Resume previous decision",
                                "Current stored decisions remain valid" if ok else "→ dead end, return to previous stored decision, and try next candidate")
                
            elif et == 'reset_cell':
                r, c = event['cell']
                if self.mode == 'While':
                    self.set_status(
                        f"Reset r{r+1}, c{c+1} to 0",
                        "This candidate path failed, so clear the cell and continue from the previous saved decision."
                    )
                else:
                    self.set_status(
                        f"Reset r{r+1}, c{c+1}",
                        "Undo the previous placement."
                    )
            elif et == 'backtrack':
                r,c = event['cell']
                self.set_status(f"Backtrack: clear r{r+1}, c{c+1}",
                                "Undo the last placement, resume the previous decision in stack.")
            elif et == 'dead_end':
                r,c = event['cell']; cands = event['cands']
                self.set_status(f"All candidates exhausted at r{r+1}, c{c+1}",
                                f"Candidate list {cands} exhausted, return to previous stored decision.")


        if replay_info is not None:
            self.left_info.set_text(replay_info)
        self.highlight_pc(pc_idx)
        return pc_idx, self.left_info.get_text(), k

    def apply_event(self, event):
        et = event['type']

        if et in ('base_case', 'solved'):
            self.set_cell_highlight(None)
            self.clear_conflicts()

        elif et == 'select':
            r, c = event['cell']
            self.set_cell_highlight((r, c))
            self.clear_conflicts()

            # Recur mode: selection creates a new call-frame
            if self.mode == 'Recur':
                self.stack_frames.append({
                    'depth': event['depth'],
                    'r': r,
                    'c': c,
                    'd': None,
                    'ret': None
                })
                self.render_stack()

            # While mode: do NOT push here
            # decision_stack should only be updated by push_decision

        elif et == 'try':
            self.clear_conflicts()
            if not event['valid'] and event.get('reason'):
                self.show_conflicts(event['reason'].get('conflicts', []))

        elif et == 'place':
            r, c = event['cell']
            d = event['digit']
            self.board[r][c] = d
            self.draw_numbers()
            self.clear_conflicts()

            # Recur mode: update top recursive frame with chosen digit
            if self.mode == 'Recur':
                if self.stack_frames and self.stack_frames[-1]['r'] == r and self.stack_frames[-1]['c'] == c:
                    self.stack_frames[-1]['d'] = d
                    self.stack_frames[-1]['ret'] = None   # clear stale False/True
                    self.render_stack()

            # While mode: board changes here, but stack push happens only at push_decision

        elif et == 'push_decision':
            if self.mode == 'While':
                r, c = event['cell']
                d = event['digit']
                cands = event['cands']
                next_idx = event['next_idx']

                self.stack_frames.append({
                    'r': r,
                    'c': c,
                    'cands': list(cands),
                    'next_idx': next_idx,
                    'digit': d,
                    'ret': None
                })
                self.render_stack()

        elif et == 'recurse_return':
            ok = event['ok']

            if self.mode == 'Recur':
                if self.stack_frames:
                    self.stack_frames[-1]['ret'] = ok
                    self.render_stack()

            # While mode: do not mutate stack here
            # this is just a teaching/status event

        elif et == 'reset_cell':
            r, c = event['cell']
            self.board[r][c] = 0
            self.draw_numbers()
            self.clear_conflicts()

            # Do not pop here.
            # Let backtrack handle the logical removal from stack.

        elif et == 'backtrack':
            r, c = event['cell']
            self.board[r][c] = 0
            self.draw_numbers()
            self.clear_conflicts()

            if self.mode == 'Recur':
                # keep current frame, but restore active-cell color back to green
                self.set_cell_highlight((r, c))
                self.set_cell_highlight_color(CELL_HL_COLOR)

                if self.stack_frames and self.stack_frames[-1]['r'] == r and self.stack_frames[-1]['c'] == c:
                    self.stack_frames[-1]['d'] = None
                    self.stack_frames[-1]['ret'] = False
                    self.render_stack()

            elif self.mode == 'While':
                self.set_cell_highlight((r, c))
                self.set_cell_highlight_color(CELL_HL_COLOR)

                if self.stack_frames:
                    self.stack_frames.pop()
                    self.render_stack()

        elif et == 'dead_end':
            self.clear_conflicts()

            # show current cell as red on dead-end
            if 'cell' in event:
                r, c = event['cell']
                self.set_cell_highlight((r, c))
                self.set_cell_highlight_color(CONFLICT_COLOR)

            if self.mode == 'Recur':
                if self.stack_frames:
                    self.stack_frames[-1]['ret'] = False
                    self.render_stack()
                    self.stack_frames.pop()
                    self.render_stack()
            # While mode: do not pop here
            # dead_end is explanatory; actual stack change happens at backtrack

        return self.describe_event(event)
    # ---- Buttons ----
    def on_next(self, _):
        if self.finished:
            return

        while True:
            nxt = self.index + 1
            if nxt >= len(self.events):
                self.set_status("Finished. Press Reset to run again.", self.left_info.get_text())
                self.autoplay = False
                self.timer.stop()
                return

            ev = self.events[nxt]

            # Skip While-only teaching events when in Recur mode
            if self.mode == 'Recur' and ev.get('type') in {
                'init_candidates',
                'advance_candidate_index',
                'push_decision',
                'reset_cell',
            }:
                self.index = nxt
                continue

            self.apply_event(ev)

            if ev.get('type') == 'solved':
                self.finished = True
                self.autoplay = False
                try:
                    self.timer.stop()
                except Exception:
                    pass

            hi_cell = None
            if self.cell_hl.get_visible():
                x, y = self.cell_hl.get_xy()
                hi_cell = (int(y), int(x))

            self.history = self.history[:self.index+1]
            self.history.append((
                copy.deepcopy(self.board),
                copy.deepcopy(ev),
                hi_cell,
                copy.deepcopy(self.stack_frames)
            ))
            self.index = nxt
            break

    def on_prev(self, _):
        if self.index < 0:
            self.reset_visual_state_only()
            return

        while True:
            self.index -= 1

            if self.index < 0:
                self.reset_visual_state_only()
                return

            board, ev, hi_cell, stack_frames = self.history[self.index]

            # Skip While-only teaching events when in Recur mode
            if self.mode == 'Recur' and ev.get('type') in {
                'init_candidates',
                'advance_candidate_index',
                'push_decision',
                'reset_cell',
            }:
                continue

            self.board = copy.deepcopy(board)
            self.draw_numbers()
            self.set_cell_highlight(hi_cell)
            self.clear_conflicts()
            self.stack_frames = copy.deepcopy(stack_frames)
            self.render_stack()
            self.describe_event(ev)
            return

    def on_auto(self, _):
        self.autoplay = not self.autoplay
        self.btn_auto.label.set_text('Stop' if self.autoplay else 'Auto')
        (self.timer.start() if self.autoplay else self.timer.stop())

    def on_mode(self, _):
        # cycle modes (visual only) safely
        self.autoplay = False
        try: self.timer.stop()
        except Exception: pass
        self.mode_idx = (self.mode_idx + 1) % len(self.mode_cycle)
        self.mode = self.mode_cycle[self.mode_idx]
        self.btn_mode.label.set_text(f"Mode: {self.mode}")
        self._draw_pseudocode()
        # keep the same event trace (engine unchanged), just reset visuals
        self.reset_visual_state_only()

    def auto_tick(self):
        if self.autoplay: self.on_next(None)

    def on_reset(self, _):
        self.autoplay = False; self.btn_auto.label.set_text('Auto'); self.timer.stop()
        self.new_run()

# ---------- Run ----------
if __name__ == "__main__":
    viz = SudokuTeachViz(PUZZLE)
    plt.show()
