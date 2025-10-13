# sudoku_teach_left_right_stack_only_v14.py
# Stack-only visualizer with:
# - Divider aligned under the last pseudocode line
# - Manual OR auto-fit spacing for the call stack (set STACK_AUTO_FIT)
# - Buttons under the grid; givens black, solver fills blue; current cell green; conflicts red

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

PC_FONTSIZE_BASE  = 12
STACK_FONTSIZE    = 11

# >>> Call stack spacing controls:
STACK_AUTO_FIT = False      # set to False to use the manual y0/dy below
STACK_Y0       = 0.68       # top Y for first stack line (used when AUTO_FIT=False)
STACK_DY       = 0.20       # vertical step between stack lines (used when AUTO_FIT=False)
STACK_BOTTOM   = 0.10       # min bottom margin (auto-fit mode only)
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
    for j in range(9):
        if board[r][j] == d:
            return False, {'rule':'row', 'conflicts':[(r,j)]}
    for i in range(9):
        if board[i][c] == d:
            return False, {'rule':'col', 'conflicts':[(i,c)]}
    br, bc = (r//3)*3, (c//3)*3
    for i in range(br, br+3):
        for j in range(bc, bc+3):
            if board[i][j] == d:
                return False, {'rule':'box', 'conflicts':[(i,j)]}
    return True, None

def candidates_for(board, r, c):
    return [d for d in range(1,10) if check_reason(board, r, c, d)[0]]

# ---------- Step generator ----------
def solve_steps(board, select_cell=select_cell_first_empty):
    def _solve(depth=0):
        k = count_empty(board)
        pos = select_cell(board)
        if pos is None:
            yield {'type':'base_case','k':k,'depth':depth}
            yield {'type':'solved','k':k,'depth':depth}
            return True

        r,c = pos
        yield {'type':'select','cell':(r,c),'k':k,'depth':depth}
        cands = candidates_for(board, r, c)
        yield {'type':'candidates','cell':(r,c),'cands':cands,'k':k,'depth':depth}

        for d in range(1,10):
            ok, reason = check_reason(board, r, c, d)
            yield {'type':'try','cell':(r,c),'digit':d,'valid':ok,'reason':reason,'k':k,'depth':depth}
            if not ok:
                continue

            board[r][c] = d
            yield {'type':'place','cell':(r,c),'digit':d,'k':k-1,'depth':depth}

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

        self.fig = plt.figure(figsize=(13.4, 7.4))
        plt.subplots_adjust(left=0.03, right=0.99, top=0.95, bottom=0.12, wspace=0.06)

        outer = GridSpec(nrows=1, ncols=2, figure=self.fig, width_ratios=[1.45, 1.45])

        left = GridSpecFromSubplotSpec(3, 1, subplot_spec=outer[0,0],
                                       height_ratios=[0.76, 0.08, 0.16], hspace=0.10)
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

        # --- Pseudocode
        pc_fs = PC_FONTSIZE_BASE
        self.pc_lines = [
            "def solve(board):",
            "    k ← empty_cells(board)        # input size",
            "",
            "    # BASE CASE",
            "    if k == 0: return True        # solved",
            "",
            "    # GENERAL CASE",
            "    (r, c) ← select_cell(board)    # selects next empty cell",
            "    C ← candidates(board, r, c)    # find set of valid guesses",
            "    for d in 1 to 9:",
            "        if d ∈ C:",
            "            board[r][c] ← d        # guess",
            "            # RECURSIVE CALL ↓",
            "            if solve(board): return True",
            "            board[r][c] ← 0     # backtrack, try next guess",
            "    return False",
        ]
        n = len(self.pc_lines)
        self.pc_y0 = 0.95
        self.pc_dy = min(0.07, (self.pc_y0 - 0.06) / max(1, n))
        self.pc_texts = []
        for i, line in enumerate(self.pc_lines):
            y = self.pc_y0 - i*self.pc_dy
            t = self.ax_pc.text(0.02, y, line, fontsize=pc_fs, family='monospace',
                                ha='left', va='center')
            self.pc_texts.append(t)
        self.pc_hl = Rectangle((0.015, self.pc_y0 - self.pc_dy), 0.97, self.pc_dy*0.9,
                               facecolor='yellow', alpha=0.18, zorder=0)
        self.ax_pc.add_patch(self.pc_hl); self.pc_hl.set_visible(False)

        # k badge
        self.k_text = self.ax_pc.text(0.965, 0.92, "k: —",
            fontsize=pc_fs, ha='right', va='center', family='monospace',
            bbox=dict(boxstyle="round,pad=0.25", fc=(0.95,0.95,0.95), ec="0.7"))

        # Divider directly under pseudocode (just below "return False")
        last_y = self.pc_y0 - (n-1)*self.pc_dy
        divider_y = last_y - self.pc_dy*0.55
        self.ax_pc.plot([0.02, 0.98], [divider_y, divider_y], color='0.85', lw=1)

        # Info
        self.left_info = self.ax_info.text(0.02, 0.5, "", fontsize=10, ha='left', va='center', wrap=True)

        # Stack header (no divider here)
        self.ax_stack.text(0.02, 0.98, "Call stack (bottom = root):",
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
        self.events  = []
        self.history = []
        self.index = -1
        self.autoplay = False
        self.timer = self.fig.canvas.new_timer(interval=180)
        self.timer.add_callback(self.auto_tick)

        self._init_buttons_under_grid()
        self.new_run()

    # ---- Buttons under grid (figure coords) ----
    def _init_buttons_under_grid(self):
        gb = self.ax_grid.get_position(self.fig)
        h = 0.06
        y = max(0.03, gb.y0 - 0.08)
        total_w = gb.x1 - gb.x0
        pad = 0.015
        bw = (total_w - pad*3) / 4.0
        xs = [gb.x0 + i*(bw+pad) for i in range(4)]
        def silence(ax):
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_visible(False)
        ax_next  = self.fig.add_axes([xs[0], y, bw, h]); silence(ax_next)
        ax_prev  = self.fig.add_axes([xs[1], y, bw, h]); silence(ax_prev)
        ax_auto  = self.fig.add_axes([xs[2], y, bw, h]); silence(ax_auto)
        ax_reset = self.fig.add_axes([xs[3], y, bw, h]); silence(ax_reset)
        self.btn_next = Button(ax_next,  'Next')
        self.btn_prev = Button(ax_prev,  'Prev')
        self.btn_auto = Button(ax_auto,  'Auto')
        self.btn_reset= Button(ax_reset, 'Reset')
        self.btn_next.on_clicked(self.on_next)
        self.btn_prev.on_clicked(self.on_prev)
        self.btn_auto.on_clicked(self.on_auto)
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

    def set_cell_highlight(self, cell):
        if cell is None: self.cell_hl.set_visible(False)
        else:
            r,c = cell; self.cell_hl.set_xy((c,r)); self.cell_hl.set_visible(True)

    def clear_conflicts(self):
        for p in self.conflict_patches: p.remove()
        self.conflict_patches = []; self.fig.canvas.draw_idle()

    def show_conflicts(self, cells):
        self.clear_conflicts()
        for (r,c) in cells or []:
            rect = Rectangle((c,r),1,1, fill=True, alpha=0.14, color=CONFLICT_COLOR)
            self.ax_grid.add_patch(rect); self.conflict_patches.append(rect)
        self.fig.canvas.draw_idle()

    def highlight_pc(self, idx):
        for t in self.pc_texts: t.set_color('black'); t.set_fontweight('normal')
        if idx is None: self.pc_hl.set_visible(False)
        else:
            self.pc_texts[idx].set_color('tab:blue'); self.pc_texts[idx].set_fontweight('bold')
            y = self.pc_y0 - idx*self.pc_dy - self.pc_dy*0.45
            self.pc_hl.set_y(y); self.pc_hl.set_visible(True)
        self.fig.canvas.draw_idle()

    def render_stack(self):
        for t in self.stack_drawn: t.remove()
        self.stack_drawn = []
        rows = self.stack_frames[-7:]

        if STACK_AUTO_FIT:
            max_rows = max(1, min(7, len(rows)))
            top, bottom = 0.86, STACK_BOTTOM
            dy = (top - bottom) / max_rows
            y0 = top
        else:
            y0, dy = STACK_Y0, STACK_DY

        for i, fr in enumerate(reversed(rows)):
            y = y0 - i*dy
            d = fr.get('d'); ret = fr.get('ret')
            disp = f"[{fr['depth']}] r{fr['r']+1},c{fr['c']+1} = {'?' if d is None else d}"
            color = STACK_NEUTRAL
            if ret is True:  disp += "  ← True";  color = STACK_TRUE_COLOR
            if ret is False: disp += "  ← False"; color = STACK_FALSE_COLOR
            self.stack_drawn.append(
                self.ax_stack.text(0.02, y, disp, fontsize=STACK_FONTSIZE,
                                   family='monospace', ha='left', va='center', color=color)
            )
        self.fig.canvas.draw_idle()

    def set_k(self, k): self.k_text.set_text(f"k: {k}")

    # ---- State/reset ----
    def reset_visual_state_only(self):
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
        pc_idx = None
        if et == 'base_case':
            self.set_status("BASE CASE reached", "k == 0 → no empty cells remain."); pc_idx = 4
        elif et == 'solved':
            self.set_status("Solved!", "All cells filled consistently."); pc_idx = 4
        elif et == 'select':
            r,c = event['cell']
            self.set_status(f"Select next empty cell → r{r+1}, c{c+1}",
                            "We found a position to fill (problem shrinks by one cell).")
            pc_idx = 7
        elif et == 'candidates':
            r,c = event['cell']; cands = event['cands']
            self.set_status(f"Candidates for r{r+1}, c{c+1}: {cands}",
                            "Digits allowed by row, column, and 3×3 box rules.")
            pc_idx = 8
        elif et == 'try':
            r,c = event['cell']; d = event['digit']
            if event['valid']:
                self.set_status(f"Try {d} at r{r+1}, c{c+1} ✓ valid",
                                "No conflicts found → we may place and recurse.")
            else:
                reason = event['reason']; cells = reason['conflicts']
                where = ", ".join([f"(r{ri+1},c{cj+1})" for (ri, cj) in cells])
                self.set_status(f"Try {d} at r{r+1}, c{c+1} ✗ invalid", f"Violates {reason['rule']}: {where}.")
            pc_idx = 10
        elif et == 'place':
            r,c = event['cell']; d = event['digit']
            self.set_status(f"Place {d} at r{r+1}, c{c+1}", "Commit choice; solve the smaller subproblem.")
            pc_idx = 11
        elif et == 'recurse_enter':
            k_after = k; k_before = k_after + 1
            self.set_status("RECURSIVE CALL", f"k decreases: {k_before} → {k_after} (solve subproblem)")
            pc_idx = 13
        elif et == 'recurse_return':
            ok = event['ok']
            self.set_status("Return from recursion",
                            "→ True (propagate success)" if ok else "→ False (backtrack & try next d)")
            pc_idx = 13 if ok else 14
        elif et == 'backtrack':
            r,c = event['cell']
            self.set_status(f"Backtrack: clear r{r+1}, c{c+1}", "Undo the last placement; explore other options.")
            pc_idx = 14
        elif et == 'dead_end':
            r,c = event['cell']; cands = event['cands']
            self.set_status(f"Dead end at r{r+1}, c{c+1}", f"No candidate leads to a solution (legal set was {cands}).")
            pc_idx = 15
        if replay_info is not None:
            self.left_info.set_text(replay_info)
        self.highlight_pc(pc_idx)
        return pc_idx, self.left_info.get_text(), k

    def apply_event(self, event):
        et = event['type']
        if et in ('base_case','solved'):
            self.set_cell_highlight(None); self.clear_conflicts()
        elif et == 'select':
            r,c = event['cell']
            self.stack_frames.append({'depth':event['depth'],'r':r,'c':c,'d':None,'ret':None})
            self.render_stack()
            self.set_cell_highlight((r,c)); self.clear_conflicts()
        elif et == 'try':
            self.clear_conflicts()
            if not event['valid'] and event.get('reason'):
                self.show_conflicts(event['reason'].get('conflicts', []))
        elif et == 'place':
            r,c = event['cell']; d = event['digit']
            self.board[r][c] = d
            if self.stack_frames and self.stack_frames[-1]['r']==r and self.stack_frames[-1]['c']==c:
                self.stack_frames[-1]['d'] = d; self.render_stack()
            self.draw_numbers(); self.clear_conflicts()
        elif et == 'recurse_return':
            ok = event['ok']
            if self.stack_frames:
                self.stack_frames[-1]['ret'] = ok; self.render_stack()
        elif et == 'dead_end':
            if self.stack_frames and self.stack_frames[-1]['d'] is None:
                self.stack_frames[-1]['ret'] = False; self.render_stack()
                self.stack_frames.pop(); self.render_stack()
            self.clear_conflicts()
        elif et == 'backtrack':
            r,c = event['cell']
            self.board[r][c] = 0; self.draw_numbers()
            if self.stack_frames and self.stack_frames[-1]['r']==r and self.stack_frames[-1]['c']==c:
                self.stack_frames[-1]['ret'] = False; self.render_stack()
                self.stack_frames.pop(); self.render_stack()
            self.clear_conflicts()
        return self.describe_event(event)

    # ---- Buttons ----
    def on_next(self, _):
        nxt = self.index + 1
        if nxt >= len(self.events):
            self.set_status("Finished. Press Reset to run again.", self.left_info.get_text())
            self.autoplay = False; self.timer.stop(); return
        ev = self.events[nxt]
        self.apply_event(ev)
        hi_cell = None
        if self.cell_hl.get_visible():
            x,y = self.cell_hl.get_xy(); hi_cell = (int(y), int(x))
        self.history = self.history[:self.index+1]
        self.history.append((copy.deepcopy(self.board), copy.deepcopy(ev), hi_cell,
                             copy.deepcopy(self.stack_frames)))
        self.index = nxt

    def on_prev(self, _):
        if self.index < 0:
            self.reset_visual_state_only(); return
        self.index -= 1
        if self.index >= 0:
            board, ev, hi_cell, stack_frames = self.history[self.index]
            self.board = copy.deepcopy(board); self.draw_numbers()
            self.set_cell_highlight(hi_cell); self.clear_conflicts()
            self.stack_frames = copy.deepcopy(stack_frames); self.render_stack()
            self.describe_event(ev)
        else:
            self.reset_visual_state_only()

    def on_auto(self, _):
        self.autoplay = not self.autoplay
        self.btn_auto.label.set_text('Stop' if self.autoplay else 'Auto')
        (self.timer.start() if self.autoplay else self.timer.stop())

    def auto_tick(self):
        if self.autoplay: self.on_next(None)

    def on_reset(self, _):
        self.autoplay = False; self.btn_auto.label.set_text('Auto'); self.timer.stop()
        self.new_run()

# ---------- Run ----------
if __name__ == "__main__":
    viz = SudokuTeachViz(PUZZLE)
    plt.show()
