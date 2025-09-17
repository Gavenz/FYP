# sudoku_teach_left_right_v3.py
# Sudoku backtracking teaching visualizer
# - Left panel: pseudocode (no numbering), explicit "Input size", "BASE CASE", "GENERAL CASE"
# - Live input size k = # empty cells
# - Right panel: grid with selection + conflict highlights
# - Controls: Next / Prev / Auto / Reset
# - Improvements:
#     * Better left-panel layout (no overlap) + divider
#     * Fixed title ("Backtracking Recursive Function on Sudoku")
#     * Prev works without Reset using a precomputed event trace

import copy
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button

# =========================
# Sample puzzle (0 = empty)
# =========================
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

# =========================
# Helpers
# =========================
def count_empty(board):
    return sum(1 for r in range(9) for c in range(9) if board[r][c] == 0)

def select_cell_first_empty(board):
    for r in range(9):
        for c in range(9):
            if board[r][c] == 0:
                return (r, c)
    return None

def check_reason(board, r, c, d):
    """Return (ok, reason or None). reason={'rule':'row'|'col'|'box','conflicts':[(ri,cj),...]}"""
    # Row
    for j in range(9):
        if board[r][j] == d:
            return False, {'rule': 'row', 'conflicts': [(r, j)]}
    # Column
    for i in range(9):
        if board[i][c] == d:
            return False, {'rule': 'col', 'conflicts': [(i, c)]}
    # Box
    br, bc = (r // 3) * 3, (c // 3) * 3
    for i in range(br, br + 3):
        for j in range(bc, bc + 3):
            if board[i][j] == d:
                return False, {'rule': 'box', 'conflicts': [(i, j)]}
    return True, None

def candidates_for(board, r, c):
    return [d for d in range(1, 10) if check_reason(board, r, c, d)[0]]

# =========================
# Step generator (yields micro-steps with 'k')
# =========================
def solve_steps(board, select_cell=select_cell_first_empty):
    """
    Events include a live 'k' (input size = empty cells).
    Types: base_case, solved, select, candidates, try, place,
           recurse_enter, recurse_return, backtrack, dead_end
    """
    def _solve(depth=0):
        k = count_empty(board)  # <-- INPUT SIZE
        pos = select_cell(board)
        if pos is None:
            yield {'type': 'base_case', 'k': k, 'depth': depth}
            yield {'type': 'solved', 'k': k, 'depth': depth}
            return True

        r, c = pos
        yield {'type': 'select', 'cell': (r, c), 'k': k, 'depth': depth}
        cands = candidates_for(board, r, c)
        yield {'type': 'candidates', 'cell': (r, c), 'cands': cands, 'k': k, 'depth': depth}

        for d in range(1, 10):
            ok, reason = check_reason(board, r, c, d)
            yield {'type': 'try', 'cell': (r, c), 'digit': d, 'valid': ok, 'reason': reason, 'k': k, 'depth': depth}
            if not ok:
                continue

            board[r][c] = d
            yield {'type': 'place', 'cell': (r, c), 'digit': d, 'k': k - 1, 'depth': depth}

            yield {'type': 'recurse_enter', 'k': k - 1, 'depth': depth + 1}
            outcome = None
            for ev in _solve(depth + 1):
                yield ev
                if ev.get('type') == 'solved':
                    outcome = True
            yield {'type': 'recurse_return', 'ok': bool(outcome), 'k': count_empty(board), 'depth': depth}

            if outcome:
                return True

            board[r][c] = 0
            yield {'type': 'backtrack', 'cell': (r, c), 'k': count_empty(board), 'depth': depth}

        yield {'type': 'dead_end', 'cell': (r, c), 'cands': cands, 'k': k, 'depth': depth}
        return False

    yield from _solve()

def generate_event_trace(puzzle, select_cell=select_cell_first_empty):
    """Precompute the full event list so Prev->Next works without Reset."""
    temp = copy.deepcopy(puzzle)
    return list(solve_steps(temp, select_cell=select_cell))

# =========================
# Visualization
# =========================
class SudokuTeachViz:
    def __init__(self, puzzle):
        self.orig = copy.deepcopy(puzzle)
        self.board = copy.deepcopy(puzzle)

        # ----- Figure: Left (pseudocode) | Right (grid)
        self.fig = plt.figure(figsize=(11, 6.8))

        gs = gridspec.GridSpec(2, 2, height_ratios=[18, 4], width_ratios=[1.05, 1.6])
        self.ax_left  = self.fig.add_subplot(gs[0, 0])
        self.ax_grid  = self.fig.add_subplot(gs[0, 1])
        self.ax_btns  = self.fig.add_subplot(gs[1, :])
        self.ax_btns.axis('off')

        plt.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.12, wspace=0.12, hspace=0.12)

        # Status banner (dynamic)
        self.status_text = self.fig.text(0.50, 0.965, "Ready: Press Next",
                                         ha='center', va='center', fontsize=14, fontweight='bold')

        # ----- Left panel: pseudocode (tighter spacing) + divider + live k
        self.ax_left.axis('off')
        self.ax_left.set_xlim(0, 1)
        self.ax_left.set_ylim(0, 1)

        self.pc_lines = [
            "Input size  k = empty_cells(board)",            # 0
            "",                                              # 1 spacer
            "# BASE CASE",                                   # 2
            "if k == 0: return True   # solved",             # 3
            "",                                              # 4 spacer
            "# GENERAL CASE",                                # 5
            "(r, c) ← select_cell(board)",                   # 6
            "C ← candidates(board, r, c)",                   # 7
            "for d in C:",                                   # 8
            "    board[r][c] ← d",                           # 9
            "    if solve(board): return True",              # 10
            "    board[r][c] ← 0",                           # 11
            "return False",                                  # 12
        ]
        self.pc_texts = []
        self.pc_y0 = 0.93         # tightened spacing
        self.pc_dy = 0.070
        for i, line in enumerate(self.pc_lines):
            y = self.pc_y0 - i * self.pc_dy
            t = self.ax_left.text(0.04, y, line, fontsize=12, family='monospace', ha='left', va='center')
            self.pc_texts.append(t)

        # highlight rectangle behind active line
        self.pc_hl = Rectangle((0.02, 0.90), 0.96, self.pc_dy*0.9, facecolor='yellow', alpha=0.18, zorder=0)
        self.ax_left.add_patch(self.pc_hl)
        self.pc_hl.set_visible(False)

        # Live input size k display
        self.k_text = self.ax_left.text(0.96, 0.92, "k: —",
                                        fontsize=12, ha='right', va='center', family='monospace',
                                        bbox=dict(boxstyle="round,pad=0.25", fc=(0.95,0.95,0.95), ec="0.7"))

        # Divider to prevent overlap
        self.ax_left.plot([0.04, 0.96], [0.08, 0.08], color='0.82', lw=1, clip_on=False)

        # Brief explanatory line (moved lower)
        self.left_info = self.ax_left.text(0.04, 0.02, "", fontsize=10, ha='left', va='bottom', wrap=True)

        # ----- Grid area
        self.ax_grid.set_xlim(0, 9)
        self.ax_grid.set_ylim(0, 9)
        self.ax_grid.set_aspect('equal')
        self.ax_grid.invert_yaxis()
        self.ax_grid.axis('off')

        # draw lines
        for i in range(10):
            lw = 2.4 if i % 3 == 0 else 0.8
            self.ax_grid.plot([0, 9], [i, i], linewidth=lw, color='black')
            self.ax_grid.plot([i, i], [0, 9], linewidth=lw, color='black')

        # current cell highlight
        self.cell_hl = Rectangle((0, 0), 1, 1, fill=True, alpha=0.15, color='tab:blue')
        self.ax_grid.add_patch(self.cell_hl)
        self.cell_hl.set_visible(False)

        # conflict patches
        self.conflict_patches = []

        # numbers
        self.number_texts = [[None] * 9 for _ in range(9)]
        self.draw_numbers()

        # ----- Buttons
        self.btn_next = Button(self.fig.add_axes([0.10, 0.04, 0.15, 0.06]), 'Next')
        self.btn_prev = Button(self.fig.add_axes([0.30, 0.04, 0.15, 0.06]), 'Prev')
        self.btn_auto = Button(self.fig.add_axes([0.50, 0.04, 0.15, 0.06]), 'Auto')
        self.btn_reset= Button(self.fig.add_axes([0.70, 0.04, 0.15, 0.06]), 'Reset')

        self.btn_next.on_clicked(self.on_next)
        self.btn_prev.on_clicked(self.on_prev)
        self.btn_auto.on_clicked(self.on_auto)
        self.btn_reset.on_clicked(self.on_reset)

        # ----- Event trace + history (no generator)
        self.events = []      # full list of events for this run
        self.history = []     # snapshots of applied events
        self.index = -1       # pointer into history/events
        self.autoplay = False
        self.timer = self.fig.canvas.new_timer(interval=180)
        self.timer.add_callback(self.auto_tick)

        self.new_run()

    # ========= Drawing helpers =========
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
                    is_given = (self.orig[r][c] != 0)
                    self.number_texts[r][c] = self.ax_grid.text(
                        c + 0.5, r + 0.65, str(v),
                        ha='center', va='center',
                        fontsize=16, fontweight='bold' if is_given else 'normal'
                    )
        self.fig.canvas.draw_idle()

    def set_status(self, headline, info=""):
        self.status_text.set_text(headline)
        self.left_info.set_text(info)

    def set_cell_highlight(self, cell):
        if cell is None:
            self.cell_hl.set_visible(False)
        else:
            r, c = cell
            self.cell_hl.set_xy((c, r))
            self.cell_hl.set_visible(True)

    def clear_conflicts(self):
        for p in self.conflict_patches:
            p.remove()
        self.conflict_patches = []
        self.fig.canvas.draw_idle()

    def show_conflicts(self, cells):
        self.clear_conflicts()
        for (r, c) in cells or []:
            rect = Rectangle((c, r), 1, 1, fill=True, alpha=0.14, color='red')
            self.ax_grid.add_patch(rect)
            self.conflict_patches.append(rect)
        self.fig.canvas.draw_idle()

    def highlight_pc(self, idx):
        """Highlight pseudocode line by index (0-based)."""
        for t in self.pc_texts:
            t.set_color('black')
            t.set_fontweight('normal')
        if idx is None:
            self.pc_hl.set_visible(False)
        else:
            self.pc_texts[idx].set_color('tab:blue')
            self.pc_texts[idx].set_fontweight('bold')
            y = self.pc_y0 - idx * self.pc_dy - self.pc_dy*0.45
            self.pc_hl.set_y(y)
            self.pc_hl.set_visible(True)
        self.fig.canvas.draw_idle()

    def set_k(self, k):
        self.k_text.set_text(f"k: {k}")

    # ========= State resets =========
    def reset_visual_state_only(self):
        """Reset visuals and board to the start, but keep current self.events."""
        self.board = copy.deepcopy(self.orig)
        self.draw_numbers()
        self.set_cell_highlight(None)
        self.clear_conflicts()
        self.set_status("Ready: Press Next", "")
        self.highlight_pc(None)
        self.set_k(count_empty(self.board))
        self.history = []
        self.index = -1

    def new_run(self):
        """Full reset: recompute event trace and reset visuals."""
        self.events = generate_event_trace(self.orig)
        self.reset_visual_state_only()

    # ========= Mapping event -> UI updates =========
    def describe_event(self, event, replay_info=None):
        et = event['type']
        k  = event.get('k', count_empty(self.board))
        self.set_k(k)

        pc_idx = None

        if et == 'base_case':
            self.set_status("BASE CASE reached", "k == 0 → no empty cells remain.")
            pc_idx = 3

        elif et == 'solved':
            self.set_status("Solved! 🎉", "All cells filled consistently.")
            pc_idx = 3

        elif et == 'select':
            r, c = event['cell']
            self.set_status(f"Select next empty cell → r{r+1}, c{c+1}",
                            "We found a position to fill (problem shrinks by one cell).")
            pc_idx = 6

        elif et == 'candidates':
            r, c = event['cell']; cands = event['cands']
            self.set_status(f"Candidates for r{r+1}, c{c+1}: {cands}",
                            "Digits allowed by row, column, and 3×3 box rules.")
            pc_idx = 7

        elif et == 'try':
            r, c = event['cell']; d = event['digit']
            if event['valid']:
                self.set_status(f"Try {d} at r{r+1}, c{c+1} ✓ valid",
                                "No conflicts found → we may place and recurse.")
            else:
                reason = event['reason']; rule = reason['rule']; cells = reason['conflicts']
                where = ", ".join([f"(r{ri+1},c{cj+1})" for (ri, cj) in cells])
                self.set_status(f"Try {d} at r{r+1}, c{c+1} ✗ invalid",
                                f"Violates {rule}: same digit already at {where}.")
            pc_idx = 8

        elif et == 'place':
            r, c = event['cell']; d = event['digit']
            self.set_status(f"Place {d} at r{r+1}, c{c+1}",
                            "Commit choice; solve the smaller subproblem.")
            pc_idx = 9

        elif et == 'recurse_enter':
            self.set_status("Recurse", "Call solve(board) on the remaining empty cells.")
            pc_idx = 10

        elif et == 'recurse_return':
            ok = event['ok']
            self.set_status("Return from recursion",
                            "Success bubbles up." if ok else "Branch failed → try next candidate.")
            pc_idx = 10 if event['ok'] else 11

        elif et == 'backtrack':
            r, c = event['cell']
            self.set_status(f"Backtrack: clear r{r+1}, c{c+1}",
                            "Undo the last placement; explore other options.")
            pc_idx = 11

        elif et == 'dead_end':
            r, c = event['cell']; cands = event['cands']
            self.set_status(f"Dead end at r{r+1}, c{c+1}",
                            f"No candidate leads to a solution here (legal set was {cands}).")
            pc_idx = 12

        if replay_info is not None:
            self.left_info.set_text(replay_info)
        self.highlight_pc(pc_idx)
        return pc_idx, self.left_info.get_text(), k

    def apply_event(self, event):
        et = event['type']
        conflicts_to_show = []

        if et in ('base_case', 'solved'):
            self.set_cell_highlight(None)
            self.clear_conflicts()

        elif et == 'select':
            self.set_cell_highlight(event['cell'])
            self.clear_conflicts()

        elif et == 'candidates':
            pass

        elif et == 'try':
            self.clear_conflicts()
            if not event['valid'] and event.get('reason'):
                conflicts_to_show = event['reason'].get('conflicts', [])
                self.show_conflicts(conflicts_to_show)

        elif et == 'place':
            r, c = event['cell']; d = event['digit']
            self.board[r][c] = d
            self.draw_numbers()
            self.clear_conflicts()

        elif et == 'recurse_enter':
            pass

        elif et == 'recurse_return':
            pass

        elif et == 'dead_end':
            self.clear_conflicts()

        elif et == 'backtrack':
            r, c = event['cell']
            self.board[r][c] = 0
            self.draw_numbers()
            self.clear_conflicts()

        pc_idx, info_text, k_val = self.describe_event(event)
        return conflicts_to_show, pc_idx, info_text, k_val

    # ========= Buttons =========
    def on_next(self, _):
        nxt = self.index + 1
        if nxt >= len(self.events):
            self.set_status("Finished. Press Reset to run again.", self.left_info.get_text())
            self.autoplay = False
            self.timer.stop()
            return

        event = self.events[nxt]
        conflicts, pc_idx, info_text, k_val = self.apply_event(event)

        # capture highlight cell after apply
        hi_cell = None
        if self.cell_hl.get_visible():
            x, y = self.cell_hl.get_xy()
            hi_cell = (int(y), int(x))

        # trim redo-branch if we had moved back and then go forward anew
        self.history = self.history[:self.index+1]
        self.history.append((copy.deepcopy(self.board),
                             copy.deepcopy(event),
                             hi_cell,
                             copy.deepcopy(conflicts),
                             pc_idx,
                             info_text,
                             k_val))
        self.index = nxt

    def on_prev(self, _):
        if self.index < 0:
            # already at the beginning; just ensure visuals are reset
            self.reset_visual_state_only()
            return
        # move one step back; restore previous snapshot, or initial state if none
        self.index -= 1
        if self.index >= 0:
            snap = self.history[self.index]
            (board, event, hi_cell, conflicts, pc_idx, info_text, k_val) = snap
            self.board = copy.deepcopy(board)
            self.draw_numbers()
            self.set_cell_highlight(hi_cell)
            self.clear_conflicts()
            self.show_conflicts(conflicts)
            self.highlight_pc(pc_idx)
            self.k_text.set_text(f"k: {k_val}")
            # Re-describe the event for status/left_info
            self.describe_event(event, replay_info=info_text)
        else:
            self.reset_visual_state_only()

    def on_auto(self, _):
        self.autoplay = not self.autoplay
        self.btn_auto.label.set_text('Stop' if self.autoplay else 'Auto')
        if self.autoplay:
            self.timer.start()
        else:
            self.timer.stop()

    def auto_tick(self):
        if self.autoplay:
            self.on_next(None)

    def on_reset(self, _):
        self.autoplay = False
        self.btn_auto.label.set_text('Auto')
        self.timer.stop()
        # Fresh trace + visual reset
        self.new_run()

# =========================
# Run
# =========================
if __name__ == "__main__":
    viz = SudokuTeachViz(PUZZLE)
    plt.show()
