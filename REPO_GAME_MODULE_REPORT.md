# Repository Game Module Report

## Maze (`maze.py`)

### Algorithm Engine
- `cell_cost` — Returns traversal cost for a terrain cell using the cost table.
- `new_counters` — Creates the operation-counter dictionary used to track algorithm actions.
- `graph_sizes` — Computes approximate passable vertex/edge counts for the active maze graph.
- `set_maze` — Loads a maze variant, enforces borders, and reinitializes start/goal openings.
- `in_bounds` — Checks whether a coordinate lies inside maze dimensions.
- `passable` — Tests whether a maze coordinate is not a wall.
- `neighbors` — Iterates valid 4-neighbor passable coordinates for graph expansion.
- `gen_dfs` — Emits stepwise DFS state snapshots for instructional playback.
- `gen_bfs` — Emits stepwise BFS state snapshots for instructional playback.
- `gen_dijkstra` — Emits stepwise Dijkstra state snapshots including distance/relaxation behavior.

### Visualisation Module
- `MazeViz` — Owns matplotlib figure/axes and renders maze, overlays, pseudocode, and legend views.
- `MazeViz.__init__` — Constructs the visualization layout, artists, state caches, and event bindings.
- `MazeViz.draw_custom_legend` — Draws the compact legend describing cell/overlay color meanings.
- `MazeViz.draw_static_maze` — Paints terrain tiles plus start/goal glyphs on the maze axis.
- `MazeViz.reconstruct_path` — Rebuilds a route from parent links to show solution path.
- `MazeViz.path_cost` — Computes total weighted cost of a candidate path.
- `MazeViz.depth_of` — Computes node depth from parent links for display metrics.
- `MazeViz._sets_from_state` — Derives parent/visited/path/frontier sets from current algorithm state.
- `MazeViz.draw_graph_overlay` — Draws dynamic frontier/visited/current/path overlays on the maze.
- `MazeViz._draw_section` — Draws formatted text blocks in the left instructional panel.
- `MazeViz.update_overlay` — Refreshes all dynamic artists and text after state changes.
- `MazeViz.on_resize` — Handles resize events to recompute responsive geometry.
- `MazeViz.apply_responsive_layout` — Repositions panels and controls based on figure dimensions.
- `MazeViz._init_code_blocks` — Initializes pseudocode text artists for all supported algorithms.
- `MazeViz.highlight_pseudocode` — Highlights active pseudocode lines for the current action.
- `MazeViz._get_frontier_preview_lines` — Formats frontier preview text from algorithm state.
- `MazeViz._draw_pseudocode_panel` — Renders the pseudocode panel content and line emphasis.

### UI Module
- `MazeViz._instruction_message` — Provides the keyboard-help instruction string for users.
- `MazeViz.set_algo` — Switches algorithm mode and resets run state for the new selection.
- `MazeViz.cycle_maze` — Cycles among maze presets and resets the run context.
- `MazeViz._snapshot` — Captures a restorable visualization+state snapshot for undo/redo behavior.
- `MazeViz._msg_clear` — Resets message panel text and per-step emphasis.
- `MazeViz.set_message` — Writes status/action feedback into the top message panel.
- `MazeViz.fmt_add` — Formats optional status text fragments for display strings.
- `MazeViz.step_once` — Advances one algorithm micro-step and updates history.
- `MazeViz.retract_once` — Undoes one step by restoring the previous snapshot.
- `MazeViz.on_key` — Dispatches keyboard shortcuts to step/play/mode/maze/undo commands.
- `MazeViz.autoplay` — Timer callback for continuous stepping while autoplay is enabled.

## Sudoku (`sudokuv.py`)

### Algorithm Engine
- `count_empty` — Counts unsolved cells to quantify search progress.
- `select_cell_first_empty` — Selects the next target cell using first-empty order.
- `check_reason` — Validates a candidate digit and returns conflict context when invalid.
- `candidates_for` — Lists all legal digits for a specific empty cell.
- `solve_steps` — Generates recursive backtracking events for each logical solver action.
- `generate_event_trace` — Materializes the full event stream from a fresh puzzle copy.

### Visualisation Module
- `SudokuTeachViz` — Manages all rendered panels (grid, pseudocode, info, stack) and playback state.
- `SudokuTeachViz.__init__` — Builds axes/layout/artists, configures modes, and initializes controls/events.
- `SudokuTeachViz._draw_pseudocode` — Draws the active mode’s pseudocode lines and highlight canvas.
- `SudokuTeachViz.draw_numbers` — Renders givens/filled numbers onto the Sudoku grid.
- `SudokuTeachViz.set_cell_highlight` — Updates the currently highlighted grid cell overlay.
- `SudokuTeachViz.clear_conflicts` — Removes any conflict highlight overlays from the grid.
- `SudokuTeachViz.show_conflicts` — Draws conflict highlights for invalid candidate attempts.
- `SudokuTeachViz.highlight_pc` — Activates pseudocode-line highlighting for the current event.
- `SudokuTeachViz.render_stack` — Draws the conceptual call-stack panel for recursion/backtracking.
- `SudokuTeachViz.apply_event` — Applies one solver event to visualization state and overlays.

### UI Module
- `SudokuTeachViz._init_buttons_under_grid` — Creates and places interactive control buttons below the grid.
- `SudokuTeachViz.set_status` — Updates the top status text shown to the learner.
- `SudokuTeachViz.set_k` — Updates the remaining-empty-cells indicator.
- `SudokuTeachViz.reset_visual_state_only` — Clears transient UI/overlay state without replacing the puzzle.
- `SudokuTeachViz.new_run` — Resets board/event playback and starts a new trace session.
- `SudokuTeachViz.describe_event` — Produces user-facing narration text for each solver event.
- `SudokuTeachViz.on_next` — Handles the Next button to advance one event.
- `SudokuTeachViz.on_prev` — Handles the Prev button to step backward through history.
- `SudokuTeachViz.on_auto` — Toggles autoplay mode from the Auto button.
- `SudokuTeachViz.on_mode` — Cycles displayed pseudocode mode (Recur/While/For).
- `SudokuTeachViz.auto_tick` — Timer callback that advances playback while autoplay is on.
- `SudokuTeachViz.on_reset` — Handles full reset interaction from the Reset button.

## Word Search (`wordpuzzle.py`)

### Algorithm Engine
- `empty_grid` — Allocates a blank character grid initialized with the empty marker.
- `in_bounds` — Checks whether a grid coordinate is valid.
- `canon` — Converts a word to canonical orientation for forward/reverse equivalence.
- `fits_and_overlap` — Tests if a word fits at a position/direction and reports overlap score.
- `place` — Commits word letters into selected grid cells.
- `fill_random` — Fills unused grid cells with random uppercase letters.
- `best_placement` — Searches placements and chooses the highest-overlap candidate for a word.
- `_sift_down_minheap` — Performs sift-down on heap arrays while recording demo snapshots.
- `push_with_snapshots` — Simulates heap insertion and records step-by-step heap states.
- `heapify_with_snapshots` — Heapifies an array while recording instructional snapshots.
- `pop_with_snapshots` — Simulates heap delete-max behavior (via min-heap keys) with snapshots.
- `HeapGenerator` — State machine that builds/updates heap-driven generation phases.
- `HeapGenerator.__init__` — Initializes generator state for heap demos and grid generation.
- `HeapGenerator.clone` — Deep-copies generation state for undo/history support.
- `HeapGenerator.step` — Advances one generation phase and returns render-ready state output.
- `TrieNode` — Node container for trie children/terminal/found flags.
- `TrieNode.__init__` — Initializes trie node storage fields.
- `Trie` — Prefix tree used by the DFS solver for pruning and word detection.
- `Trie.__init__` — Initializes trie root node.
- `Trie.insert` — Inserts one uppercase word into trie structure.
- `Trie.has_prefix` — Checks whether a prefix exists in trie.
- `Trie.is_word` — Checks whether a complete word exists in trie.
- `Trie.mark_found_word` — Marks a discovered word as found in trie metadata.
- `build_trie` — Builds a trie from a list of candidate words.
- `GlobalDFSSolver` — Stateful DFS engine that explores the board against trie constraints.
- `GlobalDFSSolver.__init__` — Initializes DFS stacks, found-set tracking, and traversal state.
- `GlobalDFSSolver.clone` — Deep-copies DFS solver state for reversible playback.
- `GlobalDFSSolver._push` — Pushes a DFS frame and updates visitation/path bookkeeping.
- `GlobalDFSSolver._pop` — Pops a DFS frame and restores bookkeeping state.
- `GlobalDFSSolver._done_if_all_found` — Checks global termination when all target words are found.
- `GlobalDFSSolver.step` — Advances DFS by one micro-step and returns event/render data.

### Visualisation Module
- `Viz` — Renders grid, max-heap, trie, and DFS logs within a multi-panel matplotlib view.
- `Viz.__init__` — Creates figure axes, widgets, and persistent artist containers.
- `Viz.draw_grid` — Draws/refreshes the word-search board and found-word highlights.
- `Viz.draw_maxheap_array_blocks` — Renders heap array cells and active comparison markers.
- `Viz.draw_heap_tree` — Draws heap as a binary tree view with highlighted nodes/edges.
- `Viz.draw_trie_full` — Renders the full trie with prefix/found visual emphasis.
- `Viz.draw_dfs_log` — Draws the DFS stack/log text panel for solver progression.
- `Viz.draw_pq_array_in_log` — Draws priority-queue array detail lines in the log panel.

### UI Module
- `Controller` — Coordinates mode switching, timer stepping, history, and widget callbacks.
- `Controller.__init__` — Wires visualization with generator/solver state and initializes controls.
- `Controller.bind` — Connects matplotlib button/timer events to controller handlers.
- `Controller._toggle_mode` — Switches between generation and solving interaction modes.
- `Controller._tick` — Timer callback for autoplay stepping.
- `Controller._arm` — Starts/stops timer state based on play mode.
- `Controller._step` — Executes one controlled step and records history snapshots.
- `Controller._toggle` — Handles Play/Pause button behavior.
- `Controller._restart` — Resets controller and engine state to initial conditions.
- `Controller._undo` — Restores the previous snapshot from history.
- `Controller.wait` — Blocks on the matplotlib event loop for interactive session.
- `Controller.set_state_refs` — Rebinds active engine references after mode/state changes.
- `Controller._make_snapshot` — Creates a full snapshot bundle for undo support.
- `Controller._restore_snapshot` — Restores controller/engine/viz state from a saved snapshot.
- `run` — Entry-point function that instantiates visualization/controller and starts the app.
