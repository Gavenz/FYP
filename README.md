# Gamified Learning of Data Structures & Algorithm (IE2108) - Puzzle Games
### Credits to Gaven & Dr S Supraja (Supervisor)  

This repository contains **interactive, visual puzzle games** designed to help students taking IE2108 to learn key DSAI concepts through playing puzzle games and respective functions like: step-by-step execution, pseudocode and visualisation panels.  

---
🎥 Watch 3-minute demo: 

https://github.com/user-attachments/assets/ff94cc6b-62b8-45d0-99a6-da52eea9c198


## 🎮 Included Games

- **Maze**  
  DFS / BFS / Dijkstra  
  Includes weighted maze mode and complexity counters

- **Sudoku**  
  Backtracking (Recursive & While-loop visual modes)

- **Word Search**  
  Max-Heap operations (heapify, insertion, deletion)  
  DFS solver visualization

---

## Requirements

- **Python 3.10+** (recommended)
- [`uv`](https://github.com/astral-sh/uv) as the package manager
- GUI support for Matplotlib (most systems work out of the box)

Python libraries:
- `matplotlib`

> All other imports are Python standard library modules.

---

## Installation Guide (for all Operating Systems)
---
## Step 1: Get the Project Files
1. Click the green **Code** button on GitHub
2. Click **Download ZIP**
3. Extract the folder to your Desktop

Example path (Windows):
```
C:\Users\PC\Desktop\project_file
```

---

## Step 2: Install uv

### Windows (PowerShell)
In your terminal:
```bash
pip install uv
```

If pip fails, reinstall Python from python.org

---

### macOS / Ubuntu
In your terminal:
```bash
curl -Ls https://astral.sh/uv/install.sh | sh
```

Restart terminal, then verify:

```bash
uv --version
```

---

## Quick Start
### If you are using VScode (recommended)
1. Open the project folder in VS Code
2. Open Terminal inside VS Code
3. Run:

```bash
uv sync (installs project dependencies)
uv run python mazev3.py
```
### If you are using Jupyter
These games uses interactive windows which are **not designed to run inside Jupyter.** If you do not use VScode, run the code inside your **terminal**. 
1. Navigate to the location of where you downloaded the project folder
2. Open terminal in the project folder (right click within the file and select 'open in terminal')
3. Run:

```bash
uv sync
uv run python mazev3.py
