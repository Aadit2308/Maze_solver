import os
import random
import time
from collections import deque

import numpy as np

# ==========================================================================
# SETTINGS - change these
# ==========================================================================

NUM_MAZES = 5000         # total mazes to generate
WIDTH = 50               # increased for a denser maze
HEIGHT = 50              # increased for a denser maze
CORRIDOR = 1             # 1 creates continuous walls instead of isolated pillars
WALL_SIZE = 1            # wall thickness in tiles
LOOP_PROB = 0.5          # 0.0 ensures a perfect maze with no loops
ROOMS =  0               # open rooms (0 = none)
ROOM_MAX = 3
BASE_SEED = 0            # same BASE_SEED = same dataset every time
TRAIN_FRAC = 0.8         # 80% train
VAL_FRAC = 0.1           # 10% validation, remaining 10% test
OUT_DIR = "."            # where to save the .npz files
NUM_SAMPLE_IMAGES = 10   # pictures saved to sample_mazes/ (0 = none)

WALL = 1
PATH = 0

# Action meanings: 0 = up, 1 = down, 2 = left, 3 = right
MOVES = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1),
}


# ==========================================================================
# Maze generation (same style as your picture)
# ==========================================================================
def generate_maze(width, height, seed=None, loop_prob=0.0,
                  corridor=3, wall=1, rooms=0, room_max=3):
    """
    Returns (grid, start, goal)
      grid  : numpy int8 array, 1 = wall, 0 = free
      start : (row, col) entrance, top-left on the border
      goal  : (row, col) exit, bottom-right on the border
    """
    rng = random.Random(seed)

    step = corridor + wall
    rows = height * step + wall
    cols = width * step + wall
    grid = np.ones((rows, cols), dtype=np.int8)

    def origin(i, j):
        return wall + i * step, wall + j * step

    def open_cells(i1, j1, i2, j2):
        r1, c1 = origin(i1, j1)
        r2, c2 = origin(i2, j2)
        grid[min(r1, r2):max(r1, r2) + corridor,
             min(c1, c2):max(c1, c2) + corridor] = PATH

    # DFS carve from top-left cell
    visited = {(0, 0)}
    open_cells(0, 0, 0, 0)
    stack = [(0, 0)]
    while stack:
        i, j = stack[-1]
        nbrs = []
        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ni, nj = i + di, j + dj
            if 0 <= ni < height and 0 <= nj < width and (ni, nj) not in visited:
                nbrs.append((ni, nj))
        if nbrs:
            ni, nj = rng.choice(nbrs)
            open_cells(i, j, ni, nj)
            visited.add((ni, nj))
            stack.append((ni, nj))
        else:
            stack.pop()

    # optional loops
    if loop_prob > 0:
        for i in range(height):
            for j in range(width):
                r, c = origin(i, j)
                if j + 1 < width and grid[r, c + corridor] == WALL:
                    if rng.random() < loop_prob:
                        open_cells(i, j, i, j + 1)
                if i + 1 < height and grid[r + corridor, c] == WALL:
                    if rng.random() < loop_prob:
                        open_cells(i, j, i + 1, j)

    # optional rooms
    for _ in range(rooms):
        rw = rng.randint(min(2, width), min(room_max, width))
        rh = rng.randint(min(2, height), min(room_max, height))
        i0 = rng.randint(0, height - rh)
        j0 = rng.randint(0, width - rw)
        r0 = wall + i0 * step
        c0 = wall + j0 * step
        r1 = wall + (i0 + rh - 1) * step + corridor
        c1 = wall + (j0 + rw - 1) * step + corridor
        grid[r0:r1, c0:c1] = PATH

    # entrance + exit gaps in the border
    grid[0, wall:wall + corridor] = PATH
    grid[rows - 1, cols - wall - corridor:cols - wall] = PATH

    start = (0, wall + corridor // 2)
    goal = (rows - 1, cols - wall - corridor + corridor // 2)
    return grid, start, goal


def distance_map(grid, target):
    """BFS from target. Returns steps-to-target for every tile (-1 for walls)."""
    rows, cols = grid.shape
    dist = np.full((rows, cols), -1, dtype=np.int32)
    dist[target] = 0
    queue = deque([target])
    while queue:
        r, c = queue.popleft()
        for dr, dc in MOVES.values():
            nr, nc = r + dr, c + dc
            if (0 <= nr < rows and 0 <= nc < cols
                    and grid[nr, nc] == PATH and dist[nr, nc] == -1):
                dist[nr, nc] = dist[r, c] + 1
                queue.append((nr, nc))
    return dist


def solve_maze(grid, start, goal):
    """
    Returns (path, actions)
      path    : list of (row, col) from start to goal
      actions : list of ints (0 up, 1 down, 2 left, 3 right)
    """
    dist = distance_map(grid, goal)
    if dist[start] < 0:
        raise ValueError("Maze is not solvable")

    path = [start]
    actions = []
    pos = start
    while pos != goal:
        for a, (dr, dc) in MOVES.items():
            nr, nc = pos[0] + dr, pos[1] + dc
            if (0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]
                    and dist[nr, nc] == dist[pos] - 1):
                pos = (nr, nc)
                path.append(pos)
                actions.append(a)
                break
    return path, actions


# ==========================================================================
# Bulk generation
# ==========================================================================
def generate_bulk(count, width, height, corridor, wall, loop_prob,
                  rooms, room_max, base_seed):
    step = corridor + wall
    rows = height * step + wall
    cols = width * step + wall

    grids = np.zeros((count, rows, cols), dtype=np.int8)
    solutions = np.zeros((count, rows, cols), dtype=np.int8)  # 1 on the solution path
    starts = np.zeros((count, 2), dtype=np.int32)
    goals = np.zeros((count, 2), dtype=np.int32)
    lengths = np.zeros(count, dtype=np.int32)
    seeds = np.zeros(count, dtype=np.int64)
    all_actions = []

    t0 = time.time()
    for i in range(count):
        seed = base_seed + i
        grid, start, goal = generate_maze(
            width, height, seed=seed, loop_prob=loop_prob,
            corridor=corridor, wall=wall, rooms=rooms, room_max=room_max,
        )
        path, actions = solve_maze(grid, start, goal)

        grids[i] = grid
        starts[i] = start
        goals[i] = goal
        lengths[i] = len(actions)
        seeds[i] = seed
        for (r, c) in path:
            solutions[i, r, c] = 1
        all_actions.append(actions)

        if (i + 1) % 100 == 0 or i + 1 == count:
            elapsed = time.time() - t0
            print(f"  {i + 1}/{count} mazes  ({elapsed:.1f}s)")

    # pad action lists so they fit in one array (-1 = padding)
    max_len = int(lengths.max())
    actions_arr = np.full((count, max_len), -1, dtype=np.int8)
    for i, acts in enumerate(all_actions):
        actions_arr[i, :len(acts)] = acts

    return {
        "grids": grids,            # (N, rows, cols)  1 = wall, 0 = free
        "starts": starts,          # (N, 2)  (row, col)
        "goals": goals,            # (N, 2)  (row, col)
        "solutions": solutions,    # (N, rows, cols)  1 = on the shortest path
        "actions": actions_arr,    # (N, max_len)  correct moves, -1 = padding
        "lengths": lengths,        # (N,)  shortest path length in steps
        "seeds": seeds,            # (N,)  seed used for each maze
    }


def split_and_save(data, out_dir, train_frac, val_frac):
    n = len(data["grids"])
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    splits = {
        "train": slice(0, n_train),
        "val": slice(n_train, n_train + n_val),
        "test": slice(n_train + n_val, n),
    }

    os.makedirs(out_dir, exist_ok=True)
    for name, sl in splits.items():
        part = {k: v[sl] for k, v in data.items()}
        path = os.path.join(out_dir, f"mazes_{name}.npz")
        np.savez_compressed(path, **part)
        size_mb = os.path.getsize(path) / 1e6
        print(f"  saved {path}: {len(part['grids'])} mazes ({size_mb:.1f} MB)")


def save_sample_images(data, count, folder="sample_mazes"):
    if count <= 0:
        return
    import matplotlib
    matplotlib.use("Agg")  # save to file only, no window
    import matplotlib.pyplot as plt

    os.makedirs(folder, exist_ok=True)
    for i in range(min(count, len(data["grids"]))):
        grid = data["grids"][i]
        start = data["starts"][i]
        goal = data["goals"][i]
        sol = data["solutions"][i]

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        for ax, show_solution in zip(axes, (False, True)):
            ax.imshow(grid, cmap="gray_r", interpolation="nearest")
            if show_solution:
                ys, xs = np.where(sol == 1)
                ax.scatter(xs, ys, s=4, color="red")
            ax.plot(start[1], start[0], "v", color="blue", markersize=10)
            ax.plot(goal[1], goal[0], "v", color="green", markersize=10)
            ax.axis("off")
        axes[0].set_title("Maze")
        axes[1].set_title(f"Solution ({data['lengths'][i]} steps)")
        fig.tight_layout()
        fig.savefig(os.path.join(folder, f"maze_{i}.png"), dpi=100)
        plt.close(fig)
    print(f"  saved {min(count, len(data['grids']))} pictures to {folder}/")


def verify(data, check=200):
    """Replay the stored actions on random mazes to make sure they reach the goal."""
    n = len(data["grids"])
    idx = random.Random(0).sample(range(n), min(check, n))
    for i in idx:
        grid = data["grids"][i]
        pos = tuple(data["starts"][i])
        goal = tuple(data["goals"][i])
        for a in data["actions"][i]:
            if a < 0:
                break
            dr, dc = MOVES[int(a)]
            pos = (pos[0] + dr, pos[1] + dc)
            assert grid[pos] == PATH, f"maze {i}: walked into a wall"
        assert pos == goal, f"maze {i}: did not reach the goal"
    print(f"  verified {len(idx)} random mazes: stored solutions all reach the goal")


# ==========================================================================
# Main
# ==========================================================================
if __name__ == "__main__":
    print(f"Generating {NUM_MAZES} mazes ({WIDTH}x{HEIGHT} cells, corridor={CORRIDOR})...")
    data = generate_bulk(
        NUM_MAZES, WIDTH, HEIGHT, CORRIDOR, WALL_SIZE, LOOP_PROB,
        ROOMS, ROOM_MAX, BASE_SEED,
    )

    print("\nChecking...")
    verify(data)

    print("\nSaving...")
    split_and_save(data, OUT_DIR, TRAIN_FRAC, VAL_FRAC)
    save_sample_images(data, NUM_SAMPLE_IMAGES)

    print("\nDataset info:")
    print("  grid shape per maze:", data["grids"].shape[1:])
    print("  shortest path length: min", data["lengths"].min(),
          "| avg", round(float(data["lengths"].mean()), 1),
          "| max", data["lengths"].max())
    print("\nDone.")
