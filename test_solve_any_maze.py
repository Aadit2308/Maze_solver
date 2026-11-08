"""
Tests for solve_any_maze.py.   Run:  python -m pytest test_solve_any_maze.py   (or just: python test_solve_any_maze.py)
"""
import os
import tempfile

import numpy as np

import solve_any_maze as sam
from maze_genrator import generate_maze, distance_map

HERE = os.path.dirname(os.path.abspath(__file__))


def check_path(walls, path, start, goal):
    assert path[0] == start and path[-1] == goal
    for (r1, c1), (r2, c2) in zip(path, path[1:]):
        assert abs(r1 - r2) + abs(c1 - c2) == 1, "path jumps"
        assert not walls[r2, c2], "path goes through a wall"


def test_many_sizes_and_all_algorithms_are_shortest():
    sizes = [(1, 1), (1, 7), (7, 1), (2, 2), (5, 9), (17, 3), (40, 40), (63, 21)]
    for w, h in sizes:
        for loops in (0.0, 0.3):
            grid, s, g = generate_maze(w, h, seed=w * 100 + h, loop_prob=loops, corridor=1, wall=1)
            walls = grid == 1
            best = int(distance_map(grid, g)[s])
            for algo in ("bfs", "astar", "dijkstra"):
                path = sam.solve(walls, s, g, algo)
                check_path(walls, path, s, g)
                assert len(path) - 1 == best, (w, h, algo)


def test_pure_python_fallback_matches_scipy():
    grid, s, g = generate_maze(30, 20, seed=5, loop_prob=0.2, corridor=1, wall=1)
    walls = grid == 1
    fast = sam.solve_bfs(walls, s, g)
    saved = sam.HAVE_SCIPY
    sam.HAVE_SCIPY = False
    try:
        slow = sam.solve_bfs(walls, s, g)
    finally:
        sam.HAVE_SCIPY = saved
    assert len(fast) == len(slow)


def test_large_maze():
    grid, s, g = generate_maze(400, 300, seed=1, corridor=1, wall=1)   # 601 x 801 grid
    walls = grid == 1
    path = sam.solve(walls, s, g)
    check_path(walls, path, s, g)
    assert sam.find_endpoints(sam.Maze(walls)) == (s, g)


def test_unsolvable_returns_none():
    walls = np.zeros((5, 5), dtype=bool)
    walls[2, :] = True
    for algo in ("bfs", "astar", "dijkstra"):
        assert sam.solve(walls, (0, 0), (4, 4), algo) is None


def test_endpoint_detection_without_border_gaps():
    walls = np.zeros((4, 6), dtype=bool)
    walls[1, 1:5] = True
    s, g = sam.find_endpoints(sam.Maze(walls))
    assert s == (0, 0) and g == (3, 5)


def test_text_maze_with_markers_and_double_width():
    txt = "\n".join([
        "##########",
        "#S   #   #",
        "# ## # # #",
        "#      #E#",
        "##########",
    ])
    double = "\n".join("".join(ch * 2 if ch == "#" else ch + (" " if ch in "SE" else ch)
                               for ch in line) for line in txt.splitlines())
    for content in (txt, double):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write(content)
        try:
            m = sam.load_maze(f.name)
        finally:
            os.unlink(f.name)
        assert m.shape == (5, 10)
        s, g = sam.find_endpoints(m)
        assert (s, g) == ((1, 1), (3, 8))
        check_path(m.walls, sam.solve(m.walls, s, g), s, g)


def test_bundled_files():
    for name in ("maze.png", "my_maze.npy", "maze.txt", "mazes_test.npz"):
        path = os.path.join(HERE, name)
        if not os.path.exists(path):
            continue
        m = sam.load_maze(path)
        s, g = sam.find_endpoints(m)
        p = sam.solve(m.walls, s, g, centre=m.image is not None)
        assert p is not None, name
        check_path(m.walls, p, s, g)


def test_actions_replay_to_goal():
    grid, s, g = generate_maze(12, 12, seed=3, corridor=1, wall=1)
    path = sam.solve(grid == 1, s, g)
    pos = s
    for a in sam.path_to_actions(path):
        dr, dc = sam.MOVES[a]
        pos = (pos[0] + dr, pos[1] + dc)
    assert pos == g


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
