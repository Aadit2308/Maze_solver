"""
Solve ANY maze of ANY size.

Unlike the PPO agent (which is trained on one fixed grid size), this solver works on
mazes of any width/height and in several formats:

    python solve_any_maze.py maze.png                    # picture of a maze (any resolution)
    python solve_any_maze.py my_maze.npy                 # numpy grid, 1 = wall
    python solve_any_maze.py maze.txt                    # ASCII / block-character maze
    python solve_any_maze.py mazes_test.npz --index 7    # one maze from a dataset
    python solve_any_maze.py --generate 200x150          # make a random 200x150-cell maze and solve it

Useful options:
    --start R,C --goal R,C     set the endpoints yourself (row, column in the loaded grid)
    --algo bfs|astar|dijkstra  search algorithm (all return a shortest path)
    --algo ppo                 the trained RL agent (maze_rl_any_size.py), BFS finishes if it gets stuck
    --out solution.png         where to save the picture of the solution
    --save-path path.npy       save the path as an (N, 2) array of (row, col)
    --ascii                    print the solved maze in the terminal (small mazes only)

Start and goal are found automatically, in this order:
    1. --start / --goal
    2. markers in the file (S/E letters in text; red/green marks in pictures)
    3. gaps in the outer wall (the gap nearest the top-left is the start,
       the one nearest the bottom-right is the goal)
    4. the first and last open cells
"""
import argparse
import heapq
import os
import sys
import time
from collections import deque

import numpy as np

try:
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import breadth_first_order, dijkstra
    from scipy.ndimage import distance_transform_edt
    HAVE_SCIPY = True
except ImportError:  # everything still works, just slower on huge mazes
    HAVE_SCIPY = False

MOVES = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}  # up, down, left, right (same as training code)
WALL_CHARS = set("#█▓▒X1@+-|=")
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp"}


# ==========================================================================
# Loading
# ==========================================================================
class Maze:
    """walls: 2D bool array (True = wall). start/goal: (row, col) or None."""

    def __init__(self, walls, start=None, goal=None, image=None, offset=(0, 0), scale=1.0):
        self.walls = np.ascontiguousarray(walls, dtype=bool)
        self.start, self.goal = start, goal
        self.image = image      # original RGB picture (for drawing), or None
        self.offset = offset    # top-left of the crop inside the picture
        self.scale = scale      # picture pixels per grid cell

    @property
    def shape(self):
        return self.walls.shape


def load_maze(path, index=0, one_is_wall=True, max_side=1500):
    ext = os.path.splitext(path)[1].lower()
    if ext in IMAGE_EXT:
        return load_image(path, max_side)
    if ext == ".npy":
        return grid_to_maze(np.load(path), one_is_wall)
    if ext == ".npz":
        data = np.load(path, allow_pickle=False)
        key = "grids" if "grids" in data.files else next(
            k for k in data.files if data[k].ndim in (2, 3) and "action" not in k)
        arr = data[key]
        if arr.ndim == 3:
            arr = arr[index]
        maze = grid_to_maze(arr, one_is_wall)
        if {"starts", "goals"} <= set(data.files) and data[key].ndim == 3:
            maze.start = tuple(int(v) for v in data["starts"][index])
            maze.goal = tuple(int(v) for v in data["goals"][index])
        return maze
    return load_text(path)


def grid_to_maze(arr, one_is_wall=True):
    arr = np.asarray(arr)
    if arr.ndim == 3:  # an image stored as an array
        arr = arr[..., :3].mean(-1)
    if arr.max() > 1 or arr.min() < 0:  # grey levels: dark = wall
        walls = arr < (float(arr.min()) + float(arr.max())) / 2
    else:
        walls = (arr == 1) if one_is_wall else (arr == 0)
    return Maze(walls)


def load_text(path):
    with open(path, encoding="utf-8") as f:
        lines = [ln.rstrip("\n\r") for ln in f if ln.strip()]
    width = max(len(ln) for ln in lines)
    lines = [ln.ljust(width) for ln in lines]
    # "██" / "  " style: every cell is drawn twice side by side -> collapse it
    if width % 2 == 0 and all(ln[i] == ln[i + 1] or ln[i] in "SEG" for ln in lines for i in range(0, width, 2)):
        lines = [ln[::2] for ln in lines]
    walls = np.array([[ch in WALL_CHARS for ch in ln] for ln in lines], dtype=bool)
    start = goal = None
    for r, ln in enumerate(lines):
        for c, ch in enumerate(ln):
            if ch == "S":
                start = (r, c)
            elif ch in "EG":
                goal = (r, c)
    return Maze(walls, start, goal)


def load_image(path, max_side=1500):
    from PIL import Image
    rgb = np.asarray(Image.open(path).convert("RGB")).astype(np.int16)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    sat = rgb.max(-1) - rgb.min(-1)
    walls = (lum < 128) & (sat < 60)          # dark AND grey = wall (coloured markers are not walls)
    red = (r > 150) & (g < 110) & (b < 110)
    green = (g > 90) & (r < 110) & (b < 110) & (g > r + 40)

    # crop to the maze itself (drop the empty margin around it)
    ys, xs = np.where(walls)
    if len(ys) == 0:
        raise ValueError("No walls found in the picture.")
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    walls, red, green = walls[y0:y1, x0:x1], red[y0:y1, x0:x1], green[y0:y1, x0:x1]

    # shrink very large pictures (min-pooling keeps thin walls)
    k = max(1, int(np.ceil(max(walls.shape) / max_side)))
    if k > 1:
        H, W = (walls.shape[0] // k) * k, (walls.shape[1] // k) * k
        pool = lambda m: m[:H, :W].reshape(H // k, k, W // k, k).any(axis=(1, 3))
        walls, red, green = pool(walls), pool(red), pool(green)

    def centre(mask):
        pts = np.argwhere(mask)
        return tuple(int(v) for v in pts.mean(0).round()) if len(pts) > 20 // k else None

    return Maze(walls, centre(red), centre(green), image=rgb.astype(np.uint8), offset=(y0, x0), scale=k)


# ==========================================================================
# Start / goal detection
# ==========================================================================
def perimeter_openings(walls):
    """Middle cell of every gap in the outer border, walking clockwise."""
    H, W = walls.shape
    ring = ([(0, c) for c in range(W)] + [(r, W - 1) for r in range(1, H)] +
            [(H - 1, c) for c in range(W - 2, -1, -1)] + [(r, 0) for r in range(H - 2, 0, -1)])
    gaps, cur = [], []
    for p in ring:
        if not walls[p]:
            cur.append(p)
        elif cur:
            gaps.append(cur)
            cur = []
    if cur:
        if gaps and not walls[ring[0]]:  # gap wraps around the corner where the walk started
            gaps[0] = cur + gaps[0]
        else:
            gaps.append(cur)
    if len(gaps) == 1 and len(gaps[0]) == len(ring):  # no outer wall at all
        return []
    return [g[len(g) // 2] for g in gaps]


def nearest_free(walls, p):
    p = (min(max(p[0], 0), walls.shape[0] - 1), min(max(p[1], 0), walls.shape[1] - 1))
    if not walls[p]:
        return p
    free = np.argwhere(~walls)
    i = int(np.abs(free - np.array(p)).sum(1).argmin())
    return tuple(int(v) for v in free[i])


def find_endpoints(maze):
    walls = maze.walls
    start, goal = maze.start, maze.goal
    if start is None or goal is None:
        gaps = perimeter_openings(walls)
        H, W = walls.shape
        if len(gaps) >= 2:
            by_tl = sorted(gaps, key=lambda p: p[0] + p[1])
            start = start or by_tl[0]
            goal = goal or by_tl[-1]
        else:
            free = np.argwhere(~walls)
            if len(free) == 0:
                raise ValueError("Maze has no open cells (are walls/paths inverted? try --zero-is-wall)")
            start = start or tuple(int(v) for v in free[0])
            goal = goal or tuple(int(v) for v in free[-1])
    return nearest_free(walls, start), nearest_free(walls, goal)


# ==========================================================================
# Search
# ==========================================================================
def _build_graph(free, weights=None):
    H, W = free.shape
    idx = np.full(free.shape, -1, dtype=np.int64)
    n = int(free.sum())
    idx[free] = np.arange(n)
    pairs = []
    for a, b in ((idx[:, :-1], idx[:, 1:]), (idx[:-1, :], idx[1:, :])):
        m = (a >= 0) & (b >= 0)
        pairs.append((a[m], b[m]))
    src = np.concatenate([p[0] for p in pairs])
    dst = np.concatenate([p[1] for p in pairs])
    if weights is None:
        w = np.ones(len(src), dtype=np.float32)
    else:
        flat = weights[free]
        w = ((flat[src] + flat[dst]) / 2).astype(np.float64)
    g = csr_matrix((np.concatenate([w, w]), (np.concatenate([src, dst]), np.concatenate([dst, src]))),
                   shape=(n, n))
    return g, idx


def _walk_back(pred, s, t, cells, none_value):
    if t != s and pred[t] == none_value:
        return None
    path = [t]
    while path[-1] != s:
        path.append(int(pred[path[-1]]))
    return [tuple(int(v) for v in cells[i]) for i in reversed(path)]


def solve_bfs(walls, start, goal):
    free = ~walls
    if HAVE_SCIPY:
        g, idx = _build_graph(free)
        s, t = int(idx[start]), int(idx[goal])
        _, pred = breadth_first_order(g, s, directed=False, return_predecessors=True)
        return _walk_back(pred, s, t, np.argwhere(free), -9999)
    # pure-Python fallback on flat indices
    H, W = walls.shape
    flat = free.ravel()
    pred = np.full(H * W, -1, dtype=np.int64)
    s, t = start[0] * W + start[1], goal[0] * W + goal[1]
    pred[s] = s
    q = deque([s])
    while q:
        u = q.popleft()
        if u == t:
            break
        r, c = divmod(u, W)
        for v, ok in ((u - W, r > 0), (u + W, r < H - 1), (u - 1, c > 0), (u + 1, c < W - 1)):
            if ok and flat[v] and pred[v] < 0:
                pred[v] = u
                q.append(v)
    if pred[t] < 0:
        return None
    path = [t]
    while path[-1] != s:
        path.append(int(pred[path[-1]]))
    return [divmod(u, W) for u in reversed(path)]


def solve_astar(walls, start, goal, cost=None):
    """A* with Manhattan distance. `cost` (optional) = extra price for stepping on each cell."""
    H, W = walls.shape
    flat = (~walls).ravel()
    extra = None if cost is None else cost.ravel()
    s, t = start[0] * W + start[1], goal[0] * W + goal[1]
    gr, gc = goal
    g = np.full(H * W, np.inf)
    pred = np.full(H * W, -1, dtype=np.int64)
    g[s] = 0
    heap = [(abs(start[0] - gr) + abs(start[1] - gc), 0.0, s)]
    while heap:
        _, gu, u = heapq.heappop(heap)
        if u == t:
            break
        if gu > g[u]:
            continue
        r, c = divmod(u, W)
        for v, ok in ((u - W, r > 0), (u + W, r < H - 1), (u - 1, c > 0), (u + 1, c < W - 1)):
            if not ok or not flat[v]:
                continue
            nv = gu + 1.0 + (0.0 if extra is None else extra[v])
            if nv < g[v]:
                g[v], pred[v] = nv, u
                vr, vc = divmod(v, W)
                heapq.heappush(heap, (nv + abs(vr - gr) + abs(vc - gc), nv, v))
    if not np.isfinite(g[t]):
        return None
    path = [t]
    while path[-1] != s:
        path.append(int(pred[path[-1]]))
    return [divmod(int(u), W) for u in reversed(path)]


def centre_cost(walls, strength=4.0):
    """Higher cost close to walls, so paths through wide (picture) corridors run down the middle."""
    if HAVE_SCIPY:
        clearance = distance_transform_edt(~walls)
    else:
        clearance = np.ones(walls.shape)
    return strength / np.maximum(clearance, 1.0) ** 2


def solve_dijkstra(walls, start, goal, cost=None):
    if not HAVE_SCIPY:
        return solve_astar(walls, start, goal, cost)
    free = ~walls
    weights = None if cost is None else 1.0 + cost
    g, idx = _build_graph(free, weights)
    s, t = int(idx[start]), int(idx[goal])
    _, pred = dijkstra(g, directed=False, indices=s, return_predecessors=True)
    return _walk_back(pred, s, t, np.argwhere(free), -9999)


def solve(walls, start, goal, algo="bfs", centre=False):
    cost = centre_cost(walls) if centre else None
    if algo == "astar":
        return solve_astar(walls, start, goal, cost)
    if algo == "dijkstra" or cost is not None:
        return solve_dijkstra(walls, start, goal, cost)
    return solve_bfs(walls, start, goal)


def corridor_width(walls):
    """Typical corridor width in tiles (short open runs that have a wall at both ends)."""
    runs = []
    for m in (walls, walls.T):
        for row in m[:: max(1, m.shape[0] // 300)]:
            edges = np.flatnonzero(np.diff(row.astype(np.int8)))
            for a, b in zip(edges[:-1], edges[1:]):
                if row[a] and not row[a + 1]:  # wall -> open ... -> wall
                    runs.append(b - a)
    return float(np.percentile(runs, 20)) if runs else 1.0


def shrink_for_agent(maze, start, goal):
    """The PPO agent is trained on corridors 1-3 tiles wide; shrink wider ones (e.g. pictures)."""
    k = max(1, int(corridor_width(maze.walls) // 2))
    if k == 1:
        return maze, start, goal
    H, W = (maze.shape[0] // k) * k, (maze.shape[1] // k) * k
    walls = maze.walls[:H, :W].reshape(H // k, k, W // k, k).any(axis=(1, 3))
    small = Maze(walls, image=maze.image, offset=maze.offset, scale=maze.scale * k)
    fix = lambda p: nearest_free(walls, (min(p[0] // k, walls.shape[0] - 1), min(p[1] // k, walls.shape[1] - 1)))
    return small, fix(start), fix(goal)


def path_to_actions(path):
    lookup = {v: k for k, v in MOVES.items()}
    return [lookup[(b[0] - a[0], b[1] - a[1])] for a, b in zip(path, path[1:])]


# ==========================================================================
# Output
# ==========================================================================
def draw_solution(maze, path, start, goal, out):
    from PIL import Image, ImageDraw
    if maze.image is not None:
        img = Image.fromarray(maze.image)
        oy, ox = maze.offset
        k = maze.scale
        pts = [(ox + (c + 0.5) * k, oy + (r + 0.5) * k) for r, c in path]
        width = max(2, int(round(min(img.size) / 250)))
    else:
        H, W = maze.shape
        k = max(1, min(20, 1600 // max(H, W)))
        pix = np.where(maze.walls, 0, 255).astype(np.uint8)
        img = Image.fromarray(pix).resize((W * k, H * k), Image.NEAREST).convert("RGB")
        pts = [((c + 0.5) * k, (r + 0.5) * k) for r, c in path]
        width = max(1, k // 3)
    d = ImageDraw.Draw(img)
    if len(pts) > 1:
        d.line(pts, fill=(230, 30, 30), width=width, joint="curve")
    rad = max(3, width * 2)
    for (x, y), colour in ((pts[0], (40, 90, 230)), (pts[-1], (20, 160, 60))):  # blue = start, green = goal
        d.ellipse((x - rad, y - rad, x + rad, y + rad), fill=colour)
    img.save(out)


def ascii_solution(walls, path):
    on = set(path)
    rows = []
    for r in range(walls.shape[0]):
        rows.append("".join("██" if walls[r, c] else ("··" if (r, c) in on else "  ")
                            for c in range(walls.shape[1])))
    return "\n".join(rows)


def parse_point(s):
    r, c = (int(v) for v in s.replace(" ", "").split(","))
    return r, c


# ==========================================================================
# Main
# ==========================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description="Solve a maze of any size.")
    ap.add_argument("maze", nargs="?", help="maze file: .png/.jpg, .npy, .npz, .txt")
    ap.add_argument("--generate", metavar="WxH", help="generate a random maze with W x H cells instead")
    ap.add_argument("--loops", type=float, default=0.0, help="with --generate: chance to knock extra holes")
    ap.add_argument("--seed", type=int, default=None, help="with --generate: random seed")
    ap.add_argument("--index", type=int, default=0, help="which maze inside an .npz dataset")
    ap.add_argument("--start", type=parse_point, help="start as ROW,COL")
    ap.add_argument("--goal", type=parse_point, help="goal as ROW,COL")
    ap.add_argument("--algo", choices=["bfs", "astar", "dijkstra", "ppo"], default="bfs",
                    help="ppo = the trained RL agent (any maze size), with BFS as a back-up")
    ap.add_argument("--model", default="maze_ppo_any_size", help="with --algo ppo: saved agent")
    ap.add_argument("--max-steps", type=int, help="with --algo ppo: moves the agent may make before BFS helps")
    ap.add_argument("--no-fallback", action="store_true", help="with --algo ppo: never let BFS help")
    ap.add_argument("--zero-is-wall", action="store_true", help="for 0/1 grids where 0 means wall")
    ap.add_argument("--max-side", type=int, default=1500, help="pictures larger than this are shrunk first")
    ap.add_argument("--no-centre", action="store_true", help="pictures: don't keep the path in the corridor middle")
    ap.add_argument("--out", default="solution.png", help="picture of the solution ('' = none)")
    ap.add_argument("--save-path", help="save the path as an (N, 2) .npy of (row, col)")
    ap.add_argument("--ascii", action="store_true", help="print the solved maze in the terminal")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):  # Windows consoles can't print "██" otherwise
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    t0 = time.time()
    if args.generate:
        from maze_genrator import generate_maze
        w, h = (int(v) for v in args.generate.lower().split("x"))
        grid, s, g = generate_maze(w, h, seed=args.seed, loop_prob=args.loops, corridor=1, wall=1)
        maze = Maze(grid == 1, s, g)
        name = f"random {w}x{h}-cell maze"
    elif args.maze:
        maze = load_maze(args.maze, args.index, not args.zero_is_wall, args.max_side)
        name = args.maze
    else:
        ap.error("give a maze file or --generate WxH")
    if args.start:
        maze.start = args.start
    if args.goal:
        maze.goal = args.goal

    start, goal = find_endpoints(maze)
    rl = None
    if args.algo == "ppo":
        from maze_rl_any_size import load_model, rl_solve
        maze, start, goal = shrink_for_agent(maze, start, goal)
        model = load_model(args.model)
        t1 = time.time()
        rl = rl_solve(model, maze.walls, start, goal, args.max_steps, fallback=not args.no_fallback)
        path = rl["path"]
    else:
        t1 = time.time()
        path = solve(maze.walls, start, goal, args.algo, centre=(maze.image is not None and not args.no_centre))
    t2 = time.time()

    H, W = maze.shape
    print(f"Maze: {name} | grid {H} x {W} ({H * W:,} cells) | start {start} -> goal {goal}")
    if path is None and rl is not None and not rl["solved_by_agent"] and args.no_fallback:
        print(f"The agent did not reach the goal in {rl['agent_steps']} moves (BFS back-up turned off).")
        return 1
    if path is None:
        print(f"NO PATH: the goal cannot be reached from the start. (load {t1 - t0:.2f}s, search {t2 - t1:.2f}s)")
        return 1
    print(f"Solved with {args.algo}: {len(path) - 1} steps | load {t1 - t0:.2f}s, search {t2 - t1:.2f}s")
    if rl is not None:
        if rl["solved_by_agent"]:
            print(f"The agent reached the goal on its own in {rl['agent_steps']} moves "
                  f"(path after removing dead ends: {len(path) - 1} steps).")
        else:
            print(f"The agent made {rl['agent_steps']} moves without reaching the goal; BFS finished the route. "
                  f"{rl['agent_share'] * 100:.0f}% of the final path came from the agent.")

    if args.out:
        draw_solution(maze, path, start, goal, args.out)
        print(f"Saved picture: {args.out}")
    if args.save_path:
        np.save(args.save_path, np.array(path, dtype=np.int32))
        print(f"Saved path: {args.save_path}")
    if args.ascii:
        if W <= 150:
            print(ascii_solution(maze.walls, path))
        else:
            print("(maze too wide to print, see the picture instead)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
