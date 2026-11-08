"""
PPO maze agent that works on mazes of ANY size.

Why the original agent can't do this: it looks at the whole 41 x 41 maze at once, and its
CNN ends in a Flatten -> Linear layer whose size is fixed by that 41 x 41 input.

This agent instead sees:
  * a fixed VIEW x VIEW window of the maze centred on itself (walls, where it has been, the goal
    if it is in view), and
  * which direction the goal is in, and roughly how far.
That input has the same shape for a 10 x 10 maze and a 2000 x 2000 maze, so one trained
model runs on any of them. It is trained on randomly generated mazes of many sizes,
corridor widths, wall thicknesses, loop amounts and start/goal positions, using a reverse
curriculum (start near the goal, then further back, then bigger mazes).

When it is used to solve a maze, BFS takes over if the agent runs out of steps, so every
solvable maze still gets solved (the output says how much of the route the agent found itself).

    python maze_rl_any_size.py train                    # train (uses the GPU if there is one)
    python maze_rl_any_size.py train --resume           # continue training a saved model
    python maze_rl_any_size.py benchmark                # agent vs BFS on 10..160-cell mazes
    python maze_rl_any_size.py solve maze.png           # agent solves a maze file (any format)
    python solve_any_maze.py maze.png --algo ppo        # same thing through the main solver

pip install stable-baselines3 torch gymnasium numpy scipy pillow
"""
import argparse
import sys
import time

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn

from maze_genrator import generate_maze
import solve_any_maze as sam

# ==========================================================================
# SETTINGS
# ==========================================================================
MODEL_PATH = "maze_ppo_any_size"
VIEW = 21                 # the agent sees a VIEW x VIEW window around itself (odd number)
R = VIEW // 2
N_CH = 6

# Curriculum levels: (difficulty, largest maze in cells). Difficulty = how far back from
# the goal the agent starts (0.15 = 15% of the shortest route, 1.0 = the real start).
LEVELS = [
    (0.15, 6), (0.3, 8), (0.5, 10), (0.75, 12), (1.0, 12), (1.0, 16),
    (1.0, 20), (1.0, 25), (1.0, 30), (1.0, 40), (1.0, 50), (1.0, 60),
]
MIN_CELLS = 3
REPLAY_PROB = 0.3         # share of episodes from an easier, already-passed level (stops forgetting)
RANDOM_ENDPOINTS = 0.5    # share of mazes with start/goal at random open cells instead of the border

N_ENVS = 16
CHUNK_STEPS = 50_000
MAX_CHUNKS_PER_LEVEL = 30
PASS_RATE = 0.85
EVAL_EPISODES = 100

GOAL_REWARD = 10.0
STEP_PENALTY = 0.01
WALL_PENALTY = 0.05
REVISIT_PENALTY = 0.02
SHAPING_SCALE = 0.05

MOVES = sam.MOVES  # 0 up, 1 down, 2 left, 3 right


# ==========================================================================
# Helpers
# ==========================================================================
def distance_to(walls, target):
    """Steps from every cell to `target` (-1 = wall / unreachable). Fast on big grids."""
    free = ~walls
    out = np.full(walls.shape, -1, dtype=np.int32)
    if sam.HAVE_SCIPY:
        from scipy.sparse.csgraph import dijkstra
        g, idx = sam._build_graph(free)
        d = dijkstra(g, directed=False, indices=int(idx[target]), unweighted=True)
        ok = np.isfinite(d)
        cells = np.argwhere(free)[ok]
        out[cells[:, 0], cells[:, 1]] = d[ok].astype(np.int32)
        return out
    from maze_genrator import distance_map
    return distance_map((walls).astype(np.int8), target)


def random_maze(rng, max_cells):
    """A random maze with random size / corridor / wall / loops / endpoints / orientation."""
    w = int(rng.integers(MIN_CELLS, max_cells + 1))
    h = int(rng.integers(MIN_CELLS, max_cells + 1))
    corridor = int(rng.choice([1, 1, 1, 2, 3]))
    wall = int(rng.choice([1, 1, 2]))
    loops = float(rng.choice([0.0, 0.0, 0.05, 0.15, 0.3]))
    grid, s, g = generate_maze(w, h, seed=int(rng.integers(2 ** 31)), loop_prob=loops,
                               corridor=corridor, wall=wall)
    walls = grid == 1
    if rng.random() < RANDOM_ENDPOINTS:
        free = np.argwhere(~walls)
        a, b = rng.choice(len(free), size=2, replace=False)
        s, g = tuple(int(v) for v in free[a]), tuple(int(v) for v in free[b])
    # random flips / transpose so the agent doesn't learn "always go down-right"
    H, W = walls.shape
    if rng.random() < 0.5:
        walls, s, g = walls[::-1], (H - 1 - s[0], s[1]), (H - 1 - g[0], g[1])
    if rng.random() < 0.5:
        walls, s, g = walls[:, ::-1], (s[0], W - 1 - s[1]), (g[0], W - 1 - g[1])
    if rng.random() < 0.5:
        walls, s, g = walls.T, (s[1], s[0]), (g[1], g[0])
    return np.ascontiguousarray(walls), s, g


def loop_erase(path):
    """Remove the loops from a walk, leaving a simple path with the same ends."""
    out, where = [], {}
    for p in path:
        if p in where:
            cut = where[p]
            for q in out[cut + 1:]:
                del where[q]
            out = out[:cut + 1]
        else:
            where[p] = len(out)
            out.append(p)
    return out


# ==========================================================================
# Environment
# ==========================================================================
class AnySizeMazeEnv(gym.Env):
    """
    Observation (N_CH, VIEW, VIEW), agent always in the centre:
      0  walls (outside the maze counts as wall)
      1  how often each tile was visited (0, 1/3, 2/3, 1)
      2  the goal, if it is inside the window
      3  goal direction, rows    (dr / (|dr| + |dc|)), same value everywhere
      4  goal direction, columns (dc / (|dr| + |dc|)), same value everywhere
      5  how far the goal is, log-scaled to 0..1, same value everywhere
    """
    metadata = {"render_modes": []}

    def __init__(self, level=0, train=True):
        super().__init__()
        self.observation_space = spaces.Box(-1.0, 1.0, (N_CH, VIEW, VIEW), dtype=np.float32)
        self.action_space = spaces.Discrete(4)
        self.level = level
        self.train = train

    def set_level(self, level):
        self.level = level

    # ----------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        rng = self.np_random
        if "walls" in options:                        # a given maze (solving / evaluation)
            walls, s, g = options["walls"], tuple(options["start"]), tuple(options["goal"])
            diff = options.get("difficulty", 1.0)
            need_dist = self.train or diff < 1.0
        else:                                         # a random training maze
            lvl = self.level
            if lvl > 0 and not options.get("no_replay") and rng.random() < REPLAY_PROB:
                lvl = int(rng.integers(0, lvl))
            diff, cells = LEVELS[lvl]
            diff = options.get("difficulty", diff)
            cells = options.get("cells", cells)
            while True:
                walls, s, g = random_maze(rng, cells)
                if s != g:
                    break
            need_dist = True

        self.walls, self.goal = walls, g
        self.H, self.W = walls.shape
        self.dist = distance_to(walls, g) if need_dist else None
        if self.dist is not None and self.dist[s] < 0:
            if "walls" in options:
                raise ValueError("The goal cannot be reached from the start.")
            return self.reset(options=options)

        # reverse curriculum: start part of the way back along the shortest route
        if self.dist is not None and diff < 1.0:
            target = max(1, int(round(diff * self.dist[s])))
            cur = s
            while self.dist[cur] > target:
                for dr, dc in MOVES.values():
                    n = (cur[0] + dr, cur[1] + dc)
                    if 0 <= n[0] < self.H and 0 <= n[1] < self.W and self.dist[n] == self.dist[cur] - 1:
                        cur = n
                        break
            s = cur
        self.start = self.agent = s
        self.start_dist = int(self.dist[s]) if self.dist is not None else -1
        self.max_steps = options.get("max_steps", 4 * max(self.start_dist, 1) + 50)

        self.pad_walls = np.pad(walls, R, constant_values=True)
        self.pad_visits = np.zeros(self.pad_walls.shape, dtype=np.uint8)
        self.pad_visits[s[0] + R, s[1] + R] = 1
        self.steps = 0
        self.trail = [s]
        return self._obs(), {"optimal_length": self.start_dist}

    def _obs(self):
        r, c = self.agent
        o = np.zeros((N_CH, VIEW, VIEW), dtype=np.float32)
        o[0] = self.pad_walls[r:r + VIEW, c:c + VIEW]
        o[1] = np.minimum(self.pad_visits[r:r + VIEW, c:c + VIEW], 3) / 3.0
        dr, dc = self.goal[0] - r, self.goal[1] - c
        if abs(dr) <= R and abs(dc) <= R:
            o[2, R + dr, R + dc] = 1.0
        n = abs(dr) + abs(dc)
        if n:
            o[3] = dr / n
            o[4] = dc / n
        o[5] = min(1.0, np.log1p(n) / np.log1p(5000))
        return o

    def step(self, action):
        self.steps += 1
        dr, dc = MOVES[int(action)]
        r, c = self.agent
        nr, nc = r + dr, c + dc
        reward = -STEP_PENALTY
        if not (0 <= nr < self.H and 0 <= nc < self.W) or self.walls[nr, nc]:
            reward -= WALL_PENALTY
        else:
            if self.dist is not None:
                reward += SHAPING_SCALE * (self.dist[r, c] - self.dist[nr, nc])
            if self.pad_visits[nr + R, nc + R]:
                reward -= REVISIT_PENALTY
            self.agent = (nr, nc)
            self.pad_visits[nr + R, nc + R] = min(255, self.pad_visits[nr + R, nc + R] + 1)
            self.trail.append(self.agent)
        terminated = self.agent == self.goal
        if terminated:
            reward += GOAL_REWARD
        truncated = (not terminated) and self.steps >= self.max_steps
        return self._obs(), float(reward), bool(terminated), bool(truncated), {"steps": self.steps}


# ==========================================================================
# Network (input is always VIEW x VIEW, whatever the maze size)
# ==========================================================================
def make_extractor():
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

    class LocalViewCNN(BaseFeaturesExtractor):
        def __init__(self, observation_space, features_dim=256):
            super().__init__(observation_space, features_dim)
            ch = observation_space.shape[0]
            self.cnn = nn.Sequential(
                nn.Conv2d(ch, 32, 3, 1, 1), nn.ReLU(),
                nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(),
                nn.Conv2d(64, 64, 3, 2, 1), nn.ReLU(),
                nn.Conv2d(64, 64, 3, 2, 1), nn.ReLU(),
                nn.Flatten(),
            )
            with torch.no_grad():
                n = self.cnn(torch.zeros(1, *observation_space.shape)).shape[1]
            self.linear = nn.Sequential(nn.Linear(n, features_dim), nn.ReLU())

        def forward(self, x):
            return self.linear(self.cnn(x))

    return LocalViewCNN


def load_model(path=MODEL_PATH, device="auto"):
    from stable_baselines3 import PPO
    return PPO.load(path, device=device, custom_objects={"policy_kwargs": dict(
        features_extractor_class=make_extractor(),
        features_extractor_kwargs=dict(features_dim=256), normalize_images=False)})


# ==========================================================================
# Solving a given maze with the agent
# ==========================================================================
def rl_solve(model, walls, start, goal, max_steps=None, fallback=True, deterministic=False, seed=0):
    """
    Let the agent walk from start to goal. Returns a dict:
      path          simple path start -> goal (loops removed), or None
      solved_by_agent  True if the agent reached the goal on its own
      agent_steps   moves the agent made (including dead ends / bumps)
      agent_share   fraction of the final path that the agent found itself
    """
    walls = np.asarray(walls, dtype=bool)
    start, goal = tuple(start), tuple(goal)
    if max_steps is None:
        max_steps = int(min(300_000, max(2_000, 3 * (~walls).sum())))
    env = AnySizeMazeEnv(train=False)
    obs, _ = env.reset(seed=seed, options=dict(walls=walls, start=start, goal=goal, max_steps=max_steps))
    done = term = False
    while not done:
        a, _ = model.predict(obs, deterministic=deterministic)
        obs, _, term, trunc, _ = env.step(int(a))
        done = term or trunc
    agent_path = loop_erase(env.trail)
    result = dict(solved_by_agent=bool(term), agent_steps=env.steps, path=agent_path if term else None)
    if not term and fallback:
        rest = sam.solve_bfs(walls, env.agent, goal)
        if rest is not None:
            result["path"] = loop_erase(agent_path + rest[1:])
    if result["path"] is not None:
        on_agent = set(agent_path)
        result["agent_share"] = sum(p in on_agent for p in result["path"]) / len(result["path"])
    else:
        result["agent_share"] = 0.0
    return result


# ==========================================================================
# Training / evaluation
# ==========================================================================
def run_batch(model, episodes, deterministic=False):
    """
    Run many episodes side by side, with ONE network call per step for all of them.
    episodes: list of (env, reset_options, seed). Returns a list of (reached_goal, env).
    """
    obs = [env.reset(seed=seed, options=opts)[0] for env, opts, seed in episodes]
    done = [None] * len(episodes)
    active = list(range(len(episodes)))
    while active:
        acts, _ = model.predict(np.stack([obs[i] for i in active]), deterministic=deterministic)
        still = []
        for i, a in zip(active, np.atleast_1d(acts)):
            obs[i], _, term, trunc, _ = episodes[i][0].step(int(a))
            if term or trunc:
                done[i] = term
            else:
                still.append(i)
        active = still
    return [(bool(done[i]), episodes[i][0]) for i in range(len(episodes))]


def evaluate(model, level, n=EVAL_EPISODES, seed=123):
    """Share of random mazes at this level the agent solves on its own (no BFS help)."""
    eps = []
    for i in range(n):
        env = AnySizeMazeEnv(level=level)
        env.level = level
        eps.append((env, {"difficulty": LEVELS[level][0], "cells": LEVELS[level][1], "no_replay": True}, seed + i))
    res = run_batch(model, eps)
    ratios = [(len(loop_erase(env.trail)) - 1) / max(1, env.start_dist) for ok, env in res if ok]
    return sum(ok for ok, _ in res) / n, (float(np.mean(ratios)) if ratios else float("nan"))


def train(args):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}. Levels: {len(LEVELS)}, chunk {args.chunk:,} steps, {args.envs} envs.")

    def make(i):
        return lambda: AnySizeMazeEnv(level=args.start_level)
    vec_cls = SubprocVecEnv if args.subproc else DummyVecEnv
    vec = vec_cls([make(i) for i in range(args.envs)])
    vec.seed(args.seed)

    if args.resume:
        model = load_model(args.model, device=device)
        model.set_env(vec)
    else:
        model = PPO(
            "CnnPolicy", vec,
            policy_kwargs=dict(features_extractor_class=make_extractor(),
                               features_extractor_kwargs=dict(features_dim=256),
                               normalize_images=False),
            learning_rate=3e-4, n_steps=256, batch_size=1024, n_epochs=4,
            gamma=0.995, gae_lambda=0.95, ent_coef=0.01, verbose=0, device=device, seed=args.seed,
        )

    t0 = time.time()
    for lvl in range(args.start_level, min(args.end_level, len(LEVELS))):
        diff, cells = LEVELS[lvl]
        print(f"\n=== Level {lvl}: start {diff * 100:.0f}% of the way back, mazes up to {cells}x{cells} cells ===")
        vec.env_method("set_level", lvl)
        for chunk in range(1, args.max_chunks + 1):
            model.learn(total_timesteps=args.chunk, reset_num_timesteps=False)
            rate, ratio = evaluate(model, lvl, args.eval)
            print(f"  chunk {chunk}: solved {rate * 100:.0f}% | path vs shortest {ratio:.2f}x | "
                  f"total {model.num_timesteps:,} | {(time.time() - t0) / 60:.1f} min", flush=True)
            model.save(args.model)
            if rate >= args.pass_rate:
                print("  passed")
                break
        else:
            print("  did not reach the pass rate, moving on anyway")
    vec.close()
    print(f"\nSaved {args.model}.zip")
    if not args.no_benchmark:
        benchmark(model)


def benchmark(model, sizes=(10, 20, 40, 80, 160), n=20, seed=777):
    """Agent alone (no BFS help) vs BFS on square mazes of growing size, border start/goal."""
    print("\nBenchmark (agent on its own, real entrance -> exit, unseen mazes):", flush=True)
    print(f"  {'cells':>9} {'grid':>9} {'solved':>7} {'path/BFS':>9} {'moves/BFS':>10}")
    rng = np.random.default_rng(seed)
    for size in sizes:
        eps, best = [], []
        for i in range(n):
            grid, s, g = generate_maze(size, size, seed=int(rng.integers(2 ** 31)), corridor=1, wall=1,
                                       loop_prob=float(rng.choice([0.0, 0.1])))
            walls = grid == 1
            b = len(sam.solve_bfs(walls, s, g)) - 1
            best.append(b)
            eps.append((AnySizeMazeEnv(train=False),
                        dict(walls=walls, start=s, goal=g, max_steps=10 * b + 200), i))
        res = run_batch(model, eps)
        plen = [(len(loop_erase(env.trail)) - 1) / b for (ok, env), b in zip(res, best) if ok]
        moves = [env.steps / b for (ok, env), b in zip(res, best) if ok]
        f = lambda v: f"{np.mean(v):.2f}x" if v else "-"
        print(f"  {size:>4}x{size:<4} {walls.shape[0]:>4}x{walls.shape[1]:<4} "
              f"{len(plen) / n * 100:>6.0f}% {f(plen):>9} {f(moves):>10}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description="PPO maze agent for mazes of any size.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--model", default=MODEL_PATH)
    t.add_argument("--resume", action="store_true")
    t.add_argument("--start-level", type=int, default=0)
    t.add_argument("--end-level", type=int, default=len(LEVELS), help="stop before this level")
    t.add_argument("--no-benchmark", action="store_true")
    t.add_argument("--chunk", type=int, default=CHUNK_STEPS)
    t.add_argument("--max-chunks", type=int, default=MAX_CHUNKS_PER_LEVEL)
    t.add_argument("--envs", type=int, default=N_ENVS)
    t.add_argument("--eval", type=int, default=EVAL_EPISODES)
    t.add_argument("--pass-rate", type=float, default=PASS_RATE)
    t.add_argument("--subproc", action="store_true", help="run envs in separate processes")
    t.add_argument("--seed", type=int, default=0)
    b = sub.add_parser("benchmark")
    b.add_argument("--model", default=MODEL_PATH)
    b.add_argument("--sizes", default="10,20,40,80,160")
    b.add_argument("--n", type=int, default=20)
    s = sub.add_parser("solve")
    s.add_argument("maze")
    s.add_argument("--model", default=MODEL_PATH)
    args, rest = ap.parse_known_args(argv)

    if args.cmd == "train":
        train(args)
    elif args.cmd == "benchmark":
        benchmark(load_model(args.model), tuple(int(v) for v in args.sizes.split(",")), args.n)
    else:
        return sam.main([args.maze, "--algo", "ppo", "--model", args.model] + rest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
