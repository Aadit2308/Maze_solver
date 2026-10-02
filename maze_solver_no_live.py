"""
PPO maze solver, ADAPTIVE reverse curriculum, difficulty-scaled rewards, GPU forced, NO live drawing (fast).
Uses mazes_train.npz (training), mazes_val.npz (progress checks), mazes_test.npz (final test).

    python train_maze_reverse.py --inspect    # check the npz is read correctly
    python train_maze_reverse.py              # train (console output only, fast)
    python train_maze_reverse.py --watch      # watch the saved model solve mazes

pip install stable-baselines3 torch gymnasium numpy matplotlib
"""
import sys
import time
from collections import deque

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv

DEVICE = "cuda"
if not torch.cuda.is_available():
    raise SystemExit("No CUDA GPU found: PyTorch is the CPU-only version or the driver is missing. "
                     "Fix: pip uninstall torch -y  then  "
                     "pip install torch --index-url https://download.pytorch.org/whl/cu121")
print("Using GPU:", torch.cuda.get_device_name(0))

# ==========================================================================
# SETTINGS
# ==========================================================================
TRAIN_NPZ = "mazes_train.npz"
VAL_NPZ = "mazes_val.npz"
TEST_NPZ = "mazes_test.npz"
MAZE_KEY = None          # array name inside the npz; None = auto-detect
ONE_IS_WALL = True       # for 0/1 arrays: True if 1 = wall.  (Images 0-255: dark = wall)
DOWNSAMPLE = 1           # keep 1 for your 101x101 mazes (1-cell corridors).
                         # Only use 2 for mazes with 3-cell corridors (e.g. 81x81 or 101x101 with 25 cells).
MAX_TRAIN_MAZES = 400000

# ---- Adaptive reverse curriculum ----
# Difficulty = how far back from the goal the agent starts (0.05 = 5% of the route, 1.0 = real entrance)
START_DIFF = 0.05        # first difficulty level
START_STEP = 0.05        # first jump size after a pass (5%)
MAX_STEP = 0.15          # biggest jump allowed (15%)
MIN_STEP = 0.01          # smallest jump allowed (1%)
MAX_STUCK_ROUNDS = 3     # stop after this many failed levels in a row
REPLAY_PROB = 0.3        # share of training episodes started from a random EASIER level (stops forgetting)
MIN_REPLAY_DIFF = 0.05   # easiest level used for replay
FULL_START_PROB = 0.05   # share of training episodes that always start at the real entrance

N_ENVS = 8
CHUNK_STEPS = 10_000
MAX_CHUNKS_PER_LEVEL = 20
PASS_RATE = 0.85
EVAL_MAZES = 60
FINAL_TEST_MAZES = 200
MODEL_PATH = "maze_ppo_reverse"

SHOW_EVERY_N_STEPS = 2   # (only for --watch) draw every n-th agent step
PAUSE = 0.02             # (only for --watch) seconds to pause for GUI event processing

# ---- Rewards: (value at difficulty 0.0, value at difficulty 1.0), blended in between ----
GOAL_REWARD_RANGE = (12.0, 30.0)     # bigger prize for longer routes
STEP_PENALTY_RANGE = (0.005, 0.02)   # stronger push to be efficient as it gets harder
WALL_PENALTY_RANGE = (0.03, 0.08)    # bumping into walls hurts more on hard levels
SHAPING_SCALE = 0.05

WALL, PATH = 1, 0
MOVES = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}  # up, down, left, right


def lerp(rng, d):
    return rng[0] + (rng[1] - rng[0]) * d


# ==========================================================================
# Data
# ==========================================================================
def load_grids(path, limit=None):
    data = np.load(path, allow_pickle=True)
    key = MAZE_KEY
    if key is None:
        for k in data.files:
            a = data[k]
            if "action" in k.lower() or "seed" in k.lower():
                continue
            if a.ndim in (3, 4) and a.shape[0] > 1 and a.dtype != object:
                key = k
                break
    if key is None:
        raise KeyError(f"No maze array found. Keys: { {k: data[k].shape for k in data.files} }. Set MAZE_KEY.")
    arr = np.asarray(data[key])[:limit]
    print(f"{path}: key '{key}', shape {arr.shape}, dtype {arr.dtype}")
    if arr.ndim == 4:
        arr = arr[..., :3].mean(-1) if arr.shape[-1] in (1, 3, 4) else arr[:, :3].mean(1)
    vals = np.unique(arr[:50])
    print(f"   values found: {vals[:10]}")
    if arr.max() > 10 or arr.min() < 0:      # looks like an image (0-255)
        a = arr.astype(np.float32)
        a = a - a.min()
        a = a / max(a.max(), 1.0)
        walls = a < 0.5                      # dark = wall
    else:                                    # small integer codes: 1 = wall, 0/2/3... = open
        walls = (arr == 1) if ONE_IS_WALL else (arr == 0)
    grids = walls.astype(np.int8)
    frac = grids.mean()
    print(f"   wall fraction: {frac:.2f} (normal mazes are roughly 0.5-0.7)")
    if frac < 0.3 or frac > 0.85:
        print("   WARNING: this looks inverted. Flip ONE_IS_WALL and try again.")
    if DOWNSAMPLE > 1:
        grids = grids[:, ::DOWNSAMPLE, ::DOWNSAMPLE]
        if grids[:5].min() == 1:
            raise RuntimeError("DOWNSAMPLE removed every open cell. Set DOWNSAMPLE = 1.")
    return np.ascontiguousarray(grids)


def bfs_dist(grid, target):
    rows, cols = grid.shape
    dist = np.full((rows, cols), -1, dtype=np.int16)
    dist[target] = 0
    q = deque([target])
    while q:
        r, c = q.popleft()
        for dr, dc in MOVES.values():
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == PATH and dist[nr, nc] == -1:
                dist[nr, nc] = dist[r, c] + 1
                q.append((nr, nc))
    return dist


def find_endpoints(g):
    H, W = g.shape
    top, bot = np.where(g[0] == PATH)[0], np.where(g[-1] == PATH)[0]
    if len(top) and len(bot):
        return (0, int(round(top.mean()))), (H - 1, int(round(bot.mean())))
    free = np.argwhere(g == PATH)
    if len(free) == 0:
        raise RuntimeError("Maze has no open cells: walls are read wrongly. Flip ONE_IS_WALL.")
    return tuple(free[0]), tuple(free[-1])


# ==========================================================================
# Environment
# ==========================================================================
class MazeEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, grids, difficulty=START_DIFF, full_start_prob=FULL_START_PROB):
        super().__init__()
        self.grids = grids
        self.H, self.W = grids.shape[1:]
        self.difficulty = difficulty
        self.full_start_prob = full_start_prob
        self.ep_diff = difficulty
        self.goal_r = lerp(GOAL_REWARD_RANGE, difficulty)
        self.step_p = lerp(STEP_PENALTY_RANGE, difficulty)
        self.wall_p = lerp(WALL_PENALTY_RANGE, difficulty)
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(0.0, 1.0, (4, self.H, self.W), dtype=np.float32)

    def set_difficulty(self, d):
        self.difficulty = d

    def _obs(self):
        o = np.zeros((4, self.H, self.W), dtype=np.float32)
        o[0] = self.grid
        o[1][self.agent] = 1.0
        o[2][self.goal] = 1.0
        o[3] = self.visited
        return o

    def reset(self, seed=None, options=None, maze_index=None, difficulty=None, full=False):
        super().reset(seed=seed)
        while True:
            mi = maze_index if maze_index is not None else int(self.np_random.integers(len(self.grids)))
            g = self.grids[mi]
            s, t = find_endpoints(g)
            d = bfs_dist(g, t)
            if g[s] == PATH and d[s] > 0:
                break
            if maze_index is not None:
                raise RuntimeError("Maze has no path from start to goal")
        self.grid, self.dist, self.goal, self.real_start = g, d, t, s
        self.optimal_length = int(d[s])

        diff = self.difficulty if difficulty is None else difficulty
        if difficulty is None:
            # TRAINING episode only: sometimes replay an easier level, sometimes start at the real entrance.
            # (Evaluation / watching pass `difficulty` explicitly, so they are never changed by this.)
            if self.np_random.random() < REPLAY_PROB:
                diff = float(self.np_random.uniform(MIN_REPLAY_DIFF, max(self.difficulty, MIN_REPLAY_DIFF)))
            if self.np_random.random() < self.full_start_prob:
                diff = 1.0
        if full:
            diff = 1.0
        self.ep_diff = diff

        # rewards for this episode depend on its difficulty
        self.goal_r = lerp(GOAL_REWARD_RANGE, diff)
        self.step_p = lerp(STEP_PENALTY_RANGE, diff)
        self.wall_p = lerp(WALL_PENALTY_RANGE, diff)

        target = max(1, int(round(diff * self.optimal_length)))
        cur = s
        while d[cur] > target:  # walk along the shortest path toward the goal
            for dr, dc in MOVES.values():
                n = (cur[0] + dr, cur[1] + dc)
                if 0 <= n[0] < self.H and 0 <= n[1] < self.W and d[n] == d[cur] - 1:
                    cur = n
                    break
        self.start = cur
        self.start_dist = int(d[cur])
        self.max_steps = 4 * self.start_dist + 50
        self.agent = cur
        self.visited = np.zeros((self.H, self.W), dtype=np.float32)
        self.visited[cur] = 1.0
        self.steps = 0
        self.trail = [cur]
        return self._obs(), {"steps": 0, "optimal_length": self.start_dist}

    def step(self, action):
        self.steps += 1
        dr, dc = MOVES[int(action)]
        nr, nc = self.agent[0] + dr, self.agent[1] + dc
        reward = -self.step_p
        if not (0 <= nr < self.H and 0 <= nc < self.W) or self.grid[nr, nc] == WALL:
            reward -= self.wall_p
        else:
            reward += SHAPING_SCALE * (self.dist[self.agent] - self.dist[nr, nc])
            self.agent = (nr, nc)
            self.visited[self.agent] = 1.0
            self.trail.append(self.agent)
        terminated = self.agent == self.goal
        if terminated:
            reward += self.goal_r
        truncated = (not terminated) and self.steps >= self.max_steps
        info = {"steps": self.steps, "optimal_length": self.start_dist}
        return self._obs(), float(reward), bool(terminated), bool(truncated), info


class MazeCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        ch = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(ch, 32, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1, dilation=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 2, dilation=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 4, dilation=4), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flat = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flat, features_dim), nn.ReLU())

    def forward(self, x):
        return self.linear(self.cnn(x))


# ==========================================================================
# Evaluation (no drawing)
# ==========================================================================
def run_episode(model, env, **reset_kw):
    obs, info = env.reset(**reset_kw)
    done = term = False
    while not done:
        a, _ = model.predict(obs, deterministic=False)
        obs, _, term, trunc, info = env.step(int(a))
        done = term or trunc
    return term, info


def evaluate(model, val_grids, difficulty, n, seed=123):
    env = MazeEnv(val_grids)
    picks = np.random.default_rng(seed).choice(len(val_grids), size=min(n, len(val_grids)), replace=False)
    solved, ratios = 0, []
    for mi in picks:
        term, info = run_episode(model, env, maze_index=int(mi), difficulty=difficulty, full=(difficulty >= 1.0))
        if term:
            solved += 1
            ratios.append(info["steps"] / max(1, info["optimal_length"]))
    return solved / len(picks), (float(np.mean(ratios)) if ratios else float("nan"))


# ==========================================================================
# Viewer (only used by --watch, never during training)
# ==========================================================================
class Viewer:
    def __init__(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(7, 7))
        self.cur_grid = None
        self.fig.canvas.draw()
        plt.show(block=False)
        plt.pause(0.05)

    def draw_env(self, env, title):
        if self.cur_grid is not env.grid:
            self.ax.clear()
            self.ax.imshow(env.grid, cmap="gray_r", interpolation="nearest")
            self.trail_line, = self.ax.plot([], [], "-", color="orange", lw=1.2, alpha=0.8)
            self.start_pt, = self.ax.plot([], [], "s", color="blue", ms=7)
            self.goal_pt, = self.ax.plot([], [], "v", color="green", ms=10)
            self.agent_pt, = self.ax.plot([], [], "o", color="red", ms=7)
            self.ax.axis("off")
            self.cur_grid = env.grid
        ys, xs = zip(*env.trail)
        self.trail_line.set_data(xs, ys)
        self.start_pt.set_data([env.start[1]], [env.start[0]])
        self.goal_pt.set_data([env.goal[1]], [env.goal[0]])
        self.agent_pt.set_data([env.agent[1]], [env.agent[0]])
        self.ax.set_title(title)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(PAUSE)

    def watch(self, model, grids, difficulty, label=""):
        env = MazeEnv(grids)
        obs, info = env.reset(difficulty=difficulty, full=(difficulty >= 1.0))
        done = term = False
        self.draw_env(env, f"{label} | step 0 (shortest {env.start_dist})")
        while not done:
            a, _ = model.predict(obs, deterministic=False)
            obs, _, term, trunc, info = env.step(int(a))
            done = term or trunc
            if env.steps % SHOW_EVERY_N_STEPS == 0 or done:
                self.draw_env(env, f"{label} | step {env.steps} (shortest {env.start_dist})"
                                   + ("  SOLVED" if term else ""))
        plt.pause(0.7)


# ==========================================================================
# Main
# ==========================================================================
if __name__ == "__main__":
    train = load_grids(TRAIN_NPZ, MAX_TRAIN_MAZES)
    val = load_grids(VAL_NPZ)
    print(f"Maze grid {train.shape[1]}x{train.shape[2]} | train {len(train)} | val {len(val)}")

    if "--inspect" in sys.argv:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        env = MazeEnv(train)
        for ax, diff in zip(axes, (0.1, 0.5, 1.0)):
            env.reset(difficulty=diff, full=(diff >= 1.0))
            ax.imshow(env.grid, cmap="gray_r", interpolation="nearest")
            ax.plot(env.start[1], env.start[0], "s", color="blue")
            ax.plot(env.goal[1], env.goal[0], "v", color="green")
            ax.set_title(f"start at {diff * 100:.0f}% (path {env.start_dist})")
            ax.axis("off")
        print("Walls should be black. Blue = start, green = goal. If inverted, flip ONE_IS_WALL.")
        plt.show()
        sys.exit()

    if "--watch" in sys.argv:
        model = PPO.load(MODEL_PATH, device=DEVICE)
        viewer = Viewer()
        print("Watching saved model. Close the window or press Ctrl+C to stop.")
        while plt.fignum_exists(viewer.fig.number):
            viewer.watch(model, val, 1.0, label="saved model")
        sys.exit()

    vec_env = DummyVecEnv([lambda: MazeEnv(train, START_DIFF) for _ in range(N_ENVS)])
    model = PPO(
        "CnnPolicy", vec_env,
        policy_kwargs=dict(features_extractor_class=MazeCNN,
                           features_extractor_kwargs=dict(features_dim=256),
                           normalize_images=False),
        learning_rate=3e-4, n_steps=256, batch_size=256, n_epochs=4,
        gamma=0.995, gae_lambda=0.95, ent_coef=0.01, verbose=0, device=DEVICE,
    )
    print("Model is running on:", model.device)

    t0 = time.time()

    # ---- adaptive curriculum ----
    diff = START_DIFF        # level we are training on right now
    last_good = 0.0          # highest level that was passed
    step_size = START_STEP   # how big the next jump is
    stuck = 0                # failed levels in a row

    while True:
        print(f"\n=== Start {diff * 100:.1f}% of the way back from the goal "
              f"(last passed: {last_good * 100:.1f}%, jump size {step_size * 100:.1f}%) ===")
        vec_env.env_method("set_difficulty", diff)
        passed = False
        for chunk in range(1, MAX_CHUNKS_PER_LEVEL + 1):
            model.learn(total_timesteps=CHUNK_STEPS, reset_num_timesteps=False)
            rate, ratio = evaluate(model, val, diff, EVAL_MAZES)
            print(f"  chunk {chunk}: solved {rate * 100:.0f}% | steps vs shortest {ratio:.2f}x | "
                  f"total {model.num_timesteps} | {(time.time() - t0) / 60:.1f} min")
            model.save(MODEL_PATH)
            if rate >= PASS_RATE:
                passed = True
                print("  passed")
                break

        if passed:
            last_good = diff
            stuck = 0
            if diff >= 1.0:
                print("\nReached the real entrance and passed. Training done.")
                break
            step_size = min(step_size * 1.5, MAX_STEP)       # success -> bigger jump
            diff = min(1.0, round(last_good + step_size, 3))
        else:
            stuck += 1
            if stuck > MAX_STUCK_ROUNDS:
                print(f"\nStuck {stuck} times in a row near {last_good * 100:.1f}%. Stopping training.")
                break
            step_size = max(step_size / 2, MIN_STEP)         # failure -> smaller jump from last good level
            diff = round(last_good + step_size, 3)
            print(f"  did not reach pass rate. Going back to {last_good * 100:.1f}% "
                  f"and trying a smaller jump to {diff * 100:.1f}%")

    test = load_grids(TEST_NPZ)
    rate, ratio = evaluate(model, test, 1.0, FINAL_TEST_MAZES, seed=999)
    print(f"\nFINAL TEST (real entrance, unseen mazes): solved {rate * 100:.0f}% "
          f"of {min(FINAL_TEST_MAZES, len(test))} | {ratio:.2f}x shortest. Saved {MODEL_PATH}.zip")
