# 🧩 Maze Solver using Reinforcement Learning

A maze-solving project that combines **classical pathfinding algorithms** with **reinforcement learning**.

The project has two main parts:

1. **Maze generation and shortest-path calculation**
   - Mazes are generated using **Depth-First Search (DFS)**.
   - **Breadth-First Search (BFS)** is used to calculate the shortest path from the entrance to the goal.
   - These mazes are stored as datasets for training, validation, and testing.

2. **Learning to solve mazes**
   - A reinforcement-learning agent is trained using **Proximal Policy Optimization (PPO)**.
   - A custom **Convolutional Neural Network (CNN)** processes the maze.
   - A **reverse curriculum** gradually increases the difficulty of the task.
   - The trained agent is evaluated on unseen mazes.

The important idea is:

```text
              MAZE GENERATION
                     │
                     ▼
              DFS Maze Generator
                     │
                     ▼
              Generated Mazes
                     │
                     ├───────────────┐
                     ▼               ▼
                  BFS Solver       Dataset
                     │               │
                     ▼               ▼
              Shortest Path      Train / Val / Test
                                     │
                                     ▼
                              Reinforcement Learning
                                     │
                                     ▼
                                    PPO
                                     │
                                     ▼
                               Trained Agent
                                     │
                                     ▼
                              Unseen Test Mazes
```

---

# 📌 What is this project?

Imagine giving a computer a maze like this:

```text
████████████████
█S█       █    █
█ █ █████ █ ██ █
█ █     █ █    █
█ █████ █ ██████
█     █ █      █
█████ █ ██████ █
█     █       G█
████████████████
```

Here:

- `S` = starting point
- `G` = goal
- `█` = wall
- empty space = path

A traditional algorithm such as BFS can directly search the maze and find a shortest path.

This project goes one step further.

Instead of telling the agent exactly which direction to move, we train an **RL agent** to look at the maze and learn which action is useful.

The agent can choose:

```text
0 → UP
1 → DOWN
2 → LEFT
3 → RIGHT
```

The agent receives rewards and penalties while interacting with the maze and gradually learns a navigation policy.

---

# 🎯 Project Goal

The goal is **not simply to find a path through one maze**.

The goal is to train an agent that can learn a general navigation strategy and apply it to many different maze layouts.

During training:

```text
Maze #1 ─┐
Maze #2  │
Maze #3  │
Maze #4  ├──► PPO Agent ──► Learned Navigation Policy
Maze #5  │
...      │
Maze #N ─┘
```

The final model is then tested on mazes that were not used for training.

---

# 🧠 Algorithms Used

This project uses several algorithms. Each algorithm has a different job.

| Algorithm / Technique | Purpose |
|---|---|
| **DFS** | Generate mazes |
| **BFS** | Calculate shortest paths |
| **CNN** | Understand the spatial structure of the maze |
| **PPO** | Learn the maze-navigation policy |
| **Reverse Curriculum** | Gradually increase training difficulty |
| **Reward Shaping** | Encourage movement toward the goal |

Understanding this separation is important:

> **BFS knows how to solve the maze. PPO learns how to solve the maze.**

BFS is used to create the reference shortest-path information and to calculate distances. The PPO agent does not receive the BFS action sequence as its policy input.

---

# 🗂️ Project Structure

```text
Maze_solver/
│
├── maze_genrator.py
│   └── Generates mazes and calculates their shortest paths
│
├── maze_solver.py
│   └── Gymnasium environment + PPO training + evaluation
│
├── maze_solver_no_live.py
│   └── Non-live version of the solver
│
├── mazes_train.npz
│   └── Training dataset
│
├── mazes_val.npz
│   └── Validation dataset
│
├── mazes_test.npz
│   └── Test dataset
│
├── maze.png
│   └── Example maze image
│
├── maze.txt
│   └── Example maze representation
│
├── LICENSE
│
└── README.md
```

The current repository contains these files on the `main` branch.

> **Note:** The existing generator filename is `maze_genrator.py`. The spelling is currently part of the repository, so the commands below use that exact filename.

---

# 🧱 Part 1 — Maze Generation

Maze generation is implemented in:

```text
maze_genrator.py
```

The generator represents the maze as a NumPy array:

```text
1 = WALL
0 = PATH
```

The current configuration generates:

```text
Number of mazes : 5000
Maze size       : 50 × 50 logical cells
Corridor width  : 1
Wall size       : 1
Loop probability: 0.5
Rooms           : 0
```

The dataset is split into:

```text
80% → Training
10% → Validation
10% → Testing
```

These values are configurable in the generator.

---

# 🌳 DFS Maze Generation

## What is DFS?

DFS stands for **Depth-First Search**.

It is usually taught as a graph traversal algorithm, but it can also be used to generate mazes.

The basic idea is:

```text
1. Start at the first cell.
2. Mark it as visited.
3. Pick an unvisited neighbour.
4. Move to that neighbour.
5. Repeat.
6. If there are no unvisited neighbours:
       go back to the previous cell.
```

That last step is called **backtracking**.

---

## Simple Example

Imagine these cells:

```text
A B C
D E F
G H I
```

DFS might explore:

```text
A → B → C → F → I → H → E → D → G
```

When it reaches a cell where it cannot continue, it goes backwards until it finds another unvisited cell.

The project implements this using a stack:

```python
stack = [(0, 0)]

while stack:
    current = stack[-1]

    # Find unvisited neighbours

    if neighbours:
        # Choose one
        # Carve a path
        # Push it onto the stack
    else:
        # Backtrack
        stack.pop()
```

This is exactly the maze-carving strategy implemented in the repository.

---

# 🔁 Adding Loops

A pure DFS maze tends to produce a tree-like maze with very few alternative routes.

This project optionally opens additional walls to create loops.

The important parameter is:

```python
LOOP_PROB = 0.5
```

Conceptually:

```text
LOOP_PROB = 0.0
        │
        ▼
Almost no loops

LOOP_PROB = 0.5
        │
        ▼
More alternative routes

LOOP_PROB = 1.0
        │
        ▼
Many additional connections
```

The current generator checks potential horizontal and vertical connections and opens some of them according to the probability.

---

# 🚪 Entrance and Exit

The generated maze has:

```text
START
  ↓
Top border
  │
  │
 MAZE
  │
  │
  ↓
GOAL
Bottom border
```

The generator creates an entrance on the top border and an exit on the bottom border.

---

# 🧭 Part 2 — Finding the Shortest Path with BFS

After generating a maze, the project needs to know whether the maze is solvable and what the shortest path is.

For this, it uses **Breadth-First Search (BFS)**.

---

# 🔍 What is BFS?

BFS explores nodes level by level.

Suppose the agent starts here:

```text
S . . . 
. . . .
. . . G
```

BFS first checks cells one step away:

```text
distance = 0 → S
distance = 1 → neighbours of S
distance = 2 → neighbours of those cells
distance = 3 → ...
```

It continues until it reaches the goal.

Because every valid movement costs exactly one step, the first time BFS reaches the goal, it has found a shortest path.

---

# 🌊 BFS Distance Map

The project actually performs BFS **backwards from the goal**.

For example:

```text
Goal
 ↓

0 1 2 3
1 2 3 4
2 3 4 5
3 4 5 6
```

Each number represents:

> "How many moves are required to reach the goal from this cell?"

The implementation stores unreachable cells as `-1`.

---

# 🧮 Why BFS Gives the Shortest Path

Every movement has the same cost:

```text
UP    = 1 step
DOWN  = 1 step
LEFT  = 1 step
RIGHT = 1 step
```

Therefore BFS explores:

```text
0 steps away
      ↓
1 step away
      ↓
2 steps away
      ↓
3 steps away
      ↓
...
```

It cannot reach a cell in 10 steps before checking all reachable cells that are 9 steps away.

Therefore the first distance assigned to a reachable cell is its shortest distance from the goal.

---

# 🛣️ Reconstructing the Path

Once the BFS distance map is available, the solver starts at `START`.

At every step it looks for a neighbouring cell whose distance is:

```text
current_distance - 1
```

Example:

```text
Current cell = distance 7

Neighbours:

UP    → 8
DOWN  → 6  ← choose this
LEFT  → wall
RIGHT → 7
```

So the solver chooses `DOWN`.

It continues:

```text
7 → 6 → 5 → 4 → 3 → 2 → 1 → 0
```

until it reaches the goal.

This is how the generator creates the stored shortest-path actions and path mask.

---

# 📦 Dataset Format

For every generated maze, the dataset stores information such as:

```text
grids
starts
goals
solutions
actions
lengths
seeds
```

### `grids`

The actual maze:

```text
1 = wall
0 = path
```

### `starts`

The starting coordinate:

```text
(row, column)
```

### `goals`

The goal coordinate:

```text
(row, column)
```

### `solutions`

A mask showing the cells belonging to the shortest path.

### `actions`

The shortest-path actions:

```text
0 = up
1 = down
2 = left
3 = right
```

Padding is represented by:

```text
-1
```

### `lengths`

The shortest-path length measured in number of moves.

### `seeds`

The random seed used to generate each maze.

The generator saves all of these arrays into compressed `.npz` datasets.

---

# 📚 Train / Validation / Test Split

The dataset is divided into three groups:

```text
                 All Mazes
                     │
          ┌──────────┼──────────┐
          │          │          │
          ▼          ▼          ▼
       Training   Validation   Test
         80%         10%        10%
```

### Training set

Used by PPO to learn.

### Validation set

Used during training to determine whether the model is improving.

### Test set

Used at the end to measure performance on unseen mazes.

This separation is important because evaluating on the same mazes used for training would not tell us whether the agent generalizes to new mazes. The generator implements the 80/10/10 split directly.

---

# 🤖 Part 3 — Reinforcement Learning

Now we get to the main part of the project.

The maze is converted into a **reinforcement-learning environment**.

The environment is implemented using **Gymnasium**.

```text
Maze
 │
 ▼
MazeEnv
 │
 ├── Observation
 ├── Action
 ├── Reward
 └── Done
```

---

# 🎮 What is an RL Environment?

In reinforcement learning, an agent interacts with an environment.

The loop looks like this:

```text
       ┌──────────────┐
       │    Agent     │
       └──────┬───────┘
              │
            Action
              │
              ▼
       ┌──────────────┐
       │ Environment  │
       └──────┬───────┘
              │
       Observation
        + Reward
              │
              └──────────► Agent
```

The agent repeatedly:

1. Looks at the maze.
2. Chooses an action.
3. Moves.
4. Receives a reward.
5. Observes the new state.
6. Chooses another action.

---

# 👀 Observation

The environment gives the neural network **four channels**.

```text
Channel 0 → Maze
Channel 1 → Agent position
Channel 2 → Goal position
Channel 3 → Visited cells
```

So the observation has the form:

```text
(4, height, width)
```

The four channels allow the CNN to separately understand:

- where the walls are,
- where the agent is,
- where the goal is,
- where the agent has already been.

This observation representation is implemented directly in `MazeEnv._obs()`.

---

# 🎮 Action Space

The agent has four possible actions:

| Action | Movement |
|---:|---|
| `0` | Up |
| `1` | Down |
| `2` | Left |
| `3` | Right |

This is a **discrete action space of size 4**.

---

# 🏆 Reward Function

The agent needs a way to understand whether an action was useful.

That is what the reward function does.

The project uses four main reward components:

```text
Goal reward        = +10.0
Step penalty       = -0.01
Wall penalty       = -0.05
Distance shaping   = 0.05 × distance improvement
```

These values are configurable in `maze_solver.py`.

---

## 1. Reaching the Goal

When the agent reaches the goal:

```text
+10.0
```

This strongly encourages successful completion.

---

## 2. Step Penalty

Every action has a small cost:

```text
-0.01
```

Why?

Without a step penalty, the agent could potentially wander around the maze before reaching the goal without being strongly discouraged.

The penalty encourages shorter solutions.

---

## 3. Wall Penalty

If the agent tries to move into a wall or outside the maze:

```text
-0.05
```

This teaches the agent that invalid movements are undesirable.

---

## 4. Distance-Based Reward Shaping

The environment also knows the BFS distance to the goal.

Suppose:

```text
Current distance = 20
New distance     = 19
```

The agent moved closer to the goal.

Therefore:

```text
distance improvement = 20 - 19
                     = 1
```

and the shaping reward becomes:

```text
0.05 × 1 = +0.05
```

If the agent moves farther away:

```text
20 → 21
```

the shaping contribution becomes negative.

This gives the agent a useful learning signal even before it reaches the goal.

---

# 🧠 Part 4 — CNN

The maze is an image-like grid.

A normal fully-connected neural network would have difficulty efficiently understanding spatial relationships such as:

```text
Wall next to agent
        ↓
Open path two cells away
        ↓
Goal around the corner
```

A **Convolutional Neural Network (CNN)** is better suited to this type of spatial data.

---

# 🔬 What Does a CNN Do?

A CNN uses small filters to scan across an image/grid.

For example:

```text
┌───┬───┬───┐
│   │ █ │   │
├───┼───┼───┤
│ A │   │ G │
├───┼───┼───┤
│   │ █ │   │
└───┴───┴───┘
```

The CNN can learn patterns such as:

- nearby walls,
- corridors,
- corners,
- intersections,
- agent-to-goal relationships,
- local spatial structures.

---

# 🏗️ Custom CNN Architecture

The project defines a custom feature extractor called:

```python
MazeCNN
```

It uses multiple convolutional layers followed by a fully connected layer.

Simplified:

```text
4-channel observation
        │
        ▼
Conv2D: 32 channels
        │
       ReLU
        │
        ▼
Conv2D: 64 channels
        │
       ReLU
        │
        ▼
Conv2D: 64 channels
        │
       ReLU
        │
        ▼
Conv2D: 64 channels
        │
       ReLU
        │
        ▼
Dilated convolutions
        │
        ▼
Flatten
        │
        ▼
Linear Layer
        │
        ▼
256 features
```

The CNN also uses progressively larger dilation rates in later layers, allowing it to capture information over a wider spatial area.

---

# 🔎 Why Dilated Convolutions?

A normal convolution mainly sees a local neighbourhood.

A dilated convolution inserts gaps between kernel elements.

Conceptually:

```text
Normal:

X X X
X X X
X X X


Dilated:

X . X . X
. . . . .
X . X . X
. . . . .
X . X . X
```

This gives the network a larger **receptive field** without requiring extremely large kernels.

That is useful for maze navigation because a good decision may depend on structures that are farther away than the immediately adjacent cells.

The implementation uses dilation rates `1`, `2`, and `4` in the later convolutional layers.

---

# 🤝 Part 5 — PPO

The actual reinforcement-learning algorithm is:

# Proximal Policy Optimization (PPO)

PPO is a policy-gradient reinforcement-learning algorithm.

You do not need to understand the mathematics to understand the basic idea.

The agent has a policy:

```text
Maze observation
       │
       ▼
     CNN
       │
       ▼
     PPO
       │
       ▼
Probability of each action

Up    → 10%
Down  → 60%
Left  → 5%
Right → 25%
```

The agent then selects an action based on these probabilities.

Over many episodes, PPO updates the policy so that actions leading to better long-term rewards become more likely.

---

# 🔄 How PPO Learns

Imagine the agent tries:

```text
RIGHT → RIGHT → DOWN → WALL → LEFT → ...
```

The episode produces rewards.

PPO uses the collected experience to determine:

```text
Which actions were useful?
Which actions were bad?
Which actions should become more likely?
```

The policy is then updated.

After many iterations:

```text
Random behaviour
       ↓
Some useful movement
       ↓
More goal-directed movement
       ↓
Consistent navigation
       ↓
Learned policy
```

---

# 🛡️ Why "Proximal" Policy Optimization?

The main problem with policy-gradient algorithms is that a large update can destroy a previously useful policy.

PPO limits how aggressively the policy changes.

In simplified terms:

```text
Old policy
    │
    │ small controlled update
    ▼
New policy
```

instead of:

```text
Old policy
    │
    │ huge update
    ▼
Completely different policy
```

This helps make training more stable.

The project uses Stable-Baselines3's PPO implementation.

---

# ⚙️ PPO Configuration

The current solver uses:

```text
Algorithm       : PPO
Policy          : CnnPolicy
Learning rate   : 3e-4
Rollout steps   : 256
Batch size      : 256
PPO epochs      : 4
Gamma           : 0.995
GAE lambda      : 0.95
Entropy coeff.  : 0.01
Features        : 256
Parallel envs   : 8
```

These settings are defined in the training code.

---

# 🔀 Part 6 — Reverse Curriculum Learning

Training the agent directly from the entrance of a large maze can be difficult.

Imagine asking a beginner to solve:

```text
100-step maze
```

on their first attempt.

That is a difficult learning problem.

Instead, this project uses a **reverse curriculum**.

---

# 🎓 What is Curriculum Learning?

Curriculum learning means:

> Start with easier problems and gradually increase the difficulty.

For this project:

```text
Easy
 ↓
5%
 ↓
10%
 ↓
20%
 ↓
30%
 ↓
40%
 ↓
50%
 ↓
60%
 ↓
70%
 ↓
80%
 ↓
90%
 ↓
100%
Hard
```

These are the configured curriculum levels.

---

# 🔄 Why is it Called Reverse Curriculum?

The real maze entrance might be far away from the goal.

Instead of always starting here:

```text
START ───────────────────────────────► GOAL
```

the agent can initially start closer to the goal:

```text
START ───────► AGENT ────────────────► GOAL
                ↑
             easier task
```

Once the agent becomes good at that task, the starting point is moved farther away.

Eventually:

```text
REAL START ──────────────────────────► GOAL
```

This is the "reverse" curriculum.

The environment calculates the shortest-path distance to the goal using BFS and moves the temporary starting position along that shortest path according to the selected difficulty.

---

# 🧪 Example of Curriculum

Suppose the shortest path contains:

```text
100 steps
```

At different curriculum levels, the agent may effectively start at different distances from the goal.

Conceptually:

```text
Difficulty = 0.1

START ──────────────────────── AGENT ───── GOAL
                                10 steps


Difficulty = 0.5

START ───────────── AGENT ───────────────── GOAL
                  50 steps


Difficulty = 1.0

AGENT = REAL START

START ─────────────────────────────────── GOAL
                  100 steps
```

The exact starting position is selected from the BFS distance map.

---

# 🔁 Full Training Process

The training process is:

```text
Load training mazes
        │
        ▼
Create 8 environments
        │
        ▼
Start curriculum at 5%
        │
        ▼
Train PPO
        │
        ▼
Evaluate on validation mazes
        │
        ▼
Solved ≥ 90% ?
     /       \
   YES        NO
    │          │
    ▼          ▼
Next level   Continue
    │
    └───────────┐
                ▼
          Next difficulty
                │
                ▼
              100%
                │
                ▼
          Final test set
```

The implementation trains in `10,000`-step chunks and can use up to `10` chunks at each difficulty level. It moves to the next difficulty when the validation solve rate reaches the configured `90%` threshold; otherwise it moves on after the configured maximum chunks.

---

# 📊 Evaluation

The model is evaluated using two main measurements.

## 1. Solve Rate

The solve rate answers:

> How many mazes did the agent successfully solve?

For example:

```text
40 validation mazes

32 solved

Solve rate = 32 / 40
          = 80%
```

The current configuration evaluates up to:

```text
40 validation mazes
```

during training.

---

# 2. Steps vs Shortest Path

Solving the maze is not the whole story.

Suppose BFS says:

```text
Shortest path = 100 steps
```

but the PPO agent takes:

```text
140 steps
```

Then:

```text
140 / 100 = 1.40×
```

So the agent used `1.40×` the shortest-path length.

The evaluation code calculates:

```text
agent steps / optimal BFS steps
```

for successful episodes.

---

# 🧪 Final Test

After curriculum training finishes, the model is evaluated on the separate test dataset.

The current configuration uses up to:

```text
200 test mazes
```

and forces evaluation from the real entrance:

```text
difficulty = 1.0
```

The final output reports:

```text
percentage of mazes solved
+
steps / shortest-path ratio
```

This is the most important evaluation because these mazes were not used for PPO training.

---

# 👁️ Live Visualization

The project includes a Matplotlib-based live viewer.

The visualization shows:

```text
Blue square   → Start
Green marker  → Goal
Red circle    → Current agent
Orange line   → Agent's trail
Black/white   → Maze
```

During training, the viewer can display the current training environment and validation progress.

---

# 🛠️ Installation

Clone the repository:

```bash
git clone https://github.com/Aadit2308/Maze_solver.git
cd Maze_solver
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it.

### Windows

```powershell
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install stable-baselines3 torch gymnasium numpy matplotlib
```

These are the core packages imported by the solver.

---

# ▶️ Running the Project

## Step 1 — Generate the dataset

Run:

```bash
python maze_genrator.py
```

This generates:

```text
mazes_train.npz
mazes_val.npz
mazes_test.npz
```

It also generates sample maze images according to the configured settings.

---

# 🔍 Step 2 — Inspect the Dataset

Before training, it is useful to verify that the maze is being interpreted correctly.

Run:

```bash
python maze_solver.py --inspect
```

The program displays different curriculum starting positions.

You should see:

```text
Black → Walls
Blue  → Start
Green → Goal
```

If the maze appears inverted, check:

```python
ONE_IS_WALL = True
```

The inspection mode exists specifically to catch this type of dataset-orientation problem.

---

# 🏋️ Step 3 — Train the Agent

Run:

```bash
python maze_solver.py
```

The training script will:

1. Load the training dataset.
2. Load the validation dataset.
3. Create 8 parallel environments.
4. Create the PPO model.
5. Start the reverse curriculum.
6. Train for chunks of steps.
7. Evaluate the model.
8. Display validation performance.
9. Save the model.
10. Move through increasingly difficult curriculum levels.
11. Run the final test.

The trained model is saved using:

```text
maze_ppo_reverse.zip
```

The exact training and saving flow is implemented in the main section of `maze_solver.py`.

---

# 👀 Step 4 — Watch the Trained Model

After training:

```bash
python maze_solver.py --watch
```

The saved PPO model is loaded and run on validation mazes from the real entrance.

---

# 🧪 Understanding `maze_solver.py`

The solver can be understood as five major components:

```text
maze_solver.py
│
├── 1. Dataset loader
│
├── 2. BFS distance calculation
│
├── 3. MazeEnv
│      ├── Observation
│      ├── Action
│      ├── Reward
│      └── Episode termination
│
├── 4. MazeCNN
│
├── 5. PPO training + evaluation
```

This separation makes the project easier to understand and modify.

---

# 🧩 MazeEnv

`MazeEnv` is the interface between the maze and PPO.

Its job is to answer four questions:

### What does the agent see?

```text
4-channel maze observation
```

### What can the agent do?

```text
Up / Down / Left / Right
```

### How good was that action?

```text
Reward
```

### Is the episode finished?

```text
Goal reached
OR
Maximum number of steps exceeded
```

The environment uses Gymnasium's standard `reset()` and `step()` interface.

---

# ⏱️ Episode Termination

An episode ends when:

### Case 1 — Goal reached

```text
agent == goal
```

The episode is successful.

### Case 2 — Too many steps

The maximum number of steps is calculated as:

```text
max_steps = 4 × shortest_path_length + 50
```

If the agent does not reach the goal before this limit, the episode is truncated.

---

# 🧮 Why Keep BFS if PPO Is the Solver?

This is an important design question.

It may look strange to use BFS in a project whose purpose is to train PPO.

BFS is useful for **training support and evaluation**.

It provides:

### 1. Solvability checking

If BFS cannot reach the start:

```text
Maze is not solvable
```

the dataset generator rejects the maze.

### 2. Optimal path length

The project knows:

```text
What is the shortest possible path?
```

This gives us a baseline for evaluating PPO.

### 3. Reward shaping

The distance-to-goal information provides a learning signal.

### 4. Curriculum learning

The distance map allows the environment to place the agent at different distances from the goal.

So BFS is **not replacing PPO**.

It is providing useful information around the RL problem.

---

# 🆚 BFS vs PPO

| Property | BFS | PPO |
|---|---|---|
| Type | Search algorithm | Reinforcement learning |
| Needs full maze | Yes | Yes |
| Finds shortest path | Yes | Not guaranteed |
| Learns from experience | No | Yes |
| Neural network | No | Yes |
| Generalizes through learned policy | No | Yes |
| Used in this project | Dataset/reference | Main learning agent |

The project therefore provides an interesting comparison between:

```text
Explicit planning
      vs
Learned navigation
```

---

# 🧠 Important Concept: PPO Does NOT Get the BFS Answer

During dataset generation, BFS calculates:

```text
shortest path
```

But PPO's action-selection policy is learned from environment interaction.

The agent receives the observation:

```text
Maze
Agent
Goal
Visited cells
```

and chooses:

```text
Up / Down / Left / Right
```

The BFS distance is used by the environment for shaping, curriculum placement, and evaluation rather than simply handing the agent the correct action sequence.

---

# 📁 File-by-File Explanation

## `maze_genrator.py`

Responsible for:

```text
DFS maze generation
       ↓
BFS shortest path
       ↓
Dataset creation
       ↓
Train / validation / test split
```

---

## `maze_solver.py`

Responsible for:

```text
Load dataset
      ↓
Create Gymnasium environment
      ↓
CNN
      ↓
PPO
      ↓
Reverse curriculum
      ↓
Validation
      ↓
Final testing
```

---

## `maze_solver_no_live.py`

A version of the solver without the live visualization workflow.

This is useful when you want to train without continuously rendering the maze window.

---

## `mazes_train.npz`

Training examples used by PPO.

---

## `mazes_val.npz`

Mazes used to measure progress during training.

---

## `mazes_test.npz`

Unseen mazes used for final evaluation.

---

# 🔬 Complete Technical Pipeline

The complete project can be summarized as:

```text
                         ┌─────────────────┐
                         │  DFS Generator  │
                         └────────┬────────┘
                                  │
                                  ▼
                         Generated Maze
                                  │
                                  ▼
                         ┌─────────────────┐
                         │       BFS       │
                         └────────┬────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                    ▼                           ▼
             Shortest Path              Distance Map
                    │                           │
                    ▼                           ▼
              Dataset                 Curriculum + Reward
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
      Train        Val         Test
        │
        ▼
     MazeEnv
        │
        ▼
  4-channel observation
        │
        ▼
     MazeCNN
        │
        ▼
   256 features
        │
        ▼
      PPO
        │
        ▼
   Action 0–3
        │
        ▼
     MazeEnv
        │
        └──────────────► Reward
                            │
                            ▼
                       PPO Update
                            │
                            ▼
                    Better Policy
                            │
                            ▼
                    Validation Test
                            │
                            ▼
                     Final Test Set
```

---

# ⚙️ Important Configuration

The main configuration is located near the top of `maze_solver.py`.

```python
TRAIN_NPZ = "mazes_train.npz"
VAL_NPZ = "mazes_val.npz"
TEST_NPZ = "mazes_test.npz"

MAX_TRAIN_MAZES = 40000

DIFFICULTIES = [
    0.05, 0.1, 0.2, 0.3, 0.4,
    0.5, 0.6, 0.7, 0.8, 0.9, 1.0
]

FULL_START_PROB = 0.2

N_ENVS = 8

CHUNK_STEPS = 10_000

MAX_CHUNKS_PER_LEVEL = 10

PASS_RATE = 0.90

EVAL_MAZES = 40

FINAL_TEST_MAZES = 200

MODEL_PATH = "maze_ppo_reverse"
```

These settings control the curriculum, training size, validation threshold, parallel environments, and final testing.

---

# 🎛️ Main Parameters You Can Experiment With

## Maze complexity

In `maze_genrator.py`:

```python
WIDTH
HEIGHT
CORRIDOR
WALL_SIZE
LOOP_PROB
ROOMS
ROOM_MAX
```

For example:

```python
LOOP_PROB = 0.0
```

produces a more tree-like maze.

Increasing:

```python
LOOP_PROB
```

creates more alternative routes.

---

## RL difficulty

In `maze_solver.py`:

```python
DIFFICULTIES = [
    0.05,
    0.10,
    0.20,
    ...
    1.00
]
```

You can change these values to experiment with different curriculum schedules.

---

## Reward function

You can experiment with:

```python
GOAL_REWARD
STEP_PENALTY
WALL_PENALTY
SHAPING_SCALE
```

Changing these values changes what behaviour the agent is encouraged to learn.

---

# ⚠️ Limitations

This project is useful for studying RL-based navigation, but there are important limitations.

### 1. PPO does not guarantee the shortest path

BFS explicitly finds a shortest path.

PPO only learns a policy that maximizes expected reward.

Therefore:

```text
PPO solution ≠ guaranteed optimal solution
```

The project measures the difference using:

```text
agent_steps / shortest_path_steps
```

---

### 2. BFS information is part of the training environment

The environment uses BFS distance information for:

- curriculum positioning,
- reward shaping,
- evaluation.

Therefore this is not a completely "BFS-free" RL experiment.

If the goal is to study pure end-to-end RL, these components would need to be removed or replaced.

---

### 3. Generalization depends on the maze distribution

An agent trained on one family of mazes may not perform equally well on completely different maze structures.

For example:

```text
Training:
50 × 50 DFS mazes

Testing:
100 × 100 rooms + loops + irregular geometry
```

would be a substantially harder generalization problem.

---

### 4. Training is computationally expensive

The project uses:

```text
8 environments
CNN
PPO
large maze grids
multiple curriculum levels
```

The current configuration also specifies:

```python
device="cuda"
```

so a CUDA-capable GPU is expected for the configured training setup.

---

# 🚀 Possible Improvements

Some useful future experiments would be:

- [ ] Compare PPO against BFS
- [ ] Compare PPO against A*
- [ ] Compare PPO against Dijkstra
- [ ] Add DQN
- [ ] Add A2C
- [ ] Remove BFS-based reward shaping
- [ ] Compare curriculum vs no curriculum
- [ ] Test different maze sizes
- [ ] Test unseen maze-generation algorithms
- [ ] Add TensorBoard logging
- [ ] Add training reward graphs
- [ ] Add automated benchmarks
- [ ] Save checkpoints at each curriculum level
- [ ] Add configurable CPU/GPU selection
- [ ] Add unit tests for maze generation
- [ ] Add unit tests for BFS
- [ ] Add automated dataset validation
- [ ] Rename `maze_genrator.py` to `maze_generator.py`

---

# 📚 Beginner's Guide to the Project

If you are new to AI or reinforcement learning, understand the project in this order.

### Step 1 — Understand a grid

A maze is just a matrix:

```text
1 1 1 1 1
1 0 0 0 1
1 0 1 0 1
1 0 0 0 1
1 1 1 1 1
```

---

### Step 2 — Understand DFS

DFS creates the maze.

```text
DFS
 ↓
Carves passages
 ↓
Maze
```

---

### Step 3 — Understand BFS

BFS finds the shortest route.

```text
Maze
 ↓
BFS
 ↓
Shortest path
```

---

### Step 4 — Understand RL

The agent interacts with the maze.

```text
Observation
     ↓
  Action
     ↓
  Reward
     ↓
New observation
```

---

### Step 5 — Understand CNN

The CNN converts the maze image-like representation into useful numerical features.

```text
Maze
 ↓
CNN
 ↓
Features
```

---

### Step 6 — Understand PPO

PPO uses those features to learn which actions are useful.

```text
Features
   ↓
 PPO
   ↓
Action probabilities
   ↓
Movement
```

---

### Step 7 — Understand Curriculum Learning

Start easy:

```text
5%
```

Then progressively increase:

```text
10%
20%
30%
...
100%
```

---

### Step 8 — Evaluate

Finally ask:

```text
Did it solve the maze?
        +
How close was it to the shortest path?
```

---

# 📌 One-Sentence Summary

> **This project generates mazes with DFS, computes shortest-path references using BFS, and trains a CNN-based PPO agent with reverse curriculum learning to navigate those mazes from the entrance to the goal.**

---

# 👤 Author

**Aadit2308**

GitHub:

https://github.com/Aadit2308

Repository:

https://github.com/Aadit2308/Maze_solver

---

# 📄 License

See the [`LICENSE`](./LICENSE) file for the license applicable to this project.

---

# ⭐ Final Architecture

```text
             ┌─────────────────────┐
             │    DFS Generator   │
             └──────────┬──────────┘
                        │
                        ▼
                  Maze Dataset
                        │
                        ▼
             ┌─────────────────────┐
             │        BFS          │
             │ Shortest Path +     │
             │ Distance Map        │
             └──────────┬──────────┘
                        │
                        ▼
             ┌─────────────────────┐
             │     Gymnasium       │
             │      MazeEnv        │
             └──────────┬──────────┘
                        │
                 4-channel input
                        │
                        ▼
             ┌─────────────────────┐
             │       MazeCNN       │
             │ Spatial Features    │
             └──────────┬──────────┘
                        │
                  256 features
                        │
                        ▼
             ┌─────────────────────┐
             │        PPO          │
             │ Reinforcement       │
             │ Learning Agent      │
             └──────────┬──────────┘
                        │
                  Up/Down/Left/Right
                        │
                        ▼
             ┌─────────────────────┐
             │       Maze          │
             └──────────┬──────────┘
                        │
                 Reward + State
                        │
                        └──────────► PPO
                                      │
                                      ▼
                              Learned Policy
                                      │
                                      ▼
                              Unseen Test Mazes
```

**The key learning objective is simple:** the agent sees a maze, chooses movements, receives feedback, and gradually learns a policy that takes it from the entrance to the goal. The classical algorithms provide the infrastructure and reference measurements; PPO is responsible for learning the navigation behaviour.
