# Reinforcement Learning: Tabular Methods to 2048

A working repository from a self-study project on reinforcement learning. It moves from tabular value-based methods on toy environments, through policy gradients, to studying a state-of-the-art temporal-difference approach for the game **2048**.

This is a **study repository**, not a library. The code is a mix of exercises I worked through and reference implementations I read, ran, and annotated. Third-party sources are credited in each file and summarised at the bottom.

---

## Why 2048

2048 is a compact but genuinely hard RL environment:

- **Stochastic transitions.** After every move, a new tile (2 with probability ~0.9, 4 with ~0.1) spawns in a random empty cell. The same action in the same state does not lead to the same next state, so the agent must learn *expected* value, not a fixed plan.
- **Enormous state space.** A 4×4 grid with ~16 possible tile values per cell gives on the order of 10^18 reachable configurations. Tabular Q-learning is hopeless; function approximation is mandatory.
- **Delayed credit assignment.** A move that quietly breaks a monotonic row may cost nothing for another twenty moves.
- **Illegal actions.** In many states, one or more directions do nothing, and the agent has to learn to avoid them.

The path through this repo is essentially the path to understanding why each of those properties rules out the simpler methods.

---

## Roadmap

Work through the files in this order — each one exists because the previous one wasn't enough.

### 1. Python warm-ups — `gfg1.py`, `gfg2.py`, `gfg3.py`

Tuple filtering, deduplicating nested tuples, flattening to unique elements. No RL content; kept because tuple/set manipulation shows up constantly once you start indexing states.

### 2. Tabular Q-learning — `gfg6.py`

The Q-learning update in its barest form on a 16-state ring world:

```
Q(s,a) ← Q(s,a) + α · [ r + γ · max_a' Q(s',a') − Q(s,a) ]
```

α = 0.8, γ = 0.95, ε = 0.2, 1000 epochs. Ends with a heatmap of `max_a Q(s,a)` reshaped to a 4×4 grid. Useful as the smallest possible thing that actually learns.

### 3. Q-learning on a real grid world — `Notebook_for_Topic_08_Video_Q_Learning_A_Complete_Example_in_Python.ipynb`

An 11×11 warehouse: the robot learns the shortest path from any aisle square to the packaging area, with −100 for shelves, −1 per step, +100 at the goal. This is the first example where the reward structure itself does the teaching — the step penalty is what produces *short* paths rather than merely successful ones.

### 4. SARSA — `gfg7.py`

The same GridWorld idea, but on-policy. The update bootstraps from the action actually taken next, not the greedy one:

```
Q(s,a) ← Q(s,a) + α · [ r + γ · Q(s',a') − Q(s,a) ]
```

Put side by side with `gfg6.py`, this is the cleanest way to see the on-policy / off-policy distinction: SARSA learns the value of the ε-greedy policy it's actually following, so it behaves more conservatively near cliffs.

### 5. Monte Carlo policy evaluation — `gfg5.py`

No bootstrapping at all — run full episodes, then average the observed returns per state. Shows why TD methods are usually preferred: MC needs episodes to terminate and has much higher variance, but it's unbiased and needs no model.

### 6. FrozenLake — `frozen_lake.ipynb`

Gym's FrozenLake-v1 with a Q-table and hyperparameters set up (α = 0.1, γ = 0.99, ε annealed 1 → 0.01, 10,000 episodes, 100 steps max).

> **Status: incomplete.** The training loop is not yet written. The notebook currently only runs 3 rendered episodes against an all-zero Q-table, so the agent moves arbitrarily. Adding the training loop before the play loop is the open TODO here.

### 7. Policy gradients — `gfg4.py`

REINFORCE on CartPole-v1 with TensorFlow/Keras: a two-layer network outputs a softmax over actions, and the loss is `−E[log π(a|s) · G_t]` with returns normalised per episode. The first departure from value-based methods — the policy is learned directly.

> **Known issue:** the training loop is duplicated in the file, so it trains for 2000 episodes rather than the 1000 the config implies. Worth deleting the second copy.

### 8. TD learning for 2048 — `2048.py`

The centrepiece. **This file is a copy of the TDL2048-Demo reference implementation** (see Attribution). I read it line by line and ran it rather than writing it; the notes below are my understanding of how it works.

**Bitboard representation.** The whole 4×4 board is packed into a single 64-bit integer — 4 bits per cell, storing the *exponent* rather than the tile value (so `3` means the tile 8). Row-level slide/merge results are precomputed into a lookup table (`board.lookup.init()`), which makes a move a handful of table reads instead of a loop over cells.

**N-tuple network.** Instead of a neural net, the value function is a sum of lookup tables over overlapping tile patterns. The default configuration is the 4×6-tuple network:

```python
tdl.add_feature(pattern([0, 1, 2, 3, 4, 5]))
tdl.add_feature(pattern([4, 5, 6, 7, 8, 9]))
tdl.add_feature(pattern([0, 1, 2, 4, 5, 6]))
tdl.add_feature(pattern([4, 5, 6, 8, 9, 10]))
```

Each 6-cell pattern indexes a table of 16^6 = 16,777,216 weights, and each is evaluated over all 8 board symmetries (4 rotations × reflection) with the weights shared. `V(s)` is the sum across all features; a TD update distributes the correction equally among them.

**Action selection.** For each of the four directions, compute the afterstate (post-slide, pre-spawn), and pick the move maximising `reward + V(afterstate)`. Evaluating afterstates rather than states is what keeps the random tile spawn out of the value estimate.

**Learning — TD(0) on afterstates, backwards.** After an episode, `learn_from_episode` discards the terminal record and walks the trajectory in reverse:

```
error  = target − V(s'_t)
target = r_t + V_updated(s'_t)
```

Going backwards means each update already sees the corrected value of the following state, so credit propagates through the whole episode in a single pass. α = 0.1 by default.

**Statistics.** Every 1000 games it prints average and maximum score plus the max-tile distribution, in the format:

```
100000  avg = 68663.7   max = 177508
        2048    91.2%   (22.5%)
```

meaning 91.2% of those games reached a 2048 tile at some point, and 22.5% *ended* with 2048 as their largest tile.

**Persistence.** Weights are loaded from and saved to `2048.bin` via `learning.load()` / `learning.save()`, so training resumes across runs.

---

## Running it

### Requirements

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install numpy matplotlib gym tensorflow jupyter
```

`2048.py` itself is pure standard library — numpy and TensorFlow are only needed for the earlier exercises.

### The 2048 agent

```bash
python 2048.py
```

**Practical notes before you start this:**

- **Memory.** Four features × 16^6 weights × 4 bytes ≈ **256 MB of weight tables**, allocated up front. The script prints the size of each feature as it's added.
- **Runtime.** This is a Python port of a C++ demo, and 100,000 self-play episodes is *slow* — expect hours, not minutes. Lower `total` in `__main__` for a first run.
- **`2048.bin` must not be committed.** It's on the order of 256 MB, well past GitHub's 100 MB per-file hard limit. It's in `.gitignore`.

### The exercises

```bash
python gfg6.py          # tabular Q-learning + heatmap
python gfg7.py          # SARSA on GridWorld
python gfg5.py          # Monte Carlo policy evaluation
python gfg4.py          # REINFORCE on CartPole
jupyter notebook        # for the two .ipynb files
```

---

## Results

Fill this in after your own training run — the table below is a placeholder.

| Configuration | Episodes | Avg score | Max score | 2048 win rate |
|---|---|---|---|---|
| 4×6-tuple, α = 0.1 | — | — | — | — |

The reference figures quoted in TDL2048-Demo for the 4×6-tuple network after 100k games are roughly 68k average score and a ~91% 2048 win rate; landing near those confirms a run is behaving correctly.

---

## Open TODOs

- [ ] Write the missing training loop in `frozen_lake.ipynb`
- [ ] Remove the duplicated training loop in `gfg4.py`
- [ ] Log 2048 training curves to a file and plot average score against episodes
- [ ] Try the 8×6-tuple network and compare against 4×6
- [ ] Implement an expectimax baseline to measure how much of the gap to strong play TD closes

---

## Attribution

This repository contains third-party code, kept for study purposes:

- **`2048.py`** — TDL2048-Demo, from the Computer Games and Intelligence (CGI) Lab, NYCU Taiwan and the Reinforcement Learning and Games (RLG) Lab, Academia Sinica: https://github.com/moporgic/TDL2048-Demo. The original authorship header is retained in the file. Not my implementation.
- **`Notebook_for_Topic_08_...ipynb`** — Q-learning teaching notebook by Dr. Daniel Soper.
- **`gfg1.py`–`gfg7.py`** — adapted from GeeksforGeeks tutorials.

### Papers behind the 2048 approach

- M. Szubert and W. Jaśkowski, *Temporal difference learning of N-tuple networks for the game 2048*, CIG 2014.
- I-C. Wu, K.-H. Yeh, C.-C. Liang, C.-C. Chang, H. Chiang, *Multi-stage temporal difference learning for 2048*, TAAI 2014.
- K. Matsuzaki, *Systematic selection of N-tuple networks with consideration of interinfluence for game 2048*, TAAI 2016.

2048 was created by Gabriele Cirulli (2014).

---

## License

The exercise and notebook code here belongs to its original authors under their respective terms. My own notes and additions are released under the MIT License.
