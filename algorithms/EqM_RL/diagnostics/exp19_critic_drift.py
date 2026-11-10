"""Experiment 19: where did the critic of a seed change, and where does the policy go with it? Markdown to stdout.

Reads the checkpoints of Aware/train_ieqm.py (checkpoint_<step>.pt, every 100k) of the Exp19_IEQM runs on
humanoidmaze-medium task 1. No training.

  Part 1  V on a fixed sample of dataset states, averaged per maze cell (the maze is a grid of 4 x 4 cells; a state's
          cell is the cell of its x-y position). For every seed and checkpoint: "walk uphill" on these cell means from
          the start cell, always to the neighbouring free cell with the highest V, until no neighbour is higher. A
          critic that leads to the goal ends this walk in the goal cell. A walk that ends elsewhere has found a cell
          whose V is higher than all its neighbours' (a false peak).
  Part 2  The change of the cell means between two checkpoints, as a map (for the pairs in --maps).
  Part 3  Rollouts (for the seed:step pairs in --rollouts): the run's frozen EqM-BC actor proposes 32 actions at K = 20
          and the checkpoint's critic picks one, as in the run's own 'frozen' eval (same start states). Reported: success,
          and in which cells the episodes that failed spent their last 500 steps.

    python diagnostics/exp19_critic_drift.py > diagnostics/out/exp19_critic_drift/summary.md
"""

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

EQM_RL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EQM_RL / "Aware"))
import ogbench_data  # noqa: E402
import train_ieqm as T  # noqa: E402
from actor_models import Actor  # noqa: E402
from utils.networks import ConditionalMLP  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--env_name", default="humanoidmaze-medium-navigate-singletask-v0")
ap.add_argument("--root", default="/workspace/Trained_Models/checkpoints")
ap.add_argument("--prefix", default="Exp19_IEQM")
ap.add_argument("--tag", default="", help="the part of the run name between the env name and _seed, e.g. _expectile0.7")
ap.add_argument("--seeds", default="1,2,3")
ap.add_argument("--steps", default=",".join(str(s) for s in range(100000, 1000001, 100000)))
ap.add_argument("--sample", type=int, default=300000, help="dataset states used for the cell means")
ap.add_argument("--maps", default="1:700000:800000,1:700000:900000,2:700000:900000,3:900000:1000000",
                help="seed:step_from:step_to, the maps of part 2")
ap.add_argument("--rollouts", default="1:700000,1:800000,1:900000,3:900000,3:1000000,2:1000000",
                help="seed:step, the rollouts of part 3 ('' = none)")
ap.add_argument("--episodes", type=int, default=50)
ap.add_argument("--test_seed", type=int, default=69)
ap.add_argument("--device", default="cuda")
args = ap.parse_args()
dev = args.device
seeds = [int(s) for s in args.seeds.split(",")]
steps = [int(s) for s in args.steps.split(",")]

env, data, mean, std, _, _ = ogbench_data.load_ogbench(args.env_name)
u = env.unwrapped
maze, (H, W) = u.maze_map, u.maze_map.shape
state_dim, action_dim = data["observations"].shape[1], data["actions"].shape[1]
low, high = float(env.action_space.low[0]), float(env.action_space.high[0])
mean_t = torch.tensor(mean, dtype=torch.float32, device=dev)
std_t = torch.tensor(std, dtype=torch.float32, device=dev)
start_cell = tuple(int(x) for x in u.xy_to_ij(env.reset(seed=0)[0][:2]))
goal_cell = tuple(int(x) for x in u.xy_to_ij(u.cur_goal_xy))


def cells_of(xy: np.ndarray) -> np.ndarray:
    return np.array([u.xy_to_ij(p) for p in xy], dtype=int)


def run_dir(seed: int) -> str:
    d = glob.glob(f"{args.root}/{args.prefix}_env_name{args.env_name}{args.tag}_seed{seed}-{args.env_name}-*")
    assert len(d) == 1, f"expected one run for seed {seed}, found {d}"
    return d[0]


def load(seed: int, step: int):
    cfg = yaml.safe_load(open(f"{run_dir(seed)}/config.yaml"))
    ck = torch.load(f"{run_dir(seed)}/checkpoint_{step}.pt", map_location=dev)
    value = T.IFQLMLP(state_dim, cfg["critic_hidden_dim"], cfg["critic_n_hidden"]).to(dev)
    critic = T.IFQLCritic(state_dim, action_dim, cfg["critic_hidden_dim"], cfg["critic_n_hidden"]).to(dev)
    value.load_state_dict(ck["value"])
    critic.load_state_dict(ck["critic"])
    return value.eval().requires_grad_(False), critic.eval().requires_grad_(False), cfg


# ---------------------------------------------------------------- part 1: V per maze cell, and the uphill walk
rng = np.random.default_rng(0)
idx = rng.choice(len(data["observations"]), args.sample, replace=False)
S = torch.tensor(data["observations"][idx], dtype=torch.float32, device=dev)
ij = cells_of(data["observations"][idx, :2])
free = [(i, j) for i in range(H) for j in range(W) if maze[i, j] == 0]
in_cell = {c: (ij[:, 0] == c[0]) & (ij[:, 1] == c[1]) for c in free}


@torch.no_grad()
def cell_means(seed: int, step: int) -> dict:
    v = load(seed, step)[0](S).cpu().numpy()
    return {c: float(v[m].mean()) for c, m in in_cell.items()}


def neighbours(c):
    return [(c[0] + di, c[1] + dj) for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)) if maze[c[0] + di, c[1] + dj] == 0]


def uphill(vc: dict) -> list:
    path = [start_cell]
    while True:
        best = max(neighbours(path[-1]), key=lambda c: vc[c])
        if vc[best] <= vc[path[-1]] or len(path) > len(free):
            return path
        path.append(best)


def grid(vals: dict, fmt="{:7.1f}") -> str:
    rows = []
    for i in range(1, H - 1):
        rows.append(" ".join("   ####" if maze[i, j] == 1 else fmt.format(vals[(i, j)]) for j in range(1, W - 1)))
    return "```\n" + "\n".join(rows) + "\n```"


VC = {(sd, st): cell_means(sd, st) for sd in seeds for st in steps}

print("# Experiment 19: the critic per maze cell, and where the policy goes\n")
print(f"`{args.env_name}`, runs `{args.prefix}{args.tag}`, seeds {seeds}. The maze is a grid of cells of 4 × 4 length units. "
      f"A cell is named (row, column). Start cell {start_cell}, goal cell {goal_cell}. `####` = wall.\n")
layout = {c: (" start" if c == start_cell else "  goal" if c == goal_cell else f" ({c[0]},{c[1]})") for c in free}
print("Cell names:\n\n" + grid(layout, "{:>7}") + "\n")
print(f"V is averaged over the dataset states of each cell ({args.sample:,} sampled states, 3,900 to 28,000 per cell).\n")
ref = VC[(seeds[0], 700000)] if (seeds[0], 700000) in VC else next(iter(VC.values()))
print(f"V per cell, seed {seeds[0]} at 700k (higher = the critic expects the goal sooner):\n\n" + grid(ref) + "\n")

print("## Part 1: walking uphill on V from the start cell\n")
print("From the start cell, always step to the neighbouring cell with the highest V, until no neighbour is higher. "
      "**goal** = the walk ends in the goal cell. Otherwise the cell where it ends is given: its V is higher than "
      "all its neighbours', so the critic pulls the policy there.\n")
print("| step | " + " | ".join(f"seed {sd}" for sd in seeds) + " |")
print("|---|" + "---|" * len(seeds))
for st in steps:
    cellstr = []
    for sd in seeds:
        p = uphill(VC[(sd, st)])
        cellstr.append("goal" if p[-1] == goal_cell else f"**stops at {p[-1]}** after {len(p) - 1} cells")
    print(f"| {st // 1000}k | " + " | ".join(cellstr) + " |")
p_ok = uphill(ref)
print("\nThe walk that reaches the goal goes through: " + " → ".join(f"({i},{j})" for i, j in p_ok) + ".\n")

# ---------------------------------------------------------------- part 2: maps of the change
print("## Part 2: change of V per cell between two checkpoints\n")
for spec in [m for m in args.maps.split(",") if m]:
    sd, a, b = (int(x) for x in spec.split(":"))
    d = {c: VC[(sd, b)][c] - VC[(sd, a)][c] for c in free}
    top = sorted(free, key=lambda c: -abs(d[c]))[:3]
    print(f"Seed {sd}, V at {b // 1000}k minus V at {a // 1000}k. Largest changes: "
          + ", ".join(f"({c[0]},{c[1]}) {d[c]:+.1f}" for c in top) + ".\n\n" + grid(d) + "\n")

# ---------------------------------------------------------------- part 3: rollouts with the frozen actor
specs = [r for r in args.rollouts.split(",") if r]
if specs:
    print("## Part 3: rollouts with the frozen EqM-BC actor and each critic\n")
    print(f"{args.episodes} episodes of task 1 (the same start states as the runs' evals). At every step the frozen actor of "
          "the seed proposes 32 actions (K = 20) and the checkpoint's critic picks the one with the highest min(Q1, Q2). "
          "An episode ends at the goal or after 2,000 steps. **Last 500 steps of the failed episodes** = in which cells "
          "the episodes that did not reach the goal spent their last 500 steps (share of those steps).\n")
    print("| seed | critic at | success | mean length | last 500 steps of the failed episodes |")
    print("|---|---|---|---|---|")
    tasks = ogbench_data.parse_tasks("own", env)
    envs = ogbench_data.make_eval_envs(args.env_name, tasks, args.episodes)
    for spec in specs:
        sd, st = (int(x) for x in spec.split(":"))
        _, critic, cfg = load(sd, st)
        bc = cfg["frozen_bc_config"]
        model = ConditionalMLP(state_dim + action_dim, action_dim, bc["hidden_dim"], bc["num_hidden_layers"]).to(dev)
        model.load_state_dict(torch.load(f"{cfg['frozen_bc_run']}/{cfg['bc_checkpoint']}", map_location=dev)["model"])
        model.eval().requires_grad_(False)
        smp = Actor(model=model, critic_1=None, critic_2=None, action_dim=action_dim, min_action=low, max_action=high,
                    ebm=None, opt_type=bc["actor"]["opt_type"], step_size=bc["actor"]["step_size"],
                    num_step=cfg["num_step"], moment=bc["actor"]["moment"], sampler_type="implicit",
                    ood_threshold=0.01, early_stop=None)
        N, n = cfg["eval_candidates"], len(envs)
        gen = torch.Generator(device=dev).manual_seed(sd + 2000)
        glob_rng = np.random.get_state()
        obs = np.stack([ogbench_data.seeded_reset(e, args.test_seed + 1000 * k + j)
                        for j, (k, e) in enumerate(envs)]).astype(np.float64)
        done, success, length = np.zeros(n, bool), np.zeros(n), np.zeros(n, int)
        track = [[] for _ in range(n)]                       # x-y of every step, per episode
        with torch.no_grad():
            while not done.all():
                act = np.flatnonzero(~done)
                s = torch.tensor(obs[act], dtype=torch.float32, device=dev)
                x = torch.randn(len(s) * N, action_dim, device=dev, generator=gen)
                cand = smp._implicit(x=x, state=((s - mean_t) / std_t).repeat_interleave(N, 0))
                cand = cand.clamp(low, high).view(len(s), N, action_dim)
                best = critic(s[:, None].expand(-1, N, -1), cand).argmax(1)
                a = cand[torch.arange(len(s), device=dev), best].cpu().numpy()
                for m, i in enumerate(act):
                    ob, _, term, trunc, info = envs[i][1].step(a[m])
                    obs[i] = ob
                    length[i] += 1
                    track[i].append(ob[:2].copy())
                    if term or trunc:
                        done[i], success[i] = True, float(info["success"])
        np.random.set_state(glob_rng)
        failed = [np.array(track[i][-500:]) for i in range(n) if not success[i]]
        if failed:
            c = cells_of(np.concatenate(failed))
            keys, cnt = np.unique(c, axis=0, return_counts=True)
            order = np.argsort(-cnt)[:4]
            where = ", ".join(f"({keys[o][0]},{keys[o][1]}) {100 * cnt[o] / cnt.sum():.0f} %" for o in order)
        else:
            where = "–"
        print(f"| {sd} | {st // 1000}k | {100 * success.mean():.0f} % | {length.mean():.0f} | {where} |", flush=True)
