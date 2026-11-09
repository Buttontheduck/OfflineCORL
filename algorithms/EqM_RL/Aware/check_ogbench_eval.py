"""Check ogbench_data.evaluate (the OGBench scoring used by fit_actor.py) against an independent, plain
evaluation loop written like OGBench's reference evaluation (FQL's utils/evaluation.py): one episode at a time,
until terminated or truncated, success = info["success"] at the last step, actions clipped to [-1, 1].
Both run the same deterministic policies from the same seeds, so every episode must match exactly
(success, return, length).

Policies (pointmaze, whose observation is the agent's xy):
  oracle   steers to the next cell of the shortest grid path to the task's goal (breadth-first search on
           maze_map). It should solve almost every task: tests the success path (termination at the goal,
           return = -(steps before the goal)).
  zero     always action 0: never moves, never succeeds (unless it starts at the goal): tests the failure path
           (truncation at max_episode_steps, return = -max_episode_steps).
Also checks the dataset conversion: terminals = 1 - masks, rewards in {-1, 0}, episode ids from OGBench's
trajectory ends. Run with the ogbench env:

    /workspace/miniconda3/envs/ogbench/bin/python check_ogbench_eval.py [--env pointmaze-medium-navigate-singletask-v0]
"""

import argparse
import sys
from collections import deque

import numpy as np

import ogbench_data

ap = argparse.ArgumentParser()
ap.add_argument("--env", default="pointmaze-medium-navigate-singletask-v0")   # the policies need a point maze
ap.add_argument("--episodes", type=int, default=4, help="per task")
ap.add_argument("--seed", type=int, default=69)
args = ap.parse_args()
failed = []

# ---------------------------------------------------------------- data conversion
env, data, mean, std, ep, info = ogbench_data.load_ogbench(args.env)
print(f"data: {info}")
ok = (data["rewards"].max() <= 0 and np.array_equal(data["terminals"] == 1, data["rewards"] == 0)
      and ep[-1] == info["episodes"] - 1 and np.all(np.diff(ep) >= 0))
failed += [] if ok else ["data conversion"]
print(f"data conversion (terminals = 1 - masks, rewards, episode ids): {'ok' if ok else 'FAIL'}\n")


# ---------------------------------------------------------------- policies (raw xy in, actions out)
def make_oracle(e):
    u = e.unwrapped
    walls = u.maze_map

    def next_xy(xy):
        start, goal = u.xy_to_ij(xy), u.xy_to_ij(u.cur_goal_xy)
        if start == goal:
            return np.array(u.cur_goal_xy)
        prev, q = {start: None}, deque([start])
        while q:
            c = q.popleft()
            if c == goal:
                break
            for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nb = (c[0] + d[0], c[1] + d[1])
                if nb not in prev and walls[nb] == 0:
                    prev[nb] = c
                    q.append(nb)
        c = goal
        while prev[c] != start:
            c = prev[c]
        return np.array(u.ij_to_xy(c))

    return lambda xy: np.clip(2.0 * (next_xy(xy) - xy), -1, 1)


def batched_act(envs, kind):
    """act() for ogbench_data.evaluate: normalized states -> actions; the oracle needs each env's goal, so the
    batch is matched to the unfinished envs in order (evaluate passes them in order)."""
    oracles = [make_oracle(e) for _, e in envs]
    state = {"done": np.zeros(len(envs), bool)}

    def act(s):
        xy = s.astype(np.float64) * std + mean
        if kind == "zero":
            return np.zeros((len(s), 2), np.float32)
        active = [i for i, (_, e) in enumerate(envs) if not _finished(e)]
        assert len(active) == len(s)
        return np.stack([oracles[i](xy[m]) for m, i in enumerate(active)]).astype(np.float32)

    return act


def _finished(e):
    return getattr(e, "_check_finished", False)


class Marked:
    """Wraps an env so the oracle can tell which envs are still running (set when an episode ends)."""
    def __init__(self, e):
        self.e, self._check_finished = e, False
        self.unwrapped, self.spec = e.unwrapped, e.spec

    def reset(self, **kw):
        self._check_finished = False
        return self.e.reset(**kw)

    def step(self, a):
        out = self.e.step(a)
        if out[2] or out[3]:
            self._check_finished = True
        return out


def reference(envs, kind, seed):
    """OGBench-style plain loop: one episode at a time, success = last info['success']."""
    res, j = [], {}
    for k, e in envs:
        j[k] = j.get(k, -1) + 1
        s = seed + 1000 * k + j[k]
        np.random.seed(s)                 # the maze start noise comes from numpy's global RNG
        ob, _ = e.reset(seed=s)
        pol = make_oracle(e)
        done, ret, length = False, 0.0, 0
        while not done:
            # the policy sees what fit_actor's act() gets: the state normalized and cast to float32
            xy = ((np.asarray(ob, np.float64) - mean) / std).astype(np.float32).astype(np.float64) * std + mean
            a = np.zeros(2) if kind == "zero" else pol(xy)
            ob, r, term, trunc, inf = e.step(np.clip(np.asarray(a, np.float32), -1, 1))
            done = term or trunc
            ret += r
            length += 1
        res.append((k, float(inf["success"]), ret, length, bool(term)))
    return res


tasks = ogbench_data.parse_tasks("all", env)
max_steps = info["max_episode_steps"]
for kind in ("oracle", "zero"):
    envs = [(k, Marked(e)) for k, e in ogbench_data.make_eval_envs(args.env, tasks, args.episodes)]
    got = ogbench_data.evaluate(envs, batched_act(envs, kind), mean, std, args.seed, -1.0, 1.0)
    ref_envs = [(k, e) for k, e in ogbench_data.make_eval_envs(args.env, tasks, args.episodes)]
    ref = reference(ref_envs, kind, args.seed)
    print(f"{kind}: normalized_score {got['normalized_score']:.1f} %, mean return {got['score']:.1f}, "
          f"mean length {got['episode_length']:.1f}")
    for k in tasks:
        r = [x for x in ref if x[0] == k]
        rs, rr, rl = 100 * np.mean([x[1] for x in r]), np.mean([x[2] for x in r]), np.mean([x[3] for x in r])
        same = (np.isclose(got[f"success_task{k}"], rs) and np.isclose(got[f"return_task{k}"], rr)
                and np.isclose(got[f"length_task{k}"], rl))
        # single-task rules: success <=> terminated before the limit; return = -(steps before the goal)
        rules = all((s == 1.0) == term and (rt == -(ln - 1) if s == 1.0 else rt == -ln) and ln <= max_steps
                    for _, s, rt, ln, term in r)
        failed += [] if same and rules else [f"{kind} task {k}"]
        print(f"   task {k}: batched {got[f'success_task{k}']:5.1f} % / {got[f'return_task{k}']:7.1f} / "
              f"{got[f'length_task{k}']:6.1f}   reference {rs:5.1f} % / {rr:7.1f} / {rl:6.1f}   "
              f"{'same' if same else 'DIFFERENT'}, rules {'ok' if rules else 'FAIL'}")
    print()
mean_ok = np.isclose(got["normalized_score"], np.mean([got[f"success_task{k}"] for k in tasks]))
failed += [] if mean_ok else ["normalized_score = mean over tasks"]

# repeatable: the same policy and seed give the same numbers; the caller's global numpy RNG is left as it was
envs = [(k, Marked(e)) for k, e in ogbench_data.make_eval_envs(args.env, tasks, args.episodes)]
np.random.seed(123)
before = np.random.get_state()[1].copy()
a1 = ogbench_data.evaluate(envs, batched_act(envs, "oracle"), mean, std, args.seed, -1.0, 1.0)
rng_kept = np.array_equal(np.random.get_state()[1], before)
a2 = ogbench_data.evaluate(envs, batched_act(envs, "oracle"), mean, std, args.seed, -1.0, 1.0)
same = all(np.isclose(a1[k], a2[k]) for k in a1)
failed += [] if same and rng_kept else ["repeatability / global RNG"]
print(f"repeat eval identical: {same}; caller's numpy RNG unchanged: {rng_kept}")
print(f"normalized_score = mean over tasks: {mean_ok}")
print("ALL CHECKS PASSED" if not failed else f"FAILED: {failed}")
sys.exit(1 if failed else 0)
