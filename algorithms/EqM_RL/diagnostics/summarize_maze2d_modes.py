"""Markdown tables from the JSON files of maze2d_modes.py: means over seeds per dataset, method
(Aware temperature) and checkpoint. Usage: python diagnostics/summarize_maze2d_modes.py <folder>"""

import glob
import json
import sys
from collections import defaultdict

import numpy as np

DATASETS = ["maze2d-umaze-v1", "maze2d-medium-v1", "maze2d-large-v1"]
SAMPLER_COLS = ["f_end", "spread", "n_modes", "n_modes_1", "top_mass", "recall", "precision", "energy",
                "off_support", "at_bound"]


def tag(name: str) -> str:
    step = int(name.split("_")[-1].split(".")[0])
    return "1M" if step >= 1_000_000 else f"{round(step / 1000)}k"


def ms(v, d=3):
    v = np.array(v, float)
    return f"{v.mean():.{d}f}" if len(v) == 1 else f"{v.mean():.{d}f} ± {v.std():.{d}f}"


runs = [json.load(open(f)) for f in sorted(glob.glob(f"{sys.argv[1]}/*.json"))]
samp, roll = defaultdict(list), defaultdict(list)
data, awac = defaultdict(list), defaultdict(list)
for r in runs:
    env, T = r["info"]["env"], r["info"]["temperature"]
    method = f"Aware T = {T:g}"
    data[env].append(r["data"])
    if r.get("awac"):
        awac[env].append(r["awac"])
    for name, c in r["checkpoints"].items():
        # Experiment 8's runs stopped at 440k-540k: pool them as "~500k"
        t = tag(name)
        t = t if t == "1M" else "~500k"
        for K, m in c["sampler"].items():
            samp[(env, method, t, int(K))].append(m)
        for K, m in c["rollouts"].items():
            roll[(env, method, t, int(K))].append(m)

print("# Maze2D action diagnostics (maze2d_modes.py)\n")
print("Means ± std over seeds. Distances are in action units (actions in [-1, 1]^2). n_modes and "
      "top_mass are meaningful only where f_end ≈ 0 (converged samples).\n")
for env in DATASETS:
    if env not in data:
        continue
    d = data[env][0]
    print(f"## {env}\n")
    print(f"Data reference (actions of the 128 nearest dataset states; median neighbour radius "
          f"{d['nn_radius_median']:.3f} in normalized state space): spread {d['spread']:.3f}, at ±1 "
          f"{d['at_bound']:.3f}, half vs half: recall {d['ref']['recall']:.3f}, precision "
          f"{d['ref']['precision']:.3f}, energy {d['ref']['energy']:.3f}, off_support 0.05 (by construction).")
    if awac[env]:
        a = awac[env]
        print(f"Gaussian AWAC mean action ({len(a)} runs, {a[0]['checkpoint']} or nearest): precision "
              f"{ms([x['precision'] for x in a])}, off_support {ms([x['off_support'] for x in a])}, at ±1 "
              f"{ms([x['at_bound'] for x in a])}.")
    print("\n| method | checkpoint | K | " + " | ".join(SAMPLER_COLS) + " | seeds |")
    print("|" + "---|" * (len(SAMPLER_COLS) + 4))
    for key in sorted(k for k in samp if k[0] == env):
        v = samp[key]
        print(f"| {key[1]} | {key[2]} | {key[3]} | " +
              " | ".join(ms([m[c] for m in v], 2 if c in ("n_modes",) else 3) for c in SAMPLER_COLS) +
              f" | {len(v)} |")
    print("\n**Rollouts** (D4RL normalized score over the episodes; action change = mean |a_t − a_t−1| summed over dims)\n")
    print("| method | checkpoint | K | score | episode std | worst episode | executed at ±1 | action change | seeds |")
    print("|---|---|---|---|---|---|---|---|---|")
    for key in sorted(k for k in roll if k[0] == env):
        v = roll[key]
        print(f"| {key[1]} | {key[2]} | {key[3]} | {ms([m['score'] for m in v], 1)} | "
              f"{ms([m['score_std'] for m in v], 1)} | {ms([m['score_min'] for m in v], 1)} | "
              f"{ms([m['at_bound'] for m in v])} | {ms([m['action_change'] for m in v])} | {len(v)} |")
    print()
