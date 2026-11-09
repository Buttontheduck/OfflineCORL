"""Compare the critics of three humanoidmaze-medium runs over their first training steps, from the local wandb
files in Aware/wandb (no network). Read-only; prints markdown.

  Exp18  Aware's own twin critic (Q of the current EqM policy), V from 16 samples, gamma 0.999, 4 x 512, LayerNorm,
         no target clip                                   (group Exp18_aware_humanoidmaze-medium, running)
  Exp13  Aware's own twin critic, V from 1 sample, gamma 0.995, 4 x 256, no LayerNorm, no target clip
                                                          (group Exp13_awac_humanoidmaze-medium, stopped at 100-200k)
  Exp14  QIPO in-support softmax critic for the frozen EqM-BC actor, gamma 0.995, 4 x 512, LayerNorm, target clipped
         to [-200, 0]                                     (group Exp14_Critic_humanoidmaze-medium, 1M steps)

Rewards are -1 per step and 0 at the goal, so every Q lies in [floor, 0] with floor = -1 / (1 - gamma): -1000 for
gamma 0.999, -200 for 0.995. "Q / floor" makes the two discounts comparable: 0 = goal reached at once, 1 = goal never
reached. Above 1 = below the floor (impossible for a true Q: divergence downward).
TD loss: Aware logs train/critic_loss = MSE(Q1) + MSE(Q2); the QIPO critic logs 0.5 (MSE(Q1) + MSE(Q2)). Both are
shown per network (Aware's halved), and as RMS error / |floor| so the two discounts compare.

  /workspace/miniconda3/envs/ogbench/bin/python diagnostics/compare_critics_early.py [--runs Exp18,Exp14]
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent / "Aware" / "wandb"
RUNS = {  # name: (wandb group, gamma, key of Q at dataset actions, key of the TD loss, loss factor to per network)
    "Exp18": ("Exp18_aware_humanoidmaze-medium", 0.999, "weighting/q_mean", "train/critic_loss", 0.5),
    "Exp13": ("Exp13_awac_humanoidmaze-medium", 0.995, "weighting/q_mean", "train/critic_loss", 0.5),
    "Exp14": ("Exp14_Critic_humanoidmaze-medium", 0.995, "train/q_data", "train/q_loss", 1.0),
}
STEPS = [1000, 2000, 5000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 150000, 200000,
         250000, 300000, 500000, 1000000]
_p = argparse.ArgumentParser()
_p.add_argument("--runs", default="Exp18,Exp13,Exp14", help="which of Exp18, Exp13, Exp14 to show")
_p.add_argument("--max_step", type=int, default=None, help="drop rows after this step")
_a = _p.parse_args()
RUNS = {n: RUNS[n] for n in _a.runs.split(",")}
if _a.max_step:
    STEPS = [t for t in STEPS if t <= _a.max_step]


def run_args(run_dir):
    try:
        meta = json.loads(Path(run_dir, "files", "wandb-metadata.json").read_text())
    except (OSError, ValueError):
        return {}
    return dict(a.lstrip("-").split("=", 1) for a in meta.get("args", []) if "=" in a)


def read(run_dir):
    from wandb.proto import wandb_internal_pb2 as pb
    from wandb.sdk.internal import datastore
    files = glob.glob(run_dir + "/*.wandb")
    out = {}
    if not files:
        return out
    ds = datastore.DataStore()
    ds.open_for_scan(files[0])
    try:
        while (raw := ds.scan_data()) is not None:
            rec = pb.Record()
            rec.ParseFromString(raw)
            if rec.WhichOneof("record_type") != "history":
                continue
            h = {it.key or "/".join(it.nested_key): it.value_json for it in rec.history.item}
            if "_step" not in h:
                continue
            for k, v in h.items():
                try:
                    out.setdefault(k, []).append((int(json.loads(h["_step"])), float(json.loads(v))))
                except (TypeError, ValueError):
                    pass
    except Exception:   # a run still writing can end in a partial record
        pass
    return {k: np.array(v) for k, v in out.items()}


data = {}
for d in sorted(glob.glob(f"{ROOT}/run-*")):   # sorted by date-time, so the newest run of a group/seed wins
    a = run_args(d)
    for name, (group, *_rest) in RUNS.items():
        if a.get("group") == group and a.get("seed", "").isdigit():
            data[(name, int(a["seed"]))] = read(d)


def at(name, seed, key, step):
    """Value logged at the last step <= step (logs come every 1000 steps), or None if the run isn't there yet."""
    x = data.get((name, seed), {}).get(key)
    if x is None or len(x) == 0 or x[:, 0].max() < step:
        return None
    i = np.searchsorted(x[:, 0], step, side="right") - 1
    return None if i < 0 else x[i, 1]


def cell(vals, fmt="{:.2f}"):
    v = [x for x in vals if x is not None]
    return " / ".join(fmt.format(x) for x in v) if v else "–"


seeds = (1, 2, 3)
last = {n: max([int(data[(n, s)][RUNS[n][2]][:, 0].max()) for s in seeds if (n, s) in data and RUNS[n][2] in data[(n, s)]]
               or [0]) for n in RUNS}
print("Runs found:", ", ".join(f"{n}: seeds {[s for s in seeds if (n, s) in data]}, logged up to {last[n]:,} steps"
                               for n in RUNS), "\n")

print("## Q at dataset actions, as Q / floor (per seed 1 / 2 / 3; 1 = the floor, > 1 = below it)\n")
print("| step | " + " | ".join(f"{n} (γ {RUNS[n][1]}, floor {-1 / (1 - RUNS[n][1]):.0f})" for n in RUNS) + " |")
print("|---|" + "---|" * len(RUNS))
for t in STEPS:
    row = [cell([(lambda q: None if q is None else q * (1 - RUNS[n][1]) * -1)(at(n, s, RUNS[n][2], t)) for s in seeds])
           for n in RUNS]
    if any(c != "–" for c in row):
        print(f"| {t:,} | " + " | ".join(row) + " |")

print("\n## Q at dataset actions, raw (per seed)\n")
print("| step | " + " | ".join(RUNS) + " |")
print("|---|" + "---|" * len(RUNS))
for t in STEPS:
    row = [cell([at(n, s, RUNS[n][2], t) for s in seeds], "{:.1f}") for n in RUNS]
    if any(c != "–" for c in row):
        print(f"| {t:,} | " + " | ".join(row) + " |")

print("\n## TD error: RMS per network / |floor| (per seed)\n")
print("| step | " + " | ".join(RUNS) + " |")
print("|---|" + "---|" * len(RUNS))
for t in STEPS:
    row = []
    for n in RUNS:
        g, _, _, lk, f = RUNS[n]
        row.append(cell([None if (L := at(n, s, lk, t)) is None else np.sqrt(max(L * f, 0)) * (1 - RUNS[n][1])
                         for s in seeds], "{:.4f}"))
    if any(c != "–" for c in row):
        print(f"| {t:,} | " + " | ".join(row) + " |")

print("\n## Aware only: the advantage weights (per seed)\n")
keys = ["weighting/v_mean", "weighting/adv_mean", "weighting/adv_std", "weighting/weights_mean",
        "weighting/weights_mean_1sample", "weighting/v_sample_std", "weighting/weights_clip_frac",
        "weighting/weights_median"]
for n in [r for r in ("Exp18", "Exp13") if r in RUNS]:
    print(f"\n**{n}**\n")
    print("| step | " + " | ".join(k.split("/")[1] for k in keys) + " |")
    print("|---|" + "---|" * len(keys))
    for t in STEPS:
        row = [cell([at(n, s, k, t) for s in seeds], "{:.3g}") for k in keys]
        if any(c != "–" for c in row):
            print(f"| {t:,} | " + " | ".join(row) + " |")

if "Exp14" in RUNS:
    print("\n## Exp14 only: the softmax target (per seed)\n")
    keys = ["train/target", "support/eff_n", "support/w_data", "support/lift", "support/q_range"]
    print("| step | " + " | ".join(k.split("/")[1] for k in keys) + " |")
    print("|---|" + "---|" * len(keys))
    for t in STEPS:
        row = [cell([at("Exp14", s, k, t) for s in seeds], "{:.3g}") for k in keys]
        if any(c != "–" for c in row):
            print(f"| {t:,} | " + " | ".join(row) + " |")

print("\n## Evals (success %, per seed)\n")
print("| step | " + " | ".join(RUNS) + " |")
print("|---|" + "---|" * len(RUNS))
for t in (0, 100000, 200000, 500000, 1000000):
    row = [cell([at(n, s, "eval/normalized_score", t) if t > 0 else
                 (lambda x: None if x is None or len(x) == 0 else x[0, 1])(data.get((n, s), {}).get("eval/normalized_score"))
                 for s in seeds], "{:.0f}") for n in RUNS]
    if any(c != "–" for c in row):
        print(f"| {t:,} | " + " | ".join(row) + " |")
