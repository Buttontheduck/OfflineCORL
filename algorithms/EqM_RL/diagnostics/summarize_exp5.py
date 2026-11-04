"""Summary of Experiment 5 (live Aware at T = 1, 2, 3, 4 on D4RL halfcheetah-medium-expert-v2), as
Markdown: learning curves and final scores from each run's local wandb file, the live weights'
effective batch share from the per-step weighting/* logs, and the flip counts at checkpoint_310000.pt
from diagnostics/interventions.py (diagnostics/out/interv/exp5_T<T>_s<seed>.json). Means are over
seeds (± population std); per-seed values in brackets.

    python diagnostics/summarize_exp5.py [--sweep_dir sweep_runs/exp5_temperature_<date-time>] > summary.md
"""

import argparse
import csv
import glob
import json
import re
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

EQM_RL = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--sweep_dir", type=Path, help="default: the newest sweep_runs/exp5_temperature_*")
ap.add_argument("--checkpoints", type=Path, default=Path("/workspace/Trained_Models/checkpoints"))
ap.add_argument("--wandb", type=Path, default=EQM_RL / "Aware" / "wandb")
ap.add_argument("--interv", type=Path, default=EQM_RL / "diagnostics" / "out" / "interv")
args = ap.parse_args()

CURVE = (50000, 100000, 151000, 200000, 250000, 310000)   # 10k-step windows up to these steps
WKEYS = ("weighting/weights_mean", "weighting/weights_var", "weighting/weights_clip_frac",
         "weighting/adv_mean", "weighting/adv_std")


def read_wandb(run_dir: str) -> dict:
    """eval rows (step, normalized score) and per-step weighting rows from a local .wandb file."""
    from wandb.proto import wandb_internal_pb2 as pb
    from wandb.sdk.internal import datastore

    ds = datastore.DataStore()
    ds.open_for_scan(glob.glob(run_dir + "/*.wandb")[0])
    ev, wt = [], []
    while (raw := ds.scan_data()) is not None:
        rec = pb.Record()
        rec.ParseFromString(raw)
        if rec.WhichOneof("record_type") != "history":
            continue
        h = {it.key or "/".join(it.nested_key): it.value_json for it in rec.history.item}
        if "_step" not in h:
            continue
        step = int(json.loads(h["_step"]))
        if "eval/normalized_score" in h:
            ev.append((step, float(json.loads(h["eval/normalized_score"]))))
        if WKEYS[0] in h:
            wt.append([step] + [float(json.loads(h[k])) for k in WKEYS])
    return dict(eval=np.array(ev), weights=np.array(wt))


def out(line):
    print(line.replace("–%", "–"))


def ms(v, fmt="{:.1f}"):
    v = [x for x in v if x is not None and np.isfinite(x)]
    if not v:
        return "–"
    return (fmt + " ± " + fmt).format(np.mean(v), np.std(v)) if len(v) > 1 else fmt.format(v[0])


def per_seed(v, fmt="{:.1f}"):
    return " / ".join("–" if x is None else fmt.format(x) for x in v)


# ---------------------------------------------------------------- runs: T, seed, checkpoint dir, wandb dir
sweep = args.sweep_dir or sorted(EQM_RL.glob("sweep_runs/exp5_temperature_2*"))[-1]
runs = []
for row in csv.DictReader(open(sweep / "runs.csv")):
    ov = dict(kv.split("=", 1) for kv in row["overrides"].split())
    ck = sorted(args.checkpoints.glob(f"{ov['name']}-*"), key=lambda p: p.stat().st_mtime)
    ck = ck[-1] if ck else None
    wb = sorted(glob.glob(str(args.wandb / f"run-*-{ck.name.rsplit('-', 1)[-1]}-*"))) if ck else []
    runs.append(dict(T=float(ov["temperature"]), seed=int(ov["seed"]), status=row["status"], ck=ck,
                     wandb=wb[-1] if wb else None))
with Pool(min(12, len(runs))) as pool:
    hist = pool.map(read_wandb, [r["wandb"] for r in runs if r["wandb"]])
for r, h in zip([r for r in runs if r["wandb"]], hist):
    r.update(h)

by_T = defaultdict(dict)
for r in runs:
    by_T[r["T"]][r["seed"]] = r
seeds = sorted({r["seed"] for r in runs})


def window(r, lo, hi):
    e = r.get("eval")
    if e is None or not len(e):
        return None
    sel = e[(e[:, 0] > lo) & (e[:, 0] <= hi), 1]
    return float(sel.mean()) if len(sel) else None


def flips(T, s):
    f = args.interv / f"exp5_T{T:.1f}_s{s}.json"
    return json.loads(f.read_text())["treatments"]["aware_b1_n1"]["summary"] if f.exists() else None


print(f"# Experiment 5: live Aware at T = 1, 2, 3, 4 (310k steps)\n\nSweep: `{sweep}`; "
      f"run status: {dict(Counter(r['status'] for r in runs))}\n")

print("## Score at 310k (mean of the last 20 evals, 291k–310k) and flips (100 new episodes per seed, "
      "env seeds 1000+, checkpoint_310000.pt)\n")
print("| T | score | per seed | return (100 episodes) | flips | flips per seed | speed (m/s) | executed dims at the bound |")
print("|---|---|---|---|---|---|---|---|")
for T in sorted(by_T):
    sc = [window(by_T[T][s], 290000, 310000) if s in by_T[T] else None for s in seeds]
    fs = [flips(T, s) for s in seeds]
    ok = [f for f in fs if f]
    out(f"| {T:g} | {ms(sc)} | {per_seed(sc)} | {ms([f['returns'] for f in ok], '{:,.0f}')} | "
          f"{ms([100 * f['flip_rate'] for f in ok], '{:.0f}')}% | "
          f"{per_seed([None if f is None else 100 * f['flip_rate'] for f in fs], '{:.0f}')}% | "
          f"{ms([f['speed'] for f in ok])} | {ms([100 * f['at_bound'] for f in ok])}% |")
print("\nReferences: the 1M-step Aware run (T = 1, Experiment 4) scored 87.4 ± 1.7 at 310k with 8% flips; "
      "EqM on frozen Aware weights: 86.2 (T = 1) and 92.2 (matched T = 2.3–3.9); expert-only BC 92.6; "
      "Gaussian AWAC (151k) 93.4.\n")

print("## Learning curves (mean over seeds of the evals in the 10k steps up to each point)\n")
print("| T | " + " | ".join(f"{c // 1000}k" for c in CURVE) + " |")
print("|---|" + "---|" * len(CURVE))
for T in sorted(by_T):
    cells = []
    for c in CURVE:
        v = [window(r, c - 10000, c) for r in by_T[T].values()]
        v = [x for x in v if x is not None]
        cells.append(f"{np.mean(v):.1f}" if v else "–")
    print(f"| {T:g} | " + " | ".join(cells) + " |")
print("\nReference, the 1M-step run at T = 1 (20k windows): 59.7 at 151k, 87.4 at 310k.\n")

print("## Live weights (per-step logs; effective batch share = mean² / (mean² + var) within the batch "
      "of 256; × 256 = rows that effectively count)\n")
print("| T | share, steps 1k–50k | share, 50k–151k | share, 151k–310k | at the cap, 151k–310k | A mean, 151k–310k | A std, 151k–310k |")
print("|---|---|---|---|---|---|---|")
for T in sorted(by_T):
    def stat(lo, hi, f):
        v = []
        for r in by_T[T].values():
            w = r.get("weights")
            if w is None or not len(w):
                continue
            w = w[(w[:, 0] > lo) & (w[:, 0] <= hi)]
            if len(w):
                v.append(float(np.mean(f(w))))
        return v
    share = lambda w: w[:, 1] ** 2 / (w[:, 1] ** 2 + w[:, 2])
    out(f"| {T:g} | {ms(stat(1000, 50000, share), '{:.3f}')} | {ms(stat(50000, 151000, share), '{:.3f}')} | "
          f"{ms(stat(151000, 310000, share), '{:.3f}')} | {ms(stat(151000, 310000, lambda w: 100 * w[:, 3]))}% | "
          f"{ms(stat(151000, 310000, lambda w: w[:, 4]), '{:.2f}')} | {ms(stat(151000, 310000, lambda w: w[:, 5]), '{:.2f}')} |")
print()
missing = [f"T{r['T']:g}_s{r['seed']}" for r in runs if not r["wandb"]]
if missing:
    print(f"No local wandb file for: {', '.join(missing)}\n")
