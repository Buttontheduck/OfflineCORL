"""Summary of the 1M-step Aware runs at T = 1 (Experiment 4, exp4_aware_1m) and T = 4 (Experiment 5,
exp5_aware_1m_T4) on D4RL halfcheetah-medium-expert-v2, as Markdown: scores along training (local
wandb files), flip counts at several checkpoints (diagnostics/interventions.py outputs), and the
live weights' effective batch share. Means over seeds (± population std), per seed in brackets.

    python diagnostics/summarize_exp5_1m.py > summary.md
"""

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from wandb_history import read_aware_history, wandb_dir_of

EQM_RL = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--run", action="append",
                help="LABEL=CHECKPOINT_PREFIX:FLIP_PREFIX, repeatable; checkpoint folders <prefix>_seed<s>-*, "
                     "flip files <flip prefix>_ck<step>_s<s>.json (default: T = 1 and T = 4)")
ap.add_argument("--seeds", default="1,2,3")
ap.add_argument("--title", default="1M-step Aware: T = 1 vs T = 4 (D4RL halfcheetah-medium-expert-v2)")
ap.add_argument("--window", type=int, default=20000, help="score = mean of the evals in this many steps up to each step")
ap.add_argument("--refs", default="References: CORL's AWAC after 1M steps 95.0 ± 0.6 (README); our Gaussian AWAC at 151k "
                "93.4 (11,261 return, 3% flips on the same episodes); expert-only BC (EqM) 92.6.")
ap.add_argument("--steps", default="151000,310000,500000,700000,1000000", help="score windows end here")
ap.add_argument("--flip_steps", default="151000,310000,500000,1000000", help="'' = no flip table")
ap.add_argument("--checkpoints", type=Path, default=Path("/workspace/Trained_Models/checkpoints"))
ap.add_argument("--wandb", type=Path, default=EQM_RL / "Aware" / "wandb")
ap.add_argument("--interv", type=Path, default=EQM_RL / "diagnostics" / "out" / "interv")
args = ap.parse_args()
specs = args.run or ["T = 1=Aware_1M:aware1m", "T = 4=Aware_1M_T4:aware1m_T4"]
seeds = [int(s) for s in args.seeds.split(",")]
steps = [int(s) for s in args.steps.split(",")]
flip_steps = [int(s) for s in args.flip_steps.split(",") if s]


def ms(v, fmt="{:.1f}"):
    v = [x for x in v if x is not None and np.isfinite(x)]
    if not v:
        return "–"
    return (fmt + " ± " + fmt).format(np.mean(v), np.std(v)) if len(v) > 1 else fmt.format(v[0])


def per_seed(v, fmt="{:.1f}"):
    return " / ".join("–" if x is None else fmt.format(x) for x in v)


runs = []   # one per (label, seed)
for spec in specs:
    label, rest = spec.rsplit("=", 1)
    ck_prefix, flip_prefix = rest.split(":")
    for s in seeds:
        cks = sorted(args.checkpoints.glob(f"{ck_prefix}_seed{s}-*"), key=lambda p: p.stat().st_mtime)
        ck = cks[-1] if cks else None
        runs.append(dict(label=label.strip(), seed=s, ck=ck, flip_prefix=flip_prefix,
                         wandb=wandb_dir_of(ck, args.wandb) if ck else None))
todo = [r for r in runs if r["wandb"]]
with Pool(max(1, len(todo))) as pool:
    for r, h in zip(todo, pool.map(read_aware_history, [r["wandb"] for r in todo])):
        r.update(h)
labels = list(dict.fromkeys(r["label"] for r in runs))
group = {lab: [r for r in runs if r["label"] == lab] for lab in labels}


def window(r, lo, hi):
    e = r.get("eval")
    if e is None or not len(e):
        return None
    sel = e[(e[:, 0] > lo) & (e[:, 0] <= hi), 1]
    return float(sel.mean()) if len(sel) else None


def flips(r, step):
    f = args.interv / f"{r['flip_prefix']}_ck{step}_s{r['seed']}.json"
    return json.loads(f.read_text())["treatments"]["aware_b1_n1"]["summary"] if f.exists() else None


print(f"# {args.title}\n")
for r in runs:
    if r["wandb"] is None:
        print(f"- no checkpoint or local wandb file for {r['label']}, seed {r['seed']}")
last = {lab: min((int(r["eval"][-1, 0]) for r in g if "eval" in r and len(r["eval"])), default=0) for lab, g in group.items()}
print("Last eval step per setting: " + ", ".join(f"{lab}: {s:,}" for lab, s in last.items()) + "\n")

print(f"## Score (mean of the evals in the {args.window // 1000}k steps up to each step)\n")
print("| T | " + " | ".join(f"{s // 1000}k" for s in steps) + " | best single eval |")
print("|---|" + "---|" * (len(steps) + 1))
for lab, g in group.items():
    cells = [ms([window(r, s - args.window, s) for r in g]) for s in steps]
    best = [float(r["eval"][:, 1].max()) for r in g if "eval" in r and len(r["eval"])]
    print(f"| {lab} | " + " | ".join(cells) + f" | {per_seed(best)} |")
print("\nPer seed:\n")
print("| T | " + " | ".join(f"{s // 1000}k" for s in steps) + " |")
print("|---|" + "---|" * len(steps))
for lab, g in group.items():
    print(f"| {lab} | " + " | ".join(per_seed([window(r, s - args.window, s) for r in g]) for s in steps) + " |")

if flip_steps:
    print("\n## Checkpoints: 100 new episodes per seed (env seeds 1000+), trained sampler\n")
    print("| T | step | return | flips | flips per seed | speed (m/s) | executed dims at the bound |")
    print("|---|---|---|---|---|---|---|")
for lab, g in group.items():
    for s in flip_steps:
        fs = [flips(r, s) for r in g]
        ok = [f for f in fs if f]
        if not ok:
            continue
        fl = per_seed([None if f is None else 100 * f["flip_rate"] for f in fs], "{:.0f}")
        print(f"| {lab} | {s // 1000}k | {ms([f['returns'] for f in ok], '{:,.0f}')} | "
              f"{ms([100 * f['flip_rate'] for f in ok], '{:.0f}')}% | {fl} (%) | {ms([f['speed'] for f in ok])} | "
              f"{ms([100 * f['at_bound'] for f in ok])}% |")

print("\n## Live weights (effective batch share = mean² / (mean² + var) within the batch of 256; "
      "× 256 = rows that effectively count)\n")
bins = [(1000, 50000), (50000, 151000), (151000, 310000), (310000, 500000), (500000, 1000000)]
print("| T | " + " | ".join(f"{lo // 1000}k–{hi // 1000}k" for lo, hi in bins) + " | A std, last bin |")
print("|---|" + "---|" * (len(bins) + 1))
for lab, g in group.items():
    def stat(lo, hi, f):
        v = []
        for r in g:
            w = r.get("weights")
            if w is not None and len(w):
                w = w[(w[:, 0] > lo) & (w[:, 0] <= hi)]
                if len(w):
                    v.append(float(np.mean(f(w))))
        return v
    share = lambda w: w[:, 1] ** 2 / (w[:, 1] ** 2 + w[:, 2])
    print(f"| {lab} | " + " | ".join(ms(stat(lo, hi, share), "{:.3f}") for lo, hi in bins)
          + f" | {ms(stat(*bins[-1], lambda w: w[:, 5]), '{:.2f}')} |")
print("\n" + args.refs)
