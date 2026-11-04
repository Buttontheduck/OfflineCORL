"""Summary tables of Experiment 4, priority 1 and 2 (EqM_RL.md, "Planned next experiments"), as
Markdown, from the outputs run_exp4.sh leaves: diagnostics/out/{interv,handoff,roll}/*.json, the
weight files' metadata, and the eval lines in the training sweeps' stdout logs. Means are over
seeds (± population std over seeds); each seed's value is itself a mean over its episodes.

    python diagnostics/summarize_exp4.py [--orchestrator_dir sweep_runs/exp4_<date-time>] > summary.md
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

EQM_RL = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--out", type=Path, default=EQM_RL / "diagnostics" / "out")
ap.add_argument("--sweeps", type=Path, default=EQM_RL / "sweep_runs")
ap.add_argument("--weights", type=Path, default=Path("/workspace/Trained_Models/weights"))
ap.add_argument("--orchestrator_dir", type=Path, help="run_exp4.sh output folder (its log is quoted at the end)")
args = ap.parse_args()


def ms(v, fmt="{:.0f}"):
    v = [x for x in v if x is not None and np.isfinite(x)]
    if not v:
        return "–"
    return (fmt + " ± " + fmt).format(np.mean(v), np.std(v)) if len(v) > 1 else fmt.format(v[0])


def pct(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return "–" if not v else (f"{100 * np.mean(v):.0f}% ± {100 * np.std(v):.0f}" if len(v) > 1 else f"{100 * v[0]:.0f}%")


def seed_of(name):
    m = re.search(r"_s(\d+)", name)
    return int(m.group(1)) if m else None


print("# Experiment 4: priority 1 and 2, summary\n")

# ------------------------------------------------------------------ 1a interventions
print("## 1a. Clipping and averaging (D4RL medium-expert, existing checkpoints)\n")
for split, suffix in (("screening, 20 episodes per seed (env seeds 69+)", ""),
                      ("confirmation, 100 new episodes per seed (env seeds 1000+)", "_confirm")):
    files = sorted((args.out / "interv").glob(f"me_*_s[0-9]{suffix}.json"))
    if not files:
        continue
    print(f"### {split}\n")
    by_run = defaultdict(list)
    for f in files:
        by_run[re.sub(r"_s\d+(_confirm)?$", "", f.stem)].append(json.loads(f.read_text()))
    for run, rs in sorted(by_run.items()):
        print(f"**{run}** ({len(rs)} seeds)\n")
        print("| treatment | return | flips | speed | raw outside box | executed at bound | "
              "pre-flip at bound: flip / no flip | flips per seed |")
        print("|---|---|---|---|---|---|---|---|")
        for name in rs[0]["treatments"]:
            S = [r["treatments"][name]["summary"] for r in rs if name in r["treatments"]]
            P = [r["treatments"][name]["pre_flip"]["at_bound"] for r in rs if name in r["treatments"]]
            fl = [p["flipped"] for p in P if p["flipped"] is not None]
            nf = [p["matched_no_flip"] for p in P if p["matched_no_flip"] is not None]
            pre = f"{np.mean(fl):.3f} / {np.mean(nf):.3f}" if fl and nf else "–"
            per_seed = " / ".join(f"{100 * s['flip_rate']:.0f}%" for s in S)
            print(f"| {name} | {ms([s['returns'] for s in S])} | {pct([s['flip_rate'] for s in S])} | "
                  f"{ms([s['speed'] for s in S], '{:.2f}')} | {ms([s['outside'] for s in S], '{:.3f}')} | "
                  f"{ms([s['at_bound'] for s in S], '{:.3f}')} | {pre} | {per_seed} |")
        print()

# ------------------------------------------------------------------ 1b handoff
files = sorted((args.out / "handoff").glob("me_*_s[0-9].json"))
if files:
    print("## 1b. Handoff (2 × 2: who generated the state × who continues, a fixed horizon after the switch)\n")
    by_run = defaultdict(list)
    for f in files:
        by_run[re.sub(r"_s\d+$", "", f.stem)].append(json.loads(f.read_text()))
    for run, rs in sorted(by_run.items()):
        chk = max(r["restore_check_max_abs_diff"] for r in rs)
        print(f"**{run}** ({len(rs)} seeds; restore check max |diff| {chk:.1e}). Flip rate and return over "
              f"the {rs[0]['args']['horizon']} steps after the switch, upright states only.\n")
        print("| state from → continued by | t | upright states | flips | return | already flipped |")
        print("|---|---|---|---|---|---|")
        for cell in rs[0]["cells"]:
            cs = [r["cells"][cell] for r in rs]
            c0 = cs[0]
            print(f"| {c0['source']} → {c0['continued_by']} | {c0['t']} | {sum(c['n_upright'] for c in cs)} | "
                  f"{pct([c['flip_rate_upright'] for c in cs])} | {ms([c['return_upright'] for c in cs])} | "
                  f"{sum(c['n_flipped_before'] for c in cs)} |")
        print()

# ------------------------------------------------------------------ 1c Minari flips
files = sorted((args.out / "roll").glob("medium_*_s[0-9].json"))
if files:
    print("## 1c. Flips on Minari medium-v0 (20 episodes per seed, env seeds 1000+)\n")
    print("| run | seeds | return | flips | flips per seed |")
    print("|---|---|---|---|---|")
    by_run = defaultdict(list)
    for f in files:
        by_run[re.sub(r"_s\d+$", "", f.stem)].append(json.loads(f.read_text()))
    for run, rs in sorted(by_run.items()):
        fr = [np.mean(np.array(r["flip_step"]) >= 0) for r in rs]
        print(f"| {run} | {len(rs)} | {ms([np.mean(r['returns']) for r in rs])} | {pct(fr)} | "
              f"{' / '.join(f'{100 * x:.0f}%' for x in fr)} |")
    print("\nReference: the medium-v0 data itself flips in 20.6% of its episodes (Experiment 2, diagnostic C).\n")

# ------------------------------------------------------------------ priority 2: weights
wfiles = sorted(args.weights.glob("*_seed[0-9].npz"))
if wfiles:
    print("## 2. Frozen weights (make_weights.py)\n")
    print("| file | source checkpoint | T | effective batch share | at cap | near zero | A > 0 |")
    print("|---|---|---|---|---|---|---|")
    for f in wfiles:
        m = json.loads(str(np.load(f)["meta"]))
        st = m["stats"]
        print(f"| {f.stem} | {m['run'][:40]}/{m['checkpoint']} | {m['temperature']:.3g} | {st['batch_share']:.3f} | "
              f"{st['cap_share']:.3f} | {st['near_zero_share']:.3f} | {st['positive_adv_share']:.3f} |")
    print()


# ------------------------------------------------------------------ priority 2: training sweeps
def newest(pattern):
    ds = sorted(args.sweeps.glob(pattern), key=lambda p: p.stat().st_mtime)
    return ds[-1] if ds else None


fit_line = re.compile(r"^\s*(\d+)\s+score\s+(-?[\d.]+)\s+normalized\s+(-?[\d.]+)\s+flips\s+(\d+)%")
print("## 2. Training runs (310k steps): mean of the last 20 evals (last 20k steps), then over seeds\n")
print("| sweep | actor | seeds done | normalized score | flips (eval episodes) | last step | per seed |")
print("|---|---|---|---|---|---|---|")
for sweep in ("exp4_fit_bc", "exp4_fit_iql", "exp4_fit_aware", "exp4_fit_aware_matched"):
    d = newest(f"{sweep}_2*")
    if d is None:
        print(f"| {sweep} | – | not run | | | | |")
        continue
    groups = defaultdict(list)
    for log in sorted(d.glob("*/stdout.log")):
        m = re.search(r"actor(\w+?)_seed(\d+)", log.parent.name)
        rows = [fit_line.match(l) for l in log.read_text(errors="ignore").splitlines()]
        rows = [(int(r[1]), float(r[3]), float(r[4]) / 100) for r in rows if r]
        if m and rows:
            last = rows[-20:]
            groups[m.group(1)].append((np.mean([r[1] for r in last]), np.mean([r[2] for r in last]), rows[-1][0]))
    for actor, v in sorted(groups.items()):
        print(f"| {sweep} | {actor} | {len(v)} | {ms([x[0] for x in v], '{:.1f}')} | {pct([x[1] for x in v])} | "
              f"{min(x[2] for x in v)} | {' / '.join(f'{x[0]:.1f}' for x in v)} |")

d = newest("exp4_iql_2*")
if d is not None:
    iql = []
    for log in sorted(d.glob("*/stdout.log")):
        txt = log.read_text(errors="ignore")
        steps = [int(s) for s in re.findall(r"Time steps: (\d+)", txt)]
        scores = [float(s) for s in re.findall(r"D4RL score: (-?[\d.]+)", txt)]
        if scores:
            iql.append((np.mean(scores[-4:]), steps[-1] if steps else 0))
    if iql:
        print(f"| exp4_iql (IQL's own Gaussian policy, β = 3; reference) | gaussian | {len(iql)} | "
              f"{ms([x[0] for x in iql], '{:.1f}')} | – | {min(x[1] for x in iql)} | "
              f"{' / '.join(f'{x[0]:.1f}' for x in iql)} |")
print("\nIQL's own policy is evaluated every 5k steps, so its value is the mean of the last 4 evals "
      "(last 20k steps). References at 151k steps (Experiment 3): Gaussian AWAC 93.4, Aware 59.5, "
      "Aware with BC weights 50.1; dataset halves: medium 40.7, expert 88.1.\n")

if args.orchestrator_dir and (args.orchestrator_dir / "orchestrator.log").exists():
    print("## Orchestrator log\n\n```")
    print((args.orchestrator_dir / "orchestrator.log").read_text().rstrip())
    print("```")
