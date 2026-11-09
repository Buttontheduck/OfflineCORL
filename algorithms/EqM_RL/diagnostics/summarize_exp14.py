"""Summary of Experiment 14 (the QIPO pipeline on OGBench: EqM-BC -> in-support softmax critic -> QIPO fine-tuning),
as Markdown. Default tasks of humanoidmaze-medium (task1), antsoccer-arena (task4) and cube-double (task2).

Reads the runs' local wandb files in Aware/wandb, without the network; a run is found by its wandb group
(Exp14_BC_<env>, Exp14_Critic_<env>, Exp14_FT_<env>_beta<b>) from the command line in wandb-metadata.json.
Success rates in %, 50 episodes of the default task per eval. A phase's score is the mean of its last three evals
(as FQL / OGBench report). Means over seeds ± population std.

    python diagnostics/summarize_exp14.py > summary.md
"""

import glob
import json
from pathlib import Path

import numpy as np

EQM_RL = Path(__file__).resolve().parents[1]
ROOT = EQM_RL / "Aware" / "wandb"
ENVS = ["humanoidmaze-medium", "antsoccer-arena", "cube-double"]
TASK = {"humanoidmaze-medium": 1, "antsoccer-arena": 4, "cube-double": 2}
BETAS = [3, 10, 0]
SEEDS = [1, 2, 3]
# FQL (Park et al. 2025), Table 3, default task: BC / IQL / FAWAC / FQL
REF = {"humanoidmaze-medium": "1 / 32 / 6 / 19", "antsoccer-arena": "1 / 3 / 12 / 39", "cube-double": "0 / 1 / 2 / 36"}
MAIN = "eval/normalized_score"


def run_args(run_dir: str) -> dict:
    try:
        meta = json.loads(Path(run_dir, "files", "wandb-metadata.json").read_text())
    except (OSError, ValueError):
        return {}
    return dict(a.lstrip("-").split("=", 1) for a in meta.get("args", []) if "=" in a)


def read(run_dir: str) -> dict:
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
                if k.startswith(("eval/", "train/", "support/", "ft/")):
                    try:
                        out.setdefault(k, []).append((int(json.loads(h["_step"])), float(json.loads(v))))
                    except (TypeError, ValueError):
                        pass
    except Exception:   # a run still writing can end in a partial record
        pass
    return {k: np.array(v) for k, v in out.items()}


runs = {}
for d in sorted(glob.glob(f"{ROOT}/run-*")):   # sorted by date-time, so the newest wins
    a = run_args(d)
    g = a.get("group", "")
    if g.startswith("Exp14_") and a.get("seed", "").isdigit():
        runs[(g, int(a["seed"]))] = d
data = {k: read(d) for k, d in runs.items()}


def series(group, s, key):
    return data.get((group, s), {}).get(key, np.zeros((0, 2)))


def at(e, step):
    sel = e[e[:, 0] == step, 1] if len(e) else []
    return float(sel[0]) if len(sel) else None


def mean_at(e, steps):
    v = [at(e, s) for s in steps]
    return float(np.mean(v)) if v and all(x is not None for x in v) else None


def last(e):
    return float(e[-1, 1]) if len(e) else None


def first(e):
    return float(e[0, 1]) if len(e) else None


def ms(v, fmt="{:.1f}"):
    v = [x for x in v if x is not None]
    if not v:
        return "–"
    return (fmt + " ± " + fmt).format(np.mean(v), np.std(v)) if len(v) > 1 else fmt.format(v[0])


def seeds(v, fmt="{:.0f}"):
    return " / ".join("–" if x is None else fmt.format(x) for x in v)


print("# Experiment 14: QIPO pipeline on OGBench (EqM-BC → critic → fine-tuning), 3 seeds\n")
print(f"Runs found: {len(runs)} (9 BC, 9 critics, 27 fine-tunes expected). Success in %, 50 episodes of the default "
      "task per eval. FQL default-task references: BC / IQL / FAWAC / FQL.\n")

# ---------------------------------------------------------------- phase 1
print("## Phase 1: EqM behavior cloning (500k steps)\n")
print("Score = mean of the evals at 400k, 450k, 500k; one sample per step at K = 20 (as trained), 10 and 30.\n")
print("| Environment (task) | K = 10 | K = 20 | K = 30 | K = 10 per seed | references |")
print("|---|---|---|---|---|---|")
B = [400000, 450000, 500000]
for e in ENVS:
    g = f"Exp14_BC_{e}"
    k10 = [mean_at(series(g, s, "eval/normalized_score_K10"), B) for s in SEEDS]
    k20 = [mean_at(series(g, s, MAIN), B) for s in SEEDS]
    k30 = [mean_at(series(g, s, "eval/normalized_score_K30"), B) for s in SEEDS]
    print(f"| {e} (task{TASK[e]}) | {ms(k10)} | {ms(k20)} | {ms(k30)} | {seeds(k10)} | {REF[e]} |")

# ---------------------------------------------------------------- phase 2
print("\n## Phase 2: critics (1M steps); the critic's best of 16 BC samples (K = 10) at every step\n")
print("Score = mean of the evals at 800k, 900k, 1M. **q_data** = min(Q1, Q2) at dataset actions (must stay ≤ 0; "
      "rewards are ≤ 0), **max q_data** over training flags divergence. **eff n**, **w data**, **lift**: how the "
      "softmax target used the 17 candidates (last log).\n")
print("| Environment | 100k | 300k | 600k | **score** | per seed | q_data (last) | max q_data | eff n | w data | lift |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
C = [800000, 900000, 1000000]
for e in ENVS:
    g = f"Exp14_Critic_{e}"
    ev = [series(g, s, MAIN) for s in SEEDS]
    sc = [mean_at(x, C) for x in ev]
    qd = [series(g, s, "train/q_data") for s in SEEDS]
    cells = [ms([at(x, st) for x in ev]) for st in (100000, 300000, 600000)]
    print(f"| {e} | " + " | ".join(cells) + f" | **{ms(sc)}** | {seeds(sc)} | {ms([last(x) for x in qd])} | "
          f"{ms([float(x[:, 1].max()) if len(x) else None for x in qd])} | "
          + " | ".join(ms([last(series(g, s, k)) for s in SEEDS], "{:.2f}")
                       for k in ("support/eff_n", "support/w_data", "support/lift")) + " |")

# ---------------------------------------------------------------- phase 3
print("\n## Phase 3: QIPO fine-tuning (400k steps, support at K = 20, renewed every 40k)\n")
print("One sample per step at K = 10 (main) and K = 20; the critic's best of 16 every 100k. Step 0 = the BC actor; "
      "40k = the last eval before the first renewal. **Score** = mean of the K = 10 evals at 320k, 360k, 400k.\n")
print("| Environment | β | step 0 | 40k | 80k | 200k | **score (K = 10)** | per seed | K = 20 (last 3) | best of 16: 0 → 400k |")
print("|---|---|---|---|---|---|---|---|---|---|")
F = [320000, 360000, 400000]
for e in ENVS:
    for b in BETAS:
        g = f"Exp14_FT_{e}_beta{b}"
        ev = [series(g, s, MAIN) for s in SEEDS]
        sc = [mean_at(x, F) for x in ev]
        k20 = [mean_at(series(g, s, "eval/normalized_score_K20"), F) for s in SEEDS]
        bo = [series(g, s, "eval/normalized_score_best16") for s in SEEDS]
        cells = [ms([at(x, st) for x in ev]) for st in (0, 40000, 80000, 200000)]
        print(f"| {e} | {b} | " + " | ".join(cells) + f" | **{ms(sc)}** | {seeds(sc)} | {ms(k20)} | "
              f"{ms([at(x, 0) for x in bo])} → {ms([at(x, 400000) for x in bo])} |")

print("\n### The support set at the first and the last renewal (mean ± std over seeds)\n")
print("| Environment | β | renewal | Q samples − Q data | w data | eff n | spread | dist data | at bound |")
print("|---|---|---|---|---|---|---|---|---|")
for e in ENVS:
    for b in BETAS:
        g = f"Exp14_FT_{e}_beta{b}"
        for lab, pick in (("first", first), ("last", last)):
            gap = []
            for s in SEEDS:
                qs, qd = pick(series(g, s, "ft/q_samples")), pick(series(g, s, "ft/q_data"))
                gap.append(None if qs is None or qd is None else qs - qd)
            v = lambda k, fmt="{:.2f}": ms([pick(series(g, s, k)) for s in SEEDS], fmt)  # noqa: E731
            print(f"| {e} | {b} | {lab} | {ms(gap, '{:.2f}')} | {v('ft/w_data')} | {v('ft/eff_n', '{:.1f}')} | "
                  f"{v('ft/spread')} | {v('ft/dist_data')} | {v('ft/at_bound')} |")
