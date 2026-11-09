"""Summary of Experiment 15 (eval-only best-of-N with the frozen Experiment 14 critics and EqM-BC actors), as Markdown.

Reads the 'RESULT {json}' lines of Aware/eval_best_of_n.py from the stdout.log files of the newest
sweep_runs/exp15_ogbench_bestofn_*/ (or the folder given as the first argument). Success in %, mean over seeds
± population std.

    python diagnostics/summarize_exp15.py > sweep_runs/exp15_ogbench_bestofn_<date-time>/summary.md
"""

import glob
import json
import sys
from pathlib import Path

import numpy as np

EQM_RL = Path(__file__).resolve().parents[1]
ENVS = ["humanoidmaze-medium", "antsoccer-arena", "cube-double"]
SEEDS = [1, 2, 3]
REF = {"humanoidmaze-medium": "1 / 32 / 6 / 19", "antsoccer-arena": "1 / 3 / 12 / 39", "cube-double": "0 / 1 / 2 / 36"}
# Experiment 14, 50 episodes: phase 1 one BC sample (mean of the last 3 evals) at K = 20 / 30;
# phase 3 step 0 = this eval at K = 20, N = 16 (seeds 1 / 2 / 3); phase 2 critic's own eval at K = 10, N = 16 (1M)
EXP14 = {"humanoidmaze-medium": ("0.4 / 0.7", "30 / 34 / 56", "0 / 0 / 0"),
         "antsoccer-arena": ("0.2 / 0.4", "12 / 4 / 4", "0 / 0 / 0"),
         "cube-double": ("0.0 / 0.0", "0 / 0 / 0", "0 / 0 / 0")}

sweep = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(sorted(glob.glob(str(EQM_RL / "sweep_runs" /
                                                                               "exp15_ogbench_bestofn_2*")))[-1])
res = {}
for log in sorted(sweep.glob("*/stdout.log")):
    for line in log.read_text(errors="replace").splitlines():
        if line.startswith("RESULT "):
            r = json.loads(line[7:])
            env = next(e for e in ENVS if r["env_name"].startswith(e))
            res[(env, r["seed"], r["K"], r["N"])] = r
cells = sorted({(k[2], k[3]) for k in res})


def ms(v, fmt="{:.1f}"):
    v = [x for x in v if x is not None]
    if not v:
        return "–"
    return (fmt + " ± " + fmt).format(np.mean(v), np.std(v)) if len(v) > 1 else fmt.format(v[0])


def get(env, s, K, N, key):
    r = res.get((env, s, K, N))
    return None if r is None else r[key]


eps = sorted({r["episodes"] for r in res.values()})
print("# Experiment 15: best-of-N with the frozen Experiment 14 critics and EqM-BC actors (no training)\n")
print(f"Folder: {sweep}. Results found: {len(res)} of {len(ENVS) * len(SEEDS) * len(cells)}. Episodes per cell: "
      f"{', '.join(map(str, eps))} (default task). At every step N samples of the BC actor at K sampler steps; the "
      "critic's min(Q1, Q2) picks one. Success in %, mean ± std over seeds 1-3.\n")
print("| Environment | " + " | ".join(f"K={K}, N={N}" for K, N in cells) + " | FQL refs (BC / IQL / FAWAC / FQL) |")
print("|---|" + "---|" * len(cells) + "---|")
for e in ENVS:
    print(f"| {e} | " + " | ".join(f"**{ms([get(e, s, K, N, 'success') for s in SEEDS])}**" for K, N in cells)
          + f" | {REF[e]} |")

print("\n### Per seed (1 / 2 / 3)\n")
print("| Environment | " + " | ".join(f"K={K}, N={N}" for K, N in cells) + " |")
print("|---|" + "---|" * len(cells))
for e in ENVS:
    print(f"| {e} | " + " | ".join(" / ".join("–" if (x := get(e, s, K, N, "success")) is None else f"{x:.0f}"
                                             for s in SEEDS) for K, N in cells) + " |")

print("\n### Experiment 14 for comparison (50 episodes)\n")
print("| Environment | one BC sample, K = 20 / 30 | best of 16 at K = 20 (phase-3 step 0) | best of 16 at K = 10 "
      "(critic's own eval, 1M) |")
print("|---|---|---|---|")
for e in ENVS:
    print(f"| {e} | {EXP14[e][0]} | {EXP14[e][1]} | {EXP14[e][2]} |")

print("\n### The critic's view (mean over seeds)\n")
print("**q_gain** = mean over all steps of Q(chosen) − mean Q(candidates): how far apart the critic puts the "
      "candidates. **q_start** = Q of the chosen action at the start states. **length** = mean episode length "
      "(shorter = the goal was reached earlier).\n")
print("| Environment | K | N | q_gain | q_start | length | return | seconds |")
print("|---|---|---|---|---|---|---|---|")
for e in ENVS:
    for K, N in cells:
        v = lambda k, fmt="{:.2f}": ms([get(e, s, K, N, k) for s in SEEDS], fmt)  # noqa: E731
        print(f"| {e} | {K} | {N} | {v('q_gain', '{:.3f}')} | {v('q_start', '{:.1f}')} | "
              f"{v('episode_length', '{:.0f}')} | {v('score', '{:.0f}')} | {v('seconds', '{:.0f}')} |")
