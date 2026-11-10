"""Summary of Experiment 19 (IEQM: IFQL's critic with our EqM actor, Aware/train_ieqm.py), as Markdown.

Reads the 'RESULT {json}' lines (one per eval and actor) from the stdout.log files of the newest
sweep_runs/exp19_ogbench_ieqm_*/ (or the folder given as the first argument). No wandb needed.

Score as FQL / OGBench: success in %, the mean of the evals at the last three eval steps (800k, 900k and 1M when
the run is complete), then mean ± population std over the seeds. A run that has not reached 1M yet is shown with
the evals it has, and marked.

    python diagnostics/summarize_exp19.py > sweep_runs/exp19_<date-time>/summary.md
"""

import glob
import json
import sys
from pathlib import Path

import numpy as np

EQM_RL = Path(__file__).resolve().parents[1]
ACTORS = {"trained": "EqM actor trained in the run (4 × 512, same steps as the critic)",
          "frozen": "Experiment 14's frozen EqM-BC actor (4 × 256, 500k steps)"}
FINAL_STEPS = [800000, 900000, 1000000]

sweep = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(sorted(glob.glob(str(EQM_RL / "sweep_runs" /
                                                                               "exp19_ogbench_ieqm_2*")))[-1])
res = {}                                                    # (actor, seed, step) -> the RESULT dict
for log in sorted(sweep.glob("*/stdout.log")):
    for line in log.read_text(errors="replace").splitlines():
        if line.startswith("RESULT "):
            r = json.loads(line[7:])
            res[(r["actor"], r["seed"], r["step"])] = r
seeds = sorted({k[1] for k in res})
steps = sorted({k[2] for k in res})
actors = [a for a in ACTORS if any(k[0] == a for k in res)]


def ms(v, fmt="{:.1f}"):
    v = [x for x in v if x is not None]
    if not v:
        return "–"
    return (fmt + " ± " + fmt).format(np.mean(v), np.std(v)) if len(v) > 1 else fmt.format(v[0])


def score(actor, seed):
    """(mean success of the seed's last three evals, True if those are the evals at 800k, 900k and 1M)."""
    have = sorted(s for (a, sd, s) in res if a == actor and sd == seed)[-3:]
    if not have:
        return None, False
    return float(np.mean([res[(actor, seed, s)]["success"] for s in have])), have == FINAL_STEPS


print("# Experiment 19: IEQM, the IFQL recipe with our EqM actor (humanoidmaze-medium task 1)\n")
if not res:
    print(f"No RESULT lines found in {sweep}.")
    sys.exit(0)
any_r = next(iter(res.values()))
print(f"Folder: {sweep}. Seeds found: {seeds}. Every eval: {any_r['episodes']} episodes, the best of N = {any_r['N']} "
      f"actor samples at K = {any_r['K']} sampler steps, chosen by the trained critic's min(Q1, Q2). One critic per "
      "seed (IFQL's: expectile 0.9, 4 × 512 with LayerNorm, γ = 0.995) scores both actors.\n")
print("**Score** = success in %, the mean of the evals at 800k, 900k and 1M, then mean ± std over the seeds. "
      "References on the same task: IFQL 69 ± 19 (FQL paper, 8 seeds); Experiment 15, the same frozen actor with the "
      "QIPO critic, best of 32 at K = 20: 39 (3 seeds, 100 episodes at the final checkpoints).\n")
print("| Actor | Score | per seed (" + " / ".join(map(str, seeds)) + ") |")
print("|---|---|---|")
for a in actors:
    sc = {s: score(a, s) for s in seeds}
    flag = "" if all(ok for _, ok in sc.values()) else " (not all runs have reached 1M: the last evals they have)"
    print(f"| {ACTORS[a]} | **{ms([v for v, _ in sc.values()])}**{flag} | "
          + " / ".join("–" if v is None else f"{v:.1f}" for v, _ in sc.values()) + " |")

print("\n## Success over training (mean over the seeds that have the eval)\n")
print("| Actor | " + " | ".join(f"{s // 1000}k" for s in steps) + " |")
print("|---|" + "---|" * len(steps))
for a in actors:
    print(f"| {a} | " + " | ".join(ms([res[(a, sd, s)]["success"] for sd in seeds if (a, sd, s) in res], "{:.0f}")
                                     .split(" ±")[0] for s in steps) + " |")

print("\n## The critic and the actor at each eval (mean over seeds)\n")
print("**v_data, q_data** = V(s) and min(Q1, Q2)(s, a) on dataset rows (the floor of Q is −200). **q_loss, v_loss, "
      "actor_loss** = the three training losses. **q_start** = Q of the chosen action at the start states. **q_gain** "
      "= mean over all steps of Q(chosen) − mean Q(the 32 candidates): how far apart the critic puts the candidates. "
      "**length** = mean episode length (2000 = the goal was never reached).\n")
print("| step | v_data | q_data | q_loss | v_loss | actor_loss | "
      + " | ".join(f"q_start ({a}) | q_gain ({a}) | length ({a})" for a in actors) + " |")
print("|---|---|---|---|---|---|" + "---|---|---|" * len(actors))
for s in steps:
    col = lambda a, k, fmt="{:.2f}": ms([res[(a, sd, s)][k] for sd in seeds if (a, sd, s) in res], fmt)  # noqa: E731
    a0 = actors[0]
    print(f"| {s // 1000}k | {col(a0, 'v_data', '{:.1f}')} | {col(a0, 'q_data', '{:.1f}')} | {col(a0, 'q_loss')} | "
          f"{col(a0, 'v_loss', '{:.3f}')} | {col(a0, 'actor_loss', '{:.0f}')} | "
          + " | ".join(f"{col(a, 'q_start', '{:.1f}')} | {col(a, 'q_gain', '{:.3f}')} | "
                       f"{col(a, 'episode_length', '{:.0f}')}" for a in actors) + " |")
