"""Summary of Experiment 20 (IEQM with expectile 0.7 and 0.8) together with the baseline (Experiment 19, expectile 0.9,
seeds 1-3, and its two extra seeds 4 and 5), as Markdown.

Reads the 'RESULT {json}' lines of Aware/train_ieqm.py from the stdout.log files of the newest sweep folders
  sweep_runs/exp19_ogbench_ieqm_2*            baseline, seeds 1-3 (these lines have no expectile field: it is 0.9)
  sweep_runs/exp19_ogbench_ieqm_seeds45_2*    baseline, seeds 4 and 5
  sweep_runs/exp20_ogbench_ieqm_expectile_2*  expectile 0.7 and 0.8, seeds 1-5
A folder that does not exist yet is skipped, and runs that are not finished are shown with the evals they have.
Success in %, 50 episodes per eval. Score = the mean of a seed's last three evals (800k, 900k and 1M when finished).

    python diagnostics/summarize_exp20.py > sweep_runs/exp20_<date-time>/summary.md
"""

import glob
import json
from pathlib import Path

import numpy as np

EQM_RL = Path(__file__).resolve().parents[1]
PATTERNS = ["exp19_ogbench_ieqm_2*", "exp19_ogbench_ieqm_seeds45_2*", "exp20_ogbench_ieqm_expectile_2*"]
PLATEAU = [300000, 400000, 500000, 600000, 700000]
FINAL = [800000, 900000, 1000000]

res, folders = {}, []                       # (expectile, actor, seed, step) -> RESULT dict
for pat in PATTERNS:
    found = sorted(glob.glob(str(EQM_RL / "sweep_runs" / pat)))
    if not found:
        continue
    folders.append(Path(found[-1]).name)
    for log in sorted(Path(found[-1]).glob("*/stdout.log")):
        for line in log.read_text(errors="replace").splitlines():
            if line.startswith("RESULT "):
                r = json.loads(line[7:])
                res[(r.get("expectile", 0.9), r["actor"], r["seed"], r["step"])] = r
exps = sorted({k[0] for k in res})
steps = sorted({k[3] for k in res})


def seeds_of(e, actor):
    return sorted({k[2] for k in res if k[0] == e and k[1] == actor})


def mean_at(e, actor, seed, which):
    v = [res[(e, actor, seed, s)]["success"] for s in which if (e, actor, seed, s) in res]
    return float(np.mean(v)) if len(v) == len(which) else None


def ms(v, fmt="{:.1f}"):
    v = [x for x in v if x is not None]
    if not v:
        return "–"
    return (fmt + " ± " + fmt).format(np.mean(v), np.std(v)) if len(v) > 1 else fmt.format(v[0])


print("# Experiment 20: IEQM with expectile 0.7 and 0.8, against the baseline 0.9 (humanoidmaze-medium task 1)\n")
print("Folders read: " + ", ".join(f"`{f}`" for f in folders) + ". Only the expectile differs between the settings. "
      "Success in %, 50 episodes per eval, best of 32 actor samples at K = 20.\n")
print("- **Score** = mean of the evals at 800k, 900k and 1M (FQL's protocol), then mean ± std over the seeds.\n"
      "- **300k–700k** = mean of the 5 evals from 300k to 700k, then mean ± std over the seeds (in Experiment 19 "
      "this was the stretch where all seeds worked).\n"
      "- **Fell** = seeds whose score is below half of their own 300k–700k mean.\n"
      "- A seed counts only where it has all the evals of that column; unfinished runs show as –.\n")
for actor, title in (("trained", "The EqM actor trained in the run"),
                     ("frozen", "Experiment 14's frozen EqM-BC actor (seeds 1–3 only)")):
    if not any(k[1] == actor for k in res):
        continue
    print(f"## {title}\n")
    print("| expectile | seeds | score | score per seed | 300k–700k | fell |")
    print("|---|---|---|---|---|---|")
    for e in exps:
        sd = seeds_of(e, actor)
        sc = [mean_at(e, actor, s, FINAL) for s in sd]
        pl = [mean_at(e, actor, s, PLATEAU) for s in sd]
        fell = [s for s, a, b in zip(sd, sc, pl) if a is not None and b is not None and a < 0.5 * b]
        print(f"| {e} | {', '.join(map(str, sd))} | **{ms(sc)}** | "
              + " / ".join("–" if x is None else f"{x:.1f}" for x in sc) + f" | {ms(pl)} | "
              + (", ".join(f"seed {s}" for s in fell) if fell else "none") + " |")
    print()

print("## Success at every eval, the trained actor, per seed\n")
for e in exps:
    sd = seeds_of(e, "trained")
    print(f"Expectile {e}:\n")
    print("| step | " + " | ".join(f"seed {s}" for s in sd) + " | mean |")
    print("|---|" + "---|" * (len(sd) + 1))
    for st in steps:
        v = [res[(e, "trained", s, st)]["success"] if (e, "trained", s, st) in res else None for s in sd]
        if all(x is None for x in v):
            continue
        print(f"| {st // 1000}k | " + " | ".join("–" if x is None else f"{x:.0f}" for x in v)
              + f" | {np.mean([x for x in v if x is not None]):.0f} |")
    print()

print("## The critic at the last eval of each seed (mean ± std over seeds)\n")
print("**v_data** = V on dataset rows (the floor is −200; a lower expectile gives lower values). **q_gain** = mean "
      "over all steps of Q(chosen) − mean Q(the 32 candidates): how far apart the critic puts the candidates.\n")
print("| expectile | last eval at | v_data | q_loss | v_loss | q_gain | episode length |")
print("|---|---|---|---|---|---|---|")
for e in exps:
    last = {s: max(k[3] for k in res if k[:3] == (e, "trained", s)) for s in seeds_of(e, "trained")}
    col = lambda k, fmt="{:.2f}": ms([res[(e, "trained", s, st)][k] for s, st in last.items()], fmt)  # noqa: E731
    at = sorted(set(last.values()))
    print(f"| {e} | {' / '.join(f'{a // 1000}k' for a in at)} | {col('v_data', '{:.1f}')} | {col('q_loss')} | "
          f"{col('v_loss', '{:.3f}')} | {col('q_gain', '{:.3f}')} | {col('episode_length', '{:.0f}')} |")
