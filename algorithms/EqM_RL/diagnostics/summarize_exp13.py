"""Summary of Experiment 13 (Aware on OGBench single-task default tasks, AWAC vs IQL critic), as Markdown.

Reads the runs' local wandb files in Aware/wandb, without the network. A run is found by its wandb group
(Exp13_<critic>_<env>), read from the command line in the run's wandb-metadata.json (hydra overrides).

Score as FQL / OGBench: the success rate in % (eval/normalized_score, 50 episodes of the trained task per eval),
the mean of the evals at 800k, 900k and 1M steps. Means over seeds ± population std.

    python diagnostics/summarize_exp13.py > summary.md
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np

EQM_RL = Path(__file__).resolve().parents[1]
ROOT = EQM_RL / "Aware" / "wandb"
ENVS = ["humanoidmaze-medium", "antsoccer-arena", "cube-double"]
CRITICS = ["awac", "iql"]
# FQL (Park et al. 2025), Table 3, the default task of each environment: IQL / ReBRAC / FAWAC / FQL
REF = {"humanoidmaze-medium": "32 / 16 / 6 / 19", "antsoccer-arena": "3 / 0 / 12 / 39", "cube-double": "1 / 7 / 2 / 36"}
TASK = {"humanoidmaze-medium": 1, "antsoccer-arena": 4, "cube-double": 2}
MAIN = "eval/normalized_score"
DIAG = ["weighting/q_mean", "weighting/adv_std", "weighting/weights_clip_frac", "train/critic_loss",
        "sampler/f_norm_final", "sampler/spread", "sampler/clip_frac"]

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--seeds", default="1,2,3")
ap.add_argument("--final", default="800000,900000,1000000", help="the evals averaged into the score")
args = ap.parse_args()
seeds = [int(s) for s in args.seeds.split(",")]
final = [int(s) for s in args.final.split(",")]


def run_args(run_dir: str) -> dict:
    try:
        meta = json.loads(Path(run_dir, "files", "wandb-metadata.json").read_text())
    except (OSError, ValueError):
        return {}
    return dict(a.lstrip("-").split("=", 1) for a in meta.get("args", []) if "=" in a)


def read(run_dir: str) -> dict:
    """{key: (n, 2) array of (step, value)} for eval/, weighting/, train/ and sampler/ keys."""
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
                if k.startswith(("eval/", "weighting/", "train/", "sampler/")):
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
    if g.startswith("Exp13_") and a.get("seed", "").isdigit():
        runs[(g, int(a["seed"]))] = d
data = {k: read(d) for k, d in runs.items()}


def series(env, critic, s, key):
    return data.get((f"Exp13_{critic}_{env}", s), {}).get(key, np.zeros((0, 2)))


def at(e, step):
    sel = e[e[:, 0] == step, 1] if len(e) else []
    return float(sel[0]) if len(sel) else None


def score(e):
    v = [at(e, st) for st in final]
    return float(np.mean(v)) if all(x is not None for x in v) else None


def last(e):
    return float(e[-1, 1]) if len(e) else None


def ms(v, fmt="{:.1f}"):
    v = [x for x in v if x is not None]
    if not v:
        return "–"
    return (fmt + " ± " + fmt).format(np.mean(v), np.std(v)) if len(v) > 1 else fmt.format(v[0])


print("# Experiment 13: Aware on OGBench single-task, AWAC vs IQL critic (3 seeds)\n")
print("Last eval step per run: " + "; ".join(
    f"{e} {c}: " + "/".join(f"{int(x[-1, 0]) // 1000}k" if len(x := series(e, c, s, MAIN)) else "–" for s in seeds)
    for e in ENVS for c in CRITICS) + "\n")
print(f"Success rate in %, 50 episodes of the trained (default) task per eval. **Score** = mean of the evals at "
      f"{', '.join(f'{s // 1000}k' for s in final)} (FQL's protocol), mean ± std over seeds.\n")
print("| Environment (task) | Critic | Score | per seed | best eval (per seed) | FQL default task: IQL / ReBRAC / FAWAC / FQL |")
print("|---|---|---|---|---|---|")
for e in ENVS:
    for c in CRITICS:
        es = [series(e, c, s, MAIN) for s in seeds]
        sc = [score(x) for x in es]
        print(f"| {e} (task{TASK[e]}) | {c} | {ms(sc)} | " + " / ".join("–" if x is None else f"{x:.0f}" for x in sc)
              + " | " + " / ".join(f"{x[:, 1].max():.0f}" if len(x) else "–" for x in es) + f" | {REF[e]} |")

print("\n## Success over training (mean over seeds)\n")
steps = list(range(0, 1000001, 100000))
print("| Environment | Critic | " + " | ".join(f"{s // 1000}k" for s in steps) + " |")
print("|---|---|" + "---|" * len(steps))
for e in ENVS:
    for c in CRITICS:
        es = [series(e, c, s, MAIN) for s in seeds]
        cells = []
        for st in steps:
            v = [x for x in (at(z, st) for z in es) if x is not None]
            cells.append(f"{np.mean(v):.0f}" if v else "–")
        print(f"| {e} | {c} | " + " | ".join(cells) + " |")

print("\n## Training diagnostics at the last logged step (mean ± std over seeds)\n")
print("- **q_mean, adv_std, weights_clip_frac:** the critic's Q at dataset actions, the advantage spread, and the "
      "share of weights at the cap. **critic_loss:** TD loss. **f_norm_final, spread, clip_frac:** the sampler on "
      "256 dataset states (‖f‖ at the endpoints, per-state spread, share of coordinates beyond the action box).\n")
print("| Environment | Critic | " + " | ".join(k.split("/")[1] for k in DIAG) + " |")
print("|---|---|" + "---|" * len(DIAG))
for e in ENVS:
    for c in CRITICS:
        print(f"| {e} | {c} | " + " | ".join(ms([last(series(e, c, s, k)) for s in seeds], "{:.3g}") for k in DIAG) + " |")
