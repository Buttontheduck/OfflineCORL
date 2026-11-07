"""Summary of Experiment 10 (EqM-BC vs Gaussian BC on D4RL AntMaze), as Markdown.

Reads the eval scores from the runs' local wandb files in Aware/wandb (fit_actor.py runs), without the
network. A run is found by its wandb group (Exp10_EqMBC_<dataset> / Exp10_GaussBC_<dataset>), read
from the command line stored in the run's wandb-metadata.json. Scores are success rates in %
(D4RL normalized score); each cell is the mean of the evals in the `--window` steps up to that step.
Means over seeds (± population std).

    python diagnostics/summarize_exp10.py > summary.md
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np

EQM_RL = Path(__file__).resolve().parents[1]
ROOT = EQM_RL / "Aware" / "wandb"
ARMS = {  # label: (group prefix, eval key)
    "EqM-BC, K = 20 (as trained)": ("Exp10_EqMBC_", "eval/normalized_score"),
    "EqM-BC, K = 10": ("Exp10_EqMBC_", "eval/normalized_score_K10"),
    "EqM-BC, K = 30": ("Exp10_EqMBC_", "eval/normalized_score_K30"),
    "Gaussian BC (mean)": ("Exp10_GaussBC_", "eval/normalized_score"),
}
DATASETS = ["antmaze-umaze-v2", "antmaze-umaze-diverse-v2", "antmaze-medium-diverse-v2", "antmaze-large-diverse-v2"]
# CORL README after 1M steps (BC / AWAC / IQL), and Aware T = 4 with its own critic (Exp 7 at 400k-500k; Exp 6 for large-diverse)
REF = {"antmaze-umaze-v2": ("55.3 / 56.8 / 77.0", "66.5"), "antmaze-umaze-diverse-v2": ("47.3 / 54.8 / 54.3", "50.3"),
       "antmaze-medium-diverse-v2": ("0.8 / 0.0 / 73.8", "0.3"), "antmaze-large-diverse-v2": ("0.0 / 0.0 / 30.3", "0.0")}

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--steps", default="100000,250000,500000", help="score windows end here")
ap.add_argument("--window", type=int, default=100000)
ap.add_argument("--seeds", default="1,2,3")
args = ap.parse_args()
steps = [int(s) for s in args.steps.split(",")]
seeds = [int(s) for s in args.seeds.split(",")]


def run_args(run_dir: str) -> dict:
    try:
        meta = json.loads(Path(run_dir, "files", "wandb-metadata.json").read_text())
    except (OSError, ValueError):
        return {}
    return dict(a.lstrip("-").split("=", 1) for a in meta.get("args", []) if "=" in a)


def read(run_dir: str) -> dict:
    """{key: (n, 2) array of (step, value)} for every eval/* key."""
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
            if b"eval/" not in raw:
                continue
            rec = pb.Record()
            rec.ParseFromString(raw)
            if rec.WhichOneof("record_type") != "history":
                continue
            h = {it.key or "/".join(it.nested_key): it.value_json for it in rec.history.item}
            if "_step" not in h:
                continue
            for k, v in h.items():
                if k.startswith("eval/"):
                    out.setdefault(k, []).append((int(json.loads(h["_step"])), float(json.loads(v))))
    except Exception:   # a run still writing can end in a partial record
        pass
    return {k: np.array(v) for k, v in out.items()}


runs = {}
for d in sorted(glob.glob(f"{ROOT}/run-*")):   # sorted by date-time, so the newest wins
    a = run_args(d)
    g = a.get("group", "")
    if g.startswith("Exp10_") and a.get("seed", "").isdigit():
        runs[(g, int(a["seed"]))] = d
data = {k: read(d) for k, d in runs.items()}


def window(e, lo, hi):
    sel = e[(e[:, 0] > lo) & (e[:, 0] <= hi), 1] if len(e) else []
    return float(np.mean(sel)) if len(sel) else None


def ms(v):
    v = [x for x in v if x is not None]
    if not v:
        return "–"
    return f"{np.mean(v):.1f} ± {np.std(v):.1f}" if len(v) > 1 else f"{v[0]:.1f}"


def series(arm, ds, s):
    prefix, key = ARMS[arm]
    return data.get((prefix + ds, s), {}).get(key, np.zeros((0, 2)))


print("# Experiment 10: EqM-BC vs Gaussian BC on D4RL AntMaze (success rate in %, 3 seeds)\n")
print("Last eval step per run: " + "; ".join(
    f"{p.split('_')[1]} {d.replace('antmaze-', '').replace('-v2', '')}: " + "/".join(
        f"{int(series(a, d, s)[-1, 0]) // 1000}k" if len(series(a, d, s)) else "–" for s in seeds)
    for d in DATASETS for a, (p, _) in ARMS.items() if a.endswith("trained)") or a.startswith("Gaussian")) + "\n")
print(f"Each cell: mean of the evals in the {args.window // 1000}k steps up to that step (100 episodes per eval).\n")
print("| Dataset | Arm | " + " | ".join(f"{s // 1000}k" for s in steps) + " | best single eval (per seed) | CORL BC / AWAC / IQL | Aware T = 4 (Exp 6/7) |")
print("|---|---|" + "---|" * (len(steps) + 3))
for d in DATASETS:
    for a in ARMS:
        es = [series(a, d, s) for s in seeds]
        cells = [ms([window(e, st - args.window, st) for e in es]) for st in steps]
        best = " / ".join(f"{e[:, 1].max():.0f}" if len(e) else "–" for e in es)
        print(f"| {d} | {a} | " + " | ".join(cells) + f" | {best} | {REF[d][0]} | {REF[d][1]} |")
print("\nPer seed, last window (seeds " + " / ".join(map(str, seeds)) + "):\n")
print("| Dataset | Arm | " + " | ".join(f"{s // 1000}k" for s in steps) + " |")
print("|---|---|" + "---|" * len(steps))
for d in DATASETS:
    for a in ARMS:
        es = [series(a, d, s) for s in seeds]
        print(f"| {d} | {a} | " + " | ".join(" / ".join("–" if (x := window(e, st - args.window, st)) is None else f"{x:.1f}" for e in es) for st in steps) + " |")
