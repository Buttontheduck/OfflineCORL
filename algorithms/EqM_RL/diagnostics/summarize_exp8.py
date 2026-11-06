"""Summary of Experiments 8 and 9 on D4RL Maze2D, as Markdown: Aware at T = 0.1 and 1 (Exp 9) and
T = 4 (Exp 8) vs CORL's Gaussian AWAC (Exp 8). Arms with no local runs show "–".

Reads the eval scores from the runs' local wandb files, without the network. A run is found by
its wandb group (Exp<n>_<algorithm>_<dataset>), read from the command line stored in the run's
wandb-metadata.json; Aware's runs are in Aware/wandb, AWAC's in algorithms/offline/wandb.
Scores are D4RL normalized scores; each cell is the mean of the evals in the `--window` steps up
to that step. Means over seeds (± population std), per seed in brackets.

    python diagnostics/summarize_exp8.py > summary.md
"""

import argparse
import glob
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

EQM_RL = Path(__file__).resolve().parents[1]
ALGOS = {   # label: (group prefix, wandb root, eval key, offset added to the logged step)
    "Aware, T = 0.1": ("Exp9_Aware-T0.1_", EQM_RL / "Aware" / "wandb", "eval/normalized_score", 0),
    "Aware, T = 1": ("Exp9_Aware-T1_", EQM_RL / "Aware" / "wandb", "eval/normalized_score", 0),
    "Aware, T = 4": ("Exp8_Aware-T4_", EQM_RL / "Aware" / "wandb", "eval/normalized_score", 0),
    # awac.py logs with Aware's keys, its evals at 20,000, 40,000, ... (updates done)
    "Gaussian AWAC": ("Exp8_AWAC_", EQM_RL.parent / "offline" / "wandb", "eval/normalized_score", 0),
}
DATASETS = ["maze2d-umaze-v1", "maze2d-medium-v1", "maze2d-large-v1"]
CORL = {"maze2d-umaze-v1": "0.4 / 65.7", "maze2d-medium-v1": "0.8 / 84.6", "maze2d-large-v1": "2.3 / 215.5"}

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--steps", default="100000,250000,500000,750000,1000000", help="score windows end here")
ap.add_argument("--window", type=int, default=100000)
ap.add_argument("--seeds", default="1,2,3")
args = ap.parse_args()
steps = [int(s) for s in args.steps.split(",")]
seeds = [int(s) for s in args.seeds.split(",")]


def run_args(run_dir: str) -> dict:
    """key=value pairs of the run's command line (hydra 'k=v' or pyrallis '--k=v')."""
    try:
        meta = json.loads(Path(run_dir, "files", "wandb-metadata.json").read_text())
    except (OSError, ValueError):
        return {}
    return dict(a.lstrip("-").split("=", 1) for a in meta.get("args", []) if "=" in a)


def read_evals(job) -> np.ndarray:
    run_dir, key, offset = job
    from wandb.proto import wandb_internal_pb2 as pb
    from wandb.sdk.internal import datastore
    files = glob.glob(run_dir + "/*.wandb")
    if not files:
        return np.zeros((0, 2))
    ds = datastore.DataStore()
    ds.open_for_scan(files[0])
    ev, kb = [], key.encode()
    try:
        while (raw := ds.scan_data()) is not None:
            if kb not in raw:   # most records are training steps; skip them unparsed
                continue
            rec = pb.Record()
            rec.ParseFromString(raw)
            if rec.WhichOneof("record_type") != "history":
                continue
            h = {it.key or "/".join(it.nested_key): it.value_json for it in rec.history.item}
            if key in h and "_step" in h:
                ev.append((int(json.loads(h["_step"])) + offset, float(json.loads(h[key]))))
    except Exception:   # a run still writing (or killed) can end in a partial record
        pass
    return np.array(ev) if ev else np.zeros((0, 2))


# (algo, dataset, seed) -> newest wandb run folder of that group and seed
runs = {}
for algo, (prefix, root, key, offset) in ALGOS.items():
    for d in sorted(glob.glob(f"{root}/run-*")):   # sorted by date-time, so the newest wins
        a = run_args(d)
        g = a.get("group", "")
        if g.startswith(prefix) and g[len(prefix):] in DATASETS and a.get("seed", "").isdigit():
            runs[(algo, g[len(prefix):], int(a["seed"]))] = (d, key, offset)
keys = list(runs)
with Pool(max(1, min(len(keys), 18))) as pool:
    evals = dict(zip(keys, pool.map(read_evals, [runs[k] for k in keys])))


def window(e, lo, hi):
    sel = e[(e[:, 0] > lo) & (e[:, 0] <= hi), 1] if len(e) else []
    return float(np.mean(sel)) if len(sel) else None


def ms(v):
    v = [x for x in v if x is not None]
    if not v:
        return "–"
    return f"{np.mean(v):.1f} ± {np.std(v):.1f}" if len(v) > 1 else f"{v[0]:.1f}"


def per_seed(v):
    return " / ".join("–" if x is None else f"{x:.1f}" for x in v)


print("# Experiments 8 and 9: Aware (T = 0.1, 1, 4) vs Gaussian AWAC on D4RL Maze2D (2-D actions, 3 seeds)\n")
missing = [f"{a} {d} seed {s}" for a in ALGOS for d in DATASETS for s in seeds if (a, d, s) not in evals]
if missing:
    print("Not found (no local wandb run): " + ", ".join(missing) + "\n")
print("Last eval step per run: " + "; ".join(
    f"{a} {d.split('-')[1]}: " + "/".join(f"{int(evals[(a, d, s)][-1, 0]) // 1000}k"
                                         if (a, d, s) in evals and len(evals[(a, d, s)]) else "–" for s in seeds)
    for d in DATASETS for a in ALGOS) + "\n")

print(f"## D4RL normalized score (mean of the evals in the {args.window // 1000}k steps up to each step)\n")
print("| Dataset | Method | " + " | ".join(f"{s // 1000}k" for s in steps) + " | last 100k | best single eval | CORL BC / AWAC |")
print("|---|---|" + "---|" * (len(steps) + 3))
for d in DATASETS:
    for a in ALGOS:
        es = [evals.get((a, d, s), np.zeros((0, 2))) for s in seeds]
        cells = [ms([window(e, st - args.window, st) for e in es]) for st in steps]
        last = [window(e, e[-1, 0] - args.window, e[-1, 0]) if len(e) else None for e in es]
        best = [float(e[:, 1].max()) if len(e) else None for e in es]
        print(f"| {d} | {a} | " + " | ".join(cells) + f" | {ms(last)} | {per_seed(best)} | {CORL[d]} |")

print("\nPer seed (seeds " + " / ".join(map(str, seeds)) + "):\n")
print("| Dataset | Method | " + " | ".join(f"{s // 1000}k" for s in steps) + " |")
print("|---|---|" + "---|" * len(steps))
for d in DATASETS:
    for a in ALGOS:
        es = [evals.get((a, d, s), np.zeros((0, 2))) for s in seeds]
        print(f"| {d} | {a} | " + " | ".join(per_seed([window(e, st - args.window, st) for e in es]) for st in steps) + " |")

print("\nCORL after 1M steps (README, last eval), BC / AWAC: umaze 0.4 / 65.7, medium 0.8 / 84.6, large 2.3 / 215.5. "
      "Maze2D episodes do not end at the goal, so scores above 100 are possible.")
