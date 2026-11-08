"""Summary of Experiment 12 (QIPO phase 3: EqM-BC actors fine-tuned with the frozen Experiment 11
critics), as Markdown.

Reads the runs' local wandb files in Aware/wandb (finetune_actor.py runs), without the network. A run is
found by its wandb group (Exp12_FT_<dataset>_beta<beta>), read from the command line stored in the run's
wandb-metadata.json. Success rates in %; means over seeds (± population std).

  step 0       the EqM-BC actor before fine-tuning (K = 10 one sample; best of 16 by the critic)
  K = 10       one sample per step from the fine-tuned (EMA) field, the main score; K = 20 likewise
  best of 16   the critic's pick among 16 samples of the fine-tuned field (every 100k)
Windows: mean of the evals in the --window steps up to each step in --steps.

    python diagnostics/summarize_exp12.py > summary.md
    python diagnostics/summarize_exp12.py --prefix Exp12b_FT --title "Experiment 12b: ... support_K = 20" > summary.md
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np

EQM_RL = Path(__file__).resolve().parents[1]
ROOT = EQM_RL / "Aware" / "wandb"
DATASETS = ["antmaze-medium-diverse-v2", "antmaze-umaze-v2"]
BETAS = ["3", "10", "0"]
# Experiment 11 at 1M (EqM-BC + critic, best of 16, mean of the last 300k), CORL IQL, QIPO-OT (paper, best eval)
REF = {"antmaze-medium-diverse-v2": ("78.8", "73.8", "86.4"), "antmaze-umaze-v2": ("94.6", "77.0", "93.6")}
KEYS = ("eval/", "ft/", "train/actor_loss")
MAIN, K20, BEST = "eval/normalized_score", "eval/normalized_score_K20", "eval/normalized_score_best16"

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--steps", default="100000,200000,400000", help="score windows end here")
ap.add_argument("--window", type=int, default=60000)
ap.add_argument("--seeds", default="1,2,3")
ap.add_argument("--prefix", default="Exp12_FT", help="wandb group prefix: Exp12_FT (Exp 12), Exp12b_FT (support_K = 20)")
ap.add_argument("--title", default="Experiment 12: QIPO phase 3, EqM-BC fine-tuned with the frozen Experiment 11 "
                                   "critics (3 seeds)")
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
    """{key: (n, 2) array of (step, value)} for the keys starting with one of KEYS."""
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
                if k.startswith(KEYS):
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
    if g.startswith(f"{args.prefix}_") and a.get("seed", "").isdigit():
        runs[(g, int(a["seed"]))] = d
data = {k: read(d) for k, d in runs.items()}


def series(ds, beta, s, key):
    return data.get((f"{args.prefix}_{ds}_beta{beta}", s), {}).get(key, np.zeros((0, 2)))


def window(e, lo, hi):
    sel = e[(e[:, 0] > lo) & (e[:, 0] <= hi), 1] if len(e) else []
    return float(np.mean(sel)) if len(sel) else None


def at(e, step):
    sel = e[e[:, 0] == step, 1] if len(e) else []
    return float(sel[0]) if len(sel) else None


def last(e):
    return float(e[-1, 1]) if len(e) else None


def ms(v, fmt="{:.1f}"):
    v = [x for x in v if x is not None]
    if not v:
        return "–"
    return (fmt + " ± " + fmt).format(np.mean(v), np.std(v)) if len(v) > 1 else fmt.format(v[0])


short = lambda d: d.replace("antmaze-", "").replace("-v2", "")  # noqa: E731
print(f"# {args.title}\n")
print("Last eval step per run: " + "; ".join(
    f"{short(d)} β={b}: " + "/".join(f"{int(e[-1, 0]) // 1000}k" if len(e := series(d, b, s, MAIN)) else "–"
                                       for s in seeds) for d in DATASETS for b in BETAS) + "\n")
print(f"Success rate in %, 100 episodes per eval. Fine-tuned columns: mean of the evals in the {args.window // 1000}k "
      f"steps up to that step (every 20k; best of 16 every 100k). Step 0 = the EqM-BC actor before fine-tuning.\n")
print("| Dataset | β | step 0: K = 10 | step 0: best of 16 | " + " | ".join(f"K = 10, {s // 1000}k" for s in steps)
      + f" | K = 20, {steps[-1] // 1000}k | best of 16, {steps[-1] // 1000}k | best single K = 10 eval (per seed) "
        "| Exp 11 best of 16 / CORL IQL / QIPO-OT |")
print("|---|---|---|---|" + "---|" * (len(steps) + 4))
for d in DATASETS:
    for b in BETAS:
        main = [series(d, b, s, MAIN) for s in seeds]
        cells = [ms([window(e, st - args.window, st) for e in main]) for st in steps]
        k20 = ms([window(series(d, b, s, K20), steps[-1] - args.window, steps[-1]) for s in seeds])
        best = ms([window(series(d, b, s, BEST), steps[-1] - args.window, steps[-1]) for s in seeds])
        top = " / ".join(f"{e[:, 1].max():.0f}" if len(e) else "–" for e in main)
        print(f"| {d} | {b} | {ms([at(e, 0) for e in main])} | {ms([at(series(d, b, s, BEST), 0) for s in seeds])} | "
              + " | ".join(cells) + f" | {k20} | {best} | {top} | {' / '.join(REF[d])} |")

print("\n## The support set at the first and the last renewal (mean ± std over seeds)\n")
print("- **Q samples − Q data:** the critic's Q at the actor's samples minus at the dataset action (how far the "
      "actor has moved up the critic). **w data:** softmax weight on the dataset action (1/17 ≈ 0.06 if uniform). "
      "**eff n:** exp(entropy) of the weights over the 17 candidates.")
print("- **spread:** std of the 16 samples per state, per dimension. **dist data:** distance (per √dim) from the "
      "dataset action to the nearest sample. **at bound:** share of sample coordinates beyond ±0.99.\n")
print("| Dataset | β | renewal | Q samples − Q data | w data | eff n | spread | dist data | at bound |")
print("|---|---|---|---|---|---|---|---|---|")
for d in DATASETS:
    for b in BETAS:
        for lab, pick in (("first", lambda e: float(e[0, 1]) if len(e) else None), ("last", last)):
            gap = []
            for s in seeds:
                qs, qd = pick(series(d, b, s, "ft/q_samples")), pick(series(d, b, s, "ft/q_data"))
                gap.append(None if qs is None or qd is None else qs - qd)
            v = lambda k, fmt="{:.2f}": ms([pick(series(d, b, s, k)) for s in seeds], fmt)  # noqa: E731
            print(f"| {d} | {b} | {lab} | {ms(gap, '{:.2f}')} | {v('ft/w_data')} | {v('ft/eff_n', '{:.1f}')} | "
                  f"{v('ft/spread')} | {v('ft/dist_data')} | {v('ft/at_bound')} |")

print("\nPer seed, K = 10 (seeds " + " / ".join(map(str, seeds)) + "):\n")
print("| Dataset | β | step 0 | " + " | ".join(f"{s // 1000}k" for s in steps) + " |")
print("|---|---|---|" + "---|" * len(steps))
for d in DATASETS:
    for b in BETAS:
        es = [series(d, b, s, MAIN) for s in seeds]
        print(f"| {d} | {b} | " + " / ".join("–" if (x := at(e, 0)) is None else f"{x:.0f}" for e in es) + " | "
              + " | ".join(" / ".join("–" if (x := window(e, st - args.window, st)) is None else f"{x:.1f}"
                                      for e in es) for st in steps) + " |")
