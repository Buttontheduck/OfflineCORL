"""Summary of Experiment 16 (three pulls toward the dataset actions for QIPO fine-tuning on OGBench), as Markdown.

Variants (beta 10, frozen Experiment 14 critics and BC actors): plain QIPO, data16 (the dataset action counts as 16
candidates), bc1 (+ plain EqM-BC loss, weight 1); early stopping = each variant's evals at the fixed steps 80k and
120k. Reads the runs' local wandb files in Aware/wandb (group Exp16_FT_<env>_<variant>), without the network.
Success in %, 50 episodes of the default task per eval, mean over seeds ± population std.

    python diagnostics/summarize_exp16.py > sweep_runs/exp16_<date-time>/summary.md
"""

import glob
import json
from pathlib import Path

import numpy as np

EQM_RL = Path(__file__).resolve().parents[1]
ROOT = EQM_RL / "Aware" / "wandb"
ENVS = ["humanoidmaze-medium", "antsoccer-arena"]
VARIANTS = [("plain", "plain QIPO"), ("data16", "data ×16"), ("bc1", "+ BC loss")]
SEEDS = [1, 2, 3]
REF = {"humanoidmaze-medium": "1 / 32 / 6 / 19", "antsoccer-arena": "1 / 3 / 12 / 39"}
# Experiment 15 (100 episodes): the BC actor + critic, best of N at K = 20, no fine-tuning
BEST_N = {"humanoidmaze-medium": "N=16: 32.7, N=64: 42.3", "antsoccer-arena": "N=16: 5.7, N=64: 4.3"}
METRICS = [("eval/normalized_score", "one sample, K = 10"), ("eval/normalized_score_K20", "one sample, K = 20"),
           ("eval/normalized_score_best16", "critic's best of 16 at K = 20")]
STEPS = [0, 40000, 80000, 120000, 200000, 400000]
LAST = [320000, 360000, 400000]


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
                if k.startswith(("eval/", "train/", "ft/")):
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
    if g.startswith("Exp16_FT_") and a.get("seed", "").isdigit():
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


def first(e):
    return float(e[0, 1]) if len(e) else None


def last(e):
    return float(e[-1, 1]) if len(e) else None


def ms(v, fmt="{:.1f}"):
    v = [x for x in v if x is not None]
    if not v:
        return "–"
    return (fmt + " ± " + fmt).format(np.mean(v), np.std(v)) if len(v) > 1 else fmt.format(v[0])


def peak(group, key):
    """The best eval step of the seed-mean curve (an oracle stop: picked on the test evals, so optimistic)."""
    curves = [series(group, s, key) for s in SEEDS]
    steps = sorted(set.intersection(*[set(c[:, 0].astype(int)) for c in curves if len(c)])) if any(
        len(c) for c in curves) else []
    if not steps:
        return "–"
    means = [(np.mean([at(c, st) for c in curves if len(c)]), st) for st in steps]
    m, st = max(means)
    return f"{m:.1f} @ {st // 1000}k"


print("# Experiment 16: pulling QIPO fine-tuning toward the dataset actions (OGBench, β = 10, 3 seeds)\n")
print(f"Runs found: {len(runs)} of 18. Frozen Experiment 14 critics and EqM-BC actors. **plain** = QIPO as Experiment "
      "14; **data ×16** = the dataset action counts as 16 candidates in the softmax; **+ BC loss** = plain EqM-BC loss "
      "on dataset actions added with weight 1. **Early stop** = the eval at a fixed step (80k or 120k, chosen "
      "before the runs); **last 3** = mean of the evals at 320k, 360k, 400k (Experiment 14's score); **peak** = best "
      "step of the seed-mean curve (picked on the test evals, so optimistic). Success in %, 50 episodes per eval, "
      "mean ± std over seeds. Step 0 = the BC actor.\n")
for key, title in METRICS:
    print(f"## {title}\n")
    print("| Environment | variant | " + " | ".join(f"{s // 1000}k" for s in STEPS) + " | **last 3** | peak | per seed "
          "(80k / 120k / last 3) |")
    print("|---|---|" + "---|" * (len(STEPS) + 3))
    for e in ENVS:
        for v, label in VARIANTS:
            g = f"Exp16_FT_{e}_{v}"
            cells = []
            for st in STEPS:
                c = ms([at(series(g, s, key), st) for s in SEEDS])
                cells.append(f"**{c}**" if st in (80000, 120000) else c)
            lst = [mean_at(series(g, s, key), LAST) for s in SEEDS]
            per = "; ".join(" / ".join("–" if x is None else f"{x:.0f}" for x in (
                at(series(g, s, key), 80000), at(series(g, s, key), 120000), lst[i])) for i, s in enumerate(SEEDS))
            print(f"| {e} | {label} | " + " | ".join(cells) + f" | **{ms(lst)}** | {peak(g, key)} | {per} |")
    print()

print("Reference: FQL default-task scores (BC / IQL / FAWAC / FQL): "
      + "; ".join(f"{e} {REF[e]}" for e in ENVS) + ". Experiment 15, BC actor + critic best of N at K = 20, no "
      "fine-tuning (100 episodes): " + "; ".join(f"{e} {BEST_N[e]}" for e in ENVS) + ".\n")

print("## Does the field still collapse? Support set at the first and the last renewal (mean ± std over seeds)\n")
print("**spread** = std over the 16 samples (BC actor at K = 20: about 0.15; plain QIPO ended at 0.01-0.025 in "
      "Experiment 14). **w data** = weight on the dataset action. **eff n** = exp(entropy) of the weights over the 17 "
      "candidates. **dist data** = distance from the dataset action to its nearest sample, per action dim.\n")
print("| Environment | variant | renewal | spread | w data | eff n | dist data | Q samples − Q data | BC loss (last) |")
print("|---|---|---|---|---|---|---|---|---|")
for e in ENVS:
    for v, label in VARIANTS:
        g = f"Exp16_FT_{e}_{v}"
        for lab, pick in (("first", first), ("last", last)):
            gap = []
            for s in SEEDS:
                qs, qd = pick(series(g, s, "ft/q_samples")), pick(series(g, s, "ft/q_data"))
                gap.append(None if qs is None or qd is None else qs - qd)
            f = lambda k, fmt="{:.3f}": ms([pick(series(g, s, k)) for s in SEEDS], fmt)  # noqa: E731
            bc = f("train/bc_loss", "{:.1f}") if lab == "last" else ""
            print(f"| {e} | {label} | {lab} | {f('ft/spread')} | {f('ft/w_data', '{:.2f}')} | "
                  f"{f('ft/eff_n', '{:.1f}')} | {f('ft/dist_data')} | {ms(gap, '{:.2f}')} | {bc} |")
