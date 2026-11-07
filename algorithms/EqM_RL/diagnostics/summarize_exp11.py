"""Summary of Experiment 11 (in-support softmax critics for the Experiment 10 BC actors), as Markdown.

Reads the runs' local wandb files in Aware/wandb (train_critic.py and fit_actor.py runs), without the
network. A run is found by its wandb group (Exp11_Critic_EqMBC_<dataset> / Exp11_Critic_GaussBC_<dataset>,
and Exp10_EqMBC_ / Exp10_GaussBC_ for the BC reference), read from the command line stored in the
run's wandb-metadata.json. Success rates in %; means over seeds (± population std).

  BC alone        Experiment 10, mean of the evals in 400k-500k (the checkpoint the critic uses):
                  EqM at K = 10 (the critic's candidate sampler), Gaussian with its mean action.
  best of 16      the critic's eval: at each step the highest-Q of 16 BC candidates; mean of the evals
                  in the --window steps up to each step in --steps.

    python diagnostics/summarize_exp11.py > summary.md
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np

EQM_RL = Path(__file__).resolve().parents[1]
ROOT = EQM_RL / "Aware" / "wandb"
ACTORS = {"EqM-BC": ("Exp11_Critic_EqMBC_", "Exp10_EqMBC_", "eval/normalized_score_K10"),
          "Gaussian BC": ("Exp11_Critic_GaussBC_", "Exp10_GaussBC_", "eval/normalized_score")}
DATASETS = ["antmaze-umaze-v2", "antmaze-umaze-diverse-v2", "antmaze-medium-diverse-v2", "antmaze-large-diverse-v2"]
# CORL README after 1M steps: IQL (in-sample critic, Gaussian AWR actor)
IQL = {"antmaze-umaze-v2": 77.0, "antmaze-umaze-diverse-v2": 54.3, "antmaze-medium-diverse-v2": 73.8,
       "antmaze-large-diverse-v2": 30.3}
KEYS = ("eval/", "train/q_data", "support/")

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--steps", default="300000,600000,1000000", help="score windows end here")
ap.add_argument("--window", type=int, default=300000)
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
    if g.startswith(("Exp11_Critic_", "Exp10_")) and a.get("seed", "").isdigit():
        runs[(g, int(a["seed"]))] = d
data = {k: read(d) for k, d in runs.items()}


def series(group, s, key):
    return data.get((group, s), {}).get(key, np.zeros((0, 2)))


def window(e, lo, hi):
    sel = e[(e[:, 0] > lo) & (e[:, 0] <= hi), 1] if len(e) else []
    return float(np.mean(sel)) if len(sel) else None


def last(e):
    return float(e[-1, 1]) if len(e) else None


def ms(v, fmt="{:.1f}"):
    v = [x for x in v if x is not None]
    if not v:
        return "–"
    return (fmt + " ± " + fmt).format(np.mean(v), np.std(v)) if len(v) > 1 else fmt.format(v[0])


print("# Experiment 11: in-support softmax critics for the Experiment 10 BC actors (AntMaze, 3 seeds)\n")
print("Last eval step per critic: " + "; ".join(
    f"{lab} {d.replace('antmaze-', '').replace('-v2', '')}: " + "/".join(
        f"{int(e[-1, 0]) // 1000}k" if len(e := series(c + d, s, "eval/normalized_score")) else "–" for s in seeds)
    for d in DATASETS for lab, (c, _, _) in ACTORS.items()) + "\n")
print(f"Success rate in %. 'Best of 16': at each step the highest-Q of 16 BC candidates; mean of the evals "
      f"in the {args.window // 1000}k steps up to that step (100 episodes per eval). 'BC alone': Experiment 10, "
      f"400k-500k (EqM at K = 10, Gaussian mean).\n")
print("| Dataset | BC actor | BC alone | " + " | ".join(f"best of 16, {s // 1000}k" for s in steps)
      + " | best single eval (per seed) | CORL IQL |")
print("|---|---|---|" + "---|" * (len(steps) + 2))
for d in DATASETS:
    for lab, (c, b, bkey) in ACTORS.items():
        bc = ms([window(series(b + d, s, bkey), 400000, 500000) for s in seeds])
        es = [series(c + d, s, "eval/normalized_score") for s in seeds]
        cells = [ms([window(e, st - args.window, st) for e in es]) for st in steps]
        best = " / ".join(f"{e[:, 1].max():.0f}" if len(e) else "–" for e in es)
        print(f"| {d} | {lab} | {bc} | " + " | ".join(cells) + f" | {best} | {IQL[d]} |")

print("\n## Critic and support set (latest values, mean ± std over seeds)\n")
print("- **spread:** std of the 16 support actions per state, per dimension. **nn dist:** distance (per √dim) from the "
      "dataset next action to the nearest support action; in brackets, the same to a random other row's next action.")
print("- **Q data:** min(Q1, Q2) at dataset actions (Q ∈ [−100, 0]; −100·(1−0.99ⁿ) for a goal n steps away). "
      "**Q start:** the chosen candidate's Q at the eval start states.")
print("- **w data:** softmax weight on the dataset next action (1/17 ≈ 0.06 if all candidates were equal). "
      "**eff n:** exp(entropy) of the softmax over 17 candidates. **Q range:** max − min Q over the candidates. "
      "**lift:** the target's soft maximum minus Q at the dataset next action.\n")
print("| Dataset | BC actor | spread | nn dist (random) | Q data | Q start | w data | eff n | Q range | lift |")
print("|---|---|---|---|---|---|---|---|---|---|")
for d in DATASETS:
    for lab, (c, _, _) in ACTORS.items():
        g = c + d
        v = lambda k, fmt="{:.2f}": ms([last(series(g, s, k)) for s in seeds], fmt)  # noqa: E731
        print(f"| {d} | {lab} | {v('support/spread')} | {v('support/nn_dist_data')} ({v('support/nn_dist_random')}) | "
              f"{v('train/q_data', '{:.1f}')} | {v('eval/q_start', '{:.1f}')} | {v('support/w_data')} | "
              f"{v('support/eff_n', '{:.1f}')} | {v('support/q_range')} | {v('support/lift')} |")

print("\nPer seed, best of 16 (seeds " + " / ".join(map(str, seeds)) + "):\n")
print("| Dataset | BC actor | " + " | ".join(f"{s // 1000}k" for s in steps) + " |")
print("|---|---|" + "---|" * len(steps))
for d in DATASETS:
    for lab, (c, _, _) in ACTORS.items():
        es = [series(c + d, s, "eval/normalized_score") for s in seeds]
        print(f"| {d} | {lab} | " + " | ".join(" / ".join(
            "–" if (x := window(e, st - args.window, st)) is None else f"{x:.1f}" for e in es) for st in steps) + " |")
