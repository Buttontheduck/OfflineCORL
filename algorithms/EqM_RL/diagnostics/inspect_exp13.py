"""Mid-run health check of Experiment 13 (Aware on OGBench): not only the evals, but whether the critic, the
advantage weights and the EqM actor are learning. Reads the local wandb files (Aware/wandb) of the Exp 13 runs.

Per environment and critic (mean over the seeds that reached each step; per seed where it matters):
  eval        success % (and mean episode length) at the evals so far
  critic      critic_loss (TD), q_mean = min(Q1, Q2) at dataset actions, v_mean = V(s) used in the advantage;
              rewards are <= 0, so Q > 0 means divergence; the reachable floor is r_min / (1 - gamma)
  weights     adv_std, weights_mean / median / max, clip_frac (share at the cap 100), and the effective batch share
              mean^2 / (mean^2 + var) (1 = all samples count equally (BC), small = a few samples carry the update)
  actor       actor_loss (the weighted EqM loss) and actor_loss / weights_mean (roughly the unweighted EqM loss)
  sampler     f_norm_final (field norm at the sampler's endpoints), f_norm_data (at dataset actions), spread
              (per-state std of 8 samples), clip_frac (sample coordinates outside the action box)

    python diagnostics/inspect_exp13.py
"""

import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "Aware" / "wandb"
ENVS = ["humanoidmaze-medium", "antsoccer-arena", "cube-double"]
FLOOR = {"humanoidmaze-medium": -1 / (1 - 0.995), "antsoccer-arena": -1 / (1 - 0.995), "cube-double": -2 / (1 - 0.99)}


def run_args(d):
    try:
        meta = json.loads(Path(d, "files", "wandb-metadata.json").read_text())
    except (OSError, ValueError):
        return {}
    return dict(a.lstrip("-").split("=", 1) for a in meta.get("args", []) if "=" in a)


def read(d):
    from wandb.proto import wandb_internal_pb2 as pb
    from wandb.sdk.internal import datastore
    out = {}
    files = glob.glob(d + "/*.wandb")
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
                if not k.startswith("_"):
                    try:
                        out.setdefault(k, []).append((int(json.loads(h["_step"])), float(json.loads(v))))
                    except (TypeError, ValueError):
                        pass
    except Exception:
        pass
    return {k: np.array(v) for k, v in out.items()}


runs = {}
for d in sorted(glob.glob(f"{ROOT}/run-*")):
    a = run_args(d)
    g = a.get("group", "")
    if g.startswith("Exp13_") and a.get("seed", "").isdigit():
        crit = a.get("agent.critic", "awac")
        env = next((e for e in ENVS if e in a.get("env_name", "")), None)
        runs[(env, crit, int(a["seed"]))] = read(d)


def val(h, k, step):
    """The value logged at the last step <= step (training keys are logged every 1000 steps)."""
    e = h.get(k)
    if e is None or not len(e):
        return None
    sel = e[e[:, 0] <= step]
    return float(sel[-1, 1]) if len(sel) else None


def share(h, step):
    m, v = val(h, "weighting/weights_mean", step), val(h, "weighting/weights_var", step)
    return None if m is None or v is None else m * m / (m * m + v)


def fmt(x, f="{:.3g}"):
    return "–" if x is None else f.format(x)


def mean(v):
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else None


last_step = {k: int(max(e[-1, 0] for e in h.values() if len(e))) for k, h in runs.items() if h}
print(f"runs found: {len(runs)}; last logged step per run: "
      + ", ".join(f"{e[:4]}-{c}-s{s}: {last_step.get((e, c, s), 0) // 1000}k" for e, c, s in sorted(runs)))
for env in ENVS:
    print(f"\n=================== {env}   (Q floor r_min/(1-gamma) = {FLOOR[env]:.0f})")
    for crit in ("awac", "iql"):
        hs = {s: runs[(env, crit, s)] for s in (1, 2, 3) if (env, crit, s) in runs}
        if not hs:
            continue
        top = min(last_step.get((env, crit, s), 0) for s in hs)
        steps = [s for s in (1000, 25000, 50000, 100000, 150000, 200000, 250000) if s <= max(top, 1000)]
        print(f"\n--- {crit} critic, seeds {sorted(hs)} (common progress {top // 1000}k)")
        print(f"{'step':>8} | {'crit_loss':>9} {'q_mean':>8} {'v_mean':>8} | {'adv_std':>7} {'w_mean':>7} {'w_med':>7} "
              f"{'w_max':>6} {'clip':>6} {'share':>6} | {'act_loss':>8} {'loss/w':>7}"
              + (f" | {'v_loss':>7}" if crit == "iql" else ""))
        for st in steps:
            g = lambda k: mean([val(h, k, st) for h in hs.values()])  # noqa: E731
            al, wm = g("train/actor_loss"), g("weighting/weights_mean")
            print(f"{st // 1000:>7}k | {fmt(g('train/critic_loss')):>9} {fmt(g('weighting/q_mean'), '{:.1f}'):>8} "
                  f"{fmt(g('weighting/v_mean'), '{:.1f}'):>8} | {fmt(g('weighting/adv_std')):>7} {fmt(wm):>7} "
                  f"{fmt(g('weighting/weights_median')):>7} {fmt(g('weighting/weights_max'), '{:.0f}'):>6} "
                  f"{fmt(g('weighting/weights_clip_frac')):>6} {fmt(mean([share(h, st) for h in hs.values()]), '{:.2f}'):>6} | "
                  f"{fmt(al):>8} {fmt(None if al is None or not wm else al / wm):>7}"
                  + (f" | {fmt(g('train/value_loss')):>7}" if crit == "iql" else ""))
        # evals and sampler stats, per seed
        print("  evals (success % per seed; mean episode length):")
        for st in sorted({int(x) for h in hs.values() for x in h.get("eval/normalized_score", np.zeros((0, 2)))[:, 0]}):
            sc = [val(h, "eval/normalized_score", st) if st <= last_step.get((env, crit, s), 0) else None for s, h in hs.items()]
            ln = mean([val(h, "eval/episode_length", st) for h in hs.values()])
            sm = {k: mean([val(h, f"sampler/{k}", st) for h in hs.values()]) for k in ("f_norm_final", "f_norm_data", "spread", "clip_frac")}
            print(f"    {st // 1000:>5}k  success {' / '.join(fmt(x, '{:.0f}') for x in sc):<14} length {fmt(ln, '{:.0f}'):>5} | "
                  f"sampler f_final {fmt(sm['f_norm_final'], '{:.2f}')}  f_data {fmt(sm['f_norm_data'], '{:.2f}')}  "
                  f"spread {fmt(sm['spread'], '{:.3f}')}  clip {fmt(sm['clip_frac'], '{:.3f}')}")
        qmax = [float(h["weighting/q_mean"][:, 1].max()) for h in hs.values() if "weighting/q_mean" in h]
        print(f"  max q_mean over training per seed: {' / '.join(f'{x:.1f}' for x in qmax)}  (> 0 = diverged)")
