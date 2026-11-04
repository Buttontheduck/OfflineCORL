"""Evaluation-only action interventions on one trained Aware run (EqM_RL.md, "Planned next
experiments", section 1): does a small change to the executed actions reduce flips?

Treatments, all at the run's trained sampler setting (its K, eta and momentum):
  Aware  bound b (--bounds): one EqM sample, executed clipped to [-b, b].
  Aware  average n (--avg, n > 1): the mean of n independent samples, each clipped to [-1, 1]
         first; executed with b = 1. Tested separately from the inward bounds.
  AWAC   bound b (--bounds, with --awac): the Gaussian mean, clipped to [-b, b]. --awac can also be
         a Gaussian fit of Aware/fit_actor.py, and the main run an EqM fit of it.
Every treatment runs the same episodes (env seeds seed0, seed0 + 1, ...), all side by side as one
batch. Aware treatments share their sampler noise: at a given step, episode j's first sample
starts from the same noise in every treatment.

Per episode: return, flip step (body angle obs[1] beyond --flip_angle; -1 = no flip), mean speed
obs[8], mean speed before the flip, share of raw action dims outside [-1, 1] (sampler endpoints;
for AWAC its unclipped mean), share of executed dims at the bound (|a| >= 0.99 b), and the mean
action change |a_t - a_(t-1)|. Before flips: the same per-step quantities over the --window steps
before each flip, against the same steps of the episodes of that treatment that did not flip
(matched windows).

    # screening: 20 episodes per treatment, the training eval seeds
    python diagnostics/interventions.py <Aware run dir> diagnostics/out/interv/<name>.json \
        --awac <AWAC run dir of the same seed>
    # confirmation: new episodes, selected treatments only
    python diagnostics/interventions.py ... --n_eps 100 --seed0 1000 --only aware_b0.9_n1,awac_b0.9
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from common import AwacPolicy, Policy

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("run_dir", type=Path, help="checkpoint folder of one Aware run (has config.yaml)")
ap.add_argument("out", type=Path, help="output JSON")
ap.add_argument("--checkpoint", help="file name, default: the latest checkpoint_*.pt")
ap.add_argument("--awac", type=Path, help="checkpoint folder of the Gaussian reference on the same dataset: a CORL "
                "AWAC run or a fit_actor.py Gaussian fit (its treatments are named awac_*)")
ap.add_argument("--bounds", default="1.0,0.95,0.9", help="executed action bounds b")
ap.add_argument("--avg", default="4,16", help="numbers of samples averaged (with b = 1)")
ap.add_argument("--only", help="comma-separated treatment names to run (default: all)")
ap.add_argument("--n_eps", type=int, default=20, help="episodes per treatment")
ap.add_argument("--seed0", type=int, default=69, help="env seed of the first episode")
ap.add_argument("--max_steps", type=int, default=1000, help="episode length cap (1000 = full episodes)")
ap.add_argument("--window", type=int, default=50, help="steps before a flip in the pre-flip statistics")
ap.add_argument("--flip_angle", type=float, default=2.0, help="|body angle| in rad that counts as a flip")
ap.add_argument("--save_steps", action="store_true", help="also save the per-step arrays (.npz next to the JSON)")
args = ap.parse_args()
t0 = time.time()

pol = Policy(args.run_dir, args.checkpoint)
awac = AwacPolicy(args.awac, pol) if args.awac else None
dev, lo, hi, ad = pol.device, pol.low, pol.high, pol.action_dim
K = pol.cfg["actor"]["num_step"]

# ---------------------------------------------------------------- treatments
bounds = [float(b) for b in args.bounds.split(",")]
avgs = [int(n) for n in args.avg.split(",") if n and int(n) > 1]
treat = [dict(name=f"aware_b{b:g}_n1", policy="aware", bound=b, n=1) for b in bounds]
treat += [dict(name=f"aware_b1_n{n}", policy="aware", bound=1.0, n=n) for n in avgs]
if awac is not None:
    treat += [dict(name=f"awac_b{b:g}", policy="awac", bound=b, n=1) for b in bounds]
if args.only:
    keep = set(args.only.split(","))
    unknown = keep - {t["name"] for t in treat}
    if unknown:
        raise SystemExit(f"unknown treatments {sorted(unknown)}; available: {[t['name'] for t in treat]}")
    treat = [t for t in treat if t["name"] in keep]

n_eps, T_n = args.n_eps, len(treat)
E = T_n * n_eps
row_t = np.repeat(np.arange(T_n), n_eps)            # treatment of each env row
row_ep = np.tile(np.arange(n_eps), T_n)              # episode index of each env row
bound_row = torch.tensor([treat[i]["bound"] for i in row_t], dtype=torch.float32, device=dev)
n_max = max(t["n"] for t in treat)
groups = {}                                          # (policy, n) -> env rows
for i, t in enumerate(treat):
    groups.setdefault((t["policy"], t["n"]), []).extend(np.flatnonzero(row_t == i).tolist())
groups = {k: torch.tensor(v, device=dev) for k, v in groups.items()}
row_ep_t = torch.tensor(row_ep, device=dev)

# ---------------------------------------------------------------- rollouts
envs = [pol.make_env() for _ in range(E)]
obs = np.stack([pol.reset(e, int(args.seed0 + j)) for e, j in zip(envs, row_ep)]).astype(np.float64)
done = np.zeros(E, bool)
gen = torch.Generator(device=dev).manual_seed(0)
ANG, SPD, REW, OUT, ATB, DACT, ALIVE = [], [], [], [], [], [], []
ACT = []
prev_a = None
for t in range(args.max_steps):
    if done.all():
        break
    S = pol.states(obs)
    a = torch.zeros(E, ad, device=dev)
    out = torch.zeros(E, device=dev)
    # one noise draw per (episode, sample) per step, shared by all Aware treatments
    z = torch.randn(n_eps, n_max, ad, device=dev, generator=gen)
    with torch.no_grad():
        for (policy, n), rows in groups.items():
            if policy == "aware":
                x0 = z[row_ep_t[rows], :n].reshape(-1, ad)
                raw = pol.sample(S[rows].repeat_interleave(n, 0), K, x0=x0)[0].view(len(rows), n, ad)
                out[rows] = ((raw < lo) | (raw > hi)).float().mean((1, 2))
                a[rows] = raw.clamp(lo, hi).mean(1)
            else:
                raw = awac.mean(S[rows])
                out[rows] = ((raw < lo) | (raw > hi)).float().mean(1)
                a[rows] = raw
        b = bound_row[:, None]
        a = torch.maximum(torch.minimum(a, b), -b)
        atb = (a.abs() >= 0.99 * b).float().mean(1)
        dact = torch.zeros(E, device=dev) if prev_a is None else (a - prev_a).abs().mean(1)
    prev_a = a
    a_np, r = a.cpu().numpy(), np.zeros(E)
    ALIVE.append(~done)
    for i in np.flatnonzero(~done):
        obs[i], r[i], done[i] = pol.step(envs[i], a_np[i])
    ANG.append(obs[:, 1].copy())
    SPD.append(obs[:, 8].copy())
    REW.append(r)
    OUT.append(out.cpu().numpy())
    ATB.append(atb.cpu().numpy())
    DACT.append(dact.cpu().numpy())
    if args.save_steps:
        ACT.append(a_np.astype(np.float32))
del envs
ANG, SPD, REW, OUT, ATB, DACT, ALIVE = map(np.array, (ANG, SPD, REW, OUT, ATB, DACT, ALIVE))  # (T, E)
T_len = len(ANG)


# ---------------------------------------------------------------- per-episode statistics
def masked_mean(x, m, axis=0):
    s = m.sum(axis)
    return np.where(s > 0, (x * m).sum(axis) / np.maximum(s, 1), np.nan)


flipped_at = np.abs(ANG) > args.flip_angle
flip_step = np.where(flipped_at.any(0), flipped_at.argmax(0), -1)
alive = ALIVE.astype(float)
before = (np.arange(T_len)[:, None] < np.where(flip_step >= 0, flip_step, T_len)[None, :]) & ALIVE
per_ep = dict(
    returns=(REW * alive).sum(0), flip_step=flip_step,
    speed=masked_mean(SPD, alive), speed_before_flip=masked_mean(SPD, before.astype(float)),
    outside=masked_mean(OUT, alive), at_bound=masked_mean(ATB, alive), dact=masked_mean(DACT[1:], alive[1:]),
)
W = args.window
step_q = dict(outside=OUT, at_bound=ATB, dact=DACT, speed=SPD, abs_angle=np.abs(ANG))

res = dict(pol.info(), awac_run=args.awac.name if args.awac else None, awac_kind=awac.kind if awac else None, K=K,
           args={k: str(v) for k, v in vars(args).items()}, treatments={})
for i, tr in enumerate(treat):
    cols = np.flatnonzero(row_t == i)
    fs = flip_step[cols]
    ep = {k: v[cols].tolist() for k, v in per_ep.items()}
    summ = {k: float(np.nanmean(v[cols])) for k, v in per_ep.items() if k != "flip_step"}
    summ["flip_rate"] = float((fs >= 0).mean())
    # pre-flip windows vs the same steps of this treatment's episodes without a flip
    pre = {}
    flips = [(c, s) for c, s in zip(cols, fs) if s >= W]
    calm = cols[fs < 0]
    for name, Q in step_q.items():
        pre[name] = dict(
            flipped=float(np.mean([Q[s - W:s, c].mean() for c, s in flips])) if flips else None,
            matched_no_flip=float(np.mean([Q[s - W:s, calm].mean() for _, s in flips]))
            if flips and len(calm) else None)
    pre["n_flips_with_window"], pre["n_no_flip"] = len(flips), int(len(calm))
    res["treatments"][tr["name"]] = dict(tr, episodes=ep, summary=summ, pre_flip=pre)

res["minutes"] = (time.time() - t0) / 60
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(res))
if args.save_steps:
    np.savez_compressed(args.out.with_suffix(".npz"), angle=ANG.astype(np.float32), speed=SPD.astype(np.float32),
                        reward=REW.astype(np.float32), outside=OUT, at_bound=ATB, dact=DACT, alive=ALIVE,
                        actions=np.array(ACT), row_treatment=row_t, row_episode=row_ep,
                        treatments=np.array([t["name"] for t in treat]))

print(f"{pol.run_dir.name}  ({res['minutes']:.1f} min, {n_eps} episodes per treatment)")
print(f"{'treatment':<16}{'return':>9}{'flips':>7}{'speed':>7}{'outside':>9}{'at bound':>9}"
      f"{'  pre-flip at bound (flip / no flip)':>36}")
for name, r in res["treatments"].items():
    s, p = r["summary"], r["pre_flip"]["at_bound"]
    pf = f"{p['flipped']:.3f} / {p['matched_no_flip']:.3f}" if p["flipped"] is not None and p["matched_no_flip"] is not None else "-"
    print(f"{name:<16}{s['returns']:>9.0f}{s['flip_rate']:>7.0%}{s['speed']:>7.2f}{s['outside']:>9.3f}"
          f"{s['at_bound']:>9.3f}{pf:>36}")
