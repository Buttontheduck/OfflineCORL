"""Fixed advantage weights for every transition of a D4RL dataset, from a frozen critic
(EqM_RL.md, "Planned next experiments", section 2). Both actors of the 2 x 2 frozen-weights
study train on the same saved array, so they see identical targets.

    w(s, a) = min(exp(A(s, a) / T), exp_adv_max),   A = min(Q1, Q2)(s, a) - V(s)

--source iql    Q and V of a CORL IQL run (algorithms/offline/iql.py): TwinQ "qf" and
                ValueFunction "vf". The online Q is used; iql.py doesn't save its target Q.
--source aware  An Aware run, frozen as a whole (both critics and the EqM field). V(s) as in
                training (agents.py, _offline_weights): min(Q1, Q2) at the run's own sampler's
                clipped actions, averaged over --n_samples samples (training used 1) with a fixed
                seed, so V(s) isn't noisy.
--match_share F Instead of --temperature, choose T so that the weights' effective batch share
                (mean^2 / (mean^2 + var) within batches of 256, averaged; 1 = BC) equals that of
                the weights in file F (the matched-concentration refit); --target_share gives the
                share directly. With the cap, the share is not monotone in T (very small T puts
                weights at the cap or near 0, middle T spreads them, large T makes them all ~1),
                so T is found by scanning a log grid over [1e-3, 1e3] for every crossing, refining
                each, and taking the one closest to --temperature in log T. All crossings are kept
                in `meta`. If no T reaches the target within --match_tol, nothing is saved.

Rows follow d4rl.qlearning_dataset order. Output: an .npz with weights, adv, q, v (float32) and
`meta` (JSON: source, run, checkpoint, env, T, statistics). Run in the hawk env:

    python make_weights.py --source iql --run_dir <IQL run dir> --out <dir>/iql_seed1.npz
    python make_weights.py --source aware --run_dir <Aware run dir> --checkpoint checkpoint_310000.pt \\
        --out <dir>/aware_seed1.npz
    python make_weights.py --source aware --run_dir ... --checkpoint ... \\
        --match_share <dir>/iql_seed1.npz --out <dir>/aware_matched_seed1.npz
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--source", required=True, choices=["iql", "aware"])
ap.add_argument("--run_dir", type=Path, required=True, help="checkpoint folder of the run (has config.yaml)")
ap.add_argument("--checkpoint", help="file name, default: the latest checkpoint_*.pt")
ap.add_argument("--out", type=Path, required=True, help="output .npz")
ap.add_argument("--temperature", type=float, default=1.0, help="T: w = exp(A / T)")
ap.add_argument("--exp_adv_max", type=float, default=100.0, help="weight cap")
ap.add_argument("--match_share", type=Path, help="choose T to match this weight file's effective batch share")
ap.add_argument("--target_share", type=float, help="choose T to reach this effective batch share")
ap.add_argument("--match_tol", type=float, default=0.005, help="largest accepted |achieved - target| share")
ap.add_argument("--n_samples", type=int, default=32, help="EqM samples per state for V(s) (aware)")
ap.add_argument("--seed", type=int, default=0, help="sampler seed for V(s) (aware)")
ap.add_argument("--chunk", type=int, default=8192, help="states per GPU batch")
args = ap.parse_args()
t0 = time.time()
dev = "cuda"
sys.path.insert(0, str(HERE.parent / "diagnostics"))
from common import latest_checkpoint  # noqa: E402

ckpt = args.run_dir / args.checkpoint if args.checkpoint else latest_checkpoint(args.run_dir)
cfg = yaml.safe_load(open(args.run_dir / "config.yaml"))

# ---------------------------------------------------------------- Q(s, a) and V(s) for every row
if args.source == "aware":
    from common import Policy  # noqa: E402

    pol = Policy(args.run_dir, ckpt.name, load_critic=True)
    env_name, data = pol.cfg["env_name"], pol.data
    S_all = pol.states(data["observations"])
    A_all = torch.tensor(data["actions"], dtype=torch.float32, device=dev)
    K = pol.cfg["actor"]["num_step"]
    torch.manual_seed(args.seed)
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    q_all, v_all = [], []
    with torch.no_grad():
        for b in range(0, len(S_all), args.chunk):
            s, a = S_all[b:b + args.chunk], A_all[b:b + args.chunk]
            q_all.append(pol.q(s, a))
            s_rep = s.repeat(args.n_samples, 1)                       # (n * B, state_dim)
            x0 = torch.randn(len(s_rep), pol.action_dim, device=dev, generator=gen)
            a_pi = pol.sample(s_rep, K, x0=x0)[0].clamp(pol.low, pol.high)
            v_all.append(pol.q(s_rep, a_pi).view(args.n_samples, -1).mean(0))
    run_info = dict(n_samples=args.n_samples, seed=args.seed, K=K)
else:
    sys.path.insert(0, str(HERE.parent.parent / "offline"))
    import iql as corl_iql  # noqa: E402  (CORL's algorithms/offline/iql.py)
    from offline_data import load_d4rl  # noqa: E402

    env_name = cfg["env"]
    assert cfg.get("normalize", True), "the IQL run must normalize states, like Aware"
    _, data, mean, std, _ = load_d4rl(env_name)
    sd, ad = data["observations"].shape[1], data["actions"].shape[1]
    S_all = torch.tensor((data["observations"] - mean) / std, dtype=torch.float32, device=dev)
    A_all = torch.tensor(data["actions"], dtype=torch.float32, device=dev)
    state = torch.load(ckpt, map_location=dev)
    qf, vf = corl_iql.TwinQ(sd, ad).to(dev).eval(), corl_iql.ValueFunction(sd).to(dev).eval()
    qf.load_state_dict(state["qf"])
    vf.load_state_dict(state["vf"])
    q_all, v_all = [], []
    with torch.no_grad():
        for b in range(0, len(S_all), args.chunk):
            q_all.append(qf(S_all[b:b + args.chunk], A_all[b:b + args.chunk]))
            v_all.append(vf(S_all[b:b + args.chunk]))
    run_info = dict(iql_tau=cfg.get("iql_tau"), iql_beta=cfg.get("beta"))

q, v = torch.cat(q_all), torch.cat(v_all)
adv = q - v
if not torch.isfinite(adv).all():
    sys.exit(f"non-finite Q or V for {int((~torch.isfinite(adv)).sum())} rows of {ckpt}. Nothing saved.")


# ---------------------------------------------------------------- weights and their statistics
def weights_at(T: float) -> torch.Tensor:
    return torch.clamp_max(torch.exp(adv / T), args.exp_adv_max)


batch_idx = torch.randint(0, len(adv), (1000, 256), generator=torch.Generator(device=dev).manual_seed(0), device=dev)


def batch_share(w: torch.Tensor) -> float:
    """Effective batch share as logged in training: mean^2 / (mean^2 + var) within each batch."""
    wb = w[batch_idx]
    m, var = wb.mean(1), wb.var(1, unbiased=False)
    return float((m ** 2 / (m ** 2 + var)).mean())


def stats(w: torch.Tensor) -> dict:
    return dict(batch_share=batch_share(w), dataset_share=float(w.mean() ** 2 / (w.mean() ** 2 + w.var(unbiased=False))),
                cap_share=float((w >= args.exp_adv_max).float().mean()), near_zero_share=float((w < 0.01).float().mean()),
                median=float(w.median()), mean=float(w.mean()), adv_mean=float(adv.mean()), adv_std=float(adv.std()),
                positive_adv_share=float((adv > 0).float().mean()))


def share_crossings(target: float) -> tuple:
    """Every T in [1e-3, 1e3] where the effective batch share equals target: a log grid finds
    the brackets with a sign change, bisection on log T refines each. -> (crossings, grid, shares)"""
    grid = np.logspace(-3, 3, 241)
    g = np.array([batch_share(weights_at(float(t))) for t in grid]) - target
    out = []
    for i in range(len(grid)):
        if g[i] == 0:
            out.append(float(grid[i]))
        elif i + 1 < len(grid) and g[i] * g[i + 1] < 0:
            lo_T, hi_T, f_lo = float(grid[i]), float(grid[i + 1]), g[i]
            for _ in range(50):
                mid = float(np.sqrt(lo_T * hi_T))
                f_mid = batch_share(weights_at(mid)) - target
                if f_mid * f_lo > 0:
                    lo_T, f_lo = mid, f_mid
                else:
                    hi_T = mid
            out.append(float(np.sqrt(lo_T * hi_T)))
    return out, grid, g + target


T = args.temperature
target, crossings = None, []
if args.match_share and args.target_share is not None:
    sys.exit("give either --match_share or --target_share")
if args.match_share:
    target = json.loads(str(np.load(args.match_share)["meta"]))["stats"]["batch_share"]
elif args.target_share is not None:
    target = args.target_share
if target is not None:
    crossings, grid, shares = share_crossings(target)
    if not crossings:
        sys.exit(f"target share {target:.4f} is not reachable: on T in [1e-3, 1e3] the share ranges "
                 f"from {shares.min():.4f} to {shares.max():.4f}. Nothing saved.")
    T = min(crossings, key=lambda c: abs(np.log(c / args.temperature)))

w = weights_at(T)
st = stats(w)
if target is not None and not abs(st["batch_share"] - target) <= args.match_tol:  # also fails on NaN
    sys.exit(f"match failed: T = {T:.4g} gives share {st['batch_share']:.4f}, target {target:.4f} "
             f"(tolerance {args.match_tol}). Nothing saved.")
meta = dict(source=args.source, run=args.run_dir.name, run_dir=str(args.run_dir), checkpoint=ckpt.name,
            env_name=env_name, rows=len(w), temperature=T, exp_adv_max=args.exp_adv_max,
            matched_to=str(args.match_share) if args.match_share else None, matched_target_share=target,
            matched_crossings=[dict(temperature=c, batch_share=batch_share(weights_at(c))) for c in crossings],
            stats=st, **run_info)
args.out.parent.mkdir(parents=True, exist_ok=True)
np.savez(args.out, weights=w.cpu().numpy().astype(np.float32), adv=adv.cpu().numpy().astype(np.float32),
         q=q.cpu().numpy().astype(np.float32), v=v.cpu().numpy().astype(np.float32), meta=json.dumps(meta))
print(f"{args.out.name}: {args.source} {args.run_dir.name}/{ckpt.name}, {len(w)} rows, T = {T:.4g}, "
      f"{time.time() - t0:.0f} s")
if target is not None:
    print(f"  target share {target:.4f}; crossings at T = " + ", ".join(f"{c:.4g}" for c in crossings)
          + f"; chose the one closest to T = {args.temperature:g}")
print(f"  effective batch share {st['batch_share']:.3f} | at cap {st['cap_share']:.3f} | "
      f"near zero {st['near_zero_share']:.3f} | median {st['median']:.3f} | A mean {st['adv_mean']:.2f} "
      f"std {st['adv_std']:.2f} | A > 0 {st['positive_adv_share']:.3f}")
