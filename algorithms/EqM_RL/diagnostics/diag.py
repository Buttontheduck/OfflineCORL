"""Low-dimensionality diagnostics for one trained Aware_Minari run (EqM_RL.md, Experiment 2).

A: env return vs the number of sampler steps K (same sampler as training and eval).
B: on random dataset states, many samples per state, tracked over K: field norm, spread across
   one state's samples, error to the dataset action, movement between K values, clipping.
   Plus references: the field at dataset / uniform actions and the dataset's conditional spread.

    python diagnostics/diag.py <run checkpoint dir> diagnostics/out/diag/<name>.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from common import Policy

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("run_dir", type=Path, help="checkpoint folder of one run (has config.yaml)")
ap.add_argument("out", type=Path, help="output JSON")
ap.add_argument("--checkpoint", help="file name, default: the latest checkpoint_*.pt")
ap.add_argument("--n_eps", type=int, default=5, help="episodes per K in A")
ap.add_argument("--Ks_env", default="5,10,20,40,100,300,1000", help="K values for A")
ap.add_argument("--Ks_act", default="5,10,20,40,100,300,1000,3000", help="K values for B")
ap.add_argument("--n_states", type=int, default=2048, help="dataset states in B")
ap.add_argument("--n_samples", type=int, default=32, help="samples per state in B")
args = ap.parse_args()
t0 = time.time()

pol = Policy(args.run_dir, args.checkpoint)
dev, lo, hi = pol.device, pol.low, pol.high
obs, act = pol.data["observations"], pol.data["actions"]
res = pol.info()

# ---------------- A: return vs K (episode seeds 69, 70, ... as the training evals)
envs = [pol.minari_dataset.recover_environment() for _ in range(args.n_eps)]
res["returns"] = {}
for K in [int(k) for k in args.Ks_env.split(",")]:
    torch.manual_seed(0)
    o = [e.reset(seed=69 + i)[0] for i, e in enumerate(envs)]
    ret, done = np.zeros(args.n_eps), np.zeros(args.n_eps, bool)
    while not done.all():
        a = pol.sample(pol.states(np.stack(o)), K)[0].clamp(lo, hi).cpu().numpy()
        for i, e in enumerate(envs):
            if done[i]:
                continue
            o[i], r, term, trunc, _ = e.step(a[i])
            ret[i] += r
            done[i] = term or trunc
    res["returns"][K] = ret.tolist()

# ---------------- B: sampled actions on dataset states
rng = np.random.default_rng(0)
idx = rng.choice(len(obs), args.n_states, replace=False)
S, A = pol.states(obs[idx]), torch.tensor(act[idx], dtype=torch.float32, device=dev)
N, M, ad = args.n_states, args.n_samples, pol.action_dim
S_rep = S.repeat_interleave(M, 0)
Ks = [int(k) for k in args.Ks_act.split(",")]
torch.manual_seed(1)
_, snaps = pol.sample(S_rep, max(Ks), snaps=set(Ks))

with torch.no_grad():
    model = pol.model
    res["f_at_data"] = model(A, S).norm(dim=-1).mean().item()
    U = torch.rand(N, ad, device=dev) * (hi - lo) + lo
    res["f_at_uniform"] = model(U, S).norm(dim=-1).mean().item()

    # Dataset reference: conditional spread from each state's nearest dataset state
    # (skipping itself and its neighbouring time steps), and saturation.
    all_S, all_A = pol.states(obs), torch.tensor(act, dtype=torch.float32, device=dev)
    best_d = torch.full((N,), float("inf"), device=dev)
    best_j = torch.zeros(N, dtype=torch.long, device=dev)
    idx_t = torch.tensor(idx, device=dev)
    for c in range(0, len(obs), 100_000):
        d = torch.cdist(S, all_S[c:c + 100_000])
        j = torch.arange(c, c + d.shape[1], device=dev)
        d[(idx_t[:, None] - j[None, :]).abs() <= 1] = float("inf")
        dm, jm = d.min(1)
        better = dm < best_d
        best_d[better], best_j[better] = dm[better], jm[better] + c
    res["nn_state_dist_median"] = best_d.median().item()
    res["data_cond_std_nn"] = ((A - all_A[best_j]).pow(2).mean().sqrt() / np.sqrt(2)).item()
    res["data_sat_frac"] = (A.abs() > 0.99 * hi).float().mean().item()
    res["data_action_std_marginal"] = A.std(0).mean().item()

    res["act"] = {}
    prev = None
    for K in Ks:
        x = snaps[K]
        f = model(x, S_rep).norm(dim=-1)
        xc = x.clamp(lo, hi).view(N, M, ad)
        res["act"][K] = dict(
            f_norm=f.mean().item(),                                     # field norm at the endpoint
            f_norm_p90=f.quantile(0.9).item(),
            spread=xc.std(1).mean().item(),                             # per-dim std over one state's samples
            rmse=(xc - A[:, None]).pow(2).mean().sqrt().item(),         # sample vs the state's dataset action
            bias=(xc.mean(1) - A).pow(2).mean().sqrt().item(),          # per-state sample mean vs dataset action
            clip=(x.abs() > hi).float().mean().item(),                  # action dims outside the bounds
            sat=(xc.abs() > 0.99 * hi).float().mean().item(),           # action dims at the bounds after clipping
            move_from_prev=(x - prev).norm(dim=-1).mean().item() if prev is not None else None,
        )
        prev = x

res["seconds"] = time.time() - t0
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(res, indent=1))
print("done", res["run"], f"{res['seconds']:.0f}s")
