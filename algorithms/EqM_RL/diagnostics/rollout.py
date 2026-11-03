"""Roll out one trained Aware_Minari run and record return, flips and speed (EqM_RL.md,
Experiment 2, diagnostic C). A flip is HalfCheetah's body angle obs[1] beyond 2 rad;
obs[8] is the forward speed.

    python diagnostics/rollout.py <run checkpoint dir> diagnostics/out/roll/<name>_K20.json --K 20
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from common import Policy

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("run_dir", type=Path, help="checkpoint folder of one run (has config.yaml)")
ap.add_argument("out", type=Path, help="output JSON")
ap.add_argument("--checkpoint", help="file name, default: the latest checkpoint_*.pt")
ap.add_argument("--K", type=int, default=20, help="sampler steps")
ap.add_argument("--n_eps", type=int, default=10, help="episodes (seeds 1000, 1001, ...)")
ap.add_argument("--flip_angle", type=float, default=2.0, help="|body angle| in rad that counts as a flip")
args = ap.parse_args()

pol = Policy(args.run_dir, args.checkpoint)
envs = [pol.minari_dataset.recover_environment() for _ in range(args.n_eps)]
torch.manual_seed(0)
o = [e.reset(seed=1000 + i)[0] for i, e in enumerate(envs)]
n = args.n_eps
ret, done = np.zeros(n), np.zeros(n, bool)
flip_step, ret_at_flip = np.full(n, -1), np.full(n, np.nan)
speeds = [[] for _ in range(n)]
t = 0
while not done.all():
    a = pol.sample(pol.states(np.stack(o)), args.K)[0].clamp(pol.low, pol.high).cpu().numpy()
    for i, e in enumerate(envs):
        if done[i]:
            continue
        o[i], r, term, trunc, _ = e.step(a[i])
        ret[i] += r
        speeds[i].append(o[i][8])
        if flip_step[i] < 0 and abs(o[i][1]) > args.flip_angle:
            flip_step[i], ret_at_flip[i] = t, ret[i]
        done[i] = term or trunc
    t += 1

res = dict(pol.info(), K=args.K, returns=ret.tolist(), flip_step=flip_step.tolist(),
           ret_at_flip=ret_at_flip.tolist(),
           speed_before_flip=[float(np.mean(sp[:fs] if fs > 0 else sp)) for sp, fs in zip(speeds, flip_step)])
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(res))
print("done", pol.run_dir.name, args.K)
